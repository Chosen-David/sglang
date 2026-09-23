"""TLI sparse attention backend（prototype：correctness-first）。

--attention-backend tli 启动。选择逻辑 = tli/indexer.py（与
two-level-attention/exp/trace 实测口径一致，mass coverage 0.9967–1.0000）。

prototype 边界（后续 kernel 化路线见 TWO_LEVEL_INDEXER_DESIGN.md §5/§4.3）：
  - decode：gather 全量 K（page_size=1 连续布局假设）→ 两级选择 → torch 稀疏前向
  - extend（prefill）：退化为 dense torch attention（QSA 的行分块稀疏 prefill
    路径是 M2 工作；短序列 dense 与 DSA 的阈值策略一致）
  - CUDA graph：占位未支持（init_cuda_graph_state no-op）
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.tli.config import TLIProfile
from sglang.srt.layers.attention.tli.indexer import TLIIndexer


class TLISparseAttnBackend(AttentionBackend):
    def __init__(self, runner=None) -> None:
        super().__init__()
        self.runner = runner
        self.profile = TLIProfile()
        self.indexers: dict[int, TLIIndexer] = {}
        # (layer_id, req_pool_idx) -> block index dict（TLIIndexer.build_block_index 产物）
        self.block_indices: dict[tuple[int, int], dict] = {}
        self.layer_skip: list[bool] | None = None
        if runner is not None:
            self._init_from_runner(runner)

    def _init_from_runner(self, runner) -> None:
        model_config = runner.model_config
        self.head_dim = model_config.head_dim if model_config.head_dim else 128
        self.num_kv_heads = getattr(model_config, "num_key_value_heads", None) or 1
        n_layers = model_config.num_hidden_layers
        if self.profile.layer_skip_path:
            self.layer_skip = self.profile.load_layer_skip(n_layers)
        self.dense_threshold = self.profile.dense_threshold

    def _get_indexer(self, layer_id: int) -> TLIIndexer:
        if layer_id not in self.indexers:
            idx = TLIIndexer(self.profile, head_dim=self.head_dim).to(
                self.runner.device if self.runner else "cuda"
            )
            if self.layer_skip is not None:
                idx.skip_far = self.layer_skip[
                    layer_id - (self._layer_offset or 0) if layer_id >= (self._layer_offset or 0) else 0
                ]
            self.indexers[layer_id] = idx
        return self.indexers[layer_id]

    _layer_offset = 0

    # ---------------- AttentionBackend 必须实现 ---------------- #

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        # prototype 未支持 CUDA graph decode
        pass

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def init_forward_metadata(self, forward_batch):
        pass

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: torch.nn.Module,
        forward_batch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        if save_kv_cache:
            self._save_kv_cache(k, v, layer, forward_batch)
        pool = forward_batch.token_to_kv_pool
        bs = q.shape[0]
        H = q.shape[1]
        Hkv = self.num_kv_heads
        G = H // Hkv
        out = torch.empty_like(q)
        layer_id = layer.layer_id
        indexer = self._get_indexer(layer_id)
        for i in range(bs):
            req = int(forward_batch.req_pool_indices[i])
            seq_len = int(forward_batch.seq_lens[i])  # 含当前 token
            t = seq_len - 1
            k_all = pool.get_kv_buffer(layer_id)[0][
                req : req + seq_len
            ].float()  # [S, Hkv, D]（page_size=1 假设）
            if seq_len <= self.dense_threshold:
                out[i] = self._dense_attn(q[i], k_all, v_all := None, req, pool, layer_id)
                continue
            key = (layer_id, req)
            if key not in self.block_indices or self.block_indices[key]["S"] != seq_len:
                # 新请求 prefill 后首步（或对拍重建）：全量 build
                self.block_indices[key] = indexer.build_block_index(k_all)
            sel = indexer.select(self.block_indices[key], q[i : i + 1].float(), t)  # [Hkv,K2]
            out[i] = self._sparse_attn(q[i].float(), sel, req, pool, layer_id, Hkv, G)
        return out.view(1, bs, H, self.head_dim)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: torch.nn.Module,
        forward_batch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        if save_kv_cache:
            self._save_kv_cache(k, v, layer, forward_batch)
        # prototype：prefill 走 dense（M2 工作 = QSA 式行分块稀疏 prefill）
        return self._dense_extend(q, k, v, forward_batch)

    # ---------------- 内部工具 ---------------- #

    def _save_kv_cache(self, k, v, layer, forward_batch):
        pool = forward_batch.token_to_kv_pool
        out_cache_loc = forward_batch.out_cache_loc
        pool.set_kv_buffer(layer, k, v, out_cache_loc)

    def _dense_attn(self, q_i, k_all, _, req, pool, layer_id):
        v_all = pool.get_kv_buffer(layer_id)[1][
            req : req + k_all.shape[0]
        ].float()
        H = q_i.shape[0]
        Hkv = k_all.shape[1]
        G = H // Hkv
        q_g = q_i.reshape(Hkv, G, -1)
        k_e = k_all.transpose(0, 1)  # [Hkv, S, D]
        v_e = v_all.transpose(0, 1)
        att = torch.einsum("hgd,hsd->hgs", q_g, k_e) * (self.head_dim**-0.5)
        att = torch.softmax(att, dim=-1)
        o = torch.einsum("hgs,hsd->hgd", att, v_e)
        return o.reshape(H, -1)

    def _sparse_attn(self, q_i, sel, req, pool, layer_id, Hkv, G):
        """每 kv head 在自己的 K2 候选上做 GQA attention。"""
        H = q_i.shape[0]
        outs = []
        v_buf, k_buf = pool.get_kv_buffer(layer_id)
        for h in range(Hkv):
            pos = sel[h]  # [K2]
            k_sel = k_buf[req + pos, h].float()  # [K2, D]
            v_sel = v_buf[req + pos, h].float()
            q_h = q_i[h * G : (h + 1) * G]  # [G, D]
            att = q_h @ k_sel.T * (self.head_dim**-0.5)  # [G, K2]
            att = torch.softmax(att, dim=-1)
            outs.append(att @ v_sel)  # [G, D]
        return torch.cat(outs, dim=0)

    def _dense_extend(self, q, k, v, forward_batch):
        """varlen dense attention（prototype：不分块，依赖上层 chunked prefill 限长）。"""
        # q/k/v: [total_tokens, H, D]
        cu = forward_batch.extend_seq_lens_cumulative
        H = q.shape[1]
        Hkv = k.shape[1]
        G = H // Hkv
        out = torch.empty_like(q)
        starts = cu[:-1] if cu.shape[0] > 1 else torch.zeros(1, dtype=torch.long, device=q.device)
        ends = cu
        for b in range(len(ends)):
            qs = q[starts[b] : ends[b]].float().reshape(-1, Hkv, G, self.head_dim)
            ks = k[starts[b] : ends[b]].float().transpose(0, 1)  # [Hkv,S,D]
            vs = v[starts[b] : ends[b]].float().transpose(0, 1)
            att = torch.einsum("bhgd,hsd->bhgs", qs, ks) * (self.head_dim**-0.5)
            S = ks.shape[1]
            causal = torch.ones(S, S, dtype=torch.bool, device=q.device).tril()
            att = att.masked_fill(~causal.view(1, 1, S, S), float("-inf"))
            att = torch.softmax(att, dim=-1)
            o = torch.einsum("bhgs,hsd->bhgd", att, vs)
            out[starts[b] : ends[b]] = o.reshape(-1, H, self.head_dim).to(q.dtype)
        return out


__all__ = ["TLISparseAttnBackend"]
