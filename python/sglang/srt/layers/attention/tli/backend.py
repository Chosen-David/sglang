"""TLI sparse attention backend（M2：paged 寻址 + 增量索引 + 稀疏 prefill）。

--attention-backend tli 启动。选择逻辑 = tli/indexer.py（与
two-level-attention/exp/trace 实测口径一致，mass coverage 0.9967–1.0000）。

M2 关键设计（correctness-first，kernel 化路线见 TWO_LEVEL_INDEXER_DESIGN.md §5）：
  - paged 寻址：所有 KV 读取经 req_to_token[req, :S] 间接寻址（不再假设
    page_size=1 的 req 起始连续布局）；选择返回的逻辑位置先映射到 pool 槽位
  - decode 增量索引：首步全量 build，之后每步只取新 token 调
    update_block_index（O(n)）——修复 E5b 实测的每步全量重建 O(S) 问题
    （gov_report 186min 根因）
  - extend（prefill）：S > dense_threshold 时走两级稀疏（build 全量索引 +
    select_batched 批量选择 + torch 稀疏前向）；短序列 dense
  - CUDA graph：占位未支持（init_cuda_graph_state no-op）
"""

from __future__ import annotations

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.tli.config import TLIProfile
from sglang.srt.layers.attention.tli.indexer import TLIIndexer

# 增量维护的单步上限：decode n=1 常态；超过（如 chunked/speculative 或
# req_pool_idx 被新请求复用导致 S 倒退/跳变）则全量重建
_MAX_INCREMENTAL_NEW = 512


class TLISparseAttnBackend(AttentionBackend):
    def __init__(self, runner=None) -> None:
        super().__init__()
        self.runner = runner
        self.profile = TLIProfile()
        self.indexers: dict[int, TLIIndexer] = {}
        # (layer_id, req_pool_idx) -> block index dict（TLIIndexer.build_block_index 产物）
        self.block_indices: dict[tuple[int, int], dict] = {}
        self.layer_skip: list[bool] | None = None
        self.token_to_kv_pool = None  # _init_from_runner 填充（新版挂在 model_runner 上）
        self.req_to_token = None
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
        # 新版 sglang：pool 挂在 model_runner 上（ForwardBatch 不再携带）
        self.token_to_kv_pool = runner.token_to_kv_pool
        self.req_to_token = runner.req_to_token_pool.req_to_token

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
        # 新版 q/k/v 是 2D [T, H*D] / [T, Hkv*D]（RoPE 后），统一 view 成 3D
        bs = q.shape[0]
        H = q.shape[1] // self.head_dim
        Hkv = self.num_kv_heads
        G = H // Hkv
        q = q.view(bs, H, self.head_dim)
        k = k.view(bs, Hkv, self.head_dim)
        v = v.view(bs, Hkv, self.head_dim)
        if save_kv_cache:
            self._save_kv_cache(k, v, layer, forward_batch)
        pool = self.token_to_kv_pool
        req_to_token = self.req_to_token
        out = torch.empty_like(q)
        layer_id = layer.layer_id
        indexer = self._get_indexer(layer_id)
        k_buf = pool.get_kv_buffer(layer_id)[0]
        for i in range(bs):
            req = int(forward_batch.req_pool_indices[i])
            seq_len = int(forward_batch.seq_lens[i])  # 含当前 token
            t = seq_len - 1
            locs = req_to_token[req, :seq_len]  # [S] 逻辑位置 → pool 槽位
            if seq_len <= self.dense_threshold:
                out[i] = self._dense_attn(q[i], locs, pool, layer_id)
                continue
            key = (layer_id, req)
            idx = self.block_indices.get(key)
            if (
                idx is None
                or idx["S"] >= seq_len
                or seq_len - idx["S"] > _MAX_INCREMENTAL_NEW
            ):
                # 新请求首步 / req 槽位复用 / 大跳变：全量 build
                k_all = k_buf[locs].float()  # [S, Hkv, D]（RoPE 后）
                self.block_indices[key] = indexer.build_block_index(k_all)
            elif seq_len > idx["S"]:
                # 增量：只取新 token（O(n)，E5b gov_report 186min 根因修复）
                k_new = k_buf[req_to_token[req, idx["S"] : seq_len]].float()
                indexer.update_block_index(idx, k_new)
            sel = indexer.select(
                self.block_indices[key], q[i : i + 1].float(), t,
                use_l1_kernel=self.profile.use_l1_kernel,
            )  # [Hkv,K2]
            out[i] = self._sparse_attn(q[i].float(), sel, locs, pool, layer_id, Hkv, G)
        # 返回约定：[T, H*D]（与 triton backend 的 reshape(-1, H*D) 一致）
        return out.reshape(bs, H * self.head_dim)

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
        # 新版 q/k/v 是 2D [T, H*D] / [T, Hkv*D]（RoPE 后），统一 view 成 3D
        T = q.shape[0]
        H = q.shape[1] // self.head_dim
        Hkv = self.num_kv_heads
        G = H // Hkv
        q = q.view(T, H, self.head_dim)
        k = k.view(T, Hkv, self.head_dim)
        v = v.view(T, Hkv, self.head_dim)
        if save_kv_cache:
            self._save_kv_cache(k, v, layer, forward_batch)
        pool = self.token_to_kv_pool
        req_to_token = self.req_to_token
        extend_seq_lens = forward_batch.extend_seq_lens
        extend_prefix_lens = forward_batch.extend_prefix_lens
        out = torch.empty(T, H * self.head_dim, dtype=q.dtype, device=q.device)
        layer_id = layer.layer_id
        # 新版 ForwardBatch 无 extend_seq_lens_cumulative：自行 cumsum
        ends = torch.cumsum(extend_seq_lens, dim=0).tolist()
        starts = [0] + ends[:-1]  # starts[b]..ends[b] = 第 b 个请求的 token 范围
        for b in range(len(starts)):
            req = int(forward_batch.req_pool_indices[b])
            nq = ends[b] - starts[b]
            prefix = int(extend_prefix_lens[b]) if extend_prefix_lens is not None else 0
            S = prefix + nq  # 已写池总长（前缀 + 当前 chunk）
            locs = req_to_token[req, :S]
            q_b = q[starts[b] : ends[b]].float()
            if S <= self.dense_threshold:
                out[starts[b] : ends[b]] = self._dense_extend_one(
                    q_b, locs, pool, layer_id, Hkv, G
                ).to(q.dtype)
                continue
            indexer = self._get_indexer(layer_id)
            k_buf = pool.get_kv_buffer(layer_id)[0]
            k_all = k_buf[locs].float()  # [S, Hkv, D]
            index = indexer.build_block_index(k_all)
            self.block_indices[(layer_id, req)] = index  # decode 增量起点
            t_arr = torch.arange(prefix, S, device=q.device)
            sel = indexer.select_batched(index, q_b, t_arr)  # [nq, Hkv, K2] 逻辑位置
            out[starts[b] : ends[b]] = self._sparse_extend_one(
                q_b, sel, locs, pool, layer_id, Hkv, G
            ).to(q.dtype)
        # 返回约定：[T, H*D]（helper 已按此形状返回）
        return out

    # ---------------- 内部工具 ---------------- #

    def _save_kv_cache(self, k, v, layer, forward_batch):
        self.token_to_kv_pool.set_kv_buffer(
            layer, forward_batch.out_cache_loc, k, v
        )

    def _dense_attn(self, q_i, locs, pool, layer_id):
        """decode 短序列 dense（GQA）。locs: [S] pool 槽位。"""
        k_all, v_all = pool.get_kv_buffer(layer_id)
        k_all = k_all[locs].float()  # [S, Hkv, D]
        v_all = v_all[locs].float()
        q_i = q_i.float()
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

    def _sparse_attn(self, q_i, sel, locs, pool, layer_id, Hkv, G):
        """decode 稀疏：每 kv head 在 K2 候选上做 GQA attention。

        sel: [Hkv, K2] 逻辑位置；locs[sel] → pool 槽位。
        """
        H = q_i.shape[0]
        outs = []
        v_buf, k_buf = pool.get_kv_buffer(layer_id)
        for h in range(Hkv):
            pos = sel[h]  # [K2] 逻辑位置
            pool_pos = locs[pos]  # [K2] pool 槽位
            k_sel = k_buf[pool_pos, h].float()  # [K2, D]
            v_sel = v_buf[pool_pos, h].float()
            q_h = q_i[h * G : (h + 1) * G]  # [G, D]
            att = q_h @ k_sel.T * (self.head_dim**-0.5)  # [G, K2]
            att = torch.softmax(att, dim=-1)
            outs.append(att @ v_sel)  # [G, D]
        return torch.cat(outs, dim=0)

    def _dense_extend_one(self, q_b, locs, pool, layer_id, Hkv, G):
        """单个请求的 dense causal attention（短序列 prefill / chunk）。"""
        k_all, v_all = pool.get_kv_buffer(layer_id)
        k_e = k_all[locs].float().transpose(0, 1)  # [Hkv, S, D]
        v_e = v_all[locs].float().transpose(0, 1)
        nq = q_b.shape[0]
        S = locs.shape[0]
        q_g = q_b.reshape(nq, Hkv, G, self.head_dim)
        att = torch.einsum("ahgd,hsd->ahgs", q_g, k_e) * (self.head_dim**-0.5)
        # q 行 r 的全局位置 = S - nq + r；因果 mask 相对块尾
        qpos = torch.arange(S - nq, S, device=q_b.device).view(nq, 1)
        causal = torch.arange(S, device=q_b.device).view(1, S) <= qpos
        att = att.masked_fill(~causal.view(nq, 1, 1, S), float("-inf"))
        att = torch.softmax(att, dim=-1)
        o = torch.einsum("ahgs,hsd->ahgd", att, v_e)
        return o.reshape(nq, -1)

    def _sparse_extend_one(self, q_b, sel, locs, pool, layer_id, Hkv, G):
        """单个请求的稀疏 prefill（select_batched 选择 + gather 前向）。

        sel: [nq, Hkv, K2] 逻辑位置（far+near 拼接，scatter 口径无重复）。
        按 kv head 循环 + 行分块控制 gather 峰值显存。
        """
        k_buf, v_buf = pool.get_kv_buffer(layer_id)
        nq, H = q_b.shape[0], q_b.shape[1]
        K2 = sel.shape[-1]
        pool_sel = locs[sel]  # [nq, Hkv, K2] pool 槽位
        out = torch.empty(nq, H, self.head_dim, device=q_b.device, dtype=q_b.dtype)
        row_chunk = max(1, min(nq, 512))  # [512, K2=1024, D=128] fp32 ≈ 268MB/head
        for h in range(Hkv):
            q_h = q_b[:, h * G : (h + 1) * G]  # [nq, G, D]
            for r0 in range(0, nq, row_chunk):
                r1 = min(r0 + row_chunk, nq)
                pp = pool_sel[r0:r1, h]  # [n, K2]
                k_sel = k_buf[pp, h].float()  # [n, K2, D]
                v_sel = v_buf[pp, h].float()
                att = torch.einsum("ngd,nkd->ngk", q_h[r0:r1], k_sel) * (
                    self.head_dim**-0.5
                )
                att = torch.softmax(att, dim=-1)
                out[r0:r1, h * G : (h + 1) * G] = torch.einsum(
                    "ngk,nkd->ngd", att, v_sel
                ).to(q_b.dtype)
        return out


__all__ = ["TLISparseAttnBackend"]
