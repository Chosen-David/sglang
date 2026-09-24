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

import os
import time

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.tli.config import TLIProfile
from sglang.srt.layers.attention.tli.indexer import TLIIndexer

# 增量维护的单步上限：decode n=1 常态；超过（如 chunked/speculative 或
# req_pool_idx 被新请求复用导致 S 倒退/跳变）则全量重建
_MAX_INCREMENTAL_NEW = 512

# M3-b 开销归因：TLI_PROFILE_TIMING=1 时累计 decode 各阶段墙钟，
# 每 64 步打一行（torch.cuda.synchronize 后计时，含同步代价，仅供归因）
_TIMING = os.environ.get("TLI_PROFILE_TIMING", "0") == "1"


class _PhaseTimer:
    def __init__(self):
        self.acc = {}  # phase -> 秒（累计）
        self.steps = 0

    def add(self, phase, dt):
        self.acc[phase] = self.acc.get(phase, 0.0) + dt

    def tick(self):
        if _TIMING:
            torch.cuda.synchronize()
        return time.time()

    def maybe_report(self):
        if not _TIMING:
            return
        self.steps += 1
        if self.steps % 64 == 0:
            total = sum(self.acc.values())
            parts = " ".join(
                f"{k}={v * 1000:.1f}ms" for k, v in sorted(self.acc.items())
            )
            print(
                f"[TLI timing] steps={self.steps} total={total * 1000:.1f}ms "
                f"({total / self.steps * 1000:.2f}ms/step) {parts}"
            )
            for k in self.acc:
                self.acc[k] = 0.0


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
        self.timer = _PhaseTimer()
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
                t0 = self.timer.tick()
                out[i] = self._dense_attn(q[i], locs, pool, layer_id)
                self.timer.add("dense", time.time() - t0)
                continue
            key = (layer_id, req)
            idx = self.block_indices.get(key)
            if (
                idx is None
                or idx["S"] >= seq_len
                or seq_len - idx["S"] > _MAX_INCREMENTAL_NEW
            ):
                # 新请求首步 / req 槽位复用 / 大跳变：全量 build
                t0 = self.timer.tick()
                k_all = k_buf[locs].float()  # [S, Hkv, D]（RoPE 后）
                self.block_indices[key] = indexer.build_block_index(k_all)
                self.timer.add("build", time.time() - t0)
            elif seq_len > idx["S"]:
                # 增量：只取新 token（O(n)，E5b gov_report 186min 根因修复）
                t0 = self.timer.tick()
                k_new = k_buf[req_to_token[req, idx["S"] : seq_len]].float()
                indexer.update_block_index(idx, k_new)
                self.timer.add("increment", time.time() - t0)
            t0 = self.timer.tick()
            sel = indexer.select(
                self.block_indices[key], q[i : i + 1].float(), t,
                use_l1_kernel=self.profile.use_l1_kernel,
            )  # [Hkv,K2]
            self.timer.add("select", time.time() - t0)
            t0 = self.timer.tick()
            out[i] = self._sparse_attn(q[i].float(), sel, locs, pool, layer_id, Hkv, G)
            self.timer.add("sparse_attn", time.time() - t0)
        self.timer.maybe_report()
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
        """decode 稀疏：每 kv head 在 K2 候选上做 GQA attention（批量向量化）。

        sel: [Hkv, K2] 逻辑位置；locs[sel] → pool 槽位。
        M3-b：Hkv Python 循环（每 head 4-5 个小 kernel × 8 head ≈ 40 launch）
        → 展平成 [P, Hkv*D] 后一次 flat gather + 两个批量 einsum（~6 launch）。
        数值与循环版逐位一致（同 fp32 累加顺序）。
        """
        H = q_i.shape[0]
        # 注意顺序：get_kv_buffer 返回 (k, v)——原实现写反（v_buf, k_buf），
        # 因 smoke 的 decode 全走 dense 路径（S<2048）而潜伏，稀疏 decode
        # 首次触发（M3 吞吐基线）才暴露
        k_buf, v_buf = pool.get_kv_buffer(layer_id)
        pool_pos = locs[sel]  # [Hkv, K2] pool 槽位
        K2 = pool_pos.shape[-1]
        D = self.head_dim
        # flat 索引：槽位 s、kv head h、维 d → s*(Hkv*D) + h*D + d
        d_off = torch.arange(D, device=pool_pos.device)
        h_off = torch.arange(Hkv, device=pool_pos.device).view(Hkv, 1, 1) * D
        flat = pool_pos.unsqueeze(-1) * (Hkv * D) + h_off + d_off.view(1, 1, D)
        # 注意：必须 reshape(-1) 成 1D 再索引——reshape(-1, Hkv*D)[idx] 取的是
        # 整行（[N, 1024]），第一次实现就栽在这（gathered 1024× 大小）
        k_sel = k_buf.reshape(-1)[flat.view(-1)].view(Hkv, K2, D).float()
        v_sel = v_buf.reshape(-1)[flat.view(-1)].view(Hkv, K2, D).float()
        q_g = q_i.view(Hkv, G, D)  # [Hkv, G, D]
        att = torch.einsum("hgd,hkd->hgk", q_g, k_sel) * (D**-0.5)
        att = torch.softmax(att, dim=-1)
        o = torch.einsum("hgk,hkd->hgd", att, v_sel)  # [Hkv, G, D]
        return o.reshape(H, D)

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
        # forward_extend 的 out 是 2D [T, H*D]，此处须展平返回
        # （smoke 测试长 prompt 未过 dense_threshold，稀疏 prefill 路径
        #   首次被 e2e 触发时暴露的形状 bug）
        return out.view(nq, H * self.head_dim)


__all__ = ["TLISparseAttnBackend"]
