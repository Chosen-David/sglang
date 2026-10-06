"""MoBA sparse attention backend（E89 复现臂，e2e 延迟对比）。

--attention-backend moba 启动。改造基底 = Quest backend（复用其
req_to_token/pool 接线、forward_decode/forward_extend 流程、
_dense_attn / _sparse_attn_batched / _dense_extend_one / _sparse_extend_one
消费端、CUDA graph veto），差异只在索引与选择：

  - 索引：chunk=64 全维（D=128）key 块和（fp32），chunk-mean = ksum/64
    ——E89 零 pad mean 的精确等价；pool 只存 ksum（无 min/max 上界 /
    4bit 量化 / far-near 分区 / L2 细筛）
  - 选择：chunk-mean gate（training-free，GQA 组内 mean 到 kv-head 级）
    top-(K2/BS)=16 块全展开 + sink(128) / swa(128) 保送（与 PSI 同口径）
  - attention 消费端：复用 TLI 的 fused gather+attn kernel
    （tli_sparse_gather_attn_dot，quest 同款）——同 kernel = 公平对照，
    各臂差异只在索引与选择
  - CUDA graph：不支持（恒 veto；三臂 e2e bench 脚本均 disable cuda
    graph，口径不受影响，kernel 级延迟另有 microbench）

口径声明：原版 MoBA（Moonshot）的 gate 是训练参数；本复现为
training-free chunk-mean gate（E89 质量侧验证 LongBench 13 任务 49.19，
run_scripts/e89_moba_smoke.sh：K2=1024 / chunk=64 / bf16 前向）。

索引维护成本诚实计入：prefill 每 chunk 建索引（块和增量合并，O(nq)）+
decode 每步增量 O(1)，全部 GPU 同步完成、计入 prefill/decode 墙钟。
"""

from __future__ import annotations

import time

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.moba.config import MoBAProfile
from sglang.srt.layers.attention.moba.indexer import (
    MoBAIndexer,
    _MAX_INCREMENTAL_NEW,
)
from sglang.srt.layers.attention.quest.backend import QuestSparseAttnBackend

import os

_TIMING = os.environ.get("SGLANG_MOBA_TIMING", "0") == "1"


class _PhaseTimer:
    """SGLANG_MOBA_TIMING=1 时累计 decode 各阶段墙钟（归因用，含同步代价）。"""

    def __init__(self):
        self.acc = {}
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
                f"[MoBA timing] steps={self.steps} total={total * 1000:.1f}ms "
                f"({total / self.steps * 1000:.2f}ms/step) {parts}"
            )
            for k in self.acc:
                self.acc[k] = 0.0


class MoBASparseAttnBackend(QuestSparseAttnBackend):
    """MoBA training-free 复现臂（quest backend 子类：仅换索引与选择）。

    继承自 quest 的部分（零改动复用）：
      _save_kv_cache / _dense_attn / _sparse_attn_batched /
      _dense_extend_one / _sparse_extend_one（sel+哨兵约定通用）/
      init_forward_metadata（行回收）/ veto_cuda_graph（恒 True）/
      init_cuda_graph_state（显式报错）/ get_cuda_graph_seq_len_fill_value
    """

    def __init__(self, runner=None) -> None:
        # 不走 quest.__init__（其 profile/token_to_kv_pool 布线一致但
        # profile 类型不同）；init 布线与 quest 逐行同构
        AttentionBackend.__init__(self)
        self.runner = runner
        self.profile = MoBAProfile()
        self.indexers: dict[int, MoBAIndexer] = {}
        # per-layer 共享 index pool：ksum [R, Hkv, NBLK_CAP, D] fp32
        self.index_pools: dict[int, dict] = {}
        self.token_to_kv_pool = None
        self.req_to_token = None
        self._num_layers = None
        self.head_dim = 128
        self.timer = _PhaseTimer()
        if runner is not None:
            self._init_from_runner(runner)

    def _tick(self):
        return self.timer.tick()

    def _add(self, phase, dt):
        self.timer.add(phase, dt)

    def _init_from_runner(self, runner) -> None:
        # 与 quest._init_from_runner 同构（TP 兼容取 per-rank kv head 数）
        model_config = runner.model_config
        self.head_dim = model_config.head_dim if model_config.head_dim else 128
        try:
            from sglang.srt.distributed import get_parallel

            self.num_kv_heads = model_config.get_num_kv_heads(
                get_parallel().attn_tp_size, get_parallel().attn_dcp_size
            )
        except Exception:
            self.num_kv_heads = (
                getattr(model_config, "num_key_value_heads", None) or 1
            )
        self._num_layers = model_config.num_hidden_layers
        self.dense_threshold = self.profile.dense_threshold
        self.token_to_kv_pool = runner.token_to_kv_pool
        self.req_to_token = runner.req_to_token_pool.req_to_token

    def _get_indexer(self, layer_id: int) -> MoBAIndexer:
        if layer_id not in self.indexers:
            self.indexers[layer_id] = MoBAIndexer(
                self.profile, head_dim=self.head_dim
            )
        return self.indexers[layer_id]

    # ---------------- 共享 index pool（ksum 单键，layout [R,Hkv,NC,D]） ---------------- #

    def _get_pool(self, layer_id: int) -> dict:
        if layer_id not in self.index_pools:
            p = self.profile
            Hkv = self.num_kv_heads
            dev = self.runner.device if self.runner else "cuda"
            s_cap = max(4096, p.dense_threshold + 1, p.pool_s_cap)
            nblk_cap = (s_cap + p.chunk_size - 1) // p.chunk_size
            r_cap = p.pool_rows
            # 块和存 fp32（gate 打分恒 fp32 精确；8B/token/kv-head 摊销）
            self.index_pools[layer_id] = {
                "ksum": torch.zeros(
                    r_cap, Hkv, nblk_cap, self.head_dim,
                    dtype=torch.float32, device=dev,
                ),
                "S_cap": s_cap,
                "R_cap": r_cap,
                "free": list(range(r_cap)),
                "row_of": {},  # req_pool_idx -> row
                "S": [-1] * r_cap,  # row -> 有效长度（-1 = 空）
            }
        return self.index_pools[layer_id]

    def _alloc_row(self, pool_l: dict, req: int) -> int:
        row = pool_l["row_of"].get(req)
        if row is None:
            if not pool_l["free"]:
                self._grow_pool_r(pool_l)
            row = pool_l["free"].pop()
            pool_l["row_of"][req] = row
            pool_l["S"][row] = -1
        return row

    def _ensure_pool_s(self, pool_l: dict, need: int) -> None:
        """S 维几何扩容（need ≤ S_cap 时 no-op）；旧行前缀保留。"""
        if need <= pool_l["S_cap"]:
            return
        old = pool_l["S_cap"]
        cs = self.profile.chunk_size
        new_cap = max(need + 2048, old * 2)
        old_nblk = (old + cs - 1) // cs
        new_nblk = (new_cap + cs - 1) // cs
        t = pool_l["ksum"]
        new = t.new_zeros((t.shape[0], t.shape[1], new_nblk, t.shape[3]))
        new[:, :, :old_nblk] = t
        pool_l["ksum"] = new
        pool_l["S_cap"] = new_cap

    def _grow_pool_r(self, pool_l: dict, add: int = 16) -> None:
        r0 = pool_l["R_cap"]
        r1 = r0 + add
        t = pool_l["ksum"]
        new = t.new_zeros((r1, *t.shape[1:]))
        new[:r0] = t
        pool_l["ksum"] = new
        pool_l["free"].extend(range(r0, r1))
        pool_l["S"].extend([-1] * add)
        pool_l["R_cap"] = r1

    # ---------------- AttentionBackend 必须实现 ---------------- #

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
        # q/k/v 是 2D [T, H*D] / [T, Hkv*D]（RoPE 后），统一 view 成 3D
        bs = q.shape[0]
        H = q.shape[1] // self.head_dim
        Hkv = self.num_kv_heads
        G = H // Hkv
        q = q.view(bs, H, self.head_dim)
        k = k.view(bs, Hkv, self.head_dim)
        v = v.view(bs, Hkv, self.head_dim)
        if save_kv_cache:
            self._save_kv_cache(k, v, layer, forward_batch)
        kv_pool = self.token_to_kv_pool
        req_to_token = self.req_to_token
        out = torch.empty_like(q)
        layer_id = layer.layer_id
        indexer = self._get_indexer(layer_id)
        k_buf = kv_pool.get_kv_buffer(layer_id)[0]
        pool_l = None
        reqs_l = forward_batch.req_pool_indices.tolist()
        lens_l = forward_batch.seq_lens.tolist()
        sparse_rows: list[tuple[int, int, int]] = []  # (batch_row, pool_row, seq_len)
        inc_rows: list[int] = []
        inc_S_old: list[int] = []
        inc_reqs: list[int] = []
        for i in range(bs):
            req = reqs_l[i]
            seq_len = lens_l[i]  # 含当前 token
            if seq_len <= self.dense_threshold:
                t0 = self._tick()
                out[i] = self._dense_attn(
                    q[i], req_to_token[req, :seq_len], kv_pool, layer_id
                )
                self._add("dense", time.time() - t0)
                continue
            pool_l = self._get_pool(layer_id)
            row = self._alloc_row(pool_l, req)
            S_st = pool_l["S"][row]
            if (
                S_st < 0
                or S_st >= seq_len
                or seq_len - S_st > _MAX_INCREMENTAL_NEW
            ):
                # 新请求首步 / 行复用 / 大跳变：全量 build 写入 pool 行
                t0 = self._tick()
                self._ensure_pool_s(pool_l, seq_len)
                k_all = k_buf[req_to_token[req, :seq_len]]
                idx_new = indexer.build_chunk_index(k_all)
                nblk = idx_new["nblk"]
                pool_l["ksum"][row, :, :nblk] = idx_new["ksum"]
                pool_l["S"][row] = seq_len
                self._add("build", time.time() - t0)
            elif seq_len > S_st:
                # 增量：只取新 token（块和加法合并，O(1)）
                if seq_len - S_st == 1:
                    # decode 常态（每行恰 1 新 token）→ 收集后批量维护
                    inc_rows.append(row)
                    inc_S_old.append(S_st)
                    inc_reqs.append(req)
                elif S_st > 0:
                    t0 = self._tick()
                    self._ensure_pool_s(pool_l, seq_len)
                    k_new = k_buf[req_to_token[req, S_st:seq_len]]
                    self._update_row_full(pool_l, row, S_st, seq_len, k_new)
                    self._add("increment", time.time() - t0)
                else:
                    # S_st == 0 兜底：直接全量重建（_update_row_full 无旧尾块）
                    t0 = self._tick()
                    self._ensure_pool_s(pool_l, seq_len)
                    k_all = k_buf[req_to_token[req, :seq_len]]
                    idx_new = indexer.build_chunk_index(k_all)
                    nblk = idx_new["nblk"]
                    pool_l["ksum"][row, :, :nblk] = idx_new["ksum"]
                    self._add("build", time.time() - t0)
                pool_l["S"][row] = seq_len
            sparse_rows.append((i, row, seq_len))
        if inc_rows:
            # 批量增量维护（单次 flat scatter，launch 数与 bs 无关）
            t0 = self._tick()
            self._ensure_pool_s(pool_l, max(lens_l))
            reqs_t = torch.tensor(inc_reqs, device=q.device)
            S_old_t = torch.tensor(inc_S_old, device=q.device)
            slots = req_to_token[reqs_t, S_old_t]  # [n]（新 token 槽位）
            k_new_b = k_buf[slots]  # [n, Hkv, D]
            indexer.update_pool_rows_decode(
                pool_l, torch.tensor(inc_rows, device=q.device), inc_S_old, k_new_b
            )
            self._add("increment", time.time() - t0)
        if sparse_rows:
            t0 = self._tick()
            rows_t = torch.tensor([r for _, r, _ in sparse_rows], device=q.device)
            S_list = [S for _, _, S in sparse_rows]
            idx_t = torch.tensor(
                [i for i, _, _ in sparse_rows], device=q.device
            )
            sel = indexer.select_decode_batched(
                pool_l, rows_t, S_list, q[idx_t]
            )  # [n, Hkv, K]（无效 lane = per-row 哨兵 S_i）
            self._add("select", time.time() - t0)
            t0 = self._tick()
            out_sparse = self._sparse_attn_batched(
                q,
                [i for i, _, _ in sparse_rows],
                sel,
                [S for _, _, S in sparse_rows],
                forward_batch, req_to_token, kv_pool, layer_id, Hkv, G,
            )
            rows = [i for i, _, _ in sparse_rows]
            out[rows] = out_sparse.to(out.dtype)
            self._add("sparse_attn", time.time() - t0)
        self.timer.maybe_report()
        # 返回约定：[T, H*D]
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
        ends = torch.cumsum(extend_seq_lens, dim=0).tolist()
        starts = [0] + ends[:-1]
        for b in range(len(starts)):
            req = int(forward_batch.req_pool_indices[b])
            nq = ends[b] - starts[b]
            prefix = (
                int(extend_prefix_lens[b]) if extend_prefix_lens is not None else 0
            )
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
            pool_l = self._get_pool(layer_id)
            self._ensure_pool_s(pool_l, S)
            row = self._alloc_row(pool_l, req)
            S_st = pool_l["S"][row]
            t0 = self._tick()
            if S_st == prefix and prefix > 0:
                # chunked prefill 增量：前缀块已建，追加本 chunk 新 token
                # （块和加法合并，O(nq)）
                k_new = k_buf[locs[prefix:S]]
                self._update_row_full(pool_l, row, prefix, S, k_new)
            else:
                # 首 chunk / 行复用 / 跳变：全量重建（成本计入 prefill）
                k_all = k_buf[locs]
                idx_new = indexer.build_chunk_index(k_all)
                nblk = idx_new["nblk"]
                pool_l["ksum"][row, :, :nblk] = idx_new["ksum"]
            pool_l["S"][row] = S
            self._add("build", time.time() - t0)
            nblk = (S + self.profile.chunk_size - 1) // self.profile.chunk_size
            t_arr = torch.arange(prefix, S, device=q.device)
            t0 = self._tick()
            sel = indexer.select_extend(
                pool_l["ksum"][row, :, :nblk],
                nblk,
                q_b,
                t_arr,
                S,
            )  # [nq, Hkv, K]（哨兵 S；含 query 级因果掩蔽）
            self._add("select", time.time() - t0)
            t0 = self._tick()
            out[starts[b] : ends[b]] = self._sparse_extend_one(
                q_b, sel, locs, pool, layer_id, Hkv, G,
                q_raw=q[starts[b] : ends[b]],
            ).to(q.dtype)
            self._add("sparse_attn", time.time() - t0)
        # 返回约定：[T, H*D]
        return out

    # ---------------- 内部工具 ---------------- #

    def _update_row_full(
        self, pool_l: dict, row: int,
        S_old: int, S: int, k_new: torch.Tensor,
    ) -> None:
        """多 token 增量（chunked prefill / decode 跳变兜底）：块和加法合并。

        三段式（比 quest 的 min/max 结合律合并更简单——和是加法半群）：
          头段：并入旧尾块（S_old 非 chunk 对齐时）
          中段：对齐整 chunk 段直接重建块和
          尾段：不足一 chunk 的尾（S 非 chunk 对齐时）
        k_new: [n, Hkv, D]（全局位置 S_old..S，KV dtype）。
        """
        cs = self.profile.chunk_size
        n = S - S_old
        kf = k_new.float()
        Hkv, D = kf.shape[1], kf.shape[2]
        blk_old = S_old // cs
        # 头段：不足一个完整 chunk 的前缀（S_old 对齐时为 0）
        head = min(n, (-S_old) % cs)
        if head > 0:
            pool_l["ksum"][row, :, blk_old] += kf[:head].sum(0)
        # 中段：对齐整 chunk 段
        mid_start = S_old + head
        mid_n = ((S - mid_start) // cs) * cs
        if mid_n > 0:
            seg = kf[head : head + mid_n].reshape(mid_n // cs, cs, Hkv, D).sum(1)
            # seg [nb, Hkv, D] → pool layout [R,Hkv,NC,D] 的 [Hkv, nb, D]
            pool_l["ksum"][row, :, blk_old + 1 : blk_old + 1 + mid_n // cs] += (
                seg.permute(1, 0, 2)
            )
        # 尾段：不足一 chunk 的尾（S 非对齐时）
        tail_start = mid_start + mid_n
        if tail_start < S:
            pool_l["ksum"][row, :, tail_start // cs] += kf[tail_start - S_old :].sum(0)


__all__ = ["MoBASparseAttnBackend"]
