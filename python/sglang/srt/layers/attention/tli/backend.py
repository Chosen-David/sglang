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
  - M4 批量化 decode：per-layer 共享 index pool（kq/kmin/kmax 预分配 +
    几何扩容，行随请求生命周期回收）+ n≥2 时两级选择批量 eager 化
    （select_decode_batched，launch 数与 bs 无关）+ 批量稀疏前向；
    n==1 保留 per-request L1/L2 fused kernel 路径（bs=1 延迟优势）
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
        # M4：per-layer 共享 index pool（kq/kmin/kmax 预分配 [R, cap, ...]，
        # 几何扩容），替代 per-request block_indices dict——跨请求批量
        # select/gather 的寻址前提。有效长度由 pool["S"][row] 跟踪，
        # [S:cap) 是垃圾，消费方必须按 S/nblk 切片或掩码访问。
        self.index_pools: dict[int, dict] = {}
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

    # ---------------- 共享 index pool（M4） ---------------- #

    def _get_pool(self, layer_id: int) -> dict:
        if layer_id not in self.index_pools:
            p = self.profile
            Hkv = self.num_kv_heads
            dev = self.runner.device if self.runner else "cuda"
            s_cap = max(4096, p.dense_threshold + 1)
            nblk_cap = (s_cap + p.block_size - 1) // p.block_size
            r_cap = p.pool_rows
            self.index_pools[layer_id] = {
                "kq": torch.zeros(r_cap, s_cap, Hkv, 2 * p.delta, device=dev),
                "kmin": torch.zeros(r_cap, nblk_cap, Hkv, p.coarse_dim, device=dev),
                "kmax": torch.zeros(r_cap, nblk_cap, Hkv, p.coarse_dim, device=dev),
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
        """S 维几何扩容（need ≤ S_cap 时 no-op）。扩容后旧行内容前缀保留，
        [old:cap) 新容量零初始化。"""
        if need <= pool_l["S_cap"]:
            return
        old = pool_l["S_cap"]
        new_cap = max(need + 2048, old * 2)
        bs = self.profile.block_size
        old_nblk = (old + bs - 1) // bs
        new_nblk = (new_cap + bs - 1) // bs
        for key, width in (("kq", old), ("kmin", old_nblk), ("kmax", old_nblk)):
            t = pool_l[key]
            new = t.new_zeros((t.shape[0], new_cap if key == "kq" else new_nblk, *t.shape[2:]))
            new[:, :width] = t
            pool_l[key] = new
        pool_l["S_cap"] = new_cap

    def _grow_pool_r(self, pool_l: dict, add: int = 16) -> None:
        r0 = pool_l["R_cap"]
        r1 = r0 + add
        for key in ("kq", "kmin", "kmax"):
            t = pool_l[key]
            new = t.new_zeros((r1, *t.shape[1:]))
            new[:r0] = t
            pool_l[key] = new
        pool_l["free"].extend(range(r0, r1))
        pool_l["S"].extend([-1] * add)
        pool_l["R_cap"] = r1

    def _row_views(self, pool_l: dict, row: int, S: int) -> dict:
        """构造 per-request select()/update_block_index() 兼容的 view dict。

        张量是 pool 行的 view（[cap,...] 带容量 padding）——调用方须保证
        pool 容量足够（update 内的 _ensure 不触发，否则会静默脱离 pool）。
        """
        nblk = (S + self.profile.block_size - 1) // self.profile.block_size
        return {
            "kmin": pool_l["kmin"][row],
            "kmax": pool_l["kmax"][row],
            "kq": pool_l["kq"][row],
            "nblk": nblk,
            "S": S,
        }

    # ---------------- AttentionBackend 必须实现 ---------------- #

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        # prototype 未支持 CUDA graph decode
        pass

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def init_forward_metadata(self, forward_batch):
        # M4：lazy 回收——不在当前 batch 的请求释放 pool 行（请求结束/
        # req 槽位被新请求复用前的清理；row_of 的 S 检查兜底漏网情形）
        if not self.index_pools:
            return
        try:
            active = {int(x) for x in forward_batch.req_pool_indices.tolist()}
        except Exception:
            return
        for pool_l in self.index_pools.values():
            for req in list(pool_l["row_of"]):
                if req not in active:
                    row = pool_l["row_of"].pop(req)
                    pool_l["S"][row] = -1
                    pool_l["free"].append(row)

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
        kv_pool = self.token_to_kv_pool
        req_to_token = self.req_to_token
        out = torch.empty_like(q)
        layer_id = layer.layer_id
        indexer = self._get_indexer(layer_id)
        k_buf = kv_pool.get_kv_buffer(layer_id)[0]
        pool_l = None
        # M4：稀疏请求的索引写入 per-layer 共享 pool 行（增量 O(n)），
        # n≥2 时两级选择批量 eager 化（launch 数与 bs 无关）；dense 逐请求
        # phase-3：req/seq_len 一次 .tolist()（2 次同步替代逐行 int() 的
        # 2×bs 次）；单 token 增量行收集后批量维护（update_pool_rows_decode）
        reqs_l = forward_batch.req_pool_indices.tolist()
        lens_l = forward_batch.seq_lens.tolist()
        sparse_rows: list[tuple[int, int, int]] = []  # (batch_row, pool_row, seq_len)
        inc_rows: list[int] = []  # 批量增量维护的 (pool 行, 旧 S, req)
        inc_S_old: list[int] = []
        inc_reqs: list[int] = []
        for i in range(bs):
            req = reqs_l[i]
            seq_len = lens_l[i]  # 含当前 token
            if seq_len <= self.dense_threshold:
                t0 = self.timer.tick()
                out[i] = self._dense_attn(q[i], req_to_token[req, :seq_len], kv_pool, layer_id)
                self.timer.add("dense", time.time() - t0)
                continue
            pool_l = self._get_pool(layer_id)
            row = self._alloc_row(pool_l, req)
            S_st = pool_l["S"][row]
            if S_st < 0 or S_st >= seq_len or seq_len - S_st > _MAX_INCREMENTAL_NEW:
                # 新请求首步 / req 槽位复用 / 大跳变：全量 build 写入 pool 行
                t0 = self.timer.tick()
                self._ensure_pool_s(pool_l, seq_len)
                k_all = k_buf[req_to_token[req, :seq_len]].float()  # [S, Hkv, D]
                idx_new = indexer.build_block_index(k_all)
                nblk = idx_new["nblk"]
                pool_l["kq"][row, :seq_len] = idx_new["kq"]
                pool_l["kmin"][row, :nblk] = idx_new["kmin"]
                pool_l["kmax"][row, :nblk] = idx_new["kmax"]
                pool_l["S"][row] = seq_len
                self.timer.add("build", time.time() - t0)
            elif seq_len > S_st:
                # 增量：只取新 token（O(n)，E5b gov_report 186min 根因修复）
                if seq_len - S_st == 1:
                    # decode 常态（每行恰 1 新 token）→ 收集后批量维护
                    inc_rows.append(row)
                    inc_S_old.append(S_st)
                    inc_reqs.append(req)
                else:
                    t0 = self.timer.tick()
                    self._ensure_pool_s(pool_l, seq_len)
                    k_new = k_buf[req_to_token[req, S_st:seq_len]].float()
                    indexer.update_block_index(self._row_views(pool_l, row, S_st), k_new)
                    self.timer.add("increment", time.time() - t0)
                pool_l["S"][row] = seq_len
            sparse_rows.append((i, row, seq_len))
        if inc_rows:
            # 批量增量维护（~10 launch 总量）：新 token K 从 pool 槽位一次
            # gather 出 [n, Hkv, D]，flat 索引 scatter 写 kq/kmin/kmax
            t0 = self.timer.tick()
            self._ensure_pool_s(pool_l, max(lens_l))
            reqs_t = torch.tensor(inc_reqs, device=q.device)
            S_old_t = torch.tensor(inc_S_old, device=q.device)
            slots = req_to_token[reqs_t, S_old_t]  # [n]（req_to_token 已含新 token 槽位）
            k_new_b = k_buf[slots].float()  # [n, Hkv, D]
            indexer.update_pool_rows_decode(
                pool_l, torch.tensor(inc_rows, device=q.device), inc_S_old, k_new_b
            )
            self.timer.add("increment", time.time() - t0)
        if sparse_rows:
            t0 = self.timer.tick()
            if len(sparse_rows) >= 2 and self.profile.use_batch_select:
                rows_t = torch.tensor([r for _, r, _ in sparse_rows], device=q.device)
                S_list = [S for _, _, S in sparse_rows]
                idx_t = torch.tensor(
                    [i for i, _, _ in sparse_rows], device=q.device
                )
                sel = indexer.select_decode_batched(
                    pool_l["kq"], pool_l["kmin"], pool_l["kmax"],
                    rows_t, S_list, q[idx_t].float(),
                )  # [n, Hkv, K2']（池不足槽位为哨兵 S_cap）
            else:
                # n==1 / A/B 对拍：per-request select（含 L1/L2 fused kernel 路径）
                sels = []
                for i, row, seq_len in sparse_rows:
                    sels.append(
                        indexer.select(
                            self._row_views(pool_l, row, seq_len),
                            q[i : i + 1].float(),
                            seq_len - 1,
                            use_l1_kernel=self.profile.use_l1_kernel,
                            use_l2_kernel=self.profile.use_l2_kernel,
                        )
                    )
                sel = torch.stack(sels)  # [n, Hkv, K2]
            self.timer.add("select", time.time() - t0)
            t0 = self.timer.tick()
            out_sparse = self._sparse_attn_batched(
                q,
                [i for i, _, _ in sparse_rows],
                sel,
                [S for _, _, S in sparse_rows],
                forward_batch, req_to_token, kv_pool, layer_id, Hkv, G,
            )
            rows = [i for i, _, _ in sparse_rows]
            out[rows] = out_sparse.to(out.dtype)  # 张量索引赋值要求 dtype 一致
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
            # M4：写入共享 index pool（decode 增量起点；不再存 per-request dict）
            pool_l = self._get_pool(layer_id)
            self._ensure_pool_s(pool_l, S)
            row = self._alloc_row(pool_l, req)
            pool_l["kq"][row, :S] = index["kq"]
            pool_l["kmin"][row, : index["nblk"]] = index["kmin"]
            pool_l["kmax"][row, : index["nblk"]] = index["kmax"]
            pool_l["S"][row] = S
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
        # 哨兵 pad 处理（L2 fused kernel 路径返回 [Hkv, K2+pad]，pad=S）：
        # 位置合法值域 [0, locs.shape[0])，== S 即 pad，softmax 前屏蔽
        S_loc = locs.shape[0]
        valid = sel < S_loc  # [Hkv, K2p]（eager 路径全 True，零开销约定）
        sel_c = sel.clamp(max=S_loc - 1)
        pool_pos = locs[sel_c]  # [Hkv, K2p] pool 槽位
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
        att = att.masked_fill(~valid.unsqueeze(1), float("-inf"))
        att = torch.softmax(att, dim=-1)
        o = torch.einsum("hgk,hkd->hgd", att, v_sel)  # [Hkv, G, D]
        return o.reshape(H, D)

    def _sparse_attn_batched(
        self, q, row_idx, sel, seq_lens, forward_batch, req_to_token, pool, layer_id, Hkv, G
    ):
        """M4：批量稀疏前向（所有稀疏请求一次 gather + 两个批量 einsum）。

        row_idx: 批内行号列表；sel: [n, Hkv, K2]（批量或 per-request stack，
        哨兵可为 S_i 或 S_cap——只需 ≥ seq_len_i）；seq_lens: list[int]。
        哨兵 pad：valid = sel < seq_len_i（逐行界），softmax 前屏蔽。
        数值与逐请求版一致（同 fp32 累加顺序）。
        """
        n = sel.shape[0]
        H, D = q.shape[1], self.head_dim
        rows = torch.tensor(row_idx, device=q.device)
        seq_lens_t = torch.tensor(seq_lens, device=q.device, dtype=torch.long).view(
            -1, 1, 1
        )
        valid = sel < seq_lens_t  # [n, Hkv, K2]
        # Python max（避免 .item() 的 GPU 同步；seq_lens 是 Python list）
        sel_c = sel.clamp(max=max(seq_lens) - 1)
        # req_to_token 是 2D [max_req, max_S]：一次 gather 出全部请求槽位
        reqs = forward_batch.req_pool_indices[rows]  # [n]
        locs_full = req_to_token[reqs]  # [n, max_S]（每行 = 该请求逻辑位置 → 槽位）
        pool_pos = torch.gather(
            locs_full, 1, sel_c.view(n, -1)
        ).view(n, Hkv, sel.shape[-1])  # [n, Hkv, K2] pool 槽位
        K2 = pool_pos.shape[-1]
        k_buf, v_buf = pool.get_kv_buffer(layer_id)
        # flat 索引：槽位 s、kv head h、维 d → s*(Hkv*D) + h*D + d
        d_off = torch.arange(D, device=q.device)
        h_off = torch.arange(Hkv, device=q.device).view(1, Hkv, 1, 1) * D
        flat = (
            pool_pos.unsqueeze(-1) * (Hkv * D) + h_off + d_off.view(1, 1, 1, D)
        )  # [n, Hkv, K2, D]
        flat_v = flat.view(n, -1)
        k_sel = k_buf.reshape(-1)[flat_v].view(n, Hkv, K2, D).float()
        v_sel = v_buf.reshape(-1)[flat_v].view(n, Hkv, K2, D).float()
        q_g = q[rows].float().view(n, Hkv, G, D)  # [n, Hkv, G, D]
        att = torch.einsum("nhgd,nhkd->nhgk", q_g, k_sel) * (D**-0.5)
        att = att.masked_fill(~valid.unsqueeze(2), float("-inf"))
        att = torch.softmax(att, dim=-1)
        o = torch.einsum("nhgk,nhkd->nhgd", att, v_sel)  # [n, Hkv, G, D]
        return o.view(n, H, D)

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
