"""Quest sparse attention backend（审稿 C3 修复：Quest e2e 对照臂）。

--attention-backend quest 启动。改造基底 = TLI backend（复用其 req_to_token/
pool 接线、forward_decode/forward_extend 流程、_sparse_attn 消费端），差异：

  - 索引：page=64 全维（D=128）min/max 上界，无量化（Quest 论文口径）；
    pool 只存 kmin/kmax（无 kq 4bit / far-near 分区 / D' 层跳 / PCA）
  - 选择：单级 page top-k（top-16 页 = 1024 token 预算，与 TLI@1024
    同预算），GQA group-mean 上界分，当前页强制入选；无 sink/swa
    （Quest 原版行为——baseline 诚实口径，论文 Limitations 记录）
  - attention 消费端：复用 TLI 的 fused gather+attn kernel
    （tli_sparse_gather_attn_dot）——同 kernel = 公平对照，两臂差异
    只在索引与选择；eager 回退路径与 TLI 逐位同构
  - CUDA graph：不支持（init_cuda_graph_state 显式报错 + veto_cuda_graph
    恒 True）。三臂 bench 脚本（单请求 / TP2 bs16）均 disable cuda graph，
    口径不受影响；kernel 级延迟已有独立 microbench
    （bench_quest_score.py + /tmp/bench_quest_select.cu，官方
    decode_select_k.cuh 原样编译，0.107ms@131K）

索引维护成本诚实计入：prefill 每 chunk 建索引（增量 O(n)，chunk 边界
min/max 结合律合并精确）+ decode 每步增量 O(1)，全部在 GPU 上同步完成、
计入 prefill/decode 墙钟（无后台预建）。
"""

from __future__ import annotations

import os
import time

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.quest.config import QuestProfile
from sglang.srt.layers.attention.quest.indexer import QuestIndexer, _MAX_INCREMENTAL_NEW
from sglang.srt.layers.attention.tli.kernels import tli_sparse_gather_attn_dot

_TIMING = os.environ.get("SGLANG_QUEST_TIMING", "0") == "1"


class _PhaseTimer:
    """SGLANG_QUEST_TIMING=1 时累计 decode 各阶段墙钟（归因用，含同步代价）。"""

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
                f"[Quest timing] steps={self.steps} total={total * 1000:.1f}ms "
                f"({total / self.steps * 1000:.2f}ms/step) {parts}"
            )
            for k in self.acc:
                self.acc[k] = 0.0


class QuestSparseAttnBackend(AttentionBackend):
    def __init__(self, runner=None) -> None:
        super().__init__()
        self.runner = runner
        self.profile = QuestProfile()
        self.indexers: dict[int, QuestIndexer] = {}
        # per-layer 共享 index pool：kmin/kmax [R, NBLK_CAP, Hkv, D]（KV dtype）
        self.index_pools: dict[int, dict] = {}
        self.token_to_kv_pool = None
        self.req_to_token = None
        self._num_layers = None
        self.head_dim = 128
        self.timer = _PhaseTimer()
        if runner is not None:
            self._init_from_runner(runner)

    def _init_from_runner(self, runner) -> None:
        model_config = runner.model_config
        self.head_dim = model_config.head_dim if model_config.head_dim else 128
        # TP 兼容（对齐 TLI #58）：取 per-rank kv head 数
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

    def _get_indexer(self, layer_id: int) -> QuestIndexer:
        if layer_id not in self.indexers:
            self.indexers[layer_id] = QuestIndexer(
                self.profile, head_dim=self.head_dim
            )
        return self.indexers[layer_id]

    # ---------------- 共享 index pool ---------------- #

    def _get_pool(self, layer_id: int) -> dict:
        if layer_id not in self.index_pools:
            p = self.profile
            Hkv = self.num_kv_heads
            dev = self.runner.device if self.runner else "cuda"
            s_cap = max(4096, p.dense_threshold + 1, p.pool_s_cap)
            nblk_cap = (s_cap + p.page_size - 1) // p.page_size
            r_cap = p.pool_rows
            # 界存储 fp32（S=64K/TP2 单行 ≈ 4MB，trivial；打分恒 fp32 精确）
            self.index_pools[layer_id] = {
                "kmin": torch.zeros(
                    r_cap, nblk_cap, Hkv, self.head_dim, device=dev
                ),
                "kmax": torch.zeros(
                    r_cap, nblk_cap, Hkv, self.head_dim, device=dev
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
        pg = self.profile.page_size
        new_cap = max(need + 2048, old * 2)
        old_nblk = (old + pg - 1) // pg
        new_nblk = (new_cap + pg - 1) // pg
        for key in ("kmin", "kmax"):
            t = pool_l[key]
            new = t.new_zeros((t.shape[0], new_nblk, *t.shape[2:]))
            new[:, :old_nblk] = t
            pool_l[key] = new
        pool_l["S_cap"] = new_cap

    def _grow_pool_r(self, pool_l: dict, add: int = 16) -> None:
        r0 = pool_l["R_cap"]
        r1 = r0 + add
        for key in ("kmin", "kmax"):
            t = pool_l[key]
            new = t.new_zeros((r1, *t.shape[1:]))
            new[:r0] = t
            pool_l[key] = new
        pool_l["free"].extend(range(r0, r1))
        pool_l["S"].extend([-1] * add)
        pool_l["R_cap"] = r1

    # ---------------- AttentionBackend 必须实现 ---------------- #

    def get_cuda_graph_seq_len_fill_value(self):
        # ForwardBatch 构建期无条件调用（base_runner），必须可返回
        return 1

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        raise NotImplementedError(
            "quest backend 不支持 CUDA graph（审稿 C3 对照臂为 eager 口径；"
            "请在启动参数 disable cuda graph，与三臂 bench 脚本一致）"
        )

    def veto_cuda_graph(self, forward_batch) -> bool:
        """恒 veto：quest 不提供图内路径（选择含 topk 动态宽度处理，
        图化收益不在本任务范围）。decode 恒走 eager。"""
        return True

    def init_forward_metadata(self, forward_batch):
        """eager 入口：行生命周期回收（请求退出 → pool 行归还）。"""
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
                t0 = self.timer.tick()
                out[i] = self._dense_attn(
                    q[i], req_to_token[req, :seq_len], kv_pool, layer_id
                )
                self.timer.add("dense", time.time() - t0)
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
                t0 = self.timer.tick()
                self._ensure_pool_s(pool_l, seq_len)
                k_all = k_buf[req_to_token[req, :seq_len]]
                idx_new = indexer.build_page_index(k_all)
                nblk = idx_new["nblk"]
                pool_l["kmin"][row, :nblk] = idx_new["kmin"]
                pool_l["kmax"][row, :nblk] = idx_new["kmax"]
                pool_l["S"][row] = seq_len
                self.timer.add("build", time.time() - t0)
            elif seq_len > S_st:
                # 增量：只取新 token（O(1)；契约=每行恰 1 新 token）
                if seq_len - S_st == 1:
                    inc_rows.append(row)
                    inc_S_old.append(S_st)
                    inc_reqs.append(req)
                elif S_st > 0:
                    t0 = self.timer.tick()
                    self._ensure_pool_s(pool_l, seq_len)
                    k_new = k_buf[req_to_token[req, S_st:seq_len]]
                    self._update_row_full(
                        pool_l, indexer, row, S_st, seq_len, k_new
                    )
                    self.timer.add("increment", time.time() - t0)
                else:
                    # S_st == 0 兜底：直接全量重建（_update_row_full 无旧尾页）
                    t0 = self.timer.tick()
                    self._ensure_pool_s(pool_l, seq_len)
                    k_all = k_buf[req_to_token[req, :seq_len]]
                    idx_new = indexer.build_page_index(k_all)
                    nblk = idx_new["nblk"]
                    pool_l["kmin"][row, :nblk] = idx_new["kmin"]
                    pool_l["kmax"][row, :nblk] = idx_new["kmax"]
                    self.timer.add("build", time.time() - t0)
                pool_l["S"][row] = seq_len
            sparse_rows.append((i, row, seq_len))
        if inc_rows:
            # 批量增量维护（单次 flat scatter，launch 数与 bs 无关）
            t0 = self.timer.tick()
            self._ensure_pool_s(pool_l, max(lens_l))
            reqs_t = torch.tensor(inc_reqs, device=q.device)
            S_old_t = torch.tensor(inc_S_old, device=q.device)
            slots = req_to_token[reqs_t, S_old_t]  # [n]（新 token 槽位）
            k_new_b = k_buf[slots]  # [n, Hkv, D]
            indexer.update_pool_rows_decode(
                pool_l, torch.tensor(inc_rows, device=q.device), inc_S_old, k_new_b
            )
            self.timer.add("increment", time.time() - t0)
        if sparse_rows:
            t0 = self.timer.tick()
            rows_t = torch.tensor([r for _, r, _ in sparse_rows], device=q.device)
            S_list = [S for _, _, S in sparse_rows]
            idx_t = torch.tensor(
                [i for i, _, _ in sparse_rows], device=q.device
            )
            sel = indexer.select_decode_batched(
                pool_l, rows_t, S_list, q[idx_t]
            )  # [n, Hkv, K]（无效 lane = per-row 哨兵 S_i）
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
            out[rows] = out_sparse.to(out.dtype)
            self.timer.add("sparse_attn", time.time() - t0)
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
            t0 = self.timer.tick()
            if S_st == prefix and prefix > 0:
                # chunked prefill 增量：前缀页已建，追加本 chunk 新 token
                # （页边界 min/max 结合律合并 = 精确，O(nq)）
                k_new = k_buf[locs[prefix:S]]
                self._update_row_full(pool_l, indexer, row, prefix, S, k_new)
            else:
                # 首 chunk / 行复用 / 跳变：全量重建（成本计入 prefill）
                k_all = k_buf[locs]
                idx_new = indexer.build_page_index(k_all)
                nblk = idx_new["nblk"]
                pool_l["kmin"][row, :nblk] = idx_new["kmin"]
                pool_l["kmax"][row, :nblk] = idx_new["kmax"]
            pool_l["S"][row] = S
            self.timer.add("build", time.time() - t0)
            nblk = (S + self.profile.page_size - 1) // self.profile.page_size
            t_arr = torch.arange(prefix, S, device=q.device)
            t0 = self.timer.tick()
            sel = indexer.select_extend(
                pool_l["kmin"][row, :nblk],
                pool_l["kmax"][row, :nblk],
                nblk,
                q_b,
                t_arr,
                S,
            )  # [nq, Hkv, K]（哨兵 S；含 query 级因果掩蔽）
            self.timer.add("select", time.time() - t0)
            t0 = self.timer.tick()
            out[starts[b] : ends[b]] = self._sparse_extend_one(
                q_b, sel, locs, pool, layer_id, Hkv, G,
                q_raw=q[starts[b] : ends[b]],
            ).to(q.dtype)
            self.timer.add("sparse_attn", time.time() - t0)
        # 返回约定：[T, H*D]
        return out

    # ---------------- 内部工具 ---------------- #

    def _update_row_full(
        self, pool_l: dict, indexer: QuestIndexer, row: int,
        S_old: int, S: int, k_new: torch.Tensor,
    ) -> None:
        """多 token 增量（chunked prefill / decode 跳变兜底）：
        首尾页之外的整页段直接重建 min/max，边界页结合律合并。
        k_new: [n, Hkv, D]（全局位置 S_old..S）。"""
        pg = self.profile.page_size
        n = S - S_old
        nblk_old = (S_old + pg - 1) // pg
        # 完整新页段（起点对齐页边界）
        seg_start = (S_old + pg - 1) // pg * pg  # 首个完整新页起点
        if seg_start < S:
            seg = k_new[seg_start - S_old :]  # [S - seg_start, Hkv, D]
            nb = (S - seg_start) // pg
            if nb:
                kseg = seg[: nb * pg].reshape(nb, pg, *seg.shape[1:])
                w = seg_start // pg
                pool_l["kmin"][row, w : w + nb] = kseg.amin(1)
                pool_l["kmax"][row, w : w + nb] = kseg.amax(1)
        # 前段：并入旧尾页（S_old 非页对齐时）
        head = min(n, seg_start - S_old)
        if head > 0:
            kh = k_new[:head]
            pool_l["kmin"][row, nblk_old - 1] = torch.minimum(
                pool_l["kmin"][row, nblk_old - 1], kh.amin(0)
            )
            pool_l["kmax"][row, nblk_old - 1] = torch.maximum(
                pool_l["kmax"][row, nblk_old - 1], kh.amax(0)
            )
        # 尾段：不足一页的尾（S 非页对齐）
        tail_start = seg_start + ((S - seg_start) // pg) * pg
        if tail_start < S:
            kt = k_new[tail_start - S_old :]
            w = tail_start // pg
            pool_l["kmin"][row, w] = kt.amin(0)
            pool_l["kmax"][row, w] = kt.amax(0)

    def _save_kv_cache(self, k, v, layer, forward_batch):
        self.token_to_kv_pool.set_kv_buffer(
            layer, forward_batch.out_cache_loc, k, v
        )

    def _dense_attn(self, q_i, locs, pool, layer_id):
        """decode 短序列 dense（GQA）。locs: [S] pool 槽位。"""
        k_all, v_all = pool.get_kv_buffer(layer_id)
        k_all = k_all[locs].float()
        v_all = v_all[locs].float()
        q_i = q_i.float()
        H = q_i.shape[0]
        Hkv = k_all.shape[1]
        G = H // Hkv
        q_g = q_i.reshape(Hkv, G, -1)
        k_e = k_all.transpose(0, 1)
        v_e = v_all.transpose(0, 1)
        att = torch.einsum("hgd,hsd->hgs", q_g, k_e) * (self.head_dim**-0.5)
        att = torch.softmax(att, dim=-1)
        o = torch.einsum("hgs,hsd->hgd", att, v_e)
        return o.reshape(H, -1)

    def _sparse_attn_batched(
        self, q, row_idx, sel, seq_lens, forward_batch, req_to_token, pool, layer_id, Hkv, G
    ):
        """decode 批量稀疏前向（TLI 同构：一次 gather + 批量 einsum /
        fused kernel；哨兵 = per-row S_i，valid = sel < S_i 掩 softmax）。"""
        n = sel.shape[0]
        H, D = q.shape[1], self.head_dim
        if torch.is_tensor(row_idx):
            rows = row_idx.to(torch.long)
        else:
            rows = torch.tensor(row_idx, device=q.device)
        if torch.is_tensor(seq_lens):
            seq_lens_t = seq_lens.to(torch.long)
        else:
            seq_lens_t = torch.tensor(seq_lens, device=q.device, dtype=torch.long)
        seq_v = seq_lens_t.view(-1, 1, 1)
        valid = sel < seq_v  # [n, Hkv, K]
        sel_c = torch.minimum(sel, seq_v - 1)
        reqs = forward_batch.req_pool_indices[rows]  # [n]
        locs_full = req_to_token[reqs]  # [n, max_S]
        pool_pos = torch.gather(
            locs_full, 1, sel_c.view(n, -1)
        ).view(n, Hkv, sel.shape[-1])  # [n, Hkv, K] pool 槽位
        K2 = pool_pos.shape[-1]
        k_buf, v_buf = pool.get_kv_buffer(layer_id)
        # fused kernel 路径（与 TLI decode 同款 kernel = 公平对照）
        q_raw = q[rows]
        if (
            self.profile.use_sparse_attn_kernel
            and q_raw.dtype in (torch.bfloat16, torch.float16)
            and q_raw.is_contiguous()
            and (G & (G - 1)) == 0
            and (D & (D - 1)) == 0
        ):
            return tli_sparse_gather_attn_dot(
                q_raw, pool_pos, k_buf, v_buf, G,
                S_loc=k_buf.shape[0], valid=valid,
            )
        # eager 回退（与 TLI 数值同构：fp32 累加顺序一致）
        d_off = torch.arange(D, device=q.device)
        h_off = torch.arange(Hkv, device=q.device).view(1, Hkv, 1, 1) * D
        flat = (
            pool_pos.unsqueeze(-1) * (Hkv * D) + h_off + d_off.view(1, 1, 1, D)
        )
        flat_v = flat.view(n, -1)
        k_sel = k_buf.reshape(-1)[flat_v].view(n, Hkv, K2, D).float()
        v_sel = v_buf.reshape(-1)[flat_v].view(n, Hkv, K2, D).float()
        q_g = q[rows].float().view(n, Hkv, G, D)
        att = torch.einsum("nhgd,nhkd->nhgk", q_g, k_sel) * (D**-0.5)
        att = att.masked_fill(~valid.unsqueeze(2), float("-inf"))
        att = torch.softmax(att, dim=-1)
        o = torch.einsum("nhgk,nhkd->nhgd", att, v_sel)
        return o.view(n, H, D)

    def _dense_extend_one(self, q_b, locs, pool, layer_id, Hkv, G):
        """单个请求的 dense causal attention（短序列 prefill / chunk）。"""
        k_all, v_all = pool.get_kv_buffer(layer_id)
        k_e = k_all[locs].float().transpose(0, 1)
        v_e = v_all[locs].float().transpose(0, 1)
        nq = q_b.shape[0]
        S = locs.shape[0]
        q_g = q_b.reshape(nq, Hkv, G, self.head_dim)
        att = torch.einsum("ahgd,hsd->ahgs", q_g, k_e) * (self.head_dim**-0.5)
        qpos = torch.arange(S - nq, S, device=q_b.device).view(nq, 1)
        causal = torch.arange(S, device=q_b.device).view(1, S) <= qpos
        att = att.masked_fill(~causal.view(nq, 1, 1, S), float("-inf"))
        att = torch.softmax(att, dim=-1)
        o = torch.einsum("ahgs,hsd->ahgd", att, v_e)
        return o.reshape(nq, -1)

    def _sparse_extend_one(self, q_b, sel, locs, pool, layer_id, Hkv, G, q_raw=None):
        """单个请求的稀疏 prefill。

        sel: [nq, Hkv, K] 逻辑位置（哨兵 S / 因果未来位 —— valid 掩掉）。
        fused kernel 路径：把 <S 的 valid 折进 valid 掩码传 kernel
        （per-lane 掩 softmax，与 eager 语义一致）；eager 路径同款掩码。
        """
        k_buf, v_buf = pool.get_kv_buffer(layer_id)
        nq, H = q_b.shape[0], q_b.shape[1]
        S = locs.shape[0]
        valid = sel < S  # [nq, Hkv, K]（哨兵 + 未来位置统一无效）
        sel_c = sel.clamp(max=S - 1)
        pool_sel = locs[sel_c]  # [nq, Hkv, K] pool 槽位
        if (
            q_raw is not None
            and self.profile.use_sparse_attn_kernel
            and q_raw.dtype in (torch.bfloat16, torch.float16)
            and (G & (G - 1)) == 0
            and (self.head_dim & (self.head_dim - 1)) == 0
        ):
            # q_raw 是父张量切片 view，行 stride 可能 ≠ H*D（TLI 踩坑记录）
            q_c = q_raw if q_raw.is_contiguous() else q_raw.contiguous()
            out_k = tli_sparse_gather_attn_dot(
                q_c, pool_sel, k_buf, v_buf, G,
                S_loc=k_buf.shape[0], valid=valid,
            )
            return out_k.view(nq, H * self.head_dim)
        K2 = sel.shape[-1]
        out = torch.empty(nq, H, self.head_dim, device=q_b.device, dtype=q_b.dtype)
        row_chunk = max(1, min(nq, 512))
        for h in range(Hkv):
            q_h = q_b[:, h * G : (h + 1) * G]
            for r0 in range(0, nq, row_chunk):
                r1 = min(r0 + row_chunk, nq)
                pp = pool_sel[r0:r1, h]
                vd = valid[r0:r1, h]
                k_sel = k_buf[pp, h].float()
                v_sel = v_buf[pp, h].float()
                att = torch.einsum("ngd,nkd->ngk", q_h[r0:r1], k_sel) * (
                    self.head_dim**-0.5
                )
                att = att.masked_fill(~vd.unsqueeze(1), float("-inf"))
                att = torch.softmax(att, dim=-1)
                out[r0:r1, h * G : (h + 1) * G] = torch.einsum(
                    "ngk,nkd->ngd", att, v_sel
                ).to(q_b.dtype)
        return out.view(nq, H * self.head_dim)


__all__ = ["QuestSparseAttnBackend"]
