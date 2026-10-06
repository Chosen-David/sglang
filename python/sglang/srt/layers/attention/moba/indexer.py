"""MoBA training-free 索引器：chunk-mean gate 选块全展开（E89 复现臂）。

算法口径 = two-level-attention/sparse_attn/indexer/tli_indexer.py 的
moba_gate 路径（E89，LongBench 13 任务 49.19）：
  - 索引：每 chunk=64 token 建全维（D=128）key **和**（fp32），
    chunk-mean = ksum / chunk_size——E89 零 pad 后 mean(dim) 的精确等价
    （零填充不改和；除以 64 是 2 的幂，fp32 除法无舍入）
  - gate：score = (q·scale) · chunk-mean，先 per-q-head 点积再 GQA 组内
    mean（kv-head 级），与 E89 score_moba 的算子序列同构
  - 选择：全部块（含 sink/swa/当前块）参与 top-K 排名，K = K2/chunk；
    选中块内全部 token 展开；sink（前 128 tok）+ swa（尾 128 tok）保送

sglang paged 适配（quest 同款共享 pool + 批量张量化）：
  - pool 存 ksum [R, Hkv, NBLK_CAP, D] fp32（和可增量合并：逐 token
    加法 / 对齐开新块；layout 取 [R,Hkv,NC,D] 使 gate einsum 免转置拷贝）
  - decode 增量 O(1)/token；chunked prefill 多 token 增量走 backend 的
    _update_row_full（头/中/尾三段，加法合并）
  - select 输出 [n, Hkv, K] token 逻辑位置（K = nb·bs + sink + swa 静态
    宽），无效 lane = per-row 哨兵 S_i（TLI/quest 约定，下游 valid 掩掉）

与 E89 的已知实现级偏差（如实记录）：
  1. E89 每 decode 步全量重建 chunk-mean（transformers monkeypatch 无增量
     索引），本实现增量维护块和——fp32 加法结合顺序不同，块和有 ulp 级
     漂移（topk 顺序理论上可翻转，随机数据下概率 ~0；单测对拍覆盖）
  2. E89 的 chunk-mean 在 bf16 上算（transformers bf16 前向）再 cast fp32；
     本实现 ksum 全程 fp32（gate 精度更高口径）。fp32 输入下单测对拍逐位
  3. E89 transformers 臂 prefill 走 dense；本 sglang 臂与 harness 其他稀疏
     臂（tli/quest）同口径：S > dense_threshold 时 prefill 也走稀疏选择
     （每 query 独立因果排名），保证三臂 e2e 延迟对比公平
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.moba.config import MoBAProfile

# decode 增量维护的单步上限（对齐 tli/quest backend 口径；超过则全量重建）
_MAX_INCREMENTAL_NEW = 512


class MoBAIndexer:
    """MoBA chunk-mean gate 索引（PyTorch 批量化实现，pool 行间接寻址）。"""

    def __init__(
        self,
        profile: MoBAProfile | None = None,
        head_dim: int = 128,
    ) -> None:
        self.profile = profile or MoBAProfile()
        self.head_dim = head_dim

    # ------------------------------------------------------------------ #
    # 索引维护（prefill 全量 / decode 增量）
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def build_chunk_index(self, k: torch.Tensor) -> dict:
        """k: [S, Hkv, D]（KV cache dtype，RoPE 后）→ 每 chunk 的 key 和。

        返回 {"ksum": [Hkv, nblk, D] fp32, "nblk", "S"}。
        尾块零 pad：零不改块和，ksum/chunk_size 恒等于 E89 零 pad mean。
        （存储口径对照：fp32 全维块和 = D·4B / chunk / kv-head =
        8B/token/kv-head，摊销后与 Quest fp32 界同量级）
        """
        p = self.profile
        bs = p.chunk_size
        S, Hkv, D = k.shape
        nblk = (S + bs - 1) // bs
        pad = nblk * bs - S
        kf = k.float()
        if pad:
            kf = F.pad(kf, (0, 0, 0, 0, 0, pad))
        ksum = kf.reshape(nblk, bs, Hkv, D).sum(1)  # [nblk, Hkv, D]
        # pool layout [R, Hkv, NC, D]：转成 [Hkv, nblk, D] 直接整片写入
        return {"ksum": ksum.permute(1, 0, 2).contiguous(), "nblk": nblk, "S": S}

    @torch.no_grad()
    def update_pool_rows_decode(
        self,
        pool_l: dict,
        rows: torch.Tensor,
        S_old,
        k_new: torch.Tensor,
    ) -> None:
        """decode 批量增量维护（**每行恰追加 1 个新 token**，decode 常态）。

        块和语义：S_old 对齐 chunk 边界 → 开新块（ksum = 新 token）；
        非对齐 → 并入旧尾块（ksum += 新 token）。flat scatter 一次完成
        （对齐 tli/quest update_pool_rows_decode 的口径）。
        pool_l["ksum"]: [R, Hkv, NC, D] fp32；k_new: [n, Hkv, D]。
        前提：pool 容量足够（backend 先 _ensure_pool_s）；调用方更新
        pool_l["S"][row]。
        """
        p = self.profile
        bs = p.chunk_size
        device = k_new.device
        k_new = k_new.float()
        if torch.is_tensor(S_old):
            S_old_t = S_old.to(torch.long)
            n = S_old_t.shape[0]
        else:
            n = len(S_old)
            S_old_t = torch.tensor(S_old, device=device, dtype=torch.long)
        if n == 0:
            return
        assert k_new.shape[0] == n, "moba 增量维护要求每行恰 1 个新 token"
        Hkv, D = k_new.shape[1], k_new.shape[2]
        nc = pool_l["ksum"].shape[2]
        rows_l = rows.to(torch.long)

        # 目标块：对齐 → 新块 nblk_old；非对齐 → 旧尾块 nblk_old-1
        aligned = S_old_t % bs == 0  # [n]
        nblk_old = (S_old_t + bs - 1) // bs
        blk = torch.where(aligned, nblk_old, nblk_old - 1)  # [n]
        # flat 偏移（layout [R, Hkv, NC, D]）
        off = (
            rows_l.view(n, 1, 1, 1) * (Hkv * nc * D)
            + torch.arange(Hkv, device=device).view(1, Hkv, 1, 1) * (nc * D)
            + blk.view(n, 1, 1, 1) * D
            + torch.arange(D, device=device).view(1, 1, 1, D)
        )
        off_f = off.reshape(-1)
        old = pool_l["ksum"].reshape(-1)[off_f].view(n, Hkv, D)
        new = torch.where(aligned.view(n, 1, 1), k_new, old + k_new)
        pool_l["ksum"].reshape(-1)[off_f] = new.reshape(-1)

    # ------------------------------------------------------------------ #
    # gate 打分（E89 score_moba 同构）
    # ------------------------------------------------------------------ #

    def _gate_scores(self, q: torch.Tensor, kmean: torch.Tensor) -> torch.Tensor:
        """E89 口径 gate 分：全维 (q·scale)·chunk-mean，GQA 组内 mean。

        q: [a, H, D]（任意 dtype，RoPE 后，head 布局 kv-head-major：H=h·G+g）；
        kmean: [a, Hkv, m, D] fp32。返回 [a, Hkv, m] fp32。
        算子序列对齐 E89 tli_indexer.py L356-367：
          b_q_full_gate = (q * softmax_scale).float()          # 先乘 scale 再 cast
          score_moba = einsum("h d, kt h d -> h kt")            # M=1 GEMV 路径
          sm = 组内 (h g) → h 的 mean
        逐位一致性关键：本实现的 matmul 广播形状 [a,Hkv,G,1,D] @
        [a,Hkv,1,D,m] 让每个 (h,g) 片走 M=1 GEMV（与 E89 einsum 同
        dispatch 路径，d 维归约顺序相同）；若把 g 并进矩阵 M 维
        （[G,D]x[D,m] GEMM）则归约顺序不同，产生 ~1e-7 ulp 漂移。
        kmean.transpose 免物化（stride 视图）。scale 只影响分数量级
        不影响 topk 顺序，保留是为对拍口径一致。
        """
        a, H, D = q.shape
        Hkv = kmean.shape[1]
        G = H // Hkv
        qg = (q * (D**-0.5)).to(torch.float32).view(a, Hkv, G, D)
        k_t = kmean.transpose(-1, -2)  # [a, Hkv, D, m]（视图）
        score_g = torch.matmul(qg.unsqueeze(3), k_t.unsqueeze(2))  # [a,Hkv,G,1,m]
        return score_g.squeeze(3).mean(dim=2)  # 组内 mean → [a, Hkv, m]

    # ------------------------------------------------------------------ #
    # decode 批量选择（共享 pool 行）
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def select_decode_batched(
        self,
        pool_l: dict,
        rows: torch.Tensor,
        S_list,
        q: torch.Tensor,
    ) -> torch.Tensor:
        """decode 批量 chunk 选择（E89 _moba_mask 的 sel 化移植）。

        pool_l: 共享 pool；rows: [n] 行号；S_list: 每行有效长度（int 列表
        或 device tensor，含当前 token）；q: [n, H, D]（t = S-1）。
        返回 [n, Hkv, K] token 逻辑位置，K = nb·bs + sink + swa（静态宽）。
        无效 lane = per-row 哨兵 S_i（TLI/quest 约定）。

        语义（对齐 E89 _moba_mask，L414-446）：
          - 全部块（块起点 < S，含 sink/swa/当前块）按 gate 分 top-K；
          - 选中块整块展开（pos < S 的 lane 有效）；
          - sink [0, sink_tok) 与 swa [max(0,S-swa), S) 保送（E89 mask
            直接置位语义）。E89 返回 bool mask（OR 天然去重）；本 sel 化
            版本用「top-K lane 剔除 forced 区 token + forced 区独立 lane」
            实现同一集合（三组 lane 两两不交，softmax 无重复计数）：
              top-K lane: sink ≤ pos < S-swa 且 pos < S
              sink lane:  [0, min(sink, S))
              swa lane:   [max(S-swa, sink), S)
            三者并集 = E89 mask 的 True 集合（集合等价性单测覆盖）。
        """
        p = self.profile
        bs, nb = p.chunk_size, p.select_blocks
        sink_t, swa_t = p.sink_tokens, p.sliding_window
        device = q.device
        n, H, D = q.shape
        ksum_pool = pool_l["ksum"]
        Hkv, nblk_cap = ksum_pool.shape[1], ksum_pool.shape[2]
        if torch.is_tensor(S_list):
            S_t = S_list.to(torch.long)
        else:
            S_t = torch.tensor(S_list, device=device, dtype=torch.long)
        S_v = S_t.view(-1, 1, 1)  # [n,1,1] 哨兵/界

        # gate 分数（chunk-mean = ksum / bs，2 的幂除法无舍入）
        kmean = ksum_pool[rows.to(torch.long)].float() / bs  # [n, Hkv, NC, D]
        score = self._gate_scores(q, kmean)  # [n, Hkv, NC]
        # 可排名块：块起点 < S（= E89 kt=ceil(S/bs) 块全集）
        m_idx = torch.arange(nblk_cap, device=device)
        rankable = m_idx.view(1, 1, -1) * bs < S_t.view(-1, 1, 1)
        score = score.masked_fill(~rankable, float("-inf"))
        # top-K（K > 有效块数时 -inf lane 由哨兵兜住，集合与 E89 的
        # min(nb_sel, kt) 截断一致）
        k = min(nb, nblk_cap)
        vals, blk = torch.topk(score, k, dim=-1)  # [n, Hkv, k]
        lane_ok = torch.isfinite(vals)

        # ---- top-K lane：块展开 + 因果界 + 剔除 forced 区（去重） ----
        off = torch.arange(bs, device=device)
        pos = blk.unsqueeze(-1) * bs + off.view(1, 1, 1, bs)  # [n,Hkv,k,bs]
        swa_lo = torch.clamp(S_t - swa_t, min=0)  # [n]
        in_forced = (pos < sink_t) | (pos >= swa_lo.view(-1, 1, 1, 1))
        ok = (
            lane_ok.unsqueeze(-1)
            & (pos < S_v.unsqueeze(-1))
            & ~in_forced
        )
        sel_tk = torch.where(ok, pos, S_v.unsqueeze(-1).expand_as(pos))
        sel_tk = sel_tk.reshape(n, Hkv, k * bs)
        # K 截断时 pad 回静态宽（哨兵）
        if k < nb:
            sel_tk = torch.cat(
                [sel_tk, S_v.expand(n, Hkv, (nb - k) * bs)], dim=-1
            )

        # ---- forced lane（保送区，与 top-K lane 两两不交） ----
        # sink lane：[0, sink_tokens)（越 S 无效）
        sink_pos = torch.arange(sink_t, device=device).view(1, 1, -1)
        sel_sink = torch.where(
            sink_pos < S_v, sink_pos, S_v
        ).expand(n, Hkv, sink_t)
        # swa lane：起点 max(S-swa, sink)（防与 sink lane 重叠），宽 swa_t
        swa_start = torch.clamp(S_t - swa_t, min=sink_t)  # [n]
        swa_pos = (
            swa_start.view(-1, 1, 1)
            + torch.arange(swa_t, device=device).view(1, 1, -1)
        )  # [n,1,swa_t]
        sel_swa = torch.where(swa_pos < S_v, swa_pos, S_v).expand(n, Hkv, swa_t)

        return torch.cat([sel_tk, sel_sink, sel_swa], dim=-1)  # [n, Hkv, K]

    # ------------------------------------------------------------------ #
    # extend（prefill）选择：单请求 chunk 批量化（query 级因果）
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def select_extend(
        self,
        ksum: torch.Tensor,
        nblk: int,
        q: torch.Tensor,
        t_arr: torch.Tensor,
        S: int,
        row_chunk: int = 512,
    ) -> torch.Tensor:
        """prefill 单请求选择（每 query 行独立因果块排名）。

        ksum: [Hkv, nblk, D] fp32（pool 行切片或 build 输出）；q: [nq, H, D]；
        t_arr: [nq] query 全局位置（= prefix..S-1）；S: 请求已写池总长。
        返回 [nq, Hkv, K] token 位置（哨兵 S；因果由 select 内部掩掉，
        下游 valid = sel < S 只处理哨兵）。

        prefill 口径（与 harness 其他稀疏臂一致；E89 transformers 臂
        prefill 走 dense，见模块 docstring 偏差声明 3）：
          - 每 query t：可排名块 = 块起点 ≤ t（当前块也参与——与 decode
            排名口径一致），块内未来 token 由 lane 级因果（pos ≤ t）掩掉；
          - forced 区相对 query t：sink [0, sink) ∩ [0, t] +
            swa [max(0, t+1-swa), t]；
          - 三组 lane（top-K 剔 forced / sink / swa）两两不交，同 decode。
        """
        p = self.profile
        bs, nb = p.chunk_size, p.select_blocks
        sink_t, swa_t = p.sink_tokens, p.sliding_window
        device = q.device
        nq, H, D = q.shape
        Hkv = ksum.shape[0]
        kmean = ksum.float() / bs  # [Hkv, nblk, D]
        m_idx = torch.arange(nblk, device=device)
        off = torch.arange(bs, device=device)
        sels = []
        for r0 in range(0, nq, row_chunk):
            r1 = min(r0 + row_chunk, nq)
            rc = r1 - r0
            score = self._gate_scores(
                q[r0:r1], kmean.unsqueeze(0).expand(rc, *kmean.shape)
            )  # [rc, Hkv, nblk]
            t_c = t_arr[r0:r1].to(torch.long)  # [rc]
            # 可排名块：块起点 ≤ t
            rankable = m_idx.view(1, 1, -1) * bs <= t_c.view(-1, 1, 1)
            score = score.masked_fill(~rankable, float("-inf"))
            k = min(nb, nblk)
            vals, blk = torch.topk(score, k, dim=-1)  # [rc, Hkv, k]
            lane_ok = torch.isfinite(vals)
            # top-K lane：块展开 + 因果（pos ≤ t）+ 剔除 forced 区
            pos = blk.unsqueeze(-1) * bs + off.view(1, 1, 1, bs)  # [rc,Hkv,k,bs]
            swa_lo_t = torch.clamp(t_c + 1 - swa_t, min=0)  # [rc]
            in_forced = (pos < sink_t) | (pos >= swa_lo_t.view(-1, 1, 1, 1))
            ok = (
                lane_ok.unsqueeze(-1)
                & (pos <= t_c.view(-1, 1, 1, 1))
                & ~in_forced
            )
            sel_tk = torch.where(ok, pos, torch.full_like(pos, S))
            sel_tk = sel_tk.reshape(rc, Hkv, k * bs)
            if k < nb:
                sel_tk = torch.cat(
                    [
                        sel_tk,
                        torch.full(
                            (rc, Hkv, (nb - k) * bs), S,
                            dtype=sel_tk.dtype, device=device,
                        ),
                    ],
                    dim=-1,
                )
            # sink lane：[0, sink) 内 pos ≤ t 有效
            sink_pos = torch.arange(sink_t, device=device).view(1, 1, -1)
            sink_ok = (sink_pos <= t_c.view(-1, 1, 1)) & (sink_pos < S)
            sel_sink = torch.where(
                sink_ok, sink_pos, torch.full_like(sink_pos, S)
            ).expand(rc, Hkv, sink_t)
            # swa lane：起点 max(t+1-swa, sink)，宽 swa_t，pos ≤ t 有效
            swa_start = torch.clamp(t_c + 1 - swa_t, min=sink_t)  # [rc]
            swa_pos = (
                swa_start.view(-1, 1, 1)
                + torch.arange(swa_t, device=device).view(1, 1, -1)
            )
            sel_swa = torch.where(
                swa_pos <= t_c.view(-1, 1, 1),
                swa_pos,
                torch.full_like(swa_pos, S),
            ).expand(rc, Hkv, swa_t)
            sels.append(
                torch.cat([sel_tk, sel_sink, sel_swa], dim=-1)
            )
        return torch.cat(sels, dim=0)  # [nq, Hkv, nb*bs + sink + swa]


__all__ = ["MoBAIndexer", "_MAX_INCREMENTAL_NEW"]
