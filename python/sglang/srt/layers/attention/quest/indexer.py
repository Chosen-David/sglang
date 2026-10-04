"""Quest 索引器：page 级全维 min/max 上界选择（审稿 C3 对照臂）。

算法口径 = two-level-attention/sparse_attn/indexer/quest_indexer.py：
  - prepare_index: k 重排 [b, t, bs, h, d] → page min/max（全维，无量化）
  - compute_score: score = sum_d max(q·k_min, q·k_max)，GQA group-mean
  - compute_mask: top-(k-1) page + 当前 page 强制入选（quest 原版语义）

本实现是批量张量化版本（sglang paged 寻址 + 共享 pool），关键恒等式：

    sum_d max(q_d·kmin_d, q_d·kmax_d)
      = sum_d [ q_d≥0 ? q_d·kmax_d : q_d·kmin_d ]
      = q·kmax + sum_d relu(-q_d)·(kmax_d - kmin_d)

两个标准收缩（einsum/bmm，不物化 [.., m, D] 逐维中间张量——直接算
逐维 max 的参考式要 [n, Hkv, G, m, D] fp32，131K 时 100+GB 不可行）。
数值：恒等式在精确算术下成立，fp32 舍入与参考式同量级（只用于排序）。
"""

from __future__ import annotations

import torch

from sglang.srt.layers.attention.quest.config import QuestProfile

# decode 增量维护的单步上限（对齐 TLI backend 口径；超过则全量重建）
_MAX_INCREMENTAL_NEW = 512


class QuestIndexer:
    """Quest page 级索引（PyTorch 批量化实现，pool 行间接寻址）。"""

    def __init__(
        self,
        profile: QuestProfile | None = None,
        head_dim: int = 128,
    ) -> None:
        self.profile = profile or QuestProfile()
        self.head_dim = head_dim

    # ------------------------------------------------------------------ #
    # 索引维护（prefill 全量 / decode 增量）
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def build_page_index(self, k: torch.Tensor) -> dict:
        """k: [S, Hkv, D]（KV cache dtype，RoPE 后）→ page min/max。

        返回 {"kmin": [nblk, Hkv, D], "kmax": ..., "nblk", "S"}。
        存储用 k.dtype（bound 是 KV 值的 min/max，同 dtype 表示精确无损；
        Quest 论文口径 = 无量化全维 min/max）。
        尾块精确界：不足一 page 的尾块只对真实 token 取界（零 pad 会把 0
        混进界，增量 merge 无法恢复——与 TLI build 同款修正）。
        """
        p = self.profile
        S, Hkv, D = k.shape
        nblk = (S + p.page_size - 1) // p.page_size
        pad = nblk * p.page_size - S
        if pad:
            k = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, pad))
        kp = k.reshape(nblk, p.page_size, Hkv, D)
        kmin = kp.amin(1)  # [nblk, Hkv, D]
        kmax = kp.amax(1)
        if pad:
            valid_tail = S - (nblk - 1) * p.page_size
            kmin = kmin.clone()
            kmax = kmax.clone()
            kmin[-1] = kp[-1, :valid_tail].amin(0)
            kmax[-1] = kp[-1, :valid_tail].amax(0)
        return {"kmin": kmin, "kmax": kmax, "nblk": nblk, "S": S}

    @torch.no_grad()
    def update_pool_rows_decode(
        self,
        pool_l: dict,
        rows: torch.Tensor,
        S_old,
        k_new: torch.Tensor,
    ) -> None:
        """decode 批量增量维护（**每行恰追加 1 个新 token**，decode 常态）。

        多 token 跳变（chunked/spec/行复用导致 S 倒退或跳变）由 backend
        走全量重建路径，不进本函数（与 TLI 批量增量同契约）。
        pool_l: backend 共享 pool（kmin/kmax [R, NBLK_CAP, Hkv, D]）；
        rows: [n] 行号；S_old: 每行旧有效长度（int 列表或 device tensor）；
        k_new: [n, Hkv, D]（KV dtype）。
        语义 = 逐行 page min/max 结合律合并 / 对齐开新页，flat scatter
        一次完成（对齐 TLI update_pool_rows_decode 的口径，去掉 kq 维）。
        前提：pool 容量足够（backend 先 _ensure_pool_s）；调用方更新
        pool_l["S"][row]。
        """
        p = self.profile
        pg = p.page_size
        device = k_new.device
        if torch.is_tensor(S_old):
            S_old_t = S_old.to(torch.long)
            n = S_old_t.shape[0]
        else:
            n = len(S_old)
            S_old_t = torch.tensor(S_old, device=device, dtype=torch.long)
        if n == 0:
            return
        assert k_new.shape[0] == n, "quest 增量维护要求每行恰 1 个新 token"
        Hkv, D = k_new.shape[1], k_new.shape[2]
        nblk_cap = pool_l["kmin"].shape[1]
        rows_l = rows.to(torch.long)

        # ---- page min/max：对齐行开新页（min=max=新 token），非对齐行
        # 与旧尾页 min/max 合并（结合律；全维 D 版，无子空间截取）----
        aligned = S_old_t % pg == 0  # [n]
        nblk_old = (S_old_t + pg - 1) // pg
        blk_idx = torch.where(aligned, nblk_old, nblk_old - 1)  # 目标页
        base = rows_l * (nblk_cap * Hkv * D) + blk_idx * (Hkv * D)
        off = (
            base.view(n, 1, 1)
            + torch.arange(Hkv, device=device).view(1, Hkv, 1) * D
            + torch.arange(D, device=device).view(1, 1, D)
        )
        off_f = off.reshape(-1)
        old_mn = pool_l["kmin"].reshape(-1)[off_f].view(n, Hkv, D)
        old_mx = pool_l["kmax"].reshape(-1)[off_f].view(n, Hkv, D)
        mn_val = torch.where(aligned.view(n, 1, 1), k_new, torch.minimum(old_mn, k_new))
        mx_val = torch.where(aligned.view(n, 1, 1), k_new, torch.maximum(old_mx, k_new))
        pool_l["kmin"].reshape(-1)[off_f] = mn_val.reshape(-1)
        pool_l["kmax"].reshape(-1)[off_f] = mx_val.reshape(-1)

    # ------------------------------------------------------------------ #
    # 上界打分（恒等式版，见模块 docstring）
    # ------------------------------------------------------------------ #

    def _upper_bound(
        self,
        qg: torch.Tensor,
        kmin_f: torch.Tensor,
        kmax_f: torch.Tensor,
        batched_rows: bool,
    ) -> torch.Tensor:
        """qg: [a, Hkv, G, D] fp32；kmin_f/kmax_f: [a, m, Hkv, D]（batched_rows）
        或 [m, Hkv, D]（单请求 prefill）。返回 [a, Hkv, G, m] 上界分。"""
        if batched_rows:
            spec = "ahgd,amhd->ahgm"
        else:
            spec = "ahgd,mhd->ahgm"
        s_max = torch.einsum(spec, qg, kmax_f)
        s_gap = torch.einsum(spec, torch.relu(-qg), kmax_f - kmin_f)
        return s_max + s_gap

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
        """decode 批量 page 选择。

        pool_l: 共享 pool；rows: [n] 行号；S_list: 每行有效长度（int 列表
        或 device tensor）；q: [n, H, D]（t = S-1，decode 单步）。
        返回 [n, Hkv, K] token 逻辑位置，K = topk_pages × page_size（静态
        宽度）。无效 lane（页数不足 topk / 当前页内越界位置）填 per-row
        哨兵 S_i，下游 valid = sel < S_i 统一处理（TLI 约定）。

        语义（对齐 quest_indexer.py decode 口径）：
          - 每行每 kv head：当前页（含 t 的页）强制入选；
            其余因果完成页按 GQA group-mean 上界分取 top-(topk-1)
          - 无 sink/swa：Quest 原版行为（当前页 = 最近 64 token 局部性）
        """
        p = self.profile
        pg = p.page_size
        device = q.device
        n, H, D = q.shape
        kmin_pool, kmax_pool = pool_l["kmin"], pool_l["kmax"]
        Hkv = kmin_pool.shape[2]
        G = H // Hkv
        nblk_cap = kmin_pool.shape[1]
        if torch.is_tensor(S_list):
            S_t = S_list.to(torch.long)
        else:
            S_t = torch.tensor(S_list, device=device, dtype=torch.long)
        t_t = S_t - 1  # [n] 当前 query 位置
        cur = torch.div(t_t, pg, rounding_mode="floor")  # [n] 当前页
        # 静态 topk 宽度（M5 风格；页数不足的行 topk 会选到 -inf lane → 掩掉）
        # 宽度下界守卫：topk 要求 k ≤ 搜索宽度（dense_threshold ≥ 预算时恒成立，
        # 但 dense_threshold 可配小，防御性 clamp）
        k1 = max(1, min(p.topk_pages - 1, nblk_cap - 1))

        rows_l = rows.to(torch.long)
        # per-row 界张量（[n, NBLK_CAP, Hkv, D] fp32 物化 ×2；bs16×64K×TP2
        # ≈ 64MB，131K 单卡 bs32 约 536MB——bench 口径内可控）
        kmin_f = kmin_pool[rows_l].float()
        kmax_f = kmax_pool[rows_l].float()
        qg = q.float().view(n, Hkv, G, D)
        ub = self._upper_bound(qg, kmin_f, kmax_f, batched_rows=True)
        score = ub.mean(dim=2)  # GQA group-mean → [n, Hkv, m]

        m_idx = torch.arange(nblk_cap, device=device)
        # 因果完成页：页起点 ≤ t 且非当前页（当前页强制，不参与排名）
        rankable = (m_idx.view(1, 1, -1) < cur.view(-1, 1, 1)) & (
            m_idx.view(1, 1, -1) * pg < S_t.view(-1, 1, 1)
        )
        score = score.masked_fill(~rankable, float("-inf"))
        vals, idx = torch.topk(score, k1, dim=-1)  # [n, Hkv, k1]
        lane_ok = vals > float("-inf")
        # 强制当前页
        pages = torch.cat(
            [idx, cur.view(n, 1, 1).expand(n, Hkv, 1)], dim=-1
        )  # [n, Hkv, topk]
        page_ok = torch.cat(
            [lane_ok, torch.ones_like(lane_ok[:, :, :1])], dim=-1
        )
        # 页展开成 token 位置
        off = torch.arange(pg, device=device)
        pos = pages.unsqueeze(-1) * pg + off.view(1, 1, 1, pg)  # [n,Hkv,kp,pg]
        ok = page_ok.unsqueeze(-1) & (pos < S_t.view(n, 1, 1, 1))
        sel = torch.where(ok, pos, S_t.view(n, 1, 1, 1))  # per-row 哨兵
        # k1 收缩时 pad 回静态宽度（哨兵 = per-row S_i）
        pad = p.topk_pages - (k1 + 1)
        if pad > 0:
            sel = torch.cat(
                [sel, S_t.view(n, 1, 1, 1).expand(n, Hkv, pad, pg)], dim=2
            )
        return sel.reshape(n, Hkv, p.topk_pages * pg)

    # ------------------------------------------------------------------ #
    # extend（prefill）选择：单请求 chunk 批量化
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def select_extend(
        self,
        kmin: torch.Tensor,
        kmax: torch.Tensor,
        nblk: int,
        q: torch.Tensor,
        t_arr: torch.Tensor,
        S: int,
        row_chunk: int = 512,
    ) -> torch.Tensor:
        """prefill 单请求选择（每 query 行独立因果页排名）。

        kmin/kmax: [nblk, Hkv, D]（本 chunk 索引切片）；q: [nq, H, D]；
        t_arr: [nq] query 全局位置（= prefix..S-1）；S: 请求已写池总长。
        返回 [nq, Hkv, K] token 位置（哨兵 S；含因果掩蔽：当前页内
        pos > t 的 lane 填哨兵——page 展开必须叠加 query 级因果）。

        语义对齐 quest_indexer.py 的 prefill 口径：b_mask = (q//bs) > k
        （严格早于当前页的页可排名）+ 当前页强制入选 + top-(k-1)。
        """
        p = self.profile
        pg = p.page_size
        device = q.device
        nq, H, D = q.shape
        Hkv = kmin.shape[1]
        G = H // Hkv
        # 宽度下界守卫（同 decode 版；nblk < topk-1 时收缩避免 topk 报错，
        # 下方 pad 回静态宽度）
        k1 = max(1, min(p.topk_pages - 1, nblk - 1))
        kmin_f = kmin.float()
        kmax_f = kmax.float()
        m_idx = torch.arange(nblk, device=device)
        off = torch.arange(pg, device=device)
        sels = []
        for r0 in range(0, nq, row_chunk):
            r1 = min(r0 + row_chunk, nq)
            q_c = q[r0:r1].float().view(r1 - r0, Hkv, G, D)
            ub = self._upper_bound(q_c, kmin_f, kmax_f, batched_rows=False)
            score = ub.mean(dim=2)  # [rc, Hkv, nblk]
            t_c = t_arr[r0:r1]
            cur = torch.div(t_c, pg, rounding_mode="floor")  # [rc]
            rankable = m_idx.view(1, 1, -1) < cur.view(-1, 1, 1)
            score = score.masked_fill(~rankable, float("-inf"))
            vals, idx = torch.topk(score, k1, dim=-1)
            lane_ok = vals > float("-inf")
            pages = torch.cat(
                [idx, cur.view(-1, 1, 1).expand(r1 - r0, Hkv, 1)], dim=-1
            )
            page_ok = torch.cat(
                [lane_ok, torch.ones_like(lane_ok[:, :, :1])], dim=-1
            )
            pos = pages.unsqueeze(-1) * pg + off.view(1, 1, 1, pg)
            # 因果 + 界：当前页内 pos > t 的位置掩掉（未来 token 不参与）
            ok = page_ok.unsqueeze(-1) & (pos <= t_c.view(-1, 1, 1, 1))
            sel = torch.where(ok, pos, torch.full_like(pos, S))
            # k1 收缩时 pad 回静态宽度（哨兵 S = 无效 lane）
            pad = p.topk_pages - (k1 + 1)
            if pad > 0:
                pad_t = torch.full(
                    (sel.shape[0], Hkv, pad, sel.shape[-1]), S,
                    dtype=sel.dtype, device=sel.device,
                )
                sel = torch.cat([sel, pad_t], dim=2)
            sels.append(sel.reshape(r1 - r0, Hkv, p.topk_pages * pg))
        return torch.cat(sels, dim=0)


__all__ = ["QuestIndexer"]
