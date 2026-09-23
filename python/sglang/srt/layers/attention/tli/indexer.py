"""TLI (Two-Level Indexer) 核心选择算法。

纯 tensor 接口（不依赖 sglang 运行时），便于在 trace 上单测对拍
（exp/trace/analyze_e3b.py 的 two_level_pipeline 即本实现的参考口径，
实测 mass coverage 0.9967–1.0000 vs dense top-1024）。

两级结构（TIA 语义 + 实测 Go 的创新点 A/B/D'）：
  L1: 子空间(d') 块 min/max 上界 → top-K1 块 + 滑窗块强制入选
  L2: 4bit 量化部分维 token 精筛 → top-K2 token + 滑窗 128 强制
  B(可选): 远端候选由 kmeans 聚类中心分数给出（E4: recall 4–10× 于 minmax）
  D'(可选): 被校准掩码跳过的层直接只保留 sink/滑窗/近端
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.tli.config import TLIProfile


def quant4(x: torch.Tensor) -> torch.Tensor:
    """per-token 4bit 对称格点量化（TIA 语义，dequant 后的近似值）。"""
    mx = x.amax(-1, keepdim=True)
    mn = x.amin(-1, keepdim=True)
    sc = (mx - mn).clamp(min=1e-9) / 15
    return torch.clamp(torch.round((x - mn) / sc), 0, 15) * sc + mn


def gpu_kmeans(x: torch.Tensor, K: int, niter: int = 20, seed: int = 0):
    """GEMM 距离 kmeans（创新点 B；复用 KMeans/CPU 调研的 GEMM 技巧）。"""
    g = torch.Generator().manual_seed(seed)
    N, d = x.shape
    c = x[torch.randperm(N, generator=g)[:K]].clone()
    for _ in range(niter):
        assign = (x @ c.T).argmax(dim=1)
        cnt = torch.bincount(assign, minlength=K).float()
        sums = torch.zeros(K, d, device=x.device).index_add_(0, assign, x)
        nonempty = cnt > 0
        c[nonempty] = sums[nonempty] / cnt[nonempty, None]
    return c


class TLIIndexer:
    """两级索引器（PyTorch 参考实现；kernel 化路径见导师 TileLang 两级 kernel）。"""

    def __init__(self, profile: TLIProfile | None = None, head_dim: int = 128) -> None:
        self.profile = profile or TLIProfile()
        p = self.profile
        self.register_buffer_idx(torch.arange(head_dim))
        self.idx1 = torch.tensor(p.subspace_idx(head_dim), dtype=torch.long)
        self.idx2 = torch.tensor(p.refine_idx(head_dim), dtype=torch.long)
        self.skip_far: bool = False  # D'：由 backend 按 layer 掩码置位

    def register_buffer_idx(self, _):
        pass

    def to(self, device):
        self.idx1 = self.idx1.to(device)
        self.idx2 = self.idx2.to(device)
        return self

    # ------------------------------------------------------------------ #
    # 索引维护（prefill 全量 / decode 增量），由 backend 调用
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def build_block_index(self, k: torch.Tensor) -> dict:
        """k: [S, Hkv, D] fp32（RoPE 后）→ 块 min/max + 4bit 精筛缓存 + 可选聚类。

        返回 dict，decode 增量更新用 update_block_index。
        """
        p = self.profile
        S, Hkv, D = k.shape
        device = k.device
        nblk = (S + p.block_size - 1) // p.block_size
        pad = nblk * p.block_size - S
        if pad:
            k = F.pad(k, (0, 0, 0, 0, 0, pad))
        kc = k[..., self.idx1].reshape(nblk, p.block_size, Hkv, p.coarse_dim)
        kmin = kc.amin(1)  # [nblk, Hkv, d']
        kmax = kc.amax(1)
        # 尾块精确界：零 pad 会把 0 混进 min/max（增量路径的界永久变宽且
        # 无法恢复 == 增量≠全量重建）。尾块在两条 select 路径恒被因果 mask
        # （blk_end > t），此修正不改变选择语义，只使增量维护可精确对拍。
        if pad:
            valid_tail = S - (nblk - 1) * p.block_size
            kmin[-1] = kc[-1, :valid_tail].amin(0)
            kmax[-1] = kc[-1, :valid_tail].amax(0)
        # 4bit 部分维（L2 精筛用；存 dequant 近似，生产版应存 uint4+scale）
        kq = quant4(k[:S][..., self.idx2])  # [S, Hkv, 2*delta]
        index = {
            "kmin": kmin,
            "kmax": kmax,
            "kq": kq,
            "nblk": nblk,
            "S": S,
        }
        # 创新点 B：远端聚类（prefill 一次；E7 实测 decode 增量 assign 衰减 ≤0.03）
        if p.far_kmeans and S > p.dense_threshold:
            far_hi = max(64, S - 2048)
            kfar_sub = k[:S][..., self.idx2][64:far_hi]  # [Tfar, Hkv, 2*delta]
            centroids, assign = [], []
            for h in range(Hkv):
                c = gpu_kmeans(kfar_sub[:, h, :], p.far_clusters)
                centroids.append(c)
                assign.append((kfar_sub[:, h, :] @ c.T).argmax(1))
            index["far_centroids"] = torch.stack(centroids)  # [Hkv, K_c, 2d]
            index["far_assign"] = torch.stack(assign)  # [Hkv, Tfar]
        return index

    @torch.no_grad()
    def update_block_index(self, index: dict, k_new: torch.Tensor) -> dict:
        """decode 增量：新 token [n, Hkv, D] 追加。

        精确增量（O(n) 而非 O(S)）：
        - kq 直接 append（4bit 逐 token 无块依赖）
        - 尾块 min/max 只与新 token 比较（min/max 的结合律）；
          跨块边界则新开块（min=max=新 token）
        E5b 实测 eager 每步全量重建是 gov_report 186min 的根因，此处修复。
        """
        p = self.profile
        n_new = k_new.shape[0]
        S_old = index["S"]
        S = S_old + n_new
        kq_new = quant4(k_new[..., self.idx2])
        index["kq"] = torch.cat([index["kq"], kq_new], dim=0)
        ks = k_new[..., self.idx1]  # [n, Hkv, d']
        Hkv = ks.shape[1]

        def _append_blocks(ks_seg: torch.Tensor) -> None:
            """把 ks_seg（从全局位置 s0 起）的块界精确 append（无零 pad 污染）。"""
            n = ks_seg.shape[0]
            nb_full = n // p.block_size
            if nb_full:
                kc = ks_seg[: nb_full * p.block_size].reshape(
                    nb_full, p.block_size, Hkv, p.coarse_dim
                )
                index["kmin"] = torch.cat([index["kmin"], kc.amin(1)])
                index["kmax"] = torch.cat([index["kmax"], kc.amax(1)])
            rem = n - nb_full * p.block_size
            if rem:
                # 部分尾块：界 = rem 个 token 的精确 min/max（与 build 的
                # 尾块精确界口径一致，保证 增量 == 全量重建 逐位成立）
                r = ks_seg[nb_full * p.block_size :]
                index["kmin"] = torch.cat([index["kmin"], r.amin(0)[None]])
                index["kmax"] = torch.cat([index["kmax"], r.amax(0)[None]])

        if S_old % p.block_size == 0:
            # 对齐边界：新 token 直接开新块（decode n=1 时恰好一块）
            _append_blocks(ks)
        else:
            # 尾块更新（min/max 结合律，只与新 token 比较）
            tail = S_old % p.block_size  # 尾块已有 token 数
            take = min(n_new, p.block_size - tail)
            index["kmin"][-1] = torch.minimum(index["kmin"][-1], ks[:take].amin(0))
            index["kmax"][-1] = torch.maximum(index["kmax"][-1], ks[:take].amax(0))
            rest = n_new - take
            if rest > 0:
                _append_blocks(ks[take:])
        index["S"] = S
        index["nblk"] = (S + p.block_size - 1) // p.block_size
        return index

    # ------------------------------------------------------------------ #
    # 两级选择（与 analyze_e3b.two_level_pipeline 同口径）
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def select(
        self,
        index: dict,
        q: torch.Tensor,
        t: int,
        tail_k: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """index: build_block_index 产物；q: [1, H, D]；t: 当前 query 位置。

        返回 [Hkv, K2] 的 token 位置。
        tail_k: [S, Hkv, D]（D' 跳过远端时用于近端/滑窗精筛；None 则用 index）
        """
        p = self.profile
        S = index["S"]
        Hkv = index["kmin"].shape[1]
        H = q.shape[1]
        G = H // Hkv
        device = q.device
        kmin, kmax = index["kmin"], index["kmax"]
        nblk = index["nblk"]
        K1 = min(p.k1_blocks, nblk)

        # ---- L1: 子空间块上界（创新点 A：只算 d' 维）----
        qs = q[..., self.idx1]  # [1, H, d']
        qg = qs.clamp(min=0).reshape(1, Hkv, G, p.coarse_dim)
        qn = qs.clamp(max=0).reshape(1, Hkv, G, p.coarse_dim)
        sc1 = (
            torch.einsum("bhgd,nhd->bhgn", qg, kmax)
            + torch.einsum("bhgd,nhd->bhgn", qn, kmin)
        ).sum(-2)  # [1, Hkv, nblk]
        blk_end = (torch.arange(nblk, device=device) + 1) * p.block_size - 1
        sc1 = sc1.masked_fill(blk_end.view(1, 1, -1) > t, float("-inf"))

        # D'：跳过远端的层——只保留 sink(块0) + 近端窗块
        if self.skip_far:
            near_blks = max(1, (t + 1 - 2048) // p.block_size)
            keep = torch.zeros(nblk, dtype=torch.bool, device=device)
            keep[: min(2, nblk)] = True
            keep[max(0, near_blks) :] = True
            sc1 = sc1.masked_fill(~keep.view(1, 1, -1), float("-inf"))
            # D' 真正兑现省算：topk 截断到有效块数（否则 -inf 块填满 K1 白算）
            K1 = min(K1, max(1, int(keep.sum().item())))

        cand_blk = torch.topk(sc1, K1, dim=-1).indices[0]  # [Hkv, K1]（不含滑窗）
        # TIA 语义：滑窗块强制入选
        sw_blks = p.sliding_blocks
        last_blk = t // p.block_size
        force_blks = torch.arange(
            max(0, last_blk - sw_blks + 1), last_blk + 1, device=device
        )
        cand_blk = torch.cat(
            [cand_blk, force_blks.unsqueeze(0).expand(Hkv, -1)], dim=1
        )  # 重复无碍（mask 化）

        # ---- L2: 4bit 部分维 token 精筛 ----
        nd2 = 2 * p.delta
        kq = index["kq"]  # [S, Hkv, nd2]
        q2 = q[..., self.idx2].reshape(1, Hkv, G, nd2).sum(2)  # [1, Hkv, nd2]
        blk_onehot = torch.zeros(Hkv, nblk, dtype=torch.bool, device=device)
        blk_onehot.scatter_(1, cand_blk, True)
        sel_mask = blk_onehot.any(0).repeat_interleave(p.block_size)[:S]
        cand_pos = torch.nonzero(sel_mask).squeeze(1)
        kq_h = kq[cand_pos]  # [Tc, Hkv, nd2]
        s2 = torch.einsum("hd,thd->ht", q2[0], kq_h)  # [Hkv, Tc]
        fine = torch.full((Hkv, S), float("-inf"), device=device)
        fine[:, cand_pos] = s2
        fine = fine.masked_fill(
            torch.arange(S, device=device).view(1, S) > t, float("-inf")
        )
        # TIA 语义：最后 sliding_window token 强制入选
        forced = torch.arange(max(0, t - p.sliding_window + 1), t + 1, device=device)
        fine[:, forced] = float("inf")
        # ---- B'：far/near 分区 top-K2（独立预算，防远端被近端高分挤出）----
        # far 捕获 128 tok 即饱和（e5b_far_tokens_sensitivity），且近端至少
        # 保留滑窗 + sink（far_tokens ≥ K2 时近端预算归零的边界已保护）
        if not self.skip_far:
            far_tok_lo = p.sink_blocks * p.block_size
            far_tok_hi = max(far_tok_lo, t + 1 - p.near_len)
            near_floor = p.sliding_window + far_tok_lo
            far_cap = max(0, p.token_budget - near_floor)
            if far_tok_hi > far_tok_lo:
                k2_far = min(p.far_tokens, far_tok_hi - far_tok_lo, far_cap)
                far_f = fine[:, far_tok_lo:far_tok_hi]
                i_f = torch.topk(far_f, k2_far, dim=-1).indices + far_tok_lo
                near_f = fine.clone()
                near_f[:, far_tok_lo:far_tok_hi] = float("-inf")
                k2_near = max(0, p.token_budget - k2_far)
                i_n = torch.topk(near_f, min(k2_near, S), dim=-1).indices
                return torch.cat([i_f, i_n], dim=-1)  # [Hkv, K2]（far + near 拼接）
        return torch.topk(fine, p.token_budget, dim=-1).indices  # [Hkv, K2]

    @torch.no_grad()
    def select_batched(
        self,
        index: dict,
        q: torch.Tensor,
        t_arr: torch.Tensor,
        row_chunk: int = 64,
    ) -> torch.Tensor:
        """prefill 批量两级选择（M2 稀疏 prefill 用）。

        q: [Nq, H, D]；t_arr: [Nq] 每行因果位置（全局逻辑位置）。
        返回 [Nq, Hkv, K2] token 位置（far + near 拼接，与 select 同语义）。

        与 select 的对齐（逐行对拍须位置集合一致）：
        - L2 候选池 = 各 head L1 选择块 ∪ 滑窗块的**并集**（select 的
          blk_onehot.any(0) 语义：跨 head 共享候选池）
        - L2 打分 scatter 到 [n, Hkv, S] 的 fine 矩阵后再 topk（S 维索引
          → 位置天然去重，候选重复 overwrite 无害）
        - 无 sink 强制（sink 只经 far_lo 边界进 near 池竞争，与 select 一致）
        - 远端区为空的行（t+1 ≤ near_len+far_lo）退化为整体 topk
        row_chunk 控制 kq gather [n, Tc, Hkv, nd2] 峰值显存（Tc ≈ K1*bs：
        64 行 ≈ 1.1GB@S=32K）。
        """
        p = self.profile
        S = index["S"]
        Hkv = index["kmin"].shape[1]
        Nq, H, D = q.shape
        G = H // Hkv
        device = q.device
        kmin, kmax = index["kmin"], index["kmax"]
        nblk = index["nblk"]
        K1 = min(p.k1_blocks, nblk)
        nd2 = 2 * p.delta
        bs = p.block_size
        kq = index["kq"]  # [S, Hkv, nd2]
        t_arr = t_arr.to(device)
        pos = torch.arange(S, device=device)
        out = []
        for r0 in range(0, Nq, row_chunk):
            r1 = min(r0 + row_chunk, Nq)
            q_c = q[r0:r1]
            t_c = t_arr[r0:r1]
            n = r1 - r0
            # ---- L1: 子空间块上界（a=行 / m=块，避免维度名冲突）----
            qs = q_c[..., self.idx1]  # [n, H, d']
            qg = qs.clamp(min=0).reshape(n, Hkv, G, p.coarse_dim)
            qn = qs.clamp(max=0).reshape(n, Hkv, G, p.coarse_dim)
            sc1 = torch.einsum("ahgd,mhd->ahm", qg, kmax) + torch.einsum(
                "ahgd,mhd->ahm", qn, kmin
            )  # [n, Hkv, nblk]
            blk_end = (torch.arange(nblk, device=device) + 1) * bs - 1
            sc1 = sc1.masked_fill(
                blk_end.view(1, 1, -1) > t_c.view(-1, 1, 1), float("-inf")
            )
            # D'：跳过远端层——只保留 sink + 近端窗块
            if self.skip_far:
                near_blks = ((t_c + 1 - p.near_len) // bs).clamp(min=0)
                keep = torch.arange(nblk, device=device).view(1, -1) >= near_blks.view(-1, 1)
                keep[:, : min(2, nblk)] = True
                sc1 = sc1.masked_fill(~keep.unsqueeze(1), float("-inf"))
                K1 = min(K1, max(1, int(keep.sum(1).max().item())))
            cand_blk = torch.topk(sc1, K1, dim=-1).indices  # [n, Hkv, K1]

            # ---- 候选块并集（select 的 blk_onehot.any(0) 语义）----
            # topk 块 ∪ 滑窗块（per row），跨 head 展开为共享 token 池
            onehot = torch.zeros(n, nblk, dtype=torch.bool, device=device)
            onehot.scatter_(1, cand_blk.reshape(n, -1), True)
            f_blk = (
                t_c.view(-1, 1) // bs - torch.arange(p.sliding_blocks, device=device)
            ).clamp(min=0)  # 块级滑窗（select 的 force_blks 同语义）
            onehot.scatter_(1, f_blk, True)
            sel_mask = onehot.repeat_interleave(bs, dim=1)[:, :S]  # [n, S]
            # 掩码位置 → 定长候选张量（topk 最小值技巧：哨兵 S 排最后补 pad）
            seq = pos.view(1, S).expand(n, S)
            seq_m = torch.where(sel_mask, seq, torch.full_like(seq, S))
            Tc = int(sel_mask.sum(1).max().item())
            tok = torch.topk(seq_m, Tc, dim=-1, largest=False).values  # [n, Tc]
            valid = tok < S
            tok_c = tok.clamp(max=S - 1)

            # ---- L2: 4bit 部分维精筛 → scatter 进 fine [n, Hkv, S] ----
            kq_c = kq[tok_c]  # [n, Tc, Hkv, nd2]（候选池跨 head 共享）
            q2 = q_c[..., self.idx2].reshape(n, Hkv, G, nd2).sum(2)  # [n, Hkv, nd2]
            s2 = torch.einsum("ahd,athd->aht", q2, kq_c)  # [n, Hkv, Tc]
            causal = (tok <= t_c.view(-1, 1)) & valid  # [n, Tc]
            fine = torch.full((n, Hkv, S), float("-inf"), device=device)
            fine.scatter_(
                2,
                tok_c.unsqueeze(1).expand(n, Hkv, Tc),
                torch.where(causal.unsqueeze(1), s2, torch.full_like(s2, float("-inf"))),
            )
            # 强制 token：滑窗 [t-sw+1, t]（clamp 产生的重复位置由 scatter
            # overwrite 天然去重；无 sink 强制，与 select 一致）
            sw_off = torch.arange(p.sliding_window, device=device)
            f_sw = (t_c.view(-1, 1) - sw_off).clamp(min=0)  # [n, sw]
            fine.scatter_(2, f_sw.unsqueeze(1).expand(n, Hkv, -1), float("inf"))

            # ---- B'：far/near 分区 topk（批量位置掩码，topk 直接返回位置）----
            if not self.skip_far:
                far_lo = p.sink_blocks * bs
                far_hi = (t_c + 1 - p.near_len).clamp(min=far_lo)  # [n]
                in_far = (pos.view(1, S) >= far_lo) & (
                    pos.view(1, S) < far_hi.view(-1, 1)
                )  # [n, S]（滑窗 ≥ far_hi、sink < far_lo，强制位天然不进 far 池）
                near_floor = p.sliding_window + far_lo
                far_cap = max(0, p.token_budget - near_floor)
                k2_far = min(p.far_tokens, far_cap)
                far_f = fine.masked_fill(~in_far.unsqueeze(1), float("-inf"))
                i_f = torch.topk(far_f, k2_far, dim=-1).indices
                near_f = fine.masked_fill(in_far.unsqueeze(1), float("-inf"))
                k2_near = max(0, p.token_budget - k2_far)
                i_n = torch.topk(near_f, min(k2_near, S), dim=-1).indices
                res = torch.cat([i_f, i_n], dim=-1)  # [n, Hkv, K2]
                # 远端区为空的行（t+1 ≤ near_len+far_lo，首 chunk 早段行）：
                # 与 select 同语义 = 整体 topk(fine, budget)（此时 [far_lo,S)
                # 全部按定义属于近端，分区只会选出 -inf 垃圾位）
                empty = far_hi <= far_lo  # [n]
                if bool(empty.any()):
                    i_g = torch.topk(fine, min(p.token_budget, S), dim=-1).indices
                    res = torch.where(empty.view(n, 1, 1), i_g, res)
                out.append(res)
            else:
                out.append(torch.topk(fine, min(p.token_budget, S), dim=-1).indices)
        return torch.cat(out, dim=0)  # [Nq, Hkv, K2]

    @torch.no_grad()
    def select_far_kmeans(self, index: dict, q: torch.Tensor, t: int, budget: int):
        """创新点 B：远端候选由聚类中心分数展开（E4/E7 实测口径）。

        返回 [Hkv, ≤budget] 远端 token 位置（供与 select 的 L1 远端部分交换）。
        """
        p = self.profile
        centroids = index["far_centroids"]  # [Hkv, K_c, nd2]
        far_assign = index["far_assign"]  # [Hkv, Tfar]
        Hkv, K_c, nd2 = centroids.shape
        H = q.shape[1]
        G = H // Hkv
        q2 = q[..., self.idx2].reshape(1, Hkv, G, nd2).sum(2)[0]  # [Hkv, nd2]
        sc = torch.einsum("hd,hkd->hk", q2, centroids)  # [Hkv, K_c]
        out = []
        far_lo, far_hi = 64, index["S"] - 2048
        for h in range(Hkv):
            order = torch.argsort(sc[h], descending=True)
            cands, taken = [], 0
            for ci in order.tolist():
                if taken >= budget:
                    break
                members = torch.nonzero(far_assign[h] == ci).squeeze(1) + far_lo
                cands.append(members)
                taken += members.numel()
            out.append(torch.cat(cands) if cands else torch.empty(0, dtype=torch.long, device=q.device))
        return out


__all__ = ["TLIIndexer", "quant4", "gpu_kmeans"]
