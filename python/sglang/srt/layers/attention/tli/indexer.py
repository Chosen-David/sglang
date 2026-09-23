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
        """decode 增量：新 token [n, Hkv, D] 追加（重算所在块 + 增量聚类 assign）。"""
        p = self.profile
        S_old = index["S"]
        S = S_old + k_new.shape[0]
        # prototype 简化：重算受影响块（生产版 kernel 只碰最后一个块）
        nblk = (S + p.block_size - 1) // p.block_size
        device = k_new.device
        # 把新 token 拼到缓存（kq 直接 append；kmin/kmax 重算尾部块）
        kq_new = quant4(k_new[..., self.idx2])
        index["kq"] = torch.cat([index["kq"], kq_new], dim=0)
        # 尾块重算：这里需要原始 k——prototype 由 backend 传入整段尾部
        # （backend 负责 gather 尾部 block_size 个 token）
        index["S"] = S
        index["nblk"] = nblk
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
