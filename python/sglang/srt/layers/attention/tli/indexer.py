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


def quant4_pack(x: torch.Tensor):
    """M6：kq 真 4bit 存储——量化为 (grid uint8, sc fp32, mn fp32)。

    x: [..., nd2] → grid: uint8 同形（0-15 格点）；sc/mn: [...,]（尾维压掉）。
    重建 kq_unpack(grid, sc, mn) 与 quant4(x) **逐位一致**：格点值是
    [0,15] 整数（fp32 精确表示），float(grid)*sc+mn 与 round(...)*sc+mn
    是同操作数同序的 IEEE 运算。
    存储 128 → 40 B/token-head（32B grid + 8B 双 scale fp32）。
    """
    mx = x.amax(-1, keepdim=True)
    mn = x.amin(-1, keepdim=True)
    sc = (mx - mn).clamp(min=1e-9) / 15
    grid = torch.clamp(torch.round((x - mn) / sc), 0, 15).to(torch.uint8)
    return grid, sc.squeeze(-1), mn.squeeze(-1)


def kq_unpack(grid: torch.Tensor, sc: torch.Tensor, mn: torch.Tensor) -> torch.Tensor:
    """M6：uint8 格点 + scale 重建 fp32 kq（与 quant4 逐位一致，见上）。"""
    return grid.float() * sc.unsqueeze(-1) + mn.unsqueeze(-1)


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
        # 4bit 部分维（L2 精筛用；M6 真 4bit 存储：uint8 格点 + fp32 scale，
        # 128→40B/token-head——S=131K pool 显存硬前提）
        kq_q, kq_sc, kq_mn = quant4_pack(k[:S][..., self.idx2])  # [S, Hkv, 2δ] uint8 + [S, Hkv] ×2
        index = {
            "kmin": kmin,
            "kmax": kmax,
            "kq_q": kq_q,
            "kq_sc": kq_sc,
            "kq_mn": kq_mn,
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

        M3-c：kq/kmin/kmax 用几何扩容的预分配 buffer（cat 版每步 O(S) 拷贝，
        S=131K 时 kq 134MB × 36 层 ≈ 4.8GB/步纯 memcpy）。buffer 第 0 维
        是容量，有效长度由 S/nblk 跟踪；[S:cap)/[nblk:cap) 是垃圾，
        消费方必须按 S/nblk 切片或掩码访问（select/select_batched 已切）。
        """
        p = self.profile
        n_new = k_new.shape[0]
        S_old = index["S"]
        S = S_old + n_new

        def _ensure(buf: torch.Tensor, need: int) -> torch.Tensor:
            cap = buf.shape[0]
            if cap >= need:
                return buf
            nb = buf.new_empty((max(need, cap * 2), *buf.shape[1:]))
            nb[:cap] = buf
            return nb

        kq_q_new, kq_sc_new, kq_mn_new = quant4_pack(k_new[..., self.idx2])
        index["kq_q"] = _ensure(index["kq_q"], S)
        index["kq_sc"] = _ensure(index["kq_sc"], S)
        index["kq_mn"] = _ensure(index["kq_mn"], S)
        index["kq_q"][S_old:S] = kq_q_new
        index["kq_sc"][S_old:S] = kq_sc_new
        index["kq_mn"][S_old:S] = kq_mn_new
        ks = k_new[..., self.idx1]  # [n, Hkv, d']
        Hkv = ks.shape[1]
        nblk_old = index["nblk"]

        def _append_blocks(ks_seg: torch.Tensor) -> None:
            """把 ks_seg（从全局位置 s0 起）的块界精确 append（无零 pad 污染）。"""
            n = ks_seg.shape[0]
            nb_full = n // p.block_size
            n_add = nb_full + (1 if n % p.block_size else 0)
            if n_add == 0:
                return
            index["kmin"] = _ensure(index["kmin"], nblk_old + n_add)
            index["kmax"] = _ensure(index["kmax"], nblk_old + n_add)
            w = nblk_old
            if nb_full:
                kc = ks_seg[: nb_full * p.block_size].reshape(
                    nb_full, p.block_size, Hkv, p.coarse_dim
                )
                index["kmin"][w : w + nb_full] = kc.amin(1)
                index["kmax"][w : w + nb_full] = kc.amax(1)
                w += nb_full
            rem = n - nb_full * p.block_size
            if rem:
                # 部分尾块：界 = rem 个 token 的精确 min/max（与 build 的
                # 尾块精确界口径一致，保证 增量 == 全量重建 逐位成立）
                r = ks_seg[nb_full * p.block_size :]
                index["kmin"][w] = r.amin(0)
                index["kmax"][w] = r.amax(0)

        if S_old % p.block_size == 0:
            # 对齐边界：新 token 直接开新块（decode n=1 时恰好一块）
            _append_blocks(ks)
        else:
            # 尾块更新（min/max 结合律，只与新 token 比较）；
            # 显式下标 nblk_old-1——buffer 可能有容量 padding，[-1] 会写错行
            tail = S_old % p.block_size  # 尾块已有 token 数
            take = min(n_new, p.block_size - tail)
            index["kmin"][nblk_old - 1] = torch.minimum(
                index["kmin"][nblk_old - 1], ks[:take].amin(0)
            )
            index["kmax"][nblk_old - 1] = torch.maximum(
                index["kmax"][nblk_old - 1], ks[:take].amax(0)
            )
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
        use_l1_kernel: bool = False,
        use_l2_kernel: bool = False,
    ) -> torch.Tensor:
        """index: build_block_index 产物；q: [1, H, D]；t: 当前 query 位置。

        返回 [Hkv, K2] 的 token 位置（use_l2_kernel 时池不足的槽位填
        哨兵 S，下游 valid = sel < S 统一处理）。
        tail_k: [S, Hkv, D]（D' 跳过远端时用于近端/滑窗精筛；None 则用 index）
        use_l1_kernel: True 时 L1 用 Triton fused kernel（E8-2 实测
        3.6×/1.6×；并列截断多选的块由 L2 4bit 精筛自然淘汰，质量不降）
        use_l2_kernel: True 且 far 区非空时 L2 分区精筛也走 fused kernel
        （单 launch/head，E8-2 原型 1.63×）
        """
        p = self.profile
        S = index["S"]
        Hkv = index["kmin"].shape[1]
        H = q.shape[1]
        G = H // Hkv
        device = q.device
        nblk = index["nblk"]
        # 增量路径的 kmin/kmax 带容量 padding（update_block_index 预分配），
        # 第 0 维 ≥ nblk 的尾部是垃圾——按 nblk 切片
        kmin, kmax = index["kmin"][:nblk], index["kmax"][:nblk]
        K1 = min(p.k1_blocks, nblk)
        last_blk = t // p.block_size
        sw_blks = p.sliding_blocks
        force_blks = torch.arange(
            max(0, last_blk - sw_blks + 1), last_blk + 1, device=device
        )

        if use_l1_kernel:
            # ---- L1 fused kernel 路径（kernels.tli_l1_topk）----
            from sglang.srt.layers.attention.tli.kernels import tli_l1_topk

            if self.skip_far:
                near_blks = max(1, (t + 1 - 2048) // p.block_size)
                far_lo_blk, far_hi_blk = 2, max(2, near_blks)
            else:
                far_lo_blk = far_hi_blk = 0
            q_sub = q[..., self.idx1].reshape(H, -1).contiguous()  # [H, d']
            mask = tli_l1_topk(
                q_sub, kmin, kmax, K1,
                far_lo_blk, far_hi_blk, last_blk,
            )
            blk_onehot = mask[:, :nblk].bool()
            blk_onehot[:, force_blks] = True  # 滑窗块强制
        else:
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
            cand_blk = torch.cat(
                [cand_blk, force_blks.unsqueeze(0).expand(Hkv, -1)], dim=1
            )  # 重复无碍（mask 化）
            blk_onehot = torch.zeros(Hkv, nblk, dtype=torch.bool, device=device)
            blk_onehot.scatter_(1, cand_blk, True)

        # ---- L2: 4bit 部分维 token 精筛 ----
        # fused 分区 kernel（单 launch/head：精筛分数 + far/near 分区 topk +
        # 滑窗强制 + 直接写位置）。D' 跳层 / far 区为空时退回 eager。
        if use_l2_kernel and not self.skip_far:
            far_tok_lo = p.sink_blocks * p.block_size
            far_tok_hi = max(far_tok_lo, t + 1 - p.near_len)
            if far_tok_hi > far_tok_lo:
                sel_mask = blk_onehot.any(0).repeat_interleave(p.block_size)[:S]
                cand_pos = torch.nonzero(sel_mask).squeeze(1)
                if cand_pos.numel() > 0:
                    from sglang.srt.layers.attention.tli.kernels import (
                        tli_l2_partition_topk,
                    )

                    nd2 = 2 * p.delta
                    q_sub = q[..., self.idx2].reshape(H, nd2).contiguous()
                    near_floor = p.sliding_window + far_tok_lo
                    far_cap = max(0, p.token_budget - near_floor)
                    k2_far = min(p.far_tokens, far_tok_hi - far_tok_lo, far_cap)
                    k2_near = max(0, p.token_budget - k2_far)
                    sw_lo = max(0, t - p.sliding_window + 1)
                    return tli_l2_partition_topk(
                        q_sub,
                        kq_unpack(index["kq_q"][:S], index["kq_sc"][:S], index["kq_mn"][:S]).contiguous(),
                        cand_pos,
                        S,
                        k2_far,
                        k2_near,
                        far_tok_lo,
                        far_tok_hi,
                        sw_lo,
                        t,
                    )

        nd2 = 2 * p.delta
        kq = index["kq_q"]  # M6 uint8 格点 [S, Hkv, nd2]（容量 padding 靠 cand_pos < S 规避）
        q2 = q[..., self.idx2].reshape(1, Hkv, G, nd2).sum(2)  # [1, Hkv, nd2]
        sel_mask = blk_onehot.any(0).repeat_interleave(p.block_size)[:S]
        cand_pos = torch.nonzero(sel_mask).squeeze(1)
        kq_h = kq_unpack(
            kq[cand_pos], index["kq_sc"][cand_pos], index["kq_mn"][cand_pos]
        )  # [Tc, Hkv, nd2]（逐位 == fp32 存储版）
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
        kmin, kmax = kmin[:nblk], kmax[:nblk]  # 容量 padding 垃圾行切除
        K1 = min(p.k1_blocks, nblk)
        nd2 = 2 * p.delta
        bs = p.block_size
        kq = index["kq_q"]  # M6 uint8 格点（tok_c < S，容量 padding 无害）
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
            kq_c = kq_unpack(
                kq[tok_c], index["kq_sc"][tok_c], index["kq_mn"][tok_c]
            )  # [n, Tc, Hkv, nd2]（候选池跨 head 共享）
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
    def update_pool_rows_decode(
        self,
        pool_l: dict,
        rows: torch.Tensor,
        S_old,
        k_new: torch.Tensor,
    ) -> None:
        """M4 phase-3：decode 批量增量维护（每行恰追加 1 个新 token）。

        pool_l: backend 共享 pool dict；rows: [n] 行号；S_old: 每行旧有效
        长度（Python int 列表或 device tensor——M5 CUDA graph 路径必须传
        tensor：torch.tensor(list, device=...) 的 H2D 不能出现在 capture
        区域内）；k_new: [n, Hkv, D] fp32（各行新 token）。
        语义等价于逐行 update_block_index(_row_views(pool_l,row,S_old_r),
        k_new_r[None])——对齐开新块 / 非对齐尾块 min/max 合并，全部
        flat 索引 scatter 完成（~10 launch 总量，与行数无关地替代
        O(n) 次逐行调用）。

        前提：pool 容量已足够（backend 先 _ensure_pool_s）；调用方负责
        更新 pool_l["S"][row]。
        """
        p = self.profile
        bs = p.block_size
        if torch.is_tensor(S_old):
            S_old_t = S_old.to(torch.long)
            n = S_old_t.shape[0]
        else:
            n = len(S_old)
            S_old_t = None
        if n == 0:
            return
        Hkv = k_new.shape[1]
        nd2 = 2 * p.delta
        d1 = p.coarse_dim
        device = k_new.device
        S_cap = pool_l["kq_q"].shape[1]
        nblk_cap = pool_l["kmin"].shape[1]
        rows_l = rows.to(torch.long)
        if S_old_t is None:
            S_old_t = torch.tensor(S_old, device=device)

        # ---- kq 追加（M6 uint8+scale 三张量 flat scatter）----
        kq_q_new, kq_sc_new, kq_mn_new = quant4_pack(k_new[..., self.idx2])
        base_q = rows_l * (S_cap * Hkv * nd2) + S_old_t * (Hkv * nd2)
        off_q = (
            base_q.view(n, 1, 1)
            + torch.arange(Hkv, device=device).view(1, Hkv, 1) * nd2
            + torch.arange(nd2, device=device).view(1, 1, nd2)
        )
        pool_l["kq_q"].reshape(-1)[off_q.reshape(-1)] = kq_q_new.reshape(-1)
        off_s = (
            (rows_l * (S_cap * Hkv) + S_old_t * Hkv).view(n, 1)
            + torch.arange(Hkv, device=device).view(1, Hkv)
        )
        pool_l["kq_sc"].reshape(-1)[off_s.reshape(-1)] = kq_sc_new.reshape(-1)
        pool_l["kq_mn"].reshape(-1)[off_s.reshape(-1)] = kq_mn_new.reshape(-1)

        # ---- kmin/kmax：对齐行开新块（min=max=新 token），非对齐行
        # 与旧尾块 min/max 合并（结合律，与逐行 update 逐位一致）----
        ks = k_new[..., self.idx1]  # [n, Hkv, d']
        aligned = S_old_t % bs == 0  # [n]
        nblk_old = (S_old_t + bs - 1) // bs
        blk_idx = torch.where(aligned, nblk_old, nblk_old - 1)  # 目标块
        base_m = rows_l * (nblk_cap * Hkv * d1) + blk_idx * (Hkv * d1)
        off_m = (
            base_m.view(n, 1, 1)
            + torch.arange(Hkv, device=device).view(1, Hkv, 1) * d1
            + torch.arange(d1, device=device).view(1, 1, d1)
        )
        off_mf = off_m.reshape(-1)
        old_mn = pool_l["kmin"].reshape(-1)[off_mf].view(n, Hkv, d1)
        old_mx = pool_l["kmax"].reshape(-1)[off_mf].view(n, Hkv, d1)
        mn_val = torch.where(aligned.view(n, 1, 1), ks, torch.minimum(old_mn, ks))
        mx_val = torch.where(aligned.view(n, 1, 1), ks, torch.maximum(old_mx, ks))
        pool_l["kmin"].reshape(-1)[off_mf] = mn_val.reshape(-1)
        pool_l["kmax"].reshape(-1)[off_mf] = mx_val.reshape(-1)

    @torch.no_grad()
    def select_decode_batched(
        self,
        pool_l: dict,
        rows: torch.Tensor,
        S_list: list[int],
        q: torch.Tensor,
        row_chunk_bytes: int = 256 << 20,
    ) -> torch.Tensor:
        """M4：decode 批量两级选择（共享 index pool 行上的全 eager 批量化）。

        M6 起直接收 pool dict（kq 三张量 uint8+scale：kq_q [R,S_cap,Hkv,nd2]
        uint8、kq_sc/kq_mn [R,S_cap,Hkv] fp32；kmin/kmax [R,NBLK_CAP,Hkv,d']）；
        rows: [n] pool 行号；S_list: 每行有效长度（Python int 列表或 device
        tensor——M5 CUDA graph 路径传 tensor，杜绝 H2D 同步）；
        q: [n, H, D] fp32（t = S-1，decode 单步）。
        返回 [n, Hkv, K2'] token 逻辑位置，池不足槽位 = 哨兵 S_cap
        （≥ 任意 S_i，下游 valid = sel < S_i 统一处理）。

        与 per-request select（eager / L2-kernel 两路径）的集合语义对齐：
        - L1 候选块 = 各 head top-K1 块 ∪ 滑窗块的**跨 head 并集**
          （select 的 blk_onehot.any(0) 语义）；越界/非因果/D' 掩蔽块剔除
        - L2 打分后按 L2-kernel 路径口径分区：far 池 [far_lo, far_hi)、
          near 池 pos < sw_lo（不含滑窗），滑窗 [sw_lo, t] 显式 append，
          近端配额扣减 F = t+1-sw_lo → 每行实选恰 token_budget
        - per-row far 预算 k2_far = min(far_tokens, span, far_cap)；span
          不足的行以哨兵 pad 到统一宽度（保证跨行可 stack）
        - 池不足（候选 < 配额）→ 哨兵（L2-kernel 路径语义；eager 路径
          此处会选出 -inf 垃圾实位置，批量版采用更安全的哨兵口径）
        - skip_far 层退化为单池整体 topk + 滑窗 +inf（= eager 语义）

        全程 launch 数与 bs 无关（~15 个：2 einsum + topk×3 + gather 若干），
        替代 per-request select 的 O(n) 次调用——M3-c 线性放大的根因修复。

        M5 静态宽度改造（CUDA graph 可录制的硬前提）：所有随 S 值变化的
        量改为 device 张量运算（nblk_t / far_hi / sw_lo / F / k2_far /
        k2n），topk 宽度改为**静态上界**（W_far/W_near/W_forced 常量），
        per-row 配额裁剪由 rank 掩码（device）完成——被裁剪行/池不足槽位
        落哨兵。宽度上界推导：
          W_far   = min(far_tokens, far_cap)        （k2_far 的上界）
          W_near  = token_budget                     （k2n = budget-k2_far-F 的上界）
          W_forced = sliding_window                   （F = min(t+1, sw) 的上界）
        静态宽度比动态宽度多 ~25-40% 的 -inf/哨兵 lane（softmax 前屏蔽），
        换取形状与 S 值无关。
        """
        p = self.profile
        device = q.device
        n = q.shape[0]
        kq_pool, kq_sc_pool, kq_mn_pool = pool_l["kq_q"], pool_l["kq_sc"], pool_l["kq_mn"]
        kmin_pool, kmax_pool = pool_l["kmin"], pool_l["kmax"]
        Hkv = kmin_pool.shape[2]
        H = q.shape[1]
        G = H // Hkv
        S_cap = kq_pool.shape[1]
        NBLK_CAP = kmin_pool.shape[1]
        bs = p.block_size
        nd2 = 2 * p.delta
        if torch.is_tensor(S_list):
            S_t = S_list.to(torch.long)
        else:
            S_t = torch.tensor(S_list, device=device, dtype=torch.long)
        t_t = S_t - 1
        nblk_t = (S_t + bs - 1) // bs
        # K1 静态（原 min(k1_blocks, nblk_max)：nblk_max 随 S 变化）。
        # K1 > nblk_t 的行选出垃圾块 → valid_blk 掩码 + scatter 源剔除
        # （下方既有机制），语义不变
        K1 = min(p.k1_blocks, NBLK_CAP)

        # ---- L1: 子空间块上界（批量 einsum；a=请求行 / m=块）----
        kmin_b = kmin_pool[rows]  # [n, NBLK_CAP, Hkv, d']（容量行含垃圾，靠掩码）
        kmax_b = kmax_pool[rows]
        qs = q[..., self.idx1]  # [n, H, d']
        qg = qs.clamp(min=0).reshape(n, Hkv, G, p.coarse_dim)
        qn = qs.clamp(max=0).reshape(n, Hkv, G, p.coarse_dim)
        sc1 = torch.einsum("ahgd,amhd->ahm", qg, kmax_b) + torch.einsum(
            "ahgd,amhd->ahm", qn, kmin_b
        )  # [n, Hkv, NBLK_CAP]
        blk_id = torch.arange(NBLK_CAP, device=device)
        blk_end = (blk_id + 1) * bs - 1
        valid_blk = (blk_id.view(1, -1) < nblk_t.view(-1, 1)) & (
            blk_end.view(1, -1) <= t_t.view(-1, 1)
        )
        if self.skip_far:
            near_blks = ((t_t + 1 - p.near_len) // bs).clamp(min=0)
            keep = blk_id.view(1, -1) >= near_blks.view(-1, 1)
            keep[:, : min(2, NBLK_CAP)] = True
            valid_blk = valid_blk & keep
        sc1 = sc1.masked_fill(~valid_blk.unsqueeze(1), float("-inf"))
        cand_blk = torch.topk(sc1, K1, dim=-1).indices  # [n, Hkv, K1]
        onehot = torch.zeros(n, NBLK_CAP, dtype=torch.bool, device=device)
        # 越界 / 非因果 / D' 掩蔽的垃圾块选择剔除在 scatter 源上完成——
        # 不能对 onehot 整体 & 掩码：滑窗强制块（含非对齐 S 的非因果尾块）
        # 必须保留（per-request 的 force_blks 无视 L1 因果 mask，块内 > t
        # 的位置由 L2 层 causal 掩掉；per-request 靠 K1=min(K1,nblk) +
        # -inf 排序天然规避垃圾块，批量 K1 全局统一须显式掩）
        sel_src = torch.gather(valid_blk, 1, cand_blk.reshape(n, -1))
        onehot.scatter_(1, cand_blk.reshape(n, -1), sel_src)
        f_blk = (
            t_t.view(-1, 1) // bs - torch.arange(p.sliding_blocks, device=device)
        ).clamp(min=0)
        onehot.scatter_(1, f_blk, True)

        # ---- 候选 token 位置（并集块展开 + pos < S 截断，topk-min 压实）----
        sel_mask = onehot.repeat_interleave(bs, dim=1)[:, :S_cap]  # [n, S_cap]
        pos = torch.arange(S_cap, device=device)
        cand = sel_mask & (pos.view(1, S_cap) < S_t.view(-1, 1))
        Tc = min((K1 * Hkv + p.sliding_blocks) * bs, NBLK_CAP * bs, S_cap)
        seq_m = torch.where(cand, pos.view(1, S_cap), torch.full_like(pos, S_cap))
        tok = torch.topk(seq_m, Tc, dim=-1, largest=False).values  # [n, Tc] 升序
        valid = tok < S_cap
        tok_c = tok.clamp(max=S_cap - 1)

        # ---- L2: 4bit 精筛打分（M6：uint8 格点 + scale 的 flat gather + 重建）----
        # 重建 kq_c = grid*sc+mn 与 fp32 存储版逐位一致（IEEE 同运算序），
        # 打分数值零漂移；pool 常驻显存 128→40 B/token-head
        q2 = q[..., self.idx2].reshape(n, Hkv, G, nd2).sum(2)  # [n, Hkv, nd2]
        s2 = torch.empty(n, Hkv, Tc, dtype=torch.float32, device=device)
        rows_l = rows.to(torch.long)
        d_off = torch.arange(nd2, device=device)
        h_off = torch.arange(Hkv, device=device).view(1, Hkv, 1) * nd2
        h_off_s = torch.arange(Hkv, device=device)
        chunk = max(1, row_chunk_bytes // max(Tc * Hkv * nd2 * 4, 1))
        for r0 in range(0, n, chunk):
            r1 = min(r0 + chunk, n)
            m = r1 - r0
            # 槽位 s、kv head h、维 d → r*S_cap*Hkv*nd2 + s*Hkv*nd2 + h*nd2 + d
            flat = (
                rows_l[r0:r1].view(m, 1, 1, 1) * (S_cap * Hkv * nd2)
                + tok_c[r0:r1].view(m, Tc, 1, 1) * (Hkv * nd2)
                + h_off.view(1, Hkv, 1)
                + d_off.view(1, 1, nd2)
            )
            # scale/mn：r*S_cap*Hkv + s*Hkv + h（无维偏移）
            flat_s = (
                rows_l[r0:r1].view(m, 1, 1) * (S_cap * Hkv)
                + tok_c[r0:r1].view(m, Tc, 1) * Hkv
                + h_off_s.view(1, 1, Hkv)
            )
            grid_c = kq_pool.reshape(-1)[flat.view(-1)].view(m, Tc, Hkv, nd2)
            sc_c = kq_sc_pool.reshape(-1)[flat_s.view(-1)].view(m, Tc, Hkv)
            mn_c = kq_mn_pool.reshape(-1)[flat_s.view(-1)].view(m, Tc, Hkv)
            kq_c = grid_c.float() * sc_c.unsqueeze(-1) + mn_c.unsqueeze(-1)
            s2[r0:r1] = torch.einsum("ahd,athd->aht", q2[r0:r1], kq_c)

        causal = tok <= t_t.view(-1, 1)
        SENT = S_cap

        if self.skip_far:
            # 单池口径（= per-request eager 的整体 topk + 滑窗 +inf）
            sw_lo_t = (t_t - p.sliding_window + 1).clamp(min=0)
            sc = s2.masked_fill(~(valid & causal).unsqueeze(1), float("-inf"))
            sc = sc.masked_fill((tok >= sw_lo_t.view(-1, 1)).unsqueeze(1), float("inf"))
            k = min(p.token_budget, Tc)
            i_g = torch.topk(sc, k, dim=-1).indices
            sc_g = torch.gather(sc, 2, i_g)
            sel_g = torch.gather(tok_c.unsqueeze(1).expand(n, Hkv, Tc), 2, i_g)
            return torch.where(
                sc_g == float("-inf"), torch.full_like(sel_g, SENT), sel_g
            )  # [n, Hkv, budget]

        # ---- B'：far/near 分区（L2-kernel 路径口径；M5 全程 device 张量 + 静态宽度）----
        far_lo = p.sink_blocks * bs
        far_hi_t = (t_t + 1 - p.near_len).clamp(min=far_lo)
        sw_lo_t = (t_t - p.sliding_window + 1).clamp(min=0)
        near_floor = p.sliding_window + far_lo
        far_cap = max(0, p.token_budget - near_floor)
        # per-row 配额（随 S 值变化但形状静态 [n]）：
        #   k2_far = min(far_tokens, far_span, far_cap)
        #   F      = min(t+1, sliding_window)
        #   k2n    = budget - k2_far - F（near_floor 预算保护下 ≥0）
        k2_far_t = (far_hi_t - far_lo).clamp(max=min(p.far_tokens, far_cap))
        F_t = (t_t + 1 - sw_lo_t).clamp(min=0)
        k2n_t = (p.token_budget - k2_far_t - F_t).clamp(min=0)
        # topk / 拼接宽度 = 静态上界（与 S 值无关；Tc 由 pool 容量等静态量决定）：
        #   W_far    ≤ min(far_tokens, far_cap)
        #   W_near   ≤ token_budget
        #   W_forced ≤ sliding_window
        # 被裁剪的 lane 落哨兵，softmax 前由消费方屏蔽
        W_far = min(p.far_tokens, far_cap, Tc)
        W_near = min(p.token_budget, Tc)
        W_forced = min(p.sliding_window, Tc)

        in_far = (tok >= far_lo) & (tok < far_hi_t.view(-1, 1)) & valid & causal
        in_near = valid & causal & ~in_far & (tok < sw_lo_t.view(-1, 1))
        far_sc = s2.masked_fill(~in_far.unsqueeze(1), float("-inf"))
        near_sc = s2.masked_fill(~in_near.unsqueeze(1), float("-inf"))
        tok_e = tok_c.unsqueeze(1).expand(n, Hkv, Tc)
        parts = []
        if W_far > 0:
            i_f = torch.topk(far_sc, W_far, dim=-1).indices
            sc_f = torch.gather(far_sc, 2, i_f)
            sel_f = torch.gather(tok_e, 2, i_f)
            # per-row 配额裁剪：W_far 是静态上界（统一宽度），rank ≥ 该行
            # k2_far 的槽位（topk 降序 = 低分尾部）转哨兵；-inf 槽位（池
            # 不足）同样转哨兵
            rank_f = torch.arange(W_far, device=device).view(1, 1, -1) < k2_far_t.view(
                -1, 1, 1
            )
            keep_f = (sc_f != float("-inf")) & rank_f
            parts.append(torch.where(~keep_f, torch.full_like(sel_f, SENT), sel_f))
        if W_near > 0:
            i_n = torch.topk(near_sc, W_near, dim=-1).indices
            sc_n = torch.gather(near_sc, 2, i_n)
            sel_n = torch.gather(tok_e, 2, i_n)
            rank_n = torch.arange(W_near, device=device).view(1, 1, -1) < k2n_t.view(
                -1, 1, 1
            )
            keep_n = (sc_n != float("-inf")) & rank_n
            parts.append(torch.where(~keep_n, torch.full_like(sel_n, SENT), sel_n))
        if W_forced > 0:
            f_pos = sw_lo_t.view(-1, 1) + torch.arange(W_forced, device=device)
            f_pad = torch.arange(W_forced, device=device).view(1, -1) >= F_t.view(-1, 1)
            forced = torch.where(f_pad, torch.full_like(f_pos, SENT), f_pos)
            parts.append(forced.unsqueeze(1).expand(n, Hkv, W_forced))
        return torch.cat(parts, dim=-1)  # [n, Hkv, W_far+W_near+W_forced] 静态宽度

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
