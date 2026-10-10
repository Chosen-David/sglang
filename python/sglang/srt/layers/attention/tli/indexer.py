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

import os

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.tli.config import TLIProfile


def quant4(x: torch.Tensor) -> torch.Tensor:
    """per-token 4bit 对称格点量化（TIA 语义，dequant 后的近似值）。"""
    mx = x.amax(-1, keepdim=True)
    mn = x.amin(-1, keepdim=True)
    sc = (mx - mn).clamp(min=1e-9) / 15
    return torch.clamp(torch.round((x - mn) / sc), 0, 15) * sc + mn


# ---- #64：DeepSelect 官方 topk kernel 替换（torch.topk 占 select 62%）----
_DS = None
_DS_TRIED = False


def _ds_available() -> bool:
    """deep_select 编译装在 ~/.local/pylibs（H20 sm_90a 版）；懒加载。"""
    global _DS, _DS_TRIED
    if not _DS_TRIED:
        _DS_TRIED = True
        try:
            import deep_select  # noqa: F401

            _DS = deep_select
        except Exception:
            _DS = None
    return _DS is not None


def ds_topk(x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """torch.topk(x, k, dim=-1).indices 的 DeepSelect 替代。
    x: [R, C] fp32（内部 cast bf16——4bit 量化的分数本就粗粒度，tie 容忍
    口径见 §8b-31 microbench jaccard 0.986-1.0）；返回 (idx, xp)：
    idx [R, k] int64、xp = pad 后 bf16 张量（pad 列 -inf）。
    约束：DeepSelect 要求 stride(0) 1024B 对齐（bf16 = 512 列倍数）→
    列 pad 到 512 倍数，且**恒 pad ≥ 512**（对齐时也补）——保证 clamp
    出口落在 -inf 区；k ≤ 4096。
    **坑一（30B 64K 崩溃真根因）：行有限值 < k 时阈值退化到 -inf，
    kernel 随机块序会把 pad 列的 -inf 选进输出，实测 idx 甚至可超
    C+pad（33307 > 33280）——32K 活是因 Tc_k=32768 恰为 512 倍数
    （pad=0），非真安全。故返回前统一 clamp(0, C+pad-1)**。
    **坑二：pad 区必须显式填 -inf**（torch.empty 垃圾撞 NaN bit
    pattern → abort_when_nan_found 中止 kernel → 输出不写）。
    调用方约定：只要 idx → 自行 clamp(max=C-1)；要 gather 分数 →
    从 xp（而非原 x）gather，越界位得 -inf 由下游 keep 掩码转哨兵，
    有效集与 torch.topk 逐位一致。
    """
    R, C = x.shape
    pad = (-C) % 512 or 512
    xp = torch.empty((R, C + pad), device=x.device, dtype=torch.bfloat16)
    xp[:, C:] = float("-inf")
    xp[:, :C] = x
    end = torch.full((R,), C, device=x.device, dtype=torch.int32)
    _, idx = _DS.topk(xp, k, end=end, return_value=False, indices_type=torch.int64)
    if os.environ.get("SGLANG_TLI_DS_DEBUG"):
        # 诊断插桩：NaN 输入 / kernel 退化行越界索引在这暴露（clamp 前）
        nan = int(torch.isnan(xp).sum())
        bad = (idx < 0) | (idx >= C + pad)
        nb = int(bad.sum())
        print(f"[ds_debug] R={R} C={C} pad={pad} k={k} NaN={nan} "
              f"idx_bad={nb} idx_min={int(idx.min())} idx_max={int(idx.max())}",
              flush=True)
        if nan:
            raise RuntimeError(f"ds_topk debug trip: NaN={nan}")
    # 30B 64K 实测：退化行（有限值<k、阈值=-inf）会输出 [C, C+pad) 乃至
    # > C+pad 的越界索引（idx_max 观测 33307 > 33280）——统一 clamp 到
    # pad 区（恒 -inf），下游 keep 掩码转哨兵，有效集不受影响
    idx = idx.clamp_(min=0, max=C + pad - 1)
    return idx, xp


def ds_topk_padded(xp: torch.Tensor, C: int, k: int) -> torch.Tensor:
    """#65：输入已是 pad 后 bf16（宽 C+pad，pad 列 -inf，(C+pad)%512==0，
    由 tli_l2_score_batched_dual(out_bf16_pad=) 直出）——省去 ds_topk 的
    xp 物化（末 chunk fp32→bf16 复制链 ~4GB 流量）。返回 idx [R, k]
    int64（clamp 到 pad 区；调用方从 xp gather 分数使越界位 -inf），
    有效列数 C 通过 end 传给 kernel（pad 列不参与）。"""
    R = xp.shape[0]
    end = torch.full((R,), C, device=xp.device, dtype=torch.int32)
    _, idx = _DS.topk(xp, k, end=end, return_value=False,
                      indices_type=torch.int64)
    if os.environ.get("SGLANG_TLI_DS_DEBUG"):
        nan = int(torch.isnan(xp).sum())
        bad = (idx < 0) | (idx >= xp.shape[1])
        print(f"[ds_debug_padded] R={R} C={C} W={xp.shape[1]} k={k} "
              f"NaN={nan} idx_bad={int(bad.sum())} "
              f"idx_max={int(idx.max())}", flush=True)
        if nan:
            raise RuntimeError(f"ds_topk_padded debug trip: NaN={nan}")
    idx = idx.clamp_(min=0, max=xp.shape[1] - 1)
    return idx


def _ds_pad_for(C: int) -> int:
    """DS 直出 pad 宽：使 C+pad 为 512 倍数且恒 ≥512（对齐也补）。"""
    return (-C) % 512 or 512


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
    # 存量 bug 修复（#127 单测触发）：N < K 时 randperm(N)[:K] 静默只给
    # N 个中心，bincount(minlength=K) 掩码与中心数不匹配 → IndexError。
    # clamp 后 = 少样本退化（每 token 近自成中心）；N ≥ K 的既有路径
    # （实验全部 S >> far_clusters）行为零变化。
    K = min(K, N)
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

    def __init__(
        self,
        profile: TLIProfile | None = None,
        head_dim: int = 128,
        basis: torch.Tensor | None = None,
    ) -> None:
        """basis: [Hkv, D, r] fp32（M9 PCA 投影基，离线校准）。
        给定时 L2 精筛表示从「维度选择（idx2）」切换为「PCA 投影」：
        kq 存 K@basis 的 4bit（r 维），打分维 2δ→r、存储 40→r+8 B/token-head。
        """
        self.profile = profile or TLIProfile()
        p = self.profile
        self.register_buffer_idx(torch.arange(head_dim))
        self.idx1 = torch.tensor(p.subspace_idx(head_dim), dtype=torch.long)
        self.idx2 = torch.tensor(p.refine_idx(head_dim), dtype=torch.long)
        assert basis is None or not p.far_kmeans, "PCA 投影与 kmeans 消融路径互斥"
        self.basis = basis
        self.nd2 = int(basis.shape[-1]) if basis is not None else 2 * p.delta
        self.skip_far: bool = False  # D'：由 backend 按 layer 掩码置位
        # #60：prefill 动态测层——末 chunk 统计的 per-layer far mass
        # （None = 尚未 prefill，decode 侧不置位）
        self.dyn_far_stat: float | None = None

    def register_buffer_idx(self, _):
        pass

    def to(self, device):
        self.idx1 = self.idx1.to(device)
        self.idx2 = self.idx2.to(device)
        if self.basis is not None:
            self.basis = self.basis.to(device)
        return self

    # ------------------------------------------------------------------ #
    # E112（#149）：TASK.md 口径分区选择（α/β/γ + far/near method 组合）
    # ------------------------------------------------------------------ #

    def _need_avg_score(self) -> bool:
        """avg 分数源条件（对齐 two-level tli_indexer._need_avg_score，E109a 修复版）。

        far avg：单池（α=0）与分区（α>0）的 far 池都要用 → 永远需要；
        near avg：near 池仅在分区（α>0 且 β>0）存在 → 仅分区需要。
        taskmd 关闭时恒 False（旧 B' 路径零 kavg 维护开销，回归保护）。
        """
        p = self.profile
        if not getattr(p, "taskmd", False):
            return False
        return p.far_method == "avg" or (
            p.near_method == "avg" and p.alpha > 0 and p.beta > 0
        )

    def _taskmd_regions(self, S: int) -> dict:
        """TASK.md 区域边界（host 标量版；decode per-request / 对拍用）。

        与 two-level tli_indexer.compute_mask 的边界公式逐项对齐
        （L779-799 区域 / L836-839 L1 块配额 / L984-1019 L2 mid 预算）：
        - e64_partition = α>0 且 β>0：mid = S − sink − swa；
          near_len = max(bs, α·mid)；nb_near = max(1, round(k1·β))；
          nb_far = k1 − nb_near（严格无保底，topk k=0 安全）
        - K2_mid = K2 − sink_tok − swa_tok（sink/swa 正交，不占预算）；
          nt_near = min(int(nb_near·bs·γ), K2_mid)；
          far_budget = max(0, K2_mid − nt_near)（γ 严格语义，无 64 保底）
        - 单池（α=0 或 β=0）：far_method 独占全管线，near 区边界量
          不进单池语义（仅 region 报告用，消费方不读）
        """
        p = self.profile
        bs = p.block_size
        nblk = (S + bs - 1) // bs
        sink_tok = p.sink_blocks * bs
        swa_tok = p.sliding_window
        e64 = p.alpha > 0 and p.beta > 0
        # 【B10 修复 2026-10-10】e64 分区臂 near 左界从 swa 起点推
        # （near 区实际宽 = α·mid_len，TASK.md L137 口径；旧口径从序列末尾
        # 推 → swa 被扣两次）；单池 (α=0 或 β=0) 分支保持旧式逐位不变。
        if e64:
            mid_len = max(0, S - sink_tok - swa_tok)
            near_len_dyn = max(bs, int(p.alpha * mid_len))
            near_base = S - swa_tok
        else:
            near_len_dyn = swa_tok
            near_base = S
        near_blks = max(p.sink_blocks, (near_base - near_len_dyn) // bs)
        far_lo_blk, far_hi_blk = p.sink_blocks, near_blks
        # near 池上界 = swa 起点（块级）：swa 块完全排除出双池，纯靠输出强制
        swa_lo_blk = max(near_blks, nblk - max(1, swa_tok // bs))
        swa_lo_tok = max(0, S - swa_tok)
        nb_near = nb_far = nt_near = far_budget = K2_mid = 0
        if e64:
            nb_near = max(1, int(round(p.k1_blocks * p.beta)))
            nb_far = max(0, p.k1_blocks - nb_near)
            K2_mid = max(0, p.token_budget - sink_tok - (S - swa_lo_tok))
            nt_near = min(int(nb_near * bs * p.gamma), K2_mid)
            far_budget = max(0, K2_mid - nt_near)
        return {
            "e64": e64,
            "nblk": nblk,
            "sink_tok": sink_tok,
            "swa_tok": swa_tok,
            "near_blks": near_blks,
            "far_lo_blk": far_lo_blk,
            "far_hi_blk": far_hi_blk,
            "swa_lo_blk": swa_lo_blk,
            "swa_lo_tok": swa_lo_tok,
            "nb_near": nb_near,
            "nb_far": nb_far,
            "nt_near": nt_near,
            "far_budget": far_budget,
            "K2_mid": K2_mid,
        }

    # ---- M9：L2 精筛表示（选择 idx2 / PCA 投影，统一出口）----

    def _k_refine(self, k: torch.Tensor) -> torch.Tensor:
        """k: [..., Hkv, D] → [..., Hkv, nd2]（写入 kq 前的精筛表示）。"""
        if self.basis is None:
            return k[..., self.idx2]
        return torch.einsum("...hd,hdr->...hr", k, self.basis)

    def _q_refine(self, q: torch.Tensor) -> torch.Tensor:
        """q: [..., H, D] → [..., H, nd2]（逐 q-head 精筛表示；GQA sum
        由调用方 reshape(..., Hkv, G, nd2).sum(2) 完成——投影线性，
        先投影后求和 == 先求和后投影）。"""
        if self.basis is None:
            return q[..., self.idx2]
        Hkv, D, r = self.basis.shape
        G = q.shape[-2] // Hkv
        # 注意 einsum 标签：head 维必须出现在输出（hgd,hdr->hgr），
        # 否则 h 变归约维 = 对全部 head 的基求和（曾踩：q 路径全错）
        qp = torch.einsum(
            "...hgd,hdr->...hgr", q.reshape(*q.shape[:-2], Hkv, G, D), self.basis
        )
        return qp.reshape(*q.shape[:-1], r)

    def _pad_refine_full(self, x: torch.Tensor, D: int) -> torch.Tensor:
        """E112 对拍口径：[..., nd2] → [..., D]（refine 维零填充回全维）。

        two-level 权威实现的 k_qat/b_q_full 是全 D 维张量（refine 外全 0），
        其 L2 精筛 einsum 沿 128 维归约；直接用 nd2=32 维操作数做 einsum
        的归约树与之不同（实测 ulp 级 max|Δ|≈2.4e-7，会在 topk 边界翻转
        近并列 token → 选择集合不一致）。零填充回全维后逐位一致
        （D4 对拍 0/65536 不等）。eager 路径正确性优先；M8/M10 fused
        kernel 侧的逐位对齐另行处理（kernel 本身不做维数选择展开）。
        投影基（basis）路径 nd2=r 为 PCA 投影维而非 D 的子集，本方法
        仅在 basis is None 时被调用。
        """
        assert self.basis is None, "投影路径无全维零填充口径"
        shape = list(x.shape)
        shape[-1] = D
        out = torch.zeros(shape, dtype=x.dtype, device=x.device)
        out[..., self.idx2] = x
        return out

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
        # M9：basis 存在时存 PCA 投影 K@basis 的 4bit（r 维，16+8 B/token-head）
        kq_q, kq_sc, kq_mn = quant4_pack(self._k_refine(k[:S]))  # [S, Hkv, nd2] uint8 + [S, Hkv] ×2
        # E112：avg method 分数源（块 token 和；pad 零不贡献和）。
        # 消费端 /bs 即 two-level 的 padded-mean 口径（k_coarse.mean(dim=2)
        # 分母恒 bs）——存和而非均值是为了增量维护的精确性（尾块累加）。
        kavg_sum = kc.sum(1) if self._need_avg_score() else None  # [nblk, Hkv, d']
        index = {
            "kmin": kmin,
            "kmax": kmax,
            "kq_q": kq_q,
            "kq_sc": kq_sc,
            "kq_mn": kq_mn,
            "kavg_sum": kavg_sum,
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

        kq_q_new, kq_sc_new, kq_mn_new = quant4_pack(self._k_refine(k_new))
        index["kq_q"] = _ensure(index["kq_q"], S)
        index["kq_sc"] = _ensure(index["kq_sc"], S)
        index["kq_mn"] = _ensure(index["kq_mn"], S)
        index["kq_q"][S_old:S] = kq_q_new
        index["kq_sc"][S_old:S] = kq_sc_new
        index["kq_mn"][S_old:S] = kq_mn_new
        ks = k_new[..., self.idx1]  # [n, Hkv, d']
        Hkv = ks.shape[1]
        nblk_old = index["nblk"]
        # E112：kavg_sum 增量维护（taskmd + avg method 时存在；
        # 全量 build 的尾块和含零 pad 贡献 = 真实 token 和，故增量累加口径一致）
        _kavg_on = index.get("kavg_sum") is not None

        def _append_blocks(ks_seg: torch.Tensor) -> None:
            """把 ks_seg（从全局位置 s0 起）的块界精确 append（无零 pad 污染）。"""
            n = ks_seg.shape[0]
            nb_full = n // p.block_size
            n_add = nb_full + (1 if n % p.block_size else 0)
            if n_add == 0:
                return
            index["kmin"] = _ensure(index["kmin"], nblk_old + n_add)
            index["kmax"] = _ensure(index["kmax"], nblk_old + n_add)
            if _kavg_on:
                index["kavg_sum"] = _ensure(index["kavg_sum"], nblk_old + n_add)
            w = nblk_old
            if nb_full:
                kc = ks_seg[: nb_full * p.block_size].reshape(
                    nb_full, p.block_size, Hkv, p.coarse_dim
                )
                index["kmin"][w : w + nb_full] = kc.amin(1)
                index["kmax"][w : w + nb_full] = kc.amax(1)
                if _kavg_on:
                    index["kavg_sum"][w : w + nb_full] = kc.sum(1)
                w += nb_full
            rem = n - nb_full * p.block_size
            if rem:
                # 部分尾块：界 = rem 个 token 的精确 min/max（与 build 的
                # 尾块精确界口径一致，保证 增量 == 全量重建 逐位成立）
                r = ks_seg[nb_full * p.block_size :]
                index["kmin"][w] = r.amin(0)
                index["kmax"][w] = r.amax(0)
                if _kavg_on:
                    index["kavg_sum"][w] = r.sum(0)

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
            if _kavg_on:
                index["kavg_sum"][nblk_old - 1] = (
                    index["kavg_sum"][nblk_old - 1] + ks[:take].sum(0)
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
        # E112（#149）：TASK.md 模式（α/β/γ + method 组合）→ 专用路径。
        # 该模式旁路 L1/L2 fused kernel（avg 分数源 + sink 正交语义超出
        # kernel 出口径）与旧 B' 分区预算；D' skip_far 未集成（E111 层
        # gate 合并主树后再接，此处不消费）。
        if getattr(self.profile, "taskmd", False):
            return self._select_taskmd(index, q, t)
        p = self.profile
        # #60 D' 动态 gate：decode 侧按 prefill 统计的 per-layer far mass
        # 置 skip_far（e60：prefill→decode corr 0.86-0.99，无静态掩码
        # 泛化假设）。幂等置位，静态掩码存在时优先动态口径（阈值可关）。
        if getattr(p, "dyn_far_gate", False) and self.dyn_far_stat is not None:
            self.skip_far = self.dyn_far_stat < p.dyn_far_thresh
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
            _skip_src = None  # skip_far：topk 结果的 rank 截断源（见下）
            if self.skip_far:
                near_blks = max(1, (t + 1 - 2048) // p.block_size)
                keep = torch.zeros(nblk, dtype=torch.bool, device=device)
                keep[: min(2, nblk)] = True
                keep[max(0, near_blks) :] = True
                sc1 = sc1.masked_fill(~keep.view(1, 1, -1), float("-inf"))
                # F5（#129）：原版 K1 = min(K1, max(1, int(keep.sum().item())))
                # 是 GPU 同步。改法与 select_batched 的 skip_far 分支同构：
                # K1 保持静态（不收缩——t 小时 keep=全部块），旧动态宽
                # w = min(K1, keep 计数) 用 device 标量 rank 截断复刻（单行
                # 无 sub-max 泄漏，rank < w 的截断集 = 旧 topk(w) 选择集）。
                # 2048 是本分支的硬编码近端长（保留原口径不动）。
                _skip_src = (
                    torch.arange(K1, device=device).view(1, -1)
                    < keep.sum().clamp(min=1, max=K1)
                ).expand(Hkv, K1)  # [Hkv, K1]（device 截断，零同步）

            cand_blk = torch.topk(sc1, K1, dim=-1).indices[0]  # [Hkv, K1]（不含滑窗）
            cand_blk = torch.cat(
                [cand_blk, force_blks.unsqueeze(0).expand(Hkv, -1)], dim=1
            )  # 重复无碍（mask 化）
            blk_onehot = torch.zeros(Hkv, nblk, dtype=torch.bool, device=device)
            if _skip_src is not None:
                # 滑窗强制块恒 True；topk 段按 rank 截断（见上注释）
                src = torch.cat(
                    [
                        _skip_src,
                        torch.ones(
                            force_blks.shape[0], dtype=torch.bool, device=device
                        ).unsqueeze(0).expand(Hkv, -1),
                    ],
                    dim=1,
                )
                blk_onehot.scatter_(1, cand_blk, src)
            else:
                blk_onehot.scatter_(1, cand_blk, True)

        # ---- L2: 4bit 部分维 token 精筛 ----
        # fused 分区 kernel（单 launch/head：精筛分数 + far/near 分区 topk +
        # 滑窗强制 + 直接写位置）。D' 跳层 / far 区为空 / q_agg=max（kernel
        # 内 group-sum，max 聚合不支持，bug2 修复新增旁路条件）时退回 eager。
        q_agg_max = getattr(p, "q_agg", "sum") == "max"
        if use_l2_kernel and not self.skip_far and not q_agg_max:
            far_tok_lo = p.sink_blocks * p.block_size
            far_tok_hi = max(far_tok_lo, t + 1 - p.near_len)
            if far_tok_hi > far_tok_lo:
                sel_mask = blk_onehot.any(0).repeat_interleave(p.block_size)[:S]
                cand_pos = torch.nonzero(sel_mask).squeeze(1)
                if cand_pos.numel() > 0:
                    from sglang.srt.layers.attention.tli.kernels import (
                        tli_l2_partition_topk,
                    )

                    nd2 = self.nd2
                    q_sub = self._q_refine(q).reshape(H, nd2).contiguous()
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

        nd2 = self.nd2
        kq = index["kq_q"]  # M6 uint8 格点 [S, Hkv, nd2]（容量 padding 靠 cand_pos < S 规避）
        # bug2 修复（GPT 复查 2026-10-08）：q_agg=max 原在 decode per-request
        # 路径未生效（此处写死 sum，仅 prefill L1003/L1042/L1211 有分支）。
        # max = 逐 q-head 打分取组内 max（#58 30B G=8 投影符号冲突时
        # sum 互相抵消 → 打分失真）。
        q_g = self._q_refine(q).reshape(1, Hkv, G, nd2)  # [1, Hkv, G, nd2]
        sel_mask = blk_onehot.any(0).repeat_interleave(p.block_size)[:S]
        cand_pos = torch.nonzero(sel_mask).squeeze(1)
        kq_h = kq_unpack(
            kq[cand_pos], index["kq_sc"][cand_pos], index["kq_mn"][cand_pos]
        )  # [Tc, Hkv, nd2]（逐位 == fp32 存储版）
        if q_agg_max:
            s2 = torch.einsum("hgd,thd->hgt", q_g[0], kq_h).max(1).values  # [Hkv, Tc]
        else:
            s2 = torch.einsum("hd,thd->ht", q_g.sum(2)[0], kq_h)  # [Hkv, Tc]
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
    def _select_taskmd(self, index: dict, q: torch.Tensor, t: int) -> torch.Tensor:
        """E112（#149）：TASK.md α/β/γ 分区 + far/near method 组合（per-request）。

        口径与 two-level tli_indexer.compute_mask（two-level-indexer 分支
        54d93a4c6，avg/minmax method 组合子集）逐项对齐：
        - e64_partition = α>0 且 β>0：L1 far/near 双池（far 池分数源 =
          far_method，near 池分数源 = near_method；near 池上界 swa_lo_blk，
          swa 块纯靠输出强制）+ L2 mid 预算按 γ 切分（nt_near =
          nb_near·bs·γ，far_budget = K2_mid − nt_near，严格无保底）
        - 单池（α=0 或 β=0）：far_method 独占全管线（L1 单池 topk +
          当前块强制 inf + L2 整行 topk + swa 强制）；sink 走管线竞争
          （two-level 单池语义，sink 无正交置位）
        - sink/swa 正交强制区（e64 路径）：不进任何池、不占预算，
          输出显式拼接（two-level topk_mask[..., :sink]=True; [..., swa:]=True）
        - 池不足槽位 → 哨兵 S（下游 _sparse_attn_batched 的
          valid = sel < seq_len 统一屏蔽）
        - 输出恒宽 token_budget（e64：far_budget+nt_near+sink+swa = K2，
          decode t = S-1 > dense_threshold 保证 swa 满宽；单池：K2）
          ——M5 CUDA graph 静态宽前提
        - M8/M10 fused kernel 旁路：avg 分数源 + sink 正交语义超出
          kernel 出口径，eager 保正确性（性能优化属后续工作）
        - GQA 聚合差异（固有，非对齐项）：本实现 L1 avg 分数 = group-sum(q)
          ·kavg（≡ two-level group-mean 的 G 倍，topk 等价）；L2 精筛 =
          group-sum(q2)·kq（two-level 为 softmax 后 group-mean——G=1 时
          softmax 单调变换使 topk 逐位等价，G>1 存在固有聚合口径差异）
        """
        p = self.profile
        S = index["S"]
        Hkv = index["kmin"].shape[1]
        H = q.shape[1]
        G = H // Hkv
        device = q.device
        nblk = index["nblk"]
        bs = p.block_size
        kmin, kmax = index["kmin"][:nblk], index["kmax"][:nblk]
        r = self._taskmd_regions(S)
        e64 = r["e64"]
        sink_tok, swa_tok = r["sink_tok"], r["swa_tok"]
        near_blks, far_lo_blk, far_hi_blk = (
            r["near_blks"], r["far_lo_blk"], r["far_hi_blk"]
        )
        swa_lo_blk, swa_lo_tok = r["swa_lo_blk"], r["swa_lo_tok"]

        # ---- L1 分数：minmax 块上界 + 可选 avg 块均值 ----
        qs = q[..., self.idx1]  # [1, H, d']
        qg = qs.clamp(min=0).reshape(1, Hkv, G, p.coarse_dim)
        qn = qs.clamp(max=0).reshape(1, Hkv, G, p.coarse_dim)
        sc1 = (
            torch.einsum("bhgd,nhd->bhgn", qg, kmax)
            + torch.einsum("bhgd,nhd->bhgn", qn, kmin)
        ).sum(-2)  # [1, Hkv, nblk]
        blk_end = (torch.arange(nblk, device=device) + 1) * bs - 1
        causal_blk = blk_end <= t  # [nblk]
        sc1 = sc1.masked_fill(~causal_blk.view(1, 1, -1), float("-inf"))
        sc_avg = None
        if self._need_avg_score():
            kavg = index["kavg_sum"][:nblk] / bs  # [nblk, Hkv, d']（padded-mean 口径）
            qa = qs.reshape(1, Hkv, G, p.coarse_dim).sum(2)  # [1, Hkv, d']
            sc_avg = torch.einsum("ahd,nhd->ahn", qa, kavg)
            sc_avg = sc_avg.masked_fill(~causal_blk.view(1, 1, -1), float("-inf"))

        if e64:
            # ---- E64 L1 双池：far/near 独立块配额（严格无保底）----
            nb_near, nb_far = r["nb_near"], r["nb_far"]
            # far 池分数源：far_method（avg=块均值 / minmax=块上界）
            src_far = sc_avg if (p.far_method == "avg" and sc_avg is not None) else sc1
            sc_far = src_far[..., far_lo_blk:far_hi_blk]
            i_f = (
                torch.topk(sc_far, min(nb_far, sc_far.shape[-1]), dim=-1).indices
                + far_lo_blk
            )  # [1, Hkv, nb_far']
            # near 池分数源：near_method；上界 swa_lo_blk（swa 块排除出双池）
            src_near = (
                sc1 if (p.near_method == "minmax" or sc_avg is None) else sc_avg
            )
            sc_near = src_near[..., near_blks:swa_lo_blk]
            if sc_near.shape[-1] > 0:
                i_n = (
                    torch.topk(
                        sc_near, min(nb_near, sc_near.shape[-1]), dim=-1
                    ).indices
                    + near_blks
                )  # [1, Hkv, nb_near']
            else:
                i_n = i_f[..., :0]  # near 池空（mid 不足一块）→ 不选
            blk_onehot = torch.zeros(Hkv, nblk, dtype=torch.bool, device=device)
            blk_onehot.scatter_(
                1, torch.cat([i_f[0], i_n[0]], dim=-1), True
            )
        else:
            # ---- 单池：far_method 独占全管线（E109a 修复口径）----
            sc_pool = (
                sc1 if (p.far_method != "avg" or sc_avg is None) else sc_avg
            )
            sc_pool = sc_pool.clone()
            # TIA 语义：当前块强制（two-level sc_pool[..., -1]=inf 的 t 精确版；
            # 因果 mask 已打 -inf 的非对齐尾块由此恢复强制）
            sc_pool[..., t // bs] = float("inf")
            k1e = min(p.k1_blocks, nblk)
            i_pool = torch.topk(sc_pool, k1e, dim=-1).indices[0]  # [Hkv, k1e]
            blk_onehot = torch.zeros(Hkv, nblk, dtype=torch.bool, device=device)
            blk_onehot.scatter_(1, i_pool, True)

        # ---- L2：4bit 部分维精筛（与旧 eager 路径同 einsum 口径）----
        nd2 = self.nd2
        kq = index["kq_q"]
        # bug2 修复（GPT 复查 2026-10-08）：q_agg=max 原在 decode per-request
        # taskmd 路径未生效（写死 sum）。max = 逐 q-head 打分取组内 max
        # （#58，与 select() / prefill L1042/L1211 同款分支）。
        q_g = self._q_refine(q).reshape(1, Hkv, G, nd2)  # [1, Hkv, G, nd2]
        q_agg_max = getattr(p, "q_agg", "sum") == "max"
        q2 = None if q_agg_max else q_g.sum(2)  # [1, Hkv, nd2]
        sel_mask = blk_onehot.any(0).repeat_interleave(bs)[:S]
        cand_pos = torch.nonzero(sel_mask).squeeze(1)
        kq_h = kq_unpack(
            kq[cand_pos], index["kq_sc"][cand_pos], index["kq_mn"][cand_pos]
        )  # [Tc, Hkv, nd2]
        if self.basis is None:
            # E112 对拍口径：L2 einsum 零填充回全 D 维（归约树与
            # two-level 的全维 k_qat 一致 → 分数逐位同；32 维直接归约
            # 有 ulp 差异会在 topk 边界翻转近并列 token，见 _pad_refine_full）
            kq_full = self._pad_refine_full(kq_h, q.shape[-1])  # [Tc, Hkv, D]
            if q_agg_max:
                q_gf = self._pad_refine_full(q_g, q.shape[-1])  # [1, Hkv, G, D]
                s2 = torch.einsum(
                    "hgd,thd->hgt", q_gf[0], kq_full
                ).max(1).values  # [Hkv, Tc]
            else:
                q2f = self._pad_refine_full(q2, q.shape[-1])  # [1, Hkv, D]
                s2 = torch.einsum("hd,thd->ht", q2f[0], kq_full)  # [Hkv, Tc]
        else:
            if q_agg_max:
                s2 = torch.einsum("hgd,thd->hgt", q_g[0], kq_h).max(1).values
            else:
                s2 = torch.einsum("hd,thd->ht", q2[0], kq_h)  # [Hkv, Tc]
        fine = torch.full((Hkv, S), float("-inf"), device=device)
        fine[:, cand_pos] = s2
        fine = fine.masked_fill(
            torch.arange(S, device=device).view(1, S) > t, float("-inf")
        )
        # E112 per-head L1 掩码：two-level 的 score_fine 被本 head 自己的
        # L1 topk_mask 掩码后才进 softmax/L2 池；blk_onehot.any(0) 的跨
        # head 并集只是 gather 效率优化——不掩回 per-head 语义时，其它
        # head 选中的块会带有效分进入本 head 的 L2 池竞争（far 池非空
        # 且 L1 有绑定的配置集合发散）。单池路径的 swa +inf 强制在其后
        # 覆盖此掩码（two-level p[..., -swa:]=1.0 同样不受 L1 限制）
        fine = fine.masked_fill(
            ~blk_onehot.repeat_interleave(bs, dim=-1)[:, :S], float("-inf")
        )
        if e64:
            # ---- E64 L2：mid 预算按 γ 切分（sink/swa 正交不占预算）----
            K2_mid, nt_near, far_budget = r["K2_mid"], r["nt_near"], r["far_budget"]
            far_tok_lo = sink_tok
            far_tok_hi = min(near_blks * bs, S)
            k2_far = min(far_budget, far_tok_hi - far_tok_lo, K2_mid)
            # S3 修复（Kimi3 审查 2026-10-08）：池不足时 topk 的 -inf 槽位
            # 是池内真实位置（< S）→ 消费端判有效并以真实 logit 计入；
            # 批量版用 keep = sc > -inf 转哨兵——本处补齐同款口径，
            # n==1 与 n≥2 decode 语义统一。
            f2_r = fine[:, far_tok_lo:far_tok_hi]
            i_f2_raw = torch.topk(f2_r, k2_far, dim=-1).indices
            sc_f2 = torch.gather(f2_r, 1, i_f2_raw)
            i_f2 = torch.where(
                sc_f2 > float("-inf"), i_f2_raw + far_tok_lo, S
            )  # [Hkv, k2_far]（k2_far=0 时 topk 空，已验证安全）
            k2_near = max(0, K2_mid - k2_far)
            near_f = fine[:, far_tok_hi:swa_lo_tok]
            k2n_sel = min(k2_near, near_f.shape[-1])
            i_n2_raw = torch.topk(near_f, k2n_sel, dim=-1).indices
            sc_n2 = torch.gather(near_f, 1, i_n2_raw)
            i_n2 = torch.where(
                sc_n2 > float("-inf"), i_n2_raw + far_tok_hi, S
            )  # [Hkv, ≤k2_near]（-inf 槽位转哨兵；宽度不足由下 w_pad 补）
            # 哨兵 pad：i_n2 欠宽槽位 → S（下游 valid 掩码屏蔽；
            # two-level 同位是 softmax 后 p=0 的垃圾 True 位，注意力权重 0
            # ≈ 不选，集合语义一致）
            w_pad = k2_near - i_n2.shape[-1]
            if w_pad > 0:
                i_n2 = torch.cat(
                    [
                        i_n2,
                        torch.full(
                            (Hkv, w_pad), S, dtype=i_n2.dtype, device=device
                        ),
                    ],
                    dim=-1,
                )
            # 正交强制区显式拼接（必选不占预算）：sink 头部 + swa 尾部
            # （decode t = S-1 > dense_threshold 保证 swa 段满宽）
            sink_pos = torch.arange(sink_tok, device=device).unsqueeze(0).expand(Hkv, -1)
            swa_pos = (
                torch.arange(max(0, t - swa_tok + 1), t + 1, device=device)
                .unsqueeze(0)
                .expand(Hkv, -1)
            )
            return torch.cat([i_f2, i_n2, sink_pos, swa_pos], dim=-1)  # [Hkv, K2]
        # 单池 L2：swa 强制 + 整行 topk（sink 走管线竞争）
        forced = torch.arange(max(0, t - swa_tok + 1), t + 1, device=device)
        fine[:, forced] = float("inf")
        return torch.topk(fine, min(p.token_budget, S), dim=-1).indices  # [Hkv, K2]

    @torch.no_grad()
    def select_batched(
        self,
        index: dict,
        q: torch.Tensor,
        t_arr: torch.Tensor,
        row_chunk: int = 64,
        t_min_hint: int | None = None,
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

        M7 快路径：真实数据下 K1=128 块 × Hkv 个 head 的并集在 S ≲ 8K*Hkv
        时覆盖全部因果 token（实测 S=10K 时 Tc==S）——此时逐 chunk 的
        「掩码→topk 提取候选→gather→反量化→scatter 回 S 宽度」全是绕路，
        直接对全宽 kq 反量化表做一次 einsum 即得同一 fine 矩阵（池==因果区
        时 scatter 版语义逐位等价）。反量化表 [S,Hkv,nd2] 每次调用建一张、
        全部 chunk 共享（S=10K 仅 10MB；131K 134MB 瞬态）。prefill 归因
        （M7 微基准）：S=10K 时 select 865→~160 ms/层，瓶颈占比从 ~95%
        降到次要项。
        """
        # E112（#149）：TASK.md 模式（α/β/γ + method 组合）→ 专用路径
        # （eager，M7 快路径/M10 kernel 旁路——avg 分数源 + sink 正交 +
        # 双池 L1 语义超出现有出口径）
        if getattr(self.profile, "taskmd", False):
            return self._select_batched_taskmd(
                index, q, t_arr, row_chunk=row_chunk, t_min_hint=t_min_hint
            )
        p = self.profile
        # #60 修复跨请求锁死 bug（2026-09-27）：动态 gate 语义 = prefill 测层
        # / decode 跳 far。此前 skip_far 由 decode 的 select() 置位后跨请求
        # 残留 → 本条 prefill 走 skip_far 分支（far 区为空）→ 统计代码在
        # not-skip_far 分支内永不执行 → dyn_far_stat 冻结在低值 → skip_far
        # 永真 → 自增强锁死（E5b musique 第 8 条起 on 臂全空输出的根因）。
        # 修法：动态 gate 打开时 prefill 入口强制重置（静态掩码口径由动态
        # 优先的设计覆盖，见 select() 幂等置位注释）。
        if getattr(p, "dyn_far_gate", False):
            self.skip_far = False
        S = index["S"]
        Hkv = index["kmin"].shape[1]
        Nq, H, D = q.shape
        G = H // Hkv
        device = q.device
        # M10：kernel 慢路径不物化 kq_c（寄存器内 dequant）→ row_chunk 的
        # 显存约束解除，放大到 512 摊销 launch/topk 固定项（247.5→145.8ms
        # 实测@30K 末chunk）。eager 回退罕见（Hkv 非二次幂/不连续）不保护
        rc = row_chunk
        if (
            getattr(p, "use_prefill_kernel", False)
            and not self.skip_far
            and Nq >= 2
            and (Hkv & (Hkv - 1)) == 0
            and (self.nd2 & (self.nd2 - 1)) == 0
        ):
            rc = max(row_chunk, 512)
        kmin, kmax = index["kmin"], index["kmax"]
        nblk = index["nblk"]
        kmin, kmax = kmin[:nblk], kmax[:nblk]  # 容量 padding 垃圾行切除
        K1 = min(p.k1_blocks, nblk)
        nd2 = self.nd2
        bs = p.block_size
        kq = index["kq_q"]  # M6 uint8 格点（tok_c < S，容量 padding 无害）
        t_arr = t_arr.to(device)
        pos = torch.arange(S, device=device)
        # M7：共享反量化表（快路径直接 einsum；慢路径 gather 打分也用）。
        # F3（#127）：惰性构建——M10 kernel 路径（寄存器内 dequant）不
        # 需要全宽 fp32 表（[S,Hkv,nd2]，131K 时 ~134MB/层/调用纯白付），
        # 只在 fast path / eager 慢路径 / kernel 路径 empty 行兜底首次
        # 实际使用时才物化；多次调用同一表仍只建一次。
        kq_f_l: list[torch.Tensor | None] = [None]

        def _kq_f() -> torch.Tensor:
            if kq_f_l[0] is None:
                kq_f_l[0] = kq_unpack(kq[:S], index["kq_sc"][:S], index["kq_mn"][:S])
            return kq_f_l[0]

        out = []
        for r0 in range(0, Nq, rc):
            r1 = min(r0 + rc, Nq)
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
            _skip_rank_src = None  # skip_far：topk 结果的 rank 截断源（见下）
            if self.skip_far:
                near_blks = ((t_c + 1 - p.near_len) // bs).clamp(min=0)
                keep = torch.arange(nblk, device=device).view(1, -1) >= near_blks.view(-1, 1)
                keep[:, : min(2, nblk)] = True
                sc1 = sc1.masked_fill(~keep.unsqueeze(1), float("-inf"))
                # F5（#129，host 同步消除）：原版 K1 = min(K1, max(1,
                # int(keep.sum(1).max().item()))——每 chunk 一次 GPU 同步
                # （#126 ⑥）。改法：K1 保持静态（min(k1_blocks, nblk)，
                # 不收缩——t 小时 keep=全部块，收缩上界反而小于旧宽），
                # 旧动态宽 w = min(K1, max(1, keep 计数最大值)) 改用 device
                # 标量作 rank 截断：rank < w 的槽位 scatter True，其余
                # False——复刻旧版 topk(w) 的前缀选择集（含旧版 sub-max
                # 行的 -inf 泄漏块——忠实保留而非"修正"，泄漏块也是旧
                # D' 臂口径的一部分）。tie 风险：截断区 -inf 并列的
                # torch.topk sorted 输出按索引升序（实测稳定），
                # test_async_prefill.py 有 skip_far 对拍兜底。DS 路径
                # （use_ds_topk）的前缀性质未验证，默认关。
                _skip_rank_src = (
                    torch.arange(K1, device=device).view(1, 1, -1)
                    < keep.sum(1).max().clamp(min=1, max=K1)
                ).expand(n, Hkv, K1)  # [n, Hkv, K1]（广播 device 截断）
            # #64：L1 块 topk → DeepSelect（torch.topk 占 select CUDA 62%）。
            # bf16 cast 的 tie 翻转由候选池并集语义吸收（onehot 展开多选无害）
            if getattr(p, "use_ds_topk", False) and _ds_available():
                n_hkv, nb = sc1.shape[0] * sc1.shape[1], sc1.shape[2]
                # clamp：首 chunk 早段行 finite<K1 时 DS 会选进 pad 列
                # （-inf 垃圾块位），clamp 到末块语义同 torch.topk 选
                # in-range -inf——并集多一块由 fast path 吸收
                cand_blk = ds_topk(
                    sc1.reshape(n_hkv, nb), K1
                )[0].clamp(max=nb - 1).reshape(
                    sc1.shape[0], sc1.shape[1], K1
                )
            else:
                cand_blk = torch.topk(sc1, K1, dim=-1).indices  # [n, Hkv, K1]

            # ---- 候选块并集（select 的 blk_onehot.any(0) 语义）----
            # topk 块 ∪ 滑窗块（per row），跨 head 展开为共享 token 池
            onehot = torch.zeros(n, nblk, dtype=torch.bool, device=device)
            if _skip_rank_src is not None:
                # skip_far：rank 截断源（复刻旧动态宽度选择集，见上注释）
                onehot.scatter_(
                    1, cand_blk.reshape(n, -1), _skip_rank_src.reshape(n, -1)
                )
            else:
                onehot.scatter_(1, cand_blk.reshape(n, -1), True)
            f_blk = (
                t_c.view(-1, 1) // bs - torch.arange(p.sliding_blocks, device=device)
            ).clamp(min=0)  # 块级滑窗（select 的 force_blks 同语义）
            onehot.scatter_(1, f_blk, True)
            sel_mask = onehot.repeat_interleave(bs, dim=1)[:, :S]  # [n, S]
            # #58 GQA 聚合：sum（8B G=4 校准）vs max（30B G=8 投影符号冲突
            # 时 sum 互相抵消 → 打分失真）。max = 逐 q-head 打分取组内 max。
            q_agg_max = getattr(p, "q_agg", "sum") == "max"
            q_g = self._q_refine(q_c).reshape(n, Hkv, G, nd2)  # [n, Hkv, G, nd2]
            q2 = q_g.sum(2) if not q_agg_max else None  # [n, Hkv, nd2]
            far_lo_k = p.sink_blocks * bs  # M10 慢路径 kernel 分支提前用（原版在 B' 段定义）
            # M7 快路径判定：池 ⊇ 因果区（池 ⊆ 因果区恒成立 → 即 ==）。
            # 真实数据 S ≲ K1*bs*Hkv 量级时并集全选，Tc==S，scatter 是纯绕路。
            # M10 修正认知：nblk > K1（S ≳ 8K×Hkv/K1…即 S>~32K/Hkv 边界附近）
            # 后每 head 仅选 nblk 的 K1/nblk 比例块，并集 < 100% → 慢路径
            # 必然触发（30K 实测末 chunk 128/128 慢路径，q 是否真实无关）
            fast_path = False
            # #65：kernel 路径可用时不再走 eager fast path——Tc==S 是 M10
            # 慢路径的特例（候选并集=全块，Tc_k==S，kernel 语义等价），而
            # kernel+DS（bf16 直出）已快于 eager 的全宽 einsum + 2×torch.topk
            # （首 chunk 实测 67ms vs 慢路径 ~24ms 量级）。dyn_far_gate 统计
            # 仅 eager 路径有 → 开 gate 时保留 fast path。
            # （同步坑：fast_path 判定的 .all().item() 是 GPU 同步——生产
            # 每请求每层每 forward 一次 × 16 req × 36 层 × 60 forward ≈ 3.5 万
            # 次队列排空；_kern_ok 时直接跳过判定）
            _kern_ok = (
                getattr(p, "use_prefill_kernel", False)
                and not self.skip_far
                and not q_agg_max
                and n >= 2
                and (Hkv & (Hkv - 1)) == 0
                and (nd2 & (nd2 - 1)) == 0
                and index["kq_q"].is_contiguous()
                and not getattr(p, "dyn_far_gate", False)
            )
            if not _kern_ok:
                # F5 豁免说明：此处 .all() → bool 是 GPU 同步，但仅当 kernel
                # 路径不可用（_kern_ok=False：dyn_far_gate 开 / q_agg=max /
                # Hkv 或 nd2 非二次幂 / n<2 / kq 非连续）才执行——生产默认
                # 配置（use_prefill_kernel=True，mavg 臂 q_agg=sum、gate 关）
                # 恒走 _kern_ok，此同步零触发。fast/slow 两路径计算结构
                # 不同（全宽 einsum vs 候选 gather），无法用 device 掩码
                # 无条件合并（须双算两遍才可消除，负收益），故保留分支同步
                # 并以 _kern_ok 短路守卫。
                fast_path = bool((sel_mask.sum(1) >= t_c + 1).all())
            if fast_path:
                if q_agg_max:
                    fine = torch.einsum(
                        "ahgd,shd->ahgs", q_g, _kq_f()
                    ).max(2).values  # [n, Hkv, S]
                else:
                    fine = torch.einsum("ahd,shd->ahs", q2, _kq_f())  # [n, Hkv, S]
                causal_full = pos.view(1, S) <= t_c.view(-1, 1)  # [n, S]
                fine = fine.masked_fill(~causal_full.unsqueeze(1), float("-inf"))
            elif _kern_ok:
                # M10：慢路径 kernel 化（M8 decode 侧全套移植，t_t → per-row
                # t_c，pool 以 [1,...] 视图 + rows=0 寻址）。消除三处 eager 大头：
                # ①seq_m where [n,S] + topk 全排序 → tli_compact 块展开；
                # ②kq_f[tok_c] gather + einsum → KernelC fused gather+dequant+GEMV
                #   双池直写；③fine [n,Hkv,S] 物化 + masked_fill 链 → 池表
                # [n,Hkv,Tc]。B01 修复后：无效槽位保留哨兵=S（下游
                # _sparse_extend_one 以 valid = sel < S 屏蔽——原转 0 使
                # token 0 重复计权；empty 行（far 区空）走原版整体 topk
                # 特判保语义）
                from sglang.srt.layers.attention.tli.kernels import (
                    tli_compact,
                    tli_l2_score_batched_dual,
                )

                Tc_k = min((K1 * Hkv + p.sliding_blocks) * bs, nblk * bs, S)
                S_t_r = t_c + 1  # [n] per-row 因果长度
                tok = tli_compact(onehot, S_t_r, bs, S, Tc_k)  # [n, Tc_k] 哨兵=S
                tok_ck = tok.clamp(max=S - 1)
                far_hi_t = (t_c + 1 - p.near_len).clamp(min=far_lo_k)
                sw_lo_t = (t_c - p.sliding_window + 1).clamp(min=0)
                rows0 = torch.zeros(n, dtype=torch.long, device=device)
                _ds_on = getattr(p, "use_ds_topk", False) and _ds_available()
                # #65：DS 开时 dual kernel 直出 bf16+pad（消灭 fp32 [n,Hkv,Tc]
                # ×2 物化 + ds_topk cast/pad 复制链；分数 fp32 累加后单点转
                # bf16，rounding 与原整体 cast 一致）
                far_sc, near_sc = tli_l2_score_batched_dual(
                    q2,
                    index["kq_q"][:S].unsqueeze(0),
                    index["kq_sc"][:S].unsqueeze(0),
                    index["kq_mn"][:S].unsqueeze(0),
                    rows0,
                    tok_ck,
                    far_lo_k,
                    far_hi_t,
                    sw_lo_t,
                    out_bf16_pad=_ds_pad_for(Tc_k) if _ds_on else 0,
                )
                # B'：host 常量宽度（prefill 无 CUDA graph 约束，对齐原版输出
                # K2 = token_budget：far k2_far + near (k2_near-F) + forced F；
                # 原版 near topk k2_near 席含滑窗 +inf 位 → 近端实选 k2_near-F，
                # kernel 版显式 forced 段补 F —— 预算分配等价）
                near_floor = p.sliding_window + far_lo_k
                far_cap = max(0, p.token_budget - near_floor)
                W_far = min(p.far_tokens, far_cap)
                W_forced = p.sliding_window
                W_near = max(0, p.token_budget - W_far - W_forced)
                F_t = (t_c + 1 - sw_lo_t).clamp(min=0)
                SENT = S
                tok_e = tok_ck.unsqueeze(1).expand(n, Hkv, Tc_k)
                parts = []
                if W_far > 0:
                    if _ds_on:
                        # #65：far_sc 已是 bf16+pad 直出（宽 Tc_k+pad）——
                        # DS 直用，分数 gather 越界位自动 -inf → keep 兜住
                        i_f = ds_topk_padded(
                            far_sc.reshape(-1, far_sc.shape[-1]), Tc_k, W_far
                        ).reshape(far_sc.shape[0], far_sc.shape[1], W_far)
                        sc_f = torch.gather(far_sc, 2, i_f)
                        # idx 可落 pad 区（≥Tc_k）→ clamp 到候选表末位；
                        # 该位分数为 -inf，keep_f 会转哨兵，语义不变
                        sel_f = torch.gather(
                            tok_e, 2, i_f.clamp(max=Tc_k - 1)
                        )
                    else:
                        i_f = torch.topk(far_sc, W_far, dim=-1).indices
                        sc_f = torch.gather(far_sc, 2, i_f)
                        sel_f = torch.gather(tok_e, 2, i_f)
                    keep_f = sc_f != float("-inf")
                    # B01 修复：池不足槽位保留哨兵 SENT（原转 0 使
                    # token 0 重复计权；下游 valid = sel < S 屏蔽）
                    parts.append(torch.where(~keep_f, SENT, sel_f))
                if W_near > 0:
                    if _ds_on:
                        i_n = ds_topk_padded(
                            near_sc.reshape(-1, near_sc.shape[-1]), Tc_k, W_near
                        ).reshape(near_sc.shape[0], near_sc.shape[1], W_near)
                        sc_n = torch.gather(near_sc, 2, i_n)
                        sel_n = torch.gather(
                            tok_e, 2, i_n.clamp(max=Tc_k - 1)
                        )
                    else:
                        i_n = torch.topk(near_sc, W_near, dim=-1).indices
                        sc_n = torch.gather(near_sc, 2, i_n)
                        sel_n = torch.gather(tok_e, 2, i_n)
                    keep_n = sc_n != float("-inf")
                    # B01 修复：near 池同款哨兵 SENT
                    parts.append(torch.where(~keep_n, SENT, sel_n))
                if W_forced > 0:
                    f_pos = sw_lo_t.view(-1, 1) + torch.arange(W_forced, device=device)
                    f_pad = torch.arange(W_forced, device=device).view(1, -1) >= F_t.view(-1, 1)
                    # B01 修复：forced 段 pad 位哨兵 SENT（原转 0）
                    forced = torch.where(f_pad, SENT, f_pos)
                    parts.append(forced.unsqueeze(1).expand(n, Hkv, -1))
                res_k = torch.cat(parts, dim=-1)  # [n, Hkv, K2]（B01 修复后：无效槽位哨兵 S）
                # empty 行（far 区空）：走原版整体 topk 口径（对拍锚定）。
                # #65：fine_e 只对 empty 行子集计算（旧版全 n 行 einsum+
                # 全宽 topk——首 chunk 41% 行 empty 时 chunk0 67ms vs 其余
                # 24ms 的主因）；逐行结果不变（纯子集化）。
                # t_min_hint（forward_extend 的 prefix，host int）：t ≥
                # near_len+far_lo 时无 empty 行、t ≥ K2 时无 early 行——
                # 跳过 .any() GPU 同步（生产 ~60 forward×16 req×36 层，
                # 每次同步排空队列 = e2e 隐藏大头）
                # F5（#129）豁免说明：此处的 bool(empty.any()) 保留——
                # 兜底计算需要 q2[empty]/t_c[empty] 的**数据依赖形状**
                # 布尔索引（CUDA 上无论 any() 还是掩码索引本身都是 host
                # 同步；改全行 einsum = #65 的 67ms/首chunk 回归）。缓解
                # 已就位：① t_min_hint（prefix ≥ near_len+far_lo 的 chunk，
                # 即除首 chunk 外全部）整段跳过；② F3 惰性化后 kq_f 仅在
                # 该兜底实际触发时构建。残余同步 = 每请求每层首 chunk
                # 恰一次（36 层 × 1 = 每请求 36 次，对比原版每 chunk）。
                empty = far_hi_t <= far_lo_k  # [n] device
                # t_c < K2 的行末尾被 #58 uniform grid 整行覆盖 → 无需算
                _no_empty = (
                    t_min_hint is not None
                    and t_min_hint >= p.near_len + far_lo_k
                )
                if not _no_empty:
                    empty &= t_c >= res_k.shape[-1]  # device op 无同步
                    if bool(empty.any()):
                        q2_e = q2[empty]           # [ne, Hkv, nd2]
                        t_e = t_c[empty]           # [ne]
                        fine_e = torch.einsum("ahd,shd->ahs", q2_e, _kq_f())
                        causal_full = pos.view(1, S) <= t_e.view(-1, 1)
                        fine_e = fine_e.masked_fill(~causal_full.unsqueeze(1), float("-inf"))
                        sw_off = torch.arange(p.sliding_window, device=device)
                        f_sw = (t_e.view(-1, 1) - sw_off).clamp(min=0)
                        fine_e.scatter_(2, f_sw.unsqueeze(1).expand(-1, Hkv, -1), float("inf"))
                        i_g = torch.topk(fine_e, min(p.token_budget, S), dim=-1).indices
                        # 行 scatter 回全宽（i_g 宽 min(budget,S) 与 res_k 的
                        # parts 拼接宽对齐；正常配置相等=token_budget）
                        w = res_k.shape[-1]
                        if i_g.shape[-1] != w:
                            if i_g.shape[-1] > w:
                                i_g = i_g[..., :w]
                            else:
                                i_g = torch.nn.functional.pad(
                                    i_g, (0, w - i_g.shape[-1]))
                        res_k[empty] = i_g
                out.append(res_k)
                continue
            else:
                # 掩码位置 → 定长候选张量（topk 最小值技巧：哨兵 S 排最后补 pad）
                seq = pos.view(1, S).expand(n, S)
                seq_m = torch.where(sel_mask, seq, torch.full_like(seq, S))
                # F5（#129，host 同步消除）：原版 Tc = int(sel_mask.sum(1)
                # .max().item()) 是每 chunk 一次 GPU 同步（#126 ⑥ 同步点）。
                # 改静态上界（与 select_decode_batched / M10 慢路径 kernel 的
                # Tc_k 同一推导）：候选并集 = 每 head top-K1 块 ∪ 滑窗块，
                # 跨 head 并集上界 (K1*Hkv + sliding_blocks)*bs，另受 nblk*bs
                # 与 S 截断。多付 topk-min 全排序宽度（动态值→上界），换取
                # 队列不排空；valid 掩码（tok < S）已消化多出的哨兵 pad 行
                # ——topk largest=False 时哨兵 S 恰好排在末尾被 pad，与原版
                # 动态宽度逐位一致（torch.topk 对 -inf/哨兵的排序稳定）。
                Tc = min((K1 * Hkv + p.sliding_blocks) * bs, nblk * bs, S)
                tok = torch.topk(seq_m, Tc, dim=-1, largest=False).values  # [n, Tc]
                valid = tok < S
                tok_c = tok.clamp(max=S - 1)

                # ---- L2: 4bit 部分维精筛 → scatter 进 fine [n, Hkv, S] ----
                kq_c = _kq_f()[tok_c]  # [n, Tc, Hkv, nd2]（共享表 gather，替代逐 chunk 反量化）
                if q_agg_max:
                    s2 = torch.einsum(
                        "ahgd,athd->ahgt", q_g, kq_c
                    ).max(2).values  # [n, Hkv, Tc]
                else:
                    s2 = torch.einsum("ahd,athd->aht", q2, kq_c)  # [n, Hkv, Tc]
                causal = (tok <= t_c.view(-1, 1)) & valid  # [n, Tc]
                fine = torch.full((n, Hkv, S), float("-inf"), device=device)
                fine.scatter_(
                    2,
                    tok_c.unsqueeze(1).expand(n, Hkv, Tc),
                    torch.where(
                        causal.unsqueeze(1), s2, torch.full_like(s2, float("-inf"))
                    ),
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
                # F5（#129）：残余同步 bool(empty.any()) 消除——整体 topk
                # 无条件计算（形状静态 [n, Hkv, budget]，正常行结果被
                # torch.where 丢弃），ne=0 时多付一次 topk（μs 级，远小于
                # 队列排空代价）。语义与 if-gated 版逐位一致。
                empty = far_hi <= far_lo  # [n]
                i_g = torch.topk(fine, min(p.token_budget, S), dim=-1).indices
                res = torch.where(empty.view(n, 1, 1), i_g, res)
                # #60：末 row chunk 统计 per-layer far mass（行×Hkv 平均，
                # softmax 在因果区归一化）→ decode 动态 skip_far 依据。
                # M10 kernel 慢路径不物化 fine（far_sc 池表口径不同），
                # 统计仅 eager/fast_path 路径——dyn gate 实验先跑 eager。
                if getattr(p, "dyn_far_gate", False) and r1 == Nq:
                    # F5 豁免：float(fm.mean()) 是有意的一次 host 同步——
                    # gate 统计量本身是 host 标量（decode 侧阈值判断用），
                    # 且仅在 dyn_far_gate 开 + 末 row chunk 触发（每请求每层
                    # 恰一次），非热点路径。
                    # 滑窗强制位是 +inf（topk 强制语义）→ softmax(inf)=nan，
                    # 统计前剔除（权重 0；sw=128 << far 区尺度，口径影响 <1%）
                    fine_s = fine.masked_fill(fine == float("inf"), float("-inf"))
                    pm = torch.softmax(fine_s, dim=-1)  # [n, Hkv, S]
                    fm = (pm * in_far.unsqueeze(1)).sum(-1)  # [n, Hkv]
                    stat = float(fm.mean())
                    if stat == stat:  # nan 防护：全空 far 行不覆盖
                        self.dyn_far_stat = stat
                out.append(res)
            else:
                out.append(torch.topk(fine, min(p.token_budget, S), dim=-1).indices)
        res_all = torch.cat(out, dim=0)  # [Nq, Hkv, K2]
        # #58 行级因果修复（30B 崩坏根因）：早期行（t_r < K2，仅 prefill 首
        # chunk 存在——后续 chunk t_arr 从 prefix 起均 ≥ K2）的因果 token 数
        # 小于 K2，topk 的 -inf 填充槽位会返回 S 维内任意位置（含未来 token）；
        # M10 kernel 路径哨兵转 0 同样不保证行级因果。下游 _sparse_extend_one
        # softmax 无掩码 → 早期行直接看到本 chunk 未来 token，逐层传播污染
        # 全序列（30B 64-token 生成陷入重复循环；审计 caus_over=2095104 全部
        # 来自前 K2-1 行）。修复：这些行整行替换为 identity grid。
        # B02 修复（GPT 审查 2026-10-08）：原 uniform grid 在 K2 不整除 L 时
        # 前 K2%L 个 token 重复 floor+1 次——重复 m 次等价 softmax logit
        # +log(m) 偏置（K2=1024/L=1000 反例：前 24 token 权重翻倍，算术
        # 输出 0.047 vs dense 0.024），并非注释原称的「数学等价 dense」。
        # 改为每 token 恰一次（j < L → j）+ 尾部哨兵 S（下游
        # valid = sel < S 屏蔽）——softmax 有效位恰为全部 L 个因果
        # token，真正 dense 等价。t_r ≥ K2 的行越界数为 0（审计验证），不动。
        early = t_arr < res_all.shape[-1]  # [Nq]
        # #65：t_min_hint ≥ K2（输出宽）时无 early 行——跳过 .any() 同步
        _no_early = (
            t_min_hint is not None and t_min_hint >= res_all.shape[-1]
        )
        # F5（#129）：残余同步 bool(early.any()) 消除——无条件执行向量
        # 化构造（全部 device op；ne=0 时 nonzero 返回空张量，scatter
        # no-op，多付 ~5 个微 launch 远小于队列排空）。语义与
        # if-gated 版逐位一致。
        if not _no_early:
            # #65 向量化：旧版逐行 Python 循环（arange+repeat_interleave+
            # cat+index_put × 每早期行，首 chunk 1024 行 → ~5K launch +
            # 3K 次 .item() 同步，chunk0 65ms 的主因）。
            K2 = res_all.shape[-1]
            Hkv2 = res_all.shape[1]
            e_idx = early.nonzero().squeeze(-1)  # [ne]
            L = t_arr[e_idx] + 1                 # [ne] 因果位置数
            j = torch.arange(K2, device=res_all.device).view(1, -1)
            # B02：identity grid + 哨兵 S 尾垫（哨兵 ≥ 任意有效位，
            # 下游 valid = sel < S 屏蔽；行恒含位置 0 → softmax 无 NaN）
            grid = torch.where(
                j < L.view(-1, 1), j, S
            )  # [ne, K2]
            res_all[e_idx] = grid.unsqueeze(1).expand(-1, Hkv2, K2)
        return res_all

    @torch.no_grad()
    def _select_batched_taskmd(
        self,
        index: dict,
        q: torch.Tensor,
        t_arr: torch.Tensor,
        row_chunk: int = 64,
        t_min_hint: int | None = None,
    ) -> torch.Tensor:
        """E112（#149）：prefill 批量 TASK.md 分区选择（eager，per-row device 边界）。

        与 _select_taskmd 逐行集合一致（对拍锚：per-row 区域边界 /
        双池分数源 / γ 预算切分全部按 t_r 独立计算）。M7 快路径与
        M10 kernel 旁路（avg 分数源 + sink 正交 + 双池 L1 语义超出
        出口径）。输出 [Nq, Hkv, K2]。B01 修复（GPT 审查 2026-10-08）：
        池不足/越界槽位 = 哨兵 S_r（原 M10 口径转 0 会把 token 0 重复
        计权——槽位重复 m 次等价 softmax logit +log(m) 偏置，S=4096
        反例实测算术输出 0.376 vs 去重 0.0016）；下游 _sparse_extend_one
        以 valid = sel < S 逐槽屏蔽（与 decode 侧哨兵协议统一）。
        早期行（t < K2）由尾部 grid 修复覆盖：每 token 恰一次 + 哨兵
        尾垫（B02：原 uniform grid 在 K2 不整除 L 时前 K2%L 个 token
        重复 2 次，非 dense 等价）。

        per-row 区域边界（two-level compute_mask 公式的 t_r 化）：
        - near_len_dyn_r = max(bs, α·mid_len_r)，mid_len_r 按因果长度
          S_r = t_r+1 计算（two-level 是 t=末端的单 query 版）
        - near_blks_r = max(sink_blks, (S_r − swa_tok − near_len_dyn_r)//bs)
          （B10 修复 2026-10-10：near 左界从 swa 起点推，near 区实际宽 = α·mid_len）
        - far 池（L1 块级）：[sink_blks, near_blks_r)；near 池：
          [near_blks_r, swa_lo_blk_r)（swa_lo_blk_r = 当前块 − 1，
          swa 块排除出双池）
        - L2 far 池（token 级）：[sink_tok, min(near_blks_r·bs, S_r))；
          near 池：[far_hi, swa_lo_tok_r)；swa 强制 [t_r−swa+1, t_r]
        """
        p = self.profile
        S = index["S"]
        Hkv = index["kmin"].shape[1]
        Nq, H, D = q.shape
        G = H // Hkv
        device = q.device
        nblk = index["nblk"]
        bs = p.block_size
        kmin, kmax = index["kmin"][:nblk], index["kmax"][:nblk]
        kavg = (
            index["kavg_sum"][:nblk] / bs if self._need_avg_score() else None
        )
        kq = index["kq_q"]
        nd2 = self.nd2
        t_arr = t_arr.to(device)
        pos = torch.arange(S, device=device)
        blk_id = torch.arange(nblk, device=device)
        sink_tok = p.sink_blocks * bs
        swa_tok = p.sliding_window
        swa_blks = max(1, swa_tok // bs)
        e64 = p.alpha > 0 and p.beta > 0
        # 静态配额（host 常量；S 满额口径——sparse 行 S > dense_threshold
        # 保证 swa/sink 段满额，输出恒宽 token_budget）
        nb_near = nb_far = 0
        K2_mid_cap = nt_near_cap = far_budget_cap = 0
        if e64:
            nb_near = max(1, int(round(p.k1_blocks * p.beta)))
            nb_far = max(0, p.k1_blocks - nb_near)
            K2_mid_cap = max(0, p.token_budget - sink_tok - swa_tok)
            nt_near_cap = min(int(nb_near * bs * p.gamma), K2_mid_cap)
            far_budget_cap = max(0, K2_mid_cap - nt_near_cap)
        # 共享反量化表（全宽 einsum fine 矩阵；与 M7 fast path 同口径）
        kq_f = kq_unpack(kq[:S], index["kq_sc"][:S], index["kq_mn"][:S])
        out = []
        for r0 in range(0, Nq, row_chunk):
            r1 = min(r0 + row_chunk, Nq)
            q_c = q[r0:r1]
            t_c = t_arr[r0:r1]
            n = r1 - r0
            S_r = t_c + 1  # [n] per-row 因果长度
            # ---- L1 分数：minmax + 可选 avg ----
            qs = q_c[..., self.idx1]  # [n, H, d']
            qg = qs.clamp(min=0).reshape(n, Hkv, G, p.coarse_dim)
            qn = qs.clamp(max=0).reshape(n, Hkv, G, p.coarse_dim)
            sc1 = torch.einsum("ahgd,mhd->ahm", qg, kmax) + torch.einsum(
                "ahgd,mhd->ahm", qn, kmin
            )  # [n, Hkv, nblk]
            blk_end = (torch.arange(nblk, device=device) + 1) * bs - 1
            causal_blk = blk_end.view(1, -1) <= t_c.view(-1, 1)  # [n, nblk]
            sc1 = sc1.masked_fill(~causal_blk.unsqueeze(1), float("-inf"))
            sc_avg = None
            if kavg is not None:
                qa = qs.reshape(n, Hkv, G, p.coarse_dim).sum(2)  # [n, Hkv, d']
                sc_avg = torch.einsum("ahd,mhd->ahm", qa, kavg)
                sc_avg = sc_avg.masked_fill(
                    ~causal_blk.unsqueeze(1), float("-inf")
                )

            if e64:
                # ---- per-row 区域边界（device）----
                # 【B10 修复 2026-10-10】near 左界从 swa 起点推
                # （near_base = S_r − swa_tok），与 _taskmd_regions /
                # _select_decode_taskmd / two-level compute_mask 同源。
                mid_len_r = (S_r - sink_tok - swa_tok).clamp(min=0)
                near_len_dyn_r = torch.maximum(
                    torch.full_like(S_r, bs),
                    (p.alpha * mid_len_r.float()).long(),
                )
                near_blks_r = torch.maximum(
                    torch.full_like(S_r, p.sink_blocks),
                    (S_r - swa_tok - near_len_dyn_r) // bs,
                )
                swa_lo_blk_r = torch.maximum(
                    near_blks_r, t_c // bs - swa_blks + 1
                )
                in_far_blk = (blk_id.view(1, -1) >= p.sink_blocks) & (
                    blk_id.view(1, -1) < near_blks_r.view(-1, 1)
                ) & causal_blk  # [n, nblk]
                in_near_blk = (blk_id.view(1, -1) >= near_blks_r.view(-1, 1)) & (
                    blk_id.view(1, -1) < swa_lo_blk_r.view(-1, 1)
                ) & causal_blk
                # far 池 topk（静态宽 nb_far；池不足/非 far 位 -inf → keep 剔除）
                src_far = (
                    sc_avg if (p.far_method == "avg" and sc_avg is not None) else sc1
                )
                sc_far = src_far.masked_fill(
                    ~in_far_blk.unsqueeze(1), float("-inf")
                )
                i_f = torch.topk(sc_far, min(nb_far, nblk), dim=-1).indices
                keep_f = (
                    torch.gather(sc_far, 2, i_f) > float("-inf")
                )  # [n, Hkv, nb_far]
                # near 池 topk
                src_near = (
                    sc1
                    if (p.near_method == "minmax" or sc_avg is None)
                    else sc_avg
                )
                sc_near = src_near.masked_fill(
                    ~in_near_blk.unsqueeze(1), float("-inf")
                )
                i_n = torch.topk(sc_near, min(nb_near, nblk), dim=-1).indices
                keep_n = torch.gather(sc_near, 2, i_n) > float("-inf")
                # E112 per-head L1 池（two-level topk_mask 的 per-head 语义；
                # 并集 onehot 仅作全宽 einsum 的效率口径，fine 掩码用 per-head）。
                # far/near 须分别 scatter 到独立零张量再 OR：topk 池不足时的
                # 欠宽槽位指向 -inf 并列位（tie-break 无保证、索引任意），
                # 第二个 scatter 的 False 会把第一个 scatter 已置 True 的
                # 块覆写回 False（mavg_faract 实测每 head 丢 ~16 个 far 块
                # → fine 有限位少 8192 → far 池 topk 集合发散）。
                # 分池写再 OR 恒安全：False 落在零张量是无操作。
                oh_f = torch.zeros(n, Hkv, nblk, dtype=torch.bool, device=device)
                oh_f.scatter_(2, i_f, keep_f)
                oh_n = torch.zeros(n, Hkv, nblk, dtype=torch.bool, device=device)
                oh_n.scatter_(2, i_n, keep_n)
                onehot_h = oh_f | oh_n
                onehot = onehot_h.any(1)  # [n, nblk] 候选并集
            else:
                # ---- 单池：far_method 独占 + 当前块 per-row 强制 inf ----
                sc_pool = (
                    sc1 if (p.far_method != "avg" or sc_avg is None) else sc_avg
                )
                sc_pool = sc_pool.clone()
                cur_blk_r = (t_c // bs).view(n, 1, 1)
                sc_pool.scatter_(
                    -1,
                    cur_blk_r.expand(n, Hkv, 1),
                    torch.full(
                        (n, Hkv, 1),
                        float("inf"),
                        dtype=sc_pool.dtype,
                        device=device,
                    ),
                )
                k1e = min(p.k1_blocks, nblk)
                i_pool = torch.topk(sc_pool, k1e, dim=-1).indices
                keep_p = torch.gather(sc_pool, 2, i_pool) > float("-inf")
                # E112 per-head L1 池（同 e64 分支；swa 块由 L2 +inf 强制
                # 覆盖 per-head 掩码，无须入池）
                onehot_h = torch.zeros(
                    n, Hkv, nblk, dtype=torch.bool, device=device
                )
                onehot_h.scatter_(2, i_pool, keep_p)
                onehot = onehot_h.any(1)  # [n, nblk] 候选并集

            # ---- L2：全宽 einsum fine 矩阵（候选并集语义 → per-head 掩码）----
            # bug2 修复（GPT 复查 2026-10-08）：q_agg=max 原在 taskmd
            # prefill 批量路径未生效（写死 sum）——补 max 分支（#58）。
            q_g = self._q_refine(q_c).reshape(n, Hkv, G, nd2)  # [n, Hkv, G, nd2]
            q_agg_max = getattr(p, "q_agg", "sum") == "max"
            q2 = None if q_agg_max else q_g.sum(2)  # [n, Hkv, nd2]
            if self.basis is None:
                # E112 对拍口径：零填充回全 D 维 + 逐行 "hd,thd->ht" 形式
                # （与 two-level / per-request 归约树逐位一致——批量形式
                # "ahd,shd->ahs" 的 einsum 降维顺序有 ulp 差异，会在 topk
                # 边界翻转近并列 token；见 _pad_refine_full）
                kq_ff = self._pad_refine_full(kq_f, D)  # [S, Hkv, D]
                if q_agg_max:
                    q_gf = self._pad_refine_full(q_g, D)  # [n, Hkv, G, D]
                    fine = torch.stack(
                        [
                            torch.einsum("hgd,thd->hgt", q_gf[a], kq_ff)
                            .max(1)
                            .values
                            for a in range(n)
                        ]
                    )  # [n, Hkv, S]
                else:
                    q2f = self._pad_refine_full(q2, D)  # [n, Hkv, D]
                    fine = torch.stack(
                        [
                            torch.einsum("hd,thd->ht", q2f[a], kq_ff)
                            for a in range(n)
                        ]
                    )  # [n, Hkv, S]
            elif q_agg_max:
                fine = torch.einsum("ahgd,shd->ahgs", q_g, kq_f).max(
                    2
                ).values  # [n, Hkv, S]
            else:
                fine = torch.einsum("ahd,shd->ahs", q2, kq_f)  # [n, Hkv, S]
            causal_full = pos.view(1, S) <= t_c.view(-1, 1)
            # E112 per-head L1 掩码（two-level per-head topk_mask 语义；
            # 全宽 einsum + 并集只是效率口径）。单池路径的 swa +inf
            # scatter 在其后覆盖此掩码（two-level p[..., -swa:]=1.0 同款）
            blk_mask_h = onehot_h.repeat_interleave(bs, dim=-1)[
                :, :, :S
            ]  # [n, Hkv, S]
            fine = fine.masked_fill(
                ~(blk_mask_h & causal_full.unsqueeze(1)), float("-inf")
            )

            if e64:
                # ---- L2 双池：γ 预算切分（静态宽 + rank 掩码，M5 模式）----
                far_hi_t = torch.minimum(near_blks_r * bs, S_r)  # [n]
                swa_lo_t = (t_c + 1 - swa_tok).clamp(min=0)  # [n]
                swa_w_r = torch.minimum(
                    torch.full_like(S_r, swa_tok), S_r
                )  # swa 段宽（早行 < swa_tok；#58 修复覆盖）
                K2_mid_r = (
                    p.token_budget - sink_tok - swa_w_r
                ).clamp(min=0)
                nt_near_r = torch.minimum(
                    torch.full_like(K2_mid_r, nt_near_cap), K2_mid_r
                )
                far_budget_r = (K2_mid_r - nt_near_r).clamp(min=0)
                span_r = (far_hi_t - sink_tok).clamp(min=0)
                k2_far_r = torch.minimum(
                    torch.minimum(far_budget_r, span_r), K2_mid_r
                )
                # far 池 topk（静态宽 far_budget_cap；fine 全宽矩阵 → 索引即位置）
                in_far_tok = (pos.view(1, S) >= sink_tok) & (
                    pos.view(1, S) < far_hi_t.view(-1, 1)
                )  # [n, S]
                far_f = fine.masked_fill(
                    ~in_far_tok.unsqueeze(1), float("-inf")
                )
                i_f2 = torch.topk(far_f, far_budget_cap, dim=-1).indices
                sc_f2 = torch.gather(far_f, 2, i_f2)
                rank_f2 = (
                    torch.arange(far_budget_cap, device=device)
                    .view(1, 1, -1)
                    < k2_far_r.view(-1, 1, 1)
                )
                keep_f2 = (sc_f2 > float("-inf")) & rank_f2
                # B01 修复：无效槽位哨兵全局 S（下游 valid = sel < S 屏蔽；
                # 原转 0 使 token 0 被重复计权。S1 修复（GPT F01/Kimi3 双
                # 审查 2026-10-08）：原用逐行 S_r=t_r+1，中间行 S_r<S 被
                # 消费端判有效 → 读入未来 token t_r+1（因果泄漏）；哨兵值
                # 必须与消费端界同源 = 全局 S）
                sel_f2 = torch.where(
                    ~keep_f2, S, i_f2
                )
                # near 池 topk（静态宽 nt_near_cap）
                in_near_tok = (pos.view(1, S) >= far_hi_t.view(-1, 1)) & (
                    pos.view(1, S) < swa_lo_t.view(-1, 1)
                )
                near_f = fine.masked_fill(
                    ~in_near_tok.unsqueeze(1), float("-inf")
                )
                i_n2 = torch.topk(near_f, nt_near_cap, dim=-1).indices
                sc_n2 = torch.gather(near_f, 2, i_n2)
                k2_near_r = (K2_mid_r - k2_far_r).clamp(min=0)
                rank_n2 = (
                    torch.arange(nt_near_cap, device=device)
                    .view(1, 1, -1)
                    < k2_near_r.view(-1, 1, 1)
                )
                keep_n2 = (sc_n2 > float("-inf")) & rank_n2
                # B01 修复：无效槽位哨兵全局 S（S1 同 far 池口径）
                sel_n2 = torch.where(
                    ~keep_n2, S, i_n2
                )
                # 正交强制区：sink 头部（宽 sink_tok）+ swa 尾部（宽 swa_tok，
                # 早行 t_r < swa-1 的越界槽位转哨兵全局 S 而非 0——S1 修复）
                sink_pos = torch.arange(sink_tok, device=device).view(1, -1)
                sink_pad = (
                    torch.arange(sink_tok, device=device).view(1, -1)
                    >= S_r.view(-1, 1)
                )
                sink_out = torch.where(
                    sink_pad, S, sink_pos
                ).unsqueeze(1).expand(n, Hkv, -1)
                f_pos = swa_lo_t.view(-1, 1) + torch.arange(
                    swa_tok, device=device
                ).view(1, -1)
                F_t = (t_c + 1 - swa_lo_t).clamp(min=0)  # [n] 实际 swa 段宽
                f_pad = (
                    torch.arange(swa_tok, device=device).view(1, -1)
                    >= F_t.view(-1, 1)
                )
                forced_out = torch.where(
                    f_pad, S, f_pos
                ).unsqueeze(1).expand(n, Hkv, -1)
                res = torch.cat(
                    [sel_f2, sel_n2, sink_out, forced_out], dim=-1
                )  # [n, Hkv, K2] 静态宽
            else:
                # ---- 单池 L2：swa 强制 + 整行 topk ----
                sw_off = torch.arange(swa_tok, device=device)
                f_sw = (t_c.view(-1, 1) - sw_off).clamp(min=0)  # [n, swa]
                fine.scatter_(
                    2, f_sw.unsqueeze(1).expand(n, Hkv, -1), float("inf")
                )
                res = torch.topk(
                    fine, min(p.token_budget, S), dim=-1
                ).indices  # [n, Hkv, K2]
            out.append(res)
        res_all = torch.cat(out, dim=0)  # [Nq, Hkv, K2]
        # #58 行级因果修复 + B02 修复（GPT 审查 2026-10-08）：早期行
        # t < K2 用 identity grid（每 token 恰一次）+ 哨兵 S 尾垫——
        # 原 uniform grid 在 K2 不整除 L 时前 K2%L 个 token 重复 2 次
        # （等价 +log(2) logit 偏置，非 dense 等价）；哨兵槽位由下游
        # valid = sel < S 屏蔽，与 _select_taskmd 尾部口径一致。
        early = t_arr < res_all.shape[-1]
        if bool(early.any()):
            K2 = res_all.shape[-1]
            Hkv2 = res_all.shape[1]
            e_idx = early.nonzero().squeeze(-1)
            L = t_arr[e_idx] + 1
            j = torch.arange(K2, device=res_all.device).view(1, -1)
            grid = torch.where(
                j < L.view(-1, 1), j, S
            )  # [ne, K2] identity + 哨兵尾垫
            res_all[e_idx] = grid.unsqueeze(1).expand(-1, Hkv2, K2)
        return res_all

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
        nd2 = self.nd2
        d1 = p.coarse_dim
        device = k_new.device
        S_cap = pool_l["kq_q"].shape[1]
        nblk_cap = pool_l["kmin"].shape[1]
        rows_l = rows.to(torch.long)
        if S_old_t is None:
            S_old_t = torch.tensor(S_old, device=device)

        # ---- kq 追加（M6 uint8+scale 三张量 flat scatter；M9 投影先于量化）----
        kq_q_new, kq_sc_new, kq_mn_new = quant4_pack(self._k_refine(k_new))
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
        # E112：kavg_sum 批量增量（taskmd + avg method 时 pool 带 kavg_sum，
        # 形状 [R, NBLK_CAP, Hkv, d1] 与 kmin 同构 → 复用 off_mf flat 索引；
        # 对齐行开新块（和 = 新 token），非对齐行尾块累加）
        if "kavg_sum" in pool_l:
            old_sum = (
                pool_l["kavg_sum"].reshape(-1)[off_mf].view(n, Hkv, d1)
            )
            sum_val = torch.where(
                aligned.view(n, 1, 1), ks, old_sum + ks
            )
            pool_l["kavg_sum"].reshape(-1)[off_mf] = sum_val.reshape(-1)

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
        # E112（#149）：TASK.md 模式（α/β/γ + method 组合）→ 专用路径
        # （eager，M8 KernelD/KernelC/compact/topk kernel 全旁路——avg
        # 分数源 + sink 正交 + 双池 L1 语义超出现有出口径；静态宽推导
        # 沿用 M5 模式保证 CUDA graph 可录制）
        if getattr(self.profile, "taskmd", False):
            return self._select_decode_taskmd(pool_l, rows, S_list, q)
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
        nd2 = self.nd2
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

        # ---- L1: 子空间块上界（a=请求行 / m=块）----
        # M8-KernelD：非 skip_far 层走 fused gather+GEMV（rows 行间接直读 pool，
        # 消除 P1 行 gather 262μs + P2 einsum permute 拷贝 ~346μs@bs32/131K；
        # 归约同 eager 分组，数值 1e-7 级）
        rows_l = rows.to(torch.long)
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
        if (
            getattr(p, "use_l1_batched_kernel", False)
            and not self.skip_far
            and n >= 2
            and (Hkv & (Hkv - 1)) == 0
            and (p.coarse_dim & (p.coarse_dim - 1)) == 0
            and q.is_contiguous()
        ):
            if getattr(p, "use_l1_tc_kernel", False) and p.coarse_dim >= 16:
                # M8-TC：tl.dot tf32 MMA 版（SGLANG_TLI_L1TC_KERNEL=1）
                from sglang.srt.layers.attention.tli.kernels import (
                    tli_l1_score_batched_dot,
                )

                sc1 = tli_l1_score_batched_dot(
                    q, self.idx1, kmin_pool, kmax_pool, rows_l, nblk_t, t_t, bs
                )
            else:
                from sglang.srt.layers.attention.tli.kernels import (
                    tli_l1_score_batched,
                )

                sc1 = tli_l1_score_batched(
                    q, self.idx1, kmin_pool, kmax_pool, rows_l, nblk_t, t_t, bs
                )  # 垃圾块 -inf 已在 kernel 内烘焙（skip_far 的 keep 掩码走 eager）
        else:
            kmin_b = kmin_pool[rows]  # [n, NBLK_CAP, Hkv, d']（容量行含垃圾，靠掩码）
            kmax_b = kmax_pool[rows]
            qs = q[..., self.idx1]  # [n, H, d']
            qg = qs.clamp(min=0).reshape(n, Hkv, G, p.coarse_dim)
            qn = qs.clamp(max=0).reshape(n, Hkv, G, p.coarse_dim)
            sc1 = torch.einsum("ahgd,amhd->ahm", qg, kmax_b) + torch.einsum(
                "ahgd,amhd->ahm", qn, kmin_b
            )  # [n, Hkv, NBLK_CAP]
            sc1 = sc1.masked_fill(~valid_blk.unsqueeze(1), float("-inf"))
        cand_blk = torch.topk(sc1, K1, dim=-1).indices  # [n, Hkv, K1]
        # #65 decode 侧 DS：CUDA graph 捕获中回退 torch.topk（无图路径
        # ——收益区口径——直用；DS 退化行 pad 泄漏由 clamp+valid_blk 掩码
        # 吸收，与 prefill L1 同语义）
        _ds_dec = (
            getattr(p, "use_ds_topk", False)
            and _ds_available()
            and not torch.cuda.is_current_stream_capturing()
        )
        if _ds_dec:
            cand_blk = ds_topk(
                sc1.reshape(n * Hkv, NBLK_CAP), K1
            )[0].clamp(max=NBLK_CAP - 1).reshape(n, Hkv, K1)
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
        # M8：kernel 版 = cumsum 前缀 + 块展开（消除 [n,S_cap] int64 物化与
        # topk-min 全排序；哨兵可在中段——下游 valid 掩掉，有效集逐行一致）
        Tc = min((K1 * Hkv + p.sliding_blocks) * bs, NBLK_CAP * bs, S_cap)
        if (
            getattr(p, "use_compact_kernel", False)
            and n >= 2
            and onehot.is_contiguous()
        ):
            from sglang.srt.layers.attention.tli.kernels import tli_compact

            tok = tli_compact(onehot, S_t, bs, S_cap, Tc)  # [n, Tc] 块升序
        else:
            sel_mask = onehot.repeat_interleave(bs, dim=1)[:, :S_cap]  # [n, S_cap]
            pos = torch.arange(S_cap, device=device)
            cand = sel_mask & (pos.view(1, S_cap) < S_t.view(-1, 1))
            seq_m = torch.where(cand, pos.view(1, S_cap), torch.full_like(pos, S_cap))
            tok = torch.topk(seq_m, Tc, dim=-1, largest=False).values  # [n, Tc] 升序
        # valid/causal 掩码惰性计算：KernelC 双池直写路径池界即因果界，
        # 无需 [n,Tc] int64 逐元素掩码（省 2 个 ~130μs op@bs32/131K）
        tok_c = tok.clamp(max=S_cap - 1)

        # ---- B' 边界（per-row device 张量，静态形状；双池 kernel 与
        # skip_far / eager 掩码路径共用；提前到 L2 之前供 kernel 直写）----
        far_lo = p.sink_blocks * bs
        far_hi_t = (t_t + 1 - p.near_len).clamp(min=far_lo)
        sw_lo_t = (t_t - p.sliding_window + 1).clamp(min=0)

        # ---- L2: 4bit 精筛打分（M6：uint8 格点 + scale 的 flat gather + 重建）----
        # 重建 kq_c = grid*sc+mn 与 fp32 存储版逐位一致（IEEE 同运算序），
        # 打分数值零漂移；pool 常驻显存 128→40 B/token-head
        # M8：n≥2 走 fused 批量 kernel（P5 瓶颈 87.6%@bs32/131K，
        # 20.5→0.48ms 43×，s2 对拍 2.4e-07；哨兵位置垃圾分数同 eager 语义）
        # M8-KernelC：非 skip_far 层走双池直写（far/near -inf 烘进写出口径，
        # 消除 P6 masked_fill 链——池边界即因果边界，topk 输入逐位一致）
        # M8-topk：near 池压缩直写（near 有限项仅 ~920/65728——near topk 98.6%
        # 在扫 -inf；静态上界 WNCAP = sink + (near_len - sliding_window) = 2048，
        # topk 宽度 30×↓；slot 确定性（前缀 sink slot=c / 后缀近带 slot=ps+c-pf）
        # 保持 CUDA graph replay 逐位一致）
        # bug2 修复（GPT 复查 2026-10-08）：q_agg=max 原在 decode 批量
        # 路径未生效（写死 sum）——补 max 分支；M8 fused kernel 输入是
        # group-sum 后的 q2 且 kernel 内做 group-sum，max 聚合不支持
        # → q_agg=max 时旁路 kernel 走 eager。
        q_g = self._q_refine(q).reshape(n, Hkv, G, nd2)  # [n, Hkv, G, nd2]
        q_agg_max = getattr(p, "q_agg", "sum") == "max"
        q2 = None if q_agg_max else q_g.sum(2)  # [n, Hkv, nd2]
        far_sc = near_sc = None
        near_tok_c = None
        WNCAP = far_lo + max(0, p.near_len - p.sliding_window)
        if (
            getattr(p, "use_l2_batched_kernel", False)
            and not q_agg_max
            and n >= 2
            and (Hkv & (Hkv - 1)) == 0
            and (nd2 & (nd2 - 1)) == 0
            and tok_c.is_contiguous()
        ):
            if not self.skip_far and getattr(p, "use_l2_dual_kernel", False):
                from sglang.srt.layers.attention.tli.kernels import (
                    tli_l2_score_batched_dual,
                )

                use_nc = (
                    getattr(p, "use_near_compact", False)
                    and WNCAP > 0
                    and WNCAP < Tc
                )
                if use_nc:
                    far_sc, near_sc, near_tok_c = tli_l2_score_batched_dual(
                        q2, kq_pool, kq_sc_pool, kq_mn_pool, rows_l, tok_c,
                        far_lo, far_hi_t, sw_lo_t,
                        near_compact=True, wncap=WNCAP, sent=S_cap,
                    )
                else:
                    far_sc, near_sc = tli_l2_score_batched_dual(
                        q2, kq_pool, kq_sc_pool, kq_mn_pool, rows_l, tok_c,
                        far_lo, far_hi_t, sw_lo_t,
                    )
            else:
                from sglang.srt.layers.attention.tli.kernels import (
                    tli_l2_score_batched,
                )

                s2 = tli_l2_score_batched(q2, kq_pool, kq_sc_pool, kq_mn_pool,
                                          rows_l, tok_c)
        else:
            s2 = torch.empty(n, Hkv, Tc, dtype=torch.float32, device=device)
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
                if q_agg_max:
                    # bug2 修复：max = 逐 q-head 打分取组内 max（#58）
                    s2[r0:r1] = torch.einsum(
                        "ahgd,athd->ahgt", q_g[r0:r1], kq_c
                    ).max(2).values
                else:
                    s2[r0:r1] = torch.einsum("ahd,athd->aht", q2[r0:r1], kq_c)

        SENT = S_cap

        if self.skip_far:
            # 单池口径（= per-request eager 的整体 topk + 滑窗 +inf）
            causal = tok <= t_t.view(-1, 1)
            valid = tok < S_cap
            sc = s2.masked_fill(~(valid & causal).unsqueeze(1), float("-inf"))
            # S2 修复（Kimi3 审查 2026-10-08）：原条件 (tok >= sw_lo_t) 对
            # 哨兵 lane（tok = S_cap 未 clamp ≥ sw_lo_t 恒真）把刚打的 -inf
            # 复活为 +inf → topk 被 +inf 哨兵挤占、退化「只看滑窗」、
            # 最坏 0 有效 lane → softmax 全 -inf = NaN。sw_lo ≤ t 已蕴含
            # causal，补 & valid 即可。
            sc = sc.masked_fill(
                ((tok >= sw_lo_t.view(-1, 1)) & valid).unsqueeze(1),
                float("inf"),
            )
            k = min(p.token_budget, Tc)
            i_g = torch.topk(sc, k, dim=-1).indices
            sc_g = torch.gather(sc, 2, i_g)
            sel_g = torch.gather(tok_c.unsqueeze(1).expand(n, Hkv, Tc), 2, i_g)
            return torch.where(
                sc_g == float("-inf"), torch.full_like(sel_g, SENT), sel_g
            )  # [n, Hkv, budget]

        # ---- B'：far/near 分区（L2-kernel 路径口径；M5 全程 device 张量 + 静态宽度）----
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
        # near 压缩路径：topk 宽度 = 静态压缩宽（≤ WNCAP），输出拼接宽度
        # 与压缩路径绑定（CUDA graph 捕获时固定）
        W_near = min(p.token_budget, WNCAP if near_tok_c is not None else Tc)
        W_forced = min(p.sliding_window, Tc)

        if far_sc is None:
            # eager / s2-kernel 路径：此处物化双池掩码表
            # （KernelC 直写路径已带 -inf，池界等价于 valid & causal 掩码）
            causal = tok <= t_t.view(-1, 1)
            valid = tok < S_cap
            in_far = (tok >= far_lo) & (tok < far_hi_t.view(-1, 1)) & valid & causal
            in_near = valid & causal & ~in_far & (tok < sw_lo_t.view(-1, 1))
            far_sc = s2.masked_fill(~in_far.unsqueeze(1), float("-inf"))
            near_sc = s2.masked_fill(~in_near.unsqueeze(1), float("-inf"))
        tok_e = tok_c.unsqueeze(1).expand(n, Hkv, Tc)  # far 分支 gather 源
        near_tok_e = (
            near_tok_c.unsqueeze(1).expand(n, Hkv, WNCAP)
            if near_tok_c is not None
            else tok_e
        )
        parts = []
        if W_far > 0:
            if _ds_dec:
                # #65：DS 直用（分数从 pad 后 bf16 xp gather：越界位 -inf
                # → keep 转哨兵，与 torch.topk 有效集逐位一致）
                i_f, xp_f = ds_topk(
                    far_sc.reshape(-1, far_sc.shape[-1]), W_far
                )
                i_f = i_f.reshape(n, Hkv, W_far)
                sc_f = torch.gather(xp_f.view(n, Hkv, -1), 2, i_f)
                sel_f = torch.gather(
                    tok_e, 2, i_f.clamp(max=far_sc.shape[-1] - 1)
                )
            else:
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
            # #65：near 池不接 DS——压缩表 ~2048 有限项挤在 4bit 格点
            # 极少数分数值上（tie 组巨大），DS 块序 tie 打破使选中集合
            # 与 torch 索引序差异过大（实测 jaccard 0.72 vs L1/far 的
            # 0.999）；池仅 WNCAP 宽、torch.topk 本身便宜，保守保留
            i_n = torch.topk(near_sc, W_near, dim=-1).indices
            sc_n = torch.gather(near_sc, 2, i_n)
            # near 压缩路径：gather 源 = near_tok_c（静态宽 WNCAP）；
            # 否则 tok_c（宽 Tc）。topk 索引域与 gather 源宽度一致
            sel_n = torch.gather(near_tok_e, 2, i_n)
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
    def _select_decode_taskmd(
        self,
        pool_l: dict,
        rows: torch.Tensor,
        S_list,
        q: torch.Tensor,
        row_chunk_bytes: int = 256 << 20,
    ) -> torch.Tensor:
        """E112（#149）：decode 批量 TASK.md 分区选择（eager + M5 静态宽度）。

        与 _select_taskmd 逐行集合一致（对拍锚）。结构沿用
        select_decode_batched 的 M5 模式：
        - per-row 区域边界（near_blks_t / swa_lo_blk_t / far_hi_t /
          swa_lo_t / k2_far_t / k2n_t）全部 device 张量运算
        - topk / 输出宽度 = 静态上界（W_far = far_budget_cap、
          W_near = nt_near_cap、W_sink / W_forced 恒定），per-row 配额
          裁剪由 rank 掩码完成——CUDA graph 可录制
        - 输出 [n, Hkv, K2] 恒宽 = far_budget + nt_near + sink + swa
          = token_budget（decode t = S-1 > dense_threshold 保证满额）；
          池不足/越界槽位 = 哨兵 S_cap（下游 _sparse_attn_batched 的
          valid = sel < seq_len 统一屏蔽）
        - M8 全部 fused kernel 旁路（avg 分数源需 kavg_sum 读 pool +
          双池 L1 分数源选择，超出现有 kernel 出口径）
        """
        p = self.profile
        device = q.device
        n = q.shape[0]
        kq_pool, kq_sc_pool, kq_mn_pool = (
            pool_l["kq_q"], pool_l["kq_sc"], pool_l["kq_mn"]
        )
        kmin_pool, kmax_pool = pool_l["kmin"], pool_l["kmax"]
        kavg_pool = pool_l.get("kavg_sum")  # taskmd+avg 时存在（backend 建池）
        Hkv = kmin_pool.shape[2]
        H = q.shape[1]
        G = H // Hkv
        S_cap = kq_pool.shape[1]
        NBLK_CAP = kmin_pool.shape[1]
        bs = p.block_size
        nd2 = self.nd2
        d1 = p.coarse_dim
        if torch.is_tensor(S_list):
            S_t = S_list.to(torch.long)
        else:
            S_t = torch.tensor(S_list, device=device, dtype=torch.long)
        t_t = S_t - 1
        nblk_t = (S_t + bs - 1) // bs
        K1 = min(p.k1_blocks, NBLK_CAP)
        sink_tok = p.sink_blocks * bs
        swa_tok = p.sliding_window
        swa_blks = max(1, swa_tok // bs)
        e64 = p.alpha > 0 and p.beta > 0

        rows_l = rows.to(torch.long)
        blk_id = torch.arange(NBLK_CAP, device=device)
        blk_end = (blk_id + 1) * bs - 1
        valid_blk = (blk_id.view(1, -1) < nblk_t.view(-1, 1)) & (
            blk_end.view(1, -1) <= t_t.view(-1, 1)
        )  # [n, NBLK_CAP]

        # ---- L1 分数（eager einsum：minmax 上界 + 可选 avg 块均值）----
        kmin_b = kmin_pool[rows]  # [n, NBLK_CAP, Hkv, d']（容量行垃圾靠掩码）
        kmax_b = kmax_pool[rows]
        qs = q[..., self.idx1]  # [n, H, d']
        qg = qs.clamp(min=0).reshape(n, Hkv, G, d1)
        qn = qs.clamp(max=0).reshape(n, Hkv, G, d1)
        sc1 = torch.einsum("ahgd,amhd->ahm", qg, kmax_b) + torch.einsum(
            "ahgd,amhd->ahm", qn, kmin_b
        )  # [n, Hkv, NBLK_CAP]
        sc1 = sc1.masked_fill(~valid_blk.unsqueeze(1), float("-inf"))
        sc_avg = None
        if kavg_pool is not None:
            kavg_b = kavg_pool[rows] / bs  # [n, NBLK_CAP, Hkv, d']
            qa = qs.reshape(n, Hkv, G, d1).sum(2)  # [n, Hkv, d']
            sc_avg = torch.einsum("ahd,amhd->ahm", qa, kavg_b)
            sc_avg = sc_avg.masked_fill(
                ~valid_blk.unsqueeze(1), float("-inf")
            )

        if e64:
            # ---- per-row 区域边界（device）----
            # 【B10 修复 2026-10-10】near 左界从 swa 起点推
            # （near_base = S_t − swa_tok），与 _taskmd_regions /
            # _select_batched_taskmd / two-level compute_mask 同源。
            mid_len_t = (S_t - sink_tok - swa_tok).clamp(min=0)
            near_len_dyn_t = torch.maximum(
                torch.full_like(S_t, bs),
                (p.alpha * mid_len_t.float()).long(),
            )
            near_blks_t = torch.maximum(
                torch.full_like(S_t, p.sink_blocks),
                (S_t - swa_tok - near_len_dyn_t) // bs,
            )
            swa_lo_blk_t = torch.maximum(
                near_blks_t, t_t // bs - swa_blks + 1
            )
            in_far_blk = (blk_id.view(1, -1) >= p.sink_blocks) & (
                blk_id.view(1, -1) < near_blks_t.view(-1, 1)
            ) & valid_blk
            in_near_blk = (blk_id.view(1, -1) >= near_blks_t.view(-1, 1)) & (
                blk_id.view(1, -1) < swa_lo_blk_t.view(-1, 1)
            ) & valid_blk
            nb_near = max(1, int(round(K1 * p.beta)))
            nb_far = max(0, K1 - nb_near)
            # far 池
            src_far = (
                sc_avg if (p.far_method == "avg" and sc_avg is not None) else sc1
            )
            sc_far = src_far.masked_fill(
                ~in_far_blk.unsqueeze(1), float("-inf")
            )
            i_f = torch.topk(sc_far, min(nb_far, NBLK_CAP), dim=-1).indices
            keep_f = torch.gather(sc_far, 2, i_f) > float("-inf")
            # near 池
            src_near = (
                sc1
                if (p.near_method == "minmax" or sc_avg is None)
                else sc_avg
            )
            sc_near = src_near.masked_fill(
                ~in_near_blk.unsqueeze(1), float("-inf")
            )
            i_n = torch.topk(sc_near, min(nb_near, NBLK_CAP), dim=-1).indices
            keep_n = torch.gather(sc_near, 2, i_n) > float("-inf")
            # E112 per-head L1 池（two-level topk_mask 的 per-head 语义；
            # 并集 onehot 仅作候选压实的效率口径，s2 打分后掩回 per-head）。
            # far/near 分池 scatter 再 OR（同 _select_batched_taskmd：第二个
            # scatter 的 -inf 并列槽位 False 会覆写掉第一个 scatter 的 True）
            oh_f = torch.zeros(n, Hkv, NBLK_CAP, dtype=torch.bool, device=device)
            oh_f.scatter_(2, i_f, keep_f)
            oh_n = torch.zeros(n, Hkv, NBLK_CAP, dtype=torch.bool, device=device)
            oh_n.scatter_(2, i_n, keep_n)
            onehot_h = oh_f | oh_n
            onehot = onehot_h.any(1)  # [n, NBLK_CAP] 候选并集
        else:
            # ---- 单池：far_method 独占 + 当前块 per-row 强制 inf ----
            sc_pool = (
                sc1 if (p.far_method != "avg" or sc_avg is None) else sc_avg
            )
            sc_pool = sc_pool.clone()
            cur_blk_t = (t_t // bs).view(n, 1, 1)
            sc_pool.scatter_(
                -1,
                cur_blk_t.expand(n, Hkv, 1),
                torch.full(
                    (n, Hkv, 1),
                    float("inf"),
                    dtype=sc_pool.dtype,
                    device=device,
                ),
            )
            i_pool = torch.topk(sc_pool, K1, dim=-1).indices
            keep_p = torch.gather(sc_pool, 2, i_pool) > float("-inf")
            # E112 per-head L1 池（同 e64 分支；swa 块不入 per-head 池——
            # L2 的 swa +inf 强制覆盖 per-head 掩码，two-level p[...,-swa:]
            # =1.0 同样不受 L1 限制）
            onehot_h = torch.zeros(
                n, Hkv, NBLK_CAP, dtype=torch.bool, device=device
            )
            onehot_h.scatter_(2, i_pool, keep_p)
            onehot = onehot_h.any(1)  # [n, NBLK_CAP] 候选并集
            # swa 块并入候选并集（two-level 单池语义：swa token 由 L2
            # +inf 强制必选，但其块须先入候选——S > K1·bs 时 swa 前块
            # 可能不入 topk，swa token 会整个丢失。e64 路径 swa 由
            # forced_out 按位置直接拼接，不受此约束；per-request 路径
            # 同理由全宽 fine 矩阵 + forced=inf 兜住）
            swa_lo_blk_row = ((t_t + 1 - swa_tok).clamp(min=0) // bs).view(
                n, 1
            )
            in_swa_blk = (blk_id.view(1, -1) >= swa_lo_blk_row) & (
                blk_id.view(1, -1) <= cur_blk_t.view(n, 1)
            )
            onehot |= in_swa_blk

        # ---- 候选压实（topk-min，与既有 eager 慢路径同口径）----
        Tc = min((K1 * Hkv + p.sliding_blocks) * bs, NBLK_CAP * bs, S_cap)
        sel_mask = onehot.repeat_interleave(bs, dim=1)[:, :S_cap]  # [n, S_cap]
        pos_cap = torch.arange(S_cap, device=device)
        cand = sel_mask & (pos_cap.view(1, S_cap) < S_t.view(-1, 1))
        seq_m = torch.where(
            cand, pos_cap.view(1, S_cap), torch.full_like(pos_cap, S_cap)
        )
        tok = torch.topk(seq_m, Tc, dim=-1, largest=False).values  # [n, Tc]
        tok_c = tok.clamp(max=S_cap - 1)

        # ---- L2 精筛打分（eager flat gather：M8 fused kernel 旁路）----
        # bug2 修复（GPT 复查 2026-10-08）：q_agg=max 原在 decode 批量
        # taskmd 路径未生效（写死 sum）——补 max 分支（#58）。
        q_g = self._q_refine(q).reshape(n, Hkv, G, nd2)  # [n, Hkv, G, nd2]
        q_agg_max = getattr(p, "q_agg", "sum") == "max"
        q2 = None if q_agg_max else q_g.sum(2)  # [n, Hkv, nd2]
        # E112 对拍口径：零填充回全 D 维（归约树与 two-level / per-request
        # / prefill 批量一致，见 _pad_refine_full；chunk 循环内按 chunk 物化）
        D_eff = q.shape[-1]
        if q_agg_max:
            q_use = (
                self._pad_refine_full(q_g, D_eff) if self.basis is None else q_g
            )  # [n, Hkv, G, D 或 nd2]
        else:
            q_use = (
                self._pad_refine_full(q2, D_eff) if self.basis is None else q2
            )  # [n, Hkv, D 或 nd2]
        D_eff = q_use.shape[-1]
        s2 = torch.empty(n, Hkv, Tc, dtype=torch.float32, device=device)
        d_off = torch.arange(nd2, device=device)
        h_off = torch.arange(Hkv, device=device).view(1, Hkv, 1) * nd2
        h_off_s = torch.arange(Hkv, device=device)
        chunk = max(1, row_chunk_bytes // max(Tc * Hkv * nd2 * 4, 1))
        for rr0 in range(0, n, chunk):
            rr1 = min(rr0 + chunk, n)
            m = rr1 - rr0
            flat = (
                rows_l[rr0:rr1].view(m, 1, 1, 1) * (S_cap * Hkv * nd2)
                + tok_c[rr0:rr1].view(m, Tc, 1, 1) * (Hkv * nd2)
                + h_off.view(1, Hkv, 1)
                + d_off.view(1, 1, nd2)
            )
            flat_s = (
                rows_l[rr0:rr1].view(m, 1, 1) * (S_cap * Hkv)
                + tok_c[rr0:rr1].view(m, Tc, 1) * Hkv
                + h_off_s.view(1, 1, Hkv)
            )
            grid_c = (
                kq_pool.reshape(-1)[flat.view(-1)].view(m, Tc, Hkv, nd2)
            )
            sc_c = kq_sc_pool.reshape(-1)[flat_s.view(-1)].view(m, Tc, Hkv)
            mn_c = kq_mn_pool.reshape(-1)[flat_s.view(-1)].view(m, Tc, Hkv)
            kq_c = grid_c.float() * sc_c.unsqueeze(-1) + mn_c.unsqueeze(-1)
            if self.basis is None:
                # E112 对拍口径：零填充回全 D 维 + 逐行 "hd,thd->ht" 形式
                # （与 two-level / per-request 归约树逐位一致——批量形式
                # "ahd,athd->aht" 有 ulp 差异，见 _pad_refine_full）
                kq_c = self._pad_refine_full(kq_c, D_eff)
                for j in range(m):
                    if q_agg_max:
                        s2[rr0 + j] = torch.einsum(
                            "hgd,thd->hgt", q_use[rr0 + j], kq_c[j]
                        ).max(1).values
                    else:
                        s2[rr0 + j] = torch.einsum(
                            "hd,thd->ht", q_use[rr0 + j], kq_c[j]
                        )
            elif q_agg_max:
                s2[rr0:rr1] = torch.einsum(
                    "ahgd,athd->ahgt", q_use[rr0:rr1], kq_c
                ).max(2).values
            else:
                s2[rr0:rr1] = torch.einsum(
                    "ahd,athd->aht", q_use[rr0:rr1], kq_c
                )
        # E112 per-head L1 掩码：s2 恢复 per-head topk_mask 语义（并集
        # 候选仅 gather 效率口径）；单池路径的 swa +inf 强制在其后覆盖
        # 此掩码（two-level p[..., -swa:]=1.0 同样不受 L1 限制）
        blk_c = tok_c // bs  # [n, Tc]（垃圾槽位 clamp 后指向尾块，由
        # 后续 valid/pool 掩码统一屏蔽）
        in_pool_h = onehot_h.gather(
            2, blk_c.unsqueeze(1).expand(n, Hkv, Tc)
        )  # [n, Hkv, Tc]
        s2 = s2.masked_fill(~in_pool_h, float("-inf"))

        SENT = S_cap
        causal = tok <= t_t.view(-1, 1)
        valid = tok < S_cap
        # swa 起点（per-row）：e64 的 near 池上界与单池的 +inf 强制共用
        swa_lo_t = (t_t + 1 - swa_tok).clamp(min=0)  # [n]

        if e64:
            # ---- L2 双池：γ 预算切分（静态宽 + rank 掩码 + 哨兵）----
            far_hi_t = torch.minimum(near_blks_t * bs, S_t)  # [n]
            swa_w_t = torch.minimum(
                torch.full_like(S_t, swa_tok), S_t
            )  # decode S > dense_threshold 恒满额（防御短行）
            K2_mid_t = (p.token_budget - sink_tok - swa_w_t).clamp(min=0)
            K2_mid_cap = max(0, p.token_budget - sink_tok - swa_tok)
            nt_near_cap = min(int(nb_near * bs * p.gamma), K2_mid_cap)
            far_budget_cap = max(0, K2_mid_cap - nt_near_cap)
            nt_near_t = torch.minimum(
                torch.full_like(K2_mid_t, nt_near_cap), K2_mid_t
            )
            far_budget_t = (K2_mid_t - nt_near_t).clamp(min=0)
            span_t = (far_hi_t - sink_tok).clamp(min=0)
            k2_far_t = torch.minimum(
                torch.minimum(far_budget_t, span_t), K2_mid_t
            )
            in_far = (tok >= sink_tok) & (
                tok < far_hi_t.view(-1, 1)
            ) & valid & causal  # [n, Tc]
            in_near = valid & causal & ~in_far & (
                tok < swa_lo_t.view(-1, 1)
            )
            far_sc = s2.masked_fill(~in_far.unsqueeze(1), float("-inf"))
            near_sc = s2.masked_fill(~in_near.unsqueeze(1), float("-inf"))
            tok_e = tok_c.unsqueeze(1).expand(n, Hkv, Tc)
            W_far = min(far_budget_cap, Tc)
            W_near = min(nt_near_cap, Tc)
            parts = []
            if W_far > 0:
                i_f2 = torch.topk(far_sc, W_far, dim=-1).indices
                sc_f2 = torch.gather(far_sc, 2, i_f2)
                rank_f2 = (
                    torch.arange(W_far, device=device).view(1, 1, -1)
                    < k2_far_t.view(-1, 1, 1)
                )
                keep_f2 = (sc_f2 > float("-inf")) & rank_f2
                sel_f2 = torch.gather(tok_e, 2, i_f2)
                parts.append(
                    torch.where(
                        ~keep_f2, torch.full_like(sel_f2, SENT), sel_f2
                    )
                )
            if W_near > 0:
                i_n2 = torch.topk(near_sc, W_near, dim=-1).indices
                sc_n2 = torch.gather(near_sc, 2, i_n2)
                k2_near_t = (K2_mid_t - k2_far_t).clamp(min=0)
                rank_n2 = (
                    torch.arange(W_near, device=device).view(1, 1, -1)
                    < k2_near_t.view(-1, 1, 1)
                )
                keep_n2 = (sc_n2 > float("-inf")) & rank_n2
                sel_n2 = torch.gather(tok_e, 2, i_n2)
                parts.append(
                    torch.where(
                        ~keep_n2, torch.full_like(sel_n2, SENT), sel_n2
                    )
                )
            # 正交强制区：sink 头部（静态宽 sink_tok；pad 行/短行越界槽位
            # → 哨兵，softmax 前屏蔽——pad 行 forced 恒含位置 0 保无 NaN）
            sink_pos = torch.arange(sink_tok, device=device).view(1, -1)
            sink_pad = (
                torch.arange(sink_tok, device=device).view(1, -1)
                >= S_t.view(-1, 1)
            )
            sink_out = (
                torch.where(sink_pad, torch.full_like(sink_pos, SENT), sink_pos)
                .unsqueeze(1)
                .expand(n, Hkv, -1)
            )
            parts.append(sink_out)
            # swa 尾部（静态宽 swa_tok，M10 forced 段同款）
            W_forced = min(swa_tok, Tc)
            f_pos = swa_lo_t.view(-1, 1) + torch.arange(
                W_forced, device=device
            )
            F_t = (t_t + 1 - swa_lo_t).clamp(min=0)
            f_pad = (
                torch.arange(W_forced, device=device).view(1, -1)
                >= F_t.view(-1, 1)
            )
            forced_out = (
                torch.where(f_pad, torch.full_like(f_pos, SENT), f_pos)
                .unsqueeze(1)
                .expand(n, Hkv, -1)
            )
            parts.append(forced_out)
            return torch.cat(parts, dim=-1)  # [n, Hkv, K2] 静态宽

        # ---- 单池：整行 topk + swa +inf（two-level 单池语义）----
        sc = s2.masked_fill(~(valid & causal).unsqueeze(1), float("-inf"))
        # S2 修复第二处（Kimi3 审查 2026-10-08）：与 _select_batched 的
        # L1975-1978 同款 bug——原条件 (tok >= swa_lo_t) 对哨兵 lane
        # （tok = S_cap 未 clamp ≥ swa_lo_t 恒真）把刚打的 -inf 复活为
        # +inf → topk 被 +inf 哨兵挤占、退化「只看滑窗」、最坏 0 有效
        # lane → softmax 全 -inf = NaN。swa_lo ≤ t 已蕴含 causal，补
        # & valid 即可。taskmd 单池臂（α=0/β=0，含 (0,0) 退化点）的
        # 批量 decode/CUDA graph 路径每步都走这里。
        sc = sc.masked_fill(
            ((tok >= swa_lo_t.view(-1, 1)) & valid).unsqueeze(1),
            float("inf"),
        )
        k = min(p.token_budget, Tc)
        i_g = torch.topk(sc, k, dim=-1).indices
        sc_g = torch.gather(sc, 2, i_g)
        sel_g = torch.gather(tok_c.unsqueeze(1).expand(n, Hkv, Tc), 2, i_g)
        return torch.where(
            sc_g == float("-inf"), torch.full_like(sel_g, SENT), sel_g
        )  # [n, Hkv, budget]

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
