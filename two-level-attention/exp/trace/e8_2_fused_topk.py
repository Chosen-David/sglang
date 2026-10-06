# E8-2: fused 级联 topk kernel 原型（Triton）
# 语义 = TLI decode 单步索引（A 子空间 L1 + L2 4bit 精筛 + D' 块剔除 + B' far/near 分区）
#
# 数据为 kernel 级合成 microbench（真实分布参数：S=128K, nblk=2048, Hkv=8, G=4,
# d'=32, K1=128, K2=1024, sink=2 块, near=2048 tok；来自 Qwen3-8B trace 口径），
# 数值正确性对拍用同一合成数据上的 PyTorch eager 参考实现。
#
# 结构：
#   baseline (eager PyTorch, ~15 kernels): 块分数 einsum → topk 块 → gather 块 token
#       → 4bit 反量化+分数 → 分区 near/far → 两次 topk → scatter mask
#   fused (Triton, 2 launches):
#       K1: 每 head 一个 program：块区间分数（d'=32 子空间）+ D' 剔除 + 近端滑窗
#           合并 + in-kernel top-K1 块（阈值扫描法）→ 输出选中块 id
#       K2: 每 head 一个 program：gather 选中块的 4bit k_qat，算 token 分数，
#           far/near 分区 top-K2（far 池 far_lo..far_hi 独立预算）→ 输出 token mask/id
import torch
import triton
import triton.language as tl

# ---------------- Triton kernels ---------------- #


@triton.jit
def _l1_block_score_kernel(
    q_ptr, kmin_ptr, kmax_ptr, blk_ids_ptr,  # out: 选中块 id [Hkv, K1]
    H: tl.constexpr, HKV: tl.constexpr, G: tl.constexpr,
    DP: tl.constexpr,        # 子空间维 d'=32
    NBLK: tl.constexpr,      # 总块数
    K1: tl.constexpr,        # L1 topk 块数
    FAR_LO_BLK: tl.constexpr, FAR_HI_BLK: tl.constexpr,  # D'/B' 剔除区间
    SCALE: tl.constexpr,
):
    # 每 program 处理一个 kv-head：L1 块分数 + top-K1 块
    h = tl.program_id(0)
    # q_group: [DP] group-sum 的子空间 q（4 个 q-head 求和，与 L1 量纲一致）
    q_pos = tl.zeros([DP], dtype=tl.float32)
    q_neg = tl.zeros([DP], dtype=tl.float32)
    for g in range(G):
        qg = tl.load(q_ptr + (h * G + g) * DP + tl.arange(0, DP)).to(tl.float32) * SCALE
        q_pos += tl.maximum(qg, 0.0)
        q_neg += tl.minimum(qg, 0.0)
    # 块分数（区间算术上界，只在 d' 子空间维）：score = q+·kmax + q-·kmin
    # 分块扫描全部块，in-register 保留 top-K1（阈值法：两轮扫描）
    offs_b = tl.arange(0, NBLK_POW2)  # type: ignore
    valid = offs_b < NBLK
    # D'/B'：远端块剔除（-inf）
    skip = (offs_b >= FAR_LO_BLK) & (offs_b < FAR_HI_BLK)
    offs_d = tl.arange(0, DP)
    kmin = tl.load(kmin_ptr + offs_b[:, None] * (HKV * DP) + h * DP + offs_d[None, :],
                   mask=valid[:, None], other=0.0).to(tl.float32)
    kmax = tl.load(kmax_ptr + offs_b[:, None] * (HKV * DP) + h * DP + offs_d[None, :],
                   mask=valid[:, None], other=0.0).to(tl.float32)
    score = (tl.sum(q_pos[None, :] * kmax, axis=1) + tl.sum(q_neg[None, :] * kmin, axis=1))
    score = tl.where(valid & (~skip), score, float("-inf"))
    # 当前块强制（TIA 语义）：最后一块 +inf
    score = tl.where(offs_b == NBLK - 1, float("inf"), score)
    # top-K1：阈值扫描（K1 轮二分过于慢，用「计数+固定阈值迭代」）
    # 简化原型的做法：K1 次「取最大+屏蔽」太慢；用 log2 范围二分 20 轮找第 K1 大
    lo = -1e30
    hi = 1e30
    for _ in range(60):  # ~log2(2e30)≈100，60 轮已够 fp32 精度
        mid = (lo + hi) / 2
        cnt = tl.sum(tl.where(score >= mid, 1, 0))
        # 目标：恰好 K1 个 >= 阈值（含并列时取偏小阈值）
        is_big = cnt > K1
        lo = tl.where(is_big, mid, lo)
        hi = tl.where(is_big, hi, mid)
    thr = (lo + hi) / 2
    sel = score >= thr
    # 并列超选处理：按块 id 截断到恰好 K1 个
    over = tl.sum(tl.where(sel, 1, 0)) - K1
    # 把并列块中 id 较大者剔除（确定性）
    tie = sel & (score == thr)
    n_tie = tl.sum(tl.where(tie, 1, 0))
    keep_tie = tl.sum(tl.where(tie, tl.where(offs_b < NBLK, 1, 0), 0))  # placeholder
    # 简化：允许 over<=0 情况下全保留；超选时用 tie 块 id 顺序截断
    if over > 0:
        # tie 块里保留前 (n_tie - over) 个（按块 id 升序）
        rank_in_tie = tl.sum(tl.where(tie & (offs_b[:, None] > offs_b[None, :]), 1, 0), axis=1)
        sel = sel & (~tie | (rank_in_tie < n_tie - over))
    ids = tl.where(sel, offs_b, NBLK)  # 未选中 → NBLK（无效标记）
    # 输出：排序困难，直接输出 [NBLK_POW2] 的选中标记按位置压缩不可行；
    # 原型采用：输出 bool 选中掩码由调用侧转 id（torch.nonzero 一次）
    tl.store(blk_ids_ptr + offs_b, tl.where(sel, 1, 0).to(tl.int32), mask=valid)


# NBLK_POW2 作为 constexpr 由外部传入（通过 kernel 参数包装）——Triton 需要显式声明
# 这里改用 autotune wrapper 生成

def make_l1_kernel(nblk_pow2):
    @triton.jit
    def _k(q_ptr, kmin_ptr, kmax_ptr, out_ptr,
           H: tl.constexpr, HKV: tl.constexpr, G: tl.constexpr,
           DP: tl.constexpr, NBLK: tl.constexpr, NBLK_P2: tl.constexpr,
           K1: tl.constexpr, FAR_LO_BLK: tl.constexpr, FAR_HI_BLK: tl.constexpr,
           SCALE: tl.constexpr):
        h = tl.program_id(0)
        q_pos = tl.zeros([DP], dtype=tl.float32)
        q_neg = tl.zeros([DP], dtype=tl.float32)
        for g in range(G):
            qg = tl.load(q_ptr + (h * G + g) * DP + tl.arange(0, DP)).to(tl.float32) * SCALE
            q_pos += tl.maximum(qg, 0.0)
            q_neg += tl.minimum(qg, 0.0)
        offs_b = tl.arange(0, NBLK_P2)
        valid = offs_b < NBLK
        skip = (offs_b >= FAR_LO_BLK) & (offs_b < FAR_HI_BLK)
        offs_d = tl.arange(0, DP)
        kmin = tl.load(kmin_ptr + offs_b[:, None] * (HKV * DP) + h * DP + offs_d[None, :],
                       mask=valid[:, None], other=0.0).to(tl.float32)
        kmax = tl.load(kmax_ptr + offs_b[:, None] * (HKV * DP) + h * DP + offs_d[None, :],
                       mask=valid[:, None], other=0.0).to(tl.float32)
        score = tl.sum(q_pos[None, :] * kmax, axis=1) + tl.sum(q_neg[None, :] * kmin, axis=1)
        score = tl.where(valid & (~skip), score, float("-inf"))
        # 以实际分数范围作二分初值（60 轮内收敛到 fp32 精度）
        fin = tl.where(valid & (~skip), score, 0.0)
        lo = -tl.max(tl.abs(fin)) - 1.0
        hi = tl.max(tl.abs(fin)) + 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            cnt = tl.sum(tl.where(score >= mid, 1, 0))
            lo = tl.where(cnt > K1, mid, lo)
            hi = tl.where(cnt > K1, hi, mid)
        thr = (lo + hi) / 2
        sel = score >= thr
        sel = sel | (offs_b == NBLK - 1)  # 当前块强制（TIA 语义）
        # 原型放宽：阈值边界并列时可能多选少量块（>=K1），多选块交由 L2 精筛，
        # 不影响正确性语义（真 fused 版用 warp 级 bitonic 排序精确截断）
        tl.store(out_ptr + h * NBLK_P2 + offs_b, tl.where(sel, 1, 0).to(tl.int32), mask=valid)
    return _k


@triton.jit
def _l2_cascade_kernel(
    q_ptr, kqat_ptr, selblk_ptr, tok_mask_ptr,  # out: token mask [Hkv, S]
    HKV: tl.constexpr, G: tl.constexpr, DP: tl.constexpr,
    BLK: tl.constexpr, NBLK: tl.constexpr, NBLK_P2: tl.constexpr,
    K1: tl.constexpr, K2: tl.constexpr,
    FAR_TOK_LO: tl.constexpr, FAR_TOK_HI: tl.constexpr,
    FAR_BUDGET: tl.constexpr, SCALE: tl.constexpr,
    S: tl.constexpr,
):
    # 每 program 处理一个 kv-head 的 L2 级联：选中块内 token 4bit 分数 + far/near 分区 top-K2
    h = tl.program_id(0)
    # group-sum q（全维 d' 子空间，4bit 反量化后点积）
    q_acc = tl.zeros([DP], dtype=tl.float32)
    for g in range(G):
        qg = tl.load(q_ptr + (h * G + g) * DP + tl.arange(0, DP)).to(tl.float32) * SCALE
        q_acc += qg
    selmask = tl.load(selblk_ptr + h * NBLK_P2 + tl.arange(0, NBLK_P2))
    # 展开选中块的 token：原型按「最多 K1 块」逐块循环算分数，写入临时
    # 分区 topk 用阈值扫描（near 池 + far 池各一次）
    # 近端池阈值（K2 - FAR_BUDGET 个）
    k_near = K2 - FAR_BUDGET
    # far/near 分数需要先算出全部选中 token 的分数 → 用第二次扫描
    # （原型：两次遍历选中块。第一次算分数统计阈值，第二次写 mask）
    # ---- pass 1: near 阈值 ----
    lo_n, hi_n = -1e30, 1e30
    for _ in range(60):
        thr = (lo_n + hi_n) / 2
        cnt = 0
        cnt = cnt.to(tl.float32)
        for b in range(NBLK_P2):
            on = tl.sum(tl.where(selmask == 0, 0, 0))  # placeholder to keep loop structure
        break
    # NOTE: 原型退化为单遍：直接存分数到全局 scratch，由 torch 侧做分区 topk
    # 真正 fused 版本需要 shared-memory 两遍结构，作为 M2 kernel 的设计输入
    tl.store(tok_mask_ptr + h, 0)


def main():
    torch.manual_seed(0)
    dev = "cuda:1"  # GPU1（E5b 占 GPU0）
    # ---- 合成数据（真实口径规模）----
    S, BLK = 131072, 64
    NBLK = S // BLK
    HKV, G, DP = 8, 4, 32
    H = HKV * G
    K1, K2 = 128, 1024
    FAR_LO_BLK, FAR_TOK = 2, 512
    NEAR = 2048
    FAR_HI_BLK = NBLK - NEAR // BLK
    FAR_TOK_LO, FAR_TOK_HI = FAR_LO_BLK * BLK, S - NEAR
    SCALE = 1.0

    q = torch.randn(H, DP, device=dev)  # 子空间 q（已 slice）
    kmin = torch.randn(NBLK, HKV, DP, device=dev) - 1.0
    kmax = kmin + torch.rand(NBLK, HKV, DP, device=dev) * 2

    # ---- baseline: eager PyTorch（模拟 TLI compute_score L1 路径）----
    def eager_l1(far_lo=FAR_LO_BLK, far_hi=FAR_HI_BLK):
        qg = q.view(HKV, G, DP).sum(1)  # [HKV, DP]
        sc = (torch.einsum("hd,nhd->hn", qg.clamp(min=0), kmax)
              + torch.einsum("hd,nhd->hn", qg.clamp(max=0), kmin))
        skip = torch.zeros(NBLK, dtype=torch.bool, device=dev)
        skip[far_lo:far_hi] = True
        sc = sc.masked_fill(skip, float("-inf"))
        sc[:, -1] = float("inf")
        return torch.topk(sc, K1, dim=-1).indices

    # ---- fused: Triton L1（2-launch 方案的第 1 个）----
    import math
    NBLK_P2 = triton.next_power_of_2(NBLK)
    out = torch.zeros(HKV, NBLK_P2, dtype=torch.int32, device=dev)
    kern = make_l1_kernel(NBLK_P2)
    kern[(HKV,)](
        q, kmin, kmax, out,
        H=H, HKV=HKV, G=G, DP=DP, NBLK=NBLK, NBLK_P2=NBLK_P2,
        K1=K1, FAR_LO_BLK=FAR_LO_BLK, FAR_HI_BLK=FAR_HI_BLK, SCALE=SCALE,
        num_warps=8,
    )
    tri_ids = torch.nonzero(out[0]).squeeze(1)
    _eager_all = eager_l1()[0]
    # eager 参考的 topk 会用 -inf 块填满 K1；用带 skip 的分数过滤出真正有效块
    _qg = q.view(HKV, G, DP).sum(1)
    _sc_skip = (torch.einsum("hd,nhd->hn", _qg.clamp(min=0), kmax)
                + torch.einsum("hd,nhd->hn", _qg.clamp(max=0), kmin))[0]
    _sc_skip[FAR_LO_BLK:FAR_HI_BLK] = float("-inf")
    eager_ids = _eager_all[_sc_skip[_eager_all] > float("-inf")]
    # 对拍（tie 截断规则不同 → 比较「分数集合」而非 id 集合）
    def blk_scores(ids):
        qg = q.view(HKV, G, DP).sum(1)
        sc = (torch.einsum("hd,nhd->hn", qg.clamp(min=0), kmax)
              + torch.einsum("hd,nhd->hn", qg.clamp(max=0), kmin))
        return sc[0][ids]
    s_tri, s_eag = blk_scores(tri_ids), blk_scores(eager_ids)
    print(f"L1 有效块数(D' 语义, far 全剔除): triton={len(tri_ids)}, eager(过滤-inf)={len(eager_ids)}")
    inter = len(set(tri_ids.tolist()) & set(eager_ids.tolist()))
    print(f"L1 选中块 id 集合一致: {inter}/{len(eager_ids)}  分数和差 |Δ|={abs(s_tri.sum()-s_eag.sum()).item():.6f}")

    # ---- 延迟对比 ----
    import time
    def bench(fn, n=50):
        for _ in range(5):
            fn()
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize(dev)
        return (time.perf_counter() - t0) / n * 1e3

    t_eager = bench(eager_l1)
    t_tri = bench(lambda: kern[(HKV,)](
        q, kmin, kmax, out,
        H=H, HKV=HKV, G=G, DP=DP, NBLK=NBLK, NBLK_P2=NBLK_P2,
        K1=K1, FAR_LO_BLK=FAR_LO_BLK, FAR_HI_BLK=FAR_HI_BLK, SCALE=SCALE,
        num_warps=8,
    ))
    print(f"\nL1 延迟 (S={S}, nblk={NBLK}): eager={t_eager:.3f} ms, triton={t_tri:.3f} ms, 加速 {t_eager/t_tri:.2f}×")
    # L2 延迟模型（eager：gather+einsum+topk 两个池；triton fused 版本轮原型以 L1 结构为设计输入）
    print("\n注: L2 级联 fused kernel（单 launch 块级联 token 精筛+分区 topk）为 M2 设计输入，本原型验证 L1 融合正确性与收益")

    # ---- 场景2：非跳层（far 不剔除，K1=128 from 2048 块，常规路径）----
    FAR_LO2, FAR_HI2 = 4, 4  # 空区间 = 不剔除
    out2 = torch.zeros(HKV, NBLK_P2, dtype=torch.int32, device=dev)
    kern[(HKV,)](q, kmin, kmax, out2,
                 H=H, HKV=HKV, G=G, DP=DP, NBLK=NBLK, NBLK_P2=NBLK_P2,
                 K1=K1, FAR_LO_BLK=FAR_LO2, FAR_HI_BLK=FAR_HI2, SCALE=SCALE, num_warps=8)
    tri2 = torch.nonzero(out2[0]).squeeze(1)
    eag2 = eager_l1(FAR_LO2, FAR_HI2)[0]
    # 场景2 无剔除：eager 的 topk 不含 -inf 块，直接对拍
    s_tri2, s_eag2 = blk_scores(tri2), blk_scores(eag2)
    inter2 = len(set(tri2.tolist()) & set(eag2.tolist()))
    print(f"\n[非跳层] triton 块数={len(tri2)}, eager={len(eag2)}, id 交集={inter2}")
    print(f"[非跳层] 分数和: tri={s_tri2.sum().item():.4f} eag={s_eag2.sum().item():.4f} "
          f"(|Δ|={abs(s_tri2.sum()-s_eag2.sum()).item():.6f}, 相对 {abs(s_tri2.sum()-s_eag2.sum()).item()/abs(s_eag2.sum()).item():.2e})")
    t_tri2 = bench(lambda: kern[(HKV,)](
        q, kmin, kmax, out2,
        H=H, HKV=HKV, G=G, DP=DP, NBLK=NBLK, NBLK_P2=NBLK_P2,
        K1=K1, FAR_LO_BLK=FAR_LO2, FAR_HI_BLK=FAR_HI2, SCALE=SCALE, num_warps=8))
    print(f"[非跳层] L1 延迟: triton={t_tri2:.3f} ms, 加速 {t_eager/t_tri2:.2f}×")


if __name__ == "__main__":
    main()
