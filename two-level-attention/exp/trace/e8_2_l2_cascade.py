# E8-2 下半场：L2 级联 fused kernel（Triton）
# 语义 = TLI decode 的 L2 阶段：选中块 token 的 4bit 精筛分数 + far/near 分区 top-K2
# 结构（2 launches + 1 个 2048 元素的小 compaction）：
#   L1 fused kernel（已验证）→ cand_pos compaction → 本 kernel（单 launch/head）
#   每 program 一个 kv-head：候选 token 分数全部驻留寄存器（~8320 fp32），
#   far/near 两个池各做阈值二分 topk，最后 scatter 写 token mask
# 对拍：与 eager 分区 topk 语义对齐（PyTorch 参考实现）
import time

import torch
import triton
import triton.language as tl

dev = "cuda:0"  # 由 CUDA_VISIBLE_DEVICES 控制（测试时用空闲 GPU1）


@triton.jit
def _l2_partition_topk_kernel(
    q_ptr, kq_ptr, cand_ptr, mask_out_ptr,
    HKV: tl.constexpr, G: tl.constexpr, ND2: tl.constexpr,
    TC: tl.constexpr, TC_P2: tl.constexpr, S: tl.constexpr,
    K2_FAR: tl.constexpr, K2_NEAR: tl.constexpr,
    FAR_LO: tl.constexpr, FAR_HI: tl.constexpr,
    SW_LO: tl.constexpr,  # 滑窗起点（强制入选）
    SCALE,
):
    h = tl.program_id(0)
    # group-sum q（4bit 精筛子空间，G 个 q-head 求和）
    q2 = tl.zeros([ND2], dtype=tl.float32)
    for g in range(G):
        q2 += tl.load(q_ptr + (h * G + g) * ND2 + tl.arange(0, ND2)).to(tl.float32) * SCALE
    # 候选 token 分数（全驻寄存器）
    offs_c = tl.arange(0, TC_P2)
    valid = offs_c < TC
    pos = tl.load(cand_ptr + offs_c, mask=valid, other=0)  # token 位置
    offs_d = tl.arange(0, ND2)
    kq = tl.load(kq_ptr + pos[:, None] * (HKV * ND2) + h * ND2 + offs_d[None, :],
                 mask=valid[:, None], other=0.0).to(tl.float32)  # [TC_P2, ND2]
    score = tl.sum(q2[None, :] * kq, axis=1)
    score = tl.where(valid, score, float("-inf"))
    # 滑窗强制（TIA 语义）
    forced = pos >= SW_LO
    far = (pos >= FAR_LO) & (pos < FAR_HI) & (~forced)
    near = valid & (~far) & (~forced)
    # ---- far 池 top-K2_FAR（阈值二分）----
    fin = tl.where(far, score, 0.0)
    rng = tl.max(tl.abs(fin)) + 1.0
    lo, hi = -rng, rng
    for _ in range(60):
        mid = (lo + hi) / 2
        cnt = tl.sum(tl.where(far & (score >= mid), 1, 0))
        lo = tl.where(cnt > K2_FAR, mid, lo)
        hi = tl.where(cnt > K2_FAR, hi, mid)
    thr_f = (lo + hi) / 2
    sel_far = far & (score >= thr_f)
    # ---- near 池 top-K2_NEAR ----
    fin_n = tl.where(near, score, 0.0)
    rng_n = tl.max(tl.abs(fin_n)) + 1.0
    lo_n, hi_n = -rng_n, rng_n
    for _ in range(60):
        mid = (lo_n + hi_n) / 2
        cnt = tl.sum(tl.where(near & (score >= mid), 1, 0))
        lo_n = tl.where(cnt > K2_NEAR, mid, lo_n)
        hi_n = tl.where(cnt > K2_NEAR, hi_n, mid)
    thr_n = (lo_n + hi_n) / 2
    sel_near = near & (score >= thr_n)
    sel = (sel_far | sel_near | forced) & valid
    # scatter 写 token mask（per-head：mask_out [HKV, S]）
    tl.store(mask_out_ptr + h * S + pos, 1, mask=sel)


def l2_partition_topk(
    q2: torch.Tensor,        # [H, nd2]（已 slice 的 4bit 精筛子空间 q）
    kq: torch.Tensor,        # [S, Hkv, nd2] dequant 4bit
    cand_pos: torch.Tensor,  # [Tc] 选中块展开的 token 位置（已含滑窗块）
    S: int,
    k2_far: int, k2_near: int,
    far_lo: int, far_hi: int,
    sw_lo: int,
    scale: float = 1.0,
) -> torch.Tensor:
    HKV = kq.shape[1]
    ND2 = kq.shape[-1]
    G = q2.shape[0] // HKV
    TC = cand_pos.shape[0]
    TC_P2 = triton.next_power_of_2(max(TC, 16))
    mask_out = torch.zeros(HKV, S, dtype=torch.int32, device=q2.device)
    _l2_partition_topk_kernel[(HKV,)](
        q2, kq, cand_pos, mask_out,
        HKV=HKV, G=G, ND2=ND2, TC=TC, TC_P2=TC_P2, S=S,
        K2_FAR=k2_far, K2_NEAR=k2_near,
        FAR_LO=far_lo, FAR_HI=far_hi, SW_LO=sw_lo,
        SCALE=scale, num_warps=8,
    )
    return mask_out


def main():
    torch.manual_seed(0)
    # 合成数据（真实口径）
    S, BLK = 131072, 64
    NBLK = S // BLK
    HKV, G, ND2 = 8, 4, 32
    H = HKV * G
    K1, K2, FAR_TOKENS = 128, 1024, 256
    FAR_LO = 2 * BLK
    NEAR = 2048
    FAR_HI = S - NEAR
    SW = 128
    SW_LO = S - SW

    q2 = torch.randn(H, ND2, device=dev)
    kq = torch.randn(S, HKV, ND2, device=dev)

    # 选中块：随机 K1 块 + 滑窗块 + 当前块
    selblk = torch.randperm(NBLK, device=dev)[:K1]
    selblk = torch.cat([selblk, torch.arange(NBLK - 3, NBLK, device=dev)])
    blk_mask = torch.zeros(NBLK, dtype=torch.bool, device=dev)
    blk_mask[selblk] = True
    cand_pos = torch.nonzero(blk_mask.repeat_interleave(BLK)).squeeze(1)[: S]
    TC = cand_pos.shape[0]

    # ---- eager 参考（TLI compute_mask 分区语义，per-head）----
    def eager():
        qg = q2.view(HKV, G, ND2).sum(1)  # [HKV, ND2]
        fine = torch.einsum("hd,thd->ht", qg, kq[cand_pos])  # [HKV, TC]
        pos = cand_pos
        forced = pos >= SW_LO
        far = (pos >= FAR_LO) & (pos < FAR_HI) & (~forced)
        near = (~far) & (~forced)
        m = torch.zeros(HKV, S, dtype=torch.bool, device=dev)
        m[:, pos[forced]] = True
        for h in range(HKV):
            i_f = torch.topk(torch.where(far, fine[h], float("-inf")),
                             min(FAR_TOKENS, int(far.sum())), dim=-1).indices
            i_n = torch.topk(torch.where(near, fine[h], float("-inf")),
                             min(K2 - FAR_TOKENS, int(near.sum())), dim=-1).indices
            m[h, pos[i_f]] = True
            m[h, pos[i_n]] = True
        return m

    t0 = time.perf_counter()
    m_eager = eager()
    torch.cuda.synchronize(dev)
    print(f"eager L2: {(time.perf_counter()-t0)*1e3:.3f} ms")

    # ---- triton fused ----
    t0 = time.perf_counter()
    m_tri = l2_partition_topk(q2, kq, cand_pos, S,
                              min(FAR_TOKENS, int(((cand_pos >= FAR_LO) & (cand_pos < FAR_HI)).sum())),
                              K2 - FAR_TOKENS, FAR_LO, FAR_HI, SW_LO)
    torch.cuda.synchronize(dev)
    print(f"triton L2: {(time.perf_counter()-t0)*1e3:.3f} ms")

    # ---- 对拍 ----
    m_t = m_tri > 0
    inter = int((m_eager & m_t).sum())
    only_e = int((m_eager & ~m_t).sum())
    only_t = int((~m_eager & m_t).sum())
    print(f"mask: eager={int(m_eager.sum())}, triton={int(m_t.sum())}, 交集={inter}, 仅eager={only_e}, 仅triton={only_t}")

    # 延迟 benchmark（不含 compaction：cand_pos 预先算好）
    def bench(fn, n=50):
        for _ in range(5):
            fn()
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize(dev)
        return (time.perf_counter() - t0) / n * 1e3

    t_e = bench(eager)
    nfar = int(((cand_pos >= FAR_LO) & (cand_pos < FAR_HI)).sum())
    t_t = bench(lambda: l2_partition_topk(q2, kq, cand_pos, S, min(FAR_TOKENS, nfar),
                                          K2 - FAR_TOKENS, FAR_LO, FAR_HI, SW_LO))
    print(f"\nL2 延迟 (Tc={TC}): eager={t_e:.3f} ms, triton={t_t:.3f} ms, 加速 {t_e/t_t:.2f}×")


if __name__ == "__main__":
    main()
