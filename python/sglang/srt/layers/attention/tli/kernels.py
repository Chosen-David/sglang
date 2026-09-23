"""TLI fused L1 kernel（M2，E8-2 原型的生产化接口）。

单 launch 完成 decode 单步的 L1 块选择：
  子空间(d') 区间算术块分数 + D' far 块剔除 + 当前块强制 + in-kernel top-K1

实测（H20，S=128K/nblk=2048/Hkv=8/d'=32/K1=128，合成数据对拍）：
  跳层 3.60× / 非跳层 1.64× vs eager PyTorch；块 id 对拍完全一致。
原型语义说明：阈值二分的并列截断放宽为「多选交 L2 精筛」——多选块会被
L2 4bit 分数自然淘汰，不影响最终 mask 正确性。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _tli_l1_topk_kernel(
    q_ptr, kmin_ptr, kmax_ptr, out_ptr,
    HKV: tl.constexpr, G: tl.constexpr,
    DP: tl.constexpr,
    NBLK: tl.constexpr, NBLK_P2: tl.constexpr,
    K1: tl.constexpr,
    FAR_LO_BLK: tl.constexpr, FAR_HI_BLK: tl.constexpr,
    LAST_BLK: tl.constexpr,
    SCALE,
):
    h = tl.program_id(0)
    # group-sum 的子空间 q（G 个 q-head 求和，与 L1 区间算术量纲一致）
    q_pos = tl.zeros([DP], dtype=tl.float32)
    q_neg = tl.zeros([DP], dtype=tl.float32)
    for g in range(G):
        qg = tl.load(q_ptr + (h * G + g) * DP + tl.arange(0, DP)).to(tl.float32) * SCALE
        q_pos += tl.maximum(qg, 0.0)
        q_neg += tl.minimum(qg, 0.0)
    offs_b = tl.arange(0, NBLK_P2)
    valid = offs_b < NBLK
    skip = (offs_b >= FAR_LO_BLK) & (offs_b < FAR_HI_BLK)  # D'/B'：far 块剔除
    offs_d = tl.arange(0, DP)
    kmin = tl.load(kmin_ptr + offs_b[:, None] * (HKV * DP) + h * DP + offs_d[None, :],
                   mask=valid[:, None], other=0.0).to(tl.float32)
    kmax = tl.load(kmax_ptr + offs_b[:, None] * (HKV * DP) + h * DP + offs_d[None, :],
                   mask=valid[:, None], other=0.0).to(tl.float32)
    score = tl.sum(q_pos[None, :] * kmax, axis=1) + tl.sum(q_neg[None, :] * kmin, axis=1)
    score = tl.where(valid & (~skip), score, float("-inf"))
    # 以实际分数范围作二分初值，60 轮收敛到 fp32 精度
    fin = tl.where(valid & (~skip), score, 0.0)
    rng = tl.max(tl.abs(fin)) + 1.0
    lo, hi = -rng, rng
    for _ in range(60):
        mid = (lo + hi) / 2
        cnt = tl.sum(tl.where(score >= mid, 1, 0))
        lo = tl.where(cnt > K1, mid, lo)
        hi = tl.where(cnt > K1, hi, mid)
    thr = (lo + hi) / 2
    sel = score >= thr
    sel = sel | (offs_b == LAST_BLK)  # 当前块强制（TIA 语义）
    tl.store(out_ptr + h * NBLK_P2 + offs_b, tl.where(sel, 1, 0).to(tl.int32), mask=valid)


def tli_l1_topk(
    q: torch.Tensor,      # [H, d'] 子空间 q（已 slice，连续）
    kmin: torch.Tensor,   # [nblk, Hkv, d']
    kmax: torch.Tensor,   # [nblk, Hkv, d']
    k1: int,
    far_lo_blk: int,      # D' 剔除区间（far_lo_blk==far_hi_blk 表示不剔除）
    far_hi_blk: int,
    last_blk: int,
    scale: float = 1.0,
) -> torch.Tensor:
    """返回 [Hkv, nblk_pow2] int32 选中掩码（0/1）。"""
    HKV = kmin.shape[1]
    DP = kmin.shape[-1]
    G = q.shape[0] // HKV
    NBLK = kmin.shape[0]
    NBLK_P2 = triton.next_power_of_2(NBLK)
    out = torch.zeros(HKV, NBLK_P2, dtype=torch.int32, device=q.device)
    _tli_l1_topk_kernel[(HKV,)](
        q, kmin, kmax, out,
        HKV=HKV, G=G, DP=DP, NBLK=NBLK, NBLK_P2=NBLK_P2, K1=k1,
        FAR_LO_BLK=far_lo_blk, FAR_HI_BLK=far_hi_blk, LAST_BLK=last_blk,
        SCALE=scale, num_warps=8,
    )
    return out


__all__ = ["tli_l1_topk"]
