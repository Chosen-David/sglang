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


__all__ = ["tli_l1_topk", "tli_l2_partition_topk"]


@triton.jit
def _tli_l2_score_kernel(
    q_ptr, kq_ptr, cand_ptr, far_scr_ptr, near_scr_ptr,
    HKV: tl.constexpr, G: tl.constexpr, ND2: tl.constexpr,
    TC, TC_P2: tl.constexpr, CHUNK: tl.constexpr,
    FAR_LO, FAR_HI, SW_LO, S,
    SCALE,
):
    """L2 级联精筛 pass1（E8-2 生产化修正版）：单 launch/head 对候选 token
    打 4bit 精筛分数，分 far/near 池写 scratch（池外 -inf）。

    top-K2 交给 torch.topk（纯 kernel 版实测两败：①阈值二分在 4bit 量化
    分数上并列极多、最后一级无人兜底 → 每 head 过选 130-256；②120 轮
    二分 × 逐 chunk 串行读，8 program 打不满 78 SM，延迟受限反慢 4.5×）。
    本 pass1 的价值 = 融合 gather+GEMV，省掉 eager 的 [Tc,Hkv,nd2] fp32
    中间量（131K 时 ~51MB 写+读）与 [Hkv,S] fine 矩阵 scatter。

    因果性：FAR_HI ≤ t+1-near_len < t、SW_LO = t-sw+1 < t，两池上界
    均严格 ≤ t——L1 topk 在 nblk < K1 时选中的 -inf 垃圾块（位置 > t）
    天然落在两池之外（eager 靠 fine 矩阵因果 mask 兜底，此处靠池边界）。
    """
    h = tl.program_id(0)
    # group-sum q（4bit 精筛子空间，G 个 q-head 求和）
    q2 = tl.zeros([ND2], dtype=tl.float32)
    for g in range(G):
        q2 += tl.load(q_ptr + (h * G + g) * ND2 + tl.arange(0, ND2)).to(tl.float32) * SCALE
    offs_d = tl.arange(0, ND2)
    for c0 in range(0, TC_P2, CHUNK):
        offs_c = c0 + tl.arange(0, CHUNK)
        valid = offs_c < TC
        pos = tl.load(cand_ptr + offs_c, mask=valid, other=0)
        kq = tl.load(kq_ptr + pos[:, None] * (HKV * ND2) + h * ND2 + offs_d[None, :],
                     mask=valid[:, None], other=0.0).to(tl.float32)
        sc = tl.sum(q2[None, :] * kq, axis=1)
        far = valid & (pos >= FAR_LO) & (pos < FAR_HI)
        near = valid & (~far) & (pos < SW_LO)
        tl.store(far_scr_ptr + h * TC_P2 + offs_c,
                 tl.where(far, sc, float("-inf")), mask=valid)
        tl.store(near_scr_ptr + h * TC_P2 + offs_c,
                 tl.where(near, sc, float("-inf")), mask=valid)


def tli_l2_partition_topk(
    q: torch.Tensor,      # [H, nd2] 4bit 精筛子空间 q（已 slice，未 G-sum）
    kq: torch.Tensor,     # [S, Hkv, nd2] dequant 4bit（须连续）
    cand_pos: torch.Tensor,  # [Tc] 选中块展开的 token 位置（升序，含滑窗块）
    S: int,
    k2_far: int,
    k2_near: int,
    far_lo: int,
    far_hi: int,
    sw_lo: int,
    t: int,
    scale: float = 1.0,
) -> torch.Tensor:
    """返回 [Hkv, K2] 选中 token 位置（int64，池不足时 pad = 哨兵 S）。

    哨兵约定：位置合法值域 [0, S)，S 为 pad；下游 valid = sel < S。
    topk 池不足时选到 -inf pad lane → gather 出哨兵 S，天然去 junk。

    与 eager select 的精确对齐（两处语义修正的来源）：
    - 滑窗 [sw_lo, t] 全部位置强制入选（不管是否候选——eager 的 +inf
      语义），近端配额相应扣减 F = t+1-sw_lo → 输出长度恰为 K2
    - far/near 池边界即因果边界（见 kernel docstring）
    """
    HKV = kq.shape[1]
    ND2 = kq.shape[-1]
    G = q.shape[0] // HKV
    TC = cand_pos.shape[0]
    TC_P2 = triton.next_power_of_2(max(TC, 16))
    far_scr = torch.full((HKV, TC_P2), float("-inf"), dtype=torch.float32,
                         device=q.device)
    near_scr = torch.full((HKV, TC_P2), float("-inf"), dtype=torch.float32,
                          device=q.device)
    _tli_l2_score_kernel[(HKV,)](
        q, kq, cand_pos, far_scr, near_scr,
        HKV=HKV, G=G, ND2=ND2, TC=TC, TC_P2=TC_P2, CHUNK=1024,
        FAR_LO=far_lo, FAR_HI=far_hi, SW_LO=sw_lo, S=S,
        SCALE=scale, num_warps=8,
    )
    # 候选 pad 到 TC_P2（哨兵 S）——topk 落到 -inf pad lane 时 gather 出哨兵
    cand_pad = torch.full((TC_P2,), S, dtype=torch.int64, device=q.device)
    cand_pad[:TC] = cand_pos
    # 滑窗强制：窗口 token 全入选（head 无关），近端配额扣减
    F = t + 1 - sw_lo  # ≤ sliding_window
    k2_near_eff = max(0, k2_near - F)
    i_f = torch.topk(far_scr, k2_far, dim=-1).indices  # [HKV, k2_far] 精确
    i_n = torch.topk(near_scr, k2_near_eff, dim=-1).indices
    sel_f = torch.gather(cand_pad.expand(HKV, -1), 1, i_f)
    sel_n = torch.gather(cand_pad.expand(HKV, -1), 1, i_n)
    forced = torch.arange(sw_lo, t + 1, device=q.device).unsqueeze(0).expand(HKV, -1)
    return torch.cat([sel_f, sel_n, forced], dim=-1)  # [HKV, K2]（池足时）
