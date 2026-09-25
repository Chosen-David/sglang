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


__all__ = [
    "tli_l1_topk",
    "tli_l1_score_batched",
    "tli_l2_partition_topk",
    "tli_l2_score_batched",
    "tli_l2_score_batched_dual",
    "tli_compact",
]


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


# ---------------- M8：批量 L2 fused gather+dequant+GEMV ----------------
# select_decode_batched 的 P5 瓶颈（87.6%@bs32/131K）：eager 路径物化
# kq_c fp32 2.1GB×2(rw) + 逐元素 flat gather（~240GB/s 有效）。
# 本 kernel：grid (n, Tc/CHUNK)，每 program 对 CHUNK 个候选 token 直接从
# pool uint8 gather（每 token 的 HKV*ND2 字节连续段）→ 寄存器内反量化 →
# GEMV 打分 → 写 s2，消除全部中间物化。实测 20.5→0.48ms（43×，1.55TB/s）。
# CHUNK 必须 ≤512：1024 时 tile fp32 化 262144 元素寄存器溢出到 local memory，
# 反而 3.2ms（6.7× 慢）。对拍口径 s2 max diff 2.4e-07（归约顺序级）。
@triton.jit
def _tli_l2_score_batched_kernel(
    q2_ptr, kq_ptr, sc_ptr, mn_ptr, rows_ptr, tok_ptr, s2_ptr,
    HKV: tl.constexpr, ND2: tl.constexpr,
    TC: tl.constexpr,
    S_CAP,
    CHUNK: tl.constexpr,
):
    a = tl.program_id(0)
    c0 = tl.program_id(1) * CHUNK
    offs_c = c0 + tl.arange(0, CHUNK)
    cm = offs_c < TC
    row = tl.load(rows_ptr + a).to(tl.int64)
    pos = tl.load(tok_ptr + a * TC + offs_c, mask=cm, other=0).to(tl.int64)
    offs_h = tl.arange(0, HKV)
    offs_d = tl.arange(0, ND2)
    # uint8 gather [CHUNK, HKV, ND2]：每 token 的 HKV*ND2 字节连续
    base = row * (S_CAP * HKV * ND2) + pos * (HKV * ND2)  # [CHUNK] int64
    addr = base[:, None, None] + (offs_h[None, :, None] * ND2 + offs_d[None, None, :])
    kq = tl.load(kq_ptr + addr, mask=cm[:, None, None], other=0).to(tl.float32)
    # scale / mn [CHUNK, HKV]
    base_s = row * (S_CAP * HKV) + pos * HKV  # [CHUNK]
    addr_s = base_s[:, None] + offs_h[None, :]
    sc = tl.load(sc_ptr + addr_s, mask=cm[:, None], other=0.0)
    mn = tl.load(mn_ptr + addr_s, mask=cm[:, None], other=0.0)
    # 寄存器内反量化（与 eager grid*sc+mn 同运算序，格点级逐位一致）
    kq_c = kq * sc[:, :, None] + mn[:, :, None]
    # GEMV：q2 [HKV, ND2]，对每 (chunk, h) 归约 d
    q2 = tl.load(q2_ptr + a * HKV * ND2 + offs_h[:, None] * ND2 + offs_d[None, :])
    s = tl.sum(kq_c * q2[None, :, :], axis=2)  # [CHUNK, HKV]
    # 转置写 s2[a, h, c]（下游 [n, Hkv, Tc] 布局不变）
    tl.store(s2_ptr + a * HKV * TC + offs_h[None, :] * TC + offs_c[:, None],
             s, mask=cm[:, None])


def tli_l2_score_batched(
    q2: torch.Tensor,     # [n, Hkv, nd2] fp32 连续（G-sum 后）
    kq_q: torch.Tensor,   # [R, S_cap, Hkv, nd2] uint8（pool 常驻）
    kq_sc: torch.Tensor,  # [R, S_cap, Hkv] fp32
    kq_mn: torch.Tensor,  # [R, S_cap, Hkv] fp32
    rows: torch.Tensor,   # [n] pool 行号
    tok_c: torch.Tensor,  # [n, Tc] int64 已 clamp 的候选位置（连续）
    chunk: int = 128,
) -> torch.Tensor:
    """返回 s2 [n, Hkv, Tc] fp32 精筛分数（哨兵位置为垃圾分数——下游
    valid & causal 掩掉，与 eager 相同语义；形状静态可进 CUDA graph）。
    CHUNK=128 实测最优（0.35ms vs 512 的 0.49ms）：更小 tile 降寄存器
    压力提占用率。"""
    n, Hkv, nd2 = q2.shape
    Tc = tok_c.shape[1]
    s2 = torch.empty(n, Hkv, Tc, dtype=torch.float32, device=q2.device)
    grid = (n, triton.cdiv(Tc, chunk))
    _tli_l2_score_batched_kernel[grid](
        q2, kq_q, kq_sc, kq_mn, rows, tok_c, s2,
        HKV=Hkv, ND2=nd2, TC=Tc, S_CAP=kq_q.shape[1],
        CHUNK=chunk, num_warps=8,
    )
    return s2


# ---------------- M8-KernelD：批量 L1 fused gather+GEMV（P1+P2）----------------
# eager 的 P1（kmin/kmax 行 gather 各 67MB clone）+ P2（einsum 内部 permute 拷贝
# + gemv）合计 ~0.6ms@bs32/131K。本 kernel 经 rows 行间接寻址直读 pool，省掉
# 全部中间物化（读 128MB 写 0.5MB）。sc1 直接带 valid_blk 的 -inf（垃圾块
# 剔除语义不变：topk 落 -inf lane → 下游 sel_src gather 剔除）。
# 归约分组与 eager einsum 相同（先 G-sum 后点积），数值 1e-7 级（tie 翻转
# 概率可忽略）。skip_far 层不适用（near_keep 掩码语义走 eager）。
@triton.jit
def _tli_l1_score_batched_kernel(
    q_ptr, idx1_ptr, kmin_ptr, kmax_ptr, rows_ptr, nblk_ptr, t_ptr, sc1_ptr,
    H, D, BS,
    HKV: tl.constexpr, G: tl.constexpr, DP: tl.constexpr,
    NBLK, MBLK: tl.constexpr,
):
    ah = tl.program_id(0)  # a*HKV + h
    j = tl.program_id(1)
    a = ah // HKV
    h = ah % HKV
    offs_d = tl.arange(0, DP)
    d_idx = tl.load(idx1_ptr + offs_d)  # int64 子空间维度索引
    # group-sum q（先 G-sum 后点积，与 eager einsum 同分组）
    q_pos = tl.zeros([DP], dtype=tl.float32)
    q_neg = tl.zeros([DP], dtype=tl.float32)
    for g in range(G):
        qv = tl.load(q_ptr + (a * H + h * G + g) * D + d_idx).to(tl.float32)
        q_pos += tl.maximum(qv, 0.0)
        q_neg += tl.minimum(qv, 0.0)
    row = tl.load(rows_ptr + a).to(tl.int64)
    offs_m = j * MBLK + tl.arange(0, MBLK)
    mm = offs_m < NBLK
    # kmin/kmax [R, NBLK, Hkv, d'] 直读（行间接寻址，无中间物化）
    addr = (row * NBLK + offs_m[:, None]).to(tl.int64) * (HKV * DP) + h * DP + offs_d[None, :]
    kmin = tl.load(kmin_ptr + addr, mask=mm[:, None], other=0.0).to(tl.float32)
    kmax = tl.load(kmax_ptr + addr, mask=mm[:, None], other=0.0).to(tl.float32)
    sc = tl.sum(q_pos[None, :] * kmax, axis=1) + tl.sum(q_neg[None, :] * kmin, axis=1)
    # valid_blk = 块首 < nblk & 块尾 ≤ t（垃圾块 -inf，同 eager masked_fill）
    nblk_a = tl.load(nblk_ptr + a).to(tl.int32)
    t_a = tl.load(t_ptr + a).to(tl.int32)
    valid = mm & (offs_m < nblk_a) & ((offs_m + 1) * BS - 1 <= t_a)
    sc = tl.where(valid, sc, float("-inf"))
    tl.store(sc1_ptr + ah.to(tl.int64) * NBLK + offs_m, sc, mask=mm)


def tli_l1_score_batched(
    q: torch.Tensor,     # [n, H, D] fp32 连续
    idx1: torch.Tensor,  # [d'] int64 子空间维度索引（device）
    kmin_pool: torch.Tensor,  # [R, NBLK, Hkv, d'] fp32（pool 常驻）
    kmax_pool: torch.Tensor,  # [R, NBLK, Hkv, d']
    rows: torch.Tensor,  # [n] pool 行号
    nblk_t: torch.Tensor,  # [n] int64 每行块数上界（含非对齐尾块）
    t_t: torch.Tensor,   # [n] int64 每行 t = S-1
    block_size: int,
    mblk: int = 256,
) -> torch.Tensor:
    """返回 sc1 [n, Hkv, NBLK] fp32 块上界分数（垃圾块已 -inf；形状静态）。"""
    n, H, D = q.shape
    HKV = kmin_pool.shape[2]
    DP = kmin_pool.shape[-1]
    NBLK = kmin_pool.shape[1]
    G = H // HKV
    sc1 = torch.empty(n, HKV, NBLK, dtype=torch.float32, device=q.device)
    grid = (n * HKV, triton.cdiv(NBLK, mblk))
    _tli_l1_score_batched_kernel[grid](
        q, idx1, kmin_pool, kmax_pool,
        rows.to(torch.long).contiguous(),
        nblk_t.to(torch.long).contiguous(), t_t.to(torch.long).contiguous(),
        sc1, H, D, block_size,
        HKV=HKV, G=G, DP=DP, NBLK=NBLK, MBLK=mblk, num_warps=4,
    )
    return sc1


# ---------------- M8-KernelC：双池直写（P6+P7 融合进 KernelA）----------------
# P6（masked_fill 链 0.41ms）+ P7 的 far_sc/near_sc 物化消除：打分 kernel 直接
# 按 far/near 池边界写两张 -inf 掩码后的分数表，下游 topk 无需再物化。
# 池边界即因果边界（FAR_HI ≤ t+1-near_len、SW_LO ≤ t）——tok 数组取值仅为
# [0,S_t) 实位置或哨兵 S_cap（clamp 后 S_cap-1），两者均天然落两池之外，
# 无需显式 valid/causal 掩码（与 eager in_far = 池界 & valid & causal 逐位
# 等价：池界 ⇒ tok < S_t ⇒ valid & causal）。skip_far 层不适用（单池口径）。
@triton.jit
def _tli_l2_score_batched_dual_kernel(
    q2_ptr, kq_ptr, sc_ptr, mn_ptr, rows_ptr, tok_ptr,
    far_ptr, near_ptr, far_hi_ptr, sw_lo_ptr,
    HKV: tl.constexpr, ND2: tl.constexpr,
    TC: tl.constexpr,
    S_CAP, FAR_LO, CHUNK: tl.constexpr,
):
    a = tl.program_id(0)
    c0 = tl.program_id(1) * CHUNK
    offs_c = c0 + tl.arange(0, CHUNK)
    cm = offs_c < TC
    row = tl.load(rows_ptr + a).to(tl.int64)
    pos = tl.load(tok_ptr + a * TC + offs_c, mask=cm, other=0).to(tl.int64)
    offs_h = tl.arange(0, HKV)
    offs_d = tl.arange(0, ND2)
    base = row * (S_CAP * HKV * ND2) + pos * (HKV * ND2)  # [CHUNK] int64
    addr = base[:, None, None] + (offs_h[None, :, None] * ND2 + offs_d[None, None, :])
    kq = tl.load(kq_ptr + addr, mask=cm[:, None, None], other=0).to(tl.float32)
    base_s = row * (S_CAP * HKV) + pos * HKV  # [CHUNK]
    addr_s = base_s[:, None] + offs_h[None, :]
    sc = tl.load(sc_ptr + addr_s, mask=cm[:, None], other=0.0)
    mn = tl.load(mn_ptr + addr_s, mask=cm[:, None], other=0.0)
    kq_c = kq * sc[:, :, None] + mn[:, :, None]
    q2 = tl.load(q2_ptr + a * HKV * ND2 + offs_h[:, None] * ND2 + offs_d[None, :])
    s = tl.sum(kq_c * q2[None, :, :], axis=2)  # [CHUNK, HKV]
    # far/near 池归属（per-row 边界；哨兵与池外位置两表均落 -inf）
    far_hi = tl.load(far_hi_ptr + a)  # int64
    sw_lo = tl.load(sw_lo_ptr + a)
    in_far = (pos >= FAR_LO) & (pos < far_hi)  # [CHUNK]
    in_near = (pos < sw_lo) & (~in_far)
    far_v = tl.where(in_far[:, None], s, float("-inf"))  # [CHUNK, HKV]
    near_v = tl.where(in_near[:, None], s, float("-inf"))
    o_off = a * HKV * TC + offs_h[None, :] * TC + offs_c[:, None]
    tl.store(far_ptr + o_off, far_v, mask=cm[:, None])
    tl.store(near_ptr + o_off, near_v, mask=cm[:, None])


def tli_l2_score_batched_dual(
    q2: torch.Tensor,     # [n, Hkv, nd2] fp32 连续（G-sum 后）
    kq_q: torch.Tensor,   # [R, S_cap, Hkv, nd2] uint8（pool 常驻）
    kq_sc: torch.Tensor,  # [R, S_cap, Hkv] fp32
    kq_mn: torch.Tensor,  # [R, S_cap, Hkv] fp32
    rows: torch.Tensor,   # [n] pool 行号
    tok_c: torch.Tensor,  # [n, Tc] int64 已 clamp 的候选位置（连续）
    far_lo: int,          # sink 区上界（host 常量 = sink_blocks*block_size）
    far_hi: torch.Tensor, # [n] int64 far 池上界（per-row device）
    sw_lo: torch.Tensor,  # [n] int64 滑窗下界（per-row device）
    chunk: int = 128,
):
    """返回 (far_sc, near_sc) [n, Hkv, Tc] fp32（池外 -inf，与 eager 的
    s2.masked_fill(~in_far/~in_near) 逐位一致；topk 输入等价 → 输出 torch.equal）。
    CHUNK=128 实测最优（0.43ms vs 512 的 1.06ms）：双输出 tile 使寄存器
    压力比单输出版更早触顶，512 即溢出到 local memory。"""
    n, Hkv, nd2 = q2.shape
    Tc = tok_c.shape[1]
    far_sc = torch.empty(n, Hkv, Tc, dtype=torch.float32, device=q2.device)
    near_sc = torch.empty(n, Hkv, Tc, dtype=torch.float32, device=q2.device)
    grid = (n, triton.cdiv(Tc, chunk))
    _tli_l2_score_batched_dual_kernel[grid](
        q2, kq_q, kq_sc, kq_mn, rows, tok_c,
        far_sc, near_sc, far_hi.to(torch.long).contiguous(),
        sw_lo.to(torch.long).contiguous(),
        HKV=Hkv, ND2=nd2, TC=Tc, S_CAP=kq_q.shape[1], FAR_LO=far_lo,
        CHUNK=chunk, num_warps=8,
    )
    return far_sc, near_sc


# ---------------- M8：候选压实块展开 kernel（P4）----------------
# eager：repeat_interleave [n,S_cap] + where 物化 [n,S_cap] int64 + topk-min
# (k=Tc) 全排序。kernel：cumsum(onehot) 前缀定槽位 + grid (n, NBLK/BPC) 逐块
# 展开 BS 个 token（块升序天然保持位置升序），非因果尾 token 写哨兵，
# 尾部 pad 预填充。语义差异：eager 哨兵全在尾部；kernel 哨兵可在中段（非因果
# 尾块内）——下游 valid = tok < S_cap 掩掉与位置无关，有效集逐行一致（对拍 PASS）。
@triton.jit
def _tli_compact_kernel(
    onehot_ptr, prefix_ptr, tok_ptr, s_t_ptr,
    NBLK: tl.constexpr, BS: tl.constexpr, TC: tl.constexpr,
    BPC: tl.constexpr, SENT,
):
    a = tl.program_id(0)
    j = tl.program_id(1)
    offs_b = j * BPC + tl.arange(0, BPC)
    bm = offs_b < NBLK
    oh = tl.load(onehot_ptr + a * NBLK + offs_b, mask=bm, other=0)
    pref = tl.load(prefix_ptr + a * NBLK + offs_b, mask=bm, other=0)  # 含 cumsum
    s_t = tl.load(s_t_ptr + a)  # int64
    # 选中块首 token 输出槽 = (pref-1)*BS（union 上界保证槽位不越界；
    # store 掩码再带防御性上界）
    slot = (pref - 1) * BS
    offs_i = tl.arange(0, BS)
    p = offs_b[:, None] * BS + offs_i[None, :]  # [BPC, BS]
    v = tl.where(p < s_t, p, SENT)  # 非因果尾 → 哨兵
    st_mask = (oh > 0)[:, None] & bm[:, None] & ((slot[:, None] + BS) <= TC)
    tl.store(tok_ptr + a * TC + slot[:, None] + offs_i[None, :], v, mask=st_mask)


def tli_compact(
    onehot: torch.Tensor,   # [n, NBLK] bool 连续（跨 head 并集 + 滑窗强制）
    S_t: torch.Tensor,      # [n] int64 每行有效长度
    block_size: int,
    S_cap: int,             # 哨兵值（pool 容量）
    Tc: int,
) -> torch.Tensor:
    """返回 tok [n, Tc] int64 候选位置（哨兵 S_cap，有效集与 eager topk-min
    逐行一致；块升序）。CUDA graph 兼容（形状静态）。"""
    n, NBLK = onehot.shape
    dev = onehot.device
    tok = torch.full((n, Tc), S_cap, dtype=torch.int64, device=dev)
    prefix = torch.cumsum(onehot.to(torch.int32), dim=1)
    BPC = 32
    grid = (n, triton.cdiv(NBLK, BPC))
    _tli_compact_kernel[grid](
        onehot.view(torch.uint8), prefix, tok, S_t.to(torch.long).contiguous(),
        NBLK=NBLK, BS=block_size, TC=Tc, BPC=BPC, SENT=S_cap,
        num_warps=4,
    )
    return tok
