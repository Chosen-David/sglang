#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E113：sim_greedy 增量贪心聚类的 Triton kernel（方案1：单 kernel 顺序化）。

== 地位（E113b e2e 集成）==
本文件是 **生产路径 kernel 本体**（tli_indexer.py `_greedy_cluster_pass`
调度器的 CUDA 路径，env 开关 SGLANG_TLI_GREEDY_KERNEL，默认 1）。
原型/单测/bench 脚本在 two-level-attention/exp/trace/e113_*.py（exp/trace
的 e113_greedy_triton.py 是本文件的转发入口，单一事实源在这里——远程部署
只需 rsync sparse_attn 目录即可携带 kernel）。
设计文档：research/docs/e113_method_kernel_design.md（实证篇）+
research/docs/sim_greedy_kernel_design.md（设计篇）。

== 背景 ==
E110 落地的 `_greedy_cluster_pass`（two-level-attention sparse_attn/indexer/
tli_indexer.py）是逐 token 的 Python 循环：每步 bmm/argmax/gather/index_add
约 10 次 kernel launch，E108 probe 实测 T≈16K 时 ~5.8 s/层（sim=0.9 时 27 s），
是远程 sim_greedy e2e 臂 13 h/臂 的直接根因。

== 本文件语义目标（对齐铁律）==
与 `_greedy_cluster_pass` 同签名同语义：
    greedy_pass_triton(x, sim, sums, cnt, sq, k_live)
        -> (sums, cnt, sq, k_live, assign)
- 贪心时序语义逐 token 精确保留：token i 与「i 时刻的簇心」（running mean）
  比较余弦，argmax + sim 阈值 → 归并或新建；
- 增量续跑（传入非零 k_live 的既有状态）≡ 全量重放（E110 T3 口径）；
- fp 注记：dot 用 tl.sum 逐 tile 归约，与 torch bmm 的归约顺序可能在
  1e-7 量级上不同 → 近并列边界可能翻转（E108 probe 已有先例：sim≥0.85
  逐步精确 norm vs 增量 sq 维护 0.03% 边界差）。单测报告逐位一致率。

== kernel 设计（grid=(H,) 单程序一头，token 循环在 kernel 内）==
- 每 program 独占一个 kv-head 的全部簇状态（sums/cnt/sq），无跨 program
  竞争，无需原子操作；
- 每 token：K 分 tile 扫描 sums 行（HBM/L2 读，tile [BT,DD]），cos =
  dot/(norm·xn+EPS)，tile 内 argmax（first-occurrence tie-break）+
  跨 tile 严格 > 比较（早 tile 优先 = 全局 first-occurrence，对齐 torch
  argmax 语义）；
- 决策与更新 branchless：dst = upd ? best_k : k_live；sums[dst] = upd ?
  row+x : x；cnt/sq 同式（新建簇槽必为零槽，与参考实现「零权重
  index_add 落零槽/活簇」的语义一致）；
- 容量由 wrapper 保证 K ≥ max(k_live)+T（最坏每 token 新建一簇），
  与参考实现的 F.pad 扩容点一致；
- 状态体量小（[H,K,dd] fp32，T=16K 时 ~16MB×H/8…见设计文档 roofline），
  L2 常驻，K 扫描走 L2 带宽。

用法（对拍单测见 exp/trace/test_e113_greedy_triton.py）：
    from sparse_attn.indexer.greedy_triton import greedy_pass_triton, greedy_build_triton
（兼容旧路径：exp/trace/e113_greedy_triton.py 转发导出本模块。）
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ---------------------------------------------------------------- kernel 本体
@triton.jit
def _greedy_pass_kernel(
    x_ptr,          # [T, H, DD] fp32 连续（与参考实现 x [T,H,dd] 同布局）
    sums_ptr,       # [H, K, DD] fp32 连续（原地更新）
    cnt_ptr,        # [H, K]     fp32 连续（原地更新）
    sq_ptr,         # [H, K]     fp32 连续（原地更新，||sums||^2 增量维护）
    klive_ptr,      # [H] int64：入=初始活簇数，出=终态活簇数
    assign_ptr,     # [H, T] int64 出口
    H: tl.constexpr, DD: tl.constexpr, BT: tl.constexpr,
    K,              # 簇容量（runtime，≥ max(初始 k_live)+T 由 wrapper 保证）
    T,              # 本趟 token 数（runtime）
    SIM,
    EPS: tl.constexpr,
):
    h = tl.program_id(0)
    k_live = tl.load(klive_ptr + h).to(tl.int32)
    offs_d = tl.arange(0, DD)
    for i in range(T):
        x = tl.load(x_ptr + (i * H + h) * DD + offs_d).to(tl.float32)
        xn = tl.sqrt(tl.sum(x * x))
        # ---- K 分 tile 扫描：cos = dot/(norm·xn+EPS)，死簇 -inf ----
        best_cos = float("-inf")
        best_k = tl.zeros((), dtype=tl.int32)
        best_dot = tl.zeros((), dtype=tl.float32)
        for k0 in range(0, k_live, BT):
            offs_k = k0 + tl.arange(0, BT)
            m = offs_k < k_live
            sums_t = tl.load(
                sums_ptr + (h * K + offs_k[:, None]) * DD + offs_d[None, :],
                mask=m[:, None], other=0.0,
            )
            dot_t = tl.sum(sums_t * x[None, :], axis=1)
            sq_t = tl.load(sq_ptr + h * K + offs_k, mask=m, other=0.0)
            norm_t = tl.sqrt(tl.maximum(sq_t, 0.0))
            cos_t = dot_t / (norm_t * xn + EPS)
            cos_t = tl.where(m, cos_t, float("-inf"))
            am = tl.argmax(cos_t, axis=0)          # tile 内 first-occurrence
            tmax = tl.max(cos_t, axis=0)
            lane = tl.arange(0, BT)
            dot_am = tl.sum(tl.where(lane == am, dot_t, 0.0))
            take = tmax > best_cos                 # 严格 >：早 tile 赢并列
            best_cos = tl.where(take, tmax, best_cos)
            best_k = tl.where(take, (k0 + am).to(tl.int32), best_k)
            best_dot = tl.where(take, dot_am, best_dot)
        # ---- 决策 + 更新（branchless；新建簇槽必为零槽）----
        upd = best_cos >= SIM
        dst = tl.where(upd, best_k, k_live)
        row = tl.load(sums_ptr + (h * K + dst) * DD + offs_d)
        tl.store(sums_ptr + (h * K + dst) * DD + offs_d,
                 tl.where(upd, row + x, x))
        c_old = tl.load(cnt_ptr + h * K + dst)
        tl.store(cnt_ptr + h * K + dst, tl.where(upd, c_old + 1.0, 1.0))
        s_old = tl.load(sq_ptr + h * K + dst)
        # 归并：sq += 2·(sums_a·x) + ||x||^2（dot_a 取扫描时的 tile 值，更新前口径）
        tl.store(sq_ptr + h * K + dst,
                 tl.where(upd, s_old + 2.0 * best_dot + xn * xn, xn * xn))
        tl.store(assign_ptr + h * T + i, dst.to(tl.int64))
        k_live = tl.where(upd, k_live, k_live + 1)
    tl.store(klive_ptr + h, k_live.to(tl.int64))


# ---------------------------------------------------------------- wrapper（对齐参考签名）
def greedy_pass_triton(x, sim, sums, cnt, sq, k_live, BT=512, num_warps=4):
    # BT/num_warps 默认值 = E113 GPU1 干净口径扫描最优（T=16384，struct K≈2K）：
    #   BT=128/nw=8: 41.0μs/token → BT=512/nw=4: 9.8μs/token（4.2×）；
    #   BT=1024/nw=4 与 512 持平（9.6 vs 9.8），K=16K 单例臂 512 略优（41 vs 43）；
    #   BT 不影响语义：dot 归约沿 dd 维（行内 32 元素，与 BT 无关），
    #   跨 tile 严格 > 的 first-occurrence tie-break 与 BT 无关 → 改 BT 逐位不变。
    # num_warps=4 时单 tile [512,32] = 64KB fp32，寄存器/L1 平衡点。
    """与 TLIIndexer._greedy_cluster_pass 同签名同语义的 Triton 版。

    x: [T, H, dd]（按序）；sums [H,K,dd] / cnt [H,K] / sq [H,K] / k_live [H]
    为既有簇状态（全零 = 冷启动；非零 = 续跑，与全量重放逐位一致——
    贪心决策只依赖先验状态）。返回 (sums, cnt, sq, k_live, assign[H,T])；
    容量不足时返回扩容新张量（与参考实现的 F.pad 扩容同语义）。
    注意：与参考实现一致，未扩容时 sums/cnt/sq 被原地更新。
    """
    T, H, dd = x.shape
    assert x.is_cuda, "Triton 路径需要 CUDA"
    x = x.float().contiguous()
    need = int(k_live.max().item()) + T
    K = sums.shape[1]
    if need > K:  # 扩容（新簇上界 = T，与参考实现同判据）
        pad = need - K
        sums = F.pad(sums, (0, 0, 0, pad))
        cnt = F.pad(cnt, (0, pad))
        sq = F.pad(sq, (0, pad))
        K = sums.shape[1]
    sums = sums.contiguous()
    cnt = cnt.contiguous()
    sq = sq.contiguous()
    kl = k_live.to(torch.int64).contiguous().clone()
    assign = torch.empty(H, T, dtype=torch.int64, device=x.device)
    BT = min(BT, triton.next_power_of_2(max(K, 16)))
    _greedy_pass_kernel[(H,)](
        x, sums, cnt, sq, kl, assign,
        H=H, DD=dd, BT=BT, K=K, T=T, SIM=float(sim), EPS=1e-9,
        num_warps=num_warps,
    )
    return sums, cnt, sq, kl, assign


def greedy_build_triton(x, sim, chunk=16384, BT=512, num_warps=4):
    """冷启动全序列构建（prefill 路径）：分段喂 greedy_pass_triton。

    分段不改变语义（贪心时序跨段精确延续，同 _update_far_greedy 的
    增量续跑论证）；chunk 只是单 launch 的 token 上限（防超长序列单
    kernel 占用与编译期 K 循环过长）。E113 实测 chunk=16384 比 8192
    快 1.75×（9.8 vs 17.1μs/token，launch 间隙效应），默认 16384。
    返回 (sums, cnt, sq, k_live, assign)。
    """
    T, H, dd = x.shape
    K = T
    sums = x.new_zeros(H, K, dd)
    cnt = x.new_zeros(H, K)
    sq = x.new_zeros(H, K)
    k_live = torch.zeros(H, dtype=torch.int64, device=x.device)
    assign = torch.empty(H, T, dtype=torch.int64, device=x.device)
    for c0 in range(0, T, chunk):
        seg = x[c0:c0 + chunk]
        sums, cnt, sq, k_live, a = greedy_pass_triton(
            seg, sim, sums, cnt, sq, k_live, BT=BT, num_warps=num_warps)
        assign[:, c0:c0 + seg.shape[0]] = a
    return sums, cnt, sq, k_live, assign


__all__ = ["greedy_pass_triton", "greedy_build_triton", "_greedy_pass_kernel"]
