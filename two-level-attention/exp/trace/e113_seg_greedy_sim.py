#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E113：SEG-GREEDY（段式推测 + Cauchy-Schwarz 界验证）仿真——方案2 的
可证明跳过率与精确性实证（PyTorch 级实现，kernel 化前的设计验证件）。

== 核心思想（设计文档 §2.3 方案2）==
贪心是时序串行，但段内可以用「冻结状态的批量 GEMM」推测每个 token 的
决策，再用**区间界**证明哪些推测决策与精确时序语义必然一致（免重算）：

  设段首快照 s^0_k（簇和向量）、D_j(k) = x_j·s^0_k（批量 GEMM）、
  d_k = s_k − s^0_k（段内漂移，join 时增量维护）、δ_k = ||d_k||：
    |dot_j(k) − D_j(k)| = |x_j·d_k| ≤ ||x_j||·δ_k      （Cauchy–Schwarz）
    ||s_k|| ∈ [n0_k − δ_k, n0_k + δ_k]                  （三角不等式，n0=||s^0||）
  ⇒ cos_j(k) 的上下界：
    UB_k = (D + xn·δ) / ((n0−δ)·xn + EPS)，LB_k = (D − xn·δ) / ((n0+δ)·xn + EPS)
  （n0−δ ≤ 0 时 UB=+inf 保守处理）

  正确性判据（精确，非近似）：
    a = argmax_k UB_k；若 LB_a > max_{k≠a} UB_k（且 > 段内新建簇的精确 cos），
    则 a 是真 argmax（cos_a ≥ LB_a > UB_k ≥ cos_k）；
    阈值决策用 a 的精确 cos（单行点积）→ join/create 决策精确。
    若不可证明：候选集 C = {k: UB_k ≥ LB_a} ∪ 新建簇——真 argmax 必在 C 内
    （C 外的 k 满足 cos_k ≤ UB_k < LB_a ≤ cos_a），对 C 精确重算即得真值。
  ⇒ 整个算法与逐 token 精确贪心**决策级等价**（bound 只用于跳过重算，
  不可证明的 token 走精确路径），不是近似变体。

  段内新建簇（k ≥ K0）无快照 → 一律精确处理（对其当前 sums 行直接点积）。
  新建簇数超 NEW_CAP 时段提前重启（重新 GEMM 快照）——生产设计的段重启。

== 输出 ==
exp/trace/results/e113_seg_greedy_sim.json：
  - 每配置：skip_rate（可证明跳过比例）、avg/mean 候选集规模、段重启数、
    assign 与参考逐位不一致数（期望 0，fp 边界另报）、双方 wall time。
  - 数据：真实 trace（/tmp/trace/qwen3-8b，tail32 子空间 mid 区）+ 合成结构数据。

用法：CUDA_VISIBLE_DEVICES=0 python3 e113_seg_greedy_sim.py [--T 8192]
"""
import argparse
import importlib
import json
import os
import sys
import time
import types

sys.dont_write_bytecode = True   # 主树只读 import

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from e113_greedy_triton import greedy_build_triton   # noqa: E402  (Triton 参考也一并计时)

MAIN_TREE = "/home/wangyuanshuo02/sglang/two-level-attention"
TRACE = "/tmp/trace/qwen3-8b"
EPS = 1e-9
BOUND_INFLATE = 1e-6   # 界的 fp 余量（保守方向放大界，防浮点误差导致误判「可证明」）
SINK, SWA = 128, 1024
D2I = list(range(48, 64)) + list(range(112, 128))   # tail32（e64a 口径）


def load_sparse_attn(name, root):
    mod = types.ModuleType(name)
    mod.__path__ = [os.path.join(root, "sparse_attn")]
    sys.modules[name] = mod
    return importlib.import_module(f"{name}.indexer")


IDX = load_sparse_attn("sparse_attn_e113sim", MAIN_TREE)
greedy_ref = IDX.TLIIndexer._greedy_cluster_pass_python   # E113b：参考实现固定取 Python 路径（_greedy_cluster_pass 已是调度器）


# ---------------------------------------------------------------- SEG-GREEDY 本体
def seg_greedy(x, sim, L=512, new_cap=256, stats=None):
    """段式推测贪心（精确）。x: [T,H,dd] fp32 cuda。返回 (sums,cnt,sq,klive,assign,stats)。"""
    T, H, dd = x.shape
    K = T
    dev = x.device
    sums = x.new_zeros(H, K, dd)
    cnt = x.new_zeros(H, K)
    sq = x.new_zeros(H, K)
    klive = torch.zeros(H, dtype=torch.long, device=dev)
    assign = torch.zeros(H, T, dtype=torch.long, device=dev)
    ar_h = torch.arange(H, device=dev)
    ar_new = torch.arange(new_cap, device=dev)
    st = {"skip": 0, "exact": 0, "cand_sum": 0, "cand_max": 0, "restarts": 0,
          "tokens": T, "new_sum": 0, "gemm_flop": 0.0}
    seg = 0
    c0 = 0
    while c0 < T:
        c1 = min(c0 + L, T)
        # ---- 段首快照 ----
        K0 = klive.clone()                              # [H] 各头旧簇上界
        Kmax = int(K0.max().item())
        s0 = sums[:, :Kmax].clone() if Kmax > 0 else None
        n0 = sq[:, :Kmax].clamp(min=0).sqrt() if Kmax > 0 else None
        xseg = x[c0:c1]                                 # [L,H,dd]
        if Kmax > 0:
            D = torch.einsum("lhd,hkd->hlk", xseg, s0)  # [H,L,Kmax] 冻结 GEMM
            st["gemm_flop"] += 2.0 * H * xseg.shape[0] * Kmax * dd
        d = x.new_zeros(H, Kmax, dd) if Kmax > 0 else None
        dnorm = x.new_zeros(H, Kmax) if Kmax > 0 else None
        broke = False
        for j in range(c1 - c0):
            xj = xseg[j]                                # [H,dd]
            xn = xj.norm(dim=-1)                        # [H]
            kl = klive                                  # 当前各头活簇数
            # ---- 旧簇界 ----
            if Kmax > 0:
                Dj = D[:, j, :]                         # [H,Kmax]
                old_live = torch.arange(Kmax, device=dev)[None, :] < K0[:, None]
                ub = (Dj + xn[:, None] * dnorm) / \
                     ((n0 - dnorm).clamp(min=BOUND_INFLATE) * xn[:, None] + EPS)
                ub = ub + BOUND_INFLATE                  # fp 余量：放大上界（保守）
                ub = torch.where(old_live, ub, float("-inf"))
                lb = (Dj - xn[:, None] * dnorm) / \
                     ((n0 + dnorm) * xn[:, None] + EPS)
                lb = lb - BOUND_INFLATE                  # 缩小下界（保守）
                lb = torch.where(old_live, lb, float("-inf"))
            else:
                ub = torch.full((H, 0), float("-inf"), device=dev)
                lb = torch.full((H, 0), float("-inf"), device=dev)
            # ---- 段内新建簇：精确 cos ----
            n_new = (kl - K0).clamp(min=0)              # [H]
            new_idx = K0[:, None] + ar_new[None, :]     # [H,new_cap]
            new_valid = (ar_new[None, :] < n_new[:, None]) & (n_new[:, None] > 0)
            if new_valid.any():
                rows = sums[ar_h[:, None], new_idx.clamp(max=K - 1)]  # [H,new_cap,dd]
                dots_n = torch.einsum("hnd,hd->hn", rows, xj)
                norm_n = sq[ar_h[:, None], new_idx.clamp(max=K - 1)].clamp(min=0).sqrt()
                cos_n = dots_n / (norm_n * xn[:, None] + EPS)
                cos_n = torch.where(new_valid, cos_n, float("-inf"))
            else:
                cos_n = torch.full((H, new_cap), float("-inf"), device=dev)
            max_new = cos_n.max(dim=-1).values           # [H]（无新建簇 = -inf）
            has_new = n_new > 0
            # ---- a = argmax UB（每头独立）----
            if ub.shape[1] > 0:
                a = ub.argmax(dim=-1)                    # [H]
                ub_a = ub.gather(1, a[:, None]).squeeze(1)
                lb_a = lb.gather(1, a[:, None]).squeeze(1)
                ub_masked = ub.scatter(-1, a[:, None], float("-inf"))
                max_other = ub_masked.max(dim=-1).values
            else:
                a = torch.zeros(H, dtype=torch.long, device=dev)
                ub_a = torch.full((H,), float("-inf"), device=dev)
                lb_a = torch.full((H,), float("-inf"), device=dev)
                max_other = torch.full((H,), float("-inf"), device=dev)
            # ---- 可证明判定：LB_a > 其余 UB 且 > 新建簇精确 max ----
            provable = (lb_a > max_other) & (lb_a > max_new) & (ub_a > float("-inf"))
            # 阈值决策需要 a 的精确 cos：单行点积（当前 sums 状态）
            def exact_cos(c):
                rows_c = sums[ar_h, c]                                    # [H,dd]
                dot_c = (rows_c * xj).sum(-1)
                norm_c = sq[ar_h, c].clamp(min=0).sqrt()
                return dot_c / (norm_c * xn + EPS), dot_c
            if provable.any():
                a_p = torch.where(provable, a, torch.zeros_like(a))
                cos_a, dot_a = exact_cos(a_p)
                join_p = provable & (cos_a >= sim)
                st["skip"] += int(provable.sum())
            else:
                join_p = provable  # 全 False
            # ---- 不可证明头：候选集精确重算 ----
            need = (~provable) & ((ub_a > float("-inf")) | has_new)
            # 冷启动头（无旧簇无新簇）→ 必然新建（参考语义：全 -inf argmax=0, m=-inf<sim）
            if need.any():
                cand = (ub >= lb_a[:, None]) | provable[:, None]         # a 必在候选内
                cand = cand & old_live if Kmax > 0 else cand[:, :0]
                # 每头候选数与索引（向量化：取 top 候选数最大的头做上限会浪费，
                # 仿真直接对逐头循环——生产 kernel 用 warp 归约）
                for h in torch.nonzero(need).flatten().tolist():
                    idx_old = torch.nonzero(cand[h]).flatten()
                    idx_new = torch.nonzero(new_valid[h] & (cos_n[h] > float("-inf"))).flatten()
                    rows_o = sums[h, idx_old] if idx_old.numel() else x.new_zeros(0, dd)
                    rows_v = sums[h, new_idx[h, idx_new]] if idx_new.numel() else x.new_zeros(0, dd)
                    rows_all = torch.cat([rows_o, rows_v], 0)             # [C,dd]
                    dots = rows_all @ xj[h]
                    norms = torch.cat([
                        sq[h, idx_old].clamp(min=0).sqrt() if idx_old.numel() else x.new_zeros(0),
                        sq[h, new_idx[h, idx_new]].clamp(min=0).sqrt() if idx_new.numel() else x.new_zeros(0),
                    ], 0)
                    cos_all = dots / (norms * xn[h] + EPS)
                    am = int(cos_all.argmax()) if cos_all.numel() else -1
                    st["exact"] += 1
                    st["cand_sum"] += int(cos_all.numel())
                    st["cand_max"] = max(st["cand_max"], int(cos_all.numel()))
                    if am >= 0:
                        a[h] = (idx_old[am] if am < idx_old.numel()
                                else new_idx[h, idx_new[am - idx_old.numel()]]).item()
                # 需要头逐头重算 a 后统一精确 cos（向量化一遍）
            # ---- 统一决策 + 更新（join 头用精确 cos）----
            cos_f, dot_f = exact_cos(torch.where(need | provable, a, torch.zeros_like(a)))
            join = (provable & join_p) | (need & (cos_f >= sim))
            # 不可证明且无候选可用（cold）：join=False → 新建 ✓
            dst = torch.where(join, a, kl)
            # 更新 sums/cnt/sq（join: 增量；新建: 写入）
            upd_row = sums[ar_h, dst]
            sums[ar_h, dst] = torch.where(join[:, None], upd_row + xj, xj)
            cnt[ar_h, dst] = torch.where(join, cnt[ar_h, dst] + 1.0,
                                         torch.ones_like(cnt[ar_h, dst]))
            sq[ar_h, dst] = torch.where(
                join, sq[ar_h, dst] + 2.0 * dot_f + xn * xn, xn * xn)
            assign[:, c0 + j] = dst
            klive = kl + (~join).long()
            # d/dnorm 增量维护（join 到旧簇的头）
            if Kmax > 0:
                is_old = join & (dst < K0)
                if is_old.any():
                    dst_o = torch.where(is_old, dst, torch.zeros_like(dst))
                    d[ar_h, dst_o] = d[ar_h, dst_o] + xj
                    dnorm[ar_h, dst_o] = d[ar_h, dst_o].norm(dim=-1)
            # 新建簇超上限 → 段提前重启
            if int((klive - K0).max().item()) >= new_cap:
                st["restarts"] += 1
                st["new_sum"] += int((klive - K0).sum().item())
                broke = True
                c0 = c0 + j + 1
                break
        if not broke:
            st["new_sum"] += int((klive - K0).sum().item())
            c0 = c1
        seg += 1
    st["segments"] = seg
    if stats is not None:
        stats.update(st)
    return sums, cnt, sq, klive, assign, st


# ---------------------------------------------------------------- 数据源
def load_trace_x(sample, layer, T_cap, device="cuda"):
    d = torch.load(f"{TRACE}/{sample}/layer{layer:02d}.pt", map_location="cpu",
                   weights_only=False)
    k = d["k"].to(device).float()
    qpos = d["qpos"]
    mid_hi = int(qpos[-1]) + 1 - SWA
    idx = torch.tensor(D2I, device=device)
    x = k[SINK:mid_hi][..., idx]      # [T,Hkv,32] tail32 mid 区（贪心聚类口径）
    if T_cap and x.shape[0] > T_cap:
        x = x[:T_cap]
    return x, d["k"].shape[1], int(k.shape[1])


def gen_structured(T, H, dd, n_base, noise, seed, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    base = torch.randn(n_base, H, dd, generator=g, device=device)
    idx = torch.randint(0, n_base, (T,), generator=g, device=device)
    return (base[idx] + noise * torch.randn(T, H, dd, generator=g, device=device)).contiguous()


# ---------------------------------------------------------------- 主流程
def run_case(name, x, sim, L=512):
    T, H, dd = x.shape
    print(f"\n---- {name}: T={T} H={H} dd={dd} sim={sim} L={L} ----", flush=True)
    # 计时口径：三处均显式 synchronize——Triton launch 异步，不同步只测到 launch 开销
    torch.cuda.synchronize()
    t0 = time.time()
    r = greedy_ref(x, sim, x.new_zeros(H, T, dd), x.new_zeros(H, T),
                   x.new_zeros(H, T), torch.zeros(H, dtype=torch.long, device=x.device))
    torch.cuda.synchronize()
    t_ref = time.time() - t0
    t0 = time.time()
    s, c, q, kl, a, st = seg_greedy(x, sim, L=L)
    t_seg = time.time() - t0
    torch.cuda.synchronize()
    t0 = time.time()
    tk = greedy_build_triton(x, sim, chunk=8192)
    torch.cuda.synchronize()
    t_tri = time.time() - t0
    # Triton 冷启动含 JIT 编译：再热跑一次取稳定值（编译缓存后）
    torch.cuda.synchronize()
    t0 = time.time()
    tk2 = greedy_build_triton(x, sim, chunk=8192)
    torch.cuda.synchronize()
    t_tri_hot = time.time() - t0
    assert int((tk[4] != tk2[4]).sum()) == 0, "Triton 冷/热两次 assign 不一致（非确定性）"
    mism = int((a != r[4]).sum())
    live = int(r[3].max().item())
    cen_ok = torch.allclose(s[:, :live], r[0][:, :live], atol=1e-5)
    rec = {
        "T": T, "H": H, "dd": dd, "sim": sim, "L": L,
        "clusters_max_head": live, "C_over_N": round(live / T, 4),
        "assign_mismatch": mism, "centroids_allclose": bool(cen_ok),
        # 口径注记：skip/exact 均为 head-token 计数（×H 后除总位）
        "skip_rate_headtok": round(st["skip"] / (T * H), 4),
        "exact_rate_headtok": round(st["exact"] / (T * H), 4),
        "cand_mean": round(st["cand_sum"] / max(st["exact"], 1), 1),
        "cand_over_clusters": round(
            (st["cand_sum"] / max(st["exact"], 1)) / max(live, 1), 3),
        "cand_max": st["cand_max"],
        "restarts": st["restarts"], "segments": st["segments"],
        "gemm_GFLOP": round(st["gemm_flop"] / 1e9, 2),
        "wall_s_ref_python": round(t_ref, 2),
        "wall_s_seg_pytorch": round(t_seg, 2),
        "wall_s_triton_kernel_cold": round(t_tri, 3),
        "wall_s_triton_kernel_hot": round(t_tri_hot, 4),
    }
    print(json.dumps(rec, ensure_ascii=False, indent=1), flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=8192)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "e113_seg_greedy_sim.json"))
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    assert torch.cuda.is_available(), "需 GPU"
    results = {"meta": {
        "probe": "E113 SEG-GREEDY 段式推测+界验证：可证明跳过率与精确性实证",
        "bound": "Cauchy-Schwarz |Δdot|≤||x||·||d|| + 三角不等界 ||s||∈[n0−δ,n0+δ]",
        "exactness": "bound 仅用于跳过重算；不可证明 token 走候选集精确重算 → 决策级等价",
        "inflate": BOUND_INFLATE, "new_cap": 256, "L": 512,
        "trace": TRACE, "started": time.strftime("%F %T"),
    }, "cases": []}

    # 真实 trace：hotpotqa（S=16957）layer18，tail32 mid 区
    x, S, Hkv = load_trace_x("lb_hotpotqa_0", 18, args.T)
    for sim in (0.80, 0.90):
        results["cases"].append(run_case(f"trace_hotpotqa_L18 (S={S})", x, sim))
    # 合成结构数据（中簇密度对照）
    xs = gen_structured(args.T, 8, 32, n_base=max(args.T // 8, 1), noise=0.10, seed=7)
    for sim in (0.80, 0.90):
        results["cases"].append(run_case("synthetic_mixture", xs, sim))

    results["finished"] = time.strftime("%F %T")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=1, ensure_ascii=False)
    print("\nsaved ->", args.out)


if __name__ == "__main__":
    main()
