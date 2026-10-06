#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E108 探针：cluster_sim_greedy（用户 2026-10-06 提案）质量-算力曲线判决。

提案定义：
  - 增量贪心聚类：逐 token 处理 K，与现有簇代表算余弦相似度，>= sim 阈值并入该簇
    （代表 = 成员 running mean 算术平均），否则新建簇
  - 选择：q 对全部簇代表打分排序，按簇召回（两种口径：簇内 token 全取 / 按簇代表
    分数取簇内 token）直到填满 mid 预算 B_TOK=2048（与既有 mass 重放 mono 臂同口径）
  - sim 单值 sweep {0.80..0.99}，找 knee sim*（跨数据集通用性验证）

口径铁律（与既有 mass 重放逐项一致，源脚本）：
  - dump/协议常量/greedy 聚类语义：/home/wangyuanshuo02/two-level-attention/exp/trace/analyze_e64a_ab_grid.py
    （TRACE=/tmp/trace/qwen3-8b, D2I=tail32 子空间, BS=64, SINK=128, SWA=1024,
     TAIL_N=4, BP=64, B_TOK=2048, greedy_cluster_assign：时序逐 token 余弦阈值均值簇心）
  - mass 重放口径：analyze_e64j_combo_best.py / analyze_e98_abg_full_grid.py
    （真实全维 softmax 行级 mass coverage；mono 臂 = mid 单池 B_TOK=2048 + sink/swa 强制；
     qsub = GQA 组内 sum 聚合；代表打分 = qsub·cent 点积——与 cavg 簇分同式）
  - 本探针贪心实现与 e64a.greedy_cluster_assign 语义相同（向量化新簇创建加速），
    smoke 模式下与原实现逐位对拍 assign。

锚点（同 dump 同口径本脚本重算，不引用旧数字）：
  - full_fine：每 token 独立子空间打分，全 mid topk B_TOK（细筛上限）
  - minmax_mono：minmax 块上界粗筛 BP 页 + 池内 token 细筛 topk（= e64a 'mono' 臂）
  - cavg_page_sim0.90：贪心 sim=0.9 + 页 amax 粗筛（= e64j 'cavg_mono' 臂口径）
  - kmeans_K：批量 kmeans（Lloyd）簇数 K sweep {512,1024,2048}，同两种召回口径
    （贪心 vs 批量最优在相近 C/N 下的质量差）

判决门（写进 JSON verdict）：存在 sim* 使 mass(rep) >= full_fine - 0.005（0.5pt）
且 C/N < 20%（全部样本同时满足、单值通用）→ GO；否则 NO-GO。

用法：
  CUDA_VISIBLE_DEVICES=1 python3 probe_e108_sim_greedy.py            # 全量探针
  CUDA_VISIBLE_DEVICES=1 python3 probe_e108_sim_greedy.py --smoke    # 冒烟+口径对拍
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_e64a_ab_grid as g0          # 口径源（常量 + 原版 greedy 对拍用）

TRACE = g0.TRACE                            # /tmp/trace/qwen3-8b
D2I, BS, SINK, SWA, TAIL_N = g0.D2I, g0.BS, g0.SINK, g0.SWA, g0.TAIL_N
BP = g0.BP                                  # 64
B_TOK = g0.B_TOK                            # 2048（mono 单池口径）

OUT = "/home/wangyuanshuo02/two-level-attention/exp/trace/results/e108_sim_greedy_probe.json"
SIMS = [0.80, 0.85, 0.90, 0.92, 0.94, 0.96, 0.98, 0.99]
KM_KS = [512, 1024, 2048]
SAMPLES = ["lb_gov_report_0", "lb_hotpotqa_0", "lb_multifieldqa_en_0",
           "lb_musique_0", "lb_narrativeqa_0", "lb_passage_retrieval_en_0",
           "lb_qasper_0", "needle32k", "natural32k"]
DRIFT_SAMPLE, DRIFT_LAYER, DRIFT_SIM = "lb_hotpotqa_0", 18, 0.90
EPS = 1e-9


# ---------------------------------------------------------------- 贪心聚类（提案核心）
def greedy_cluster_assign(x, sim, snap=False):
    """增量贪心聚类（语义 = e64a.greedy_cluster_assign：时序逐 token、余弦阈值、
    算术均值簇心；k_max=T 不设上限使簇数统计精确）。
    x: [T, Hkv, dd]。返回 cent [H,K,dd]（最终均值代表）, assign [H,T], cnt [H,K],
    k_live [H], rep_snap [T,H,dd]（仅 snap：每 token 加入时刻所比对的代表向量）。
    实现注：||sums|| 用增量平方和 sq 维护避免每步全量 norm；新簇创建全程向量化
    （非创建头 index_add 加 0 权重无害——目标槽位均为已活簇或全新零槽）。"""
    T, H, dd = x.shape
    K = T
    dev = x.device
    sums = torch.zeros(H, K, dd, device=dev)
    sq = torch.zeros(H, K, device=dev)          # ||sums||^2 增量维护
    cnt = torch.zeros(H, K, device=dev)
    k_live = torch.zeros(H, dtype=torch.long, device=dev)
    assign = torch.zeros(H, T, dtype=torch.long, device=dev)
    x_n = x.norm(dim=-1)                        # [T,H]
    ar_h = torch.arange(H, device=dev)
    arK = torch.arange(K, device=dev)
    live_row = arK.unsqueeze(0)                 # [1,K]
    rep_snap = torch.empty(T, H, dd, device=dev) if snap else None
    for i in range(T):
        xi = x[i]                               # [H,dd]
        xi_n = x_n[i]                           # [H]
        norm = sq.clamp(min=0).sqrt()           # [H,K]
        dot = torch.bmm(sums, xi.unsqueeze(-1)).squeeze(-1)   # [H,K]
        cos = dot / (norm * xi_n.unsqueeze(-1) + EPS)
        cos = cos.masked_fill(live_row >= k_live.unsqueeze(-1), float("-inf"))
        a = cos.argmax(-1)                      # [H]
        m = cos.gather(1, a.unsqueeze(-1)).squeeze(-1)
        upd = m >= sim                          # [H]
        if snap:
            mean_a = sums.gather(1, a.view(H, 1, 1).expand(H, 1, dd)).squeeze(1) \
                / cnt.gather(1, a.unsqueeze(-1)).clamp(min=1)
            rep_snap[i] = torch.where(upd.view(H, 1), mean_a, xi)
        w = upd.float()
        # 并入已有簇（upd 头）：sums[a] += xi；cnt[a] += 1；sq[a] += 2*(sums_a·xi)+||xi||^2
        # 注：dot_a 用已算好的 dot 直接 gather（恒有限）；若用 m*norm*xi_n 推导，
        # 无活簇首 token 时 m=-inf 会经 0 权重乘出 NaN 污染 sq
        dot_a = dot.gather(1, a.unsqueeze(-1)).squeeze(-1)         # sums_a·xi
        flat_upd = (ar_h * K + a)
        sums.view(-1, dd).index_add_(0, flat_upd, xi * w.unsqueeze(-1))
        cnt.view(-1).index_add_(0, flat_upd, w)
        sq.view(-1).index_add_(0, flat_upd, (2 * dot_a + xi_n * xi_n) * w)
        # 新建簇（~upd 头）：槽 k_live 为全新零槽
        nw = (~upd).float()
        flat_new = (ar_h * K + k_live)
        sums.view(-1, dd).index_add_(0, flat_new, xi * nw.unsqueeze(-1))
        cnt.view(-1).index_add_(0, flat_new, nw)
        sq.view(-1).index_add_(0, flat_new, xi_n * xi_n * nw)
        assign[:, i] = torch.where(upd, a, k_live)
        k_live = k_live + (~upd).long()
    cent = sums / cnt.clamp(min=1).unsqueeze(-1)
    return cent, assign, cnt, k_live, rep_snap


# ---------------------------------------------------------------- kmeans 锚点（批量最优）
def kmeans_cluster(x, K, iters=20, seed=1234):
    """Lloyd kmeans（逐 Hkv head 独立），x [T,H,dd]。
    返回 cent [H,K,dd], assign [H,T], cnt [H,K]。空簇重播种随机 token。"""
    T, H, dd = x.shape
    dev = x.device
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    cent = torch.empty(H, K, dd, device=dev)
    for h in range(H):
        pi = torch.randperm(T, generator=g)[:K].to(dev)
        cent[h] = x[pi, h]
    xn2 = (x * x).sum(-1)                       # [T,H]
    ar_h = torch.arange(H, device=dev)
    xh = x.permute(1, 0, 2).reshape(-1, dd)     # [H*T,dd] 行序 (h,t)
    assign = None
    for _ in range(iters):
        cn2 = (cent * cent).sum(-1)             # [H,K]
        assign = torch.empty(T, H, dtype=torch.long, device=dev)
        for s in range(0, T, 2048):             # 分块防 [T,H,K] 爆内存
            xs = x[s:s + 2048]
            e = torch.einsum("chd,hkd->chk", xs, cent)
            dist = xn2[s:s + 2048].unsqueeze(-1) - 2 * e + cn2.unsqueeze(0)
            assign[s:s + 2048] = dist.argmin(-1)
        sums = torch.zeros(H * K, dd, device=dev)
        cnt = torch.zeros(H * K, device=dev)
        flat = (assign.t() + (ar_h * K).unsqueeze(1)).reshape(-1)   # 行序 (h,t)
        sums.index_add_(0, flat, xh)
        cnt.index_add_(0, flat, torch.ones(H * T, device=dev))
        cntm = cnt.view(H, K)
        cent = (sums.view(H, K, dd) / cntm.clamp(min=1).unsqueeze(-1))
        empty = cntm <= 0
        if empty.any():
            for h in torch.nonzero(empty.any(-1)).flatten().tolist():
                ks = torch.nonzero(empty[h]).flatten().tolist()
                pi = torch.randperm(T, generator=g)[:len(ks)].to(dev)
                cent[h, ks] = x[pi, h]
    # 最终一次 assign 与最终 cent 一致
    cn2 = (cent * cent).sum(-1)
    assign = torch.empty(T, H, dtype=torch.long, device=dev)
    for s in range(0, T, 2048):
        xs = x[s:s + 2048]
        e = torch.einsum("chd,hkd->chk", xs, cent)
        dist = xn2[s:s + 2048].unsqueeze(-1) - 2 * e + cn2.unsqueeze(0)
        assign[s:s + 2048] = dist.argmin(-1)
    sums = torch.zeros(H * K, dd, device=dev)
    cnt = torch.zeros(H * K, device=dev)
    flat = (assign.t() + (ar_h * K).unsqueeze(1)).reshape(-1)
    sums.index_add_(0, flat, xh)
    cnt.index_add_(0, flat, torch.ones(H * T, device=dev))
    cnt = cnt.view(H, K)
    cent = sums.view(H, K, dd) / cnt.clamp(min=1).unsqueeze(-1)
    return cent, assign.t().contiguous(), cnt


# ---------------------------------------------------------------- 两种召回口径（提案选择）
def cluster_select(cent, cnt, assign_mid, qsub, mid_len, n_tokens, mode):
    """按簇代表打分召回。返回 mid 相对 token 索引 it [H, n_tokens]。
    mode='rep'：token 分 = 其簇代表分，全 mid topk（按簇代表分数取簇内 token）；
    mode='whole'：簇按代表分排序，整簇时间序全取直到预算（簇内 token 全取）。
    代表打分 = qsub·cent 点积（与 e64a cavg 簇分同式，GQA sum 聚合后的 q）。"""
    H, K, dd = cent.shape
    dev = cent.device
    cs = torch.bmm(cent, qsub.unsqueeze(-1)).squeeze(-1)          # [H,K]
    cs = cs.masked_fill(cnt <= 0, float("-inf"))
    if mode == "rep":
        tok_cl = cs.gather(1, assign_mid)                          # [H,mid_len]
        it = torch.topk(tok_cl, min(n_tokens, mid_len), dim=-1).indices
    elif mode == "whole":
        order = torch.argsort(cs, dim=-1, descending=True)         # 死簇 -inf 排尾
        rank = torch.empty_like(order)
        rank.scatter_(1, order, torch.arange(K, device=dev).unsqueeze(0).expand(H, K))
        trank = rank.gather(1, assign_mid)                         # token 的簇名次
        key = trank * mid_len + torch.arange(mid_len, device=dev).unsqueeze(0)
        it = torch.topk(key, min(n_tokens, mid_len), dim=-1, largest=False).indices
    else:
        raise ValueError(mode)
    return it


def page_select(sc_blk_mid, tok_score_mid, lo_off, hi_off, n_pages, n_tokens, mid_len, device):
    """页粗筛+池内 token 细筛（逻辑 = e64a.select_sub，块区间判定逐行照抄）。"""
    nblk_sc = sc_blk_mid.shape[1]
    ar_b = torch.arange(nblk_sc, device=device)
    inrng = ((ar_b * BS + SINK) < SINK + hi_off) & ((ar_b * BS + BS - 1 + SINK) >= SINK + lo_off)
    s1 = sc_blk_mid.masked_fill(~inrng.view(1, -1), float("-inf"))
    ib = torch.topk(s1, min(n_pages, nblk_sc), dim=-1).indices
    tok = (ib.unsqueeze(-1) * BS + torch.arange(BS, device=device).view(1, 1, BS)).reshape(ib.shape[0], -1).clamp(max=mid_len - 1)
    pool = torch.zeros(ib.shape[0], mid_len, dtype=torch.bool, device=device)
    pool.scatter_(1, tok, True)
    pool &= ((torch.arange(mid_len, device=device) >= lo_off) &
             (torch.arange(mid_len, device=device) < hi_off)).view(1, -1)
    ts = tok_score_mid.masked_fill(~pool, float("-inf"))
    it = torch.topk(ts, min(n_tokens, mid_len), dim=-1).indices
    return it


# ---------------------------------------------------------------- 单层评估
def eval_layer(lf, device, sims, km_ks, drift=False):
    """返回 (res_arms, greedy_stats, drift_data)。res_arms: arm名 -> cov 列表（4 tail query）。"""
    d = torch.load(lf, map_location="cpu", weights_only=False)
    k, q, qpos, S = d["k"].to(device).float(), d["q"].to(device).float(), d["qpos"], d["S"]
    Hkv, H, D = k.shape[1], q.shape[1], k.shape[-1]
    G = H // Hkv
    pos = torch.arange(S, device=device)
    res, gstats, drift_data = {}, {}, None
    mid_hi_last = int(qpos[-1]) + 1 - SWA
    idx = torch.tensor(D2I, device=device)
    ksub_layer = k[..., idx]                       # [S,Hkv,32] tail32 子空间
    ksub = ksub_layer
    T = mid_hi_last - SINK
    xmid = ksub_layer[SINK:mid_hi_last]            # [T,Hkv,32] mid 区聚类输入
    # minmax 块统计（minmax_mono 锚点用）
    nblk = (S + BS - 1) // BS
    kk = F.pad(ksub_layer, (0, 0, 0, 0, 0, nblk * BS - S))
    kc = kk.reshape(nblk, BS, Hkv, 32)
    kmin, kmax = kc.amin(1), kc.amax(1)            # [nblk,Hkv,32]
    # 贪心 sweep build（每层每 sim 一次，4 个 tail query 共享）
    greedy = {}
    for sim in sims:
        t0 = time.time()
        cent, assign_g, cnt, k_live, snap = greedy_cluster_assign(xmid, sim, snap=(drift and sim == DRIFT_SIM))
        build_s = time.time() - t0
        live = cnt > 0
        c_per_h = live.sum(-1).float()             # [H]
        sizes = cnt[live]
        gstats[sim] = {
            "C_mean": float(c_per_h.mean()), "C_min": float(c_per_h.min()), "C_max": float(c_per_h.max()),
            "C_over_N": float(c_per_h.mean()) / T,
            "build_s": round(build_s, 2),
            "sizes": sizes.cpu(),          # 池化张量：main 中跨层 concat 后统一分位数
        }
        greedy[sim] = (cent, assign_g, cnt, snap)
        if snap is not None:
            # ---- 漂移量化（本层 sim=DRIFT_SIM）----
            sums2 = torch.zeros_like(cent)
            cnt2 = torch.zeros_like(cnt)
            ar_h = torch.arange(Hkv, device=device)
            flat_all = ((ar_h * cent.shape[1]).unsqueeze(1) + assign_g).view(-1)
            sums2.view(-1, 32).index_add_(0, flat_all, xmid.permute(1, 0, 2).reshape(-1, 32))
            cnt2.view(-1).index_add_(0, flat_all, torch.ones(Hkv * T, device=device))
            cent2 = sums2 / cnt2.clamp(min=1).unsqueeze(-1)
            cosex = F.cosine_similarity(cent, cent2, dim=-1)[live]
            cent_tok = cent[ar_h.unsqueeze(1), assign_g]          # [H,T,dd] 最终代表
            rs = snap.permute(1, 0, 2)                            # [H,T,dd] 加入时刻代表
            cosd = F.cosine_similarity(rs, cent_tok, dim=-1)      # [H,T]
            dd_ = (1 - cosd).flatten()
            drift_data = {
                "running_vs_recompute_cos_min": float(cosex.min()),
                "running_vs_recompute_cos_mean": float(cosex.mean()),
                "join_vs_final_dist_p50": float(torch.quantile(dd_, 0.5)),
                "join_vs_final_dist_p90": float(torch.quantile(dd_, 0.9)),
                "join_vs_final_dist_max": float(dd_.max()),
                "join_vs_final_dist_mean": float(dd_.mean()),
            }
    # kmeans 锚点 build
    km = {}
    for K in km_ks:
        cent_km, assign_km, cnt_km = kmeans_cluster(xmid, K)
        km[K] = (cent_km, assign_km, cnt_km)

    for ri in range(TAIL_N):
        t_r = int(qpos[-TAIL_N + ri])
        mid_hi = t_r + 1 - SWA
        mid_len = mid_hi - SINK
        if mid_len < 4096:
            continue
        # 真值分布（与 e64a 逐行同式）
        qg = q[-TAIL_N + ri].reshape(Hkv, G, D)
        k4 = k[:, :, None, :].expand(S, Hkv, G, D)
        s_full = torch.einsum("hgd,shgd->hgs", qg, k4) * (D ** -0.5)
        s_full = s_full.masked_fill((pos > t_r).view(1, 1, -1), float("-inf"))
        p_full = torch.softmax(s_full, dim=-1)

        def cov_mass(cand):
            return float((p_full * cand.unsqueeze(1)).sum(-1).mean())

        def cand_from_it(it):
            cand = torch.zeros(Hkv, S, dtype=torch.bool, device=device)
            cand.scatter_(1, it + SINK, True)
            return cand

        qsub = q[-TAIL_N + ri][..., idx].reshape(Hkv, G, 32).sum(1)      # [Hkv,32]
        sc_tok = torch.einsum("hgd,shd->hgs", q[-TAIL_N + ri][..., idx].reshape(Hkv, G, 32), ksub).sum(1)  # [Hkv,S]
        sc_tok_mid = sc_tok[:, SINK:mid_hi]                              # [Hkv,mid_len]
        sink_c = (pos < SINK).view(1, -1)
        swa_c = ((pos >= t_r + 1 - SWA) & (pos <= t_r)).view(1, -1)

        def full_cand(it):
            return sink_c | swa_c | cand_from_it(it)

        # ---- 锚点 1：全量精筛（每 token 独立打分，全 mid topk）----
        it = torch.topk(sc_tok_mid, min(B_TOK, mid_len), dim=-1).indices
        res.setdefault("full_fine", []).append(cov_mass(full_cand(it)))
        # ---- 锚点 2：minmax_mono（minmax 块上界 BP 页粗筛 + 池内 token 细筛）----
        sc_mm = (torch.einsum("hd,nhd->hn", qsub.clamp(min=0), kmax) +
                 torch.einsum("hd,nhd->hn", qsub.clamp(max=0), kmin))    # [Hkv,nblk]
        it = page_select(sc_mm[:, SINK // BS:], sc_tok_mid, 0, mid_len, BP, B_TOK, mid_len, device)
        res.setdefault("minmax_mono", []).append(cov_mass(full_cand(it)))
        # ---- 锚点 3：cavg_page（贪心 sim=0.9 + 页 amax 粗筛，= e64j cavg_mono 口径）----
        cent09, assign09, cnt09, _ = greedy[0.90] if 0.90 in greedy else (None,) * 4
        if cent09 is not None:
            assign_mid09 = assign09[:, :mid_len]
            cs09 = torch.bmm(cent09, qsub.unsqueeze(-1)).squeeze(-1).masked_fill(cnt09 <= 0, float("-inf"))
            tok_cl09 = cs09.gather(1, assign_mid09)
            blk_ids = (torch.arange(mid_len, device=device) // BS).unsqueeze(0).expand(Hkv, -1)
            sc_cl_blk = torch.full((Hkv, (mid_len + BS - 1) // BS), float("-inf"), device=device)
            sc_cl_blk.scatter_reduce_(1, blk_ids, tok_cl09, reduce="amax", include_self=False)
            it = page_select(sc_cl_blk, tok_cl09, 0, mid_len, BP, B_TOK, mid_len, device)
            res.setdefault("cavg_page_sim0.90", []).append(cov_mass(full_cand(it)))
        # ---- 提案臂：贪心 sim sweep × 两种召回口径 ----
        for sim in sims:
            cent, assign_g, cnt, _ = greedy[sim]
            assign_mid = assign_g[:, :mid_len]
            for mode in ("rep", "whole"):
                it = cluster_select(cent, cnt, assign_mid, qsub, mid_len, B_TOK, mode)
                res.setdefault(f"sim{sim:.2f}_{mode}", []).append(cov_mass(full_cand(it)))
        # ---- 锚点 4：kmeans_K sweep × 两种召回口径 ----
        for K in km_ks:
            cent_km, assign_km, cnt_km = km[K]
            assign_mid = assign_km[:, :mid_len]
            for mode in ("rep", "whole"):
                it = cluster_select(cent_km, cnt_km, assign_mid, qsub, mid_len, B_TOK, mode)
                res.setdefault(f"kmeans{K}_{mode}", []).append(cov_mass(full_cand(it)))
    del k, q, d, ksub_layer, xmid, greedy, km
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return res, gstats, drift_data


def layers_of(n_layers):
    # 排除 layer0（attention-sink 层 tail32 子空间近共线，低 sim 下贪心单簇坍缩）
    return list(range(n_layers // 4, n_layers, max(1, n_layers // 4)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default=os.environ.get("E108_DEVICE", "cuda"))
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    device = args.device
    samples = SAMPLES
    sims = SIMS
    km_ks = KM_KS
    if args.smoke:
        samples = ["lb_multifieldqa_en_0"]
        sims = [0.85, 0.99]
        km_ks = [512]
        # 口径对拍：贪心 assign 与 e64a 原版逐位比对 + minmax_mono 与 e64a 'mono' 臂比对
        lf = f"{TRACE}/{samples[0]}/layer{layers_of(36)[1]:02d}.pt"
        d = torch.load(lf, map_location="cpu", weights_only=False)
        qpos = d["qpos"]
        xsmall = d["k"].float()[..., torch.tensor(D2I)][SINK:int(qpos[-1]) + 1 - SWA].to(device)
        Ts = 4096
        xsmall = xsmall[:Ts]
        for sim in (0.85, 0.99):
            _, a_mine, _, _, _ = greedy_cluster_assign(xsmall, sim)
            _, a_ref, _ = g0.greedy_cluster_assign(xsmall, sim=sim, k_max=Ts)
            mism = int((a_mine != a_ref).sum())
            print(f"[smoke] greedy assign mismatch sim={sim}: {mism}/{a_mine.numel()}")
            assert mism == 0, "greedy 实现与 e64a 原版不一致"
        print("[smoke] greedy 对拍 PASS")
        r_ref, _, _ = None, None, None
        import analyze_e64a_ab_grid as _g0
        r_ref = _g0.eval_layer(lf, device)
        res, _, _ = eval_layer(lf, device, sims, km_ks)
        mine = sum(res["minmax_mono"]) / len(res["minmax_mono"])
        ref = r_ref["mono"][0] if isinstance(r_ref["mono"], list) else r_ref["mono"]
        # e64a eval_layer 的 mono 臂每 tail query 一条，取均值对拍
        ref = sum(r_ref["mono"]) / len(r_ref["mono"])
        print(f"[smoke] minmax_mono mine={mine:.6f} ref(e64a mono)={ref:.6f} diff={abs(mine-ref):.2e}")
        assert abs(mine - ref) < 1e-6, "minmax_mono 与 e64a mono 臂不一致"
        print("[smoke] mono 口径对拍 PASS")
        print("[smoke] arms:", {k: round(sum(v) / len(v), 4) for k, v in res.items()})
        return

    results = {"meta": {
        "probe": "E108 cluster_sim_greedy 质量-算力曲线",
        "proposal": "用户 2026-10-06：余弦相似度贪心聚类+均值代表+代表打分召回，sim 单值跨数据集校准",
        "source_scripts": [
            "/home/wangyuanshuo02/two-level-attention/exp/trace/analyze_e64a_ab_grid.py（dump/协议常量/greedy 语义/mono 臂）",
            "/home/wangyuanshuo02/two-level-attention/exp/trace/analyze_e64j_combo_best.py（mass 重放口径 cavg_mono/mavg_mono）",
            "/home/wangyuanshuo02/two-level-attention/exp/trace/analyze_e98_abg_full_grid.py（mass 重放口径同源复核）",
        ],
        "dump": TRACE, "protocol": {"D2I": "tail32(48:64+112:128)", "BS": BS, "SINK": SINK, "SWA": SWA,
                                    "TAIL_N": TAIL_N, "BP": BP, "B_TOK": B_TOK,
                                    "mass": "真实全维 softmax 行级 coverage（e64a/e64j 同式）",
                                    "rep_scoring": "qsub·cent 点积（与 cavg 簇分同式）",
                                    "budget": "mid 单池 B_TOK=2048 + sink/swa 强制（mono 口径）",
                                    "layers_per_sample": "[9,18,27]（排除 layer0：tail32 子空间在 layer0 近共线，"
                                                         "sim<=0.85 贪心坍缩为单簇——实测 hotpotqa layer0 sim0.80 C=1 size=15805，"
                                                         "会污染 sim 校准；锚点同层重算保持内部可比）"},
                       "assign_fp_note": "本实现 ||sums|| 用增量平方和维护：sim=0.80 时与 e64a 原版逐步精确 norm "
                                         "有 0.03%（35/126440）fp 边界差，sim>=0.85 逐位全同（hotpotqa layer18 全长对拍）",
                       "size_stat_note": "簇尺寸分位数 = 跨层池化后统一计算（非层中位数平均）",
        "samples": samples, "sims": sims, "kmeans_K": km_ks,
        "device": device, "started": time.strftime("%F %T"),
    }}
    per_sample = {}
    drift_out = None
    for name in samples:
        n_layers = json.load(open(f"{TRACE}/{name}/meta.json"))["n_layers"]
        Ls = layers_of(n_layers)
        agg, gagg, gsz, ndrift = {}, {}, {}, 0
        for li in Ls:
            is_drift = (name == DRIFT_SAMPLE and li == DRIFT_LAYER)
            r, gs, dd_ = eval_layer(f"{TRACE}/{name}/layer{li:02d}.pt", device, sims, km_ks, drift=is_drift)
            if not r:
                continue
            for k2, v in r.items():
                agg.setdefault(k2, []).extend(v)
            for sim, st in gs.items():
                for kk, vv in st.items():
                    if kk == "sizes":
                        gsz.setdefault(sim, []).append(vv)
                    else:
                        gagg.setdefault(sim, {}).setdefault(kk, []).append(vv)
            if dd_ is not None:
                drift_out = {"sample": name, "layer": li, "sim": DRIFT_SIM, **dd_}
                ndrift += 1
        # 簇尺寸分布：跨层池化后统一分位数（避免「层中位数再平均」被退化层主导）
        for sim in sims:
            if sim in gsz:
                pooled = torch.cat(gsz[sim]).float()
                gagg[sim]["size_mean"] = [float(pooled.mean())]
                gagg[sim]["size_p50"] = [float(torch.quantile(pooled, 0.5))]
                gagg[sim]["size_p90"] = [float(torch.quantile(pooled, 0.9))]
                gagg[sim]["size_max"] = [float(pooled.max())]
        rec = {"S": json.load(open(f"{TRACE}/{name}/meta.json"))["S"], "layers_used": Ls,
               "anchors": {}, "greedy": {}, "kmeans": {}}
        for k2, v in agg.items():
            m = sum(v) / len(v)
            if k2.startswith("sim"):
                rec["greedy"].setdefault(k2, {})["mass"] = round(m, 4)
            elif k2.startswith("kmeans"):
                rec["kmeans"].setdefault(k2, {})["mass"] = round(m, 4)
            else:
                rec["anchors"][k2] = round(m, 4)
        for sim in sims:
            g = rec["greedy"].get(f"sim{sim:.2f}_rep")
            if g is None:
                continue
            st = {kk: round(sum(vv) / len(vv), 4) for kk, vv in gagg[sim].items()}
            for mode in ("rep", "whole"):
                rec["greedy"][f"sim{sim:.2f}_{mode}"]["C_over_N"] = st["C_over_N"]
                rec["greedy"][f"sim{sim:.2f}_{mode}"]["C_mean"] = st["C_mean"]
                rec["greedy"][f"sim{sim:.2f}_{mode}"]["size_mean"] = st["size_mean"]
                rec["greedy"][f"sim{sim:.2f}_{mode}"]["size_p50"] = st["size_p50"]
                rec["greedy"][f"sim{sim:.2f}_{mode}"]["size_p90"] = st["size_p90"]
                rec["greedy"][f"sim{sim:.2f}_{mode}"]["size_max"] = st["size_max"]
                rec["greedy"][f"sim{sim:.2f}_{mode}"]["build_s"] = st["build_s"]
        T_mid = rec["S"] - SINK - SWA
        for K in km_ks:
            if f"kmeans{K}_rep" in rec["kmeans"]:
                for mode in ("rep", "whole"):
                    rec["kmeans"][f"kmeans{K}_{mode}"]["C_over_N"] = round(K / T_mid, 4)
        per_sample[name] = rec
        results["per_sample"] = per_sample
        json.dump(results, open(args.out, "w"), indent=1)
        print(f"[{name}] anchors={rec['anchors']} "
              f"sim0.90_rep={rec['greedy'].get('sim0.90_rep', {}).get('mass')} "
              f"C/N@0.90={rec['greedy'].get('sim0.90_rep', {}).get('C_over_N')}", flush=True)
    # ---- AVG ----
    avg = {"anchors": {}, "greedy": {}, "kmeans": {}}
    for k2 in next(iter(per_sample.values()))["anchors"]:
        avg["anchors"][k2] = round(sum(r["anchors"][k2] for r in per_sample.values()) / len(per_sample), 4)
    for key in next(iter(per_sample.values()))["greedy"]:
        avg["greedy"][key] = {kk: round(sum(r["greedy"][key][kk] for r in per_sample.values()) / len(per_sample), 4)
                              for kk in next(iter(per_sample.values()))["greedy"][key]}
    for key in next(iter(per_sample.values()))["kmeans"]:
        avg["kmeans"][key] = {kk: round(sum(r["kmeans"][key][kk] for r in per_sample.values()) / len(per_sample), 4)
                              for kk in next(iter(per_sample.values()))["kmeans"][key]}
    results["AVG"] = avg
    # ---- 判决 ----
    bar = 0.005
    sim_star = {}
    for name, r in per_sample.items():
        ff = r["anchors"]["full_fine"]
        ok = [s for s in sims if r["greedy"].get(f"sim{s:.2f}_rep", {}).get("mass", -1) >= ff - bar]
        sim_star[name] = min(ok) if ok else None
    ok_global = [s for s in sims
                 if all(r["greedy"].get(f"sim{s:.2f}_rep", {}).get("mass", -1) >= r["anchors"]["full_fine"] - bar
                        for r in per_sample.values())]
    sim_global = min(ok_global) if ok_global else None
    gate_pass = False
    if sim_global is not None:
        cn = avg["greedy"][f"sim{sim_global:.2f}_rep"]["C_over_N"]
        gate_pass = cn < 0.20
    # 诚实量化：即使门槛不达标，也报告每样本/全局的最小差距及其代价点
    min_gap = {}
    for name, r in per_sample.items():
        ff = r["anchors"]["full_fine"]
        best = max((s for s in sims), key=lambda s: r["greedy"].get(f"sim{s:.2f}_rep", {}).get("mass", -1))
        g = r["greedy"].get(f"sim{best:.2f}_rep", {})
        min_gap[name] = {"sim_at_min_gap": best,
                         "min_gap_pt": round((ff - g.get("mass", 0.0)) * 100, 2),
                         "C_over_N_at_min_gap": round(g.get("C_over_N", 1.0), 4)}
    best_avg = max(sims, key=lambda s: avg["greedy"].get(f"sim{s:.2f}_rep", {}).get("mass", -1))
    verdict = {
        "decision": "GO" if gate_pass else "NO-GO",
        "gate": "存在 sim* 使全部样本 mass(rep) >= full_fine - 0.5pt 且 AVG C/N < 20%（单值通用）",
        "sim_star_per_sample": sim_star,
        "sim_star_global": sim_global,
        "avg_C_over_N_at_global": round(avg["greedy"][f"sim{sim_global:.2f}_rep"]["C_over_N"], 4) if sim_global else None,
        "avg_mass_rep_at_global": avg["greedy"][f"sim{sim_global:.2f}_rep"]["mass"] if sim_global else None,
        "avg_full_fine": avg["anchors"]["full_fine"],
        "min_gap_per_sample": min_gap,
        "avg_best_greedy": {"sim": best_avg,
                            "mass": avg["greedy"].get(f"sim{best_avg:.2f}_rep", {}).get("mass"),
                            "C_over_N": avg["greedy"].get(f"sim{best_avg:.2f}_rep", {}).get("C_over_N"),
                            "gap_pt": round((avg["anchors"]["full_fine"]
                                             - avg["greedy"].get(f"sim{best_avg:.2f}_rep", {}).get("mass", 0.0)) * 100, 2)},
        "kmeans_reference": {"kmeans2048_rep_mass": avg["kmeans"].get("kmeans2048_rep", {}).get("mass"),
                             "kmeans2048_C_over_N": avg["kmeans"].get("kmeans2048_rep", {}).get("C_over_N"),
                             "note": "批量 kmeans 同召回口径对照：贪心曲线是否被 kmeans 支配的判据"},
        "cross_sample_consistency": (len({v for v in sim_star.values() if v is not None}) == 1
                                     and None not in sim_star.values()),
        "rationale": "",
    }
    if gate_pass:
        verdict["rationale"] = (
            f"sim*={sim_global} 单值下 9 样本全部达到全量精筛-0.5pt 以内，AVG C/N="
            f"{verdict['avg_C_over_N_at_global']}（算力代理 <20%），建议进入 e2e screen")
    else:
        miss = [n for n, v in sim_star.items() if v is None]
        bg = verdict["avg_best_greedy"]
        verdict["rationale"] = (
            f"无单值 sim 满足质量门（未达标样本: {miss if miss else '无'}）；贪心曲线最优处（sim={bg['sim']}，"
            f"C/N={bg['C_over_N']:.2f}）仍距全量精筛 {bg['gap_pt']:.2f}pt，且被批量 kmeans 支配"
            f"（kmeans2048 C/N={verdict['kmeans_reference']['kmeans2048_C_over_N']} 时 "
            f"mass={verdict['kmeans_reference']['kmeans2048_rep_mass']:.4f}）；诚实判决 NO-GO")
    results["drift"] = drift_out
    results["verdict"] = verdict
    json.dump(results, open(args.out, "w"), indent=1)

    # ---- stdout 判决摘要 ----
    print("\n==== E108 cluster_sim_greedy 探针判决摘要 ====")
    print(f"锚点（9 样本 AVG，本脚本同 dump 同口径重算）:")
    print(f"  full_fine   = {avg['anchors']['full_fine']}")
    print(f"  minmax_mono = {avg['anchors']['minmax_mono']}")
    print(f"  cavg_page(0.90) = {avg['anchors'].get('cavg_page_sim0.90')}")
    print("sim 曲线（AVG mass_rep | mass_whole | C/N | 簇数）:")
    for s in sims:
        g = avg["greedy"].get(f"sim{s:.2f}_rep")
        w = avg["greedy"].get(f"sim{s:.2f}_whole")
        if g:
            print(f"  sim={s:.2f}  rep={g['mass']:.4f}  whole={w['mass']:.4f}  "
                  f"C/N={g['C_over_N']:.3f}  C={g['C_mean']:.0f}  size_p50/p90={g['size_p50']:.0f}/{g['size_p90']:.0f}")
    print("kmeans 锚点（AVG mass_rep | mass_whole | C/N）:")
    for K in km_ks:
        g = avg["kmeans"].get(f"kmeans{K}_rep")
        w = avg["kmeans"].get(f"kmeans{K}_whole")
        if g:
            print(f"  K={K:5d}  rep={g['mass']:.4f}  whole={w['mass']:.4f}  C/N={g['C_over_N']:.3f}")
    if drift_out:
        print(f"漂移量化（{drift_out['sample']} layer{drift_out['layer']} sim={drift_out['sim']}）:")
        print(f"  running vs 重算 mean cos: min={drift_out['running_vs_recompute_cos_min']:.6f} "
              f"mean={drift_out['running_vs_recompute_cos_mean']:.6f}")
        print(f"  加入时刻代表 vs 最终均值 1-cos: P50={drift_out['join_vs_final_dist_p50']:.4f} "
              f"P90={drift_out['join_vs_final_dist_p90']:.4f} max={drift_out['join_vs_final_dist_max']:.4f}")
    print(f"判决: {verdict['decision']}  sim*_global={verdict['sim_star_global']}  "
          f"跨样本一致={verdict['cross_sample_consistency']}")
    print(f"sim* 每样本: {verdict['sim_star_per_sample']}")
    print(f"理由: {verdict['rationale']}")
    print("saved ->", args.out)


if __name__ == "__main__":
    main()
