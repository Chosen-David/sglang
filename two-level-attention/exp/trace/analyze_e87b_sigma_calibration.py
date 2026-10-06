# E87b：合理的 σ 校准（用户 2026-10-02「你找个合理的sigma呢」）
# 思路：σ 的本质 = 质量-预算旋钮。固定 σ 要"合理"，必须满足预算匹配——
#   该区域选中 token 数 ≈ 该区域 token 预算。因此对每行直接反解：
#   σ_implied = sink_max − s_(k)，其中 s_(k) = 该区细筛分第 k 大（k = 区域预算）
# 输出：
#   1) σ_implied 分布（p10/p50/p90，far/near 两侧，group-sum 与 group-mean 双口径）
#      —— group-mean 口径可直接迁移到 e2e（e2e sigma 分支用 mean）
#   2) 扩展 σ 网格 {16,32,48,64,96,128,256} 的 cov / sel_tok（补 e87 网格只到 32 的盲区）
#   3) 预算匹配直接 topk（无块粗筛）cov —— per-row σ* 的覆盖上限参照
#   4) 噪声分析（用户 2026-10-02「加入噪声的考虑」）：
#      a. 子空间估计噪声 σ_n：32 维细筛分对全维真分数的残差 std（最小二乘尺度对齐后）
#         —— σ 阈值在噪声带内的 token 选/不选是随机的，σ 应含噪声余量
#      b. 噪声鲁棒臂：thr = sink_max − σ_implied − κ·σ_n（κ∈{1,2}），看 sel/cov 变化
#      c. sink 锚噪声：子空间 sink_max 与全维 sink_max 的差分布（锚本身漂移多大）
# 双口径注记：sink_max−σ 阈值卡死的机理判断也在此验证（σ_implied 分布宽 = 固定 σ 控不住预算）
import json
import os

import torch
import torch.nn.functional as F

import analyze_e64a_ab_grid as base

TRACE = base.TRACE
D2I, BS, SINK, SWA, TAIL_N = base.D2I, base.BS, base.SINK, base.SWA, base.TAIL_N
BP = 64
GAMMA = 1.0
ALPHA, BETA = 0.125, 0.25
SIGMAS_EXT = [16.0, 32.0, 48.0, 64.0, 96.0, 128.0, 256.0]
OUT = "/home/wangyuanshuo02/two-level-attention/exp/trace/results/e87b_sigma_calibration.json"


def eval_layer(lf, device):
    d = torch.load(lf, map_location="cpu", weights_only=False)
    k, q, qpos, S = d["k"].to(device).float(), d["q"].to(device).float(), d["qpos"], d["S"]
    Hkv, H, D = k.shape[1], q.shape[1], k.shape[-1]
    G = H // Hkv
    pos = torch.arange(S, device=device)
    res = {}
    cost = {}
    sig_imp = {"far": [], "near": []}   # group-sum 口径隐含 σ
    sig_norm = {"far": [], "near": []}  # 归一化 σ：σ_implied / 该行 mid 分数 std（自校准形式）
    cov_direct = []                     # 预算匹配直接 topk cov
    noise_stats = {"sigma_n": [], "sink_gap": []}  # 噪声量化
    for ri in range(TAIL_N):
        t_r = int(qpos[-TAIL_N + ri])
        mid_hi = t_r + 1 - SWA
        mid_len = mid_hi - SINK
        if mid_len < 4096:
            continue
        qg = q[-TAIL_N + ri].reshape(Hkv, G, D)
        k4 = k[:, :, None, :].expand(S, Hkv, G, D)
        s_full = torch.einsum("hgd,shgd->hgs", qg, k4) * (D ** -0.5)
        s_full = s_full.masked_fill((pos > t_r).view(1, 1, -1), float("-inf"))
        p_full = torch.softmax(s_full, dim=-1)
        # 全维真分数（group-sum、带 √D 缩放；与 sc_tok 尺度用回归对齐）
        s_true = s_full.sum(1)                                   # [Hkv, S]（causal mask 已含 -inf）

        def cov_mass(cand):
            return float((p_full * cand.unsqueeze(1)).sum(-1).mean())

        # 共享打分（e87 同款）
        idx = torch.tensor(D2I, device=device)
        ksub = k[..., idx]
        qsub = q[-TAIL_N + ri][..., idx].reshape(Hkv, G, 32).sum(1)
        sc_tok = torch.einsum("hgd,shd->hgs",
                              q[-TAIL_N + ri][..., idx].reshape(Hkv, G, 32), ksub).sum(1)
        sink_c = (pos < SINK).view(1, -1)
        swa_c = ((pos >= t_r + 1 - SWA) & (pos <= t_r)).view(1, -1)
        sink_max = sc_tok[:, :SINK].max(dim=-1).values          # [Hkv] group-sum 口径

        # 预算结构（e87 同款）
        near_L = int(ALPHA * mid_len)
        nb_near = int(round(BP * BETA))
        nb_far = BP - nb_near
        nt_near = min(int(nb_near * BS * GAMMA), 2048)
        nt_far = max(64, 2048 - nt_near)
        near_lo_off, near_hi_off = mid_len - near_L, mid_len
        far_hi_off = near_lo_off

        # ---- 1) 预算匹配隐含 σ ----
        sc_far = sc_tok[:, SINK:SINK + far_hi_off]              # [Hkv, far_L]
        sc_near = sc_tok[:, SINK + near_lo_off: SINK + mid_len]
        kth_far = sc_far.kthvalue(sc_far.shape[-1] - nt_far + 1, dim=-1).values
        kth_near = sc_near.kthvalue(sc_near.shape[-1] - nt_near + 1, dim=-1).values
        sig_imp["far"].extend((sink_max - kth_far).tolist())
        sig_imp["near"].extend((sink_max - kth_near).tolist())
        # ---- 1b) 噪声量化：子空间分 vs 全维真分数残差（最小二乘尺度对齐） ----
        sc_mid = sc_tok[:, SINK:SINK + mid_len]                 # [Hkv, mid_L]
        # 归一化 σ：隐含 σ / 该行 mid 分数 std —— 若比值跨行稳定则存在自校准 σ = c·std
        row_std = sc_mid.std(dim=-1)                            # [Hkv]
        sig_norm["far"].extend(((sink_max - kth_far) / row_std).tolist())
        sig_norm["near"].extend(((sink_max - kth_near) / row_std).tolist())
        st_mid = s_true[:, SINK:SINK + mid_len]
        a_hat = (st_mid * sc_mid).sum(-1) / (sc_mid * sc_mid).sum(-1).clamp(min=1e-9)
        resid = st_mid - a_hat.unsqueeze(1) * sc_mid
        sig_n = resid.std(dim=-1)                               # [Hkv] 每头估计噪声
        noise_stats["sigma_n"].extend(sig_n.tolist())
        # sink 锚噪声：对齐尺度下 全维 sink_max − 子空间 sink_max
        sink_gap = (s_true[:, :SINK].max(-1).values
                    - a_hat * sink_max)
        noise_stats["sink_gap"].extend(sink_gap.tolist())

        # ---- 1c) 噪声鲁棒臂：阈值下探 κ·σ_n（κ=1,2），看预算/覆盖弹性 ----
        for kap in [1.0, 2.0]:
            keep_f = sc_far >= (kth_far - kap * sig_n).unsqueeze(1)
            keep_n = sc_near >= (kth_near - kap * sig_n).unsqueeze(1)
            sf_ = int(keep_f.sum(-1).float().mean())
            sn_ = int(keep_n.sum(-1).float().mean())
            cf = torch.zeros(Hkv, S, dtype=torch.bool, device=device)
            cf.scatter_(1, (torch.arange(far_hi_off, device=device).view(1, -1)
                            .expand(Hkv, -1) * keep_f).masked_fill(~keep_f, 0) + SINK, keep_f)
            cn = torch.zeros(Hkv, S, dtype=torch.bool, device=device)
            cn.scatter_(1, (torch.arange(near_lo_off, near_hi_off, device=device).view(1, -1)
                            .expand(Hkv, -1) * keep_n).masked_fill(~keep_n, 0) + SINK, keep_n)
            res.setdefault(f"noise_k{int(kap)}", []).append(cov_mass(cf | cn | sink_c | swa_c))
            cost.setdefault(f"noise_k{int(kap)}", []).append(
                [mid_len * 32, float(sf_ + sn_)])

        # ---- 2) 预算匹配直接 topk cov（per-row σ* 覆盖上限） ----
        i_f = sc_far.topk(nt_far, dim=-1).indices
        i_n = sc_near.topk(nt_near, dim=-1).indices + near_lo_off
        cand = torch.zeros(Hkv, S, dtype=torch.bool, device=device)
        cand.scatter_(1, i_f + SINK, True)
        cand.scatter_(1, i_n + SINK, True)
        cov_direct.append(cov_mass(cand | sink_c | swa_c))

        # ---- 3) 扩展 σ 网格 ----
        for sg in SIGMAS_EXT:
            keep_f = sc_far >= (sink_max - sg).unsqueeze(1)
            keep_n = sc_near >= (sink_max - sg).unsqueeze(1)
            sf = int(keep_f.sum(-1).float().mean())
            sn = int(keep_n.sum(-1).float().mean())
            cf = torch.zeros(Hkv, S, dtype=torch.bool, device=device)
            cf.scatter_(1, (torch.arange(far_hi_off, device=device).view(1, -1)
                            .expand(Hkv, -1) * keep_f).masked_fill(~keep_f, 0) + SINK, keep_f)
            cn = torch.zeros(Hkv, S, dtype=torch.bool, device=device)
            cn.scatter_(1, (torch.arange(near_lo_off, near_hi_off, device=device).view(1, -1)
                            .expand(Hkv, -1) * keep_n).masked_fill(~keep_n, 0) + SINK, keep_n)
            res.setdefault(f"mid_sigma_{sg}", []).append(cov_mass(cf | cn | sink_c | swa_c))
            cost.setdefault(f"mid_sigma_{sg}", []).append(
                [mid_len * 32, float(sf + sn)])
            res.setdefault(f"near_sigma_{sg}", []).append(cov_mass(cf | sink_c | swa_c))
            cost.setdefault(f"near_sigma_{sg}", []).append(
                [far_hi_off * 32 + 1024 * 32, float(sf + 1024)])
    del k, q, d, ksub
    torch.cuda.empty_cache()
    return res, cost, sig_imp, cov_direct, noise_stats, sig_norm


def main():
    shard = os.environ.get("E87B_SHARD", "0")
    device = "cuda:0"
    names = sorted(n for n in os.listdir(TRACE)
                   if os.path.isfile(os.path.join(TRACE, n, "meta.json")))
    names = [n for i, n in enumerate(names) if i % 2 == int(shard)]
    results, costs = {}, {}
    all_sig = {"far": [], "near": []}
    all_norm = {"far": [], "near": []}
    all_direct = []
    all_noise = {"sigma_n": [], "sink_gap": []}
    for name in names:
        n_layers = json.load(open(f"{TRACE}/{name}/meta.json"))["n_layers"]
        agg, agg_c = {}, {}
        for li in range(0, n_layers, max(1, n_layers // 12)):
            out = eval_layer(f"{TRACE}/{name}/layer{li:02d}.pt", device)
            if not out:
                continue
            r, c, si, cd, ns, sn = out
            for k2, v in r.items():
                agg.setdefault(k2, []).extend(v)
                for vv in c[k2]:
                    agg_c.setdefault(k2, []).append(vv)
            all_sig["far"].extend(si["far"])
            all_sig["near"].extend(si["near"])
            all_norm["far"].extend(sn["far"])
            all_norm["near"].extend(sn["near"])
            all_direct.extend(cd)
            all_noise["sigma_n"].extend(ns["sigma_n"])
            all_noise["sink_gap"].extend(ns["sink_gap"])
        results[name] = {k2: round(sum(v) / len(v), 4) for k2, v in agg.items()}
        costs[name] = {k2: [round(float(sum(x[0] for x in agg_c[k2]) / len(agg_c[k2])), 1),
                            round(float(sum(x[1] for x in agg_c[k2]) / len(agg_c[k2])), 1)]
                       for k2 in agg_c}
        print(f"[shard{shard} {name}] done", flush=True)
    # 隐含 σ 分布（group-sum 口径；e2e group-mean 口径 = 此值 / G，G=4）
    import numpy as np
    G = 4

    dist = {}
    for side in ["far", "near"]:
        a = np.array(all_norm[side])
        dist[side + "_norm"] = {
            "p10": round(float(np.percentile(a, 10)), 2),
            "p50": round(float(np.percentile(a, 50)), 2),
            "p90": round(float(np.percentile(a, 90)), 2),
            "mean": round(float(a.mean()), 2), "std": round(float(a.std()), 2),
            "cv": round(float(a.std() / abs(a.mean())), 3),
        }
        a = np.array(all_sig[side])
        dist[side] = {
            "p10": round(float(np.percentile(a, 10)), 2),
            "p50": round(float(np.percentile(a, 50)), 2),
            "p90": round(float(np.percentile(a, 90)), 2),
            "mean": round(float(a.mean()), 2),
            "std": round(float(a.std()), 2),
            "e2e_mean_scale_div_G": round(float(a.mean()) / G, 2),
        }
    nd = {}
    for kk in ["sigma_n", "sink_gap"]:
        a = np.array(all_noise[kk])
        nd[kk] = {"p10": round(float(np.percentile(a, 10)), 3),
                  "p50": round(float(np.percentile(a, 50)), 3),
                  "p90": round(float(np.percentile(a, 90)), 3),
                  "mean": round(float(a.mean()), 3), "std": round(float(a.std()), 3)}
    out = {"cov": results, "cost": costs, "sigma_implied": dist, "noise": nd,
           "cov_direct_budget_matched": round(float(np.mean(all_direct)), 4)}
    fname = OUT.replace(".json", f"_s{shard}.json") if "{}" not in OUT else OUT.format(shard)
    json.dump(out, open(fname, "w"), indent=1)
    print(json.dumps(dist, indent=1))
    print("noise:", json.dumps(nd, indent=1))
    print("cov_direct(budget-matched topk):", out["cov_direct_budget_matched"])
    print("saved ->", fname)


if __name__ == "__main__":
    main()
