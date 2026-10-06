# E64j：五种 method 组合各自最佳配置对比（用户 2026-09-29 定义命名：
#   mminmax=(minmax,minmax) / mavg=(minmax,avg) / ccluster=(cluster,cluster) /
#   cavg=(cluster,avg) / aavg=(avg,avg)）
# 用户指令：探索每个组合的最佳 (α,β) 配置后，在各自最佳配置上对比组合得分。
# 背景：E64i 只测了 α=0.125 单点；E4d 证明纯贪心簇 token 级 far 召回 16/16
#   胜 minmax（@512 +0.28）——cluster 强在 far 召回口径，需在组合框架下
#   复验「每组合调到各自最优后谁最高」。
# 设计：5 组合 × α∈{0.125,0.25,0.375} × β∈{0.25,0.375,0.5}（9 格点）
#   + far 单池对照（α=0，组合间共享）；γ=1、B_TOK=2048、BP=64 协议同 E64g/i。
# 覆盖：复用 e64i 的 eval_layer 全部计算骨架，仅参数化 PAIRS 与 PTS。
import json
import os

import torch

import analyze_e64i_same_method as base

TRACE = base.TRACE
OUT = "/home/wangyuanshuo02/two-level-attention/exp/trace/results/e64j_combo_best.json"
METHS = ["mavg", "cavg", "aavg"]
PAIRS = [("mavg", "mavg"), ("mavg", "aavg"), ("cavg", "cavg"),
         ("cavg", "aavg"), ("aavg", "aavg")]          # 用户五种组合
PTS = [(a, b) for a in (0.125, 0.25, 0.375) for b in (0.25, 0.375, 0.5)]


def eval_layer(lf):
    """复用 base.eval_layer 的张量计算，仅替换臂循环（PAIRS×PTS + mono）。"""
    import torch.nn.functional as F
    import analyze_e64a_ab_grid as g0
    d = torch.load(lf, map_location="cpu", weights_only=False)
    k, q, qpos, S = d["k"].float(), d["q"].float(), d["qpos"], d["S"]
    Hkv, H, D = k.shape[1], q.shape[1], k.shape[-1]
    G = H // Hkv
    pos = torch.arange(S)
    res = {}
    mid_hi_last = int(qpos[-1]) + 1 - base.SWA
    idx = torch.tensor(base.D2I)
    ksub_layer = k[..., idx]
    cent_layer, assign_mid_layer, _ = g0.greedy_cluster_assign(ksub_layer[base.SINK:mid_hi_last])
    for ri in range(base.TAIL_N):
        t_r = int(qpos[-base.TAIL_N + ri])
        mid_hi = t_r + 1 - base.SWA
        mid_len = mid_hi - base.SINK
        if mid_len < 4096:
            continue
        qg = q[-base.TAIL_N + ri].reshape(Hkv, G, D)
        k4 = k[:, :, None, :].expand(S, Hkv, G, D)
        s_full = torch.einsum("hgd,shgd->hgs", qg, k4) * (D ** -0.5)
        s_full = s_full.masked_fill((pos > t_r).view(1, 1, -1), float("-inf"))
        p_full = torch.softmax(s_full, dim=-1)

        def cov_mass(cand):
            return float((p_full * cand.unsqueeze(1)).sum(-1).mean())

        ksub = ksub_layer
        qsub = q[-base.TAIL_N + ri][..., idx].reshape(Hkv, G, 32).sum(1)
        nblk = (S + base.BS - 1) // base.BS
        kk = F.pad(ksub, (0, 0, 0, 0, 0, nblk * base.BS - S))
        kc = kk.reshape(nblk, base.BS, Hkv, 32)
        kmin, kmax, kavg = kc.amin(1), kc.amax(1), kc.mean(1)
        sc_mm = (torch.einsum("hd,nhd->hn", qsub.clamp(min=0), kmax) +
                 torch.einsum("hd,nhd->hn", qsub.clamp(max=0), kmin))
        sc_av = torch.einsum("hd,nhd->hn", qsub, kavg)
        sc_tok = torch.einsum("hgd,shd->hgs", q[-base.TAIL_N + ri][..., idx].reshape(Hkv, G, 32), ksub).sum(1)
        assign_mid = assign_mid_layer[:, :mid_len]
        cs = torch.einsum("hd,hkd->hk", qsub, cent_layer)
        tok_cl = cs.gather(1, assign_mid)
        blk_ids_mid = (torch.arange(mid_len) // base.BS).unsqueeze(0).expand(Hkv, -1)
        sc_cl_blk = torch.full((Hkv, (mid_len + base.BS - 1) // base.BS), float("-inf"))
        sc_cl_blk.scatter_reduce_(1, blk_ids_mid, tok_cl, reduce="amax", include_self=False)
        sc_blk = {"mavg": sc_mm[:, base.SINK // base.BS:], "aavg": sc_av[:, base.SINK // base.BS:], "cavg": sc_cl_blk}
        tok_score = {"mavg": sc_tok[:, base.SINK:mid_hi], "aavg": sc_tok[:, base.SINK:mid_hi], "cavg": tok_cl}
        sink_c = (pos < base.SINK).view(1, -1)
        swa_c = ((pos >= t_r + 1 - base.SWA) & (pos <= t_r)).view(1, -1)

        def select_sub(lo_off, hi_off, method, n_pages, n_tokens):
            s1 = sc_blk[method].masked_fill(
                ~(((torch.arange(sc_blk[method].shape[1]) * base.BS + base.SINK) < base.SINK + hi_off) &
                  ((torch.arange(sc_blk[method].shape[1]) * base.BS + base.BS - 1 + base.SINK) >= base.SINK + lo_off)).view(1, -1),
                float("-inf"))
            nblk_sub = s1.shape[-1]
            ib = torch.topk(s1, min(n_pages, nblk_sub), dim=-1).indices
            tok = (ib.unsqueeze(-1) * base.BS + torch.arange(base.BS).view(1, 1, base.BS)).reshape(Hkv, -1).clamp(max=mid_len - 1)
            pool = torch.zeros(Hkv, mid_len, dtype=torch.bool)
            pool.scatter_(1, tok, True)
            pool &= ((torch.arange(mid_len) >= lo_off) &
                     (torch.arange(mid_len) < hi_off)).view(1, -1)
            ts = tok_score[method].masked_fill(~pool, float("-inf"))
            it = torch.topk(ts, min(n_tokens, mid_len), dim=-1).indices
            cand = torch.zeros(Hkv, S, dtype=torch.bool)
            cand.scatter_(1, it + base.SINK, True)
            return cand

        for a, b in PTS:
            near_L = int(a * mid_len)
            near_lo_off, near_hi_off = mid_len - near_L, mid_len
            nb_near = int(round(base.BP * b))
            nb_far = base.BP - nb_near
            nt_near = min(int(nb_near * base.BS * base.GAMMA), base.B_TOK)
            nt_far = max(64, base.B_TOK - nt_near)
            for f_m, n_m in PAIRS:
                cand = torch.zeros(Hkv, S, dtype=torch.bool)
                cand |= sink_c | swa_c
                cand |= select_sub(near_lo_off, near_hi_off, n_m, nb_near, nt_near)
                cand |= select_sub(0, near_lo_off, f_m, nb_far, nt_far)
                res.setdefault(f"{f_m}+{n_m}_a{a}_b{b}", []).append(cov_mass(cand))
        for f_m in ("mavg", "cavg", "aavg"):
            cand = torch.zeros(Hkv, S, dtype=torch.bool)
            cand |= sink_c | swa_c
            cand |= select_sub(0, mid_len, f_m, base.BP, base.B_TOK)
            res.setdefault(f"{f_m}_mono", []).append(cov_mass(cand))
    return res


def main():
    torch.set_num_threads(10)
    names = sorted(n for n in os.listdir(TRACE)
                   if os.path.isfile(os.path.join(TRACE, n, "meta.json")))
    results = {}
    for name in names:
        n_layers = json.load(open(f"{TRACE}/{name}/meta.json"))["n_layers"]
        agg = {}
        for li in range(0, n_layers, max(1, n_layers // 8)):
            r = eval_layer(f"{TRACE}/{name}/layer{li:02d}.pt")
            if not r:
                continue
            for k2, v in r.items():
                agg.setdefault(k2, []).extend(v)
        rec = {k2: round(sum(v) / len(v), 4) for k2, v in agg.items()}
        results[name] = rec
        print(f"[{name}] done {len(rec)} arms", flush=True)
        json.dump(results, open(OUT, "w"), indent=1)
    keys = sorted({k for rec in results.values() for k in rec})
    avg = {k: round(sum(rec[k] for rec in results.values() if k in rec)
                    / sum(1 for rec in results.values() if k in rec), 4) for k in keys}
    results["AVG"] = avg
    json.dump(results, open(OUT, "w"), indent=1)
    # 每组合最佳配置对比表
    print("\n== 五组合各自最佳配置（16 样本 AVG）==")
    best = {}
    for f_m, n_m in PAIRS:
        cands = {k: v for k, v in avg.items() if k.startswith(f"{f_m}+{n_m}_")}
        bk, bv = max(cands.items(), key=lambda x: x[1])
        best[(f_m, n_m)] = (bk, bv)
        print(f"  {f_m}+{n_m:5s} best={bk:32s} {bv:.4f}")
    for f_m in ("mavg", "cavg", "aavg"):
        print(f"  {f_m}_mono（α=0 单池）                          {avg[f'{f_m}_mono']:.4f}")
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
