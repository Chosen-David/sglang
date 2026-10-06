# E64i：同 method 分区 vs 异 method 分区直接对照（用户 2026-09-29 指令：
#   「如果无论怎么改 alpha beta 配比，不同method分区不如同method分区的话，
#     不同method分区的创新点或许可以删除」——需实测判决）
#
# E64g 盲区：三族（mavg/cavg/aavg）枚举的只是 far 侧 method，near 侧恒 aavg
#   → 全部是「异 method 分区」臂，缺 far=near 同 method 分区对照。
# 本实验：near 侧 method 同样枚举 {mavg, aavg, cavg}，
#   far ∈ {mavg, cavg, aavg} × near ∈ {mavg, aavg, cavg} × α=0.125 × β∈{0.25, 0.375}
#   （分区冠军点，E64g 最优内部臂恒 α=0.125；β 0.25=B7 实配 / 0.375=网格冠军）
#   + 单池对照臂（far method 单池，与 E64g α=0 一致）
# 协议与 E64g 完全一致（γ=1、B_TOK=2048、BP=64、真实全维 softmax 行级 mass
#   coverage、TAIL_N 末尾 q、12 层抽样→CPU 跑 8 层与 E64h 一致）。
# 判决规则：同 method 分区显著 > 异 method 分区（同 far 侧）→
#   「不同 method 分区」创新点删除；否则保留（near 侧 method 是自由度）。
import json
import os

import torch
import torch.nn.functional as F

import analyze_e64a_ab_grid as base

TRACE = base.TRACE
OUT = "/home/wangyuanshuo02/two-level-attention/exp/trace/results/e64i_same_method.json"
D2I, BS, SINK, SWA, TAIL_N = base.D2I, base.BS, base.SINK, base.SWA, base.TAIL_N
B_TOK = 2048
BP = 64
GAMMA = 1.0
METHS = ["mavg", "cavg", "aavg"]
PTS = [(0.125, 0.25), (0.125, 0.375)]   # (α, β) 分区冠军点


def eval_layer(lf, device="cpu"):
    d = torch.load(lf, map_location=device, weights_only=False)
    k, q, qpos, S = d["k"].float(), d["q"].float(), d["qpos"], d["S"]
    Hkv, H, D = k.shape[1], q.shape[1], k.shape[-1]
    G = H // Hkv
    pos = torch.arange(S)
    res = {}
    mid_hi_last = int(qpos[-1]) + 1 - SWA
    idx = torch.tensor(D2I)
    ksub_layer = k[..., idx]
    cent_layer, assign_mid_layer, _ = base.greedy_cluster_assign(ksub_layer[SINK:mid_hi_last])
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

        def cov_mass(cand):
            return float((p_full * cand.unsqueeze(1)).sum(-1).mean())

        ksub = ksub_layer
        qsub = q[-TAIL_N + ri][..., idx].reshape(Hkv, G, 32).sum(1)
        nblk = (S + BS - 1) // BS
        kk = F.pad(ksub, (0, 0, 0, 0, 0, nblk * BS - S))
        kc = kk.reshape(nblk, BS, Hkv, 32)
        kmin, kmax, kavg = kc.amin(1), kc.amax(1), kc.mean(1)
        sc_mm = (torch.einsum("hd,nhd->hn", qsub.clamp(min=0), kmax) +
                 torch.einsum("hd,nhd->hn", qsub.clamp(max=0), kmin))
        sc_av = torch.einsum("hd,nhd->hn", qsub, kavg)
        sc_tok = torch.einsum("hgd,shd->hgs", q[-TAIL_N + ri][..., idx].reshape(Hkv, G, 32), ksub).sum(1)
        assign_mid = assign_mid_layer[:, :mid_len]
        cs = torch.einsum("hd,hkd->hk", qsub, cent_layer)
        tok_cl = cs.gather(1, assign_mid)
        blk_ids_mid = (torch.arange(mid_len) // BS).unsqueeze(0).expand(Hkv, -1)
        sc_cl_blk = torch.full((Hkv, (mid_len + BS - 1) // BS), float("-inf"))
        sc_cl_blk.scatter_reduce_(1, blk_ids_mid, tok_cl, reduce="amax", include_self=False)
        sc_blk = {"mavg": sc_mm[:, SINK // BS:], "aavg": sc_av[:, SINK // BS:], "cavg": sc_cl_blk}
        tok_score = {"mavg": sc_tok[:, SINK:mid_hi], "aavg": sc_tok[:, SINK:mid_hi], "cavg": tok_cl}
        sink_c = (pos < SINK).view(1, -1)
        swa_c = ((pos >= t_r + 1 - SWA) & (pos <= t_r)).view(1, -1)

        def select_sub(lo_off, hi_off, method, n_pages, n_tokens):
            s1 = sc_blk[method].masked_fill(
                ~(((torch.arange(sc_blk[method].shape[1]) * BS + SINK) < SINK + hi_off) &
                  ((torch.arange(sc_blk[method].shape[1]) * BS + BS - 1 + SINK) >= SINK + lo_off)).view(1, -1),
                float("-inf"))
            nblk_sub = s1.shape[-1]
            ib = torch.topk(s1, min(n_pages, nblk_sub), dim=-1).indices
            tok = (ib.unsqueeze(-1) * BS + torch.arange(BS).view(1, 1, BS)).reshape(Hkv, -1).clamp(max=mid_len - 1)
            pool = torch.zeros(Hkv, mid_len, dtype=torch.bool)
            pool.scatter_(1, tok, True)
            pool &= ((torch.arange(mid_len) >= lo_off) &
                     (torch.arange(mid_len) < hi_off)).view(1, -1)
            ts = tok_score[method].masked_fill(~pool, float("-inf"))
            it = torch.topk(ts, min(n_tokens, mid_len), dim=-1).indices
            cand = torch.zeros(Hkv, S, dtype=torch.bool)
            cand.scatter_(1, it + SINK, True)
            return cand

        for a, b in PTS:
            near_L = int(a * mid_len)
            near_lo_off, near_hi_off = mid_len - near_L, mid_len
            nb_near = int(round(BP * b))
            nb_far = BP - nb_near
            nt_near = min(int(nb_near * BS * GAMMA), B_TOK)
            nt_far = max(64, B_TOK - nt_near)
            for f_m in METHS:
                for n_m in METHS:
                    cand = torch.zeros(Hkv, S, dtype=torch.bool)
                    cand |= sink_c | swa_c
                    cand |= select_sub(near_lo_off, near_hi_off, n_m, nb_near, nt_near)
                    cand |= select_sub(0, near_lo_off, f_m, nb_far, nt_far)
                    res.setdefault(f"{f_m}+{n_m}_a{a}_b{b}", []).append(cov_mass(cand))
        # 单池对照（far method 单池 = E64g α=0 口径）
        for f_m in METHS:
            cand = torch.zeros(Hkv, S, dtype=torch.bool)
            cand |= sink_c | swa_c
            cand |= select_sub(0, mid_len, f_m, BP, B_TOK)
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
        print(f"[{name}] " + " ".join(f"{k2}={v}" for k2, v in sorted(rec.items())), flush=True)
        json.dump(results, open(OUT, "w"), indent=1)
    keys = sorted({k for rec in results.values() for k in rec})
    avg = {k: round(sum(rec[k] for rec in results.values() if k in rec)
                    / sum(1 for rec in results.values() if k in rec), 4) for k in keys}
    results["AVG"] = avg
    json.dump(results, open(OUT, "w"), indent=1)
    print("\nAVG:")
    print(json.dumps(avg, indent=1))
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
