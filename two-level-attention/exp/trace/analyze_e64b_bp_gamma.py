# E64b：冠军配置 bp × gamma 细扫（用户指令「bp 多组补充扫描」+「gamma 若不用需说明原因」）
# 固定 method ∈ {mavg, cavg}（E64a 中期冠军法）、(alpha, beta) = (0.125, 0.25)（E64a 冠军点，全量后可用环境变量覆盖）；
# 扫描 bp ∈ {16,32,64,128,256} × gamma ∈ {0.5,0.75,1.0} + mono 同 bp 对照（gamma 单池无意义，bp 对照臂）。
# far token 预算 = B_TOK − near_bp·64·gamma（用户公式）；near 页池 = bp·beta、far 页池 = bp·(1-beta)。
# 防爆炸：贪心 build 层级共享（E64a 提速版同款）。
# 口径：真实全维 softmax 行级 mass coverage（sink+swa+mid 全覆盖）；16 样本×12 层。
import json
import os

import torch
import torch.nn.functional as F

import analyze_e64a_ab_grid as base   # 复用 greedy_cluster_assign/共享打分量

TRACE = base.TRACE
OUT = "/home/wangyuanshuo02/two-level-attention/exp/trace/results/e64b_bp_gamma.json"
D2I, BS, SINK, SWA, TAIL_N = base.D2I, base.BS, base.SINK, base.SWA, base.TAIL_N
B_TOK = 2048
BPS = [16, 32, 64, 128, 256]
GAMMAS = [0.5, 0.75, 1.0]
ALPHA = float(os.environ.get("E64B_ALPHA", "0.125"))
BETA = float(os.environ.get("E64B_BETA", "0.25"))
METHODS = ["mavg", "cavg"]


def eval_layer(lf, device):
    d = torch.load(lf, map_location="cpu", weights_only=False)
    k, q, qpos, S = d["k"].to(device).float(), d["q"].to(device).float(), d["qpos"], d["S"]
    Hkv, H, D = k.shape[1], q.shape[1], k.shape[-1]
    G = H // Hkv
    pos = torch.arange(S, device=device)
    res = {}
    mid_hi_last = int(qpos[-1]) + 1 - SWA
    idx = torch.tensor(D2I, device=device)
    ksub_layer = k[..., idx]
    cent_layer, assign_mid_layer, k_live_layer = base.greedy_cluster_assign(ksub_layer[SINK:mid_hi_last])
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
        blk_ids_mid = (torch.arange(mid_len, device=device) // BS).unsqueeze(0).expand(Hkv, -1)
        sc_cl_blk = torch.full((Hkv, (mid_len + BS - 1) // BS), float("-inf"), device=device)
        sc_cl_blk.scatter_reduce_(1, blk_ids_mid, tok_cl, reduce="amax", include_self=False)
        sc_blk = {"mavg": sc_mm[:, SINK // BS:], "cavg": sc_cl_blk, "aavg": sc_av[:, SINK // BS:]}
        tok_score = {"mavg": sc_tok[:, SINK:mid_hi], "cavg": tok_cl, "aavg": sc_tok[:, SINK:mid_hi]}
        sink_c = (pos < SINK).view(1, -1)
        swa_c = ((pos >= t_r + 1 - SWA) & (pos <= t_r)).view(1, -1)

        def select_sub(lo_off, hi_off, method, n_pages, n_tokens):
            s1 = sc_blk[method].masked_fill(
                ~(((torch.arange(sc_blk[method].shape[1], device=device) * BS + SINK) < SINK + hi_off) &
                  ((torch.arange(sc_blk[method].shape[1], device=device) * BS + BS - 1 + SINK) >= SINK + lo_off)).view(1, -1),
                float("-inf"))
            nblk_sub = s1.shape[-1]
            ib = torch.topk(s1, min(n_pages, nblk_sub), dim=-1).indices
            tok = (ib.unsqueeze(-1) * BS + torch.arange(BS, device=device).view(1, 1, BS)).reshape(Hkv, -1).clamp(max=mid_len - 1)
            pool = torch.zeros(Hkv, mid_len, dtype=torch.bool, device=device)
            pool.scatter_(1, tok, True)
            pool &= ((torch.arange(mid_len, device=device) >= lo_off) &
                     (torch.arange(mid_len, device=device) < hi_off)).view(1, -1)
            ts = tok_score[method].masked_fill(~pool, float("-inf"))
            it = torch.topk(ts, min(n_tokens, mid_len), dim=-1).indices
            cand = torch.zeros(Hkv, S, dtype=torch.bool, device=device)
            cand.scatter_(1, it + SINK, True)
            return cand

        near_L = int(ALPHA * mid_len)
        near_lo_off, near_hi_off = mid_len - near_L, mid_len
        for meth in METHODS:
            for bp in BPS:
                nb_near = max(1, int(round(bp * BETA)))
                nb_far = max(1, bp - nb_near)
                for gam in GAMMAS:
                    nt_near = int(nb_near * BS * gam)
                    nt_far = max(64, B_TOK - nt_near)
                    cand = torch.zeros(Hkv, S, dtype=torch.bool, device=device)
                    cand |= sink_c | swa_c
                    cand |= select_sub(near_lo_off, near_hi_off, "aavg", nb_near, nt_near)
                    cand |= select_sub(0, near_lo_off, meth, nb_far, nt_far)
                    res.setdefault(f"{meth}_bp{bp}_g{gam}", []).append(cov_mass(cand))
        for bp in BPS:  # mono 同 bp 对照（单池 minmax 全 mid）
            cand = torch.zeros(Hkv, S, dtype=torch.bool, device=device)
            cand |= sink_c | swa_c
            sc_m = {"mavg": sc_mm[:, SINK // BS:]}
            # mono：单池直接内联（select_sub 但 method 池全 mid）
            s1 = sc_mm[:, SINK // BS:].masked_fill(
                ~(((torch.arange(sc_mm[:, SINK // BS:].shape[1], device=device) * BS + SINK) < SINK + mid_len) &
                  ((torch.arange(sc_mm[:, SINK // BS:].shape[1], device=device) * BS + BS - 1 + SINK) >= SINK)).view(1, -1),
                float("-inf"))
            ib = torch.topk(s1, min(bp, s1.shape[-1]), dim=-1).indices
            tok = (ib.unsqueeze(-1) * BS + torch.arange(BS, device=device).view(1, 1, BS)).reshape(Hkv, -1).clamp(max=mid_len - 1)
            pool = torch.zeros(Hkv, mid_len, dtype=torch.bool, device=device)
            pool.scatter_(1, tok, True)
            ts = sc_tok[:, SINK:mid_hi].masked_fill(~pool, float("-inf"))
            it = torch.topk(ts, min(B_TOK, mid_len), dim=-1).indices
            cand.scatter_(1, it + SINK, True)
            res.setdefault(f"mono_bp{bp}", []).append(cov_mass(cand))
    del k, q, d, ksub
    torch.cuda.empty_cache()
    return res


def main():
    device = "cuda:" + os.environ.get("E64B_GPU", "1")
    names = sorted(n for n in os.listdir(TRACE)
                   if os.path.isfile(os.path.join(TRACE, n, "meta.json")))
    results = {}
    for name in names:
        n_layers = json.load(open(f"{TRACE}/{name}/meta.json"))["n_layers"]
        agg = {}
        for li in range(0, n_layers, max(1, n_layers // 12)):
            r = eval_layer(f"{TRACE}/{name}/layer{li:02d}.pt", device)
            if not r:
                continue
            for k2, v in r.items():
                agg.setdefault(k2, []).extend(v)
        rec = {k2: round(sum(v) / len(v), 4) for k2, v in agg.items()}
        results[name] = rec
        print(f"[{name}] " + " ".join(f"{k2}={v}" for k2, v in sorted(rec.items())), flush=True)
    json.dump(results, open(OUT, "w"), indent=1)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
