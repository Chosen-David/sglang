# E64d：cluster 完整形态进网格——hybrid = minmax 块粗筛 + 池内贪心簇分 token 细筛（E4d 结论 3 + §8b-36 结论 3 的待办消融）
# E64a 的 cavg 垫底是块级 scatter-amax 粗筛形态；E4d 证明贪心簇分的优势形态是 token 级。
# 本实验把 cluster 塞进两级框架的正确位置：粗筛用 minmax 上界（保住漏选理论性质），细筛用贪心簇分。
# 臂（冠军点 α=0.125 β=0.25，bp ∈ {64,128,256}，gamma=1）：
#   mavg_bp{N}   ：minmax 粗筛 + 精确子空间 token 分细筛（E64a 冠军对照）
#   hybrid_bp{N} ：minmax 粗筛 + 池内贪心簇分细筛（本实验主臂）
#   mono_bp{N}   ：单池对照
# 口径：真实全维 softmax 行级 mass coverage；16 样本×12 层；贪心 build 层级共享。
import json
import os

import torch
import torch.nn.functional as F

import analyze_e64a_ab_grid as base

TRACE = base.TRACE
OUT = "/home/wangyuanshuo02/two-level-attention/exp/trace/results/e64d_hybrid.json"
D2I, BS, SINK, SWA, TAIL_N = base.D2I, base.BS, base.SINK, base.SWA, base.TAIL_N
B_TOK = 2048
BPS = [64, 128, 256]
ALPHA, BETA, GAMMA = 0.125, 0.25, 1.0


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
        cs = torch.einsum("hd,hkd->hk", qsub, cent_layer)     # 簇分
        tok_cl = cs.gather(1, assign_mid)                       # token 簇分（贪心簇分的 token 级形态）
        sc_mm_mid = sc_mm[:, SINK // BS:]
        sc_av_mid = sc_av[:, SINK // BS:]
        tok_exact = sc_tok[:, SINK:mid_hi]                      # 精确子空间 token 分
        sink_c = (pos < SINK).view(1, -1)
        swa_c = ((pos >= t_r + 1 - SWA) & (pos <= t_r)).view(1, -1)

        def select_sub(lo_off, hi_off, sc_blk, tok_score, n_pages, n_tokens):
            """通用区选择：块分粗筛页池 → 池内 token 分细筛。sc_blk [Hkv,nblk_mid]，tok_score [Hkv,mid_len]。"""
            nblk_mid = sc_blk.shape[1]
            rng = torch.arange(nblk_mid, device=device)
            s1 = sc_blk.masked_fill(~(((rng * BS) < hi_off) & ((rng * BS + BS - 1) >= lo_off)).view(1, -1), float("-inf"))
            ib = torch.topk(s1, min(n_pages, nblk_mid), dim=-1).indices
            tok = (ib.unsqueeze(-1) * BS + torch.arange(BS, device=device).view(1, 1, BS)).reshape(Hkv, -1).clamp(max=mid_len - 1)
            pool = torch.zeros(Hkv, mid_len, dtype=torch.bool, device=device)
            pool.scatter_(1, tok, True)
            pool &= ((torch.arange(mid_len, device=device) >= lo_off) &
                     (torch.arange(mid_len, device=device) < hi_off)).view(1, -1)
            ts = tok_score.masked_fill(~pool, float("-inf"))
            it = torch.topk(ts, min(n_tokens, mid_len), dim=-1).indices
            cand = torch.zeros(Hkv, S, dtype=torch.bool, device=device)
            cand.scatter_(1, it + SINK, True)
            return cand

        near_L = int(ALPHA * mid_len)
        near_lo_off, near_hi_off = mid_len - near_L, mid_len
        for bp in BPS:
            nb_near = max(1, int(round(bp * BETA)))
            nb_far = max(1, bp - nb_near)
            nt_near = int(nb_near * BS * GAMMA)
            nt_far = max(64, B_TOK - nt_near)
            for meth, tok_sc in (("mavg", tok_exact), ("hybrid", tok_cl)):
                cand = torch.zeros(Hkv, S, dtype=torch.bool, device=device)
                cand |= sink_c | swa_c
                cand |= select_sub(near_lo_off, near_hi_off, sc_av_mid, tok_exact, nb_near, nt_near)  # near 固定 avg+精确
                cand |= select_sub(0, near_lo_off, sc_mm_mid, tok_sc, nb_far, nt_far)                 # far 各臂
                res.setdefault(f"{meth}_bp{bp}", []).append(cov_mass(cand))
            cand = torch.zeros(Hkv, S, dtype=torch.bool, device=device)
            cand |= sink_c | swa_c
            cand |= select_sub(0, mid_len, sc_mm_mid, tok_exact, bp, B_TOK)  # mono 同 bp
            res.setdefault(f"mono_bp{bp}", []).append(cov_mass(cand))
    del k, q, d, ksub
    torch.cuda.empty_cache()
    return res


def main():
    device = "cuda:" + os.environ.get("E64D_GPU", "0")
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
