# E88：分区收益强化——紧预算扫描（用户 2026-10-02「分区本身要有收益」）
# 假设：分区防挤出收益随预算紧度放大。单池挤出带宽 = 预算总量 B_TOK，
#   预算越紧 near 高分挤掉 far 关键 token 越狠，分区隔离价值越大。
# 设计：B_TOK ∈ {256, 512, 768, 1024, 2048} × {mono, 分区} × 16 样本
#   mono   = 单池 mavg（minmax 两级全 mid，top B_TOK 个块 → token）
#   part   = 分区 mavg α.125/β.25（far/near 双池，预算按 β 分）
#   γ = 0.25（e2e 同款等比缩放；near token = nb_near·BS·γ，far = B_TOK − near）
# 口径：e64g 同款真实全维 softmax 行级 mass coverage（可横向比）。
# 判决：Δ = part − mono 随 B_TOK 的曲线——若紧预算端显著正，分区叙事成立。
# 另加分区最优 β oracle（每行扫 β ∈ {0.125..0.875}，看分区上界距离）。
import json
import os

import torch
import torch.nn.functional as F

import analyze_e64a_ab_grid as base

TRACE = base.TRACE
D2I, BS, SINK, SWA, TAIL_N = base.D2I, base.BS, base.SINK, base.SWA, base.TAIL_N
B_TOKS = [256, 512, 768, 1024, 2048]
GAMMA = 0.25
ALPHA = 0.125
BETAS = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
OUT = "/home/wangyuanshuo02/two-level-attention/exp/trace/results/e88_tight_budget_partition_s{}.json"


def eval_layer(lf, device):
    d = torch.load(lf, map_location="cpu", weights_only=False)
    k, q, qpos, S = d["k"].to(device).float(), d["q"].to(device).float(), d["qpos"], d["S"]
    Hkv, H, D = k.shape[1], q.shape[1], k.shape[-1]
    G = H // Hkv
    pos = torch.arange(S, device=device)
    res = {}
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

        # 共享打分（mavg：far/near 都用 minmax 粗筛 + avg 细筛——mono 与分区唯一差异是预算组织）
        idx = torch.tensor(D2I, device=device)
        ksub = k[..., idx]
        qsub = q[-TAIL_N + ri][..., idx].reshape(Hkv, G, 32).sum(1)
        nblk = (S + BS - 1) // BS
        kk = F.pad(ksub, (0, 0, 0, 0, 0, nblk * BS - S))
        kc = kk.reshape(nblk, BS, Hkv, 32)
        kmin, kmax = kc.amin(1), kc.amax(1)
        sc_mm = (torch.einsum("hd,nhd->hn", qsub.clamp(min=0), kmax) +
                 torch.einsum("hd,nhd->hn", qsub.clamp(max=0), kmin))
        sc_tok = torch.einsum("hgd,shd->hgs",
                              q[-TAIL_N + ri][..., idx].reshape(Hkv, G, 32), ksub).sum(1)
        sink_c = (pos < SINK).view(1, -1)
        swa_c = ((pos >= t_r + 1 - SWA) & (pos <= t_r)).view(1, -1)

        nblk_mid = (mid_len + BS - 1) // BS
        near_L = int(ALPHA * mid_len)

        # 两级选择子（块 minmax topk → token 细筛 topk，区域裁剪；mono/分区共用）
        def select_sub(lo_off, hi_off, n_pages, n_tokens):
            s1 = sc_mm[:, SINK // BS: SINK // BS + nblk_mid]
            blk_lo, blk_hi = lo_off // BS, (hi_off + BS - 1) // BS
            m = ((torch.arange(nblk_mid, device=device) >= blk_lo) &
                 (torch.arange(nblk_mid, device=device) < blk_hi)).view(1, -1)
            s1 = s1.masked_fill(~m, float("-inf"))
            n_pages = min(n_pages, int(m.sum().item() * BS) // BS + 1, s1.shape[-1])
            ib = torch.topk(s1, n_pages, dim=-1).indices
            tok = (ib.unsqueeze(-1) * BS + torch.arange(BS, device=device).view(1, 1, BS)).reshape(Hkv, -1)
            tok = tok.clamp(max=mid_len - 1)
            pool = torch.zeros(Hkv, mid_len, dtype=torch.bool, device=device)
            pool.scatter_(1, tok, True)
            pool &= ((torch.arange(mid_len, device=device) >= lo_off) &
                     (torch.arange(mid_len, device=device) < hi_off)).view(1, -1)
            ts = sc_tok[:, SINK:SINK + mid_len].masked_fill(~pool, float("-inf"))
            n_tokens = min(n_tokens, mid_len)
            it = torch.topk(ts, n_tokens, dim=-1).indices
            cand = torch.zeros(Hkv, S, dtype=torch.bool, device=device)
            cand.scatter_(1, it + SINK, True)
            return cand

        for B_TOK in B_TOKS:
            # mono：单池全 mid，bp 块预算 = B_TOK/BS·(1/γ 折算到块) —— 与 e2e C0 同构：
            # 块预算按 token 预算/γ 放大（细筛有 γ 折扣），封顶 nblk_mid
            bp = min(int(B_TOK / (BS * GAMMA)), nblk_mid)
            nt = min(B_TOK, mid_len)
            c_m = select_sub(0, mid_len, bp, nt)
            res.setdefault(f"mono_b{B_TOK}", []).append(cov_mass(c_m | sink_c | swa_c))

            # 分区（部署臂 β=0.25）+ β oracle
            best_oracle = -1.0
            for beta in BETAS:
                nb_near = int(round(B_TOK / (BS * GAMMA) * beta))
                nb_near = min(nb_near, max(1, near_L // BS))
                nb_far = max(1, min(int(B_TOK / (BS * GAMMA)) - nb_near, (mid_len - near_L) // BS))
                nt_near = min(int(nb_near * BS * GAMMA), B_TOK - 64, near_L)
                nt_near = max(nt_near, 1)
                nt_far = max(64, B_TOK - nt_near)
                c_f = select_sub(0, mid_len - near_L, nb_far, nt_far)
                c_n = select_sub(mid_len - near_L, mid_len, nb_near, nt_near)
                v = cov_mass(c_f | c_n | sink_c | swa_c)
                if beta == 0.25:
                    res.setdefault(f"part_b{B_TOK}", []).append(v)
                best_oracle = max(best_oracle, v)
            res.setdefault(f"part_oracle_b{B_TOK}", []).append(best_oracle)
    del k, q, d, ksub
    torch.cuda.empty_cache()
    return res


def main():
    shard = os.environ.get("E88_SHARD", "0")
    device = "cpu" if os.environ.get("E88_CPU") else "cuda:0"
    names = sorted(n for n in os.listdir(TRACE)
                   if os.path.isfile(os.path.join(TRACE, n, "meta.json")))
    names = [n for i, n in enumerate(names) if i % 2 == int(shard)]
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
        results[name] = {k2: round(sum(v) / len(v), 4) for k2, v in agg.items()}
        print(f"[shard{shard} {name}] done", flush=True)
    json.dump(results, open(OUT.format(shard), "w"), indent=1)
    print("saved ->", OUT.format(shard))


if __name__ == "__main__":
    main()
