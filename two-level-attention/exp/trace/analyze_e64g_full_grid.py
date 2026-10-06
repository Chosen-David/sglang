# E64g：α×β 全组合网格（用户指令 2026-09-28：步长 0.125 从 0 到 1，含边界角点）
# 角点语义（用户定义）：
#   ab(1,1) = 全部 near_method（near 区=全 mid + near 拿全部页池/预算）
#   ab(0,0) = 全部 far_method（far 区=全 mid + far 拿全部页池/预算）
#   ab(1,0) = 无意义（near 负责全部但没有选择权利）→ 数据上仅剩 sink+swa，预期低 cov
#   ab(0,1) = 无意义（far 负责全部但没有选择权利）→ 同上
# 实现要点：α=0 或 β=0 → 跳过 near 选择（近区/近预算为空）；α=1 或 β=1 → 跳过 far 选择。
# method ∈ {mavg, cavg, aavg}（far 侧；near 侧恒 avg+精确细筛，(1,1) 角点即全 near_method 形态）。
# bp=64、gamma=1、B_TOK=2048（E64a 同协议连续性）；分片跑（E64G_SHARD=0/1 各半样本，双 GPU）。
# 口径：真实全维 softmax 行级 mass coverage；输出每样本每臂 cov（用户要的「不同数据集下的精度数据」）。
import json
import os

import torch
import torch.nn.functional as F

import analyze_e64a_ab_grid as base

TRACE = base.TRACE
OUT = "/home/wangyuanshuo02/two-level-attention/exp/trace/results/e64g_full_grid_s{}.json"
D2I, BS, SINK, SWA, TAIL_N = base.D2I, base.BS, base.SINK, base.SWA, base.TAIL_N
B_TOK = 2048
BP = 64
GAMMA = 1.0
GRID = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
METHODS = ["mavg", "cavg", "aavg"]


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
        sc_blk = {"mavg": sc_mm[:, SINK // BS:], "aavg": sc_av[:, SINK // BS:], "cavg": sc_cl_blk}
        tok_score = {"mavg": sc_tok[:, SINK:mid_hi], "aavg": sc_tok[:, SINK:mid_hi], "cavg": tok_cl}
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

        for meth in METHODS:
            for a in GRID:
                near_L = int(a * mid_len)
                near_lo_off, near_hi_off = mid_len - near_L, mid_len
                for b in GRID:
                    # 角点守卫（用户语义）：α=0/β=0 → near 空；α=1/β=1 → far 空
                    do_near = (a > 0) and (b > 0)
                    do_far = (a < 1) and (b < 1)
                    nb_near = int(round(BP * b)) if do_near else 0
                    nb_far = BP - nb_near if do_far else 0
                    nt_near = min(int(nb_near * BS * GAMMA), B_TOK) if do_near else 0
                    nt_far = max(64, B_TOK - nt_near) if do_far else 0
                    cand = torch.zeros(Hkv, S, dtype=torch.bool, device=device)
                    cand |= sink_c | swa_c
                    if do_near:
                        cand |= select_sub(near_lo_off, near_hi_off, "aavg", nb_near, nt_near)
                    if do_far:
                        cand |= select_sub(0, near_lo_off, meth, nb_far, nt_far)
                    res.setdefault(f"{meth}_a{a}_b{b}", []).append(cov_mass(cand))
    del k, q, d, ksub
    torch.cuda.empty_cache()
    return res


def main():
    shard = os.environ.get("E64G_SHARD", "0")
    device = "cuda:0"
    names = sorted(n for n in os.listdir(TRACE)
                   if os.path.isfile(os.path.join(TRACE, n, "meta.json")))
    names = [n for i, n in enumerate(names) if i % 2 == int(shard)]   # 双 GPU 分片
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
        print(f"[shard{shard} {name}] done {len(rec)} arms", flush=True)
    json.dump(results, open(OUT.format(shard), "w"), indent=1)
    print("saved ->", OUT.format(shard))


if __name__ == "__main__":
    main()
