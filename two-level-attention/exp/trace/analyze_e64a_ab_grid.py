# E64a：method × alpha × beta 网格消融（用户命名：cavg=(cluster,avg) mavg=(minmax,avg) aavg=(avg,avg)）
# 阶段1：固定 bp=64（页池总量）、gamma=1（ab 模式默认）、budget_token=2048。
#   alpha ∈ {0.125,0.25,0.5,0.75}：near 子区长度 = alpha·mid_L（紧邻 swa 的 mid 段）；
#   beta ∈ {0.125,0.25,0.5,0.75}：near 页池 = bp·beta，far 页池 = bp·(1-beta)；
#   near token 预算 = near_bp·64·gamma，far token 预算 = budget_token − near token（用户公式）。
# 三 method 臂的 far 粗筛/细筛：
#   cavg：far 粗筛 = 贪心簇分 scatter-amax 页分；far 细筛 = 池内 token 簇分 topk
#   mavg：far 粗筛 = minmax 块上界；far 细筛 = 池内 token 子空间精确分 topk（TIA 式）
#   aavg：far 粗筛 = avg 块均值；far 细筛 = 池内 token 子空间精确分 topk
#   near（三 method 同）：avg 块粗筛 → 池内 token 子空间分 topk（gamma 折扣）
# 对照：mono（alpha=0 全 mid 单池 minmax）；tli_anchor（mid 近端免打分 1024+far minmax 单池）。
# 防爆炸：每层共享量算一次（minmax/avg 块分 + 贪心聚类 build + token 子空间分），48 臂仅区域切分+topk。
# 口径：真实全维 softmax 行级 mass coverage（sink+swa+mid 全覆盖）；16 样本×12 层，swa=1024、sink=128。
# 输出：e64a_ab_grid.json（每样本每臂 cov）。
import json
import os
import time

import torch
import torch.nn.functional as F

TRACE = "/tmp/trace/qwen3-8b"
OUT = "/home/wangyuanshuo02/two-level-attention/exp/trace/results/e64a_ab_grid.json"
D2I = list(range(48, 64)) + list(range(112, 128))
BS = 64
SINK = 128
SWA = 1024
TAIL_N = 4
BP = 64            # 页池总量 budget_page_topk
B_TOK = 2048       # 细筛 token 总预算（用户示例值）
GAMMA = 1.0
ALPHAS = [0.125, 0.25, 0.5, 0.75]
BETAS = [0.125, 0.25, 0.5, 0.75]
SIM = 0.9
K_MAX = 4096
METHODS = ["cavg", "mavg", "aavg"]


def greedy_cluster_assign(x, sim=SIM, k_max=K_MAX):
    """增量贪心聚类（时序逐 token、余弦阈值、算术均值簇心）。返回簇分/assign。"""
    T, Hkv, dd = x.shape
    dev = x.device
    sums = torch.zeros(Hkv, k_max, dd, device=dev)
    cnt = torch.zeros(Hkv, k_max, device=dev)
    k_live = torch.zeros(Hkv, dtype=torch.long, device=dev)
    assign = torch.zeros(Hkv, T, dtype=torch.long, device=dev)
    x_n = x.norm(dim=-1)
    ar = torch.arange(Hkv, device=dev)
    ones_hk = torch.ones(Hkv, device=dev)
    for i in range(T):
        xi = x[i]
        norms = sums.norm(dim=-1)
        cos = torch.einsum("hkd,hd->hk", sums, xi) / (norms * x_n[i].unsqueeze(-1) + 1e-9)
        live_mask = torch.arange(k_max, device=dev).unsqueeze(0) >= k_live.unsqueeze(-1)
        cos = cos.masked_fill(live_mask, float("-inf"))
        a = cos.argmax(-1)
        m = cos.gather(1, a.unsqueeze(-1)).squeeze(-1)
        upd = m >= sim
        flat = ar * k_max + a
        sums.view(-1, dd).index_add_(0, flat[upd], xi[upd])
        cnt.view(-1).index_add_(0, flat[upd], ones_hk[upd])
        a_final = a.clone()
        for h in torch.nonzero(~upd).flatten().tolist():
            k = int(k_live[h])
            if k < k_max:
                sums[h, k] = xi[h]; cnt[h, k] = 1; k_live[h] += 1; a_final[h] = k
            else:
                sums[h, a[h]] += xi[h]; cnt[h, a[h]] += 1; a_final[h] = a[h]
        assign[:, i] = a_final
    cent = sums / cnt.clamp(min=1).unsqueeze(-1)     # 算术均值簇心 [Hkv,K,dd]
    return cent, assign, k_live


def eval_layer(lf, device):
    d = torch.load(lf, map_location="cpu", weights_only=False)
    k, q, qpos, S = d["k"].to(device).float(), d["q"].to(device).float(), d["qpos"], d["S"]
    Hkv, H, D = k.shape[1], q.shape[1], k.shape[-1]
    G = H // Hkv
    pos = torch.arange(S, device=device)
    res = {}
    # ---- 层级共享：贪心聚类 build（只依赖 token，用末位 query 的 mid 界；行间 mid_hi 仅差 ≤4 token）----
    mid_hi_last = int(qpos[-1]) + 1 - SWA
    mid_len = mid_hi_last - SINK
    idx = torch.tensor(D2I, device=device)
    ksub_layer = k[..., idx]
    t0 = time.time()
    cent_layer, assign_mid_layer, k_live_layer = greedy_cluster_assign(ksub_layer[SINK:mid_hi_last])
    t_build_layer = time.time() - t0
    for ri in range(TAIL_N):
        t_r = int(qpos[-TAIL_N + ri])
        mid_hi = t_r + 1 - SWA
        mid_len = mid_hi - SINK
        if mid_len < 4096:
            continue
        # 真值分布
        qg = q[-TAIL_N + ri].reshape(Hkv, G, D)
        k4 = k[:, :, None, :].expand(S, Hkv, G, D)
        s_full = torch.einsum("hgd,shgd->hgs", qg, k4) * (D ** -0.5)
        s_full = s_full.masked_fill((pos > t_r).view(1, 1, -1), float("-inf"))
        p_full = torch.softmax(s_full, dim=-1)

        def cov_mass(cand):
            return float((p_full * cand.unsqueeze(1)).sum(-1).mean())

        # ---- 共享打分量（全 mid 区一次）----
        idx = torch.tensor(D2I, device=device)
        ksub = k[..., idx]                           # [S,Hkv,32]
        qsub = q[-TAIL_N + ri][..., idx].reshape(Hkv, G, 32).sum(1)   # [Hkv,32]
        nblk = (S + BS - 1) // BS
        kk = F.pad(ksub, (0, 0, 0, 0, 0, nblk * BS - S))
        kc = kk.reshape(nblk, BS, Hkv, 32)
        kmin, kmax, kavg = kc.amin(1), kc.amax(1), kc.mean(1)
        sc_mm = (torch.einsum("hd,nhd->hn", qsub.clamp(min=0), kmax) +
                 torch.einsum("hd,nhd->hn", qsub.clamp(max=0), kmin))     # [Hkv,nblk]
        sc_av = torch.einsum("hd,nhd->hn", qsub, kavg)
        sc_tok = torch.einsum("hgd,shd->hgs", q[-TAIL_N + ri][..., idx].reshape(Hkv, G, 32), ksub).sum(1)  # [Hkv,S] token 子空间分
        # 贪心聚类：复用层级 build（token 侧与 query 无关）；簇分 = qsub·簇心
        cent, k_live = cent_layer, k_live_layer
        t_build = t_build_layer
        assign_mid = assign_mid_layer[:, :mid_len]
        cs = torch.einsum("hd,hkd->hk", qsub, cent)               # [Hkv,K] 簇分
        tok_cl = cs.gather(1, assign_mid)                          # [Hkv, mid_len] token 簇分
        blk_ids_mid = (torch.arange(mid_len, device=device) // BS).unsqueeze(0).expand(Hkv, -1)
        sc_cl_blk = torch.full((Hkv, (mid_len + BS - 1) // BS), float("-inf"), device=device)
        sc_cl_blk.scatter_reduce_(1, blk_ids_mid, tok_cl, reduce="amax", include_self=False)
        sc_blk = {"mavg": sc_mm[:, SINK // BS:], "aavg": sc_av[:, SINK // BS:], "cavg": sc_cl_blk}
        tok_score = {"mavg": sc_tok[:, SINK:mid_hi], "aavg": sc_tok[:, SINK:mid_hi], "cavg": tok_cl}
        sink_c = (pos < SINK).view(1, -1)
        swa_c = ((pos >= t_r + 1 - SWA) & (pos <= t_r)).view(1, -1)

        def select_sub(lo_off, hi_off, method, n_pages, n_tokens):
            """mid 区 [lo_off, hi_off)（相对 SINK 偏移）内：method 粗筛页 → 池内 token 细筛 top n_tokens。
            返回绝对位置 bool [Hkv, S]。"""
            s1 = sc_blk[method].masked_fill(
                ~(((torch.arange(sc_blk[method].shape[1], device=device) * BS + SINK) < SINK + hi_off) &
                  ((torch.arange(sc_blk[method].shape[1], device=device) * BS + BS - 1 + SINK) >= SINK + lo_off)).view(1, -1),
                float("-inf"))
            nblk_sub = s1.shape[-1]
            ib = torch.topk(s1, min(n_pages, nblk_sub), dim=-1).indices   # [Hkv,np]
            pool = torch.zeros(Hkv, mid_len, dtype=torch.bool, device=device)
            pool.scatter_(1, (ib * BS).clamp(max=mid_len - 1), True)
            # 展开页到 token（页长 64，clamp 尾部）
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

        # ---- 48 网格臂 ----
        for meth in METHODS:
            for a in ALPHAS:
                near_L = int(a * mid_len)
                near_lo_off, near_hi_off = mid_len - near_L, mid_len
                for b in BETAS:
                    nb_near = max(1, int(round(BP * b)))
                    nb_far = max(1, BP - nb_near)
                    nt_near = int(nb_near * BS * GAMMA)
                    nt_far = max(64, B_TOK - nt_near)
                    cand = torch.zeros(Hkv, S, dtype=torch.bool, device=device)
                    cand |= sink_c | swa_c
                    # near 子区：method=avg（三 method 的 near 都是 avg）
                    cand |= select_sub(near_lo_off, near_hi_off, "aavg", nb_near, nt_near)
                    # far 子区：method 各异
                    cand |= select_sub(0, near_lo_off, meth, nb_far, nt_far)
                    res.setdefault(f"{meth}_a{a}_b{b}", []).append(cov_mass(cand))
        # ---- 对照臂 ----
        cand = torch.zeros(Hkv, S, dtype=torch.bool, device=device)
        cand |= sink_c | swa_c
        cand |= select_sub(0, mid_len, "mavg", BP, B_TOK)     # mono（alpha=0）
        res.setdefault("mono", []).append(cov_mass(cand))
        keep = ((pos >= mid_hi - 1024) & (pos < mid_hi)).view(1, -1)
        cand = torch.zeros(Hkv, S, dtype=torch.bool, device=device)
        cand |= sink_c | swa_c | keep
        cand |= select_sub(0, mid_len - 1024, "mavg", BP, 256)  # tli_anchor
        res.setdefault("tli_anchor", []).append(cov_mass(cand))
        res.setdefault("build_s", []).append(round(t_build, 2))
    del k, q, d, ksub
    torch.cuda.empty_cache()
    return res


def main():
    device = "cuda:0"
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
