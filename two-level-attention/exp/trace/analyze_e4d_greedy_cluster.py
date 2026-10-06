# E4d：cluster 方法族消融——增量贪心聚类（用户指定：sim=0.9 余弦阈值、算术均值簇心、
#   sim_dims = NoPE 代理（低频尾维 32）或全维 128）vs kmeans（E4c 旧基线，点积 argmax）vs
#   minmax 块上界（冠军基线）vs oracle。
# 贪心语义：按时间顺序逐 token，与现有簇心的余弦相似 ≥ sim 则归入最优簇（算术均值在线更新），
#   否则新开簇（K_MAX=4096 上限，溢出强制归最优簇）。
# 口径：与 E4c 一致——严格 token 预算、per-head far mass 加权捕获、末位 query、far=[64, t-2048)；
#   采样：12 层 × 16 个 8B trace。输出 e4d_greedy_cluster.json（含贪心 build 耗时与簇数）。
import json
import os
import time

import torch
import torch.nn.functional as F

TRACE = "/tmp/trace/qwen3-8b"
OUT = "/home/wangyuanshuo02/two-level-attention/exp/trace/results/e4d_greedy_cluster.json"
D2I = list(range(48, 64)) + list(range(112, 128))   # 低频尾维 = NoPE 代理 = 压缩子空间 d'=32
BUDGETS = [512, 1024]
SIM = 0.9
K_MAX = 4096
BS = 64


def gpu_kmeans(x, K, niter=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    N, d = x.shape
    c = x[torch.randperm(N, generator=g)[:K]].clone()
    for _ in range(niter):
        a = (x @ c.T).argmax(dim=1)
        cnt = torch.bincount(a, minlength=K).float()
        sums = torch.zeros(K, d, device=x.device).index_add_(0, a, x)
        ne = cnt > 0
        c[ne] = sums[ne] / cnt[ne, None]
    return c, a


def greedy_cluster_assign(x, sim=SIM, k_max=K_MAX):
    """增量贪心聚类：x [T,Hkv,d] 时序 token。返回 sums[Hkv,K,d]/cnt[Hkv,K]/k_live[Hkv]/assign[Hkv,T]。"""
    T, Hkv, dd = x.shape
    dev = x.device
    sums = torch.zeros(Hkv, k_max, dd, device=dev)
    cnt = torch.zeros(Hkv, k_max, device=dev)
    k_live = torch.zeros(Hkv, dtype=torch.long, device=dev)
    assign = torch.zeros(Hkv, T, dtype=torch.long, device=dev)
    x_n = x.norm(dim=-1)                            # [T, Hkv]
    ar = torch.arange(Hkv, device=dev)
    ones_hk = torch.ones(Hkv, device=dev)
    for i in range(T):
        xi = x[i]                                   # [Hkv, d]
        norms = sums.norm(dim=-1)                   # [Hkv, K]
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
        new_h = torch.nonzero(~upd).flatten().tolist()
        for h in new_h:
            k = int(k_live[h])
            if k < k_max:
                sums[h, k] = xi[h]; cnt[h, k] = 1; k_live[h] += 1; a_final[h] = k
            else:  # 溢出保护：强制归当前最优簇
                sums[h, a[h]] += xi[h]; cnt[h, a[h]] += 1; a_final[h] = a[h]
        assign[:, i] = a_final
    return sums, cnt, k_live, assign


def eval_layer(lf, device):
    d = torch.load(lf, map_location="cpu", weights_only=False)
    k, q, qpos, S = d["k"].to(device).float(), d["q"].to(device).float(), d["qpos"], d["S"]
    t = int(qpos[-1])
    Hkv, H, D = k.shape[1], q.shape[1], k.shape[-1]
    G = H // Hkv
    qg = q[-1:].reshape(1, Hkv, G, D)
    s = torch.einsum("bhgd,chd->bhgc", qg, k).sum(-2) * (D ** -0.5)
    s = s.masked_fill(torch.arange(S, device=device).view(1, 1, S) > t, float("-inf"))
    p = torch.softmax(s, dim=-1)[0]                # [Hkv, S]
    far_lo, far_hi = 64, t - 2048
    if far_hi - far_lo < 4096:
        del k, q, s, p, d
        return None
    far_h = p[:, far_lo:far_hi].sum(-1)
    total_far = float(far_h.sum())
    if total_far < 1e-6:
        del k, q, s, p, d
        return None

    idx = torch.tensor(D2I, device=device)
    ksub = k[..., idx]                              # [S, Hkv, 32]
    qsub = q[-1:][..., idx].reshape(Hkv, G, 32).sum(1)      # [Hkv, 32]
    q_full = q[-1:].reshape(Hkv, G, D).sum(1)               # [Hkv, 128]
    # ---- minmax 块（冠军基线）----
    nblk = (S + BS - 1) // BS
    kk = F.pad(ksub, (0, 0, 0, 0, 0, nblk * BS - S))
    kc = kk.reshape(nblk, BS, Hkv, 32)
    kmin, kmax = kc.amin(1), kc.amax(1)
    sc_mm = (torch.einsum("hd,nhd->hn", qsub.clamp(min=0), kmax) +
             torch.einsum("hd,nhd->hn", qsub.clamp(max=0), kmin))
    # ---- kmeans 基线（E4c 同款）----
    km_c, km_a = [], []
    for h in range(Hkv):
        c, a = gpu_kmeans(ksub[far_lo:far_hi, h, :], 256, niter=20, seed=0)
        km_c.append(c); km_a.append(a)
    km_c = torch.stack(km_c)                        # [Hkv, 256, 32]
    cscore_km = torch.einsum("hd,hkd->hk", qsub, km_c)
    tok_km = torch.stack([cscore_km[h][km_a[h]] for h in range(Hkv)])  # [Hkv, Tfar]
    # ---- 贪心 lf32 臂（sim_dims=NoPE 代理低频尾维 32）----
    t0 = time.time()
    s_lf, c_lf, kl_lf, a_lf = greedy_cluster_assign(ksub[far_lo:far_hi], sim=SIM)
    t_lf = time.time() - t0
    cent_lf = s_lf / c_lf.clamp(min=1).unsqueeze(-1)         # 算术均值簇心
    cs_lf = torch.einsum("hd,hkd->hk", qsub, cent_lf)
    tok_gl = cs_lf.gather(1, a_lf)                           # [Hkv, Tfar]
    # ---- 贪心 full 臂（sim_dims=全维 128，打分也用全维均值簇心）----
    t0 = time.time()
    s_fu, c_fu, kl_fu, a_fu = greedy_cluster_assign(k[far_lo:far_hi], sim=SIM)
    t_fu = time.time() - t0
    cent_fu = s_fu / c_fu.clamp(min=1).unsqueeze(-1)
    cs_fu = torch.einsum("hd,hkd->hk", q_full, cent_fu)
    tok_gf = cs_fu.gather(1, a_fu)

    # 块级 scatter-amax（km_blk / gl_blk / gf_blk 同款公式）
    nblk_far = (far_hi - far_lo + BS - 1) // BS
    blk_ids = (torch.arange(far_hi - far_lo, device=device) // BS).unsqueeze(0).expand(Hkv, -1)

    def blk_amax(tok_score):
        sc = torch.full((Hkv, nblk_far), float("-inf"), device=device)
        sc.scatter_reduce_(1, blk_ids, tok_score, reduce="amax", include_self=False)
        return sc

    sc_kmblk, sc_glblk, sc_gfblk = blk_amax(tok_km), blk_amax(tok_gl), blk_amax(tok_gf)
    sc_mmf = sc_mm[:, far_lo // BS: far_lo // BS + nblk_far]
    order_far = torch.argsort(s[0, :, far_lo:far_hi], dim=-1, descending=True)

    strat_blk = {"minmax_blk": sc_mmf, "km_blk": sc_kmblk, "gl_blk": sc_glblk, "gf_blk": sc_gfblk}
    strat_tok = {"km_tok": tok_km, "gl_tok": tok_gl, "gf_tok": tok_gf}
    out = {"build_ms_gl": round(t_lf * 1000, 1), "build_ms_gf": round(t_fu * 1000, 1),
           "k_gl": int(kl_lf.float().mean()), "k_gf": int(kl_fu.float().mean())}
    for b in BUDGETS:
        nb = b // BS
        for name, sc in strat_blk.items():
            captured = 0.0
            for h in range(Hkv):
                if far_h[h].item() < 1e-8:
                    continue
                topb = torch.topk(sc[h], min(nb, sc.shape[-1])).indices
                toks = torch.cat([torch.arange(i * BS, min((i + 1) * BS, far_hi - far_lo), device=device) + far_lo for i in topb])
                captured += float(p[h][toks].sum())
            out[f"{name}@{b}"] = round(captured / total_far, 4)
        for name, ts in strat_tok.items():
            captured = 0.0
            for h in range(Hkv):
                if far_h[h].item() < 1e-8:
                    continue
                toks = torch.topk(ts[h], min(b, far_hi - far_lo)).indices + far_lo
                captured += float(p[h][toks].sum())
            out[f"{name}@{b}"] = round(captured / total_far, 4)
        captured = 0.0
        for h in range(Hkv):
            if far_h[h].item() >= 1e-8:
                captured += float(p[h][order_far[h][:b] + far_lo].sum())
        out[f"oracle@{b}"] = round(captured / total_far, 4)
    del k, q, s, p, d, ksub
    torch.cuda.empty_cache()
    return out


def main():
    device = "cuda:0"
    names = sorted(n for n in os.listdir(TRACE)
                   if os.path.isfile(os.path.join(TRACE, n, "meta.json")))
    results = {}
    for name in names:
        n_layers = json.load(open(f"{TRACE}/{name}/meta.json"))["n_layers"]
        acc = {}
        for li in range(0, n_layers, max(1, n_layers // 12)):
            r = eval_layer(f"{TRACE}/{name}/layer{li:02d}.pt", device)
            if r is None:
                continue
            for k2, v in r.items():
                acc.setdefault(k2, []).append(v)
        rec = {k2: round(sum(v) / len(v), 4) for k2, v in acc.items() if v}
        results[name] = rec
        print(f"[{name}] " + " ".join(f"{k2}={v}" for k2, v in sorted(rec.items())), flush=True)
    json.dump(results, open(OUT, "w"), indent=1)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
