# E4: 创新点 B Go/No-Go——远端区域三种代表策略同预算对比
#   (a) block mean（HISA 式均值代表）
#   (b) block min/max 上界（TIA）
#   (c) kmeans 聚类中心（proposal 创新点 B）+ 增量 assign 模拟
#   (d) exact token 分数（oracle 上界）
# 口径：远端 = [64, t-2048)；gt = dense top-1024 中的远端 token；预算 C_far tokens/head
# 输出：far token recall / far mass 捕获 / kmeans 构建时间
import os
import torch
import torch.nn.functional as F
import glob
import json
import time
import statistics as st

TRACE = "/tmp/trace/qwen3-8b"
OUT = "/home/wangyuanshuo02/two-level-attention/exp/trace/results"
D2I = list(range(48, 64)) + list(range(112, 128))  # d'=32 低频子空间


def gpu_kmeans(x, K, niter=20, seed=0):
    """x:[N,d] fp32 -> centroids [K,d], assign [N]。GEMM 距离 + argmin + scatter 均值"""
    g = torch.Generator().manual_seed(seed)
    N, d = x.shape
    c = x[torch.randperm(N, generator=g)[:K]].clone()
    for _ in range(niter):
        # 距离 = |x|² - 2x·cᵀ + |c|²（只需 argmin，可省 |x|²）
        sims = x @ c.T                                   # [N,K]
        assign = sims.argmax(dim=1)                       # 余弦最近（等价 L2 最近 when 归一化尺度一致；这里用点积因为打分也是点积语义）
        # 均值更新（空簇保留原中心）
        cnt = torch.bincount(assign, minlength=K).float()
        sums = torch.zeros(K, d, device=x.device).index_add_(0, assign, x)
        nonempty = cnt > 0
        c[nonempty] = sums[nonempty] / cnt[nonempty, None]
    return c, assign


def main():
    C_FAR = 2048        # 每 kv head 远端候选 token 预算
    NBLK_BUDGET = C_FAR // 64  # 块策略预算 = 32 块
    KC_LIST = [256, 1024]
    results = {}
    for pdir in sorted(glob.glob(f"{TRACE}/*")):
        name = os.path.basename(pdir)
        if not name.startswith(("needle", "natural")):
            continue
        acc = {m: {"recall": [], "mass": []} for m in
               ["mean", "minmax", "km256", "km1024", "exact"]}
        build_times = {256: [], 1024: []}
        for lf in sorted(glob.glob(f"{pdir}/layer*.pt")):
            d = torch.load(lf, map_location="cuda:0")
            k, q, qpos, S = d["k"].float(), d["q"].float(), d["qpos"].cuda(), d["S"]
            t = qpos[-1].item()
            Hkv, H, D = k.shape[1], q.shape[1], k.shape[-1]
            G = H // Hkv
            # dense gt
            qg = q[-1:].reshape(1, Hkv, G, D)
            s = torch.einsum("bhgd,chd->bhgc", qg, k).sum(-2) * (D ** -0.5)
            s = s.masked_fill(torch.arange(S, device="cuda").view(1, 1, S) > t, float("-inf"))
            p = torch.softmax(s, dim=-1)
            pm = p.mean(dim=(0, 1))
            far_lo, far_hi = 64, t - 2048
            far_mask_idx = torch.arange(far_lo, far_hi, device="cuda")
            far_mass = pm[far_lo:far_hi]
            total_far_mass = far_mass.sum().item()
            if total_far_mass < 1e-6:
                continue  # 该层远端无质量，跳过
            # gt far tokens（每 head 的 dense top-1024 中落在远端的）
            gt = torch.topk(s, 1024, dim=-1).indices[0]           # [Hkv,K]
            gt_far = [set(x.item() for x in gt[h] if far_lo <= x.item() < far_hi) for h in range(Hkv)]
            # 子空间张量
            idx = torch.tensor(D2I, device="cuda")
            ksub = k[..., idx]                                     # [S,Hkv,32]
            qsub = q[-1:][..., idx].reshape(1, Hkv, G, 32).sum(2)  # [1,Hkv,32]
            nblk = (S + 63) // 64
            kk = F.pad(ksub, (0, 0, 0, 0, 0, nblk * 64 - S)) if nblk * 64 > S else ksub
            kc = kk.reshape(nblk, 64, Hkv, 32)
            # (a) block mean
            kmean = kc.mean(1)                                     # [nblk,Hkv,32]
            sc_mean = torch.einsum("hd,nhd->hn", qsub[0], kmean)
            # (b) min/max 上界
            kmin, kmax = kc.amin(1), kc.amax(1)
            sc_mm = (torch.einsum("hd,nhd->hn", qsub[0].clamp(min=0), kmax) +
                     torch.einsum("hd,nhd->hn", qsub[0].clamp(max=0), kmin))
            blk_lo, blk_hi = far_lo // 64, far_hi // 64            # 远端块范围
            blk_range = torch.arange(blk_lo, blk_hi, device="cuda")
            far_sel_mask = torch.zeros(S, dtype=torch.bool, device="cuda")
            cand_exact = set()
            # (d) exact：远端 token 精确分数 top C_FAR（块对齐换算：top 块使 token 数≈预算）
            # 各策略统一接口：给出每 head 的远端候选 token 集合
            def take_blocks(sc, budget_blk):
                cand = [set() for _ in range(Hkv)]
                scb = sc[:, blk_range]                             # [Hkv, nfarblk]
                topb = torch.topk(scb, min(budget_blk, scb.shape[1]), dim=-1).indices
                for h in range(Hkv):
                    for b in topb[h].tolist():
                        cand[h].update(range(blk_range[b].item() * 64, min((blk_range[b].item() + 1) * 64, far_hi)))
                return cand

            cand_mean = take_blocks(sc_mean, NBLK_BUDGET)
            cand_mm = take_blocks(sc_mm, NBLK_BUDGET)
            # (d) exact oracle：按 token 精确分选 top C_FAR
            s_far = s[0, :, far_lo:far_hi]                         # [Hkv, Tfar]
            topk_exact = torch.topk(s_far, min(C_FAR, s_far.shape[1]), dim=-1).indices + far_lo
            cand_exact = [set(x.item() for x in topk_exact[h]) for h in range(Hkv)]
            # (c) kmeans
            cands_km = {}
            for Kc in KC_LIST:
                t0 = time.time()
                kfar = ksub[far_lo:far_hi]                          # [Tfar,Hkv,32]
                cand_km = [set() for _ in range(Hkv)]
                for h in range(Hkv):
                    x = kfar[:, h, :]                               # [Tfar,32]
                    c, assign = gpu_kmeans(x, Kc, niter=20, seed=0)
                    cscore = qsub[0, h] @ c.T                       # [Kc]
                    order = torch.argsort(cscore, descending=True)
                    cnt = torch.bincount(assign, minlength=Kc)
                    taken = 0
                    for ci in order.tolist():
                        if taken >= C_FAR:
                            break
                        members = torch.nonzero(assign == ci).squeeze(1) + far_lo
                        cand_km[h].update(members.tolist())
                        taken += len(members)
                build_times[Kc].append(time.time() - t0)
                cands_km[Kc] = cand_km
            # 评分
            def score_cand(cand):
                r, m = [], []
                for h in range(Hkv):
                    g = gt_far[h]
                    if g:
                        r.append(len(cand[h] & g) / len(g))
                    cl = torch.tensor(sorted(cand[h]), device="cuda", dtype=torch.long)
                    m.append(far_mass[cl - far_lo].sum().item() / total_far_mass)
                return (sum(r) / len(r) if r else float("nan"),
                        sum(m) / len(m))
            acc["mean"]["recall"].append(score_cand(cand_mean)[0]); acc["mean"]["mass"].append(score_cand(cand_mean)[1])
            acc["minmax"]["recall"].append(score_cand(cand_mm)[0]); acc["minmax"]["mass"].append(score_cand(cand_mm)[1])
            for Kc in KC_LIST:
                r, m = score_cand(cands_km[Kc])
                acc[f"km{Kc}"]["recall"].append(r); acc[f"km{Kc}"]["mass"].append(m)
            acc["exact"]["recall"].append(score_cand(cand_exact)[0]); acc["exact"]["mass"].append(score_cand(cand_exact)[1])
            del k, q, s, p
            torch.cuda.empty_cache()
        results[name] = {
            m: {"recall": st.mean([x for x in v["recall"] if x == x]),
                "mass": st.mean(v["mass"])}
            for m, v in acc.items()}
        results[name]["kmeans_build_ms"] = {
            str(Kc): st.mean(build_times[Kc]) * 1000 for Kc in KC_LIST}
        print(f"=== {name} (C_far={C_FAR} tok/head) ===")
        for m in ["mean", "minmax", "km256", "km1024", "exact"]:
            print(f"  {m:8s}: far_recall={results[name][m]['recall']:.3f}  far_mass={results[name][m]['mass']:.3f}")
        print(f"  kmeans build: {results[name]['kmeans_build_ms']}")
    json.dump(results, open(f"{OUT}/e4_representatives.json", "w"), indent=1)

if __name__ == "__main__":
    main()
