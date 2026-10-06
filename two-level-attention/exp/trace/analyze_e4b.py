# E4b: 远端代表 Pareto 曲线——预算扫描 C_far ∈ {512,1024,2048,4096} tok/head
# 策略：block mean / block minmax / kmeans(K_c=256,1024) / exact oracle
# 每层一次加载，多预算复用张量
import os
import torch
import torch.nn.functional as F
import glob
import json
import time
import statistics as st

TRACE = "/tmp/trace/qwen3-8b"
OUT = "/home/wangyuanshuo02/two-level-attention/exp/trace/results"
D2I = list(range(48, 64)) + list(range(112, 128))
BUDGETS = [512, 1024, 2048, 4096]
STRATS = ["mean", "minmax", "km256", "km1024", "exact"]


def gpu_kmeans(x, K, niter=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    N, d = x.shape
    c = x[torch.randperm(N, generator=g)[:K]].clone()
    for _ in range(niter):
        sims = x @ c.T
        assign = sims.argmax(dim=1)
        cnt = torch.bincount(assign, minlength=K).float()
        sums = torch.zeros(K, d, device=x.device).index_add_(0, assign, x)
        nonempty = cnt > 0
        c[nonempty] = sums[nonempty] / cnt[nonempty, None]
    return c, assign


def main():
    results = {}
    for pdir in sorted(glob.glob(f"{TRACE}/*")):
        name = os.path.basename(pdir)
        if not name.startswith(("needle", "natural", "lb_")):
            continue
        # acc[strat][budget] -> list over layers
        acc = {m: {b: {"recall": [], "mass": []} for b in BUDGETS} for m in STRATS}
        build_ms = []
        for lf in sorted(glob.glob(f"{pdir}/layer*.pt")):
            d = torch.load(lf, map_location="cuda:0")
            k, q, qpos, S = d["k"].float(), d["q"].float(), d["qpos"].cuda(), d["S"]
            t = qpos[-1].item()
            Hkv, H, D = k.shape[1], q.shape[1], k.shape[-1]
            G = H // Hkv
            qg = q[-1:].reshape(1, Hkv, G, D)
            s = torch.einsum("bhgd,chd->bhgc", qg, k).sum(-2) * (D ** -0.5)
            s = s.masked_fill(torch.arange(S, device="cuda").view(1, 1, S) > t, float("-inf"))
            p = torch.softmax(s, dim=-1)
            pm = p.mean(dim=(0, 1))
            far_lo, far_hi = 64, t - 2048
            total_far_mass = pm[far_lo:far_hi].sum().item()
            if total_far_mass < 1e-6:
                continue
            gt = torch.topk(s, 1024, dim=-1).indices[0]
            gt_far = [set(x.item() for x in gt[h] if far_lo <= x.item() < far_hi) for h in range(Hkv)]
            idx = torch.tensor(D2I, device="cuda")
            ksub = k[..., idx]
            qsub = q[-1:][..., idx].reshape(1, Hkv, G, 32).sum(2)
            nblk = (S + 63) // 64
            kk = F.pad(ksub, (0, 0, 0, 0, 0, nblk * 64 - S)) if nblk * 64 > S else ksub
            kc = kk.reshape(nblk, 64, Hkv, 32)
            kmean = kc.mean(1)
            sc_mean = torch.einsum("hd,nhd->hn", qsub[0], kmean)
            kmin, kmax = kc.amin(1), kc.amax(1)
            sc_mm = (torch.einsum("hd,nhd->hn", qsub[0].clamp(min=0), kmax) +
                     torch.einsum("hd,nhd->hn", qsub[0].clamp(max=0), kmin))
            blk_lo, blk_hi = far_lo // 64, far_hi // 64
            blk_range = torch.arange(blk_lo, blk_hi, device="cuda")
            sc_mean_f = sc_mean[:, blk_range]
            sc_mm_f = sc_mm[:, blk_range]
            s_far = s[0, :, far_lo:far_hi]
            order_far = torch.argsort(s_far, dim=-1, descending=True)  # [Hkv,Tfar] exact 排序
            # kmeans（每层一次，预算复用：按中心分数排序展开）
            t0 = time.time()
            kfar = ksub[far_lo:far_hi]
            km_data = {}
            for Kc in [256, 1024]:
                per_head = []
                for h in range(Hkv):
                    c, assign = gpu_kmeans(kfar[:, h, :], Kc, niter=20, seed=0)
                    cscore = qsub[0, h] @ c.T
                    order = torch.argsort(cscore, descending=True)
                    members = [torch.nonzero(assign == ci).squeeze(1) + far_lo for ci in range(Kc)]
                    per_head.append((order, members))
                km_data[Kc] = per_head
            build_ms.append(time.time() - t0)

            def cand_by_budget(strat, budget):
                cands = [set() for _ in range(Hkv)]
                if strat in ("mean", "minmax"):
                    nb = max(1, budget // 64)
                    sc = sc_mean_f if strat == "mean" else sc_mm_f
                    topb = torch.topk(sc, min(nb, sc.shape[1]), dim=-1).indices
                    for h in range(Hkv):
                        for b in topb[h].tolist():
                            cands[h].update(range(blk_range[b] * 64, min((blk_range[b] + 1) * 64, far_hi)))
                elif strat == "exact":
                    for h in range(Hkv):
                        cands[h] = set(x.item() for x in (order_far[h][:budget] + far_lo))
                else:
                    Kc = int(strat[2:])
                    for h in range(Hkv):
                        order, members = km_data[Kc][h]
                        taken = 0
                        for ci in order.tolist():
                            if taken >= budget:
                                break
                            cands[h].update(members[ci].tolist())
                            taken += len(members[ci])
                return cands

            def score(cands):
                r, m = [], []
                for h in range(Hkv):
                    g = gt_far[h]
                    if g:
                        r.append(len(cands[h] & g) / len(g))
                    cl = torch.tensor(sorted(cands[h]), device="cuda", dtype=torch.long)
                    m.append(pm[cl].sum().item() / total_far_mass)
                return (sum(r) / len(r) if r else float("nan"), sum(m) / len(m))

            for strat in STRATS:
                for b in BUDGETS:
                    r, m = score(cand_by_budget(strat, b))
                    acc[strat][b]["recall"].append(r)
                    acc[strat][b]["mass"].append(m)
            del k, q, s, p
            torch.cuda.empty_cache()
        results[name] = {"build_ms_mean": st.mean(build_ms) if build_ms else None, "curves": {}}
        for strat in STRATS:
            results[name]["curves"][strat] = {
                str(b): {"recall": st.mean([x for x in acc[strat][b]["recall"] if x == x]),
                         "mass": st.mean(acc[strat][b]["mass"])}
                for b in BUDGETS}
        print(f"=== {name}（kmeans build {results[name]['build_ms_mean']*1000:.0f} ms/层）===")
        hdr = "budget: " + "  ".join(f"{b:>14d}" for b in BUDGETS)
        print("  recall  " + hdr)
        for strat in STRATS:
            print(f"    {strat:6s} " + "  ".join(f"{results[name]['curves'][strat][str(b)]['recall']:14.3f}" for b in BUDGETS))
        print("  mass    " + hdr)
        for strat in STRATS:
            print(f"    {strat:6s} " + "  ".join(f"{results[name]['curves'][strat][str(b)]['mass']:14.3f}" for b in BUDGETS))
    json.dump(results, open(f"{OUT}/e4b_pareto.json", "w"), indent=1)

if __name__ == "__main__":
    main()
