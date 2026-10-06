# E4c（修正 E4b 超选 bug）：严格 token 预算下远端策略对比
# 口径修正：
#   1. 预算严格 = 恰好 N 个 token（块级=块数×64；token 级=topk N）
#   2. 捕获评估：per-head 自己分布的 far mass 捕获（加权平均 Σcaptured/Σfar）
# 策略：minmax块 / kmeans块(scatter-amax) / kmeans token / oracle token
import glob
import json
import os
import statistics as st
import time

import torch
import torch.nn.functional as F

TRACE = "/tmp/trace/qwen3-8b"
OUT = "/home/wangyuanshuo02/two-level-attention/exp/trace/results"
D2I = list(range(48, 64)) + list(range(112, 128))
BUDGETS = [512, 1024, 2048]
STRATS = ["minmax_blk", "km_blk", "km_tok", "oracle"]


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


def main():
    results = {}
    for pdir in sorted(glob.glob(f"{TRACE}/*")):
        name = os.path.basename(pdir)
        if not name.startswith(("needle", "natural", "lb_")):
            continue
        acc = {m: {b: [] for b in BUDGETS} for m in STRATS}  # 每层加权捕获
        for lf in sorted(glob.glob(f"{pdir}/layer*.pt")):
            d = torch.load(lf, map_location="cuda:0")
            k, q, qpos, S = d["k"].float(), d["q"].float(), d["qpos"].cuda(), d["S"]
            t = qpos[-1].item()
            Hkv, H, D = k.shape[1], q.shape[1], k.shape[-1]
            G = H // Hkv
            qg = q[-1:].reshape(1, Hkv, G, D)
            s = torch.einsum("bhgd,chd->bhgc", qg, k).sum(-2) * (D ** -0.5)
            s = s.masked_fill(torch.arange(S, device="cuda").view(1, 1, S) > t, float("-inf"))
            p = torch.softmax(s, dim=-1)[0]  # [Hkv, S]
            far_lo, far_hi = 64, t - 2048
            far_h = p[:, far_lo:far_hi].sum(dim=-1)  # [Hkv] 每 head far mass
            total_far = far_h.sum().item()
            if total_far < 1e-6:
                continue
            idx = torch.tensor(D2I, device="cuda")
            ksub = k[..., idx]                       # [S, Hkv, 32]
            qsub = q[-1:][..., idx].reshape(Hkv, G, 32).sum(1)  # [Hkv, 32]
            # 块级 minmax / 块分数
            nblk = (S + 63) // 64
            kk = F.pad(ksub, (0, 0, 0, 0, 0, nblk * 64 - S)) if nblk * 64 > S else ksub
            kc = kk.reshape(nblk, 64, Hkv, 32)
            kmin, kmax = kc.amin(1), kc.amax(1)
            sc_mm = (torch.einsum("hd,nhd->hn", qsub.clamp(min=0), kmax) +
                     torch.einsum("hd,nhd->hn", qsub.clamp(max=0), kmin))
            # kmeans（K_c=256, niter=20）
            t0 = time.time()
            km_c, km_a = [], []
            for h in range(Hkv):
                c, a = gpu_kmeans(ksub[far_lo:far_hi, h, :], 256, niter=20, seed=0)
                km_c.append(c)
                km_a.append(a)
            build_ms = (time.time() - t0) * 1000
            km_c = torch.stack(km_c)      # [Hkv, Kc, 32]
            cscore = torch.einsum("hd,hkd->hk", qsub, km_c)  # [Hkv, Kc]
            # token → 块 scatter-amax（B 原设计：块分数 = 块内簇分最大值）
            nblk_far = (far_hi - far_lo + 63) // 64  # ceil：含末尾不满块
            blk_ids = (torch.arange(far_hi - far_lo, device="cuda") // 64)
            blk_ids = blk_ids.unsqueeze(0).expand(Hkv, -1)
            tok_score = torch.stack([cscore[h][km_a[h]] for h in range(Hkv)])  # [Hkv, Tfar]
            sc_kmblk = torch.full((Hkv, nblk_far), float("-inf"), device="cuda")
            sc_kmblk.scatter_reduce_(1, blk_ids, tok_score, reduce="amax", include_self=False)
            # oracle 排序
            s_far = s[0, :, far_lo:far_hi]
            order_far = torch.argsort(s_far, dim=-1, descending=True)

            blk_lo = far_lo // 64
            for b in BUDGETS:
                nb = b // 64
                for strat in STRATS:
                    captured = 0.0
                    for h in range(Hkv):
                        if far_h[h].item() < 1e-8:
                            continue
                        if strat == "minmax_blk":
                            sc_mmf = sc_mm[h, blk_lo:blk_lo + nblk_far]
                            topb = torch.topk(sc_mmf, min(nb, sc_mmf.shape[-1])).indices
                            toks = torch.cat([torch.arange(i * 64, min((i + 1) * 64, far_hi - far_lo), device="cuda") + far_lo for i in topb])
                        elif strat == "km_blk":
                            topb = torch.topk(sc_kmblk[h], min(nb, nblk_far)).indices
                            toks = torch.cat([torch.arange(i * 64, min((i + 1) * 64, far_hi - far_lo), device="cuda") + far_lo for i in topb])
                        elif strat == "km_tok":
                            toks = torch.topk(tok_score[h], min(b, far_hi - far_lo)).indices + far_lo
                        else:
                            toks = order_far[h][:b] + far_lo
                        captured += p[h][toks].sum().item()
                    acc[strat][b].append(captured / total_far)
            del k, q, s, p, d
            torch.cuda.empty_cache()
        results[name] = {strat: {str(b): round(st.mean(v), 4) if v else None
                                  for b, v in acc[strat].items()} for strat in STRATS}
        print(f"=== {name} ===")
        print("  budget:  " + "  ".join(f"{b:>8d}" for b in BUDGETS))
        for strat in STRATS:
            print(f"  {strat:10s}" + "  ".join(f"{results[name][strat][str(b)]:8.3f}" for b in BUDGETS))
    json.dump(results, open(f"{OUT}/e4c_strict_budget.json", "w"), indent=1)
    print(f"\nsaved -> {OUT}/e4c_strict_budget.json")


if __name__ == "__main__":
    main()
