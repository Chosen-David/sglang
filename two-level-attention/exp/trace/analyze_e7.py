# E7: kmeans 增量维护模拟——「prefill 时全量聚类 + 远端增长后不重聚类」的质量衰减
# 模拟：聚类只用 far 区前半段 token（prefill 时刻），后半段 token 增量 assign 到旧中心（decode 期新 token），
# 在最终位置评估 far recall/mass —— 对比 fresh（用全部 far token 聚类）
# 同时测增量 assign 的单 token 开销（GEMV 批量折算）
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
KC = 256
BUDGETS = [512, 1024, 2048]


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
    return c


def cand_from_centroids(c, cscore_order, assign_all, far_lo, far_hi, budget):
    """按中心分数排序取簇成员直到预算。assign_all: [Tfar] 全部 far token 的 assign"""
    cands = set()
    taken = 0
    for ci in cscore_order:
        if taken >= budget:
            break
        members = (assign_all == ci).nonzero().squeeze(1) + far_lo
        cands.update(members.tolist())
        taken += members.numel()
    return cands


def main():
    results = {}
    for pdir in sorted(glob.glob(f"{TRACE}/*")):
        name = os.path.basename(pdir)
        if not name.startswith(("needle", "natural", "lb_")):
            continue
        acc = {m: {b: {"recall": [], "mass": []} for b in BUDGETS} for m in ["fresh", "stale"]}
        assign_us_per_tok = []
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
            # 只取 far 质量最重的头（kmeans 逐头做，取 top-2 头省时间，每层同口径对比 fresh/stale）
            far_mass_head = pm[far_lo:far_hi]  # 头平均口径与 e4 一致，用平均
            # 对全部头逐一做（与 e4 相同口径）
            mid = (far_lo + far_hi) // 2       # 「prefill 时刻」：far 前半
            for h in range(Hkv):
                kfar = ksub[far_lo:far_hi, h, :]           # [Tfar,32]
                qh = qsub[0, h]
                n_fresh = kfar.shape[0]
                n_half = mid - far_lo
                # fresh: 全量聚类
                c_f = gpu_kmeans(kfar, KC, niter=20, seed=0)
                assign_f = (kfar @ c_f.T).argmax(1)
                sc_f = qh @ c_f.T
                order_f = torch.argsort(sc_f, descending=True).tolist()
                # stale: 只用前半聚类，后半增量 assign
                c_s = gpu_kmeans(kfar[:n_half], KC, niter=20, seed=0)
                assign_s = (kfar @ c_s.T).argmax(1)         # 全部 token assign 到旧中心（后半=增量）
                sc_s = qh @ c_s.T
                order_s = torch.argsort(sc_s, descending=True).tolist()
                # 增量 assign 单 token 开销（后半段 token 数）
                t0 = time.time()
                _ = (kfar[n_half:] @ c_s.T).argmax(1)
                torch.cuda.synchronize()
                assign_us_per_tok.append((time.time() - t0) / max(1, n_fresh - n_half) * 1e6)
                g = gt_far[h]
                for b in BUDGETS:
                    for mode, c_, order_, assign_ in [("fresh", c_f, order_f, assign_f),
                                                      ("stale", c_s, order_s, assign_s)]:
                        cands = cand_from_centroids(c_, order_, assign_, far_lo, far_hi, b)
                        if g:
                            acc[mode][b]["recall"].append(len(cands & g) / len(g))
                        cl = torch.tensor(sorted(cands), device="cuda", dtype=torch.long)
                        acc[mode][b]["mass"].append(pm[cl].sum().item() / total_far_mass)
            del k, q, s, p
            torch.cuda.empty_cache()
        results[name] = {
            m: {str(b): {"recall": st.mean([x for x in acc[m][b]["recall"] if x == x]),
                         "mass": st.mean(acc[m][b]["mass"])} for b in BUDGETS}
            for m in ["fresh", "stale"]
        }
        results[name]["assign_us_per_tok"] = st.mean(assign_us_per_tok)
        print(f"=== {name} (K_c={KC}) ===")
        for m in ["fresh", "stale"]:
            r = results[name][m]
            print(f"  {m:5s}: " + "  ".join(
                f"b{b}: recall={r[str(b)]['recall']:.3f} mass={r[str(b)]['mass']:.3f}" for b in BUDGETS))
        print(f"  增量 assign 开销: {results[name]['assign_us_per_tok']:.1f} us/token/head")
    json.dump(results, open(f"{OUT}/e7_incremental_kmeans.json", "w"), indent=1)


if __name__ == "__main__":
    main()
