# E3 v2: 子空间粗筛 recall——多预算 × 多维度选择策略 × 双口径（entry recall + mass-weighted）
# 关键消融：低频尾维（TIA 规则） vs 随机同数维（证明「子空间保留」不是随便选维度）
import os
import json
import glob
import torch
import torch.nn.functional as F

TRACE_DIR = "/tmp/trace/qwen3-8b"
OUT_DIR = "/home/wangyuanshuo02/two-level-attention/exp/trace/results"
BLOCK = 64
NEAR = 2048

def qwen_subspace_idx(d_prime, D=128, mode="lowfreq"):
    d = d_prime // 2
    if mode == "lowfreq":
        return list(range(64 - d, 64)) + list(range(128 - d, 128))
    if mode == "random":
        g = torch.Generator().manual_seed(7)
        return [int(x) for x in torch.randperm(D, generator=g)[:d_prime]]
    if mode == "highfreq":  # 反例：高频头维（预期差）
        return list(range(d)) + list(range(64, 64 + d))
    raise ValueError(mode)

def coarse_scores(k, q, dim_idx, block=BLOCK):
    """k:[S,Hkv,D] q:[1,H,D] -> 块上界分数 [1,Hkv,nblk]（group-sum）"""
    S, Hkv, D = k.shape
    H = q.shape[1]
    G = H // Hkv
    nblk = (S + block - 1) // block
    idx = torch.tensor(dim_idx, device="cuda")
    ksub = k[..., idx]
    if nblk * block > S:
        ksub = F.pad(ksub, (0, 0, 0, 0, 0, nblk * block - S))
    kc = ksub.view(nblk, block, Hkv, -1)
    kmin, kmax = kc.amin(1).float(), kc.amax(1).float()
    qsub = q[..., idx].float()
    qg_pos = qsub.clamp(min=0).view(1, Hkv, G, -1)
    qg_neg = qsub.clamp(max=0).view(1, Hkv, G, -1)
    sc = (torch.einsum("bhgd,nhd->bhgn", qg_pos, kmax) +
          torch.einsum("bhgd,nhd->bhgn", qg_neg, kmin)).sum(-2)
    return sc

def main():
    budgets_blk = [32, 64, 128, 256]      # 候选块数（×64 token）：2K/4K/8K/16K token
    dprimes = [32, 64, 96, 128]
    modes = ["lowfreq", "random", "highfreq"]
    results = {}
    for pdir in sorted(glob.glob(f"{TRACE_DIR}/*")):
        name = os.path.basename(pdir)
        # 每个 (budget, d', mode) 收集 per-layer entry recall / mass recall
        acc = {(b, dp, m): {"entry": [], "mass": []} for b in budgets_blk for dp in dprimes for m in modes}
        # dense 基准的 per-layer 全概率质量（分母）
        for lf in sorted(glob.glob(f"{pdir}/layer*.pt")):
            d = torch.load(lf, map_location="cuda:0")
            k, q, qpos, S = d["k"].bfloat16(), d["q"].bfloat16(), d["qpos"].cuda(), d["S"]
            t = qpos[-1].item()
            Hkv, H, D = k.shape[1], q.shape[1], k.shape[-1]
            G = H // Hkv
            qg = q[-1:].float().view(1, Hkv, G, D)
            scale = D ** -0.5
            s = torch.einsum("bhgd,chd->bhgc", qg, k.float()).sum(-2) * scale
            s = s.masked_fill(torch.arange(S, device="cuda").view(1, 1, S) > t, float("-inf"))
            # dense top-1024（gt）与全 softmax 质量
            gt = torch.topk(s, 1024, dim=-1).indices          # [1,Hkv,1024]
            p_full = torch.softmax(s, dim=-1)                  # [1,Hkv,S]
            total_mass = p_full.sum(dim=(0, 1))                # [S] 跨 head 汇总质量
            nblk = (S + BLOCK - 1) // BLOCK
            blk_end = (torch.arange(nblk, device="cuda") + 1) * BLOCK - 1
            for b in budgets_blk:
                for dp in dprimes:
                    for m in modes:
                        sc = coarse_scores(k, q[-1:], qwen_subspace_idx(dp, mode=m))
                        sc = sc.masked_fill(blk_end.view(1, 1, -1) > t, float("-inf"))
                        cand_blk = torch.topk(sc, min(b, nblk), dim=-1).indices[0]  # [Hkv,b]
                        entry_r, mass_r = [], []
                        for h in range(Hkv):
                            cand = set()
                            for blk in cand_blk[h].tolist():
                                cand.update(range(blk * BLOCK, min((blk + 1) * BLOCK, S)))
                            gts = set(gt[0, h].tolist())
                            entry_r.append(len(cand & gts) / len(gts))
                            mass_r.append(total_mass[sorted(cand)].sum().item() / total_mass.sum().item())
                        acc[(b, dp, m)]["entry"].append(sum(entry_r) / len(entry_r))
                        acc[(b, dp, m)]["mass"].append(sum(mass_r) / len(mass_r))
            del k, q, s, p_full, gt
            torch.cuda.empty_cache()
        results[name] = {}
        for (b, dp, m), v in acc.items():
            results[name][f"blk{b}_d{dp}_{m}"] = {
                "entry_recall": sum(v["entry"]) / len(v["entry"]),
                "entry_p5": sorted(v["entry"])[max(0, int(0.05 * len(v["entry"])))],
                "mass_recall": sum(v["mass"]) / len(v["mass"]),
                "mass_p5": sorted(v["mass"])[max(0, int(0.05 * len(v["mass"])))],
            }
        # 打印关键行
        print(f"=== {name} ===")
        for b in budgets_blk:
            row = [f"blk{b}"]
            for dp in dprimes:
                for m in ["lowfreq", "random", "highfreq"]:
                    r = results[name][f"blk{b}_d{dp}_{m}"]
                    row.append(f"{m[:4]}{dp}:e{r['entry_recall']:.2f}/m{r['mass_recall']:.3f}")
            print("  " + " | ".join(row))
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(results, open(f"{OUT_DIR}/e3_subspace_recall_v2.json", "w"), indent=1)

if __name__ == "__main__":
    main()
