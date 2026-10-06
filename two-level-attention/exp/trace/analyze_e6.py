# E6 (Innovation D'): 层自适应级联跳过——远端质量低的层直接跳过远端检索
# 依据：H1_full 显示 ~44% 层是 sink-dominated（far mass < 0.02）
# 问题：用「上一 prompt 的层轮廓」或「在线 L1 信号」能否在推理时正确判定哪些层可跳过？
# 口径：
#   (1) 离线 oracle：每层自己 far mass 是否 < 阈值 → 跳过的层的 far 质量总量（质量损失上界）
#   (2) 跨 prompt 泛化：用 prompt A 的层轮廓预测 prompt B 的可跳层 → 预测准确率（precision/recall）
#   (3) 在线 L1 信号近似：用 L1 块上界 top 块分数之和（sink+near 之外）归一化近似 far mass
import os
import torch
import glob
import json
import statistics as st

TRACE = "/tmp/trace/qwen3-8b"
OUT = "/home/wangyuanshuo02/two-level-attention/exp/trace/results"
THRESH = 0.02   # far mass 阈值：低于此认为层是 sink-dominated


def layer_stats(lf):
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
    far = pm[64:t - 2048].sum().item()
    # 在线 L1 信号：d'=32 子空间块上界，far 块 top-32 分数和 / 全部块分数和
    idx1 = torch.tensor(list(range(48, 64)) + list(range(112, 128)), device="cuda")
    nblk = (S + 63) // 64
    kc = k[..., idx1]
    if nblk * 64 > S:
        kc = torch.nn.functional.pad(kc, (0, 0, 0, 0, 0, nblk * 64 - S))
    kc = kc.reshape(nblk, 64, Hkv, 32)
    kmin, kmax = kc.amin(1), kc.amax(1)
    qs = q[-1:][..., idx1]
    qg2 = qs.clamp(min=0).reshape(1, Hkv, G, 32)
    qn2 = qs.clamp(max=0).reshape(1, Hkv, G, 32)
    sc1 = (torch.einsum("bhgd,nhd->bhgn", qg2, kmax) +
           torch.einsum("bhgd,nhd->bhgn", qn2, kmin)).sum(-2)  # [1,Hkv,nblk]
    blk_end = (torch.arange(nblk, device="cuda") + 1) * 64 - 1
    sc1 = sc1.masked_fill(blk_end.view(1, 1, -1) > t, float("-inf"))
    # far 块 = 排除前 1 块(64 tok sink) 和最后 32 块(2048 near)
    far_blk_lo, far_blk_hi = 1, max(1, (t - 2048) // 64)
    sc_far = sc1[0, :, far_blk_lo:far_blk_hi]
    sc_all = sc1[0]
    # 每头取 far 块 top-32 分数和（softmax 前的原始分和，近似 far 吸引度）
    k1 = min(32, sc_far.shape[1])
    top_far = torch.topk(sc_far, k1, dim=-1).values.clamp(min=0).sum(-1)  # [Hkv]
    tot = sc_all.clamp(min=0).sum(-1)
    l1_signal = (top_far / tot.clamp(min=1e-9)).mean().item()
    del k, q, s, p, kc, sc1
    torch.cuda.empty_cache()
    return far, l1_signal


def main():
    profiles = {}   # name -> [(far_mass, l1_signal) per layer]
    for pdir in sorted(glob.glob(f"{TRACE}/*")):
        name = os.path.basename(pdir)
        if not os.path.exists(f"{pdir}/meta.json"):
            continue
        rows = []
        for lf in sorted(glob.glob(f"{pdir}/layer*.pt")):
            rows.append(layer_stats(lf))
        profiles[name] = rows
        n_skip = sum(1 for f, _ in rows if f < THRESH)
        far_sum = sum(f for f, _ in rows)
        far_skip_sum = sum(f for f, _ in rows if f < THRESH)
        # L1 信号与 far mass 的相关性
        import math
        a = [f for f, _ in rows]; b = [x for _, x in rows]
        n = len(a)
        ma, mb = sum(a) / n, sum(b) / n
        cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
        va = sum((a[i] - ma) ** 2 for i in range(n))
        vb = sum((b[i] - mb) ** 2 for i in range(n))
        corr = cov / math.sqrt(va * vb) if va > 0 and vb > 0 else 0.0
        print(f"{name}: 可跳层 {n_skip}/{n} | 跳过层 far 质量合计 {far_skip_sum:.4f}"
              f" / 全层 far {far_sum:.4f} | L1信号相关 {corr:.3f}")
    # 跨 prompt 泛化：A 预测 B 的可跳层
    names = list(profiles.keys())
    results = {"per_prompt": {}, "cross_prompt_pred": {}, "l1_online_pred": {}}
    for name, rows in profiles.items():
        n_skip = sum(1 for f, _ in rows if f < THRESH)
        results["per_prompt"][name] = {
            "n_layers": len(rows),
            "n_skippable": n_skip,
            "far_mass_total": sum(f for f, _ in rows),
            "far_mass_skipped": sum(f for f, _ in rows if f < THRESH),
        }
    # 用平均轮廓（跨 prompt 离线 profile）预测每个 prompt
    n_min = min(len(r) for r in profiles.values())
    avg_profile = [st.mean([profiles[nm][i][0] for nm in names]) for i in range(n_min)]
    for name, rows in profiles.items():
        pred_skip = [i for i in range(n_min) if avg_profile[i] < THRESH]
        true_skip = [i for i in range(n_min) if rows[i][0] < THRESH]
        tp = len(set(pred_skip) & set(true_skip))
        prec = tp / len(pred_skip) if pred_skip else 1.0
        rec = tp / len(true_skip) if true_skip else 1.0
        # 跳错层的质量代价：预测跳过但实际 far 高的层
        miss_mass = sum(rows[i][0] for i in pred_skip if rows[i][0] >= THRESH)
        results["cross_prompt_pred"][name] = {
            "precision": prec, "recall": rec,
            "n_pred": len(pred_skip),
            "missed_far_mass": miss_mass,
        }
        print(f"  平均轮廓→{name}: pred={len(pred_skip)} prec={prec:.2f} rec={rec:.2f} "
              f"漏掉far质量={miss_mass:.4f}")
    # 在线 L1 信号阈值判定：l1_signal < 阈值 → 跳过。扫阈值
    all_l1 = [x for rows in profiles.values() for _, x in rows]
    l1_th = st.mean(all_l1)  # 简单取均值作演示阈值
    for name, rows in profiles.items():
        pred_skip = [i for i, (f, l1s) in enumerate(rows) if l1s < l1_th]
        true_skip = [i for i, (f, _) in enumerate(rows) if f < THRESH]
        tp = len(set(pred_skip) & set(true_skip))
        miss_mass = sum(rows[i][0] for i in pred_skip if rows[i][0] >= THRESH)
        results["l1_online_pred"][name] = {
            "threshold": l1_th, "n_pred": len(pred_skip),
            "precision": tp / len(pred_skip) if pred_skip else 1.0,
            "missed_far_mass": miss_mass,
        }
        print(f"  L1在线信号→{name}: pred={len(pred_skip)} "
              f"missed_far={miss_mass:.4f}")
    json.dump(results, open(f"{OUT}/e6_layer_skip.json", "w"), indent=1)


if __name__ == "__main__":
    main()
