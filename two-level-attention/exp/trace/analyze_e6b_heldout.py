# E6b: D' 层跳过掩码的严格 held-out（leave-one-out）验证
# 动机：E6 的「平均轮廓→每 prompt」预测把目标 trace 混入校准集（非 held-out），
# 审稿必问「校准掩码在未见负载上的 precision」。本脚本对 16 条 trace 做
# leave-one-out：掩码由其余 15 条平均轮廓阈值化得出，在留出 trace 上测
#   precision = |{pred skip 且 far mass < THRESH}| / |pred skip|
#   missed_far = 留出 trace 上被跳过层的 far mass 合计（质量代价）
# 扫三档 THRESH（0.02 E6 原值 / 0.01 / 0.005 保守），找 precision ≥0.98 的稳健点。
# far mass 口径与 E6 完全一致（pm 平均 + far = pm[64:t-2048]），trace 级 GPU 重算。
import glob
import json
import os

import torch

TRACE = "/tmp/trace/qwen3-8b"
OUT = "/home/wangyuanshuo02/two-level-attention/exp/trace/results"


def layer_far(lf):
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
    del k, q, s, p, d
    torch.cuda.empty_cache()
    return far


def main():
    # 1) 重算每条 trace 的 36 层 far 轮廓（缓存到 json，二次跑免重算）
    cache_path = f"{OUT}/e6b_far_profiles.json"
    if os.path.exists(cache_path):
        profiles = json.load(open(cache_path))
        print(f"profiles cached: {len(profiles)} traces")
    else:
        profiles = {}
        for pdir in sorted(glob.glob(f"{TRACE}/*")):
            name = os.path.basename(pdir)
            if not os.path.exists(f"{pdir}/meta.json"):
                continue
            rows = [layer_far(lf) for lf in sorted(glob.glob(f"{pdir}/layer*.pt"))]
            profiles[name] = rows
            print(f"{name}: far range {min(rows):.4f}-{max(rows):.4f}")
        json.dump(profiles, open(cache_path, "w"), indent=1)

    names = list(profiles.keys())
    L = min(len(r) for r in profiles.values())
    print(f"\n{len(names)} traces × {L} layers; leave-one-out:")

    results = {}
    for thresh in [0.02, 0.01, 0.005]:
        rows_out = []
        for held in names:
            others = [n for n in names if n != held]
            avg = [sum(profiles[n][i] for n in others) / len(others) for i in range(L)]
            pred = [i for i in range(L) if avg[i] < thresh]
            true = [i for i in range(L) if profiles[held][i] < thresh]
            tp = len(set(pred) & set(true))
            prec = tp / len(pred) if pred else 1.0
            rec = tp / len(true) if true else 1.0
            miss = sum(profiles[held][i] for i in pred if profiles[held][i] >= thresh)
            rows_out.append({
                "held_out": held, "n_pred": len(pred), "n_true": len(true),
                "precision": round(prec, 4), "recall": round(rec, 4),
                "missed_far_mass": round(miss, 5),
            })
            print(f"  TH={thresh} {held}: pred={len(pred)} prec={prec:.3f} "
                  f"rec={rec:.2f} missed_far={miss:.5f}")
        precs = [r["precision"] for r in rows_out]
        results[str(thresh)] = {
            "rows": rows_out,
            "precision_min": min(precs),
            "precision_mean": sum(precs) / len(precs),
            "n_pred_mean": sum(r["n_pred"] for r in rows_out) / len(rows_out),
            "missed_far_max": max(r["missed_far_mass"] for r in rows_out),
        }
        print(f"  TH={thresh} 汇总: prec min/mean = {min(precs):.3f}/"
              f"{sum(precs)/len(precs):.3f}, pred均值 "
              f"{results[str(thresh)]['n_pred_mean']:.1f} 层\n")

    json.dump(results, open(f"{OUT}/e6b_heldout_gate.json", "w"), indent=1)
    print("saved e6b_heldout_gate.json")


if __name__ == "__main__":
    main()
