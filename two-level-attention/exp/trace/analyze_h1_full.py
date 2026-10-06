# H1 完整版：全部 trace（含真实 LongBench）的 sink/near/far 质量分解 + 逐层异质性
import os
import torch
import glob
import json
import statistics as st

TRACE = "/tmp/trace/qwen3-8b"
OUT = "/home/wangyuanshuo02/two-level-attention/exp/trace/results"


def main():
    results = {}
    far_profiles = {}
    for pdir in sorted(glob.glob(f"{TRACE}/*")):
        name = os.path.basename(pdir)
        if not os.path.exists(f"{pdir}/meta.json"):
            continue
        sink_l, near_l, far_l, cov1024_l = [], [], [], []
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
            sink_l.append(pm[:64].sum().item())
            near_l.append(pm[t - 2048:t].sum().item())
            far_l.append(pm[64:t - 2048].sum().item())
            gt = torch.topk(s, 1024, dim=-1).indices[0]
            cov1024_l.append(p[0].gather(-1, gt).sum().item() / p.sum().item())
            del k, q, s, p
            torch.cuda.empty_cache()
        results[name] = {
            "sink_mean": st.mean(sink_l), "near_mean": st.mean(near_l),
            "far_mean": st.mean(far_l), "far_max": max(far_l),
            "layers_far_gt_0.02": sum(1 for x in far_l if x > 0.02),
            "n_layers": len(far_l),
            "dense_top1024_cov": st.mean(cov1024_l),
            "far_profile": far_l,
        }
        far_profiles[name] = far_l
        print(f"{name}: sink={st.mean(sink_l):.3f} near={st.mean(near_l):.3f} "
              f"far={st.mean(far_l):.3f} (max {max(far_l):.2f}, >0.02 的层 {sum(1 for x in far_l if x>0.02)}/{len(far_l)}) "
              f"dense1024cov={st.mean(cov1024_l):.4f}")
    # 跨 prompt 的层轮廓相关性矩阵
    names = list(far_profiles.keys())
    corr = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = far_profiles[names[i]], far_profiles[names[j]]
            n = min(len(a), len(b))
            ma, mb = sum(a) / n, sum(b) / n
            cov = sum((a[x] - ma) * (b[x] - mb) for x in range(n))
            va = sum((a[x] - ma) ** 2 for x in range(n))
            vb = sum((b[x] - mb) ** 2 for x in range(n))
            import math
            corr[f"{names[i]}~{names[j]}"] = cov / math.sqrt(va * vb) if va > 0 and vb > 0 else 0.0
    results["layer_profile_corr"] = corr
    import json
    json.dump(results, open(f"{OUT}/h1_full_decomposition.json", "w"), indent=1)
    print("层轮廓相关性（远端质量）:")
    for k2, v in sorted(corr.items()):
        print(f"  {k2}: {v:.3f}")


if __name__ == "__main__":
    main()
