# Qwen3-32B 泛化复验（论文待办 #5）：A 子空间 + D' 层掩码 + far 总量 gate 判据
# 数据：/tmp/trace/qwen3-32b（7 任务 × 2 样本：4 校准口径 + 3 多跳 gate 任务）
# 三个问题：
#   (1) A 泛化：lowfreq d'=32 块上界 recall ≈ 全维 128？（random/highfreq 崩溃佐证选择性）
#   (2) D' 泛化：层 far-mass 轮廓双峰 + 平均轮廓静态跳过掩码 precision
#   (3) gate 泛化：T_far 多跳任务（musique/qasper/mfq）vs 安全任务 gap 是否
#       复现 8B 的 <0.3 / ≥0.5 判据边界
import os
import glob
import json
import math
import statistics as st
import torch
import torch.nn.functional as F

TRACE = "/tmp/trace/qwen3-32b"
OUT = "/home/wangyuanshuo02/two-level-attention/exp/trace/results"
THRESH = 0.02
BLOCK = 64
GATE_CAL = ["hotpotqa", "narrativeqa", "passage_retrieval_en", "gov_report"]
GATE_MULTIHOP = ["musique", "qasper", "multifieldqa_en"]


def qwen_subspace_idx(d_prime, D=128, mode="lowfreq"):
    d = d_prime // 2
    if mode == "lowfreq":
        return list(range(64 - d, 64)) + list(range(128 - d, 128))
    if mode == "random":
        g = torch.Generator().manual_seed(7)
        return [int(x) for x in torch.randperm(D, generator=g)[:d_prime]]
    if mode == "highfreq":
        return list(range(d)) + list(range(64, 64 + d))
    raise ValueError(mode)


def layer_far(lf):
    """E6 同口径：末行 q 的 per-layer far mass（[64, t-2048) 区，pm 平均）。"""
    d = torch.load(lf, map_location="cuda:0")
    k, q, qpos, S = d["k"].float(), d["q"].float(), d["qpos"].cuda(), d["S"]
    t = qpos[-1].item()
    Hkv, H, D = k.shape[1], q.shape[1], k.shape[-1]
    G = H // Hkv
    qg = q[-1:].reshape(1, Hkv, G, D)
    s = torch.einsum("bhgd,chd->bhgc", qg, k).sum(-2) * (D**-0.5)
    s = s.masked_fill(torch.arange(S, device="cuda").view(1, 1, S) > t, float("-inf"))
    p = torch.softmax(s, dim=-1)
    pm = p.mean(dim=(0, 1))
    far = pm[64 : t - 2048].sum().item()
    sink = pm[:64].sum().item()
    near = pm[t - 2048 :].sum().item()
    del k, q, s, p
    torch.cuda.empty_cache()
    return far, sink, near


def subspace_recall(lf, budget_blk=128):
    """E3 核心消融：d'∈{32,128} × mode∈{lowfreq,random,highfreq} 的块候选 recall。

    双口径与 8B E3（analyze_e3.py）完全对齐：
    - entry recall: per-head (候选块 token ∩ dense top-1024)/1024，head 平均
    - mass recall:  候选并集 token 的 mass 占比（跨 head 汇总 total_mass）
    32B 上 mass 口径易饱和（sink 主导层 cov→1），entry 口径保留判别力。
    """
    d = torch.load(lf, map_location="cuda:0")
    k, q, qpos, S = d["k"].float(), d["q"].float(), d["qpos"].cuda(), d["S"]
    t = qpos[-1].item()
    Hkv, H, D = k.shape[1], q.shape[1], k.shape[-1]
    G = H // Hkv
    qg = q[-1:].view(1, Hkv, G, D)
    s = torch.einsum("bhgd,chd->bhgc", qg, k).sum(-2) * (D**-0.5)
    s = s.masked_fill(torch.arange(S, device="cuda").view(1, 1, S) > t, float("-inf"))
    gt = torch.topk(s, 1024, dim=-1).indices  # [1, Hkv, 1024]
    p_full = torch.softmax(s, dim=-1)
    total_mass = p_full.sum(dim=(0, 1))  # [S]
    nblk = (S + BLOCK - 1) // BLOCK
    blk_end = (torch.arange(nblk, device="cuda") + 1) * BLOCK - 1
    res = {}
    for dp in (32, 128):
        for m in ("lowfreq", "random", "highfreq"):
            dim_idx = qwen_subspace_idx(dp, mode=m)
            idx = torch.tensor(dim_idx, device="cuda")
            ksub = k[..., idx]
            if nblk * BLOCK > S:
                ksub = F.pad(ksub, (0, 0, 0, 0, 0, nblk * BLOCK - S))
            kc = ksub.view(nblk, BLOCK, Hkv, -1)
            kmin, kmax = kc.amin(1), kc.amax(1)
            qsub = q[-1:][..., idx]
            qg_pos = qsub.clamp(min=0).view(1, Hkv, G, -1)
            qg_neg = qsub.clamp(max=0).view(1, Hkv, G, -1)
            sc = (torch.einsum("bhgd,nhd->bhgn", qg_pos, kmax)
                  + torch.einsum("bhgd,nhd->bhgn", qg_neg, kmin)).sum(-2)
            sc = sc.masked_fill(blk_end.view(1, 1, -1) > t, float("-inf"))
            cand_blk = torch.topk(sc, min(budget_blk, nblk), dim=-1).indices[0]
            # entry recall（per-head，E3 同口径）
            entry_r = []
            for h in range(Hkv):
                cand = set()
                for blk in cand_blk[h].tolist():
                    cand.update(range(blk * BLOCK, min((blk + 1) * BLOCK, S)))
                gts = set(gt[0, h].tolist())
                entry_r.append(len(cand & gts) / len(gts))
            # mass recall（并集口径）
            onehot = torch.zeros(Hkv, nblk, dtype=torch.bool, device="cuda")
            onehot.scatter_(1, cand_blk, True)
            mask = onehot.repeat_interleave(BLOCK, 1)
            mask = mask[:, :S] & (torch.arange(S, device="cuda").view(1, S) <= t)
            cov = total_mass[mask.any(0)].sum().item() / total_mass.sum().item()
            res[f"d{dp}_{m}"] = cov
            res[f"d{dp}_{m}_entry"] = sum(entry_r) / len(entry_r)
            del ksub, kc, sc
    del k, q, s, p_full, gt
    torch.cuda.empty_cache()
    return res


def main():
    pdirs = sorted(p for p in glob.glob(f"{TRACE}/lb_*") if os.path.exists(f"{p}/meta.json"))
    assert pdirs, f"no trace under {TRACE}"
    print(f"traces: {len(pdirs)}")
    results = {"subspace": {}, "layers": {}, "gate": {}}
    layer_profiles = {}  # name -> [far per layer]
    for pdir in pdirs:
        name = os.path.basename(pdir)
        lfs = sorted(glob.glob(f"{pdir}/layer*.pt"))
        # (1) 子空间：抽 8 层（均匀）控制时长
        sel_lfs = [lfs[i] for i in sorted(set(int(x) for x in torch.linspace(0, len(lfs) - 1, 8).tolist()))]
        acc = {}
        for lf in sel_lfs:
            r = subspace_recall(lf)
            for k, v in r.items():
                acc.setdefault(k, []).append(v)
        results["subspace"][name] = {k: st.mean(v) for k, v in acc.items()}
        print(f"[A] {name}: mass lowfreq32={st.mean(acc['d32_lowfreq']):.4f} "
              f"full128={st.mean(acc['d128_lowfreq']):.4f} "
              f"rand32={st.mean(acc['d32_random']):.4f} "
              f"hifreq32={st.mean(acc['d32_highfreq']):.4f}")
        print(f"[A] {name}: entry lowfreq32={st.mean(acc['d32_lowfreq_entry']):.4f} "
              f"full128={st.mean(acc['d128_lowfreq_entry']):.4f} "
              f"rand32={st.mean(acc['d32_random_entry']):.4f} "
              f"hifreq32={st.mean(acc['d32_highfreq_entry']):.4f}")
        # (2)+(3) 层轮廓 / gate
        rows = [layer_far(lf) for lf in lfs]
        layer_profiles[name] = [f for f, _, _ in rows]
        task = name.split("_")[1]
        far_total = sum(f for f, _, _ in rows)
        n_skip = sum(1 for f in layer_profiles[name] if f < THRESH)
        sink_av = st.mean([si for _, si, _ in rows])
        near_av = st.mean([ne for _, _, ne in rows])
        results["layers"][name] = {
            "n_layers": len(rows), "n_skippable": n_skip,
            "far_total": far_total, "sink_mean": sink_av, "near_mean": near_av,
        }
        results["gate"][task] = results["gate"].get(task, []) + [far_total]
        print(f"[D'] {name}: 跳层 {n_skip}/{len(rows)}, T_far={far_total:.3f}, "
              f"sink={sink_av:.3f}, near={near_av:.3f}")

    # (2) 平均轮廓静态掩码 precision（D' 全局掩码口径）
    names = list(layer_profiles.keys())
    n_min = min(len(v) for v in layer_profiles.values())
    avg = [st.mean([layer_profiles[nm][i] for nm in names]) for i in range(n_min)]
    skip = [i for i in range(n_min) if avg[i] < THRESH]
    prec_all, miss_all = [], []
    for nm in names:
        true_skip = {i for i in range(n_min) if layer_profiles[nm][i] < THRESH}
        tp = len(set(skip) & true_skip)
        prec = tp / len(skip) if skip else 1.0
        miss = sum(layer_profiles[nm][i] for i in skip if layer_profiles[nm][i] >= THRESH)
        prec_all.append(prec)
        miss_all.append(miss)
    results["avg_mask"] = {
        "n_skip": len(skip), "skip_idx": skip, "threshold": THRESH,
        "precision_mean": st.mean(prec_all), "missed_far_mass_max": max(miss_all),
    }
    print(f"[D'] 平均轮廓: 跳 {len(skip)}/{n_min} 层, precision={st.mean(prec_all):.3f}, "
          f"max 漏 far 质量={max(miss_all):.4f}")

    # (3) gate 判据——per-layer 归一化 far（跨模型可比；8B 为 36 层总和、
    # 32B 为 64 层总和，直接比较会误导）。8B 对照数据来自
    # e6_layer_skip.json / tli_layer_skip_mask_*.json（同 pm 平均口径）
    B8_FAR = {  # task key 与 results["gate"] 一致（name.split("_")[1]）
        "hotpotqa": [0.571, 0.661], "narrativeqa": [7.711],
        "passage": [0.986, 0.759], "gov": [8.948, 10.595],
        "musique": [0.786], "qasper": [0.547], "multifieldqa": [0.946],
    }
    B8_LAYERS = 36
    n32 = results["layers"][next(iter(results["layers"]))]["n_layers"]
    cal = [st.mean(v) for t, v in results["gate"].items() if t in GATE_CAL and v]
    multi = [st.mean(v) for t, v in results["gate"].items() if t in GATE_MULTIHOP and v]
    per_layer = {
        t: {v / n32 for v in []} if False else {
            "far_per_layer_32b": st.mean(v) / n32,
            "far_per_layer_8b": st.mean(B8_FAR[t]) / B8_LAYERS,
        }
        for t, v in results["gate"].items()
    }
    results["gate_summary"] = {
        "cal_tasks_T_far": {t: st.mean(v) for t, v in results["gate"].items() if t in GATE_CAL},
        "multihop_T_far": {t: st.mean(v) for t, v in results["gate"].items() if t in GATE_MULTIHOP},
        "far_per_layer": per_layer,
        "n_layers_32b": n32,
        # 原「0.3 边界」判据（任务级 far 总量）在 8B 上即不干净（hotpotqa
        # 0.57 vs musique 0.79 重叠），32B 同样重叠——判据修正为层轮廓 τ 阈值
        "gap_clean_0.3_boundary": (max(cal) < 0.3 and min(multi) >= 0.3) if cal and multi else None,
    }
    print(f"[gate] 安全任务 T_far(64层和): {[f'{v:.3f}' for v in cal]} (max {max(cal):.3f})")
    print(f"[gate] 多跳任务 T_far(64层和): {[f'{v:.3f}' for v in multi]} (min {min(multi):.3f})")
    for t, d in per_layer.items():
        print(f"[gate] {t}: far/层 32B={d['far_per_layer_32b']:.4f} vs 8B={d['far_per_layer_8b']:.4f}")
    print(f"[gate] 0.3 边界干净复现: {results['gate_summary']['gap_clean_0.3_boundary']}"
          f"（8B 上同口径 hotpotqa 0.57 vs musique 0.79 本就重叠）")

    json.dump(results, open(f"{OUT}/generalize_32b.json", "w"), indent=1)
    print("saved", f"{OUT}/generalize_32b.json")


if __name__ == "__main__":
    main()
