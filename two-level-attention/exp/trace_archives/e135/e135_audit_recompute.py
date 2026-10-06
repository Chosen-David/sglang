#!/usr/bin/env python3
# E135 / EXP0 raw 数据可复算性审计 —— 独立重算脚本（只读，不修改任何实验代码/结果文件）
# 复算对象：E98 主表臂 per-sample 分数、FullKV per-sample 分数、13 任务 AVG、E100 bootstrap CI
# 口径：与 benchmark.LongBench.eval.scorer 逐字一致（per-sample = max over ground_truths，triviaqa 截断）
import glob
import json
import sys

import numpy as np
from scipy import stats

ROOT = "/home/wangyuanshuo02/two-level-attention"
sys.path.insert(0, ROOT)
from benchmark.LongBench.eval import dataset2metric  # noqa: E402

PSI_DIR = "/tmp/e98_full/pred_E98BEST_mavg_a0.125_b0.375_g0.625"
FKV_DIR = f"{ROOT}/exp/results_longbench/Qwen3-8B/pred_1024"
TASKS = ["hotpotqa", "2wikimqa", "musique", "passage_retrieval_en", "qasper",
         "multifieldqa_en", "gov_report", "qmsum", "multi_news", "narrativeqa",
         "triviaqa", "lcc", "repobench"]
B = 10000
SEED = 20261004


def per_sample_scores(task, pred_file):
    predictions, answers, all_classes = [], [], None
    with open(pred_file) as f:
        for line in f:
            data = json.loads(line)
            predictions.append(data["pred"])
            answers.append(data["answers"])
            all_classes = data["all_classes"]
    scores = []
    for prediction, ground_truths in zip(predictions, answers):
        score = 0.0
        if task in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip("\n").split("\n")[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[task](prediction, ground_truth, all_classes=all_classes))
        scores.append(score)
    return np.asarray(scores, dtype=np.float64), answers


def bootstrap_avg_diff(per_task_diffs, task_list, rng):
    boot_means = []
    for t in task_list:
        d = per_task_diffs[t]
        idx = rng.integers(0, len(d), size=(B, len(d)))
        boot_means.append(d[idx].mean(axis=1))
    return np.mean(np.stack(boot_means, axis=0), axis=0)


def main():
    per_task = {}
    pairing_ok = True
    for task in TASKS:
        psi_files = sorted(glob.glob(f"{PSI_DIR}/{task}-tli_*.jsonl"))
        fkv_files = sorted(glob.glob(f"{FKV_DIR}/{task}-none-*.jsonl"))
        assert psi_files and fkv_files, task
        psi, pa = per_sample_scores(task, psi_files[-1])
        fkv, fa = per_sample_scores(task, fkv_files[-1])
        assert len(psi) == len(fkv), task
        assert pa == fa, f"{task} answers 不配对"
        per_task[task] = {
            "n": int(len(psi)),
            "psi": float(np.round(psi.mean() * 100, 2)),
            "fkv": float(np.round(fkv.mean() * 100, 2)),
            "diff": float(np.round((psi - fkv).mean() * 100, 2)),
            "d": psi - fkv,
        }

    psi_avg = round(float(np.mean([per_task[t]["psi"] for t in TASKS])), 2)
    fkv_avg = round(float(np.mean([per_task[t]["fkv"] for t in TASKS])), 2)

    # bootstrap（复现原脚本 rng 消耗顺序：main → heldout → musique → hotpotqa）
    rng = np.random.default_rng(SEED)
    main_dist = bootstrap_avg_diff({t: per_task[t]["d"] for t in TASKS}, TASKS, rng)
    heldout_tasks = [t for t in TASKS if t not in ("musique", "hotpotqa")]
    heldout_dist = bootstrap_avg_diff({t: per_task[t]["d"] for t in TASKS}, heldout_tasks, rng)
    mu_dist = bootstrap_avg_diff({t: per_task[t]["d"] for t in TASKS}, ["musique"], rng)
    hq_dist = bootstrap_avg_diff({t: per_task[t]["d"] for t in TASKS}, ["hotpotqa"], rng)

    def ci(dist):
        lo, hi = np.percentile(dist, [2.5, 97.5])
        return {"mean": round(float(dist.mean()) * 100, 2),
                "lo": round(float(lo) * 100, 2), "hi": round(float(hi) * 100, 2),
                "contains_zero": bool(lo <= 0 <= hi)}

    wins = sum(1 for t in TASKS if per_task[t]["diff"] > 0)
    losses = sum(1 for t in TASKS if per_task[t]["diff"] < 0)
    sign_p = float(stats.binomtest(wins, wins + losses, 0.5).pvalue)

    out = {
        "psi_task_scores": {t: per_task[t]["psi"] for t in TASKS},
        "fkv_task_scores": {t: per_task[t]["fkv"] for t in TASKS},
        "psi_avg": psi_avg, "fkv_avg": fkv_avg, "diff": round(psi_avg - fkv_avg, 2),
        "ns": {t: per_task[t]["n"] for t in TASKS},
        "main_ci": ci(main_dist),
        "heldout_ci": ci(heldout_dist),
        "musique_ci": ci(mu_dist),
        "hotpotqa_ci": ci(hq_dist),
        "sign_test": {"wins": wins, "losses": losses, "p": round(sign_p, 4)},
        "pairing_ok": pairing_ok,
    }
    print(json.dumps(out, indent=1, ensure_ascii=False))
    json.dump(out, open("/tmp/e135_audit_recompute.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
