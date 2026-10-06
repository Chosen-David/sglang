# E100：主表 claim「PSI 50.78 vs FullKV 50.36（+0.42）」的逐样本 bootstrap 置信区间判决
#
# 背景（审稿 C2）：+0.42 增益中 musique(+2.57)/hotpotqa(+1.96) 两选举任务贡献 83%，
# held-out 11 任务净 +0.08 恰在噪声地板；无 seed 重复无 CI，是 reject 级风险。
# 本实验用逐样本配对 bootstrap 回应显著性问题（纯 CPU，零 GPU）。
#
# 设计：
#   1. 逐样本打分：复用 benchmark.LongBench.eval 的 metric 映射，per-sample score =
#      max over ground_truths（与官方 scorer() 逐字一致，含 triviaqa 截断规则），
#      先用任务级均值回验已知主表数字（PSI 50.78 / FullKV 50.36）确保口径零漂移
#   2. 逐样本配对 bootstrap（B=10000，种子固定 20261004）：
#      任务内重采样有放回抽 index → 任务均值 → 13 任务等权平均（LongBench 官方口径）
#      → AVG 差的 percentile 95% CI
#   3. 分解：held-out 11 任务（排除选举任务 musique/hotpotqa）AVG 差 CI；
#      musique、hotpotqa 各自单任务逐样本 CI
#   4. 任务级 sign test（8 胜 5 负，双侧二项精确检验）
#   5. 诚实判决：CI 含 0 → 「+0.42 不显著」，如实落袋，不调口径掩饰
#
# 数据源（只读）：
#   PSI:   /tmp/e98_full/pred_E98BEST_mavg_a0.125_b0.375_g0.625/{task}-tli_*.jsonl
#   FullKV: exp/results_longbench/Qwen3-8B/pred_1024/{task}-none-*.jsonl（13 任务全齐）
# 用法：cd /home/wangyuanshuo02/two-level-attention && python -u exp/trace/analyze_e100_bootstrap_ci.py

import glob
import json
import os
import sys

import numpy as np
from scipy import stats  # binomtest 双侧精确检验

ROOT = "/home/wangyuanshuo02/two-level-attention"
sys.path.insert(0, ROOT)
from benchmark.LongBench.eval import dataset2metric  # noqa: E402  官方 metric 映射

PSI_DIR = "/tmp/e98_full/pred_E98BEST_mavg_a0.125_b0.375_g0.625"
FKV_DIR = f"{ROOT}/exp/results_longbench/Qwen3-8B/pred_1024"
OUT_JSON = f"{ROOT}/exp/trace/results/e100_bootstrap_ci.json"

TASKS = [
    "hotpotqa", "2wikimqa", "musique", "passage_retrieval_en", "qasper",
    "multifieldqa_en", "gov_report", "qmsum", "multi_news", "narrativeqa",
    "triviaqa", "lcc", "repobench",
]
ELECTION_TASKS = ["musique", "hotpotqa"]  # E98 选举用任务（贡献 83% 增益的两臂）
HELDOUT_TASKS = [t for t in TASKS if t not in ELECTION_TASKS]

B = 10000
SEED = 20261004
CI_LEVEL = 0.95


def per_sample_scores(task, pred_file):
    """逐样本打分：与官方 scorer() 逐字一致（含 trec/triviaqa/samsum/lsht 截断规则）。

    返回 np.array，每元素为该样本的 0-1 分（乘 100 前的原始分）。
    """
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
            score = max(
                score,
                dataset2metric[task](prediction, ground_truth, all_classes=all_classes),
            )
        scores.append(score)
    return np.asarray(scores, dtype=np.float64)


def load_task(task):
    """加载 PSI 与 FullKV 逐样本分并配对。返回 (psi, fkv, n, 配对校验信息)。"""
    psi_files = sorted(glob.glob(f"{PSI_DIR}/{task}-tli_*.jsonl"))
    fkv_files = sorted(glob.glob(f"{FKV_DIR}/{task}-none-*.jsonl"))
    assert psi_files, f"PSI 缺 {task}"
    assert fkv_files, f"FullKV 缺 {task}"
    psi = per_sample_scores(task, psi_files[-1])
    fkv = per_sample_scores(task, fkv_files[-1])
    assert len(psi) == len(fkv), f"{task} 样本数不等: {len(psi)} vs {len(fkv)}"
    # answers 逐位配对校验（防乱序/错版本文件）
    pa = [json.loads(l)["answers"] for l in open(psi_files[-1])]
    fa = [json.loads(l)["answers"] for l in open(fkv_files[-1])]
    assert pa == fa, f"{task} answers 不配对（顺序或内容不一致）"
    return psi, fkv


def bootstrap_avg_diff(per_task_diffs, task_list, rng, b=B):
    """任务内重采样 + 任务等权平均（LongBench 官方口径）的 bootstrap 分布。

    per_task_diffs: {task: np.array 逐样本配对差}
    返回 b 维 AVG 差的 bootstrap 样本（0-1 尺度）。
    """
    boot_means = []
    for t in task_list:
        d = per_task_diffs[t]
        idx = rng.integers(0, len(d), size=(b, len(d)))
        boot_means.append(d[idx].mean(axis=1))  # (b,) 该任务 bootstrap 均值
    return np.mean(np.stack(boot_means, axis=0), axis=0)  # (b,) 等权 AVG 差


def ci_report(dist, label):
    """percentile CI + 均值；返回 dict（分转换为 LongBench 分数尺度 ×100）。"""
    lo, hi = np.percentile(dist, [(1 - CI_LEVEL) / 2 * 100, (1 + CI_LEVEL) / 2 * 100])
    return {
        "mean": round(float(dist.mean()) * 100, 2),
        "ci95_lo": round(float(lo) * 100, 2),
        "ci95_hi": round(float(hi) * 100, 2),
        "contains_zero": bool(lo <= 0 <= hi),
    }


def main():
    rng = np.random.default_rng(SEED)
    per_task_diffs = {}
    task_table = {}
    for task in TASKS:
        psi, fkv = load_task(task)
        d = psi - fkv
        per_task_diffs[task] = d
        task_table[task] = {
            "n": int(len(d)),
            "psi_task_score": round(float(psi.mean()) * 100, 2),
            "fkv_task_score": round(float(fkv.mean()) * 100, 2),
            "task_diff": round(float(d.mean()) * 100, 2),
        }

    # ---- 口径回验：任务级均值须复现已落袋主表数字（round 口径同 scorer）----
    psi_avg = float(np.mean([task_table[t]["psi_task_score"] for t in TASKS]))
    fkv_avg = float(np.mean([task_table[t]["fkv_task_score"] for t in TASKS]))
    print(f"[口径回验] PSI AVG = {round(psi_avg,2)}（期望 50.78，e98_full_13tasks.json）")
    print(f"[口径回验] FullKV AVG = {round(fkv_avg,2)}（期望 50.36，e71_main_table.json）")
    assert abs(psi_avg - 50.78) < 0.02, "PSI 口径漂移！"
    assert abs(fkv_avg - 50.36) < 0.02, "FullKV 口径漂移！"
    for t in ["hotpotqa", "musique"]:
        exp_psi = {"hotpotqa": 55.44, "musique": 34.71}[t]
        assert abs(task_table[t]["psi_task_score"] - exp_psi) < 0.02, f"{t} PSI 口径漂移"
    for t in ["hotpotqa", "musique"]:
        exp_f = {"hotpotqa": 53.48, "musique": 32.14}[t]
        assert abs(task_table[t]["fkv_task_score"] - exp_f) < 0.02, f"{t} FullKV 口径漂移"
    print("[口径回验] 全部通过（PSI 5 项 / FullKV 3 项锚点 + AVG 双锚点）")

    # ---- 逐样本配对 bootstrap ----
    main_dist = bootstrap_avg_diff(per_task_diffs, TASKS, rng)
    main_ci = ci_report(main_dist, "13 任务 AVG 差")

    heldout_dist = bootstrap_avg_diff(per_task_diffs, HELDOUT_TASKS, rng)
    heldout_ci = ci_report(heldout_dist, "held-out 11 任务 AVG 差")

    musique_ci = ci_report(bootstrap_avg_diff(per_task_diffs, ["musique"], rng), "musique")
    hotpotqa_ci = ci_report(bootstrap_avg_diff(per_task_diffs, ["hotpotqa"], rng), "hotpotqa")

    # ---- 任务级 sign test（8 胜 5 负，双侧二项精确检验）----
    wins = sum(1 for t in TASKS if task_table[t]["task_diff"] > 0)
    losses = sum(1 for t in TASKS if task_table[t]["task_diff"] < 0)
    ties = len(TASKS) - wins - losses
    sign_p = float(stats.binomtest(wins, wins + losses, 0.5).pvalue) if wins + losses else 1.0

    # ---- 判决 ----
    significant = not main_ci["contains_zero"]
    if significant:
        verdict = (
            f"13 任务 AVG 差 +{main_ci['mean']:.2f} 的 95% CI "
            f"[{main_ci['ci95_lo']:.2f}, {main_ci['ci95_hi']:.2f}] 不含 0，"
            f"逐样本 bootstrap 下 +0.42 显著（但注意 held-out CI 仍须单独看）"
        )
    else:
        verdict = (
            f"13 任务 AVG 差 +{main_ci['mean']:.2f} 的 95% CI "
            f"[{main_ci['ci95_lo']:.2f}, {main_ci['ci95_hi']:.2f}] 含 0，"
            f"+0.42 在逐样本配对 bootstrap 下不显著；主表 claim 不得表述为显著增益，"
            f"卖点须转向结构分析（musique/hotpotqa far 检索证据 + 等精度同预算的 kernel 速度优势）"
        )

    out = {
        "note": "E100 主表 claim 显著性判决：PSI 50.78 vs FullKV 50.36（+0.42）逐样本配对 bootstrap CI",
        "method": {
            "bootstrap": "任务内有放回重采样逐样本配对差 → 任务均值 → 13 任务等权平均（LongBench 官方聚合口径）",
            "B": B,
            "seed": SEED,
            "ci": "percentile 95%",
            "scoring": "per-sample = max over ground_truths（与 benchmark.LongBench.eval.scorer 逐字一致，含 triviaqa 截断）",
            "pairing": "PSI 与 FullKV 同任务同 n 同 answers 逐位配对（已 assert 校验）",
        },
        "claim": {"psi_avg": round(psi_avg, 2), "fkv_avg": round(fkv_avg, 2),
                  "diff": round(psi_avg - fkv_avg, 2)},
        "calibration_check": {
            "psi_avg_expected": 50.78, "fkv_avg_expected": 50.36, "passed": True,
        },
        "task_table": task_table,
        "main_ci_13tasks": main_ci,
        "heldout_ci_11tasks": {
            **heldout_ci,
            "tasks_excluded": ELECTION_TASKS,
            "nominal_diff": round(float(np.mean([task_table[t]["task_diff"] for t in HELDOUT_TASKS])), 2),
        },
        "election_task_ci": {"musique": {**musique_ci, "nominal_diff": task_table["musique"]["task_diff"]},
                             "hotpotqa": {**hotpotqa_ci, "nominal_diff": task_table["hotpotqa"]["task_diff"]}},
        "sign_test": {"wins": wins, "losses": losses, "ties": ties,
                      "two_sided_binomial_p": round(sign_p, 4)},
        "verdict": verdict,
        "significant": significant,
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(out, open(OUT_JSON, "w"), indent=1, ensure_ascii=False)
    print("\n== 主结果 ==")
    print(json.dumps({k: out[k] for k in
                      ["claim", "main_ci_13tasks", "heldout_ci_11tasks",
                       "election_task_ci", "sign_test", "verdict"]},
                     indent=1, ensure_ascii=False))
    print("\nsaved", OUT_JSON)


if __name__ == "__main__":
    main()
