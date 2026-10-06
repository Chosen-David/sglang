# E100 tail32 L1 臂 vs full-L1 主表臂的逐样本配对 bootstrap CI（审稿 C1 判决配套）
#
# 背景：审稿 C1「被评测系统≠描述系统」——主表臂 L1 全维 128 无量化 vs 论文描述 tail32。
# E100 tail 臂 13 任务全量（E98 best 配置 + --tli_subspace tail）落袋后运行本脚本。
# 判决口径：tail/full AVG 差的 95% CI 含 0 → 「L1 上界维度口径对主表结论统计不可分」，
# 主表臂可安全换 tail 口径（了结 C1）；不含 0 且为负 → 如实报告并保留 full 口径 + 口径注。
# 顺带输出 tail vs FullKV（论文主 claim 的 tail 口径版），与 e100_bootstrap_ci.json 可并读。
#
# 设计（与 analyze_e100_bootstrap_ci.py 同法）：
#   1. 逐样本打分与官方 scorer() 逐字一致（含 triviaqa 截断）
#   2. 任务内重采样 → 任务均值 → 13 任务等权平均 → percentile 95% CI（B=10000，种子固定）
#   3. answers 逐位配对校验（tail/full 同 pred.py 同数据顺序，防乱序）
#   4. 口径回验：full 侧锚点须复现 e98_full_13tasks.json（50.78/55.44/34.71）
#
# 数据源（只读）：
#   tail: /tmp/e100_tail/pred_E100TAIL_mavg_a0.125_b0.375_g0.625/{task}-tli_*.jsonl
#   full: /tmp/e98_full/pred_E98BEST_mavg_a0.125_b0.375_g0.625/{task}-tli_*.jsonl
#   FullKV: exp/results_longbench/Qwen3-8B/pred_1024/{task}-none-*.jsonl
# 用法：cd /home/wangyuanshuo02/two-level-attention && python -u exp/trace/analyze_e100_tail_ci.py

import glob
import json
import os
import sys

import numpy as np
from scipy import stats

ROOT = "/home/wangyuanshuo02/two-level-attention"
sys.path.insert(0, ROOT)
from benchmark.LongBench.eval import dataset2metric  # noqa: E402

TAIL_DIR = "/tmp/e100_tail/pred_E100TAIL_mavg_a0.125_b0.375_g0.625"
FULL_DIR = "/tmp/e98_full/pred_E98BEST_mavg_a0.125_b0.375_g0.625"
FKV_DIR = f"{ROOT}/exp/results_longbench/Qwen3-8B/pred_1024"
OUT_JSON = f"{ROOT}/exp/trace/results/e100_tail_ci.json"

TASKS = [
    "hotpotqa", "2wikimqa", "musique", "passage_retrieval_en", "qasper",
    "multifieldqa_en", "gov_report", "qmsum", "multi_news", "narrativeqa",
    "triviaqa", "lcc", "repobench",
]

B = 10000
SEED = 20261004
CI_LEVEL = 0.95


def per_sample_scores(task, pred_file):
    """逐样本打分：与官方 scorer() 逐字一致（含 trec/triviaqa/samsum/lsht 截断规则）。"""
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
    """加载 tail/full 逐样本分并配对（answers 逐位校验）。"""
    tail_files = sorted(glob.glob(f"{TAIL_DIR}/{task}-tli_*.jsonl"))
    full_files = sorted(glob.glob(f"{FULL_DIR}/{task}-tli_*.jsonl"))
    assert tail_files, f"tail 缺 {task}"
    assert full_files, f"full 缺 {task}"
    tail = per_sample_scores(task, tail_files[-1])
    full = per_sample_scores(task, full_files[-1])
    assert len(tail) == len(full), f"{task} 样本数不等: {len(tail)} vs {len(full)}"
    ta = [json.loads(l)["answers"] for l in open(tail_files[-1])]
    fa = [json.loads(l)["answers"] for l in open(full_files[-1])]
    assert ta == fa, f"{task} answers 不配对（顺序或内容不一致）"
    return tail, full


def load_fkv(task):
    fkv_files = sorted(glob.glob(f"{FKV_DIR}/{task}-none-*.jsonl"))
    assert fkv_files, f"FullKV 缺 {task}"
    return per_sample_scores(task, fkv_files[-1])


def bootstrap_avg_diff(per_task_diffs, task_list, rng, b=B):
    """任务内重采样 + 任务等权平均的 bootstrap 分布（LongBench 官方口径）。"""
    boot_means = []
    for t in task_list:
        d = per_task_diffs[t]
        idx = rng.integers(0, len(d), size=(b, len(d)))
        boot_means.append(d[idx].mean(axis=1))
    return np.mean(np.stack(boot_means, axis=0), axis=0)


def ci_report(dist):
    lo, hi = np.percentile(dist, [(1 - CI_LEVEL) / 2 * 100, (1 + CI_LEVEL) / 2 * 100])
    return {
        "mean": round(float(dist.mean()) * 100, 2),
        "ci95_lo": round(float(lo) * 100, 2),
        "ci95_hi": round(float(hi) * 100, 2),
        "contains_zero": bool(lo <= 0 <= hi),
    }


def main():
    rng = np.random.default_rng(SEED)
    task_table = {}
    diffs_tf = {}   # tail - full
    diffs_tk = {}   # tail - FullKV
    for task in TASKS:
        tail, full = load_task(task)
        fkv = load_fkv(task)
        assert len(tail) == len(fkv), f"{task} tail/FullKV 样本数不等"
        diffs_tf[task] = tail - full
        diffs_tk[task] = tail - fkv
        task_table[task] = {
            "n": int(len(tail)),
            "tail": round(float(tail.mean()) * 100, 2),
            "full": round(float(full.mean()) * 100, 2),
            "fkv": round(float(fkv.mean()) * 100, 2),
            "tail_minus_full": round(float((tail - full).mean()) * 100, 2),
            "tail_minus_fkv": round(float((tail - fkv).mean()) * 100, 2),
        }

    # ---- 口径回验：full 侧锚点须复现 e98_full_13tasks.json ----
    full_avg = float(np.mean([task_table[t]["full"] for t in TASKS]))
    tail_avg = float(np.mean([task_table[t]["tail"] for t in TASKS]))
    print(f"[口径回验] full AVG = {round(full_avg,2)}（期望 50.78）")
    assert abs(full_avg - 50.78) < 0.02, "full 口径漂移！"
    for t, exp in [("hotpotqa", 55.44), ("musique", 34.71)]:
        assert abs(task_table[t]["full"] - exp) < 0.02, f"{t} full 口径漂移"
    # tail 侧与 e100_tail_full.json（若已落袋）交叉核对
    e100 = f"{ROOT}/exp/trace/results/e100_tail_full.json"
    if os.path.exists(e100):
        ref = json.load(open(e100))["tasks"]
        for t in TASKS:
            assert abs(task_table[t]["tail"] - ref[t]) < 0.02, f"{t} tail 与落袋 JSON 不一致"
        print(f"[口径回验] tail 侧与 e100_tail_full.json 13 任务交叉核对全过（AVG {round(tail_avg,2)}）")

    # ---- bootstrap CI ----
    tf_ci = ci_report(bootstrap_avg_diff(diffs_tf, TASKS, rng))
    tk_ci = ci_report(bootstrap_avg_diff(diffs_tk, TASKS, rng))

    # ---- 任务级 sign test（tail vs full）----
    wins = sum(1 for t in TASKS if task_table[t]["tail_minus_full"] > 0)
    losses = sum(1 for t in TASKS if task_table[t]["tail_minus_full"] < 0)
    sign_p = float(stats.binomtest(wins, wins + losses, 0.5).pvalue) if wins + losses else 1.0

    # ---- 判决 ----
    if tf_ci["contains_zero"]:
        verdict = (
            f"tail-full AVG 差 {tf_ci['mean']:+.2f} 的 95% CI "
            f"[{tf_ci['ci95_lo']:.2f}, {tf_ci['ci95_hi']:.2f}] 含 0："
            f"L1 上界维度口径（tail32 vs 全维 128）对主表结论统计不可分，"
            f"主表臂可换 tail 口径与论文描述系统对齐（了结审稿 C1）；"
            f"tail 口径下 vs FullKV 差 {tk_ci['mean']:+.2f}（CI [{tk_ci['ci95_lo']:.2f}, {tk_ci['ci95_hi']:.2f}]）"
        )
    else:
        direction = "tail 显著更优" if tf_ci["mean"] > 0 else "tail 显著更差"
        verdict = (
            f"tail-full AVG 差 {tf_ci['mean']:+.2f} 的 95% CI "
            f"[{tf_ci['ci95_lo']:.2f}, {tf_ci['ci95_hi']:.2f}] 不含 0（{direction}）："
            f"维度口径不可忽略，论文须如实报告并择优口径 + 保留另一口径消融"
        )

    out = {
        "note": "E100 审稿 C1 判决配套：tail32-L1 臂 vs full-L1 主表臂逐样本配对 bootstrap CI",
        "method": {
            "bootstrap": "任务内有放回重采样逐样本配对差 → 任务均值 → 13 任务等权平均",
            "B": B, "seed": SEED, "ci": "percentile 95%",
            "scoring": "per-sample = max over ground_truths（与 scorer 逐字一致，含 triviaqa 截断）",
            "pairing": "tail/full 同 pred.py 同数据顺序，answers 逐位 assert 校验",
        },
        "claim": {"tail_avg": round(tail_avg, 2), "full_avg": round(full_avg, 2),
                  "diff": round(tail_avg - full_avg, 2)},
        "task_table": task_table,
        "tail_vs_full_ci": tf_ci,
        "tail_vs_fullkv_ci": tk_ci,
        "sign_test_tail_vs_full": {"wins": wins, "losses": losses,
                                   "two_sided_binomial_p": round(sign_p, 4)},
        "verdict": verdict,
    }

    json.dump(out, open(OUT_JSON, "w"), indent=1, ensure_ascii=False)
    print("\n== 主结果 ==")
    print(json.dumps({k: out[k] for k in
                      ["claim", "tail_vs_full_ci", "tail_vs_fullkv_ci",
                       "sign_test_tail_vs_full", "verdict"]},
                     indent=1, ensure_ascii=False))
    print("\nsaved", OUT_JSON)


if __name__ == "__main__":
    main()
