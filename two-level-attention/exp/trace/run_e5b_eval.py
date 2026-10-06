# E5b 打分：TLI vs baseline（none/quest/tia_c4/twi）
# 用法：python -u exp/trace/run_e5b_eval.py
# 原理：eval.py 会给整个 pred 目录打分，所以为每个方法建软链目录后分别调用
import glob
import json
import os
import sys

ROOT = "/home/wangyuanshuo02/two-level-attention"
sys.path.insert(0, ROOT)
PRED = f"{ROOT}/exp/results_longbench/Qwen3-8B/pred_1024"
TMP = "/tmp/e5b_eval"
OUT_JSON = f"{ROOT}/exp/trace/results/e5b_main_table.json"

# 方法 → 文件名模式（tia 用 _c4 同口径；排除 async 变体）
METHODS = {
    "FullKV": "{task}-none-*.jsonl",
    "Quest": "{task}-quest_64_16-*.jsonl",
    "TIA": "{task}-tia_64_128_1024_c4-01231105.jsonl",
    "TWI": "{task}-twi_64_128_0.95-*.jsonl",
    "TLI": "{task}-tli_64_128_1024_c4_ABD-09231659.jsonl",
}
TASKS = [
    "hotpotqa", "2wikimqa", "musique", "passage_retrieval_en", "qasper",
    "multifieldqa_en", "gov_report", "qmsum", "multi_news", "narrativeqa",
    "triviaqa", "lcc", "repobench",
]  # 与 run_e5b.sh 完全一致的 13 任务

results = {}
for method, pat in METHODS.items():
    # 建软链目录（eval.py 从目录名取 pred_postfix）
    tag = f"_{method.lower()}"
    d = f"{TMP}/pred{tag}"
    os.makedirs(d, exist_ok=True)
    for f in glob.glob(f"{d}/*"):
        os.remove(f)
    missing = []
    for task in TASKS:
        files = sorted(glob.glob(os.path.join(PRED, pat.format(task=task))))
        if not files:
            missing.append(task)
            continue
        os.symlink(files[-1], f"{d}/{task}.jsonl")
    if missing:
        print(f"[{method}] 缺少任务: {missing}")
        continue
    # 直接调用 scorer 打分（与 eval.py 完全同一实现）
    scores = {}
    from benchmark.LongBench.eval import scorer  # noqa
    import numpy as np
    for task in TASKS:
        predictions, answers, all_classes = [], [], None
        with open(f"{d}/{task}.jsonl") as f:
            for line in f:
                data = json.loads(line)
                predictions.append(data["pred"])
                answers.append(data["answers"])
                all_classes = data["all_classes"]
        scores[task] = scorer(task, predictions, answers, all_classes)
    avg = round(float(np.mean([scores[t] for t in TASKS])), 2)
    scores["AVG"] = avg
    results[method] = scores
    print(f"== {method} ==")
    print(json.dumps(scores, indent=1))

os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
json.dump(results, open(OUT_JSON, "w"), indent=1)
print("\nsaved", OUT_JSON)

# markdown 主表
md = ["| task | " + " | ".join(results.keys()) + " |",
      "|" + "---|" * (len(results) + 1)]
for task in TASKS + ["AVG"]:
    md.append("| " + task + " | " + " | ".join(
        str(results[m][task]) for m in results) + " |")
print("\n".join(md))
open(f"{ROOT}/exp/trace/results/e5b_main_table.md", "w").write("\n".join(md) + "\n")
