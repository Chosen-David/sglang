# E71 打分：C0（新主推，α=0 单池 bp128）vs 基线（FullKV/Quest/TIA/TWI/TLI-gated 老口径）
# 用法：python -u exp/trace/run_e71_eval.py [--quick 检查缺什么]
# 原理：eval.py 同一 scorer，为每个方法建软链目录后分别调用
import glob
import json
import os
import sys

ROOT = "/home/wangyuanshuo02/two-level-attention"
sys.path.insert(0, ROOT)
BASE = f"{ROOT}/exp/results_longbench/Qwen3-8B"
TMP = "/tmp/e71_eval"
OUT_JSON = f"{ROOT}/exp/trace/results/e71_main_table.json"

# 方法 → (pred 目录, 文件名模式)
METHODS = {
    "FullKV": (f"{BASE}/pred_1024", "{task}-none-*.jsonl"),
    "Quest": (f"{BASE}/pred_1024", "{task}-quest_64_16-*.jsonl"),
    "TIA": (f"{BASE}/pred_1024", "{task}-tia_64_128_1024_c4-01231105.jsonl"),
    "TWI": (f"{BASE}/pred_1024", "{task}-twi_64_128_0.95-*.jsonl"),
    "TLI_old": (f"{BASE}/pred_1024", "{task}-tli_64_128_1024_c4_ABD-09231659.jsonl"),
    # E71 新主推：C0 = α0 单池 bp128 K2=1024（hotpotqa 冒烟 + 12 任务放量，postfix _c0）
    # 注意：pred_postfix 只进目录名，文件名是 {task}-{method}-{t}.jsonl（t 为时间戳）
    # → 目录已区分配置，模式放宽到任意批次取最新（sorted[-1]）
    "TLI_C0": (f"{BASE}/pred_c0", "{task}-tli_64_128_1024_c4_A-*.jsonl"),
    # E71 B 配置（d8 投影，已知崩坏，仅供 negative result 对照）
    "TLI_B": (f"{BASE}/pred_b3", "{task}-tli_64_128_1024_c4_A-*.jsonl"),
    # E71 B7 修复版分区（α.125/β.25/γ.125，sink 注入+γ 门控双 bug 修复后）
    "TLI_B7": (f"{BASE}/pred_b7", "{task}-tli_64_128_1024_c4_A-*.jsonl"),
    # E72 mavg 冠军臂（α.125/β.375/γ.125）
    "TLI_E72": (f"{BASE}/pred_e72f_mavg", "{task}-tli_64_128_1024_c4_AB-*.jsonl"),
    # E85f：静态 pair e2e 判决（E72 配置 + --tli_static_pair，目录隔离臂）
    "TLI_E85F": (f"{BASE}/pred_e85f", "{task}-tli_*.jsonl"),
    # E81：KVCache-Factory 统一口径 baseline（sink 保送适配版）
    # 注意：pred_kvcf.py 保存用完整 dataset 名（repobench-p-*.jsonl），
    # 而 pred.py 截断为 repobench——pattern 须兼容两种前缀
    "SnapKV": (f"{BASE}/pred_kvcf/snapkv", "{task}*-snapkv-*.jsonl"),
    "H2O": (f"{BASE}/pred_kvcf/h2o", "{task}*-h2o-*.jsonl"),
    "PyramidKV": (f"{BASE}/pred_kvcf/pyramidkv", "{task}*-pyramidkv-*.jsonl"),
    # E89：StreamingLLM（kvcf 适配 patch，静态稀疏对照）
    "StreamingLLM": (f"{BASE}/pred_kvcf/streamingllm", "{task}*-streamingllm-*.jsonl"),
}
TASKS = [
    "hotpotqa", "2wikimqa", "musique", "passage_retrieval_en", "qasper",
    "multifieldqa_en", "gov_report", "qmsum", "multi_news", "narrativeqa",
    "triviaqa", "lcc", "repobench",
]
# 注意：repobench 的 dataset 名是 repobench-p，但文件名经 dataset.split("-")[0]
# 截断为 repobench 前缀——TASKS 键用 repobench 即可匹配所有方法的输出文件

quick = "--quick" in sys.argv
results = {}
for method, (pred_dir, pat) in METHODS.items():
    if not os.path.isdir(pred_dir):
        print(f"[{method}] 目录不存在: {pred_dir}")
        continue
    d = f"{TMP}/pred_{method.lower()}"
    os.makedirs(d, exist_ok=True)
    for f in glob.glob(f"{d}/*"):
        os.remove(f)
    missing = []
    for task in TASKS:
        files = sorted(glob.glob(os.path.join(pred_dir, pat.format(task=task))))
        if not files:
            missing.append(task)
            continue
        os.symlink(files[-1], f"{d}/{task}.jsonl")
    if missing:
        print(f"[{method}] 缺少任务: {missing}")
        if quick:
            continue
    have = [t for t in TASKS if t not in missing]
    if not have:
        continue
    if quick:
        continue
    from benchmark.LongBench.eval import scorer  # noqa
    import numpy as np
    scores = {}
    for task in have:
        predictions, answers, all_classes = [], [], None
        with open(f"{d}/{task}.jsonl") as f:
            for line in f:
                data = json.loads(line)
                predictions.append(data["pred"])
                answers.append(data["answers"])
                all_classes = data["all_classes"]
        scores[task] = scorer(task, predictions, answers, all_classes) if predictions else None
    scores = {t: s for t, s in scores.items() if s is not None}
    have = [t for t in have if t in scores]
    if len(have) == len(TASKS):
        scores["AVG"] = round(float(np.mean([scores[t] for t in TASKS])), 2)
    results[method] = scores
    print(f"== {method} ==")
    print(json.dumps(scores, indent=1))

if quick:
    sys.exit(0)

os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
json.dump(results, open(OUT_JSON, "w"), indent=1)
print("\nsaved", OUT_JSON)

if results:
    md = ["| task | " + " | ".join(results.keys()) + " |",
          "|" + "---|" * (len(results) + 1)]
    for task in TASKS + ["AVG"]:
        row = []
        for m in results:
            row.append(str(results[m].get(task, "-")))
        if any(r != "-" for r in row):
            md.append("| " + task + " | " + " | ".join(row) + " |")
    print("\n".join(md))
    open(f"{ROOT}/exp/trace/results/e71_main_table.md", "w").write("\n".join(md) + "\n")
