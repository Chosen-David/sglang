# E87 e2e 打分：top-σ 臂 vs twolvl 基线（hotpotqa/musique screen 口径）
# 用法：python -u exp/trace/run_e87_eval.py
import glob
import json
import os
import sys

ROOT = "/home/wangyuanshuo02/two-level-attention"
sys.path.insert(0, ROOT)
E2E = "/tmp/e87_e2e"
OUT_JSON = f"{ROOT}/exp/trace/results/e87_e2e_screen.json"
TASKS = ["hotpotqa", "musique"]

# 臂 → pred 目录（pred_postfix 只进目录名）
ARMS = {
    "near_sigma_2": f"{E2E}/pred_signear_2",
    "near_sigma_8": f"{E2E}/pred_signear_8",
    "near_sigma_32": f"{E2E}/pred_signear_32",
    "mid_sigma_8": f"{E2E}/pred_sigmid_8",
    "far_sigma_8": f"{E2E}/pred_sigfar_8",
    "far_sigma_32": f"{E2E}/pred_sigfar_32",
    "mid_sigma_32": f"{E2E}/pred_sigmid_32",
}

results = {}
for arm, pred_dir in ARMS.items():
    if not os.path.isdir(pred_dir):
        print(f"[{arm}] 目录不存在，跳过")
        continue
    from benchmark.LongBench.eval import scorer  # noqa
    scores = {}
    for task in TASKS:
        files = sorted(glob.glob(os.path.join(pred_dir, f"{task}-tli_*.jsonl")))
        if not files:
            print(f"[{arm}] 缺 {task}")
            continue
        predictions, answers, all_classes = [], [], None
        with open(files[-1]) as f:
            for line in f:
                data = json.loads(line)
                predictions.append(data["pred"])
                answers.append(data["answers"])
                all_classes = data["all_classes"]
        if predictions:
            scores[task] = round(scorer(task, predictions, answers, all_classes), 2)
    if scores:
        results[arm] = scores
        print(f"== {arm} == {json.dumps(scores)}")

json.dump(results, open(OUT_JSON, "w"), indent=1)
print("saved ->", OUT_JSON)
# twolvl 参照（B7s 严格口径同配置）：hotpotqa 54.23 / musique 32.82
print("参照 twolvl(B7s): hotpotqa 54.23 / musique 32.82; C0 单池: 54.93 / 29.74")
