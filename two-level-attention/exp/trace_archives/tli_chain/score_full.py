# best 臂 12 任务全量打分 → full_scores.json（复用 B7s 时直接读 e71_main_table TLI_B7）
import glob, json, os, sys

ROOT = "/home/wangyuanshuo02/two-level-attention"
sys.path.insert(0, ROOT)
from benchmark.LongBench.eval import scorer

best = open("/tmp/tli_chain/best_arm.txt").read().strip()
core = best.replace("(B7s)", "")
TASKS = ["hotpotqa", "2wikimqa", "musique", "passage_retrieval_en", "qasper",
         "multifieldqa_en", "gov_report", "qmsum", "multi_news", "narrativeqa",
         "triviaqa", "lcc", "repobench"]
BASE = f"{ROOT}/exp/results_longbench/Qwen3-8B"

if best == "mavg(B7s)":
    t = json.load(open(f"{ROOT}/exp/trace/results/e71_main_table.json"))["TLI_B7"]
    scores = {k: v for k, v in t.items() if k != "AVG"}
    scores["AVG"] = t.get("AVG")
    print("(复用 B7s 12 任务分数)")
else:
    d = f"{BASE}/pred_e72f_{core}"
    scores = {}
    for task in TASKS:
        files = sorted(glob.glob(f"{d}/{task}-tli_*.jsonl"))
        if not files:
            print("缺", task); continue
        preds, answers, all_classes = [], [], None
        for line in open(files[-1]):
            rec = json.loads(line)
            preds.append(rec["pred"]); answers.append(rec["answers"]); all_classes = rec["all_classes"]
        scores[task] = round(scorer(task, preds, answers, all_classes), 2)
    if len(scores) == len(TASKS):
        import numpy as np
        scores["AVG"] = round(float(np.mean([scores[t] for t in TASKS])), 2)
json.dump(scores, open("/tmp/tli_chain/full_scores.json", "w"), indent=1)
print(json.dumps(scores, indent=1))
