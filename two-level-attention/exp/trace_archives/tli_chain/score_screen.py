# 快筛打分：各臂 hotpotqa/musique F1 + B7s(mavg 基准) → screen_scores.json
import glob, json, os, sys

ROOT = "/home/wangyuanshuo02/two-level-attention"
sys.path.insert(0, ROOT)
from benchmark.LongBench.eval import scorer

meta = json.load(open("/tmp/tli_chain/screen_arms.json"))
BASE = f"{ROOT}/exp/results_longbench/Qwen3-8B"
dirs = {"mavg(B7s)": (f"{BASE}/pred_b7", "{t}-tli_*.jsonl")}
for arm in meta:
    if meta[arm].get("reuse_b7s"):
        continue
    dirs[arm] = (f"{BASE}/pred_e72_{arm}", "{t}-tli_*.jsonl")

results = {}
for arm, (d, pat) in dirs.items():
    scores = {}
    for t in ("hotpotqa", "musique"):
        files = sorted(glob.glob(os.path.join(d, pat.format(t=t))))
        if not files:
            scores[t] = None
            continue
        preds, answers, all_classes = [], [], None
        for line in open(files[-1]):
            rec = json.loads(line)
            preds.append(rec["pred"]); answers.append(rec["answers"]); all_classes = rec["all_classes"]
        scores[t] = round(scorer(t, preds, answers, all_classes), 2)
    if all(v is not None for v in scores.values()):
        scores["screen_avg"] = round((scores["hotpotqa"] + scores["musique"]) / 2, 2)
    results[arm] = scores
    print(arm, scores)

json.dump(results, open("/tmp/tli_chain/screen_scores.json", "w"), indent=1)
ranked = sorted(((v["screen_avg"], k) for k, v in results.items() if "screen_avg" in v), reverse=True)
print("\n== 快筛排名 ==")
for s, k in ranked:
    print(f"  {k:14s} {s}")
open("/tmp/tli_chain/best_arm.txt", "w").write(ranked[0][1] + "\n")
print("best arm ->", ranked[0][1])
