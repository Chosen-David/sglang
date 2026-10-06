# #60 E5b 双臂打分：gate-on vs gate-off（musique/qasper/multifieldqa_en 全量 200）
# 输入：/home/wangyuanshuo02/sglang/pred_dyngate_{on,off}/<task>.jsonl
# 打分与 E5b 主表同一实现（benchmark.LongBench.eval.scorer）
import json
import sys

ROOT = "/home/wangyuanshuo02/two-level-attention"
sys.path.insert(0, ROOT)
sys.path.insert(0, "/home/wangyuanshuo02/.local/pylibs")

TASKS = ["musique", "qasper", "multifieldqa_en"]
OUT = "/home/wangyuanshuo02/sglang/pred_dyngate_score.json"


def score_dir(arm):
    from benchmark.LongBench.eval import scorer
    scores = {}
    for task in TASKS:
        path = f"/home/wangyuanshuo02/sglang/pred_dyngate_{arm}/{task}.jsonl"
        preds, answers, all_classes = [], [], None
        for line in open(path):
            d = json.loads(line)
            preds.append(d["pred"])
            answers.append(d["answers"])
            all_classes = d["all_classes"]
        scores[task] = round(scorer(task, preds, answers, all_classes), 2)
        print(f"[{arm}/{task}] n={len(preds)} score={scores[task]}", flush=True)
    return scores


def main():
    import numpy as np
    res = {}
    for arm in ["on", "off"]:
        try:
            res[f"gate_{arm}"] = score_dir(arm)
        except FileNotFoundError as e:
            print(f"[gate_{arm}] missing: {e}")
    # E5b 主表参考（transformers 平台）：主表 JSON 无 TLI 键时取 TIA 行
    # （B' 在三任务上 ≈ TIA，qasper 完全同分）；另有 sglang triton dense
    # 平台基线（pred_e5b_triton/，musique 30.32）作口径定界参考
    try:
        ref = {t: json.load(
            open(f"{ROOT}/exp/trace/results/e5b_main_table.json"))["TLI"][t] for t in TASKS}
    except KeyError:
        ref = {t: json.load(
            open(f"{ROOT}/exp/trace/results/e5b_main_table.json"))["TIA"][t] for t in TASKS}
    res["e5b_TIA_ref_transformers"] = ref
    for arm in ["on", "off"]:
        if f"gate_{arm}" in res:
            res[f"gate_{arm}"]["AVG"] = round(float(np.mean(
                [res[f"gate_{arm}"][t] for t in TASKS])), 2)
    res["e5b_TIA_ref_transformers"]["AVG"] = round(float(np.mean([ref[t] for t in TASKS])), 2)
    # sglang 平台 triton dense 基线（musique 定界参考，若存在）
    try:
        tr = []
        for line in open("/home/wangyuanshuo02/sglang/pred_e5b_triton/musique.jsonl"):
            tr.append(json.loads(line))
        res["sglang_triton_dense_musique"] = 30.32  # 2026-09-27 打分落盘值
    except FileNotFoundError:
        pass
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res, indent=1))
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
