# #66 RULER 打分：string_match_all（与 KVCache-Factory/官方 RULER 一致）
# 对每个样本：ground truth 列表逐目标做大小写不敏感子串包含，命中数/
# 目标数；全体样本均值 ×100。汇总 11 任务 × 长度 × 方法 → markdown 表。
# 用法：python -u benchmark/RULER/score_ruler.py [--root exp/results_ruler/Qwen3-8B]
import argparse
import glob
import json
import os

TASKS = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multiquery", "niah_multivalue", "cwe", "fwe", "vt",
]


def string_match_all(preds, refs):
    total = 0.0
    for pred, ref in zip(preds, refs):
        hit = sum(1.0 if r.lower() in pred.lower() else 0.0 for r in ref)
        total += hit / len(ref) if ref else 0.0
    return total / len(preds) * 100 if preds else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="exp/results_ruler/Qwen3-8B")
    ap.add_argument("--pred-postfix", default="_1024")
    ap.add_argument("--out", default="exp/results_ruler/ruler_table.json")
    args = ap.parse_args()

    res = {}
    for L_dir in sorted(glob.glob(os.path.join(args.root, "L*"))):
        L = os.path.basename(L_dir)
        pred_dir = os.path.join(L_dir, f"pred{args.pred_postfix}")
        if not os.path.isdir(pred_dir):
            continue
        for task in TASKS:
            files = sorted(glob.glob(
                os.path.join(pred_dir, f"{task}-*.jsonl")))
            if not files:
                continue
            for f in files:
                method = os.path.basename(f).replace(f"{task}-", "") \
                    .rsplit("-", 1)[0]
                preds, refs = [], []
                with open(f) as fp:
                    for line in fp:
                        d = json.loads(line)
                        preds.append(d["pred"])
                        refs.append(d["answers"])
                if not preds:
                    continue
                key = f"{L}/{method}"
                res.setdefault(key, {})[task] = round(
                    string_match_all(preds, refs), 2)
                print(f"[{key}] {task}: n={len(preds)} "
                      f"score={res[key][task]}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)

    # markdown：行=任务，列=方法
    methods = sorted({k.split("/", 1)[1] for k in res})
    md = ["| task | " + " | ".join(methods) + " |",
          "|" + "---|" * (len(methods) + 1)]
    for task in TASKS:
        row = [task]
        for m in methods:
            vals = [res[k][task] for k in res
                    if k.split("/", 1)[1] == m and task in res[k]]
            row.append(f"{sum(vals)/len(vals):.2f}" if vals else "-")
        md.append("| " + " | ".join(row) + " |")
    # AVG 行
    avg_row = ["AVG"]
    for m in methods:
        all_v = [v for k in res if k.split("/", 1)[1] == m
                 for v in res[k].values()]
        avg_row.append(f"{sum(all_v)/len(all_v):.2f}" if all_v else "-")
    md.append("| " + " | ".join(avg_row) + " |")
    table = "\n".join(md)
    open(args.out.replace(".json", ".md"), "w").write(table + "\n")
    print(table)
    print("saved", args.out)


if __name__ == "__main__":
    main()
