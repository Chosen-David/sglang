# #66 RULER 打分：string_match_all（与 KVCache-Factory/官方 RULER 一致）
# 对每个样本：ground truth 列表逐目标做大小写不敏感子串包含，命中数/
# 目标数；全体样本均值 ×100。汇总 11 任务 × 长度 × 方法 → markdown 表。
# 用法：python -u benchmark/RULER/score_ruler.py [--root exp/results_ruler/Qwen3-8B]
# B08 修复（GPT 审查 2026-10-08）：原版不验证样本数、不隔离轮次——同目录
# 残留旧轮/部分结果的 cell 会混入 AVG（缺任务也照出分）。修复：①逐 cell
# 记录样本数 n，n < min-samples 的 cell 打 WARN 且不计入 AVG（分数仍展示）
# ②每方法统计 33 cell 完整度，缺失 cell 打印警告，AVG 只在完整时可信。
import argparse
import glob
import json
import os

TASKS = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multiquery", "niah_multivalue", "cwe", "fwe", "vt",
]
NTASK = len(TASKS)


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
    ap.add_argument("--min-samples", type=int, default=100,
                    help="cell 样本数低于该值不计入 AVG（默认 100=满格）")
    args = ap.parse_args()

    res = {}   # key "L/method" -> {task: score}
    resn = {}  # key "L/method" -> {task: n}
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
                resn.setdefault(key, {})[task] = len(preds)
                print(f"[{key}] {task}: n={len(preds)} "
                      f"score={res[key][task]}")

    # B08：不完整 cell 警示（不计入 AVG；分数仍展示在表内）
    incomplete = [(k, t, resn[k][t]) for k in resn for t in resn[k]
                  if resn[k][t] < args.min_samples]
    for k, t, n in incomplete:
        print(f"WARN incomplete cell {k}/{t}: n={n} < {args.min_samples} "
              f"(excluded from AVG)")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    json.dump({"scores": res, "n": resn, "incomplete_cells": incomplete},
              open(args.out, "w"), indent=1)

    # markdown：行=任务，列=方法
    methods = sorted({k.split("/", 1)[1] for k in res})
    md = ["| task | " + " | ".join(methods) + " |",
          "|" + "---|" * (len(methods) + 1)]
    for task in TASKS:
        row = [task]
        for m in methods:
            # B08：任务内均值也只用满足样本数门的 cell（长度间求均值）
            vals = [res[k][task] for k in res
                    if k.split("/", 1)[1] == m and task in res[k]
                    and resn[k][task] >= args.min_samples]
            row.append(f"{sum(vals)/len(vals):.2f}" if vals else "-")
        md.append("| " + " | ".join(row) + " |")
    # AVG 行：只统计满足样本数门的 cell；打印每方法完整度
    avg_row = ["AVG"]
    for m in methods:
        ok_cells = [k for k in res if k.split("/", 1)[1] == m
                    and all(resn[k][t] >= args.min_samples for t in resn[k])]
        have_cells = [k for k in res if k.split("/", 1)[1] == m]
        all_v = [v for k in ok_cells for v in res[k].values()]
        avg_row.append(f"{sum(all_v)/len(all_v):.2f}" if all_v else "-")
        n_ok = sum(len(res[k]) for k in ok_cells)
        n_have = sum(len(res[k]) for k in have_cells)
        if n_ok < NTASK * 3:
            print(f"WARN method={m}: only {n_ok}/{NTASK*3} complete "
                  f"cells ({n_have} present incl. partial) — AVG not "
                  f"comparable across methods")
    md.append("| " + " | ".join(avg_row) + " |")
    table = "\n".join(md)
    open(args.out.replace(".json", ".md"), "w").write(table + "\n")
    print(table)
    print("saved", args.out)


if __name__ == "__main__":
    main()
