# #66 RULER 打分：string_match_all（与 KVCache-Factory/官方 RULER 一致）
# 对每个样本：ground truth 列表逐目标做大小写不敏感子串包含，命中数/
# 目标数；全体样本均值 ×100。汇总 11 任务 × 长度 × 方法 → markdown 表。
# 用法：python -u benchmark/RULER/score_ruler.py [--root exp/results_ruler/Qwen3-8B]
# B08 修复（GPT 审查 2026-10-08）：原版不验证样本数、不隔离轮次——同目录
# 残留旧轮/部分结果的 cell 会混入 AVG（缺任务也照出分）。修复：①逐 cell
# 记录样本数 n，n < min-samples 的 cell 打 WARN 且不计入 AVG（分数仍展示）
# ②每方法统计 33 cell 完整度，缺失 cell 打印警告，AVG 只在完整时可信。
# E116c（GPT 0826 审计 TL-RULER-SAMPLE-GATE-026）fail-closed 门禁：
#   ① 同 task/method-key 检测到 >1 个文件 → SystemExit 非零（列出冲突文件，
#      要求先显式合并）；--merge-best 时按 行数最多（并列取文件名时间戳
#      最新）选 best-file，落盘单一规范文件 {task}-{method}-merged.jsonl
#      并打印每格选择的文件名+SHA256 后再评分
#   ② --manifest <json>：{task: {"ids": [...], "answers_sha": {id: sha}}}，
#      给定时做集合闭包校验（缺/多/重复/answers hash 错配全部 fail closed）；
#      文件无 _id 却给了 manifest → fail closed（拒绝静默跳过校验）
#   ③ --expect-tasks N：每个方法 key 必须覆盖预期任务集合，缺任一任务 →
#      非零退出，不输出正式结果
#   ④ 评分结果 JSON 每格记录源文件名 + 文件 SHA256（结果可追溯）
# 兼容：无 manifest 且同键单文件时行为与旧版一致（老数据不带 _id 不强制）。
import argparse
import glob
import hashlib
import json
import os
import shutil

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


def _answers_sha16(ans):
    """answers canonical JSON SHA256 前 16 位（与 pred_ruler._answers_sha 及
    LongBench manifest 口径一致）。"""
    return hashlib.sha256(json.dumps(
        ans, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def _file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _nlines(path):
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def _ts_of(fname):
    """文件名最后一段（时间戳）；merged 规范文件返回 'merged'（字典序 >
    纯数字时间戳，行数并列时天然胜出）。"""
    return fname[:-len(".jsonl")].rsplit("-", 1)[-1]


def _resolve_group(files, task, method, key, merge_best):
    """同 (L, task, method) 键的文件集合 → 单一评分文件。
    >1 文件且未开 --merge-best → fail closed；
    开 --merge-best → best-file（行数最多，并列取时间戳最新），并复制为
    单一规范文件 {task}-{method}-merged.jsonl 落盘（打印选择依据）。"""
    if len(files) == 1:
        return files[0]
    listing = "\n".join(
        f"    {os.path.basename(f)}  n={_nlines(f)}  sha256={_file_sha256(f)}"
        for f in files)
    if not merge_best:
        raise SystemExit(
            f"[GATE-FAIL] {key}/{task}: 同 task/method-key 检测到 "
            f"{len(files)} 个文件——fail closed，拒绝静默覆盖（后读覆盖先读"
            f"的旧行为已废除）。冲突文件清单：\n{listing}\n"
            f"请先显式合并（--merge-best：行数最多、并列取时间戳最新）或清理"
            f"重复文件后重跑。"
        )
    # --merge-best：best-file 选择 + 规范文件落盘
    best = max(files, key=lambda f: (_nlines(f), _ts_of(os.path.basename(f))))
    canon = os.path.join(os.path.dirname(best),
                         f"{task}-{method}-merged.jsonl")
    if os.path.exists(canon) and os.path.samefile(best, canon):
        pass  # 幂等重跑：best 恰为已落盘规范文件本身，无需拷贝
    else:
        shutil.copyfile(best, canon)
    skipped = [f for f in files if f != best]
    print(f"[merge-best] {key}/{task}: 选择 best-file "
          f"{os.path.basename(best)} (n={_nlines(best)}, "
          f"sha256={_file_sha256(best)}) → 规范文件 "
          f"{os.path.basename(canon)}；跳过 "
          f"{[os.path.basename(f) for f in skipped]}")
    return canon


def _score_cell(path, task, key, manifest):
    """读单一文件 → (score, n)。含 _id 重复检测与 manifest 闭包校验。"""
    records = [json.loads(l) for l in open(path, encoding="utf-8")]
    if not records:
        return None, 0
    preds = [d["pred"] for d in records]
    refs = [d["answers"] for d in records]
    ids = [d.get("_id") for d in records]
    have_ids = [i for i in ids if i is not None]
    # ---- 门禁 1：_id 无重复 / 不许部分行缺失（新数据全带 _id；无 _id 的
    # 旧文件且未给 manifest 时跳过，保持旧行为兼容）----
    if have_ids:
        if len(have_ids) != len(ids):
            raise SystemExit(
                f"[GATE-FAIL] {os.path.basename(path)}: 部分行缺失 _id"
                f"（{len(have_ids)}/{len(ids)}）——fail closed，不写结果"
            )
        if len(ids) != len(set(ids)):
            dup = sorted({i for i in ids if ids.count(i) > 1})[:5]
            raise SystemExit(
                f"[GATE-FAIL] {os.path.basename(path)}: 检测到重复 _id"
                f"（示例 {dup}）——fail closed，不写结果"
            )
        # 行内 _answers_sha 与 answers 重算不一致 = 记录被篡改/混行
        # （只依赖记录自身；老数据无该字段不受影响，故无条件执行）
        for d, a in zip(records, refs):
            if "_answers_sha" in d and d["_answers_sha"] != _answers_sha16(a):
                raise SystemExit(
                    f"[GATE-FAIL] {os.path.basename(path)}: _id={d.get('_id')}"
                    f" 行内 _answers_sha 与 answers 重算不一致（记录被篡改/"
                    f"混行）——fail closed，不写结果"
                )
    # ---- 门禁 2：manifest 集合闭包 + answers hash 一致 ----
    if manifest is not None:
        if not have_ids:
            raise SystemExit(
                f"[GATE-FAIL] {os.path.basename(path)}: 预测缺 _id 字段，"
                f"无法对 manifest 校验——fail closed（拒绝静默跳过），不写结果"
            )
        m_task = manifest.get(task)
        if m_task is None:
            print(f"[GATE-WARN] {os.path.basename(path)}: 任务 {task} 不在 "
                  f"manifest——跳过该文件")
            return None, 0
        exp_ids = set(m_task["ids"])
        got_ids = set(ids)
        if got_ids != exp_ids:
            missing = sorted(exp_ids - got_ids)[:5]
            extra = sorted(got_ids - exp_ids)[:5]
            raise SystemExit(
                f"[GATE-FAIL] {os.path.basename(path)}: _id 集合与 manifest "
                f"不等（缺 {len(exp_ids - got_ids)} 例 {missing}；多 "
                f"{len(got_ids - exp_ids)} 例 {extra}）——fail closed，不写结果"
            )
        exp_sha = m_task.get("answers_sha", {})
        for d, i, a in zip(records, ids, refs):
            if i in exp_sha and _answers_sha16(a) != exp_sha[i][:16]:
                raise SystemExit(
                    f"[GATE-FAIL] {os.path.basename(path)}: _id={i} 的 "
                    f"answers hash 与 manifest 不一致（答案错配/混行）"
                    f"——fail closed，不写结果"
                )
    return round(string_match_all(preds, refs), 2), len(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="exp/results_ruler/Qwen3-8B")
    ap.add_argument("--pred-postfix", default="_1024")
    ap.add_argument("--out", default="exp/results_ruler/ruler_table.json")
    ap.add_argument("--min-samples", type=int, default=100,
                    help="cell 样本数低于该值不计入 AVG（默认 100=满格）")
    # E116c fail-closed 门禁参数
    ap.add_argument("--merge-best", action="store_true",
                    help="同键多文件时按行数最多（并列取时间戳最新）选 "
                         "best-file 并落盘单一规范文件后再评分")
    ap.add_argument("--manifest", type=str, default="",
                    help="样本身份 manifest（JSON，{task: {ids, answers_sha}}），"
                         "给定时做集合闭包校验")
    ap.add_argument("--expect-tasks", type=int, default=0,
                    help="每个方法 key 必须覆盖的任务数（按 TASKS 前 N 个为"
                         "预期集合；缺任一任务非零退出）")
    args = ap.parse_args()

    manifest = json.load(open(args.manifest)) if args.manifest else None
    if args.expect_tasks > NTASK:
        raise SystemExit(f"[GATE-FAIL] --expect-tasks {args.expect_tasks} > "
                         f"任务总数 {NTASK}")

    res = {}    # key "L/method" -> {task: score}
    resn = {}   # key "L/method" -> {task: n}
    resrc = {}  # key "L/method" -> {task: {"file": ..., "sha256": ...}}
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
            # 同 task 下按 method-key 分组（同键多文件 = 多机重复跑/残留轮次）
            groups = {}
            for f in files:
                method = os.path.basename(f).replace(f"{task}-", "") \
                    .rsplit("-", 1)[0]
                groups.setdefault(method, []).append(f)
            for method, gfiles in groups.items():
                key = f"{L}/{method}"
                f_use = _resolve_group(gfiles, task, method, key,
                                       args.merge_best)
                score, n = _score_cell(f_use, task, key, manifest)
                if score is None:
                    continue
                res.setdefault(key, {})[task] = score
                resn.setdefault(key, {})[task] = n
                resrc.setdefault(key, {})[task] = {
                    "file": os.path.basename(f_use),
                    "sha256": _file_sha256(f_use)}
                print(f"[{key}] {task}: n={n} score={score} "
                      f"src={resrc[key][task]['file']} "
                      f"sha256={resrc[key][task]['sha256'][:16]}...")

    # ---- 门禁 3：--expect-tasks 任务闭包（缺任一任务 fail closed）----
    if args.expect_tasks > 0:
        exp_set = set(TASKS[:args.expect_tasks])
        for key in sorted(res):
            missing = sorted(exp_set - set(res[key]))
            if missing:
                raise SystemExit(
                    f"[GATE-FAIL] {key}: 缺 {len(missing)}/{args.expect_tasks}"
                    f" 个预期任务（{missing}）——fail closed，不输出正式结果"
                    f"（--expect-tasks {args.expect_tasks}）"
                )

    # B08：不完整 cell 警示（不计入 AVG；分数仍展示在表内）
    incomplete = [(k, t, resn[k][t]) for k in resn for t in resn[k]
                  if resn[k][t] < args.min_samples]
    for k, t, n in incomplete:
        print(f"WARN incomplete cell {k}/{t}: n={n} < {args.min_samples} "
              f"(excluded from AVG)")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    json.dump({"scores": res, "n": resn, "incomplete_cells": incomplete,
               "sources": resrc},
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
                    and all(resn[k][t] >= args.min_samples for t in res[k])]
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
