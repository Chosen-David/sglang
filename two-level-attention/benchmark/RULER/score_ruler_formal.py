# E116d：RULER 生产级正式打分入口（两阶段，fail-closed）
# 阶段一「冻结」：遍历 root/L*/pred{postfix}/，逐 (L, task, method) 格从
#   原始候选全集（排除 *-merged.jsonl 派生物）仲裁 best-file（行数最多，
#   并列取文件名时间戳最新）；
# 阶段二「评分」：以 --manifest --expect-tasks --merge-best 子进程调用
#   score_ruler.py，退出码非零 → 本脚本以同码失败（不吞 scorer 的门禁）。
#
# ===== E116e（GPT 1033 审计 030-036 七项修复，本版全落地）=====
#   030 min-samples 升格为发布硬门禁：任一 (L, method, task) 格 n <
#       min-samples → 非零退出、不发布任何产物、清理 staging；
#   031 staging 原子发布：全部校验（manifest + scorer + JSON/MD + receipt）
#       先在 {out}.staging-{run_id}/ 临时目录完成，成功后逐文件 os.replace
#       原子发布（receipt 最后落盘=提交信号）；失败写独立 failure receipt
#       （{out}.failure-{run_id}.json，明确「旧产物属于上一轮成功」），
#       不覆盖不删除旧成功产物，只清理未发布 staging；
#   032 身份扩展：manifest 逐格绑定 length（L 目录档位 + 行内实际 token 数
#       统计）、method、源数据文件 SHA256（{data_root}/{L}/{task}.jsonl）、
#       model_path、关键参数（--yarn/--yarn-factor/--extra-param）；
#       长度身份门禁 = ① 行 length ≤ L 档位（128K 数据混入 32K 目录拒）
#       ② 同 task 跨 method 逐行 length 一致（同 row index/answers 不同
#       length 拒）；legacy 事后行号身份标 identity_mode=legacy-partial，
#       原生 _id 数据标 native；
#   033 --out 强制小写 .json 后缀（无后缀/.JSON/路径中间含 .json 全拒），
#       MD 路径用后缀精确推导（不用 replace），并断言 JSON/MD/manifest/
#       receipt 四路径两两不同；
#   036 不改原始文件：原始 pred 文件全程只读；legacy 补刻写到 staging 派生
#       副本（成功后保留在 {out}.run-{run_id}/ 版本化派生目录）；receipt 记录
#       source_sha256 → derived_sha256 与逐字段不变量断言（pred/answers/
#       length/budget 逐位不变）；失败保留源文件原样。
# 用法（单臂单 root）：
#   python -u benchmark/RULER/score_ruler_formal.py \
#     --root exp/results_ruler/e109_full_Qwen3-8B/L32768 \
#     --pred-postfix _E109_FULLKV \
#     --data-root /home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/RULER \
#     --out /tmp/ruler_formal_fullkv.json \
#     [--manifest-out ...] [--expect-tasks 11] [--min-samples 100] \
#     [--model-path ~/Qwen3-8B] [--yarn] [--extra-param k=v ...]
# 三臂各跑一次（postfix 分别为 _E109_mavg_a0.25_b0.125_g0.625 /
#   _E109_aavg_a0_b0_g0 / _E109_FULLKV）。
# 发布产物（成功时）：{out}（结果 JSON）、{out 去后缀}.md（表格）、
#   {out}.manifest.json（冻结身份 manifest）、{out}.receipt.json（运行
#   receipt=提交信号）、{out}.run-{run_id}/（版本化派生目录：补刻副本 +
#   merged 规范文件 + scorer manifest）。
# 失败产物：{out}.failure-{run_id}.json（独立 failure receipt，不动旧产物）。
import argparse
import glob
import json
import os
import random
import re
import shutil
import subprocess
import sys
from datetime import datetime

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from benchmark.RULER.score_ruler import (  # noqa: E402
    TASKS, _answers_sha16, _file_sha256, _nlines, _ts_of,
    _validate_manifest_schema,
)

FORMAL_PATH = os.path.abspath(__file__)
SCORER_PATH = os.path.join(os.path.dirname(FORMAL_PATH), "score_ruler.py")
# 036 逐字段不变量：legacy 补刻只允许新增 _id/_answers_sha 两键，
# 这四个业务字段必须逐位不变
INVARIANT_FIELDS = ("pred", "answers", "length", "budget")


def _fail(msg):
    raise SystemExit(f"[GATE-FAIL] {msg}")


def _stamp_or_copy(src, dst, task, stamp):
    """036：legacy 补刻写到派生副本 dst（源文件 src 全程只读）。
    返回 (stamped, rows)。stamped=True 表示发生了补刻（dst 含新增
    _id/_answers_sha 两字段）。写后重读做逐字段不变量断言。
    混合状态（部分行有 _id）→ fail closed（score_ruler 门禁 1 同款）。"""
    rows = [json.loads(l) for l in open(src, encoding="utf-8")]
    if not rows:
        _fail(f"{src}: 空预测文件——fail closed")
    have = sum(1 for r in rows if r.get("_id") is not None)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if have == len(rows):
        shutil.copyfile(src, dst)   # native：原样拷贝
        return False, rows
    if have > 0:
        _fail(f"{src}: 部分行有 _id（{have}/{len(rows)}）——混合状态非法，"
              f"fail closed")
    if not stamp:
        _fail(f"{src}: legacy 数据缺 _id 且 --no-stamp-legacy-ids——无法做"
              f"manifest 校验，fail closed")
    out_rows = []
    for i, r in enumerate(rows):
        r2 = dict(r)
        # 与 pred_ruler.py E116c 原生口径一致：_id = {task}:{row_index}
        r2["_id"] = f"{task}:{i}"
        r2.setdefault("_answers_sha", _answers_sha16(r["answers"]))
        out_rows.append(r2)
    with open(dst, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # 逐字段不变量断言（写后重读，防转换脚本自身破坏业务字段）
    dst_rows = [json.loads(l) for l in open(dst, encoding="utf-8")]
    if len(dst_rows) != len(rows):
        _fail(f"{src} → {dst}: 补刻后行数变化（{len(rows)}→{len(dst_rows)}）"
              f"——不变量破坏，fail closed")
    for a, b in zip(rows, dst_rows):
        for k in INVARIANT_FIELDS:
            if a.get(k) != b.get(k):
                _fail(f"{src} → {dst}: 补刻后字段 {k} 发生变化"
                      f"（{a.get(k)!r} → {b.get(k)!r}）——不变量破坏，"
                      f"fail closed")
    return True, out_rows


def freeze_and_stage(root, postfix, expect_tasks, stamp, staging,
                     min_samples, data_root):
    """阶段一（E116e 重构）：原始候选只读拷贝进 staging 派生目录 →
    legacy 补刻（派生副本上）→ 长度身份门禁 → min-samples 硬门禁 →
    best-file 仲裁 → 跨 method 身份（ids/answers_sha/lengths 逐行）一致性
    → 源数据 SHA256 绑定 → 冻结 manifest。
    返回 (tasks_manifest, identity_info)。"""
    staged_root = os.path.join(staging, "pred_root")
    tasks_manifest = {}   # {task: {ids, answers_sha, lengths, identity_mode}}
    cells_info = {}       # {key: {length_dir, identity_mode, tasks: {...}}}
    src_data_sha = {}     # {"{Lnum}/{task}": {path, sha256}}
    n_stamped = 0
    for L_dir in sorted(glob.glob(os.path.join(root, "L*"))):
        Lname = os.path.basename(L_dir)
        m = re.fullmatch(r"L(\d+)", Lname)
        if not m:
            _fail(f"非法长度目录名 {Lname}（须形如 L32768）——fail closed")
        Lnum = int(m.group(1))
        pred_dir = os.path.join(L_dir, f"pred{postfix}")
        if not os.path.isdir(pred_dir):
            continue
        for task in TASKS:
            files = sorted(glob.glob(
                os.path.join(pred_dir, f"{task}-*.jsonl")))
            files = [f for f in files
                     if not os.path.basename(f).endswith("-merged.jsonl")]
            if not files:
                continue
            # 032 源数据身份绑定：{data_root}/{L}/{task}.jsonl 必须存在
            data_file = os.path.join(data_root, str(Lnum), f"{task}.jsonl")
            if not os.path.isfile(data_file):
                _fail(f"源数据文件缺失: {data_file}（task={task}, L={Lname}）"
                      f"——身份闭包不完整，fail closed")
            src_data_sha.setdefault(f"{Lnum}/{task}", {
                "path": data_file, "sha256": _file_sha256(data_file)})
            groups = {}
            for f in files:
                method = os.path.basename(f).replace(f"{task}-", "") \
                    .rsplit("-", 1)[0]
                groups.setdefault(method, []).append(f)
            for method, gfiles in groups.items():
                # ---- 036：只读拷贝 + legacy 补刻（派生副本）----
                staged_files = []
                stamped_flags = {}
                for f in gfiles:
                    dst = os.path.join(staged_root, Lname, f"pred{postfix}",
                                       os.path.basename(f))
                    stamped, rows = _stamp_or_copy(f, dst, task, stamp=stamp)
                    staged_files.append(dst)
                    stamped_flags[dst] = stamped
                    if stamped:
                        n_stamped += 1
                        print(f"[formal] legacy 补刻 _id（派生副本）: {f}")
                    # ---- 032 长度身份门禁 ①：行 length ≤ L 档位 ----
                    # （length=实际 token 数；128K 数据混入 32K 目录 → 拒）
                    for r in rows:
                        if r.get("length") is None:
                            _fail(f"{os.path.basename(f)}: 行缺 length 字段"
                                  f"——无法做长度身份门禁，fail closed")
                        if r["length"] > Lnum:
                            _fail(
                                f"{Lname}/{method}/{task}: 行 length="
                                f"{r['length']} > 目录档位 {Lnum}（疑似 "
                                f"{r['length']} 档数据混入 {Lname} 目录）"
                                f"——长度身份门禁 fail closed")
                # ---- best-file 仲裁（行数最多，并列取时间戳最新）----
                best = max(staged_files, key=lambda f: (
                    _nlines(f), _ts_of(os.path.basename(f))))
                rows = [json.loads(l)
                        for l in open(best, encoding="utf-8")]
                ids = [r["_id"] for r in rows]
                shas = {r["_id"]: r["_answers_sha"] for r in rows}
                lengths = [r["length"] for r in rows]
                if len(ids) != len(set(ids)):
                    _fail(f"{os.path.basename(best)}: _id 重复——fail closed")
                # ---- 030：min-samples 发布硬门禁 ----
                if len(rows) < min_samples:
                    _fail(
                        f"{Lname}/{method}/{task}: n={len(rows)} < "
                        f"--min-samples {min_samples}——样本数不完整，"
                        f"拒绝发布（min-samples 是发布门禁不是展示开关），"
                        f"不发布任何产物")
                identity_mode = ("native" if not stamped_flags[best]
                                 else "legacy-partial")
                key = f"{Lname}/{method}"
                print(f"[formal] {key}/{task}: best-file "
                      f"{os.path.basename(best)} (n={len(rows)}) 冻结进 "
                      f"manifest（identity_mode={identity_mode}）")
                # ---- 跨 method/L 身份一致性（032：ids + answers_sha +
                # 逐行 length 三重比对）----
                if task in tasks_manifest:
                    prev = tasks_manifest[task]
                    if set(prev["ids"]) != set(ids) or \
                            prev["answers_sha"] != shas:
                        _fail(f"task={task}: 跨方法/跨 L 目录的样本身份不一致"
                              f"（ids 或 answers_sha 漂移）——fail closed，"
                              f"manifest 拒绝生成")
                    if prev["lengths"] != lengths:
                        _fail(
                            f"task={task}: 跨方法逐行 length 不一致（同 "
                            f"row index/answers 而 length 漂移——输入身份"
                            f"不闭合）——fail closed（032 长度身份门禁）")
                else:
                    tasks_manifest[task] = {
                        "ids": ids, "answers_sha": shas,
                        "lengths": lengths,
                        "identity_mode": identity_mode,
                    }
                cells_info.setdefault(key, {
                    "length_dir": Lnum,
                    "identity_mode": identity_mode,
                    "tasks": {},
                })["tasks"][task] = {
                    "n": len(rows),
                    "best_file": os.path.basename(best),
                    "source_path": os.path.abspath(
                        # best 与 staged 同名，映射回原始源文件
                        os.path.join(pred_dir, os.path.basename(best))),
                    "source_sha256": _file_sha256(os.path.join(
                        pred_dir, os.path.basename(best))),
                    "derived_sha256": _file_sha256(best),
                }
    if expect_tasks > 0 and len(tasks_manifest) < expect_tasks:
        missing = sorted(set(TASKS[:expect_tasks]) - set(tasks_manifest))
        _fail(f"root={root} postfix={postfix}: manifest 只覆盖 "
              f"{len(tasks_manifest)}/{expect_tasks} 个任务（缺 {missing}）"
              f"——fail closed")
    if n_stamped:
        print(f"[formal] 共 {n_stamped} 个 legacy 文件在派生副本上补刻身份"
              f"（_id/_answers_sha；源文件零改动，逐字段不变量已断言）")
    _validate_manifest_schema(tasks_manifest)
    return tasks_manifest, cells_info, src_data_sha


def main():
    ap = argparse.ArgumentParser(
        description="RULER 生产级正式打分入口（E116e：staging 原子发布 + "
                    "min-samples 硬门禁 + 身份闭包 + 源文件只读）")
    ap.add_argument("--root", required=True,
                    help="结果根目录（其下 L*/pred{postfix}/，全程只读）")
    ap.add_argument("--pred-postfix", required=True,
                    help="pred 目录后缀（区分臂，如 _E109_FULLKV）")
    ap.add_argument("--out", required=True,
                    help="结果 JSON 输出路径（必须以小写 .json 结尾）")
    ap.add_argument("--manifest-out", default="",
                    help="冻结 manifest 落盘路径（默认 <out>.manifest.json）")
    ap.add_argument("--expect-tasks", type=int, default=11,
                    help="预期任务闭包数（默认 11=RULER 全任务）")
    ap.add_argument("--min-samples", type=int, default=100,
                    help="发布硬门禁：任一格 n 低于该值 → 非零退出不发布")
    ap.add_argument("--data-root", required=True,
                    help="RULER 源数据根目录（其下 {L}/{task}.jsonl；"
                         "manifest 绑定逐任务源数据 SHA256）")
    ap.add_argument("--model-path", default="",
                    help="声明生成该预测所用模型路径（身份绑定；legacy 数据"
                         "可留空，manifest 记 null 不冒充）")
    ap.add_argument("--yarn", action="store_true",
                    help="声明生成时启用 YaRN（身份绑定）")
    ap.add_argument("--yarn-factor", type=float, default=None,
                    help="声明 YaRN factor（身份绑定）")
    ap.add_argument("--extra-param", action="append", default=[],
                    metavar="K=V",
                    help="附加身份参数（可重复，如 --extra-param "
                         "tia_level1_topk=1024）")
    ap.add_argument("--no-stamp-legacy-ids", action="store_true",
                    help="禁用 legacy _id 补刻（缺 _id 数据将 fail closed）")
    args = ap.parse_args()

    # ---- 033：--out 后缀门禁 + 四路径两两不同（在任何文件创建之前）----
    basename = os.path.basename(args.out)
    if not basename.endswith(".json") or basename == ".json":
        _fail(f"--out 必须以非空小写 .json 后缀结尾，得到 {args.out!r}"
              f"（无后缀/大写 .JSON/路径中间含 .json 均拒绝——防止 MD "
              f"覆盖 JSON 同一文件）")
    md_path = args.out[:-len(".json")] + ".md"   # 后缀精确推导，不用 replace
    manifest_path = args.manifest_out or (args.out + ".manifest.json")
    receipt_path = args.out + ".receipt.json"
    four = [os.path.abspath(p) for p in
            (args.out, md_path, manifest_path, receipt_path)]
    if len(set(four)) != len(four):
        _fail(f"JSON/MD/manifest/receipt 路径必须两两不同，得到 "
              f"{four}（--manifest-out 不得与 --out 或其派生路径相同）")

    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    run_id = f"{ts}-{os.getpid()}-{random.randint(1000, 9999)}"
    # staging/run_dir 绝对化：scorer 子进程以 cwd=REPO 运行，相对路径会
    # 相对 REPO 解析而非调用者 cwd——统一用绝对路径消除歧义
    staging = os.path.abspath(f"{args.out}.staging-{run_id}")
    run_dir = os.path.abspath(f"{args.out}.run-{run_id}")   # 成功后改名

    extra_params = {}
    for kv in args.extra_param:
        if "=" not in kv:
            _fail(f"--extra-param 须为 K=V 形式，得到 {kv!r}")
        k, v = kv.split("=", 1)
        extra_params[k] = v

    try:
        if not os.path.isdir(args.data_root):
            _fail(f"--data-root 不是目录: {args.data_root}")
        os.makedirs(staging, exist_ok=True)

        # ---- 阶段一：staging 派生 + 冻结 manifest（含身份扩展 032）----
        tasks_manifest, cells_info, src_data_sha = freeze_and_stage(
            args.root, args.pred_postfix, args.expect_tasks,
            stamp=not args.no_stamp_legacy_ids, staging=staging,
            min_samples=args.min_samples, data_root=args.data_root)

        manifest_full = {
            "manifest_version": 2,
            "run_id": run_id,
            "generated": datetime.now().isoformat(),
            "root": os.path.abspath(args.root),
            "pred_postfix": args.pred_postfix,
            "expect_tasks": args.expect_tasks,
            "min_samples": args.min_samples,
            "run_identity": {
                "data_root": os.path.abspath(args.data_root),
                "model_path": args.model_path or None,
                "yarn": bool(args.yarn),
                "yarn_factor": args.yarn_factor,
                "extra_params": extra_params,
                "formal_script_sha256": _file_sha256(FORMAL_PATH),
                "scorer_sha256": _file_sha256(SCORER_PATH),
                # legacy 声明口径：model/yarn 等为操作者事后声明，
                # 可能不可恢复——以 receipt/manifest 记录为准，不冒充完整
                "note": ("model_path/yarn 等为操作者声明值；legacy 数据无法"
                         "从文件恢复完整输入身份（identity_mode=legacy-"
                         "partial），native 数据由 pred_ruler.py 落盘"),
            },
            "source_data_sha256": src_data_sha,
            "cells": cells_info,
            # score_ruler.py 兼容子集（ids + answers_sha；lengths/
            # identity_mode 为 E116e 身份扩展，scorer 调用时剥离）
            "tasks": tasks_manifest,
        }
        staged_manifest = os.path.join(staging, "manifest.json")
        json.dump(manifest_full, open(staged_manifest, "w"), indent=1,
                  ensure_ascii=False)

        # ---- 阶段二：scorer 子进程（root=staging 派生副本；merged 规范
        # 文件也只落在 staging 内，原始目录零写入）----
        scorer_manifest = os.path.join(staging, "scorer.manifest.json")
        json.dump({t: {"ids": m["ids"], "answers_sha": m["answers_sha"]}
                   for t, m in tasks_manifest.items()},
                  open(scorer_manifest, "w"))
        staged_result = os.path.join(staging, "result.json")
        argv = [sys.executable, "-u", "-m", "benchmark.RULER.score_ruler",
                "--root", os.path.join(staging, "pred_root"),
                "--pred-postfix", args.pred_postfix,
                "--out", staged_result,
                "--min-samples", str(args.min_samples),
                "--manifest", scorer_manifest,
                "--expect-tasks", str(args.expect_tasks), "--merge-best"]
        r = subprocess.run(argv, cwd=REPO,
                           env={**os.environ, "PYTHONPATH": REPO})
        if r.returncode != 0:
            _fail(f"scorer 退出码 {r.returncode} —— 正式打分失败（scorer "
                  f"阶段门禁触发，不发布）")

        # ---- 发布前终验（staging 内完成；031 原子发布前置条件）----
        staged_md = os.path.join(staging, "result.md")
        if not os.path.isfile(staged_result) or not os.path.isfile(staged_md):
            _fail("scorer 成功但 staging 产物缺失（result.json/result.md）"
                  "——不发布")
        res = json.load(open(staged_result))
        for key, tasks in res["n"].items():
            for t, n in tasks.items():
                if n < args.min_samples:   # 030 复核（双保险）
                    _fail(f"{key}/{t}: scorer 结果 n={n} < min-samples "
                          f"{args.min_samples}——拒绝发布")
        # 每格源→派生 sha 追溯 + 逐字段不变量已由 _stamp_or_copy 断言；
        # receipt 汇总 cells 级 source→derived 映射
        avgs = {k: round(sum(s.values()) / len(s), 2)
                for k, s in res["scores"].items()}

        receipt = {
            "run_id": run_id,
            "status": "success",
            "generated": datetime.now().isoformat(),
            "formal": {"script": "benchmark/RULER/score_ruler_formal.py",
                       "sha256": _file_sha256(FORMAL_PATH)},
            "scorer": {"script": "benchmark/RULER/score_ruler.py",
                       "sha256": _file_sha256(SCORER_PATH)},
            "inputs": {
                "root": os.path.abspath(args.root),
                "pred_postfix": args.pred_postfix,
                "data_root": os.path.abspath(args.data_root),
                "expect_tasks": args.expect_tasks,
                "min_samples": args.min_samples,
                "run_identity": manifest_full["run_identity"],
            },
            "outputs": {"json": os.path.abspath(args.out),
                        "md": os.path.abspath(md_path),
                        "manifest": os.path.abspath(manifest_path),
                        "derived_dir": os.path.abspath(run_dir)},
            "manifest_sha256": _file_sha256(staged_manifest),
            "result_sha256": _file_sha256(staged_result),
            "source_data_sha256": src_data_sha,
            "cells": cells_info,
            "avg": avgs,
            "legacy_stamp": {
                "n_stamped": sum(
                    1 for c in cells_info.values()
                    for t in c["tasks"]
                    if c["identity_mode"] == "legacy-partial"),
                "invariant_fields": list(INVARIANT_FIELDS),
                "invariant_asserted": True,
                "note": ("补刻只发生在 staging 派生副本；源文件 SHA256 记录"
                         "于 cells[].tasks[].source_sha256，全程只读"),
            },
        }
        staged_receipt = os.path.join(staging, "receipt.json")
        json.dump(receipt, open(staged_receipt, "w"), indent=1,
                  ensure_ascii=False)

        # ---- 031：原子发布（逐文件 os.replace；receipt 最后=提交信号）----
        for p in (args.out, md_path, manifest_path):
            d = os.path.dirname(p)
            if d:
                os.makedirs(d, exist_ok=True)
        os.replace(staged_result, args.out)
        os.replace(staged_md, md_path)
        os.replace(staged_manifest, manifest_path)
        os.replace(staged_receipt, receipt_path)
        os.rename(staging, run_dir)   # 版本化派生目录（补刻副本+merged）
        print(f"[formal] 发布成功（原子）：{args.out} / {md_path} / "
              f"{manifest_path} / {receipt_path}")
        print(f"[formal] 派生目录（含补刻副本+merged 规范文件）: {run_dir}")
        print(f"DONE {args.out}")
    except SystemExit as e:
        # ---- 031/030/036：失败路径——独立 failure receipt + 清理 staging，
        # 不触碰旧成功产物与源文件 ----
        msg = str(e.code) if e.code is not None and str(e.code) else \
            f"exit({e.code})"
        print(msg, file=sys.stderr)   # 拒绝原因同时输出到 stderr（可观测）
        fpath = f"{args.out}.failure-{run_id}.json"
        try:
            d = os.path.dirname(fpath)
            if d:
                os.makedirs(d, exist_ok=True)
            json.dump({
                "run_id": run_id,
                "status": "failed",
                "generated": datetime.now().isoformat(),
                "error": msg,
                "argv": sys.argv[1:],
                "note": ("本轮失败，未发布任何产物；输出路径下既有产物"
                         "（如有）属于上一轮成功运行，请以最新 "
                         "*.receipt.json 为准"),
                "staging_cleaned": True,
            }, open(fpath, "w"), indent=1, ensure_ascii=False)
            print(f"[formal] failure receipt 落盘: {fpath}", file=sys.stderr)
        except OSError:
            pass
        shutil.rmtree(staging, ignore_errors=True)
        code = e.code if isinstance(e.code, int) and e.code != 0 else 1
        sys.exit(code)


if __name__ == "__main__":
    main()
