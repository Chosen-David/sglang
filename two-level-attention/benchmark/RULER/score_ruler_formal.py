# E116d：RULER 生产级正式打分入口（两阶段，fail-closed）
# 阶段一「冻结」：遍历 root/L*/pred{postfix}/，逐 (L, task, method) 格从
#   原始候选全集（排除 *-merged.jsonl 派生物）仲裁 best-file（行数最多，
#   并列取文件名时间戳最新）；
#   - 原始文件缺 _id（E116c 之前的 legacy 数据）→ 原地补刻身份
#     _id={task}:{row_index}（与 pred_ruler.py 原生口径一致）+ _answers_sha，
#     原子替换（临时文件 + os.replace），pred/answers/length/budget 零改动；
#   - 由 best-file 逐行提取 ids 与 answers_sha 生成冻结 identity manifest，
#     同 task 跨方法/跨 L 目录身份不一致 → fail closed（跨臂样本身份漂移）；
#   - manifest 落盘。
# 阶段二「评分」：以 --manifest --expect-tasks --merge-best 子进程调用
#   score_ruler.py，退出码非零 → 本脚本以同码失败（不吞 scorer 的门禁）。
# 成功时打印 DONE + 结果 JSON 路径。
#
# 用法（单臂单 root）：
#   python -u benchmark/RULER/score_ruler_formal.py \
#     --root exp/results_ruler/e109_full_Qwen3-8B/L32768 \
#     --pred-postfix _E109_FULLKV \
#     --out /tmp/ruler_formal_fullkv.json \
#     [--manifest-out /tmp/ruler_formal_manifest.json] \
#     [--expect-tasks 11] [--min-samples 100] [--no-stamp-legacy-ids]
# 三臂各跑一次（postfix 分别为 _E109_mavg_a0.25_b0.125_g0.625 /
#   _E109_aavg_a0_b0_g0 / _E109_FULLKV）。
# 注意：--merge-best 会在 root 的 pred 目录落盘 {task}-{method}-merged.jsonl
#   规范文件；legacy 补刻会原子改写缺 _id 的原始 pred 文件（仅增两个字段）。
#   对不可改动的存档数据请先复制到工作目录再跑（测试即如此）。
import argparse
import glob
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from benchmark.RULER.score_ruler import (  # noqa: E402
    TASKS, _answers_sha16, _file_sha256, _nlines, _ts_of,
    _validate_manifest_schema,
)


def _stamp_legacy_ids(path, task, stamp=True):
    """原始 pred 文件缺 _id（E116c 前 legacy 数据）→ 原子补刻身份字段。
    返回 True 表示发生了改写。混合状态（部分行有 _id）→ fail closed
    （score_ruler 门禁 1 同样会拒）。"""
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    if not rows:
        return False
    have = sum(1 for r in rows if r.get("_id") is not None)
    if have == len(rows):
        return False
    if have > 0:
        raise SystemExit(
            f"[GATE-FAIL] {path}: 部分行有 _id（{have}/{len(rows)}）——混合"
            f"状态非法，fail closed"
        )
    if not stamp:
        raise SystemExit(
            f"[GATE-FAIL] {path}: legacy 数据缺 _id 且 --no-stamp-legacy-ids"
            f"——无法做 manifest 校验，fail closed"
        )
    out = []
    for i, r in enumerate(rows):
        r = dict(r)
        # 与 pred_ruler.py E116c 原生口径一致：_id = {task}:{row_index}
        r["_id"] = f"{task}:{i}"
        r.setdefault("_answers_sha", _answers_sha16(r["answers"]))
        out.append(r)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    return True


def freeze_manifest(root, postfix, expect_tasks, stamp):
    """阶段一：逐格 best-file 仲裁 + legacy 补刻 + manifest 生成与跨臂
    身份断言。返回 manifest dict。"""
    manifest = {}
    n_stamped = 0
    for L_dir in sorted(glob.glob(os.path.join(root, "L*"))):
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
            groups = {}
            for f in files:
                method = os.path.basename(f).replace(f"{task}-", "") \
                    .rsplit("-", 1)[0]
                groups.setdefault(method, []).append(f)
            for method, gfiles in groups.items():
                for f in gfiles:
                    if _stamp_legacy_ids(f, task, stamp=stamp):
                        n_stamped += 1
                        print(f"[formal] legacy 补刻 _id: {f}")
                best = max(gfiles, key=lambda f: (
                    _nlines(f), _ts_of(os.path.basename(f))))
                rows = [json.loads(l)
                        for l in open(best, encoding="utf-8")]
                ids = [r["_id"] for r in rows]
                shas = {r["_id"]: r["_answers_sha"] for r in rows}
                if len(ids) != len(set(ids)):
                    raise SystemExit(
                        f"[GATE-FAIL] {best}: _id 重复——fail closed"
                    )
                key = f"{os.path.basename(L_dir)}/{method}"
                print(f"[formal] {key}/{task}: best-file "
                      f"{os.path.basename(best)} (n={len(rows)}) 冻结进 manifest")
                if task in manifest:
                    if set(manifest[task]["ids"]) != set(ids) or \
                            manifest[task]["answers_sha"] != shas:
                        raise SystemExit(
                            f"[GATE-FAIL] task={task}: 跨方法/跨 L 目录的"
                            f"样本身份不一致（ids 或 answers_sha 漂移）"
                            f"——fail closed，manifest 拒绝生成"
                        )
                else:
                    manifest[task] = {"ids": ids, "answers_sha": shas}
    if expect_tasks > 0 and len(manifest) < expect_tasks:
        missing = sorted(set(TASKS[:expect_tasks]) - set(manifest))
        raise SystemExit(
            f"[GATE-FAIL] root={root} postfix={postfix}: manifest 只覆盖 "
            f"{len(manifest)}/{expect_tasks} 个任务（缺 {missing}）"
            f"——fail closed"
        )
    if n_stamped:
        print(f"[formal] 共补刻 {n_stamped} 个 legacy 文件的身份字段"
              f"（_id/_answers_sha，pred/answers 零改动，原子替换）")
    _validate_manifest_schema(manifest)
    return manifest


def main():
    ap = argparse.ArgumentParser(
        description="RULER 生产级正式打分入口（冻结 manifest + 门禁评分）")
    ap.add_argument("--root", required=True,
                    help="结果根目录（其下 L*/pred{postfix}/）")
    ap.add_argument("--pred-postfix", required=True,
                    help="pred 目录后缀（区分臂，如 _E109_FULLKV）")
    ap.add_argument("--out", required=True, help="结果 JSON 输出路径")
    ap.add_argument("--manifest-out", default="",
                    help="冻结 manifest 落盘路径（默认 <out>.manifest.json）")
    ap.add_argument("--expect-tasks", type=int, default=11,
                    help="预期任务闭包数（默认 11=RULER 全任务）")
    ap.add_argument("--min-samples", type=int, default=100)
    ap.add_argument("--no-stamp-legacy-ids", action="store_true",
                    help="禁用 legacy _id 补刻（缺 _id 数据将 fail closed）")
    args = ap.parse_args()

    manifest_path = args.manifest_out or (args.out + ".manifest.json")
    manifest = freeze_manifest(args.root, args.pred_postfix,
                               args.expect_tasks,
                               stamp=not args.no_stamp_legacy_ids)
    mdir = os.path.dirname(manifest_path)
    if mdir:
        os.makedirs(mdir, exist_ok=True)
    json.dump(manifest, open(manifest_path, "w"), indent=1, ensure_ascii=False)
    print(f"[formal] manifest 落盘: {manifest_path} "
          f"({len(manifest)} tasks)")

    # 阶段二：子进程调用 score_ruler.py，退出码原样传播
    argv = [sys.executable, "-u", "-m", "benchmark.RULER.score_ruler",
            "--root", args.root, "--pred-postfix", args.pred_postfix,
            "--out", args.out, "--min-samples", str(args.min_samples),
            "--manifest", manifest_path,
            "--expect-tasks", str(args.expect_tasks), "--merge-best"]
    r = subprocess.run(argv, cwd=REPO,
                       env={**os.environ, "PYTHONPATH": REPO})
    if r.returncode != 0:
        print(f"[formal] scorer 退出码 {r.returncode} —— 正式打分失败",
              file=sys.stderr)
        sys.exit(r.returncode)
    print(f"DONE {args.out}")


if __name__ == "__main__":
    main()
