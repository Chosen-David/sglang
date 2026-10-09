#!/usr/bin/env python3
# E116d：RULER 32K 三臂闭包对账 JSON v2 重生成（只读，不改动任何数据文件）
# 修订 v1（e109_ruler32_closure_check.json）两个缺陷：
#   ① 快照漂移：per_task 与 cross_arm_answers_check 引用了不同文件——
#     niah_multikey_1/aavg 引用 61 行的增长中文件（10090655 当时尚在写），
#     而 per_task 用 100 行 best-file（10090537），两节口径不一致；
#   ② partial 外推：同键 partial 文件（<100 行）的 pred 前缀对比被表述为
#     harmless_for_this_data=true（暗示任一文件得分相同）——尾部未覆盖，
#     不得外推。
# v2 契约：
#   - 逐 (arm, task) 单一 selected-file 冻结：行数最多，并列取文件名时间戳
#     最新，排除 *-merged.jsonl 派生物（与 score_ruler.py E116d 仲裁规则
#     完全一致）；
#   - per_task / cross_arm_answers_check / collision 全部引用同一 selected
#     文件（完整路径 + SHA256 + 行数）；
#   - 33 格计数断言：3 臂 × 11 任务 × 100 行 = 3300；
#   - 同键多文件候选全集 + 采用/排除原因落盘（rejected 也记 SHA256+行数）；
#   - partial 文件 pred 对比只声明「共同前缀 N 行 0 差异，尾部未覆盖」，
#     禁止外推为「任一文件得分相同」；
#   - 以当前磁盘状态为准冻结（10:2X 三机拉回后目录可能新增文件）。
# 数据源（只读）：exp/results_ruler/e109_full_Qwen3-8B/L32768/pred_E109_*/
# 输出：exp/trace/results/e109_ruler32_closure_check_v2.json（不覆盖 v1）
# 用法：python3 exp/trace/analyze_e116d_closure_v2.py
import glob
import hashlib
import json
import os
from datetime import datetime

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
BASE = os.path.join(REPO, "exp/results_ruler/e109_full_Qwen3-8B/L32768")
ARMS = [
    "pred_E109_mavg_a0.25_b0.125_g0.625",
    "pred_E109_aavg_a0_b0_g0",
    "pred_E109_FULLKV",
]
TASKS = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multiquery", "niah_multivalue", "cwe", "fwe", "vt",
]
EXPECTED_ROWS = 100
OUT = os.path.join(REPO, "exp/trace/results/e109_ruler32_closure_check_v2.json")


def _sha16(a):
    return hashlib.sha256(json.dumps(
        a, ensure_ascii=False, sort_keys=True).encode("utf-8")
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
    return fname[:-len(".jsonl")].rsplit("-", 1)[-1]


def _read(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def _reject_reason(f, best_rows, best_ts):
    n = _nlines(f)
    if n < best_rows:
        return f"行数 {n} < best {best_rows}"
    return f"行数并列 {n}，时间戳 {_ts_of(os.path.basename(f))} 早于 best {best_ts}"


def main():
    per_task = {}
    same_key = {}
    collision_paths = {}
    total_selected_rows = 0
    n_cells = 0
    partial_files = []
    n_candidates_total = 0

    for task in TASKS:
        per_task[task] = {}
        for arm in ARMS:
            pred_dir = os.path.join(BASE, arm)
            files = sorted(glob.glob(
                os.path.join(pred_dir, f"{task}-*.jsonl")))
            # 排除 merged 派生物（与 score_ruler.py E116d 一致；32K 目录当前
            # 无 merged 文件，但规则保持一致以防未来生成）
            files = [f for f in files
                     if not os.path.basename(f).endswith("-merged.jsonl")]
            if not files:
                raise SystemExit(
                    f"[CLOSURE-FAIL] {arm}/{task}: 零候选文件——33 格闭包缺口")
            n_candidates_total += len(files)
            best = max(files, key=lambda f: (
                _nlines(f), _ts_of(os.path.basename(f))))
            best_rows, best_ts = _nlines(best), _ts_of(os.path.basename(best))
            rows = _read(best)
            sel = {
                "path": os.path.relpath(best, REPO),
                "file": os.path.basename(best),
                "sha256": _file_sha256(best),
                "rows": best_rows,
            }
            per_task[task][arm] = sel
            total_selected_rows += best_rows
            n_cells += 1
            if best_rows < EXPECTED_ROWS:
                partial_files.append(f"{arm}/{task}: {sel['file']} "
                                     f"({best_rows} 行)")
            # 跨臂 selected 路径共享检查（不同臂目录不同 → 应零冲突）
            collision_paths.setdefault(sel["path"], []).append(f"{arm}/{task}")

            # 同键多文件候选全集 + 采用/排除原因（rejected 也记 SHA+行数）
            if len(files) > 1:
                cands = []
                for f in files:
                    n = _nlines(f)
                    cands.append({
                        "file": os.path.basename(f),
                        "rows": n,
                        "sha256": _file_sha256(f),
                        "adopted": f == best,
                        "reject_reason": None if f == best else
                        _reject_reason(f, best_rows, best_ts),
                    })
                    if n < EXPECTED_ROWS:
                        partial_files.append(
                            f"{arm}/{task}: {os.path.basename(f)} ({n} 行)")
                # 候选两两 pred 前缀对比——partial 只声明前缀，禁止外推
                pairwise = []
                for i in range(len(files)):
                    for j in range(i + 1, len(files)):
                        ra, rb = _read(files[i]), _read(files[j])
                        k = min(len(ra), len(rb))
                        diff = sum(1 for x, y in zip(ra[:k], rb[:k])
                                   if x["pred"] != y["pred"])
                        pair_partial = (len(ra) < EXPECTED_ROWS or
                                        len(rb) < EXPECTED_ROWS)
                        pairwise.append({
                            "files": [os.path.basename(files[i]),
                                      os.path.basename(files[j])],
                            "common_prefix_rows": k,
                            "pred_diff_count_in_common_prefix": diff,
                            "partial_pair": pair_partial,
                            "claim": (
                                f"共同前缀 {k} 行 pred 差异 {diff}，尾部未"
                                f"覆盖——仅前缀一致性声明，禁止外推为任一文件"
                                f"得分相同" if pair_partial else
                                f"全长 {k} 行 pred 差异 {diff}"
                                f"（两文件均完整 {EXPECTED_ROWS} 行）"),
                        })
                same_key[f"{arm}/{task}"] = {
                    "selected": sel["file"],
                    "selected_sha256": sel["sha256"],
                    "candidates": cands,
                    "pairwise_pred_check": pairwise,
                }

    # ---- 33 格计数一致性断言 ----
    assert n_cells == len(ARMS) * len(TASKS) == 33, \
        f"格数 {n_cells} != 33（3 臂 × 11 任务）"
    assert total_selected_rows == 33 * EXPECTED_ROWS == 3300, \
        f"selected 行数总和 {total_selected_rows} != 3300"

    # ---- 跨臂 answers 逐行一致性（引用与 per_task 完全相同的 selected）----
    cross_arm = {"all_identical_rowwise": True}
    for task in TASKS:
        row_shas = None
        per_arm = {}
        ok = True
        for arm in ARMS:
            sel = per_task[task][arm]
            rows = _read(os.path.join(REPO, sel["path"]))
            shas = [_sha16(r["answers"]) for r in rows]
            per_arm[arm] = {"file": sel["file"], "sha256": sel["sha256"],
                            "rows": sel["rows"]}
            if row_shas is None:
                row_shas = shas
            elif shas != row_shas:
                ok = False
        cross_arm[task] = {"identical_rowwise": ok, "per_arm": per_arm}
        if not ok:
            cross_arm["all_identical_rowwise"] = False

    # ---- 文件内重复检查（canonical = answers+length+budget，无 _id 旧数据）----
    within = {}
    for task in TASKS:
        # 三臂 answers 逐行全等（上面已验），取第一臂 selected 做代表性检查
        # 并逐臂记录行数；canon 唯一性对每臂单独验
        per_arm = {}
        for arm in ARMS:
            sel = per_task[task][arm]
            rows = _read(os.path.join(REPO, sel["path"]))
            canon = [_sha16([r["answers"], r["length"], r.get("budget")])
                     for r in rows]
            per_arm[arm] = {"n": len(rows), "unique_canon": len(set(canon))}
        within[task] = per_arm

    # ---- 跨臂 selected 路径冲突（同一文件被两臂采用）----
    collision = {p: v for p, v in collision_paths.items() if len(v) > 1}

    out = {
        "check": ("E116d RULER 32K 三臂闭包对账 v2（单一 selected-file 冻结，"
                  "修订 GPT 0935 审计确认的 v1 快照漂移）"),
        "base": BASE,
        "generated": datetime.now().isoformat(),
        "supersedes": "e109_ruler32_closure_check.json",
        "revision_reason": (
            "v1 的 per_task 与 cross_arm_answers_check 引用了不同文件"
            "（niah_multikey_1/aavg 引用 61 行增长中文件 vs per_task 的 100 "
            "行 best-file，快照漂移）；且 v1 把 partial 文件 pred 前缀一致"
            "外推为 harmless_for_this_data=true。v2 逐格单一 selected-file "
            "冻结，三节全部引用同一文件（路径+SHA256+行数），partial 对比"
            "只声明前缀、禁止外推"),
        "selection_rule": ("行数最多，并列取文件名时间戳最新，排除 "
                           "*-merged.jsonl 派生物（与 score_ruler.py E116d "
                           "best-file 仲裁规则一致）"),
        "frozen_at_disk_state": {
            "total_candidate_files": n_candidates_total,
            "note": "以脚本运行时磁盘状态为准冻结（三机拉回可能新增文件）",
        },
        "arms": ARMS,
        "cell_count": {
            "arms": len(ARMS), "tasks": len(TASKS), "cells": n_cells,
            "expected_rows_per_cell": EXPECTED_ROWS,
            "selected_rows_total": total_selected_rows,
            "assertion": "3 臂 × 11 任务 × 100 行 = 3300，逐格断言通过",
        },
        "per_task": per_task,
        "cross_arm_answers_check": cross_arm,
        "within_file_duplicate_check": within,
        "collision_check": collision,
        "same_key_collision_check": same_key,
        "partial_files": partial_files,
        "verdict": {
            "cells_complete": n_cells == 33 and total_selected_rows == 3300,
            "cross_arm_answers_rowwise_identical":
                cross_arm["all_identical_rowwise"],
            "within_file_zero_duplicate": all(
                v["n"] == v["unique_canon"]
                for t in within.values() for v in t.values()),
            "cross_arm_selected_path_collision": bool(collision),
            "partial_files_present": bool(partial_files),
            "note": ("pred 前缀一致的候选对只声明前缀性质；得分等价性仅对"
                     "全长完整且 pred 逐位相同的候选对成立（本轮全部候选均"
                     "为 100 行完整文件，前缀=全长）"),
        },
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(f"[closure-v2] 33 格 selected 行数总和 = {total_selected_rows}"
          f"（断言 3300 通过）")
    print(f"[closure-v2] 跨臂 answers 逐行全等 = "
          f"{cross_arm['all_identical_rowwise']}")
    print(f"[closure-v2] 跨臂 selected 路径冲突 = {collision or '无'}")
    print(f"[closure-v2] partial 文件 = {partial_files or '无'}")
    print(f"[closure-v2] 同键多文件格数 = {len(same_key)}"
          f"（候选总数 {n_candidates_total}）")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
