#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E119 最小脱敏 fixture 生成器（GPT 1429 审计 TL-E119-TEST-PORTABILITY-048①）。

用途：为 test_e119_crossarm_identity.py 提供干净检出即可运行的最小合成
三臂产物（不包含任何生产预测内容——比「脱敏复制少量行」更强的口径：
全部行都是合成数据）。生成的产物与真实 E119 64K 三臂的结构逐字段
同构（result/manifest/receipt + generation 派生目录 + 各臂源预测目录），
全部 SHA 闭合，可通过 analyze_e119_ruler64k_formal.py 的全部门禁。

生成内容（dest 目录下）：
  data_root/65536/{task}.jsonl                          —— RULER 源数据（合成）
  pred_root/{mavg,fullkv,aavg}/L65536/pred_1024/*.jsonl   —— 各臂源预测（legacy，无 _id）
  results/e119_ruler64k_formal_{arm}.json{,.manifest.json,.receipt.json}
  results/{arm}.json.run-{run_id}/scorer.manifest.json
  results/{arm}.json.run-{run_id}/pred_root/L65536/pred_1024/*.jsonl
                                                          —— 补刻派生副本（含 _id）

口径说明：
  - 4 任务 × 3 行/任务（min_samples=3：analyzer 的基数门禁为
    n == len(ids) == len(lengths) == len(answers_sha) 且 n >= manifest
    min_samples，生产 manifest min_samples=100 时等价于旧的 n==100 门禁）；
  - receipt 无 publish_protocol 键（与真实 64K 三臂一致的 legacy 协议，
    analyzer 据此标 legacy_protocol=true 并按固定命名构造 generation 路径）；
  - 三臂 treatment 与 analyze 的 ARM_CONTRACT 逐字段一致；
    分数 mavg 40.0 > FullKV 35.0 > aavg 30.0（与生产排序方向一致）；
  - 三臂 formal/scorer 脚本 SHA 互相一致（047 公平门禁的正例口径）。

用法： python3 gen_e119_min_fixture.py [dest]   # 缺省 dest = 本脚本所在目录
生成产物逐字节确定（可重复运行校验入库文件无漂移）。
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

TASKS = ["niah_single_1", "niah_multiquery", "cwe", "vt"]
ROWS = 3                       # 每任务行数（= manifest min_samples）
LNUM, LNAME = 65536, "L65536"
PRED_POSTFIX = "_1024"
PRED_DIR = "pred" + PRED_POSTFIX
METHOD_TLI = "tli_64_128_1024_c4_A"
RUN_IDS = {"mavg": "20261009120000-000001-0001",
           "FullKV": "20261009120000-000002-0002",
           "aavg": "20261009120000-000003-0003"}
ARM_DIRS = {"mavg": "mavg", "FullKV": "fullkv", "aavg": "aavg"}
ARM_CONTRACT = {
    "mavg": {"far_method": "minmax", "near_method": "avg",
             "alpha": "0.25", "beta": "0.125", "gamma": "0.625"},
    "aavg": {"far_method": "avg", "near_method": "avg",
             "alpha": "0", "beta": "0", "gamma": "0"},
    "FullKV": {"method": "none"},
}
# 与生产排序方向一致的合成分数（mavg > FullKV > aavg）
SCORES = {
    "mavg": {"niah_single_1": 40.0, "niah_multiquery": 44.0,
             "cwe": 36.0, "vt": 40.0},
    "FullKV": {t: 35.0 for t in TASKS},
    "aavg": {t: 30.0 for t in TASKS},
}
FORMAL_SHA = "f" * 64          # 三臂一致（047 门禁正例）
SCORER_SHA = "e" * 64


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha16(obj):
    """合成 answers_sha（16 hex，与 score_ruler._answers_sha16 同长度口径）。"""
    return hashlib.sha1(json.dumps(obj, ensure_ascii=False,
                                    sort_keys=True).encode("utf-8")
                        ).hexdigest()[:16]


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)


def _write_rows(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def generate(dest):
    dest = os.path.abspath(dest)
    data_root = os.path.join(dest, "data_root")
    pred_root = os.path.join(dest, "pred_root")
    results = os.path.join(dest, "results")

    # ---- RULER 源数据（合成）----
    src_data_sha = {}
    for t in TASKS:
        p = os.path.join(data_root, str(LNUM), f"{t}.jsonl")
        _write_rows(p, [{"index": i,
                         "input": f"synthetic input for {t} row {i}",
                         "answers": [f"gt{i}"], "length": LNUM}
                        for i in range(ROWS)])
        src_data_sha[f"{LNUM}/{t}"] = {
            "path": p, "sha256": _sha256_file(p)}

    # ---- 各臂源预测（legacy：无 _id/_answers_sha）+ 补刻派生副本 ----
    arm_files = {}   # arm -> {task: {src_fp, der_fp, best_file, ids, shas, lengths}}
    for arm, arm_dir in ARM_DIRS.items():
        method = "none" if arm == "FullKV" else METHOD_TLI
        cell_key = f"{LNAME}/{'none' if arm == 'FullKV' else METHOD_TLI}"
        arm_files[arm] = {"cell_key": cell_key, "tasks": {}}
        for t in TASKS:
            best_file = f"{t}-{method}-10090633.jsonl"
            src_fp = os.path.join(pred_root, arm_dir, LNAME, PRED_DIR,
                                  best_file)
            src_rows = [{"index": i,
                         "input": f"synthetic input for {t} row {i}",
                         "answers": [f"gt{i}"], "length": LNUM,
                         "pred": f"synthetic pred {arm}/{t}/{i}",
                         "budget": 1024}
                        for i in range(ROWS)]
            _write_rows(src_fp, src_rows)
            der_rows = [dict(r, _id=f"{t}:{i}",
                             _answers_sha=_sha16(r["answers"]))
                        for i, r in enumerate(src_rows)]
            run_dir = os.path.join(
                results, f"e119_ruler64k_formal_{arm.lower()}.json"
                f".run-{RUN_IDS[arm]}")
            der_fp = os.path.join(run_dir, "pred_root", LNAME, PRED_DIR,
                                  best_file)
            _write_rows(der_fp, der_rows)
            arm_files[arm]["tasks"][t] = {
                "best_file": best_file,
                "source_path": src_fp,
                "source_sha256": _sha256_file(src_fp),
                "derived_sha256": _sha256_file(der_fp),
                "ids": [r["_id"] for r in der_rows],
                "shas": {r["_id"]: r["_answers_sha"] for r in der_rows},
                "lengths": [r["length"] for r in der_rows],
            }

    # ---- 三臂 result / manifest / receipt / scorer manifest ----
    for arm in ARM_DIRS:
        cell_key = arm_files[arm]["cell_key"]
        result_name = f"e119_ruler64k_formal_{arm.lower()}.json"
        result_fp = os.path.join(results, result_name)
        run_dir = os.path.join(results, result_name + ".run-" + RUN_IDS[arm])
        tasks_scores = SCORES[arm]
        cells = {
            cell_key: {
                "length_dir": LNUM,
                "identity_mode": "legacy-partial",
                "tasks": {
                    t: {"n": ROWS,
                        "best_file": arm_files[arm]["tasks"][t]["best_file"],
                        "source_path":
                            arm_files[arm]["tasks"][t]["source_path"],
                        "source_sha256":
                            arm_files[arm]["tasks"][t]["source_sha256"],
                        "derived_sha256":
                            arm_files[arm]["tasks"][t]["derived_sha256"]}
                    for t in TASKS},
            }
        }
        tasks_manifest = {
            t: {"ids": arm_files[arm]["tasks"][t]["ids"],
                "answers_sha": arm_files[arm]["tasks"][t]["shas"],
                "lengths": arm_files[arm]["tasks"][t]["lengths"],
                "identity_mode": "legacy-partial"}
            for t in TASKS}
        result = {
            "scores": {cell_key: dict(tasks_scores)},
            "n": {cell_key: {t: ROWS for t in TASKS}},
            "incomplete_cells": [],
            "sources": {cell_key: {
                t: {"file": f"{t}-"
                    f"{'none' if arm == 'FullKV' else METHOD_TLI}-merged.jsonl",
                    "sha256": arm_files[arm]["tasks"][t]["derived_sha256"]}
                for t in TASKS}},
        }
        _write_json(result_fp, result)
        run_identity = {
            "data_root": data_root,
            "model_path": os.path.join(dest, "Qwen3-8B-fixture"),
            "yarn": True,
            "yarn_factor": 2.0,
            "extra_params": dict(ARM_CONTRACT[arm]),
            "formal_script_sha256": FORMAL_SHA,
            "scorer_sha256": SCORER_SHA,
            "note": ("synthetic fixture：E119 最小脱敏产物（048①），全部行"
                     "为合成数据；结构与真实 legacy 产物逐字段同构"),
        }
        manifest = {
            "manifest_version": 2,
            "run_id": RUN_IDS[arm],
            "generated": "2026-10-09T12:00:00",
            "root": os.path.join(pred_root, ARM_DIRS[arm]),
            "pred_postfix": PRED_POSTFIX,
            "expect_tasks": len(TASKS),
            "min_samples": ROWS,
            "run_identity": run_identity,
            "source_data_sha256": src_data_sha,
            "cells": cells,
            "tasks": tasks_manifest,
        }
        manifest_fp = result_fp + ".manifest.json"
        _write_json(manifest_fp, manifest)
        avg = round(sum(tasks_scores.values()) / len(tasks_scores), 2)
        receipt = {
            "run_id": RUN_IDS[arm],
            "status": "success",
            "generated": "2026-10-09T12:00:00",
            # legacy receipt：无 publish_protocol / outputs.generation_files
            # （与真实 64K 三臂一致——E116e 及更早的发布协议）
            "formal": {"script": "benchmark/RULER/score_ruler_formal.py",
                       "sha256": FORMAL_SHA},
            "scorer": {"script": "benchmark/RULER/score_ruler.py",
                       "sha256": SCORER_SHA},
            "inputs": {
                "root": os.path.join(pred_root, ARM_DIRS[arm]),
                "pred_postfix": PRED_POSTFIX,
                "data_root": data_root,
                "expect_tasks": len(TASKS),
                "min_samples": ROWS,
                "run_identity": dict(run_identity),
            },
            "outputs": {
                "json": result_fp,
                "md": result_fp[:-len(".json")] + ".md",
                "manifest": manifest_fp,
                "derived_dir": run_dir,
            },
            "manifest_sha256": _sha256_file(manifest_fp),
            "result_sha256": _sha256_file(result_fp),
            "source_data_sha256": src_data_sha,
            "cells": cells,
            "avg": {cell_key: avg},
            "legacy_stamp": {
                "n_stamped": len(TASKS),
                "invariant_fields": ["pred", "answers", "length", "budget"],
                "invariant_asserted": True,
                "note": ("synthetic fixture：补刻只发生在派生副本；源文件"
                         " SHA256 记录于 cells[].tasks[].source_sha256"),
            },
        }
        _write_json(result_fp + ".receipt.json", receipt)
        # generation 内 scorer manifest（ids + answers_sha 闭包子集）
        _write_json(os.path.join(run_dir, "scorer.manifest.json"),
                    {t: {"ids": tasks_manifest[t]["ids"],
                         "answers_sha": tasks_manifest[t]["answers_sha"]}
                     for t in TASKS})
    print(f"e119_min fixture 生成完毕: {dest}")
    for arm in ARM_DIRS:
        print(f"  {arm}: cell={arm_files[arm]['cell_key']} "
              f"run_id={RUN_IDS[arm]}")


if __name__ == "__main__":
    generate(sys.argv[1] if len(sys.argv) > 1 else HERE)
