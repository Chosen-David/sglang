#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E117b：LB v2 解析器官方口径统一后的 503×3 全量重评（GPT TL-LBV2-PARSER-037）。

只读重放：不复制/改写解析判断逻辑——新口径直接 import 真实模块
benchmark.LongBench.lbv2_choice.extract_choice_official（E117b 修复后）；
历史口径用冻结内联实现（修复前 pred.py/eval.py 共享的兜底逻辑，仅作差异表对照）。

闭包断言（fail-closed）：
  ① 每臂 503 行、_id 无重复；
  ② 三臂 _id 集合完全相等；
  ③ 三臂行序与源 lbv2.json _id 序逐位一致；
  ④ 每行 answers 与源 answer 一致。

产物：
  exp/trace/results/e109_full_lbv2_v2.json          —— 修复后三臂成绩（supersedes v1）
  exp/trace/results/e109_full_lbv2_parser_diff.json —— 逐题差异表
      （_id → raw_pred → old_choice → official_choice → answer → old/new judge）
用法： python3 exp/trace/analyze_e117b_lbv2_rescore.py
"""
import json
import os
import re
import sys
import hashlib

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

RESULTS_DIR = os.path.join(ROOT, "exp", "trace", "results")

ARM_PATHS = {
    "FullKV": "/tmp/e109_scan_v2/pred_FULLKV/lbv2-none-benchmarklaunchtime.jsonl",
    "mavg": "/tmp/e109_scan_v2/pred_E109_mavg_a0.25_b0.125_g0.625/lbv2-tli_64_128_1024_c4_A-benchmarklaunchtime.jsonl",
    "aavg": "/tmp/e109_scan_v2/pred_E109_aavg_a0_b0_g0/lbv2-tli_64_128_1024_c4_A-benchmarklaunchtime.jsonl",
}
SOURCE_JSON = os.path.expanduser("~/datasets/LongBench/data/lbv2.json")


# ---------- 冻结的历史实现（修复前，仅对照） ----------
def extract_choice_legacy(text):
    if not text:
        return None
    t = text.replace("*", "")
    m = re.search(r"The correct answer is \(([A-D])\)", t, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"The correct answer is ([A-D])", t, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([ABCD])\b", t, flags=re.IGNORECASE)
    return m.group(1).upper() if m else None


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    from benchmark.LongBench.lbv2_choice import extract_choice_official, LBV2_PARSER_VERSION

    source = json.load(open(SOURCE_JSON))
    src_ids = [r["_id"] for r in source]
    src_ans = {r["_id"]: r["answer"] for r in source}
    assert len(src_ids) == 503 and len(set(src_ids)) == 503, "源 503 题 _id 有重复"

    per_arm, diff_table = {}, {}
    for arm, path in ARM_PATHS.items():
        rows = [json.loads(l) for l in open(path)]
        # 闭包断言 ①③④
        assert len(rows) == 503, (arm, len(rows))
        ids = [r["_id"] for r in rows]
        assert len(set(ids)) == 503, f"{arm} _id 重复"
        assert ids == src_ids, f"{arm} 行序与源 lbv2.json 不一致"
        for r in rows:
            assert list(r["answers"]) == [src_ans[r["_id"]]], (arm, r["_id"], r["answers"])

        old_c, new_c, diffs = 0, 0, []
        for r in rows:
            gt = src_ans[r["_id"]]
            o = extract_choice_legacy(r["pred"])
            n = extract_choice_official(r["pred"])
            oj = 1.0 if (o is not None and o == gt) else 0.0
            nj = 1.0 if (n is not None and n == gt) else 0.0
            old_c += oj
            new_c += nj
            if o != n:
                diffs.append({
                    "_id": r["_id"],
                    "raw_pred": r["pred"],
                    "old_choice": o,
                    "official_choice": n,
                    "answer": gt,
                    "old_judge": oj,
                    "new_judge": nj,
                })
        # 闭包断言 ② 在此臂收集后做集合比较（下方统一）
        per_arm[arm] = {
            "rows": 503,
            "old_score": round(old_c / 503 * 100, 2),
            "old_correct": int(old_c),
            "official_score": round(new_c / 503 * 100, 2),
            "official_correct": int(new_c),
            "parse_diff_count": len(diffs),
            "false_positives_removed": sum(1 for d in diffs if d["old_judge"] == 1.0 and d["new_judge"] == 0.0),
            "false_negatives_introduced": sum(1 for d in diffs if d["old_judge"] == 0.0 and d["new_judge"] == 1.0),
        }
        diff_table[arm] = diffs

    # 闭包断言 ② 三臂 _id 集合完全相等
    id_sets = None
    for arm, path in ARM_PATHS.items():
        s = {json.loads(l)["_id"] for l in open(path)}
        id_sets = s if id_sets is None else id_sets
        assert s == id_sets, "三臂 _id 集合不相等"

    v1 = json.load(open(os.path.join(RESULTS_DIR, "e109_full_lbv2.json")))
    out = {
        "experiment": "E117b_lbv2_parser_official_rescore",
        "date": "2026-10-09",
        "supersedes": {
            "file": "e109_full_lbv2.json",
            "reason": "GPT TL-LBV2-PARSER-037：历史解析器 IGNORECASE 独立字母兜底把冠词 a 判成选项 A"
                      "（假阳性）。正式口径统一为官方两条大小写敏感模式后全量重评，三臂排序反转。",
        },
        "parser": {
            "version": LBV2_PARSER_VERSION,
            "module": "benchmark/LongBench/lbv2_choice.py",
            "semantics": "官方（THUDM/LongBench）两条大小写敏感模式，无兜底，无匹配记 0 分",
            "star_strip": "off（官方无此行为；重放前实测 503×3 剥离与否解析逐位等价）",
        },
        "closure": {
            "rows_per_arm": 503,
            "three_arm_id_sets_equal": True,
            "row_order_matches_source": True,
            "answers_match_source": True,
            "source_json": SOURCE_JSON,
            "source_sha256": sha256(SOURCE_JSON),
            "arm_file_sha256": {a: sha256(p) for a, p in ARM_PATHS.items()},
        },
        "arms": {},
        "verdict": {},
    }
    fullkv_new = per_arm["FullKV"]["official_score"]
    ranking = sorted(per_arm.items(), key=lambda kv: -kv[1]["official_score"])
    for arm, st in per_arm.items():
        out["arms"][arm] = {
            **st,
            "delta_vs_fullkv_official": round(st["official_score"] - fullkv_new, 2),
            "v1_reported_score": v1["arms"].get(
                "FullKV" if arm == "FullKV" else
                ("mavg_a0.25_b0.125_g0.625" if arm == "mavg" else "aavg_a0_b0_g0"),
                {}).get("score"),
        }
    out["verdict"] = {
        "old_ranking": "aavg 32.60 > FullKV 32.21 > mavg 32.01（aavg +0.39 唯一超）",
        "official_ranking": " > ".join(f"{a} {s['official_score']}" for a, s in ranking),
        "conclusion": "排序反转：官方口径下无 TLI 臂超过 FullKV；aavg 因 4 个兜底假阳性虚高。"
                      "原「LB v2 冠军=aavg +0.39」结论撤销，以本文件口径为准。",
        "diff_total": sum(len(v) for v in diff_table.values()),
    }

    p_v2 = os.path.join(RESULTS_DIR, "e109_full_lbv2_v2.json")
    p_diff = os.path.join(RESULTS_DIR, "e109_full_lbv2_parser_diff.json")
    json.dump(out, open(p_v2, "w"), ensure_ascii=False, indent=2)
    json.dump({
        "experiment": "E117b_lbv2_parser_diff_table",
        "parser_version": LBV2_PARSER_VERSION,
        "spec": "_id → raw_pred → old_choice → official_choice → answer → old_judge → new_judge",
        "arms": diff_table,
    }, open(p_diff, "w"), ensure_ascii=False, indent=2)

    print(json.dumps({k: v for k, v in out["arms"].items()}, ensure_ascii=False, indent=2))
    print("verdict:", json.dumps(out["verdict"], ensure_ascii=False))
    print(f"saved: {p_v2}")
    print(f"saved: {p_diff}")


if __name__ == "__main__":
    main()
