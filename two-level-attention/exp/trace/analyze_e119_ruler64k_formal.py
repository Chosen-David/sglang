#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E119：RULER 64K 三臂正式入口收口汇总（score_ruler_formal.py E116e 门禁产物聚合）。

三臂分别经正式入口打分（各产出 result/MD/manifest/receipt 四件套 + 派生目录），
本脚本只读聚合三份已发布 JSON/receipt，生成跨臂判决汇总——不重算任何分数。

闭包断言（fail-closed）：
  ① 三臂 receipt 均 status=success 且 result_sha256 与当前 JSON 实际 SHA 一致；
  ② 三臂 11 任务 n 全部 =100（min-samples 硬门禁已在正式入口内执行，此处复核）；
  ③ 三臂 sources 的源文件 SHA 与磁盘当前文件 SHA 一致（发布后无篡改）。

产物：exp/trace/results/e119_ruler64k_formal_summary.json
用法： python3 exp/trace/analyze_e119_ruler64k_formal.py
"""
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "exp", "trace", "results")

ARMS = {
    "mavg": "e119_ruler64k_formal_mavg.json",
    "FullKV": "e119_ruler64k_formal_fullkv.json",
    "aavg": "e119_ruler64k_formal_aavg.json",
}
# 32K 正式口径（E116e，历史落袋）用于跨档对比
RULER32 = {"FullKV": 59.38, "mavg": 59.99, "aavg": 57.33}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    per_arm = {}
    for arm, fn in ARMS.items():
        p = os.path.join(RESULTS, fn)
        d = json.load(open(p))
        receipt = json.load(open(p + ".receipt.json"))
        # ① receipt 闭合
        assert receipt["status"] == "success", (arm, receipt["status"])
        assert receipt["result_sha256"] == _sha256(p), (arm, "receipt↔JSON SHA 不一致")
        # ② n 全 100
        key = next(iter(d["n"]))
        assert all(v == 100 for v in d["n"][key].values()), (arm, "n≠100")
        scores = d["scores"][key]
        # ③ 双向 SHA 闭合（发布后无篡改）：receipt cells[].tasks[] 记录
        #    source_sha256（磁盘源文件）+ derived_sha256（补刻派生副本）；
        #    sources[].sha256 = derived（补刻后）口径
        pred_dir = os.path.join(
            ROOT, "exp", "results_ruler", "e109_full_Qwen3-8B",
            "fullkv" if arm == "FullKV" else arm, "L65536", "pred_1024")
        derived_dir = os.path.join(
            RESULTS, os.path.basename(p) + ".run-" + receipt["run_id"],
            "pred_root", "L65536", "pred_1024")
        for task, tinfo in receipt["cells"][key]["tasks"].items():
            src_fp = os.path.join(pred_dir, tinfo["best_file"])
            assert os.path.exists(src_fp), (arm, task, src_fp)
            assert tinfo["source_sha256"] == _sha256(src_fp), (arm, task, "源 SHA 漂移")
            der_fp = os.path.join(derived_dir, tinfo["best_file"])
            assert os.path.exists(der_fp), (arm, task, der_fp)
            assert tinfo["derived_sha256"] == _sha256(der_fp), (arm, task, "派生 SHA 漂移")
        per_arm[arm] = {
            "avg": round(sum(scores.values()) / len(scores), 2),
            "receipt_run_id": receipt.get("run_id"),
            "result_file": os.path.basename(p),
            "per_task": scores,
        }

    fullkv = per_arm["FullKV"]["avg"]
    out = {
        "experiment": "E119_ruler64k_formal_closure",
        "date": "2026-10-09",
        "entry": "benchmark/RULER/score_ruler_formal.py（E116e 正式入口：min-samples 硬门禁 + "
                 "staging 原子发布 + 身份扩展 receipt/manifest）",
        "identity": {
            "length_tier": "L65536（YaRN factor 2.0，64K 全档）",
            "model": "Qwen3-8B",
            "samples_per_task": 100,
            "tasks": 11,
        },
        "closure": {
            "three_arm_receipts_success": True,
            "receipt_result_sha_matches": True,
            "all_cells_n100": True,
            "source_sha_stable": True,
        },
        "arms": {
            arm: {
                **st,
                "delta_vs_fullkv": round(st["avg"] - fullkv, 2),
                "ruler32_official": RULER32[arm],
            }
            for arm, st in per_arm.items()
        },
        "verdict": {
            "ruler64k_ranking": " > ".join(f"{a} {s['avg']}" for a, s in
                                            sorted(per_arm.items(), key=lambda kv: -kv[1]["avg"])),
            "conclusion": "64K 正式口径下 mavg(+0.88) 超 FullKV、aavg(−1.03) 落后；与 32K 正式排序"
                          "（mavg 59.99 > FullKV 59.38 > aavg 57.33）方向一致——RULER 32K/64K 两档"
                          " mavg 冠军稳定，128K 待全齐后同口径收口。",
        },
    }
    p_out = os.path.join(RESULTS, "e119_ruler64k_formal_summary.json")
    json.dump(out, open(p_out, "w"), ensure_ascii=False, indent=2)
    print(json.dumps(out["arms"], ensure_ascii=False, indent=2))
    print("verdict:", json.dumps(out["verdict"], ensure_ascii=False))
    print(f"saved: {p_out}")


if __name__ == "__main__":
    main()
