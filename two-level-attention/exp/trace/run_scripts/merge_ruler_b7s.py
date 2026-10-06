#!/usr/bin/env python3
# RULER B7s（严格口径）三长度终表合并：C0 vs B7s + v1caliber 口径差列
# 等 b7s_ruler_relay.sh 产出 ruler_b7s.json 后运行
import json
import os
import sys

import numpy as np

ROOT = "/home/wangyuanshuo02/two-level-attention"
B7S = f"{ROOT}/exp/results_ruler/ruler_b7s.json"
if not os.path.exists(B7S):
    print("ruler_b7s.json 不存在，B7s RULER 尚未打分")
    sys.exit(1)

TASKS = ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1",
         "niah_multikey_2", "niah_multikey_3", "niah_multiquery",
         "niah_multivalue", "cwe", "fwe", "vt"]


def avg3(merged, key):
    vals = {t: round(float(np.mean([merged[f"L{L}/{key}"][t] for L in (4096, 8192, 16384)])), 2)
            for t in TASKS}
    vals["AVG"] = round(float(np.mean(list(vals.values()))), 2)
    return vals


merged = {}
for f in ("ruler_table_final.json", "ruler_table_e71_final.json"):
    p = f"{ROOT}/exp/results_ruler/{f}"
    if os.path.exists(p):
        merged.update(json.load(open(p)))
b7s = json.load(open(B7S))
merged.update(b7s)
out = {}
for k in ["none", "quest_64_16", "tia_64_128_1024_c4"]:
    out[k] = avg3(merged, k)
out["TLI_C0"] = avg3(merged, "tli_64_128_1024_c4_A")
b7s_keys = sorted({v.split("/")[1] for v in b7s})
out["TLI_B7s"] = avg3(merged, b7s_keys[0]) if b7s_keys else None
# v1caliber 对照（旧口径 B7）
if os.path.exists(f"{ROOT}/exp/results_ruler/ruler_b7.json"):
    v1 = json.load(open(f"{ROOT}/exp/results_ruler/ruler_b7.json"))
    merged.update(v1)
    v1_keys = sorted({v.split("/")[1] for v in v1})
    out["TLI_B7_v1caliber"] = avg3(merged, v1_keys[0]) if v1_keys else None
json.dump(out, open(f"{ROOT}/exp/results_ruler/ruler_table_b7s_final.json", "w"), indent=1)
print(json.dumps({k: v["AVG"] if v else None for k, v in out.items()}, indent=1))
print("saved -> ruler_table_b7s_final.json")
