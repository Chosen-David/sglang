#!/usr/bin/env python3
# RULER E71 终表合并：C0（单池）vs B7（分区）三长度消融列
# 等 score_after_ruler_b7.sh 产出 ruler_b7.json 后运行
import json
import os
import sys

import numpy as np

ROOT = "/home/wangyuanshuo02/two-level-attention"
B7 = f"{ROOT}/exp/results_ruler/ruler_b7.json"
if not os.path.exists(B7):
    print("ruler_b7.json 不存在，B7 RULER 尚未打分")
    sys.exit(1)

TASKS = ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1",
         "niah_multikey_2", "niah_multikey_3", "niah_multiquery",
         "niah_multivalue", "cwe", "fwe", "vt"]
base = json.load(open(f"{ROOT}/exp/results_ruler/ruler_table_final.json"))
b7 = json.load(open(B7))
merged = dict(base)
for k, v in b7.items():
    merged[k] = v


def avg3(key):
    vals = {t: round(float(np.mean([merged[f"L{L}/{key}"][t] for L in (4096, 8192, 16384)])), 2)
            for t in TASKS}
    vals["AVG"] = round(float(np.mean(list(vals.values()))), 2)
    return vals


cols = ["none", "quest_64_16", "tia_64_128_1024_c4",
        "tli_64_128_1024_c4_A", "tli_64_128_1024_c4_A"]
# 最后一列换成 B7 的 key（b7 json 的 key 形如 L4096/tli_...）
b7_keys = {v.split("/")[1] for v in b7}
b7_key = sorted(b7_keys)[0] if b7_keys else None
out = {}
for k in ["none", "quest_64_16", "tia_64_128_1024_c4"]:
    out[k] = avg3(k)
out["TLI_C0"] = avg3("tli_64_128_1024_c4_A")
if b7_key:
    out["TLI_B7"] = avg3(b7_key)

names = list(out.keys())
md = ["| task | " + " | ".join(names) + " |", "|" + "---|" * (len(names) + 1)]
for t in TASKS + ["AVG"]:
    md.append("| " + t + " | " + " | ".join(str(out[c][t]) for c in names) + " |")
# per-length AVG 行
for L in (4096, 8192, 16384):
    row = [round(float(np.mean(list(merged[f"L{L}/{k}"].values()))), 2)
           for k in ["none", "quest_64_16", "tia_64_128_1024_c4",
                     "tli_64_128_1024_c4_A"] + ([b7_key] if b7_key else [])]
    md.append(f"| AVG@L{L} | " + " | ".join(str(x) for x in row) + " |")

txt = "\n".join(md)
open(f"{ROOT}/exp/results_ruler/ruler_table_e71_final.md", "w").write(txt + "\n")
json.dump({k: out[k] for k in names},
          open(f"{ROOT}/exp/results_ruler/ruler_table_e71_final.json", "w"), indent=1)
print(txt)
