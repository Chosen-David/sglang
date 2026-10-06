# E64j 收尾：提取每组合最佳 (α,β) + mono 对照 → e64j_best.json（供 e2e 快筛编排读取）
import json

RES = "/home/wangyuanshuo02/two-level-attention/exp/trace/results/e64j_combo_best.json"
OUT = "/tmp/tli_chain/e64j_best.json"
PAIRS = [("mavg", "mavg"), ("mavg", "aavg"), ("cavg", "cavg"),
         ("cavg", "aavg"), ("aavg", "aavg")]

d = json.load(open(RES))
if "AVG" not in d:
    raise SystemExit("E64j not finished (no AVG)")
avg = d["AVG"]

best = {}
print("== E64j 五组合各自最佳配置（16 样本 AVG, mass coverage）==")
for f_m, n_m in PAIRS:
    cands = {k: v for k, v in avg.items() if k.startswith(f"{f_m}+{n_m}_")}
    if not cands:
        continue
    bk, bv = max(cands.items(), key=lambda x: x[1])
    # 解析 a{a}_b{b}
    tail = bk.split("_a")[1]
    a_s, b_s = tail.split("_b")
    best[f"{f_m}+{n_m}"] = {"alpha": float(a_s), "beta": float(b_s), "mass": bv, "key": bk}
    print(f"  {f_m}+{n_m:5s} best a={a_s} b={b_s} mass={bv:.4f}")
for m in ("mavg", "cavg", "aavg"):
    print(f"  {m}_mono {avg.get(f'{m}_mono')}")
mono_max = max(avg.get(f"{m}_mono", 0) for m in ("mavg", "cavg", "aavg"))
best["_meta"] = {"mono_max": mono_max,
                 "part_max": max(v["mass"] for v in best.values() if isinstance(v, dict))}
json.dump(best, open(OUT, "w"), indent=1)
print("saved ->", OUT)
