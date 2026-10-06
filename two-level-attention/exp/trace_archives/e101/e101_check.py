import glob, json, os

TAG = "E101RULER_mavg_a0.125_b0.375_g0.625"
OUTROOT = "/tmp/e101_ruler"
TASKS = ["niah_single_1", "niah_single_2", "niah_single_3",
         "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
         "niah_multiquery", "niah_multivalue", "cwe", "fwe", "vt"]
LENGTHS = [4096, 8192, 16384]
N = 100
ROOT = "/home/wangyuanshuo02/two-level-attention"


def string_match_all(preds, refs):
    # 与 benchmark/RULER/score_ruler.py 逐位一致（官方 RULER 口径）
    total = 0.0
    for pred, ref in zip(preds, refs):
        hit = sum(1.0 if r.lower() in pred.lower() else 0.0 for r in ref)
        total += hit / len(ref) if ref else 0.0
    return total / len(preds) * 100 if preds else 0.0


def load_old(path, e72_style):
    """旧臂 JSON 归一化为 {L: {task: score}}（不含 AVG 键）
    b7s_ruler_final.json: {"L4096": {task: v, "AVG": v}, ...}（论文 87.93 主源，β.25/γ.125）
    ruler_e72mavg.json:   {"L4096/tli_64_128_1024_c4_A": {task: v}, ...}（β.375/γ.125）"""
    if not os.path.exists(path):
        return None
    raw = json.load(open(path))
    out = {}
    if e72_style:
        for k, tasks in raw.items():
            L = k.split("/")[0]
            out[L] = {t: v for t, v in tasks.items() if t != "AVG"}
    else:
        for L, tasks in raw.items():
            out[L] = {t: v for t, v in tasks.items() if t != "AVG"}
    return out


scores = {}
for L in LENGTHS:
    pred_dir = f"{OUTROOT}/L{L}/pred_{TAG}"
    row = {}
    for t in TASKS:
        files = sorted(glob.glob(f"{pred_dir}/{t}-*.jsonl"))
        ok = [f for f in files if sum(1 for _ in open(f)) >= N]
        if not ok:
            row[t] = None
            print(f"[WARN] L{L} {t}: no complete pred (n>={N})")
            continue
        preds, refs = [], []
        for line in open(ok[-1]):
            d = json.loads(line)
            preds.append(d["pred"])
            refs.append(d["answers"])
        row[t] = round(string_match_all(preds, refs), 2)
    vals = [v for v in row.values() if v is not None]
    row["AVG"] = round(sum(vals) / len(vals), 2) if vals else None
    scores[f"L{L}"] = row

# 总 AVG = 33 任务值等权均值（论文口径：三长度 AVG 的均值，二者在 11 任务齐全时相等）
all_v = [v for L in LENGTHS for t, v in scores[f"L{L}"].items()
         if t != "AVG" and v is not None]
overall = round(sum(all_v) / len(all_v), 2) if all_v else None

old_b25 = load_old("/tmp/tli_chain/b7s_ruler_final.json", e72_style=False)   # β.25/γ.125 论文主源
old_b375 = load_old(f"{ROOT}/exp/results_ruler/ruler_e72mavg.json", e72_style=True)  # β.375/γ.125


def delta(old):
    if old is None:
        return None
    out = {}
    for L in LENGTHS:
        d = {}
        for t in TASKS:
            new_v = scores[f"L{L}"].get(t)
            old_v = old.get(f"L{L}", {}).get(t)
            d[t] = round(new_v - old_v, 2) if (new_v is not None and old_v is not None) else None
        nv = scores[f"L{L}"].get("AVG")
        ov = old.get(f"L{L}", {})
        ovals = [v for t, v in ov.items() if t != "AVG" and isinstance(v, (int, float))]
        oavg = round(sum(ovals) / len(ovals), 2) if ovals else None
        d["AVG"] = round(nv - oavg, 2) if (nv is not None and oavg is not None) else None
        out[f"L{L}"] = d
    return out


def old_overall(old):
    if old is None:
        return None
    vals = [v for L in LENGTHS for t, v in old.get(f"L{L}", {}).items()
            if t != "AVG" and isinstance(v, (int, float))]
    return round(sum(vals) / len(vals), 2) if vals else None


out = {
    "note": ("E101 审稿C4修复：E98 best 主臂配置（mavg α=0.125 β=0.375 γ=0.625，"
             "L1 全维 full 默认口径）RULER 11任务×3长度×n=100 重扫，"
             "统一论文 RULER 双表与 LongBench 主表臂（50.78）配置口径"),
    "TAG": TAG,
    "config": {"alpha": 0.125, "beta": 0.375, "gamma": 0.625,
               "far_method": "minmax", "near_method": "avg",
               "subspace": "full(默认未传)", "n_per_task": N,
               "lengths": LENGTHS,
               "budget": "level1_topk 128 / level2_topk 1024 / cmp_ratio 4"},
    "scores": scores,
    "overall_AVG": overall,
    "old_refs": {
        "b25_g0125_b7s_paper_arm": {"path": "/tmp/tli_chain/b7s_ruler_final.json",
                                     "overall_AVG": old_overall(old_b25)},
        "b375_g0125_e72mavg": {"path": "exp/results_ruler/ruler_e72mavg.json",
                               "overall_AVG": old_overall(old_b375)},
    },
    "delta_vs_b25_g0125": delta(old_b25),
    "delta_vs_b375_g0125": delta(old_b375),
}
json.dump(out, open(f"{ROOT}/exp/trace/results/e101_ruler_g0625.json", "w"),
          indent=1, ensure_ascii=False)
print(json.dumps(out, ensure_ascii=False, indent=1))
print("saved exp/trace/results/e101_ruler_g0625.json")
