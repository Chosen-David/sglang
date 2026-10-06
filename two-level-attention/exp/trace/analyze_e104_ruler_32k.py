# E104 修正版打分：pred_ruler 实际输出在 {OUTROOT}/L32768/pred_{TAG}/（自动加 L{length} 层级），
# 原脚本 /tmp/e104_ruler_32k.sh 尾部打分 glob 路径少了该层级——跑完后本脚本覆盖重打。
# E104b 扩展：同时打五方法臂（FullKV/Quest/TIA/单池，/tmp/e104b_ruler_32k_5arms.sh），
# 凑齐主表 tab:ruler 的 32K 行；六臂不齐时缺失臂记 null（部分收割可用）。
# 打分逻辑（string_match_all）与 score_ruler.py / E101 逐字一致。
import glob
import json

OUTROOT = "/tmp/e104_ruler_32k/L32768"
TASKS = ["niah_single_1", "niah_single_2", "niah_single_3",
         "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
         "niah_multiquery", "niah_multivalue", "cwe", "fwe", "vt"]
ARMS = {"main_g0625": "E104MAIN_mavg_a0.125_b0.375_g0.625",
        "ref_g0125": "E104REF_mavg_a0.125_b0.25_g0.125",
        "fullkv": "E104B_FULLKV",
        "quest": "E104B_QUEST",
        "tia": "E104B_TIA",
        "single_pool": "E104B_C0"}
N = 100
ROOT = "/home/wangyuanshuo02/two-level-attention"


def string_match_all(preds, refs):
    # 与 benchmark/RULER/score_ruler.py 逐位一致（官方 RULER 口径）
    total = 0.0
    for pred, ref in zip(preds, refs):
        hit = sum(1.0 if r.lower() in pred.lower() else 0.0 for r in ref)
        total += hit / len(ref) if ref else 0.0
    return total / len(preds) * 100 if preds else 0.0


scores = {}
for arm, tag in ARMS.items():
    row = {}
    for t in TASKS:
        files = sorted(glob.glob(f"{OUTROOT}/pred_{tag}/{t}-*.jsonl"))
        ok = [f for f in files if sum(1 for _ in open(f)) >= N]
        if not ok:
            row[t] = None
            print(f"[WARN] {arm} {t}: no complete pred (n>={N})")
            continue
        preds, refs = [], []
        for line in open(ok[-1]):
            d = json.loads(line)
            preds.append(d["pred"])
            refs.append(d["answers"])
        row[t] = round(string_match_all(preds, refs), 2)
    vals = [v for v in row.values() if v is not None]
    row["AVG"] = round(sum(vals) / len(vals), 2) if vals else None
    scores[arm] = row

# 主臂 vs 参照臂 delta（E101 判决口径：伤害集中 cwe/fwe）
delta = {}
if scores["main_g0625"].get("AVG") is not None and scores["ref_g0125"].get("AVG") is not None:
    for t in TASKS + ["AVG"]:
        m, r = scores["main_g0625"].get(t), scores["ref_g0125"].get(t)
        delta[t] = round(m - r, 2) if (m is not None and r is not None) else None

# 参照臂 vs FullKV（「PSI 持平 FullKV」32K 延伸）与 vs Quest（下界崩塌第四点）
delta_ref_vs_fullkv = {}
delta_ref_vs_quest = {}
if scores.get("fullkv", {}).get("AVG") is not None:
    for t in TASKS + ["AVG"]:
        a, b = scores["ref_g0125"].get(t), scores["fullkv"].get(t)
        delta_ref_vs_fullkv[t] = round(a - b, 2) if (a is not None and b is not None) else None
if scores.get("quest", {}).get("AVG") is not None:
    for t in TASKS + ["AVG"]:
        a, b = scores["ref_g0125"].get(t), scores["quest"].get(t)
        delta_ref_vs_quest[t] = round(a - b, 2) if (a is not None and b is not None) else None

out = {
    "note": ("E104/E104b round4 MAJOR #111：RULER 32768 档补点——双 TLI 臂"
             "（主臂 mavg α=0.125 β=0.375 γ=0.625，E101 同配置；参照臂 mavg α=0.125 "
             "β=0.25 γ=0.125，b7s 论文 87.93 主源口径）+ 五方法臂（FullKV/Quest/TIA/"
             "单池，凑主表 32K 行），11 任务×n=100，官方生成器产数据"),
    "arms": {
        "main_g0625": {"alpha": 0.125, "beta": 0.375, "gamma": 0.625},
        "ref_g0125": {"alpha": 0.125, "beta": 0.25, "gamma": 0.125},
        "fullkv": {"method": "none"},
        "quest": {"method": "quest_64_16 (defaults)"},
        "tia": {"method": "tia_64_128_1024_c4"},
        "single_pool": {"method": "tli α=0/β=0/γ=1 (C0 defaults)"},
    },
    "context_length": 32768,
    "budget": "level1_topk 128 / level2_topk 1024 / cmp_ratio 4 / subspace full(默认)",
    "scores": scores,
    "delta_main_minus_ref": delta,
    "delta_ref_minus_fullkv": delta_ref_vs_fullkv,
    "delta_ref_minus_quest": delta_ref_vs_quest,
    "e101_context": ("16K 及以下：主臂 overall 85.83 vs 参照臂 87.93（−2.10），"
                     "伤害集中 cwe（82.0/68.6/55.5 随长度放大）+ fwe 微伤，其余 8 任务持平；"
                     "Quest 4K/8K/16K = 87.91/81.76/70.18 随长度崩塌"),
}
json.dump(out, open(f"{ROOT}/exp/trace/results/e104_ruler_32k.json", "w"),
          indent=1, ensure_ascii=False)
print(json.dumps(out, ensure_ascii=False, indent=1))
print("saved exp/trace/results/e104_ruler_32k.json")
