#!/bin/bash
# E101（审稿 C4 修复）：E98 best 主臂配置（mavg α=0.125 β=0.375 γ=0.625）RULER 全量重扫
# ——统一论文 RULER 双表（tab:ruler16k 与正文 87.93 等）与 LongBench 主表臂的配置口径。
# 触发条件：/tmp/e100_tail_full.log 出现 E100_TAIL_FULL_DONE（E100 双卡跑完释放 GPU）
#
# ===== 历史 RULER 命令原始口径（可追溯，本脚本除 α/β/γ 外照抄）=====
# B7s 臂（β=0.25 γ=0.125，exp/trace/run_scripts/run_ruler_b7.sh）：
#   CUDA_VISIBLE_DEVICES=$GPU python -u -m benchmark.RULER.pred_ruler \
#     --model Qwen3-8B --model_path $MODEL \
#     --task $T --context_length $L --method tli --t $TS \
#     --data-root /home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/RULER \
#     --output-dir exp/results_ruler/Qwen3-8B \
#     --tia_level1_topk 128 --tia_level2_topk 1024 --tia_level2_cmp_ratio 4 \
#     --tli_enable_kmeans false --tli_enable_layer_skip false \
#     --tli_alpha 0.125 --tli_beta 0.25 --tli_gamma 0.125 \
#     --pred_postfix _b7 --max-num 100
# E72mavg 臂（β=0.375 γ=0.125，/tmp/tli_chain/run_ruler_e72m.sh）：同上，仅
#   --tli_alpha 0.125 --tli_beta 0.375 --tli_gamma 0.125 --pred_postfix _e72mavg --t 0930e72m
# 两历史命令均未传 --tli_far_method/--tli_near_method（默认 minmax/avg = mavg）与
# --tli_subspace（默认 full = 主表口径）；本脚本显式传 method（语义等价）但不传 subspace。
# 任务集 11 任务 × 长度 {4096,8192,16384}（pred_ruler.py choices 上限即此三档）× n=100
# （数据集每任务 500 条，--max-num 100 取前 100 条，确定性）。
# 双卡分工照抄历史：GPU0 短长度（4096+8192）/ GPU1 L16384。
# ============================================================
cd /home/wangyuanshuo02/two-level-attention
export PYTHONPATH=/home/wangyuanshuo02/two-level-attention:/home/wangyuanshuo02/.local/pylibs

TAG=E101RULER_mavg_a0.125_b0.375_g0.625
OUTROOT=/tmp/e101_ruler
TS=e101ruler   # 固定 tag：partial 重跑时同路径覆写（pred_ruler 以 "w" 模式打开）
MODEL=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B
DATA=/home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/RULER
N=100
TASKS="niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multiquery niah_multivalue cwe fwe vt"

# ---- 等 E100 双卡跑完 ----
while ! grep -aq "E100_TAIL_FULL_DONE" /tmp/e100_tail_full.log 2>/dev/null; do
  echo "[$(date +%H:%M:%S)] waiting E100_TAIL_FULL_DONE in /tmp/e100_tail_full.log ..."
  sleep 120
done
echo "[$(date +%H:%M:%S)] E100 done, start E101 RULER"

mkdir -p "$OUTROOT"

run_gpu () {
  local GPU=$1; shift
  for L in $@; do
    for T in $TASKS; do
      local PRED_DIR=$OUTROOT/L$L/pred_${TAG}
      if ls "$PRED_DIR"/${T}-tli_*.jsonl >/dev/null 2>&1 && \
         [ "$(cat "$PRED_DIR"/${T}-tli_*.jsonl 2>/dev/null | wc -l)" -ge $N ]; then
        echo "==== SKIP L$L $T (done) ===="; continue
      fi
      echo "==== GPU$GPU L$L $T $(date +%H:%M:%S) ===="
      CUDA_VISIBLE_DEVICES=$GPU python -u -m benchmark.RULER.pred_ruler \
        --model Qwen3-8B --model_path $MODEL \
        --task $T --context_length $L --method tli --t $TS \
        --data-root $DATA \
        --output-dir $OUTROOT \
        --tia_level1_topk 128 --tia_level2_topk 1024 --tia_level2_cmp_ratio 4 \
        --tli_enable_kmeans false --tli_enable_layer_skip false \
        --tli_alpha 0.125 --tli_beta 0.375 --tli_gamma 0.625 \
        --tli_far_method minmax --tli_near_method avg \
        --pred_postfix _${TAG} --max-num $N 2>&1 | tail -2
    done
  done
}

# 双卡分工照抄历史（GPU0 短长度 / GPU1 L16384）
run_gpu 0 4096 8192 &
P0=$!
run_gpu 1 16384 &
P1=$!
wait $P0 $P1

# ---- 打分落袋 e101_ruler_g0625.json（打分逻辑照抄 benchmark/RULER/score_ruler.py 的
#      string_match_all）+ 旧臂对照 delta ----
python3 - << 'PYEOF'
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
PYEOF
echo "E101_RULER_DONE"
