#!/bin/bash
# E104（round4 MAJOR #111）：RULER 32K 档双臂补点
# ——论文 RULER 双表（主臂 γ.625 85.83 / 参照臂 87.93）目前只覆盖 {4096,8192,16384}，
#   补 32768 档：主臂（mavg α.125 β.375 γ.625，E101 同配置）+ 参照臂（b7s β.25 γ.125，
#   论文 87.93 主源口径），验证「伤害集中 cwe/fwe 随长度放大、其余任务持平」叙事在 32K 的延伸。
# 数据：官方 RULER 生成器产 11 任务×100 条（/tmp/ruler_gen.log），已归位 data-root 32768/。
# 命令口径：除 α/β/γ 与 L 外逐字照抄 E101（/tmp/e101_ruler_g0625.sh）与 b7s 历史命令
#   （exp/trace/run_scripts/run_ruler_b7.sh 注释）——tia_level1_topk 128 / level2_topk 1024 /
#   cmp_ratio 4 / method 显式 mavg / subspace 默认 full / max-num 100。
# 双卡分工：GPU0 主臂 γ.625 / GPU1 参照臂 γ.125（各 11 任务串行）。
cd /home/wangyuanshuo02/two-level-attention
export PYTHONPATH=/home/wangyuanshuo02/two-level-attention:/home/wangyuanshuo02/.local/pylibs

L=32768
N=100
MODEL=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B
DATA=/home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/RULER
TASKS="niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multiquery niah_multivalue cwe fwe vt"

run_arm () {
  local GPU=$1; local TAG=$2; local TS=$3; local ALPHA=$4; local BETA=$5; local GAMMA=$6
  local OUTROOT=/tmp/e104_ruler_32k
  mkdir -p "$OUTROOT"
  for T in $TASKS; do
    local PRED_DIR=$OUTROOT/pred_${TAG}
    if ls "$PRED_DIR"/${T}-tli_*.jsonl >/dev/null 2>&1 && \
       [ "$(cat "$PRED_DIR"/${T}-tli_*.jsonl 2>/dev/null | wc -l)" -ge $N ]; then
      echo "==== SKIP $TAG $T (done) ===="; continue
    fi
    echo "==== GPU$GPU $TAG $T $(date +%H:%M:%S) ===="
    CUDA_VISIBLE_DEVICES=$GPU python -u -m benchmark.RULER.pred_ruler \
      --model Qwen3-8B --model_path $MODEL \
      --task $T --context_length $L --method tli --t $TS \
      --data-root $DATA \
      --output-dir $OUTROOT \
      --tia_level1_topk 128 --tia_level2_topk 1024 --tia_level2_cmp_ratio 4 \
      --tli_enable_kmeans false --tli_enable_layer_skip false \
      --tli_alpha $ALPHA --tli_beta $BETA --tli_gamma $GAMMA \
      --tli_far_method minmax --tli_near_method avg \
      --pred_postfix _${TAG} --max-num $N 2>&1 | tail -2
  done
}

# GPU0 主臂 γ.625（E98 best / E101 同配置）
run_arm 0 E104MAIN_mavg_a0.125_b0.375_g0.625 e104main 0.125 0.375 0.625 &
P0=$!
# GPU1 参照臂 γ.125（b7s 论文 87.93 主源口径 β.25/γ.125）
run_arm 1 E104REF_mavg_a0.125_b0.25_g0.125 e104ref 0.125 0.25 0.125 &
P1=$!
wait $P0 $P1

# ---- 打分落袋 e104_ruler_32k.json（string_match_all 逐字照抄 score_ruler.py）----
python3 - << 'PYEOF'
import glob, json

OUTROOT = "/tmp/e104_ruler_32k"
TASKS = ["niah_single_1", "niah_single_2", "niah_single_3",
         "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
         "niah_multiquery", "niah_multivalue", "cwe", "fwe", "vt"]
ARMS = {"main_g0625": "E104MAIN_mavg_a0.125_b0.375_g0.625",
        "ref_g0125": "E104REF_mavg_a0.125_b0.25_g0.125"}
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

out = {
    "note": ("E104 round4 MAJOR #111：RULER 32768 档双臂补点——主臂（mavg α=0.125 "
             "β=0.375 γ=0.625，E101 同配置）+ 参照臂（mavg α=0.125 β=0.25 γ=0.125，"
             "b7s 论文 87.93 主源口径），11 任务×n=100，官方生成器产数据"),
    "arms": {
        "main_g0625": {"alpha": 0.125, "beta": 0.375, "gamma": 0.625},
        "ref_g0125": {"alpha": 0.125, "beta": 0.25, "gamma": 0.125},
    },
    "context_length": 32768,
    "budget": "level1_topk 128 / level2_topk 1024 / cmp_ratio 4 / subspace full(默认)",
    "scores": scores,
    "delta_main_minus_ref": delta,
    "e101_context": ("16K 及以下：主臂 overall 85.83 vs 参照臂 87.93（−2.10），"
                     "伤害集中 cwe（82.0/68.6/55.5 随长度放大）+ fwe 微伤，其余 8 任务持平"),
}
json.dump(out, open(f"{ROOT}/exp/trace/results/e104_ruler_32k.json", "w"),
          indent=1, ensure_ascii=False)
print(json.dumps(out, ensure_ascii=False, indent=1))
print("saved exp/trace/results/e104_ruler_32k.json")
PYEOF
echo "E104_RULER_32K_DONE"
