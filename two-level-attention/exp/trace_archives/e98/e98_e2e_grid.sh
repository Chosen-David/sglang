#!/bin/bash
# E98 第二阶段：e2e 精度网格（用户指令：不仅 trace 回放，实际精度数据必须测）
# 触发条件：① E98 mass 网格完成（e98_abg_full_grid.json 落袋）② GPU1 pyramidkv sinkguard 完成
# 设计：4 个 e2e 可跑 method 组合 × mass 排名 top-3 (α,β,γ) × {hotpotqa, musique}
#   + 参照臂（当前部署 mavg α.125/β.375/γ.125）成对对照
#   ccluster（near 侧 cluster）e2e 无实现路径——只有 mass 数据，报告如实注明
# 用法：nohup bash /tmp/e98_e2e_grid.sh > /tmp/e98_e2e_grid.log 2>&1 &
cd /home/wangyuanshuo02/two-level-attention
export PYTHONPATH=/home/wangyuanshuo02/two-level-attention:/home/wangyuanshuo02/.local/pylibs
export CUDA_VISIBLE_DEVICES=1
E98_JSON=exp/trace/results/e98_abg_full_grid.json

# ---- 条件等待：E98 mass 网格 ----
while [ ! -f "$E98_JSON" ]; do
  echo "[$(date +%H:%M:%S)] waiting E98 mass grid..."; sleep 300
done
echo "[$(date +%H:%M:%S)] E98 mass grid ready"

# ---- 条件等待：GPU1 pyramidkv 完成 ----
while ! grep -q "M6_SINKGUARD_GPU1_PYRAMIDKV_DONE" /tmp/m6_pyramidkv_gpu1.log 2>/dev/null; do
  echo "[$(date +%H:%M:%S)] waiting GPU1 pyramidkv..."; sleep 300
done
echo "[$(date +%H:%M:%S)] GPU1 free, launching e2e grid"

# ---- 从 mass 网格选各组合 top-3 (α,β,γ) → 生成运行清单 ----
python3 - << 'EOF'
import json
d = json.load(open("exp/trace/results/e98_abg_full_grid.json"))
mean = d["mean"]
runs = []
for combo in ["mavg", "mminmax", "aavg", "cavg"]:
    arms = {k: v for k, v in mean.items()
            if k.startswith(f"{combo}_") }
    # 过滤：α/β 必须在开区间（角点无意义臂 mass 反而高是池合并假象）
    valid = {}
    for k, v in arms.items():
        parts = k.split("_")  # combo_a0.125_b0.25_g1.0
        a, b = float(parts[1][1:]), float(parts[2][1:])
        if 0 < a < 1 and 0 < b < 1:
            valid[k] = v
    top3 = sorted(valid.items(), key=lambda x: -x[1])[:3]
    for k, v in top3:
        parts = k.split("_")
        a, b, g = parts[1][1:], parts[2][1:], parts[3][1:]
        runs.append({"combo": combo, "a": a, "b": b, "g": g, "mass": v})
# 参照臂（当前部署配置）
runs.append({"combo": "mavg", "a": "0.125", "b": "0.375", "g": "0.125",
             "mass": mean.get("mavg_a0.125_b0.375_g0.125"), "ref": True})
json.dump(runs, open("/tmp/e98_e2e_runs.json", "w"), indent=1)
print(f"{len(runs)} runs:", *[f"{r['combo']} a{r['a']} b{r['b']} g{r['g']}" for r in runs], sep="\n  ")
EOF

# ---- 运行 e2e（每配置 × 2 任务）----
# method 映射：mavg=(minmax,avg) mminmax=(minmax,minmax) aavg=(avg,avg)
#   cavg=(cluster,avg)=--tli_far_select cluster + kmeans；预算与主表同口径 K1=128 K2=1024
while read -r line; do
  combo=$(echo "$line" | python3 -c "import json,sys; r=json.load(sys.stdin); print(r['combo'])")
  a=$(echo "$line" | python3 -c "import json,sys; r=json.load(sys.stdin); print(r['a'])")
  b=$(echo "$line" | python3 -c "import json,sys; r=json.load(sys.stdin); print(r['b'])")
  g=$(echo "$line" | python3 -c "import json,sys; r=json.load(sys.stdin); print(r['g'])")
  case $combo in
    mavg)     FM=minmax; NM=avg;    EXTRA="";;
    mminmax)  FM=minmax; NM=minmax; EXTRA="";;
    aavg)     FM=avg;    NM=avg;    EXTRA="";;
    cavg)     FM=minmax; NM=avg;    EXTRA="--tli_enable_kmeans true --tli_far_select cluster";;
  esac
  TAG=${combo}_a${a}_b${b}_g${g}
  for task in hotpotqa musique; do
    # 已有完整输出则跳过（断点续跑）
    OUTDIR=/tmp/e98_e2e/pred_${TAG}
    mkdir -p "$OUTDIR"
    if ls "$OUTDIR"/${task}-tli_*.jsonl >/dev/null 2>&1 && \
       [ "$(cat "$OUTDIR"/${task}-tli_*.jsonl 2>/dev/null | wc -l)" -ge 200 ]; then
      echo "==== SKIP $task $TAG (done) ===="
      continue
    fi
    echo "==== $task $TAG $(date +%H:%M:%S) ===="
    python -u -m benchmark.LongBench.pred \
      --model Qwen3-8B \
      --model_path /mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B \
      --task $task --method tli --e 0 \
      --dataset-path /home/wangyuanshuo02/datasets/LongBench/data \
      --config-path benchmark/LongBench/config \
      --output-dir /tmp/e98_e2e --pred_postfix _${TAG} \
      --tia_level1_topk 128 --tia_level2_topk 1024 --tia_level2_cmp_ratio 4 \
      --tli_enable_kmeans false --tli_enable_layer_skip false \
      --tli_alpha $a --tli_beta $b --tli_gamma $g \
      --tli_far_method $FM --tli_near_method $NM $EXTRA \
      2>&1 | tail -2
  done
done < <(python3 -c "
import json
for r in json.load(open('/tmp/e98_e2e_runs.json')):
    print(json.dumps(r))
")

# ---- 打分 → e98_e2e_grid.json ----
python3 - << 'EOF'
import glob, json, sys
sys.path.insert(0, "/home/wangyuanshuo02/two-level-attention")
from benchmark.LongBench.eval import scorer

runs = json.load(open("/tmp/e98_e2e_runs.json"))
out = {"note": "E98 e2e 精度网格：mass top-3 (a,b,g) × 4 组合 × screen 任务 + 部署参照臂",
       "protocol": "K1=128 K2=1024, L1 上界全维口径（与主表臂同），n=200"}
for r in runs:
    tag = f"{r['combo']}_a{r['a']}_b{r['b']}_g{r['g']}"
    d = f"/tmp/e98_e2e/pred_{tag}"
    row = {"mass": r.get("mass")}
    if r.get("ref"):
        row["ref"] = True
    for task in ["hotpotqa", "musique"]:
        fs = sorted(glob.glob(f"{d}/{task}-tli_*.jsonl"))
        fs = [f for f in fs if sum(1 for _ in open(f)) >= 200]
        if not fs:
            row[task] = None
            continue
        preds, answers, allc = [], [], None
        for line in open(fs[-1]):
            j = json.loads(line)
            preds.append(j["pred"]); answers.append(j["answers"]); allc = j["all_classes"]
        row[task] = round(scorer(task, preds, answers, allc), 2)
    out[tag] = row
    print(tag, row)
json.dump(out, open("/home/wangyuanshuo02/two-level-attention/exp/trace/results/e98_e2e_grid.json", "w"),
          indent=1, ensure_ascii=False)
print("saved e98_e2e_grid.json")
EOF
echo "E98_E2E_GRID_DONE"
