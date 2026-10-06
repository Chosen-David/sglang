#!/bin/bash
# E98 e2e 精度网格 v2（fix：v1 的 mass top-3 在 e2e K2=1024 预算下 γ 截断坍缩为同配置）
# v2 按「有效配置 (α,β,nt_near 截断后)」去重重选 top-3（/tmp/e98_e2e_runs_v2.json）
# v1 已完成的 mavg a.125/b.25/g.75 两任务由 SKIP 逻辑断点续跑
cd /home/wangyuanshuo02/two-level-attention
export PYTHONPATH=/home/wangyuanshuo02/two-level-attention:/home/wangyuanshuo02/.local/pylibs
export CUDA_VISIBLE_DEVICES=1

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
for r in json.load(open('/tmp/e98_e2e_runs_v2.json')):
    print(json.dumps(r))
")

# ---- 打分 → e98_e2e_grid.json ----
python3 - << 'EOF'
import glob, json, sys
sys.path.insert(0, "/home/wangyuanshuo02/two-level-attention")
from benchmark.LongBench.eval import scorer

runs = json.load(open("/tmp/e98_e2e_runs_v2.json"))
out = {"note": "E98 e2e 精度网格 v2：mass 有效配置 top-3（γ 截断去重，nt_near=min(nb_near*BS*γ,K2)）"
                "× 4 组合 × screen 任务 + 部署参照臂；v1 坍缩臂教训见 note",
       "v1_fix": "v1 选 mass top-3 但 γ≥0.5 臂在 e2e K2=1024 下全部截断为 nt_near=1024 同配置"
                 "（g0.75/g0.875 同分 54.16/35.57 实证），v2 按 (α,β,nt_near) 去重",
       "protocol": "K1=128 K2=1024, n=200"}
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
echo "E98_E2E_GRID_V2_DONE"
