#!/bin/bash
# E98 第三阶段：best (α,β,γ)×method 13 任务全量（LongBench e2e 新主表）
# 触发条件：e98_best_election.json 落袋（best 选举完成）
# 双卡分工：GPU0 前半任务、GPU1 后半（每任务 200 样本）
cd /home/wangyuanshuo02/two-level-attention
export PYTHONPATH=/home/wangyuanshuo02/two-level-attention:/home/wangyuanshuo02/.local/pylibs

while [ ! -f exp/trace/results/e98_best_election.json ]; do
  echo "[$(date +%H:%M:%S)] waiting best election..."; sleep 120
done

# 从选举 JSON 提取 best 配置
read -r COMBO A B G < <(python3 -c "
import json
b = json.load(open('exp/trace/results/e98_best_election.json'))['best']
tag = b['tag']
parts = tag.split('_')
print(parts[0], parts[1][1:], parts[2][1:], parts[3][1:])
")
case $COMBO in
  mavg)     FM=minmax; NM=avg;    EXTRA="";;
  mminmax)  FM=minmax; NM=minmax; EXTRA="";;
  aavg)     FM=avg;    NM=avg;    EXTRA="";;
  cavg)     FM=minmax; NM=avg;    EXTRA="--tli_enable_kmeans true --tli_far_select cluster";;
esac
TAG=E98BEST_${COMBO}_a${A}_b${B}_g${G}
echo "[$(date +%H:%M:%S)] BEST=$TAG FM=$FM NM=$NM EXTRA=$EXTRA"

# 13 任务（与 E71/E72 主表口径严格一致：含 multi_news，无 trec/samsum；multifieldqa_en 全集 150）
# 任务键集已从 e71_main_table.json TLI_E72 键核实
run_gpu () {
  local GPU=$1; shift
  local TL=$1; shift
  for t in $TL; do
    OUTDIR=/tmp/e98_full/pred_${TAG}
    mkdir -p "$OUTDIR"
    if ls "$OUTDIR"/${t}-tli_*.jsonl >/dev/null 2>&1 && \
       [ "$(cat "$OUTDIR"/${t}-tli_*.jsonl 2>/dev/null | wc -l)" -ge 200 ]; then
      echo "==== SKIP $t (done) ===="
      continue
    fi
    echo "==== GPU$GPU $t $(date +%H:%M:%S) ===="
    CUDA_VISIBLE_DEVICES=$GPU python -u -m benchmark.LongBench.pred \
      --model Qwen3-8B \
      --model_path /mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B \
      --task $t --method tli --e 0 \
      --dataset-path /home/wangyuanshuo02/datasets/LongBench/data \
      --config-path benchmark/LongBench/config \
      --output-dir /tmp/e98_full --pred_postfix _${TAG} \
      --tia_level1_topk 128 --tia_level2_topk 1024 --tia_level2_cmp_ratio 4 \
      --tli_enable_kmeans false --tli_enable_layer_skip false \
      --tli_alpha $A --tli_beta $B --tli_gamma $G \
      --tli_far_method $FM --tli_near_method $NM $EXTRA \
      2>&1 | tail -2
  done
}

# 按耗时均衡分卡（长任务分摊）：GPU0 前段、GPU1 后段
run_gpu 0 "hotpotqa musique narrativeqa qasper multifieldqa_en gov_report qmsum multi_news" &
P0=$!
run_gpu 1 "triviaqa passage_retrieval_en lcc repobench-p 2wikimqa" &
P1=$!
wait $P0 $P1

# ---- 打分 → e98_full_13tasks.json ----
python3 - << 'EOF'
import glob, json, sys
sys.path.insert(0, "/home/wangyuanshuo02/two-level-attention")
from benchmark.LongBench.eval import scorer

best = json.load(open("exp/trace/results/e98_best_election.json"))["best"]
tag = best["tag"]
TAG = f"E98BEST_{tag}"
d = f"/tmp/e98_full/pred_{TAG}"
# 任务集从 E71 主表键取
e71 = json.load(open("exp/trace/results/e71_main_table.json"))
TASKS = [k for k in e71.get("TLI_E72", {}).keys() if k != "AVG"]
out = {"note": "E98 best 配置 13 任务全量（LongBench e2e 主表臂）",
       "best": best, "TAG": TAG}
row = {}
for t in TASKS:
    fs = sorted(glob.glob(f"{d}/{t}-tli_*.jsonl"))
    ok = []
    for f in fs:
        n = sum(1 for _ in open(f))
        if n >= 200 or (t == "multifieldqa_en" and n >= 150):
            ok.append(f)
    if not ok:
        row[t] = None
        continue
    preds, answers, allc = [], [], None
    for line in open(ok[-1]):
        j = json.loads(line)
        preds.append(j["pred"]); answers.append(j["answers"]); allc = j["all_classes"]
    # eval 键名映射：repobench-p 文件 → repobench 键
    key = "repobench" if t == "repobench-p" else t
    row[t] = round(scorer(key, preds, answers, allc), 2)
vals = [v for v in row.values() if v is not None]
row["AVG"] = round(sum(vals) / len(vals), 2) if vals else None
out["tasks"] = row
print(json.dumps(out, ensure_ascii=False, indent=1))
json.dump(out, open("exp/trace/results/e98_full_13tasks.json", "w"), indent=1, ensure_ascii=False)
print("saved e98_full_13tasks.json")
EOF
echo "E98_FULL_13TASKS_DONE"
