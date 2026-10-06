#!/bin/bash
# E100（审稿 C1 修复）：tail32 L1 臂 13 任务全量——把评测系统对齐论文描述系统
# 配置 = E98 best（mavg α.125/β.375/γ.625）+ --tli_subspace tail（L1 块 min/max 在 tail32 上算）
# 双卡分工同 e98_full_13tasks.sh；输出 /tmp/e100_tail/pred_E100TAIL_*
cd /home/wangyuanshuo02/two-level-attention
export PYTHONPATH=/home/wangyuanshuo02/two-level-attention:/home/wangyuanshuo02/.local/pylibs

TAG=E100TAIL_mavg_a0.125_b0.375_g0.625
run_gpu () {
  local GPU=$1; shift
  for t in $@; do
    OUTDIR=/tmp/e100_tail/pred_${TAG}
    mkdir -p "$OUTDIR"
    if ls "$OUTDIR"/${t}-tli_*.jsonl >/dev/null 2>&1 && \
       [ "$(cat "$OUTDIR"/${t}-tli_*.jsonl 2>/dev/null | wc -l)" -ge 200 ]; then
      echo "==== SKIP $t (done) ===="; continue
    fi
    echo "==== GPU$GPU $t $(date +%H:%M:%S) ===="
    CUDA_VISIBLE_DEVICES=$GPU python -u -m benchmark.LongBench.pred \
      --model Qwen3-8B \
      --model_path /mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B \
      --task $t --method tli --e 0 \
      --dataset-path /home/wangyuanshuo02/datasets/LongBench/data \
      --config-path benchmark/LongBench/config \
      --output-dir /tmp/e100_tail --pred_postfix _${TAG} \
      --tia_level1_topk 128 --tia_level2_topk 1024 --tia_level2_cmp_ratio 4 \
      --tli_enable_kmeans false --tli_enable_layer_skip false \
      --tli_alpha 0.125 --tli_beta 0.375 --tli_gamma 0.625 \
      --tli_far_method minmax --tli_near_method avg \
      --tli_subspace tail \
      2>&1 | tail -2
  done
}
run_gpu 0 "hotpotqa musique narrativeqa qasper multifieldqa_en gov_report qmsum multi_news" &
P0=$!
run_gpu 1 "triviaqa passage_retrieval_en lcc repobench-p 2wikimqa" &
P1=$!
wait $P0 $P1

# 打分落袋 e100_tail_full.json（对照 E98 best full 臂逐任务差值）
python3 - << 'PYEOF'
import glob, json, sys
sys.path.insert(0, "/home/wangyuanshuo02/two-level-attention")
from benchmark.LongBench.eval import scorer
TAG = "E100TAIL_mavg_a0.125_b0.375_g0.625"
d = f"/tmp/e100_tail/pred_{TAG}"
e71 = json.load(open("exp/trace/results/e71_main_table.json"))
TASKS = [k for k in e71.get("TLI_E72", {}).keys() if k != "AVG"]
out = {"note": "E100 审稿C1修复：tail32 L1 臂 13 任务全量（E98 best 配置 + tli_subspace tail），对照 full-L1 主表臂 50.78",
       "TAG": TAG}
row = {}
for t in TASKS:
    fs = sorted(glob.glob(f"{d}/{t}-tli_*.jsonl"))
    ok = [f for f in fs if sum(1 for _ in open(f)) >= 200 or (t == "multifieldqa_en" and sum(1 for _ in open(f)) >= 150)]
    if not ok:
        row[t] = None; continue
    preds, answers, allc = [], [], None
    for line in open(ok[-1]):
        j = json.loads(line)
        preds.append(j["pred"]); answers.append(j["answers"]); allc = j["all_classes"]
    key = "repobench" if t == "repobench-p" else t
    row[t] = round(scorer(key, preds, answers, allc), 2)
vals = [v for v in row.values() if v is not None]
row["AVG"] = round(sum(vals) / len(vals), 2) if vals else None
out["tasks"] = row
# 对照 E98 best full 臂
e98 = json.load(open("exp/trace/results/e98_full_13tasks.json"))["tasks"]
out["delta_vs_full_L1"] = {k: (round(row[k] - e98[k], 2) if row.get(k) is not None and e98.get(k) is not None else None) for k in e98 if k != "AVG"}
if row.get("AVG") and e98.get("AVG"):
    out["delta_vs_full_L1"]["AVG"] = round(row["AVG"] - e98["AVG"], 2)
json.dump(out, open("exp/trace/results/e100_tail_full.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(out, ensure_ascii=False, indent=1))
print("saved e100_tail_full.json")
PYEOF
echo "E100_TAIL_FULL_DONE"
