#!/bin/bash
# E72 mavg 冠军臂（α.125 β.375 γ.125，far=minmax/near=avg 默认）RULER 33 任务补全
# ——论文主表 TLI_E72 RULER 空格（Limitations 补全项）。双卡拆分：GPU0 短长度 / GPU1 L16384。
# 用法：bash /tmp/tli_chain/run_ruler_e72m.sh <gpu_id> <lengths>
cd /home/wangyuanshuo02/two-level-attention
MODEL=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B
DATA=/home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/RULER
GPU=${1:-0}
LENS=${2:-"4096 8192"}
N=100
TS=0930e72m
TASKS="niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multiquery niah_multivalue cwe fwe vt"
for L in $LENS; do
  for T in $TASKS; do
    echo "=== [$(date +%H:%M:%S)] L=$L task=$T ==="
    CUDA_VISIBLE_DEVICES=$GPU python -u -m benchmark.RULER.pred_ruler \
      --model Qwen3-8B --model_path $MODEL \
      --task $T --context_length $L --method tli --t $TS \
      --data-root $DATA \
      --output-dir exp/results_ruler/Qwen3-8B \
      --tia_level1_topk 128 --tia_level2_topk 1024 --tia_level2_cmp_ratio 4 \
      --tli_enable_kmeans false --tli_enable_layer_skip false \
      --tli_alpha 0.125 --tli_beta 0.375 --tli_gamma 0.125 \
      --pred_postfix _e72mavg --max-num $N 2>&1 | tail -2
  done
done
echo "RULER E72MAVG GPU$GPU DONE"
