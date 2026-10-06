#!/bin/bash
# #66 RULER 批量调度：11 任务 × 3 长度 × 方法
# 用法：bash benchmark/RULER/run_ruler.sh <method> <gpu_id> [max_num]
# 例：bash benchmark/RULER/run_ruler.sh tli 0 100
cd /home/wangyuanshuo02/two-level-attention
MODEL=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B
DATA=/home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/RULER
METHOD=$1
GPU=$2
N=${3:-100}
TS=$(date +%m%d%H%M)
TASKS="niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multiquery niah_multivalue cwe fwe vt"
for L in 4096 8192 16384; do
  for T in $TASKS; do
    echo "=== [$(date +%H:%M:%S)] L=$L task=$T ==="
    CUDA_VISIBLE_DEVICES=$GPU python -u -m benchmark.RULER.pred_ruler \
      --model Qwen3-8B --model_path $MODEL \
      --task $T --context_length $L --method $METHOD --t $TS \
      --data-root $DATA \
      --output-dir exp/results_ruler/Qwen3-8B \
      --tia_level2_topk 1024 --tia_level2_cmp_ratio 4 \
      --pred_postfix _1024 --max-num $N 2>&1 | tail -2
  done
done
echo "ALL DONE $METHOD"
