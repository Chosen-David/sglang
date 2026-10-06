#!/bin/bash
# B7 严格口径全量重跑·GPU1 臂：RULER 三长度 → LongBench 尾部三任务
cd /home/wangyuanshuo02/two-level-attention
MODEL=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B
DATA=/home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/RULER
TS=09291030
TASKS_R="niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multiquery niah_multivalue cwe fwe vt"
for L in 4096 8192 16384; do
  for T in $TASKS_R; do
    echo "=== [B7s GPU1 $(date +%H:%M:%S)] L=$L task=$T ==="
    CUDA_VISIBLE_DEVICES=1 python -u -m benchmark.RULER.pred_ruler \
      --model Qwen3-8B --model_path $MODEL \
      --task $T --context_length $L --method tli --t $TS \
      --data-root $DATA \
      --output-dir exp/results_ruler/Qwen3-8B \
      --tia_level1_topk 128 --tia_level2_topk 1024 --tia_level2_cmp_ratio 4 \
      --tli_enable_kmeans false --tli_enable_layer_skip false \
      --tli_alpha 0.125 --tli_beta 0.25 --tli_gamma 0.125 \
      --pred_postfix _b7 --max-num 100 2>&1 | tail -2
  done
done
echo "B7s GPU1 RULER DONE"
for T in triviaqa lcc repobench-p; do
  echo "=== [B7s GPU1 $(date +%H:%M:%S)] task=$T ==="
  CUDA_VISIBLE_DEVICES=1 python -u -m benchmark.LongBench.pred \
    --model Qwen3-8B --model_path $MODEL \
    --task $T --method tli --e 0 --t $TS \
    --dataset-path /home/wangyuanshuo02/datasets/LongBench/data \
    --config-path benchmark/LongBench/config \
    --output-dir exp/results_longbench/Qwen3-8B \
    --tia_level1_topk 128 --tia_level2_topk 1024 --tia_level2_cmp_ratio 4 \
    --tli_enable_kmeans false --tli_enable_layer_skip false \
    --tli_alpha 0.125 --tli_beta 0.25 --tli_gamma 0.125 \
    --pred_postfix _b7 2>&1 | tail -2
done
echo "B7s GPU1 ALL DONE"
