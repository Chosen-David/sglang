#!/bin/bash
# E103 kv-head 共享消融 —— GPU1：musique 臂 B → 臂 C
# 共享臂 baseline 不重跑：e98_full_13tasks.json mu=34.71
cd /home/wangyuanshuo02/two-level-attention
export PYTHONPATH=/home/wangyuanshuo02/two-level-attention:/home/wangyuanshuo02/.local/pylibs
export CUDA_VISIBLE_DEVICES=1

for CFG in "1024 _B" "256 _C"; do
  set -- $CFG
  K2=$1; POST=$2
  python -u -m benchmark.LongBench.pred \
    --model Qwen3-8B \
    --model_path /mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B \
    --task musique --method tli --e 0 \
    --dataset-path /home/wangyuanshuo02/datasets/LongBench/data \
    --config-path benchmark/LongBench/config \
    --output-dir /tmp/e103_perqhead --pred_postfix $POST \
    --tia_level1_topk 128 --tia_level2_topk $K2 --tia_level2_cmp_ratio 4 \
    --tli_enable_kmeans false --tli_enable_layer_skip false \
    --tli_alpha 0.125 --tli_beta 0.375 --tli_gamma 0.625 \
    --tli_far_method minmax --tli_near_method avg \
    --tli_per_q_head true >> /tmp/e103_mu.log 2>&1
  echo "MU_${POST}_DONE K2=$K2" >> /tmp/e103_mu.log
done
echo "E103_GPU1_MU_ALL_DONE" >> /tmp/e103_mu.log
