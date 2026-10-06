#!/bin/bash
# B7 严格口径全量重跑·GPU0 臂（2026-09-29 用户指示：直接挂掉全部重新跑）
# 严格口径 = commit 7ea756c：sink/swa 完全保送不进双池，
# K2_mid = K2 − sink_tok − swa_tok 全给 mid 创新管线
cd /home/wangyuanshuo02/two-level-attention
MODEL=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B
TS=09291030
for T in hotpotqa 2wikimqa musique passage_retrieval_en qasper multifieldqa_en gov_report qmsum multi_news narrativeqa; do
  echo "=== [B7s GPU0 $(date +%H:%M:%S)] task=$T ==="
  CUDA_VISIBLE_DEVICES=0 python -u -m benchmark.LongBench.pred \
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
echo "B7s GPU0 DONE"
