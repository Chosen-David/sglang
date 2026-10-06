#!/bin/bash
# E71 B7 全量放量（修复版分区 α.125/β.25/γ.125）：11 任务（hotpotqa 已有）
cd /home/wangyuanshuo02/two-level-attention
MODEL=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B
TS=09290152  # 与 B7 冒烟同时间戳
for T in 2wikimqa musique passage_retrieval_en qasper multifieldqa_en gov_report qmsum multi_news narrativeqa triviaqa lcc repobench-p; do
  echo "=== [B7 $(date +%H:%M:%S)] task=$T ==="
  CUDA_VISIBLE_DEVICES=$1 python -u -m benchmark.LongBench.pred \
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
echo "B7 FULL DONE"
