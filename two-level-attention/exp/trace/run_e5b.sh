#!/bin/bash
# E5b: TLI(A+B'+D') 在 LongBench 13 英文子集上的真实精度评估
# 与 baseline 同口径：K2=1024, cmp_ratio=4, pred_1024 目录
cd /home/wangyuanshuo02/two-level-attention
MODEL=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B
TS=$(date +%m%d%H%M)
TASKS="hotpotqa 2wikimqa musique passage_retrieval_en qasper multifieldqa_en gov_report qmsum multi_news narrativeqa triviaqa lcc repobench"
for T in $TASKS; do
  echo "=== [$(date +%H:%M:%S)] task=$T ==="
  python -u -m benchmark.LongBench.pred \
    --model Qwen3-8B --model_path $MODEL \
    --task $T --method tli --e 0 --t $TS \
    --dataset-path /home/wangyuanshuo02/datasets/LongBench/data \
    --config-path benchmark/LongBench/config \
    --output-dir exp/results_longbench/Qwen3-8B \
    --tia_level2_topk 1024 --tia_level2_cmp_ratio 4 \
    --pred_postfix _1024 2>&1 | tail -3
done
echo "ALL DONE"
