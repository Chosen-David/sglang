#!/bin/bash
# E81 收尾 + E85f 提前启动（GPU0 空闲，2026-10-01 00:20 决策）：
#   ① 补跑 snapkv/h2o 的 multifieldqa_en（各只有 150 行残缺文件，须先删）
#   ② E85f：per-layer 静态 pair e2e 判决 13 任务（E72 mavg 冠军配置 +
#      --tli_static_pair，musique/qasper 前置出判决信号）
# 原 GPU1 waiter 已杀；GPU1 继续跑 pyramidkv 剩余（musique 在跑）。
cd /home/wangyuanshuo02/two-level-attention
MODEL=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B
BASE=exp/results_longbench/Qwen3-8B/pred_kvcf
LOG=/tmp/e81_e85f_gpu0.log

# ---- ① E81 残缺补跑（删 150 行文件重跑）----
for m in snapkv h2o; do
  rm -f $BASE/$m/multifieldqa_en-$m-*.jsonl
  echo "=== [repair $m multifieldqa_en $(date +%H:%M:%S)] ===" >> $LOG
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python -u exp/trace/pred_kvcf.py --task multifieldqa_en --method $m \
    --max-capacity 1024 --n 200 >> $LOG 2>&1
done
echo "E81_GPU0_REPAIR_DONE" >> $LOG

# ---- ② E85f e2e 判决 13 任务 ----
TS=0930e85f
for T in musique qasper hotpotqa 2wikimqa passage_retrieval_en multifieldqa_en gov_report qmsum multi_news narrativeqa triviaqa lcc repobench-p; do
  echo "=== [E85f GPU0 $(date +%H:%M:%S)] task=$T ===" >> $LOG
  CUDA_VISIBLE_DEVICES=0 python -u -m benchmark.LongBench.pred \
    --model Qwen3-8B --model_path $MODEL \
    --task $T --method tli --e 0 --t $TS \
    --dataset-path /home/wangyuanshuo02/datasets/LongBench/data \
    --config-path benchmark/LongBench/config \
    --output-dir exp/results_longbench/Qwen3-8B \
    --tia_level1_topk 128 --tia_level2_topk 1024 --tia_level2_cmp_ratio 4 \
    --tli_enable_kmeans false --tli_enable_layer_skip false \
    --tli_alpha 0.125 --tli_beta 0.375 --tli_gamma 0.125 \
    --tli_subspace tail --tli_static_pair \
    --pred_postfix _e85f 2>&1 | tail -2 >> $LOG
done
echo "E85F_GPU0_DONE" >> $LOG
