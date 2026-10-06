#!/bin/bash
# RULER 32K 档数据生成驱动脚本（纯 CPU，CUDA_VISIBLE_DEVICES="")
# 11 任务 × max_seq_length=32768 × num_samples=100
# 语料：PaulGrahamEssays.json（KVCache-Factory 49 篇散文拼接，官方格式）
#       english_words.json（wonderwords 池占位，32K 档 cwe 不用 randle_words）
set -u
export PYTHONPATH=/home/wangyuanshuo02/.local/pylibs
export CUDA_VISIBLE_DEVICES=""
cd /tmp/ruler_official/scripts/data

MODEL=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B
RAW=/tmp/ruler_32k_raw
TASKS="niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multiquery niah_multivalue vt cwe fwe"

for T in $TASKS; do
  echo "[$(date '+%F %T')] === START $T ==="
  python3 prepare.py \
    --save_dir $RAW \
    --benchmark synthetic \
    --task $T \
    --tokenizer_path $MODEL \
    --tokenizer_type hf \
    --max_seq_length 32768 \
    --model_template_type base \
    --num_samples 100 \
    --random_seed 42 2>&1 | tail -3
  echo "[$(date '+%F %T')] === DONE $T ==="
done
echo "[$(date '+%F %T')] ALL TASKS DONE"
