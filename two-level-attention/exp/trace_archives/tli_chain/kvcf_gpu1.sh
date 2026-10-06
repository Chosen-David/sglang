#!/bin/bash
# E81 放量 GPU1：h2o 全 13 任务 + pyramidkv 轻载 6 任务
cd /home/wangyuanshuo02/two-level-attention
TASKS="hotpotqa 2wikimqa musique qasper multifieldqa_en passage_retrieval_en gov_report narrativeqa triviaqa lcc repobench-p qmsum multi_news"
for t in $TASKS; do
  CUDA_VISIBLE_DEVICES=1 python -u exp/trace/pred_kvcf.py --task $t --method h2o --max-capacity 1024 --n 200 >> /tmp/kvcf_gpu1.log 2>&1
done
for t in hotpotqa 2wikimqa musique qasper multifieldqa_en passage_retrieval_en; do
  CUDA_VISIBLE_DEVICES=1 python -u exp/trace/pred_kvcf.py --task $t --method pyramidkv --max-capacity 1024 --n 200 >> /tmp/kvcf_gpu1.log 2>&1
done
echo KVCF_GPU1_DONE >> /tmp/kvcf_gpu1.log
