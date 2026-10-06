#!/bin/bash
# E81 放量 GPU0：snapkv 全 13 任务 + pyramidkv 重载 7 任务
cd /home/wangyuanshuo02/two-level-attention
TASKS="hotpotqa 2wikimqa musique qasper multifieldqa_en passage_retrieval_en gov_report narrativeqa triviaqa lcc repobench-p qmsum multi_news"
for t in $TASKS; do
  CUDA_VISIBLE_DEVICES=0 python -u exp/trace/pred_kvcf.py --task $t --method snapkv --max-capacity 1024 --n 200 >> /tmp/kvcf_gpu0.log 2>&1
done
for t in narrativeqa multi_news lcc repobench-p gov_report qmsum triviaqa; do
  CUDA_VISIBLE_DEVICES=0 python -u exp/trace/pred_kvcf.py --task $t --method pyramidkv --max-capacity 1024 --n 200 >> /tmp/kvcf_gpu0.log 2>&1
done
echo KVCF_GPU0_DONE >> /tmp/kvcf_gpu0.log
