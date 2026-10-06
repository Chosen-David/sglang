#!/bin/bash
# E81 GPU1 接力：等 1313328（h2o passage_retrieval_en）退出后，跑 h2o 剩余
# 11 任务 + pyramidkv 6 轻载任务。h2o 已做分块修复（CHUNK=4096），narrativeqa
# 30K 可跑。expandable_segments 防碎片化 OOM。
cd /home/wangyuanshuo02/two-level-attention
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 等 1313328 退出（最多等 2h）
while kill -0 1313328 2>/dev/null; do sleep 60; done
sleep 30

TASKS="hotpotqa musique qasper multifieldqa_en gov_report narrativeqa triviaqa lcc repobench-p qmsum multi_news"
for t in $TASKS; do
  CUDA_VISIBLE_DEVICES=1 python -u exp/trace/pred_kvcf.py --task $t --method h2o --max-capacity 1024 --n 200 >> /tmp/kvcf_gpu1.log 2>&1
done
for t in hotpotqa 2wikimqa musique qasper multifieldqa_en passage_retrieval_en; do
  CUDA_VISIBLE_DEVICES=1 python -u exp/trace/pred_kvcf.py --task $t --method pyramidkv --max-capacity 1024 --n 200 >> /tmp/kvcf_gpu1.log 2>&1
done
echo KVCF_GPU1_DONE >> /tmp/kvcf_gpu1.log
