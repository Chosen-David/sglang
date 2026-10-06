#!/bin/bash
# E81 GPU0 补漏：等主脚本 KVCF_GPU0_DONE 后，补跑 snapkv 崩掉的两个任务
# （multifieldqa_en 曾留 150 行缓冲残留、passage_retrieval_en 曾 OOM 空文件，
#  已删，重跑）。
cd /home/wangyuanshuo02/two-level-attention
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

while ! grep -q KVCF_GPU0_DONE /tmp/kvcf_gpu0.log; do sleep 120; done
sleep 30

for t in multifieldqa_en passage_retrieval_en; do
  CUDA_VISIBLE_DEVICES=0 python -u exp/trace/pred_kvcf.py --task $t --method snapkv --max-capacity 1024 --n 200 >> /tmp/kvcf_gpu0.log 2>&1
done
echo KVCF_GPU0_REPAIR_DONE >> /tmp/kvcf_gpu0.log
