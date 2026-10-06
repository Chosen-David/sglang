#!/bin/bash
# M6 sink-guard 全量·GPU1 并行臂：pyramidkv（GPU0 队列最后一个方法，预跑免重复）
# pred_kvcf.py 已加「完整输出跳过」逻辑：GPU0 之后跑到 pyramidkv 时会自动跳过已完成任务
cd /home/wangyuanshuo02/two-level-attention/exp/trace
export CUDA_VISIBLE_DEVICES=1
TASKS="hotpotqa musique 2wikimqa passage_retrieval_en qasper multifieldqa_en gov_report qmsum narrativeqa triviaqa multi_news lcc repobench-p"
for t in $TASKS; do
  echo "==== pyramidkv $t ===="
  PYTHONPATH=~/.local/pylibs:/home/wangyuanshuo02/two-level-attention \
  python3 -u pred_kvcf.py --task $t --method pyramidkv --sink-guard 128 \
    --max-capacity 1024 --n 200 --gpu 1 --output-suffix _sinkguard \
    2>&1 | tail -2
done
echo "M6_SINKGUARD_GPU1_PYRAMIDKV_DONE"
