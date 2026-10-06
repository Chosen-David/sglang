#!/bin/bash
# M6 sink-guard 全量：SnapKV/H2O/PyramidKV + sink_guard 128（PSI 口径对齐）
# GPU0 顺序跑（MoBA repobench 占 GPU1）；hotpotqa 已冒烟 20 样本，全量重跑覆盖
cd /home/wangyuanshuo02/two-level-attention/exp/trace
export CUDA_VISIBLE_DEVICES=0
TASKS="hotpotqa musique 2wikimqa passage_retrieval_en qasper multifieldqa_en gov_report qmsum narrativeqa triviaqa multi_news lcc repobench-p"
for m in snapkv h2o pyramidkv; do
  for t in $TASKS; do
    echo "==== $m $t ===="
    PYTHONPATH=~/.local/pylibs:/home/wangyuanshuo02/two-level-attention \
    python3 -u pred_kvcf.py --task $t --method $m --sink-guard 128 \
      --max-capacity 1024 --n 200 --gpu 0 --output-suffix _sinkguard \
      2>&1 | tail -1
  done
done
echo "M6_SINKGUARD_GPU0_DONE"
