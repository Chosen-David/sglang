#!/bin/bash
# M6 sink-guard 冒烟：SnapKV+sink_guard 128（PSI sink 口径对齐）hotpotqa n=20
cd /home/wangyuanshuo02/two-level-attention/exp/trace
export CUDA_VISIBLE_DEVICES=0
PYTHONPATH=~/.local/pylibs:/home/wangyuanshuo02/two-level-attention \
python3 -u pred_kvcf.py --task hotpotqa --method snapkv --sink-guard 128 \
  --max-capacity 1024 --n 20 --gpu 0 \
  --output-suffix _sinkguard 2>&1 | tail -15
echo "M6_SMOKE_DONE"
