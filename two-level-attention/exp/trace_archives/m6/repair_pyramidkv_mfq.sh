#!/bin/bash
# 补跑 pyramidkv_sinkguard multifieldqa_en（150/200 截断）；等 GPU0 h2o 完成后用 GPU0
cd /home/wangyuanshuo02/two-level-attention/exp/trace
export PYTHONPATH=/home/wangyuanshuo02/two-level-attention:/home/wangyuanshuo02/.local/pylibs
while ! grep -q "M6_SINKGUARD_GPU0_DONE" /tmp/m6_sinkguard_full.log 2>/dev/null; do
  sleep 120
done
rm -f exp/results_longbench/Qwen3-8B/pred_kvcf/pyramidkv_sinkguard/multifieldqa_en-pyramidkv_sinkguard-080346.jsonl
python3 -u pred_kvcf.py --task multifieldqa_en --method pyramidkv --sink-guard 128 \
  --max-capacity 1024 --n 200 --gpu 0 --output-suffix _sinkguard 2>&1 | tail -2
echo "PYRAMIDKV_MFQ_REPAIR_DONE"
