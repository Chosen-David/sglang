#!/bin/bash
# B7s 严格口径 RULER 自动链：GPU1 RULER 全完成 → 打分 → 三长度合并终表
cd /home/wangyuanshuo02/two-level-attention
while true; do
  if grep -q "B7s GPU1 ALL DONE" /tmp/tli_b7s_gpu1.log 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] B7s RULER+tail done, scoring RULER"
    PYTHONPATH=/home/wangyuanshuo02/.local/pylibs:. python -u benchmark/RULER/score_ruler.py \
      --pred-postfix _b7 --out exp/results_ruler/ruler_b7s.json 2>&1 | tail -20
    echo "SCORE DONE"
    python3 /tmp/merge_ruler_b7s.py
    echo "MERGE DONE"
    break
  fi
  sleep 180
done
