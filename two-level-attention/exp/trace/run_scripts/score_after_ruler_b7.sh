#!/bin/bash
# B7 RULER 三长度完成后自动打分（与 C0 同口径对比表）
while true; do
  if grep -q "RULER B7 ALL DONE" /tmp/tli_ruler_b7.log 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] B7 RULER 完成，开始打分"
    cd /home/wangyuanshuo02/two-level-attention
    PYTHONPATH=/home/wangyuanshuo02/.local/pylibs:. python -u benchmark/RULER/score_ruler.py --pred-postfix _b7 --out exp/results_ruler/ruler_b7.json 2>&1 | tail -20
    echo "RULER SCORE DONE"
    break
  fi
  sleep 300
done
