#!/bin/bash
# RULER B7 打分完成后自动合并终表（三长度消融列）
while true; do
  if grep -q "RULER SCORE DONE" /tmp/tli_score_ruler_b7.log 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] B7 RULER scored, merging final table"
    python3 /tmp/merge_ruler_b7.py
    echo "MERGE DONE"
    break
  fi
  sleep 120
done
