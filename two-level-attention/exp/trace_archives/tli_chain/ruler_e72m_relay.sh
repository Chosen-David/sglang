#!/bin/bash
# E72 mavg RULER 打分 relay：双卡 DONE 后自动打分出 ruler_e72mavg.json
LOG0=/tmp/ruler_e72m_gpu0.log
LOG1=/tmp/ruler_e72m_gpu1.log
OUT=/home/wangyuanshuo02/two-level-attention/exp/results_ruler/ruler_e72mavg.json
[ -f /tmp/tli_chain/RULER_E72M_SCORED ] && exit 0
while true; do
  D0=$(grep -c "RULER E72MAVG GPU0 DONE" $LOG0 2>/dev/null || echo 0)
  D1=$(grep -c "RULER E72MAVG GPU1 DONE" $LOG1 2>/dev/null || echo 0)
  if [ "$D0" -ge 1 ] && [ "$D1" -ge 1 ]; then
    echo "[$(date +%H:%M:%S)] 双卡完成，开始打分"
    cd /home/wangyuanshuo02/two-level-attention
    PYTHONPATH=/home/wangyuanshuo02/.local/pylibs:. python -u benchmark/RULER/score_ruler.py \
      --pred-postfix _e72mavg --out $OUT 2>&1 | tail -25
    touch /tmp/tli_chain/RULER_E72M_SCORED
    echo "RULER E72MAVG SCORED"
    break
  fi
  sleep 300
done
