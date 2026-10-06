#!/bin/bash
# E65b 链式启动：等 E65 主实验落盘后自动跑自训练降维矩阵对比
cd /home/wangyuanshuo02/two-level-attention/exp/trace
while [ ! -f results/e65_dim_reduction.json ]; do
  sleep 120
  # E65 主进程死且无产物 → 直接启动 E65b（防死锁）
  if ! pgrep -f analyze_e65_dim_reduce.py > /dev/null; then
    sleep 60
    [ -f results/e65_dim_reduction.json ] || break
  fi
done
python3 -u analyze_e65b_sup_proj.py >> /tmp/e65b_sup.log 2>&1
