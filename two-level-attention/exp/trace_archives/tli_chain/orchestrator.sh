#!/bin/bash
# TLI 任务链编排器（2026-09-29 用户指令：耗时短优先 + 幂等状态机，可反复调用）
# 链：E64j 提最佳配置(CPU) → GPU0 空闲后 e2e 五组合快筛(hotpotqa+musique) → best 臂 12 任务全量 → 终表
CD=/tmp/tli_chain
LOG=$CD/orchestrator.log
echo "=== [orchestrator $(date '+%m-%d %H:%M:%S')] ===" >> $LOG
cd $CD

# ---- 阶段 1：E64j 完成 → 提取每组合最佳 αβ ----
if [ ! -f $CD/E64J_DONE ]; then
  if python3 -c "import json;d=json.load(open('/home/wangyuanshuo02/two-level-attention/exp/trace/results/e64j_combo_best.json'));exit(0 if 'AVG' in d else 1)" 2>/dev/null; then
    PYTHONPATH=/home/wangyuanshuo02/.local/pylibs:. python3 e64j_best.py >> $LOG 2>&1 && touch $CD/E64J_DONE && echo "[T1] E64j best 提取完成" >> $LOG
  else
    n=$(python3 -c "import json;d=json.load(open('/home/wangyuanshuo02/two-level-attention/exp/trace/results/e64j_combo_best.json'));print(len([k for k in d if k!='AVG']))" 2>/dev/null)
    echo "[T1] E64j 进行中 $n/16" >> $LOG
  fi
fi

# ---- 阶段 2：GPU0 空闲 + E64j 完成 → 生成并启动快筛 ----
if [ -f $CD/E64J_DONE ] && [ ! -f $CD/SCREEN_LAUNCHED ]; then
  if grep -q "B7s GPU0 DONE" /tmp/tli_b7s_gpu0.log 2>/dev/null; then
    PYTHONPATH=/home/wangyuanshuo02/.local/pylibs:. python3 gen_screen.py >> $LOG 2>&1
    if [ -f $CD/run_screen.sh ]; then
      nohup bash $CD/run_screen.sh > $CD/screen.log 2>&1 &
      touch $CD/SCREEN_LAUNCHED
      echo "[T2] 快筛已启动 (pid $!)" >> $LOG
    else
      echo "[T2] gen_screen 失败" >> $LOG
    fi
  else
    echo "[T2] 等 GPU0（B7s LongBench 主段未完成）" >> $LOG
  fi
fi

# ---- 阶段 3：快筛完成 → 打分排名 ----
if [ -f $CD/SCREEN_LAUNCHED ] && [ ! -f $CD/SCREEN_SCORED ]; then
  if grep -q "SCREEN ALL DONE" $CD/screen.log 2>/dev/null; then
    PYTHONPATH=/home/wangyuanshuo02/.local/pylibs:. python3 score_screen.py >> $LOG 2>&1 && touch $CD/SCREEN_SCORED && echo "[T3] 快筛打分完成" >> $LOG
  else
    echo "[T3] 快筛进行中：$(grep -c '=== \[screen' $CD/screen.log 2>/dev/null)/4 臂完成，落盘 $(find /home/wangyuanshuo02/two-level-attention/exp/results_longbench/Qwen3-8B/ -maxdepth 1 -name 'pred_e72_*' -exec sh -c 'ls $1 | wc -l' _ {} \; | paste -sd+ | bc 2>/dev/null || echo 0)/8 任务" >> $LOG
  fi
fi

# ---- 阶段 4：best 臂 13 任务全量（双卡拆分，GPU0+GPU1 并行）----
if [ -f $CD/SCREEN_SCORED ] && [ ! -f $CD/FULL_LAUNCHED ]; then
  PYTHONPATH=/home/wangyuanshuo02/.local/pylibs:. python3 gen_full.py >> $LOG 2>&1
  if [ -f $CD/run_full_gpu0.sh ]; then
    nohup bash $CD/run_full_gpu0.sh > $CD/full0.log 2>&1 &
    nohup bash $CD/run_full_gpu1.sh > $CD/full1.log 2>&1 &
    touch $CD/FULL_LAUNCHED
    echo "[T4] best 臂双卡全量已启动 (pid $!)" >> $LOG
  fi
fi
if [ -f $CD/FULL_LAUNCHED ] && [ ! -f $CD/FULL_DONE ]; then
  if grep -q "FULL GPU0 DONE" $CD/full0.log 2>/dev/null && grep -q "FULL GPU1 DONE" $CD/full1.log 2>/dev/null; then
    touch $CD/FULL_DONE && echo "[T4] best 臂双卡全量完成" >> $LOG
  else
    echo "[T4] 全量进行中：GPU0 $(grep -c '=== \[full0' $CD/full0.log 2>/dev/null)/6 段，GPU1 $(grep -c '=== \[full1' $CD/full1.log 2>/dev/null)/7 段" >> $LOG
  fi
fi

# ---- 阶段 5：全量打分 + 终表 ----
if [ -f $CD/FULL_DONE ] && [ ! -f $CD/FULL_SCORED ]; then
  cd /home/wangyuanshuo02/two-level-attention && PYTHONPATH=/home/wangyuanshuo02/.local/pylibs:. python3 /tmp/tli_chain/score_full.py >> $LOG 2>&1 && touch $CD/FULL_SCORED && echo "[T5] best 臂全量打分完成" >> $LOG
fi
if [ -f $CD/FULL_SCORED ] && [ ! -f $CD/CHAIN_DONE ]; then
  cd /home/wangyuanshuo02/two-level-attention && PYTHONPATH=/home/wangyuanshuo02/.local/pylibs:. python3 /tmp/tli_chain/make_final_table.py >> $LOG 2>&1 && touch $CD/CHAIN_DONE && echo "[T5] 任务链全部完成" >> $LOG
fi

cat $CD/state 2>/dev/null; for m in E64J_DONE SCREEN_LAUNCHED SCREEN_SCORED FULL_LAUNCHED FULL_DONE CHAIN_DONE; do
  [ -f $CD/$m ] && echo "$m"
done >> $LOG
