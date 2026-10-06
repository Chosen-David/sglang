#!/bin/bash
# 双卡拆分守卫：①GPU0 循环进入 triviaqa 时杀掉（避免与 GPU1 尾部撞车）
# ②12 任务文件全齐后跑终版打分（替代旧 score_after_b7.sh 的单 DONE 判据）
EXPECT="2wikimqa:200 gov_report:200 hotpotqa:200 multifieldqa_en:150 multi_news:200 musique:200 narrativeqa:200 passage_retrieval_en:200 qasper:200 qmsum:200 lcc:500 repobench:500"
DIR=/home/wangyuanshuo02/two-level-attention/exp/results_longbench/Qwen3-8B/pred_b7
killed=0
while true; do
  # ① GPU0 循环进入 triviaqa（GPU1 尾部的任务）→ 杀原循环+该 python
  if [ $killed -eq 0 ] && grep -q "task=triviaqa" /tmp/tli_e71_b7.log 2>/dev/null; then
    LOOP_PID=$(ps aux | grep "bash /tmp/run_e71_b7_full.sh" | grep -v grep | awk '{print $2}')
    PY_PID=$(ps aux | grep "task triviaqa" | grep -v grep | awk '{print $2}')
    for p in $LOOP_PID $PY_PID; do kill $p 2>/dev/null && echo "killed $p"; done
    killed=1
    echo "[$(date +%H:%M:%S)] GPU0 loop killed at triviaqa (GPU1 owns tail)"
  fi
  # ② 文件齐备判定
  all_ok=1
  for spec in $EXPECT; do
    t=${spec%%:*}; n=${spec##*:}
    f=$(ls $DIR/${t}-tli_*.jsonl 2>/dev/null | head -1)
    if [ -z "$f" ] || [ "$(wc -l < $f)" -lt "$n" ]; then all_ok=0; break; fi
  done
  if [ $all_ok -eq 1 ]; then
    echo "[$(date +%H:%M:%S)] all 12 files complete, scoring"
    cd /home/wangyuanshuo02/two-level-attention
    PYTHONPATH=/home/wangyuanshuo02/.local/pylibs:. python -u exp/trace/run_e71_eval.py 2>&1 | tail -30
    echo "SCORE DONE"
    break
  fi
  sleep 120
done
