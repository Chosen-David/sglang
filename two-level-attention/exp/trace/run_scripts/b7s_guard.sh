#!/bin/bash
# B7 严格口径守卫：12 LongBench 文件齐 → 终版打分；RULER 齐后提示（打分由 relay 接）
EXPECT="2wikimqa:200 gov_report:200 hotpotqa:200 multifieldqa_en:150 multi_news:200 musique:200 narrativeqa:200 passage_retrieval_en:200 qasper:200 qmsum:200 lcc:500 repobench:500"
DIR=/home/wangyuanshuo02/two-level-attention/exp/results_longbench/Qwen3-8B/pred_b7
while true; do
  all_ok=1
  for spec in $EXPECT; do
    t=${spec%%:*}; n=${spec##*:}
    f=$(ls $DIR/${t}-tli_*.jsonl 2>/dev/null | head -1)
    if [ -z "$f" ] || [ "$(wc -l < $f)" -lt "$n" ]; then all_ok=0; break; fi
  done
  if [ $all_ok -eq 1 ]; then
    echo "[$(date +%H:%M:%S)] all 12 LongBench files complete, scoring"
    cd /home/wangyuanshuo02/two-level-attention
    PYTHONPATH=/home/wangyuanshuo02/.local/pylibs:. python -u exp/trace/run_e71_eval.py 2>&1 | tail -30
    echo "B7s LONGBENCH SCORE DONE"
    break
  fi
  sleep 120
done
