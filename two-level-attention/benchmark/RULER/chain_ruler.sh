#!/bin/bash
# #66 RULER 接力调度器：GPU1 FullKV 完成 → Quest → TIA 依次自动启动
# B06/B07 修复（GPT 审查 2026-10-08）：①逐任务 PIPESTATUS 检查（原版
# `python | tail` 退出码来自 tail，python 失败被吞，末尾仍打 ALL DONE）
# ② cd 失败即退出 ③ marker 带 FAILED 计数。
# 已知残留（B07 本体）：等待条件 grep "ALL DONE none" /tmp/ruler_fullkv.log
# 是字符串匹配，不绑定本轮 run ID/commit——旧日志可放行。结构化 run-ID
# 绑定待 RULER 正式阶段重构，此处先修复失败传播。
cd /home/wangyuanshuo02/two-level-attention || exit 1
MODEL=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B
DATA=/home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/RULER
TASKS="niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multiquery niah_multivalue cwe fwe vt"
FAILED=0

# 等 FullKV 完成
while ! grep -q "ALL DONE none" /tmp/ruler_fullkv.log 2>/dev/null; do
  sleep 120
done
echo "[$(date +%H:%M:%S)] FullKV done, launching Quest"

TS=$(date +%m%d%H%M)
for L in 4096 8192 16384; do
  for T in $TASKS; do
    echo "=== [$(date +%H:%M:%S)] Quest L=$L task=$T ==="
    CUDA_VISIBLE_DEVICES=1 python -u -m benchmark.RULER.pred_ruler \
      --model Qwen3-8B --model_path $MODEL \
      --task $T --context_length $L --method quest --t $TS \
      --data-root $DATA \
      --output-dir exp/results_ruler/Qwen3-8B \
      --quest_block_size 64 --quest_topk 16 \
      --pred_postfix _1024 --max-num 100 2>&1 | tail -n 2
    rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
      echo "==== FAILED Quest L=$L task=$T rc=$rc ===="
      FAILED=$((FAILED+1))
    fi
  done
done
echo "ALL DONE quest (FAILED=$FAILED)"

TS=$(date +%m%d%H%M)
for L in 4096 8192 16384; do
  for T in $TASKS; do
    echo "=== [$(date +%H:%M:%S)] TIA L=$L task=$T ==="
    CUDA_VISIBLE_DEVICES=1 python -u -m benchmark.RULER.pred_ruler \
      --model Qwen3-8B --model_path $MODEL \
      --task $T --context_length $L --method tia --t $TS \
      --data-root $DATA \
      --output-dir exp/results_ruler/Qwen3-8B \
      --tia_level2_topk 1024 --tia_level2_cmp_ratio 4 \
      --pred_postfix _1024 --max-num 100 2>&1 | tail -n 2
    rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
      echo "==== FAILED TIA L=$L task=$T rc=$rc ===="
      FAILED=$((FAILED+1))
    fi
  done
done
if [ "$FAILED" -ne 0 ]; then
  echo "ALL DONE tia (FAILED=$FAILED/$((2*3*11)))"
  exit 1
fi
echo "ALL DONE tia"
