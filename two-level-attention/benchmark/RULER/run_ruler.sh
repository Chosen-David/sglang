#!/bin/bash
# #66 RULER 批量调度：11 任务 × 3 长度 × 方法
# 用法：bash benchmark/RULER/run_ruler.sh <method> <gpu_id> [max_num]
# 例：bash benchmark/RULER/run_ruler.sh tli 0 100
# B06 修复（GPT 审查 2026-10-08）：原版 `python ... | tail -2` 无 pipefail，
# 退出码来自 tail 而非 python；cd 失败也继续跑；末尾无条件 ALL DONE——
# 33 次 python 失败仍退出 0。修复：逐任务 PIPESTATUS 检查 + FAILED 计数 +
# 失败时 exit 1 且 marker 带 FAILED 标记（下游 grep "ALL DONE" 仍会匹配旧
# marker 字符串，B07 的结构化 run-ID 绑定待 RULER 阶段重构）。
cd /home/wangyuanshuo02/two-level-attention || exit 1
MODEL=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B
DATA=/home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/RULER
METHOD=$1
GPU=$2
N=${3:-100}
TS=$(date +%m%d%H%M)
TASKS="niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multiquery niah_multivalue cwe fwe vt"
FAILED=0
for L in 4096 8192 16384; do
  for T in $TASKS; do
    echo "=== [$(date +%H:%M:%S)] L=$L task=$T ==="
    CUDA_VISIBLE_DEVICES=$GPU python -u -m benchmark.RULER.pred_ruler \
      --model Qwen3-8B --model_path $MODEL \
      --task $T --context_length $L --method $METHOD --t $TS \
      --data-root $DATA \
      --output-dir exp/results_ruler/Qwen3-8B \
      --tia_level2_topk 1024 --tia_level2_cmp_ratio 4 \
      --pred_postfix _1024 --max-num $N 2>&1 | tail -2
    rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
      echo "==== FAILED L=$L task=$T method=$METHOD rc=$rc ===="
      FAILED=$((FAILED+1))
    fi
  done
done
if [ "$FAILED" -ne 0 ]; then
  echo "ALL DONE $METHOD (FAILED=$FAILED/$((3*11)))"
  exit 1
fi
echo "ALL DONE $METHOD"
