#!/bin/bash
# ============================================================================
# E123 cavg 两配置 vs 海选冠军 GPU 小试（S-T018 / 任务 #202，用户授权起跑）
# ============================================================================
# 新口径（E121 R01 BOS-aware + B04 首 token EOS + B10 near 边界已全部合入主仓
# 384dabe41；E122 γ=off 自由竞争已实现并通过 6/6 python±-O）。
#
# 4 臂 × 5 任务（海选任务子集）：
#   mavg     海选冠军 (.25,.125,.625)——far=minmax/near=avg，新口径锚点
#   cavg_g   cavg(.25,.5,.125)——far_select=cluster + kmeans（E105 历史口径）
#   cavg_off cavg(.126,.126,γ off)——γ off = 取消 L2 near/far 配额分割，
#            mid 单池 topk 自由竞争（E122 新路径，量纲与 near avg 一致）
#   fullkv   FullKV 锚点（--method none）
#
# 注意（E121 B09 口径）：method_name 现含 α/β/γ（tli_64_128_1024_c4a0.25_b0.125_
# g0.625 风格），四臂文件名天然互异，无同名互覆风险；γ off 渲染 goff。
# ============================================================================
set -u

REPO_ROOT=/home/wangyuanshuo02/sglang/two-level-attention
MODEL=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B
DATA=/home/wangyuanshuo02/datasets/LongBench/data
OUTROOT=/tmp/e123_trial
T_STAMP=$(date +%m%d%H%M%S)
TASKS="qasper hotpotqa gov_report repobench-p musique"

min_rows () {
  case "$1" in
    repobench-p) echo 500 ;;
    *) echo 200 ;;
  esac
}

run_arm () {
  local GPU=$1; local ARM=$2
  cd "$REPO_ROOT"
  export PYTHONPATH="$REPO_ROOT"
  local OUTDIR="$OUTROOT/pred_${ARM}"
  mkdir -p "$OUTDIR"
  for t in $TASKS; do
    local MIN; MIN=$(min_rows "$t")
    local DONE=0
    for f in "$OUTDIR"/${t}-*.jsonl; do
      [ -e "$f" ] || continue
      if [ "$(cat "$f" 2>/dev/null | wc -l)" -ge "$MIN" ]; then DONE=1; fi
    done
    if [ "$DONE" = 1 ]; then echo "==== SKIP $ARM/$t (done ≥$MIN rows) ===="; continue; fi
    echo "==== GPU$GPU arm=$ARM task=$t $(date +%H:%M:%S) ===="
    local COMMON="--model Qwen3-8B --model_path $MODEL --task $t --e 0 \
      --dataset-path $DATA --config-path benchmark/LongBench/config \
      --output-dir $OUTROOT --pred_postfix _${ARM} \
      --tia_level1_topk 128 --tia_level2_topk 1024 --tia_level2_cmp_ratio 4 \
      --tli_enable_layer_skip false --t $T_STAMP"
    case "$ARM" in
      mavg)
        CUDA_VISIBLE_DEVICES=$GPU python -u -m benchmark.LongBench.pred $COMMON \
          --method tli --tli_far_method minmax --tli_near_method avg \
          --tli_alpha 0.25 --tli_beta 0.125 --tli_gamma 0.625 \
          --tli_enable_kmeans false \
          2>&1 | tee -a "$OUTROOT/e123_${ARM}_gpu${GPU}.log" | tail -2 ;;
      cavg_g)
        CUDA_VISIBLE_DEVICES=$GPU python -u -m benchmark.LongBench.pred $COMMON \
          --method tli --tli_far_method minmax --tli_near_method avg \
          --tli_far_select cluster --tli_enable_kmeans true \
          --tli_alpha 0.25 --tli_beta 0.5 --tli_gamma 0.125 \
          2>&1 | tee -a "$OUTROOT/e123_${ARM}_gpu${GPU}.log" | tail -2 ;;
      cavg_off)
        CUDA_VISIBLE_DEVICES=$GPU python -u -m benchmark.LongBench.pred $COMMON \
          --method tli --tli_far_method minmax --tli_near_method avg \
          --tli_far_select cluster --tli_enable_kmeans true \
          --tli_alpha 0.126 --tli_beta 0.126 --tli_gamma off \
          2>&1 | tee -a "$OUTROOT/e123_${ARM}_gpu${GPU}.log" | tail -2 ;;
      fullkv)
        CUDA_VISIBLE_DEVICES=$GPU python -u -m benchmark.LongBench.pred $COMMON \
          --method none \
          2>&1 | tee -a "$OUTROOT/e123_${ARM}_gpu${GPU}.log" | tail -2 ;;
      *) echo "unknown arm $ARM"; exit 1 ;;
    esac
  done
}

mkdir -p "$OUTROOT"
# 本地 2 卡分工：GPU0 = mavg → cavg_off（判决最相关两臂）；
#               GPU1 = cavg_g → fullkv。
run_arm 0 mavg && run_arm 0 cavg_off &
P0=$!
run_arm 1 cavg_g && run_arm 1 fullkv &
P1=$!
wait $P0 $P1
echo "==== E123 四臂 5 任务全部结束 $(date) ===="
