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
#
# 2026-10-11 审计修复（GPT 4303fb9 audit，两项 P2）：
#   TL-E123-RESUME-PREFIX-086：SKIP 判断改用归一化前缀 ${t%%-*}——producer
#     benchmark/LongBench/pred.py 文件名取 task.split("-")[0]（repobench-p 落盘
#     repobench-*），原 repobench-p-* glob 永不命中 → SKIP 失效；且只在
#     「唯一且完整」候选上 SKIP：同前缀多于一个 jsonl = 身份歧义（081 合入后
#     文件名带 _h<hash10> 尾段属正常口径，同臂同任务多文件即歧义），显式警告
#     后重跑暴露，绝不静默任选。行数门不变（repobench-p=500 其余=200）。
#   TL-E2E-FAILMASK-008（本脚本新增可达入口）：失败必须传播到总退出码——
#     ① set -o pipefail + 每条预测管道显式保存退出状态（原 python|tee|tail
#     的状态来自 tail，恒 0）；② run_arm 失败任务列表非空即返回非零；
#     ③ 双 GPU 链 wait 后逐 PID 检查子 shell 退出状态，任一非零 → echo 失败
#     摘要 + 总脚本非零退出；④ 全部成功才打印「全部结束」。原
#     `run_arm A && run_arm B &` 的 && 语义（A 失败会静默短路掉 B）已废除，
#     改为 run_chain 顺序执行后统一判定。
# ============================================================================
set -u
set -o pipefail   # 008：python|tee|tail 管道中 python 的失败不再被 tail 的 0 掩盖

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
  local FAILED=""   # 008：失败任务显式收集，函数末统一判非零
  for t in $TASKS; do
    local MIN; MIN=$(min_rows "$t")
    # 086：归一化前缀（producer pred.py 用 task.split("-")[0] 生成文件名，
    # repobench-p 的产物是 repobench-*）——与 E116b lib 修复同口径
    local P="${t%%-*}"
    local DONE=0 CANDS=0
    for f in "$OUTDIR"/${P}-*.jsonl; do
      [ -e "$f" ] || continue
      CANDS=$((CANDS+1))
      if [ "$(wc -l < "$f" 2>/dev/null || echo 0)" -ge "$MIN" ]; then DONE=1; fi
    done
    if [ "$CANDS" -gt 1 ]; then
      # 086：同前缀多候选 = 身份歧义，不 SKIP，重跑让歧义暴露（事后由
      # analyzer find_pred 的唯一性门禁拒收，绝不静默任选其一）
      echo "==== AMBIGUOUS $ARM/$t: 前缀 $P 命中 $CANDS 个 jsonl 候选（身份歧义），不 SKIP，重跑暴露 ===="
    elif [ "$DONE" = 1 ]; then
      echo "==== SKIP $ARM/$t (唯一候选 ≥$MIN rows) ===="
      continue
    fi
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
    # 008：显式保存预测管道退出状态（pipefail 已开，python 失败传导到整条
    # 管道；不能依赖管道隐式状态）
    local RC=$?
    if [ "$RC" -ne 0 ]; then
      FAILED="$FAILED $t(rc=$RC)"
      echo "==== PRED-FAIL $ARM/$t rc=$RC ===="
    fi
  done
  # 008：run_arm 失败标记——失败任务列表非空即返回非零
  if [ -n "$FAILED" ]; then
    echo "==== ARM-FAIL $ARM 失败任务:$FAILED ===="
    return 1
  fi
  return 0
}

# 008：GPU 链 runner——顺序执行所有臂，任一臂失败即链失败。
# 不用 `run_arm A && run_arm B`：&& 会让 A 失败时静默短路掉 B，
# 后者的问题就永远不会暴露。
run_chain () {
  local GPU=$1; shift
  local FAIL=0 ARM
  for ARM in "$@"; do
    run_arm "$GPU" "$ARM" || FAIL=1
  done
  return $FAIL
}

mkdir -p "$OUTROOT"
# 本地 2 卡分工：GPU0 = mavg → cavg_off（判决最相关两臂）；
#               GPU1 = cavg_g → fullkv。
run_chain 0 mavg cavg_off &
P0=$!
run_chain 1 cavg_g fullkv &
P1=$!
# 008：逐 PID 检查子 shell 退出状态（原 `wait $P0 $P1` 只回传最后一个）
wait "$P0"; S0=$?
wait "$P1"; S1=$?
if [ "$S0" -ne 0 ] || [ "$S1" -ne 0 ]; then
  echo "==== E123 派单存在失败（GPU0 链 rc=$S0；GPU1 链 rc=$S1）——失败任务见上方 PRED-FAIL/ARM-FAIL 行，日志在 $OUTROOT/e123_*.log ===="
  exit 1
fi
echo "==== E123 四臂 5 任务全部结束 $(date) ===="
