#!/bin/bash
# E109 三件套 #3：RULER 32K/64K/128K 冠军链批量调度（2026-10-09）
# 不改动旧版 run_ruler.sh（B06 修复版继续服务 4K-16K 档）。
#
# 与旧版差异：
#   - 三长度档：32768（原生 RoPE，不加 --yarn）/ 65536 / 131072（YaRN 自动加）
#   - 三臂参数化：mavg 冠军 / aavg Pareto 候选 / fullkv baseline（--method none）
#   - SKIP 幂等：输出文件已有行数 ≥ 当前 max_num 则跳过（断点续跑；
#     冒烟 max-num 2 的残文件在全量 max-num 100 时会因 2<100 重跑覆盖）
#   - 输出目录：exp/results_ruler/e109_full_Qwen3-8B/{arm}/L{len}/pred_1024/
#   - B06 同款纪律：逐任务 PIPESTATUS 检查 + FAILED 计数 + 失败 exit 1
#
# 用法：bash benchmark/RULER/run_ruler_e109.sh <arm> <gpu_id> [max_num] [lens]
#   arm ∈ {mavg, aavg, fullkv}；lens 默认 "32768 65536 131072"（空格分隔可 subset）
#   例：bash benchmark/RULER/run_ruler_e109.sh mavg 0 100 "65536 131072"
cd /home/wangyuanshuo02/sglang/two-level-attention || exit 1
MODEL=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B
DATA=/home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/RULER
ARM=$1
GPU=$2
N=${3:-100}
LENS=${4:-"32768 65536 131072"}
TS=$(date +%m%d%H%M)
OUTROOT=exp/results_ruler/e109_full_Qwen3-8B
POSTFIX=_1024
TASKS="niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multiquery niah_multivalue cwe fwe vt"

# ---- 三臂参数（E109 海选终判 2026-10-09：mavg 冠军 46.95 / aavg(0,0) Pareto hq 王）----
# 共同参数：--tia_level1_topk 128 --tia_level2_topk 1024 --tia_level2_cmp_ratio 4
#           --tli_enable_layer_skip false（tli 臂）
case $ARM in
  mavg)
    METHOD=tli
    TLI_ARGS="--tli_far_method minmax --tli_near_method avg --tli_enable_kmeans false --tli_alpha 0.25 --tli_beta 0.125 --tli_gamma 0.625"
    ;;
  aavg)
    METHOD=tli
    TLI_ARGS="--tli_far_method avg --tli_near_method avg --tli_enable_kmeans false --tli_alpha 0 --tli_beta 0 --tli_gamma 0"
    ;;
  fullkv)
    METHOD=none
    TLI_ARGS=""
    ;;
  *)
    echo "usage: $0 <mavg|aavg|fullkv> <gpu_id> [max_num] [lens]"
    exit 1
    ;;
esac

FAILED=0
SKIPPED=0
DONE=0
for L in $LENS; do
  # 超原生 40960 的档位自动加 --yarn（Qwen3 官方配方；32768 原生档零扰动）
  YARN=""
  if [ "$L" -gt 40960 ]; then YARN="--yarn"; fi
  # 输出文件名由 pred_ruler 决定：{task}-{method_name}-{t}.jsonl
  # method_name 含 method+参数信息，这里用 glob 匹配任意 t
  for T in $TASKS; do
    OUTDIR=$OUTROOT/$ARM/L$L/pred$POSTFIX
    mkdir -p "$OUTDIR"
    # SKIP 幂等：已有文件行数 >= N 则跳过
    EXIST=$(ls $OUTDIR/$T-*.jsonl 2>/dev/null | head -1)
    if [ -n "$EXIST" ]; then
      NLINES=$(wc -l < "$EXIST")
      if [ "$NLINES" -ge "$N" ]; then
        echo "=== SKIP L=$L task=$T arm=$ARM (exist $NLINES >= $N) ==="
        SKIPPED=$((SKIPPED+1))
        continue
      fi
    fi
    echo "=== [$(date +%H:%M:%S)] L=$L task=$T arm=$ARM n=$N ==="
    CUDA_VISIBLE_DEVICES=$GPU python -u -m benchmark.RULER.pred_ruler \
      --model Qwen3-8B --model_path $MODEL \
      --task $T --context_length $L --method $METHOD --t $TS \
      --data-root $DATA \
      --output-dir $OUTROOT/$ARM \
      --tia_level1_topk 128 --tia_level2_topk 1024 \
      --tia_level2_cmp_ratio 4 --tli_enable_layer_skip false \
      $TLI_ARGS $YARN \
      --pred_postfix $POSTFIX --max-num $N 2>&1 | tail -2
    rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
      echo "==== FAILED L=$L task=$T arm=$ARM rc=$rc ===="
      FAILED=$((FAILED+1))
    else
      DONE=$((DONE+1))
    fi
  done
done
echo "E109 RULER $ARM DONE (done=$DONE skip=$SKIPPED failed=$FAILED)"
if [ "$FAILED" -ne 0 ]; then
  exit 1
fi
echo "E109 RULER $ARM ALL DONE"
