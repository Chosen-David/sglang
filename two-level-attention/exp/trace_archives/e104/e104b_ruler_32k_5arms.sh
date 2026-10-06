#!/bin/bash
# E104b（#111 配套）：RULER 32K 档五方法臂补齐——FullKV(none)/Quest/TIA/单池
# 触发条件：/tmp/e104_ruler_32k.log 出现 E104_RULER_32K_DONE（双 TLI 臂跑完释放 GPU）
# 目的：主表 tab:ruler 32K 行需要 FullKV 锚点（「PSI 持平 FullKV」延伸）与 Quest 锚点
#   （「下界漏选随长度放大」延伸：87.91→81.76→70.18 的 32K 第四点）；TIA/单池补全五列。
# 命令口径逐字对齐历史产物命名（exp/results_ruler/Qwen3-8B/L*/pred_1024/）：
#   none = --method none；quest_64_16 = --method quest（quest_block_size 64/topk 16 均默认）；
#   tia_64_128_1024_c4 = --method tia --tia_level1_topk 128 --tia_level2_topk 1024
#     --tia_level2_cmp_ratio 4（tia_block_size 64 默认）；
#   单池 = --method tli + 预算三参 + αβγ 全默认（α=0/β=0/γ=1，C0 口径，不传 tli_alpha）
# 顺序（优先级从高到低）：GPU0 none→GPU1 quest→GPU0 tia→GPU1 c0（单池）
cd /home/wangyuanshuo02/two-level-attention
export PYTHONPATH=/home/wangyuanshuo02/two-level-attention:/home/wangyuanshuo02/.local/pylibs

while ! grep -aq "E104_RULER_32K_DONE" /tmp/e104_ruler_32k.log 2>/dev/null; do
  echo "[$(date +%H:%M:%S)] waiting E104_RULER_32K_DONE ..."; sleep 180
done
echo "[$(date +%H:%M:%S)] E104 done, start E104b five-method arms"

L=32768
N=100
MODEL=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B
DATA=/home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/RULER
TASKS="niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multiquery niah_multivalue cwe fwe vt"
OUTROOT=/tmp/e104_ruler_32k
mkdir -p "$OUTROOT"

run_arm () {
  local GPU=$1; local POSTFIX=$2; local TS=$3; shift 3
  local EXTRA="$@"
  for T in $TASKS; do
    local PRED_DIR=$OUTROOT/L$L/pred_${POSTFIX}
    if ls "$PRED_DIR"/${T}-*.jsonl >/dev/null 2>&1 && \
       [ "$(cat "$PRED_DIR"/${T}-*.jsonl 2>/dev/null | wc -l)" -ge $N ]; then
      echo "==== SKIP $POSTFIX $T (done) ===="; continue
    fi
    echo "==== GPU$GPU $POSTFIX $T $(date +%H:%M:%S) ===="
    CUDA_VISIBLE_DEVICES=$GPU python -u -m benchmark.RULER.pred_ruler \
      --model Qwen3-8B --model_path $MODEL \
      --task $T --context_length $L --t $TS \
      --data-root $DATA \
      --output-dir $OUTROOT \
      --pred_postfix _${POSTFIX} --max-num $N $EXTRA 2>&1 | tail -2
  done
}

# ---- 阶段 1：FullKV + Quest（核心锚点，双卡并行）----
run_arm 0 E104B_FULLKV e104full --method none &
P0=$!
run_arm 1 E104B_QUEST e104quest --method quest &
P1=$!
wait $P0 $P1
echo "[$(date +%H:%M:%S)] stage1 fullkv+quest done"

# ---- 阶段 2：TIA + 单池（表完整性，双卡并行）----
run_arm 0 E104B_TIA e104tia --method tia \
  --tia_level1_topk 128 --tia_level2_topk 1024 --tia_level2_cmp_ratio 4 &
P0=$!
run_arm 1 E104B_C0 e104c0 --method tli \
  --tia_level1_topk 128 --tia_level2_topk 1024 --tia_level2_cmp_ratio 4 \
  --tli_enable_kmeans false --tli_enable_layer_skip false &
P1=$!
wait $P0 $P1
echo "[$(date +%H:%M:%S)] stage2 tia+c0 done"

echo "E104B_RULER_32K_DONE"
