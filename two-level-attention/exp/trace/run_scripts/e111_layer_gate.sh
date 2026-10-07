#!/bin/bash
# E111（用户 2026-10-07 指令）：D' 层 gate 三臂 e2e 对照（no-skip / static / dynamic）
# =====================================================================
# !! 本脚本为草案，未执行（GPU 被 E109 扫描占用）——留待 GPU 空闲后跑 !!
# =====================================================================
# 设计：
#   任务 hotpotqa + musique（n=200，E98 v2 screen 同协议），K1=128 K2=1024 cmp4
#   method = E109 最优组合【占位：E98 best mavg α.125 β.375 γ.625——E109 收官后
#   替换下方 METHOD_* 四个变量即可，臂结构与 gate 参数正交】
#   四臂：
#     noskip  : --tli_layer_gate none   （主表口径对照，= --tli_enable_layer_skip false）
#     static  : --tli_layer_gate static （旧 DEFAULT_MASK 静态层掩码 [0,1,7-16,35]）
#     dyn01   : --tli_layer_gate dynamic --tli_layer_gate_tau 0.1（E67 离线：跳 ~31% 层）
#     dyn02   : --tli_layer_gate dynamic --tli_layer_gate_tau 0.2（E67 离线：跳 ~12% 层，
#                mass 损失 1.7%——保守臂）
#   验收口径：atten mass ≠ e2e 铁律——精度以 run_e71_eval.py 打分为准；
#   速度侧另需 kernel/e2e 延迟对比（留 GPU 空闲后，参照 E107 三段式口径）。
#
# 注意：
#   * 路径指向 dev-e111-layerskip worktree（主树被 E109 占用，臂进程即时加载
#     工作树代码）；**主会话合并 patch 到 two-level-indexer 后改回主树路径**
#   * gate flag 与 --tli_enable_layer_skip false 并传是显式覆盖写法（gate 优先）；
#     只传 --tli_layer_gate 也等价（旧 flag 默认 True 会被覆盖），双写更防误读
#   * CPU 单测已验证：gate=none 与 HEAD 逐位一致、dynamic 判定 τ 行为、
#     static ≡ 旧 flag、省算路径 ≡ 完整计算+far -inf（exp/trace/test_e111_layer_gate.py）
set -u
cd /home/wangyuanshuo02/wt-e111-layerskip/two-level-attention
export PYTHONPATH=/home/wangyuanshuo02/wt-e111-layerskip/two-level-attention:/home/wangyuanshuo02/.local/pylibs
MODEL=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B
OUTROOT=/tmp/e111_layer_gate
mkdir -p $OUTROOT

# ---- method 组合占位（E109 未收官；E98 best mavg 作占位臂）----
METHOD_FAR=avg      # mavg = (minmax→已废, far avg) ; E109 最优出来后替换
METHOD_NEAR=minmax
METHOD_ALPHA=0.125
METHOD_BETA=0.375
METHOD_GAMMA=0.625

run_arms () {
  local GPU=$1; shift
  for SPEC in "$@"; do
    set -- $SPEC
    local TAG=$1 GATE=$2 TAU=$3; shift 3
    local REST="$*"
    local OUTDIR=$OUTROOT/pred_$TAG
    mkdir -p "$OUTDIR"
    local NEED=0
    for t in hotpotqa musique; do
      if ! ls "$OUTDIR"/${t}-tli_*.jsonl >/dev/null 2>&1 || \
         [ "$(cat "$OUTDIR"/${t}-tli_*.jsonl 2>/dev/null | wc -l)" -lt 200 ]; then
        NEED=1
      fi
    done
    if [ $NEED -eq 0 ]; then echo "==== SKIP $TAG (done) ===="; continue; fi
    echo "==== $TAG GPU$GPU $(date +%m-%d\ %H:%M:%S) ===="
    for t in hotpotqa musique; do
      if ls "$OUTDIR"/${t}-tli_*.jsonl >/dev/null 2>&1 && \
         [ "$(cat "$OUTDIR"/${t}-tli_*.jsonl 2>/dev/null | wc -l)" -ge 200 ]; then
        continue
      fi
      CUDA_VISIBLE_DEVICES=$GPU TLI_DEBUG=1 python -u -m benchmark.LongBench.pred \
        --model Qwen3-8B --model_path $MODEL \
        --task $t --method tli --e 0 \
        --dataset-path /home/wangyuanshuo02/datasets/LongBench/data \
        --config-path benchmark/LongBench/config \
        --output-dir $OUTROOT --pred_postfix _$TAG \
        --tia_level1_topk 128 --tia_level2_topk 1024 --tia_level2_cmp_ratio 4 \
        --tli_enable_layer_skip false \
        --tli_layer_gate $GATE --tli_layer_gate_tau $TAU \
        --tli_far_method $METHOD_FAR --tli_near_method $METHOD_NEAR \
        --tli_alpha $METHOD_ALPHA --tli_beta $METHOD_BETA --tli_gamma $METHOD_GAMMA \
        $REST \
        2>&1 | tee "$OUTDIR/${t}_run.log" | grep -E "\[E111\]|far_frac" | head -200 \
        > "$OUTDIR/${t}_gate_trace.log" || true
      # 注：TLI_DEBUG=1 时 dynamic 臂每请求逐层打印 [E111] far_frac/skip 决策
      #     （打分侧可统计实际跳层率，与 E67 离线推算对照）
    done
  done
  echo "==== GPU$GPU ARMS DONE $(date +%m-%d\ %H:%M:%S) ===="
}

# 四臂两卡分摊（noskip+static 一卡，dyn01+dyn02 一卡；TAU 占位 none 的臂不消费）
run_arms 0 \
  'E111_noskip none 0' \
  'E111_static static 0' &
P0=$!
run_arms 1 \
  'E111_dyn01 dynamic 0.1' \
  'E111_dyn02 dynamic 0.2' &
P1=$!
wait $P0 $P1

# ---- 打分（拉回本机口径，参照既有 run_e71_eval.py 流程；此处只列占位）----
# python exp/trace/run_e71_eval.py --pred-root $OUTROOT --tasks hotpotqa,musique \
#     --arms E111_noskip,E111_static,E111_dyn01,E111_dyn02
echo "[$(date +%m-%d\ %H:%M:%S)] E111_LAYER_GATE_DONE（打分与速度对比留 GPU 空闲后补）"
