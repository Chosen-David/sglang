#!/bin/bash
# E110 e2e 草案：ccluster / cluster_sim_greedy vs ccluster_kmeans vs cavg 对照臂
# 【草案——不要现在执行！】须等 E109 扫描收官（主树双卡 + 远程双卡 ~1.5 天）后，
#   由主会话择机：① 先 merge dev-e110-ccluster 分支到主树 ② 再从主树跑本脚本
#   （或直接在 worktree 跑——benchmark.LongBench.pred 会加载当前工作树代码）
#
# 交付背景（2026-10-07 用户指令）：ccluster (cluster,cluster) 与 cluster_sim_greedy
#   的 e2e 实现（分支 dev-e110-ccluster，单测 9/9 过）。E108 mass probe 的
#   sim_greedy 9/9 未过质量门只是 mass 侧记录——本脚本就是补 e2e 实测的铁律步骤。
#
# 口径：与 /tmp/e109_scan_local.sh 严格一致（E98 v2 screen 同协议）
#   hotpotqa + musique，n=200，K1=128 K2=1024 cmp4，Qwen3-8B，e=0
#   （--tli_subspace 不传 = 默认 full：E98/E105/E109 全部 e2e 臂同口径）
#
# 新 flag（dev-e110-ccluster）：
#   --tli_near_select {4bit,cluster,sim_greedy}   near 侧 L2 选择（默认 4bit=现状）
#   --tli_far_select  {4bit,cluster,sim_greedy}   far 侧新增 sim_greedy
#   --tli_sim 0.9                                 贪心余弦归并阈值
#   --tli_sim_dims {subspace,nope,tail,full}      贪心聚类维度（默认 subspace=与 kmeans 同维）
#   注意：cluster/sim_greedy 臂须 --tli_enable_kmeans true（沿用 cavg 惯例）
#
# 预算语义（实现口径，与 cavg 的差异务必知情）：
#   cavg   （far=cluster, near=4bit ）：far=min(far_tokens=512,Tfar,K2_mid)，near=剩余；γ 死参数（现状）
#   ccluster（far=cluster, near=cluster）：γ 活化——nt_near=nb_near·bs·γ，far=max(64,K2_mid−nt_near)
#          （TASK.md L68-82 预算语义；只对新 ccluster 臂生效，cavg 原路径逐位不动）
#   near 簇覆盖=块对齐，右缘未对齐尾巴 token 用细筛原始分回退（质量 ≥ 簇代表分）
#
# 【性能预告——sim_greedy 臂慢，先冒烟再全量】
#   贪心聚类是时序逐 token 的 Python 循环：prefill 每 layer 全量建簇（far 区 ~5-20K 步），
#   decode 期增量续跑（精确，成本 ~新增段）。估算每请求 prefill 额外开销：
#   36 layers × far_tokens 步 ≈ 数十秒（hotpotqa ~9K ctx）到数分钟（长文任务）。
#   200 样本 × 2 任务的单 sim_greedy 臂可能要数小时——建议：
#   ① 先跑 SMOKE=1（n=8）测 wall time，可接受再放全量
#   ② sim 三值先只跑 0.9（用户探索值），0.85/0.95 视 0.9 结果再加
#   ③ ccluster_kmeans 臂没有此问题（kmeans 向量化，与 cavg 同量级）
#
# 臂设计（(α,β,γ) 代表点说明）：
#   ccluster_kmeans：cavg 冠军点 (0.125,0.25,0.75) + γ 低点 (0.125,0.25,0.375)——
#     ccluster 的 γ 是活参数（与 cavg 不同），必须扫 γ 才有信息量
#   cavg_sim：固定 cavg 冠军点扫 sim {0.85,0.9,0.95}——与 E109 cavg 臂只差聚类方式，
#     干净隔离「kmeans vs 增量贪心」单变量
#   ccluster_sim：同 ccluster_kmeans 的 γ 低点扫 sim——与 ccluster_kmeans 只差 near 侧
# 对照基准：E109 cavg 臂（/tmp/e109_scan/pred_E109_cavg_*）已有数据可直接对表
cd /home/wangyuanshuo02/wt-e110-ccluster/two-level-attention   # worktree 分支 dev-e110-ccluster
export PYTHONPATH=/home/wangyuanshuo02/wt-e110-ccluster/two-level-attention:/home/wangyuanshuo02/.local/pylibs
MODEL=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B
OUTROOT=/tmp/e110_ccluster
SMOKE=${SMOKE:-0}          # SMOKE=1 时只跑 n=8 冒烟（测 sim_greedy wall time）
N=$([ "$SMOKE" = "1" ] && echo 8 || echo 200)
mkdir -p $OUTROOT

run_arms () {
  local GPU=$1; shift
  for SPEC in "$@"; do
    set -- $SPEC
    local TAG=$1 A=$2 B=$3 G=$4; shift 4
    local REST="$*"
    local OUTDIR=$OUTROOT/pred_$TAG
    mkdir -p "$OUTDIR"
    local NEED=0
    for t in hotpotqa musique; do
      if ! ls "$OUTDIR"/${t}-tli_*.jsonl >/dev/null 2>&1 || \
         [ "$(cat "$OUTDIR"/${t}-tli_*.jsonl 2>/dev/null | wc -l)" -lt "$N" ]; then
        NEED=1
      fi
    done
    if [ $NEED -eq 0 ]; then echo "==== SKIP $TAG (done) ===="; continue; fi
    echo "==== $TAG GPU$GPU $(date +%m-%d\ %H:%M:%S) ===="
    for t in hotpotqa musique; do
      if ls "$OUTDIR"/${t}-tli_*.jsonl >/dev/null 2>&1 && \
         [ "$(cat "$OUTDIR"/${t}-tli_*.jsonl 2>/dev/null | wc -l)" -ge "$N" ]; then
        continue
      fi
      CUDA_VISIBLE_DEVICES=$GPU python -u -m benchmark.LongBench.pred \
        --model Qwen3-8B --model_path $MODEL \
        --task $t --method tli --e 0 \
        --dataset-path /home/wangyuanshuo02/datasets/LongBench/data \
        --config-path benchmark/LongBench/config \
        --output-dir $OUTROOT --pred_postfix _$TAG \
        --tia_level1_topk 128 --tia_level2_topk 1024 --tia_level2_cmp_ratio 4 \
        --tli_enable_layer_skip false \
        --tli_alpha $A --tli_beta $B --tli_gamma $G $REST \
        2>&1 | tail -2
    done
  done
  echo "==== GPU$GPU ARMS DONE $(date +%m-%d\ %H:%M:%S) ===="
}

# ---- 冒烟（SMOKE=1 时只跑这一个臂，测 sim_greedy 的 wall time）----
if [ "$SMOKE" = "1" ]; then
  run_arms 0 \
    'E110_cavgsim_s0.9_a0.125_b0.25_g0.75 0.125 0.25 0.75 --tli_far_select sim_greedy --tli_near_select 4bit --tli_enable_kmeans true --tli_sim 0.9'
  exit 0
fi

# ---- 全量：9 臂双卡分工（kmeans 臂快放前段，greedy 臂慢放后段排队）----
run_arms 0 \
  'E110_ccluster_km_a0.125_b0.25_g0.75 0.125 0.25 0.75 --tli_far_select cluster --tli_near_select cluster --tli_enable_kmeans true --tli_far_method minmax --tli_near_method avg' \
  'E110_ccluster_km_a0.125_b0.25_g0.375 0.125 0.25 0.375 --tli_far_select cluster --tli_near_select cluster --tli_enable_kmeans true --tli_far_method minmax --tli_near_method avg' \
  'E110_cavgsim_s0.9_a0.125_b0.25_g0.75 0.125 0.25 0.75 --tli_far_select sim_greedy --tli_near_select 4bit --tli_enable_kmeans true --tli_sim 0.9' \
  'E110_cavgsim_s0.85_a0.125_b0.25_g0.75 0.125 0.25 0.75 --tli_far_select sim_greedy --tli_near_select 4bit --tli_enable_kmeans true --tli_sim 0.85' \
  'E110_cavgsim_s0.95_a0.125_b0.25_g0.75 0.125 0.25 0.75 --tli_far_select sim_greedy --tli_near_select 4bit --tli_enable_kmeans true --tli_sim 0.95' &
P0=$!
run_arms 1 \
  'E110_ccluster_km_a0.25_b0.25_g0.5 0.25 0.25 0.5 --tli_far_select cluster --tli_near_select cluster --tli_enable_kmeans true --tli_far_method minmax --tli_near_method avg' \
  'E110_ccluster_sim_s0.9_a0.125_b0.25_g0.375 0.125 0.25 0.375 --tli_far_select sim_greedy --tli_near_select sim_greedy --tli_enable_kmeans true --tli_sim 0.9' \
  'E110_ccluster_sim_s0.85_a0.125_b0.25_g0.375 0.125 0.25 0.375 --tli_far_select sim_greedy --tli_near_select sim_greedy --tli_enable_kmeans true --tli_sim 0.85' \
  'E110_ccluster_sim_s0.95_a0.125_b0.25_g0.375 0.125 0.25 0.375 --tli_far_select sim_greedy --tli_near_select sim_greedy --tli_enable_kmeans true --tli_sim 0.95' &
P1=$!
wait $P0 $P1
echo "[$(date +%m-%d\ %H:%M:%S)] E110_LOCAL_DONE"

# ---- 打分（拉回本机跑，远程环境缺 jieba 等）：复用 run_e71_eval.py 口径 ----
# PYTHONPATH=/home/wangyuanshuo02/wt-e110-ccluster/two-level-attention:/home/wangyuanshuo02/.local/pylibs \
# python3 exp/trace/run_e71_eval.py --pred_root /tmp/e110_ccluster ...（对照 E105/E109 打分脚本调用方式）
