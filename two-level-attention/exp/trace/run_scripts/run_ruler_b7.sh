#!/bin/bash
# E71 RULER B7 配置（修复版分区 α=0.125 β=0.25 γ=0.125）11 任务 × 3 长度
# 与 C0（α=0 单池）同 harness 对比 → 「分区有用」消融列
# 用法：bash /tmp/run_ruler_b7.sh <gpu_id> [max_num]
cd /home/wangyuanshuo02/two-level-attention
MODEL=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B
DATA=/home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/RULER
GPU=${1:-1}
N=${2:-100}
TS=$(date +%m%d%H%M)
TASKS="niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multiquery niah_multivalue cwe fwe vt"
for L in 4096 8192 16384; do
  for T in $TASKS; do
    echo "=== [$(date +%H:%M:%S)] L=$L task=$T ==="
    CUDA_VISIBLE_DEVICES=$GPU python -u -m benchmark.RULER.pred_ruler \
      --model Qwen3-8B --model_path $MODEL \
      --task $T --context_length $L --method tli --t $TS \
      --data-root $DATA \
      --output-dir exp/results_ruler/Qwen3-8B \
      --tia_level1_topk 128 --tia_level2_topk 1024 --tia_level2_cmp_ratio 4 \
      --tli_enable_kmeans false --tli_enable_layer_skip false \
      --tli_alpha 0.125 --tli_beta 0.25 --tli_gamma 0.125 \
      --pred_postfix _b7 --max-num $N 2>&1 | tail -2
  done
done
echo "RULER B7 ALL DONE"
