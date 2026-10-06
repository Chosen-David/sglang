#!/bin/bash
# E87c：细筛维度口径校准臂（tail32 版关键臂）——消除 E87/E72 vs B7s 的维度混杂
# 背景：tli_subspace 默认 full（22ecb4d），E87 sigma 臂与 E72 method 组合臂均全维细筛，
#       而 twolvl 参照 B7s = tail32。本脚本补 tail 口径关键臂做干净对比：
#   1. near_sigma_8 tail 版（E87 核心新发现的口径内复验）
#   2. mavg E72 冠军臂 tail 版（far_method=mavg near_method=avg α.125/β.375）
# 用法：bash e87c_tail_caliber.sh <gpu 0|1> <arms...>   双卡各跑一部分
GPU=$1; shift
ARMS="$@"
cd /home/wangyuanshuo02/two-level-attention
while true; do
  if [ "$GPU" = "0" ] && grep -q "MOBA_FULL_GPU0_DONE" /tmp/relay2_gpu0.log 2>/dev/null; then break; fi
  if [ "$GPU" = "1" ] && grep -q "MOBA_FULL_GPU1_DONE" /tmp/relay2_gpu1.log 2>/dev/null; then break; fi
  sleep 120
done
echo "moba done, starting E87c tail caliber arms on GPU$GPU"
export CUDA_VISIBLE_DEVICES=$GPU
COMMON="--model Qwen3-8B --model_path /mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B \
  --method tli --e 0 --dataset-path /home/wangyuanshuo02/datasets/LongBench/data \
  --config-path benchmark/LongBench/config \
  --tia_level1_topk 128 --tia_level2_topk 1024 --tia_level2_cmp_ratio 4 \
  --tli_enable_kmeans false --tli_enable_layer_skip false \
  --tli_alpha 0.125 --tli_beta 0.25 --tli_gamma 0.125 \
  --tli_subspace tail"
for arm in $ARMS; do
  for t in hotpotqa musique; do
    if [ "$arm" = "sig_near8" ]; then
      python -u -m benchmark.LongBench.pred --task $t $COMMON \
        --tli_sigma_select near --tli_sigma 8 \
        --output-dir /tmp/e87c_tail --pred_postfix _signear8_tail
    elif [ "$arm" = "mavg" ]; then
      python -u -m benchmark.LongBench.pred --task $t $COMMON \
        --tli_far_method minmax --tli_near_method avg \
        --tli_alpha 0.125 --tli_beta 0.375 \
        --output-dir /tmp/e87c_tail --pred_postfix _mavg_tail
    fi
    echo "DONE e87c $arm $t"
  done
done
echo "E87C_TAIL_GPU${GPU}_DONE"
