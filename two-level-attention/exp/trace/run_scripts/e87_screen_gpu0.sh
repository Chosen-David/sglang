#!/bin/bash
# E87 e2e screen GPU0: hotpotqa × 5 臂
cd /home/wangyuanshuo02/two-level-attention
export CUDA_VISIBLE_DEVICES=0
COMMON="--model Qwen3-8B --model_path /mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B \
  --method tli --e 0 --dataset-path /home/wangyuanshuo02/datasets/LongBench/data \
  --config-path benchmark/LongBench/config \
  --tia_level1_topk 128 --tia_level2_topk 1024 --tia_level2_cmp_ratio 4 \
  --tli_enable_kmeans false --tli_enable_layer_skip false \
  --tli_alpha 0.125 --tli_beta 0.25 --tli_gamma 0.125"
run() {
  python -u -m benchmark.LongBench.pred --task hotpotqa $COMMON \
    --tli_sigma_select $1 --tli_sigma $2 \
    --output-dir /tmp/e87_e2e --pred_postfix _sig$1_$2
  echo "DONE hotpotqa sig$1_$2"
}
run near 2
run near 8
run near 32
run mid 8
run far 8
echo "E87_GPU0_ALL_DONE"
