#!/bin/bash
# 第二段接力：relay 链完成 → MoBA 全量 13 任务双卡拆分（已存在的任务跳过）
cd /home/wangyuanshuo02/two-level-attention
GPU=$1
TASKS=$2
while true; do
  if [ "$GPU" = "0" ] && grep -q "RELAY_GPU0_ALL_DONE" /tmp/relay_gpu0.log 2>/dev/null; then break; fi
  if [ "$GPU" = "1" ] && grep -q "RELAY_GPU1_ALL_DONE" /tmp/relay_gpu1.log 2>/dev/null; then break; fi
  sleep 120
done
echo "relay done, starting MoBA full on GPU$GPU"
export CUDA_VISIBLE_DEVICES=$GPU
for t in $TASKS; do
  if ls /tmp/e89_moba/pred_moba/${t}-tli_*.jsonl >/dev/null 2>&1; then
    echo "SKIP moba $t (exists)"
    continue
  fi
  python -u -m benchmark.LongBench.pred \
    --model Qwen3-8B --model_path /mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B \
    --task $t --method tli --e 0 --t moba \
    --dataset-path /home/wangyuanshuo02/datasets/LongBench/data \
    --config-path benchmark/LongBench/config --output-dir /tmp/e89_moba \
    --tia_level1_topk 128 --tia_level2_topk 1024 --tia_level2_cmp_ratio 4 \
    --tli_enable_kmeans false --tli_enable_layer_skip false \
    --tli_subspace tail --tli_moba \
    --pred_postfix _moba
  echo "DONE moba $t"
done
echo "MOBA_FULL_GPU${GPU}_DONE"
