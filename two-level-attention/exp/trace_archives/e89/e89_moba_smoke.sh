#!/bin/bash
# E89 MoBA 臂完整冒烟：hotpotqa 200 样本（GPU1 与 E87 共享）
cd /home/wangyuanshuo02/two-level-attention
export CUDA_VISIBLE_DEVICES=1
python -u -m benchmark.LongBench.pred \
  --model Qwen3-8B --model_path /mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B \
  --task hotpotqa --method tli --e 0 --t moba \
  --dataset-path /home/wangyuanshuo02/datasets/LongBench/data \
  --config-path benchmark/LongBench/config --output-dir /tmp/e89_moba \
  --tia_level1_topk 128 --tia_level2_topk 1024 --tia_level2_cmp_ratio 4 \
  --tli_enable_kmeans false --tli_enable_layer_skip false \
  --tli_subspace tail --tli_moba \
  --pred_postfix _moba
echo "MOBA_SMOKE_DONE"
python -u -m benchmark.LongBench.pred \
  --model Qwen3-8B --model_path /mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B \
  --task musique --method tli --e 0 --t moba \
  --dataset-path /home/wangyuanshuo02/datasets/LongBench/data \
  --config-path benchmark/LongBench/config --output-dir /tmp/e89_moba \
  --tia_level1_topk 128 --tia_level2_topk 1024 --tia_level2_cmp_ratio 4 \
  --tli_enable_kmeans false --tli_enable_layer_skip false \
  --tli_subspace tail --tli_moba \
  --pred_postfix _moba
echo "MOBA_SMOKE_ALL_DONE"
