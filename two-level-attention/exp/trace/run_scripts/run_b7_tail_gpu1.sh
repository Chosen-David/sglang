#!/bin/bash
# B7 LongBench 尾部任务拆到 GPU1 并行（用户指示：非性能敏感任务双卡并行提速）
# 等 RULER B7 跑完 GPU1 释放后自动启动；跑 triviaqa/lcc/repobench-p（GPU0 只剩 multi_news+narrativeqa）
cd /home/wangyuanshuo02/two-level-attention
MODEL=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B
# 等 GPU1 的 RULER 全部完成
while ! grep -q "RULER B7 ALL DONE" /tmp/tli_ruler_b7.log 2>/dev/null; do
  sleep 60
done
echo "=== [GPU1-tail $(date +%H:%M:%S)] RULER done, starting tail tasks ==="
for T in triviaqa lcc repobench-p; do
  echo "=== [GPU1-tail $(date +%H:%M:%S)] task=$T ==="
  CUDA_VISIBLE_DEVICES=1 python -u -m benchmark.LongBench.pred \
    --model Qwen3-8B --model_path $MODEL \
    --task $T --method tli --e 0 --t 09290152 \
    --dataset-path /home/wangyuanshuo02/datasets/LongBench/data \
    --config-path benchmark/LongBench/config \
    --output-dir exp/results_longbench/Qwen3-8B \
    --tia_level1_topk 128 --tia_level2_topk 1024 --tia_level2_cmp_ratio 4 \
    --tli_enable_kmeans false --tli_enable_layer_skip false \
    --tli_alpha 0.125 --tli_beta 0.25 --tli_gamma 0.125 \
    --pred_postfix _b7 2>&1 | tail -2
done
echo "B7 GPU1-TAIL DONE"
