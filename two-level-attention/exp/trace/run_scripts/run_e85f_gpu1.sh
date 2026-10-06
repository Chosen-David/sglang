#!/bin/bash
# E85f：per-layer 静态 pair 的 e2e 全量判决（E85e 重放 0.849 vs tail32 0.811 的
# e2e 验证；E66 投影基 trace 成立 e2e 崩的前车之鉴——判 q 统计静态化是否同样
# 对 decode q 漂移脆弱）。臂 = E72 mavg 冠军配置 + --tli_static_pair：
#   α=0.125 β=0.375 γ=0.125 (minmax,avg) tail32→静态pair16 K2=1024
# 任务序：musique/qasper 前置（far-heavy 多跳 = 最快出判决信号），其余 11 后续。
# 等 GPU1 的 KVCF_GPU1_DONE 尾标（E81 h2o+pyramidkv 放量收官）后自动启动。
cd /home/wangyuanshuo02/two-level-attention
MODEL=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B
TS=0930e85f

# 等 E81 GPU1 放量收官（最多等 14h，防死等）
for i in $(seq 1 840); do
  grep -q KVCF_GPU1_DONE /tmp/kvcf_gpu1.log 2>/dev/null && break
  sleep 60
done
if ! grep -q KVCF_GPU1_DONE /tmp/kvcf_gpu1.log 2>/dev/null; then
  echo "E85F_WAIT_TIMEOUT $(date)" >> /tmp/e85f_gpu1.log
  exit 1
fi
sleep 60

for T in musique qasper hotpotqa 2wikimqa passage_retrieval_en multifieldqa_en gov_report qmsum multi_news narrativeqa triviaqa lcc repobench-p; do
  echo "=== [E85f GPU1 $(date +%H:%M:%S)] task=$T ===" >> /tmp/e85f_gpu1.log
  CUDA_VISIBLE_DEVICES=1 python -u -m benchmark.LongBench.pred \
    --model Qwen3-8B --model_path $MODEL \
    --task $T --method tli --e 0 --t $TS \
    --dataset-path /home/wangyuanshuo02/datasets/LongBench/data \
    --config-path benchmark/LongBench/config \
    --output-dir exp/results_longbench/Qwen3-8B \
    --tia_level1_topk 128 --tia_level2_topk 1024 --tia_level2_cmp_ratio 4 \
    --tli_enable_kmeans false --tli_enable_layer_skip false \
    --tli_alpha 0.125 --tli_beta 0.375 --tli_gamma 0.125 \
    --tli_subspace tail --tli_static_pair \
    --pred_postfix _e85f 2>&1 | tail -2 >> /tmp/e85f_gpu1.log
done
echo "E85F_DONE" >> /tmp/e85f_gpu1.log
