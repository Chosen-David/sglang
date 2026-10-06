#!/bin/bash
# E71 主表重跑：C 配置（α=0 单池 bp128，质量最优）与 B 配置（帕累托点 bp128+sup_wsvd d8+α.125/β.25/γ.25）
# 口径修正（2026-09-28）：E64 trace 的 B_TOK=2048 vs 主表 K2=1024——γ 按总预算等比缩放 0.5→0.25
# （nt_near=32·64·γ：γ0.5 时=1024 吃光 K2=1024 致 far 仅保底 64 → F1 崩 13.51；γ0.25 → near/far=512/512）
# 双 GPU 并行：GPU0=C 全任务，GPU1=B 全任务；hotpotqa 冒烟已完成的跳过
cd /home/wangyuanshuo02/two-level-attention
MODEL=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B
BASIS=/home/wangyuanshuo02/two-level-attention/exp/trace/results/sup_wsvd_basis_qwen3-8b_r8.pt
TS=$(date +%m%d%H%M)
TASKS="2wikimqa musique passage_retrieval_en qasper multifieldqa_en gov_report qmsum multi_news narrativeqa triviaqa lcc repobench"

run_c () {
  for T in $TASKS; do
    echo "=== [C $(date +%H:%M:%S)] task=$T ==="
    CUDA_VISIBLE_DEVICES=0 python -u -m benchmark.LongBench.pred \
      --model Qwen3-8B --model_path $MODEL \
      --task $T --method tli --e 0 --t $TS \
      --dataset-path /home/wangyuanshuo02/datasets/LongBench/data \
      --config-path benchmark/LongBench/config \
      --output-dir exp/results_longbench/Qwen3-8B \
      --tia_level1_topk 128 --tia_level2_topk 1024 --tia_level2_cmp_ratio 4 \
      --tli_enable_kmeans false --tli_enable_layer_skip false \
      --pred_postfix _c0 2>&1 | tail -2
  done
  echo "C ALL DONE"
}

run_b () {
  for T in $TASKS; do
    echo "=== [B $(date +%H:%M:%S)] task=$T ==="
    CUDA_VISIBLE_DEVICES=1 python -u -m benchmark.LongBench.pred \
      --model Qwen3-8B --model_path $MODEL \
      --task $T --method tli --e 0 --t $TS \
      --dataset-path /home/wangyuanshuo02/datasets/LongBench/data \
      --config-path benchmark/LongBench/config \
      --output-dir exp/results_longbench/Qwen3-8B \
      --tia_level1_topk 128 --tia_level2_topk 1024 --tia_level2_cmp_ratio 4 \
      --tli_enable_layer_skip false \
      --tli_alpha 0.125 --tli_beta 0.25 --tli_gamma 0.25 \
      --tli_proj_basis $BASIS \
      --pred_postfix _b0 2>&1 | tail -2
  done
  echo "B ALL DONE"
}

run_c > /tmp/tli_e71_c.log 2>&1 &
run_b > /tmp/tli_e71_b.log 2>&1 &
wait
echo "E71 ALL DONE"
