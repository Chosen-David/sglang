#!/bin/bash
# GPU1 接力：E87 完成后 → StreamingLLM 后 6 任务 → E90 子空间 screen (nope/random/highfreq)
cd /home/wangyuanshuo02/two-level-attention
while ! grep -q "E87_GPU1_ALL_DONE" /tmp/e87_gpu1.log 2>/dev/null; do sleep 60; done
echo "E87 GPU1 done, starting streamingllm"
export CUDA_VISIBLE_DEVICES=1
TASKS1="narrativeqa triviaqa qmsum multi_news lcc repobench-p"
for t in $TASKS1; do
  python -u -u exp/trace/pred_kvcf.py --task $t --method streamingllm \
    --max-capacity 1024 --gpu 0
  echo "DONE streamingllm $t"
done
echo "STREAMINGLLM_GPU1_DONE"
COMMON="--model Qwen3-8B --model_path /mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B \
  --method tli --e 0 --dataset-path /home/wangyuanshuo02/datasets/LongBench/data \
  --config-path benchmark/LongBench/config \
  --tia_level1_topk 128 --tia_level2_topk 1024 --tia_level2_cmp_ratio 4 \
  --tli_enable_kmeans false --tli_enable_layer_skip false \
  --tli_alpha 0.125 --tli_beta 0.25 --tli_gamma 0.125"
for sub in nope random highfreq; do
  for t in hotpotqa musique; do
    python -u -m benchmark.LongBench.pred --task $t $COMMON \
      --tli_subspace $sub --output-dir /tmp/e90_sub --pred_postfix _sub$sub
    echo "DONE e90 $sub $t"
  done
done
echo "RELAY_GPU1_ALL_DONE"
