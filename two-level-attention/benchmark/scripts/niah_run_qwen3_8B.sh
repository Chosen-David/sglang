cd /home/hadoop-friday-llm/VSCodeProjects/sparse-attn-dev_huyuxuan09

MODEL=Qwen3-8B
MODELPATH=hf_models/downloads/Qwen/Qwen3-8B

OUTPUT_DIR=exp/results_longbench/$MODEL

# Use HF load_dataset. Modify this if you want to use local data set.
DATASET_PATH="offline_data/LongBench/data"

CONFIG_PATH=benchmark/LongBench/config/

mkdir -p $OUTPUT_DIR

method=$1
device=$2

budget_optipns_512="\
  --tia_level2_topk 512 \
  --tia_level2_cmp_ratio 4 \
  --quest_topk 8 \
  --twi_level2_topp 0.90 \
  --tia_enable_async_topk \
"

budget_optipns_1024="\
  --tia_level2_topk 1024 \
  --tia_level2_cmp_ratio 4 \
  --quest_topk 16 \
  --twi_level2_topp 0.95 \
"

budget_optipns_2048="\
  --tia_level2_topk 2048 \
  --tia_level2_cmp_ratio 4 \
  --quest_topk 32 \
  --twi_level2_topp 0.98 \
"

#   --tia_enable_async_topk \

export CUDA_VISIBLE_DEVICES=${device}

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u -m benchmark.RULER.llm_eval \
  --model_name ${MODEL} \
  --model ${MODELPATH} \
  --method $method \
  ${budget_optipns_512}
