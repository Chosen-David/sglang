cd /home/hadoop-friday-llm/VSCodeProjects/sparse-attn-dev_huyuxuan09

MODEL=Qwen3-14B
MODELPATH=hf_models/downloads/Qwen/Qwen3-14B

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
  --twi_level2_topp 0.92 \
  --pred_postfix "_512"
"

budget_optipns_1024="\
  --tia_level2_topk 1024 \
  --tia_level2_cmp_ratio 4 \
  --quest_topk 16 \
  --twi_level2_topp 0.95 \
  --pred_postfix "_1024"
"

budget_optipns_2048="\
  --tia_level2_topk 2048 \
  --tia_level2_cmp_ratio 4 \
  --quest_topk 32 \
  --twi_level2_topp 0.98 \
  --pred_postfix "_2048"
"

#   --tia_enable_async_topk \

export CUDA_VISIBLE_DEVICES=${device}

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Whether enable LongBench-E benchmark
enable_e=0

# ignore: vcsum multifieldqa_zh lsht passage_retrieval_zh dureader samsum passage_count trec
datasets=('narrativeqa' 'qasper' 'multifieldqa_en' 'hotpotqa'
          '2wikimqa' 'musique' 'gov_report' 'qmsum' 'multi_news'
          'triviaqa' 'passage_retrieval_en' 'lcc' 'repobench-p')

dataset_e=('qasper' 'multifieldqa_en' 'hotpotqa' '2wikimqa' 'gov_report'
           'multi_news' 'trec' 'triviaqa' 'passage_count'
           'passage_retrieval_en' 'lcc' 'repobench-p')

if [ "$enable_e" -eq 0 ]; then
  benchmark_dataset=("${datasets[@]}")
else
  benchmark_dataset=("${dataset_e[@]}")
fi

month=$(date +"%m")
day=$(date +"%d")
hour=$(date +"%H")
minute=$(date +"%M")
time="${month}${day}${hour}${minute}"

for task in "${benchmark_dataset[@]}"
do
    echo "Running $method $task:"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u -m benchmark.LongBench.pred \
        --model $MODEL --model_path $MODELPATH --task $task \
        --method $method \
        --e $enable_e --t $time \
        --dataset-path $DATASET_PATH \
        --output-dir $OUTPUT_DIR \
        --config-path $CONFIG_PATH \
        ${budget_optipns_512}
done

# for task in "${benchmark_dataset[@]}"
# do
#     echo "Running $method $task:"
#     PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u -m benchmark.LongBench.pred \
#         --model $MODEL --model_path $MODELPATH --task $task \
#         --method $method \
#         --e $enable_e --t $time \
#         --dataset-path $DATASET_PATH \
#         --output-dir $OUTPUT_DIR \
#         --config-path $CONFIG_PATH \
#         ${budget_optipns_1024}
# done

for task in "${benchmark_dataset[@]}"
do
    echo "Running $method $task:"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u -m benchmark.LongBench.pred \
        --model $MODEL --model_path $MODELPATH --task $task \
        --method $method \
        --e $enable_e --t $time \
        --dataset-path $DATASET_PATH \
        --output-dir $OUTPUT_DIR \
        --config-path $CONFIG_PATH \
        ${budget_optipns_2048}
done
