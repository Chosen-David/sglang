cd /home/hadoop-friday-llm/VSCodeProjects/sparse-attn-dev_huyuxuan09

MODEL=Qwen3-8B
MODELPATH=hf_models/downloads/Qwen/Qwen3-8B

OUTPUT_DIR=exp/results_longbench/$MODEL

# Use HF load_dataset. Modify this if you want to use local data set.
DATASET_PATH="offline_data/LongBench/data"

CONFIG_PATH=benchmark/LongBench/config/

mkdir -p $OUTPUT_DIR

# Whether enable LongBench-E benchmark
enable_e=0

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

python -u -m benchmark.LongBench.eval --model $MODEL --e $enable_e --t $time --output-dir $OUTPUT_DIR
