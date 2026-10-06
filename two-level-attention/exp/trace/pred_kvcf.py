# E81：KVCache-Factory 方法（SnapKV/H2O/PyramidKV）LongBench 统一口径 runner
# 口径与 pred.py（E71 主表）完全一致：同数据/同模板/同中段截断/同 question-stage
# 模拟 decode/同 greedy——唯一差异 = KV cache 压缩方法。
# 用法：
#   python -u exp/trace/pred_kvcf.py --task hotpotqa --method snapkv \
#     --max-capacity 1024 --n 8            # 冒烟
#   python -u exp/trace/pred_kvcf.py --task hotpotqa --method snapkv --max-capacity 1024   # 全量 200
# 落盘：exp/results_longbench/Qwen3-8B/pred_kvcf/{method}/{task}-{method}-{t}.jsonl
# 打分：run_e71_eval.py 的 METHODS 加对应行（pattern "{task}-{method}-*.jsonl"）
import argparse
import glob
import json
import os
import random
import sys
import time

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = "/home/wangyuanshuo02/two-level-attention"
sys.path.insert(0, ROOT)
sys.path.insert(0, f"{ROOT}/exp/trace")

from kvcf_qwen3 import replace_qwen3  # noqa: E402

ATTN_IMPL = "sdpa"

MODEL_PATH = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B"
DATA = "/home/wangyuanshuo02/datasets/LongBench/data"
CONFIG = f"{ROOT}/benchmark/LongBench/config"
MAX_LENGTH = 31500  # model2maxlen.json Qwen3-8B
OUT_BASE = f"{ROOT}/exp/results_longbench/Qwen3-8B/pred_kvcf"


def build_chat(prompt):
    # pred.py 的 Qwen3 分支（thinking 模板，与主表口径一致）
    return (f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            f"<think>\n\n</think>\n\n")


NO_CHAT = ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]


def get_pred(model, tokenizer, data, max_gen, prompt_format, dataset, method,
             prefill_full=True):
    preds = []
    for json_obj in tqdm(data):
        torch.cuda.empty_cache()
        prompt = prompt_format.format(**json_obj)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > MAX_LENGTH:
            half = int(MAX_LENGTH / 2)
            prompt = (tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)
                      + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True))
        if dataset not in NO_CHAT:
            prompt = build_chat(prompt)

        if prefill_full:
            # SnapKV/H2O/PyramidKV 论文口径：完整 prompt（含 question）一次
            # prefill 压缩——静态压缩方法的设计假设是 prefill 时已见 question。
            # （Quest 口径的 question 拆分对静态压缩系统性不利：question 从未
            # 参与压缩选择，相关段落被压掉后 query 无法找回。）
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to("cuda")
            with torch.no_grad():
                output = model(input_ids=input.input_ids, past_key_values=None,
                               use_cache=True)
                past_key_values = output.past_key_values
        else:
            # question-stage 模拟（pred.py 同款拆分，Quest 口径）
            if dataset in ["qasper", "hotpotqa"]:
                q_pos = prompt.rfind("Question:")
            elif dataset in ["multifieldqa_en", "gov_report"]:
                q_pos = prompt.rfind("Now,")
            elif dataset == "triviaqa":
                q_pos = prompt.rfind("Answer the question")
            elif dataset == "narrativeqa":
                q_pos = prompt.rfind("Do not provide")
            else:
                q_pos = -1
            q_pos = max(len(prompt) - 100, q_pos)
            question = prompt[q_pos:]
            prompt = prompt[:q_pos]

            input = tokenizer(prompt, truncation=False, return_tensors="pt").to("cuda")
            q_input = tokenizer(question, truncation=False, return_tensors="pt").to("cuda")
            q_input.input_ids = q_input.input_ids[:, 1:]

            with torch.no_grad():
                output = model(input_ids=input.input_ids, past_key_values=None, use_cache=True)
                past_key_values = output.past_key_values
                for input_id in q_input.input_ids[0]:
                    output = model(input_ids=input_id.unsqueeze(0).unsqueeze(0),
                                   past_key_values=past_key_values, use_cache=True)
                    past_key_values = output.past_key_values

        with torch.no_grad():
            pred_token_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            generated_content = [pred_token_idx.item()]
            for _ in range(max_gen - 1):
                outputs = model(input_ids=pred_token_idx,
                                past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
                pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                generated_content += [pred_token_idx.item()]
                if pred_token_idx.item() == tokenizer.eos_token_id:
                    break

        preds.append({
            "pred": tokenizer.decode(generated_content, skip_special_tokens=True),
            "answers": json_obj["answers"],
            "all_classes": json_obj["all_classes"],
            "length": json_obj["length"],
        })
    return preds


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--method", required=True,
                   choices=["snapkv", "h2o", "pyramidkv", "streamingllm"])
    ap.add_argument("--max-capacity", type=int, default=1024)
    ap.add_argument("--window-size", type=int, default=8)
    ap.add_argument("--kernel-size", type=int, default=7)
    ap.add_argument("--pooling", default="maxpool")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--attn-impl", default="sdpa", dest="attn_impl")
    ap.add_argument("--split-question", action="store_true", help="Quest 口径拆分 question（默认完整 prefill）")
    ap.add_argument("--sink-guard", type=int, default=0,
                    help="M6 sink 保送适配：压缩后 cat 回 prefill 首 N token（0=关闭，PSI 严格口径对齐）")
    ap.add_argument("--output-suffix", default="",
                    help="输出文件后缀（sink-guard 臂目录隔离，避免覆盖原 baseline 文件）")
    args = ap.parse_args()

    global ATTN_IMPL
    ATTN_IMPL = args.attn_impl
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    seed_everything(42)

    dataset2prompt = json.load(open(f"{CONFIG}/dataset2prompt.json"))
    dataset2maxlen = json.load(open(f"{CONFIG}/dataset2maxlen.json"))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map="auto", attn_implementation=ATTN_IMPL).eval()

    replace_qwen3(model, args.method, max_capacity=args.max_capacity,
                  window_size=args.window_size, kernel_size=args.kernel_size,
                  pooling=args.pooling, sink_guard=args.sink_guard)

    task_file = f"{DATA}/{args.task}.jsonl"
    data = [json.loads(l) for l in open(task_file)][: args.n]

    out_dir = f"{OUT_BASE}/{args.method}{args.output_suffix}"
    os.makedirs(out_dir, exist_ok=True)
    # M6 双卡并行加速：若同任务已有完整 n 行输出（另一卡预跑完成），直接跳过。
    # 注意冒烟文件（n=20）行数 < 全量 200 不会误触发跳过。
    for old in sorted(glob.glob(f"{out_dir}/{args.task}-{args.method}{args.output_suffix}-*.jsonl")):
        try:
            n_done = sum(1 for _ in open(old))
        except OSError:
            continue
        if n_done >= args.n:
            print(f"SKIP {args.task} (已有完整 {n_done} 行: {old})")
            return
    t = time.strftime("%H%M%S")
    out_fn = f"{out_dir}/{args.task}-{args.method}{args.output_suffix}-{t}.jsonl"
    fout = open(out_fn, "w")
    preds = get_pred(model, tokenizer, data, dataset2maxlen[args.task],
                     dataset2prompt[args.task], args.task, args.method,
                     prefill_full=not args.split_question)
    for p in preds:
        fout.write(json.dumps(p) + "\n")
    fout.close()
    print(f"DONE saved {len(preds)} preds -> {out_fn}")


if __name__ == "__main__":
    main()
