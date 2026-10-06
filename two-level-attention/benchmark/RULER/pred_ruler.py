# #66 RULER 评测适配器：把 KVCache-Factory 预生成的 RULER 数据
# （data/RULER/{4096,8192,16384,32768}/×11 任务×500 条，零外网）喂进
# 本仓库 sparse_attn 管线（TIA/TLI/Quest/FullKV 同一 monkeypatch）。
# 口径对齐官方 RULER：prompt 原样（无 chat template）、max_new=64、
# min_length=context+1 防复读、string_match_all 打分（score_ruler.py）。
# 用法（run_ruler.sh 批量调度）：
#   python -u -m benchmark.RULER.pred_ruler \
#     --model Qwen3-8B --model_path $MODEL \
#     --task niah_single_1 --context_length 8192 --method tli --t $TS \
#     --data-root /home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/RULER \
#     --output-dir exp/results_ruler/Qwen3-8B \
#     --tia_level2_topk 1024 --tia_level2_cmp_ratio 4 --pred_postfix _1024 \
#     --max-num 500
import argparse
import json
import os
import random

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from sparse_attn.arguments import add_sparse_attn_args
from sparse_attn.patches import register_patch
from sparse_attn.metrics import get_metrics
from sparse_attn.info import get_method_name_with_info

RULER_TASKS = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multiquery", "niah_multivalue", "cwe", "fwe", "vt",
]
MAX_GEN = 64  # 官方 RULER 口径（全部任务统一）


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen3-8B")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--task", type=str, choices=RULER_TASKS, required=True)
    parser.add_argument("--context_length", type=int,
                        choices=[4096, 8192, 16384, 32768], required=True)
    parser.add_argument("--data-root", type=str, required=True,
                        help="KVCache-Factory data/RULER 目录")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--t", type=str, default="ruler run")
    parser.add_argument("--pred_postfix", type=str, default="")
    parser.add_argument("--max-num", type=int, default=500,
                        help="每任务样本数（0=全量 500）")
    add_sparse_attn_args(parser)
    return parser.parse_args()


def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    args = parse_args()
    seed_everything()
    data_file = os.path.join(args.data_root, str(args.context_length),
                             f"{args.task}.jsonl")
    rows = [json.loads(l) for l in open(data_file)]
    if args.max_num and len(rows) > args.max_num:
        rows = rows[:args.max_num]  # topk 口径（确定性）
    print(f"[ruler] task={args.task} L={args.context_length} "
          f"n={len(rows)} method={args.method}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto").eval()
    register_patch(model, args)

    eos_ids = [tokenizer.eos_token_id]
    nl = tokenizer.encode("\n", add_special_tokens=False)
    if nl:
        eos_ids.append(nl[-1])

    out_dir = os.path.join(args.output_dir, f"L{args.context_length}",
                           f"pred{args.pred_postfix}")
    os.makedirs(out_dir, exist_ok=True)
    method_name = get_method_name_with_info(args)
    out_path = os.path.join(
        out_dir, f"{args.task}-{method_name}-{args.t}.jsonl")
    fout = open(out_path, "w", encoding="utf-8")

    for row in tqdm(rows):
        torch.cuda.empty_cache()
        input_ids = tokenizer(row["input"], truncation=False,
                              return_tensors="pt").input_ids.to("cuda")
        context_length = input_ids.shape[-1]
        with torch.no_grad():
            output = model(input_ids=input_ids, past_key_values=None,
                           use_cache=True)
            past = output.past_key_values
            pred_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            gen = [pred_idx.item()]
            for _ in range(MAX_GEN - 1):
                # min_length 语义：禁止在 context 结束前停（RULER 官方口径）
                if len(gen) < 2 and pred_idx.item() in eos_ids:
                    pass  # 首位不许停（近似 min_length=context+1）
                out = model(input_ids=pred_idx, past_key_values=past,
                            use_cache=True)
                past = out.past_key_values
                pred_idx = out.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                gen.append(pred_idx.item())
                if pred_idx.item() in eos_ids and len(gen) > 1:
                    break
        pred = tokenizer.decode(gen, skip_special_tokens=True)
        metrics = get_metrics()
        budget = metrics.get_select_tokens()
        metrics.clear()
        fout.write(json.dumps({
            "pred": pred, "answers": row["outputs"],
            "length": row["length"], "budget": budget,
        }, ensure_ascii=False) + "\n")
        fout.flush()
    fout.close()
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
