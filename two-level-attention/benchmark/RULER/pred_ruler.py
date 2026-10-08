# #66 RULER 评测适配器：把 KVCache-Factory 预生成的 RULER 数据
# （data/RULER/{4096,8192,16384,32768}/×11 任务×100 条，零外网；65536/131072
# 由 gen_ruler_long.py 循环体扩展生成）喂进本仓库 sparse_attn 管线
# （TIA/TLI/Quest/FullKV 同一 monkeypatch）。
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
# ---- E109 三件套 #3：YaRN 长上下文扩展（2026-10-09）----
# Qwen3-8B 原生 max_position_embeddings=40960，65536/131072 档须 YaRN。
# 配方按 Qwen3 官方 model card：rope_scaling={yarn, factor, original 40960,
# beta_fast 32, beta_slow 1}；128K 档 factor=4.0（官方原配方，effective
# 163840），64K 档取半档 factor=2.0（effective 81920，留 25% 余量；与官方
# 整档倍率风格一致，且 ramp 插值更平缓、近程精度损失更小）。
# 设计文档口径（逐层参数求解器 §10.4）：dense 与所有稀疏方法同一位置缩放/
# tokenizer/最大输出；先报告 dense 长度能力再看稀疏退化。
# YaRN 对 TLI 索引透明：indexer 消费的是 post-RoPE 的 K（qwen3_attn_patch
# 在 apply_rotary_pos_emb + cache.update 之后传入），位置缩放自动生效。
# logits 显存账（超 32K 档自动 logits_to_keep=1，等价性审计 0233 已 A/B
# 实测 bitwise 相等）：131072×151936 vocab bf16 全序列 logits ≈ 39.8GB 纯
# 浪费（只消费末 token），32K 及以下档路径逐位不变（零扰动纪律）。
import argparse
import json
import os
import random

import numpy as np
import torch
from tqdm import tqdm
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer)

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
# Qwen3-8B 原生上下文上限（config.max_position_embeddings）
QWEN3_NATIVE_MPE = 40960
# YaRN factor 自动档位（官方 128K 配方 factor 4.0 的等比半档）
YARN_FACTOR_AUTO = {65536: 2.0, 131072: 4.0}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen3-8B")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--task", type=str, choices=RULER_TASKS, required=True)
    parser.add_argument("--context_length", type=int,
                        choices=[4096, 8192, 16384, 32768, 65536, 131072],
                        required=True)
    parser.add_argument("--data-root", type=str, required=True,
                        help="KVCache-Factory data/RULER 目录")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--t", type=str, default="ruler run")
    parser.add_argument("--pred_postfix", type=str, default="")
    parser.add_argument("--max-num", type=int, default=500,
                        help="每任务样本数（0=全量）")
    parser.add_argument("--yarn", action="store_true",
                        help="启用 YaRN 位置扩展（Qwen3 官方配方；"
                             "65536/131072 档必开，32768 及以下档保持关闭"
                             "以维持既有口径零扰动）")
    parser.add_argument("--yarn_factor", type=float, default=None,
                        help="YaRN factor（默认按档位自动：65536→2.0、"
                             "131072→4.0；显式传入可覆盖做消融）")
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

    # ---- YaRN 校验与配置（E109 三件套 #3）----
    # 超过原生 40960 的档位必须显式 --yarn（防误用原生 RoPE 跑超限位置）；
    # 原生档位默认关闭（既有 4K-32K 路径零扰动），显式开也允许（消融用）。
    use_yarn = args.yarn
    if args.context_length > QWEN3_NATIVE_MPE and not use_yarn:
        raise SystemExit(
            f"[yarn] context_length={args.context_length} 超出 Qwen3-8B 原生 "
            f"{QWEN3_NATIVE_MPE}，必须加 --yarn（run_ruler_e109.sh 已自动）")
    if use_yarn:
        factor = args.yarn_factor or YARN_FACTOR_AUTO.get(
            args.context_length, 4.0)
        print(f"[yarn] enabled: factor={factor} "
              f"(effective context = {int(QWEN3_NATIVE_MPE * factor)})")

    data_file = os.path.join(args.data_root, str(args.context_length),
                             f"{args.task}.jsonl")
    rows = [json.loads(l) for l in open(data_file)]
    if args.max_num and len(rows) > args.max_num:
        rows = rows[:args.max_num]  # topk 口径（确定性）
    print(f"[ruler] task={args.task} L={args.context_length} "
          f"n={len(rows)} method={args.method}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True)
    # YaRN：在模型加载前把 rope_scaling 写进 config（transformers 从
    # config 构造 YarnRotaryEmbedding；同时写 rope_type/type 兼容新旧字段）
    config = AutoConfig.from_pretrained(args.model_path,
                                        trust_remote_code=True)
    if use_yarn:
        factor = args.yarn_factor or YARN_FACTOR_AUTO.get(
            args.context_length, 4.0)
        config.rope_scaling = {
            "rope_type": "yarn", "type": "yarn",
            "factor": factor,
            "original_max_position_embeddings": QWEN3_NATIVE_MPE,
            "beta_fast": 32, "beta_slow": 1,
        }
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, config=config, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto").eval()
    register_patch(model, args)

    # 超原生档位启用 logits_to_keep=1：prefill 只回传末 token logits
    # （131072×151936 bf16 全序列 logits ≈ 39.8GB 纯浪费；等价性由审计
    # 0233 A/B 实测 bitwise 相等保证；≤32K 档不启用，路径逐位不变）
    keep_logits = args.context_length > 32768

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
                           use_cache=True,
                           **({"logits_to_keep": 1} if keep_logits else {}))
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
