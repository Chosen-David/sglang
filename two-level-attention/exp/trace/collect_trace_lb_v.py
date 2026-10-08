# -*- coding: utf-8 -*-
# P0' 逐层参数求解器潜力检查——trace 采集脚本（带 V）。
# 基于 collect_trace_lb.py 改造：hook 中额外抓 value_states（V [S,Hkv,Dv]），
#   存入 layer pt 的 "v" 键（输出失真目标 y−ŷ=(1−mS)(μS̄−μS) 需要 V）。
# 与原版差异：
#   1) 只存 8 代表层 {4,8,12,16,20,24,28,35} + 层 1（对照，零基编号），
#      非 12 个抽样层——单样本存储量远小于全量（V 与 K 同量级，
#      原 27G trace 若翻倍是因为全 36 层；本脚本 9 层 + fp16/bf16 原 dtype
#      不升精度，单样本 ~9 层 × (K+V+Q) ≈ 1.2G/32K 样本量级）。
#   2) 层集/样本数/输出目录参数化（--layers --samples --out）。
#   3) V 抓取：sdpa attention interface 的第 4 个参数（positional value_states
#      或 kwargs），按 shape 与 key_states 的 (S,Hkv) 对齐做防御性校验。
# 用法（海选收官后 GPU 空闲时）：
#   python collect_trace_lb_v.py --out /tmp/trace/qwen3-8b-v \
#       --layers 1,4,8,12,16,20,24,28,35 --samples 2
# 注意：本脚本需要 GPU 跑真实模型前向（transformers）——写好即止，
#   正式运行须等 E109 海选链释放双卡。
import argparse
import json
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B"
LB = "/home/wangyuanshuo02/datasets/LongBench/data"
TARGET_LEN = 32768
N_TAIL = 256
N_ANCHOR = 16
DEFAULT_LAYERS = [1, 4, 8, 12, 16, 20, 24, 28, 35]   # 层 1 = 对照（早期层）
DEFAULT_DATASETS = ["hotpotqa", "narrativeqa", "passage_retrieval_en", "gov_report"]

# LongBench 官方 prompt 模板（与 collect_trace_lb.py 同源）
PROMPTS = {
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, followed by a question. Answer the question based on your understanding of the story. Only give me the answer and do not output any other words.\n\n{context}\n\nQuestion: {input}\nAnswer:",
    "passage_retrieval_en": "The following are 30 paragraphs from Wikipedia, along with an abstract of another Wikipedia article. Your task is to identify which of the 30 paragraphs the abstract is from. Only give me the answer as the number of the paragraph and do not output any other words.\n\n{context}\n\nQuestion: {input}\nAnswer:",
    # 【10-09 修复】gov_report 是摘要类任务：官方 LongBench 把 context 前置
    # 拼接（context + "\n\n" + 模板），不做 {context} 注入——原模板缺占位符
    # 导致 format 后只剩 32 个模板 token（S=32 坏样本已删除重采）
    "gov_report": "{context}\n\nYou are given a report by a government agency. Write a one-page summary of the report.\n\nNow, write a one-page summary of the report.\n\nSummary:",
}

CAPTURE = {"enabled": False, "sink": None}


def _extract_value(key_states, args, kwargs):
    """从 sdpa attention 调用中防御性提取 value_states。

    transformers 4.5x 的 attention interface 签名第 4 位是 value：
    f(module, query_states, key_states, value_states, attention_mask, ...)。
    兼容 positional / kwargs 两种传法；positional 时按 shape 与 key_states
    的 (S, Hkv) 维对齐校验（防 attention_mask 等先入 args 的版本差异）。
    """
    v = kwargs.get("value_states")
    if v is not None:
        return v
    for a in args:
        if (torch.is_tensor(a) and a.dim() == 4
                and a.shape[0] == key_states.shape[0]
                and a.shape[-2] == key_states.shape[-2]
                and a.shape[1] == key_states.shape[1]):
            return a
    return None


def install_hook():
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    orig = ALL_ATTENTION_FUNCTIONS["sdpa"]

    def hooked(module, query_states, key_states, *args, **kwargs):
        if CAPTURE["enabled"] and CAPTURE["sink"] is not None and not module.training:
            value_states = _extract_value(key_states, args, kwargs)
            CAPTURE["sink"](module.layer_idx, query_states, key_states, value_states)
        return orig(module, query_states, key_states, *args, **kwargs)

    ALL_ATTENTION_FUNCTIONS["sdpa"] = hooked


def main():
    ap = argparse.ArgumentParser(
        description="P0' trace 采集（K+Q+V，8 代表层+层1 对照；需 GPU 跑真实前向）")
    ap.add_argument("--out", default="/tmp/trace/qwen3-8b-v",
                    help="输出根目录（每样本一个子目录，含 meta.json + layerNN.pt）")
    ap.add_argument("--layers", default=",".join(map(str, DEFAULT_LAYERS)),
                    help="零基层号逗号分隔（默认 1,4,8,12,16,20,24,28,35）")
    ap.add_argument("--samples", type=int, default=2,
                    help="每数据集取最长 N 条样本（默认 2）")
    ap.add_argument("--datasets", default=",".join(DEFAULT_DATASETS),
                    help="LongBench 子集逗号分隔")
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--target-len", type=int, default=TARGET_LEN,
                    help="token 截断上限（默认 32768，Qwen3-8B 原生 40960 内）")
    ap.add_argument("--device", default="cuda:0")
    ns = ap.parse_args()

    layer_set = {int(x) for x in ns.layers.split(",") if x.strip()}
    datasets = [d.strip() for d in ns.datasets.split(",") if d.strip()]
    os.makedirs(ns.out, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(ns.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        ns.model, dtype=torch.bfloat16, device_map=ns.device,
        trust_remote_code=True, attn_implementation="sdpa").eval()
    install_hook()
    n_layers = model.config.num_hidden_layers

    for ds in datasets:
        tmpl = PROMPTS[ds]
        rows = [json.loads(l) for l in open(f"{LB}/{ds}.jsonl")]
        rows = sorted(rows, key=lambda r: -len(r["context"]))[:ns.samples]
        for si, row in enumerate(rows):
            name = f"lb_{ds}_{si}"
            pdir = os.path.join(ns.out, name)
            if os.path.exists(os.path.join(pdir, "meta.json")):
                print(f"[{name}] exists, skip")
                continue
            os.makedirs(pdir, exist_ok=True)
            text = tmpl.format(context=row["context"], input=row.get("input", ""))
            ids = tokenizer(text, add_special_tokens=False).input_ids[:ns.target_len]
            S = len(ids)
            print(f"[{name}] len={S}", flush=True)
            stored = {}

            def sink(layer_idx, q, k, v):
                # 每层只存一次；V 缺失时显式报错（P0' 输出失真目标必须用 V）
                if layer_idx in stored:
                    return
                if v is None:
                    raise RuntimeError(
                        f"layer{layer_idx}: value_states 抓取失败（hook 未识别 V 参数）"
                        "——请检查 transformers attention interface 签名")
                if layer_idx not in layer_set:
                    return
                q, k, v = q.detach(), k.detach(), v.detach()
                if q.dim() == 4:
                    q = q.squeeze(0)
                    k = k.squeeze(0)
                    v = v.squeeze(0)
                # 存储布局与 collect_trace_lb.py 对齐：k/v [S,Hkv,D]（原 dtype
                # bf16 不升精度，显存/磁盘与 K 同量级）；q [nq,H,D]
                tail_pos = list(range(S - min(N_TAIL, S), S))
                anchor_pos = [int(S * (i + 0.5) / N_ANCHOR) for i in range(N_ANCHOR)]
                qpos = sorted(set(tail_pos + anchor_pos))
                qi = q[:, qpos, :].transpose(0, 1).contiguous()
                stored[layer_idx] = (k.transpose(0, 1).contiguous().cpu(),
                                     v.transpose(0, 1).contiguous().cpu(),  # V 同 K 布局
                                     qi.cpu(), torch.tensor(qpos))
                # 采集期即时下 CPU，GPU 显存不随层累积
                torch.cuda.empty_cache()

            CAPTURE["sink"] = sink
            CAPTURE["enabled"] = True
            with torch.no_grad():
                out = model(torch.tensor([ids], device=ns.device))
            CAPTURE["enabled"] = False
            for layer_idx, (k, v, qi, qpos) in stored.items():
                torch.save({"k": k, "v": v, "q": qi, "qpos": qpos, "S": S},
                           os.path.join(pdir, f"layer{layer_idx:02d}.pt"))
            json.dump({"S": S, "n_layers": n_layers, "prompt": name,
                       "layers": sorted(layer_set),
                       "with_v": True,
                       "question": row.get("input", "")[:200],
                       "answer": str(row.get("answer", ""))[:100]},
                      open(os.path.join(pdir, "meta.json"), "w"))
            print(f"[{name}] saved {len(stored)} layers (with V)", flush=True)
            del stored
            torch.cuda.empty_cache()
            pred = tokenizer.decode(out.logits[0, -1].argmax().item())
            print(f"[{name}] pred token: {pred!r} (answer: {str(row.get('answer',''))[:50]!r})")


if __name__ == "__main__":
    main()
