# E2 trace 采集：Qwen3-8B 真实 attention 前向，捕获每层 RoPE 后的 K 与采样 q
# 用途：H1 位置偏斜 / H2 子空间冗余 / 创新点 B 代表质量 / 跨层复用 / 增量 topk 全部分析
# 约束：本机无 flash-attn，用 sdpa；32K 单卡 prefill（139GB 显存充裕）
import os
import sys
import json
import random
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B"
OUT_DIR = "/tmp/trace/qwen3-8b"
TARGET_LEN = 32768
# 采样 q 位置：最后 256 个连续（测增量 topk 漂移）+ 16 个全程锚点（测跨层/自适应维度）
N_TAIL = 256
N_ANCHOR = 16

# ---------------------------------------------------------------- prompt 构造

ADJ = ("quarrelsome vintage halting fertile nice vitamin collection fee acquisition "
       "driving spend distribute counter woman carotene statement cravat draw").split()
NOUN = ("ratepayer strive specify yell venture force-pathogenesis union visitor "
        "surface arm sake moustache bed alpenhorn peaceful trigger elf").split()

def gen_needle_prompt(tokenizer, target_len, seed=0):
    """register/needle 任务：导师 example_data.txt 格式，行级 needle 分布全程"""
    rng = random.Random(seed)
    lines, n_tok = [], 0
    n = 0
    while n_tok < target_len - 512:
        n += 1
        line = f"line {rng.choice(ADJ)}-{rng.choice(NOUN)}: REGISTER_CONTENT is <{rng.randint(1000, 99999)}>\n"
        lines.append(line)
        n_tok += 10  # 粗略估算，后面精调
    prompt = "".join(lines)
    ids = tokenizer(prompt, add_special_tokens=False).input_ids
    # 精确截断到 target_len - question
    ids = ids[: target_len - 128]
    # 问题指向 75% 深度的某行（保证有远端 needle hit）
    question = "\nWhat is the REGISTER_CONTENT of line " + lines[len(lines) * 3 // 4].split("line ")[1].split(":")[0] + "? Answer with the number only."
    q_ids = tokenizer(question, add_special_tokens=False).input_ids
    return ids, q_ids

def gen_natural_prompt(tokenizer, target_len):
    """自然文本：拼接本机真实 md/py 文件（模拟 summarization 型 broad attention）"""
    import glob
    files = []
    files += sorted(glob.glob("/home/wangyuanshuo02/sglang/*.md"))
    files += sorted(glob.glob("/home/wangyuanshuo02/CUDA-Triton-Learn/KMeans/CPU/*.md"))
    files += sorted(glob.glob("/home/wangyuanshuo02/sglang/python/sglang/srt/layers/attention/qsa/*.py"))
    files += sorted(glob.glob("/home/wangyuanshuo02/two-level-attention/sparse_attn/indexer/*.py"))
    chunks = []
    n_tok = 0
    for f in files:
        try:
            text = open(f, errors="ignore").read()
        except Exception:
            continue
        chunks.append(f"\n\n===== FILE: {os.path.basename(f)} =====\n\n" + text)
    body = "".join(chunks)
    ids = tokenizer(body, add_special_tokens=False).input_ids
    ids = ids[: target_len - 128]
    question = "\n\nSummarize the key technical points of the document above in one paragraph."
    q_ids = tokenizer(question, add_special_tokens=False).input_ids
    return ids, q_ids

# ---------------------------------------------------------------- hook 捕获

CAPTURE = {"enabled": False, "sink": None}

def install_hook():
    """包装 sdpa attention 函数：捕获 (layer_idx, q[B,H,S,D], k[B,Hkv,S,D]) 到 sink"""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    orig = ALL_ATTENTION_FUNCTIONS["sdpa"]

    def hooked(module, query_states, key_states, *args, **kwargs):
        if CAPTURE["enabled"] and CAPTURE["sink"] is not None and not module.training:
            CAPTURE["sink"](module.layer_idx, query_states, key_states)
        return orig(module, query_states, key_states, *args, **kwargs)

    ALL_ATTENTION_FUNCTIONS["sdpa"] = hooked

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="cuda:0",
        trust_remote_code=True, attn_implementation="sdpa",
    ).eval()
    install_hook()
    n_layers = model.config.num_hidden_layers

    prompts = {
        "needle32k": gen_needle_prompt(tokenizer, TARGET_LEN, seed=0),
        "natural32k": gen_natural_prompt(tokenizer, TARGET_LEN),
    }

    for name, (body_ids, q_ids) in prompts.items():
        pdir = os.path.join(OUT_DIR, name)
        os.makedirs(pdir, exist_ok=True)
        input_ids = body_ids + q_ids
        S = len(input_ids)
        print(f"[{name}] total len = {S}")

        stored = {}
        def sink(layer_idx, q, k):
            # q: [B, H, S, D] k: [B, Hkv, S, D]（RoPE 后，sdpa 路径）
            if layer_idx in stored:
                return
            q = q.detach()
            k = k.detach()
            if q.dim() == 4:
                q = q.squeeze(0)  # [H, S, D]
                k = k.squeeze(0)  # [Hkv, S, D]
            # 采样 q 位置
            tail_pos = list(range(S - N_TAIL, S))
            anchor_pos = [int(S * (i + 0.5) / N_ANCHOR) for i in range(N_ANCHOR)]
            qpos = sorted(set(tail_pos + anchor_pos))
            qi = q[:, qpos, :].transpose(0, 1).contiguous()  # [nq, H, D]
            stored[layer_idx] = (k.transpose(0, 1).contiguous().cpu(), qi.cpu(), torch.tensor(qpos))
            if layer_idx == n_layers - 1 or layer_idx % 8 == 0:
                print(f"  captured layer {layer_idx}: k={tuple(stored[layer_idx][0].shape)}")

        CAPTURE["sink"] = sink
        CAPTURE["enabled"] = True
        with torch.no_grad():
            out = model(torch.tensor([input_ids], device="cuda:0"))
        CAPTURE["enabled"] = False

        for layer_idx, (k, qi, qpos) in stored.items():
            torch.save({"k": k, "q": qi, "qpos": qpos, "S": S},
                       os.path.join(pdir, f"layer{layer_idx:02d}.pt"))
        meta = {"S": S, "n_layers": n_layers,
                "num_heads": model.config.num_attention_heads,
                "num_kv_heads": model.config.num_key_value_heads,
                "head_dim": model.config.head_dim,
                "prompt": name}
        json.dump(meta, open(os.path.join(pdir, "meta.json"), "w"))
        print(f"[{name}] saved {len(stored)} layers -> {pdir}")
        del stored
        torch.cuda.empty_cache()
        # 打印模型对 needle 的回答（sanity check）
        pred = tokenizer.decode(out.logits[0, -1].argmax().item())
        print(f"[{name}] next token prediction: {pred!r}")

if __name__ == "__main__":
    main()
