# 真实 LongBench 样本 trace 采集（学术口径对齐：hotpotqa/narrativeqa/passage_retrieval_en/gov_report）
# 每个 LongBench 子集取 2 条最长样本，32K 截断（Qwen3-8B 上限 40960）
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B"
OUT_DIR = "/tmp/trace/qwen3-8b"
LB = "/home/wangyuanshuo02/datasets/LongBench/data"
TARGET_LEN = 32768
N_TAIL = 256
N_ANCHOR = 16

# LongBench 官方 prompt 模板（与导师 pipeline 同源）
PROMPTS = {
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, followed by a question. Answer the question based on your understanding of the story. Only give me the answer and do not output any other words.\n\n{context}\n\nQuestion: {input}\nAnswer:",
    "passage_retrieval_en": "The following are 30 paragraphs from Wikipedia, along with an abstract of another Wikipedia article. Your task is to identify which of the 30 paragraphs the abstract is from. Only give me the answer as the number of the paragraph and do not output any other words.\n\n{context}\n\nQuestion: {input}\nAnswer:",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
}

CAPTURE = {"enabled": False, "sink": None}

def install_hook():
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
        trust_remote_code=True, attn_implementation="sdpa").eval()
    install_hook()
    n_layers = model.config.num_hidden_layers

    for ds, tmpl in PROMPTS.items():
        rows = [json.loads(l) for l in open(f"{LB}/{ds}.jsonl")]
        rows = sorted(rows, key=lambda r: -len(r["context"]))[:2]
        for si, row in enumerate(rows):
            name = f"lb_{ds}_{si}"
            pdir = os.path.join(OUT_DIR, name)
            if os.path.exists(os.path.join(pdir, "meta.json")):
                print(f"[{name}] exists, skip")
                continue
            os.makedirs(pdir, exist_ok=True)
            text = tmpl.format(context=row["context"], input=row.get("input", ""))
            ids = tokenizer(text, add_special_tokens=False).input_ids[:TARGET_LEN]
            S = len(ids)
            print(f"[{name}] len={S}")
            stored = {}

            def sink(layer_idx, q, k):
                if layer_idx in stored:
                    return
                q = q.detach()
                k = k.detach()
                if q.dim() == 4:
                    q = q.squeeze(0)
                    k = k.squeeze(0)
                tail_pos = list(range(S - min(N_TAIL, S), S))
                anchor_pos = [int(S * (i + 0.5) / N_ANCHOR) for i in range(N_ANCHOR)]
                qpos = sorted(set(tail_pos + anchor_pos))
                qi = q[:, qpos, :].transpose(0, 1).contiguous()
                stored[layer_idx] = (k.transpose(0, 1).contiguous().cpu(),
                                     qi.cpu(), torch.tensor(qpos))

            CAPTURE["sink"] = sink
            CAPTURE["enabled"] = True
            with torch.no_grad():
                out = model(torch.tensor([ids], device="cuda:0"))
            CAPTURE["enabled"] = False
            for layer_idx, (k, qi, qpos) in stored.items():
                torch.save({"k": k, "q": qi, "qpos": qpos, "S": S},
                           os.path.join(pdir, f"layer{layer_idx:02d}.pt"))
            json.dump({"S": S, "n_layers": n_layers, "prompt": name,
                       "question": row.get("input", "")[:200],
                       "answer": str(row.get("answer", ""))[:100]},
                      open(os.path.join(pdir, "meta.json"), "w"))
            print(f"[{name}] saved {len(stored)} layers")
            del stored
            torch.cuda.empty_cache()
            pred = tokenizer.decode(out.logits[0, -1].argmax().item())
            print(f"[{name}] pred token: {pred!r} (answer: {str(row.get('answer',''))[:50]!r})")

if __name__ == "__main__":
    main()
