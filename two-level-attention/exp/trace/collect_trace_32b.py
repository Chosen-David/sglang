# Qwen3-32B 泛化复验 trace 采集（A/D' 层掩码 + far 总量 gate 判据）
# 7 任务 × 2 最长样本（4 校准口径任务 + 3 多跳 gate 任务），32K 截断
# 与 collect_trace_lb.py / collect_trace_task.py 同语义：hook sdpa 存 K + 尾部/锚点 q
import os
import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/model/Qwen3-32B"
OUT_DIR = "/tmp/trace/qwen3-32b"
LB = "/home/wangyuanshuo02/datasets/LongBench/data"
CONF = "/home/wangyuanshuo02/two-level-attention/benchmark/LongBench/config/dataset2prompt.json"
TARGET_LEN = 32768
N_TAIL = 256
N_ANCHOR = 16

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
    # 4 个 E6 校准口径任务 + 3 个 gate 多跳任务（musique/qasper/multifieldqa_en）
    tasks = sys.argv[1:] or [
        "hotpotqa", "narrativeqa", "passage_retrieval_en", "gov_report",
        "musique", "qasper", "multifieldqa_en",
    ]
    templates = json.load(open(CONF))
    os.makedirs(OUT_DIR, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="cuda:0",
        trust_remote_code=True, attn_implementation="sdpa").eval()
    install_hook()
    n_layers = model.config.num_hidden_layers
    print(f"model loaded: {n_layers} layers, kv_heads={model.config.num_key_value_heads}", flush=True)

    for ds in tasks:
        tmpl = templates[ds]
        rows = [json.loads(l) for l in open(f"{LB}/{ds}.jsonl")]
        # 去重 context：narrativeqa 同一文档多问题，截断后 prompt 相同 →
        # trace 完全重复（8B 时代 e6_layer_skip.json 的 narrativeqa_0/1 far
        # 完全相同即此坑）。取 context 互不相同的最长 2 条
        seen, uniq = set(), []
        for r in sorted(rows, key=lambda r: -len(r["context"])):
            key = r["context"][:10000]
            if key in seen:
                continue
            seen.add(key)
            uniq.append(r)
            if len(uniq) == 2:
                break
        rows = uniq
        for si, row in enumerate(rows):
            name = f"lb_{ds}_{si}"
            pdir = os.path.join(OUT_DIR, name)
            if os.path.exists(os.path.join(pdir, "meta.json")):
                print(f"[{name}] exists, skip", flush=True)
                continue
            os.makedirs(pdir, exist_ok=True)
            text = tmpl.format(context=row["context"], input=row.get("input", ""))
            ids = tokenizer(text, add_special_tokens=False).input_ids[:TARGET_LEN]
            S = len(ids)
            print(f"[{name}] len={S}", flush=True)
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
            print(f"[{name}] saved {len(stored)} layers", flush=True)
            del stored
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
