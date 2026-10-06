# #58 收益区首测：30B（修复后）tli vs triton e2e prefill 计时。
# 32K / 64K 两档（narrativeqa 最长 context），max_new_tokens=1（纯 prefill
# 口径），disable_radix_cache，预热一次吸收 JIT。
# 用法：BACKEND=tli|triton CUDA_VISIBLE_DEVICES=x python3 test_tli_30b_bench.py
import json
import os
import sys
import time
import warnings

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
warnings.filterwarnings("ignore")

BACKEND = os.environ.get("BACKEND", "tli")


def main():
    from sglang import Engine

    MODEL = "/home/wangyuanshuo02/models/Qwen3-30B-A3B"
    eng = Engine(
        model_path=MODEL,
        attention_backend=BACKEND,
        dtype="bfloat16",
        device="cuda",
        tp_size=1,
        mem_fraction_static=0.55,
        trust_remote_code=True,
        disable_radix_cache=True,
        watchdog_timeout=1800,
        cuda_graph_config={"decode": {"backend": "disabled"}, "prefill": {"backend": "disabled"}},
    )
    rows = [json.loads(l) for l in open(
        "/home/wangyuanshuo02/datasets/LongBench/data/narrativeqa.jsonl")]
    ctx_full = sorted(rows, key=lambda r: -len(r.get("context", "")))[0]["context"]
    # 两档：~32K token（85K chars 预估）与 ~64K token（取 top-2 最长拼接近似单文档）
    tiers = {
        "32k": ctx_full[:85000],
        "64k": (ctx_full * 2)[:170000],  # 同文档重复拼接近似 64K（计时口径，非质量口径）
    }
    res = {}
    # 预热（短 prompt，吸收 Triton JIT / MoE 编译）
    eng.generate(["hello"], sampling_params={"temperature": 0.0, "max_new_tokens": 1})
    for name, ctx in tiers.items():
        prompt = ctx + "\n\nSummarize the above text in one sentence:"
        ts = []
        for _ in range(2):
            t0 = time.perf_counter()
            outs = eng.generate([prompt], sampling_params={"temperature": 0.0, "max_new_tokens": 1})
            ts.append(time.perf_counter() - t0)
        n_tok = int(len(prompt) * 0.85)  # 英文近似（Engine 无 tokenizer 属性）
        res[name] = {
            "prefill_s": round(min(ts), 3),
            "both_runs_s": [round(t, 3) for t in ts],
            "prompt_chars": len(prompt),
            "prompt_tokens_est": n_tok,
            "out_preview": outs[0]["text"][:60],
        }
        print(f"[{BACKEND}/{name}] chars={len(prompt)} prefill={res[name]['prefill_s']}s "
              f"out={res[name]['out_preview']!r}", flush=True)
    out_path = f"/tmp/tli_30b_bench_{BACKEND}.json"
    json.dump(res, open(out_path, "w"), indent=1)
    print("saved ->", out_path)
    eng.shutdown()
    print("DONE")


if __name__ == "__main__":
    main()
