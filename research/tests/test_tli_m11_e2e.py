# M11 e2e AB：SGLANG_TLI_PREFILL_KERNEL=1（Triton fused）vs 0（eager）
# 输出一致性 + prefill 计时（8B，narrativeqa 38.5K token 真实长文）。
# 用法：CUDA_VISIBLE_DEVICES=0 SGLANG_TLI_PREFILL_KERNEL=1 python3 test_tli_m11_e2e.py
import json
import os
import sys
import time
import warnings

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
warnings.filterwarnings("ignore")

MODE = os.environ.get("SGLANG_TLI_PREFILL_KERNEL", "1")


def main():
    from sglang import Engine

    MODEL = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B"
    eng = Engine(
        model_path=MODEL,
        attention_backend="tli",
        dtype="bfloat16",
        device="cuda",
        tp_size=1,
        mem_fraction_static=0.7,
        trust_remote_code=True,
        disable_radix_cache=True,
        watchdog_timeout=1800,
        cuda_graph_config={"decode": {"backend": "disabled"}, "prefill": {"backend": "disabled"}},
    )
    rows = [json.loads(l) for l in open(
        "/home/wangyuanshuo02/datasets/LongBench/data/narrativeqa.jsonl")]
    ctx = sorted(rows, key=lambda r: -len(r.get("context", "")))[2]["context"][:85000]
    prompt = ctx + "\n\nSummarize the above text in one sentence:"
    eng.generate(["hello"], sampling_params={"temperature": 0.0, "max_new_tokens": 1})
    t0 = time.perf_counter()
    outs = eng.generate([prompt], sampling_params={"temperature": 0.0, "max_new_tokens": 32})
    total = time.perf_counter() - t0
    text = outs[0]["text"]
    print(f"[m11-e2e kernel={MODE}] total={total:.2f}s")
    print(f"[m11-out] {text[:200]!r}")
    json.dump({"mode": MODE, "total_s": round(total, 2), "text": text},
              open(f"/tmp/tli_m11_e2e_{MODE}.json", "w"), ensure_ascii=False, indent=1)
    eng.shutdown()
    print("DONE")


if __name__ == "__main__":
    main()
