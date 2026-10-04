# 审稿 C3：三臂（triton dense / tli / quest）单请求 e2e 延迟对照。
# 32K / 64K 两档（narrativeqa 最长 context），带 decode 步（max_new=64），
# prefill 与 decode 分开计时（两遍法：t(max_new=1) = prefill；
# t(max_new=64) - t(max_new=1) = 63 步 decode）。
# disable_radix_cache；预热一次吸收 JIT / MoE 编译。
# 用法（GPU 空闲后逐臂跑）：
#   SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=true SGLANG_IS_FLASHINFER_AVAILABLE=false \
#   SGLANG_ENABLE_JIT_DEEPGEMM=0 CUDA_VISIBLE_DEVICES=0 \
#   PYTHONPATH=/home/wangyuanshuo02/.local/pylibs_tf512:/home/wangyuanshuo02/sglang/python \
#   BACKEND=quest python3 test_c3_3arm_bench.py
#   # BACKEND=triton / BACKEND=tli 同法（已有参考：64K dense 21.061s / TLI 16.386s
#   # 为纯 prefill 口径；本脚本新口径两遍法数字与旧口径 prefill 列可对账）
import json
import os
import sys
import time
import warnings

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
warnings.filterwarnings("ignore")

BACKEND = os.environ.get("BACKEND", "quest")
N_DECODE = 64


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
        watchdog_timeout=3600,
        cuda_graph_config={"decode": {"backend": "disabled"},
                           "prefill": {"backend": "disabled"}},
    )
    rows = [json.loads(l) for l in open(
        "/home/wangyuanshuo02/datasets/LongBench/data/narrativeqa.jsonl")]
    ctx_full = sorted(rows, key=lambda r: -len(r.get("context", "")))[0]["context"]
    tiers = {
        "32k": ctx_full[:85000],
        "64k": (ctx_full * 2)[:170000],  # 同文档重复拼接近似 64K（计时口径）
    }
    res = {"backend": BACKEND, "n_decode": N_DECODE}
    # 预热（短 prompt，吸收 Triton JIT / MoE 编译 / quest 首次建池）
    eng.generate(["hello"], sampling_params={"temperature": 0.0, "max_new_tokens": 1})
    for name, ctx in tiers.items():
        prompt = ctx + "\n\nSummarize the above text in one sentence:"
        # 两遍法：pass1 = 纯 prefill（max_new=1）；pass2 = prefill+63 步 decode
        ts1, ts2 = [], []
        for _ in range(2):
            t0 = time.perf_counter()
            eng.generate([prompt],
                         sampling_params={"temperature": 0.0, "max_new_tokens": 1})
            ts1.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            outs = eng.generate(
                [prompt],
                sampling_params={"temperature": 0.0, "max_new_tokens": N_DECODE})
            ts2.append(time.perf_counter() - t0)
        prefill_s = min(ts1)
        total_s = min(ts2)
        decode_s = max(total_s - prefill_s, 0.0)
        res[name] = {
            "prefill_s": round(prefill_s, 3),
            "total_s": round(total_s, 3),
            "decode_s": round(decode_s, 3),
            "decode_ms_per_step": round(decode_s / (N_DECODE - 1) * 1000, 2),
            "pass1_runs_s": [round(t, 3) for t in ts1],
            "pass2_runs_s": [round(t, 3) for t in ts2],
            "prompt_chars": len(prompt),
            "out_preview": outs[0]["text"][:60],
        }
        print(f"[{BACKEND}/{name}] prefill={prefill_s:.3f}s "
              f"total={total_s:.3f}s decode={decode_s:.3f}s "
              f"({res[name]['decode_ms_per_step']}ms/step) "
              f"out={res[name]['out_preview']!r}", flush=True)
    out_path = f"/tmp/c3_3arm_{BACKEND}.json"
    json.dump(res, open(out_path, "w"), indent=1, ensure_ascii=False)
    print("saved ->", out_path)
    eng.shutdown()
    print("DONE")


if __name__ == "__main__":
    main()
