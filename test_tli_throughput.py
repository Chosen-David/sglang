# M3-a: e2e 吞吐基线实测（tli vs triton backend，Qwen3-8B 真实长上下文）
# 方法：
#   prefill 时间 = generate(max_new_tokens=1) 的墙钟时间
#   decode 吞吐 = N / (generate(max_new_tokens=1+N) - prefill 时间)，N=64
# 上下文：真实 LongBench narrativeqa 样本（最长 210K chars）按字符截断出
#   ~8K / 32K / 128K chars 三档（英文 ≈ 4 chars/token → 约 2K / 8K / 32K token）
# 运行：cd /home/wangyuanshuo02/sglang && \
#   SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=true SGLANG_IS_FLASHINFER_AVAILABLE=false \
#   SGLANG_ENABLE_JIT_DEEPGEMM=0 CUDA_VISIBLE_DEVICES=1 \
#   PYTHONPATH=/home/wangyuanshuo02/.local/pylibs_tf512:/home/wangyuanshuo02/sglang/python \
#   python test_tli_throughput.py
import json
import sys
import time
import warnings

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
warnings.filterwarnings("ignore")

from sglang import Engine

MODEL = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B"
DATA = "/home/wangyuanshuo02/datasets/LongBench/data/narrativeqa.jsonl"
CHAR_LENS = [8000, 32000, 120000]  # Qwen3-8B context 40960 token 上限 → ≤~123K chars
N_DECODE = 64
OUT_JSON = "/home/wangyuanshuo02/sglang/tli_throughput_results.json"


def load_contexts():
    rows = [json.loads(l) for l in open(DATA)]
    # 取最长样本（同 context 去重口径与 32B 采集一致）
    rows.sort(key=lambda r: -len(r["context"]))
    ctx = rows[0]["context"]
    out = []
    for cl in CHAR_LENS:
        p = (
            ctx[:cl]
            + "\n\nSummarize the above text in one sentence:"
        )
        out.append((cl, p))
    return out


def run_backend(backend, prompts):
    eng = Engine(
        model_path=MODEL,
        attention_backend=backend,
        dtype="bfloat16",
        device="cuda",
        disable_cuda_graph=True,
        mem_fraction_static=0.6,  # tli 的 fp32 gather 需要瞬态显存余量（0.85 时 KV pool 吃满 OOM）
        trust_remote_code=True,
        disable_radix_cache=True,  # 关键：否则第二次同 prompt 的 prefill 被 cache 命中，差分口径失效
    )
    # 预热：吸收 Triton JIT / CUDA 模块加载等一次性开销
    eng.generate(
        ["warmup"], sampling_params={"temperature": 0.0, "max_new_tokens": 4}
    )
    results = []
    for cl, p in prompts:
        # prefill only
        t0 = time.time()
        eng.generate([p], sampling_params={"temperature": 0.0, "max_new_tokens": 1})
        t_prefill = time.time() - t0
        # prefill + N decode
        t0 = time.time()
        o = eng.generate(
            [p], sampling_params={"temperature": 0.0, "max_new_tokens": 1 + N_DECODE}
        )
        t_total = time.time() - t0
        t_decode = t_total - t_prefill
        # 尝试提取 prompt token 数（新版 meta_info）
        ptok = None
        try:
            ptok = o[0]["meta_info"]["prompt_tokens"]
        except Exception:
            pass
        results.append(
            {
                "chars": cl,
                "prompt_tokens": ptok,
                "prefill_s": round(t_prefill, 3),
                "decode_s": round(t_decode, 3),
                "decode_tps": round(N_DECODE / t_decode, 2) if t_decode > 0 else None,
                "step_ms": round(t_decode * 1000 / N_DECODE, 1),
                "text": o[0]["text"][:120],
            }
        )
        print(
            f"[{backend}] chars={cl} ptok={ptok} prefill={t_prefill:.2f}s "
            f"decode={t_decode:.2f}s ({N_DECODE} steps, "
            f"{t_decode*1000/N_DECODE:.1f} ms/step, "
            f"{N_DECODE/t_decode:.2f} tok/s)"
        )
        sys.stdout.flush()
    eng.shutdown()
    return results


if __name__ == "__main__":
    prompts = load_contexts()
    all_res = {}
    for backend in ("triton", "tli"):
        print(f"\n===== backend: {backend} =====")
        all_res[backend] = run_backend(backend, prompts)
    # 汇总
    print("\n===== 汇总（decode ms/step, tok/s）=====")
    print(f"{'chars':>8} {'triton':>18} {'tli':>18}")
    for i, (cl, _) in enumerate(prompts):
        tr = all_res["triton"][i]
        tl = all_res["tli"][i]
        print(
            f"{cl:>8} {tr['step_ms']:>8.1f} {tr['decode_tps']:>9.2f}"
            f" {tl['step_ms']:>8.1f} {tl['decode_tps']:>9.2f}"
        )
    json.dump(all_res, open(OUT_JSON, "w"), indent=1)
    print("saved", OUT_JSON)
