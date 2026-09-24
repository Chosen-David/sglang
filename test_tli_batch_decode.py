# M3-c: 高并发 decode 曲线（bs=1/8/16/32 × S≈9.9K，tli vs triton）
# 论文吞吐主表标准口径：大 batch 下权重流量被摊销、KV 流量成第一瓶颈，
# 稀疏化收益随 bs 增长的趋势曲线。
# 运行：cd /home/wangyuanshuo02/sglang && \
#   SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=true SGLANG_IS_FLASHINFER_AVAILABLE=false \
#   SGLANG_ENABLE_JIT_DEEPGEMM=0 CUDA_VISIBLE_DEVICES=1 SGLANG_TLI_L1_KERNEL=1 \
#   PYTHONPATH=/home/wangyuanshuo02/.local/pylibs_tf512:/home/wangyuanshuo02/sglang/python \
#   python test_tli_batch_decode.py
import json
import sys
import time
import warnings

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
warnings.filterwarnings("ignore")

from sglang import Engine

MODEL = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B"
DATA = "/home/wangyuanshuo02/datasets/LongBench/data/narrativeqa.jsonl"
CHARS = 32000  # ≈ 9.9K token
BATCHES = [1, 8, 16, 32]
N_DECODE = 64
OUT_JSON = "/home/wangyuanshuo02/sglang/tli_batch_decode_results.json"


def make_prompts(bs):
    # narrativeqa 同文档去重后仅 ~20 个唯一 context；不足 bs 时用其他
    # LongBench 真实子集补足（仍是真实文本，非合成）
    import glob

    rows = [json.loads(l) for l in open(DATA)]
    for extra in sorted(glob.glob("/home/wangyuanshuo02/datasets/LongBench/data/*.jsonl")):
        if len(rows) >= 512:
            break
        try:
            rows += [json.loads(l) for l in open(extra)]
        except Exception:
            pass
    rows.sort(key=lambda r: -len(r.get("context", "")))
    # 不同样本（同 context 去重），保证 bs 份真实多样本
    seen, uniq = set(), []
    for r in rows:
        ctx = r.get("context") or ""
        if len(ctx) < CHARS // 2:
            continue  # 太短撑不起目标 S
        key = ctx[:10000]
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
        if len(uniq) >= bs:
            break
    assert len(uniq) == bs, f"only {len(uniq)} unique contexts"
    return [
        r["context"][:CHARS] + "\n\nSummarize the above text in one sentence:"
        for r in uniq
    ]


def run_backend(backend):
    eng = Engine(
        model_path=MODEL,
        attention_backend=backend,
        dtype="bfloat16",
        device="cuda",
        disable_cuda_graph=True,
        mem_fraction_static=0.6,
        trust_remote_code=True,
        disable_radix_cache=True,
    )
    eng.generate(["warmup"], sampling_params={"temperature": 0.0, "max_new_tokens": 4})
    results = []
    for bs in BATCHES:
        prompts = make_prompts(bs)
        # prefill only
        t0 = time.time()
        eng.generate(prompts, sampling_params={"temperature": 0.0, "max_new_tokens": 1})
        t_prefill = time.time() - t0
        # prefill + N decode
        t0 = time.time()
        eng.generate(
            prompts,
            sampling_params={"temperature": 0.0, "max_new_tokens": 1 + N_DECODE},
        )
        t_total = time.time() - t0
        t_decode = t_total - t_prefill
        results.append(
            {
                "bs": bs,
                "prefill_s": round(t_prefill, 3),
                "decode_s": round(t_decode, 3),
                "step_ms": round(t_decode * 1000 / N_DECODE, 1),
                "tok_per_s": round(bs * N_DECODE / t_decode, 2) if t_decode > 0 else None,
            }
        )
        print(
            f"[{backend}] bs={bs:>2} prefill={t_prefill:6.2f}s decode={t_decode:5.2f}s "
            f"({t_decode * 1000 / N_DECODE:6.1f} ms/step, "
            f"{bs * N_DECODE / t_decode:8.2f} tok/s total)"
        )
        sys.stdout.flush()
    eng.shutdown()
    return results


if __name__ == "__main__":
    all_res = {}
    for backend in ("triton", "tli"):
        print(f"\n===== backend: {backend} =====")
        all_res[backend] = run_backend(backend)
    print("\n===== 汇总 =====")
    print(f"{'bs':>4} | {'triton ms/step':>15} {'tok/s':>10} | {'tli ms/step':>12} {'tok/s':>10}")
    for i, bs in enumerate(BATCHES):
        tr = all_res["triton"][i]
        tl = all_res["tli"][i]
        print(
            f"{bs:>4} | {tr['step_ms']:>15.1f} {tr['tok_per_s']:>10.2f} "
            f"| {tl['step_ms']:>12.1f} {tl['tok_per_s']:>10.2f}"
        )
    json.dump(all_res, open(OUT_JSON, "w"), indent=1)
    print("saved", OUT_JSON)
