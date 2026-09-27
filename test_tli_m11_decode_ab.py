# M11 decode e2e AB：SGLANG_TLI_SPARSE_KERNEL=1（fused）vs 0（eager 批量）
# 与 M5 e2e 完全同口径（narrativeqa 32K chars ≈9.9K token、分步计时、
# 无图——隔离验证 _sparse_attn_batched 尾段 kernel 化收益；graph 路径
# 下轮另测）。输出一致性用同 prompt 双臂生成对比。
# 用法：CUDA_VISIBLE_DEVICES=0 SGLANG_TLI_SPARSE_KERNEL=1 python3 test_tli_m11_decode_ab.py
import json
import os
import sys
import time
import warnings

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
warnings.filterwarnings("ignore")

MODEL = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B"
CHARS = 32000  # ≈ 9.9K token（M3-c/M4/M5 同曲线口径）
BATCHES = [int(x) for x in os.environ.get("M11_BATCHES", "8,16,32").split(",")]
N_DECODE = 64
MODE = os.environ.get("SGLANG_TLI_SPARSE_KERNEL", "1")
OUT_JSON = "/home/wangyuanshuo02/sglang/tli_m11_decode_ab.json"


def make_prompts(bs):
    rows = [json.loads(l) for l in open(
        "/home/wangyuanshuo02/datasets/LongBench/data/narrativeqa.jsonl")]
    rows.sort(key=lambda r: -len(r.get("context", "")))
    uniq, seen = [], set()
    for r in rows:
        ctx = r.get("context") or ""
        if len(ctx) < CHARS // 2:
            continue
        key = ctx[:10000]
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
        if len(uniq) >= bs:
            break
    # narrativeqa 唯一长 context 仅 ~20 个：不足时从其他 LongBench 子集补足
    # （M3-c 已知坑；补足样本只参与计时，质量口径不受影响）
    if len(uniq) < bs:
        import glob
        for f in sorted(glob.glob(
                "/home/wangyuanshuo02/datasets/LongBench/data/*.jsonl")):
            if "narrativeqa" in f:
                continue
            try:
                for r in map(json.loads, open(f)):
                    ctx = r.get("context") or ""
                    if len(ctx) < CHARS // 2:
                        continue
                    key = (f, ctx[:10000])
                    if key in seen:
                        continue
                    seen.add(key)
                    uniq.append({"context": ctx})
                    if len(uniq) >= bs:
                        break
            except (OSError, json.JSONDecodeError):
                continue
            if len(uniq) >= bs:
                break
    assert len(uniq) == bs, f"only {len(uniq)} unique contexts"
    return [r["context"][:CHARS] + "\n\nSummarize the above text in one sentence:"
            for r in uniq]


def main():
    from sglang import Engine

    eng = Engine(
        model_path=MODEL,
        attention_backend="tli",
        dtype="bfloat16",
        device="cuda",
        mem_fraction_static=0.6,
        trust_remote_code=True,
        disable_radix_cache=True,
        disable_cuda_graph=True,
        watchdog_timeout=1800,
    )
    eng.generate(["warmup"], sampling_params={"temperature": 0.0, "max_new_tokens": 4})
    results = []
    texts = {}
    for bs in BATCHES:
        prompts = make_prompts(bs)
        t0 = time.time()
        eng.generate(prompts, sampling_params={"temperature": 0.0, "max_new_tokens": 1})
        t_prefill = time.time() - t0
        t0 = time.time()
        outs = eng.generate(
            prompts, sampling_params={"temperature": 0.0, "max_new_tokens": 1 + N_DECODE})
        t_total = time.time() - t0
        t_decode = t_total - t_prefill
        rec = {
            "bs": bs,
            "prefill_s": round(t_prefill, 3),
            "decode_s": round(t_decode, 3),
            "step_ms": round(t_decode * 1000 / N_DECODE, 1),
            "tok_per_s": round(bs * N_DECODE / t_decode, 2),
        }
        results.append(rec)
        texts[bs] = [o["text"] for o in outs]
        print(f"[m11-dec kernel={MODE}] bs={bs:>2} prefill={t_prefill:6.2f}s "
              f"decode={t_decode:5.2f}s ({rec['step_ms']:6.1f} ms/step, "
              f"{rec['tok_per_s']:8.2f} tok/s)", flush=True)
    eng.shutdown()

    all_res = {}
    if os.path.exists(OUT_JSON):
        try:
            all_res = json.load(open(OUT_JSON))
        except Exception:
            pass
    all_res[f"kernel{MODE}"] = {"results": results}
    # 输出文本也存（跨臂一致性人工/脚本比对）
    json.dump(all_res, open(OUT_JSON, "w"), indent=1)
    json.dump(texts, open(f"/tmp/m11_dec_texts_{MODE}.json", "w"), ensure_ascii=False)
    print("saved ->", OUT_JSON)


if __name__ == "__main__":
    main()
