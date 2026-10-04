# 审稿 C3：三臂（triton dense / tli / quest）TP2 × bs16 × S≈64K e2e 对照
# （Qwen3-30B-A3B-Instruct-2507 256K context，64 decode 步，prefill/decode 分开计时）。
# 口径承接 test_tli_64k_tp2.py（一进程一 Engine、无图、预热吸收 JIT；
# 参考数字：TP2 triton 106.24s / tli 115.28s 为 total 口径）。
# 分计时两遍法：pass1(max_new=1) = prefill；pass2(max_new=64) 为 total，
# decode = total - prefill（63 步有效；disable_radix_cache 保证两遍独立 prefill）。
# 用法（GPU 空闲后逐臂跑，需双卡）：
#   SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=true SGLANG_IS_FLASHINFER_AVAILABLE=false \
#   SGLANG_ENABLE_JIT_DEEPGEMM=0 CUDA_VISIBLE_DEVICES=0,1 \
#   PYTHONPATH=/home/wangyuanshuo02/.local/pylibs_tf512:/home/wangyuanshuo02/sglang/python \
#   BACKEND=quest python3 test_c3_3arm_tp2.py
#   # BACKEND=triton / BACKEND=tli 同法
import json
import os
import sys
import time
import warnings

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
warnings.filterwarnings("ignore")

BACKEND = os.environ.get("BACKEND", "quest")
MODEL = "/tmp/huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507"
CHARS = 260000  # ≈ 60-64K token（narrativeqa 最长样本级）
BATCH = 16
N_DECODE = 64
OUT_JSON = f"/home/wangyuanshuo02/sglang/c3_3arm_tp2_{BACKEND}.json"


def make_prompts(bs):
    rows = [json.loads(l) for l in open(
        "/home/wangyuanshuo02/datasets/LongBench/data/narrativeqa.jsonl")]
    rows.sort(key=lambda r: -len(r.get("context", "")))
    uniq, seen = [], set()
    for r in rows:
        ctx = r.get("context") or ""
        if len(ctx) < 100000:
            continue
        key = ctx[:10000]
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
        if len(uniq) >= bs:
            break
    if len(uniq) < bs:
        import glob
        for f in sorted(glob.glob(
                "/home/wangyuanshuo02/datasets/LongBench/data/*.jsonl")):
            if "narrativeqa" in f:
                continue
            try:
                for r in map(json.loads, open(f)):
                    ctx = r.get("context") or ""
                    if len(ctx) < 100000:
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
    assert len(uniq) == bs, f"only {len(uniq)} unique long contexts"
    return [r["context"][:CHARS] + "\n\nSummarize the above text in one sentence:"
            for r in uniq]


def main():
    from sglang import Engine
    if BACKEND == "tli":
        # 共享 index pool 容量 ≥ bs16 × 64K（TLI 既有口径）
        os.environ.setdefault("SGLANG_TLI_POOL_R", "32")
        os.environ.setdefault("SGLANG_TLI_POOL_S_CAP", "65536")
    if BACKEND == "quest":
        # Quest pool 容量（nblk 维）：bs16 行 + S_cap 64K
        os.environ.setdefault("SGLANG_QUEST_POOL_R", "32")
        os.environ.setdefault("SGLANG_QUEST_POOL_S_CAP", "65536")
    eng = Engine(
        model_path=MODEL,
        attention_backend=BACKEND,
        dtype="bfloat16",
        device="cuda",
        tp_size=2,
        mem_fraction_static=0.8,
        trust_remote_code=True,
        disable_radix_cache=True,
        watchdog_timeout=3600,
        cuda_graph_config={"decode": {"backend": "disabled"},
                           "prefill": {"backend": "disabled"}},
    )
    prompts = make_prompts(BATCH)
    eng.generate(["hello"] * BATCH,
                 sampling_params={"temperature": 0.0, "max_new_tokens": 1})
    # pass1：纯 prefill 计时
    t0 = time.time()
    eng.generate(prompts, sampling_params={"temperature": 0.0, "max_new_tokens": 1})
    t1 = time.time()
    prefill_s = t1 - t0
    # pass2：prefill + 64 步 decode（disable_radix_cache → 独立 prefill）
    t0 = time.time()
    outs = eng.generate(prompts, sampling_params={
        "temperature": 0.0, "max_new_tokens": N_DECODE})
    t1 = time.time()
    total_s = t1 - t0
    decode_s = max(total_s - prefill_s, 0.0)
    texts = [o["text"][:80] for o in outs]
    res = {
        "backend": BACKEND, "bs": BATCH, "chars": CHARS, "n_decode": N_DECODE,
        "prefill_s": round(prefill_s, 2),
        "total_s": round(total_s, 2),
        "decode_s": round(decode_s, 2),
        "decode_ms_per_step": round(decode_s / (N_DECODE - 1) * 1000, 1),
        "tok_per_s": round(BATCH * N_DECODE / total_s, 1),
        "texts": texts,
    }
    json.dump(res, open(OUT_JSON, "w"), indent=1, ensure_ascii=False)
    print(f"[{BACKEND}] prefill={prefill_s:.2f}s total={total_s:.2f}s "
          f"decode={decode_s:.2f}s ({res['decode_ms_per_step']}ms/step) "
          f"tok/s={res['tok_per_s']}")
    for t in texts[:3]:
        print("  -", repr(t))
    print("saved ->", OUT_JSON)
    eng.shutdown()


if __name__ == "__main__":
    main()
