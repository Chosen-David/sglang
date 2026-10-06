# #58 收官：S=64K×bs16 收益区首测（Qwen3-30B-A3B-Instruct-2507 256K context，TP2）
# 预核算：S=131K×bs16/32 单卡不可行（KV 310/620GB>141GB）→ 本规格 =
# TP2 + bs16 + S≈64K（narrativeqa 真实最长样本截 260K chars ≈ 64K token）。
# 双臂：attention_backend=tli vs triton（一进程一 Engine，结果落盘）。
# 口径：M3-c/M4/M11 decode AB 同款（分步计时、无图、预热吸收 JIT）。
# 用法：CUDA_VISIBLE_DEVICES=0,1 BACKEND=tli|triton python3 test_tli_64k_tp2.py
import json
import os
import sys
import time
import warnings

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
warnings.filterwarnings("ignore")

BACKEND = os.environ.get("BACKEND", "tli")
MODEL = "/tmp/huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507"
CHARS = 260000  # ≈ 60-64K token（narrativeqa 最长样本级）
BATCH = 16
N_DECODE = 64
OUT_JSON = f"/home/wangyuanshuo02/sglang/tli_64k_tp2_{BACKEND}.json"


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
        # 超长样本不足：重复填充（性能口径，同 context 不同 book 不影响计时形态）
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
    kwargs = dict(
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
    if BACKEND == "tli":
        # 共享 index pool 容量 ≥ bs16 × 64K
        os.environ.setdefault("SGLANG_TLI_POOL_R", "32")
        os.environ.setdefault("SGLANG_TLI_POOL_S_CAP", "65536")
    eng = Engine(**kwargs)
    prompts = make_prompts(BATCH)
    eng.generate(["hello"] * BATCH, sampling_params={"temperature": 0.0, "max_new_tokens": 1})
    t0 = time.time()
    outs = eng.generate(prompts, sampling_params={
        "temperature": 0.0, "max_new_tokens": N_DECODE})
    t1 = time.time()
    texts = [o["text"][:80] for o in outs]
    res = {
        "backend": BACKEND, "bs": BATCH, "chars": CHARS,
        "total_s": round(t1 - t0, 2),
        "texts": texts,
    }
    json.dump(res, open(OUT_JSON, "w"), indent=1, ensure_ascii=False)
    print(f"[{BACKEND}] total={t1-t0:.2f}s "
          f"decode≈{(t1-t0)/(N_DECODE)*1000:.1f}ms/step "
          f"tok/s={BATCH*N_DECODE/(t1-t0):.1f}")
    for t in texts[:3]:
        print("  -", repr(t))
    print("saved ->", OUT_JSON)
    eng.shutdown()


if __name__ == "__main__":
    main()
