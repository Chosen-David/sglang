# e2e decode 差分法方差仲裁（#56）：bs=16 / S=30K / N=256。
# 问题：decode = t_total − t_prefill 两次独立 run 相减，tli prefill ~735s 的
# run-to-run 方差直接污染 decode（m10 结果 decode_s 为负是铁证；N=64 首测
# 47.7ms vs N=256 复测 67.1ms 差 40%）。
# 方案：每 backend 跑 prefill-only ×2 + prefill+N ×2 交替（P1 D1 P2 D2），
# 得两个 decode 估计（D_i − P_i 交叉配对）+ 两个 prefill 估计 → step_ms ± 区间。
# 用法：python test_tli_e2e_variance.py {tli|triton}（复用 m8_long 的环境变量集）
import json
import os
import sys
import time
import warnings

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
warnings.filterwarnings("ignore")

from sglang import Engine

MODEL = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B"
TARGET_TOKENS = int(os.environ.get("LONG_TOKENS", 30000))
BS = int(os.environ.get("VAR_BS", 16))
N_DECODE = int(os.environ.get("LONG_N_DECODE", 256))
MODE = os.environ.get("VAR_MODE", "30k")  # 30k=vcsum token 精确 | 9k=m8_e2e chars 口径
N_ROUNDS = int(os.environ.get("VAR_ROUNDS", 2))
# #57：bs16×40K 需 644K token KV，mem 0.7 下 tli pool 仅 ~497K（索引池吃 15.8GB）
# → retraction 混沌（同 run 41/125ms 双峰）。40K 档必须 0.85（pool ~653K）。
# 0.85 又遇 prefill transient OOM（capture 后仅剩 4.18GB，select_batched 的
# far/near_sc [n,Hkv,Tc] fp32 双 scratch 超 4GB）→ VAR_CHUNK 压小
# chunked_prefill：decode 差分法只依赖两次 run 同 chunking，chunk 不影响
# decode step 估计的口径。
MEM_FRAC = float(os.environ.get("VAR_MEM_FRAC", 0.7))
CHUNK = int(os.environ.get("VAR_CHUNK", 0))  # 0 = 引擎默认
OUT_JSON = "/home/wangyuanshuo02/sglang/tli_e2e_variance_results.json"


def make_prompts(bs):
    import glob

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    rows = []
    for f in sorted(glob.glob("/home/wangyuanshuo02/datasets/LongBench/data/*.jsonl")):
        try:
            rows += [json.loads(l) for l in open(f)]
        except Exception:
            pass
    rows.sort(key=lambda r: -len(r.get("context", "") or ""))
    seen, uniq = set(), []
    if MODE == "9k":
        # 与 test_tli_m8_e2e.py 完全同口径（narrativeqa 最长唯一 context，
        # CHARS=32000 截断 + 英文 summarize prompt）→ fig9a 9.9K 点同源
        for r in rows:
            ctx = r.get("context") or ""
            if len(ctx) < 32000 // 2:
                continue
            key = ctx[:10000]
            if key in seen:
                continue
            seen.add(key)
            uniq.append(r)
            if len(uniq) >= bs:
                break
        prompts = [r["context"][:32000] + "\n\nSummarize the above text in one sentence:" for r in uniq]
    else:
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
        prompts = []
        for r in uniq:
            ctx = r["context"]
            enc = tok(ctx, add_special_tokens=False, return_offsets_mapping=True)
            if len(enc["input_ids"]) <= TARGET_TOKENS:
                cut = ctx
            else:
                cut = ctx[: enc["offset_mapping"][TARGET_TOKENS - 1][1]]
            prompts.append(cut + "\n\n请用一句话总结上文。")
    assert len(uniq) == bs, f"only {len(uniq)} unique long contexts"
    n_tok = [len(tok(p)["input_ids"]) for p in prompts]
    print(f"[make_prompts] bs={bs} prompt tokens: {n_tok}")
    assert max(n_tok) < 40960 - N_DECODE - 8
    return prompts


def main():
    backend = sys.argv[1] if len(sys.argv) > 1 else "tli"
    # tag 必须含 S（TARGET_TOKENS）：否则不同 S 的 run 会互相覆盖 JSON 归档
    # （#57 教训：40K run 差点覆盖 #56 的 30K 仲裁数据）
    tag = f"var_{MODE}_S{TARGET_TOKENS}_{backend}_bs{BS}_n{N_DECODE}" + (f"_r{N_ROUNDS}" if N_ROUNDS != 2 else "")

    eng = Engine(
        model_path=MODEL,
        attention_backend=backend,
        dtype="bfloat16",
        device="cuda",
        mem_fraction_static=MEM_FRAC,
        trust_remote_code=True,
        disable_radix_cache=True,
        watchdog_timeout=1800,
        cuda_graph_config={"decode": {"backend": "full", "bs": [BS]}, "prefill": {"backend": "disabled"}},
        **({"chunked_prefill_size": CHUNK} if CHUNK else {}),
    )
    eng.generate(["warmup"], sampling_params={"temperature": 0.0, "max_new_tokens": 4})
    prompts = make_prompts(BS)

    # P1 D1 P2 D2 交替：两个 (prefill, prefill+N) 配对。
    # ignore_eos=True 是关键：greedy 下「一句话总结」prompt 会在 ~30 token EOS
    # 早停，实际 decode 步数 << max_new_tokens，差分法除以 N 会得到 7.9ms/step
    # 的假象（2.02s ÷ 256 而实际只走了 ~30 步）。ignore_eos 强制走满 N 步。
    prefill_s, decode_s, n_toks = [], [], []
    for rnd in range(N_ROUNDS):
        t0 = time.time()
        eng.generate(prompts, sampling_params={"temperature": 0.0, "max_new_tokens": 1})
        tp = time.time() - t0
        t0 = time.time()
        outs = eng.generate(
            prompts,
            sampling_params={"temperature": 0.0, "max_new_tokens": 1 + N_DECODE, "ignore_eos": True},
        )
        tt = time.time() - t0
        td = tt - tp
        ks = [o.get("meta_info", {}).get("completion_tokens") for o in outs]
        prefill_s.append(round(tp, 3))
        decode_s.append(round(td, 3))
        n_toks.append(ks)
        print(f"[{tag}] round{rnd}: prefill={tp:7.2f}s total={tt:7.2f}s "
              f"decode={td:6.2f}s ({td * 1000 / N_DECODE:.1f} ms/step, ktok={ks[:4]}...)")
        sys.stdout.flush()
    eng.shutdown()

    steps = [d * 1000 / N_DECODE for d in decode_s]
    rec = {
        "backend": backend, "bs": BS, "prompt_tokens": TARGET_TOKENS, "n_decode": N_DECODE,
        "ignore_eos": True, "completion_tokens": n_toks,
        "prefill_s": prefill_s, "decode_s": decode_s, "step_ms": [round(s, 1) for s in steps],
        "step_ms_mean": round(sum(steps) / len(steps), 1),
        "step_ms_spread": round(abs(steps[0] - steps[1]), 1),
        "prefill_spread_s": round(abs(prefill_s[0] - prefill_s[1]), 1),
    }
    all_res = {}
    if os.path.exists(OUT_JSON):
        try:
            all_res = json.load(open(OUT_JSON))
        except Exception:
            all_res = {}
    all_res[tag] = rec
    json.dump(all_res, open(OUT_JSON, "w"), indent=1)
    print(f"[{tag}] summary: {json.dumps(rec, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
