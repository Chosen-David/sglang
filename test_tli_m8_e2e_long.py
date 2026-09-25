# M8 e2e（长上下文高并发）：bs=8/16 × S≈30K token，tli graph vs triton graph。
#
# 动机（报告 §8b-13 下一步）：M5/M8 现有 e2e 只有 9.9K token（tli @10K 仍慢于
# triton，收益展示位在长 S + 大 batch）。Qwen3-8B context 上限 40960 → e2e
# 最多 ~30K token/请求。显存账（H20-3e 141GB）：
#   KV = 144KB/token（36 层 × 8 Hkv × 128 D × (k,v) × bf16）
#   bs=16 × 30K ≈ 71GB + kq 4bit index 池 ~7.4GB（S_CAP=40K）+ 权重 16GB ≈ 95GB
#   → mem_fraction_static=0.7；bs=32 需 155GB KV 不可行（超模型上下文一半也找不到
#   32 个唯一长文档：LongBench ≥157K chars 唯一文档仅 20 个，全在 vcsum——
#   中文 chars/token≈5:1（Qwen tokenizer 实测 105K chars=21K token），故 CHARS=150K
#   ≈ 30K token）。
#
# watchdog_timeout=1800：30K×16 req 的稀疏 prefill 预估数百秒，默认 300s 会被
# SIGQUIT。其余口径与 test_tli_m8_e2e.py 完全一致（graph decode full + prefill
# disabled、decode 差分法、disable_radix_cache、N=64）。
#
# 用法（GPU 空闲时依次）：
#   cd /home/wangyuanshuo02/sglang && \
#   SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=true SGLANG_IS_FLASHINFER_AVAILABLE=false \
#   SGLANG_ENABLE_JIT_DEEPGEMM=0 CUDA_VISIBLE_DEVICES=1 SGLANG_TLI_L1_KERNEL=1 \
#   SGLANG_TLI_POOL_S_CAP=40000 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
#   PYTHONPATH=/home/wangyuanshuo02/.local/pylibs_tf512:/home/wangyuanshuo02/sglang/python \
#   python test_tli_m8_e2e_long.py tli 1
#   （随后同命令 triton 1）
import json
import os
import sys
import time
import warnings

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
warnings.filterwarnings("ignore")

from sglang import Engine

MODEL = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B"
DATA = "/home/wangyuanshuo02/datasets/LongBench/data/vcsum.jsonl"
# 截断口径=精确 token 数（vcsum 中文文档间 chars/token 波动 3.1-5:1，chars 截断
# 会让 150K chars 文档达 47896 token 超模型上限 40960——首次运行实测失败教训）
TARGET_TOKENS = int(os.environ.get("LONG_TOKENS", 30000))
BATCHES = [int(x) for x in os.environ.get("LONG_BATCHES", "8,16").split(",")]
N_DECODE = int(os.environ.get("LONG_N_DECODE", 64))
OUT_JSON = "/home/wangyuanshuo02/sglang/tli_m8_e2e_long_results.json"
GRAPH_CFG = {"decode": {"backend": "full", "bs": BATCHES}, "prefill": {"backend": "disabled"}}


def make_prompts(bs):
    # 全 LongBench 子集按 context 长度排序 + ctx[:10000] 去重（同书多问），
    # 与 test_tli_m8_e2e.py 同口径；长文档集中在 vcsum（中文会议纪要）。
    # token 截断用 offset_mapping 找第 TARGET_TOKENS 个 token 的 char 边界
    # → 传给 engine 的仍是原文前缀（前缀 tokenize 不变性，无 decode-encode 往返）
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
    for r in rows:
        ctx = r.get("context") or ""
        if len(ctx) < 100000:  # chars 预筛（ratio 3-5:1 → 30K-50K token 均 ≥ 截断目标）
            continue
        key = ctx[:10000]
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
        if len(uniq) >= bs:
            break
    assert len(uniq) == bs, f"only {len(uniq)} unique long contexts"
    prompts = []
    for r in uniq:
        ctx = r["context"]
        enc = tok(ctx, add_special_tokens=False, return_offsets_mapping=True)
        if len(enc["input_ids"]) <= TARGET_TOKENS:
            cut = ctx
        else:
            cut = ctx[: enc["offset_mapping"][TARGET_TOKENS - 1][1]]
        prompts.append(cut + "\n\n请用一句话总结上文。")
    n_tok = [len(tok(p)["input_ids"]) for p in prompts]
    print(f"[make_prompts] bs={bs} prompt tokens: {n_tok}")
    assert max(n_tok) < 40960 - N_DECODE - 8, "prompt 超模型上下文上限"
    return prompts


def main():
    backend = sys.argv[1] if len(sys.argv) > 1 else "tli"
    use_graph = sys.argv[2] if len(sys.argv) > 2 else "1"
    tag = f"m8_long_{backend}_graph{use_graph}"
    if N_DECODE != 64:
        tag += f"_n{N_DECODE}"  # N≠64 不覆盖历史键（decode 信噪比复测变体）

    kwargs = dict(
        model_path=MODEL,
        attention_backend=backend,
        dtype="bfloat16",
        device="cuda",
        mem_fraction_static=float(os.environ.get("LONG_MEM_FRAC", 0.7)),
        trust_remote_code=True,
        disable_radix_cache=True,
        watchdog_timeout=1800,  # 30K×16 req 稀疏 prefill 预估数百秒
    )
    if use_graph == "1":
        kwargs["cuda_graph_config"] = GRAPH_CFG
    else:
        kwargs["disable_cuda_graph"] = True

    t_init = time.time()
    eng = Engine(**kwargs)
    print(f"[{tag}] engine init (含 graph capture) {time.time() - t_init:.1f}s")
    sys.stdout.flush()
    eng.generate(["warmup"], sampling_params={"temperature": 0.0, "max_new_tokens": 4})

    results = []
    for bs in BATCHES:
        prompts = make_prompts(bs)
        # prefill only（分步计时：prefill 与 decode 分离）
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
        rec = {
            "bs": bs,
            "prompt_tokens": TARGET_TOKENS,
            "prefill_s": round(t_prefill, 3),
            "decode_s": round(t_decode, 3),
            "step_ms": round(t_decode * 1000 / N_DECODE, 1),
            "tok_per_s": round(bs * N_DECODE / t_decode, 2),
        }
        results.append(rec)
        print(
            f"[{tag}] bs={bs:>2} prefill={t_prefill:7.2f}s decode={t_decode:6.2f}s "
            f"({rec['step_ms']:7.1f} ms/step, {rec['tok_per_s']:8.2f} tok/s)"
        )
        sys.stdout.flush()
    eng.shutdown()

    all_res = {}
    if os.path.exists(OUT_JSON):
        try:
            all_res = json.load(open(OUT_JSON))
        except Exception:
            all_res = {}
    all_res[tag] = results
    json.dump(all_res, open(OUT_JSON, "w"), indent=1)
    print(f"saved {OUT_JSON} [{tag}]")


if __name__ == "__main__":
    main()
