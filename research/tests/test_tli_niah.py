# NIAH（needle in a haystack）检索质量评测：tli vs triton（dense FullKV）。
#
# 口径对齐 RULER needle 类任务（Quest/SnapKV/HISA 论文均报）：S=32K haystack
# + magic number needle，深度 5%-95% 十档 × 每档 N_PER_DEPTH 样本，temperature=0，
# 输出含 needle 数字串计成功。haystack = 本地 LongBench gov_report + hotpotqa
# 长英文文档拼接（外网受限无法取 RULER 的 PG essays；NIAH 检索任务对 haystack
# 语义不敏感，拼接文档是社区常用替代）。needle 插在句边界（offset 精确 token 深度）。
#
# 用法（两进程分别跑，结果落盘对比）：
#   SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=true SGLANG_IS_FLASHINFER_AVAILABLE=false \
#   SGLANG_ENABLE_JIT_DEEPGEMM=0 CUDA_VISIBLE_DEVICES=1 \
#   SGLANG_TLI_L1_KERNEL=1 SGLANG_TLI_POOL_S_CAP=40000 \
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
#   PYTHONPATH=~/.local/pylibs_tf512:/home/wangyuanshuo02/sglang/python \
#   python test_tli_niah.py tli  / python test_tli_niah.py triton
import json
import os
import random
import sys
import time
import warnings

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
warnings.filterwarnings("ignore")

from sglang import Engine

MODEL = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B"
DATA_DIR = "/home/wangyuanshuo02/datasets/LongBench/data"
TARGET_TOKENS = int(os.environ.get("NIAH_TOKENS", 32000))
N_PER_DEPTH = int(os.environ.get("NIAH_N_PER_DEPTH", 2))
DEPTHS = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]  # %（RULER uniform 深度档）
OUT_JSON = "/home/wangyuanshuo02/sglang/tli_niah_results.json"
NEEDLE_TMPL = "One important piece of information is that the magic number is {num}. Remember this number."
QUESTION = "\n\nWhat is the magic number mentioned in the passage? Answer with only the number."


def build_haystack_pool(rng):
    """多源英文长文档池（去重、拼接候选）。"""
    rows = []
    for name in ("gov_report", "hotpotqa", "qasper", "multifieldqa_en", "triviaqa"):
        try:
            rows += [json.loads(l) for l in open(f"{DATA_DIR}/{name}.jsonl")]
        except Exception:
            pass
    rng.shuffle(rows)
    pool, seen = [], set()
    for r in rows:
        ctx = r.get("context") or ""
        if len(ctx) < 2000:
            continue
        key = ctx[:2000]
        if key in seen:
            continue
        seen.add(key)
        pool.append(ctx)
    return pool


def make_samples(tok, rng):
    """每样本独立 haystack + needle，深度按 token 位置精确控制。"""
    pool = build_haystack_pool(rng)
    samples = []
    pi = 0
    for depth in DEPTHS:
        for _ in range(N_PER_DEPTH):
            # 拼文档至 ~TARGET_TOKENS（留 needle+question 余量）
            parts, n_char = [], 0
            while True:
                ctx = pool[pi % len(pool)]
                pi += 1
                parts.append(ctx)
                n_char += len(ctx)
                if n_char > TARGET_TOKENS * 5.5:  # chars/token ≈4-5，超配再精确回切
                    break
            haystack = "\n\n".join(parts)
            enc = tok(haystack, add_special_tokens=False, return_offsets_mapping=True)
            n_tok = len(enc["input_ids"])
            if n_tok < TARGET_TOKENS + 256:
                continue  # 池耗尽保护（不应发生）
            # token 精确截断到 TARGET_TOKENS（前缀 tokenize 不变性）
            cut_char = enc["offset_mapping"][TARGET_TOKENS - 1][1]
            haystack = haystack[:cut_char]
            # needle 插入深度 d：找第 d*TARGET_TOKENS 个 token 的 char 边界，
            # 回退到最近的句边界（. 或换行后）避免劈开句子
            d_char = enc["offset_mapping"][int(depth / 100 * TARGET_TOKENS) - 1][1]
            seg = haystack[:d_char]
            k = max(seg.rfind(". "), seg.rfind(".\n"), seg.rfind("\n\n"))
            if k < len(seg) // 2:
                k = len(seg)
            needle = NEEDLE_TMPL.format(num=rng.randrange(10**6, 10**7))
            prompt = haystack[: k + 1].rstrip() + "\n" + needle + "\n" + haystack[k + 1 :] + QUESTION
            samples.append({"depth": depth, "needle_num": str(needle_num(needle)), "prompt": prompt})
    # 终验 token 数（needle/question 增量后须 < 40960-margin）
    final = []
    for s in samples:
        n = len(tok(s["prompt"], add_special_tokens=False)["input_ids"])
        s["prompt_tokens"] = n
        if n < 40960 - 256:
            final.append(s)
    return final


def needle_num(needle: str) -> int:
    return int(needle.split("magic number is ")[1].split(".")[0])


def main():
    backend = sys.argv[1] if len(sys.argv) > 1 else "tli"
    tag = f"niah_{backend}"
    if os.environ.get("NIAH_TAG"):  # 变体（如 far 预算扫描）不覆盖默认键
        tag = os.environ["NIAH_TAG"]
    rng = random.Random(int(os.environ.get("NIAH_SEED", 1234)))

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    samples = make_samples(tok, rng)
    print(f"[{tag}] {len(samples)} samples, token range "
          f"[{min(s['prompt_tokens'] for s in samples)}, {max(s['prompt_tokens'] for s in samples)}]")

    kwargs = dict(
        model_path=MODEL,
        attention_backend=backend,
        dtype="bfloat16",
        device="cuda",
        mem_fraction_static=0.7,
        trust_remote_code=True,
        disable_radix_cache=True,
        watchdog_timeout=3600,
        cuda_graph_config={"decode": {"backend": "full", "bs": [1]}, "prefill": {"backend": "disabled"}},
    )
    t0 = time.time()
    eng = Engine(**kwargs)
    print(f"[{tag}] engine init {time.time() - t0:.1f}s")
    eng.generate(["warmup"], sampling_params={"temperature": 0.0, "max_new_tokens": 4})

    prompts = [s["prompt"] for s in samples]
    t0 = time.time()
    outs = eng.generate(prompts, sampling_params={"temperature": 0.0, "max_new_tokens": 32})
    t_gen = time.time() - t0

    results = []
    n_ok = 0
    for s, o in zip(samples, outs):
        text = o["text"]
        ok = s["needle_num"] in text
        n_ok += ok
        results.append({"depth": s["depth"], "needle": s["needle_num"],
                        "ok": ok, "out": text.strip()[:60], "prompt_tokens": s["prompt_tokens"]})
    print(f"[{tag}] score {n_ok}/{len(results)} = {n_ok / len(results):.3f}  (gen {t_gen:.1f}s)")

    by_depth = {}
    for r in results:
        by_depth.setdefault(r["depth"], []).append(r["ok"])
    for d in sorted(by_depth):
        v = by_depth[d]
        print(f"  depth {d:>2}%: {sum(v)}/{len(v)}")

    all_res = {}
    if os.path.exists(OUT_JSON):
        try:
            all_res = json.load(open(OUT_JSON))
        except Exception:
            all_res = {}
    all_res[tag] = {"score": round(n_ok / len(results), 4), "n": len(results),
                    "gen_s": round(t_gen, 1), "samples": results}
    json.dump(all_res, open(OUT_JSON, "w"), indent=1, ensure_ascii=False)
    print(f"saved {OUT_JSON} [{tag}]")
    eng.shutdown()


if __name__ == "__main__":
    main()
