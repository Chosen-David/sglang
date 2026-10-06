# RULER 多任务质量评测：niah_multikey / niah_multivalue / niah_multiquery /
# variable_tracking（RULER 官方模板格式，Quest/HISA 论文口径）。
# 复用 NIAH 基建（同一 haystack 池 + token 精确深度插入 + 句边界对齐）。
# S=32K、深度均匀、temperature=0、n=20/任务、双方法（tli vs triton FullKV）。
#
# 用法（一进程一 Engine，两进程分别跑）：
#   SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=true SGLANG_IS_FLASHINFER_AVAILABLE=false \
#   SGLANG_ENABLE_JIT_DEEPGEMM=0 CUDA_VISIBLE_DEVICES=1 \
#   SGLANG_TLI_POOL_S_CAP=40000 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
#   PYTHONPATH=~/.local/pylibs_tf512:/home/wangyuanshuo02/sglang/python \
#   python test_tli_ruler.py tli  /  python test_tli_ruler.py triton
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
N_PER_TASK = int(os.environ.get("RULER_N", 20))
OUT_JSON = "/home/wangyuanshuo02/sglang/tli_ruler_results.json"
# RULER 官方 needle 模板（keys 用 uuid 风格词表）
KEY_POOL = [
    "Dwight", "Cece", "Cassady", "Renate", "Joetta", "Lidia", "Trudy", "Odell",
    "Kati", "Jenica", "Afton", "Dallas", "Zola", "Fleming", "Tonda", "Merna",
    "Randell", "Young", "Alfons", "Gwenn",
]


def build_haystack_pool(rng):
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


def insert_needles(haystack, enc, target_depth_frac, needles):
    """多针插入：主针插目标深度，其余均匀散布。句边界对齐（与 NIAH 同法）。"""
    n_tok = len(enc["input_ids"])
    out = []
    k_prev = 0
    for i, needle in enumerate(needles):
        frac = target_depth_frac if i == 0 else (i / len(needles)) * 0.9 + 0.05
        d_char = enc["offset_mapping"][int(frac * n_tok) - 1][1]
        d_char = min(d_char, len(haystack) - 1)
        seg = haystack[k_prev:d_char]
        k = max(seg.rfind(". "), seg.rfind(".\n"), seg.rfind("\n\n"))
        if k < len(seg) // 2:
            k = len(seg)
        out.append(haystack[k_prev : k_prev + k + 1].rstrip())
        out.append(needle)
        k_prev = k_prev + k + 1
    out.append(haystack[k_prev:])
    return "\n".join(out)


def make_task_samples(tok, rng):
    pool = build_haystack_pool(rng)
    tasks = {}

    # ---- niah_multikey_1：4 针不同 key，问其中一条 ----
    samples = []
    pi = 0
    for i in range(N_PER_TASK):
        parts, n_char = [], 0
        while True:
            ctx = pool[pi % len(pool)]
            pi += 1
            parts.append(ctx)
            n_char += len(ctx)
            if n_char > TARGET_TOKENS * 5.5:
                break
        haystack = "\n\n".join(parts)
        enc = tok(haystack, add_special_tokens=False, return_offsets_mapping=True)
        if len(enc["input_ids"]) < TARGET_TOKENS + 512:
            continue
        cut = enc["offset_mapping"][TARGET_TOKENS - 1][1]
        haystack = haystack[:cut]
        keys = rng.sample(KEY_POOL, 4)
        nums = [str(rng.randrange(10**6, 10**7)) for _ in range(4)]
        needles = [
            f"One of the special magic numbers for {k} is: {n}." for k, n in zip(keys, nums)
        ]
        depth = rng.choice([15, 35, 55, 75, 90]) / 100
        prompt = insert_needles(haystack, enc, depth, needles)
        qkey = keys[0]
        prompt += f"\n\nWhat is the special magic number for {qkey} mentioned in the provided text?"
        samples.append({"ans": [nums[0]], "prompt": prompt})
    tasks["niah_multikey"] = samples

    # ---- niah_multivalue：同 key 5 个值，列举全部 ----
    samples = []
    for i in range(N_PER_TASK):
        parts, n_char = [], 0
        while True:
            ctx = pool[pi % len(pool)]
            pi += 1
            parts.append(ctx)
            n_char += len(ctx)
            if n_char > TARGET_TOKENS * 5.5:
                break
        haystack = "\n\n".join(parts)
        enc = tok(haystack, add_special_tokens=False, return_offsets_mapping=True)
        if len(enc["input_ids"]) < TARGET_TOKENS + 512:
            continue
        cut = enc["offset_mapping"][TARGET_TOKENS - 1][1]
        haystack = haystack[:cut]
        key = rng.choice(KEY_POOL)
        nums = [str(rng.randrange(10**6, 10**7)) for _ in range(5)]
        needles = [f"One of the special magic numbers for {key} is: {n}." for n in nums]
        depth = rng.choice([15, 35, 55, 75, 90]) / 100
        prompt = insert_needles(haystack, enc, depth, needles)
        prompt += f"\n\nWhat are all the special magic numbers for {key} mentioned in the provided text?"
        samples.append({"ans": nums, "prompt": prompt})
    tasks["niah_multivalue"] = samples

    # ---- niah_multiquery：4 针不同 key，一次问全部 4 个 ----
    samples = []
    for i in range(N_PER_TASK):
        parts, n_char = [], 0
        while True:
            ctx = pool[pi % len(pool)]
            pi += 1
            parts.append(ctx)
            n_char += len(ctx)
            if n_char > TARGET_TOKENS * 5.5:
                break
        haystack = "\n\n".join(parts)
        enc = tok(haystack, add_special_tokens=False, return_offsets_mapping=True)
        if len(enc["input_ids"]) < TARGET_TOKENS + 512:
            continue
        cut = enc["offset_mapping"][TARGET_TOKENS - 1][1]
        haystack = haystack[:cut]
        keys = rng.sample(KEY_POOL, 4)
        nums = [str(rng.randrange(10**6, 10**7)) for _ in range(4)]
        needles = [
            f"One of the special magic numbers for {k} is: {n}." for k, n in zip(keys, nums)
        ]
        depth = rng.choice([15, 35, 55, 75, 90]) / 100
        prompt = insert_needles(haystack, enc, depth, needles)
        kq = ", ".join(keys[:-1]) + f" and {keys[-1]}"
        prompt += f"\n\nWhat are all the special magic numbers for {kq} mentioned in the provided text?"
        samples.append({"ans": nums, "prompt": prompt})
    tasks["niah_multiquery"] = samples

    # ---- variable_tracking：变量链（RULER 官方 VT 格式）----
    samples = []
    for i in range(N_PER_TASK):
        parts, n_char = [], 0
        while True:
            ctx = pool[pi % len(pool)]
            pi += 1
            parts.append(ctx)
            n_char += len(ctx)
            if n_char > TARGET_TOKENS * 5.5:
                break
        haystack = "\n\n".join(parts)
        enc = tok(haystack, add_special_tokens=False, return_offsets_mapping=True)
        if len(enc["input_ids"]) < TARGET_TOKENS + 512:
            continue
        cut = enc["offset_mapping"][TARGET_TOKENS - 1][1]
        haystack = haystack[:cut]
        # RULER VT：N_VALS 值 → N_CHAINS 链（每链 1 直接赋值 + 复制链）
        n_vals, n_chains = 5, 3
        vals = [str(rng.randrange(10**5, 10**6)) for _ in range(n_vals)]
        needles = []
        for v in vals:
            needles.append(f"VAR {v[0:2]}A{v[3:5]} = {v}.")
        # 复制链（每值 1 链）：X2 = X1 形式
        var_names = [f"VAR {v[0:2]}A{v[3:5]}" for v in vals]
        final_map = {}
        for name, v in zip(var_names, vals):
            prev = name
            for step in range(n_chains):
                nxt = f"VAR {v[0:2]}B{step}{v[3:5]}"
                needles.append(f"{nxt} = {prev}.")
                prev = nxt
            final_map[prev] = v
        depth = rng.choice([25, 55, 85]) / 100
        prompt = insert_needles(haystack, enc, depth, needles)
        qname = rng.choice(list(final_map))
        prompt += f"\n\nWhat is the value of {qname} mentioned in the provided text?"
        samples.append({"ans": [final_map[qname]], "prompt": prompt})
    tasks["variable_tracking"] = samples

    # 终验 token 数
    for name, ss in list(tasks.items()):
        fin = []
        for s in ss:
            n = len(tok(s["prompt"], add_special_tokens=False)["input_ids"])
            if n < 40960 - 256:
                s["prompt_tokens"] = n
                fin.append(s)
        tasks[name] = fin
    return tasks


def main():
    backend = sys.argv[1] if len(sys.argv) > 1 else "tli"
    tag = f"ruler_{backend}"
    if os.environ.get("RULER_TAG"):
        tag = os.environ["RULER_TAG"]
    rng = random.Random(int(os.environ.get("RULER_SEED", 1234)))

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    tasks = make_task_samples(tok, rng)
    for name, ss in tasks.items():
        print(f"[{tag}] {name}: {len(ss)} samples")
    total = sum(len(ss) for ss in tasks.values())
    print(f"[{tag}] total {total} samples")

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

    all_res = {}
    if os.path.exists(OUT_JSON):
        try:
            all_res = json.load(open(OUT_JSON))
        except Exception:
            all_res = {}

    task_results = {}
    for name, ss in tasks.items():
        prompts = [s["prompt"] for s in ss]
        t0 = time.time()
        outs = eng.generate(prompts, sampling_params={"temperature": 0.0, "max_new_tokens": 160})
        t_gen = time.time() - t0
        rows = []
        for s, o in zip(ss, outs):
            text = o["text"]
            # 评分：全部答案值出现在输出中（RULER 官方 multivalue/multiquery 全命中口径）
            ok = all(a in text for a in s["ans"])
            rows.append({"ok": ok, "ans": s["ans"], "out": text.strip()[:80],
                         "prompt_tokens": s["prompt_tokens"]})
        score = sum(r["ok"] for r in rows) / max(1, len(rows))
        task_results[name] = {"score": round(score, 3), "n": len(rows),
                              "gen_s": round(t_gen, 1), "samples": rows}
        print(f"[{tag}] {name}: {sum(r['ok'] for r in rows)}/{len(rows)} = {score:.3f}  (gen {t_gen:.1f}s)")
        sys.stdout.flush()

    all_res[tag] = task_results
    json.dump(all_res, open(OUT_JSON, "w"), indent=1)
    print(f"saved {OUT_JSON} [{tag}]")
    eng.shutdown()


if __name__ == "__main__":
    main()
