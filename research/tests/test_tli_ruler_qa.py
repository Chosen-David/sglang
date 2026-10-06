# RULER QA 任务（qa_1/qa_2，RULER 官方模板格式）。
# qa_1 = TriviaQA 单跳事实问答 needle（检索型，与 CWE 聚合型互补——
# 聚合型已被证伪为模型能力上限无区分度，检索型预期有区分度）；
# qa_2 = HotpotQA 两跳问题（RULER 官方 qa_2 口径）。
# needle 格式（RULER 官方）：
#   "One of the special magic questions for {word} is: {q}
#    The special magic answer for {word} is: {a}."
# 问 "What is the special magic answer for {qword} mentioned in the
# provided text?"，评分 = 答案（或别名）出现在输出中。
# 用法同 test_tli_ruler.py（RULER_SEED / RULER_TAG / RULER_N）。
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
OUT_JSON = "/home/wangyuanshuo02/sglang/tli_ruler_qa_results.json"
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


def build_qa_pool(name):
    """LongBench QA 子集 → (question, answers) 池（答案 ≤30 字符）。
    triviaqa input = "Passage:\n...\nQuestion:\n{q}\nAnswer:\n"（取中段）；
    hotpotqa input 即问题本身。"""
    rows = [json.loads(l) for l in open(f"{DATA_DIR}/{name}.jsonl")]
    qa = []
    for r in rows:
        q = (r.get("input") or "").strip()
        if "Question:\n" in q and "\nAnswer:" in q:
            q = q.split("Question:\n", 1)[1].split("\nAnswer:", 1)[0].strip()
        ans = r.get("answers") or []
        ans = [a for a in ans if a and len(a) <= 30]
        if q and ans and 10 <= len(q) <= 200:
            qa.append((q.rstrip("?") + "?", ans))
    return qa


def insert_needles(haystack, enc, target_depth_frac, needles):
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

    for tname, src in (("qa1", "triviaqa"), ("qa2", "hotpotqa")):
        qa_pool = build_qa_pool(src)
        rng.shuffle(qa_pool)
        print(f"[{tname}] {len(qa_pool)} QA pairs from {src}")
        samples = []
        pi, qi = 0, 0
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
            keys = rng.sample(KEY_POOL, 20)
            needles, qas = [], []
            for k in keys:
                q, ans = qa_pool[qi % len(qa_pool)]
                qi += 1
                needles.append(
                    f"One of the special magic questions for {k} is: {q} "
                    f"The special magic answer for {k} is: {ans[0]}."
                )
                qas.append(ans)
            depth = rng.choice([15, 35, 55, 75, 90]) / 100
            prompt = insert_needles(haystack, enc, depth, needles)
            qkey = keys[0]
            prompt += (
                f"\n\nWhat is the special magic answer for {qkey} "
                "mentioned in the provided text?"
            )
            samples.append({"ans": qas[0], "prompt": prompt})
        tasks[tname] = samples

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
    tag = f"qa_{backend}"
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
            # 评分：任一答案别名出现在输出中（RULER 官方 substring 口径）
            ok = any(a.lower() in text.lower() for a in s["ans"])
            rows.append({"ok": ok, "ans": s["ans"][:3], "out": text.strip()[:80],
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
