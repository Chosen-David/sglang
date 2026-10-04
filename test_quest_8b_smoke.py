# 审稿 C3：Quest backend 精度冒烟（Qwen3-8B，hotpotqa 前 5 样本，单请求逐条）
# 口径对齐 two-level-attention pred_1024 的 quest_64_16 臂：
#   - prompt = LongBench hotpotqa 模板 + Qwen3 build_chat 格式（pred.py 原文）
#   - max_new_tokens = 32（dataset2maxgen），greedy
#   - 参考（two-level quest_64_16 hotpotqa）：full200 qa_f1 = 0.46，
#     first5 均值 0.43（管线不同不必逐位，量级同档即通过）
# 用法（等 GPU 空闲后）：
#   SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=true SGLANG_IS_FLASHINFER_AVAILABLE=false \
#   SGLANG_ENABLE_JIT_DEEPGEMM=0 CUDA_VISIBLE_DEVICES=1 \
#   PYTHONPATH=/home/wangyuanshuo02/.local/pylibs_tf512:/home/wangyuanshuo02/sglang/python \
#   python3 test_quest_8b_smoke.py            # quest 臂
#   BACKEND=triton python3 test_quest_8b_smoke.py   # dense 对照臂（可选）
import json
import os
import re
import string
import sys
import warnings

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
warnings.filterwarnings("ignore")

BACKEND = os.environ.get("BACKEND", "quest")
MODEL = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B"
N_SAMPLES = 5
OUT_JSON = f"/tmp/quest_smoke_{BACKEND}.json"

# ---- LongBench qa_f1（英文；与 two-level metrics.py 同式，去 jieba 依赖）----


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return " ".join(remove_articles(remove_punc(s.lower())).split())


def f1_score(prediction, ground_truth):
    np_, ng = normalize_answer(prediction), normalize_answer(ground_truth)
    pt, gt = np_.split(), ng.split()
    common = [t for t in pt if t in gt]
    if not common or not pt or not gt:
        return 0.0
    precision = len(common) / len(pt)
    recall = len(common) / len(gt)
    return 2 * precision * recall / (precision + recall)


def qa_f1(pred, answers):
    return max(f1_score(pred, a) for a in answers)


def main():
    from sglang import Engine

    prompt_tpl = (
        "Answer the question based on the given passages. Only give me the "
        "answer and do not output any other words.\n\nThe following are given "
        "passages.\n{context}\n\nAnswer the question based on the given "
        "passages. Only give me the answer and do not output any other words."
        "\n\nQuestion: {input}\nAnswer:"
    )
    # Qwen3 build_chat（two-level pred.py 原文，含空 think 块 hack）
    chat_tpl = ("<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"
              "<think>\n\n</think>\n\n")

    rows = [
        json.loads(l)
        for l in open(
            "/home/wangyuanshuo02/datasets/LongBench/data/hotpotqa.jsonl"
        )
    ][:N_SAMPLES]

    eng = Engine(
        model_path=MODEL,
        attention_backend=BACKEND,
        dtype="bfloat16",
        device="cuda",
        tp_size=1,
        mem_fraction_static=0.85,
        trust_remote_code=True,
        disable_radix_cache=True,
        watchdog_timeout=1800,
        cuda_graph_config={
            "decode": {"backend": "disabled"},
            "prefill": {"backend": "disabled"},
        },
    )
    # 预热（吸收 JIT / 首次建池）
    eng.generate(["hello"], sampling_params={"temperature": 0.0, "max_new_tokens": 1})

    preds, f1s = [], []
    for r in rows:
        prompt = prompt_tpl.format(context=r["context"], input=r["input"])
        # 超长截断口径略过（前 5 样本 ~9-13K token < 31500，无截断必要）
        prompt = chat_tpl.format(p=prompt)
        outs = eng.generate(
            [prompt],
            sampling_params={"temperature": 0.0, "max_new_tokens": 32},
        )
        pred = outs[0]["text"].strip()
        # post_process：截断到首个换行/结尾符（pred.py 口径近似）
        for stopper in ("<|im_end|>", "\n"):
            if stopper in pred:
                pred = pred.split(stopper)[0].strip()
        preds.append(pred)
        f1s.append(qa_f1(pred, r["answers"]))
        print(
            f"[{BACKEND}] pred={pred[:50]!r} gold={r['answers'][:2]} f1={f1s[-1]:.2f}",
            flush=True,
        )
    mean_f1 = sum(f1s) / len(f1s)
    res = {
        "backend": BACKEND,
        "n": len(rows),
        "f1_per_sample": [round(x, 3) for x in f1s],
        "f1_mean": round(mean_f1, 4),
        "preds": preds,
        "reference": "two-level quest_64_16 hotpotqa: full200=0.46 first5=0.43",
    }
    json.dump(res, open(OUT_JSON, "w"), indent=1, ensure_ascii=False)
    print(f"[{BACKEND}] first5 qa_f1 = {mean_f1:.3f}  (quest 参考 first5 = 0.43)")
    print("saved ->", OUT_JSON)
    eng.shutdown()
    print("DONE")


if __name__ == "__main__":
    main()
