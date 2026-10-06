# #60 全量 E5b 多跳复验：musique/qasper/multifieldqa_en 三任务 200 样本全量，
# gate-on vs gate-off 双臂。与 E5b 主表同口径（官方模板 + 31500 token 中间截断 +
# dataset2maxlen 的 max_gen）。输出 jsonl 与 benchmark.LongBench.eval.scorer 对齐。
# 用法：CUDA_VISIBLE_DEVICES=0 SGLANG_TLI_DYN_GATE=1 ARM=on python3 test_tli_dyngate_e5b.py
#       CUDA_VISIBLE_DEVICES=1 SGLANG_TLI_DYN_GATE=0 ARM=off python3 test_tli_dyngate_e5b.py
import json
import os
import sys
import warnings

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
warnings.filterwarnings("ignore")

TASKS = ["musique", "qasper", "multifieldqa_en"]
MAXLEN = 31500
MAXGEN = {"musique": 32, "qasper": 128, "multifieldqa_en": 64}
ARM = os.environ.get("ARM", "on")
OUT_DIR = f"/home/wangyuanshuo02/sglang/pred_dyngate_{ARM}"
_TPL = json.load(open(
    "/home/wangyuanshuo02/two-level-attention/benchmark/LongBench/config/dataset2prompt.json"))
BATCH = 8  # 一次 generate 的样本数（prefill 内部自动 batch）

# E5b 口径修正（监督轮发现）：pred.py 的 build_chat 对 Qwen3 加 chat template
# （截断发生在 template 前）；裸 prompt 会使 musique 绝对分 8.19 vs 主表 32.28。
# 字符串与 pred.py build_chat Qwen3 分支逐字符一致（含 mind\n\n\n\n 后缀）。
_CHAT_TPL = "<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\nmind\n\n\n\n"


def build_chat(prompt):
    return _CHAT_TPL.format(p=prompt)


def main():
    from sglang import Engine

    MODEL = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B"
    eng = Engine(
        model_path=MODEL,
        attention_backend="tli",
        dtype="bfloat16",
        device="cuda",
        tp_size=1,
        mem_fraction_static=0.7,
        trust_remote_code=True,
        disable_radix_cache=True,
        watchdog_timeout=1800,
        cuda_graph_config={"decode": {"backend": "disabled"}, "prefill": {"backend": "disabled"}},
    )
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    # 预热吸收 Triton JIT
    eng.generate(["hello"], sampling_params={"temperature": 0.0, "max_new_tokens": 1})
    for task in TASKS:
        out_path = f"{OUT_DIR}/{task}.jsonl"
        done = 0
        if os.path.exists(out_path):
            done = sum(1 for _ in open(out_path))
        if done >= 200:
            print(f"[{task}] already {done}, skip", flush=True)
            continue
        rows = [json.loads(l) for l in open(
            f"/home/wangyuanshuo02/datasets/LongBench/data/{task}.jsonl")][:200]
        fout = open(out_path, "a")
        prompts = []
        metas = []
        for i in range(done, len(rows)):
            json_obj = rows[i]
            prompt = _TPL[task].format(**{k: json_obj.get(k, "") for k in ("context", "input")})
            # 与 E5b 同口径：中间截断到 max_length token
            ids = tok(prompt, truncation=False)["input_ids"]
            if len(ids) > MAXLEN:
                half = MAXLEN // 2
                prompt = tok.decode(ids[:half], skip_special_tokens=True) \
                    + tok.decode(ids[-half:], skip_special_tokens=True)
            prompt = build_chat(prompt)
            prompts.append(prompt)
            metas.append({
                "answers": json_obj["answers"],
                "all_classes": json_obj["all_classes"],
                "length": json_obj["length"],
            })
            if len(prompts) == BATCH or i == len(rows) - 1:
                outs = eng.generate(prompts, sampling_params={
                    "temperature": 0.0, "max_new_tokens": MAXGEN[task]})
                for m, o in zip(metas, outs):
                    m["pred"] = o["text"]
                    fout.write(json.dumps(m, ensure_ascii=False) + "\n")
                fout.flush()
                n = done + len(metas) + (i - done - len(metas) + 1)
                print(f"[{task}] {i+1}/{len(rows)} done", flush=True)
                prompts, metas = [], []
        fout.close()
    eng.shutdown()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
