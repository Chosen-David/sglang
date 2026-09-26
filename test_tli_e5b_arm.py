# E5b 定界臂：sglang 平台差 vs tli 配置差（musique 200 条）
# 用法：CUDA_VISIBLE_DEVICES=x BACKEND=triton python3 test_tli_e5b_arm.py   → 平台差（dense）
#       CUDA_VISIBLE_DEVICES=x BACKEND=tli FAR=512 python3 test_tli_e5b_arm.py → far_tokens 对齐主表
# 与 test_tli_dyngate_e5b.py 完全同口径（官方模板/31500 token 中间截断/max_gen）
import json
import os
import sys
import warnings

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
warnings.filterwarnings("ignore")

BACKEND = os.environ.get("BACKEND", "triton")
FAR = os.environ.get("FAR", "")  # tli 时覆盖 far_tokens（空=默认 256）
if FAR:
    # 须在 Engine（scheduler spawn 子进程）初始化前设 env
    os.environ["SGLANG_TLI_FAR_TOKENS"] = FAR
TAG = os.environ.get("TAG", BACKEND if not FAR else f"{BACKEND}_far{FAR}")
TASK = os.environ.get("TASK", "musique")
MAXGEN = {"musique": 32, "qasper": 128, "multifieldqa_en": 64}[TASK]
OUT_DIR = f"/home/wangyuanshuo02/sglang/pred_e5b_{TAG}"
MAXLEN = 31500
BATCH = 8
_TPL = json.load(open(
    "/home/wangyuanshuo02/two-level-attention/benchmark/LongBench/config/dataset2prompt.json"))
_CHAT_TPL = "<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\nmind\n\n\n\n"


def main():
    from sglang import Engine

    MODEL = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B"
    eng = Engine(
        model_path=MODEL,
        attention_backend=BACKEND,
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
    out_path = f"{OUT_DIR}/{TASK}.jsonl"
    done = sum(1 for _ in open(out_path)) if os.path.exists(out_path) else 0
    if done >= 200:
        print(f"[{TAG}] already {done}, skip", flush=True)
        eng.shutdown()
        return
    rows = [json.loads(l) for l in open(
        f"/home/wangyuanshuo02/datasets/LongBench/data/{TASK}.jsonl")][:200]
    fout = open(out_path, "a")
    prompts, metas = [], []
    eng.generate(["hello"], sampling_params={"temperature": 0.0, "max_new_tokens": 1})
    for i in range(done, len(rows)):
        json_obj = rows[i]
        prompt = _TPL[TASK].format(**{k: json_obj.get(k, "") for k in ("context", "input")})
        ids = tok(prompt, truncation=False)["input_ids"]
        if len(ids) > MAXLEN:
            half = MAXLEN // 2
            prompt = tok.decode(ids[:half], skip_special_tokens=True) \
                + tok.decode(ids[-half:], skip_special_tokens=True)
        prompt = _CHAT_TPL.format(p=prompt)
        prompts.append(prompt)
        metas.append({
            "answers": json_obj["answers"],
            "all_classes": json_obj["all_classes"],
            "length": json_obj["length"],
        })
        if len(prompts) == BATCH or i == len(rows) - 1:
            outs = eng.generate(prompts, sampling_params={
                "temperature": 0.0, "max_new_tokens": MAXGEN})
            for m, o in zip(metas, outs):
                m["pred"] = o["text"]
                fout.write(json.dumps(m, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"[{TAG}/{TASK}] {i+1}/{len(rows)} done", flush=True)
            prompts, metas = [], []
    fout.close()
    eng.shutdown()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
