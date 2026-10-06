# E67 升 τ 真判决 e2e（dyngate τ=0.01 近 no-op 复核结论的后续）：
# sglang 口径 τ = per-layer far mass（行×Hkv 平均；8B 实测安全 ~0.001-0.008、
# 多跳 ~0.015-0.026）。τ0.01 复核实测触发率仅 1.1%（近 no-op）。
# e2e 判决：far-heavy 多跳任务（musique/qasper）上 τ 上探（0.005/0.015）后
# 精度损失 vs 跳层收益。对照 = pred_dyngate_off（27.57 / 40.37）。
# 用法：CUDA_VISIBLE_DEVICES=0 ARM=tau01 python3 test_tli_e67_tau.py
#       CUDA_VISIBLE_DEVICES=1 ARM=tau02 python3 test_tli_e67_tau.py
# τ 由 SGLANG_TLI_DYN_GATE_THRESH 控制（config.py 默认 0.01）。
import json
import os
import sys
import warnings

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
warnings.filterwarnings("ignore")

ARM = os.environ.get("ARM", "tau01")            # tau01 / tau02
# τ 口径 = per-layer far mass（行×Hkv 平均；8B 实测安全任务 ~0.001-0.008、
# 多跳 ~0.015-0.026）——E67 trace 侧 0.1/0.2 是「占比」口径不通用，
# 本口径下 0.1/0.2 会跳掉几乎所有层（含多跳 far-heavy 层），无信息量。
# 取口径内梯度：0.005（安全任务内）/ 0.010（≈默认）/ 0.015（安全/多跳边界）。
THRESH = {"tau01": "0.005", "tau02": "0.015"}[ARM]
os.environ["SGLANG_TLI_DYN_GATE"] = "1"
os.environ["SGLANG_TLI_DYN_GATE_THRESH"] = THRESH

TASKS = ["musique", "qasper"]                   # far-heavy 多跳：错误跳层最伤的任务
MAXLEN = 31500
MAXGEN = {"musique": 32, "qasper": 128}
OUT_DIR = f"/home/wangyuanshuo02/sglang/pred_e67_{ARM}"
_TPL = json.load(open(
    "/home/wangyuanshuo02/two-level-attention/benchmark/LongBench/config/dataset2prompt.json"))
BATCH = 4  # 8 会在 musique 8×9K extend 的 select_batched einsum 中间张量 OOM（7.62GiB 超顶）
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
        prompts, metas = [], []
        for i in range(done, len(rows)):
            json_obj = rows[i]
            prompt = _TPL[task].format(**{k: json_obj.get(k, "") for k in ("context", "input")})
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
                print(f"[{task}] {i+1}/{len(rows)} done", flush=True)
                prompts, metas = [], []
        fout.close()
    eng.shutdown()
    print(f"E67 {ARM} DONE", flush=True)


if __name__ == "__main__":
    main()
