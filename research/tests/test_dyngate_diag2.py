# #60 口径 bug 诊断 2：sglang str vs input_ids 输入对照（特殊 token 编码差异假设）
import json
import warnings

warnings.filterwarnings("ignore")
from transformers import AutoTokenizer

MODEL = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B"
_TPL = json.load(open(
    "/home/wangyuanshuo02/two-level-attention/benchmark/LongBench/config/dataset2prompt.json"))
MIND_TPL = "<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\nmind\n\n\n\n"


def main():
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    row = json.loads(open(
        "/home/wangyuanshuo02/datasets/LongBench/data/musique.jsonl").readline())
    prompt = MIND_TPL.format(p=_TPL["musique"].format(
        **{k: row.get(k, "") for k in ("context", "input")}))
    ids = tok(prompt)["input_ids"]
    print("n_tokens:", len(ids), "im_start id:",
          tok.convert_tokens_to_ids("<|im_start|>"), "first5:", ids[:5], flush=True)

    from sglang import Engine
    eng = Engine(model_path=MODEL, attention_backend="tli", dtype="bfloat16",
                 device="cuda", tp_size=1, mem_fraction_static=0.7,
                 trust_remote_code=True, disable_radix_cache=True,
                 disable_cuda_graph=True, watchdog_timeout=1800)
    eng.generate(["warmup"], sampling_params={"temperature": 0.0, "max_new_tokens": 1})
    o1 = eng.generate([prompt], sampling_params={"temperature": 0.0, "max_new_tokens": 32})
    print("[str]   ", repr(o1[0]["text"][:120]), flush=True)
    o2 = eng.generate(input_ids=[ids], sampling_params={"temperature": 0.0, "max_new_tokens": 32})
    print("[ids]   ", repr(o2[0]["text"][:120]), flush=True)
    eng.shutdown()


if __name__ == "__main__":
    main()
