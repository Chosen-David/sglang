# e2e smoke：tli eager vs L1+L2 双 kernel 输出一致性（真实 narrativeqa 长上下文）
# 运行：CUDA_VISIBLE_DEVICES=0 SGLANG_TLI_L2_SMOKE=1 python test_tli_l2_e2e_smoke.py
import json
import sys
import warnings

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
warnings.filterwarnings("ignore")

from sglang import Engine

MODEL = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B"


def make_prompt():
    rows = [json.loads(l) for l in open("/home/wangyuanshuo02/datasets/LongBench/data/narrativeqa.jsonl")]
    rows.sort(key=lambda r: -len(r["context"]))
    p = rows[0]["context"][:32000] + "\n\nSummarize the above text in one sentence:"
    print(f"prompt chars={len(p)}")
    return p


def run(prompt):
    eng = Engine(
        model_path=MODEL,
        attention_backend="tli",
        dtype="bfloat16",
        device="cuda",
        disable_cuda_graph=True,
        mem_fraction_static=0.6,
        trust_remote_code=True,
        disable_radix_cache=True,
    )
    eng.generate(["warmup"], sampling_params={"temperature": 0.0, "max_new_tokens": 4})
    out = eng.generate([prompt], sampling_params={"temperature": 0.0, "max_new_tokens": 64})[0]["text"]
    eng.shutdown()
    return out


if __name__ == "__main__":
    prompt = make_prompt()
    import os

    if os.environ.get("SGLANG_TLI_L1_KERNEL", "0") != "1" or os.environ.get(
        "SGLANG_TLI_L2_KERNEL", "0"
    ) != "1":
        a = run(prompt)  # tli eager
        print("eager  :", repr(a[:120]))
        with open("/tmp/tli_l2_smoke_eager.txt", "w") as f:
            f.write(a)
    if os.environ.get("SGLANG_TLI_L2_KERNEL", "0") == "1":
        b = run(prompt)  # tli kernel
        print("kernels:", repr(b[:120]))
        with open("/tmp/tli_l2_smoke_kernel.txt", "w") as f:
            f.write(b)
    try:
        a = open("/tmp/tli_l2_smoke_eager.txt").read()
        b = open("/tmp/tli_l2_smoke_kernel.txt").read()
        print("identical:", a.strip() == b.strip())
    except FileNotFoundError:
        pass
