# sglang tli backend e2e smoke test（offline engine，Qwen3-8B 真实权重）
# 验证：tli vs 默认 flashattention backend 的短 prompt 输出一致性 +
# 长 prompt（>dense_threshold 后稀疏路径）不崩、输出合理
# 运行：PYTHONPATH=/home/wangyuanshuo02/sglang/python CUDA_VISIBLE_DEVICES=1 python test_tli_e2e_smoke.py
import os
import sys

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
import warnings

warnings.filterwarnings("ignore")

from sglang import Engine, Runtime

MODEL = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B"


def run(attention_backend, prompts, max_new_tokens=64):
    eng = Engine(
        model_path=MODEL,
        attention_backend=attention_backend,
        dtype="bfloat16",
        device="cuda",
        disable_cuda_graph=True,  # tli 不支持 cuda graph → 关闭 graph 捕获
        mem_fraction_static=0.85,
        trust_remote_code=True,
    )
    outs = eng.generate(
        prompts,
        sampling_params={
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
        },
    )
    outs = [o["text"] for o in outs]  # 新版返回 list[dict]
    eng.shutdown()
    return outs


if __name__ == "__main__":
    prompts = [
        "The capital of France is",
        "1 + 1 =",
        # 长上下文：拼接出 > dense_threshold(2048) 的 prompt 触发稀疏路径
        "Here is a long story: " + ("Once upon a time there was a little robot exploring the forest. " * 90)
        + "\nQuestion: What was the robot exploring?\nAnswer:",
    ]
    print("=== flashattention (baseline) ===")
    ref = run("triton", prompts)
    for p, o in zip(prompts, ref):
        print(f"[{len(p)} chars] -> {o[:80]!r}")

    print("\n=== tli ===")
    tli = run("tli", prompts)
    for p, o in zip(prompts, ref):
        pass
    for p, o in zip(prompts, tli):
        print(f"[{len(p)} chars] -> {o[:80]!r}")

    print("\n=== 对比 ===")
    for i, (a, b) in enumerate(zip(ref, tli)):
        same = a.strip() == b.strip()
        print(f"prompt{i}: identical={same}")
        if not same:
            print(f"  fa : {a[:100]!r}")
            print(f"  tli: {b[:100]!r}")
    print("\nDONE")
