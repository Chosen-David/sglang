# M9 e2e smoke：tli 选择路径（δ=16）vs PCA 投影路径（r=16）输出一致性
# 真实 narrativeqa 32K 字符长上下文（≈9.9K token > dense_threshold=2048，
# 稀疏 prefill + 稀疏 decode 全路径覆盖）。一进程一 Engine，结果落盘对比。
# 运行：CUDA_VISIBLE_DEVICES=1 python test_tli_m9_e2e.py            # baseline 落盘
#       CUDA_VISIBLE_DEVICES=1 SGLANG_TLI_PROJ_BASIS=... python ...  # PCA 落盘+对比
import json
import os
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
    mode = "pca" if os.environ.get("SGLANG_TLI_PROJ_BASIS") else "sel"
    out = run(prompt)
    print(f"{mode:>4s}:", repr(out[:120]))
    with open(f"/tmp/tli_m9_smoke_{mode}.txt", "w") as f:
        f.write(out)
    if mode == "pca":
        try:
            a = open("/tmp/tli_m9_smoke_sel.txt").read()
            print("identical:", a.strip() == out.strip())
            if a.strip() != out.strip():
                # 语义级对比（bf16 累积顺序噪声预期）
                import difflib

                d = list(difflib.unified_diff(a.split(), out.split(), lineterm=""))
                print("diff lines:", len(d))
                print("\n".join(d[:20]))
        except FileNotFoundError:
            print("baseline 未生成，先跑无 env 版本")
