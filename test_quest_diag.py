# Quest e2e 诊断：分级强制「选全页 ⇒ 数学等价 dense」，对比 triton 输出。
# L1: S=800, dense_threshold=256 → 稀疏但全选（无 chunk）
# L2: S=3500, topk=64(4096≥S) → 全选 + chunked prefill + 簿记链
# L3: S=3500, topk=16（真实 Quest 口径）——质量级对比（不逐位）
import os
import sys

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
import warnings

warnings.filterwarnings("ignore")

from sglang import Engine

MODEL = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B"

PROMPTS = {
    "L1_s800_allok": (
        "The capital of France is Paris. The capital of Japan is Tokyo. "
        * 20
        + "\nQuestion: What is the capital of Japan?\nAnswer:"
    ),
    "L2_s3500_allok": (
        "Alice and Bob live in a red house. Charlie lives in a blue house. "
        * 180
        + "\nQuestion: What color is Alice's house?\nAnswer:"
    ),
    "L3_s3500_topk16": (
        "Alice and Bob live in a red house. Charlie lives in a blue house. "
        * 180
        + "\nQuestion: What color is Alice's house?\nAnswer:"
    ),
}


def run(backend, envs, names):
    old = {k: os.environ.get(k) for k in envs}
    os.environ.update(envs)
    try:
        eng = Engine(
            model_path=MODEL,
            attention_backend=backend,
            dtype="bfloat16",
            device="cuda",
            tp_size=1,
            mem_fraction_static=0.85,
            trust_remote_code=True,
            disable_radix_cache=True,
            cuda_graph_config={"decode": {"backend": "disabled"},
                               "prefill": {"backend": "disabled"}},
        )
        outs = {}
        for n in names:
            o = eng.generate([PROMPTS[n]],
                             sampling_params={"temperature": 0.0,
                                              "max_new_tokens": 16})
            outs[n] = o[0]["text"]
            print(f"[{backend}/{n}] -> {outs[n][:60]!r}", flush=True)
        eng.shutdown()
        return outs
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


if __name__ == "__main__":
    names = ["L1_s800_allok", "L2_s3500_allok", "L3_s3500_topk16"]
    # dense 参考（triton）
    ref = run("triton", {}, names)
    # L1：稀疏全选、无 chunk
    q1 = run("quest", {"SGLANG_QUEST_DENSE_THRESHOLD": "256"}, names)
    # L2：全选 + chunked prefill（topk=64 页=4096 token ≥ S）
    q2 = run("quest", {"SGLANG_QUEST_DENSE_THRESHOLD": "256",
                       "SGLANG_QUEST_TOPK_PAGES": "64"}, names)
    print("\n==== 对比 ====")
    for n in names:
        print(f"{n}:")
        print(f"  triton        : {ref[n][:60]!r}")
        print(f"  quest L1      : {q1[n][:60]!r}")
        print(f"  quest L2(64p) : {q2[n][:60]!r}")
