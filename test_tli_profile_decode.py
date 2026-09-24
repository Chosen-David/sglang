# M3-b: decode 开销归因（TLI_PROFILE_TIMING=1，单请求 9.9K token narrativeqa）
# 运行：cd /home/wangyuanshuo02/sglang && \
#   SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=true SGLANG_IS_FLASHINFER_AVAILABLE=false \
#   SGLANG_ENABLE_JIT_DEEPGEMM=0 CUDA_VISIBLE_DEVICES=1 TLI_PROFILE_TIMING=1 \
#   PYTHONPATH=/home/wangyuanshuo02/.local/pylibs_tf512:/home/wangyuanshuo02/sglang/python \
#   python test_tli_profile_decode.py
import json
import sys
import time
import warnings

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
warnings.filterwarnings("ignore")

from sglang import Engine

MODEL = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B"
DATA = "/home/wangyuanshuo02/datasets/LongBench/data/narrativeqa.jsonl"


def main():
    rows = [json.loads(l) for l in open(DATA)]
    rows.sort(key=lambda r: -len(r["context"]))
    p = rows[0]["context"][:32000] + "\n\nSummarize the above text in one sentence:"

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
    t0 = time.time()
    eng.generate([p], sampling_params={"temperature": 0.0, "max_new_tokens": 129})
    print(f"\nTOTAL wall: {time.time() - t0:.2f}s")
    eng.shutdown()


if __name__ == "__main__":
    main()
