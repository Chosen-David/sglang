# M5 e2e smoke：CUDA graph replay vs eager 输出一致性（真实 narrativeqa 长上下文）
# 一进程一 Engine：SGLANG_M5_MODE=eager / graph 分别跑，结果落盘 /tmp 再对比。
# 运行：
#   SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=true SGLANG_IS_FLASHINFER_AVAILABLE=false \
#   SGLANG_ENABLE_JIT_DEEPGEMM=0 CUDA_VISIBLE_DEVICES=1 SGLANG_M5_MODE=graph \
#   SGLANG_TLI_POOL_S_CAP=12288 \
#   PYTHONPATH=/home/wangyuanshuo02/.local/pylibs_tf512:/home/wangyuanshuo02/sglang/python \
#   python test_tli_m5_smoke.py
import json
import os
import sys
import warnings

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
warnings.filterwarnings("ignore")

from sglang import Engine

MODEL = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B"
MODE = os.environ.get("SGLANG_M5_MODE", "graph")


def make_prompt():
    rows = [json.loads(l) for l in open("/home/wangyuanshuo02/datasets/LongBench/data/narrativeqa.jsonl")]
    rows.sort(key=lambda r: -len(r["context"]))
    # 32K chars ≈ 9.9K token > dense_threshold：稳态走图内稀疏路径
    return rows[0]["context"][:32000] + "\n\nSummarize the above text in one sentence:"


def main():
    prompt = make_prompt()
    kwargs = dict(
        model_path=MODEL,
        attention_backend="tli",
        dtype="bfloat16",
        device="cuda",
        mem_fraction_static=0.6,
        trust_remote_code=True,
        disable_radix_cache=True,
    )
    if MODE == "graph":
        # prefill 必须显式 disabled：默认 BREAKABLE 依赖 sgl_kernel.weak_ref_tensor
        # （0.3.16.post6 无此 API → ImportError → breakable 段 assert 崩）
        kwargs["cuda_graph_config"] = {
            "decode": {"backend": "full", "bs": [1, 2]},
            "prefill": {"backend": "disabled"},
        }
    else:
        kwargs["disable_cuda_graph"] = True
    eng = Engine(**kwargs)
    eng.generate(["warmup"], sampling_params={"temperature": 0.0, "max_new_tokens": 4})
    out = eng.generate([prompt], sampling_params={"temperature": 0.0, "max_new_tokens": 64})[0]["text"]
    eng.shutdown()
    path = f"/tmp/tli_m5_smoke_{MODE}.txt"
    open(path, "w").write(out)
    print(f"[{MODE}] {len(out)} chars: {out[:100]!r}")
    print("saved", path)


if __name__ == "__main__":
    main()
