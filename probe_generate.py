# 探针：确认新版 Engine.generate 返回结构（stdin 无法被 spawn，须落盘）
import warnings

warnings.filterwarnings("ignore")

from sglang import Engine

if __name__ == "__main__":
    eng = Engine(
        model_path="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B",
        attention_backend="triton", dtype="bfloat16", device="cuda",
        disable_cuda_graph=True, mem_fraction_static=0.85, trust_remote_code=True,
    )
    outs = eng.generate(
        ["The capital of France is"],
        sampling_params={"temperature": 0.0, "max_new_tokens": 8},
    )
    print("TYPE:", type(outs))
    print("OUT:", [str(o)[:200] for o in outs] if isinstance(outs, list) else str(outs)[:300])
    eng.shutdown()
