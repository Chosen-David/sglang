# Qwen3-30B-A3B-Instruct-2507（256K context）上 tli vs triton smoke（#58）。
# 目的：S=64K/128K 收益区点的载体模型（Qwen3-8B 上限 40960）。
# 验证：①tli backend 在 qwen3_moe 架构上能跑；②短 prompt + 8K 稀疏路径
# 输出与 triton 语义等价；③TP2 也过（收益区点要 TP2 装下 KV）。
# 用法：CUDA_VISIBLE_DEVICES=0[,1] python3 test_tli_30b_smoke.py [tli|triton] [tp]
import json
import sys
import warnings

if __name__ == "__main__":
    sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
    warnings.filterwarnings("ignore")
    from sglang import Engine

    backend = sys.argv[1] if len(sys.argv) > 1 else "tli"
    tp = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    MODEL = ("/home/wangyuanshuo02/"
             "models/Qwen3-30B-A3B")

    eng = Engine(
        model_path=MODEL,
        attention_backend=backend,
        dtype="bfloat16",
        device="cuda",
        tp_size=tp,
        mem_fraction_static=float(__import__("os").environ.get("VAR_MEM_FRAC", "0.85")),
        trust_remote_code=True,
        disable_radix_cache=True,
        watchdog_timeout=1800,
        cuda_graph_config={"decode": {"backend": "full", "bs": [4]}, "prefill": {"backend": "disabled"}},
    )
    outs = eng.generate(
        ["The capital of France is", "水的化学式是"],
        sampling_params={"temperature": 0.0, "max_new_tokens": 24},
    )
    print(f"[30b-{backend}-tp{tp}] short: {json.dumps([o['text'] for o in outs], ensure_ascii=False)}")

    # 8K 稀疏路径（dense_threshold 应触发两级 select；narrativeqa context 截 30K chars ≈ 7.5K tok）
    rows = [json.loads(l) for l in open(
        "/home/wangyuanshuo02/datasets/LongBench/data/narrativeqa.jsonl")]
    ctx = sorted(rows, key=lambda r: -len(r.get("context", "")))[0]["context"][:32000]
    out = eng.generate([ctx + "\n\nSummarize the above text in one sentence:"],
                       sampling_params={"temperature": 0.0, "max_new_tokens": 48})
    print(f"[30b-{backend}-tp{tp}] long: {json.dumps([out[0]['text'][:120]], ensure_ascii=False)}")
    eng.shutdown()
    print(f"[30b-{backend}-tp{tp}] OK")
