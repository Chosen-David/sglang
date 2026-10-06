# tli backend TP2 正确性 smoke（#58）：H20 双卡跑 Qwen3-8B tensor parallel 2，
# 同 prompt 双 backend 对拍（短 prompt 全链路 + 长稀疏路径）。
# 用法：CUDA_VISIBLE_DEVICES=0,1 python3 test_tli_tp2_smoke.py [tli|triton]
import json
import sys
import warnings

if __name__ == "__main__":
    sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
    warnings.filterwarnings("ignore")
    from sglang import Engine

    backend = sys.argv[1] if len(sys.argv) > 1 else "tli"

    eng = Engine(
        model_path="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B",
        attention_backend=backend,
        dtype="bfloat16",
        device="cuda",
        tp_size=2,
        mem_fraction_static=0.85,
        trust_remote_code=True,
        disable_radix_cache=True,
        watchdog_timeout=1800,
        cuda_graph_config={"decode": {"backend": "full", "bs": [4]}, "prefill": {"backend": "disabled"}},
    )
    outs = eng.generate(
        ["The capital of France is", "水的化学式是"],
        sampling_params={"temperature": 0.0, "max_new_tokens": 24},
    )
    texts = [o["text"] for o in outs]
    print(f"[tp2-{backend}] short: {json.dumps(texts, ensure_ascii=False)}")

    # 长稀疏路径（dense_threshold 触发两级 select；narrativeqa 真实 context）
    import glob as _g

    rows = [json.loads(l) for l in open(
        "/home/wangyuanshuo02/datasets/LongBench/data/narrativeqa.jsonl")]
    ctx = sorted(rows, key=lambda r: -len(r.get("context", "")))[0]["context"][:32000]
    long_prompt = ctx + "\n\nSummarize the above text in one sentence:"
    out = eng.generate([long_prompt],
                       sampling_params={"temperature": 0.0, "max_new_tokens": 48})
    print(f"[tp2-{backend}] long: {json.dumps([out[0]['text'][:120]], ensure_ascii=False)}")
    eng.shutdown()
    print(f"[tp2-{backend}] OK")
