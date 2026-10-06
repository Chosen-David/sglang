# 30B tli 稀疏 prefill 质量崩坏诊断（#58）：长 prompt 下 tli 输出 "2 2 2 2..."，
# triton 正常。二分定位：prefill 稀疏路径（select_batched）vs decode 增量路径。
# 方法：max_new_tokens=1（纯 prefill），分别用 dense_threshold=999999 但
# POOL_S_CAP 联动问题 → 换手动的分块法不行。直接对比「长 prompt prefill-only
# 的 next token」：tli 稀疏 vs triton。若 prefill-only 就错 → prefill 路径 bug；
# 若 prefill-only 对、decode 后崩 → decode 增量/稀疏 bug。
import json
import sys
import warnings

if __name__ == "__main__":
    sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
    warnings.filterwarnings("ignore")
    from sglang import Engine

    backend = sys.argv[1] if len(sys.argv) > 1 else "tli"
    n_new = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    MODEL = "/home/wangyuanshuo02/models/Qwen3-30B-A3B"

    eng = Engine(
        model_path=MODEL,
        attention_backend=backend,
        dtype="bfloat16",
        device="cuda",
        tp_size=1,
        mem_fraction_static=0.7,
        trust_remote_code=True,
        disable_radix_cache=True,
        watchdog_timeout=1800,
        cuda_graph_config={"decode": {"backend": "disabled"}, "prefill": {"backend": "disabled"}},
    )
    rows = [json.loads(l) for l in open(
        "/home/wangyuanshuo02/datasets/LongBench/data/narrativeqa.jsonl")]
    ctx = sorted(rows, key=lambda r: -len(r.get("context", "")))[0]["context"][:32000]
    prompt = ctx + "\n\nSummarize the above text in one sentence:"
    outs = eng.generate([prompt],
                        sampling_params={"temperature": 0.0, "max_new_tokens": n_new})
    print(f"[30b-{backend}-n{n_new}]: {json.dumps([o['text'][:150] for o in outs], ensure_ascii=False)}")
    eng.shutdown()
    print("DONE")
