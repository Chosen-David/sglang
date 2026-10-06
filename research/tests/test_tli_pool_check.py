# KV pool 上限仲裁（#57）：bs16×40K 需 95GB > 30K 的 71GB，验证 mem_fraction
# 各档位下 tli 后端的 max_total_num_tokens 是否容得下 16×(S+256)。
# 用法：CUDA_VISIBLE_DEVICES=1 python3 test_tli_pool_check.py [mem_fraction]
import sys

if __name__ == "__main__":
    from sglang import Engine

    mf = float(sys.argv[1]) if len(sys.argv) > 1 else 0.7
    eng = Engine(
        model_path="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B",
        attention_backend="tli", dtype="bfloat16", device="cuda",
        mem_fraction_static=mf, trust_remote_code=True, disable_radix_cache=True,
        cuda_graph_config={"decode": {"backend": "full", "bs": [16]}, "prefill": {"backend": "disabled"}},
    )
    # offline Engine: scheduler 子进程内的 model_runner 持有 pool；直接解析不行，
    # 但引擎 init 日志会打 max_total_num_tokens——靠 stderr 日志即可。
    eng.shutdown()
    print(f"[pool_check] mem_fraction={mf} done (see log above)")
