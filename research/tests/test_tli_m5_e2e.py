# M5 e2e: CUDA graph decode 基准（vs M4 phase-3 无图基线 + triton 图基线）
# M4 归因结论：批量化后剩余固定项 ~125 ms/step = ~35 launch/层 × 36 层的纯
# 调度开销——CUDA graph 是唯一解。本脚本验证 M5 图化后的实际收益。
#
# 一进程一 Engine（scheduler 限制），CLI 选择配置，结果增量落盘：
#   cd /home/wangyuanshuo02/sglang && \
#   SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=true SGLANG_IS_FLASHINFER_AVAILABLE=false \
#   SGLANG_ENABLE_JIT_DEEPGEMM=0 CUDA_VISIBLE_DEVICES=1 SGLANG_TLI_L1_KERNEL=1 \
#   SGLANG_TLI_POOL_S_CAP=12288 \
#   PYTHONPATH=/home/wangyuanshuo02/.local/pylibs_tf512:/home/wangyuanshuo02/sglang/python \
#   python test_tli_m5_e2e.py <triton|tli> <0|1>
# 依次跑：tli 1（headline）→ tli 0（同环境复现 M4）→ triton 1（公平图基线）
#
# 关键配置：
#   - cuda_graph_config: decode=full, bs=[1,8,16,32]（R_cap=33，pool 预分配
#     33 行）；prefill 显式 disabled（隔离变量：tli 稀疏 prefill 形状动态，
#     且 prefill 延迟不是本里程碑目标）
#   - SGLANG_TLI_POOL_S_CAP=12288：pool S 维封顶（prompt ≈9.9K token +
#     65 decode）。不封顶则按 req_to_token 全宽预分配（kq fp32 ≈1KB/token/
#     行 × 33 行 × 36 层），长上下文主表须等 M6 kq 4bit
import json
import os
import sys
import time
import warnings

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
warnings.filterwarnings("ignore")

from sglang import Engine

MODEL = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B"
DATA = "/home/wangyuanshuo02/datasets/LongBench/data/narrativeqa.jsonl"
CHARS = 32000  # ≈ 9.9K token（与 M3-c/M4 同曲线同 prompt 口径）
BATCHES = [int(x) for x in os.environ.get("M5_BATCHES", "1,8,16,32").split(",")]
N_DECODE = 64
OUT_JSON = "/home/wangyuanshuo02/sglang/tli_m5_e2e_results.json"
GRAPH_CFG = {"decode": {"backend": "full", "bs": BATCHES}, "prefill": {"backend": "disabled"}}


def make_prompts(bs):
    # 与 test_tli_batch_decode.py 完全同口径：narrativeqa 唯一长 context
    # 去重（同书多问），不足时其他 LongBench 真实子集补足
    import glob

    rows = [json.loads(l) for l in open(DATA)]
    for extra in sorted(glob.glob("/home/wangyuanshuo02/datasets/LongBench/data/*.jsonl")):
        if len(rows) >= 512:
            break
        try:
            rows += [json.loads(l) for l in open(extra)]
        except Exception:
            pass
    rows.sort(key=lambda r: -len(r.get("context", "")))
    seen, uniq = set(), []
    for r in rows:
        ctx = r.get("context") or ""
        if len(ctx) < CHARS // 2:
            continue
        key = ctx[:10000]
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
        if len(uniq) >= bs:
            break
    assert len(uniq) == bs, f"only {len(uniq)} unique contexts"
    return [
        r["context"][:CHARS] + "\n\nSummarize the above text in one sentence:"
        for r in uniq
    ]


def main():
    backend = sys.argv[1] if len(sys.argv) > 1 else "tli"
    use_graph = sys.argv[2] if len(sys.argv) > 2 else "1"
    tag = f"{backend}_graph{use_graph}"

    kwargs = dict(
        model_path=MODEL,
        attention_backend=backend,
        dtype="bfloat16",
        device="cuda",
        mem_fraction_static=0.6,
        trust_remote_code=True,
        disable_radix_cache=True,  # 吞吐口径：排除 radix cache 命中差分
    )
    if use_graph == "1":
        kwargs["cuda_graph_config"] = GRAPH_CFG
    else:
        kwargs["disable_cuda_graph"] = True

    t_init = time.time()
    eng = Engine(**kwargs)
    print(f"[{tag}] engine init (含 graph capture) {time.time() - t_init:.1f}s")
    eng.generate(["warmup"], sampling_params={"temperature": 0.0, "max_new_tokens": 4})

    results = []
    for bs in BATCHES:
        prompts = make_prompts(bs)
        # prefill only（分步计时：prefill 与 decode 分离）
        t0 = time.time()
        eng.generate(prompts, sampling_params={"temperature": 0.0, "max_new_tokens": 1})
        t_prefill = time.time() - t0
        # prefill + N decode
        t0 = time.time()
        eng.generate(
            prompts,
            sampling_params={"temperature": 0.0, "max_new_tokens": 1 + N_DECODE},
        )
        t_total = time.time() - t0
        t_decode = t_total - t_prefill
        rec = {
            "bs": bs,
            "prefill_s": round(t_prefill, 3),
            "decode_s": round(t_decode, 3),
            "step_ms": round(t_decode * 1000 / N_DECODE, 1),
            "tok_per_s": round(bs * N_DECODE / t_decode, 2),
        }
        results.append(rec)
        print(
            f"[{tag}] bs={bs:>2} prefill={t_prefill:6.2f}s decode={t_decode:5.2f}s "
            f"({rec['step_ms']:6.1f} ms/step, {rec['tok_per_s']:8.2f} tok/s)"
        )
        sys.stdout.flush()
    eng.shutdown()

    all_res = {}
    if os.path.exists(OUT_JSON):
        try:
            all_res = json.load(open(OUT_JSON))
        except Exception:
            all_res = {}
    all_res[tag] = results
    json.dump(all_res, open(OUT_JSON, "w"), indent=1)
    print(f"saved {OUT_JSON} [{tag}]")


if __name__ == "__main__":
    main()
