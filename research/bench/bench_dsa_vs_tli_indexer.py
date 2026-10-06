# DSA indexer vs TLI 两级 select —— kernel 级同机 microbench（诚实口径）
#
# 对比对象与口径：
# ① DSA 侧 = DeepSeek-V3.2 官方 inference/kernel.py 原样（tilelang fp8_index）
#    + torch.topk(2048)（model.py line 483 的 decode indexer 全链路）。
#    输入为合成数据（kernel 级 microbench，非端到端精度实验）：
#    q/k 用 randn 合成，量级对齐 k_norm(LayerNorm) 后的真实分布；
#    weights = randn/sqrt(64)（对应 weights_proj·n_heads^-0.5）。
#    计时包含 act_quant(q)（每步真实成本）+ fp8_index + topk(2048)。
#    k_cache 写入（每步 1 token 的 act_quant+scatter）单独报一行。
# ② TLI 侧 = sglang tli 两级 select 全链路，真实 trace（/tmp/trace/qwen3-8b）
#    q/K 权重真实，S 扩展用 repeat（成本口径不失真，M7 bench 同款技巧）。
#    计时两种形态：eager select（PyTorch 级）与 fused L1/L2 kernel 路径。
#
# 架构差异如实报告：DSA indexer h=64 头×d=128（3.2-Exp 生产配置），
# TLI 为 Qwen3-8B GQA Hkv=8；各自在生产模型配置下的 indexer 开销对比。
# 机器：H20-3e（cc9.0，78 SM），CUDA_VISIBLE_DEVICES=1。
import sys
import time
import json

sys.path.insert(0, "/tmp/papers/DeepSeek-V3.2/inference")
sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")

import os

import torch

dev = "cuda:0"
OUT = "/home/wangyuanshuo02/sglang/dsa_vs_tli_indexer.json"

S_LIST = [10000, 40000, 131072]
RESULTS = {"machine": "H20-3e cc9.0", "note": "kernel-level microbench; DSA side synthetic q/k (aligned magnitude), TLI side real trace"}


def bench(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2] * 1e3  # ms 中位数


# ============ ① DSA indexer（官方 tilelang kernel 原样） ============
# 兼容补丁：本机 tilelang 0.1.7.post3 无 TL_DISABLE_FAST_MATH key（fast-math
# pass 为新版才引入，旧版不存在 = 默认禁用）。用 importlib 源码级去掉该行
# （pass_configs 是编译器优化开关，删 no-op 开关不改 kernel 语义，其余原样）。
# 注意：tilelang 0.1.7 jit 会按首次调用形状特化缓存（symbolic n 静态化），
# 故每个 S 用独立子进程跑（本文件 `dsa S` 模式），主进程汇总。
import importlib.util  # noqa: E402
import subprocess  # noqa: E402
import sys as _sys  # noqa: E402


def run_dsa_subprocess():
    dsa_results = {}
    for S in S_LIST:
        p = subprocess.run(
            [_sys.executable, __file__, "dsa", str(S)],
            capture_output=True, text=True,
        )
        for line in p.stdout.splitlines():
            if line.startswith("DSA_RESULT"):
                dsa_results[S] = json.loads(line.split(" ", 2)[2])
        if p.returncode != 0:
            print(p.stderr[-2000:])
            raise RuntimeError(f"DSA subprocess failed at S={S}")
    return dsa_results


if len(_sys.argv) >= 3 and _sys.argv[1] == "dsa":
    # ---- 子进程模式：单个 S 的 DSA indexer 计时 ----
    S = int(_sys.argv[2])
    _src = open("/tmp/papers/DeepSeek-V3.2/inference/kernel.py").read()
    _src = _src.replace("    tilelang.PassConfigKey.TL_DISABLE_FAST_MATH: True,\n", "")
    _kpath = "/tmp/quest_min/dsa_kernel_compat.py"
    os.makedirs("/tmp/quest_min", exist_ok=True)
    with open(_kpath, "w") as f:
        f.write(_src)
    _spec = importlib.util.spec_from_file_location("dsa_kernel_compat", _kpath)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    act_quant, fp8_index = _mod.act_quant, _mod.fp8_index

    H_IDX, D_IDX, TOPK = 64, 128, 2048
    torch.manual_seed(0)
    q_dsa = (torch.randn(1, 1, H_IDX, D_IDX, device=dev) * 0.5).to(torch.bfloat16)
    k_all = (torch.randn(1, S, D_IDX, device=dev) * 0.5).to(torch.bfloat16)

    k_dsa = k_all.contiguous()
    k_fp8, k_s = act_quant(k_dsa, 128)
    k_fp8, k_s = k_fp8.contiguous(), k_s.squeeze(-1).contiguous()  # [b,S,1]→[b,S]（kernel 签名）

    def dsa_step():
        q_fp8, q_s = act_quant(q_dsa, 128)  # 每步 q 量化（真实成本）
        w = (torch.randn(1, 1, H_IDX, device=dev) * (H_IDX ** -0.5) * q_s.view(1, 1, -1) * (D_IDX ** -0.5)).contiguous()
        score = fp8_index(q_fp8.contiguous(), w.view(1, 1, H_IDX), k_fp8, k_s)
        return score.topk(min(TOPK, S), dim=-1)[1]

    ms = bench(dsa_step)

    def dsa_index_only():
        q_fp8, q_s = act_quant(q_dsa, 128)
        w = (torch.randn(1, 1, H_IDX, device=dev) * (H_IDX ** -0.5) * q_s.view(1, 1, -1) * (D_IDX ** -0.5)).contiguous()
        return fp8_index(q_fp8.contiguous(), w.view(1, 1, H_IDX), k_fp8, k_s)

    ms_index = bench(dsa_index_only)

    k_one = k_all[:, :1].contiguous()
    def dsa_write():
        kf, ks = act_quant(k_one, 128)
        return kf, ks
    ms_write = bench(dsa_write)
    res = {"total_ms": round(ms, 3), "fp8_index_ms": round(ms_index, 3),
           "topk_ms": round(ms - ms_index, 3), "k_write_ms": round(ms_write, 4)}
    print(f"[DSA indexer] S={S:6d}  total {ms:8.3f} ms  (fp8_index {ms_index:8.3f} + topk {ms-ms_index:6.3f})  k_write {ms_write:.4f}")
    print("DSA_RESULT", S, json.dumps(res))
    _sys.exit(0)


RESULTS["dsa_indexer"] = run_dsa_subprocess()
with open(OUT, "w") as f:
    json.dump(RESULTS, f, indent=2)

# ============ ② TLI 两级 select（真实 trace，eager + fused kernel） ============
from sglang.srt.layers.attention.tli.config import TLIProfile  # noqa: E402
from sglang.srt.layers.attention.tli.indexer import TLIIndexer  # noqa: E402

LAYERS = ["layer03", "layer17"]  # far-heavy + 中性层
TLI_S_LIST = [9891, 40000, 131072]


def bench_tli(use_l1_kernel: bool):
    idxer = TLIIndexer(TLIProfile())
    res = {}
    for lname in LAYERS:
        d = torch.load(f"/tmp/trace/qwen3-8b/lb_hotpotqa_0/{lname}.pt", map_location=dev)
        k_real = d["k"].float()
        q_real = d["q"].float()
        S0 = k_real.shape[0]
        for S in TLI_S_LIST:
            if S <= S0:
                k = k_real[:S]
            else:
                reps = (S + S0 - 1) // S0
                k = k_real.repeat(reps, 1, 1)[:S]
            q = q_real[-1:]  # decode 单 token
            t = S - 1
            index = idxer.build_block_index(k)  # 稳态：索引已建好（对应 k_cache 已就位）
            torch.cuda.synchronize()

            def tli_step():
                return idxer.select(index, q, t, use_l1_kernel=use_l1_kernel)  # L1 块筛 + L2 4bit 细筛全链路

            ms = bench(tli_step, warmup=3, iters=10)
            res[f"{lname}_S{S}"] = round(ms, 3)
            print(f"[TLI select {'fused' if use_l1_kernel else 'eager':5s}] {lname} S={S:6d}  {ms:8.3f} ms")
            del index, k
            torch.cuda.empty_cache()
    return res


RESULTS["tli_select_eager"] = bench_tli(False)
RESULTS["tli_select_fusedL1"] = bench_tli(True)

with open(OUT, "w") as f:
    json.dump(RESULTS, f, indent=2)
print("saved", OUT)
