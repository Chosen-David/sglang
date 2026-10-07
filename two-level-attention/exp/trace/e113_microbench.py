#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E113 microbench：sim_greedy 贪心聚类 Python 循环 vs Triton kernel 规模扫描。

GPU 空闲时跑（E109 扫描占满双卡时挂起；本脚本显存占用 <200MB 可共存，
但计时需独占才准——默认 GPU0，加 --wait 可轮询等待空闲）。

口径：
  - 数据：base+noise 混合簇结构（与单测同款，覆盖归并路径）+ 纯高斯单例
    极端臂（C/N≈1，最坏 K 扫描量）；
  - 计时：torch.cuda.synchronize() 前后 wall time，Triton 预热一次去 JIT；
  - 对照：E108 probe 实测口径（Python 循环 T≈16K ~5.8s/层，sim=0.9 27s）。

用法：CUDA_VISIBLE_DEVICES=0 python3 e113_microbench.py [--wait]
输出：exp/trace/results/e113_microbench.json
"""
import argparse
import importlib
import json
import os
import sys
import time
import types

sys.dont_write_bytecode = True   # 主树只读 import

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from e113_greedy_triton import greedy_pass_triton, greedy_build_triton  # noqa: E402

MAIN_TREE = "/home/wangyuanshuo02/sglang/two-level-attention"


def load_sparse_attn(name, root):
    mod = types.ModuleType(name)
    mod.__path__ = [os.path.join(root, "sparse_attn")]
    sys.modules[name] = mod
    return importlib.import_module(f"{name}.indexer")


IDX = load_sparse_attn("sparse_attn_e113bench", MAIN_TREE)
greedy_ref = IDX.TLIIndexer._greedy_cluster_pass_python   # E113b：参考实现固定取 Python 路径（_greedy_cluster_pass 已是调度器）


def gen_structured(T, H, dd, seed, n_base=None, noise=0.10):
    """混合簇结构（归并路径）。n_base=T//8 时 C/N≈0.125。"""
    g = torch.Generator(device="cuda").manual_seed(seed)
    n_base = n_base or max(T // 8, 1)
    base = torch.randn(n_base, H, dd, generator=g, device="cuda")
    idx = torch.randint(0, n_base, (T,), generator=g, device="cuda")
    return (base[idx] + noise * torch.randn(T, H, dd, generator=g, device="cuda")).contiguous()


def bench(fn, warmup=1, rep=3):
    """同步口径计时：预热 warmup 次后取 rep 次中位数。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(rep):
        t0 = time.time()
        fn()
        torch.cuda.synchronize()
        ts.append(time.time() - t0)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait", action="store_true", help="轮询等 GPU 空闲再跑")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "e113_microbench.json"))
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    assert torch.cuda.is_available(), "需 GPU"
    if args.wait:
        while True:
            free, total = torch.cuda.mem_get_info()
            if (total - free) / total < 0.5:   # 已用 <50% 视为空闲
                break
            print(f"GPU 占用高（free {free/1e9:.0f}GB/{total/1e9:.0f}GB），60s 后重试…", flush=True)
            time.sleep(60)

    H, dd = 8, 32   # Qwen3-8B kv-head × tail32 口径
    results = {"meta": {
        "probe": "E113 microbench：sim_greedy Python 循环 vs Triton kernel",
        "gpu": torch.cuda.get_device_name(0),
        "timing": "cuda synchronize wall, median of 3, Triton 预热去 JIT",
        "e108_probe_ref": "Python 循环 T≈16K 实测 ~5.8s/层（sim=0.9 时 27s）",
        "started": time.strftime("%F %T"),
    }, "cases": []}

    for T in (1024, 4096, 8192, 16384, 32768):
        for sim, tag, gen in ((0.9, "structured_c8", lambda: gen_structured(T, H, dd, 7)),
                              (0.9, "singleton", lambda: torch.randn(T, H, dd, device="cuda").contiguous())):
            x = gen()

            def run_ref():
                return greedy_ref(x, sim, x.new_zeros(H, T, dd), x.new_zeros(H, T),
                                  x.new_zeros(H, T),
                                  torch.zeros(H, dtype=torch.long, device="cuda"))

            def run_tri():
                return greedy_build_triton(x, sim, chunk=16384)
            # Python 参考（慢，只跑 1 次且兼做正确性基准）
            t_ref = bench(run_ref, warmup=0, rep=1)
            # Triton（预热去 JIT，中位×3；chunk=16384 为 E113 调优最优）
            t_tri = bench(run_tri, warmup=1, rep=3)
            # 正确性抽验（逐位，铁律门 1 微型版）
            r = run_ref()
            t = run_tri()
            mism = int((r[4] != t[4]).sum())
            live = int(r[3].max().item())
            rec = {"T": T, "H": H, "dd": dd, "sim": sim, "data": tag,
                   "clusters_max_head": live, "C_over_N": round(live / T, 4),
                   "assign_mismatch": mism,
                   "wall_s_python": round(t_ref, 3),
                   "wall_s_triton": round(t_tri, 4),
                   "us_per_token_python": round(1e6 * t_ref / T, 2),
                   "us_per_token_triton": round(1e6 * t_tri / T, 2),
                   "speedup": round(t_ref / t_tri, 1)}
            print(json.dumps(rec, ensure_ascii=False), flush=True)
            results["cases"].append(rec)
            del x, r, t
            torch.cuda.empty_cache()

    results["finished"] = time.strftime("%F %T")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=1, ensure_ascii=False)
    print("\nsaved ->", args.out)


if __name__ == "__main__":
    main()
