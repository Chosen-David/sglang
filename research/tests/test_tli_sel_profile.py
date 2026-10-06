# #64 select_batched 内部归因 microbench（30B 形态，合成 index 池）
# 目的：30B prefill 归因 select=10.95s@32K（81%）的内部分解——
# kq_f 反量化表 / L1 einsum+topk / onehot+sel_mask / 快路径 einsum /
# M10 慢路径 kernel / B' 分区段 各占多少。
# 形态对齐 30B 生产：Hkv=4, G=8, D=128, nd2=64；chunk 5.25K 行
# （21K prompt / 4 chunks），S 沿 chunk 增长 5K→21K。
# 方法：①torch.profiler op 级表（真实方法 unbound call，代码零复制）
# ②wall-clock 总账 vs 生产 10.95s 定标。
# 用法：CUDA_VISIBLE_DEVICES=0 python3 test_tli_sel_profile.py
import json
import os
import sys
import warnings

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
warnings.filterwarnings("ignore")

import torch

from sglang.srt.layers.attention.tli.config import TLIProfile
from sglang.srt.layers.attention.tli.indexer import (
    TLIIndexer,
    kq_unpack,
    quant4_pack,
)

OUT = "/home/wangyuanshuo02/sglang/tli_sel_profile.json"
Q_AGG = os.environ.get("SGLANG_TLI_Q_AGG", "sum")
DS = os.environ.get("SGLANG_TLI_DS_TOPK", "0") == "1"
if DS:
    sys.path.insert(0, "/home/wangyuanshuo02/.local/pylibs")


def build_index(S, Hkv, nd2, device):
    """合成 index dict（真实 build_block_index 输出结构）。"""
    bs = 64
    nblk = (S + bs - 1) // bs
    kmin = (torch.randn(nblk, Hkv, 32, device=device) * 0.5).float()
    kmax = kmin + torch.rand(nblk, Hkv, 32, device=device).float()
    kq = (torch.randn(S, Hkv, nd2, device=device) * 0.8).float()
    kq_q, kq_sc, kq_mn = quant4_pack(kq)
    return {
        "S": S, "nblk": nblk, "kmin": kmin, "kmax": kmax,
        "kq_q": kq_q, "kq_sc": kq_sc, "kq_mn": kq_mn,
    }


def main():
    dev = "cuda:0"
    torch.manual_seed(0)
    # 30B 形态
    Hkv, G, D, nd2 = 4, 8, 128, 32
    H = Hkv * G
    os.environ["SGLANG_TLI_Q_AGG"] = Q_AGG
    p = TLIProfile()
    p.q_agg = Q_AGG
    p.use_ds_topk = DS
    idxr = TLIIndexer(profile=p, head_dim=D).to(dev)
    print(f"[cfg] q_agg={Q_AGG} ds_topk={DS} k1={p.k1_blocks} far={p.far_tokens} "
          f"budget={p.token_budget} near_len={p.near_len} use_prefill_kernel={p.use_prefill_kernel}")

    # 4 chunks（21K prompt）：Nq=5250, S 因果增长 5250→21000
    CHUNKS = [5250, 10500, 15750, 21000]
    res = {"q_agg": Q_AGG, "ds_topk": DS, "chunks": []}
    # DS 臂须先对拍输出集合（SGLANG_TLI_DS_TOPK 由 TLIProfile 读 env，
    # 此处进程级已设/未设由外层 env 控制）
    # 预热（Triton JIT + cuBLAS）
    idx_w = build_index(8192, Hkv, nd2, dev)
    q_w = torch.randn(512, H, D, device=dev).float()
    t_w = torch.full((512,), 8191, device=dev, dtype=torch.long)
    for _ in range(2):
        idxr.select_batched(idx_w, q_w, t_w)
    torch.cuda.synchronize()

    from torch.profiler import ProfilerActivity, profile
    # ---- DS 对拍段（末 chunk 形态：torch 版 vs DS 版输出集合 jaccard）----
    jac_report = None
    if DS:
        S = CHUNKS[-1]
        n = 5250
        index = build_index(S, Hkv, nd2, dev)
        q = torch.randn(n, H, D, device=dev).float()
        t_arr = torch.arange(S - n, S, device=dev, dtype=torch.long)
        p.use_ds_topk = False
        r_t = idxr.select_batched(index, q, t_arr)
        p.use_ds_topk = True
        r_d = idxr.select_batched(index, q, t_arr)
        # 位置集合 jaccard per (row, head)：哨兵 0 位排除后集合比
        jacs = []
        for i in range(0, n, 250):
            for h in range(Hkv):
                a = set(r_t[i, h].tolist()) - {0}
                b = set(r_d[i, h].tolist()) - {0}
                if a or b:
                    jacs.append(len(a & b) / len(a | b))
        jac_report = round(sum(jacs) / len(jacs), 4)
        res["ds_jaccard_mean"] = jac_report
        res["ds_jaccard_min"] = round(min(jacs), 4)
        print(f"[ds-check] 输出位置集合 jaccard mean={jac_report} min={min(jacs):.4f} "
              f"n={len(jacs)}（哨兵 0 位排除）", flush=True)
    total_sel = 0.0
    for ci, S in enumerate(CHUNKS):
        n = 5250
        index = build_index(S, Hkv, nd2, dev)
        q = torch.randn(n, H, D, device=dev).float()
        t_arr = torch.arange(S - n, S, device=dev, dtype=torch.long)
        # 计时（无 profiler 的 wall-clock，3 次取中位）
        ts = []
        for _ in range(3):
            torch.cuda.synchronize()
            t0 = __import__("time").perf_counter()
            idxr.select_batched(index, q, t_arr)
            torch.cuda.synchronize()
            ts.append(__import__("time").perf_counter() - t0)
        ts.sort()
        wall = ts[1]
        total_sel += wall
        res["chunks"].append({"chunk": ci, "S": S, "n": n, "wall_ms": round(wall * 1000, 1)})
        print(f"[chunk{ci}] S={S} n={n} select={wall*1000:.1f}ms", flush=True)
    # op 级表（末 chunk，真实方法 unbound 调用零代码复制）
    S = CHUNKS[-1]
    n = 5250
    index = build_index(S, Hkv, nd2, dev)
    q = torch.randn(n, H, D, device=dev).float()
    t_arr = torch.arange(S - n, S, device=dev, dtype=torch.long)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        idxr.select_batched(index, q, t_arr)
        torch.cuda.synchronize()
    tbl = prof.key_averages().table(sort_by="cuda_time_total", row_limit=22)
    print(tbl)
    res["total_48layers_s"] = round(total_sel * 48, 2)
    res["op_table"] = tbl
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"[total] 4 chunks×48 layers ≈ {total_sel*48:.2f}s "
          f"(生产 32K 归因 select=10.95s 对照；chunk 比例看逐行)")
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
