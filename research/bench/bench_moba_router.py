# #30 MoBA indexer 级同机对比：官方 moba_naive.py 的 router（gating）原样接入
# —— router 是纯 PyTorch（块均值 + q·mean einsum + topk），无需 flash-attn
# （moba_efficient.py 的 flash-attn 只是 attention 主体，非 indexer）。
# 口径与 Quest/DSA/TLI 对比一致：decode 单 token per-layer-call，S=10K/40K/131K。
# 预算对齐：MoBA chunk=512 × topk=2 = 1024 token ≙ TLI K2=1024。
# 形状口径（如实报告差异）：MoBA router 逐 q-head（H=32）打分 vs TLI GQA-sum（Hkv=8）；
# MoBA 无 token 级细筛/滑窗/分区（chunk 粒度 512 固定）；MoBA 需从头训练。
# 注意：router 计时用合成张量（与 Quest/DSA 对比同做法，已标注）；trace 版见 rows。
import json
import math
import sys
import time

import torch

dev = "cuda:0"
H, D, CHUNK, TOPK = 32, 128, 512, 2  # Qwen3-8B 32 q-head；topk×chunk=1024 ≙ K2
Ss = [10_000, 40_000, 131_072]
res = {"config": {"H": H, "D": D, "chunk": CHUNK, "topk": TOPK,
                  "budget_tokens": CHUNK * TOPK, "synthetic": True}}
g = torch.Generator(device=dev).manual_seed(0)


def bench(fn, reps=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return sorted(ts)[reps // 2]


print(f"{'S':>7s} {'build_ms':>9s} {'decode_router_ms':>17s} {'prefill_router_ms':>18s} {'storage_B/tok':>13s}")
rows = []
for S in Ss:
    k = (torch.randn(S, H, D, generator=g, device=dev) * 0.3).bfloat16()
    N = math.ceil(S / CHUNK)
    # ---- build：块均值（官方 naive 语义 k_[b0:b1].mean(0)；增量化是工程改进，
    # 这里测官方口径的全量 build）----
    def build():
        pad = N * CHUNK - S
        kk = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, pad)) if pad else k
        return kk.reshape(N, CHUNK, H, D).mean(1)  # [N, H, D]

    t_build = bench(build, reps=10, warmup=3)
    kgw = build()

    # ---- decode router（官方 gate 语义，fp32 打分）----
    q1 = (torch.randn(1, H, D, generator=g, device=dev) * 0.3).float()

    def decode_router():
        gate = torch.einsum("shd,nhd->hsn", q1, kgw.float())  # [H,1,N]
        return torch.topk(gate, min(TOPK, N), dim=-1)

    t_dec = bench(decode_router)

    # ---- decode 增量维护：新 token 并入尾块（running mean，一次小 kernel）----
    k_new = k[-1:]

    def maintain():
        blk = (S - 1) // CHUNK
        n_in = S - blk * CHUNK
        kgw[blk] = (kgw[blk].float() * n_in + k_new[0].float()) / (n_in + 1)

    t_mnt = bench(maintain)

    # ---- prefill router（全 q 序列：einsum [H,S,N] + topk，官方口径）----
    qf = (torch.randn(S, H, D, generator=g, device=dev) * 0.3).float()

    def prefill_router():
        gate = torch.einsum("shd,nhd->hsn", qf, kgw.float())  # [H,S,N]
        return torch.topk(gate, min(TOPK, N), dim=-1)

    t_pre = bench(prefill_router, reps=5, warmup=2)
    st_bytes = N * H * D * 2 / S  # bf16 块均值摊到每 token
    rows.append({"S": S, "build_ms": round(t_build * 1e3, 3),
                 "decode_router_ms": round(t_dec * 1e3, 4),
                 "decode_maintain_ms": round(t_mnt * 1e3, 4),
                 "prefill_router_ms": round(t_pre * 1e3, 3),
                 "storage_B_per_tok": round(st_bytes, 1)})
    print(f"{S:7d} {t_build*1e3:9.2f} {t_dec*1e3:17.4f} {t_pre*1e3:18.2f} {st_bytes:13.1f}")
    del k, kgw, qf
    torch.cuda.empty_cache()

res["rows"] = rows
# TLI 参照（kernel_comparison_indexers.json，同机同口径）
kc = json.load(open("/home/wangyuanshuo02/sglang/kernel_comparison_indexers.json"))
print("\nTLI 参照（同机）：eager select 0.736/0.780/0.853 ms（10K/40K/131K，"
      "含两级+滑窗+分区；fusedL1 0.604/0.658/0.787）")
print("MoBA router（本测）：", " / ".join(f"{r['decode_router_ms']:.3f}" for r in rows), "ms")
json.dump(res, open("/home/wangyuanshuo02/sglang/moba_router_bench.json", "w"), indent=1)
print("saved moba_router_bench.json")
