# Quest 完整 decode 索引成本补测：page 分数计算（q·kmin/q·kmax GEMV + elementwise max）
# + 官方 decode_select_k radix kernel（已在 /tmp/quest_bench_select_k 单独测得 8.3-12.5µs）
# —— 两部分之和才是与 DSA fp8_index+topk、TLI select 全链路同口径的数字。
# 口径：Quest 论文配置 num_heads=32、head_dim=128、page=16 token、fp16 分数。
# k_min/k_max 是 Quest KV cache 侧维护的逐 page 逐 head min/max（合成数据，kernel 级 microbench）。
# 625/2500/8192 pages 对应 10K/40K/131K 真实 token；k=64 pages=1024 token 检索预算。
import time
import torch

dev = "cuda:0"


def bench(fn, warmup=5, iters=50):
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


print(f"{'pages':>6s} {'tokens':>7s} {'score_ms':>9s} {'HBM_MB':>8s}")
for PAGES in [625, 2500, 8192]:
    H, D = 32, 128
    k_min = torch.randn(PAGES, H, D, device=dev, dtype=torch.float16)
    k_max = torch.randn(PAGES, H, D, device=dev, dtype=torch.float16)
    q = torch.randn(H, D, device=dev, dtype=torch.float16)

    def score_step():
        # Quest decode：每 page 每 head 打分 = max(q·k_min, q·k_max)（上界分数）
        s_lo = torch.einsum("hd,phd->ph", q, k_min)
        s_hi = torch.einsum("hd,phd->ph", q, k_max)
        return torch.maximum(s_lo, s_hi)  # [pages, H] → 转置后即 select_k 输入 [H*pages]

    ms = bench(score_step)
    hbm = PAGES * H * D * 2 * 2 / 1e6  # min+max 两份 fp16
    print(f"{PAGES:6d} {PAGES*16:7d} {ms:9.4f} {hbm:8.1f}")
    # 存一份分数供 select_k 口径参考
    torch.save({"pages": PAGES, "score_ms": ms, "hbm_mb": hbm}, f"/tmp/quest_score_{PAGES}.pt")
print("note: select_k kernel (official raft radix) = 8.3/9.2/12.5 us @ 625/2500/8192 pages")
