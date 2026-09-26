# M8-TC 验证：L1 批量打分 tl.dot(tf32) 版 vs 广播版 vs eager einsum
# 三口径：①分数 maxdiff ②top-K1 块选择 jaccard ③计时（bs32/131K 同 M8 口径）
# 用法：CUDA_VISIBLE_DEVICES=x python3 test_tli_m8_l1tc.py
import sys
import warnings

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
warnings.filterwarnings("ignore")

import torch

from sglang.srt.layers.attention.tli.kernels import (
    tli_l1_score_batched,
    tli_l1_score_batched_dot,
)


def main():
    torch.manual_seed(0)
    dev = "cuda:0"
    # M8 同口径：bs=32 / S=131K / block=64 / Hkv=8 / H=32 / D=128 / d'=32
    n, S, BS = 32, 131072, 64
    Hkv, H, D, DP = 8, 32, 128, 32
    NBLK = S // BS
    G = H // Hkv

    kmin_pool = (torch.randn(64, NBLK, Hkv, DP, device=dev) * 0.5).float()
    kmax_pool = kmin_pool + torch.rand(64, NBLK, Hkv, DP, device=dev).float()
    q = (torch.randn(n, H, D, device=dev) * 0.8).float().contiguous()
    # 子空间索引 = 低频尾维（与生产 idx1 同形态）
    idx1 = torch.arange(D - DP, D, device=dev, dtype=torch.int64)
    rows = torch.randperm(48, device=dev)[:n].to(torch.int64).contiguous()
    t_t = torch.full((n,), S - 1, device=dev, dtype=torch.int64)
    # 混合 nblk（含非对齐尾块行），对拍覆盖垃圾块 -inf 口径
    nblk_t = torch.randint(NBLK - 64, NBLK + 1, (n,), device=dev, dtype=torch.int64)

    # ---- ① 正确性：分数 diff + topk jaccard ----
    sc_b = tli_l1_score_batched(q, idx1, kmin_pool, kmax_pool, rows, nblk_t, t_t, BS)
    sc_d = tli_l1_score_batched_dot(q, idx1, kmin_pool, kmax_pool, rows, nblk_t, t_t, BS)
    finite = torch.isfinite(sc_b)
    md = (sc_b[finite] - sc_d[finite]).abs().max().item()
    # -inf 位一致（垃圾块口径逐位）
    inf_match = bool((torch.isinf(sc_b) == torch.isinf(sc_d)).all())
    # topk jaccard（K1=128，同生产）
    K1 = 128
    kb = torch.topk(sc_b.view(n * Hkv, -1), K1, dim=-1).indices
    kd = torch.topk(sc_d.view(n * Hkv, -1), K1, dim=-1).indices
    jac = []
    for i in range(n * Hkv):
        jac.append(len(set(kb[i].tolist()) & set(kd[i].tolist())) / K1)
    print(f"[corr] score maxdiff={md:.3e} (tf32 预期 ~1e-3 级)")
    print(f"[corr] -inf 位逐位一致={inf_match} topk jaccard mean={sum(jac)/len(jac):.4f} min={min(jac):.4f}")

    # ---- ② 计时（bs32/131K 全池读）----
    def bench(fn, iters=50):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / iters

    tb = bench(lambda: tli_l1_score_batched(q, idx1, kmin_pool, kmax_pool, rows, nblk_t, t_t, BS))
    td = bench(lambda: tli_l1_score_batched_dot(q, idx1, kmin_pool, kmax_pool, rows, nblk_t, t_t, BS))
    # 理论地板：全池行读 2×n×NBLK×Hkv×DP×4B
    bytes_ = 2 * n * NBLK * Hkv * DP * 4
    print(f"[time] 广播版={tb*1000:.0f}us  TC版={td*1000:.0f}us  加速={tb/td:.2f}x")
    print(f"[time] 流量={bytes_/1e6:.0f}MB  广播带宽={bytes_/(tb/1e3)/1e9:.0f}GB/s  "
          f"TC带宽={bytes_/(td/1e3)/1e9:.0f}GB/s (HBM3e 峰值 ~4800GB/s)")
    print("PASS" if md < 5e-2 and inf_match else "FAIL")


if __name__ == "__main__":
    main()
