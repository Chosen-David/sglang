# M11 fused 稀疏 attention microbench：kernel vs eager（_sparse_extend_one 形态）
# eager = gather fp32 物化 + einsum + softmax（backend 现行实现）
import sys

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
import torch

from sglang.srt.layers.attention.tli.kernels import tli_sparse_gather_attn


def eager_ext(q, sel, k_buf, v_buf):
    """backend._sparse_extend_one 的核心运算（逐 head 向量化版，无 row_chunk）"""
    n, H, D = q.shape
    Hkv = k_buf.shape[1]
    G = H // Hkv
    K2 = sel.shape[-1]
    pool_sel = sel
    out = torch.empty(n, H, D, device=q.device, dtype=q.dtype)
    for h in range(Hkv):
        q_h = q[:, h * G : (h + 1) * G].float()
        pp = pool_sel[:, h]
        k_sel = k_buf[pp, h].float()
        v_sel = v_buf[pp, h].float()
        att = torch.einsum("ngd,nkd->ngk", q_h, k_sel) * (D**-0.5)
        att = torch.softmax(att, dim=-1)
        out[:, h * G : (h + 1) * G] = torch.einsum("ngk,nkd->ngd", att, v_sel).to(q.dtype)
    return out


def bench(fn, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = __import__("time").perff_counter if False else None
    import time
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def main():
    torch.manual_seed(0)
    dev = "cuda:0"
    # 形态：8B（Hkv=8,G=4）与 30B-A3B（Hkv=4,G=8）× chunk 规模
    for tag, (n, Hkv, G, D, K2, pool) in [
        ("8B  chunk 2K", (2048, 8, 4, 128, 1024, 131072)),
        ("8B  chunk 4K", (4096, 8, 4, 128, 1024, 131072)),
        ("30B chunk 2K", (2048, 4, 8, 128, 1024, 131072)),
        ("30B chunk 4K", (4096, 4, 8, 128, 1024, 131072)),
    ]:
        H = Hkv * G
        q = torch.randn(n, H, D, device=dev, dtype=torch.bfloat16)
        k_buf = torch.randn(pool, Hkv, D, device=dev, dtype=torch.bfloat16)
        v_buf = torch.randn(pool, Hkv, D, device=dev, dtype=torch.bfloat16)
        sel = torch.randint(0, pool, (n, Hkv, K2), device=dev)
        te = bench(lambda: eager_ext(q, sel, k_buf, v_buf))
        best = None
        for bk in (32, 64, 128, 256):
            tk = bench(lambda: tli_sparse_gather_attn(q, sel, k_buf, v_buf, G, pool, block_k=bk))
            if best is None or tk < best[1]:
                best = (bk, tk)
        o1 = eager_ext(q, sel, k_buf, v_buf).float()
        o2 = tli_sparse_gather_attn(q, sel, k_buf, v_buf, G, pool, block_k=best[0]).float()
        diff = (o1 - o2).abs().max().item()
        # 带宽账（fused 只读 K/V bf16 各一次）
        bytes_r = n * Hkv * K2 * D * 2 * 2
        print(f"[{tag}] eager={te:.2f}ms kernel(BK={best[0]})={best[1]:.2f}ms "
              f"speedup={te/best[1]:.2f}x maxdiff={diff:.1e} "
              f"eff_bw={bytes_r/best[1]/1e9:.0f}GB/s", flush=True)


if __name__ == "__main__":
    main()
