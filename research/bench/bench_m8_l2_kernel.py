# M8 KernelA：batched L2 fused gather+dequant+GEMV（P5 瓶颈，87.6%@bs32/131K）
# eager 路径问题：kq_c fp32 物化 2.1GB×2(rw) + 逐元素 flat gather（~240GB/s 有效）。
# kernel 方案：grid (n, Tc/CHUNK)，每 program 对 CHUNK 个候选 token 直接从 pool
# uint8 gather（每 token HKV*ND2=256B 连续段）→ 寄存器内反量化 → GEMV 打分 →
# 写 s2（消除全部中间物化）。
# 对拍口径：s2 allclose（atol 1e-3，归约顺序不同）+ 最终选择有效集合一致率。
import sys

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
import time

import torch
import triton
import triton.language as tl


@triton.jit
def _tli_l2_score_batched_kernel(
    q2_ptr, kq_ptr, sc_ptr, mn_ptr, rows_ptr, tok_ptr, s2_ptr,
    HKV: tl.constexpr, ND2: tl.constexpr,
    TC: tl.constexpr, TC_P2: tl.constexpr,
    S_CAP,  # kq 行 stride（runtime，避免 131072 常量溢出问题不大但保留灵活性）
    CHUNK: tl.constexpr,
):
    a = tl.program_id(0)
    c0 = tl.program_id(1) * CHUNK
    offs_c = c0 + tl.arange(0, CHUNK)
    cm = offs_c < TC
    row = tl.load(rows_ptr + a).to(tl.int64)
    pos = tl.load(tok_ptr + a * TC + offs_c, mask=cm, other=0).to(tl.int64)
    offs_h = tl.arange(0, HKV)
    offs_d = tl.arange(0, ND2)
    # uint8 gather [CHUNK, HKV, ND2]：每 token 的 HKV*ND2=256B 连续
    base = row * (S_CAP * HKV * ND2) + pos * (HKV * ND2)  # [CHUNK] int64
    addr = base[:, None, None] + (offs_h[None, :, None] * ND2 + offs_d[None, None, :])
    kq = tl.load(kq_ptr + addr, mask=cm[:, None, None], other=0).to(tl.float32)
    # scale / mn [CHUNK, HKV]
    base_s = row * (S_CAP * HKV) + pos * HKV  # [CHUNK]
    addr_s = base_s[:, None] + offs_h[None, :]
    sc = tl.load(sc_ptr + addr_s, mask=cm[:, None], other=0.0)
    mn = tl.load(mn_ptr + addr_s, mask=cm[:, None], other=0.0)
    kq_c = kq * sc[:, :, None] + mn[:, :, None]
    # GEMV：q2 [HKV, ND2]，对每 (chunk, h) 归约 d
    q2 = tl.load(q2_ptr + a * HKV * ND2 + offs_h[:, None] * ND2 + offs_d[None, :])
    s = tl.sum(kq_c * q2[None, :, :], axis=2)  # [CHUNK, HKV]
    # 转置写 s2[a, h, c]（下游布局不变）
    tl.store(s2_ptr + a * HKV * TC + offs_h[None, :] * TC + offs_c[:, None],
             s, mask=cm[:, None])


def tli_l2_score_batched(q2, kq_q, kq_sc, kq_mn, rows, tok_c, chunk=1024):
    """q2 [n,Hkv,nd2] fp32；pool 三张量；rows [n]；tok_c [n,Tc]（已 clamp）。
    返回 s2 [n,Hkv,Tc] fp32（哨兵位置为垃圾分数——下游 valid&causal 掩掉，
    与 eager 相同语义）。"""
    n, Hkv, nd2 = q2.shape
    Tc = tok_c.shape[1]
    s2 = torch.empty(n, Hkv, Tc, dtype=torch.float32, device=q2.device)
    grid = (n, triton.cdiv(Tc, chunk))
    _tli_l2_score_batched_kernel[grid](
        q2, kq_q, kq_sc, kq_mn, rows, tok_c, s2,
        HKV=Hkv, ND2=nd2, TC=Tc, TC_P2=triton.next_power_of_2(Tc),
        S_CAP=kq_q.shape[1], CHUNK=chunk, num_warps=8,
    )
    return s2


# ---------------- 验证与基准 ----------------
if __name__ == "__main__":
    dev = "cuda:0"
    torch.manual_seed(0)
    n, R, S_cap, Hkv, nd2 = 32, 32, 131072, 8, 32
    Tc = 65728
    pool_q = torch.randint(0, 16, (R, S_cap, Hkv, nd2), dtype=torch.uint8, device=dev)
    pool_sc = torch.rand(R, S_cap, Hkv, device=dev) * 0.01
    pool_mn = (torch.rand(R, S_cap, Hkv, device=dev) - 0.5) * 0.1
    rows = torch.arange(n, device=dev)
    q2 = torch.randn(n, Hkv, nd2, device=dev) * 0.3
    # 真实候选分布形态：每行 ~65K/131K 随机升序位置（含哨兵 clamp 语义）
    g = torch.Generator(device=dev).manual_seed(1)
    sel = torch.rand(n, S_cap, generator=g, device=dev) < 0.5
    tok = torch.where(sel, torch.arange(S_cap, device=dev).expand(n, -1),
                      torch.full((n, S_cap), S_cap, device=dev, dtype=torch.long))
    tok = torch.topk(tok, Tc, dim=-1, largest=False).values
    tok_c = tok.clamp(max=S_cap - 1).contiguous()

    # ---- eager 参照（生产代码同构：flat gather + 物化 + einsum）----
    def eager_p5():
        s2 = torch.empty(n, Hkv, Tc, dtype=torch.float32, device=dev)
        d_off = torch.arange(nd2, device=dev)
        h_off = torch.arange(Hkv, device=dev).view(1, Hkv, 1) * nd2
        h_off_s = torch.arange(Hkv, device=dev)
        chunk = max(1, (256 << 20) // max(Tc * Hkv * nd2 * 4, 1))
        for r0 in range(0, n, chunk):
            r1 = min(r0 + chunk, n)
            m = r1 - r0
            flat = (rows[r0:r1].view(m, 1, 1, 1) * (S_cap * Hkv * nd2)
                    + tok_c[r0:r1].view(m, Tc, 1, 1) * (Hkv * nd2)
                    + h_off.view(1, Hkv, 1) + d_off.view(1, 1, nd2))
            flat_s = (rows[r0:r1].view(m, 1, 1) * (S_cap * Hkv)
                      + tok_c[r0:r1].view(m, Tc, 1) * Hkv + h_off_s.view(1, 1, Hkv))
            grid_c = pool_q.reshape(-1)[flat.view(-1)].view(m, Tc, Hkv, nd2)
            sc_c = pool_sc.reshape(-1)[flat_s.view(-1)].view(m, Tc, Hkv)
            mn_c = pool_mn.reshape(-1)[flat_s.view(-1)].view(m, Tc, Hkv)
            kq_c = grid_c.float() * sc_c.unsqueeze(-1) + mn_c.unsqueeze(-1)
            s2[r0:r1] = torch.einsum("ahd,athd->aht", q2[r0:r1], kq_c)
        return s2


    s2_ref = eager_p5()
    s2_k = tli_l2_score_batched(q2, pool_q, pool_sc, pool_mn, rows, tok_c)
    torch.cuda.synchronize()
    diff = (s2_k - s2_ref).abs().max().item()
    rel = diff / s2_ref.abs().max().item()
    print(f"s2 max abs diff = {diff:.3e} (rel {rel:.3e})")
    assert diff < 1e-3, "s2 数值不一致"
    # 格点级强口径：反量化 kq 逐位由相同运算序保证，分数差异只来自归约顺序
    print("对拍 PASS（allclose, atol 1e-3）")


    def bench(fn, reps=20, warmup=3):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        return sorted(ts)[len(ts) // 2] * 1e3


    t_e = bench(eager_p5)
    t_k = bench(lambda: tli_l2_score_batched(q2, pool_q, pool_sc, pool_mn, rows, tok_c))
    print(f"eager P5: {t_e:.3f} ms | kernel: {t_k:.3f} ms | 加速 {t_e/t_k:.2f}×")
