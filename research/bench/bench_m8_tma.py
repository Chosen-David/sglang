# M8-2b：KernelC dual 带宽分解 + TMA/布局写实验（#51）
# 背景：dual kernel 0.385ms（NEARC=0 口径），总流量 ≈ 读 405MB（kq_q 270 + sc/mn 135）
#   + 写 70MB → 1.23TB/s，仅 H20 4TB/s 的 31%。嫌疑：int64 间接寻址开销 /
#   program 粒度过小（CHUNK=128 → 16.6K programs）/ 写布局。
# 实验：
#   A. 分解变体：nostore（读+算）/ noload（写 only）/ full —— 定位瓶颈在读/写/开销
#   B. CHUNK × num_warps sweep（NEARC=1 寄存器压力与 NEARC=0 不同，需重扫）
#   C. 写变体：block_ptr 写 far_sc（转置 tile [8, CHUNK]）；TMA descriptor 写
#   D. 参照：同尺寸 D2D copy 带宽上限
# 对拍口径：far_sc 逐位 equal（CUDA graph 路径要求），near_sc/near_tok 同。
import os
import sys
import time

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
import json

import torch
import triton
import triton.language as tl

dev = "cuda:0"
torch.manual_seed(0)
n, R = 32, 32
S_cap, Hkv, D = 131072, 8, 128
ND2 = 16  # M9 PCA r=16（生产口径）
Tc = (128 * Hkv + 3) * 64  # 65728
FAR_LO = 2 * 64
WNCAP = FAR_LO + (2048 - 128)  # 2048
CHUNK = 128

kq_q = torch.randint(0, 16, (R, S_cap, Hkv, ND2), dtype=torch.uint8, device=dev)
kq_sc = torch.rand(R, S_cap, Hkv, device=dev) * 0.01
kq_mn = (torch.rand(R, S_cap, Hkv, device=dev) - 0.5) * 0.1
rows_l = torch.arange(n, device=dev, dtype=torch.long)
# 模拟 compact 输出：块升序候选（66048→65728 取前 Tc），尾部哨兵 clamp
tok_c = torch.arange(Tc, device=dev, dtype=torch.long).unsqueeze(0).expand(n, Tc).contiguous()
q2 = torch.randn(n, Hkv, ND2, device=dev)
S_t = torch.full((n,), S_cap, device=dev, dtype=torch.long)
t_t = S_t - 1
far_hi_t = (t_t + 1 - 2048).clamp(min=FAR_LO)
sw_lo_t = (t_t - 128 + 1).clamp(min=0)
ps = (tok_c < FAR_LO).sum(dim=-1)
pf = (tok_c < far_hi_t.view(-1, 1)).sum(dim=-1)


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
    return sorted(ts)[len(ts) // 2] * 1e3


# ---- 生产 kernel 原样拷贝（NEARC=1 分支），做分解变体 ----
@triton.jit
def _dual_var_kernel(
    q2_ptr, kq_ptr, sc_ptr, mn_ptr, rows_ptr, tok_ptr,
    far_ptr, near_ptr, near_tok_ptr, far_hi_ptr, sw_lo_ptr, ps_ptr, pf_ptr,
    HKV: tl.constexpr, ND2: tl.constexpr,
    TC: tl.constexpr,
    S_CAP, FAR_LO: tl.constexpr, CHUNK: tl.constexpr,
    WNCAP: tl.constexpr,
    STORE: tl.constexpr, LOAD: tl.constexpr,
):
    a = tl.program_id(0)
    c0 = tl.program_id(1) * CHUNK
    offs_c = c0 + tl.arange(0, CHUNK)
    cm = offs_c < TC
    row = tl.load(rows_ptr + a).to(tl.int64)
    pos = tl.load(tok_ptr + a * TC + offs_c, mask=cm, other=0).to(tl.int64)
    offs_h = tl.arange(0, HKV)
    offs_d = tl.arange(0, ND2)
    if LOAD:
        base = row * (S_CAP * HKV * ND2) + pos * (HKV * ND2)
        addr = base[:, None, None] + (offs_h[None, :, None] * ND2 + offs_d[None, None, :])
        kq = tl.load(kq_ptr + addr, mask=cm[:, None, None], other=0).to(tl.float32)
        base_s = row * (S_CAP * HKV) + pos * HKV
        addr_s = base_s[:, None] + offs_h[None, :]
        sc = tl.load(sc_ptr + addr_s, mask=cm[:, None], other=0.0)
        mn = tl.load(mn_ptr + addr_s, mask=cm[:, None], other=0.0)
        kq_c = kq * sc[:, :, None] + mn[:, :, None]
        q2v = tl.load(q2_ptr + a * HKV * ND2 + offs_h[:, None] * ND2 + offs_d[None, :])
        s = tl.sum(kq_c * q2v[None, :, :], axis=2)
    else:
        s = tl.full((CHUNK, HKV), 1.0, dtype=tl.float32)  # 假数据，写侧带宽不依赖值
    far_hi = tl.load(far_hi_ptr + a)
    sw_lo = tl.load(sw_lo_ptr + a)
    in_far = (pos >= FAR_LO) & (pos < far_hi)
    in_near = (pos < sw_lo) & (~in_far)
    if STORE:
        far_v = tl.where(in_far[:, None], s, float("-inf"))
        o_off = a * HKV * TC + offs_h[None, :] * TC + offs_c[:, None]
        tl.store(far_ptr + o_off, far_v, mask=cm[:, None])
        ps = tl.load(ps_ptr + a)
        pf = tl.load(pf_ptr + a)
        slot = tl.where(pos < FAR_LO, offs_c, ps + (offs_c - pf))
        nm = in_near & cm
        near_v = tl.where(in_near[:, None], s, float("-inf"))
        tl.store(
            near_ptr + a * HKV * WNCAP + offs_h[None, :] * WNCAP + slot[:, None],
            near_v, mask=nm[:, None],
        )
        tl.store(near_tok_ptr + a * WNCAP + slot, pos, mask=nm)


def alloc_out():
    far_sc = torch.empty(n, Hkv, Tc, dtype=torch.float32, device=dev)
    near_sc = torch.full((n, Hkv, WNCAP), float("-inf"), dtype=torch.float32, device=dev)
    near_tok = torch.full((n, WNCAP), S_cap, dtype=torch.int64, device=dev)
    return far_sc, near_sc, near_tok


def run_var(chunk, nw, store=True, load=True):
    far_sc, near_sc, near_tok = alloc_out()
    grid = (n, triton.cdiv(Tc, chunk))
    _dual_var_kernel[grid](
        q2, kq_q, kq_sc, kq_mn, rows_l, tok_c,
        far_sc, near_sc, near_tok,
        far_hi_t, sw_lo_t,
        ps.to(torch.int32).contiguous(), pf.to(torch.int32).contiguous(),
        HKV=Hkv, ND2=ND2, TC=Tc, S_CAP=S_cap, FAR_LO=FAR_LO,
        CHUNK=chunk, WNCAP=WNCAP, STORE=store, LOAD=load, num_warps=nw,
    )
    return far_sc, near_sc, near_tok


# 对拍基准（full 变体 vs 生产 kernel）
from sglang.srt.layers.attention.tli.kernels import tli_l2_score_batched_dual

ref_far, ref_near, ref_tok = tli_l2_score_batched_dual(
    q2, kq_q, kq_sc, kq_mn, rows_l, tok_c, FAR_LO, far_hi_t, sw_lo_t,
    near_compact=True, wncap=WNCAP, sent=S_cap,
)
v_far, v_near, v_tok = run_var(CHUNK, 8)
eq = (torch.equal(v_far, ref_far) and torch.equal(v_near, ref_near)
      and torch.equal(v_tok, ref_tok))
print(f"对拍 full 变体 vs 生产 kernel: {'PASS' if eq else 'FAIL'}")
assert eq

results = {}
# ---- A. 分解（CHUNK=128, nw=8 生产口径）----
t_full = bench(lambda: run_var(CHUNK, 8))
t_nostore = bench(lambda: run_var(CHUNK, 8, store=False))
t_noload = bench(lambda: run_var(CHUNK, 8, load=False))
results["A_full_ms"] = round(t_full, 4)
results["A_nostore_ms"] = round(t_nostore, 4)
results["A_noload_ms"] = round(t_noload, 4)
print(f"\n[A 分解 @CHUNK=128/nw=8]")
print(f"  full (读+算+双写) : {t_full:6.3f} ms")
print(f"  nostore (读+算)   : {t_nostore:6.3f} ms  → 写侧成本 ≈ {t_full - t_nostore:.3f} ms")
print(f"  noload (写 only)  : {t_noload:6.3f} ms  → 读+算成本 ≈ {t_full - t_noload:.3f} ms")

# ---- B. CHUNK × num_warps sweep ----
print(f"\n[B sweep]")
best = (None, t_full)
for chunk in (64, 128, 256):
    for nw in (4, 8, 16):
        try:
            t = bench(lambda: run_var(chunk, nw))
        except Exception as e:
            print(f"  CHUNK={chunk} nw={nw}: {type(e).__name__}")
            continue
        # 正确性抽查（边界 mask 生效性）
        vf, vn, vt = run_var(chunk, nw)
        okk = torch.equal(vf, ref_far) and torch.equal(vn, ref_near) and torch.equal(vt, ref_tok)
        print(f"  CHUNK={chunk:3d} nw={nw:2d}: {t:6.3f} ms  equal={'✓' if okk else '✗'}")
        results[f"B_c{chunk}_w{nw}_ms"] = round(t, 4)
        if okk and t < best[1]:
            best = ((chunk, nw), t)
results["B_best"] = f"chunk={best[0][0]},nw={best[0][1]}" if best[0] else "baseline"
print(f"  → best: {results['B_best']} @ {best[1]:.3f} ms")

# ---- D. 带宽参照 ----
src = torch.randn(n * Hkv * Tc, dtype=torch.float32, device=dev)
dst = torch.empty_like(src)
t_copy = bench(lambda: dst.copy_(src))
bw_copy = 2 * src.numel() * 4 / t_copy / 1e9
print(f"\n[D 参照] D2D copy {t_copy:.3f} ms → {bw_copy:.0f} GB/s (R+W)")
results["D_copy_ms"] = round(t_copy, 4)
results["D_copy_GBps"] = round(bw_copy, 1)

# 流量账
rd = Tc * n * (Hkv * ND2 + Hkv * 8) / 1e9  # kq(uint8) + sc/mn(fp32×2)
wr = (n * Hkv * Tc + n * Hkv * WNCAP) * 4 / 1e9
results["traffic_read_GB"] = round(rd, 3)
results["traffic_write_GB"] = round(wr, 3)
ideal = (rd + wr) / 3.0  # 3TB/s 可达带宽口径
print(f"流量账: 读 {rd:.2f}GB + 写 {wr:.2f}GB；@3TB/s 理想 {ideal * 1e3:.0f} µs")
results["ideal_us_3TBps"] = round(ideal * 1e3, 1)

json.dump(results, open("/home/wangyuanshuo02/sglang/tli_m8_tma_breakdown.json", "w"),
          indent=1, ensure_ascii=False)
print("saved tli_m8_tma_breakdown.json")
