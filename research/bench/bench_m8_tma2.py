# M8-2b 续：far_sc 写侧三变体对比（#51）
# 已定位：dual kernel 0.359ms 中写侧 ≈0.283ms（70MB → 0.25TB/s），根因是
# [CHUNK,HKV] tile 写 [n,Hkv,Tc] 布局的转置写（512B 段 × 8，段间 264KB）。
# 变体：
#   C1 block_ptr 写 far（tile [HKV,CHUNK]，boundary_check 吞尾块——须 Tc%CHUNK==0）
#   C2 TMA descriptor 写 far（host-side TensorDescriptor，异步 bulk 事务）
#   C3 布局改 [n,Tc,Hkv] 连续写（4KB 全合并段）+ fused/permute 转置回读口径，
#       下游 topk 前 contiguous 化的成本一并计账（full-chain 对比）
#   生产参照 = C0 现状（CHUNK=128, nw=4 已是最优 sweep 点）
# 对拍口径：C1/C2/C3 的 far_sc 物化结果与 C0 逐位 equal。
import os
import sys
import time

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
import json

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

dev = "cuda:0"
torch.manual_seed(0)
n, R = 32, 32
S_cap, Hkv, D = 131072, 8, 128
ND2 = 16
Tc = (128 * Hkv + 3) * 64  # 65728 = 128*513 整除
FAR_LO = 2 * 64
WNCAP = FAR_LO + (2048 - 128)
CHUNK = 128

kq_q = torch.randint(0, 16, (R, S_cap, Hkv, ND2), dtype=torch.uint8, device=dev)
kq_sc = torch.rand(R, S_cap, Hkv, device=dev) * 0.01
kq_mn = (torch.rand(R, S_cap, Hkv, device=dev) - 0.5) * 0.1
rows_l = torch.arange(n, device=dev, dtype=torch.long)
tok_c = torch.arange(Tc, device=dev, dtype=torch.long).unsqueeze(0).expand(n, Tc).contiguous()
q2 = torch.randn(n, Hkv, ND2, device=dev)
S_t = torch.full((n,), S_cap, device=dev, dtype=torch.long)
t_t = S_t - 1
far_hi_t = (t_t + 1 - 2048).clamp(min=FAR_LO)
sw_lo_t = (t_t - 128 + 1).clamp(min=0)
ps = (tok_c < FAR_LO).sum(dim=-1)
pf = (tok_c < far_hi_t.view(-1, 1)).sum(dim=-1)
ps_i = ps.to(torch.int32).contiguous()
pf_i = pf.to(torch.int32).contiguous()


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


# ---- C0 生产参照（nw=4，B sweep 最优点）----
@triton.jit
def _dual_c0(
    q2_ptr, kq_ptr, sc_ptr, mn_ptr, rows_ptr, tok_ptr,
    far_ptr, near_ptr, near_tok_ptr, far_hi_ptr, sw_lo_ptr, ps_ptr, pf_ptr,
    HKV: tl.constexpr, ND2: tl.constexpr, TC: tl.constexpr,
    S_CAP, FAR_LO: tl.constexpr, CHUNK: tl.constexpr, WNCAP: tl.constexpr,
):
    a = tl.program_id(0)
    c0 = tl.program_id(1) * CHUNK
    offs_c = c0 + tl.arange(0, CHUNK)
    cm = offs_c < TC
    row = tl.load(rows_ptr + a).to(tl.int64)
    pos = tl.load(tok_ptr + a * TC + offs_c, mask=cm, other=0).to(tl.int64)
    offs_h = tl.arange(0, HKV)
    offs_d = tl.arange(0, ND2)
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
    far_hi = tl.load(far_hi_ptr + a)
    sw_lo = tl.load(sw_lo_ptr + a)
    in_far = (pos >= FAR_LO) & (pos < far_hi)
    in_near = (pos < sw_lo) & (~in_far)
    far_v = tl.where(in_far[:, None], s, float("-inf"))
    o_off = a * HKV * TC + offs_h[None, :] * TC + offs_c[:, None]
    tl.store(far_ptr + o_off, far_v, mask=cm[:, None])
    ps = tl.load(ps_ptr + a)
    pf = tl.load(pf_ptr + a)
    slot = tl.where(pos < FAR_LO, offs_c, ps + (offs_c - pf))
    nm = in_near & cm
    near_v = tl.where(in_near[:, None], s, float("-inf"))
    tl.store(near_ptr + a * HKV * WNCAP + offs_h[None, :] * WNCAP + slot[:, None],
             near_v, mask=nm[:, None])
    tl.store(near_tok_ptr + a * WNCAP + slot, pos, mask=nm)


# ---- C1/C2 共用主体：far 写抽离成 WRITE 变体（0=普通, 1=block_ptr, 2=TMA desc）----
@triton.jit
def _dual_c12(
    q2_ptr, kq_ptr, sc_ptr, mn_ptr, rows_ptr, tok_ptr,
    far_ptr, far_bp_ptr, far_desc,
    near_ptr, near_tok_ptr, far_hi_ptr, sw_lo_ptr, ps_ptr, pf_ptr,
    HKV: tl.constexpr, ND2: tl.constexpr, TC: tl.constexpr,
    S_CAP, FAR_LO: tl.constexpr, CHUNK: tl.constexpr, WNCAP: tl.constexpr,
    WRITE: tl.constexpr,
):
    a = tl.program_id(0)
    c0 = tl.program_id(1) * CHUNK
    offs_c = c0 + tl.arange(0, CHUNK)
    cm = offs_c < TC
    row = tl.load(rows_ptr + a).to(tl.int64)
    pos = tl.load(tok_ptr + a * TC + offs_c, mask=cm, other=0).to(tl.int64)
    offs_h = tl.arange(0, HKV)
    offs_d = tl.arange(0, ND2)
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
    far_hi = tl.load(far_hi_ptr + a)
    sw_lo = tl.load(sw_lo_ptr + a)
    in_far = (pos >= FAR_LO) & (pos < far_hi)
    in_near = (pos < sw_lo) & (~in_far)
    far_v = tl.where(in_far[:, None], s, float("-inf"))  # [CHUNK, HKV]
    if WRITE == 0:
        o_off = a * HKV * TC + offs_h[None, :] * TC + offs_c[:, None]
        tl.store(far_ptr + o_off, far_v, mask=cm[:, None])
    elif WRITE == 1:
        # block_ptr：2D [(n*HKV), TC]，块 [HKV, CHUNK]，order=(1,0) 列连续
        bp = tl.make_block_ptr(
            base=far_bp_ptr, shape=(2048 * 1024, TC),  # shape 仅用于 stride 推导的哨兵；用真实乘积
            strides=(TC, 1), offsets=(a * HKV, c0),
            block_shape=(HKV, CHUNK), order=(1, 0),
        )
        tl.store(bp, tl.trans(far_v), boundary_check=(0, 1))
    else:
        # TMA descriptor store：tile [HKV, CHUNK]
        far_desc.store([a * HKV, c0], tl.trans(far_v))
    ps = tl.load(ps_ptr + a)
    pf = tl.load(pf_ptr + a)
    slot = tl.where(pos < FAR_LO, offs_c, ps + (offs_c - pf))
    nm = in_near & cm
    near_v = tl.where(in_near[:, None], s, float("-inf"))
    tl.store(near_ptr + a * HKV * WNCAP + offs_h[None, :] * WNCAP + slot[:, None],
             near_v, mask=nm[:, None])
    tl.store(near_tok_ptr + a * WNCAP + slot, pos, mask=nm)


# ---- C3：far 写 [n, Tc, Hkv] 连续布局 ----
@triton.jit
def _dual_c3(
    q2_ptr, kq_ptr, sc_ptr, mn_ptr, rows_ptr, tok_ptr,
    far_ptr, near_ptr, near_tok_ptr, far_hi_ptr, sw_lo_ptr, ps_ptr, pf_ptr,
    HKV: tl.constexpr, ND2: tl.constexpr, TC: tl.constexpr,
    S_CAP, FAR_LO: tl.constexpr, CHUNK: tl.constexpr, WNCAP: tl.constexpr,
):
    a = tl.program_id(0)
    c0 = tl.program_id(1) * CHUNK
    offs_c = c0 + tl.arange(0, CHUNK)
    cm = offs_c < TC
    row = tl.load(rows_ptr + a).to(tl.int64)
    pos = tl.load(tok_ptr + a * TC + offs_c, mask=cm, other=0).to(tl.int64)
    offs_h = tl.arange(0, HKV)
    offs_d = tl.arange(0, ND2)
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
    far_hi = tl.load(far_hi_ptr + a)
    sw_lo = tl.load(sw_lo_ptr + a)
    in_far = (pos >= FAR_LO) & (pos < far_hi)
    in_near = (pos < sw_lo) & (~in_far)
    far_v = tl.where(in_far[:, None], s, float("-inf"))
    # 连续写：far[a, c, h] —— tile [CHUNK, HKV] 每 token 32B 连续，整 tile 4KB 连续
    o_off = a * TC * HKV + offs_c[:, None] * HKV + offs_h[None, :]
    tl.store(far_ptr + o_off, far_v, mask=cm[:, None])
    ps = tl.load(ps_ptr + a)
    pf = tl.load(pf_ptr + a)
    slot = tl.where(pos < FAR_LO, offs_c, ps + (offs_c - pf))
    nm = in_near & cm
    near_v = tl.where(in_near[:, None], s, float("-inf"))
    tl.store(near_ptr + a * HKV * WNCAP + offs_h[None, :] * WNCAP + slot[:, None],
             near_v, mask=nm[:, None])
    tl.store(near_tok_ptr + a * WNCAP + slot, pos, mask=nm)


def alloc_far(layout="hkv"):
    if layout == "hkv":
        return torch.empty(n, Hkv, Tc, dtype=torch.float32, device=dev)
    return torch.empty(n, Tc, Hkv, dtype=torch.float32, device=dev)


def alloc_near():
    near_sc = torch.full((n, Hkv, WNCAP), float("-inf"), dtype=torch.float32, device=dev)
    near_tok = torch.full((n, WNCAP), S_cap, dtype=torch.int64, device=dev)
    return near_sc, near_tok


grid = (n, triton.cdiv(Tc, CHUNK))

# C0 参照
far0 = alloc_far()
ns0, nt0 = alloc_near()
_dual_c0[grid](q2, kq_q, kq_sc, kq_mn, rows_l, tok_c, far0, ns0, nt0,
               far_hi_t, sw_lo_t, ps_i, pf_i,
               HKV=Hkv, ND2=ND2, TC=Tc, S_CAP=S_cap, FAR_LO=FAR_LO,
               CHUNK=CHUNK, WNCAP=WNCAP, num_warps=4)
t_c0 = bench(lambda: _dual_c0[grid](q2, kq_q, kq_sc, kq_mn, rows_l, tok_c, far0, ns0, nt0,
                                     far_hi_t, sw_lo_t, ps_i, pf_i,
                                     HKV=Hkv, ND2=ND2, TC=Tc, S_CAP=S_cap, FAR_LO=FAR_LO,
                                     CHUNK=CHUNK, WNCAP=WNCAP, num_warps=4))
print(f"C0 生产参照 (nw=4)        : {t_c0:6.3f} ms")

res = {"C0_ms": round(t_c0, 4)}

# C1 block_ptr
far1 = alloc_far()
ns1, nt1 = alloc_near()
try:
    _dual_c12[grid](q2, kq_q, kq_sc, kq_mn, rows_l, tok_c, far1, far1, far1,
                    ns1, nt1, far_hi_t, sw_lo_t, ps_i, pf_i,
                    HKV=Hkv, ND2=ND2, TC=Tc, S_CAP=S_cap, FAR_LO=FAR_LO,
                    CHUNK=CHUNK, WNCAP=WNCAP, WRITE=1, num_warps=4)
    eq1 = torch.equal(far1, far0) and torch.equal(ns1, ns0) and torch.equal(nt1, nt0)
    t_c1 = bench(lambda: _dual_c12[grid](q2, kq_q, kq_sc, kq_mn, rows_l, tok_c,
                                         far1, far1, far1, ns1, nt1,
                                         far_hi_t, sw_lo_t, ps_i, pf_i,
                                         HKV=Hkv, ND2=ND2, TC=Tc, S_CAP=S_cap, FAR_LO=FAR_LO,
                                         CHUNK=CHUNK, WNCAP=WNCAP, WRITE=1, num_warps=4))
    print(f"C1 block_ptr 写 far       : {t_c1:6.3f} ms  equal={'✓' if eq1 else '✗'}")
    res["C1_ms"] = round(t_c1, 4)
except Exception as e:
    print(f"C1 block_ptr 写 far       : {type(e).__name__}: {str(e)[:200]}")

# C2 TMA descriptor store
far2 = alloc_far()
far2_2d = far2.view(n * Hkv, Tc)
ns2, nt2 = alloc_near()
try:
    desc = TensorDescriptor.from_tensor(far2_2d, block_shape=[Hkv, CHUNK])
    _dual_c12[grid](q2, kq_q, kq_sc, kq_mn, rows_l, tok_c, far2, far2, desc,
                    ns2, nt2, far_hi_t, sw_lo_t, ps_i, pf_i,
                    HKV=Hkv, ND2=ND2, TC=Tc, S_CAP=S_cap, FAR_LO=FAR_LO,
                    CHUNK=CHUNK, WNCAP=WNCAP, WRITE=2, num_warps=4)
    eq2 = torch.equal(far2, far0) and torch.equal(ns2, ns0) and torch.equal(nt2, nt0)
    t_c2 = bench(lambda: _dual_c12[grid](q2, kq_q, kq_sc, kq_mn, rows_l, tok_c,
                                         far2, far2, desc, ns2, nt2,
                                         far_hi_t, sw_lo_t, ps_i, pf_i,
                                         HKV=Hkv, ND2=ND2, TC=Tc, S_CAP=S_cap, FAR_LO=FAR_LO,
                                         CHUNK=CHUNK, WNCAP=WNCAP, WRITE=2, num_warps=4))
    print(f"C2 TMA desc 写 far        : {t_c2:6.3f} ms  equal={'✓' if eq2 else '✗'}")
    res["C2_ms"] = round(t_c2, 4)
except Exception as e:
    print(f"C2 TMA desc 写 far        : {type(e).__name__}: {str(e)[:300]}")

# C3 布局 [n,Tc,Hkv] 连续写 + 转置回 [n,Hkv,Tc]
far3 = alloc_far("c_h")
ns3, nt3 = alloc_near()
_dual_c3[grid](q2, kq_q, kq_sc, kq_mn, rows_l, tok_c, far3, ns3, nt3,
               far_hi_t, sw_lo_t, ps_i, pf_i,
               HKV=Hkv, ND2=ND2, TC=Tc, S_CAP=S_cap, FAR_LO=FAR_LO,
               CHUNK=CHUNK, WNCAP=WNCAP, num_warps=4)
far3_t = far3.permute(0, 2, 1).contiguous()
eq3 = torch.equal(far3_t, far0) and torch.equal(ns3, ns0) and torch.equal(nt3, nt0)


def c3_chain():
    _dual_c3[grid](q2, kq_q, kq_sc, kq_mn, rows_l, tok_c, far3, ns3, nt3,
                   far_hi_t, sw_lo_t, ps_i, pf_i,
                   HKV=Hkv, ND2=ND2, TC=Tc, S_CAP=S_cap, FAR_LO=FAR_LO,
                   CHUNK=CHUNK, WNCAP=WNCAP, num_warps=4)
    return far3.permute(0, 2, 1).contiguous()


t_c3 = bench(lambda: _dual_c3[grid](q2, kq_q, kq_sc, kq_mn, rows_l, tok_c, far3, ns3, nt3,
                                    far_hi_t, sw_lo_t, ps_i, pf_i,
                                    HKV=Hkv, ND2=ND2, TC=Tc, S_CAP=S_cap, FAR_LO=FAR_LO,
                                    CHUNK=CHUNK, WNCAP=WNCAP, num_warps=4))
t_c3_chain = bench(c3_chain)
print(f"C3 连续布局写             : {t_c3:6.3f} ms  equal={'✓' if eq3 else '✗'}")
print(f"C3 + permute.contiguous   : {t_c3_chain:6.3f} ms（full-chain 口径）")
res["C3_ms"] = round(t_c3, 4)
res["C3_chain_ms"] = round(t_c3_chain, 4)

# ---- full-chain 语义对比：写+far topk ----
W_far = 256


def chain_c0():
    _dual_c0[grid](q2, kq_q, kq_sc, kq_mn, rows_l, tok_c, far0, ns0, nt0,
                   far_hi_t, sw_lo_t, ps_i, pf_i,
                   HKV=Hkv, ND2=ND2, TC=Tc, S_CAP=S_cap, FAR_LO=FAR_LO,
                   CHUNK=CHUNK, WNCAP=WNCAP, num_warps=4)
    return torch.topk(far0, W_far, dim=-1)


def chain_c3():
    _dual_c3[grid](q2, kq_q, kq_sc, kq_mn, rows_l, tok_c, far3, ns3, nt3,
                   far_hi_t, sw_lo_t, ps_i, pf_i,
                   HKV=Hkv, ND2=ND2, TC=Tc, S_CAP=S_cap, FAR_LO=FAR_LO,
                   CHUNK=CHUNK, WNCAP=WNCAP, num_warps=4)
    return torch.topk(far3.permute(0, 2, 1).contiguous(), W_far, dim=-1)


t_ch0 = bench(chain_c0)
t_ch3 = bench(chain_c3)
print(f"\nfull-chain（写+far topk）: C0 {t_ch0:.3f} ms vs C3 {t_ch3:.3f} ms")
v0 = chain_c0()[1]
v3 = chain_c3()[1]
print(f"topk 值逐位一致: {'✓' if torch.equal(v0, v3) else '✗'}")
res["chain_C0_ms"] = round(t_ch0, 4)
res["chain_C3_ms"] = round(t_ch3, 4)

json.dump(res, open("/home/wangyuanshuo02/sglang/tli_m8_tma_variants.json", "w"),
          indent=1, ensure_ascii=False)
print("saved tli_m8_tma_variants.json")
