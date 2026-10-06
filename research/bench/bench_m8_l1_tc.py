# M8-TC：L1 块上界打分的 TC 化实验（#28，kernel 级 microbench，合成数据标注）
# 现状诊断：_tli_l1_score_batched_kernel 的 tile 是 [MBLK, DP]、块间 stride=Hkv*DP*4=1KB，
# 每 1KB 段只用 128B —— 合并访问效率 12.5%（实测 ~360GB/s 量级）。
# 方案 A（合并访存版）：单 program 覆盖 [MBLK2, Hkv, DP] 整段连续 1KB——cache line 全用，
#   按头分组归约同 eager（先 G-sum 后点积）。
# 方案 B（tl.dot TC 版）：k_tile [MBLK2, Hkv*DP] @ W[Hkv*DP, Hkv_pad16]（W=块对角 q），
#   fp32 dot（TF32 精度风险）与 ieee dot 双口径。
# 口径：bs=32 × S=131K（NBLK=2048）合成数据（对齐 bench_m8_l2_kernel.py 同款口径），
# 对拍 = sc1 allclose（vs eager，1e-5）+ topk 块 id 集合一致率。
import sys

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
import time

import torch
import triton
import triton.language as tl


# ---- 方案 A：合并访存（[MBLK2, HKV, DP] 连续 tile + 头分组归约）----
@triton.jit
def _l1_coalesced_kernel(
    q_ptr, idx1_ptr, kmin_ptr, kmax_ptr, rows_ptr, nblk_ptr, t_ptr, sc1_ptr,
    H, D, BS,
    HKV: tl.constexpr, G: tl.constexpr, DP: tl.constexpr,
    NBLK, MBLK2: tl.constexpr,
):
    a = tl.program_id(0)  # 请求
    j = tl.program_id(1)
    offs_m = j * MBLK2 + tl.arange(0, MBLK2)
    mm = offs_m < NBLK
    offs_h = tl.arange(0, HKV)
    offs_d = tl.arange(0, DP)
    d_idx = tl.load(idx1_ptr + offs_d)
    # G-sum q（与 eager einsum 同分组：先 G-sum 后点积）
    q_pos = tl.zeros([HKV, DP], dtype=tl.float32)
    q_neg = tl.zeros([HKV, DP], dtype=tl.float32)
    for g in range(G):
        # q[a, h*G+g, d_idx] → [HKV, DP]
        qv = tl.load(
            q_ptr + (a * H + offs_h[:, None] * G + g) * D + d_idx[None, :]
        ).to(tl.float32)
        q_pos += tl.maximum(qv, 0.0)
        q_neg += tl.minimum(qv, 0.0)
    row = tl.load(rows_ptr + a).to(tl.int64)
    # [MBLK2, HKV, DP] 连续 tile（块 m 的 Hkv*DP*4=1KB 全用）
    addr = (
        (row * NBLK + offs_m[:, None, None]).to(tl.int64) * (HKV * DP)
        + offs_h[None, :, None] * DP + offs_d[None, None, :]
    )
    m3 = mm[:, None, None]
    kmin = tl.load(kmin_ptr + addr, mask=m3, other=0.0).to(tl.float32)
    kmax = tl.load(kmax_ptr + addr, mask=m3, other=0.0).to(tl.float32)
    sc = tl.sum(q_pos[None] * kmax, axis=2) + tl.sum(q_neg[None] * kmin, axis=2)
    # [MBLK2, HKV] → valid -inf → 转置写 sc1[a, h, m]
    nblk_a = tl.load(nblk_ptr + a).to(tl.int32)
    t_a = tl.load(t_ptr + a).to(tl.int32)
    valid = mm[:, None] & (offs_m[:, None] < nblk_a) & (
        (offs_m[:, None] + 1) * BS - 1 <= t_a
    )
    sc = tl.where(valid, sc, float("-inf"))
    tl.store(
        sc1_ptr + a.to(tl.int64) * HKV * NBLK + offs_h[None, :] * NBLK + offs_m[:, None],
        sc,
    )


def l1_coalesced(q, idx1, kmin_pool, kmax_pool, rows, nblk_t, t_t, bs, mblk2=32):
    n, H, D = q.shape
    HKV, DP, NBLK = kmin_pool.shape[2], kmin_pool.shape[3], kmin_pool.shape[1]
    G = H // HKV
    sc1 = torch.empty(n, HKV, NBLK, dtype=torch.float32, device=q.device)
    grid = (n, triton.cdiv(NBLK, mblk2))
    _l1_coalesced_kernel[grid](
        q, idx1, kmin_pool, kmax_pool,
        rows.to(torch.long).contiguous(), nblk_t.to(torch.long).contiguous(),
        t_t.to(torch.long).contiguous(), sc1, H, D, bs,
        HKV=HKV, G=G, DP=DP, NBLK=NBLK, MBLK2=mblk2, num_warps=8,
    )
    return sc1


# ---- 方案 B：tl.dot TC 版（块对角 W，两个独立 dot：kmax·Wpos + kmin·Wneg）----
@triton.jit
def _l1_dot_kernel(
    q_ptr, idx1_ptr, kmin_ptr, kmax_ptr, rows_ptr, nblk_ptr, t_ptr, sc1_ptr,
    H, D, BS,
    HKV: tl.constexpr, G: tl.constexpr, DP: tl.constexpr,
    NBLK, MBLK2: tl.constexpr, HPAD: tl.constexpr,
    IEEE: tl.constexpr,
):
    a = tl.program_id(0)
    j = tl.program_id(1)
    offs_m = j * MBLK2 + tl.arange(0, MBLK2)
    mm = offs_m < NBLK
    offs_h = tl.arange(0, HKV)
    offs_d = tl.arange(0, DP)
    offs_hp = tl.arange(0, HPAD)
    d_idx = tl.load(idx1_ptr + offs_d)
    q_pos = tl.zeros([HKV, DP], dtype=tl.float32)
    q_neg = tl.zeros([HKV, DP], dtype=tl.float32)
    for g in range(G):
        qv = tl.load(
            q_ptr + (a * H + offs_h[:, None] * G + g) * D + d_idx[None, :]
        ).to(tl.float32)
        q_pos += tl.maximum(qv, 0.0)
        q_neg += tl.minimum(qv, 0.0)
    row = tl.load(rows_ptr + a).to(tl.int64)
    # k tile [MBLK2, HKV*DP]（连续 1KB/块）
    offs_hd = tl.arange(0, HKV * DP)
    addr = (row * NBLK + offs_m[:, None]).to(tl.int64) * (HKV * DP) + offs_hd[None, :]
    kx = tl.load(kmax_ptr + addr, mask=mm[:, None], other=0.0).to(tl.float32)
    kn = tl.load(kmin_ptr + addr, mask=mm[:, None], other=0.0).to(tl.float32)
    # 块对角 W [HKV*DP, HPAD]：W[h*DP+d, h] = qpos[h,d]
    qf_pos = tl.reshape(q_pos, [HKV * DP])
    qf_neg = tl.reshape(q_neg, [HKV * DP])
    h_of = offs_hd // DP
    Wp = tl.where(h_of[:, None] == offs_hp[None, :], qf_pos[:, None], 0.0)
    Wn = tl.where(h_of[:, None] == offs_hp[None, :], qf_neg[:, None], 0.0)
    prec: tl.constexpr = "ieee" if IEEE else "tf32"
    acc = tl.dot(kx, Wp, input_precision=prec) + tl.dot(kn, Wn, input_precision=prec)
    nblk_a = tl.load(nblk_ptr + a).to(tl.int32)
    t_a = tl.load(t_ptr + a).to(tl.int32)
    valid = mm[:, None] & (offs_m[:, None] < nblk_a) & (
        (offs_m[:, None] + 1) * BS - 1 <= t_a
    )
    acc = tl.where(valid, acc, float("-inf"))
    # store [MBLK2, HPAD] → sc1[a, h, m]（列掩 h < HKV；pad 列丢弃）
    tl.store(
        sc1_ptr + a.to(tl.int64) * HKV * NBLK + offs_hp[None, :] * NBLK + offs_m[:, None],
        acc,
        mask=(offs_hp[None, :] < HKV),
    )


def l1_dot(q, idx1, kmin_pool, kmax_pool, rows, nblk_t, t_t, bs,
           mblk2=64, ieee=True):
    n, H, D = q.shape
    HKV, DP, NBLK = kmin_pool.shape[2], kmin_pool.shape[3], kmin_pool.shape[1]
    G = H // HKV
    sc1 = torch.empty(n, HKV, NBLK, dtype=torch.float32, device=q.device)
    grid = (n, triton.cdiv(NBLK, mblk2))
    _l1_dot_kernel[grid](
        q, idx1, kmin_pool, kmax_pool,
        rows.to(torch.long).contiguous(), nblk_t.to(torch.long).contiguous(),
        t_t.to(torch.long).contiguous(), sc1, H, D, bs,
        HKV=HKV, G=G, DP=DP, NBLK=NBLK, MBLK2=mblk2, HPAD=16, IEEE=ieee,
        num_warps=8,
    )
    return sc1


def bench(fn, iters=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def main():
    torch.manual_seed(0)
    dev = "cuda:0"
    n, S = 32, 131072
    H, HKV, D, DP = 32, 8, 128, 32
    G = H // HKV
    BS = 64
    NBLK = S // BS  # 2048
    R = 33
    # 合成数据（microbench 口径；量纲对齐生产 bs32/131K）
    q = torch.randn(n, H, D, device=dev)
    idx1 = torch.tensor(list(range(48, 64)) + list(range(112, 128)),
                        device=dev, dtype=torch.long)
    kmin_pool = (torch.randn(R, NBLK, HKV, DP, device=dev) * 0.5 - 0.3)
    kmax_pool = kmin_pool + torch.rand(R, NBLK, HKV, DP, device=dev)
    rows = torch.randperm(R, device=dev)[:n].to(torch.long)
    S_t = torch.full((n,), S, device=dev, dtype=torch.long)
    t_t = S_t - 1
    nblk_t = (S_t + BS - 1) // BS

    # eager P1+P2 基准
    def eager():
        kmin_b = kmin_pool[rows]
        kmax_b = kmax_pool[rows]
        qs = q[..., idx1]
        qg = qs.clamp(min=0).reshape(n, HKV, G, DP)
        qn = qs.clamp(max=0).reshape(n, HKV, G, DP)
        sc1 = torch.einsum("ahgd,amhd->ahm", qg, kmax_b) + torch.einsum(
            "ahgd,amhd->ahm", qn, kmin_b
        )
        blk_id = torch.arange(NBLK, device=dev)
        valid = (blk_id.view(1, -1) < nblk_t.view(-1, 1)) & (
            (blk_id + 1).view(1, -1) * BS - 1 <= t_t.view(-1, 1)
        )
        return sc1.masked_fill(~valid.unsqueeze(1), float("-inf"))

    # 生产 KernelD 基准
    from sglang.srt.layers.attention.tli.kernels import tli_l1_score_batched

    def kernel_d():
        return tli_l1_score_batched(q, idx1, kmin_pool, kmax_pool, rows,
                                     nblk_t, t_t, BS)

    sc_ref = eager()
    sc_d = kernel_d()
    K1 = 128

    def topk_ids(sc):
        return torch.topk(sc, K1, dim=-1).indices

    def match_rate(a, b):
        # per (a,h) 块 id 集合一致率
        A = topk_ids(a); B = topk_ids(b)
        eq = (torch.sort(A, dim=-1).values == torch.sort(B, dim=-1).values)
        return eq.all(dim=-1).float().mean().item()

    print("== 对拍 ==")
    print(f"KernelD vs eager: allclose={torch.allclose(sc_d, sc_ref, atol=1e-5)}, "
          f"maxdiff={(sc_d - sc_ref).abs().max().item():.2e}, "
          f"topk集合一致={match_rate(sc_d, sc_ref):.4f}")

    results = {}
    results["eager P1+P2"] = bench(eager)
    results["KernelD (生产)"] = bench(kernel_d)
    for mb in [16, 32, 64]:
        sc_a = l1_coalesced(q, idx1, kmin_pool, kmax_pool, rows, nblk_t, t_t, BS, mb)
        ok = torch.allclose(sc_a, sc_ref, atol=1e-5)
        mr = match_rate(sc_a, sc_ref)
        ms = bench(lambda mb=mb: l1_coalesced(
            q, idx1, kmin_pool, kmax_pool, rows, nblk_t, t_t, BS, mb))
        results[f"方案A 合并访存 MBLK2={mb}"] = ms
        print(f"方案A(MBLK2={mb}) vs eager: allclose={ok}, "
              f"maxdiff={(sc_a - sc_ref).abs().max().item():.2e}, "
              f"topk集合一致={mr:.4f}, {ms:.3f}ms")
    for ieee in [True, False]:
        try:
            sc_b = l1_dot(q, idx1, kmin_pool, kmax_pool, rows, nblk_t, t_t, BS,
                          64, ieee)
            ok = torch.allclose(sc_b, sc_ref, atol=1e-4)
            mr = match_rate(sc_b, sc_ref)
            ms = bench(lambda ie=ieee: l1_dot(
                q, idx1, kmin_pool, kmax_pool, rows, nblk_t, t_t, BS, 64, ie))
            tag = "ieee(FMA)" if ieee else "tf32(TC)"
            results[f"方案B tl.dot {tag}"] = ms
            print(f"方案B({tag}) vs eager: allclose(1e-4)={ok}, "
                  f"topk集合一致={mr:.4f}, {ms:.3f}ms")
        except Exception as e:
            print(f"方案B({'ieee' if ieee else 'tf32'}) 失败: {type(e).__name__}: {e}")

    print("\n== 汇总（ms，bs=32 × S=131K，合成 microbench）==")
    for k, v in results.items():
        print(f"  {k:34s} {v:8.3f}")
    base = results["eager P1+P2"]
    for k, v in results.items():
        print(f"  {k:34s} {base / v:7.2f}x vs eager")
    json_out = {k: round(v, 4) for k, v in results.items()}
    import json
    json.dump(json_out, open("/home/wangyuanshuo02/sglang/tli_l1_tc_bench.json", "w"),
              indent=1)
    print("saved tli_l1_tc_bench.json")


if __name__ == "__main__":
    main()
