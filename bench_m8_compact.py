# M8 KernelB：候选压实块展开 kernel（P4，3.8%@bs32/131K）
# eager：repeat_interleave [n,S_cap] + where + topk-min(k=Tc) 全排序 ≈ 0.88ms。
# kernel：cumsum(onehot) 前缀 + grid (n, NBLK/BPC) 逐块展开 64 token（升序天然
# 保持），非因果尾 token 写哨兵 S_cap，尾部 pad 预填充。
# 语义差异（可接受）：eager topk-min 输出哨兵全在尾部升序；kernel 哨兵可在中段
# （非因果尾块内）——下游 valid = tok < S_cap 掩掉，与位置无关，有效集一致。
import sys

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
import time

import torch
import triton
import triton.language as tl


@triton.jit
def _tli_compact_kernel(
    onehot_ptr, prefix_ptr, tok_ptr, s_t_ptr,
    NBLK: tl.constexpr, BS: tl.constexpr, TC: tl.constexpr,
    BPC: tl.constexpr, SENT,
):
    a = tl.program_id(0)
    j = tl.program_id(1)
    offs_b = j * BPC + tl.arange(0, BPC)
    bm = offs_b < NBLK
    oh = tl.load(onehot_ptr + a * NBLK + offs_b, mask=bm, other=0)  # uint8/bool
    pref = tl.load(prefix_ptr + a * NBLK + offs_b, mask=bm, other=0)  # 含 cumsum
    s_t = tl.load(s_t_ptr + a)  # int64
    # 选中块首 token 输出槽 = (pref-1)*BS
    slot = (pref - 1) * BS
    offs_i = tl.arange(0, BS)
    p = offs_b[:, None] * BS + offs_i[None, :]  # [BPC, BS]
    v = tl.where(p < s_t, p, SENT)  # 非因果尾 → 哨兵
    st_mask = (oh > 0)[:, None] & bm[:, None] & ((slot[:, None] + BS) <= TC)
    tl.store(tok_ptr + a * TC + slot[:, None] + offs_i[None, :], v, mask=st_mask)


def tli_compact(onehot: torch.Tensor, S_t: torch.Tensor, S_cap: int, Tc: int):
    """onehot [n, NBLK] bool；S_t [n] int64；返回 tok [n, Tc] int64（哨兵 S_cap）。
    prefix 须为含 cumsum（int32/int64）[n, NBLK]。"""
    n, NBLK = onehot.shape
    BS = S_cap // NBLK  # block_size（S_cap = NBLK*BS 对齐口径）
    tok = torch.full((n, Tc), S_cap, dtype=torch.int64, device=onehot.device)
    prefix = torch.cumsum(onehot.to(torch.int32), dim=1)
    BPC = 32
    grid = (n, triton.cdiv(NBLK, BPC))
    _tli_compact_kernel[grid](
        onehot.view(torch.uint8) if onehot.dtype == torch.bool else onehot,
        prefix, tok, S_t.to(torch.long).contiguous(),
        NBLK=NBLK, BS=BS, TC=Tc, BPC=BPC, SENT=S_cap,
        num_warps=4,
    )
    return tok


if __name__ == "__main__":
    dev = "cuda:0"
    torch.manual_seed(0)
    n, S_cap, bs = 32, 131072, 64
    NBLK = S_cap // bs
    K1, Hkv, sliding = 128, 8, 3
    Tc = (K1 * Hkv + sliding) * bs
    # 构造与生产同构的 onehot：每行恰好 ≤K1*Hkv 随机块 + 滑窗尾块（union 上界
    # 与生产一致：选中块数 ≤ Tc/bs，槽位恒不越界）
    g = torch.Generator(device=dev).manual_seed(1)
    S_t = torch.randint(S_cap // 2, S_cap + 1, (n,), generator=g, device=dev, dtype=torch.long)  # 有的行 S<S_cap（非满行）
    oh = torch.zeros(n, NBLK, dtype=torch.bool, device=dev)
    for a in range(n):
        nblk_a = int((S_t[a] + bs - 1) // bs)
        k = min(1024, nblk_a)
        sel = torch.randperm(nblk_a, generator=torch.Generator().manual_seed(a))[:k]
        oh[a, sel] = True
        last_blk = int((S_t[a] - 1) // bs)
        oh[a, max(0, last_blk - sliding + 1): last_blk + 1] = True
    onehot = oh.contiguous()

    # eager 参照（生产同构）
    def eager_p4():
        sel_mask = onehot.repeat_interleave(bs, dim=1)[:, :S_cap]
        pos = torch.arange(S_cap, device=dev)
        cand = sel_mask & (pos.view(1, S_cap) < S_t.view(-1, 1))
        seq_m = torch.where(cand, pos.view(1, S_cap), torch.full_like(pos, S_cap))
        return torch.topk(seq_m, Tc, dim=-1, largest=False).values

    tok_ref = eager_p4()
    tok_k = tli_compact(onehot, S_t, S_cap, Tc)
    torch.cuda.synchronize()
    # 有效集一致（逐行）
    for a in range(n):
        s0 = set(tok_ref[a].tolist()); s0.discard(S_cap)
        s1 = set(tok_k[a].tolist()); s1.discard(S_cap)
        assert s0 == s1, f"row {a} 集合不一致: |ref|={len(s0)} |k|={len(s1)}"
    print(f"有效集一致 PASS（每行 {len(s0)} token，Tc={Tc}）")

    def bench(fn, reps=30, warmup=5):
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

    t_e = bench(eager_p4)
    t_k = bench(lambda: tli_compact(onehot, S_t, S_cap, Tc))
    print(f"eager P4: {t_e:.3f} ms | kernel: {t_k:.3f} ms | 加速 {t_e/t_k:.2f}×")
