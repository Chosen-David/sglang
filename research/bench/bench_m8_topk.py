# M8-topk：select_decode_batched 剩余瓶颈归因（#50）
# 在生产形状（n=32, S=131K, Tc=66048, Hkv=8）下逐段计时：
#   L1 kernel / tli_compact / KernelC dual / far+near 两个 torch.topk / 配额与拼接尾段
# 并测 topk 候选优化：sorted=False、带宽账、以及池内有限项占比（band 结构）。
import os
import sys

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
import time

import torch

os.environ.setdefault("SGLANG_TLI_L2B_KERNEL", "1")
os.environ.setdefault("SGLANG_TLI_COMPACT_KERNEL", "1")
os.environ.setdefault("SGLANG_TLI_L1B_KERNEL", "1")
os.environ.setdefault("SGLANG_TLI_L2D_KERNEL", "1")

from sglang.srt.layers.attention.tli.config import TLIProfile
from sglang.srt.layers.attention.tli.indexer import TLIIndexer
from sglang.srt.layers.attention.tli.kernels import (
    tli_compact,
    tli_l1_score_batched,
    tli_l2_score_batched_dual,
)

dev = "cuda:0"
torch.manual_seed(0)
n, R = 32, 32
S_cap, Hkv, D = 131072, 8, 128
prof = TLIProfile()
idxer = TLIIndexer(prof, head_dim=D).to(dev)
nd2, d1, bs = idxer.nd2, prof.coarse_dim, prof.block_size
NBLK_CAP = S_cap // bs

pool = {
    "kq_q": torch.randint(0, 16, (R, S_cap, Hkv, nd2), dtype=torch.uint8, device=dev),
    "kq_sc": torch.rand(R, S_cap, Hkv, device=dev) * 0.01,
    "kq_mn": (torch.rand(R, S_cap, Hkv, device=dev) - 0.5) * 0.1,
    "kmin": (torch.rand(R, NBLK_CAP, Hkv, d1, device=dev) - 0.5) * 0.2 - 0.1,
    "kmax": (torch.rand(R, NBLK_CAP, Hkv, d1, device=dev) - 0.5) * 0.2 + 0.1,
}
rows = torch.arange(n, device=dev)
q = torch.randn(n, Hkv * 4, D, device=dev) * 0.3
S_list = [S_cap] * n

def bench(fn, reps=30, warmup=6):
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

# ---- 全函数现状 ----
t_full = bench(lambda: idxer.select_decode_batched(pool, rows, S_list, q))
print(f"select_decode_batched 全函数: {t_full:.3f} ms")

# ---- 逐段：构造生产中间量 ----
S_t = torch.tensor(S_list, device=dev, dtype=torch.long)
t_t = S_t - 1
nblk_t = (S_t + bs - 1) // bs
rows_l = rows.to(torch.long)
idx1 = idxer.idx1
K1 = min(prof.k1_blocks, NBLK_CAP)
Tc = min((K1 * Hkv + prof.sliding_blocks) * bs, NBLK_CAP * bs, S_cap)

sc1 = tli_l1_score_batched(q, idx1, pool["kmin"], pool["kmax"], rows_l, nblk_t, t_t, bs)
cand_blk = torch.topk(sc1, K1, dim=-1).indices
blk_id = torch.arange(NBLK_CAP, device=dev)
valid_blk = (blk_id.view(1, -1) < nblk_t.view(-1, 1)) & (
    (blk_id + 1).view(1, -1) * bs - 1 <= t_t.view(-1, 1)
)
onehot = torch.zeros(n, NBLK_CAP, dtype=torch.bool, device=dev)
sel_src = torch.gather(valid_blk, 1, cand_blk.reshape(n, -1))
onehot.scatter_(1, cand_blk.reshape(n, -1), sel_src)
f_blk = (t_t.view(-1, 1) // bs - torch.arange(prof.sliding_blocks, device=dev)).clamp(min=0)
onehot.scatter_(1, f_blk, True)
tok = tli_compact(onehot, S_t, bs, S_cap, Tc)
tok_c = tok.clamp(max=S_cap - 1)

q2 = idxer._q_refine(q).reshape(n, Hkv, 4, nd2).sum(2)
far_lo = prof.sink_blocks * bs
far_hi_t = (t_t + 1 - prof.near_len).clamp(min=far_lo)
sw_lo_t = (t_t - prof.sliding_window + 1).clamp(min=0)
far_sc, near_sc = tli_l2_score_batched_dual(
    q2, pool["kq_q"], pool["kq_sc"], pool["kq_mn"], rows_l, tok_c,
    far_lo, far_hi_t, sw_lo_t,
)

W_far = min(prof.far_tokens, max(0, prof.token_budget - prof.sliding_window - far_lo), Tc)
W_near = min(prof.token_budget, Tc)

# ---- 逐段计时 ----
t_l1 = bench(lambda: tli_l1_score_batched(q, idx1, pool["kmin"], pool["kmax"],
                                           rows_l, nblk_t, t_t, bs))
t_topk1 = bench(lambda: torch.topk(sc1, K1, dim=-1))  # L1 块 topk（也在链上）
t_compact = bench(lambda: tli_compact(onehot, S_t, bs, S_cap, Tc))
t_dual = bench(lambda: tli_l2_score_batched_dual(
    q2, pool["kq_q"], pool["kq_sc"], pool["kq_mn"], rows_l, tok_c,
    far_lo, far_hi_t, sw_lo_t))
t_topk_f = bench(lambda: torch.topk(far_sc, W_far, dim=-1))
t_topk_n = bench(lambda: torch.topk(near_sc, W_near, dim=-1))
t_topk_fn = bench(lambda: (torch.topk(far_sc, W_far, dim=-1, sorted=False),
                           torch.topk(near_sc, W_near, dim=-1, sorted=False)))

# 尾段（gather/where/cat）完整重放
tok_e = tok_c.unsqueeze(1).expand(n, Hkv, Tc)
k2_far_t = (far_hi_t - far_lo).clamp(max=W_far)
F_t = (t_t + 1 - sw_lo_t).clamp(min=0)
k2n_t = (prof.token_budget - k2_far_t - F_t).clamp(min=0)

def tail():
    i_f = torch.topk(far_sc, W_far, dim=-1).indices
    sc_f = torch.gather(far_sc, 2, i_f)
    sel_f = torch.gather(tok_e, 2, i_f)
    rank_f = torch.arange(W_far, device=dev).view(1, 1, -1) < k2_far_t.view(-1, 1, 1)
    keep_f = (sc_f != float("-inf")) & rank_f
    a = torch.where(~keep_f, torch.full_like(sel_f, S_cap), sel_f)
    i_n = torch.topk(near_sc, W_near, dim=-1).indices
    sc_n = torch.gather(near_sc, 2, i_n)
    sel_n = torch.gather(tok_e, 2, i_n)
    rank_n = torch.arange(W_near, device=dev).view(1, 1, -1) < k2n_t.view(-1, 1, 1)
    keep_n = (sc_n != float("-inf")) & rank_n
    b = torch.where(~keep_n, torch.full_like(sel_n, S_cap), sel_n)
    f_pos = sw_lo_t.view(-1, 1) + torch.arange(min(prof.sliding_window, Tc), device=dev)
    f_pad = torch.arange(min(prof.sliding_window, Tc), device=dev).view(1, -1) >= F_t.view(-1, 1)
    c = torch.where(f_pad, torch.full_like(f_pos, S_cap), f_pos).unsqueeze(1).expand(n, Hkv, -1)
    return torch.cat([a, b, c], dim=-1)

t_tail = bench(tail)

print(f"  L1 打分 kernel          : {t_l1:7.3f} ms")
print(f"  L1 块 topk (k={K1})       : {t_topk1:7.3f} ms")
print(f"  tli_compact             : {t_compact:7.3f} ms")
print(f"  KernelC dual (L2 打分)   : {t_dual:7.3f} ms")
print(f"  far topk (k={W_far}, Tc={Tc}) : {t_topk_f:7.3f} ms")
print(f"  near topk (k={W_near})      : {t_topk_n:7.3f} ms")
print(f"  双 topk sorted=False    : {t_topk_fn:7.3f} ms")
print(f"  尾段 (topk+gather+where) : {t_tail:7.3f} ms")

# ---- band 结构统计（topk 输入的有限项占比）----
fin_f = (far_sc > float("-inf")).float().mean().item()
fin_n = (near_sc > float("-inf")).float().mean().item()
print(f"\nfar_sc 有限项占比 {fin_f:.3f} / near_sc {fin_n:.3f}（Tc={Tc}）")
# far/near 有限项是否为 tok_c 的连续 band（sorted 前提下）
t0 = tok_c[0]
fm = far_sc[0, 0] > float("-inf")
# 检查 far band 连续性：finite 位置集合是否为一段区间
idx_fin = fm.nonzero().flatten()
if len(idx_fin):
    contiguous = (idx_fin[-1] - idx_fin[0] + 1) == len(idx_fin)
    print(f"far 有限项 band: [{idx_fin[0].item()}, {idx_fin[-1].item()}] "
          f"长度 {len(idx_fin)}, 连续={contiguous}")

import json
out = {
    "full_ms": round(t_full, 4),
    "l1_kernel_ms": round(t_l1, 4),
    "l1_block_topk_ms": round(t_topk1, 4),
    "compact_ms": round(t_compact, 4),
    "dual_kernel_ms": round(t_dual, 4),
    "far_topk_ms": round(t_topk_f, 4),
    "near_topk_ms": round(t_topk_n, 4),
    "dual_topk_nosort_ms": round(t_topk_fn, 4),
    "tail_ms": round(t_tail, 4),
    "far_finite_frac": round(fin_f, 4),
    "near_finite_frac": round(fin_n, 4),
}
json.dump(out, open("/home/wangyuanshuo02/sglang/tli_m8_topk_breakdown.json", "w"), indent=1)
print("saved tli_m8_topk_breakdown.json")
