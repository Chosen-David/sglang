# M10 归因：select_batched（prefill 路径）在 30K 尺度的分阶段成本。
# 合成 index + 真实形状（S=30K / nq 全宽 chunked），微基准口径（标注：合成数据
# 仅用于归因，对拍保证语义）。目标：解释 30K e2e prefill 787s（bs=8）中
# select_batched 占多少、哪个 phase 超线性。
#
# 计时粒度：torch profiler（CUDA kernel 级）+ 手动 phase 分解双口径——
# M8 教训：phase 插桩打点位置会骗人（clone/permute 时间被记入后续 phase），
# kernel 级 profiler 复核才算数。
import os
import sys

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")

import time

import torch

from sglang.srt.layers.attention.tli.config import TLIProfile
from sglang.srt.layers.attention.tli.indexer import TLIIndexer, kq_unpack

dev = "cuda:0"
torch.manual_seed(0)

S, Hkv, D, G = 30720, 8, 128, 4
H = Hkv * G
NQ = 8192  # 单次 select_batched 的 nq（sglang chunked prefill 每层调用口径
# 的量级；30K prompt 分 4 个 8192-token chunk，每 chunk 调一次/层）

prof = TLIProfile()
idx = TLIIndexer(prof, head_dim=D).to(dev)

# 合成 index（真实形状）：直接走 build_block_index 保证结构一致
k = torch.randn(S, Hkv, D, device=dev) * 0.3
index = idx.build_block_index(k.float())
q = torch.randn(NQ, H, D, device=dev) * 0.3
t_arr = torch.arange(NQ, device=dev) + (S - NQ)  # 末 chunk：t ∈ [S-NQ, S)
# 也测首 chunk（t ∈ [0, NQ)）——快路径判定覆盖率不同

def bench(fn, reps=5, warmup=2):
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

print(f"S={S} nq={NQ} row_chunk=64 → {NQ//64} chunks/调用")
for name, t0_off in [("末chunk(t=S-NQ..S)", S - NQ), ("首chunk(t=0..NQ)", 0)]:
    t_arr = torch.arange(NQ, device=dev) + t0_off
    ms = bench(lambda: idx.select_batched(index, q, t_arr))
    print(f"select_batched {name}: {ms:8.1f} ms")

# ---- 手动 phase 分解（末 chunk，单次调用，拆关键中间量）----
t_arr = torch.arange(NQ, device=dev) + (S - NQ)
p = prof
nd2 = idx.nd2
bs = p.block_size
nblk = index["nblk"]
print(f"\nphase 分解（末 chunk，nq={NQ}, S={S}）：")

def t_ms(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    r = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3, r

# P0: kq_f 反量化表（每 chunk 重建——M7 已知疑点）
ms, kq_f = t_ms(lambda: kq_unpack(index["kq_q"][:S], index["kq_sc"][:S], index["kq_mn"][:S]))
print(f"  P0 kq_f 反量化表 [S,Hkv,nd2]     : {ms:7.1f} ms")

# 单 chunk 内部各 phase（row_chunk=64，末 chunk 的最后一个 64 行块）
q_c = q[-64:]
t_c = t_arr[-64:]
n = 64
qs = q_c[..., idx.idx1]
qg = qs.clamp(min=0).reshape(n, Hkv, G, p.coarse_dim)
qn = qs.clamp(max=0).reshape(n, Hkv, G, p.coarse_dim)
kmin, kmax = index["kmin"][:nblk], index["kmax"][:nblk]
ms, _ = t_ms(lambda: torch.einsum("ahgd,mhd->ahm", qg, kmax) + torch.einsum("ahgd,mhd->ahm", qn, kmin))
print(f"  P1 L1 einsum [n,nblk]（单 chunk）: {ms:7.1f} ms × {NQ//64} chunks")
q2 = idx._q_refine(q_c).reshape(n, Hkv, G, nd2).sum(2)
ms, fine = t_ms(lambda: torch.einsum("ahd,shd->ahs", q2, kq_f))
print(f"  P2 快路径全宽 einsum [n,Hkv,S]   : {ms:7.1f} ms × {NQ//64} chunks")
pos = torch.arange(S, device=dev)
causal_full = pos.view(1, S) <= t_c.view(-1, 1)
ms, _ = t_ms(lambda: fine.masked_fill(~causal_full.unsqueeze(1), float("-inf")))
print(f"  P3 causal masked_fill [n,Hkv,S]  : {ms:7.1f} ms × {NQ//64} chunks")
in_far = (pos.view(1, S) >= p.sink_blocks * bs) & (pos.view(1, S) < (t_c + 1 - p.near_len).clamp(min=p.sink_blocks * bs).view(-1, 1))
fine2 = fine.clone()
ms, _ = t_ms(lambda: fine2.masked_fill(~in_far.unsqueeze(1), float("-inf")))
print(f"  P4 far masked_fill [n,Hkv,S]     : {ms:7.1f} ms × {NQ//64} chunks")
ms, _ = t_ms(lambda: torch.topk(fine2, p.far_tokens, dim=-1))
print(f"  P5 far topk k={p.far_tokens} [n,Hkv,S]: {ms:7.1f} ms × {NQ//64} chunks")

# 快路径判定本身（.all() 触发同步）
onehot_probe = torch.rand(64, nblk, device=dev) < 0.9  # 形状同 sel_mask 展开前
sel_mask_probe = onehot_probe.repeat_interleave(bs, dim=1)[:, :S]
ms, _ = t_ms(lambda: bool((sel_mask_probe.sum(1) >= t_c + 1).all()))
print(f"  P6 快路径判定 .all()（含同步）  : {ms:7.1f} ms × {NQ//64} chunks")
