# M8-2b 收尾：dual kernel num_warps 干净 A/B（真实稀疏 tok_c 形态）
# 环境漂移下 on/off 两边同涨，须同进程交替测。真实 tok_c 来自 L1 topk+compact 链。
import os
import sys
import time

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
import json

import torch
import triton

for k in ["L2B", "COMPACT", "L1B", "L2D", "NEAR_COMPACT"]:
    os.environ.setdefault(f"SGLANG_TLI_{k}_KERNEL", "1") if k != "NEAR_COMPACT" else None
os.environ["SGLANG_TLI_NEAR_COMPACT"] = "1"

from sglang.srt.layers.attention.tli.config import TLIProfile
from sglang.srt.layers.attention.tli.indexer import TLIIndexer
from sglang.srt.layers.attention.tli.kernels import (
    tli_compact,
    tli_l1_score_batched,
    _tli_l2_score_batched_dual_kernel,
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
rows_l = torch.arange(n, device=dev, dtype=torch.long)
q = torch.randn(n, Hkv * 4, D, device=dev) * 0.3

# ---- 真实 tok_c（L1 topk + compact 链产物，稀疏块形态）----
S_t = torch.full((n,), S_cap, device=dev, dtype=torch.long)
t_t = S_t - 1
nblk_t = (S_t + bs - 1) // bs
sc1 = tli_l1_score_batched(q, idxer.idx1, pool["kmin"], pool["kmax"], rows_l, nblk_t, t_t, bs)
K1 = min(prof.k1_blocks, NBLK_CAP)
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
Tc = min((K1 * Hkv + prof.sliding_blocks) * bs, NBLK_CAP * bs, S_cap)
tok_c = tli_compact(onehot, S_t, bs, S_cap, Tc).clamp(max=S_cap - 1)

q2 = idxer._q_refine(q).reshape(n, Hkv, 4, nd2).sum(2)
far_lo = prof.sink_blocks * bs
far_hi_t = (t_t + 1 - prof.near_len).clamp(min=far_lo)
sw_lo_t = (t_t - prof.sliding_window + 1).clamp(min=0)
ps = (tok_c < far_lo).sum(dim=-1)
pf = (tok_c < far_hi_t.view(-1, 1)).sum(dim=-1)
ps_i = ps.to(torch.int32).contiguous()
pf_i = pf.to(torch.int32).contiguous()
WNCAP = far_lo + max(0, prof.near_len - prof.sliding_window)


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


def dual(nw):
    far_sc = torch.empty(n, Hkv, Tc, dtype=torch.float32, device=dev)
    near_sc = torch.full((n, Hkv, WNCAP), float("-inf"), dtype=torch.float32, device=dev)
    near_tok = torch.full((n, WNCAP), S_cap, dtype=torch.int64, device=dev)
    grid = (n, triton.cdiv(Tc, 128))
    _tli_l2_score_batched_dual_kernel[grid](
        q2, pool["kq_q"], pool["kq_sc"], pool["kq_mn"], rows_l, tok_c,
        far_sc, near_sc, near_tok,
        far_hi_t, sw_lo_t, ps_i, pf_i,
        HKV=Hkv, ND2=nd2, TC=Tc, S_CAP=S_cap, FAR_LO=far_lo,
        CHUNK=128, NEARC=True, WNCAP=WNCAP, num_warps=nw,
    )
    return far_sc, near_sc, near_tok


# 对拍两种 nw 逐位一致
fa, na, ta = dual(4)
fb, nb, tb = dual(8)
eq = torch.equal(fa, fb) and torch.equal(na, nb) and torch.equal(ta, tb)
print(f"nw=4 vs nw=8 输出逐位一致: {'✓' if eq else '✗'}  (Tc={Tc}, 稀疏块形态)")

# 交替测（抗环境漂移）+ CHUNK 交叉 sweep（真实稀疏形态）
res = {}
for tag, nw in [("nw=8", 8), ("nw=4", 4), ("nw=8", 8), ("nw=4", 4), ("nw=2", 2), ("nw=4", 4)]:
    t = bench(lambda: dual(nw))
    res[tag] = min(res.get(tag, 1e9), t)
    print(f"dual NEARC=1 {tag}: {t:6.3f} ms")

def dual2(nw, chunk):
    far_sc = torch.empty(n, Hkv, Tc, dtype=torch.float32, device=dev)
    near_sc = torch.full((n, Hkv, WNCAP), float("-inf"), dtype=torch.float32, device=dev)
    near_tok = torch.full((n, WNCAP), S_cap, dtype=torch.int64, device=dev)
    grid = (n, triton.cdiv(Tc, chunk))
    _tli_l2_score_batched_dual_kernel[grid](
        q2, pool["kq_q"], pool["kq_sc"], pool["kq_mn"], rows_l, tok_c,
        far_sc, near_sc, near_tok,
        far_hi_t, sw_lo_t, ps_i, pf_i,
        HKV=Hkv, ND2=nd2, TC=Tc, S_CAP=S_cap, FAR_LO=far_lo,
        CHUNK=chunk, NEARC=True, WNCAP=WNCAP, num_warps=nw,
    )
    return far_sc, near_sc, near_tok

print("\nCHUNK×nw sweep（真实稀疏形态）：")
f0, n0, t0k = dual2(8, 128)
sweep = {}
for chunk in (64, 128, 256, 512):
    for nw in (4, 8, 16):
        try:
            ff, nn, tt = dual2(nw, chunk)
            ok = torch.equal(ff, f0) and torch.equal(nn, n0) and torch.equal(tt, t0k)
            t = bench(lambda: dual2(nw, chunk))
            sweep[f"c{chunk}_w{nw}"] = round(t, 4)
            print(f"  CHUNK={chunk:3d} nw={nw:2d}: {t:6.3f} ms  equal={'✓' if ok else '✗'}")
        except Exception as e:
            print(f"  CHUNK={chunk:3d} nw={nw:2d}: {type(e).__name__}: {str(e)[:120]}")

print("\n稳态最优：", {k: round(v, 4) for k, v in res.items()})
json.dump({"ab": {k: round(v, 4) for k, v in res.items()}, "sweep": sweep},
          open("/home/wangyuanshuo02/sglang/tli_m8_nw_ab.json", "w"), indent=1)
