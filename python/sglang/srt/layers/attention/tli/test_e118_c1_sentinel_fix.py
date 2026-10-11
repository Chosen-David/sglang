# E118 C1 修复验证（GPT 审计 sglang_twolevel_audit C1/P1，基线 ee256310b）
# 红绿口径：
#   红（修复前）：eager per-request select() 的 far/near 分区 topk 与
#     非分区兜底 topk 均不看选中分数——池内有限候选 < 配额时，-inf
#     槽位以 [0,S) 真实位置返回，消费端 valid = sel < S 无法屏蔽。
#   绿（修复后）：-inf 槽位 → 哨兵 S（与 use_l2_kernel 哨兵协议同款）。
# 运行方式：
#   绿：python test_e118_c1_sentinel_fix.py（本测试按自身位置推导
#     sglang/python 根，即随所在树加载）
#   红：E118_SGLANG_ROOT=/home/wangyuanshuo02/sglang/python \
#       python test_e118_c1_sentinel_fix.py（指向修复前基点树）
import os
import sys

_ROOT = os.environ.get("E118_SGLANG_ROOT")
if _ROOT:
    sys.path.insert(0, _ROOT)
else:
    sys.path.insert(
        0,
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
        ),
    )

import torch

from sglang.srt.layers.attention.tli.config import TLIProfile
from sglang.srt.layers.attention.tli.indexer import TLIIndexer

dev = "cuda:0"
torch.manual_seed(20261011)
Hkv, D, H = 1, 128, 1
S, T = 1024, 1023
BS = 64


def make_profile(near_len: int) -> TLIProfile:
    """C1 反例口径：k1=2 块 → 候选远小于 L2 配额，制造池饥饿。"""
    p = TLIProfile()
    p.block_size = BS
    p.sink_blocks = 1          # far_tok_lo = 64
    p.k1_blocks = 2
    p.sliding_blocks = 1       # force_blks = {最后一个块}
    p.sliding_window = 32
    p.token_budget = 256
    p.far_tokens = 64
    p.near_len = near_len
    return p


def build_cand(idxer: TLIIndexer, index: dict, q: torch.Tensor, t: int) -> set:
    """独立复算 L1 候选 token 集（与 select() L1 同口径，供断言）。"""
    p = idxer.profile
    nblk = index["nblk"]
    kmin, kmax = index["kmin"][:nblk], index["kmax"][:nblk]
    qs = q[..., idxer.idx1]                       # [1, H, d']
    qg = qs.clamp(min=0).reshape(1, Hkv, H // Hkv, p.coarse_dim)
    qn = qs.clamp(max=0).reshape(1, Hkv, H // Hkv, p.coarse_dim)
    sc1 = (
        torch.einsum("bhgd,nhd->bhgn", qg, kmax)
        + torch.einsum("bhgd,nhd->bhgn", qn, kmin)
    ).sum(-2)[0]                                  # [Hkv, nblk]
    blk_end = (torch.arange(nblk, device=dev) + 1) * p.block_size - 1
    sc1 = sc1.masked_fill(blk_end.view(1, -1) > t, float("-inf"))
    cand_blk = torch.topk(sc1, min(p.k1_blocks, nblk), dim=-1).indices[0]
    last_blk = t // p.block_size
    force = torch.arange(max(0, last_blk - p.sliding_blocks + 1), last_blk + 1, device=dev)
    onehot = torch.zeros(nblk, dtype=torch.bool, device=dev)
    onehot[cand_blk.tolist()] = True
    onehot[force.tolist()] = True
    toks = onehot.repeat_interleave(p.block_size)
    return set((toks.nonzero().squeeze(1)[:S]).tolist())


def craft_case(near_len: int, boost_blocks: list[int]):
    """构造 L1 top-K1 恰为 boost_blocks 的确定性反例。"""
    p = make_profile(near_len)
    idxer = TLIIndexer(p, head_dim=D).to(dev)
    boost = torch.zeros(D, device=dev)
    boost[idxer.idx1.tolist()] = 1.0
    k = torch.randn(S, Hkv, D, device=dev) * 0.05
    for b in boost_blocks:
        k[b * BS : (b + 1) * BS, 0] += boost * 3.0
    q = torch.randn(1, H, D, device=dev) * 0.1
    q[..., idxer.idx1.tolist()] = q[..., idxer.idx1.tolist()].abs() + 1.0
    index = idxer.build_block_index(k)
    return idxer, index, q, build_cand(idxer, index, q, T)


def check(name, idxer, index, q, cand):
    sel = idxer.select(index, q, T)                    # [Hkv, K2]
    assert sel.shape[0] == Hkv
    lanes = sel[0].tolist()
    valid = [x for x in lanes if x < S]                # 消费端 valid 口径
    bad = [x for x in valid if x not in cand]
    assert not bad, (
        f"[{name}] {len(bad)} 个非哨兵槽位不在 L1 候选集内（前 8: {bad[:8]}）"
        f"——-inf 槽位以真实位置泄漏进注意力"
    )
    dup = len(valid) - len(set(valid))
    assert dup == 0, f"[{name}] {dup} 个重复槽位（同一 token 重复计权）"
    n_sent = sum(x == S for x in lanes)
    print(f"[{name}] 有效 lane {len(valid)}（全部 ∈ L1 候选集）+ 哨兵 {n_sent} — PASS")
    return lanes


# ---- 场景 1：分区分支 near 饥饿（审计反例形态：配额 192 vs 有限 64）----
# far 区 [64,896)，L1 top-2 = 块 5/6（远端）→ near 区候选仅强制块 15
# 的 64 token < k2_near=192 → ≥128 个 -inf 槽位。
idxer1, index1, q1, cand1 = craft_case(near_len=128, boost_blocks=[5, 6])
check("C1-A 分支 near 饥饿", idxer1, index1, q1, cand1)

# ---- 场景 2：非分区兜底分支饥饿（far_tok_hi == far_tok_lo → L667 兜底）----
# near_len=4096 > S → 远端区空，走整行 topk(budget=256)；候选 2 块 +
# 强制块 = 192 token < 256 → 64 个 -inf 槽位。
idxer2, index2, q2, cand2 = craft_case(near_len=4096, boost_blocks=[5, 6])
check("C1-B 兜底分支饥饿", idxer2, index2, q2, cand2)

# ---- 场景 3：far 池饥饿（L1 top-2 全在 near 区 → far 配额全 -inf）----
# far 配额 64 vs far 区候选 0 → 修复前 64 个垃圾位置 ∈ [64,896)。
idxer3, index3, q3, cand3 = craft_case(near_len=128, boost_blocks=[14, 15])
check("C1-C far 池饥饿", idxer3, index3, q3, cand3)

print("E118 C1 全部场景 PASS：-inf 槽位一律哨兵 S，非法 token 不再进注意力")
