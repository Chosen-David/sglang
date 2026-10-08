# GPT 复查 bug1 + bug2 专项验证（2026-10-08）
# bug1（kernels.tli_l2_partition_topk）：far/near 池不足时 topk 的 -inf
#   槽位有两类——pad lane（cand_pad[S] 本就哨兵）与**池内真实候选位**
#   （far_scr 对 near/swa 候选也写 -inf）。后者 gather 出真实 token
#   位置 → 下游 valid = sel < S 判有效（错误池 token 混入，S3 同类）。
#   修复 = keep 掩码统一转哨兵 S。
#   验证：far 候选数 < k2_far 时，far 段有效槽位 ⊆ far 候选集、
#   其余全为哨兵 S（修复前 near/swa 真实位置混入）。
# bug2（q_agg=max decode 未生效）：select / _select_taskmd /
#   select_decode_batched（内部路由 _select_decode_taskmd）/
#   select_batched（taskmd prefill 批量）全部写死 group-sum。
#   修复 = 全位点补 max 分支（逐 q-head 打分取组内 max）+ fused
#   kernel 旁路（kernel 内 group-sum）。
#   验证（两不变量）：
#   ① G=1：max ≡ sum 逐位相同（单元素 max = 本身——接线正确性）
#   ② G=2 符号冲突：组内 q 为 +w/−w，sum 互相抵消而 max 保留 +w
#     → max 与 sum 输出必须不同（修复前 max 写死 sum → 相同 → FAIL）
import os

os.environ.setdefault("SGLANG_TLI_ALPHA", "0.125")
os.environ.setdefault("SGLANG_TLI_BETA", "0.375")
os.environ.setdefault("SGLANG_TLI_GAMMA", "0.625")
import sys

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
import torch

from sglang.srt.layers.attention.tli.config import TLIProfile
from sglang.srt.layers.attention.tli.indexer import TLIIndexer

dev = "cuda:0"
torch.manual_seed(20261008)
D, Hkv, S = 128, 8, 2048

# ============ bug1：tli_l2_partition_topk 池不足错误池 token ============
from sglang.srt.layers.attention.tli.kernels import tli_l2_partition_topk

nd2, G = 32, 4
H = Hkv * G
q = torch.randn(H, nd2, device=dev)
kq = torch.randn(S, Hkv, nd2, device=dev) * 0.1
far_lo, swa_tok = 256, 128
t = S - 1
far_hi = t + 1 - 512  # far 区 [256, 1536)
sw_lo = t - swa_tok + 1
# far 候选仅 10 个 token；near/swa 候选各若干（它们在 far_scr 是
# -inf 的**池内真实位**——bug1 场景）
far_cand = torch.randint(far_lo, far_hi - 8, (10,), device=dev)
near_cand = torch.randint(far_hi, sw_lo, (200,), device=dev)
swa_cand = torch.arange(sw_lo, t + 1, device=dev)
cand_pos = torch.cat([far_cand, near_cand, swa_cand]).sort().values
k2_far, k2_near = 64, 320
out = tli_l2_partition_topk(
    q, kq, cand_pos, S, k2_far, k2_near, far_lo, far_hi, sw_lo, t
)  # [Hkv, K2]
far_seg = out[:, :k2_far]
far_set = set(far_cand.tolist())
for h in range(Hkv):
    valid_h = far_seg[h][far_seg[h] < S]
    bad = [p for p in valid_h.tolist() if p not in far_set]
    assert not bad, f"head {h}：far 段混入错误池真实 token {bad[:8]}（bug1）"
    n_valid = valid_h.numel()
    assert n_valid <= len(far_set), f"head {h}：far 有效数 {n_valid} > 候选 {len(far_set)}"
    n_sent = int((far_seg[h] == S).sum().item())
    assert n_valid + n_sent == k2_far, f"head {h}：有效 {n_valid}+哨兵 {n_sent} ≠ {k2_far}"
print("[1] bug1：far 池不足槽位全部转哨兵 S，无错误池真实 token — PASS")

# 近段同款验证（near 候选远多于配额，此处验证哨兵协议零例外）
near_seg = out[:, k2_far : k2_far + max(0, k2_near - swa_tok)]
near_set = set(near_cand.tolist())
for h in range(Hkv):
    valid_h = near_seg[h][near_seg[h] < S]
    bad = [p for p in valid_h.tolist() if p not in near_set]
    assert not bad, f"head {h}：near 段混入池外 token {bad[:8]}"
print("[2] bug1：near 段有效槽位 ⊆ near 候选集 — PASS")


# ============ bug2：q_agg=max 五路径生效性 ============
def make_idxer(q_agg: str) -> TLIIndexer:
    prof = TLIProfile()
    prof.q_agg = q_agg
    return TLIIndexer(prof, head_dim=D).to(dev)


def conflicted_q(shape_prefix, w):
    """组内 +w/−w 符号冲突 q：sum 抵消、max 保留 +w。"""
    n, H_, D_ = *shape_prefix, None
    q = torch.zeros(*shape_prefix, device=dev)
    # refine 子空间（idx2，32 维）组内 head0 = +w、head1 = −w
    full = torch.zeros(*shape_prefix, D, device=dev)
    G_ = H_ // Hkv
    for h in range(Hkv):
        full[..., h * G_ + 0, idxer_refine_idx] = w
        if G_ > 1:
            full[..., h * G_ + 1, idxer_refine_idx] = -w
    return full


idxer_refine_idx = None  # 由首个 idxer 的 idx2 填充
idx_sum, idx_max = make_idxer("sum"), make_idxer("max")
idxer_refine_idx = idx_sum.idx2

k_real = torch.randn(S, Hkv, D, device=dev) * 0.1
index = idx_sum.build_block_index(k_real)


def assert_diff(out_s, out_m, tag):
    assert not torch.equal(out_s, out_m), (
        f"{tag}：q_agg=max 输出与 sum 完全相同（max 未生效——bug2 症状）"
    )
    print(f"  {tag}：max 输出 ≠ sum 输出（生效）")


# ---- B1：select（per-request，非 taskmd 由 env 决定——本测试 env 是
# taskmd 激活态，故 B1 实测走 _select_taskmd；B1' 用 batched 路径）----
w = torch.randn(nd2, device=dev) * 0.5
t = S - 1
q1_sum = conflicted_q((1, H), w)
out_s = idx_sum.select(index, q1_sum, t)
out_m = idx_max.select(index, q1_sum, t)
assert_diff(out_s, out_m, "B1 _select_taskmd（per-request）")

# G=1 逐位等价（max ≡ sum，接线正确性不变量）
H1 = Hkv
q_g1 = torch.randn(1, H1, D, device=dev) * 0.1
prof1 = TLIProfile()
prof1.q_agg = "sum"
i1s = TLIIndexer(prof1, head_dim=D).to(dev)
prof1b = TLIProfile()
prof1b.q_agg = "max"
i1m = TLIIndexer(prof1b, head_dim=D).to(dev)
o_s = i1s.select(index, q_g1, t)
o_m = i1m.select(index, q_g1, t)
assert torch.equal(o_s, o_m), "G=1：max 与 sum 应逐位相同（接线错误）"
print("  G=1：max ≡ sum 逐位相同 — PASS")

# ---- B2：select_batched（taskmd prefill 批量，_select_batched_taskmd）----
nq = 4
t_arr = torch.tensor([1023, 2047, 3071, S - 1], device=dev)
qb = conflicted_q((nq, H), w)
ob_s = idx_sum.select_batched(index, qb, t_arr)
ob_m = idx_max.select_batched(index, qb, t_arr)
assert_diff(ob_s, ob_m, "B2 _select_batched_taskmd（prefill 批量）")

# ---- B3/B4：select_decode_batched（非 taskmd pool 路径 + taskmd 路径）----
# pool 构造（backend._get_pool 同构）
R, S_cap, NBLK_CAP, bs = 4, 2048, 32, 64
pool_l = {
    "kq_q": torch.randint(0, 200, (R, S_cap, Hkv, idx_sum.nd2), dtype=torch.uint8, device=dev),
    "kq_sc": torch.ones(R, S_cap, Hkv, device=dev),
    "kq_mn": torch.zeros(R, S_cap, Hkv, device=dev),
    "kmin": torch.rand(R, NBLK_CAP, Hkv, idx_sum.profile.coarse_dim, device=dev) * 0.2,
    "kmax": torch.rand(R, NBLK_CAP, Hkv, idx_sum.profile.coarse_dim, device=dev) * 0.2,
}
rows = torch.tensor([1, 2], device=dev)
S_list = [1024, 2048]
n = len(S_list)
qd = conflicted_q((n, H), w)
od_s = idx_sum.select_decode_batched(pool_l, rows, S_list, qd)
od_m = idx_max.select_decode_batched(pool_l, rows, S_list, qd)
assert_diff(od_s, od_m, "B3/B4 select_decode_batched（decode 批量，taskmd 路由）")

# G=1 decode 批量逐位等价
qd1 = torch.randn(n, H1, D, device=dev) * 0.1
od1_s = i1s.select_decode_batched(pool_l, rows, S_list, qd1)
od1_m = i1m.select_decode_batched(pool_l, rows, S_list, qd1)
assert torch.equal(od1_s, od1_m), "G=1 decode：max 与 sum 应逐位相同"
print("  G=1 decode 批量：max ≡ sum 逐位相同 — PASS")

print("\nbug1 + bug2（fused L2 错误池 token + q_agg=max decode 生效）全部 PASS")
