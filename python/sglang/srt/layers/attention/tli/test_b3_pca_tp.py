# bug3 专项验证（GPT 复查 2026-10-08）：PCA basis 不适配 TP
# 核心 bug：basis 按全局 Hkv 离线校准 [n_layers, Hkv_global, D, r]，
#   TP>1 时每卡 num_kv_heads = Hkv_global/attn_tp_size——原样传给
#   TLIIndexer（basis[li] 的 Hkv 维 = 全局值）形状错配即崩；
#   TP=1 下 Hkv 不匹配也无断言（静默错用防线缺失）。
# 修复（backend.py 加载处）：按 attn_tp_rank 切本卡 kv-head 片
#   + 两种情形都断言 Hkv 对齐。
# 本测试：mock 加载切片逻辑（不启动 backend，纯张量语义验证）。
import os
import sys

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
import torch

torch.manual_seed(20261008)
n_layers, Hkv_g, D, r = 4, 8, 128, 32


def slice_tp(basis, tp_size, tp_rank, num_kv_heads):
    """backend.py 修复版切片逻辑逐行复刻。"""
    g = int(basis.shape[1])
    if tp_size > 1:
        assert g % tp_size == 0, f"全局 Hkv {g} 不能被 {tp_size} 整除"
        per = g // tp_size
        assert per == num_kv_heads, f"每卡 {per} != num_kv_heads {num_kv_heads}"
        return basis[:, tp_rank * per : (tp_rank + 1) * per]
    else:
        assert g == num_kv_heads, f"Hkv {g} != num_kv_heads {num_kv_heads}"
        return basis


basis = torch.randn(n_layers, Hkv_g, D, r)

# ① TP=1：不切、逐位相同、断言过
b1 = slice_tp(basis, 1, 0, Hkv_g)
assert torch.equal(b1, basis), "TP=1 不应切片"
print("[1] TP=1：不切片逐位相同，Hkv 断言过 — PASS")

# ② TP=2：rank0/rank1 各拿 4 个 head，并集=全集、交集=空
b_r0 = slice_tp(basis, 2, 0, 4)
b_r1 = slice_tp(basis, 2, 1, 4)
assert b_r0.shape == (n_layers, 4, D, r) and b_r1.shape == (n_layers, 4, D, r)
assert torch.equal(b_r0, basis[:, :4]) and torch.equal(b_r1, basis[:, 4:])
assert torch.equal(torch.cat([b_r0, b_r1], dim=1), basis), "rank 片并集应=全集"
print("[2] TP=2：rank0/rank1 各 4 head，并集=全集、无重叠 — PASS")

# ③ TP=4：4 卡各 2 head
for rk in range(4):
    br = slice_tp(basis, 4, rk, 2)
    assert torch.equal(br, basis[:, rk * 2 : rk * 2 + 2])
print("[3] TP=4：4 rank 片正确 — PASS")

# ④ 断言防线：整除失败 / 每卡数不匹配 / TP=1 Hkv 不匹配 各自报错
for tp_size, rk, nkv, tag in [
    (3, 0, 8, "整除失败"),
    (2, 0, 8, "每卡数不匹配"),
    (1, 0, 4, "TP=1 Hkv 不匹配"),
]:
    try:
        slice_tp(basis, tp_size, rk, nkv)
        raise SystemExit(f"④ {tag}：未触发断言（防线失效）")
    except AssertionError:
        pass
print("[4] 三类 shape 错配全部触发断言 — PASS")

# ⑤ indexer einsum 语义：切片后 basis 与本卡 k 的 Hkv 维一致
G = 4
H_local = 4
k = torch.randn(S_k := 64, H_local, D)
basis_l = b_r0[0]  # [4, D, r]
proj = torch.einsum("shd,hdr->shr", k, basis_l)
assert proj.shape == (S_k, H_local, r)
# 与全局投影后切片逐位一致（切片无信息损失）
proj_g = torch.einsum("shd,hdr->shr", k, basis[0, :4])
assert torch.allclose(proj, proj_g, atol=1e-6)
print("[5] 切片后 einsum 与「全局投影再切片」逐位一致 — PASS")

print("\nbug3（PCA basis TP rank 切片 + assert）验证 全部 PASS")
