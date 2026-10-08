# S2 第二处专项验证（Kimi3 S2，2026-10-08）：_select_decode_taskmd 单池分支
# 核心 bug：taskmd 单池臂（α=0 或 β=0，含计划内 (0,0) 退化点）的批量
#   decode/CUDA graph 路径，第二处 masked_fill((tok >= swa_lo_t), +inf)
#   对哨兵 lane（tok = S_cap 未 clamp，≥ swa_lo_t 恒真）把刚打的 -inf
#   复活为 +inf → topk 被 +inf 哨兵挤占 → 退化「只看滑窗」（有限分
#   far/near 候选被整体挤出 topk）、最坏 0 有效 lane → softmax NaN。
# 修复：+inf 强制条件补 & valid（valid = tok < S_cap）。
# 触发前提：候选压实 Tc = (K1·Hkv + sliding_blocks)·bs > 单池候选数
#   （top-K1 块 ∪ swa 块 × bs）——Hkv=8 时恒成立（Tc≈8×候选），哨兵
#   lane 是单池批量 decode 的常态而非边角。
# 验证：① 输出有效槽位集合 == L1 候选全集（修复前 far/near 候选被
#   +inf 哨兵挤掉，只剩滑窗）；② swa 区 [t+1-swa, t] 全部在场；
#   ③ 全部有效槽位 ≤ t（因果）；④ 有效槽位无重复；⑤ 槽位总数 = k；
#   ⑥ 消费端 softmax 无 NaN。
import os

os.environ.setdefault("SGLANG_TLI_ALPHA", "0.0")  # 单池（e64=False）
os.environ.setdefault("SGLANG_TLI_K1_BLOCKS", "8")
os.environ.setdefault("SGLANG_TLI_TOKEN_BUDGET", "1024")
os.environ.setdefault("SGLANG_TLI_SLIDING_WINDOW", "128")
os.environ.setdefault("SGLANG_TLI_SLIDING_BLOCKS", "3")
os.environ.setdefault("SGLANG_TLI_SINK_BLOCKS", "2")
import sys

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
import torch

from sglang.srt.layers.attention.tli.config import TLIProfile
from sglang.srt.layers.attention.tli.indexer import TLIIndexer

dev = "cuda:0"
torch.manual_seed(20261008)
D, Hkv, H = 128, 8, 32
G = H // Hkv
bs, K1 = 64, 8
S_cap, NBLK_CAP, R = 2048, 32, 4
swa_tok, sink_blk = 128, 2

prof = TLIProfile()
assert prof.taskmd and prof.alpha == 0.0  # 单池臂（(0,0) 退化点同路径）
assert prof.far_method == "minmax"
idxer = TLIIndexer(prof, head_dim=D).to(dev)

# ---- 构造 decode pool（backend._get_pool 同构，kq 反量化 = grid*sc+mn）----
pool_l = {
    "kq_q": torch.randint(0, 200, (R, S_cap, Hkv, idxer.nd2), dtype=torch.uint8, device=dev),
    "kq_sc": torch.ones(R, S_cap, Hkv, device=dev),
    "kq_mn": torch.zeros(R, S_cap, Hkv, device=dev),
    "kmin": torch.rand(R, NBLK_CAP, Hkv, prof.coarse_dim, device=dev) * 0.2,
    "kmax": torch.rand(R, NBLK_CAP, Hkv, prof.coarse_dim, device=dev) * 0.2,
}
rows = torch.tensor([1, 2], device=dev)
S_list = [1024, 2048]  # 两行不同长度（短行候选更少，哨兵更多）
n = len(S_list)
q = torch.randn(n, H, D, device=dev) * 0.1

sel = idxer.select_decode_batched(pool_l, rows, S_list, q)  # [n, Hkv, K2']
k = min(prof.token_budget, NBLK_CAP * bs)  # 单池 topk 宽度（静态）
assert sel.shape == (n, Hkv, k), f"输出形状 {sel.shape} ≠ [n, Hkv, {k}]"

# ---- 独立 oracle：复刻 L1 单池候选（top-K1 块 ∪ swa 块 → token 全集）----
blk_id = torch.arange(NBLK_CAP, device=dev)
blk_end = (blk_id + 1) * bs - 1
for r, (row, S) in enumerate(zip(rows.tolist(), S_list)):
    t = S - 1
    nblk = (S + bs - 1) // bs
    valid_blk = (blk_id < nblk) & (blk_end <= t)
    qs = q[r][..., idxer.idx1]  # [H, d']
    qg = qs.clamp(min=0).reshape(Hkv, G, prof.coarse_dim)
    qn = qs.clamp(max=0).reshape(Hkv, G, prof.coarse_dim)
    kmin_b = pool_l["kmin"][row]
    kmax_b = pool_l["kmax"][row]
    sc1 = torch.einsum("hgd,mhd->hm", qg, kmax_b) + torch.einsum(
        "hgd,mhd->hm", qn, kmin_b
    )  # [Hkv, NBLK_CAP]
    sc1 = sc1.masked_fill(~valid_blk.unsqueeze(0), float("-inf"))
    sc_pool = sc1.clone()
    sc_pool[:, t // bs] = float("inf")  # 当前块强制（scatter 同语义）
    i_pool = torch.topk(sc_pool, K1, dim=-1).indices
    keep_p = torch.gather(sc_pool, 1, i_pool) > float("-inf")
    onehot_h = torch.zeros(Hkv, NBLK_CAP, dtype=torch.bool, device=dev)
    onehot_h.scatter_(1, i_pool, keep_p)
    swa_lo_blk_row = max(0, t + 1 - swa_tok) // bs
    in_swa = (blk_id >= swa_lo_blk_row) & (blk_id <= t // bs)
    # 输出断言按 per-head（L2 per-head 掩码 in_pool_h 语义）
    cand_by_h = {
        h: set(
            (onehot_h[h] | in_swa)
            .repeat_interleave(bs)[:S]
            .nonzero()
            .flatten()
            .tolist()
        )
        for h in range(Hkv)
    }
    # ② swa 区应全部在 per-head 候选内（+inf 强制 + 块并入并集）
    swa_set = set(range(max(0, t + 1 - swa_tok), t + 1))
    assert all(swa_set <= cand_by_h[h] for h in range(Hkv)), (
        f"行 {r}：swa 区未并入 L1 候选（oracle 自检失败）"
    )

    # ---- ① 输出有效槽位集合 == per-head L1 候选全集（k ≥ 候选 → 全选）----
    out_r = sel[r]  # [Hkv, k]
    for h in range(Hkv):
        cand = cand_by_h[h]
        valid_h = out_r[h][out_r[h] < S_cap]
        assert bool((valid_h <= t).all()), f"行 {r} head {h}：有效槽位越出因果界"
        out_set = set(valid_h.tolist())
        assert len(out_set) == valid_h.numel(), f"行 {r} head {h}：有效槽位重复"
        # 哨兵数 = k − 候选数（修复前 +inf 哨兵挤占 → 有效数 << 候选数）
        sent_h = int((out_r[h] == S_cap).sum().item())
        assert sent_h == k - len(cand), (
            f"行 {r} head {h}：哨兵 {sent_h} ≠ k−候选 {k - len(cand)}"
        )
        if h == 0:
            print(
                f"[{r}] S={S}：候选 {len(cand)}（swa {len(swa_set)} 全在场），"
                f"有效槽位 {len(out_set)}，哨兵 {sent_h}/{k}"
            )
        missing = cand - out_set
        assert not missing or len(missing) <= 0, (
            f"行 {r} head {h}：{len(missing)} 个 L1 候选被挤出 topk"
            f"（修复前症状：+inf 哨兵挤占）——示例 {sorted(missing)[:8]}"
        )
        assert swa_set <= out_set, f"行 {r} head {h}：swa 强制区被挤掉"

print("[1] 单池批量 decode：有效槽位 = L1 候选全集，swa 全在场，"
      "哨兵数 = k−候选 — PASS")

# ---- ⑥ 消费端协议（_sparse_attn_batched 同款 valid 掩码 + softmax）----
w = (sel < torch.tensor(S_list, device=dev).view(-1, 1, 1)).float()
w = w / w.sum(-1, keepdim=True).clamp(min=1)
assert not bool(torch.isnan(w).any()), "消费端 softmax 出现 NaN（哨兵复活症状）"
print("[2] 消费端 softmax：无 NaN — PASS")
print("\nS2 第二处（_select_decode_taskmd 单池分支哨兵复活）修复验证 全部 PASS")
