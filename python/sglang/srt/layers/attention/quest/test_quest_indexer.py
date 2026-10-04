"""Quest indexer/backend 张量级单测（不启 Engine，import 级冒烟的一部分）。

覆盖（对拍口径 = two-level-attention/sparse_attn/indexer/quest_indexer.py）：
  1. 上界恒等式：q·kmax + relu(-q)·(kmax-kmin) == sum_d max(q·kmin, q·kmax)
     （参考式逐维 max——Quest 论文口径的直接实现）
  2. build_page_index 尾块精确界（零 pad 不污染）
  3. decode 增量（update_pool_rows_decode 逐 token）== 全量重建（逐位）
  4. _update_row_full 多 token 增量（chunked prefill 边界）== 全量重建
  5. select_decode_batched 选择的页集合 == 慢速参考实现（逐页直接公式
     打分 + GQA group-mean + top-(k-1) + 当前页强制）
  6. 因果性：sel 全部 ≤ t；哨兵 lane = S_i
  7. select_extend：query 级因果（当前页内未来位置被掩）

跑法：CUDA_VISIBLE_DEVICES=x python3 -m pytest ... 或直接 python3 运行。
"""

import math

import torch

from sglang.srt.layers.attention.quest.config import QuestProfile
from sglang.srt.layers.attention.quest.indexer import QuestIndexer


def _make(S, Hkv=2, H=8, D=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(S, Hkv, D, generator=g).cuda() if torch.cuda.is_available() \
        else torch.randn(S, Hkv, D, generator=g)
    q = torch.randn(1, H, D, generator=g)
    q = q.cuda() if torch.cuda.is_available() else q
    return k, q[0]


def test_upper_bound_identity():
    """恒等式 vs 参考式（quest_indexer.py 的 stack+amax+sum 直接实现）。"""
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    q = torch.randn(4, 6, 8, 16, device=dev) * 2  # [a, h, g, d]
    kmin = torch.randn(10, 6, 16, device=dev)
    kmax = kmin + torch.rand(10, 6, 16, device=dev)
    idx = QuestIndexer(QuestProfile(), head_dim=16)
    ub = idx._upper_bound(q, kmin, kmax, batched_rows=False)  # [4,6,8,10]
    # 参考式（quest_indexer.py 口径）：逐维 max 后求和——
    # b_score_min/max 保留 d 维（h d, tk h d -> h tk d），stack amax 再 sum
    p_min = torch.einsum("ahgd,mhd->ahgmd", q, kmin)
    p_max = torch.einsum("ahgd,mhd->ahgmd", q, kmax)
    ref = torch.maximum(p_min, p_max).sum(-1)
    torch.testing.assert_close(ub, ref, rtol=1e-4, atol=1e-3)
    print("[1] upper bound identity OK, max diff",
          (ub - ref).abs().max().item())


def _fake_pool(profile, S_cap, Hkv, D, R=4):
    pg = profile.page_size
    nblk_cap = (S_cap + pg - 1) // pg
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "kmin": torch.zeros(R, nblk_cap, Hkv, D, device=dev),
        "kmax": torch.zeros(R, nblk_cap, Hkv, D, device=dev),
        "S_cap": S_cap,
        "R_cap": R,
        "free": list(range(R)),
        "row_of": {},
        "S": [-1] * R,
    }


def test_incremental_equals_rebuild():
    """逐 token 增量 == 全量重建（含页边界 / 尾块精确界）。"""
    prof = QuestProfile()
    idx = QuestIndexer(prof, head_dim=16)
    S, Hkv, D = 1000, 2, 16
    k, _ = _make(S, Hkv, D=D, seed=1)
    pool = _fake_pool(prof, 2048, Hkv, D)
    full = idx.build_page_index(k)
    pool["kmin"][0, : full["nblk"]] = full["kmin"].float()
    pool["kmax"][0, : full["nblk"]] = full["kmax"].float()
    # 增量版：从 0 重建首步 + 逐步追加 1 token
    pool2 = _fake_pool(prof, 2048, Hkv, D)
    f0 = idx.build_page_index(k[:1])
    pool2["kmin"][0, : f0["nblk"]] = f0["kmin"].float()
    pool2["kmax"][0, : f0["nblk"]] = f0["kmax"].float()
    for s in range(1, S):
        # S_old = s：追加 token s 前 pool 已含 token 0..s-1 共 s 个
        idx.update_pool_rows_decode(
            pool2, torch.tensor([0], device=k.device), [s], k[s : s + 1]
        )
    nblk = full["nblk"]
    torch.testing.assert_close(
        pool2["kmin"][0, :nblk], pool["kmin"][0, :nblk], rtol=0, atol=0
    )
    torch.testing.assert_close(
        pool2["kmax"][0, :nblk], pool["kmax"][0, :nblk], rtol=0, atol=0
    )
    print("[3] incremental == rebuild OK (逐位)")


def test_update_row_full():
    """多 token 增量（chunk 边界）== 全量重建。"""
    prof = QuestProfile()
    idx = QuestIndexer(prof, head_dim=16)
    from sglang.srt.layers.attention.quest.backend import (
        QuestSparseAttnBackend,
    )

    S, Hkv, D = 2000, 2, 16
    k, _ = _make(S, Hkv, D=D, seed=2)
    full = idx.build_page_index(k)
    be = QuestSparseAttnBackend.__new__(QuestSparseAttnBackend)  # 不走 runner
    be.profile = prof
    pool = _fake_pool(prof, 2048, Hkv, D)
    # 分三段：0-1000（全量）、1000-1500、1500-2000（增量）
    f0 = idx.build_page_index(k[:1000])
    pool["kmin"][0, : f0["nblk"]] = f0["kmin"].float()
    pool["kmax"][0, : f0["nblk"]] = f0["kmax"].float()
    be._update_row_full(pool, idx, 0, 1000, 1500, k[1000:1500])
    be._update_row_full(pool, idx, 0, 1500, 2000, k[1500:2000])
    nblk = full["nblk"]
    torch.testing.assert_close(
        pool["kmin"][0, :nblk], full["kmin"].float(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        pool["kmax"][0, :nblk], full["kmax"].float(), rtol=0, atol=0
    )
    print("[4] _update_row_full == rebuild OK (逐位)")


def _ref_pages(k, q, t, pg, topk):
    """慢速参考：逐页直接公式上界 + GQA group-mean + top-(k-1) + 当前页。
    k: [S, Hkv, D] fp32；q: [H, D] fp32。返回 per-kv-head 的页集合 frozenset。"""
    S, Hkv, D = k.shape
    H = q.shape[0]
    G = H // Hkv
    nblk = (S + pg - 1) // pg
    cur = t // pg
    out = []
    for h in range(Hkv):
        sc = []
        for m in range(nblk):
            if m >= cur:
                sc.append(float("-inf"))
                continue
            ks = k[m * pg : (m + 1) * pg, h]  # [pg, D]
            kmin, kmax = ks.amin(0), ks.amax(0)
            ub = torch.stack((q @ kmin, q @ kmax), dim=-1).amax(-1).sum()
            sc.append(ub.item())
        order = sorted(range(nblk), key=lambda m: -sc[m])[: max(topk - 1, 1)]
        pages = set(m for m in order if sc[m] > float("-inf"))
        pages.add(cur)
        out.append(frozenset(pages))
    return out


def test_select_decode_batched():
    """批量选择的页集合 == 慢速参考；因果 + 哨兵语义。"""
    prof = QuestProfile()
    idx = QuestIndexer(prof, head_dim=16)
    S, Hkv, H, D = 1000, 2, 8, 16
    pg = prof.page_size
    k, q = _make(S, Hkv, H=H, D=D, seed=3)
    k32 = k.float()
    t = S - 1
    pool = _fake_pool(prof, 2048, Hkv, D)
    full = idx.build_page_index(k)
    pool["kmin"][0, : full["nblk"]] = full["kmin"].float()
    pool["kmax"][0, : full["nblk"]] = full["kmax"].float()
    sel = idx.select_decode_batched(
        pool, torch.tensor([0], device=k.device), [S], q.view(1, H, D)
    )  # [1, Hkv, K]
    ref = _ref_pages(k32, q.float(), t, pg, prof.topk_pages)
    K = prof.topk_pages * pg
    assert sel.shape == (1, Hkv, K), sel.shape
    for h in range(Hkv):
        pos = sel[0, h]
        valid = pos[pos < S]
        assert (valid <= t).all(), "因果违反"
        pages = frozenset((valid // pg).tolist())
        assert pages == ref[h], f"head {h}: {sorted(pages)} vs {sorted(ref[h])}"
    print("[5] select_decode_batched 页集合 == 参考 OK")


def test_select_extend_causal():
    """extend 选择的 query 级因果：当前页内 pos > t 被掩成哨兵。"""
    prof = QuestProfile()
    idx = QuestIndexer(prof, head_dim=16)
    S, Hkv, H, D = 1000, 2, 8, 16
    k, q = _make(S, Hkv, H=H, D=D, seed=4)
    full = idx.build_page_index(k)
    nq = 130
    t_arr = torch.arange(S - nq, S, device=k.device)
    sel = idx.select_extend(
        full["kmin"].float(), full["kmax"].float(), full["nblk"],
        q.view(1, H, D).expand(nq, H, D).contiguous().float(),
        t_arr, S,
    )
    K = prof.topk_pages * prof.page_size
    assert sel.shape == (nq, Hkv, K)
    for r in range(nq):
        t = int(t_arr[r])
        for h in range(Hkv):
            pos = sel[r, h]
            valid = pos[pos < S]
            assert (valid <= t).all(), f"行 {r} head {h} 因果违反"
            # 每行至少 1 个有效位（位置 t 本身在强制当前页内）
            assert (valid == t).any(), f"行 {r} head {h} 缺当前 query 位置"
    print("[7] select_extend query 级因果 OK")


if __name__ == "__main__":
    test_upper_bound_identity()
    test_incremental_equals_rebuild()
    test_update_row_full()
    test_select_decode_batched()
    test_select_extend_causal()
    print("ALL QUEST TENSOR TESTS PASSED")
