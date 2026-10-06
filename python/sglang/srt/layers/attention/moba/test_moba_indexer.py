"""MoBA indexer/backend 张量级单测（不启 Engine，import 级冒烟的一部分）。

对拍口径 = two-level-attention/sparse_attn/indexer/tli_indexer.py 的
moba_gate 路径（E89：L224 k_avg_full / L356-367 score_moba / L414-446
_moba_mask）。参考实现按 E89 算子序列逐行复刻：
  - chunk-mean：零 pad → reshape [kt, bs, Hkv, D] → mean(dim=2)
  - gate：repeat 到 (h g) 布局 → einsum "h d, kt h d -> h kt" → 组内 mean
  - mask：top-(K2/BS) 块展开（clamp 尾界）scatter + sink/swa 直接置位

覆盖：
  1. chunk-mean（ksum/bs）== E89 零 pad mean（逐位，fp32 输入）
  2. gate 分数 == E89 score_moba（逐位，fp32 输入）
  3. decode 选择的 token 集合 == E89 _moba_mask 的 True 集合（含多行 /
     非 chunk 对齐 S / S < sink+swa 全保送边界）
  4. decode 逐 token 增量 == 全量重建（块和 ulp 级；选择集合一致）
  5. _update_row_full 多 token 增量（chunked prefill 边界）== 全量重建
  6. select_extend：query 级因果（pos ≤ t）+ 每行至少 1 有效位 +
     集合 == 逐 query 慢速参考
  7. 预算上界：|选中 token| ≤ nb·bs + sink + swa

已知实现级偏差（见 indexer.py docstring）：E89 每 decode 步全量重建、
本实现增量维护块和 → 测试 4/5 用 allclose（ulp 级）+ 集合一致断言；
测试 1/2/3 用全量 build 路径（无增量漂移），要求逐位/集合严格一致。

跑法：CUDA_VISIBLE_DEVICES=x python3 test_moba_indexer.py（小张量，
显存 < 100MB）。
"""

import math

import torch

from sglang.srt.layers.attention.moba.config import MoBAProfile
from sglang.srt.layers.attention.moba.indexer import MoBAIndexer


def _make(S, Hkv=2, H=8, D=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    k = torch.randn(S, Hkv, D, generator=g).to(dev)
    q = torch.randn(1, H, D, generator=g).to(dev)
    return k, q[0]


def _small_profile(**kw):
    """小预算 profile（nb=4 块 + sink/swa 各 2 块，选择有区分度）。"""
    p = MoBAProfile()
    p.chunk_size = kw.get("chunk_size", 64)
    p.token_budget = kw.get("token_budget", 256)
    p.sink_tokens = kw.get("sink_tokens", 128)
    p.sliding_window = kw.get("sliding_window", 128)
    p.dense_threshold = kw.get("dense_threshold", 2048)
    return p


# ---------------------------------------------------------------------- #
# E89 参考实现（tli_indexer.py 的算子序列逐行复刻，fp32 输入口径）
# ---------------------------------------------------------------------- #

def _e89_gate(k, q, bs, scale=None):
    """E89 score_moba：chunk-mean gate 分数（组内 mean 后）。

    k: [S, Hkv, D] fp32；q: [H, D] fp32。返回 [Hkv, kt]。
    """
    S, Hkv, D = k.shape
    H = q.shape[0]
    G = H // Hkv
    if scale is None:
        scale = D**-0.5
    kt = (S + bs - 1) // bs
    pad = kt * bs - S
    pad_k = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, pad)) if pad else k
    kc = pad_k.reshape(kt, bs, Hkv, D)
    # E89 rearrange("b (t bs) h d -> b t bs h d").mean(dim=2) 是对 bs 维求
    # 均值（4D 含 batch）；本 3D 口径 reshape 后 token 维 = dim=1
    k_avg_full = kc.mean(dim=1)  # [kt, Hkv, D]（E89 L224-228）
    # einops repeat "t h d -> t (h g) d" 的 torch 等价
    b_k = (
        k_avg_full.unsqueeze(2).expand(kt, Hkv, G, D).reshape(kt, Hkv * G, D)
    )  # [kt, H, D]
    b_q = (q * scale).to(torch.float32)  # [H, D]（E89 L362：先乘 scale 再 cast）
    # E89 用 einops einsum（"kt" 是单标签）；torch.einsum 下标须逐字母，
    # 此处块维记作 m
    score_h = torch.einsum("h d, m h d -> h m", b_q, b_k)  # [H, kt]（E89 L364-366）
    return score_h.view(Hkv, G, kt).mean(dim=1)  # 组内 mean → [Hkv, kt]（E89 L422）


def _e89_mask(sm, S, bs, K2, sink_tok, swa_tok):
    """E89 _moba_mask：top-(K2/BS) 块展开 scatter + sink/swa 置位。

    sm: [Hkv, kt] gate 分数。返回 [Hkv, S] bool。
    """
    Hkv, kt = sm.shape
    nb_sel = max(1, K2 // bs)
    i_blk = torch.topk(sm, min(nb_sel, sm.shape[-1]), dim=-1).indices  # E89 L429
    m = torch.zeros(Hkv, S, dtype=torch.bool, device=sm.device)
    tok_hi = S
    tok = (
        i_blk.unsqueeze(-1) * bs
        + torch.arange(bs, device=sm.device)
    ).clamp(max=tok_hi - 1).reshape(Hkv, -1)  # E89 L434-437（尾块越界 clamp）
    m.scatter_(1, tok, True)
    m[:, :sink_tok] = True  # E89 L440
    m[:, max(0, tok_hi - swa_tok):] = True  # E89 L441
    return m


def _fake_pool(profile, S_cap, Hkv, D, R=4):
    cs = profile.chunk_size
    nblk_cap = (S_cap + cs - 1) // cs
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "ksum": torch.zeros(R, Hkv, nblk_cap, D, dtype=torch.float32, device=dev),
        "S_cap": S_cap,
        "R_cap": R,
        "free": list(range(R)),
        "row_of": {},
        "S": [-1] * R,
    }


def _sel_to_mask(sel, S):
    """sel [n, Hkv, K]（哨兵 ≥ S）→ bool mask [n, Hkv, S]。"""
    n, Hkv, _ = sel.shape
    m = torch.zeros(n, Hkv, S, dtype=torch.bool, device=sel.device)
    valid = sel < S
    idx = sel.clamp(max=S - 1)
    m.scatter_(-1, idx, valid)
    return m


# ---------------------------------------------------------------------- #
# 测试
# ---------------------------------------------------------------------- #

def test_chunk_mean_matches_e89():
    """ksum / bs == E89 零 pad mean（逐位；除以 64 是 2 的幂无舍入）。"""
    p = _small_profile()
    idx = MoBAIndexer(p, head_dim=16)
    S, Hkv, D = 1000, 2, 16
    k, _ = _make(S, Hkv, D=D, seed=1)
    full = idx.build_chunk_index(k)
    kt = (S + p.chunk_size - 1) // p.chunk_size
    pad = kt * p.chunk_size - S
    pad_k = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, pad)) if pad else k
    ref_mean = pad_k.reshape(kt, p.chunk_size, Hkv, D).mean(dim=1)  # [kt,Hkv,D]（token 维 = dim=1）
    mine = full["ksum"].permute(1, 0, 2) / p.chunk_size  # [kt,Hkv,D]
    torch.testing.assert_close(mine, ref_mean, rtol=0, atol=0)
    # 尾块（40 真实 token + 24 零 pad）均值 ≈ 真实和 / 64（E89 稀释口径）。
    # ulp 级容差：64 元素归约（含零）与 40 元素归约的加法结合顺序不同，
    # fp32 下有 ~1e-8 级漂移（数学上零不改和，浮点归约顺序是本质差异）
    tail_real = k[(kt - 1) * p.chunk_size :].sum(0) / p.chunk_size
    torch.testing.assert_close(mine[-1], tail_real, rtol=1e-5, atol=1e-7)
    print("[1] chunk-mean == E89 零 pad mean OK (逐位)")


def test_gate_scores_match_e89():
    """_gate_scores == E89 score_moba（逐位，fp32 输入）。"""
    p = _small_profile()
    idx = MoBAIndexer(p, head_dim=16)
    S, Hkv, H, D = 1000, 2, 8, 16
    k, q = _make(S, Hkv, H=H, D=D, seed=2)
    full = idx.build_chunk_index(k)
    kmean = full["ksum"].unsqueeze(0) / p.chunk_size  # [1, Hkv, kt, D]
    mine = idx._gate_scores(q.view(1, H, D), kmean)[0]  # [Hkv, kt]
    ref = _e89_gate(k, q, p.chunk_size)  # [Hkv, kt]
    diff = (mine - ref).abs().max().item()
    torch.testing.assert_close(mine, ref, rtol=0, atol=0)
    print(f"[2] gate 分数 == E89 score_moba OK (逐位, max diff {diff:.2e})")


def test_select_decode_matches_e89():
    """decode 选择的 token 集合 == E89 _moba_mask True 集合。

    覆盖：非 chunk 对齐 S（1000）+ 多行不同 S + S < sink+swa 全保送边界。
    """
    p = _small_profile()
    idx = MoBAIndexer(p, head_dim=16)
    bs, K2 = p.chunk_size, p.token_budget
    S_list = [1000, 967, 200]  # 含 < sink+swa 的全保送边界
    Hkv, H, D = 2, 8, 16
    g = torch.Generator().manual_seed(3)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n = len(S_list)
    ks = [torch.randn(S, Hkv, D, generator=g).to(dev) for S in S_list]
    qs = torch.randn(n, H, D, generator=g).to(dev)
    pool = _fake_pool(p, 2048, Hkv, D, R=n)
    for r, k in enumerate(ks):
        full = idx.build_chunk_index(k)
        pool["ksum"][r, :, : full["nblk"]] = full["ksum"]
    rows = torch.arange(n, device=dev)
    sel = idx.select_decode_batched(pool, rows, S_list, qs)  # [n,Hkv,K]
    K = p.select_blocks * bs + p.sink_tokens + p.sliding_window
    assert sel.shape == (n, Hkv, K), sel.shape
    m = _sel_to_mask(sel, max(S_list))  # 哨兵已掩；行界内位置再按 S 截
    for r, S in enumerate(S_list):
        ref = _e89_mask(_e89_gate(ks[r], qs[r], bs), S, bs, K2,
                        p.sink_tokens, p.sliding_window)  # [Hkv, S]
        mine = m[r, :, :S]
        assert torch.equal(mine, ref), (
            f"row {r} (S={S}) mask 不一致: "
            f"diff={int((mine != ref).sum())}"
        )
        n_sel = int(mine.sum())  # [Hkv, S] 总和：每 head 上界 K（互斥三组 lane）
        assert n_sel <= Hkv * K, f"row {r} 超预算: {n_sel} > {Hkv * K}"
        if S < p.sink_tokens + p.sliding_window:
            assert bool(mine.all()), f"row {r} 全保送边界应全覆盖"
    print("[3] decode 选择集合 == E89 _moba_mask OK (3 行含边界)")


def test_incremental_equals_rebuild():
    """逐 token 增量 == 全量重建（块和 ulp 级 allclose + 选择集合一致）。"""
    p = _small_profile()
    idx = MoBAIndexer(p, head_dim=16)
    S, Hkv, H, D = 1000, 2, 8, 16
    k, q = _make(S, Hkv, H=H, D=D, seed=4)
    pool1 = _fake_pool(p, 2048, Hkv, D)
    full = idx.build_chunk_index(k)
    pool1["ksum"][0, :, : full["nblk"]] = full["ksum"]
    # 增量版：首 token 全量 + 逐步追加
    pool2 = _fake_pool(p, 2048, Hkv, D)
    f0 = idx.build_chunk_index(k[:1])
    pool2["ksum"][0, :, : f0["nblk"]] = f0["ksum"]
    for s in range(1, S):
        idx.update_pool_rows_decode(
            pool2, torch.tensor([0], device=k.device), [s], k[s : s + 1]
        )
    nblk = full["nblk"]
    torch.testing.assert_close(
        pool2["ksum"][0, :, :nblk], pool1["ksum"][0, :, :nblk],
        rtol=1e-5, atol=1e-4,
    )
    # 选择集合必须一致（漂移不翻转 topk）
    rows = torch.tensor([0], device=k.device)
    sel1 = idx.select_decode_batched(pool1, rows, [S], q.view(1, H, D))
    sel2 = idx.select_decode_batched(pool2, rows, [S], q.view(1, H, D))
    assert torch.equal(
        _sel_to_mask(sel1, S), _sel_to_mask(sel2, S)
    ), "增量与全量的选择集合不一致"
    print("[4] 增量 == 全量重建 OK (allclose ulp 级 + 集合一致)")


def test_update_row_full():
    """多 token 增量（chunk 边界三段式）== 全量重建 + 集合一致。"""
    p = _small_profile()
    idx = MoBAIndexer(p, head_dim=16)
    from sglang.srt.layers.attention.moba.backend import MoBASparseAttnBackend

    S, Hkv, H, D = 2000, 2, 8, 16
    k, q = _make(S, Hkv, H=H, D=D, seed=5)
    full = idx.build_chunk_index(k)
    be = MoBASparseAttnBackend.__new__(MoBASparseAttnBackend)  # 不走 runner
    be.profile = p
    pool = _fake_pool(p, 4096, Hkv, D)
    # 分三段：0-1000（全量）、1000-1500、1500-2000（增量；边界非对齐）
    f0 = idx.build_chunk_index(k[:1000])
    pool["ksum"][0, :, : f0["nblk"]] = f0["ksum"]
    be._update_row_full(pool, 0, 1000, 1500, k[1000:1500])
    be._update_row_full(pool, 0, 1500, 2000, k[1500:2000])
    nblk = full["nblk"]
    torch.testing.assert_close(
        pool["ksum"][0, :, :nblk], full["ksum"], rtol=1e-5, atol=1e-4
    )
    rows = torch.tensor([0], device=k.device)
    sel = idx.select_decode_batched(pool, rows, [S], q.view(1, H, D))
    ref_pool = _fake_pool(p, 4096, Hkv, D)
    ref_pool["ksum"][0, :, :nblk] = full["ksum"]
    sel_ref = idx.select_decode_batched(ref_pool, rows, [S], q.view(1, H, D))
    assert torch.equal(_sel_to_mask(sel, S), _sel_to_mask(sel_ref, S))
    print("[5] _update_row_full == rebuild OK (allclose ulp 级 + 集合一致)")


def test_select_extend_causal_and_ref():
    """extend：query 级因果 + 每 query 至少 1 有效位 + 集合 == 慢速参考。

    慢速参考（E89 无 prefill 路径，参考 = decode 语义按 query t 平移）：
    可排名块（起点 ≤ t）gate top-K 展开 ∩ [0, t] ∪ sink ∩ [0,t] ∪
    swa [max(0,t+1-swa), t]，三组去重后与 sel 集合比对。
    """
    p = _small_profile()
    idx = MoBAIndexer(p, head_dim=16)
    bs, nb, sink_t, swa_t = (
        p.chunk_size, p.select_blocks, p.sink_tokens, p.sliding_window
    )
    S, Hkv, H, D = 1000, 2, 8, 16
    k, q = _make(S, Hkv, H=H, D=D, seed=6)
    full = idx.build_chunk_index(k)
    nq = 130
    t_arr = torch.arange(S - nq, S, device=k.device)
    sel = idx.select_extend(
        full["ksum"], full["nblk"], q.view(1, H, D).expand(nq, H, D).contiguous(),
        t_arr, S,
    )
    K = nb * bs + sink_t + swa_t
    assert sel.shape == (nq, Hkv, K), sel.shape
    gate = _e89_gate(k, q, bs)  # [Hkv, kt]（query 无关，chunk-mean 只依赖 k）
    kt = full["nblk"]
    for r in range(0, nq, 17):  # 抽查（含 t 跨块边界）
        t = int(t_arr[r])
        for h in range(Hkv):
            pos = sel[r, h]
            valid = pos[pos < S]
            assert (valid <= t).all(), f"行 {r} head {h} 因果违反"
            assert (valid == t).any(), f"行 {r} head {h} 缺当前 query 位置"
            # ---- 慢速参考 ----
            rankable = [m for m in range(kt) if m * bs <= t]
            sc = [gate[h, m].item() for m in rankable]
            order = sorted(
                range(len(rankable)), key=lambda j: -sc[j]
            )[: min(nb, len(rankable))]
            blk_sel = {rankable[j] for j in order}
            toks = set()
            for m in blk_sel:
                for pos_t in range(m * bs, min((m + 1) * bs, S)):
                    if pos_t <= t and not (pos_t < sink_t or pos_t >= max(0, t + 1 - swa_t)):
                        toks.add(pos_t)
            toks |= set(range(min(sink_t, t + 1)))
            toks |= set(range(max(0, t + 1 - swa_t, sink_t), t + 1))
            mine = set(valid.tolist())
            assert mine == toks, (
                f"行 {r} head {h} 集合不一致: "
                f"仅我方 {sorted(mine - toks)[:8]} 仅参考 {sorted(toks - mine)[:8]}"
            )
    print("[6] select_extend 因果 + 集合 == 慢速参考 OK")


def test_backend_import_and_pool():
    """backend 可构造（runner=None）+ pool 布局/扩容 + veto/哨兵约定。"""
    from sglang.srt.layers.attention.moba.backend import MoBASparseAttnBackend

    be = MoBASparseAttnBackend(runner=None)
    be.num_kv_heads = 2
    be.head_dim = 16
    be.profile = _small_profile()
    be.runner = None
    be._num_layers = 1
    # _get_pool 无 runner 时用 "cuda"（有卡环境）
    pool = be._get_pool(0)
    cs = be.profile.chunk_size
    assert pool["ksum"].shape == (
        be.profile.pool_rows, 2, (pool["S_cap"] + cs - 1) // cs, 16
    )
    assert pool["ksum"].dtype == torch.float32
    # S 扩容：前缀保留
    old = pool["ksum"].clone()
    be._ensure_pool_s(pool, 5000)
    assert pool["S_cap"] >= 5000
    nblk_old = old.shape[2]
    torch.testing.assert_close(pool["ksum"][:, :, :nblk_old], old, rtol=0, atol=0)
    assert be.veto_cuda_graph(None) is True
    print("[7] backend 构造 / pool 布局与扩容 / graph veto OK")


if __name__ == "__main__":
    test_chunk_mean_matches_e89()
    test_gate_scores_match_e89()
    test_select_decode_matches_e89()
    test_incremental_equals_rebuild()
    test_update_row_full()
    test_select_extend_causal_and_ref()
    test_backend_import_and_pool()
    print("ALL MOBA TENSOR TESTS PASSED")
