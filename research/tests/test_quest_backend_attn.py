"""Quest backend attention 路径 mock 级对拍（不启 Engine）。

选全页（topk 足够大）时稀疏路径数学等价 dense causal：
  - _sparse_extend_one(fused kernel 路径) vs _dense_extend_one
  - _sparse_attn_batched(decode, fused kernel) vs _dense_attn
若不一致 → 消费端 plumbing bug；一致而 e2e 乱码 → 选择/ranking bug。
"""

import sys

import torch

from sglang.srt.layers.attention.quest.backend import QuestSparseAttnBackend
from sglang.srt.layers.attention.quest.config import QuestProfile
from sglang.srt.layers.attention.quest.indexer import QuestIndexer

DEV = "cuda" if torch.cuda.is_available() else "cpu"


class _FakePool:
    def __init__(self, k, v):
        self.k, self.v = k, v

    def get_kv_buffer(self, layer_id):
        return self.k, self.v


class _FakeFB:
    def __init__(self, reqs, max_s):
        self.req_pool_indices = torch.tensor(reqs, device=DEV)
        _ = max_s


def _make_backend(profile):
    be = QuestSparseAttnBackend.__new__(QuestSparseAttnBackend)
    be.profile = profile
    be.head_dim = 16
    be.num_kv_heads = 2
    be.runner = None
    return be


def test_extend_sparse_equals_dense():
    """选全页：_sparse_extend_one == _dense_extend_one（fused kernel 路径）。"""
    torch.manual_seed(0)
    prof = QuestProfile()
    S, Hkv, H, D = 500, 2, 8, 16
    G = H // Hkv
    idx = QuestIndexer(prof, head_dim=D)
    k = (torch.randn(S, Hkv, D, device=DEV) * 3).to(torch.bfloat16)
    v = (torch.randn(S, Hkv, D, device=DEV) * 3).to(torch.bfloat16)
    pool = _FakePool(k, v)
    be = _make_backend(prof)
    locs = torch.arange(S, device=DEV)
    nq = 40
    prefix = S - nq
    q = (torch.randn(nq, H, D, device=DEV) * 2).to(torch.bfloat16)
    q_raw = q  # contiguous
    # 索引 + 选择（topk 大到选全页 → 等价 dense）
    prof_full = QuestProfile()
    prof_full.topk_pages = 32  # 32 页 > nblk(8) → 全选
    idx_full = QuestIndexer(prof_full, head_dim=D)
    full = idx_full.build_page_index(k.float())
    t_arr = torch.arange(prefix, S, device=DEV)
    sel = idx_full.select_extend(
        full["kmin"], full["kmax"], full["nblk"], q.float(), t_arr, S
    )
    out_sp = be._sparse_extend_one(
        q.float(), sel, locs, pool, 0, Hkv, G, q_raw=q_raw
    )
    out_dn = be._dense_extend_one(q.float(), locs, pool, 0, Hkv, G)
    diff = (out_sp.float() - out_dn.float()).abs().max().item()
    scale = out_dn.float().abs().max().item()
    print(f"[extend] fused-vs-dense max diff = {diff:.3e} (scale {scale:.2f})")
    assert diff < 0.05 * max(scale, 1.0), "extend 稀疏全页 != dense"

    # eager 回退路径同样对拍（关 kernel）
    prof_nk = QuestProfile()
    prof_nk.topk_pages = 32
    prof_nk.use_sparse_attn_kernel = False
    be_nk = _make_backend(prof_nk)
    out_sp2 = be_nk._sparse_extend_one(
        q.float(), sel, locs, pool, 0, Hkv, G, q_raw=q_raw
    )
    diff2 = (out_sp2.float() - out_dn.float()).abs().max().item()
    print(f"[extend] eager-vs-dense max diff = {diff2:.3e}")
    assert diff2 < 1e-4 * max(scale, 1.0), "extend eager 全页 != dense"


def test_decode_sparse_equals_dense():
    """选全页：_sparse_attn_batched == _dense_attn（decode，n=2 行）。"""
    torch.manual_seed(1)
    prof = QuestProfile()
    S, Hkv, H, D = 300, 2, 8, 16
    G = H // Hkv
    idx = QuestIndexer(prof, head_dim=D)
    be = _make_backend(prof)
    k = (torch.randn(S, Hkv, D, device=DEV) * 3).to(torch.bfloat16)
    v = (torch.randn(S, Hkv, D, device=DEV) * 3).to(torch.bfloat16)
    pool = _FakePool(k, v)
    locs = torch.arange(S, device=DEV)
    q = (torch.randn(2, H, D, device=DEV) * 2).to(torch.bfloat16)
    prof_full = QuestProfile()
    prof_full.topk_pages = 16  # 16 页 ≥ nblk(5) → 全选
    idx_full = QuestIndexer(prof_full, head_dim=D)
    fake_pool_l = _fake_pool_l(k, prof_full, idx_full)
    sel = idx_full.select_decode_batched(
        fake_pool_l, torch.tensor([0, 1], device=DEV), [S, S - 50], q
    )
    fb = _FakeFB([7, 9], S)
    req_to_token = torch.zeros(16, S, dtype=torch.long, device=DEV)
    req_to_token[7] = locs
    req_to_token[9] = locs
    out_sp = be._sparse_attn_batched(
        q, [0, 1], sel, [S, S - 50], fb, req_to_token, pool, 0, Hkv, G
    )
    out_dn0 = be._dense_attn(q[0].float(), locs[:S], pool, 0)
    out_dn1 = be._dense_attn(q[1].float(), locs[: S - 50], pool, 0)
    d0 = (out_sp[0].float() - out_dn0.float()).abs().max().item()
    d1 = (out_sp[1].float() - out_dn1.float()).abs().max().item()
    print(f"[decode] fused-vs-dense max diff = {max(d0, d1):.3e}")
    assert max(d0, d1) < 0.05, "decode 稀疏全页 != dense"


def _fake_pool_l(k, prof, idx):
    full = idx.build_page_index(k.float())
    dev = k.device
    Hkv, D = k.shape[1], k.shape[2]
    nblk_cap = max(full["nblk"], 8)
    pad = nblk_cap - full["kmin"].shape[0]
    row_min = torch.nn.functional.pad(
        full["kmin"], (0, 0, 0, 0, 0, pad)
    )  # [nblk_cap, Hkv, D]
    row_max = torch.nn.functional.pad(full["kmax"], (0, 0, 0, 0, 0, pad))
    # 两行同内容（测试只读行 0/1，第二行内容不影响 rows[0] 选择）
    return {
        "kmin": torch.stack([row_min, row_min]),
        "kmax": torch.stack([row_max, row_max]),
    }


if __name__ == "__main__":
    test_extend_sparse_equals_dense()
    test_decode_sparse_equals_dense()
    print("ALL BACKEND ATTN MOCK TESTS PASSED")
