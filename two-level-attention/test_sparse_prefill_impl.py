#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""稀疏 prefill 实现专项测试（CPU only，用户 2026-10-08 指令的验收）。

测 ops/eager_prefill.py 的 sparse_prefill_attn（TLI_SPARSE_PREFILL=1 门控的
chunk 共享选择语义，MoBA 口径）：

  T1  短序列（S<1024）：全 dense——输出 == fp32 dense 因果参考（逐位级）
  T2  S=3000 两 chunk：首 chunk（0..2047）行 == dense 参考行；
      输出无 NaN、shape 正确
  T3  S=3000 chunk1：输出 == 「末行选择 sel ∩ 因果」参考 attention；
      sel 的 sink 区 [0,128) 全 True、swa 区（块末视角最近 128）全 True
  T4  S=5000 三 chunk：中间 chunk 选择只可见前缀——prepare_mask 收到的
      k 长度 = 各 chunk 末位置+1（4096/5000），q_ids = 末行位置
  T5  E103 per_q_head 模式（mask [H,T]）路径可用，输出有限
  T6  dense 臂（sel=None 的 _chunk_rows_attn 直测）== dense 参考
  T7  TL-PREFILL-SWA-002 行级保护带：最小共享选集（sink∪块末SWA）下
      chunk 首/中/末行的自身 128-token 窗口必须恢复（GPT 审计反例）

用法：python3 test_sparse_prefill_impl.py   （two-level-attention/ 下）
"""
import os
import sys
import types

sys.dont_write_bytecode = True   # 主树正被 E109 扫描使用，只读 import 不落字节码

import torch

torch.set_num_threads(1)

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from sparse_attn.indexer import TLIIndexer          # noqa: E402
from sparse_attn.ops.eager_prefill import (         # noqa: E402
    sparse_prefill_attn,
    _chunk_rows_attn,
)

RESULTS = []


def report(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))


def make_args(**kw):
    a = types.SimpleNamespace(
        tia_block_size=64,
        tia_level1_topk=128,
        tia_level2_topk=1024,
        tia_level2_cmp_ratio=4,
        tia_enable_async_topk=False,
        tli_enable_layer_skip=False,
        tli_enable_kmeans=False,
    )
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def build_qkv(L, H=4, Hkv=2, D=128, seed=20261008):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(1, L, H, D, generator=g)
    k = torch.randn(1, L, Hkv, D, generator=g)
    v = torch.randn(1, L, Hkv, D, generator=g)
    return q, k, v


def ref_attn(q, k, v, scale, sel_h=None, sink_tok=0, swa_tok=0):
    """fp32 参考因果 attention（GQA 展开）。sel_h: [H, S] bool 或 None=dense。

    sink_tok/swa_tok > 0 时叠加行级保护带（TL-PREFILL-SWA-002 语义，与实现
    _chunk_rows_attn 同口径）：行 p 额外可见 sink 列 [0, sink_tok) 与自身窗口
    [max(0, p-swa_tok+1), p]（均 ∩ 因果）。dense 行该带 ⊆ 因果恒 no-op。
    """
    L, H, D = q.shape[1], q.shape[2], q.shape[3]
    S, Hkv = k.shape[1], k.shape[2]
    G = H // Hkv
    qf = (q[0] * scale).float()                       # [L,H,D]
    k_e = k[0].float().repeat_interleave(G, dim=1)    # [S,H,D]
    v_e = v[0].repeat_interleave(G, dim=1)            # [S,H,D]
    s = torch.einsum("lhd,shd->hls", qf, k_e)         # fp32 [H,L,S]
    causal = torch.arange(L)[:, None] >= torch.arange(S)[None, :]
    m = causal[:, None, :].expand(L, H, S)
    if sel_h is not None:
        m = sel_h[None, :, :] & causal[:, None, :]
    if sink_tok > 0 or swa_tok > 0:
        pos = torch.arange(L)
        prot = (torch.arange(S)[None, :] < sink_tok) | (
            torch.arange(S)[None, :] >= (pos - swa_tok + 1)[:, None]
        )
        m = m | (prot & causal)[:, None, :]
    s = s.permute(1, 0, 2).masked_fill(~m, float("-inf"))  # [L,H,S]
    p = torch.softmax(s, dim=-1)
    o = torch.einsum("lhs,shd->lhd", p, v_e)
    return o.unsqueeze(0)


def expand_sel(sel, Hkv, G):
    """[Hkv,S] 共享 → [H,S]（q-head 布局 (h g)）；[H,S] 原样返回。"""
    if sel.shape[0] == Hkv:
        return sel.repeat_interleave(G, dim=0)
    return sel


def expected_sel(indexer, q, k, c1, scale, pad_to=None):
    """独立重放：末行 q + k[:, :c1] 切片调 prepare_mask（与实现同口径）。

    pad_to：中间 chunk 的 sel 只覆盖 [0, c1)——与全长参考对齐时右侧补 False
    （[c1, pad_to) 对该 chunk 全体行是未来 token，因果已排除，补值不改变语义）。
    """
    cu = torch.tensor([0, c1], dtype=torch.int32)
    q_ids = cu[1:] - 1
    mask, _ = indexer.prepare_mask(q[:, c1 - 1 : c1], q_ids, k[:, :c1], cu, scale)
    sel = mask[0, 0]
    if pad_to is not None and sel.shape[-1] < pad_to:
        pad = torch.zeros(*sel.shape[:-1], pad_to, dtype=sel.dtype)
        pad[..., : sel.shape[-1]] = sel
        sel = pad
    return sel


# ================================================================ T1 短序列全 dense
def t1_short_dense():
    name = "T1 S=800(<1024) 全 dense == fp32 参考"
    try:
        q, k, v = build_qkv(800)
        idx = TLIIndexer(make_args())
        idx.layer_idx = 3
        scale = 128 ** -0.5
        o = sparse_prefill_attn(idx, q, k, v, softmax_scale=scale)
        ref = ref_attn(q, k, v, scale)
        d = float((o - ref).abs().max())
        assert o.shape == q.shape and not torch.isnan(o).any()
        assert d < 1e-5, f"max|Δ|={d:.3e}"
        report(name, True, f"max|Δ|={d:.2e}")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ T2 两 chunk 首行 dense + NaN/shape
def t2_two_chunks():
    name = "T2 S=3000 两 chunk：首 chunk 行==dense / 无 NaN / shape 正确"
    try:
        L = 3000
        q, k, v = build_qkv(L)
        idx = TLIIndexer(make_args())
        idx.layer_idx = 3
        scale = 128 ** -0.5
        o = sparse_prefill_attn(idx, q, k, v, softmax_scale=scale)
        assert o.shape == (1, L, 4, 128), f"shape={list(o.shape)}"
        assert not torch.isnan(o).any(), "输出含 NaN"
        ref = ref_attn(q, k, v, scale)
        d0 = float((o[0, :2048] - ref[0, :2048]).abs().max())
        assert d0 < 1e-5, f"首 chunk max|Δ|={d0:.3e}"
        report(name, True, f"首 chunk max|Δ|={d0:.2e}")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ T3 chunk1 = sel∩因果 参考 + sink/swa 强制
def t3_sel_causal():
    name = "T3 S=3000 chunk1 == 末行选择∩因果 参考；sink/swa 强制区（分区臂）"
    try:
        L = 3000
        q, k, v = build_qkv(L)
        # 分区臂（α/β>0）：compute_mask 的 sink/swa 正交强制区生效
        # （单池 (0,0) 口径 indexer 本就不强制 sink——既有语义非本实现关注点）
        cfg = dict(tli_alpha=0.25, tli_beta=0.25, tli_gamma=0.125)
        idx = TLIIndexer(make_args(**cfg))
        idx.layer_idx = 3
        scale = 128 ** -0.5
        o = sparse_prefill_attn(idx, q, k, v, softmax_scale=scale)
        # 独立重放末行选择（chunk1 末 = 3000 = L）
        sel = expected_sel(TLIIndexer(make_args(**cfg)), q, k, L, scale)  # [Hkv, S]
        H, Hkv, G = 4, 2, 2
        sel_h = expand_sel(sel, Hkv, G)               # [H, S]
        assert bool(sel[:, :128].all()), "sink 区 [0,128) 未全保留"
        assert bool(sel[:, -128:].all()), "swa 区（块末最近 128）未全保留"
        # 参考 oracle 含行级保护带（TL-PREFILL-SWA-002 后的正确语义；
        # 旧 oracle 只用末行 sel∩因果 会把缺陷语义复制进参考）
        ref = ref_attn(q, k, v, scale, sel_h=sel_h, sink_tok=128, swa_tok=128)
        d1 = float((o[0, 2048:] - ref[0, 2048:]).abs().max())
        assert d1 < 1e-5, f"chunk1 max|Δ|={d1:.3e}"
        report(name, True, f"chunk1 max|Δ|={d1:.2e}")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ T4 中间 chunk 因果切片
def t4_intermediate_chunk():
    name = "T4 S=5000 三 chunk：中间 chunk 选择只可见前缀（k 长度=块末+1）"
    try:
        L = 5000
        q, k, v = build_qkv(L)
        idx = TLIIndexer(make_args())
        idx.layer_idx = 3
        scale = 128 ** -0.5
        # 记录 prepare_mask 收到的 k 长度与 q_ids（验证因果切片）
        seen = []
        orig = idx.prepare_mask

        def spy(qr, qi, kr, cu, sc=None):
            seen.append((int(kr.shape[1]), int(qi[0])))
            return orig(qr, qi, kr, cu, sc)

        idx.prepare_mask = spy
        o = sparse_prefill_attn(idx, q, k, v, softmax_scale=scale)
        assert seen == [(4096, 4095), (5000, 4999)], \
            f"prepare_mask 调用序列异常: {seen}"
        # chunk1（2048..4095）参考：选择来自 k[:, :4096] 的末行（4095），
        # sel 右侧 pad False 到全长 5000（未来 token 因果已排除）
        sel1 = expected_sel(TLIIndexer(make_args()), q, k, 4096, scale, pad_to=L)
        sel1_h = expand_sel(sel1, 2, 2)
        ref1 = ref_attn(q, k, v, scale, sel_h=sel1_h, sink_tok=128, swa_tok=128)
        d1 = float((o[0, 2048:4096] - ref1[0, 2048:4096]).abs().max())
        assert d1 < 1e-5, f"chunk1 max|Δ|={d1:.3e}"
        report(name, True, f"k_len 序列={seen} chunk1 max|Δ|={d1:.2e}")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ T5 per_q_head 路径
def t5_per_q_head():
    name = "T5 per_q_head（mask [H,S]）路径输出有限且 == 参考"
    try:
        L = 3000
        q, k, v = build_qkv(L)
        args = make_args(tli_per_q_head=True)
        idx = TLIIndexer(args)
        idx.layer_idx = 3
        scale = 128 ** -0.5
        o = sparse_prefill_attn(idx, q, k, v, softmax_scale=scale)
        assert o.shape == q.shape and torch.isfinite(o).all(), "输出非有限"
        sel = expected_sel(TLIIndexer(args), q, k, L, scale)  # [H, S]
        assert sel.shape[0] == 4, f"per_q_head mask 头数={sel.shape[0]}"
        ref = ref_attn(q, k, v, scale, sel_h=sel, sink_tok=128, swa_tok=128)
        d = float((o[0, 2048:] - ref[0, 2048:]).abs().max())
        assert d < 1e-5, f"max|Δ|={d:.3e}"
        report(name, True, f"chunk1 max|Δ|={d:.2e}")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ T7 行级保护带（TL-PREFILL-SWA-002）
def t7_row_protection_band():
    name = "T7 逐行 SWA/sink 保护带：最小选集下首/中/末行自身窗口恢复"
    # GPT 审计最小验收反例：chunk [2048,3000) 的最小共享选集 = sink ∪ 块末
    # SWA（[2872,2999]）；行 2048 的自身窗口 [1921,2048] 与之零重叠——
    # 修复前其有效选集只剩 128 个 sink token，自身 128-token 局部窗口全丢。
    # 本测试用 monkeypatch 的最小选集（比真实 topk 更极端）直击该反例。
    try:
        L = 3000
        # 构造：q/k 全零 → 所有 score 相等 → softmax 均匀 → 输出 = 选中
        # token 的 v 均值。v 在三个探针行自身窗口内置 1、其余 0：
        #   行 2048 窗口 [1921,2048]（chunk 首行）、行 2500 窗口 [2373,2500]
        #   （中行）、行 2999 窗口 [2872,2999]（末行=块末 SWA 本就在选集）
        # 修复后各探针行有效选集 = 128 sink(v=0) + 128 自身窗口(v=1) → 0.5
        # 修复前（无保护带）：首/中行只余 sink → 0.0（末行不受影响 0.5）
        g = torch.Generator().manual_seed(20261009)
        H, Hkv, D = 2, 1, 8
        q = torch.zeros(1, L, H, D)
        k = torch.zeros(1, L, Hkv, D)
        v = torch.zeros(1, L, Hkv, D)
        for lo, hi in [(1921, 2049), (2373, 2501), (2872, 3000)]:
            v[0, lo:hi, 0, 0] = 1.0
        idx = TLIIndexer(make_args())
        idx.layer_idx = 3

        def minimal_sel(qr, qi, kr, cu, sc=None):
            # 最小共享选集：sink [0,128) ∪ 块末视角 SWA [c1-128, c1)
            c1 = int(kr.shape[1])
            m = torch.zeros(1, 1, Hkv, c1, dtype=torch.bool)
            m[..., :128] = True
            m[..., c1 - 128:] = True
            return m, 0

        idx.prepare_mask = minimal_sel
        scale = D ** -0.5
        o = sparse_prefill_attn(idx, q, k, v, softmax_scale=scale)
        probes = {2048: 0.5, 2500: 0.5, 2999: 0.5}
        for r, expect in probes.items():
            got = float(o[0, r, :, 0].mean())
            assert abs(got - expect) < 1e-6, \
                f"行 {r} 输出 {got:.4f} != {expect}（自身 SWA 未恢复）"
        # dense 首 chunk 行不受保护带影响：行 2000（chunk 0 内、因果全可见）
        # 期望 = 均匀 softmax 下 v=1 可见 token 数 / 因果长度 =
        # [1921,2000] 共 80 个 / 2001
        got0 = float(o[0, 2000, :, 0].mean())
        assert abs(got0 - 80.0 / 2001) < 1e-6, \
            f"dense 行 2000 输出 {got0:.6f} != 80/2001（dense 路径被扰动）"
        report(name, True, "首/中/末行自身 SWA 恢复（0.5）；dense 行不受扰")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ T6 dense 臂直测
def t6_dense_rows():
    name = "T6 _chunk_rows_attn(sel=None) == dense 参考（任意行子块）"
    try:
        L, H, Hkv = 3000, 4, 2
        q, k, v = build_qkv(L)
        scale = 128 ** -0.5
        # 直接调用内部行块函数（pos_lo=0 起点，全 dense）
        o = _chunk_rows_attn(q, k, v, None, 0, H // Hkv, scale)
        ref = ref_attn(q, k, v, scale)
        d = float((o - ref).abs().max())
        assert d < 1e-5, f"max|Δ|={d:.3e}"
        report(name, True, f"max|Δ|={d:.2e}")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


if __name__ == "__main__":
    t1_short_dense()
    t2_two_chunks()
    t3_sel_causal()
    t4_intermediate_chunk()
    t5_per_q_head()
    t6_dense_rows()
    t7_row_protection_band()
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n===== {len(RESULTS) - n_fail}/{len(RESULTS)} PASS =====")
    sys.exit(1 if n_fail else 0)
