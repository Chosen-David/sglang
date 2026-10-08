#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C3 修复专项回归测试：near 簇分 softmax_scale 量纲对齐（CPU only）。

缺陷（核验报告 research/docs/gpt_audit_verdict_20261008_round2.md C3 节）：
  compute_score 缓存的 _last_q 未乘 softmax_scale，而 near 池混合竞争中
  score_fine 回退段（b_q_full = q*softmax_scale）是缩放口径 → 簇覆盖段分数
  被放大量纲 |1/scale| 倍（D=128 时 11.31×），系统性挤出回退段高分 token。

修复：缓存处统一乘 scale（far 侧纯簇分 topk 乘正数排序不变 → far 逐位不变）
      + clear() 重置 _last_q。

红-绿对照：旧代码（git HEAD，/tmp/c3_old/）vs 新代码（工作树）。
  红：回退段正确高分 token j（缩放口径最高）在旧代码下落选；
  绿：新代码下入选；且 far 区选中集合新旧逐位相同（far 不变性）。

用法：python3 test_c3_near_scale_fix.py   （two-level-attention/ 下）
"""
import importlib
import os
import sys
import types

sys.dont_write_bytecode = True   # 主树正被 E109 扫描使用，只读 import 不落字节码

import torch

torch.set_num_threads(1)

REPO = os.path.dirname(os.path.abspath(__file__))
OLD_ROOT = "/tmp/c3_old"

RESULTS = []


def report(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))


def load_sparse_attn(name, root):
    """root/sparse_attn 作为独立命名空间包加载（无 __init__.py，手动造 __path__）。"""
    mod = types.ModuleType(name)
    mod.__path__ = [os.path.join(root, "sparse_attn")]
    sys.modules[name] = mod
    return importlib.import_module(f"{name}.indexer")


IDX_NEW = load_sparse_attn("sparse_attn_c3new", REPO)
IDX_OLD = load_sparse_attn("sparse_attn_c3old", OLD_ROOT)


def make_args(**kw):
    a = types.SimpleNamespace(
        tia_block_size=64,
        tia_level1_topk=128,
        tia_level2_topk=1024,
        tia_level2_cmp_ratio=4,
        tia_enable_async_topk=False,
    )
    for k, v in kw.items():
        setattr(a, k, v)
    return a


C3_CFG = dict(
    tli_far_select="sim_greedy", tli_near_select="sim_greedy", tli_sim=0.9,
    tli_enable_kmeans=True, tli_enable_layer_skip=False,
    tli_alpha=0.25, tli_beta=0.25, tli_gamma=0.125,
)


def build_scene(seed=20261008):
    """竞争反例场景：所有 token 与 q 同向（中等幅度）+ 两处高亮。

    k = q_dir*0.5 + 小噪声 → 未缩放簇分 ≈0.4-0.6、缩放后 ≈0.04-0.06
    k[3500] = q_dir*1.5   → 簇覆盖区高亮（修复前簇分 1.5 / 修复后 0.133）
    k[3980] = q_dir*2.0   → 回退尾巴区高亮（缩放分 0.177 = 全场正确最高）
    回退段其余随机 token 缩放分 ≈0.04 → 修复前 j(0.177) 仍低于簇段(≥0.4) 落选；
    修复后 j(0.177) > i(0.133) > 其余(≈0.05) 全场最高必入选。
    """
    S = 4096 + 30
    g = torch.Generator().manual_seed(seed)
    H, Hkv, D = 4, 2, 128
    # q 全 head 同方向（归一化）→ 点积跨 head 一致，断言可用 .all()
    q_dir_h = torch.nn.functional.normalize(torch.randn(H, D, generator=g), dim=-1)
    q_dir = q_dir_h.mean(0)
    q_dir = torch.nn.functional.normalize(q_dir, dim=-1)
    q = q_dir.unsqueeze(0).repeat(H, 1).unsqueeze(0).unsqueeze(0).clone()  # [1,1,H,D] 全 head 同向
    # k：全部 token = q_dir*0.5 + 小噪声（kv-head 维同向）
    k = (q_dir.unsqueeze(0).unsqueeze(0) * 0.5).repeat(1, S, Hkv, 1).clone()
    k += torch.randn(1, S, Hkv, D, generator=g) * 0.05
    k[0, 3500] = q_dir * 1.5
    k[0, 3980] = q_dir * 2.0
    return S, k, q


def run_mask_idx(idxpkg, args, k, q):
    idx = idxpkg.TLIIndexer(args)
    idx.layer_idx = 3
    cu = torch.tensor([0, k.shape[1]])
    q_ids = torch.tensor([k.shape[1] - 1])
    scale = q.shape[-1] ** -0.5
    mask, _ = idx.prepare_mask(q, q_ids, k, cu, scale)
    return idx, mask, scale


# ================================================================ C3-1 缓存已缩放
def c3_1_cache_scaled():
    name = "C3-1 _last_q 缓存已乘 softmax_scale（量纲源头）"
    try:
        S, k, q = build_scene()
        idx, mask, scale = run_mask_idx(IDX_NEW, make_args(**C3_CFG), k, q)
        expect = q.squeeze(0).squeeze(0).to(torch.float32) * scale
        assert idx._last_q is not None, "compute_score 未缓存 _last_q"
        assert torch.allclose(idx._last_q, expect, atol=1e-6), \
            f"_last_q 未乘 scale：max|Δ|={float((idx._last_q - expect).abs().max()):.3e}"
        # 对照：旧代码此处是未缩放缓存（红侧存在性，静态断言）
        idx_o, _, _ = run_mask_idx(IDX_OLD, make_args(**C3_CFG), k, q)
        expect_o = q.squeeze(0).squeeze(0).to(torch.float32)
        assert torch.allclose(idx_o._last_q, expect_o, atol=1e-6), \
            "旧代码缓存口径与预期不符（红侧前提失效，需复查场景）"
        report(name, True, f"新=缩放({scale:.4f}) 旧=未缩放（放大 {1/scale:.2f}×）")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ C3-2 红绿竞争反例
def c3_2_red_green():
    name = "C3-2 红绿对拍：回退段正确高分 token 新入选 / 旧落选"
    try:
        S, k, q = build_scene()
        bs, sink_tok, swa_tok = 64, 128, 128
        mid_len = S - sink_tok - swa_tok
        near_len_dyn = max(bs, int(0.25 * mid_len))
        near_blks = max(2, (S - near_len_dyn) // bs)
        far_tok_hi = near_blks * bs                     # 簇覆盖段左界（far 池右界）
        near_hi = (S - swa_tok) // bs * bs              # 3968：near 簇覆盖右界（块对齐）
        swa_lo_tok = S - swa_tok                        # 3998
        assert 3980 >= near_hi and 3980 < swa_lo_tok, "前提：3980 在回退尾巴区"
        assert 3500 >= far_tok_hi and 3500 < near_hi, "前提：3500 在簇覆盖区"

        _, mask_new, _ = run_mask_idx(IDX_NEW, make_args(**C3_CFG), k, q)
        _, mask_old, _ = run_mask_idx(IDX_OLD, make_args(**C3_CFG), k, q)
        m_new, m_old = mask_new[0, 0], mask_old[0, 0]

        j_sel_new = bool(m_new[:, 3980].all())
        j_sel_old = bool(m_old[:, 3980].all())
        i_sel_new = bool(m_new[:, 3500].any())
        i_sel_old = bool(m_old[:, 3500].any())
        # 绿：修复后回退段正确最高分 token j 必入选
        assert j_sel_new, "修复后 3980（缩放口径全场最高分）仍未入选 —— 修复未生效"
        # 红：修复前 j 被未缩放簇段（0.4~1.5 > 0.177）挤出 —— 确认缺陷曾真实存在
        assert not j_sel_old, "修复前 3980 竟已入选 —— 反例构造失效，红绿对照无判别力"
        # 簇覆盖区高亮 i：两版都应入选（0.133 与 1.5 在各自口径下均属簇段头部）
        assert i_sel_new and i_sel_old, "簇覆盖区高亮 3500 召回异常"
        # 预算守恒（两版同）：mid 选中总数 = K2_mid
        K2_mid = 1024 - sink_tok - swa_tok
        for tag, m in (("new", m_new), ("old", m_old)):
            n_mid = int(m[0, sink_tok:swa_lo_tok].sum())
            assert n_mid == K2_mid, f"{tag}: mid 选中 {n_mid} != K2_mid {K2_mid}"
        report(name, True, f"3980: old=落选(红) new=入选(绿)；3500 两版均召回；"
                           f"far_hi={far_tok_hi} near_hi={near_hi} 预算守恒")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ C3-3 far 不变性
def c3_3_far_invariant():
    name = "C3-3 far 区选中集合新旧逐位相同（乘正 scale 排序不变）"
    try:
        S, k, q = build_scene()
        bs, sink_tok = 64, 128
        mid_len = S - sink_tok - 128
        near_len_dyn = max(bs, int(0.25 * mid_len))
        far_tok_hi = max(2, (S - near_len_dyn) // bs) * bs
        _, mask_new, _ = run_mask_idx(IDX_NEW, make_args(**C3_CFG), k, q)
        _, mask_old, _ = run_mask_idx(IDX_OLD, make_args(**C3_CFG), k, q)
        far_new = mask_new[0, 0][..., sink_tok:far_tok_hi]
        far_old = mask_old[0, 0][..., sink_tok:far_tok_hi]
        assert far_new.shape == far_old.shape
        assert torch.equal(far_new, far_old), \
            f"far 区选中集合漂移：per-head 差异 {int((far_new != far_old).sum())} 位"
        n_far = int(far_new.sum())
        report(name, True, f"far 区 [128,{far_tok_hi}) 逐位一致，选中 {n_far} 位")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ C3-4 clear() 重置
def c3_4_clear_reset():
    name = "C3-4 clear() 重置 _last_q（跨请求 stale q 防护）"
    try:
        S, k, q = build_scene()
        idx, _, _ = run_mask_idx(IDX_NEW, make_args(**C3_CFG), k, q)
        assert idx._last_q is not None, "前提：跑过一轮后 _last_q 非空"
        idx.clear()
        assert getattr(idx, "_last_q", "unset") is None, "clear() 后 _last_q 应为 None"
        report(name, True)
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ C3-5 kmeans 路径冒烟 + 预算守恒
def c3_5_kmeans_path():
    name = "C3-5 ccluster_kmeans 路径冒烟（近侧 kmeans 同修复面）"
    try:
        S, k, q = build_scene()
        cfg = dict(C3_CFG)
        cfg.update(tli_far_select="cluster", tli_near_select="cluster")
        idx, mask, _ = run_mask_idx(IDX_NEW, make_args(**cfg), k, q)
        m = mask[0, 0]
        assert mask.dtype == torch.bool and mask.shape[-1] == S
        assert bool(m[..., :128].all()), "sink 区未全选"
        assert bool(m[..., S - 128:].all()), "swa 区未全选"
        # 缩放口径下 3980（q·k=2.0·scale=0.177）全场最高 → kmeans 簇代表分路径也应召回
        assert bool(m[:, 3980].any()), "kmeans 路径下回退段高分 token 未召回"
        K2_mid = 1024 - 128 - 128
        n_mid = int(m[0, 128:S - 128].sum())
        assert n_mid == K2_mid, f"mid 选中 {n_mid} != {K2_mid}"
        report(name, True, f"kmeans 路径 3980 召回，mid={n_mid}")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


if __name__ == "__main__":
    c3_1_cache_scaled()
    c3_2_red_green()
    c3_3_far_invariant()
    c3_4_clear_reset()
    c3_5_kmeans_path()
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n===== C3 专项回归：{len(RESULTS) - n_fail}/{len(RESULTS)} PASS =====")
    sys.exit(1 if n_fail else 0)
