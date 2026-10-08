#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""near-SWA 边界修复 + C-1 控制流洞修复验证测试（CPU only）——验证 /tmp/near_fix_v2。

缺陷一 N1（用户 2026-10-08 报告，独立核实属实）：
  e64 分区臂 near 左界从序列末尾（kt*bs）往前推 near_len_dyn，而 TASK.md L137
  权威定义 near_L = α·mid_L 应从 swa 起点（S-swa_tok）往前推 → near 区实际宽
  = α·mid_len - swa_tok（系统性少 128），far 区多 128；α=1 时 far 区仍留 128。
  块对齐反例：目标 near=2048，实际 1920。

缺陷二 C-1（pool_starvation_audit_20261008 §3/§5-1，P1）：
  far token 池空（near_blks==sink_blocks，α=1/短序列必然触发）时
  compute_mask 的 `if far_tok_hi > far_tok_lo:` 把 near 池选择 + sink/swa
  强制 + return 整体旁路 → 静默落单池兜底：sink 不保证、mid 超预算
  （896>768）、短序列全注意力。修复：guard 只保留 far 选择段，near 段无条件
  执行 + far 空时初始化空 i_f。

修复（分支化，仅动 e64 分区臂）：
  e64 分支 near_base = S - swa_tok；(0,0) 单池与老逻辑分支保持旧式逐位不变
  （(0,0) 的 far_hi = S - swa_tok 恰为正确单池语义——第一版"统一从 swa_lo 推"
  的公式会破坏单池 far 少 128，本测试 N3 防的就是这个）。

红-绿对照：主树（未修复，红侧）vs /tmp/near_fix_v2（绿侧，含 C-1 修复）。
用法：python3 test_near_swa_boundary.py
"""
import importlib
import os
import sys
import types

sys.dont_write_bytecode = True

import torch

torch.set_num_threads(1)

REPO = os.path.dirname(os.path.abspath(__file__))
FIX_ROOT = "/tmp/near_fix_v2"

RESULTS = []


def report(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))


def load_sparse_attn(name, root):
    mod = types.ModuleType(name)
    mod.__path__ = [os.path.join(root, "sparse_attn")]
    sys.modules[name] = mod
    return importlib.import_module(f"{name}.indexer")


IDX_OLD = load_sparse_attn("sparse_attn_nb_old", REPO)      # 主树 = 旧口径（红侧）
IDX_NEW = load_sparse_attn("sparse_attn_nb_new", FIX_ROOT)  # 修复副本（绿侧）


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


def run_mask(idxpkg, args, k, q):
    idx = idxpkg.TLIIndexer(args)
    idx.layer_idx = 3
    cu = torch.tensor([0, k.shape[1]])
    q_ids = torch.tensor([k.shape[1] - 1])
    mask, _ = idx.prepare_mask(q, q_ids, k, cu, q.shape[-1] ** -0.5)
    return idx, mask


def gen_kq(S, seed=31):
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(1, S, 2, 128, generator=g) * 0.5
    q = torch.randn(1, 1, 4, 128, generator=g) * 0.5
    return k, q


CCLUSTER_CFG = dict(
    tli_far_select="cluster", tli_near_select="cluster",
    tli_enable_kmeans=True, tli_enable_layer_skip=False,
    tli_far_method="minmax", tli_near_method="avg",
)


# ================================================================ N1 红绿对拍：块对齐反例
def n1_aligned_counterexample():
    name = "N1 红绿对拍（块对齐反例：目标 near=2048，旧实得 1920/新 2048）"
    try:
        # S = sink(128) + mid(4096) + swa(128)，全 64 块对齐；α=0.5 → near_len_dyn=2048
        S = 128 + 4096 + 128
        k, q = gen_kq(S)
        args = make_args(**CCLUSTER_CFG, tli_alpha=0.5, tli_beta=0.25, tli_gamma=0.5)
        idx_old, _ = run_mask(IDX_OLD, args, k, q)
        idx_new, _ = run_mask(IDX_NEW, args, k, q)
        # 建簇侧 far_hi（= near 左界）：旧 (S-2048)//64*64=2304，新 (S-128-2048)//64*64=2176
        fh_old = int(idx_old._km_far_hi_cached)
        fh_new = int(idx_new._km_far_hi_cached)
        assert fh_old == 2304, f"旧 far_hi 应 2304（near 宽 1920 = 目标-128），实际 {fh_old}"
        assert fh_new == 2176, f"新 far_hi 应 2176（near 宽 2048 = 目标），实际 {fh_new}"
        # near 簇覆盖区（消费侧同源）：宽 旧 1920 / 新 2048
        swa_lo = S - 128
        nw_old = swa_lo - fh_old
        nw_new = swa_lo - fh_new
        assert nw_old == 1920 and nw_new == 2048, \
            f"near 区宽 old={nw_old}（应 1920，缺陷实锤）new={nw_new}（应 2048）"
        report(name, True, f"far_hi: old={fh_old}(near 1920) new={fh_new}(near 2048)")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ N2 α=1 far 区空
def n2_alpha_one_empty_far():
    name = "N2 α=1 far 区空：建簇侧 far_hi=192 + C-1 修复后 sink 强制恢复"
    # 归因更正（pool_starvation_audit_20261008 §3）：此前该测试 FAIL 的真因
    # 不是建簇侧 far_hi 计算链（那部分 near_fix 一直是对的，fh_new=192），
    # 而是消费端控制流洞 C-1——far token 池空（near_blks==sink_blocks）时
    # `if far_tok_hi > far_tok_lo:` 把 near 池选择 + sink/swa 强制 + return
    # 整体旁路，静默落单池兜底：sink_all=False（靠 p 概率侥幸入选）、
    # mid=896 > K2_mid=768 预算失守。C-1 修复后本断言（sink 必强制）转绿，
    # 与 far_hi=192 断言构成完整红绿对。
    try:
        S = 128 + 4096 + 128
        k, q = gen_kq(S)
        args = make_args(**CCLUSTER_CFG, tli_alpha=1.0, tli_beta=0.25, tli_gamma=0.5)
        idx_old, mask_old = run_mask(IDX_OLD, args, k, q)
        idx_new, mask_new = run_mask(IDX_NEW, args, k, q)
        fh_old = int(idx_old._km_far_hi_cached)
        fh_new = int(idx_new._km_far_hi_cached)
        # 旧：far 区 [128, S-4096=256) 宽 128（α=1 仍留 far，用户报告实锤）
        assert fh_old == 256, f"旧 far_hi 应 256（far 区 128 = α=1 缺陷），实际 {fh_old}"
        # 新：near_len_dyn=mid_len=4096 → far_hi_blk=(4352-128-4096)//64=2=sink → far 区空
        assert fh_new == 192, \
            f"新 far_hi 建簇侧应 sink+1 块保底 192（far 区空），实际 {fh_new}"
        # C-1 修复后：far 空不再旁路 near 段与强制区——sink/swa 必强制，
        # mid 预算走分区路径（far 0 + near 池内截断）而非单池兜底
        assert mask_new.dtype == torch.bool and mask_new.shape[-1] == S
        assert bool(mask_new[..., :128].all()) and bool(mask_new[..., S - 128:].all()), \
            "sink/swa 强制区缺失——C-1 控制流洞未修复（far 空旁路了强制区）"
        # mid 预算恢复分区语义：mid 选中数 = min(K2_mid, near 区宽)（far 空时
        # k2_near=K2_mid=768，α=1 near 区宽=4096>768 饱食 → mid==768），
        # 而非 C-1 洞的单池兜底 896
        mm = mask_new[0, 0]
        n_mid = int(mm[0, 128:S - 128].sum())
        assert n_mid == 768, f"α=1 far 空时 mid 应 768（C-1 修复后分区预算），实际 {n_mid}"
        report(name, True, f"far_hi: old=256(far 留 128) new=192(空,建簇保底)；"
                           f"sink/swa 强制齐，mid={n_mid}（非洞口径 896）")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ N3 (0,0) 单池回归逐位不变
def n3_single_pool_invariant():
    name = "N3 (0,0) 单池点 mask 新旧逐位相同（第一版统一公式会破坏此处）"
    try:
        S = 4352
        k, q = gen_kq(S)
        args = make_args(**CCLUSTER_CFG, tli_alpha=0.0, tli_beta=0.0)
        _, m_old = run_mask(IDX_OLD, args, k, q)
        _, m_new = run_mask(IDX_NEW, args, k, q)
        assert torch.equal(m_old, m_new), \
            f"(0,0) 单池 mask 漂移 {int((m_old != m_new).sum())} 位 —— 修复破坏单池语义"
        report(name, True, "逐位一致（far 区仍 = 全 mid）")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ N4 老逻辑（α=0 非 cluster）回归
def n4_legacy_invariant():
    name = "N4 老逻辑（无 αβ 的默认 flag）mask 新旧逐位相同"
    try:
        S = 4352
        k, q = gen_kq(S)
        args = make_args(tli_enable_layer_skip=False)   # 默认 4bit/4bit，near_len=2048 老语义
        _, m_old = run_mask(IDX_OLD, args, k, q)
        _, m_new = run_mask(IDX_NEW, args, k, q)
        assert torch.equal(m_old, m_new), \
            f"老逻辑 mask 漂移 {int((m_old != m_new).sum())} 位"
        report(name, True)
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ N5 e64 分区臂预算守恒（分段断言）
def n5_budget_conservation():
    name = "N5 e64 分区臂预算守恒（分段：饱食 mid=K2_mid；饥饿 mid=池内截断）"
    # 断言口径更正（pool_starvation_audit_20261008 §2.4）：「mid 总选恒 = K2_mid」
    # 只在两池候选 ≥ 各自配额时成立，须改分段断言——
    #   饱食配置 ccluster a.5/b.25/g.5：near 区宽 2048 ≥ k2_near、far 池足
    #     → mid == K2_mid == 768；
    #   饥饿配置 mavg a.125/b.375/g.625：γ 悬崖（γ*=768/(0.375·128·64)≈0.25 <
    #     0.625）→ far_budget=0（γ 饱和合法坍缩）+ near 池候选 = α·mid_L = 512
    #     < k2_near=768 → 池内截断 mid == 512；该配置本身违反 TASK.md L145-147
    #     约束（near_budget_token = 48·64·0.625 = 1920 > near_L = 512），属
    #     L216-219「没有意义」配置集——饥饿是约束违反的正确语义体现，
    #     剩余预算不跨池回补 far（L154-155 池独立语义）。
    try:
        S = 4352
        k, q = gen_kq(S)
        K2_mid = 1024 - 128 - 128
        for tag, extra, mid_expect in [
            ("ccluster a.5/b.25/g.5", dict(tli_alpha=0.5, tli_beta=0.25, tli_gamma=0.5), K2_mid),
            ("mavg a.125/b.375/g.625", dict(tli_far_select="4bit", tli_near_select="4bit",
                                             tli_enable_kmeans=False, tli_alpha=0.125,
                                             tli_beta=0.375, tli_gamma=0.625), 512),
        ]:
            args = make_args(**CCLUSTER_CFG, **extra) if "ccluster" in tag else \
                make_args(tli_enable_layer_skip=False, tli_far_method="minmax",
                          tli_near_method="avg", **extra)
            _, m = run_mask(IDX_NEW, args, k, q)
            mm = m[0, 0]
            n_mid = int(mm[0, 128:S - 128].sum())
            assert n_mid == mid_expect, \
                f"{tag}: mid 选中 {n_mid} != {mid_expect}" + \
                ("（饱食守恒）" if mid_expect == K2_mid else "（饥饿池内截断）")
            assert bool(mm[..., :128].all()) and bool(mm[..., S - 128:].all()), \
                f"{tag}: sink/swa 强制区缺失"
        report(name, True, "ccluster mid=768 守恒；mavg mid=512 饥饿截断；sink/swa 强制齐")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


if __name__ == "__main__":
    n1_aligned_counterexample()
    n2_alpha_one_empty_far()
    n3_single_pool_invariant()
    n4_legacy_invariant()
    n5_budget_conservation()
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n===== near-SWA 边界修复验证：{len(RESULTS) - n_fail}/{len(RESULTS)} PASS =====")
    sys.exit(1 if n_fail else 0)
