#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""near-SWA 边界修复红绿历史对照（CPU only，**非门禁**）。

【10-09 拆分（GPT 复审 2026-10-09_0131，TL-TEST-SKIP-PASS-002）】
本脚本依赖未提交的 /tmp/near_fix_v2 修复副本做主树（红）vs 修复副本（绿）
对拍——按复审建议 2，这类 /tmp 依赖从发布门禁（test_near_swa_boundary.py）
中移除，单独放非门禁脚本：
  - 门禁（干净检出可复现、0 SKIP）= test_near_swa_boundary.py；
  - 本脚本 = 开发期红绿对照：副本缺失时相关项 SKIP（显式状态，
    不计 PASS 分子，退出码 2=incomplete，不冒充通过）；near_fix 合入
    主树后本脚本使命结束，N1 的块对齐断言应并入门禁直接断言。

对照内容：
  N1 红绿对拍（块对齐反例：目标 near=2048，旧实得 1920/新 2048）
  N3 (0,0) 单池点 mask 新旧逐位相同（第一版统一公式会破坏此处）
  N4 老逻辑（无 αβ 的默认 flag）mask 新旧逐位相同
  N5 e64 分区臂预算守恒（分段：饱食 mid=K2_mid；饥饿 mid=池内截断）

【10-10 B10 已合入（e121）】N1 块对齐断言已按预案并入
test_near_swa_boundary.py 门禁（N1/N2/N6 三项）；本脚本探测到合入后
N1 翻转为「合入实现 vs 原型副本」一致性核对，N3/N4/N5 转为
「合入实现 vs 原型」逐位一致（历史红绿使命完成，保留作原型对照）。

用法：python3 test_near_swa_redgreen.py   （two-level-attention/ 下）
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
HAVE_FIX = os.path.isdir(os.path.join(FIX_ROOT, "sparse_attn"))

# 状态显式化（TL-TEST-SKIP-PASS-002 修复）：PASS/FAIL/SKIP 三态字符串。
# SKIP 不进 PASS 分子；汇总分别输出 executed_pass/executed_total 与
# skipped；退出码 1=有 FAIL、2=有 SKIP（incomplete）、0=全执行全过。
RESULTS = []   # (name, status, detail)


def report(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    RESULTS.append((name, status, detail))
    print(f"{status}  {name}" + (f"  -- {detail}" if detail else ""))


def report_skip(name, reason):
    RESULTS.append((name, "SKIP", reason))
    print(f"SKIP  {name}  -- {reason}")


def load_sparse_attn(name, root):
    mod = types.ModuleType(name)
    mod.__path__ = [os.path.join(root, "sparse_attn")]
    sys.modules[name] = mod
    return importlib.import_module(f"{name}.indexer")


IDX_OLD = load_sparse_attn("sparse_attn_rg_old", REPO)      # 主树（红侧）
IDX_NEW = load_sparse_attn("sparse_attn_rg_new", FIX_ROOT) if HAVE_FIX else None  # 修复副本（绿侧）


def _repo_has_b10():
    """【10-10 B10 合入（e121）】探测 REPO 实现是否已含 near-SWA 边界修复
    （N1 场景 far_hi：缺陷态 2304 / 修复态 2176）。合入后本红绿对照使命
    结束——N1 断言已按预案并入 test_near_swa_boundary.py 门禁；此处翻转为
    「合入实现 vs 原型副本」一致性核对（e64 臂两侧同绿）。"""
    S = 128 + 4096 + 128
    k, q = gen_kq(S)
    args = make_args(**CCLUSTER_CFG, tli_alpha=0.5, tli_beta=0.25, tli_gamma=0.5)
    idx, _ = run_mask(IDX_OLD, args, k, q)
    return int(idx._km_far_hi_cached) == 2176


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

MERGED = _repo_has_b10()


# ================================================================ N1 红绿对拍：块对齐反例
def n1_aligned_counterexample():
    name = "N1 红绿对拍（块对齐反例：目标 near=2048，旧实得 1920/新 2048）"
    if MERGED:
        # 【10-10 B10 合入后】红绿对照使命结束（断言已并入门禁 N1）。
        # 翻转语义：合入实现（REPO）vs 原型副本（/tmp/near_fix_v2）一致性。
        if not HAVE_FIX:
            report_skip(name, f"B10 已合入 REPO；原型副本 {FIX_ROOT} 缺失，"
                              f"一致性核对不可执行（门禁 N1 已直接断言）")
            return
        try:
            S = 128 + 4096 + 128
            k, q = gen_kq(S)
            args = make_args(**CCLUSTER_CFG, tli_alpha=0.5, tli_beta=0.25, tli_gamma=0.5)
            idx_repo, _ = run_mask(IDX_OLD, args, k, q)
            idx_proto, _ = run_mask(IDX_NEW, args, k, q)
            fh_repo = int(idx_repo._km_far_hi_cached)
            fh_proto = int(idx_proto._km_far_hi_cached)
            assert fh_repo == 2176 and fh_proto == 2176, \
                f"B10 合入态 far_hi 应 2176，repo={fh_repo} proto={fh_proto}"
            report(name, True, f"[已合入] 一致性核对：repo=proto=2176（near 宽 2048）"
                               f"——红绿历史断言已并入 test_near_swa_boundary.py N1")
        except AssertionError as e:
            report(name, False, str(e))
        except Exception as e:
            report(name, False, f"异常: {type(e).__name__}: {e}")
        return
    if not HAVE_FIX:
        report_skip(name, f"near_fix 副本 {FIX_ROOT} 不存在（非门禁红绿对照；"
                          f"合入主树后本断言应并入 test_near_swa_boundary.py）")
        return
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


# ================================================================ N3 (0,0) 单池回归逐位不变
def n3_single_pool_invariant():
    name = "N3 (0,0) 单池点 mask 新旧逐位相同（第一版统一公式会破坏此处）"
    if not HAVE_FIX:
        report_skip(name, f"near_fix 副本 {FIX_ROOT} 不存在（非门禁红绿对照）")
        return
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
    if not HAVE_FIX:
        report_skip(name, f"near_fix 副本 {FIX_ROOT} 不存在（非门禁红绿对照）")
        return
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
    if not HAVE_FIX:
        report_skip(name, f"near_fix 副本 {FIX_ROOT} 不存在（非门禁红绿对照）")
        return
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
    n3_single_pool_invariant()
    n4_legacy_invariant()
    n5_budget_conservation()
    n_pass = sum(1 for _, s, _ in RESULTS if s == "PASS")
    n_fail = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    n_skip = sum(1 for _, s, _ in RESULTS if s == "SKIP")
    executed = n_pass + n_fail
    print(f"\n===== near-SWA 红绿对照（非门禁）：executed {n_pass}/{executed} PASS"
          f"、{n_skip} SKIP（SKIP 不计 PASS；副本 {FIX_ROOT} "
          f"{'存在' if HAVE_FIX else '缺失'}）=====")
    # 退出码：1=FAIL；2=无 FAIL 但有 SKIP（incomplete，不冒充通过）；0=全执行全过
    sys.exit(1 if n_fail else (2 if n_skip else 0))
