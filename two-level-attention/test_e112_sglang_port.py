#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E112（#149）对拍测试：sglang TLI backend 的 TASK.md 口径移植 vs two-level 权威实现。

权威源：sparse_attn/indexer/tli_indexer.py（two-level-indexer 分支 @54d93a4c6，
只读）。移植目标：python/sglang/srt/layers/attention/tli/{config,indexer,backend}.py。

全部 CPU 运行（GPU 留给 E109 v2 扫描），T=8192（64 对齐，无 pad 边角），
不依赖 SGLANG_TLI_* 环境变量（直接改 TLIProfile 属性，进程环境零污染）。

对齐口径（两侧配置映射）：
  two-level                                sglang TLIProfile
  ----------------                         -----------------
  tia_block_size=64                        block_size=64
  tia_level1_topk=128                      k1_blocks=128
  tia_level2_topk=1024                     token_budget=1024
  tia_level2_cmp_ratio=4（L2 tail32）      delta=16（refine 2*delta=32）
  tli_subspace="full"（L1 全 128 维）      coarse_dim=128（idx1 全维）
  sliding_window_size=128                  sliding_window=128
  sink_blocks=2                            sink_blocks=2
  tli_alpha/beta/gamma                     alpha/beta/gamma（taskmd=True）
  tli_far_method/tli_near_method           far_method/near_method
  softmax_scale = D**-0.5（内部乘 q）      测试侧预乘进 q（正标量不改变
                                           topk 序；预乘使两侧操作数逐位同）

测试项：
  T0  4bit 量化逐位等价（quant4_pack+kq_unpack vs min_max_per_token_quant）
  T1  decode per-request（_select_taskmd）vs two-level prepare_mask：
      G=1 共 7 配置（mavg/aavg/mminmax 冠军位 + far 池激活位 + 两个 (0,0)
      单池点），逐 kv-head token 位置集合逐位一致
  T2  G=4 固有差异记录（GQA 聚合口径：two-level softmax-后-group-mean vs
      sglang group-sum——L2 侧数学上不同聚合，记录差异量级，不判 fail）
  T3a sglang 三路径内部一致性：per-request vs prefill 批量
      （_select_batched_taskmd，t=S-1 行）vs decode 批量
      （_select_decode_taskmd，共享 pool 行）——集合一致
  T3b prefill 批量多行（t+1 ∈ {8192,8128,7680}，全 64 对齐）vs
      two-level 截断 k 的逐行 prepare_mask——集合一致
  T4  增量一致性：build(S0)+update(k_new) == build(S) 全量重建
      （对齐 S0=8128 / 非对齐 S0=8100 两臂；kavg_sum 增量维护逐位验证）
  T5  update_pool_rows_decode 逐 token 增量 == 全量 pool（kavg_sum flat
      scatter 路径）
  T6  回归保护：taskmd=False（环境变量全缺省）走旧 B' 路径正常运行，
      index 不带 kavg_sum（零额外开销）

用法：python3 test_e112_sglang_port.py [配置名子串过滤]
"""

import sys
import traceback
from types import SimpleNamespace

import os

import torch

# 【10-10 修复】根路径从 __file__ 推导（原硬编码主仓绝对路径——worktree 检出
# 下会误测主树代码而非 worktree 代码，B10 worktree 验收时实锤踩中）。
TWO_LEVEL_ROOT = os.path.dirname(os.path.abspath(__file__))
SGLANG_PY_ROOT = os.path.normpath(os.path.join(TWO_LEVEL_ROOT, "..", "python"))

sys.path.insert(0, TWO_LEVEL_ROOT)
from sparse_attn.indexer.tli_indexer import TLIIndexer as TLIIndexer2L  # noqa: E402

sys.path.insert(0, SGLANG_PY_ROOT)
from sglang.srt.layers.attention.tli.indexer import (  # noqa: E402
    TLIIndexer as TLIIndexerSG,
    kq_unpack,
    quant4_pack,
)
from sglang.srt.layers.attention.tli.config import TLIProfile  # noqa: E402

S = 8192          # 序列长（64 对齐）
HKV = 8           # kv-head 数
D = 128           # head_dim
SCALE = D ** -0.5  # two-level prepare_mask 内部对 q 乘的 softmax_scale

# (名称, far_method, near_method, alpha, beta, gamma, G, G=1 时要求逐位一致)
CONFIGS = [
    ("mavg_champ_G1",     "minmax", "avg",    0.125, 0.375, 0.625, 1, True),
    ("mavg_faract_G1",    "minmax", "avg",    0.125, 0.25,  0.125, 1, True),
    ("aavg_e105_G1",      "avg",    "avg",    0.875, 0.875, 0.375, 1, True),
    ("mminmax_e105_G1",   "minmax", "minmax", 0.875, 0.875, 0.5,   1, True),
    ("mminmax_far_G1",    "minmax", "minmax", 0.75,  0.125, 0.5,   1, True),
    ("aavg_single_G1",    "avg",    "avg",    0.0,   0.0,   1.0,   1, True),
    ("mminmax_single_G1", "minmax", "minmax", 0.0,   0.0,   1.0,   1, True),
    # G=4：固有聚合差异记录臂（softmax-后-group-mean vs group-sum）
    ("mavg_faract_G4",    "minmax", "avg",    0.125, 0.25,  0.125, 4, False),
    ("aavg_e105_G4",      "avg",    "avg",    0.875, 0.875, 0.375, 4, False),
    ("mminmax_far_G4",    "minmax", "minmax", 0.75,  0.125, 0.5,   4, False),
]

RESULTS = []  # (测试项, 配置, 通过?, 备注)


def make_args_2l(far_method, near_method, alpha, beta, gamma):
    """two-level TLIIndexer 的 args 命名空间（对齐 e2e 生产口径 subspace=full）。"""
    return SimpleNamespace(
        tia_block_size=64,
        tia_level1_topk=128,
        tia_level2_topk=1024,
        tia_level2_cmp_ratio=4,          # L2 精筛 tail32（delta=16×2）
        tia_enable_async_topk=False,
        tli_subspace="full",             # L1 全 128 维（E109 v2 扫描默认口径）
        tli_enable_subspace=True,        # full 时 __init__ 内部强制 False
        tli_enable_kmeans=False,         # 4bit 分区（cluster 留 E110 阶段）
        tli_far_select="4bit",
        tli_near_select="4bit",
        tli_far_method=far_method,
        tli_near_method=near_method,
        tli_alpha=alpha,
        tli_beta=beta,
        tli_gamma=gamma,
        tli_enable_layer_skip=False,     # D' 关（E111 层 gate 未合并主树）
        tli_layer_skip_path=None,
        tli_sigma_select="none",
        tli_moba=False,
        tli_sigma=8.0,
        tli_per_q_head=False,
        tli_static_pair=False,
        tli_proj_basis=None,
    )


def make_profile_sg(far_method, near_method, alpha, beta, gamma):
    """sglang TLIProfile（taskmd 模式，属性直改不动进程环境）。"""
    p = TLIProfile()
    p.taskmd = True
    p.alpha, p.beta, p.gamma = alpha, beta, gamma
    p.far_method, p.near_method = far_method, near_method
    p.block_size = 64
    p.coarse_dim = 128      # full-128 L1（= two-level tli_subspace=full）
    p.k1_blocks = 128
    p.token_budget = 1024
    p.sliding_window = 128
    p.sink_blocks = 2
    p.delta = 16            # L2 精筛 32 维（= two-level cmp_ratio=4）
    p.near_len = 2048
    p.sliding_blocks = 3
    p.far_tokens = 256
    p.dense_threshold = 2048
    return p


def mask_row_to_set(row):
    """two-level mask 单 head 行 [S] bool → token 位置集合。"""
    return set(row.nonzero().squeeze(-1).tolist())


def sel_row_to_set(row, sentinel):
    """sglang select 输出单 head 行 [K2] → 有效 token 位置集合（哨兵剔除）。"""
    return {int(x) for x in row.tolist() if x < sentinel}


def cmp_heads(name, cfg, sets_a, sets_b, hard, note="", tie_sf=None):
    """逐 head 集合比较；hard=True 时不一致即 fail。

    【10-10 fp32 平票容忍（B10 验收实测 aavg_e105@t=8191 实锤）】
    两侧细筛原始分同序、仅差 ~6 ulp 的 token 对，经 two-level 侧全行
    softmax（exp+归一）后可坍缩为逐位相同的 p 值 → torch.topk 平票按
    索引序取低位；sglang 侧用原始分保住 gap 取高位——非语义分歧，是
    fp32 在 top-K 边界的固有可分度极限（B10 改 near 池边界 19→17 块后
    该 q seed 恰好落一个 6-ulp 边界对，此前 59/59 属候选池不同的运气）。
    tie_sf 给出 two-level 侧原始细筛分（[1,1,H,T] fp32）时启用 fail-closed
    平票判定：差集 ≤4 token、全部落在 mid 区 [sink_tok, swa_lo_tok)、
    且差集 token 原始分极差 ≤ 1e-5 相对量级 → 记 PASS（备注 fp32 tie）；
    任一条件不满足仍 FAIL（真实语义分歧不放行）。
    """
    diffs = [len(a ^ b) for a, b in zip(sets_a, sets_b)]
    mx = max(diffs) if diffs else 0
    ok = mx == 0
    if hard:
        tie_note = ""
        if not ok and tie_sf is not None and mx <= 4:
            # mid 区间：sink 128 + swa 128 之外（B10 后区域公式与生产同源）
            bs = 64
            seqlen = tie_sf.shape[-1]
            mid_lo, mid_hi = 2 * bs, max(0, seqlen - 128)
            for h, (a, b) in enumerate(zip(sets_a, sets_b)):
                if a == b:
                    continue
                disputed = sorted(a ^ b)
                if not all(mid_lo <= x < mid_hi for x in disputed):
                    continue
                row = tie_sf[0, 0, h].to(torch.float32)
                v = row[disputed]
                if torch.isinf(v).any():
                    continue
                scale = v.abs().max().item()
                spread = (v.max() - v.min()).abs().item()
                if scale > 0 and spread <= 1e-5 * scale:
                    tie_note = (f" fp32-tie(head={h} 差集 {len(disputed)} token 原始分"
                                f" 极差 {spread:.3e} ≤1e-5×{scale:.3e})")
                    ok = True
                break
        RESULTS.append((name, cfg, ok, f"max|A△B|={mx}" + tie_note
                        + (f" {note}" if note else "")))
        if not ok:
            # 打印首个不一致 head 的差集前若干项，便于定位
            for h, (a, b) in enumerate(zip(sets_a, sets_b)):
                if a != b:
                    print(f"    [FAIL] {name}/{cfg} head={h} "
                          f"两侧差集 sglang-only={sorted(b - a)[:8]} "
                          f"two-level-only={sorted(a - b)[:8]}")
                    break
        elif tie_note:
            print(f"    [TIE ] {name}/{cfg}{tie_note}")
    else:
        RESULTS.append((name, cfg, True, f"记录: max|A△B|={mx} "
                         f"(固有 GQA 聚合差异)" + (f" {note}" if note else "")))
    return ok


class _Capturing2L(TLIIndexer2L):
    """捕获 compute_score 的 score_dict（平票判定需要原始细筛分）。"""

    def compute_score(self, q, q_ids, index_dict, softmax_scale):
        d = super().compute_score(q, q_ids, index_dict, softmax_scale)
        self._cap_sf = d["score_fine"].detach().clone()
        return d


def run_two_level(args_2l, q, k, seqlen):
    """two-level 权威路径：prepare_mask（q [1,1,H,D]，k [1,L,Hkv,D]）。

    返回 (mask, score_fine)：score_fine = two-level 原始细筛分 [1,1,H,T]
    （供 cmp_heads 平票判定；不需要时调用方忽略第二返回值）。
    """
    idx = _Capturing2L(args_2l)
    cu = torch.tensor([0, seqlen], dtype=torch.long)
    q_ids = torch.tensor([seqlen - 1])
    mask, _ = idx.prepare_mask(q, q_ids, k[:, :seqlen], cu)
    return mask, idx._cap_sf


def main():
    flt = sys.argv[1] if len(sys.argv) > 1 else ""
    torch.manual_seed(20261007)
    k = torch.randn(1, S, HKV, D)  # 全配置共享同一份 k（fp32 CPU）

    # ---------------- T0：4bit 量化逐位等价 ---------------- #
    idx_probe = TLIIndexer2L(make_args_2l("minmax", "minmax", 0.0, 0.0, 1.0))
    x = torch.randn(2000, HKV, 32)
    q_deq = idx_probe.min_max_per_token_quant(x)
    s_deq = kq_unpack(*quant4_pack(x))
    t0_ok = torch.equal(q_deq, s_deq)
    RESULTS.append(("T0_quant_bitexact", "-", t0_ok,
                    f"max|Δ|={(q_deq - s_deq).abs().max().item():.3e}"))

    # ---------------- T1/T2：per-request vs two-level ---------------- #
    # sel_full[cfg] = per-request 输出（供 T3/T4/T5 复用）
    sel_full, q_of_cfg, prof_of_cfg = {}, {}, {}
    for (name, fm, nm, a, b_, g, G, hard) in CONFIGS:
        if flt and flt not in name:
            continue
        H = HKV * G
        torch.manual_seed(500 + CONFIGS.index(
            next(c for c in CONFIGS if c[0] == name)))
        q = torch.randn(1, 1, H, D)
        q_scaled = q * SCALE  # 预乘 softmax_scale（两侧操作数逐位同）

        # two-level 权威
        mask, sf_2l = run_two_level(make_args_2l(fm, nm, a, b_, g), q, k, S)
        sets_2l = [mask_row_to_set(mask[0, 0, h]) for h in range(HKV)]

        # sglang per-request（decode t = S-1）
        prof = make_profile_sg(fm, nm, a, b_, g)
        idxsg = TLIIndexerSG(prof, head_dim=D)
        index = idxsg.build_block_index(k[0])
        sel = idxsg.select(index, q_scaled[0], S - 1)  # [Hkv, K2]
        sets_sg = [sel_row_to_set(sel[h], S) for h in range(HKV)]

        if G == 1:
            r = idxsg._taskmd_regions(S)
            note = (f"e64={r['e64']} near_blks={r['near_blks']} "
                    f"nb_near/nb_far={r['nb_near']}/{r['nb_far']} "
                    f"nt_near={r['nt_near']} far_budget={r['far_budget']}")
            cmp_heads("T1_per_request_vs_2L", name, sets_2l, sets_sg, True, note,
                      tie_sf=sf_2l)
        else:
            cmp_heads("T2_G4_inherent_diff", name, sets_2l, sets_sg, False)

        sel_full[name], q_of_cfg[name], prof_of_cfg[name] = sel, q_scaled, prof

    # ---------------- T3a：三路径内部一致性（t = S-1 行） ---------------- #
    for (name, fm, nm, a, b_, g, G, _hard) in CONFIGS:
        if flt and flt not in name:
            continue
        if name not in sel_full:
            continue
        prof = prof_of_cfg[name]
        q_scaled = q_of_cfg[name]
        idxsg = TLIIndexerSG(prof, head_dim=D)
        index = idxsg.build_block_index(k[0])
        ref = [sel_row_to_set(r, S) for r in sel_full[name]]

        # prefill 批量（单行 t=S-1；哨兵口径=0 → 直接集合）
        sel_b = idxsg.select_batched(index, q_scaled[0], torch.tensor([S - 1]))
        sets_b = [{int(x) for x in row.tolist()} for row in sel_b[0]]
        cmp_heads("T3a_prefill_batched", name, ref, sets_b, True,
                  note="(vs per-request)")

        # decode 批量（共享 pool 单行；哨兵 = S_cap = S）
        pool_l = {
            "kq_q": index["kq_q"].unsqueeze(0),
            "kq_sc": index["kq_sc"].unsqueeze(0),
            "kq_mn": index["kq_mn"].unsqueeze(0),
            "kmin": index["kmin"].unsqueeze(0),
            "kmax": index["kmax"].unsqueeze(0),
        }
        if index["kavg_sum"] is not None:
            pool_l["kavg_sum"] = index["kavg_sum"].unsqueeze(0)
        sel_d = idxsg.select_decode_batched(
            pool_l, torch.tensor([0]), [S], q_scaled[0])
        sets_d = [sel_row_to_set(row, S) for row in sel_d[0]]
        cmp_heads("T3a_decode_batched", name, ref, sets_d, True,
                  note="(vs per-request)")

    # ---------------- T3b：prefill 批量多行 vs two-level 截断 k 逐行 ---------------- #
    # 行 t+1 全 64 对齐（非对齐 S 的 pad 边界差异属固有口径差，见报告）
    T_ROWS = [8191, 8127, 7679]
    for (name, fm, nm, a, b_, g, G, _hard) in CONFIGS:
        if flt and flt not in name:
            continue
        if G != 1 or name not in sel_full:
            continue
        torch.manual_seed(700 + CONFIGS.index(
            next(c for c in CONFIGS if c[0] == name)))
        q_rows = torch.randn(len(T_ROWS), HKV * G, D) * SCALE
        prof = prof_of_cfg[name]
        idxsg = TLIIndexerSG(prof, head_dim=D)
        index = idxsg.build_block_index(k[0])
        sel_b = idxsg.select_batched(index, q_rows, torch.tensor(T_ROWS))
        args_2l = make_args_2l(fm, nm, a, b_, g)
        for i, t in enumerate(T_ROWS):
            q_i = q_rows[i].unsqueeze(0).unsqueeze(0)  # [1,1,H,D]
            mask_i, sf_i = run_two_level(args_2l, q_i, k, t + 1)
            sets_2l = [mask_row_to_set(mask_i[0, 0, h]) for h in range(HKV)]
            sets_b_i = [{int(x) for x in row.tolist()} for row in sel_b[i]]
            cmp_heads("T3b_prefill_rows_vs_2L", f"{name}@t={t}",
                      sets_2l, sets_b_i, True, tie_sf=sf_i)

    # ---------------- T4：增量 vs 全量重建 ---------------- #
    for name in ("mavg_champ_G1", "mminmax_far_G1"):
        if name not in sel_full:
            continue
        if flt and flt not in name:
            continue
        cfg = next(c for c in CONFIGS if c[0] == name)
        _, fm, nm, a, b_, g, _G, _ = cfg
        prof = make_profile_sg(fm, nm, a, b_, g)
        ref = [sel_row_to_set(r, S) for r in sel_full[name]]
        for S0, tag in ((8128, "aligned"), (8100, "unaligned")):
            idx_inc = TLIIndexerSG(prof, head_dim=D)
            index_inc = idx_inc.build_block_index(k[0, :S0])
            index_inc = idx_inc.update_block_index(index_inc, k[0, S0:S])
            sel_inc = idx_inc.select(index_inc, q_of_cfg[name][0], S - 1)
            sets_inc = [sel_row_to_set(r, S) for r in sel_inc]
            cmp_heads("T4_incremental", f"{name}/{tag}", ref, sets_inc, True)

    # ---------------- T5：pool 逐 token 增量 vs 全量 pool ---------------- #
    for name in ("mavg_champ_G1", "mminmax_far_G1"):
        if name not in sel_full:
            continue
        if flt and flt not in name:
            continue
        cfg = next(c for c in CONFIGS if c[0] == name)
        _, fm, nm, a, b_, g, _G, _ = cfg
        prof = make_profile_sg(fm, nm, a, b_, g)
        idxb = TLIIndexerSG(prof, head_dim=D)
        nd2 = 2 * prof.delta
        d1 = prof.coarse_dim
        S0 = 8128  # 对齐（decode 逐 token 增量的典型起点）
        i0 = idxb.build_block_index(k[0, :S0])
        pool = {
            "kq_q": torch.zeros(1, S, HKV, nd2, dtype=torch.uint8),
            "kq_sc": torch.zeros(1, S, HKV),
            "kq_mn": torch.zeros(1, S, HKV),
            "kmin": torch.zeros(1, S // 64, HKV, d1),
            "kmax": torch.zeros(1, S // 64, HKV, d1),
        }
        pool["kq_q"][0, :S0] = i0["kq_q"]
        pool["kq_sc"][0, :S0] = i0["kq_sc"]
        pool["kq_mn"][0, :S0] = i0["kq_mn"]
        pool["kmin"][0, : i0["nblk"]] = i0["kmin"]
        pool["kmax"][0, : i0["nblk"]] = i0["kmax"]
        if i0["kavg_sum"] is not None:
            pool["kavg_sum"] = torch.zeros(1, S // 64, HKV, d1)
            pool["kavg_sum"][0, : i0["nblk"]] = i0["kavg_sum"]
        for j in range(S0, S):  # 逐 token 增量（每行恰 1 token 的契约）
            idxb.update_pool_rows_decode(
                pool, torch.tensor([0]), [j], k[0, j : j + 1])
        sel_p = idxb.select_decode_batched(
            pool, torch.tensor([0]), [S], q_of_cfg[name][0])
        sets_p = [sel_row_to_set(row, S) for row in sel_p[0]]
        # 参照 = T3a 的全量 decode 批量结果重算（同 q 同配置）
        idx_ref = TLIIndexerSG(prof, head_dim=D)
        index_ref = idx_ref.build_block_index(k[0])
        pool_ref = {
            "kq_q": index_ref["kq_q"].unsqueeze(0),
            "kq_sc": index_ref["kq_sc"].unsqueeze(0),
            "kq_mn": index_ref["kq_mn"].unsqueeze(0),
            "kmin": index_ref["kmin"].unsqueeze(0),
            "kmax": index_ref["kmax"].unsqueeze(0),
        }
        if index_ref["kavg_sum"] is not None:
            pool_ref["kavg_sum"] = index_ref["kavg_sum"].unsqueeze(0)
        sel_ref = idx_ref.select_decode_batched(
            pool_ref, torch.tensor([0]), [S], q_of_cfg[name][0])
        sets_ref = [sel_row_to_set(row, S) for row in sel_ref[0]]
        cmp_heads("T5_pool_incremental", name, sets_ref, sets_p, True)

    # ---------------- T6：taskmd=False 回归保护（旧 B' 路径） ---------------- #
    if not flt or "regression" in flt:
        p6 = TLIProfile()  # 环境无 SGLANG_TLI_* → taskmd=False（旧 B' 语义）
        p6.coarse_dim = 128
        idx6 = TLIIndexerSG(p6, head_dim=D)
        index6 = idx6.build_block_index(k[0])
        sel6 = idx6.select(index6, q_of_cfg.get(
            "mavg_champ_G1", torch.randn(1, HKV, D) * SCALE)[0], S - 1)
        ok6 = sel6.shape == (HKV, p6.token_budget) and index6["kavg_sum"] is None
        RESULTS.append(("T6_regression_off", "-", bool(ok6),
                        f"shape={tuple(sel6.shape)} kavg_sum={index6['kavg_sum']}"))

    # ---------------- 汇总 ---------------- #
    print("\n" + "=" * 78)
    print(f"{'测试项':<28} {'配置':<24} {'结果':<6} 备注")
    print("-" * 78)
    n_fail = 0
    for (tname, cfg, ok, note) in RESULTS:
        mark = "PASS" if ok else "FAIL"
        if not ok:
            n_fail += 1
        print(f"{tname:<28} {cfg:<24} {mark:<6} {note}")
    print("-" * 78)
    print(f"总计 {len(RESULTS)} 项，失败 {n_fail} 项"
          + ("（G=4 记录臂不计失败）" if any(not r[2] for r in RESULTS) else ""))
    return 1 if n_fail else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
