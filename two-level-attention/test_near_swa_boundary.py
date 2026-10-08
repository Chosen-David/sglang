#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""near-SWA 边界 + C-1 控制流回归门禁（CPU only，干净检出可复现，0 SKIP）。

【10-09 复审修复（GPT 复审 2026-10-09_0131，TL-TEST-SKIP-PASS-002 /
TL-TEST-FAR-EMPTY-COVERAGE-003）】
  - 门禁不再依赖未提交 /tmp/near_fix_v2：红绿历史对照（N1/N3/N4/N5）拆到
    非门禁脚本 test_near_swa_redgreen.py；本文件只断言检入主树实现。
  - 结果状态显式 PASS/FAIL，汇总输出 executed_pass/executed_total；
    门禁无 SKIP 路径——任何未执行的检查直接计 FAIL（不再假绿）。
  - N2 更名：它覆盖的是 α=1 near-max 分区回归（C-1 半程 + 边界缺陷
    哨兵），far 区实为 [128,256) 宽 128 非空——不再声称 far-empty。
  - 新增 N6：短序列 far-empty 真覆盖（far_tok_hi <= far_tok_lo 路径）。

背景（两缺陷）：
  N1（用户 2026-10-08 报告，独立核实属实）：e64 分区臂 near 左界从序列末尾
  （kt*bs）往前推 near_len_dyn，而 TASK.md L137 权威定义 near_L = α·mid_L
  应从 swa 起点（S-swa_tok）往前推 → near 区实际宽 = α·mid_len - swa_tok
  （系统性少 128），far 区多 128；α=1 时 far 区仍留 128（N2 哨兵断言的就是
  这个开放缺陷）。

  C-1（pool_starvation_audit_20261008 §3/§5-1，P1，已修 bb8c6712c）：
  far token 池空（near_blks==sink_blocks）时 compute_mask 的
  `if far_tok_hi > far_tok_lo:` 曾把 near 池选择 + sink/swa 强制 + return
  整体旁路 → 静默落单池兜底。N2（长序列 near-max）与 N6（短序列
  far-empty）共同回归保护该修复。

用法：python3 test_near_swa_boundary.py   （two-level-attention/ 下）
"""
import importlib
import os
import sys
import types

sys.dont_write_bytecode = True

import torch

torch.set_num_threads(1)

REPO = os.path.dirname(os.path.abspath(__file__))

# 状态显式化（TL-TEST-SKIP-PASS-002）：PASS/FAIL 字符串，不再用 ok=bool
# 把 SKIP 混进 PASS 分子。门禁文件无 SKIP——需要 SKIP 的红绿对照在
# 非门禁脚本 test_near_swa_redgreen.py 中。
RESULTS = []   # (name, status, detail)  status ∈ {"PASS", "FAIL"}


def report(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    RESULTS.append((name, status, detail))
    print(f"{status}  {name}" + (f"  -- {detail}" if detail else ""))


def load_sparse_attn(name, root):
    mod = types.ModuleType(name)
    mod.__path__ = [os.path.join(root, "sparse_attn")]
    sys.modules[name] = mod
    return importlib.import_module(f"{name}.indexer")


IDX_OLD = load_sparse_attn("sparse_attn_nb_old", REPO)      # 主树 = C-1 已修（bb8c6712c）


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


# ================================================================ N2 α=1 near-max 分区回归
def n2_alpha_one_near_max():
    name = "N2 α=1 near-max 分区回归：C-1 半程（主树 sink/swa 强制 + mid=768）+ 边界缺陷哨兵"
    # 【10-09 更名（GPT 复审 TL-TEST-FAR-EMPTY-COVERAGE-003）】本测试的
    # far 区实为 [128,256) 宽 128 非空（边界缺陷开放中 near 从序列尾前推
    # 少 128）——它覆盖的是 far 非空 + near 最大下的分区预算回归，不是
    # far-empty。far-empty 真覆盖见 N6（短序列）。
    try:
        S = 128 + 4096 + 128
        k, q = gen_kq(S)
        args = make_args(**CCLUSTER_CFG, tli_alpha=1.0, tli_beta=0.25, tli_gamma=0.5)
        idx_old, mask_old = run_mask(IDX_OLD, args, k, q)
        fh_old = int(idx_old._km_far_hi_cached)
        # 边界缺陷哨兵：α=1 仍留 128-token far 区。合入 near_fix 后此值
        # 变 192 且 far 池（消费侧 far_tok_hi=128=far_lo）转空——届时本
        # 断言翻转，并按复审建议 3 在 N6 补 α=1 长序列的 near 配额深断言。
        assert fh_old == 256, f"主树 far_hi 应 256（边界缺陷开放中哨兵），实际 {fh_old}"
        # C-1 回归（主树直接断言）：分区路径下 sink/swa 必强制、mid 走
        # 分区预算 768（非单池兜底 896）
        assert mask_old.dtype == torch.bool and mask_old.shape[-1] == S
        assert bool(mask_old[..., :128].all()) and bool(mask_old[..., S - 128:].all()), \
            "主树 sink/swa 强制区缺失——C-1 回归失败（bb8c6712c 被破坏）"
        mm = mask_old[0, 0]
        n_mid = int(mm[0, 128:S - 128].sum())
        assert n_mid == 768, f"主树 α=1 mid 应 768（C-1 分区预算），实际 {n_mid}"
        report(name, True, f"far_hi=256(哨兵)、sink/swa 强制齐、mid={n_mid}")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ N6 短序列 far-empty（C-1 空池路径）
def n6_short_seq_far_empty():
    name = "N6 短序列 far-empty：far_tok_hi<=far_tok_lo 空池控制流回归（K2<S 判别）"
    # far-empty 在当前主树（边界缺陷开放中）的自然可达路径 = 短序列：
    # mid_len < bs 时 near_len_dyn 被 max(bs,·) 下限抬到 64；kt<=3 时
    # near_blks = max(2, (192-64)//64) = 2 = sink_blocks → far_tok_hi =
    # 128 = far_tok_lo，far 池真空（sc_far.shape[-1]==0、i_f 空）。
    # S=192 下 sink[0,128) ∪ swa[64,192) = 全序列。
    # 判别设计（红绿可分）：tia_level2_topk=100 < S——C-1 修复路径经
    # 强制区+分区 return 出全 True；若 C-1 guard 回归（整段旁路落
    # 单池 topk(p, K2)）只出 100 个 True → mask.all() 必失败。
    # 【near_fix 合入后 TODO（复审建议 3 完整形态）】α=1/S=4352 届时自然
    # 触达 far-empty（far_tok_hi=128）且 near 选择区 [128,4224) 非空——
    # 补 near 配额/sink/SWA/总预算深断言（当前缺陷开放中不可达）。
    try:
        S, bs, sink_blocks, swa_tok = 192, 64, 2, 128
        # 执行前先断言（公式重放，与 compute_mask L784-806 同源）：
        # 该配置确实落 far-empty
        kt = S // bs                                      # 3
        mid_len = max(0, kt * bs - sink_blocks * bs - swa_tok)   # 0
        near_len_dyn = max(bs, int(1.0 * mid_len))        # 64（max(bs,·) 下限）
        near_blks = max(sink_blocks, (kt * bs - near_len_dyn) // bs)  # 2
        far_tok_lo = sink_blocks * bs                     # 128
        far_tok_hi = min(near_blks * bs, kt * bs)         # 128
        assert far_tok_hi <= far_tok_lo, \
            f"公式重放 far 区非空 ({far_tok_lo},{far_tok_hi})——配置未触达 far-empty"
        far_width = far_tok_hi - far_tok_lo
        k, q = gen_kq(S)
        args = make_args(tli_enable_layer_skip=False, tli_far_method="minmax",
                         tli_near_method="avg", tli_alpha=1.0, tli_beta=0.25,
                         tli_gamma=0.5, tia_level2_topk=100)
        _, mask = run_mask(IDX_OLD, args, k, q)
        assert mask.dtype == torch.bool and mask.shape[-1] == S
        # C-1 修复路径：强制区覆盖全序列 → 全 True（判别：旁路单池只 100 True）
        n_true = int(mask[0, 0].sum())
        assert bool(mask.all()), (
            f"far-empty 路径 mask 非全 True（{n_true}/{S}）——C-1 空池控制流被"
            f"旁路（单池兜底只给 K2=100）或强制区缺失")
        assert bool(mask[..., :sink_blocks * bs].all()), "sink 强制区缺失"
        assert bool(mask[..., max(0, S - swa_tok):].all()), "swa 强制区缺失"
        report(name, True, f"far 区宽 {far_width}（空）、i_f/sc_near 空池不崩、"
                           f"强制区全 True（K2=100<S 判别过）")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


if __name__ == "__main__":
    n2_alpha_one_near_max()
    n6_short_seq_far_empty()
    n_pass = sum(1 for _, s, _ in RESULTS if s == "PASS")
    n_fail = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    assert n_pass + n_fail == len(RESULTS) == 2, "门禁不允许 SKIP/漏项"
    print(f"\n===== near-SWA/C-1 门禁：executed {n_pass}/{n_pass + n_fail} PASS"
          f"（0 SKIP；红绿对照见非门禁 test_near_swa_redgreen.py）=====")
    sys.exit(1 if n_fail else 0)
