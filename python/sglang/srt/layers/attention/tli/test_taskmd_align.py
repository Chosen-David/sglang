"""E112（#149 第一阶段）TASK.md 语义对拍单测（CPU 可跑，PYTHONPATH 直跑）。

对拍两侧：
  - sglang 侧：TLIIndexer._select_taskmd（has_abg=True 语义分支，
    python/sglang/srt/layers/attention/tli/indexer.py）
  - 权威侧：two-level-attention sparse_attn TLIIndexer.prepare_mask
    （分支 two-level-indexer，含 E109a 五个修复 commit——只读参照）

语义锚点（/home/wangyuanshuo02/sglang/TASK.md，只读权威）：
  L36-45  method 注册表：mavg=(minmax,avg)/cavg/aavg/ccluster/mminmax
  L151-172 α/β/γ：near_L=α·mid、near 块预算=β·k1、far 块预算=k1−near
          （无保底）、near_token=near·bs·γ、far_token=K2_mid−near_token
          （无保底，E109a 严格化）
  L231    ab(0,0)=全部走 far_method（单池点）

对拍口径：块对齐 S（64 的倍数）+ t=S−1 + fp32 —— 权威以 padded 宽
（kt·bs）为工作宽，块对齐下 padded == real-S，两侧几何一致；逐
kv-head 比较**token 位置集合**（sglang 侧哨兵 pad 槽位剔除后）。

已知口径差距（详见 _select_taskmd docstring 的差距清单）：
  1. L1 组内聚合：sglang sum vs 权威 mean——严格成比例 → topk 排名
     等价（exp/trace probe Q2c 实测 indices 一致）；
  2. L2 细筛：sglang 32 维紧凑收缩 vs 权威 128 维零 pad 物化 repeat
     ——~1e-7 级浮点漂移（probe Q3）；softmax 归约宽度 T vs Tc 同级
     漂移。随机数据下 topk 边界 gap >> 1e-7，集合仍逐位一致（本单测
     断言 exact set 相等；若未来出现偶发翻转，先查 tie 而非语义）。

用法：
  PYTHONPATH=<repo>/python python3 test_taskmd_align.py
  （权威侧路径 /home/wangyuanshuo02/sglang/two-level-attention 需存在）
"""

import os
import sys
import types

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
# 上面得到 python/sglang/srt/layers/attention/tli → 上溯 5 级 = repo/python
sys.path.insert(0, os.path.join(_REPO, ""))  # repo/python（sglang 包根）
sys.path.insert(0, "/home/wangyuanshuo02/sglang/two-level-attention")  # 权威仓

from sglang.srt.layers.attention.tli.config import TLIProfile  # noqa: E402
from sglang.srt.layers.attention.tli.indexer import TLIIndexer  # noqa: E402
from sparse_attn.indexer.tli_indexer import TLIIndexer as RefIndexer  # noqa: E402

# ---- 公共配置（两侧对齐；TLIProfile 默认值即权威默认）----
# block_size=64 / k1=128 / token_budget=1024 / sliding_window=128 /
# sink_blocks=2 / near_len=2048 / L2 delta=16（权威 cmp_ratio=4 → 32 维）
ENV_KEYS = (
    "SGLANG_TLI_ALPHA",
    "SGLANG_TLI_BETA",
    "SGLANG_TLI_GAMMA",
    "SGLANG_TLI_FAR_METHOD",
    "SGLANG_TLI_NEAR_METHOD",
    "SGLANG_TLI_SUBSPACE",
)


def _set_env(alpha, beta, gamma, far, near, subspace):
    for k in ENV_KEYS:
        os.environ.pop(k, None)
    if alpha is not None:
        os.environ["SGLANG_TLI_ALPHA"] = str(alpha)
    if beta is not None:
        os.environ["SGLANG_TLI_BETA"] = str(beta)
    if gamma is not None:
        os.environ["SGLANG_TLI_GAMMA"] = str(gamma)
    if far is not None:
        os.environ["SGLANG_TLI_FAR_METHOD"] = far
    if near is not None:
        os.environ["SGLANG_TLI_NEAR_METHOD"] = near
    if subspace is not None:
        os.environ["SGLANG_TLI_SUBSPACE"] = subspace


def sglang_sel(k, q, *, alpha, beta, gamma, far, near, subspace, skip=False):
    """sglang 侧：环境变量注入 → TLIProfile → build_block_index → select。

    返回 list[Hkv] 的 token 位置集合（哨兵 S 槽位剔除）。
    """
    _set_env(alpha, beta, gamma, far, near, subspace)
    prof = TLIProfile()
    assert prof.has_abg, "对拍用例必须显式设置 α/β/γ/method 进入 TASK.md 语义分支"
    ix = TLIIndexer(prof, head_dim=128)
    if skip:
        ix.skip_far = True
    index = ix.build_block_index(k)
    S = k.shape[0]
    sel = ix.select(index, q, S - 1)  # [Hkv, K2]（哨兵 = S）
    return [set(int(x) for x in sel[h][sel[h] < S]) for h in range(sel.shape[0])]


def ref_mask_sets(k, q, *, alpha, beta, gamma, far, near, subspace, skip=False):
    """权威侧：two-level-attention TLIIndexer.prepare_mask → mask 位置集合。"""
    a = types.SimpleNamespace(
        tia_block_size=64,
        tia_level1_topk=128,
        tia_level2_topk=1024,
        tia_level2_cmp_ratio=4,  # L2 细筛 32 维 = sglang delta=16 同口径
        tia_enable_async_topk=False,
        tli_alpha=alpha if alpha is not None else 0.0,
        tli_beta=beta if beta is not None else 0.0,
        tli_gamma=gamma if gamma is not None else 1.0,
        tli_far_method=far if far is not None else "minmax",
        tli_near_method=near if near is not None else "avg",
        tli_enable_kmeans=False,
        tli_enable_layer_skip=False,
        tli_subspace=subspace if subspace is not None else "full",
        tli_enable_subspace=True,
    )
    ix = RefIndexer(a)
    ix.layer_idx = 5
    if skip:
        ix.skip_far = True
    T, Hkv, D = k.shape
    H = q.shape[1]
    mask, _ = ix.prepare_mask(
        q.view(1, 1, H, D),
        torch.tensor([T - 1]),
        k.view(1, T, Hkv, D),
        torch.tensor([0, T]),
        D**-0.5,
    )
    m = mask[0, 0]  # [Hkv, T]
    return [set(int(x) for x in torch.nonzero(m[h]).squeeze(1)) for h in range(m.shape[0])]


def run_case(name, *, alpha, beta, gamma, far, near, subspace="full", skip=False):
    sets_sg = sglang_sel(K, Q, alpha=alpha, beta=beta, gamma=gamma,
                         far=far, near=near, subspace=subspace, skip=skip)
    sets_ref = ref_mask_sets(K, Q, alpha=alpha, beta=beta, gamma=gamma,
                             far=far, near=near, subspace=subspace, skip=skip)
    assert len(sets_sg) == len(sets_ref), f"{name}: head 数不一致"
    bad = [h for h in range(len(sets_sg)) if sets_sg[h] != sets_ref[h]]
    n_sg = sum(len(s) for s in sets_sg)
    n_ref = sum(len(s) for s in sets_ref)
    ok = not bad
    print(f"  {name}: {'PASS' if ok else 'FAIL'}（选中 {n_sg} vs {n_ref} token·head）"
          + (f"  差异 head={bad[:5]}" if bad else ""))
    return ok


# ---- 数据（块对齐 S=8192；Hkv=8 G=4 H=32 D=128，fp32）----
torch.manual_seed(0)
S, Hkv, G, D = 8192, 8, 4, 128
H = Hkv * G
K = torch.randn(S, Hkv, D, dtype=torch.float32)
Q = torch.randn(1, H, D, dtype=torch.float32)

if __name__ == "__main__":
    print("== E112 TASK.md 语义对拍（sglang _select_taskmd vs 权威 prepare_mask）==")
    results = []

    # T1 (0,0) 单池 minmax（mminmax 单池点；has_abg 由 method env 触发）
    results.append(run_case("T1 单池(0,0) mminmax",
                            alpha=None, beta=None, gamma=None, far="minmax", near="minmax"))
    # T2 (0,0) 单池 far=avg（aavg 单池点——E109a 修复语义）
    results.append(run_case("T2 单池(0,0) far=avg",
                            alpha=None, beta=None, gamma=None, far="avg", near="avg"))
    # T3 分区 mminmax（minmax,minmax 交叉）
    results.append(run_case("T3 分区 mminmax α.125/β.25/γ.375",
                            alpha=0.125, beta=0.25, gamma=0.375, far="minmax", near="minmax"))
    # T4 分区 mavg（E98 冠军组合形态 minmax,avg）
    results.append(run_case("T4 分区 mavg α.125/β.375/γ.625",
                            alpha=0.125, beta=0.375, gamma=0.625, far="minmax", near="avg"))
    # T5 分区 aavg（avg,avg）
    results.append(run_case("T5 分区 aavg α.125/β.25/γ.375",
                            alpha=0.125, beta=0.25, gamma=0.375, far="avg", near="avg"))
    # T6 高 γ far_budget=0 边界（γ=1 → nt_near 吃满 K2_mid → far 池空）
    results.append(run_case("T6 高γ far_budget=0 α.125/β.375/γ1.0",
                            alpha=0.125, beta=0.375, gamma=1.0, far="minmax", near="avg"))
    # T7 tail 子空间复现（B'/E71 旧口径臂）
    results.append(run_case("T7 tail 子空间 aavg α.125/β.25/γ.375",
                            alpha=0.125, beta=0.25, gamma=0.375, far="avg", near="avg",
                            subspace="tail"))
    # T8 skip_far（分区 α/β>0 + 跳 far → 权威 L812：整体退化单池）
    results.append(run_case("T8 skip_far 分区α.125/β.25/γ.375",
                            alpha=0.125, beta=0.25, gamma=0.375, far="minmax", near="avg",
                            skip=True))
    # T9 skip_far 单池（α=0 几何 = near_len 口径）
    results.append(run_case("T9 skip_far 单池(0,0) far=avg",
                            alpha=None, beta=None, gamma=None, far="avg", near="avg",
                            skip=True))

    # T10 B' 回退模式回归保护：has_abg=False 时不得进入 _select_taskmd
    for kk in ENV_KEYS:
        os.environ.pop(kk, None)
    prof = TLIProfile()
    assert not prof.has_abg, "清空 env 后必须回退 B' 模式"
    ix = TLIIndexer(prof, head_dim=128)
    _orig = TLIIndexer._select_taskmd

    def _must_not_run(*a, **kw):
        raise AssertionError("has_abg=False 时不应进入 _select_taskmd")

    TLIIndexer._select_taskmd = _must_not_run
    try:
        index = ix.build_block_index(K)
        sel_b = ix.select(index, Q, S - 1)  # B' eager 路径
        ok = sel_b.shape == (Hkv, 1024) and int((sel_b < S).all()) == 1
    finally:
        TLIIndexer._select_taskmd = _orig
    print(f"  T10 B' 回退模式（不进 _select_taskmd，宽度 K2=1024 全有效）: "
          f"{'PASS' if ok else 'FAIL'}")
    results.append(ok)

    print(f"\n总判决: {'ALL PASS' if all(results) else '存在 FAIL'}（{sum(results)}/{len(results)}）")
    sys.exit(0 if all(results) else 1)
