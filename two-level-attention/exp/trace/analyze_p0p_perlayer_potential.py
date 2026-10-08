# -*- coding: utf-8 -*-
# P0' 逐层参数求解器潜力检查（预注册判决门）——
#   设计文档：sglang/research/docs/逐层参数求解器_数学原理与实验设计.md §5.1/§7。
#
# 问题：固定 dense trace，对 Qwen3-8B 的 8 代表层 {4,8,12,16,20,24,28,35}
#   （+层 1 对照），逐层枚举 (α,β,γ) 在**输出失真目标**（projected output MSE，
#   恒等式 y−ŷ=(1−mS)(μS̄−μS)）下的层内最优 Eℓ(c*_ℓ)，对比统一配置 G* 的
#   层内失真 Eℓ(G*)，量化逐层潜力 gap。
#   判决：gap 中位 >10% → "potential-high"（值得开发逐层求解器）
#         否则 → "potential-low"（负结果收档，走速度路线）。
#
# 协议要点（对齐文档 §5.1/§6.2/§6.3）：
#   - 预注册网格：α 8 值 × β 7 值 × γ 6 值 = 336 原始（文档 §6.2，含 γ=1.0）
#     + 单池点 (0,0,0) + G*（若不在集合中）。
#   - 闭式剔除 γ 悬崖坍塌臂（far_budget=0 且 far_frac>0.25，E109 实证必崩）
#     + 非法约束（near 细筛配额 nt_near > ℓ_near，文档 §2.2）——剔除候选在
#     JSON 留 "excluded_by"（逐样本 reason，可审计）；G* 豁免（参照必须评估）。
#   - mask 复算走 sparse_attn tli_indexer 真实链路（E109a 修复后 v2 代码）：
#     prepare_index → compute_score → compute_mask；α/β/γ 是 indexer 实例属性，
#     逐候选原地改参 → 同一 index_dict/score_dict 复用（分数与候选无关，先算一次）。
#   - 输出失真：p_full 全维 softmax（s=q·k/√d，GQA 组内 q-head 各自算分，
#     共享 kv-head 选择 mask——原语义）；文档 §5.1 归一化 e=‖Δo‖²/(aℓ²+ε)，
#     aℓ=该层 dense 输出 RMS（对候选固定）；文档内 query 平均 → 样本平均。
#   - E109a 纪律：aavg 回放断言 score_coarse_avg 非空（avg 分支真生效）+
#     与 minmax 变体 mask 差异检查；γ 悬崖闭式与 TLI_DEBUG [L1dbg]/[L2dbg]
#     单点对拍（闭式 vs 真实代码预算逐项一致）。
#
# 输入：trace 目录（collect_trace_lb_v.py 产出的含 "v" 的 layer pt）。
# 干跑（CPU，不占 GPU）：--dry-run 用随机合成 k/q/v（shape 对齐 Qwen3-8B：
#   H_q=32, H_kv=8, D=128, Dv=128, S=4096）+ 12 点缩域跑通全流程 +
#   恒等式自检 + TLI_DEBUG 对拍；全 PASS 才算完成。
#
# 正式运行（海选收官 GPU 空闲后，预计单卡 ~10 分钟量级）：
#   PYTHONPATH=... python3 analyze_p0p_perlayer_potential.py \
#       --trace /tmp/trace/qwen3-8b-v --device cuda:0
import argparse
import contextlib
import io
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from sparse_attn.indexer.tli_indexer import TLIIndexer  # noqa: E402

# ---- canonical 预算（r1c_canonical_config.json：与主表臂同口径）----
BS = 64            # tia_block_size
K1 = 128           # tia_level1_topk（L1 块预算）
K2 = 1024          # tia_level2_topk（最终 token 总预算）
CMP = 4            # tia_level2_cmp_ratio → L2 tail32
SINK_BLOCKS = 2    # sink = 前 2 块 = 128 token
SWA = 128          # sliding_window_size
SINK_TOK = SINK_BLOCKS * BS

# 预注册网格（文档 §6.2；γ 含 1.0 → 8×7×6=336 原始）
DEF_ALPHAS = [0.0625, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
DEF_BETAS = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
DEF_GAMMAS = [0.125, 0.25, 0.5, 0.625, 0.75, 1.0]
DEF_GSTAR = (0.125, 0.125, 0.375)     # aavg 海选现领跑臂（far=near=avg）
DEF_LAYERS = [1, 4, 8, 12, 16, 20, 24, 28, 35]   # 层 1 = 对照
REP_LAYERS = {4, 8, 12, 16, 20, 24, 28, 35}

# 干跑缩域（12 点：4α×2β×1γ=8 + 单池 + G* + (.5,.5,.5) + 悬崖示例臂）
DRY_POINTS = [
    (0.125, 0.25, 0.25), (0.375, 0.25, 0.25), (0.625, 0.25, 0.25), (0.875, 0.25, 0.25),
    (0.125, 0.75, 0.25), (0.375, 0.75, 0.25), (0.625, 0.75, 0.25), (0.875, 0.75, 0.25),
    (0.0, 0.0, 0.0),
    DEF_GSTAR,
    (0.5, 0.5, 0.5),
    (0.125, 0.125, 0.75),   # γ 悬崖坍塌示例（far_budget=0 且 far_frac=.875）
]
DRY_LAYERS = [4, 8]


def make_indexer(far_method, near_method, alpha, beta, gamma, layer_idx):
    """构造 v2 代码路径的 TLIIndexer（subspace=full、4bit L2、无 kmeans/D'）。

    与 e2e 主表臂同口径：未传 --tli_subspace → 默认 full（L1 全维 128）；
    far_select/near_select 默认 4bit；enable_kmeans=False → use_partition
    仅由 e64_partition（α>0 且 β>0）驱动——v2 严格语义。
    """
    from types import SimpleNamespace
    args = SimpleNamespace(
        tia_block_size=BS, tia_level1_topk=K1, tia_level2_topk=K2,
        tia_level2_cmp_ratio=CMP, tia_enable_async_topk=False,
        tli_alpha=alpha, tli_beta=beta, tli_gamma=gamma,
        tli_far_method=far_method, tli_near_method=near_method,
        tli_enable_kmeans=False, tli_enable_layer_skip=False,
        tli_subspace="full",
    )
    idx = TLIIndexer(args)
    idx.layer_idx = layer_idx
    return idx


# ---------------- 闭式预算（γ 悬崖剔除 + TLI_DEBUG 对拍的参照） ---------------- #

def budget_closed_form(S, alpha, beta, gamma):
    """v2 严格口径闭式预算（对齐 tli_indexer.py compute_mask E64 分区路径）。

    far_budget = max(0, K2_mid − nt_near)（E109a-γ 严格化：无 64 保底）。
    与 r1c budget_row 的差异即此一处（r1c 快照时 canonical 臂仍是 max(64,…)）。
    单池（α=0 或 β=0）：nt_near/far_budget 不参与选择（仅记录），near_len_dyn
    走 far_select=4bit 单池路径的 near_len=2048 常量（该值不进入选择逻辑）。
    """
    kt = (S + BS - 1) // BS
    S_pad = kt * BS
    mid_len = max(0, S_pad - SINK_TOK - SWA)
    dual = alpha > 0 and beta > 0
    near_len_dyn = max(BS, int(alpha * mid_len)) if dual else 2048
    near_blks = max(SINK_BLOCKS, (S_pad - near_len_dyn) // BS)
    swa_lo_blk = max(near_blks, kt - max(1, SWA // BS))
    far_blocks_avail = near_blks - SINK_BLOCKS
    near_blocks_avail = max(0, swa_lo_blk - near_blks)
    nb_near = max(1, int(round(K1 * beta)))
    nb_far = max(0, K1 - nb_near)
    i_f = min(nb_far, far_blocks_avail)
    i_n = min(nb_near, near_blocks_avail)
    K2_eff = min(S, K2)
    K2_mid = max(0, K2_eff - SINK_TOK - SWA)
    nt_near = min(int(nb_near * BS * gamma), K2_mid)
    far_budget = max(0, K2_mid - nt_near)
    # far 区可见 token（p 宽 = S，far_tok_hi = min(near_blks·bs, S)）
    far_tok_hi = min(near_blks * BS, S)
    far_tok_avail = max(0, far_tok_hi - SINK_TOK)
    k2_far = min(far_budget, far_tok_avail, K2_mid) if far_tok_avail > 0 else 0
    k2_near = max(0, K2_mid - k2_far)
    near_finite = i_n * BS
    near_sel = min(k2_near, near_finite)
    swa_tok_eff = min(SWA, S)
    far_frac = (mid_len - near_len_dyn) / mid_len if mid_len > 0 else 0.0
    return {
        "S": S, "kt": kt, "mid_len": mid_len,
        "near_len_dyn": near_len_dyn, "near_blks": near_blks,
        "swa_lo_blk": swa_lo_blk,
        "far_blocks_avail": far_blocks_avail, "near_blocks_avail": near_blocks_avail,
        "nb_near": nb_near, "nb_far": nb_far, "i_f": i_f, "i_n": i_n,
        "K2_eff": K2_eff, "K2_mid": K2_mid,
        "nt_near": nt_near, "far_budget": far_budget,
        "k2_far": k2_far, "k2_near": k2_near, "near_sel": near_sel,
        "sink_tok": SINK_TOK, "swa_tok": swa_tok_eff,
        "total_selected": SINK_TOK + swa_tok_eff + k2_far + near_sel,
        "far_frac": far_frac,
    }


def exclusion_reason(S, alpha, beta, gamma):
    """候选在长度 S 下的闭式剔除判据（None = 合法）。

    ① γ 悬崖坍塌：far_budget=0 且 far_frac>0.25 → far 区整体零召回
      （E109 实证必崩；far_frac≤0.25 时 far_budget=0 几乎不掉分，保留）。
    ② 非法约束（文档 §2.2）：near 细筛配额 nt_near 超过 near 区长度 ℓ_near。
    仅对双池候选（α>0 且 β>0）适用；单池点不判。
    """
    if not (alpha > 0 and beta > 0):
        return None
    b = budget_closed_form(S, alpha, beta, gamma)
    if b["far_budget"] == 0 and b["far_frac"] > 0.25:
        return f"gamma_cliff_far_budget_zero(far_frac={b['far_frac']:.3f})"
    if b["nt_near"] > b["near_len_dyn"]:
        return (f"near_quota_exceeds_near_region("
                f"nt_near={b['nt_near']}>near_L={b['near_len_dyn']})")
    return None


def build_domain(alphas, betas, gammas, gstar, sample_Ss, dry_run):
    """候选域：原始网格 + 单池点 + G*；逐样本闭式剔除（任一样本触发即剔除，
    避免跨样本不公平平均；G* 豁免但记录 flag——参照必须评估）。"""
    if dry_run:
        raw = list(DRY_POINTS)
        raw_grid_note = "dry-run 12 点缩域"
    else:
        raw = [(a, b, g) for a in alphas for b in betas for g in gammas]
        raw.append((0.0, 0.0, 0.0))
        raw_grid_note = (f"{len(alphas)}×{len(betas)}×{len(gammas)}="
                         f"{len(alphas)*len(betas)*len(gammas)} 网格 + 单池点")
    if tuple(gstar) not in raw:
        raw.append(tuple(gstar))
    kept, excluded, gstar_flag = [], {}, False
    for c in raw:
        reasons = {}
        for name, S in sample_Ss.items():
            r = exclusion_reason(S, *c)
            if r:
                reasons[name] = r
        if tuple(c) == tuple(gstar):
            gstar_flag = bool(reasons)
            kept.append(c)          # G* 豁免
        elif reasons:
            excluded[c] = reasons
        else:
            kept.append(c)
    return raw, kept, excluded, gstar_flag, raw_grid_note


def cand_tag(c):
    a, b, g = c
    if a == 0.0 and b == 0.0:
        return "single_pool(0,0,0)"
    return f"a{a}_b{b}_g{g}"


# ---------------- 恒等式自检 + TLI_DEBUG 对拍 ---------------- #

def identity_selfcheck(device, seed=20261008):
    """文档 §3.2 恒等式 y−ŷ=(1−mS)(μS̄−μS) 在随机 p/v 上的残差（要求 <1e-5）。"""
    g = torch.Generator().manual_seed(seed)
    n, S, Dv = 4, 512, 64
    logits = torch.randn(n, S, generator=g).to(device) * 3.0
    p = F.softmax(logits, dim=-1)
    v = torch.randn(S, Dv, generator=g).to(device)
    m = torch.rand(S, generator=g).to(device) < 0.7
    m[0], m[-1] = True, False          # 保证 0<mS<1 非退化
    y = p @ v
    pm = p * m
    mS = pm.sum(-1, keepdim=True)
    num = pm @ v
    yhat = num / mS
    mu_S = num / mS
    mu_bar = (y - num) / (1.0 - mS)
    resid = (y - yhat) - (1.0 - mS) * (mu_bar - mu_S)
    return float(resid.abs().max())


def parse_dbg(buf):
    def _parse(ln):
        return dict(kv.split("=", 1) for kv in ln.split("]")[1].split() if "=" in kv)
    l1 = l2 = None
    for ln in buf.getvalue().splitlines():
        if ln.startswith("[L1dbg]"):
            l1 = _parse(ln)
        if ln.startswith("[L2dbg]"):
            l2 = _parse(ln)
    return l1, l2


def tli_debug_crosscheck(device, cand, far_method, near_method):
    """闭式公式 vs 真实代码单点对拍（合成 k/q，layer_idx=1 触发 debug 打印）。

    与 r1c formula_check 同协议：解析 [L1dbg]/[L2dbg]，闭式逐项对拍
    （kt/near_blks/swa_lo_blk/nb_far/nb_near/K2/K2_mid/nt_near/far_budget/k2_far）
    + far 区每 kv-head 实选 token 数 == k2_far（topk 严格预算的精确检验）
    + sink/swa 强制置位检验。
    """
    S = 4096
    torch.manual_seed(1234)
    k = torch.randn(1, S, 8, 128, device=device)
    q = torch.randn(1, 1, 32, 128, device=device)
    cu = torch.tensor([0, S], dtype=torch.long, device=device)
    q_ids = torch.tensor([S - 1], device=device)
    idx = make_indexer(far_method, near_method, *cand, layer_idx=1)
    os.environ["TLI_DEBUG"] = "1"
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            idict = idx.prepare_index(k, cu)
            sdict = idx.compute_score(q, q_ids, idict, 128 ** -0.5)
            mask = idx.compute_mask(q_ids, sdict)
    finally:
        os.environ.pop("TLI_DEBUG", None)
    l1, l2 = parse_dbg(buf)
    if l1 is None or l2 is None:
        return {"dbg_present": False,
                "error": "TLI_DEBUG 未打印 L1dbg/L2dbg（候选应为双池分区臂）"}
    b = budget_closed_form(S, *cand)
    far_hi = min(b["near_blks"] * BS, S)
    far_counts = mask[0, 0][..., SINK_TOK:far_hi].sum(-1) if far_hi > SINK_TOK \
        else torch.zeros(1, dtype=torch.long, device=device)
    checks = {
        "kt": int(l1["kt"]) == b["kt"],
        "near_blks": int(l1["near_blks"]) == b["near_blks"],
        "swa_lo_blk": int(l1["swa_lo_blk"]) == b["swa_lo_blk"],
        "nb_far": int(l1["nb_far"]) == b["nb_far"],
        "nb_near": int(l1["nb_near"]) == b["nb_near"],
        "K2": int(l2["K2"]) == b["K2_eff"],
        "K2_mid": int(l2["K2_mid"]) == b["K2_mid"],
        "nt_near": int(l2["nt_near"]) == b["nt_near"],
        "far_budget": int(l2["far_budget"]) == b["far_budget"],
        "k2_far": int(l2["k2_far"]) == b["k2_far"],
        "far_sel_exact_per_kvhead": bool((far_counts == b["k2_far"]).all()),
        "swa_forced": bool(mask[0, 0][..., max(0, S - SWA):].all()),
        "sink_forced_or_pooled": True,   # 单池路径 sink 走池竞争（v2 语义），仅记录
    }
    return {"dbg_present": True, "code_L1dbg": l1, "code_L2dbg": {
                k2: l2[k2] for k2 in ("K2", "K2_mid", "sink_tok", "swa_tok",
                                       "nt_near", "far_budget", "k2_far")},
            "closed_form": {k: b[k] for k in (
                "kt", "near_blks", "swa_lo_blk", "nb_near", "nb_far", "K2_eff",
                "K2_mid", "nt_near", "far_budget", "k2_far")},
            "checks": checks, "all_pass": all(checks.values()),
            "pass": all(checks.values())}


def avg_vs_minmax_diff_check(device, gstar, far_method, near_method):
    """E109a 纪律自检：avg 分支须真生效（回放 mask 异于 (minmax,minmax) 变体）。

    判别力前提：L1 双池预算 binding。S=4096 下 G* 池饱和（far 区 54 块 <
    nb_far=112，near 区 6 块 < nb_near=16 → L1 全选，method 只影响 L1 块分数源，
    饱和态下合法同 mask）——干跑枚举 S=4096 即饱和态，故本检查独立用 S=32768
    （far 区 446 块 > 112、near 区 62 块 > 16，双池 binding，method 差异可见）。
    各 indexer 走各自完整链路（prepare_index→compute_score→compute_mask），
    不共享 score_dict（避免分数源归属歧义）。
    """
    if (far_method, near_method) == ("minmax", "minmax"):
        return {"skipped": True, "pass": True,
                "note": "主 method 即 (minmax,minmax)，无需差异检查"}
    S = 32768
    torch.manual_seed(5678)
    k = torch.randn(1, S, 8, 128, device=device)
    q = torch.randn(1, 1, 32, 128, device=device)
    cu = torch.tensor([0, S], dtype=torch.long, device=device)
    q_ids = torch.tensor([S - 1], device=device)
    idx_a = make_indexer(far_method, near_method, *gstar, layer_idx=1)
    idx_m = make_indexer("minmax", "minmax", *gstar, layer_idx=1)
    ia = idx_a.prepare_index(k, cu)
    sa = idx_a.compute_score(q, q_ids, ia, 128 ** -0.5)
    assert sa.get("score_coarse_avg") is not None, \
        "score_coarse_avg 缺失：avg 分支未生效（E109a bug 复现）"
    ma = idx_a.compute_mask(q_ids, sa)
    im = idx_m.prepare_index(k, cu)
    sm = idx_m.compute_score(q, q_ids, im, 128 ** -0.5)
    mm = idx_m.compute_mask(q_ids, sm)
    n_diff = int((ma != mm).sum())
    del k, q, ia, sa, im, sm, ma, mm
    return {"S": S, "far/near": [far_method, near_method],
            "mask_diff_elems": n_diff, "found_diff": n_diff > 0,
            "pass": n_diff > 0,
            "note": "S=32768 使 L1 双池 binding（S=4096 饱和态下 method 差异合法"
                    "不可见）；avg mask 须异于 minmax 变体（E109a：avg 分支静默"
                    "失效时两 mask 逐位相同）"}


# ---------------- 输出失真评估（候选无关量缓存） ---------------- #

def select_query_indices(qpos, S, max_q):
    """query 位置采样（文档 §6.3：覆盖中后段；每文档至多 max_q 个，确定性）。

    过滤 S_q≥1024（K2 满额口径）；优先取 t ≥ S//2 的中后段位置，均匀取
    max_q 个；中后段为空时回退全部合法位置。
    """
    cand = [(i, t) for i, t in enumerate(qpos) if t + 1 >= 1024]
    late = [(i, t) for i, t in cand if t >= S // 2]
    pool = late if late else cand
    if not pool:
        return []
    if len(pool) <= max_q:
        return [i for i, _ in pool]
    step = (len(pool) - 1) / (max_q - 1)
    idxs, seen, out = [], set(), []
    for j in range(max_q):
        jj = min(len(pool) - 1, round(j * step))
        if jj not in seen:
            seen.add(jj)
            out.append(pool[jj][0])
    return out


def eval_layer_sample(k, v_all, q, qpos, qpos_sel, kept, indexer, device,
                      identity_replay_check):
    """单 (layer, sample) 枚举：逐 query 缓存 dense 量，逐候选只跑 compute_mask。

    返回 per-candidate err 累计（Σ_{q,h} ‖y−ŷ‖²，恒等式口径）、(q,h) 对数、
    dense 输出平方和（aℓ 用）、per-candidate mS 均值。
    k [S,Hkv,D] / v_all [S,Hkv,Dv] / q [nq,H,D]，均 fp32 on device。
    """
    D = k.shape[-1]
    Hkv, H = k.shape[1], q.shape[1]
    G = H // Hkv
    assert H == Hkv * G, f"GQA 不整除: H={H} Hkv={Hkv}"
    err = {c: 0.0 for c in kept}
    mS_acc = {c: 0.0 for c in kept}
    n_pairs, y_sq = 0, 0.0
    n_q = 0
    for qi in qpos_sel:                      # qi = q 行索引；query 位置 = qpos[qi]
        S_q = int(qpos[qi]) + 1
        k_q = k[:S_q].unsqueeze(0)              # [1,S_q,Hkv,D]（decode t 时刻口径）
        v_q = v_all[:S_q]
        cu = torch.tensor([0, S_q], dtype=torch.long, device=device)
        q_ids = torch.tensor([S_q - 1], device=device)
        q_in = q[qi:qi + 1].unsqueeze(0)        # [1,1,H,D]
        # ---- 候选无关量：索引 + 分数 + dense 分布（先算一次，跨候选复用）----
        index_dict = indexer.prepare_index(k_q, cu)
        score_dict = indexer.compute_score(q_in, q_ids, index_dict, D ** -0.5)
        # E109a 纪律：avg 分支真生效（subspace=full 下旧 bug 曾静默退化 minmax）
        if "avg" in (indexer.far_method, indexer.near_method):
            assert score_dict.get("score_coarse_avg") is not None, \
                "score_coarse_avg 缺失：avg 分支未生效（E109a bug 复现）"
        qr = q[qi].reshape(Hkv, G, D)
        s_full = torch.einsum("hgd,shd->hgs", qr, k[:S_q]) * (D ** -0.5)
        p_full = torch.softmax(s_full, dim=-1)              # [Hkv,G,S_q]
        y = torch.einsum("hgs,shd->hgd", p_full, v_q)       # [Hkv,G,Dv]
        y_sq += float((y * y).sum())
        n_pairs += Hkv * G
        n_q += 1
        # ---- 逐候选：原地改 α/β/γ → compute_mask（真实链路）→ 恒等式失真 ----
        for c in kept:
            indexer.alpha, indexer.beta, indexer.gamma = c
            mask = indexer.compute_mask(q_ids, score_dict)  # [1,1,Hkv,S_q]
            m = mask[0, 0]
            mb = m.unsqueeze(1)                              # [Hkv,1,S_q]
            p_sel = p_full * mb
            mS = p_sel.sum(-1)                               # [Hkv,G]
            num_S = torch.einsum("hgs,shd->hgd", p_sel, v_q)  # [Hkv,G,Dv]
            mS_ = mS.unsqueeze(-1)                           # [Hkv,G,1]
            w_ = (1.0 - mS).unsqueeze(-1)                    # [Hkv,G,1]
            mu_S = num_S / mS_.clamp(min=1e-12)
            mu_bar = (y - num_S) / w_.clamp(min=1e-12)
            diff = mu_bar - mu_S
            w2 = (1.0 - mS) ** 2                             # [Hkv,G]
            e_id = w2 * (diff * diff).sum(-1)                # ‖y−ŷ‖² per (h,g)
            if identity_replay_check is not None and cand_tag(c) in identity_replay_check:
                # 干跑加强：回放路径上恒等式 vs 直接 ŷ 对拍
                yhat = num_S / mS_.clamp(min=1e-12)
                e_dir = ((y - yhat) ** 2).sum(-1)
                rel = float((e_id - e_dir).abs().max() /
                            (e_dir.sum().clamp(min=1e-20)))
                identity_replay_check[cand_tag(c)] = rel
            err[c] += float(e_id.sum())
            mS_acc[c] += float(mS.mean())
        del index_dict, score_dict, p_full, y, s_full, k_q, v_q, p_sel, num_S
    return err, n_pairs, y_sq, n_q, mS_acc


# ---------------- 合成 trace（干跑） ---------------- #

def gen_synth_trace(outdir, seed=20261008):
    """CPU 合成 trace（shape 对齐 Qwen3-8B；无 GPU）。2 样本 × 层 {4,8}。"""
    S, Hkv, H, D, Dv = 4096, 8, 32, 128, 128
    qpos = [1500, 2500, 3500, 4095]
    names = []
    for si in range(2):
        name = f"synth_{si}"
        names.append(name)
        pdir = os.path.join(outdir, name)
        os.makedirs(pdir, exist_ok=True)
        for li in DRY_LAYERS:
            g = torch.Generator().manual_seed(seed + si * 100 + li)
            torch.save({
                "k": torch.randn(S, Hkv, D, generator=g),
                "v": torch.randn(S, Hkv, Dv, generator=g),
                "q": torch.randn(len(qpos), H, D, generator=g),
                "qpos": torch.tensor(qpos), "S": S,
            }, os.path.join(pdir, f"layer{li:02d}.pt"))
        json.dump({"S": S, "n_layers": 36, "prompt": name, "with_v": True,
                   "synthetic": True, "layers": DRY_LAYERS},
                  open(os.path.join(pdir, "meta.json"), "w"))
    return names


# ---------------- 主流程 ---------------- #

def main():
    ap = argparse.ArgumentParser(
        description="P0' 逐层参数求解器潜力检查（输出失真目标层内枚举 vs 统一配置 G*）")
    ap.add_argument("--trace", default="/tmp/trace/qwen3-8b-v",
                    help="trace 目录（须为 collect_trace_lb_v.py 产出，含 v）")
    ap.add_argument("--out", default=None, help="输出 JSON 路径")
    ap.add_argument("--gstar", default=",".join(map(str, DEF_GSTAR)),
                    help="统一配置 α,β,γ（默认 aavg 海选领跑臂 0.125,0.125,0.375）")
    ap.add_argument("--far-method", default="avg", help="far 池 L1 分数源（默认 avg）")
    ap.add_argument("--near-method", default="avg", help="near 池 L1 分数源（默认 avg）")
    ap.add_argument("--layers", default=",".join(map(str, DEF_LAYERS)))
    ap.add_argument("--max-queries", type=int, default=6,
                    help="每文档至多 N 个 query 位置（文档 §6.3）")
    ap.add_argument("--samples", default=None, help="样本名逗号过滤（默认全部）")
    ap.add_argument("--alphas", default=",".join(map(str, DEF_ALPHAS)))
    ap.add_argument("--betas", default=",".join(map(str, DEF_BETAS)))
    ap.add_argument("--gammas", default=",".join(map(str, DEF_GAMMAS)))
    ap.add_argument("--device", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="CPU 合成数据干跑（12 点缩域 + 全部自检；不占 GPU）")
    ap.add_argument("--seed", type=int, default=20261008)
    ns = ap.parse_args()

    dry = ns.dry_run
    device = "cpu" if dry else (
        ns.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    gstar = tuple(float(x) for x in ns.gstar.split(","))
    assert len(gstar) == 3
    far_method, near_method = ns.far_method, ns.near_method
    out_path = ns.out or os.path.join(
        REPO, "exp", "trace", "results",
        "p0p_perlayer_potential_dryrun.json" if dry
        else "p0p_perlayer_potential.json")

    t0 = time.time()
    self_checks = {}

    # ---- 自检 1：恒等式（随机 p/v）----
    resid = identity_selfcheck(device, ns.seed)
    self_checks["identity_random"] = {"max_residual": resid, "pass": resid < 1e-5}

    # ---- trace 发现 ----
    if dry:
        trace_dir = "/tmp/p0p_dryrun_trace"
        os.makedirs(trace_dir, exist_ok=True)
        sample_names = gen_synth_trace(trace_dir, ns.seed)
        layers = list(DRY_LAYERS)
    else:
        trace_dir = ns.trace
        layers = [int(x) for x in ns.layers.split(",") if x.strip()]
        sample_names = sorted(
            d for d in os.listdir(trace_dir)
            if os.path.isfile(os.path.join(trace_dir, d, "meta.json")))
        if ns.samples:
            keep = {s.strip() for s in ns.samples.split(",")}
            sample_names = [s for s in sample_names if s in keep]
        assert sample_names, f"trace 目录无样本: {trace_dir}"

    sample_Ss = {}
    for name in sample_names:
        meta = json.load(open(os.path.join(trace_dir, name, "meta.json")))
        sample_Ss[name] = int(meta["S"])

    # ---- 候选域 + 闭式剔除 ----
    if dry:
        alphas = betas = gammas = []
    else:
        alphas = [float(x) for x in ns.alphas.split(",")]
        betas = [float(x) for x in ns.betas.split(",")]
        gammas = [float(x) for x in ns.gammas.split(",")]
    raw, kept, excluded, gstar_flag, grid_note = build_domain(
        alphas, betas, gammas, gstar, sample_Ss, dry)

    # ---- TLI_DEBUG 单点对拍（合成 k/q，始终执行：闭式 vs 真实代码）----
    dbg = tli_debug_crosscheck(device, gstar, far_method, near_method)
    self_checks["tli_debug_crosscheck"] = dbg

    # ---- minmax 差异检查（E109a 纪律：独立大 S 合成，L1 binding 才有判别力）----
    self_checks["avg_vs_minmax_mask_diff"] = avg_vs_minmax_diff_check(
        device, gstar, far_method, near_method)

    # ---- 逐层枚举 ----
    indexer = make_indexer(far_method, near_method, *gstar, layer_idx=layers[0])
    acc = {}   # layer -> sample -> {err,n_pairs,y_sq,n_q,mS}
    identity_replay_check = None
    if dry:
        # 干跑加强：G* 与单池点上做恒等式 vs 直接 ŷ 对拍（tag 键，JSON 可序列化）
        identity_replay_check = {cand_tag(tuple(gstar)): None,
                                 cand_tag((0.0, 0.0, 0.0)): None}
    for layer in layers:
        acc[layer] = {}
        for name in sample_names:
            lf = os.path.join(trace_dir, name, f"layer{layer:02d}.pt")
            if not os.path.isfile(lf):
                print(f"[warn] 缺层文件 {lf}，跳过")
                continue
            d = torch.load(lf, map_location="cpu", weights_only=False)
            if "v" not in d:
                raise RuntimeError(
                    f"{lf} 无 'v' 键——须先用 collect_trace_lb_v.py 采集带 V 的 trace")
            k = d["k"].to(device).float()
            v_all = d["v"].to(device).float()
            q = d["q"].to(device).float()
            qpos = d["qpos"].tolist()
            S = int(d["S"])
            qpos_sel = select_query_indices(qpos, S, ns.max_queries)
            if not qpos_sel:
                print(f"[warn] {name} L{layer}: 无合法 query 位置（S={S}），跳过")
                continue
            err, n_pairs, y_sq, n_q, mS_acc = eval_layer_sample(
                k, v_all, q, qpos, qpos_sel, kept, indexer, device,
                identity_replay_check)
            acc[layer][name] = {"err": err, "n_pairs": n_pairs,
                                "y_sq": y_sq, "n_q": n_q, "mS": mS_acc}
            del k, v_all, q, d
            if device != "cpu":
                torch.cuda.empty_cache()
        print(f"[L{layer}] done ({time.time() - t0:.1f}s)", flush=True)

    if identity_replay_check is not None:
        vals = {t: r for t, r in identity_replay_check.items()}
        done = all(r is not None for r in vals.values())
        self_checks["identity_on_replay"] = {
            "max_relative_deviation": (max(vals.values()) if done and vals
                                       else None),
            "per_candidate": vals,
            "pass": bool(done and vals) and max(vals.values()) < 1e-4}

    # ---- 聚合：aℓ → e_{ℓ,j} → Eℓ ----
    per_layer = {}
    gap_rels, c_stars = [], []
    for layer in layers:
        if not acc[layer]:
            continue
        y_sq = sum(rec["y_sq"] for rec in acc[layer].values())
        n_y = sum(rec["n_pairs"] for rec in acc[layer].values())
        a2 = y_sq / max(n_y, 1) + 1e-9
        E = {}
        mS_mean = {}
        for c in kept:
            es = []
            for name, rec in acc[layer].items():
                if rec["n_pairs"] == 0:
                    continue
                es.append(rec["err"][c] / rec["n_pairs"] / a2)
            if es:
                E[c] = sum(es) / len(es)
                mS_mean[c] = sum(
                    acc[layer][n2]["mS"][c] for n2 in acc[layer]
                ) / max(sum(acc[layer][n2]["n_q"] for n2 in acc[layer]), 1)
        if not E:
            continue
        c_star = min(E, key=E.get)
        Eg = E.get(tuple(gstar))
        top5 = sorted(E.items(), key=lambda kv: kv[1])[:5]
        gap_abs = (Eg - E[c_star]) if Eg is not None else None
        gap_rel = (gap_abs / Eg) if (Eg is not None and Eg > 0) else None
        per_layer[str(layer)] = {
            "role": "control(早期层对照)" if layer not in REP_LAYERS else "representative",
            "a_l_rms": round(a2 ** 0.5, 6),
            "E_best": round(E[c_star], 8), "c_star": list(c_star),
            "c_star_tag": cand_tag(c_star),
            "E_gstar": round(Eg, 8) if Eg is not None else None,
            "gap_abs": round(gap_abs, 8) if gap_abs is not None else None,
            "gap_rel": round(gap_rel, 6) if gap_rel is not None else None,
            "mS_best": round(mS_mean.get(c_star, float("nan")), 6),
            "mS_gstar": round(mS_mean.get(tuple(gstar), float("nan")), 6),
            "top5": [{"cand": list(c), "tag": cand_tag(c), "E": round(e, 8)}
                     for c, e in top5],
        }
        if layer in REP_LAYERS and gap_rel is not None:
            gap_rels.append(gap_rel)
            c_stars.append(c_star)

    gap_rels_sorted = sorted(gap_rels)
    median_gap = (gap_rels_sorted[len(gap_rels_sorted) // 2]
                  if gap_rels_sorted else None)
    distinct_c = len(set(c_stars))
    spread = {}
    if c_stars:
        for i, pname in enumerate(("alpha", "beta", "gamma")):
            vals = [c[i] for c in c_stars]
            spread[pname] = {"min": min(vals), "max": max(vals)}
    verdict = None
    if median_gap is not None:
        verdict = "potential-high" if median_gap > 0.10 else "potential-low"

    all_pass = all(v.get("pass", False) for v in self_checks.values())

    result = {
        "exp": "P0' 逐层参数求解器潜力检查（预注册判决门）",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "objective": ("projected output MSE（恒等式 y−ŷ=(1−mS)(μS̄−μS)），"
                      "文档 §5.1 归一化 e=‖Δo‖²/(aℓ²+ε)，文档内 query 平均→样本平均"),
        "verdict_rule": "8 代表层 gap_rel 中位 >10% → potential-high（值得开发逐层求解器）",
        "config": {
            "trace_dir": trace_dir, "device": device, "dry_run": dry,
            "layers": layers, "samples": sample_names,
            "sample_S": sample_Ss, "max_queries_per_doc": ns.max_queries,
            "gstar": list(gstar), "far_method": far_method,
            "near_method": near_method,
            "budgets": {"bs": BS, "K1": K1, "K2": K2, "cmp_ratio": CMP,
                        "sink_blocks": SINK_BLOCKS, "swa": SWA},
            "subspace": "full（L1 全维 128，与 e2e 主表臂同口径）",
            "code_path": "sparse_attn tli_indexer v2（E109a 修复 c7a6cfc04→eb24aa8cf 后）",
            "seed": ns.seed,
        },
        "domain": {
            "grid_note": grid_note,
            "n_raw": len(raw), "n_kept": len(kept),
            "n_excluded": len(excluded),
            "exclusion_rules": [
                "γ 悬崖：far_budget=0 且 far_frac>0.25（E109 实证必崩，闭式预判）",
                "非法约束：nt_near > ℓ_near（near 细筛配额超 near 区长，文档 §2.2）",
                "剔除判据任一样本 S 触发即全局剔除（避免跨样本不公平平均）"],
            "gstar_cliff_excluded": gstar_flag,
            "excluded_candidates": [
                {"cand": list(c), "tag": cand_tag(c), "excluded_by": sorted(
                    set(rs.values())), "per_sample": rs}
                for c, rs in sorted(excluded.items())],
            "kept_candidates": [cand_tag(c) for c in kept],
            "dedup_note": "未做跨配置有效预算去重（平台区存在：不同参数可同 E 值）",
        },
        "per_layer": per_layer,
        "global": {
            "n_rep_layers": len(gap_rels),
            "gap_rel_by_layer": [round(g, 6) for g in gap_rels],
            "median_gap_rel": round(median_gap, 6) if median_gap is not None else None,
            "mean_gap_rel": round(sum(gap_rels) / len(gap_rels), 6) if gap_rels else None,
            "layer_param_discreteness": {
                "distinct_c_star": distinct_c,
                "c_star_list": [list(c) for c in c_stars],
                "param_spread": spread},
            "verdict": verdict,
            "verdict_caveat": ("固定 dense trace 的层内代理（文档 §4：非 e2e 最优；"
                               "gap 小→负结果收档走速度路线，gap 大→须过整表联合"
                               "验证三层协议后才可宣称收益）"),
        },
        "self_checks": self_checks,
        "all_self_checks_pass": all_pass,
        "wall_time_s": round(time.time() - t0, 1),
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(result, open(out_path, "w"), ensure_ascii=False, indent=1)

    # ---- 人类可读摘要 ----
    print(f"\n=== P0' 逐层潜力检查 ({'DRY-RUN 合成数据' if dry else '真实 trace'}) "
          f"method=({far_method},{near_method}) G*={list(gstar)} ===")
    print(f"候选域：{grid_note} → 剔除 {len(excluded)} → 枚举 {len(kept)}"
          f"（G* 豁免剔除，cliff_flag={gstar_flag}）")

    def _f(x, spec="{:.6f}"):
        return spec.format(x) if x is not None else "NA"

    for layer, rec in per_layer.items():
        gr = rec["gap_rel"]
        print(f"  L{layer}({rec['role'][:5]}): E*(c*)={rec['E_best']:.6f} "
              f"c*={rec['c_star']} | E(G*)={_f(rec['E_gstar'])} "
              f"gap={_f(gr, '{:+.4%}') if gr is not None else 'NA'} "
              f"mS*={rec['mS_best']:.4f} mS(G*)={rec['mS_gstar']:.4f}")
    g = result["global"]
    print(f"代表层 gap_rel 中位 = {_f(g['median_gap_rel'])}"
          f" → 判决：{g['verdict']}")
    print(f"层间最优参数离散度：{g['layer_param_discreteness']['distinct_c_star']} "
          f"个不同 c* / {g['n_rep_layers']} 代表层 {g['layer_param_discreteness']['c_star_list']}")
    for name, chk in self_checks.items():
        print(f"  [check] {name}: {'PASS' if chk.get('pass') else 'FAIL'}")
    print(f"ALL SELF-CHECKS: {'PASS' if all_pass else 'FAIL'}")
    print(f"saved -> {out_path} ({result['wall_time_s']}s)")
    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
