# -*- coding: utf-8 -*-
"""E124a（S-T021 / A1）：运行时逐序列逐层动态参数 controller v0。

语义判定（S-T021，用户指令 + GPT Runtime 方案 §0 合流口径）：
  动态 = 当前 seq 在当前层根据**当时合法可见信息**生成 α/β/γ——
  不是层号查表、不是任务名查表、不是离线静态表。本模块任何路径
  都不做 (layer → 参数) 查表；R 的逐层 seed 只是随机投影的身份标识，
  参数本身只由内容特征决定。

权威规格（逐字遵守，不改动 advice 文件）：
  agent_doc/advice/2026-10-11_Runtime_Sequence_Layerwise_Execution_Plan_by_gpt.md
  §2.1（v0 公共特征与三 profile 规则）/ §2.2（精确预算与因果时点）。

三个模块（GPT §2 表）：
  M1 公共特征   compute_features：每 seq×layer 固定 seed 的 8 维随机
                符号投影 R（元素 ±1/√8）；从非保护 middle 取 ≤16 个等距
                anchor 位置的原 K 即时投影（不另扫全历史 KV；后续若缓存
                投影 K 须另记每 token 更新/每层存储/回收成本）；当前
                query 每 Q-head 投影后与 anchor 打点积，温度 sqrt(原D)
                不是 sqrt(8)；middle 最后 1/4 = N_ref、其余 F_ref（固定
                参考边界，先于 α 定义，不自我标注）；p_hi 对全部 anchor
                共同减 max 后 softmax；s_h=log((d_N+eps)/(d_F+eps))，
                eps=1e-8 且记录 dtype；GQA 聚合 s=(1/Hkv)Σ_g[(1/|Q_g|)
                Σ_{h∈Q_g}s_h]——先同 KV 组内 Q-head 等权、再对 KV 组
                等权，**不许组大小加权**。anchor 选取只依合法位置，
                不依 method 挑中 token。非连续 gather、16·d·8 的 K 投影、
                q 投影及位置构造成本**全记账**（§2.1 条 1）。
  M2 三档规则   classify_profile：s<-ln2→P_F；s>ln2→P_N；否则 P_C
                （恰 ±ln2 归 P_C）。档位 (α,β,ρ)=P_F(1/8,1/8,1/4) /
                P_C(1/4,1/4,1/2) / P_N(1/2,1/2,3/4)。ρ 是希望给 near
                的 middle L2 比例，不是重新定义 γ。阈值首版固定，不按
                method 改（§2.1 条 4）。总预算固定下偏 near 会压缩 far，
                P_N 不能称「更保守」。
  M3 预算编译器 compile_tier / compile_fixed：Bn=max(1,round(K1β)),
                Bf=K1−Bn；Kmid=max(0,min(K2,|U|)−|P|)；Tn_target=
                round(ρ·Kmid) → 反算 γ=Tn_target/(Bn·b) → 存整数 Tn 为
                规范目标并验证浮点 floor 回译（浮点边界不少 1）；
                Tn=min(目标,Kmid), Tf=Kmid−Tn。γ>1 或实际候选容量不足
                → 该 profile 不可用 → 回退链：同 method 的 P_C → 该
                method 已通过正确性验证的固定参数档 (.25,.125,.625)
                （**绝不切换到 mavg 选择器**，只换参数）→ unsupported；
                每次回退记录原因与成本，至多两次重试（§2.1 条 5 + §2.2）。

合法性（§2.1 条 3）：任何非空区 anchor<2、数值非有限、state 身份不符
→ 返回中性 profile（P_C）并记录 unknown；空 middle 直接保护/全可见短
输入是合法路径（非失败）。禁止把没采样区当零重要性。

区域大小修正 A_r,h=|r|·d_r,h 仅作诊断记录，不作为真实 mass；保最差
head 诊断。该 s 是低维少量 anchor 代理，绝不是实际 attention mass。

与生产代码的关系：tli_indexer.py 生产静态路径**一行不改**。本模块是
独立新代码；预算公式镜像生产语义（Bn=max(1,int(round(k1·β)))、
near_len=max(bs,int(α·M))、nt_near=min(int(Bn·bs·γ),K2_mid)），
差异点显式注明（见 compile_tier 内注释）。Kmid 采用 §2.2 权威公式
max(0,min(K2,|U|)−|P|)（比生产 K2−|P| 多一层 min(K2,|U|) 对实际容量
更诚实；部署集成（A5+）时统一）。不实现 M5 D′ gate；不接 GPU；
不注册任何 CLI 参数（入口脚本 exp/trace/run_e124a_dryrun.py 自带
argparse，与 sparse_attn/arguments.py 零冲突）。
"""
import hashlib
import json
import math
import os
import time

import torch

CONTROLLER_VERSION = "e124a-dyn-controller-v0"

# ---- v0 冻结常量（GPT §2.1；预注册，不按筛选集回调） ----
FEATURE_DIM = 8            # 随机符号投影维数
N_ANCHORS_MAX = 16         # anchor 上限（有效 middle 不足取全部）
S_H_EPS = 1e-8             # s_h 的 eps（dtype 随记录）
THRESH_LN2 = math.log(2.0)  # 三档阈值 ±ln2，固定，不按 method 改
MAX_BUMP_ITERS = 128       # γ 浮点回译 bump 上限（保险，数学上远用不到）

TIER_PARAMS = {
    # (α, β, ρ)：ρ = 希望给 near 的 middle L2 比例，不是重新定义 γ
    "P_F": {"alpha": 0.125, "beta": 0.125, "rho": 0.25},
    "P_C": {"alpha": 0.25, "beta": 0.25, "rho": 0.5},
    "P_N": {"alpha": 0.5, "beta": 0.5, "rho": 0.75},
}
NEUTRAL_PROFILE = "P_C"   # 合法性失败/空 middle 的中性档

# 回退链末端的固定参数档 registry：只列「该 method 已通过正确性验证」
# 的组合（GPT §2.1 条 5）。verification 字段如实记录 A0 盘点出的证据
# 等级（e2e_full = 13 任务全量 e2e；e2e_screen = 5 任务 e2e 筛选；
# e2e_screen_singlepool = 仅 (0,0) 单池点有 e2e；e2e_smoke = n=8 冒烟）。
# v0 冻结初值；A3 入筛前按 CPU/小输入门禁逐组合验收后可增删（那时
# 才允许更新本表——不是按筛选结果回调）。
_FIXED_PARAMS = (0.25, 0.125, 0.625)
METHOD_FIXED_TIER = {
    "mavg":         {"params": _FIXED_PARAMS, "verification": "e2e_full"},
    "aavg":         {"params": _FIXED_PARAMS, "verification": "e2e_full"},
    "mminmax":      {"params": _FIXED_PARAMS, "verification": "e2e_full"},
    "cavg":         {"params": _FIXED_PARAMS, "verification": "e2e_full"},
    "ccluster":     {"params": _FIXED_PARAMS,
                     "verification": "e2e_screen_singlepool"},
    "cavgsim":      {"params": _FIXED_PARAMS, "verification": "e2e_screen"},
    "cclustersim":  {"params": _FIXED_PARAMS, "verification": "e2e_smoke"},
}
_METHOD_ALIASES = {"cavg_sim": "cavgsim", "ccluster_sim": "cclustersim"}


def canonical_method(method):
    """method 名归一（E110 草案用 cavg_sim/ccluster_sim，GPT §4 表与
    E109 screen4 用 cavgsim/cclustersim——两写法同义）。未知名原样
    返回（由固定档 registry 判 unsupported，不在这里静默替换）。"""
    m = str(method)
    return _METHOD_ALIASES.get(m, m)


# ================================================================ M1 公共特征

def _seq_layer_seed(seq_id, layer_idx):
    """每 seq×layer 的固定 seed（SHA256 派生，跨进程/跨运行可复现；
    不用 Python hash()——有盐值不可复现）。"""
    h = hashlib.sha256(f"e124a|{seq_id}|{int(layer_idx)}".encode()).digest()
    return int.from_bytes(h[:8], "big")


def make_projection(seq_id, layer_idx, dim, dtype=torch.float64):
    """M1：8 维随机符号投影 R ∈ {±1/√8}^{dim×8}，每 seq×layer 固定 seed。

    元素数学上恰为 ±1/√8——用 float64 存（float32 会舍入到
    0.353553384542…，12 位有效数字口径下不再是 ±1/√8）；下游
    compute_features 同口径 float64 计算（CPU v0；GPU 集成时精度
    契约须另行门禁）。v0 每次 compute_features 即时生成并如实记账
    （rng_elems=dim·8）；若后续缓存 R 分摊成本，须另记每层存储/回收
    （§2.1 条 1 成本条款）。
    """
    g = torch.Generator().manual_seed(_seq_layer_seed(seq_id, layer_idx))
    signs = (torch.randint(0, 2, (dim, FEATURE_DIM), generator=g) * 2 - 1)
    return signs.to(dtype) / math.sqrt(FEATURE_DIM)


def anchor_positions(m_len):
    """middle 内 ≤16 个等距 anchor 的 token 下标（0 基，相对 middle 起点）。

    只依合法位置（等距规则），不依 method 挑中 token；有效 middle
    不足 16 取全部（§2.1 条 1）。round 均距 + 去重保严格递增。
    """
    if m_len <= 0:
        return []
    n = min(N_ANCHORS_MAX, m_len)
    if n == 1:
        return [0]
    pos = sorted({int(round(i * (m_len - 1) / (n - 1))) for i in range(n)})
    return pos


def ref_split(m_len):
    """固定参考边界：middle 最后 1/4 = N_ref，其余 F_ref（§2.1 条 2）。

    split = m_len − ceil(m_len/4)；位置 ≥ split 属 N_ref。
    边界先用待预测 α 之前定义，避免自我标注。
    """
    if m_len <= 0:
        return 0
    return m_len - math.ceil(m_len / 4)


def aggregate_s_h(s_h, group_sizes):
    """GQA 聚合（§2.1 条 2，权威公式）：

        s = (1/Hkv) Σ_g [(1/|Q_g|) Σ_{h∈Q_g} s_h]

    先同 KV 组内 Q-head **等权**，再对 KV 组**等权**——不许组大小加权
    （组大小加权会静默改结果，见 test_T04 反例）。
    s_h: [H] tensor/list；group_sizes: 每 KV 组的 Q-head 数（顺序即
    head 连续分组的边界，与 GQA head→kv 映射 h//G 一致）。
    """
    total = 0.0
    n_groups = 0
    off = 0
    for gsz in group_sizes:
        gsz = int(gsz)
        if gsz <= 0:
            continue
        seg = s_h[off:off + gsz]
        # float64 聚合：float32 的 mean 会把 0.55 舍成 0.55000001192（T04
        # 实测），等权公式必须不受 dtype 舍入污染
        mean_g = float(seg.to(torch.float64).mean()) if torch.is_tensor(seg) \
            else sum(float(x) for x in seg) / gsz
        total += mean_g
        n_groups += 1
        off += gsz
    if n_groups == 0:
        return float("nan")
    return total / n_groups


def _base_cost():
    """成本记账骨架（§2.1 条 1：非连续 gather / K 投影 / q 投影 /
    位置构造 / 点积 / softmax / R 生成全部入账，全部非负）。"""
    return {
        "anchor_pos_ops": 0,   # 位置构造（等距索引计算）
        "rng_elems": 0,        # R 生成元素数（dim·8；v0 即时生成不缓存）
        "gather_elems": 0,     # 非连续 gather 元素数（n·Hkv·D）
        "k_proj_macs": 0,      # anchor K 投影 MACs（n·Hkv·D·8）
        "q_proj_macs": 0,      # q 投影 MACs（H·D·8）
        "dot_macs": 0,         # q·anchor 点积 MACs（H·n·8）
        "softmax_elems": 0,    # softmax 元素数（H·n）
        "retries": 0,          # M3 回退重试次数
        "wall_ms": 0.0,        # decide 全链墙钟（毫秒）
    }


def compute_features(q, k_mid, seq_id, layer_idx, *, phase="prefill",
                     state_epoch=0, identity_check=None):
    """M1：当前 (seq, layer) 的公共特征（无未来/无答案/无任务标签；
    所有 method 同特征、同投影、同 anchor——method 不进特征）。

    输入契约（调用方负责，M0 身份/状态模块 A1 不在范围）：
      q     : [H, D]  当前 query（全部 Q-head，原始 D 维；只含当前
                       合法可见 query——decode 期为当前步 q）
      k_mid : [T_mid, Hkv, D]  本 seq 本层**非保护 middle** 的已可见 K
                       （U\P；只含因果已见 token，禁止未来）
      identity_check: 可选 dict {seq_id, layer_idx, phase, state_epoch}
                       ——消费端声明的身份；与特征计算身份不符 →
                       unknown（§2.1 条 3 state 身份不符条款）。

    返回 dict（全部可 JSON 序列化）：
      status: "ok" | "empty_middle" | "unknown"
      s: 层级 log 密度比（聚合后标量；失败为 None）
      s_h: 每 Q-head 的 s_h（失败为 None）
      reason: 非 ok 时的原因（anchor_short/nonfinite/identity_mismatch/
              gqa_head_mismatch）
      ...特征明细与 cost（见实现）
    """
    cost = _base_cost()
    H, D = int(q.shape[0]), int(q.shape[-1])
    T_mid = 0 if k_mid is None else int(k_mid.shape[0])
    Hkv = 0 if k_mid is None else int(k_mid.shape[1])
    identity = {"seq_id": seq_id, "layer_idx": int(layer_idx),
                "phase": phase, "state_epoch": int(state_epoch)}

    def _result(status, reason=None, **extra):
        base = {
            "status": status,
            "identity": identity,
            "available_at_mid_tokens": T_mid,   # M1 验收：记录 available_at
            "s": None, "s_h": None, "reason": reason,
            "d_model": D, "temp_rule": "sqrt(D)",
            "s_h_eps": S_H_EPS, "s_h_eps_dtype": "torch.float64",
            "cost": cost,
        }
        base.update(extra)
        return base

    # ---- state 身份门禁（§2.1 条 3）----
    if identity_check is not None and dict(identity_check) != identity:
        return _result("unknown", reason="identity_mismatch")

    # ---- 空 middle：直接保护/全可见短输入的合法路径（非失败）----
    if T_mid == 0 or Hkv == 0:
        return _result("empty_middle")

    # ---- GQA 结构合法 ----
    if H % Hkv != 0:
        return _result("unknown", reason="gqa_head_mismatch")

    # ---- anchor 选取（只依合法位置）----
    positions = anchor_positions(T_mid)
    split = ref_split(T_mid)
    cost["anchor_pos_ops"] = len(positions)
    n_anchor_N = sum(1 for p in positions if p >= split)
    n_anchor_F = len(positions) - n_anchor_N
    if n_anchor_N < 2 or n_anchor_F < 2:
        # 非空区 anchor < 2 → unknown（禁止把没采样区当零重要性）
        return _result("unknown", reason="anchor_short",
                       n_anchor=len(positions), n_anchor_N=n_anchor_N,
                       n_anchor_F=n_anchor_F, split=split)

    # ---- R / gather / 投影（即时投影，不缓存，成本全记）----
    R = make_projection(seq_id, layer_idx, D)          # [D, 8]
    cost["rng_elems"] = D * FEATURE_DIM
    idx = torch.tensor(positions, dtype=torch.long)
    k_anchor = k_mid[idx].to(torch.float64)            # [n, Hkv, D] 非连续
    cost["gather_elems"] = len(positions) * Hkv * D
    k_proj = torch.einsum("nhd,de->nhe", k_anchor, R)  # [n, Hkv, 8]
    cost["k_proj_macs"] = len(positions) * Hkv * D * FEATURE_DIM
    q_proj = q.to(torch.float64) @ R                   # [H, 8]
    cost["q_proj_macs"] = H * D * FEATURE_DIM

    # ---- 每 Q-head 用其 KV 组的 anchor 打点积，温度 sqrt(原D) 不是 sqrt(8) ----
    G = H // Hkv
    k_heads = k_proj.repeat_interleave(G, dim=1)       # [n, H, 8]
    logits = torch.einsum("he,nhe->hn", q_proj, k_heads) / math.sqrt(D)
    cost["dot_macs"] = H * len(positions) * FEATURE_DIM
    cost["softmax_elems"] = H * len(positions)

    # ---- p_hi：对全部 anchor 共同减 max 后 softmax ----
    logits = logits - logits.amax(dim=-1, keepdim=True)
    p = torch.softmax(logits, dim=-1)                  # [H, n]

    is_N = torch.tensor([p_ >= split for p_ in positions],
                        dtype=torch.bool)
    d_N = p[:, is_N].sum(dim=-1) / n_anchor_N          # [H]
    d_F = p[:, ~is_N].sum(dim=-1) / n_anchor_F
    s_h = torch.log((d_N + S_H_EPS) / (d_F + S_H_EPS))  # float64，dtype 已记

    if not bool(torch.isfinite(s_h).all()):
        return _result("unknown", reason="nonfinite",
                       n_anchor=len(positions), n_anchor_N=n_anchor_N,
                       n_anchor_F=n_anchor_F, split=split)

    # ---- GQA 聚合：先组内等权、再组间等权（不许组大小加权）----
    s = aggregate_s_h(s_h, [G] * Hkv)
    if not math.isfinite(s):
        return _result("unknown", reason="nonfinite",
                       n_anchor=len(positions), n_anchor_N=n_anchor_N,
                       n_anchor_F=n_anchor_F, split=split)

    # ---- 诊断（不作为真实 mass，只记录）----
    n_N_tokens = T_mid - split
    n_F_tokens = split
    a_N_mean = float((n_N_tokens * d_N).mean())
    a_F_mean = float((n_F_tokens * d_F).mean())
    dev = (s_h - s).abs()
    worst_i = int(dev.argmax())

    return _result(
        "ok",
        s=s,
        s_h=[float(x) for x in s_h.tolist()],
        n_mid=T_mid,
        n_anchor=len(positions),
        n_anchor_N=n_anchor_N, n_anchor_F=n_anchor_F,
        anchor_positions=list(positions), split=split,
        n_N_tokens=n_N_tokens, n_F_tokens=n_F_tokens,
        n_q_heads=H, n_kv_heads=Hkv, group_size=G,
        # 区域大小修正 A_r,h=|r|·d_r,h 仅作诊断（绝不是实际 attention mass）
        a_diag={"A_N_mean": a_N_mean, "A_F_mean": a_F_mean,
                "note": "diagnostic only, not real mass"},
        # 保最差 head 诊断
        worst_head={"idx": worst_i, "s_h": float(s_h[worst_i]),
                    "dev_from_s": float(dev[worst_i])},
    )


# ================================================================ M2 三档规则

def classify_profile(s):
    """三档规则（§2.1 条 4，阈值固定不按 method 改）：
    s<-ln2→P_F；s>ln2→P_N；其余（含恰 ±ln2）→P_C。"""
    if s < -THRESH_LN2:
        return "P_F"
    if s > THRESH_LN2:
        return "P_N"
    return "P_C"


# ================================================================ M3 预算编译器

def derive_gamma(denom, target):
    """γ 反算 + 浮点 floor 回译验证（§2.1 条 5）。

    规范目标是整数 Tn；γ=target/denom 只是等价表示。IEEE 边界下
    floor(denom·γ) 可能比 target 少 1（如 (49,1)/(22,15)/(23,13) 实测
    对）——此时用 nextafter 向上微调 γ 直到回译命中，**不少 1**；
    bump 后 γ 仍 ≤ 1（target ≤ denom 时微调不会跨过 1，因 bump 步长
    是最小 ulp；断言兜底）。返回 (gamma, bumped)。
    """
    if denom <= 0:
        raise ValueError(f"denom 必须 > 0，得 {denom}")
    if target > denom:
        # 数学上 γ>1：由调用方在进本函数前拦截（gamma_gt_1），
        # 到达此处说明调用方漏检——fail loudly 而非静默给 γ>1。
        raise ValueError(f"target({target}) > denom({denom})：γ>1 应走回退")
    gamma = target / denom
    bumped = False
    it = 0
    while math.floor(denom * gamma) < target and it < MAX_BUMP_ITERS:
        gamma = math.nextafter(gamma, math.inf)
        bumped = True
        it += 1
    if math.floor(denom * gamma) != target:
        raise ValueError(f"floor 回译无法命中 target={target} "
                         f"(denom={denom}, γ={gamma!r})")
    if target <= denom and gamma > 1.0:
        # 保险：target==denom 时 γ 应恰 1.0；bump 不该跨过（ulp 步长）
        raise ValueError(f"bump 把 γ 推过 1：target={target} denom={denom} "
                         f"γ={gamma!r}")
    return gamma, bumped


def _region_sizes(alpha, m_len, bs):
    """α 切 M 尾 near（§2.2）。镜像生产 near_len_dyn=max(bs,int(α·M))
    （int 截断），再 clamp 到 M——生产对小 mid 可过度预留（bs 下限吞掉
    整个 mid），编译器如实 clamp（容量诚实口径，差异点显式注明）。
    返回 (N_len, F_len)（token 数）。"""
    n_len = max(bs, int(alpha * m_len))
    n_len = min(n_len, m_len)
    return n_len, m_len - n_len


def _compile_common(alpha, beta, k1, k2, bs, n_valid, n_protected):
    """§2.2 公共预算量（所有档位/固定档同源）：
    Bn=max(1,round(K1β))（生产 int(round()) 同口径，banker's 舍入）；
    Bf=K1−Bn；U=|U_q| 因果 valid keys；P=prefix∪SWA 去重计数；
    Kmid=max(0,min(K2,|U|)−|P|)；M=|U|−|P|。ops 计数为真实执行的
    算术/比较操作数（成本记账，非估算常数）。"""
    ops = 0
    Bn = max(1, int(round(k1 * beta))); ops += 3
    Bf = k1 - Bn; ops += 1
    Kmid = max(0, min(k2, n_valid) - n_protected); ops += 4
    m_len = max(0, n_valid - n_protected); ops += 2
    protected_ge_k2 = n_protected >= k2; ops += 1
    return Bn, Bf, Kmid, m_len, protected_ge_k2, ops


def compile_tier(tier, *, k1, k2, bs, n_valid, n_protected,
                n_prefix=None, n_swa=None):
    """M3：三档之一的编译（§2.2 精确预算 + §2.1 条 5 γ 反算）。

    返回 (ok, info)。ok=False 时 info["reason"] ∈
      {"struct", "gamma_gt_1", "floor_roundtrip",
       "near_capacity", "far_capacity"}。
    ok=True 时 info 含整数预算 Bn/Bf/Kmid/Tn/Tf、γ（浮点表示）、
    gamma_bumped、区域与容量（候选 IDs 摘要=token 区间）。

    容量语义（「实际候选容量不足 → profile 不可用」）：
      near 承诺配额 Tn_target 必须 ≤ Cn=min(Bn·b, |N|)（L1 near 块
      预算与 α 切出的 near 区宽的交集——选中页的实际有效唯一容量）；
      far 同理 Tf ≤ Cf=min(Bf·b, |F|)。padding/空池/交叠不能用
      Bn·b 蒙混（§2.2）。
    """
    if tier not in TIER_PARAMS:
        return False, {"reason": "struct", "detail": f"unknown tier {tier}"}
    tp = TIER_PARAMS[tier]
    if k1 <= 0 or bs <= 0 or n_valid < 0 or n_protected < 0:
        return False, {"reason": "struct"}

    Bn, Bf, Kmid, m_len, ge_k2, ops = _compile_common(
        tp["alpha"], tp["beta"], k1, k2, bs, n_valid, n_protected)
    info = {"tier": tier, "ops": ops}
    denom = Bn * bs; ops += 1
    Tn_target = int(round(tp["rho"] * Kmid)); ops += 2

    # γ>1：Tn_target 超过 near L1 候选上限 Bn·b → 该档不可用
    if Tn_target > denom:
        return False, {"reason": "gamma_gt_1",
                       "gamma_raw": Tn_target / denom,
                       "Tn_target": Tn_target, "denom": denom,
                       "Bn": Bn, "Bf": Bf, "Kmid": Kmid, "ops": ops + 2}
    gamma, bumped = derive_gamma(denom, Tn_target); ops += 6
    Tn = min(Tn_target, Kmid); ops += 1
    Tf = Kmid - Tn; ops += 1

    # 实际候选容量（诚实口径）
    n_len, f_len = _region_sizes(tp["alpha"], m_len, bs); ops += 4
    cn = min(denom, n_len); ops += 1
    cf = min(Bf * bs, f_len); ops += 2
    if Tn > cn:
        return False, {"reason": "near_capacity", "Tn_target": Tn_target,
                       "gamma": gamma, "gamma_bumped": bumped,
                       "Cn": cn, "N_len": n_len, "denom": denom,
                       "Bn": Bn, "Bf": Bf, "Kmid": Kmid, "ops": ops + 2}
    if Tf > cf:
        return False, {"reason": "far_capacity", "Tf": Tf, "Cf": cf,
                       "gamma": gamma, "gamma_bumped": bumped,
                       "F_len": f_len, "Bn": Bn, "Bf": Bf,
                       "Kmid": Kmid, "ops": ops + 2}

    info.update({
        "Bn": Bn, "Bf": Bf, "Kmid": Kmid,
        "Tn_target": Tn_target, "Tn": Tn, "Tf": Tf,
        "gamma": gamma, "gamma_bumped": bumped,
        "N_len": n_len, "F_len": f_len, "Cn": cn, "Cf": cf,
        "alpha": tp["alpha"], "beta": tp["beta"], "rho": tp["rho"],
        "protected_ge_k2": ge_k2,
        "ops": ops,
    })
    _attach_ranges(info, n_prefix, n_swa, n_valid, n_protected,
                   m_len, n_len, f_len)
    return True, info


def compile_fixed(method, *, k1, k2, bs, n_valid, n_protected,
                 n_prefix=None, n_swa=None):
    """M3：固定参数档编译（回退链末端）。

    与三档不同：固定档**不承诺 ρ**，配额自然截断到池宽（生产
    compute_mask 的 k=min(quota, pool width) 同语义），因此结构合法
    即可用；容量截断如实记录（Tn_effective/Tf_effective < 配额时即
    截断证据，不蒙混）。method 未通过正确性验证（不在 registry）→
    不可用（reason=method_not_registered）。
    """
    m = canonical_method(method)
    if m not in METHOD_FIXED_TIER:
        return False, {"reason": "method_not_registered", "method": str(method)}
    alpha, beta, gamma = METHOD_FIXED_TIER[m]["params"]
    if not (gamma is not None and 0 < gamma <= 1):
        return False, {"reason": "struct", "detail": f"gamma={gamma!r}"}
    if k1 <= 0 or bs <= 0 or n_valid < 0 or n_protected < 0:
        return False, {"reason": "struct"}

    Bn, Bf, Kmid, m_len, ge_k2, ops = _compile_common(
        alpha, beta, k1, k2, bs, n_valid, n_protected)
    denom = Bn * bs; ops += 1
    tn_quota = min(math.floor(denom * gamma), Kmid); ops += 3
    tf_quota = Kmid - tn_quota; ops += 1
    n_len, f_len = _region_sizes(alpha, m_len, bs); ops += 4
    cn = min(denom, n_len); ops += 1
    cf = min(Bf * bs, f_len); ops += 2
    tn_eff = min(tn_quota, cn); ops += 1
    tf_eff = min(tf_quota, cf); ops += 1
    info = {
        "Bn": Bn, "Bf": Bf, "Kmid": Kmid,
        "Tn": tn_eff, "Tf": tf_eff,
        "Tn_quota": tn_quota, "Tf_quota": tf_quota,
        "quota_truncated": (tn_eff < tn_quota) or (tf_eff < tf_quota),
        "gamma": gamma, "gamma_bumped": False,
        "N_len": n_len, "F_len": f_len, "Cn": cn, "Cf": cf,
        "alpha": alpha, "beta": beta, "rho": None,
        "protected_ge_k2": ge_k2,
        "method": m, "verification": METHOD_FIXED_TIER[m]["verification"],
        "ops": ops,
    }
    _attach_ranges(info, n_prefix, n_swa, n_valid, n_protected,
                   m_len, n_len, f_len)
    return True, info


def _attach_ranges(info, n_prefix, n_swa, n_valid, n_protected,
                   m_len, n_len, f_len):
    """候选 IDs 摘要（§6 decision schema 的「保护与候选IDs摘要」）：
    near/far 的 token 区间（0 基全序列坐标）。mid 区间按 [n_prefix,
    U−n_swa) 推导；prefix∪SWA 有去重叠（n_prefix+n_swa≠|P|）或
    区间参数缺失时区间记 None（计数仍准确，不臆造区间）。"""
    if (n_prefix is None or n_swa is None
            or n_prefix + n_swa != n_protected):
        info["near_range"] = None
        info["far_range"] = None
        info["range_note"] = ("unavailable: prefix/SWA dedup overlap "
                             "or missing n_prefix/n_swa")
        return
    far_lo = n_prefix
    far_hi = far_lo + f_len
    near_lo = far_hi
    near_hi = near_lo + n_len
    info["far_range"] = [far_lo, far_hi]
    info["near_range"] = [near_lo, near_hi]


# ================================================================ 全链编排

def decide(q, k_mid, seq_id, layer_idx, method, *, k1, k2, bs,
           n_valid, n_protected, n_prefix=None, n_swa=None,
           phase="prefill", state_epoch=0, identity_check=None):
    """M1→M2→M3 全链：返回单条 decision（可 JSON 序列化，字段闭包
    按 GPT §6：seq/layer/phase/state_epoch/features/profile/整数预算/
    保护与候选IDs摘要/attempts(fallback)/cost）。

    回退链（§2.1 条 5，至多两次重试）：
      请求档 →（≠P_C 时）同 method 的 P_C →（method 在 registry 时）
      该 method 固定参数档 (.25,.125,.625)（绝不切换到 mavg 选择器）
      → 仍不合法则 unsupported。
    合法性失败（anchor<2/非有限/身份不符）→ 中性档 P_C + unknown；
    空 middle → 合法路径（empty_middle）+ 中性档。

    v0 边界：seq-once 条件化基线（§2.2：prefill 后预测一次、decode
    复用档位、每步重编译整数预算）是 A5 部署语义——本函数只提供
    单点决策，缓存/复用由集成层包。

    几何同源绑定（GPT A1 验收反馈 2026-10-11，fail-closed）：k_mid
    实际行数必须 == max(0, n_valid − n_protected)——特征切片与预算
    编译共享同一合法前缀，不一致直接抛 ValueError，绝不静默截断/钳制
    （旧 dryrun 入口 n_valid=32768 覆盖 meta.S=16957、切片被 Python
    钳制漏切 SWA 尾段，同一决策记录里特征与预算混用两套几何，即本
    校验要堵的缺陷）。controller 从 k_mid 行数+保护集即可推导唯一
    几何，故采用「绑定」：以 k_mid 实际行数校验入参一致，而非各算各的。
    """
    t_mid = 0 if k_mid is None else int(k_mid.shape[0])
    expected_mid = max(0, int(n_valid) - int(n_protected))
    if t_mid != expected_mid:
        raise ValueError(
            f"[E124A-GEOM-MISMATCH] k_mid 行数 {t_mid} != "
            f"n_valid−n_protected = {int(n_valid)}−{int(n_protected)} "
            f"= {expected_mid}：特征切片与预算编译必须共享同一合法前缀"
            f"（fail-closed；切片越界静默钳制/两套几何混用一律拒绝）")
    t0 = time.perf_counter()
    method_c = canonical_method(method)
    feat = compute_features(q, k_mid, seq_id, layer_idx, phase=phase,
                            state_epoch=state_epoch,
                            identity_check=identity_check)
    fstat = feat["status"]

    # ---- M2：档位（合法性失败 → 中性档）----
    neutral = False
    if fstat == "ok":
        requested = classify_profile(feat["s"])
    else:
        # empty_middle（合法路径）与 unknown（合法性失败）都无有效 s
        # → 中性档 P_C（§2.1 条 3）
        requested = NEUTRAL_PROFILE
        neutral = True

    # ---- M3：编译 + 回退链（初始 + 至多两次重试）----
    attempts = []
    applied = None
    applied_info = None
    chain = [("tier", requested)]
    if requested != "P_C":
        chain.append(("tier", "P_C"))
    chain.append(("fixed", method_c))

    for step, name in chain[:3]:
        if step == "tier":
            ok, info = compile_tier(name, k1=k1, k2=k2, bs=bs,
                                    n_valid=n_valid,
                                    n_protected=n_protected,
                                    n_prefix=n_prefix, n_swa=n_swa)
            label = name
        else:
            ok, info = compile_fixed(name, k1=k1, k2=k2, bs=bs,
                                     n_valid=n_valid,
                                     n_protected=n_protected,
                                     n_prefix=n_prefix, n_swa=n_swa)
            label = "fixed_tier"
        attempts.append({"step": label, "ok": bool(ok),
                         "reason": None if ok else info.get("reason"),
                         "cost": int(info.get("ops", 0))})
        if ok:
            applied = label
            applied_info = info
            break

    retries = max(0, len(attempts) - 1)
    status = fstat
    if applied is None:
        status = "unsupported"

    feat_cost = dict(feat["cost"])
    feat_cost["retries"] = retries
    feat_cost["wall_ms"] = max(0.0, (time.perf_counter() - t0) * 1000.0)

    decision = {
        "controller_version": CONTROLLER_VERSION,
        "method": method_c,
        "seq_id": seq_id,
        "layer_idx": int(layer_idx),
        "phase": phase,
        "state_epoch": int(state_epoch),
        "identity": feat["identity"],
        "features": {k: v for k, v in feat.items() if k != "cost"},
        "status": status,
        "profile": {
            "requested": requested,
            "applied": applied,
            "neutral": neutral or fstat != "ok",
            # 恰 ±ln2 归 P_C；阈值固定 ±ln2
            "thresholds": {"P_F_below": -THRESH_LN2,
                           "P_N_above": THRESH_LN2},
        },
        "budget": applied_info,       # None = unsupported
        "protection": {
            "n_valid": n_valid, "n_protected": n_protected,
            "n_prefix": n_prefix, "n_swa": n_swa,
            # 保护 ≥ K2 的明确例外（§2.2：不许静默；记录而非失败）
            "protected_ge_k2": bool(n_protected >= k2),
        },
        "attempts": attempts,          # 初始 + 每次回退（原因+成本全记录）
        "cost": feat_cost,
    }
    if applied_info is not None:
        decision["protection"]["protected_ge_k2"] = bool(
            applied_info.get("protected_ge_k2", n_protected >= k2))
    return decision


# ================================================================ 落盘产物

def controller_spec():
    """controller_spec.json 内容（§6 最少新增产物；全可序列化、
    JSON 往返无损——参数/规则冻结快照，贯穿 producer 与 consumer）。"""
    return {
        "version": CONTROLLER_VERSION,
        "feature_dim": FEATURE_DIM,
        "n_anchors_max": N_ANCHORS_MAX,
        "s_h_eps": S_H_EPS,
        "thresholds": {"P_F_below": -THRESH_LN2, "P_N_above": THRESH_LN2},
        "threshold_note": "s<-ln2→P_F; s>ln2→P_N; 恰±ln2→P_C；固定不按method改",
        "temp_rule": "sqrt(D)",
        "neutral_profile": NEUTRAL_PROFILE,
        "tier_params": {k: [v["alpha"], v["beta"], v["rho"]]
                        for k, v in TIER_PARAMS.items()},
        "tier_params_note": "ρ=希望给near的middle L2比例，不是重新定义γ；"
                            "总预算固定下P_N压缩far，不称更保守",
        "gqa_aggregation": "s=(1/Hkv)Σ_g[(1/|Q_g|)Σ_{h∈Q_g}s_h]"
                           "（先组内等权再组间等权，不许组大小加权）",
        "ref_boundary": "middle最后1/4=N_ref，其余F_ref（先于α定义）",
        "seed_scheme": "sha256('e124a|{seq_id}|{layer_idx}')前8字节"
                       "→torch.Generator；R∈{±1/√8}^{D×8}",
        "geometry_contract": "decide(): k_mid行数必须==max(0,n_valid-"
                             "n_protected)，不一致抛 ValueError"
                             "（特征与预算同源绑定，GPT A1 验收 2026-10-11）",
        "budget_formulas": {
            "Bn": "max(1,round(K1*β))", "Bf": "K1-Bn",
            "Kmid": "max(0,min(K2,|U|)-|P|)",
            "Tn_target": "round(ρ*Kmid)", "gamma": "Tn_target/(Bn*b)",
            "Tn": "min(Tn_target,Kmid)", "Tf": "Kmid-Tn",
            "rounding": "Python round = banker's（与生产 int(round()) 同口径）",
        },
        "fallback_chain": ["请求档", "同method的P_C",
                           "该method固定参数档(.25,.125,.625)"
                           "（绝不切换mavg选择器）", "unsupported"],
        "max_retries": 2,
        "method_fixed_tier": {m: {"params": list(v["params"]),
                                  "verification": v["verification"]}
                              for m, v in METHOD_FIXED_TIER.items()},
        "method_fixed_tier_note": "只列已通过正确性验证的组合；"
                                  "A3 入筛前按 CPU/小输入门禁验收后更新",
        "not_implemented": ["M0 身份/状态模块（v0 以 identity_check 参数代）",
                            "M5 far-only D′ gate",
                            "seq-once 条件化缓存（A5 部署语义）"],
    }


def write_controller_spec(path):
    spec = controller_spec()
    with open(path, "w") as f:
        json.dump(spec, f, ensure_ascii=False, indent=2)
    return spec


class DecisionLog:
    """per_seq_layer_decisions.jsonl 追加式落盘（每行一条 decision）。"""

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self._f = open(path, "a")

    def append(self, decision):
        self._f.write(json.dumps(decision, ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self):
        self._f.close()
