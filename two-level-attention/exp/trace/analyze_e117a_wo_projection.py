# -*- coding: utf-8 -*-
# E117a —— W_O 投影逐层潜力重判（GPT 2026-10-09 Standalone 建议修正契约）。
#
# 背景：既有 P0' 判决（p0p_perlayer_potential.json，potential-low，gap_rel 中位
#   7.24%）用**未投影逐 head MSE 代理**——两 head 误差可经 W_O 抵消或放大，
#   逐头独立最小化 ≠ 全层目标。本脚本用真实 W_O 投影目标重放同一 trace 重判。
#
# 目标函数（GPT advice §4.2）：
#   Δo_l(q,c) = W_O,l · concat_h(ŷ_h − y_h)   （真实模型布局，head 拼接序
#     j = h_kv·G + g，与 HF Qwen3 attention 输出 reshape [.., H·Dv] → o_proj 一致；
#     W_O 存储为 [hidden, H·Dv]，Linear 语义 o = concat @ W^T → Δo = W @ Δconcat）
#   E_l(c) = task/doc 平衡平均 [ ‖Δo_l‖² / (a_l² + ε) ]
#   a_l 仅用**校准** dense 输出算出（固定层尺度，查看确认集前冻结）；
#   ε = 1e-9 固定常数（具输出平方量纲，不随候选/层重估分母）；
#   零尺度层单列处理；非有限目标标 failed 而非自动胜出。
#
# 协议要点：
#   - 复用 analyze_p0p_perlayer_potential.py 的候选域构建（338 原始 → 闭式剔除
#     198 γ 悬崖/非法约束臂 → 140 枚举，G* 豁免）、查询采样、trace 加载——
#     原脚本**不许改**（旁路新脚本 import 复用）。
#   - 校准/确认文档级分离（8 样本 = 4 任务 × 2 文档）：校准 5 文档
#     {gov_0, gov_1, hotpot_0, narr_0, pr_0}，确认 3 文档 {hotpot_1, narr_1, pr_1}；
#     tie 规则（相对容差 1e-12，平局取 (α,β,γ) 字典序最小）在查看确认结果前冻结。
#   - 判决门（与 P0' 同门）：8 代表层 gap_rel 中位 <10% → 计算器维持不做
#     （NO-GO，平坦性第五证）；>10% → 升级 8 层小试（GO）。
#   - median 约定修正：标准偶数中位（中间两值平均）——P0' 用 len//2 上取
#     （GPT 批评项之一），本脚本如实换用标准中位并注明。
#   - 同时报绝对误差（未归一化 ‖Δo‖²）、文档尾部/中段误差、每任务结果、
#     确认集 regret（校准冻结表 c*_l 在确认集上的表现）。
#   - W_O 权重按层从 Qwen3-8B safetensors 索引加载（o_proj），记录 shard +
#     fp32 字节 sha256（manifest 纪律）。
#
# 纯 CPU 零 GPU（纪律：不设 cuda device，不碰 nvidia-smi 上跑的进程）。
# 干跑：--dry-run 用合成 trace + 合成 W_O 跑通全流程与全部自检。
#
# 用法：
#   python3 analyze_e117a_wo_projection.py --dry-run
#   python3 analyze_e117a_wo_projection.py            # 真实 trace + 真实 W_O（CPU）
import argparse
import hashlib
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import analyze_p0p_perlayer_potential as p0p   # noqa: E402  原脚本只复用不改

MODEL_DIR_DEFAULT = ("/mnt/dolphinfs/ssd_pool/docker/user/"
                     "hadoop-mlp-ckpt/sunyueqing/Qwen3-8B")
EPS = 1e-9          # ε：固定常数，输出平方量纲，不随候选/层重估
N_TAIL_DEF = 256    # 文档尾部 query 窗口（对齐 collect_trace_lb_v.py N_TAIL）
TIE_RTOL = 1e-12    # tie 相对容差（冻结）：|E−E_best|/E_best ≤ 容差 → 平局

# 校准/确认文档级切分（冻结规则：每任务首文档进校准，gov_report 双文档进校准
# 凑 5；确认集留 hotpot/narr/pr 各 1 文档——文档级独立，非同文档切 query）
CAL_SAMPLES = ["lb_gov_report_0", "lb_gov_report_1", "lb_hotpotqa_0",
               "lb_narrativeqa_0", "lb_passage_retrieval_en_0"]
CONF_SAMPLES = ["lb_hotpotqa_1", "lb_narrativeqa_1",
                "lb_passage_retrieval_en_1"]

# 参照臂（除 G* 外记录海选冠军 mavg(.25,.125,.625) 的 E，供审阅）
CHAMP_REF = (0.25, 0.125, 0.625)


# ---------------- W_O 权重加载 ---------------- #

def load_wo_weights(model_dir, layers):
    """按层从 safetensors 加载 o_proj 权重（只开需要的 shard、只读需要的 key）。

    返回 {layer: {"W": fp32 [hidden, H*Dv], "shard", "dtype_src", "sha256"}}。
    sha256 = fp32 张量字节哈希（manifest 身份纪律）。
    """
    idx_path = os.path.join(model_dir, "model.safetensors.index.json")
    wmap = json.load(open(idx_path))["weight_map"]
    need = {f"model.layers.{l}.self_attn.o_proj.weight": l for l in layers}
    missing = [k for k in need if k not in wmap]
    assert not missing, f"o_proj 键缺失: {missing}"
    from safetensors.torch import safe_open
    out = {}
    by_shard = {}
    for key, l in need.items():
        by_shard.setdefault(wmap[key], []).append((key, l))
    for shard, items in sorted(by_shard.items()):
        with safe_open(os.path.join(model_dir, shard), framework="pt") as f:
            for key, l in items:
                w = f.get_tensor(key)          # [hidden, H*Dv]，bf16 原精度
                wf = w.to(torch.float32).contiguous()
                out[l] = {
                    "W": wf, "shard": shard,
                    "dtype_src": str(w.dtype),
                    "sha256": hashlib.sha256(wf.numpy().tobytes()).hexdigest(),
                }
    return out


def gen_synth_wo(layers, seed=20261009, hidden=4096):
    """干跑用合成 W_O（随机 [hidden, H*Dv] fp32，同真实 shape）。"""
    g = torch.Generator().manual_seed(seed)
    return {l: {"W": torch.randn(hidden, hidden, generator=g),
                "shard": "synthetic", "dtype_src": "fp32(synth)",
                "sha256": "synthetic"} for l in layers}


# ---------------- 目标函数核心（GQA concat + W_O 投影） ---------------- #

def head_concat(dy):
    """[Hkv, G, Dv] → [H*Dv]：head 拼接序 j = h_kv·G + g。

    HF Qwen3 attention：attn_output [B,H,seq,Dv] → [B,seq,H·Dv] → o_proj，
    q-head j 使用 kv-head j//G（repeat_kv 语义）——与 p0p 的
    q.reshape(Hkv,G,D) 分组口径一致。
    """
    return dy.reshape(-1)


def project_delta(W, dy):
    """Δo = W_O @ concat_h(Δy_h)（Linear o = concat @ W^T → Δo = W @ Δconcat）。"""
    return W @ head_concat(dy)


# ---------------- 自检（红绿） ---------------- #

def selfcheck_projection_identity(seed=20261009):
    """红 1：投影恒等式数值校验（小随机张量，手算 vs 代码路径逐位相等）。

    手算参照：显式按 head 序 j=h·G+g 把 Δy[h,g] 写进 concat 槽位，再手写
    矩阵-向量乘（逐元素循环）——与 project_delta 逐位比对。
    """
    g = torch.Generator().manual_seed(seed)
    Hkv, G, Dv = 3, 2, 2
    out_dim = 6
    W = torch.randn(out_dim, Hkv * G * Dv, generator=g)
    dy = torch.randn(Hkv, G, Dv, generator=g)
    # 手算 concat：head j = h*G+g → 槽 [j*Dv:(j+1)*Dv] = dy[h,g]
    x_ref = torch.zeros(Hkv * G * Dv)
    for h in range(Hkv):
        for j in range(G):
            x_ref[(h * G + j) * Dv:(h * G + j + 1) * Dv] = dy[h, j]
    # 逐位（同 op 参照）：手排 concat vs reshape 路径；手排后投影 vs 代码路径
    bit_concat = bool(torch.equal(head_concat(dy), x_ref))
    o_ref = W @ x_ref
    o_code = project_delta(W, dy)
    bit_proj = bool(torch.equal(o_ref, o_code))
    # 独立元素级手算 matvec（不同求和序，浮点容差校验，防同 op 假阴性）
    o_loop = torch.zeros(out_dim)
    for r in range(out_dim):
        o_loop[r] = float((W[r] * x_ref).sum())
    loop_dev = float((o_loop - o_code).abs().max())
    return {"bitwise_concat": bit_concat, "bitwise_projection": bit_proj,
            "loop_matvec_max_abs_diff": loop_dev,
            "pass": bool(bit_concat and bit_proj and loop_dev < 1e-5)}


def selfcheck_gqa_permutation(seed=20261010):
    """红 2：GQA concat 顺序断言——permutation 翻转 head 归属时目标必须变化。

    ① (h_kv, g) 解释翻转（Δy.transpose(0,1) = 把 G 当 kv 维）→ concat 重排；
    ② head 块循环移位（concat 块 roll 1）。
    随机 W 下目标 ‖W·Δconcat‖² 必须改变（若不变说明布局信息丢失）。
    """
    g = torch.Generator().manual_seed(seed)
    Hkv, G, Dv = 4, 2, 16
    W = torch.randn(64, Hkv * G * Dv, generator=g)
    dy = torch.randn(Hkv, G, Dv, generator=g)
    o_a = float((project_delta(W, dy) ** 2).sum())
    o_b = float((project_delta(W, dy.transpose(0, 1)) ** 2).sum())  # 解释翻转
    x = head_concat(dy)
    blocks = x.view(Hkv * G, Dv)
    x_roll = torch.cat([blocks[-1:], blocks[:-1]], dim=0).reshape(-1)  # 块移位
    o_c = float(((W @ x_roll) ** 2).sum())
    ch_b = abs(o_a - o_b) > 0
    ch_c = abs(o_a - o_c) > 0
    return {"objective_correct_order": o_a, "objective_kv_g_swapped": o_b,
            "objective_head_blocks_rolled": o_c,
            "changed_on_swap": ch_b, "changed_on_roll": ch_c,
            "pass": bool(ch_b and ch_c)}


def selfcheck_nonfinite_guard():
    """红 3（附加）：非有限目标标 failed 而非自动胜出（NaN 不得当 -inf 最小）。"""
    E = {"c_bad": float("nan"), "c_inf": float("inf"), "c_worse": 5.0, "c_best": 3.0}
    c_star, failed = guarded_argmin(E)
    ok = (c_star == "c_best" and set(failed) == {"c_bad", "c_inf"})
    return {"picked": c_star, "failed": sorted(failed), "pass": bool(ok)}


def guarded_argmin(E, tie_rtol=TIE_RTOL):
    """冻结 tie 规则的 argmin：非有限标 failed；相对容差内平局取字典序最小。"""
    failed = [c for c, v in E.items() if not (v == v and abs(v) != float("inf"))]
    valid = {c: v for c, v in E.items() if c not in failed}
    if not valid:
        return None, failed
    best = min(valid.values())
    tied = [c for c, v in valid.items()
            if (v - best) <= tie_rtol * max(abs(best), 1e-300)]
    c_star = min(tied, key=lambda c: (float(c[0]), float(c[1]), float(c[2]))
                 if isinstance(c, tuple) else (0.0, 0.0, 0.0))
    return c_star, failed


# ---------------- 单 (layer, sample) 枚举 ---------------- #

def eval_layer_sample(k, v_all, q, qpos, qpos_sel, kept, indexer, W, S,
                      replay_check_cands):
    """逐 query 缓存 dense 量，逐候选 compute_mask → Δy → W_O 投影。

    返回：
      d2/d2_tail/d2_mid: {cand: Σ_q ‖Δo‖²}（全量/尾部/中段 query）
      n_q/n_tail/n_mid, o_sq（Σ_q ‖o_dense‖²，a_l 原料）
      replay_dev: identity Δy vs 直接 ŷ 的最大相对偏差（仅指定候选首个 query）
    """
    D = k.shape[-1]
    Hkv, H = k.shape[1], q.shape[1]
    Dv = v_all.shape[-1]
    G = H // Hkv
    assert H == Hkv * G, f"GQA 不整除: H={H} Hkv={Hkv}"
    assert W.shape == (W.shape[0], H * Dv), \
        f"W_O 形状 {tuple(W.shape)} 与 concat 维 {H * Dv} 不符"
    d2 = {c: 0.0 for c in kept}
    d2_tail = {c: 0.0 for c in kept}
    d2_mid = {c: 0.0 for c in kept}
    n_q = n_tail = n_mid = 0
    o_sq = 0.0
    replay_dev = None
    tail_lo = S - N_TAIL_DEF
    for qi in qpos_sel:
        S_q = int(qpos[qi]) + 1
        k_q = k[:S_q].unsqueeze(0)
        v_q = v_all[:S_q]
        cu = torch.tensor([0, S_q], dtype=torch.long)
        q_ids = torch.tensor([S_q - 1], dtype=torch.long)
        q_in = q[qi:qi + 1].unsqueeze(0)
        # ---- 候选无关量：索引 + 分数 + dense 输出（先算一次，跨候选复用）----
        index_dict = indexer.prepare_index(k_q, cu)
        score_dict = indexer.compute_score(q_in, q_ids, index_dict, D ** -0.5)
        if "avg" in (indexer.far_method, indexer.near_method):
            assert score_dict.get("score_coarse_avg") is not None, \
                "score_coarse_avg 缺失：avg 分支未生效（E109a bug 复现）"
        qr = q[qi].reshape(Hkv, G, D)
        s_full = torch.einsum("hgd,shd->hgs", qr, k[:S_q]) * (D ** -0.5)
        p_full = torch.softmax(s_full, dim=-1)
        y = torch.einsum("hgs,shd->hgd", p_full, v_q)      # [Hkv,G,Dv] dense
        o_dense = project_delta(W, y)                        # [hidden]
        o_sq += float((o_dense * o_dense).sum())
        n_q += 1
        if qpos[qi] >= tail_lo:
            n_tail += 1
        else:
            n_mid += 1
        is_tail = qpos[qi] >= tail_lo
        # ---- 逐候选：原地改 α/β/γ → compute_mask（真实链路）→ 恒等式 Δy → 投影 ----
        for c in kept:
            indexer.alpha, indexer.beta, indexer.gamma = c
            mask = indexer.compute_mask(q_ids, score_dict)  # [1,1,Hkv,S_q]
            m = mask[0, 0].unsqueeze(1)                     # [Hkv,1,S_q]
            p_sel = p_full * m
            mS = p_sel.sum(-1)                              # [Hkv,G]
            num_S = torch.einsum("hgs,shd->hgd", p_sel, v_q)
            mS_ = mS.unsqueeze(-1)
            w_ = (1.0 - mS).unsqueeze(-1)
            mu_S = num_S / mS_.clamp(min=1e-12)
            mu_bar = (y - num_S) / w_.clamp(min=1e-12)
            # 恒等式：y−ŷ = (1−mS)·(μS̄−μS)——Δy 必须带 (1−mS) 因子（红绿
            # 自测曾抓到漏乘 bug：diff 无 w 因子时 replay 偏差 ~0.65）
            diff = w_ * (mu_bar - mu_S)                     # = y − ŷ [Hkv,G,Dv]
            if c in replay_check_cands and replay_dev is None:
                # 恒等式 vs 直接 ŷ 对拍（首个 query，指定候选，独立 oracle）
                yhat = num_S / mS_.clamp(min=1e-12)
                diff_dir = y - yhat
                denom = diff_dir.abs().max().clamp(min=1e-20)
                replay_dev = float((diff - diff_dir).abs().max() / denom)
            do = project_delta(W, diff)                     # Δo = W_O @ concat(Δy)
            e = float((do * do).sum())
            d2[c] += e
            if is_tail:
                d2_tail[c] += e
            else:
                d2_mid[c] += e
        del index_dict, score_dict, p_full, y, s_full, k_q, v_q, p_sel, num_S
    return d2, d2_tail, d2_mid, n_q, n_tail, n_mid, o_sq, replay_dev


# ---------------- 聚合 ---------------- #

def task_of(name):
    # "lb_hotpotqa_1" → "hotpotqa"；干跑 "synth_0" → "synth"
    base = name[3:] if name.startswith("lb_") else name
    return base.rsplit("_", 1)[0]


def task_balanced(doc_vals):
    """doc_vals: {sample_name: value} → task/doc 平衡平均（先 doc 后 task）。"""
    by_task = {}
    for name, v in doc_vals.items():
        by_task.setdefault(task_of(name), []).append(v)
    task_means = {t: sum(vs) / len(vs) for t, vs in by_task.items()}
    return sum(task_means.values()) / len(task_means), task_means


def median_std(vals):
    """标准中位（偶数个取中间两值平均——修正 P0' 的 len//2 约定）。"""
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return None
    if n % 2:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


# ---------------- 主流程 ---------------- #

def main():
    ap = argparse.ArgumentParser(
        description="E117a W_O 投影逐层潜力重判（真实输出目标，纯 CPU）")
    ap.add_argument("--trace", default="/tmp/trace/qwen3-8b-v")
    ap.add_argument("--model-dir", default=MODEL_DIR_DEFAULT)
    ap.add_argument("--out", default=None)
    ap.add_argument("--gstar", default=",".join(map(str, p0p.DEF_GSTAR)))
    ap.add_argument("--far-method", default="avg")
    ap.add_argument("--near-method", default="avg")
    ap.add_argument("--layers", default=",".join(map(str, p0p.DEF_LAYERS)))
    ap.add_argument("--max-queries", type=int, default=6)
    ap.add_argument("--samples", default=None)
    ap.add_argument("--alphas", default=",".join(map(str, p0p.DEF_ALPHAS)))
    ap.add_argument("--betas", default=",".join(map(str, p0p.DEF_BETAS)))
    ap.add_argument("--gammas", default=",".join(map(str, p0p.DEF_GAMMAS)))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=20261009)
    ns = ap.parse_args()

    dry = ns.dry_run
    device = "cpu"                       # 纪律：纯 CPU 零 GPU，不设 cuda 选项
    gstar = tuple(float(x) for x in ns.gstar.split(","))
    assert len(gstar) == 3
    far_method, near_method = ns.far_method, ns.near_method
    out_path = ns.out or os.path.join(
        HERE, "results", "e117a_wo_projection_dryrun.json" if dry
        else "e117a_wo_projection.json")

    t0 = time.time()
    self_checks = {}

    # ---- 自检 1：恒等式（随机 p/v，复用 P0'）----
    resid = p0p.identity_selfcheck(device, ns.seed)
    self_checks["identity_random"] = {"max_residual": resid, "pass": resid < 1e-5}

    # ---- 自检 2/3/4：投影恒等式 / GQA 排列敏感 / 非有限 guard ----
    self_checks["wo_projection_identity"] = selfcheck_projection_identity(ns.seed)
    self_checks["gqa_concat_permutation"] = selfcheck_gqa_permutation(ns.seed + 1)
    self_checks["nonfinite_guard"] = selfcheck_nonfinite_guard()

    # ---- trace 发现 + 校准/确认切分 ----
    if dry:
        trace_dir = "/tmp/e117a_dryrun_trace"
        os.makedirs(trace_dir, exist_ok=True)
        sample_names = p0p.gen_synth_trace(trace_dir, ns.seed)
        layers = list(p0p.DRY_LAYERS)
        cal_names, conf_names = [sample_names[0]], [sample_names[1]]
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
        cal_names = [s for s in CAL_SAMPLES if s in sample_names]
        conf_names = [s for s in CONF_SAMPLES if s in sample_names]
        assert len(cal_names) + len(conf_names) == len(sample_names), (
            f"切分不完整: cal={cal_names} conf={conf_names} "
            f"all={sample_names}")

    sample_Ss = {}
    for name in sample_names:
        meta = json.load(open(os.path.join(trace_dir, name, "meta.json")))
        sample_Ss[name] = int(meta["S"])

    # ---- 候选域（复用 P0'：闭式剔除 + G* 豁免；剔除判据用全部样本 S）----
    if dry:
        alphas = betas = gammas = []
    else:
        alphas = [float(x) for x in ns.alphas.split(",")]
        betas = [float(x) for x in ns.betas.split(",")]
        gammas = [float(x) for x in ns.gammas.split(",")]
    raw, kept, excluded, gstar_flag, grid_note = p0p.build_domain(
        alphas, betas, gammas, gstar, sample_Ss, dry)
    champ = CHAMP_REF if CHAMP_REF in kept else None

    # ---- 代码路径自检（复用 P0'：TLI_DEBUG 对拍 + avg 分支生效）----
    self_checks["tli_debug_crosscheck"] = p0p.tli_debug_crosscheck(
        device, gstar, far_method, near_method)
    self_checks["avg_vs_minmax_mask_diff"] = p0p.avg_vs_minmax_diff_check(
        device, gstar, far_method, near_method)

    # ---- W_O 权重（真实模型 or 干跑合成）----
    if dry:
        wo = gen_synth_wo(layers, ns.seed)
        wo_note = "合成随机 W（干跑）"
    else:
        wo = load_wo_weights(ns.model_dir, layers)
        wo_note = f"真实 o_proj（{ns.model_dir}）"
    # 布局校验（真实模式）：W 形状 vs trace head 布局（用首层文件探针）
    probe = torch.load(os.path.join(trace_dir, sample_names[0],
                                    f"layer{layers[0]:02d}.pt"),
                       map_location="cpu", weights_only=False)
    H_probe, Hkv_probe = probe["q"].shape[1], probe["k"].shape[1]
    Dv_probe = probe["v"].shape[-1]
    del probe
    layout_ok = all(wo[l]["W"].shape[1] == H_probe * Dv_probe for l in layers)
    self_checks["wo_layout_validation"] = {
        "W_shape": list(wo[layers[0]]["W"].shape),
        "H": H_probe, "Hkv": Hkv_probe, "Dv": Dv_probe,
        "hidden": wo[layers[0]]["W"].shape[0],
        "concat_dim_match": layout_ok, "pass": layout_ok}

    # ---- 逐层枚举（每层 .pt 单独加载后释放，流式防 OOM）----
    indexer = p0p.make_indexer(far_method, near_method, *gstar,
                               layer_idx=layers[0])
    acc = {}   # layer -> sample -> {d2, d2_tail, d2_mid, n_q, n_tail, n_mid, o_sq}
    replay_devs = []
    for layer in layers:
        acc[layer] = {}
        for name in sample_names:
            lf = os.path.join(trace_dir, name, f"layer{layer:02d}.pt")
            if not os.path.isfile(lf):
                print(f"[warn] 缺层文件 {lf}，跳过", flush=True)
                continue
            d = torch.load(lf, map_location="cpu", weights_only=False)
            if "v" not in d:
                raise RuntimeError(f"{lf} 无 'v' 键——须含 V 的 trace")
            k = d["k"].float()
            v_all = d["v"].float()
            q = d["q"].float()
            qpos = d["qpos"].tolist()
            S = int(d["S"])
            qpos_sel = p0p.select_query_indices(qpos, S, ns.max_queries)
            if not qpos_sel:
                print(f"[warn] {name} L{layer}: 无合法 query 位置（S={S}），跳过")
                continue
            replay_cands = {tuple(gstar)} if (layer == layers[0]
                                              and name == sample_names[0]) else set()
            (d2, d2_tail, d2_mid, n_q, n_tail, n_mid,
             o_sq, rdev) = eval_layer_sample(
                k, v_all, q, qpos, qpos_sel, kept, indexer, wo[layer]["W"], S,
                replay_cands)
            if rdev is not None:
                replay_devs.append(rdev)
            acc[layer][name] = {"d2": d2, "d2_tail": d2_tail, "d2_mid": d2_mid,
                                "n_q": n_q, "n_tail": n_tail, "n_mid": n_mid,
                                "o_sq": o_sq}
            del k, v_all, q, d
        print(f"[L{layer}] done ({time.time() - t0:.1f}s)", flush=True)

    if replay_devs:
        self_checks["identity_on_replay"] = {
            "max_relative_deviation": max(replay_devs),
            "pass": max(replay_devs) < 1e-4}

    # ---- a_l：仅校准 dense 输出（冻结）；自检：冻结性 + 敏感性 ----
    a2_by_layer, zero_scale_layers = {}, []
    for layer in layers:
        if not acc[layer]:
            continue
        o_sq_cal = sum(acc[layer][n]["o_sq"] for n in cal_names
                       if n in acc[layer])
        n_cal = sum(acc[layer][n]["n_q"] for n in cal_names if n in acc[layer])
        a2 = o_sq_cal / max(n_cal, 1)
        a2_by_layer[layer] = a2
        if not (a2 > 0.0) or a2 != a2:
            zero_scale_layers.append(layer)
    # 冻结性自检：①确定性（同序重算逐位相等）②敏感性（把确认文档混进 a_l
    # 来源会改变 a_l —— 证明冻结是有约束力的，而非空洞）
    a2_re = {}
    for layer in layers:
        if not acc[layer]:
            continue
        o_sq_cal = sum(acc[layer][n]["o_sq"] for n in cal_names
                       if n in acc[layer])
        n_cal = sum(acc[layer][n]["n_q"] for n in cal_names if n in acc[layer])
        a2_re[layer] = o_sq_cal / max(n_cal, 1)
    a2_all = {}
    for layer in layers:
        if not acc[layer]:
            continue
        o_sq_all = sum(rec["o_sq"] for rec in acc[layer].values())
        n_all = sum(rec["n_q"] for rec in acc[layer].values())
        a2_all[layer] = o_sq_all / max(n_all, 1)
    sens = [abs(a2_all[l] - a2_by_layer[l]) / max(abs(a2_by_layer[l]), 1e-300)
            for l in a2_by_layer if l in a2_all]
    self_checks["a_l_freeze"] = {
        "determinism_bitwise": all(a2_by_layer[l] == a2_re[l]
                                   for l in a2_by_layer),
        "sensitivity_if_conf_leaked": {
            "max_rel_change": max(sens) if sens else None,
            "changed": bool(sens) and max(sens) > 1e-9},
        "note": ("a_l² 只对校准文档求和（查看确认结果前冻结）；若混入确认文档"
                 "会改变 a_l（敏感性>0 证明冻结有约束力）"),
        "pass": bool(a2_by_layer) and all(a2_by_layer[l] == a2_re[l]
                                          for l in a2_by_layer)
        and sens and max(sens) > 1e-9}

    # ---- E_l：task/doc 平衡平均 ‖Δo‖²/(a_l²+ε) ----
    def split_E(layer, names):
        """返回 {cand: E}，task 平衡（先 doc 内 query 平均，再 task 平均）。"""
        a2 = a2_by_layer[layer]
        out, per_task_c = {}, {}
        for c in kept:
            doc_vals, per_task = {}, {}
            for n in names:
                if n not in acc[layer] or acc[layer][n]["n_q"] == 0:
                    continue
                doc_vals[n] = (acc[layer][n]["d2"][c]
                               / acc[layer][n]["n_q"]) / (a2 + EPS)
                per_task[n] = doc_vals[n]
            if not doc_vals:
                continue
            e, task_means = task_balanced(doc_vals)
            out[c] = e
            per_task_c[c] = task_means
        return out, per_task_c

    per_layer = {}
    gap_rels_cal, gap_rels_conf, c_stars = [], [], []
    for layer in layers:
        if not acc[layer] or layer not in a2_by_layer:
            continue
        E_cal, task_cal = split_E(layer, cal_names)
        E_conf, task_conf = split_E(layer, conf_names)
        if not E_cal:
            continue
        c_star, failed = guarded_argmin(E_cal)
        Eg_cal = E_cal.get(tuple(gstar))
        Eg_conf = E_conf.get(tuple(gstar))
        Ec_conf = E_conf.get(c_star) if c_star is not None else None
        gap_cal = ((Eg_cal - E_cal[c_star]) / Eg_cal
                   if (Eg_cal is not None and Eg_cal > 0
                       and c_star is not None) else None)
        gap_conf = ((Eg_conf - Ec_conf) / Eg_conf
                    if (Eg_conf is not None and Eg_conf > 0
                        and Ec_conf is not None) else None)
        # 绝对误差（未归一化 mean ‖Δo‖² per query，校准集）
        def _abs_err(c):
            if c is None:
                return None
            tot = sum(acc[layer][n]["d2"][c] for n in cal_names
                      if n in acc[layer])
            nq = sum(acc[layer][n]["n_q"] for n in cal_names if n in acc[layer])
            return tot / max(nq, 1)
        # 尾部/中段（校准集，归一化口径）
        def _seg(c, seg):
            if c is None:
                return None
            num = den = 0.0
            for n in cal_names:
                if n not in acc[layer]:
                    continue
                rec = acc[layer][n]
                if rec[f"n_{seg}"] == 0:
                    continue
                num += (rec[f"d2_{seg}"][c] / rec[f"n_{seg}"]) / (a2_by_layer[layer] + EPS)
                den += 1
            return num / den if den else None
        top5 = sorted(E_cal.items(), key=lambda kv: kv[1])[:5]
        rec = {
            "role": ("control(早期层对照)" if layer not in p0p.REP_LAYERS
                     else "representative"),
            "a_l_sq_cal_frozen": a2_by_layer[layer],
            "zero_scale": layer in zero_scale_layers,
            "n_failed_cands": len(failed),
            "E_best_cal": E_cal[c_star] if c_star is not None else None,
            "c_star": list(c_star) if c_star is not None else None,
            "c_star_tag": p0p.cand_tag(c_star) if c_star is not None else None,
            "E_gstar_cal": Eg_cal,
            "gap_rel_cal": gap_cal,
            "E_cstar_conf": Ec_conf,
            "E_gstar_conf": Eg_conf,
            "gap_rel_conf": gap_conf,
            "abs_err_cal": {"c_star": _abs_err(c_star), "gstar": _abs_err(tuple(gstar))},
            "tail_mid_cal": {
                "c_star": {"tail": _seg(c_star, "tail"), "mid": _seg(c_star, "mid")},
                "gstar": {"tail": _seg(tuple(gstar), "tail"),
                          "mid": _seg(tuple(gstar), "mid")}},
            "per_task_cal_cstar": task_cal.get(c_star) if c_star is not None else None,
            "per_task_cal_gstar": task_cal.get(tuple(gstar)),
            "E_champ_ref_cal": E_cal.get(champ) if champ else None,
            "top5_cal": [{"cand": list(c), "tag": p0p.cand_tag(c), "E": e}
                         for c, e in top5],
        }
        per_layer[str(layer)] = rec
        if layer in p0p.REP_LAYERS and not rec["zero_scale"]:
            if gap_cal is not None:
                gap_rels_cal.append(gap_cal)
                c_stars.append(c_star)
            if gap_conf is not None:
                gap_rels_conf.append(gap_conf)

    median_cal = median_std(gap_rels_cal)
    median_conf = median_std(gap_rels_conf)
    verdict = None
    if median_cal is not None:
        # 判决门（与 P0' 同门，median 约定修正为标准偶数中位）
        go = median_cal > 0.10
        verdict = ("GO（升级 8 层小试）" if go
                   else "NO-GO（计算器维持不做，平坦性第五证）")

    # ---- 与旧 P0' 未投影口径对照 ----
    p0p_cmp = None
    p0p_path = os.path.join(HERE, "results", "p0p_perlayer_potential.json")
    if os.path.isfile(p0p_path) and not dry:
        old = json.load(open(p0p_path))
        p0p_cmp = {
            "objective_old": ("未投影逐 head MSE 代理（GPT 批评：两 head 误差"
                              "可经 W_O 抵消或放大，逐头独立最小化 ≠ 全层目标）"),
            "median_gap_rel_old_len2_rule": old["global"]["median_gap_rel"],
            "per_layer_gap_rel_old": {
                k: v["gap_rel"] for k, v in old["per_layer"].items()},
            "verdict_old": old["global"]["verdict"],
        }

    all_pass = all(v.get("pass", False) for v in self_checks.values())

    result = {
        "exp": "E117a W_O 投影逐层潜力重判（GPT Standalone 建议修正契约）",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "objective": ("Δo_l(q,c)=W_O,l·concat_h(ŷ_h−y_h)（head 序 j=h_kv·G+g，"
                      "W 存储转置以框架实际为准）；E_l(c)=task/doc 平衡平均"
                      "‖Δo‖²/(a_l²+ε)；a_l² 仅校准 dense 输出均值 ‖o‖²（冻结）；"
                      "ε=1e-9 固定（输出平方量纲，不随候选重估分母）"),
        "verdict_rule": ("8 代表层 gap_rel 中位（标准偶数中位）>10% → GO 升级"
                         " 8 层小试；否则 NO-GO 维持不做"),
        "config": {
            "trace_dir": trace_dir, "device": device, "dry_run": dry,
            "model_dir": None if dry else ns.model_dir,
            "wo_source": wo_note,
            "layers": layers, "samples": sample_names,
            "sample_S": sample_Ss, "max_queries_per_doc": ns.max_queries,
            "split": {"calibration": cal_names, "confirmation": conf_names,
                      "rule": ("文档级切分冻结：每任务首文档进校准 + gov_report "
                               "双文档进校准凑 5；确认 3 文档（hotpot/narr/pr）"
                               "——非同文档切 query")},
            "gstar": list(gstar), "far_method": far_method,
            "near_method": near_method,
            "champion_ref": list(CHAMP_REF) if champ else None,
            "epsilon": EPS, "tie_rtol_frozen": TIE_RTOL,
            "tie_rule": ("相对容差 1e-12 内平局取 (α,β,γ) 字典序最小（查看确认"
                         "结果前冻结）"),
            "median_rule": "标准偶数中位（中间两值平均；修正 P0' len//2 约定）",
            "budgets": {"bs": p0p.BS, "K1": p0p.K1, "K2": p0p.K2,
                        "cmp_ratio": p0p.CMP, "sink_blocks": p0p.SINK_BLOCKS,
                        "swa": p0p.SWA},
            "tail_window": N_TAIL_DEF,
            "seed": ns.seed,
        },
        "wo_weights": {str(l): {"shard": wo[l]["shard"],
                                 "dtype_src": wo[l]["dtype_src"],
                                 "sha256_fp32": wo[l]["sha256"],
                                 "shape": list(wo[l]["W"].shape)}
                       for l in layers},
        "domain": {
            "grid_note": grid_note,
            "n_raw": len(raw), "n_kept": len(kept), "n_excluded": len(excluded),
            "exclusion_rules": ("复用 P0'：γ 悬崖 far_budget=0 且 far_frac>0.25；"
                                "nt_near>ℓ_near；任一样本触发即全局剔除；G* 豁免"),
            "gstar_cliff_excluded": gstar_flag,
            "n_excluded_detail": len(excluded),
            "kept_candidates": [p0p.cand_tag(c) for c in kept],
        },
        "per_layer": per_layer,
        "zero_scale_layers": zero_scale_layers,
        "global": {
            "n_rep_layers": len(gap_rels_cal),
            "gap_rel_cal_by_layer": [round(g, 6) for g in gap_rels_cal],
            "median_gap_rel_cal": round(median_cal, 6) if median_cal is not None else None,
            "median_gap_rel_conf": (round(median_conf, 6)
                                    if median_conf is not None else None),
            "gap_rel_conf_by_layer": [round(g, 6) for g in gap_rels_conf],
            "layer_param_discreteness": {
                "distinct_c_star": len(set(map(tuple, c_stars))),
                "c_star_list": [list(c) for c in c_stars]},
            "verdict": verdict,
            "verdict_caveat": ("固定 dense trace 的层内代理（W_O 投影输出失真）；"
                               "NO-GO → 平坦性第五证（E75/E79a/E79b/P0'/E117a）；"
                               "GO 亦只授权 8 层小试，不等于全模型最优"),
        },
        "p0p_unprojected_comparison": p0p_cmp,
        "self_checks": self_checks,
        "all_self_checks_pass": all_pass,
        "wall_time_s": round(time.time() - t0, 1),
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(result, open(out_path, "w"), ensure_ascii=False, indent=1)

    # ---- 人类可读摘要 ----
    print(f"\n=== E117a W_O 投影逐层潜力重判 "
          f"({'DRY-RUN 合成' if dry else '真实 trace+W_O'}) "
          f"method=({far_method},{near_method}) G*={list(gstar)} ===")
    print(f"候选域：{grid_note} → 剔除 {len(excluded)} → 枚举 {len(kept)}"
          f"（G* 豁免，cliff_flag={gstar_flag}）")
    print(f"切分：cal={cal_names} | conf={conf_names}")
    for layer, rec in per_layer.items():
        def _p(x, spec="{:+.4%}"):
            return spec.format(x) if x is not None else "NA"
        print(f"  L{layer}({rec['role'][:5]}): E*(c*)={rec['E_best_cal']:.6f} "
              f"c*={rec['c_star']} | E(G*)cal={rec['E_gstar_cal']:.6f} "
              f"gap_cal={_p(rec['gap_rel_cal'])} | gap_conf={_p(rec['gap_rel_conf'])}"
              f" | abs‖Δo‖²*: {rec['abs_err_cal']['c_star']:.4g}"
              f" G*: {rec['abs_err_cal']['gstar']:.4g}")
    g = result["global"]
    print(f"代表层 gap_rel(cal) 中位 = {g['median_gap_rel_cal']}"
          f" → 判决：{g['verdict']}")
    print(f"确认集中位 gap = {g['median_gap_rel_conf']}"
          f"（校准冻结表 c*_l 在确认集的 regret 口径）")
    if p0p_cmp:
        print(f"旧 P0' 未投影口径中位 = {p0p_cmp['median_gap_rel_old_len2_rule']}"
              f"（{p0p_cmp['verdict_old']}）")
    for name, chk in self_checks.items():
        print(f"  [check] {name}: {'PASS' if chk.get('pass') else 'FAIL'}")
    print(f"ALL SELF-CHECKS: {'PASS' if all_pass else 'FAIL'}")
    print(f"saved -> {out_path} ({result['wall_time_s']}s)")
    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
