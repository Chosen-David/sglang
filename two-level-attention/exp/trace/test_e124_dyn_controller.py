#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E124a（S-T021 / A1）动态 controller v0 红绿测试。

被测模块：sparse_attn/indexer/dyn_controller.py（M1 公共特征 + M2 三档
规则 + M3 整数预算编译器，GPT Runtime 方案 §2.1/§2.2 权威规格）。
生产静态路径（tli_indexer.py）零改动——本套件只测新模块。

用例矩阵（每条对应任务书验收项）：

  T01 三档边界    s=±ln2 恰好归 P_C；±(ln2+ε) 归 P_N/P_F；阈值固定=ln2；
  T02 anchor     等距只依合法位置（m_len 决定，与内容无关）；不足 16 取全部；
                  N_ref=middle 最后 1/4 边界 split；
  T03 投影 R      每 seq×layer 固定 seed；元素 ±1/√8；同 (seq,layer) 逐位
                  可复现、异 layer 不同；
  T04 GQA 聚合   Q-head 数 ≠ KV 组数反例（组大小 [2,1]）：等权（先组内
                  等权再组间等权）≠ 组大小加权；模块结果 = 等权；
  T05 温度        sqrt(原D) 不是 sqrt(8)：模块 s == 手工 sqrt(D) 口径，
                  ≠ 手工 sqrt(8) 口径；features 记录 d_model 与温度口径；
  T06 内容自适应  同 seq 不同层 → 不同档（P_N/P_F）；同层不同 seq 内容
                  → 不同档；混合构造 → P_C；整数预算随档位变化；
  T07 短输入     空 middle = 合法路径（empty_middle、Kmid=0、中性档）；
                  |M|=3 anchor<2 → unknown+中性档；|M|=5 合法 ok 路径；
  T08 floor 回译 浮点真边界对 (49,1)/(22,15)/(23,13)：朴素 floor 少 1，
                  模块回译后不少 1；非边界对零 bump；
  T09 回退链 γ>1  P_N γ>1 → P_C γ>1 → 固定参数档成功；两级回退逐条记录
                  （原因+成本），至多两次重试；
  T10 回退链容量  near 容量不足 → 回退 P_C 成功（γ=1.0 恰界合法 +
                  near_capacity 双重验证）；
  T11 unsupported 未注册 method + 双档失败 → 三次尝试后 unsupported；
  T12 身份门禁    state 身份（seq/layer/phase/epoch）不符 → 中性档 + unknown；
  T13 数值非有限  q 含 NaN → unknown + 中性档（禁止把非有限当合法特征）；
  T14 成本记账    全部成本字段存在且非负（gather/K 投影/q 投影/位置构造/
                  点积/softmax/重试次数/wall）；每次尝试成本非负；
  T15 spec 序列化 controller_spec() JSON 往返 + 关键字段冻结；
  T16 决策落盘    per_seq_layer_decisions.jsonl 字段闭包（seq/layer/phase/
                  state_epoch/features/s/整数预算/保护/回退/cost）；
  T17 保护>K2     |P|≥K2 明确例外标记（protected_ge_k2）+ Kmid=0；
  T18 预算镜像    Bn=max(1,round(K1β)) 与生产 int(round()) 同口径；Kmid
                  =max(0,min(K2,|U|)−|P|)（§2.2 公式）。

运行：
  python3 exp/trace/test_e124_dyn_controller.py           # 全量
  E124_ONLY=T01,T08 python3 ...                            # 子集
  python3 -O exp/trace/test_e124_dyn_controller.py        # -O 双跑
全部 _check 显式判定（045 纪律：不依赖 assert，python -O 不删除）。
"""
import json
import math
import os
import shutil
import sys
import tempfile

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

D_MODEL = 128
LN2 = math.log(2.0)


def _check(cond, msg=""):
    """045 纪律：显式判定，非 assert（python -O 不删除）。"""
    if not cond:
        raise SystemExit(f"[TEST-FAIL] {msg}")


# ---------------- 合成构造 helpers（只在本套件内） ----------------

def _mk_q(head_dims, d_model=D_MODEL, amp=16.0):
    """q [H, D]：每 head 在指定维放 amp（经 R 投影后该维自内积=amp²，
    其余 anchor 交叉项 ≤ amp·1·|RRᵀ noise|，构造性可控）。"""
    q = torch.zeros(len(head_dims), d_model)
    for h, d in enumerate(head_dims):
        q[h, d] = amp
    return q


def _mk_kmid(m_len, n_kv, dN, dF, mode, d_model=D_MODEL, amp=16.0):
    """middle K [T, Hkv, D]，anchor 位置构造可控：
      mode="N"  N_ref 区 anchor 全部放大在 dN 维（heads 瞄准 dN → P_N）
      mode="F"  F_ref 区 anchor 放大在 dF 维（→ P_F）
      mode="mixed" 双区各自放大（heads 分头瞄准 dN/dF → P_C）
    非 target anchor 放小幅 1.0 于错开维（撞 target 维也无害：logit≤amp·1）。
    """
    from sparse_attn.indexer import dyn_controller as dc
    positions = dc.anchor_positions(m_len)
    split = dc.ref_split(m_len)
    k = torch.zeros(m_len, n_kv, d_model)
    for i, p in enumerate(positions):
        in_N = p >= split
        if mode == "mixed":
            k[p, :, dN if in_N else dF] = amp
        elif (mode == "N" and in_N) or (mode == "F" and not in_N):
            k[p, :, dN if in_N else dF] = amp
        else:
            k[p, :, (i * 7 + 3) % d_model] = 1.0
    return k


def _low_cross_dims(seq_id, layer_idx, d_model=D_MODEL):
    """选 |RRᵀ[d1,d2]| ≤ 0.25 的两个维（确定性搜索；防 mixed 构造里
    非对角噪声 ±1 的最坏情形污染 head 主导性）。"""
    from sparse_attn.indexer import dyn_controller as dc
    R = dc.make_projection(seq_id, layer_idx, d_model)
    G = (R @ R.t()).float()
    for d1 in range(4, d_model // 2):
        for d2 in range(d_model // 2, d_model - 4):
            if abs(float(G[d1, d2])) <= 0.25:
                return d1, d2
    raise SystemExit("[TEST-FAIL] 找不到低串扰维对（不应发生）")


def _decide(q, k_mid, seq_id, layer_idx, method="mavg", k1=128, k2=1024,
            bs=64, n_valid=None, n_protected=256, n_prefix=128, n_swa=128,
            **kw):
    from sparse_attn.indexer import dyn_controller as dc
    if n_valid is None:
        n_valid = (0 if k_mid is None else k_mid.shape[0]) + n_protected
    return dc.decide(q, k_mid, seq_id, layer_idx, method,
                     k1=k1, k2=k2, bs=bs, n_valid=n_valid,
                     n_protected=n_protected, n_prefix=n_prefix,
                     n_swa=n_swa, **kw)


# ---------------- T01 三档边界 ----------------

def test_T01_tier_boundary():
    from sparse_attn.indexer import dyn_controller as dc
    _check(abs(dc.THRESH_LN2 - LN2) < 1e-15, f"阈值必须恰为 ln2，得 {dc.THRESH_LN2}")
    eps = 1e-9
    _check(dc.classify_profile(-LN2 - eps) == "P_F", "s < -ln2 应归 P_F")
    _check(dc.classify_profile(-LN2) == "P_C", "s 恰 = -ln2 应归 P_C（严格小于才 P_F）")
    _check(dc.classify_profile(LN2) == "P_C", "s 恰 = +ln2 应归 P_C（严格大于才 P_N）")
    _check(dc.classify_profile(LN2 + eps) == "P_N", "s > ln2 应归 P_N")
    _check(dc.classify_profile(0.0) == "P_C", "s=0 应归 P_C")
    # 档位参数冻结（GPT §2.1 条 4；阈值固定，不按 method 改）
    tp = dc.TIER_PARAMS
    _check(tp["P_F"] == {"alpha": 0.125, "beta": 0.125, "rho": 0.25},
           f"P_F 档位参数错误: {tp['P_F']}")
    _check(tp["P_C"] == {"alpha": 0.25, "beta": 0.25, "rho": 0.5},
           f"P_C 档位参数错误: {tp['P_C']}")
    _check(tp["P_N"] == {"alpha": 0.5, "beta": 0.5, "rho": 0.75},
           f"P_N 档位参数错误: {tp['P_N']}")


# ---------------- T02 anchor 等距与 N_ref 边界 ----------------

def test_T02_anchor_positions():
    from sparse_attn.indexer import dyn_controller as dc
    # 只依合法位置：同 m_len 不同内容 → 同位置（函数签名只收 m_len，行为断言）
    p100 = dc.anchor_positions(100)
    _check(p100 == dc.anchor_positions(100), "同 m_len anchor 必须确定性一致")
    _check(len(p100) == 16, f"m_len=100 应取满 16 anchor，得 {len(p100)}")
    _check(p100 == sorted(set(p100)), "anchor 必须严格递增去重")
    _check(p100[0] == 0 and p100[-1] == 99, "anchor 应覆盖 middle 两端")
    gaps = [b - a for a, b in zip(p100, p100[1:])]
    _check(max(gaps) - min(gaps) <= 1, f"等距性被破坏: {gaps}")
    # 不足 16 取全部
    _check(dc.anchor_positions(6) == [0, 1, 2, 3, 4, 5], "m_len=6 应取全部 6 个")
    _check(dc.anchor_positions(0) == [], "空 middle 无 anchor")
    # N_ref = middle 最后 1/4（固定参考边界，先于 α 定义，不自我标注）
    _check(dc.ref_split(16) == 12, "m=16 的 split 应为 12（最后 1/4=4）")
    _check(dc.ref_split(100) == 75, "m=100 的 split 应为 75")
    _check(dc.ref_split(5) == 3, "m=5 ceil(5/4)=2 → split=3")
    pos16 = dc.anchor_positions(16)
    nN = sum(p >= dc.ref_split(16) for p in pos16)
    _check(nN == 4 and len(pos16) - nN == 12,
           f"m=16 anchor 区归属应 12/4，得 F={len(pos16)-nN} N={nN}")


# ---------------- T03 投影 R ----------------

def test_T03_projection_seed():
    from sparse_attn.indexer import dyn_controller as dc
    r1 = dc.make_projection(101, 1, 64)
    r1b = dc.make_projection(101, 1, 64)
    r2 = dc.make_projection(101, 2, 64)
    r3 = dc.make_projection(202, 1, 64)
    _check(r1.shape == (64, 8), f"R 形状应 [D,8]，得 {tuple(r1.shape)}")
    _check(torch.equal(r1, r1b), "同 (seq,layer) 的 R 必须逐位一致（固定 seed）")
    _check(not torch.equal(r1, r2), "异 layer 的 R 应不同")
    _check(not torch.equal(r1, r3), "异 seq 的 R 应不同")
    vals = {round(v, 12) for v in r1.flatten().tolist()}
    expect = round(1.0 / math.sqrt(8.0), 12)
    _check(vals == {expect, -expect}, f"R 元素必须 ∈ ±1/√8，得 {vals}")


# ---------------- T04 GQA 聚合顺序 ----------------

def test_T04_gqa_aggregation_order():
    from sparse_attn.indexer import dyn_controller as dc
    # 最小反例：H=3, KV 组数=2, 组大小 [2,1]（Q-head 数 ≠ KV 组数）
    s_h = torch.tensor([1.0, 1.0, 0.0])
    got = dc.aggregate_s_h(s_h, [2, 1])
    # 先组内等权（(1+1)/2=1；(0)/1=0），再组间等权（(1+0)/2=0.5）
    expect_equal_weight = 0.5
    # 组大小加权（错误口径）：(1+1+0)/3 = 2/3
    wrong_size_weighted = 2.0 / 3.0
    _check(abs(got - expect_equal_weight) < 1e-9,
           f"等权聚合应为 0.5，得 {got}")
    _check(abs(got - wrong_size_weighted) > 1e-6,
           "等权与组大小加权在此反例必须不同（否则测试无区分力）")
    # 均匀组（H=4, [2,2]）：等权 == 加权，但公式仍须先组内后组间
    # 输入用 float64：float32 的 0.2/0.4/0.6 本身带表示误差
    # （0.55000000819 ≠ 0.55 是输入舍入，非聚合缺陷）
    s_h2 = torch.tensor([0.2, 0.4, 0.6, 1.0], dtype=torch.float64)
    got2 = dc.aggregate_s_h(s_h2, [2, 2])
    _check(abs(got2 - ((0.3 + 0.8) / 2)) < 1e-9,
           f"[2,2] 聚合应先组内 mean 再组间 mean，得 {got2}")


# ---------------- T05 温度 sqrt(D) 不是 sqrt(8) ----------------

def test_T05_temperature_sqrt_d():
    from sparse_attn.indexer import dyn_controller as dc
    D, H, Hkv, m_len = 32, 1, 1, 16
    # 构造：q 与全部 anchor 同维 dN 打点积（同维投影点积 = a_q·a_k 精确值，
    # 不经 R Rᵀ 交叉噪声）。N 区 anchor 幅 16、F 区幅 14——logit 差
    # Δ=16·(16−14)=32，除以温度后 softmax 陡度随温度变化 → sqrt(D) 与
    # sqrt(8) 两口径 s 必须显著不同，且 F 区概率远大于 eps=1e-8
    # （初版构造 F 区 logit 差 ~42，exp 后 ~e-42 ≪ eps，eps 主导把
    # 两口径压成同值——构造缺陷，非实现缺陷）。
    a_q, a_N, a_F = 16.0, 16.0, 14.0
    dN = 5
    k_mid = torch.zeros(m_len, Hkv, D)
    positions = dc.anchor_positions(m_len)
    split = dc.ref_split(m_len)
    for p in positions:
        k_mid[p, :, dN] = a_N if p >= split else a_F
    q = _mk_q([dN], d_model=D, amp=a_q)
    feat = dc.compute_features(q, k_mid, seq_id=1, layer_idx=1)
    _check(feat["status"] == "ok", f"特征应合法，得 {feat['status']} {feat.get('reason')}")
    _check(feat["d_model"] == D, "features 必须记录原 D")
    _check(feat["temp_rule"] == "sqrt(D)", f"温度口径记录错误: {feat['temp_rule']}")
    s_mod = feat["s"]
    # 手工复算：温度 sqrt(D)
    R = dc.make_projection(1, 1, D)
    k_proj = k_mid[:, 0].double() @ R           # [16, 8]
    q_proj = q.double() @ R                      # [1, 8]
    logits = (q_proj @ k_proj.t()) / math.sqrt(D)
    p = torch.softmax(logits - logits.max(), dim=-1)
    is_N = torch.tensor([pp >= split for pp in positions])
    dN_h = p[:, is_N].sum(-1) / int(is_N.sum())
    dF_h = p[:, ~is_N].sum(-1) / int((~is_N).sum())
    s_d = float(torch.log((dN_h + 1e-8) / (dF_h + 1e-8)))
    # 错误口径：温度 sqrt(8)（ sharper）
    logits8 = (q_proj @ k_proj.t()) / math.sqrt(8)
    p8 = torch.softmax(logits8 - logits8.max(), dim=-1)
    dN8 = p8[:, is_N].sum(-1) / int(is_N.sum())
    dF8 = p8[:, ~is_N].sum(-1) / int((~is_N).sum())
    s_8 = float(torch.log((dN8 + 1e-8) / (dF8 + 1e-8)))
    # F 区概率不可被 eps 吞掉（否则两口径不可区分，测试失去区分力）
    _check(float(dF_h) > 1e-6, f"F 区密度须 > eps 两个量级以上，得 {float(dF_h)}")
    _check(abs(s_mod - s_d) < 1e-4,
           f"模块 s={s_mod} 应等于 sqrt(D) 温度手工值 {s_d}")
    _check(abs(s_mod - s_8) > 0.5,
           f"sqrt(D) 与 sqrt(8) 口径在此构造必须可区分（s_mod={s_mod}, s_8={s_8}）")


# ---------------- T06 内容自适应：同 seq 异层 / 同层异 seq ----------------

def test_T06_content_adaptive_profiles():
    from sparse_attn.indexer import dyn_controller as dc
    H, Hkv, m_len = 4, 2, 2048
    seq, lay_N, lay_F, lay_M = 101, 1, 7, 2
    dN, dF = _low_cross_dims(seq, lay_M)
    dN1, _ = _low_cross_dims(seq, lay_N)
    dF7, _ = _low_cross_dims(seq, lay_F)
    dNx, _ = _low_cross_dims(202, 1)

    # 同 seq 不同层：层1 内容偏 near（P_N），层7 内容偏 far（P_F）
    k_N = _mk_kmid(m_len, Hkv, dN1, (dN1 + 40) % D_MODEL, "N")
    k_F = _mk_kmid(m_len, Hkv, (dF7 + 40) % D_MODEL, dF7, "F")
    qN = _mk_q([dN1] * H)
    qF = _mk_q([dF7] * H)
    dec1 = _decide(qN, k_N, seq, lay_N)
    dec7 = _decide(qF, k_F, seq, lay_F)
    _check(dec1["profile"]["applied"] == "P_N",
           f"seq101/L1 应 P_N（s={dec1['features']['s']}），得 {dec1['profile']['applied']}")
    _check(dec7["profile"]["applied"] == "P_F",
           f"seq101/L7 应 P_F（s={dec7['features']['s']}），得 {dec7['profile']['applied']}")
    _check(dec1["features"]["s"] > LN2, "P_N 决策的 s 应 > ln2")
    _check(dec7["features"]["s"] < -LN2, "P_F 决策的 s 应 < -ln2")
    _check(dec1["budget"]["alpha"] == 0.5 and dec7["budget"]["alpha"] == 0.125,
           "整数预算必须随档位变化（α 1/2 vs 1/8）")
    _check(dec1["budget"]["Tn"] != dec7["budget"]["Tn"],
           "不同档位的 near 整数配额应不同")

    # 同层不同 seq：内容不同 → 档不同
    dec2 = _decide(_mk_q([dNx] * H),
                   _mk_kmid(m_len, Hkv, (dNx + 40) % D_MODEL, dNx, "F"),
                   202, lay_N)
    _check(dec2["profile"]["applied"] == "P_F",
           f"seq202/L1 应 P_F（同层异 seq 反例），得 {dec2['profile']['applied']}")

    # 混合构造 → P_C（每组一半 head 瞄 N、一半瞄 F）
    k_mix = _mk_kmid(m_len, Hkv, dN, dF, "mixed")
    q_mix = _mk_q([dN, dF, dN, dF])
    decm = _decide(q_mix, k_mix, seq, lay_M)
    _check(abs(decm["features"]["s"]) < LN2,
           f"混合构造 s 应落在 (-ln2, ln2)，得 {decm['features']['s']}")
    _check(decm["profile"]["applied"] == "P_C",
           f"混合构造应 P_C，得 {decm['profile']['applied']}")
    # 无回退：单次尝试成功
    _check(len(decm["attempts"]) == 1 and decm["attempts"][0]["ok"],
           "合法档应一次尝试成功，无回退")


# ---------------- T07 短输入 / 空 middle ----------------

def test_T07_short_input_paths():
    from sparse_attn.indexer import dyn_controller as dc
    q = _mk_q([3, 4])
    # 空 middle：合法路径（直接保护/全可见短输入）
    dec0 = _decide(q, torch.zeros(0, 2, D_MODEL), 1, 1)
    _check(dec0["features"]["status"] == "empty_middle",
           f"空 middle 应记 empty_middle，得 {dec0['features']['status']}")
    _check(dec0["profile"]["applied"] == "P_C" and dec0["profile"]["neutral"],
           "空 middle 应走中性档 P_C")
    _check(dec0["budget"]["Kmid"] == 0 and dec0["budget"]["Tn"] == 0
           and dec0["budget"]["Tf"] == 0, "空 middle 预算应为 0")
    _check(dec0["status"] == "empty_middle", "决策级 status 应保留 empty_middle")
    # None k_mid 同义
    dec0b = _decide(q, None, 1, 1)
    _check(dec0b["features"]["status"] == "empty_middle", "k_mid=None 应等同空 middle")
    # |M|=3：非空区 anchor < 2 → unknown + 中性档（禁止把没采样区当零重要性）
    dec3 = _decide(q, torch.randn(3, 2, D_MODEL), 1, 1)
    _check(dec3["features"]["status"] == "unknown",
           f"|M|=3 应 unknown（N_ref anchor<2），得 {dec3['features']['status']}")
    _check("anchor" in str(dec3["features"].get("reason", "")),
           f"unknown 原因应记 anchor 不足: {dec3['features'].get('reason')}")
    _check(dec3["profile"]["neutral"], "unknown 应输出中性档")
    # |M|=5：合法 ok 路径（N=2, F=3 anchor）
    dec5 = _decide(q, torch.randn(5, 2, D_MODEL), 1, 1)
    _check(dec5["features"]["status"] == "ok",
           f"|M|=5 应合法 ok，得 {dec5['features']['status']} "
           f"{dec5['features'].get('reason')}")
    _check(dec5["features"]["n_anchor_N"] == 2 and dec5["features"]["n_anchor_F"] == 3,
           "m=5 anchor 区归属应为 F=3/N=2")


# ---------------- T08 整数 floor 回译 ----------------

def test_T08_floor_roundtrip():
    from sparse_attn.indexer import dyn_controller as dc
    # 真浮点边界对：朴素 floor(d·(t/d)) < t（IEEE 双精实测），模块必须不少 1
    for denom, target in [(49, 1), (22, 15), (23, 13), (26, 15)]:
        naive = math.floor(denom * (target / denom))
        _check(naive < target,
               f"({denom},{target}) 应是朴素回译少 1 的边界案例（得 {naive}）"
               "——若不再成立请换边界对")
        gamma, bumped = dc.derive_gamma(denom, target)
        rt = math.floor(denom * gamma)
        _check(rt == target,
               f"({denom},{target}) 回译后 floor 应恰为 {target}，得 {rt}")
        _check(bumped, f"({denom},{target}) 应发生 bump")
        _check(gamma <= 1.0, "bump 不得把 γ 推过 1（target≤denom 时）")
    # 非边界对：零 bump、γ 精确
    gamma, bumped = dc.derive_gamma(8, 3)
    _check(not bumped and gamma == 0.375 and math.floor(8 * gamma) == 3,
           "非边界对不应 bump")
    gamma, bumped = dc.derive_gamma(4096, 4096)
    _check(gamma == 1.0 and not bumped and math.floor(4096 * 1.0) == 4096,
           "γ=1.0 恰界应合法（Tn_target == Bn·b）")
    # 全链路验证：编译出的整数 Tn 不少 1
    # 构造 denom 落边界对 (22,15)：k1=176, β=1/8 → Bn=22, bs=1 →
    # denom=22；k2=60, U=128, P=0 → Kmid=min(60,128)=60，ρ=1/4 →
    # target=15。M=128 → |N|=max(1,int(128/8))=16 ≥ 15（容量须放行，
    # 初版 n_valid=60 使 |N|=7 < 15 撞 near_capacity——构造缺陷）。
    ok, info = dc.compile_tier("P_F", k1=176, k2=60, bs=1, n_valid=128,
                               n_protected=0)
    _check(ok, f"边界档应编译成功: {info}")
    _check(info["Tn"] == 15,
           f"边界档整数 Tn 应为 15（不少 1），得 {info.get('Tn')}")
    _check(math.floor(22 * info["gamma"]) == 15,
           "落盘 γ 的 floor 回译必须等于规范目标 Tn")


# ---------------- T09 回退链：γ>1 两级回退 ----------------

def test_T09_fallback_gamma_gt1():
    # 构造：k1=128, bs=64, β=1/2 → Bn=64, denom=4096；Kmid=6000
    # （k2=6000, U=8000, P=0）→ P_N: Tn_target=4500 > 4096 → γ>1；
    #    P_C: Tn_target=3000 > denom_C=2048 → γ>1；
    #    固定档 (0.25,0.125,0.625)：Bn=16, denom=1024, Tn=min(640,6000)=640，
    #    N_len=2000 → Cn=1024 ≥ 640 ✓，Tf=5360 ≤ Cf=6000 ✓ → 成功。
    # q 瞄准 k 的 target 维 dN=5（同维投影点积精确 amp²，跨维则被
    # R Rᵀ 交叉噪声随机化、档位不可控——初版 q=[3,4] 是构造缺陷）。
    q = _mk_q([5, 5])
    k_mid = _mk_kmid(2048, 2, 5, 60, "N")     # 特征 → P_N
    dec = _decide(q, k_mid, 1, 1, k1=128, k2=6000, bs=64,
                  n_valid=8000, n_protected=0, n_prefix=0, n_swa=0)
    _check(dec["profile"]["requested"] == "P_N", "请求档应为 P_N")
    _check(dec["profile"]["applied"] == "fixed_tier",
           f"两级回退后应落固定参数档，得 {dec['profile']['applied']}")
    at = dec["attempts"]
    _check(len(at) == 3, f"应恰 3 次尝试（初始+两级回退），得 {len(at)}")
    _check(at[0]["step"] == "P_N" and not at[0]["ok"]
           and at[0]["reason"] == "gamma_gt_1",
           f"第 1 次尝试应 P_N γ>1 失败: {at[0]}")
    _check(at[1]["step"] == "P_C" and not at[1]["ok"]
           and at[1]["reason"] == "gamma_gt_1",
           f"第 2 次尝试应 P_C γ>1 失败: {at[1]}")
    _check(at[2]["step"] == "fixed_tier" and at[2]["ok"],
           f"第 3 次尝试应固定档成功: {at[2]}")
    _check(dec["budget"]["alpha"] == 0.25 and dec["budget"]["beta"] == 0.125,
           "固定档参数应为 (.25,.125,.625)（不切换 mavg 选择器，只换参数）")
    for a in at:
        _check(a.get("cost", -1) >= 0, f"每次尝试成本必须非负: {a}")
    _check(dec["cost"]["retries"] == 2, "重试计数应为 2（至多两次）")
    _check(dec["status"] == "ok", "回退成功后 status 应为 ok")


# ---------------- T10 回退链：near 容量不足 ----------------

def test_T10_fallback_near_capacity():
    from sparse_attn.indexer import dyn_controller as dc
    # 构造（推导见任务书）：bs=1, k1=32, k2=14, U=28, P=0 → M=28, Kmid=14。
    # P_F: Bn=max(1,round(4))=4 → denom=4；Tn_target=round(3.5)=4（banker's）
    #      → γ=4/4=1.0 恰界合法；N_len=max(1,int(3.5))=3 → Cn=3 < 4
    #      → near_capacity 失败（同时验证 γ=1.0 不是 γ>1）。
    # P_C: Bn=8 → denom=8；Tn_target=7；N_len=max(1,7)=7 ≥ 7 ✓；
    #      Tf=7 ≤ Cf=min(24, 21)=21 ✓ → 成功。
    ok, info = dc.compile_tier("P_F", k1=32, k2=14, bs=1, n_valid=28,
                               n_protected=0)
    _check(not ok, "P_F 应失败")
    _check(info["reason"] == "near_capacity",
           f"失败原因应为 near_capacity 而非 gamma_gt_1（γ=1.0 恰界合法）: {info}")
    _check(info["gamma"] == 1.0, "γ=1.0 恰界必须放行到容量检查")
    ok2, info2 = dc.compile_tier("P_C", k1=32, k2=14, bs=1, n_valid=28,
                                 n_protected=0)
    _check(ok2, f"P_C 应编译成功: {info2}")
    _check(info2["Tn"] == 7 and info2["Tf"] == 7, "P_C 整数配额应为 7/7")
    # 全链：特征 → P_F 请求 → 回退 P_C（q 瞄准 F 构造的 target 维 dF=60）
    q = _mk_q([60, 60])
    k_mid = _mk_kmid(2048, 2, 5, 60, "F")     # 特征 → P_F
    dec = _decide(q, k_mid, 9, 1, k1=32, k2=14, bs=1, n_valid=28,
                  n_protected=0, n_prefix=0, n_swa=0)
    # 注：k_mid=2048 与 n_valid=28 不一致——decide 的特征只用 k_mid，预算只用
    # n_valid/n_protected（M0 状态来源分离，v0 契约如此）；此处验证回退链行为。
    _check(dec["profile"]["requested"] == "P_F", "请求档应为 P_F")
    _check(dec["profile"]["applied"] == "P_C",
           f"near 容量不足应回退 P_C，得 {dec['profile']['applied']}")
    at = dec["attempts"]
    _check(len(at) == 2 and at[0]["reason"] == "near_capacity" and at[1]["ok"],
           f"应恰两次尝试（P_F 容量失败 → P_C 成功）: {[(a['step'], a.get('reason'), a['ok']) for a in at]}")


# ---------------- T11 unsupported ----------------

def test_T11_unsupported():
    q = _mk_q([3, 4])
    k_mid = _mk_kmid(2048, 2, 5, 60, "N")     # → P_N
    # 未注册 method + 双档 γ>1 失败 → 三尝试后 unsupported
    dec = _decide(q, k_mid, 1, 1, method="nosuchmethod", k1=128, k2=6000,
                  bs=64, n_valid=8000, n_protected=0, n_prefix=0, n_swa=0)
    _check(dec["status"] == "unsupported", f"应判 unsupported，得 {dec['status']}")
    _check(dec["profile"]["applied"] is None, "unsupported 不得有落用档")
    at = dec["attempts"]
    _check(len(at) == 3, f"至多两次重试（3 次尝试封顶），得 {len(at)}")
    _check(at[2]["step"] == "fixed_tier"
           and at[2]["reason"] == "method_not_registered",
           f"末次尝试应为固定档不可用（method 未通过正确性验证）: {at[2]}")


# ---------------- T12 state 身份门禁 ----------------

def test_T12_identity_gate():
    q = _mk_q([3, 4])
    k_mid = _mk_kmid(2048, 2, 5, 60, "N")
    # 特征在 (seq=1, layer=1, epoch=0) 计算，消费端声明身份 (layer=3) → 不符
    dec = _decide(q, k_mid, 1, 1,
                  identity_check={"seq_id": 1, "layer_idx": 3,
                                  "phase": "prefill", "state_epoch": 0})
    _check(dec["features"]["status"] == "unknown",
           f"身份不符应 unknown，得 {dec['features']['status']}")
    _check(dec["features"]["reason"] == "identity_mismatch",
           f"原因应记 identity_mismatch: {dec['features'].get('reason')}")
    _check(dec["profile"]["neutral"] and dec["profile"]["applied"] == "P_C",
           "身份不符应输出中性档")
    # epoch 不符同理
    dec2 = _decide(q, k_mid, 1, 1,
                   identity_check={"seq_id": 1, "layer_idx": 1,
                                   "phase": "prefill", "state_epoch": 7})
    _check(dec2["features"]["status"] == "unknown", "state_epoch 不符应 unknown")
    # 一致身份 → 正常
    dec3 = _decide(q, k_mid, 1, 1,
                   identity_check={"seq_id": 1, "layer_idx": 1,
                                   "phase": "prefill", "state_epoch": 0})
    _check(dec3["features"]["status"] == "ok", "一致身份应放行")


# ---------------- T13 数值非有限 ----------------

def test_T13_nonfinite():
    q = _mk_q([3, 4])
    q[1, 3] = float("nan")
    k_mid = _mk_kmid(2048, 2, 5, 60, "N")
    dec = _decide(q, k_mid, 1, 1)
    _check(dec["features"]["status"] == "unknown",
           f"非有限特征应 unknown，得 {dec['features']['status']}")
    _check(dec["features"]["reason"] == "nonfinite",
           f"原因应记 nonfinite: {dec['features'].get('reason')}")
    _check(dec["profile"]["neutral"], "非有限应输出中性档")
    _check(dec["features"]["s"] is None, "非有限时 s 必须为 None（不得落盘假值）")


# ---------------- T14 成本记账 ----------------

def test_T14_cost_accounting():
    q = _mk_q([3, 4, 5, 6, 7, 8, 9, 10])
    m_len, Hkv = 2048, 2
    k_mid = _mk_kmid(m_len, Hkv, 5, 60, "N")
    dec = _decide(q, k_mid, 1, 1)          # 8 heads → G=4
    cost = dec["cost"]
    need = ["anchor_pos_ops", "gather_elems", "k_proj_macs", "q_proj_macs",
            "dot_macs", "softmax_elems", "rng_elems", "retries", "wall_ms"]
    for k in need:
        _check(k in cost, f"成本字段缺失: {k}")
        _check(cost[k] >= 0, f"成本字段 {k} 必须非负，得 {cost[k]}")
    # 记账口径抽查（§2.1 条 1：非连续 gather / 16·d·8 K 投影 / q 投影 / 位置构造）
    n_anchor = 16
    _check(cost["gather_elems"] == n_anchor * Hkv * D_MODEL,
           f"gather 应记 16·Hkv·D = {n_anchor * Hkv * D_MODEL}，得 {cost['gather_elems']}")
    _check(cost["k_proj_macs"] == n_anchor * Hkv * D_MODEL * 8,
           "K 投影应记 16·d·8 MACs")
    _check(cost["q_proj_macs"] == 8 * D_MODEL * 8, "q 投影应记 H·D·8 MACs")
    _check(cost["dot_macs"] == 8 * n_anchor * 8, "点积应记 H·16·8 MACs")
    _check(cost["anchor_pos_ops"] == n_anchor, "位置构造应记 16")
    # eps dtype 记录（v0 全链 float64——R 元素须精确 ±1/√8，见 T03）
    _check(dec["features"]["s_h_eps"] == 1e-8
           and dec["features"]["s_h_eps_dtype"] == "torch.float64",
           "eps=1e-8 与 dtype 必须记录")


# ---------------- T15 controller_spec 序列化 ----------------

def test_T15_controller_spec():
    from sparse_attn.indexer import dyn_controller as dc
    spec = dc.controller_spec()
    s = json.dumps(spec)                     # 可序列化
    back = json.loads(s)
    _check(back == spec, "spec JSON 往返必须无损")
    _check(back["version"] == dc.CONTROLLER_VERSION, "spec 须含版本")
    _check(back["feature_dim"] == 8 and back["n_anchors_max"] == 16,
           "spec 须冻结 8 维投影 / 16 anchor")
    _check(back["thresholds"] == {"P_F_below": -LN2, "P_N_above": LN2},
           f"spec 阈值须 ±ln2: {back['thresholds']}")
    _check(back["temp_rule"] == "sqrt(D)", "spec 须记温度口径")
    _check(back["neutral_profile"] == "P_C", "spec 须记中性档")
    _check(back["tier_params"]["P_N"] == [0.5, 0.5, 0.75], "spec 须记档位参数")
    _check("mavg" in back["method_fixed_tier"], "spec 须记固定参数档 registry")
    _check("seed_scheme" in back, "spec 须记 seed 方案")


# ---------------- T16 决策 JSONL 落盘 ----------------

def test_T16_decision_log():
    from sparse_attn.indexer import dyn_controller as dc
    base = tempfile.mkdtemp(prefix="e124a_log_")
    try:
        log = dc.DecisionLog(os.path.join(base, "per_seq_layer_decisions.jsonl"))
        for seq in (1, 2):
            for layer in (1, 5):
                # q 瞄准各构造的 target 维（层1 N 构造 dN=5、层5 F 构造
                # dF=60）——同维投影点积精确 amp²，档位确定可控
                mode = "N" if layer == 1 else "F"
                q = _mk_q([5, 5]) if mode == "N" else _mk_q([60, 60])
                dec = _decide(q, _mk_kmid(2048, 2, 5, 60, mode),
                             seq, layer, phase="prefill", state_epoch=seq)
                log.append(dec)
        with open(os.path.join(base, "per_seq_layer_decisions.jsonl")) as f:
            lines = [json.loads(x) for x in f if x.strip()]
        _check(len(lines) == 4, f"应落 4 条决策，得 {len(lines)}")
        d = lines[0]
        for k in ["seq_id", "layer_idx", "phase", "state_epoch", "features",
                  "profile", "budget", "protection", "attempts", "cost",
                  "method", "controller_version"]:
            _check(k in d, f"决策记录缺字段: {k}")
        _check(isinstance(d["features"]["s"], float) or d["features"]["s"] is None,
               "s 必须可序列化")
        for k in ["Bn", "Bf", "Kmid", "Tn", "Tf", "gamma"]:
            _check(k in d["budget"], f"整数预算缺字段: {k}")
        _check("n_protected" in d["protection"] and "protected_ge_k2" in d["protection"],
               "保护记录缺字段")
        # 同 seq 异层档必须不同（N 构造→P_N，F 构造→P_F：落盘保留
        # 逐层可变性；逐 seq 可变性已由 T06 直接覆盖，不做 seed 彩票断言）
        _check(lines[0]["seq_id"] == lines[1]["seq_id"] == 1,
               "前两行应同为 seq=1")
        _check(lines[0]["profile"]["applied"] == "P_N"
               and lines[1]["profile"]["applied"] == "P_F",
               f"同 seq 异层应落不同档（P_N/P_F），得 "
               f"{lines[0]['profile']['applied']}/{lines[1]['profile']['applied']}")
    finally:
        shutil.rmtree(base, ignore_errors=True)


# ---------------- T17 保护大于 K2 的明确例外 ----------------

def test_T17_protection_exceeds_k2():
    q = _mk_q([3, 4])
    k_mid = _mk_kmid(2048, 2, 5, 60, "N")
    # U=1200, P=1100, k2=1024 → min(k2,U)=1024 < P → Kmid=0，M=100>0
    dec = _decide(q, k_mid, 1, 1, k2=1024, n_valid=1200, n_protected=1100,
                  n_prefix=550, n_swa=550)
    _check(dec["protection"]["protected_ge_k2"] is True,
           "保护 ≥ K2 必须显式标记例外（不许静默）")
    _check(dec["budget"]["Kmid"] == 0, "Kmid 应为 0（middle 预算被保护吞没）")
    _check(dec["status"] == "ok", "保护例外是记录而非失败（合法路径）")
    _check(dec["budget"]["Tn"] == 0 and dec["budget"]["Tf"] == 0,
           "例外下 mid 预算应为 0")


# ---------------- T18 预算公式镜像（§2.2 / 生产口径） ----------------

def test_T18_budget_formula_mirror():
    from sparse_attn.indexer import dyn_controller as dc
    # Bn = max(1, round(K1β))：与生产 nb_near = max(1, int(round(k1*beta))) 同口径
    for k1, beta, expect in [(128, 0.125, 16), (128, 0.126, 16), (4, 0.125, 1),
                             (128, 0.5, 64), (128, 0.25, 32)]:
        got = max(1, int(round(k1 * beta)))
        _check(got == expect, f"Bn({k1},{beta}) 应为 {expect}，得 {got}")
    ok, info = dc.compile_tier("P_C", k1=128, k2=1024, bs=64, n_valid=2304,
                               n_protected=256)
    _check(ok, f"基准几何应编译成功: {info}")
    _check(info["Bn"] == 32 and info["Bf"] == 96, "Bn/Bf 应 32/96")
    # Kmid = max(0, min(K2,|U|)−|P|)（§2.2 公式，min(K2,|U|) 在内层）
    _check(info["Kmid"] == min(1024, 2304) - 256, "Kmid 公式镜像失败")
    _check(info["Tn"] == 384 and info["Tf"] == 384, "P_C 应 Tn=Tf=384")
    _check(abs(info["gamma"] - 384 / 2048) < 1e-12, "γ 反算应 384/2048")
    # 区域镜像生产 near_len_dyn = max(bs, int(α·mid)) 并 clamp 到 M
    _check(info["N_len"] == max(64, int(0.25 * (2304 - 256))),
           "N_len 应镜像 max(bs, int(α·M))")
    # Tn = min(floor(Bn·b·γ), Kmid) 的 floor 语义
    _check(info["Tn"] == min(math.floor(info["Bn"] * 64 * info["gamma"]),
                            info["Kmid"]), "Tn 公式镜像失败")


# ================================================================ main

PLAN = [
    ("T01", test_T01_tier_boundary),
    ("T02", test_T02_anchor_positions),
    ("T03", test_T03_projection_seed),
    ("T04", test_T04_gqa_aggregation_order),
    ("T05", test_T05_temperature_sqrt_d),
    ("T06", test_T06_content_adaptive_profiles),
    ("T07", test_T07_short_input_paths),
    ("T08", test_T08_floor_roundtrip),
    ("T09", test_T09_fallback_gamma_gt1),
    ("T10", test_T10_fallback_near_capacity),
    ("T11", test_T11_unsupported),
    ("T12", test_T12_identity_gate),
    ("T13", test_T13_nonfinite),
    ("T14", test_T14_cost_accounting),
    ("T15", test_T15_controller_spec),
    ("T16", test_T16_decision_log),
    ("T17", test_T17_protection_exceeds_k2),
    ("T18", test_T18_budget_formula_mirror),
]


def main():
    """红绿 runner：显式 _check（python -O 兼容）；E124_ONLY 过滤；
    一次跑完全部用例给红绿全景；FAIL>0 非零退出。"""
    only = os.environ.get("E124_ONLY", "")
    plan = PLAN
    if only:
        keep = {x.strip() for x in only.split(",") if x.strip()}
        plan = [p for p in PLAN if p[0] in keep]
    n_pass = n_fail = 0
    failed = []
    for name, fn in plan:
        try:
            fn()
        except SystemExit as e:
            n_fail += 1
            failed.append(name)
            print(f"[{name}] FAIL  {e}", flush=True)
            continue
        except BaseException as e:   # noqa: BLE001——一次跑完给全红绿图
            n_fail += 1
            failed.append(name)
            import traceback
            print(f"[{name}] FAIL  {type(e).__name__}: {e}\n"
                  f"{traceback.format_exc()[-1000:]}", flush=True)
            continue
        n_pass += 1
        print(f"[{name}] PASS", flush=True)
    print(f"\nE124A-DYN-CONTROLLER RESULT: PASS={n_pass} FAIL={n_fail} "
          f"(total {len(plan)})")
    if failed:
        print(f"FAILED: {failed}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
