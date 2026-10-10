#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""079（TL-E121-OUTPUT-ID-079，GPT 2026-10-11 0332 复审 P1）修复验收套件
（CPU only + torch；python 与 python -O 双跑安全——全部显式判定，无 assert）。

背景：076 canonical manifest 仍不是有效 treatment 的单射，两个残余——
  ① tia_enable_async_topk（TLI 继承 TIA：tia_indexer.py:18
     self.enable_async = args.tia_enable_async_topk；tli_indexer.py
     compute_mask async 分支缓存 prev_mask 并在下一步替换）未入
     manifest/hash → 同步/异步同名同 sidecar；
  ② tli_proj_basis / tli_layer_skip_path 只记词法路径（default=str），
     同路径不同内容 → 同 manifest/hash → LongBench 写门放行覆盖、
     RULER 复用同逻辑指针。

修复（sparse_attn/info.py）：① 生效布尔进 manifest；② 文件字段在
manifest 构造时解析为内容身份 {path, sha256, shape/n_skip}，缺失/读取
失败/解析失败 [GATE-FAIL] 079 SystemExit；LongBench sidecar、RULER
receipt（treatment_manifest 键，sha256/json 自洽 schema 校验）与
method hash 共用同一份 resolved manifest。

测试矩阵（065/067 纪律：PASS/SKIP/FAIL 三态、显式 _check、FAIL 继续跑）：
  R1  ①async 单射：两 namespace 仅差 tia_enable_async_topk →
      manifest/hash/method_name 必不同；缺省（无属性）与显式 False
      同 hash（缺省等价，B09 族 hash 侧硬约束）。
  R2  ②文件内容身份 + LongBench 三拒模式：
      a) 同路径 basis 内容 A→B → manifest/hash 必变；既有目标 +
         sidecar(A) 时 gate(B) 必拒（SystemExit + 目标字节不变），
         gate(A) 幂等放行；无 sidecar 拒；新路径放行；
      b) layer_skip 同口径（n_skip 摘要）；
      c) fail closed：文件缺失/坏 basis/坏 skip JSON → SystemExit
         带 [GATE-FAIL] 079；
      d) 值 None → manifest null；重复调用逐位稳定（确定性）。
  R3  mutation test：遍历 add_sparse_attn_args 注册的全部参数——
      manifest 字段逐个突变 → hash 必变；readable 编码字段
      （tli_enable_kmeans/layer_skip 的 B/D）→ 全名必变；白名单
      字段（quest/twi/method——非 tli treatment）→ tli 名不变；
      任何未登记参数 → FAIL（防未来新参数漏进身份）。
  R4  async 行为差异（CPU torch 确定性 marker 输入，≥2 decode 步）：
      第 1 步 async == 同步（prev_mask 尚未生效）；第 2 步 async 用
      第 1 步的块选择 → 与同步路径的 mid 区选择可不同（候选块互斥
      实证）——079① 不是性能提示而是输出语义。
  R5  RULER receipt 接线：build_yarn_receipt(treatment_manifest_json)
      → 回执含 treatment_manifest {sha256,json}；schema 校验自洽
      （篡改 json/sha → 拒收）；无该键（非 tli 臂/legacy）前向兼容。

红态对照：修复前（基点 f613ca645）R1 异步同名同 manifest、R2 同路径
内容变更同 manifest 且写门放行、R3 async/文件字段不触发 hash 变化、
R4 仍红不了（行为在但身份测不出来）——本套件对主树可作红态复跑。

用法：
  cd <repo>/two-level-attention && python3 test_e121_fix_079.py
  python -O 同上双跑；TLI079_ONLY=R1,R2 子集过滤。
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import types

sys.dont_write_bytecode = True

import torch   # info.py 模块级依赖 + R2 basis fixture + R4 行为测试

torch.set_num_threads(1)

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import sparse_attn.info as INFO                     # noqa: E402
from sparse_attn.arguments import add_sparse_attn_args  # noqa: E402
from sparse_attn.indexer.tli_indexer import TLIIndexer   # noqa: E402
from benchmark.RULER.yarn_receipt import (         # noqa: E402
    build_yarn_receipt, _validate_common_schema)


def _check(cond, msg=""):
    """065/067 纪律：显式判定，非 assert（python -O 不删除）。"""
    if not cond:
        raise SystemExit(f"[TEST-FAIL] {msg}")


def _base_ns(**kw):
    """R1/R2 用的最小 treatment namespace（getattr 兜底缺省）。"""
    base = dict(method="tli", tia_block_size=64, tia_level1_topk=128,
                tia_level2_topk=2048, tia_level2_cmp_ratio=4,
                tli_alpha=0.25, tli_beta=0.125, tli_gamma=0.625,
                tli_enable_kmeans=False, tli_enable_layer_skip=False,
                tli_far_method="minmax", tli_near_method="avg",
                tli_far_select="4bit", tli_near_select="4bit")
    base.update(kw)
    return types.SimpleNamespace(**base)


# ================================================================ R1：async 单射

def test_R1_async_identity():
    ns_off = _base_ns(tia_enable_async_topk=False)
    ns_on = _base_ns(tia_enable_async_topk=True)
    ns_def = _base_ns()   # 无该属性 → getattr 兜底 False（缺省等价）

    m_off = INFO.get_treatment_manifest_json(ns_off)
    m_on = INFO.get_treatment_manifest_json(ns_on)
    m_def = INFO.get_treatment_manifest_json(ns_def)
    _check(m_off != m_on,
           f"079① async 开关未进 manifest（同步/异步同 manifest = 079 "
           f"回归）：off={m_off!r} on={m_on!r}")
    _check(m_off == m_def,
           f"缺省（无属性）与显式 False 应同 manifest（缺省等价）："
           f"{m_def!r} vs {m_off!r}")
    _check(json.loads(m_on)["tia_enable_async_topk"] is True
           and json.loads(m_off)["tia_enable_async_topk"] is False,
           "manifest 中 async 生效布尔值不符")

    h_off, h_on = INFO._treatment_hash(ns_off), INFO._treatment_hash(ns_on)
    _check(h_off != h_on, f"async 开关未改 treatment hash：{h_off} == {h_on}")

    n_off = INFO.get_method_name_with_info(ns_off)
    n_on = INFO.get_method_name_with_info(ns_on)
    _check(n_off != n_on,
           f"async 开关未改 method_name（同名互覆 = 079 回归）："
           f"{n_off!r} == {n_on!r}")
    # 可读段逐位不动（B09 口径），分离器是 hash 段——记录口径供审计
    r_off, hs_off = INFO.split_method_name_hash(n_off)
    r_on, hs_on = INFO.split_method_name_hash(n_on)
    _check(r_off == r_on and hs_off != hs_on,
           f"async 分离应只落在 hash 段：{r_off!r} vs {r_on!r}")
    # TIA（非 TLI）可读名自带 _async——079 范围是 tli，此处锁口径
    tia_off = INFO.get_method_name_with_info(types.SimpleNamespace(
        method="tia", tia_block_size=64, tia_level1_topk=128,
        tia_level2_topk=1024, tia_level2_cmp_ratio=4,
        tia_enable_async_topk=False))
    tia_on = INFO.get_method_name_with_info(types.SimpleNamespace(
        method="tia", tia_block_size=64, tia_level1_topk=128,
        tia_level2_topk=1024, tia_level2_cmp_ratio=4,
        tia_enable_async_topk=True))
    _check(tia_off != tia_on and tia_on.endswith("_async"),
           f"tia 名 _async 回归：{tia_off!r} vs {tia_on!r}")
    return "PASS"


# ================================================================ R2：文件内容身份 + 三拒

def _mk_basis(path, nl=4, hkv=2, r=8, seed=1):
    g = torch.Generator().manual_seed(seed)
    torch.save(torch.randn(nl, hkv, 128, r, generator=g).float(), path)


def test_R2_file_identity_and_gate(base):
    d = os.path.join(base, "r2")
    os.makedirs(d, exist_ok=True)
    bp = os.path.join(d, "basis.pt")
    _mk_basis(bp, seed=1)

    # a) 同路径 basis 内容 A→B：manifest/hash 必变（修复前同 manifest）
    sha_A = hashlib.sha256(open(bp, "rb").read()).hexdigest()
    m1 = INFO.get_treatment_manifest_json(_base_ns(tli_proj_basis=bp))
    h1 = INFO._treatment_hash(_base_ns(tli_proj_basis=bp))
    _mk_basis(bp, seed=2)                       # 同路径、不同内容
    m2 = INFO.get_treatment_manifest_json(_base_ns(tli_proj_basis=bp))
    h2 = INFO._treatment_hash(_base_ns(tli_proj_basis=bp))
    _check(m1 != m2 and h1 != h2,
           f"079② 同路径 basis 内容变更未改 manifest/hash（={h1}）："
           f"m1={m1!r} m2={m2!r}")
    id1 = json.loads(m1)["tli_proj_basis"]
    _check(id1["sha256"] == sha_A,
           f"basis 身份 sha256 应等于文件字节 sha256：{id1['sha256']!r} "
           f"vs {sha_A!r}")
    _check(isinstance(id1.get("shape"), list) and len(id1["shape"]) == 4,
           f"basis 内容身份应含 shape 摘要：{id1!r}")
    _check(id1["path"] == os.path.realpath(bp),
           f"basis 身份应记规范路径（realpath）：{id1['path']!r}")

    # LongBench 三拒模式（076 套件同款）：目标 + sidecar(m1)，
    # 内容变更后 manifest=m2 → gate 必拒且目标字节不变
    out = os.path.join(d, "hotpotqa-" + INFO.get_method_name_with_info(
        _base_ns(tli_proj_basis=bp)) + "-09090909.jsonl")
    data_bytes = b'{"pred": "arm-A"}\n'
    with open(out, "wb") as f:
        f.write(data_bytes)
    sidecar = out + INFO.TREATMENT_MANIFEST_SIDECAR_SUFFIX
    with open(sidecar, "w", encoding="utf-8") as f:
        f.write(m1)
    refused = False
    try:
        INFO.gate_output_treatment_identity(out, m2)
    except SystemExit as e:
        refused = "GATE-FAIL" in str(e)
    _check(refused, "079② 同路径内容变更后写门未拒（覆盖放行 = 079 回归）")
    with open(out, "rb") as f:
        _check(f.read() == data_bytes, "拒绝路径不得改写目标字节")
    sc = INFO.gate_output_treatment_identity(out, m1)   # 同 manifest 幂等
    _check(sc == sidecar, f"sidecar 路径约定漂移：{sc!r}")

    # b) layer_skip 同口径（n_skip 摘要）
    sp = os.path.join(d, "skip.json")
    with open(sp, "w") as f:
        json.dump({"skip": [1, 3]}, f)
    ms1 = INFO.get_treatment_manifest_json(_base_ns(tli_layer_skip_path=sp))
    hs1 = INFO._treatment_hash(_base_ns(tli_layer_skip_path=sp))
    with open(sp, "w") as f:
        json.dump({"skip": [1, 3, 5]}, f)       # 同路径内容变更
    ms2 = INFO.get_treatment_manifest_json(_base_ns(tli_layer_skip_path=sp))
    hs2 = INFO._treatment_hash(_base_ns(tli_layer_skip_path=sp))
    _check(ms1 != ms2 and hs1 != hs2,
           f"079② 同路径 skip 内容变更未改 manifest/hash：{ms1!r}")
    _check(json.loads(ms1)["tli_layer_skip_path"]["n_skip"] == 2,
           "skip 身份应含 n_skip 摘要")

    # c) fail closed：缺失/坏 basis/坏 skip JSON → [GATE-FAIL] 079
    def _expect_gate_fail(ns, tag):
        try:
            INFO.get_treatment_manifest_json(ns)
        except SystemExit as e:
            _check("GATE-FAIL" in str(e) and "079" in str(e),
                   f"{tag} 拒绝消息应带 GATE-FAIL+079：{e!r}")
            return
        _check(False, f"{tag} 未 fail closed（079 回归）")

    _expect_gate_fail(_base_ns(tli_proj_basis=os.path.join(d, "missing.pt")),
                      "basis 文件缺失")
    bad_b = os.path.join(d, "bad_basis.pt")
    with open(bad_b, "wb") as f:
        f.write(b"not a tensor file")
    _expect_gate_fail(_base_ns(tli_proj_basis=bad_b), "basis 非法内容")
    bad_s = os.path.join(d, "bad_skip.json")
    with open(bad_s, "w") as f:
        f.write("{not-json")
    _expect_gate_fail(_base_ns(tli_layer_skip_path=bad_s), "skip 坏 JSON")
    no_key = os.path.join(d, "nokey_skip.json")
    with open(no_key, "w") as f:
        json.dump({"other": [1]}, f)
    _expect_gate_fail(_base_ns(tli_layer_skip_path=no_key),
                      "skip 缺 skip 键")

    # d) None → null；确定性（重复调用逐位稳定）
    m_none = INFO.get_treatment_manifest_json(_base_ns())
    _check(json.loads(m_none)["tli_proj_basis"] is None
           and json.loads(m_none)["tli_layer_skip_path"] is None,
           f"无文件配置 manifest 应为 null：{m_none!r}")
    _check(INFO.get_treatment_manifest_json(_base_ns()) == m_none,
           "manifest 重复求值不稳定（确定性破坏）")
    _check(INFO._treatment_hash(_base_ns()) == INFO._treatment_hash(_base_ns()),
           "hash 重复求值不稳定")
    return "PASS"


# ================================================================ R3：mutation test

# 白名单：add_sparse_attn_args 注册但对 tli treatment 身份惰性的参数
# （逐项理由——mutation 测试的显式豁免记录，新增豁免必须补理由）。
_R3_WHITELIST = {
    # method 是臂选择器：tli 名格式本身蕴含 method=tli；quest/twia/tia/
    # none 各自可读名已编码其全部旋钮（076 审计范围 = tli treatment 矩阵）
    "method": None,
    # quest/twi 参数在 method=tli 下不被消费（对 tli 身份惰性——测试
    # 中同时断言突变后 tli 名不变，惰性是被验证的而非被假设的）
    "quest_block_size": 32,
    "quest_topk": 8,
    "twi_block_size": 32,
    "twi_level1_topk": 64,
    "twi_level2_topp": 0.9,
}
# readable 编码字段：不进 manifest（ABD 三字符已单射），但全名必变
_R3_READABLE = {"tli_enable_kmeans": False, "tli_enable_layer_skip": False}
# manifest 文件字段：突变值 = 测试运行时在 base 下创建的真实文件
# （内容身份解析路径；同路径内容变更通道由 R2 覆盖，此处测路径切换）
_R3_FILE_FIELDS = ("tli_proj_basis", "tli_layer_skip_path")
# manifest 字段突变值（argparse 缺省 → 突变）
_R3_MUTATIONS = {
    "tia_block_size": 32,
    "tia_level1_topk": 64,
    "tia_level2_topk": 512,
    "tia_level2_cmp_ratio": 4,
    "tia_enable_async_topk": True,
    "tli_enable_subspace": False,
    "tli_subspace": "rope",
    "tli_alpha": 0.25,
    "tli_beta": 0.25,
    "tli_gamma": 0.5,
    "tli_far_method": "avg",
    "tli_near_method": "minmax",
    "tli_far_select": "cluster",
    "tli_near_select": "sim_greedy",
    "tli_far_clusters": 128,
    "tli_far_niter": 20,
    "tli_far_blocks": 8,
    "tli_far_tokens": 256,
    "tli_sim": 0.95,
    "tli_sim_dims": "nope",
    "tli_sigma_select": "far",
    "tli_sigma": 4.0,
    "tli_moba": True,
    "tli_per_q_head": True,
    "tli_static_pair": True,
}


def test_R3_mutation_over_argparse(base):
    d = os.path.join(base, "r3")
    os.makedirs(d, exist_ok=True)
    bp = os.path.join(d, "basis.pt")
    _mk_basis(bp, seed=7)
    sp = os.path.join(d, "skip.json")
    with open(sp, "w") as f:
        json.dump({"skip": [2]}, f)

    parser = argparse.ArgumentParser()
    add_sparse_attn_args(parser)
    dests = sorted(a.dest for a in parser._actions if a.dest != "help")
    ns_base = parser.parse_args(["--method", "tli"])
    m_base = INFO.get_treatment_manifest_json(ns_base)
    n_base = INFO.get_method_name_with_info(ns_base)
    h_base = INFO._treatment_hash(ns_base)

    manifest_fields = set(INFO._TREATMENT_FIELD_DEFAULTS)
    # tli_sparse_prefill 在 manifest 但不经 argparse 注册（程序化注入）——
    # argparse→manifest 完备性只查注册面；manifest 侧由 076/079 套件覆盖
    covered = (set(_R3_MUTATIONS) | set(_R3_READABLE) | set(_R3_WHITELIST)
               | set(_R3_FILE_FIELDS))
    uncovered = [x for x in dests if x not in covered]
    _check(not uncovered,
           f"add_sparse_attn_args 存在未登记参数 {uncovered}——每个可独立"
           f"改变行为的参数必须进 manifest、readable 编码或白名单（注明"
           f"理由），否则 treatment 单射出现缺口（079 教训）")
    _check(manifest_fields <= covered | {"tli_sparse_prefill"},
           f"manifest 字段未被本测试覆盖：{sorted(manifest_fields - covered)}")

    for dest in dests:
        if dest in _R3_WHITELIST and _R3_WHITELIST[dest] is None:
            continue   # method：臂选择器，突变换名格式无可比性（理由见白名单）
        ns = parser.parse_args(["--method", "tli"])
        if dest == "tli_proj_basis":
            val = bp
        elif dest == "tli_layer_skip_path":
            val = sp
        elif dest in _R3_MUTATIONS:
            val = _R3_MUTATIONS[dest]
        elif dest in _R3_READABLE:
            val = _R3_READABLE[dest]
        else:
            val = _R3_WHITELIST[dest]
        setattr(ns, dest, val)
        m = INFO.get_treatment_manifest_json(ns)
        h = INFO._treatment_hash(ns)
        n = INFO.get_method_name_with_info(ns)
        if dest in _R3_MUTATIONS or dest in _R3_FILE_FIELDS:
            _check(m != m_base and h != h_base and n != n_base,
                   f"manifest 字段 {dest} 突变未改 manifest/hash/名"
                   f"（079 单射缺口）：h={h} base={h_base}")
        elif dest in _R3_READABLE:
            _check(n != n_base,
                   f"readable 编码字段 {dest} 突变未改全名（B/D 段回归）")
        else:   # 白名单：对 tli 身份惰性（被验证而非被假设）
            _check(n == n_base,
                   f"白名单字段 {dest} 突变竟改变了 tli 名（白名单理由不"
                   f"成立，须移入 manifest）：{n!r} vs {n_base!r}")
    # γ=off（None，JSON null 渲染）独立通道
    ns = parser.parse_args(["--method", "tli"])
    ns.tli_gamma = None
    _check(INFO._treatment_hash(ns) != h_base
           and INFO.get_method_name_with_info(ns) != n_base,
           "γ=None（off）突变未改 hash/名")
    return "PASS"


# ================================================================ R4：async 行为差异

_R4_BS, _R4_S = 64, 1024


def _r4_args(async_topk):
    return types.SimpleNamespace(
        method="tli", tia_block_size=_R4_BS, tia_level1_topk=5,
        tia_level2_topk=384, tia_level2_cmp_ratio=4,
        tia_enable_async_topk=async_topk,
        tli_enable_subspace=True, tli_subspace="full",
        tli_enable_kmeans=False, tli_enable_layer_skip=False,
        tli_far_select="4bit", tli_near_select="4bit",
        tli_far_method="minmax", tli_near_method="avg",
        tli_alpha=0.0, tli_beta=0.0, tli_gamma=1.0,
        tli_per_q_head=False, tli_static_pair=False, tli_moba=False,
        tli_sigma_select="none", tli_sigma=8.0, tli_sim=0.9,
        tli_sim_dims="subspace", tli_proj_basis=None,
        tli_layer_skip_path=None, tli_sparse_prefill=False)


def _r4_marker_inputs():
    """确定性 marker 输入：block b（2..13）的 k 在细筛维 110+b 上置
    10（cmp_ratio=4 → 细筛维 112..127），q1 只在 112..115 非零（选块
    2..5），q2 只在 120..123 非零（选块 10..13）——两组块互斥，
    async 第 2 步沿用第 1 步块选择的效应可被逐位观测。"""
    S = _R4_S
    k = torch.zeros(1, S, 2, 128)
    for b in range(2, 14):
        k[0, b * _R4_BS:(b + 1) * _R4_BS, :, 110 + b] = 10.0
    q1 = torch.zeros(1, 1, 4, 128)
    q1[..., 112:116] = 100.0
    q2 = torch.zeros(1, 1, 4, 128)
    q2[..., 120:124] = 100.0
    return k, q1, q2


def _r4_decode(async_topk, qs):
    """同一 indexer 实例上跑多 decode 步（prev_mask 状态跨步保留）。"""
    idx = TLIIndexer(_r4_args(async_topk))
    idx.layer_idx = 1
    k, _, _ = _r4_marker_inputs()
    cu = torch.tensor([0, _R4_S])
    index_dict = idx.prepare_index(k, cu)
    masks = []
    for q in qs:
        sd = idx.compute_score(q, torch.tensor([_R4_S - 1]), index_dict,
                               128 ** -0.5)
        masks.append(idx.compute_mask(torch.tensor([_R4_S - 1]), sd))
    return idx, masks


def _r4_mid_any(mask, lo, hi):
    """mid 区 [lo,hi) 是否存在被选 token（sink/swa 强制区外）。"""
    m = mask.squeeze(0).squeeze(0)   # [Hkv, T]
    return bool(m[..., lo:hi].any().item())


def _r4_mid_all(mask, lo, hi):
    """mid 区 [lo,hi) 是否全 head 全被选（k1=5/K2=384 下允许块全区覆盖）。"""
    m = mask.squeeze(0).squeeze(0)   # [Hkv, T]
    return bool(m[..., lo:hi].all().item())


def test_R4_async_behavior_divergence():
    k, q1, q2 = _r4_marker_inputs()
    idx_a, m_async = _r4_decode(True, [q1, q2])
    _, m_sync = _r4_decode(False, [q1, q2])

    # ① 第 1 步 async == 同步（prev_mask 未生效；GPT 复审边界确认）
    _check(torch.equal(m_async[0], m_sync[0]),
           "async 第 1 步 mask 应与同步路径逐位一致（079 从第 2 步起触发）")
    _check(idx_a.prev_mask is not None, "async 第 1 步应缓存 prev_mask")

    # ② 同步路径的块选择随 q 变化（k1=5：当前块 15 + q 的 4 个 marker 块；
    #    K2=384：swa 128 + mid 允许块 256 token 全覆盖 → 全块观测免 tie-break）
    _check(_r4_mid_all(m_sync[0], 128, 384)
           and not _r4_mid_any(m_sync[0], 384, 896),
           "同步 step1 mid 选择应落 q1 块区 [128,384)（blocks 2..5）")
    _check(_r4_mid_all(m_sync[1], 640, 896)
           and not _r4_mid_any(m_sync[1], 128, 640),
           "同步 step2 mid 选择应落 q2 块区 [640,896)（blocks 10..13）")

    # ③ async 第 2 步候选 = 第 1 步的块选择 → 与同步第 2 步不同
    #    （079① 是输出语义而非性能提示的实证：两组候选块互斥）
    _check(_r4_mid_all(m_async[1], 128, 384),
           "async step2 应沿用 step1 的 q1 块区候选（[128,384) 全选）")
    _check(not _r4_mid_any(m_async[1], 384, 896),
           "async step2 不应选入 q2 独占块区（沿用 step1 候选集）")
    _check(not torch.equal(m_async[1], m_sync[1]),
           "async step2 mask 与同步路径相同（候选替换未生效或输入未触发差异）"
           "——079 行为差异断言失败")
    return "PASS"


# ================================================================ R5：RULER receipt 接线

def test_R5_receipt_treatment_manifest():
    ns = _base_ns(tli_proj_basis=None)
    manifest_json = INFO.get_treatment_manifest_json(ns)
    receipt = build_yarn_receipt(
        yarn_enabled=False, effective_factor=None, yarn_factor_cli=None,
        rope_scaling=None, context_length=32768, task="vt",
        model_path="/tmp/fake-model-079", model_config_sha256=None,
        native_mpe=32768,
        generation_params={"max_gen": 128, "max_num": 0, "seed": 42,
                           "method": "tli", "pred_postfix": "_x", "t": "1"},
        producer_script_path="benchmark/RULER/pred_ruler.py",
        producer_script_sha256=hashlib.sha256(b"079-test").hexdigest(),
        run_id="r5-1", prediction_basename="vt-x.jsonl",
        prediction_sha256="a" * 64, prediction_lines=1,
        treatment_manifest_json=manifest_json)
    tm = receipt.get("treatment_manifest")
    _check(tm is not None and _validate_common_schema(
        receipt, "vt-x.jsonl") is None,
        f"含 treatment_manifest 的回执应过 schema：{tm!r}")
    _check(tm["sha256"] == hashlib.sha256(
        manifest_json.encode("utf-8")).hexdigest(),
        "treatment_manifest.sha256 应等于 manifest json 字节的 sha256")

    # 篡改 json / sha → schema 拒收（fail closed）
    bad = dict(receipt)
    bad["treatment_manifest"] = {"sha256": tm["sha256"], "json": "{}"}
    _check(_validate_common_schema(bad, "vt-x.jsonl") is not None,
           "篡改 treatment_manifest.json 后 schema 应拒收")
    bad2 = dict(receipt)
    bad2["treatment_manifest"] = {"sha256": "0" * 64, "json": tm["json"]}
    _check(_validate_common_schema(bad2, "vt-x.jsonl") is not None,
           "篡改 treatment_manifest.sha256 后 schema 应拒收")

    # legacy/非 tli：无该键 → 前向兼容（不写不校验）
    legacy = build_yarn_receipt(
        yarn_enabled=False, effective_factor=None, yarn_factor_cli=None,
        rope_scaling=None, context_length=32768, task="vt",
        model_path="/tmp/fake-model-079", model_config_sha256=None,
        native_mpe=32768,
        generation_params={"max_gen": 128, "max_num": 0, "seed": 42,
                           "method": "none", "pred_postfix": "_x", "t": "1"},
        producer_script_path="benchmark/RULER/pred_ruler.py",
        producer_script_sha256=hashlib.sha256(b"079-test").hexdigest(),
        run_id="r5-2", prediction_basename="vt-x.jsonl",
        prediction_sha256="a" * 64, prediction_lines=1)
    _check("treatment_manifest" not in legacy
           and _validate_common_schema(legacy, "vt-x.jsonl") is None,
           "legacy 回执（无 treatment_manifest）应前向兼容通过")
    return "PASS"


# ================================================================ main

def main():
    """067 纪律：PASS/SKIP/FAIL 三分显式计数；异常 → FAIL 继续跑完其余
    用例；任一 SKIP/FAIL → 非全绿且非零退出。TLI079_ONLY 子集过滤。"""
    base = tempfile.mkdtemp(prefix="e121_fix_079_")
    plan = [
        ("R1", test_R1_async_identity),
        ("R2", lambda: test_R2_file_identity_and_gate(base)),
        ("R3", lambda: test_R3_mutation_over_argparse(base)),
        ("R4", test_R4_async_behavior_divergence),
        ("R5", test_R5_receipt_treatment_manifest),
    ]
    only = os.environ.get("TLI079_ONLY", "")
    if only:
        keep = {x.strip() for x in only.split(",") if x.strip()}
        plan = [p for p in plan if p[0] in keep]
    n_pass = n_fail = 0
    failed = []
    try:
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
                      f"{traceback.format_exc()[-1200:]}", flush=True)
                continue
            n_pass += 1
            print(f"[{name}] PASS", flush=True)
    finally:
        shutil.rmtree(base, ignore_errors=True)
    print(f"\nE121-FIX-079 RESULT: PASS={n_pass} FAIL={n_fail} "
          f"(total {len(plan)})")
    if failed:
        print(f"FAILED: {failed}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
