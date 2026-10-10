#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""081（TL-E121-OUTPUT-SNAPSHOT-081，GPT 2026-10-11 复审 P1）修复验收套件
（CPU only + torch；python 与 python -O 双跑安全——全部显式判定，无 assert）。

背景：079 的修复共用了解析「函数」，但没有冻结一次解析后的「值」——
同一文件在一次运行内被多次打开：
  ① info.py 的 get_treatment_manifest_json 每次调用都重新打开
     tli_proj_basis / tli_layer_skip_path 求内容身份；
  ② patch.py 对每个 attention 层分别建 TLIIndexer(args)，__init__ 的
     D′ 掩码段（每实例 open(mask_path)）与投影基段（每实例
     torch.load(bp_path)）都按层重读 → 单模型跨层混代可达；
  ③ LongBench pred.py 382 加载/patch（文件已被各层消费）→ 459-470
     生成 → 472 重新求 method_name → 501 再读求 sidecar → 生成窗口内
     文件被替换则 runtime A / 文件名 B / sidecar C 混装；RULER
     pred_ruler.py 164-190-297 同型；
  ④ 默认 D′ 掩码遗漏：缺省 enable=True + path=None 时运行时读 tracked
     DEFAULT_MASK，manifest 却恒 null（可读名含 D）→ 身份缺口；
  ⑤ 反向不对称：enable=False + 显式 path 时 manifest 强制解析、运行时
     根本不读——身份按 argv 声明而非实际消费。

修复（sparse_attn/info.py + tli_indexer.py + patch.py + 两个 benchmark
入口 + yarn_receipt.py）：
  A. resolve_treatment_snapshot(args) 一次解析 → 不可变 TreatmentSnapshot
     （manifest / manifest_json / skip_ids / basis_tensor），每文件单次
     open、SHA 与内容解析共用同一批 bytes；
  B. D′ 有效值语义：enable=True 才解析（path=None 展开 DEFAULT_MASK）；
     enable=False 时路径声明不进 manifest（null）；
  C. benchmark 入口在模型加载前冻结 snapshot，method_name / sidecar /
     receipt 全程消费同一对象；TLIIndexer(args, snapshot=...) 注入，
     不再按层重开；register_patch(model, args, snapshot) 透传；
  D. yarn_receipt 闭包校验：tm 存在时其 sha256[:10] 必须等于
     prediction_basename 的 _h<hash10> 段；无 _h 段但 manifest 存在 →
     fail closed；legacy（无 tm 键）放行；不改 effective_config_sha256。

测试矩阵（065/067 纪律：PASS/SKIP/FAIL 三态、显式 _check、FAIL 继续跑）：
  R1  混装主负例：resolve（mask=[1,3]+basis A）→ 原子替换同路径文件 →
      生成名 / sidecar / receipt hash 仍全部等于 A 的 snapshot hash
      （冻结生效）；无 snapshot 重解析已变（对照修复前混装可达）。
  R2  跨层注入：两层 TLIIndexer 共享同一 snapshot，patch 中途替换文件
      → 两层 skip_ids 相同且等于 A；投影基同口径；快照/args 失配守卫
      （skip_ids None / basis_tensor None + args 声明 → [GATE-FAIL] 081）。
  R3  默认掩码入 manifest：默认配置（path=None, enable=True）→ manifest
      含 tracked DEFAULT_MASK 的 realpath+sha256+n_skip（真实文件，
      SHA 9902254a…，1223 bytes），method_name hash 与「无掩码信息」的
      旧 hash 不同；DEFAULT_MASK 缺失/损坏（monkeypatch 常量）→
      resolve 阶段 [GATE-FAIL] 而非静默关 D′。
  R4  反向不对称修复：enable=False + 显式 path → manifest null（不强制
      存在）且 hash 等价于无路径配置；enable=True + 显式 path → 解析。
  R5  receipt 闭包：tm json 的 hash[:10] ≠ basename _h 段 → 拒收；
      相等 → 通过；无 _h 段但 manifest 存在 → 拒收；legacy 放行。
  R6  symlink retarget：resolve 后 symlink 换目标 → 快照身份不变
      （skip 与 basis 双口径），重解析得新身份。
  K1  kimi3 0316 追加修复 1：truncate 缺省 limit 245 → 230——sidecar
      链（out_fn+".jsonl"+".tli_manifest.json"）绑 ext4 255 上限
      （kimi3 实测 245 截断下 sidecar 实名 269 → OSError 36 裸
      traceback）；断言截断名全链 ≤255、sidecar 实际可创建、
      pred.py 阈值同源。
  K2  kimi3 0316 追加修复 2：tli_enable_kmeans/tli_enable_layer_skip
      入 _TREATMENT_FIELD_DEFAULTS（076 residual「可读段 B/D 已单射」
      被截断反例推翻）；cfg(True,False) vs cfg(False,True) 极端长
      prefix 截断后 trunc/hash/manifest 三互异（kimi3 反例口径）。

torch 使用口径与 079 套件同源：只做 import / torch.save / torch.load /
shape / equal（CPU 确定性），不声称 GPU 或 tensor kernel 实测。

红态对照（逻辑说明，基点 ea439fbd1）：修复前 info.py 无
resolve_treatment_snapshot / TreatmentSnapshot → 本套件 import 即
ImportError 全红；逐项而言 R1 的「生成名/sidecar/receipt 在文件替换
后仍等于 A」不成立（每层各读一次 + 生成后重读 → B 值混入）、R3 默认
配置 manifest 恒 null（身份缺口 ④）、R4 enable=False 仍强制解析
（缺口 ⑤）、R5 yarn_receipt 无 tm↔basename 比较代码（schema 只查键内
自洽）。本套件对主树可作红态复跑（git stash 后运行即红）。

用法：
  cd <repo>/two-level-attention && python3 test_e121_fix_081.py
  python -O 同上双跑；TLI081_ONLY=R1,R5 子集过滤。
"""
import hashlib
import json
import os
import shutil
import sys
import tempfile
import types

sys.dont_write_bytecode = True

import torch   # info.py 模块级依赖 + basis fixture + R2 跨层注入

torch.set_num_threads(1)

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import sparse_attn.info as INFO                     # noqa: E402
from sparse_attn.info import (                      # noqa: E402
    TreatmentSnapshot, resolve_treatment_snapshot)
from sparse_attn.indexer.tli_indexer import (       # noqa: E402
    DEFAULT_MASK, TLIIndexer)
from benchmark.RULER.yarn_receipt import (          # noqa: E402
    build_yarn_receipt, validate_producer_receipt)

# R3 锚点：tracked 默认 D′ 掩码的内容身份（exp/trace/results/
# tli_layer_skip_mask.json，1223 bytes，n_skip=13——081 身份缺口 ① 的
# 「默认配置 manifest 从 null 升级为内容身份」锚）。
_DEFAULT_MASK_SHA = ("9902254a10361cf209af70b862fb3fb8"
                     "42f6697de391888324bd156dd29f4b23")
_DEFAULT_MASK_N_SKIP = 13


def _check(cond, msg=""):
    """065/067 纪律：显式判定，非 assert（python -O 不删除）。"""
    if not cond:
        raise SystemExit(f"[TEST-FAIL] {msg}")


def _base_ns(**kw):
    """最小 treatment namespace（getattr 兜底缺省；与 079 套件同构）。"""
    base = dict(method="tli", tia_block_size=64, tia_level1_topk=128,
                tia_level2_topk=2048, tia_level2_cmp_ratio=4,
                tia_enable_async_topk=False,
                tli_alpha=0.25, tli_beta=0.125, tli_gamma=0.625,
                tli_enable_kmeans=False, tli_enable_layer_skip=False,
                tli_far_method="minmax", tli_near_method="avg",
                tli_far_select="4bit", tli_near_select="4bit")
    base.update(kw)
    return types.SimpleNamespace(**base)


def _mk_basis(path, nl=4, hkv=2, r=8, seed=1):
    g = torch.Generator().manual_seed(seed)
    torch.save(torch.randn(nl, hkv, 128, r, generator=g).float(), path)
    return torch.load(path, map_location="cpu", weights_only=True)


def _atomic_write(path, data_bytes):
    """原子替换：临时文件 + os.replace（模拟「生成窗口内文件被替换」）。"""
    tmp = path + ".swap"
    with open(tmp, "wb") as f:
        f.write(data_bytes)
    os.replace(tmp, path)


def _sha(b):
    return hashlib.sha256(b).hexdigest()


# ================================================================ R1：混装主负例

def test_R1_freeze_survives_midrun_replacement(base):
    d = os.path.join(base, "r1")
    os.makedirs(d, exist_ok=True)
    sp = os.path.join(d, "skip.json")
    with open(sp, "w") as f:
        json.dump({"skip": [1, 3]}, f)
    bp = os.path.join(d, "basis.pt")
    _mk_basis(bp, seed=1)

    ns = _base_ns(tli_enable_layer_skip=True, tli_layer_skip_path=sp,
                  tli_proj_basis=bp, tli_subspace="tail")
    # 冻结时刻（模型加载前）：A 内容的一次解析
    snap = resolve_treatment_snapshot(ns)
    frozen_name = INFO.get_method_name_with_info(ns, snap)
    frozen_mj = INFO.get_treatment_manifest_json(ns, snap)
    frozen_sha_A_skip = snap.manifest["tli_layer_skip_path"]["sha256"]
    frozen_sha_A_basis = snap.manifest["tli_proj_basis"]["sha256"]
    _check(frozen_sha_A_skip == _sha(json.dumps({"skip": [1, 3]}).encode()),
           f"冻结掩码 sha 应为 A 内容：{frozen_sha_A_skip!r}")

    # ---- 生成窗口内同路径原子替换为 B（mask [2,4] + basis seed2）----
    _atomic_write(sp, json.dumps({"skip": [2, 4]}).encode())
    _mk_basis(bp, seed=2)

    # ① 生成名/sidecar 仍全部等于 A 的 snapshot hash（冻结生效）
    _check(INFO.get_method_name_with_info(ns, snap) == frozen_name,
           "文件替换后 method_name 漂移（未消费冻结 snapshot = 081 回归）")
    _check(INFO.get_treatment_manifest_json(ns, snap) == frozen_mj,
           "文件替换后 manifest 漂移（未消费冻结 snapshot = 081 回归）")
    _check(INFO._treatment_hash(ns, snap)
           == _sha(frozen_mj.encode("utf-8"))[:10],
           "hash 与冻结 manifest_json 非同源")
    # ② 无 snapshot 重解析已读到 B（对照：修复前该值会被生成后的
    #    重读混进文件名/sidecar → runtime A / name B / sidecar C 混装）
    m_live = INFO.get_treatment_manifest_json(ns)
    _check(m_live != frozen_mj,
           "替换后重解析应读到 B（此处只为锚定对照语义，冻结断言在上面）")
    _check(json.loads(m_live)["tli_layer_skip_path"]["n_skip"] == 2,
           "重解析应读到 B 的 n_skip=2")

    # ③ LongBench sidecar 写门按冻结 A 消费：生产顺序 = 先 gate（目标
    #    不存在 → 放行返回 sidecar 路径）→ 写 sidecar → 写目标数据；
    #    同 manifest 幂等放行；用 B 的重解析 manifest → 必拒且字节不变
    out = os.path.join(d, "hotpotqa-" + frozen_name + "-09090909.jsonl")
    data_bytes = b'{"pred": "arm-A"}\n'
    sidecar = INFO.gate_output_treatment_identity(out, frozen_mj)
    with open(sidecar, "w", encoding="utf-8") as f:
        f.write(frozen_mj)
    with open(out, "wb") as f:
        f.write(data_bytes)
    sc2 = INFO.gate_output_treatment_identity(out, frozen_mj)
    _check(sc2 == sidecar, f"sidecar 幂等放行失败：{sc2!r}")
    refused = False
    try:
        INFO.gate_output_treatment_identity(out, m_live)
    except SystemExit as e:
        refused = "GATE-FAIL" in str(e)
    _check(refused, "B 的重解析 manifest 应被写门拒绝（079 门 + 081 冻结）")
    with open(out, "rb") as f:
        _check(f.read() == data_bytes, "拒绝路径不得改写目标字节")

    # ④ RULER receipt hash 段仍等于 A：basename 的 _h 段与冻结 manifest
    #    的全长 sha256 同源（081 闭包校验的结构性成立前提）
    pred_bn = f"vt-{frozen_name}-09090909.jsonl"
    receipt = build_yarn_receipt(
        yarn_enabled=False, effective_factor=None, yarn_factor_cli=None,
        rope_scaling=None, context_length=32768, task="vt",
        model_path="/tmp/fake-model-081", model_config_sha256=None,
        native_mpe=32768,
        generation_params={"max_gen": 64, "max_num": 0, "seed": 42,
                           "method": "tli", "pred_postfix": "_x",
                           "t": "09090909"},
        producer_script_path="benchmark/RULER/pred_ruler.py",
        producer_script_sha256=_sha(b"081-test"),
        run_id="r1-1", prediction_basename=pred_bn,
        prediction_sha256="a" * 64, prediction_lines=1,
        treatment_manifest_json=frozen_mj)
    err = validate_producer_receipt(receipt, os.path.join(d, pred_bn))
    _check(err is None, f"冻结 manifest 的回执应过全部校验（含 081 闭包）：{err}")
    tm = receipt["treatment_manifest"]
    _check(tm["sha256"] == _sha(frozen_mj.encode("utf-8")),
           "receipt tm.sha256 应等于冻结 manifest 字节的 sha256")
    # basename _h 段 == tm.sha256[:10] == 冻结 hash（A 的身份，非 B）
    _check(frozen_name.rsplit("_h", 1)[1] == tm["sha256"][:10],
           f"basename _h 段与冻结 manifest 身份失配：{frozen_name!r}")
    return "PASS"


# ================================================================ R2：跨层注入

def test_R2_crosslayer_shared_snapshot(base):
    d = os.path.join(base, "r2")
    os.makedirs(d, exist_ok=True)
    sp = os.path.join(d, "skip.json")
    with open(sp, "w") as f:
        json.dump({"skip": [1, 3]}, f)
    bp = os.path.join(d, "basis.pt")
    basis_A = _mk_basis(bp, seed=1)

    def mk(**kw):
        ns = _base_ns(tli_enable_layer_skip=True, tli_layer_skip_path=sp,
                      tli_proj_basis=bp, tli_subspace="tail")
        for k, v in kw.items():
            setattr(ns, k, v)
        return ns

    args = mk()
    snap = resolve_treatment_snapshot(args)
    # 层 1 构造（模拟 patch 第一个 attention 层）
    i1 = TLIIndexer(args, snapshot=snap)
    i1.layer_idx = 1
    # ---- patch 中途替换文件（旧口径：下一个层的实例会读到新内容）----
    _atomic_write(sp, json.dumps({"skip": [2, 4]}).encode())
    _mk_basis(bp, seed=2)
    # 层 3 构造（共享同一 snapshot 对象）
    i2 = TLIIndexer(args, snapshot=snap)
    i2.layer_idx = 3
    _check(i1._skip_ids == i2._skip_ids == {1, 3},
           f"两层 skip_ids 应同且等于 A：{i1._skip_ids!r} / {i2._skip_ids!r}"
           f"（081 注入失效 = 跨层混装回归）")
    _check(torch.equal(i1._basis_all, basis_A)
           and torch.equal(i2._basis_all, basis_A),
           "两层投影基应同且等于 A（081 注入失效）")
    # D′ 行为一致性：层 1/3 ∈ A 集合 → skip_far；层 0 不在
    i1._resolve_skip()
    i2._resolve_skip()
    _check(i1.skip_far and i2.skip_far,
           "两层 _resolve_skip 应一致生效（layer 1/3 ∈ A 的 {1,3}）")
    i0 = TLIIndexer(args, snapshot=snap)
    i0.layer_idx = 0
    i0._resolve_skip()
    _check(not i0.skip_far, "layer 0 ∉ {1,3} 不应 skip_far")

    # 旧路径（snapshot=None，非 benchmark 入口兼容）：替换后新实例读到 B
    # ——记录该语义作对照（不是回归：兼容路径刻意保留旧行为）
    i3 = TLIIndexer(args)
    _check(i3._skip_ids == {2, 4},
           f"无 snapshot 兼容路径应读到 B（新内容）：{i3._skip_ids!r}")

    # 失配守卫：快照/args 不一致 → [GATE-FAIL] 081（不允许静默退化）
    snap_nobasis = resolve_treatment_snapshot(mk(tli_proj_basis=None))
    try:
        TLIIndexer(args, snapshot=snap_nobasis)
        _check(False, "args 声明 basis 但快照缺 basis 未 fail closed（081）")
    except SystemExit as e:
        _check("GATE-FAIL" in str(e) and "081" in str(e),
               f"basis 失配拒绝消息应带 GATE-FAIL+081：{e!r}")
    snap_noD = resolve_treatment_snapshot(mk(tli_enable_layer_skip=False))
    try:
        TLIIndexer(mk(), snapshot=snap_noD)
        _check(False, "enable=True 但快照 skip_ids=None 未 fail closed（081）")
    except SystemExit as e:
        _check("GATE-FAIL" in str(e) and "081" in str(e),
               f"skip 失配拒绝消息应带 GATE-FAIL+081：{e!r}")
    # enable=False + 注入快照：skip_ids=None 合法（D′ 关闭语义）
    i_off = TLIIndexer(mk(tli_enable_layer_skip=False), snapshot=snap_noD)
    _check(i_off._skip_ids is None and not i_off.enable_layer_skip,
           "enable=False + 注入快照应合法关闭 D′（skip_ids=None）")
    return "PASS"


# ================================================================ R3：默认掩码入 manifest

def test_R3_default_mask_identity(base):
    # 默认配置：tli_enable_layer_skip=True（缺省）、无 tli_layer_skip_path
    # 属性（getattr → None → 展开 DEFAULT_MASK）
    ns = _base_ns(tli_enable_layer_skip=True)
    for k in ("tli_layer_skip_path",):
        if hasattr(ns, k):
            delattr(ns, k)
    snap = resolve_treatment_snapshot(ns)
    ident = snap.manifest["tli_layer_skip_path"]
    _check(isinstance(ident, dict),
           f"默认配置 manifest 的掩码字段应解析为内容身份而非 null：{ident!r}"
           f"（081 身份缺口 ① 回归：可读名含 D 而 manifest 恒 null）")
    # 081 身份口径：默认掩码是 repo tracked 常量，内容身份只记
    # {default, sha256, n_skip}——刻意不含 checkout 相关 realpath：
    # 同内容同 SHA 在 worktree/主仓两 checkout 下 realpath 不同，记
    # path 会让 hash 随 checkout 漂移（实测 h7c17d0b763/h3bbe917587
    # 两值互漂、锚点随 checkout 必红）。显式声明路径（argv
    # provenance）仍记 path——见 R4 ②。
    _check(ident.get("default") is True,
           f"默认掩码身份应带 default 标记：{ident!r}")
    _check("path" not in ident,
           f"默认掩码身份不应记 realpath（checkout 漂移防线）：{ident!r}")
    _check(ident["sha256"] == _DEFAULT_MASK_SHA,
           f"默认掩码 sha256 锚点失配（tracked 文件被改？须重算锚点）："
           f"{ident['sha256']!r}")
    _check(ident["n_skip"] == _DEFAULT_MASK_N_SKIP,
           f"默认掩码 n_skip 锚点失配：{ident['n_skip']!r}")
    # 运行时/manifest 两侧路径单一事实源（tli_indexer.DEFAULT_MASK 引用
    # info 常量）
    _check(os.path.realpath(DEFAULT_MASK)
           == os.path.realpath(INFO.DEFAULT_LAYER_SKIP_MASK_PATH),
           "tli_indexer.DEFAULT_MASK 与 info 常量不同源（081 上移失效）")
    # skip_ids 与 tracked 文件内容一致（运行时注入语义）
    with open(INFO.DEFAULT_LAYER_SKIP_MASK_PATH, "rb") as f:
        tracked_skip = set(json.loads(f.read().decode("utf-8"))["skip"])
    _check(set(snap.skip_ids) == tracked_skip,
           "快照 skip_ids 应等于 tracked 默认掩码内容")

    # method_name hash 与「无掩码信息」的旧 hash 不同（身份升级可观测）
    name = INFO.get_method_name_with_info(ns, snap)
    old_style = dict(snap.manifest)
    old_style["tli_layer_skip_path"] = None       # 081 前的 manifest 口径
    old_h = _sha(json.dumps(old_style, sort_keys=True, ensure_ascii=False,
                            default=str).encode("utf-8"))[:10]
    _check(name.endswith("_h" + old_h) is False,
           f"默认掩码内容身份未改 hash（仍等于无掩码旧口径 {old_h}）——"
           f"081 身份升级未生效：{name!r}")

    # DEFAULT_MASK 缺失/损坏 → resolve 阶段 [GATE-FAIL]（非静默关 D′）
    orig = INFO.DEFAULT_LAYER_SKIP_MASK_PATH
    try:
        INFO.DEFAULT_LAYER_SKIP_MASK_PATH = os.path.join(base, "no_such",
                                                         "mask.json")
        try:
            resolve_treatment_snapshot(ns)
            _check(False, "默认掩码缺失未 fail closed（081 静默退化回归）")
        except SystemExit as e:
            _check("GATE-FAIL" in str(e) and "079" in str(e),
                   f"缺失拒绝消息应带 GATE-FAIL+079 口径：{e!r}")
            _check("缺省 D′ 掩码" in str(e),
                   f"默认掩码展开的拒绝消息应附注 081 缺省语义：{e!r}")
        bad = os.path.join(base, "bad_default_mask.json")
        with open(bad, "w") as f:
            f.write("{not-json")
        INFO.DEFAULT_LAYER_SKIP_MASK_PATH = bad
        try:
            resolve_treatment_snapshot(ns)
            _check(False, "默认掩码损坏未 fail closed（081 静默退化回归）")
        except SystemExit as e:
            _check("GATE-FAIL" in str(e),
                   f"损坏拒绝消息应带 GATE-FAIL：{e!r}")
    finally:
        INFO.DEFAULT_LAYER_SKIP_MASK_PATH = orig
    # 恢复后确定性：重解析与冻结逐位一致
    snap2 = resolve_treatment_snapshot(ns)
    _check(snap2.manifest_json == snap.manifest_json,
           "恢复常量后重解析与首解析不一致（确定性破坏）")
    return "PASS"


# ================================================================ R4：反向不对称

def test_R4_reverse_asymmetry(base):
    d = os.path.join(base, "r4")
    os.makedirs(d, exist_ok=True)
    sp = os.path.join(d, "skip.json")
    with open(sp, "w") as f:
        json.dump({"skip": [1, 3]}, f)

    # ① enable=False + 显式 path：manifest 该字段 null（不强制存在），
    #    hash 等价于无路径配置（身份按「实际消费」而非「argv 声明」）
    ns_off = _base_ns(tli_enable_layer_skip=False, tli_layer_skip_path=sp)
    m_off = json.loads(INFO.get_treatment_manifest_json(ns_off))
    _check(m_off["tli_layer_skip_path"] is None,
           f"enable=False + 显式 path 应记 null（081 反向不对称未修）："
           f"{m_off['tli_layer_skip_path']!r}")
    h_off = INFO._treatment_hash(ns_off)
    h_none = INFO._treatment_hash(_base_ns(tli_enable_layer_skip=False))
    _check(h_off == h_none,
           f"enable=False 时路径声明不应改身份：{h_off} vs {h_none}")
    snap_off = resolve_treatment_snapshot(ns_off)
    _check(snap_off.skip_ids is None,
           "enable=False 的快照 skip_ids 应为 None（运行时不消费掩码）")

    # ② enable=True + 显式 path：解析（有效组合照常消费内容身份）
    ns_on = _base_ns(tli_enable_layer_skip=True, tli_layer_skip_path=sp)
    m_on = json.loads(INFO.get_treatment_manifest_json(ns_on))
    _check(isinstance(m_on["tli_layer_skip_path"], dict)
           and m_on["tli_layer_skip_path"]["n_skip"] == 2,
           f"enable=True + 显式 path 应解析内容身份：{m_on!r}")
    snap_on = resolve_treatment_snapshot(ns_on)
    _check(set(snap_on.skip_ids) == {1, 3},
           "enable=True 的快照 skip_ids 应等于文件内容")
    _check(h_off != INFO._treatment_hash(ns_on),
           "enable 开关翻转未改身份（D 段/hash 双通道都应可区分）")
    return "PASS"


# ================================================================ R5：receipt 闭包

def test_R5_receipt_closure():
    ns = _base_ns(tli_enable_layer_skip=True)   # 默认掩码内容身份进 manifest
    manifest_json = INFO.get_treatment_manifest_json(ns)
    name = INFO.get_method_name_with_info(ns)
    h10 = _sha(manifest_json.encode("utf-8"))[:10]

    def _rcp(basename, tm_json=manifest_json):
        return build_yarn_receipt(
            yarn_enabled=False, effective_factor=None, yarn_factor_cli=None,
            rope_scaling=None, context_length=32768, task="vt",
            model_path="/tmp/fake-model-081", model_config_sha256=None,
            native_mpe=32768,
            generation_params={"max_gen": 64, "max_num": 0, "seed": 42,
                               "method": "tli", "pred_postfix": "_x",
                               "t": "09090909"},
            producer_script_path="benchmark/RULER/pred_ruler.py",
            producer_script_sha256=_sha(b"081-test"),
            run_id="r5", prediction_basename=basename,
            prediction_sha256="a" * 64, prediction_lines=1,
            treatment_manifest_json=tm_json)

    # ① 相等 → 通过（生产形态 basename：_h 段 == tm.sha256[:10]）
    good_bn = f"vt-{name}-09090909.jsonl"
    err = validate_producer_receipt(_rcp(good_bn), "/tmp/x/" + good_bn)
    _check(err is None, f"闭包一致的回执应通过：{err}")

    # ② tm json 换成另一合法 manifest（键内自洽）但 hash ≠ basename _h 段
    #    → 拒收（修复前 schema 只查键内自洽，此混装不可检）
    other_json = INFO.get_treatment_manifest_json(
        _base_ns(tli_enable_layer_skip=True, tli_alpha=0.5))
    _check(other_json != manifest_json, "对照 manifest 应不同")
    err = validate_producer_receipt(_rcp(good_bn, tm_json=other_json),
                                    "/tmp/x/" + good_bn)
    _check(err is not None and "081" in err and "_h" in err,
           f"tm↔basename 异源应被 081 闭包拒收：{err!r}")

    # ③ basename 无 _h 段但 manifest 存在 → fail closed（防绕过）
    err = validate_producer_receipt(_rcp("vt-plain-09090909.jsonl"),
                                    "/tmp/x/vt-plain-09090909.jsonl")
    _check(err is not None and "081" in err,
           f"无 _h 段 + manifest 存在应 fail closed：{err!r}")

    # ④ 缺 prediction_basename 键 + manifest 存在 → fail closed
    r = _rcp(good_bn)
    del r["prediction_basename"]
    err = validate_producer_receipt(r, "/tmp/x/" + good_bn)
    _check(err is not None and "081" in err,
           f"manifest 存在但缺 basename 键应 fail closed：{err!r}")

    # ⑤ legacy（无 treatment_manifest 键）→ 前向兼容放行（含无 _h 段
    #    的旧 basename——既有收口数据零破坏）
    legacy = build_yarn_receipt(
        yarn_enabled=False, effective_factor=None, yarn_factor_cli=None,
        rope_scaling=None, context_length=32768, task="vt",
        model_path="/tmp/fake-model-081", model_config_sha256=None,
        native_mpe=32768,
        generation_params={"max_gen": 64, "max_num": 0, "seed": 42,
                           "method": "none", "pred_postfix": "_x",
                           "t": "09090909"},
        producer_script_path="benchmark/RULER/pred_ruler.py",
        producer_script_sha256=_sha(b"081-test"),
        run_id="r5-legacy", prediction_basename="vt-plain-09090909.jsonl",
        prediction_sha256="a" * 64, prediction_lines=1)
    err = validate_producer_receipt(legacy, "/tmp/x/vt-plain-09090909.jsonl")
    _check(err is None, f"legacy 回执应前向兼容放行：{err}")

    # ⑥ 有效值语义 + 名字里 D 段与 manifest 联动：enable 翻转 → _h 变
    name_off = INFO.get_method_name_with_info(
        _base_ns(tli_enable_layer_skip=False))
    _check(name_off != name and name_off.endswith("_h" + h10) is False,
           "enable 翻转应同时改 D 段与 _h 段（双通道身份）")
    return "PASS"


# ================================================================ R6：symlink retarget

def test_R6_symlink_retarget(base):
    d = os.path.join(base, "r6")
    os.makedirs(d, exist_ok=True)
    skip_A = os.path.join(d, "skipA.json")
    skip_B = os.path.join(d, "skipB.json")
    with open(skip_A, "w") as f:
        json.dump({"skip": [1, 3]}, f)
    with open(skip_B, "w") as f:
        json.dump({"skip": [2, 4]}, f)
    basis_A = os.path.join(d, "basisA.pt")
    basis_B = os.path.join(d, "basisB.pt")
    tA = _mk_basis(basis_A, seed=1)
    _mk_basis(basis_B, seed=2)

    # skip 经 symlink：resolve 时 L → A
    L = os.path.join(d, "skip_link.json")
    os.symlink(os.path.realpath(skip_A), L)
    ns = _base_ns(tli_enable_layer_skip=True, tli_layer_skip_path=L)
    snap = resolve_treatment_snapshot(ns)
    frozen_mj = snap.manifest_json
    frozen_sha = snap.manifest["tli_layer_skip_path"]["sha256"]
    _check(frozen_sha == _sha(json.dumps({"skip": [1, 3]}).encode()),
           f"symlink 应解析到目标 A 的内容身份：{frozen_sha!r}")
    # retarget：L → B（新链接 + os.replace 原子换目标）
    L2 = os.path.join(d, "skip_link2.json")
    os.symlink(os.path.realpath(skip_B), L2)
    os.replace(L2, L)
    _check(INFO.get_treatment_manifest_json(ns, snap) == frozen_mj,
           "symlink retarget 后冻结身份漂移（081 回归）")
    m_live = json.loads(INFO.get_treatment_manifest_json(ns))
    _check(m_live["tli_layer_skip_path"]["sha256"]
           == _sha(json.dumps({"skip": [2, 4]}).encode()),
           "retarget 后重解析应读到 B 内容（对照语义）")

    # basis 经 symlink：同口径（身份 + tensor 双冻结）
    LB = os.path.join(d, "basis_link.pt")
    os.symlink(os.path.realpath(basis_A), LB)
    ns2 = _base_ns(tli_proj_basis=LB, tli_subspace="tail")
    snap2 = resolve_treatment_snapshot(ns2)
    frozen_mj2 = snap2.manifest_json
    _check(torch.equal(snap2.basis_tensor, tA),
           "basis symlink 应解析到目标 A 的 tensor")
    LB2 = os.path.join(d, "basis_link2.pt")
    os.symlink(os.path.realpath(basis_B), LB2)
    os.replace(LB2, LB)
    _check(INFO.get_treatment_manifest_json(ns2, snap2) == frozen_mj2,
           "basis symlink retarget 后冻结身份漂移（081 回归）")
    _check(torch.equal(snap2.basis_tensor, tA),
           "basis symlink retarget 后冻结 tensor 漂移（081 回归）")
    m2_live = json.loads(INFO.get_treatment_manifest_json(ns2))
    _check(m2_live["tli_proj_basis"]["sha256"]
           != snap2.manifest["tli_proj_basis"]["sha256"],
           "basis retarget 后重解析身份应已变（对照语义）")
    return "PASS"


# ================================================================ K1：kimi3 0316 追加修复 1
# truncate_output_name_keep_hash 上限 245 → 230（sidecar 链绑 ext4 255）

def test_K1_truncate_limit_binds_sidecar_chain(base):
    """kimi3 2026-10-11 0316 hourly review 实测：out_fn 截到 245 时
    sidecar 实际文件名 = 245 + len(".jsonl")=6 + len(".tli_manifest.
    json")=18 = 269 字节 > ext4 单文件名 255 上限，open(sidecar,"w")
    抛 OSError 36 裸 traceback（破坏 076 fail-closed 统一口径）。
    修复：limit 缺省 230（= 255 − 6 − 18 − 1 余量），上限与 sidecar
    后缀链绑定。红态（基点 245）：__defaults__ 断言与长度断言双红。"""
    import inspect
    d = os.path.join(base, "k1")
    os.makedirs(d, exist_ok=True)
    # ① 缺省 limit = 230（单一事实源：info.py 函数签名）
    sig = inspect.signature(INFO.truncate_output_name_keep_hash)
    _check(sig.parameters["limit"].default == 230,
           f"truncate 缺省 limit 应为 230（sidecar 链绑定），实际 "
           f"{sig.parameters['limit'].default!r}")
    # ② 触发截断的长 prefix：截断名 + ".jsonl" + sidecar 后缀全链
    #    ≤ 255（ext4 单文件名上限），且 sidecar 可实际创建
    ns = _base_ns()
    mn = INFO.get_method_name_with_info(ns)
    prefix = "p" * 200                      # 200 + 1 + len(mn) > 230 必截断
    t = "09090909"
    out_fn = INFO.truncate_output_name_keep_hash(prefix, mn, t)
    mj = INFO.get_treatment_manifest_json(ns)
    sc_path = os.path.join(d, out_fn + ".jsonl"
                           + INFO.TREATMENT_MANIFEST_SIDECAR_SUFFIX)
    _check(len(out_fn) + len(".jsonl")
           + len(INFO.TREATMENT_MANIFEST_SIDECAR_SUFFIX) <= 255,
           f"截断名 sidecar 链超 ext4 255：{len(out_fn) + 6 + 18}")
    _check(len(sc_path.encode("utf-8")) >= len(out_fn) + 6 + 18,
           "sidecar 实名长度口径自检")
    # sidecar 实际可创建（green 态 ≤253 字节名在任何 sane FS 上成立；
    # 红态 245 口径下 269 字节名在 ext4/多数本地 FS 上 OSError 36）
    with open(sc_path, "w", encoding="utf-8") as f:
        f.write(mj)
    _check(os.path.isfile(sc_path), "sidecar 实际创建失败")
    # ③ 生产入口同源：pred.py 截断阈值与缺省 limit 一致（不传 limit 的
    #    调用约定下，>230 即走截断重建——245 阈值会让 231-245 名绕过
    #    截断后在写门处 OSError 36）
    src = os.path.join(REPO, "benchmark", "LongBench", "pred.py")
    with open(src, "r", encoding="utf-8") as f:
        pred_src = f.read()
    _check("len(out_fn) > 230" in pred_src,
           "pred.py 截断阈值应与 info.py 缺省 limit=230 同源")
    return "PASS"


# ================================================================ K2：kimi3 0316 追加修复 2
# B/D 开关入 manifest——截断吃掉可读段 B/D 位后 hash 仍单射

def test_K2_bd_switches_in_manifest_survive_truncation(base):
    """kimi3 反例推翻 076 residual「可读段 B/D 已单射」：极端长可读前缀
    触发截断时 B/D 段被 "..." 占位吃掉 → 仅在 tli_enable_kmeans/
    tli_enable_layer_skip 上不同的两 treatment 截断后文件名相同 +
    manifest 相同 + hash 相同（kimi3 实测 trunc same? True、hash same?
    True）→ 写门放行互覆。修复：两字段入 _TREATMENT_FIELD_DEFAULTS
    （默认 True，与 argparse/TLIIndexer.__init__ 同源）→ manifest/hash
    恒单射，截断只影响可读性不影响身份。红态（基点字段缺）：字段集
    断言 + manifest 互异断言 + hash 互异断言 + 截断互异断言四连红。"""
    # ① 字段集：B/D 开关必须在 manifest 字段集内
    _check("tli_enable_kmeans" in INFO._TREATMENT_FIELD_DEFAULTS
           and INFO._TREATMENT_FIELD_DEFAULTS["tli_enable_kmeans"] is True,
           "tli_enable_kmeans 应在 _TREATMENT_FIELD_DEFAULTS 且默认 True")
    _check("tli_enable_layer_skip" in INFO._TREATMENT_FIELD_DEFAULTS
           and INFO._TREATMENT_FIELD_DEFAULTS["tli_enable_layer_skip"]
           is True,
           "tli_enable_layer_skip 应在 _TREATMENT_FIELD_DEFAULTS 且默认 True")
    # ② kimi3 口径：cfg(True,False) vs cfg(False,True)，极端长可读前缀
    #    （大数值参数段拉长 readable，使 B/D 位落入截断区）
    def cfg(kmeans, layer_skip):
        ns = _base_ns(tia_block_size=10**12, tia_level1_topk=10**12,
                      tia_level2_topk=10**12, tia_level2_cmp_ratio=10**9,
                      tli_gamma=0.12345678901234,
                      tli_enable_kmeans=kmeans,
                      tli_enable_layer_skip=layer_skip)
        return ns

    a, b = cfg(True, False), cfg(False, True)
    m_a = INFO.get_treatment_manifest_json(a)
    m_b = INFO.get_treatment_manifest_json(b)
    _check(m_a != m_b, "仅 B/D 开关不同的配置 manifest 应互异（kimi3 "
           "trunc same? True 反例回归：hash 侧身份缺口）")
    h_a, h_b = INFO._treatment_hash(a), INFO._treatment_hash(b)
    _check(h_a != h_b, "仅 B/D 开关不同的配置 hash 应互异（kimi3 "
           "hash same? True 反例回归）")
    n_a = INFO.get_method_name_with_info(a)
    n_b = INFO.get_method_name_with_info(b)
    _check(n_a != n_b, "全名应互异（可读段 B vs D 双通道之一）")
    # ③ 截断互异：极端长 prefix 下 B/D 可读位被 "..." 吃掉，
    #    身份仍由 hash 尾段承载（追加修复 2 的存在理由）
    prefix = "q" * 220
    t_a = INFO.truncate_output_name_keep_hash(prefix, n_a, "09090909")
    t_b = INFO.truncate_output_name_keep_hash(prefix, n_b, "09090909")
    print(f"  K2 trunc_a -> {t_a!r}")
    print(f"  K2 trunc_b -> {t_b!r}")
    _check(t_a != t_b, "截断后仅 B/D 不同的配置仍同名（kimi3 trunc "
           "same? True 反例回归：写门放行互覆可达）")
    _check(t_a.endswith("_h" + h_a + "-09090909")
           and t_b.endswith("_h" + h_b + "-09090909"),
           "截断名应保各自 hash 尾段（身份承载）")
    # 可读 B/D 位确实被吃掉（反例成立前提自证——否则 ③ 的互异可能
    # 只来自可读位而非 hash 修复）
    ra, _ = INFO.split_method_name_hash(n_a)
    room = 230 - (len(prefix) + 1 + len("...") + 12 + 1 + len("09090909"))
    _check("B" not in ra[:max(room, 0)],
           "自检失败：可读 B 位未被截断区吃掉（K2 反例构造不成立，"
           "须加长 prefix/参数段）")
    return "PASS"


# ================================================================ main

def main():
    """067 纪律：PASS/SKIP/FAIL 三分显式计数；异常 → FAIL 继续跑完其余
    用例；任一 SKIP/FAIL → 非全绿且非零退出（SKIP 是覆盖缺口不是荣誉，
    080 口径）。TLI081_ONLY 子集过滤。"""
    base = tempfile.mkdtemp(prefix="e121_fix_081_")
    plan = [
        ("R1", lambda: test_R1_freeze_survives_midrun_replacement(base)),
        ("R2", lambda: test_R2_crosslayer_shared_snapshot(base)),
        ("R3", lambda: test_R3_default_mask_identity(base)),
        ("R4", lambda: test_R4_reverse_asymmetry(base)),
        ("R5", test_R5_receipt_closure),
        ("R6", lambda: test_R6_symlink_retarget(base)),
        ("K1", lambda: test_K1_truncate_limit_binds_sidecar_chain(base)),
        ("K2", lambda: test_K2_bd_switches_in_manifest_survive_truncation(base)),
    ]
    only = os.environ.get("TLI081_ONLY", "")
    if only:
        keep = {x.strip() for x in only.split(",") if x.strip()}
        plan = [p for p in plan if p[0] in keep]
    n_pass = n_skip = n_fail = 0
    failed = []
    try:
        for name, fn in plan:
            try:
                st = fn()
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
            if st == "SKIP":
                n_skip += 1
                print(f"[{name}] SKIP", flush=True)
            else:
                n_pass += 1
                print(f"[{name}] PASS", flush=True)
    finally:
        shutil.rmtree(base, ignore_errors=True)
    print(f"\nE121-FIX-081 RESULT: PASS={n_pass} SKIP={n_skip} "
          f"FAIL={n_fail} (total {len(plan)})")
    if failed:
        print(f"FAILED: {failed}")
    return 0 if (n_fail == 0 and n_skip == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
