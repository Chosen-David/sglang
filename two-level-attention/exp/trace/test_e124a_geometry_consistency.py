#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E124a（S-T021 / A1）几何一致性红绿测试（GPT A1 验收反馈 2026-10-11）。

被测对象：
  1. exp/trace/run_e124a_dryrun.py 实跑入口的几何一致性门禁
     （n_valid 解析/双向核对/防钳制断言/qpos 因果核对/geometry 记录）；
  2. sparse_attn/indexer/dyn_controller.py decide() 的同源几何绑定
     （k_mid 行数 == max(0, n_valid−n_protected)，不一致 fail-closed）。

缺陷背景（v1 sample 已入库的 9 条错配记录 = 拒绝负例）：
  --n-valid 旧默认硬编码 32768 使 meta.S(=16957) 回退永不触发；
  mid 切片 k_all[128:32640] 被 Python 静默钳制成 [128:16957]（SWA 尾段
  128 token 未切掉混进特征 middle）；同一决策记录特征几何 16829 与预算
  几何 32512（near_range=[16384,32640] 越出实际序列长 16957）两套口径。

用例矩阵：
  G01 负例重放    真实 trace + 显式 --n-valid 32768 重放入口 →
                  [E124A-ABORT][GEOM-MISMATCH] 非零退出（v1 错配生成路径
                  复现即被拒）。
  G02 正例回退    同 trace 不给 --n-valid → 回退 meta.S=16957，9 层全成，
                  features.n_mid == 16701 == n_valid−n_protected == budget
                  几何（near/far 区间落 [128,16829]），qpos+1 == n_valid。
  G03 正例显式    显式 --n-valid 16957（与 trace 一致）→ 成功且
                  n_valid_source == "explicit"。
  G04 decide 负例 k_mid 行数 != n_valid−n_protected 的直接调用 →
                  ValueError 含 E124A-GEOM-MISMATCH（两个规模：32768 口径
                  的 16829 行 + 小几何 2048 vs 28）。
  G05 decide 绑定 k_mid 行数 == n_valid−n_protected → 放行且 n_mid 如实。
  G06 meta/张量   合成 fixture：meta.S=16957 但 blob k 行数 16000 →
     双向核对      ABORT（meta 声明与实际张量不一致）。
  G07 qpos 因果   合成 fixture：qpos[-1]+1 != S → ABORT（query 因果位置
                  与 K 长度关系不一致）。
  G08 v1 负例闭包 已入库 9 条错配记录逐条核验：全部携带错配签名
                  （n_mid=16829 != 32512；near_range 越界）——按新绑定
                  校验口径全部会被拒绝（历史保留不动，只验不改）。
  G09 v2 正例闭包 已入库 v2 sample 9 条逐条同源核验 + v1/v2 档位分布
                  对比如实记录（几何修正后 s 重算，档位变化不要求不变，
                  断言只锁一致性不锁档位）。
  G10 保护集分量  n_prefix+n_swa != n_protected → ABORT（两分量不闭合，
                  切片与预算将两套口径）。
  G11 dry-run     不给 --n-valid 的 dry-run 预览 → run_config.n_valid=None
     预览诚实      + 各档 reason=n_valid_unspecified（不再静默默认 32768）。

trace 依赖：G01-G03 优先用真实 /tmp/trace/qwen3-8b-v/lb_hotpotqa_0；
若已清理则用合成 fixture（S=16957 + meta.json 同构，D/H 缩减版）并
如实注明「真实 trace 已清理，正例为合成同几何」。

运行：
  python3 exp/trace/test_e124a_geometry_consistency.py        # 全量
  E124G_ONLY=G01,G08 python3 exp/trace/test_e124a_geometry_consistency.py
  python3 -O exp/trace/test_e124a_geometry_consistency.py     # -O 双跑
全部 _check 显式判定（045 纪律：不依赖 assert，python -O 不删除）。
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
ENTRY = os.path.join(REPO, "exp", "trace", "run_e124a_dryrun.py")
RESULTS = os.path.join(REPO, "exp", "trace", "results")
V1_JSONL = os.path.join(RESULTS, "e124a_dryrun_sample",
                        "per_seq_layer_decisions_lb_hotpotqa_0.jsonl")
V2_JSONL = os.path.join(RESULTS, "e124a_dryrun_v2_sample",
                        "per_seq_layer_decisions_lb_hotpotqa_0.jsonl")
V2_SUMMARY = os.path.join(RESULTS, "e124a_dryrun_v2_summary.json")

REAL_TRACE = "/tmp/trace/qwen3-8b-v/lb_hotpotqa_0"
S_REAL = 16957          # trace meta.json 实际 S（v1 被静默覆盖成 32768）
N_MID_OK = S_REAL - 256  # 修正后特征/预算同源 middle = 16701
N_PREFIX, N_SWA, N_PROT = 128, 128, 256
K1, K2, BS = 128, 1024, 64


def _check(cond, msg=""):
    """045 纪律：显式判定，非 assert（python -O 不删除）。"""
    if not cond:
        raise SystemExit(f"[TEST-FAIL] {msg}")


# ---------------- trace 供给：真实优先，清理后合成同几何 ----------------

_TRACE_NOTE = {"synthetic": None}


def _build_synthetic_trace(base):
    """合成 fixture（真实 trace 被清理时的正例底座）：
    S=16957 + meta.json 同构（9 层），D/H 缩减（D=64, H=8, Hkv=2——
    几何断言与维度无关；GQA 8%2==0 特征路径合法）。固定 seed 可复现。"""
    tdir = os.path.join(base, "trace_lb_hotpotqa_0_syn")
    os.makedirs(tdir, exist_ok=True)
    meta = {"S": S_REAL, "n_layers": 36, "prompt": "lb_hotpotqa_0_synthetic",
            "layers": [1, 4, 8, 12, 16, 20, 24, 28, 35],
            "with_v": True, "question": "synthetic same-geometry fixture",
            "answer": "", "synthetic_reduced_dims": {"d_model": 64,
                                                     "n_q_heads": 8,
                                                     "n_kv_heads": 2}}
    with open(os.path.join(tdir, "meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False)
    g = torch.Generator().manual_seed(20261011)
    tq = 16
    qpos = torch.arange(S_REAL - tq, S_REAL, dtype=torch.long)
    for layer in meta["layers"]:
        k = torch.randn(S_REAL, 2, 64, generator=g)
        q = torch.randn(tq, 8, 64, generator=g)
        v = torch.randn(S_REAL, 2, 64, generator=g)
        torch.save({"k": k, "v": v, "q": q, "qpos": qpos, "S": S_REAL},
                   os.path.join(tdir, f"layer{int(layer):02d}.pt"))
    return tdir


def _trace_dir(base):
    """返回 (trace_dir, is_synthetic)。真实 trace 存在则优先；否则合成
    同几何 fixture 并如实注明。"""
    if os.path.exists(os.path.join(REAL_TRACE, "meta.json")):
        return REAL_TRACE, False
    tdir = _build_synthetic_trace(base)
    _TRACE_NOTE["synthetic"] = ("真实 trace 已清理，正例为合成同几何"
                                f"（S={S_REAL}, meta.json 同构 9 层，D/H 缩减版）")
    print(f"[fixture] {_TRACE_NOTE['synthetic']}", flush=True)
    return tdir, True


# ---------------- 入口子进程 helpers ----------------

def _run_entry(base, trace_dir, extra_args, out_name="dec.jsonl"):
    """跑修复后的入口（与当前解释器同 -O 口径）。返回 (rc, 合并输出)。"""
    out = os.path.join(base, out_name)
    argv = [sys.executable]
    if not __debug__:
        argv.append("-O")
    argv += [ENTRY, "--trace-dir", trace_dir, "--out", out] + list(extra_args)
    p = subprocess.run(argv, capture_output=True, text=True, timeout=1200)
    return p.returncode, p.stdout + p.stderr


def _load_jsonl(path):
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


# ---------------- G01 负例重放：显式 --n-valid 32768 → ABORT ----------------

def test_G01_neg_replay_explicit_nvalid():
    base = tempfile.mkdtemp(prefix="e124g01_")
    try:
        tdir, syn = _trace_dir(base)
        rc, out = _run_entry(base, tdir, ["--n-valid", "32768"])
        _check(rc != 0, f"显式 --n-valid 32768（≠ trace S=16957）必须非零退出，"
                       f"得 rc={rc}")
        _check("E124A-ABORT" in out and "GEOM-MISMATCH" in out,
               f"输出须含 [E124A-ABORT][GEOM-MISMATCH] 标记，得:\n{out}")
        _check("32768" in out and "16957" in out,
               "错误信息须给出两个几何值的说明（32768 vs 16957）")
        _check("静默覆盖" in out or "不一致" in out,
               "错误信息须含几何不一致说明")
        # 不得产出任何决策（fail-closed 在落盘前拦截）
        _check(not os.path.exists(os.path.join(base, "dec.jsonl"))
               or len(_load_jsonl(os.path.join(base, "dec.jsonl"))) == 0,
               "负例不得落任何决策记录")
    finally:
        shutil.rmtree(base, ignore_errors=True)


# ---------------- G02 正例：缺省回退 meta.S，特征/预算同源 ----------------

def _assert_record_consistent(d, ctx):
    """单条决策的几何同源断言（正例核心）。"""
    geo = d.get("geometry")
    _check(geo is not None, f"{ctx}: 决策记录缺 geometry 段")
    n_valid = geo["n_valid"]
    _check(n_valid == S_REAL, f"{ctx}: n_valid 应回退/等于 meta.S={S_REAL}，"
                              f"得 {n_valid}")
    _check(geo["k_rows"] == S_REAL and geo["S"] == S_REAL,
           f"{ctx}: meta.S 与实际 k 行数应双向一致")
    _check(geo["qpos"] + 1 == n_valid,
           f"{ctx}: 当前 query 因果可见 keys({geo['qpos']+1}) 应 == n_valid")
    # 特征几何 == 预算几何（同一合法前缀）
    n_mid = d["features"]["n_mid"]
    _check(n_mid == N_MID_OK,
           f"{ctx}: features.n_mid 应为 {N_MID_OK}（S-n_prefix-n_swa，"
           f"切掉 SWA 尾段），得 {n_mid}")
    _check(n_mid == n_valid - N_PROT,
           f"{ctx}: 特征 n_mid({n_mid}) 必须 == n_valid-n_protected"
           f"({n_valid - N_PROT})——同源绑定")
    _check(d["features"]["available_at_mid_tokens"] == n_mid,
           f"{ctx}: available_at 应与 n_mid 一致")
    b = d["budget"]
    _check(b is not None, f"{ctx}: 正例预算必须编译成功")
    _check(b["Kmid"] == max(0, min(K2, n_valid) - N_PROT),
           f"{ctx}: Kmid 应=max(0,min(K2,n_valid)-n_protected)")
    near, far = b["near_range"], b["far_range"]
    _check(far[0] == N_PREFIX and near[1] == n_valid - N_SWA,
           f"{ctx}: 候选区间应落 [n_prefix, n_valid-n_swa)="
           f"[{N_PREFIX},{n_valid - N_SWA}]，得 far={far} near={near}")
    _check(N_PREFIX <= far[0] < far[1] <= near[0] < near[1] <= n_valid - N_SWA,
           f"{ctx}: near/far 区间必须不越界且不相交: far={far} near={near}")
    _check(near[1] <= S_REAL,
           f"{ctx}: near 上界({near[1]}) 不得越出实际序列长 {S_REAL}"
           f"（v1 错配记录 near_range=[16384,32640] 即此越界）")


def test_G02_pos_fallback_meta_s():
    base = tempfile.mkdtemp(prefix="e124g02_")
    try:
        tdir, syn = _trace_dir(base)
        rc, out = _run_entry(base, tdir, [])
        _check(rc == 0, f"缺省 --n-valid（回退 meta.S）应成功，rc={rc}:\n{out}")
        recs = _load_jsonl(os.path.join(base, "dec.jsonl"))
        _check(len(recs) == 9, f"应落 9 层决策，得 {len(recs)}")
        for i, d in enumerate(recs):
            _check(d["geometry"]["n_valid_source"] == "meta.S",
                   f"rec{i}: n_valid 来源应记 meta.S（回退分支真正生效）")
            _assert_record_consistent(d, f"rec{i}")
            _check(d["budget"]["Tn"] + d["budget"]["Tf"] == d["budget"]["Kmid"],
                   f"rec{i}: Tn+Tf 应闭合 == Kmid")
        if syn:
            print(f"[G02] NOTE: {_TRACE_NOTE['synthetic']}", flush=True)
    finally:
        shutil.rmtree(base, ignore_errors=True)


# ---------------- G03 正例：显式一致的 --n-valid 放行 ----------------

def test_G03_pos_explicit_consistent():
    base = tempfile.mkdtemp(prefix="e124g03_")
    try:
        tdir, _ = _trace_dir(base)
        rc, out = _run_entry(base, tdir, ["--n-valid", str(S_REAL)])
        _check(rc == 0, f"显式 --n-valid 16957（与 trace 一致）应放行，"
                       f"rc={rc}:\n{out}")
        recs = _load_jsonl(os.path.join(base, "dec.jsonl"))
        _check(len(recs) == 9, f"应落 9 层决策，得 {len(recs)}")
        for i, d in enumerate(recs):
            _check(d["geometry"]["n_valid_source"] == "explicit",
                   f"rec{i}: 显式来源应记 explicit")
            _assert_record_consistent(d, f"rec{i}")
        # 显式一致与缺省回退两路径的决策应逐层同 s（同几何同内容）
        rc2, out2 = _run_entry(base, tdir, [], out_name="dec_fb.jsonl")
        _check(rc2 == 0, f"回退路径应成功: {out2}")
        fb = _load_jsonl(os.path.join(base, "dec_fb.jsonl"))
        for a, b in zip(recs, fb):
            _check(a["layer_idx"] == b["layer_idx"]
                   and abs((a["features"]["s"] or 0) - (b["features"]["s"] or 0)) < 1e-12
                   and a["profile"]["applied"] == b["profile"]["applied"],
                   f"layer {a['layer_idx']}: 显式一致与回退两路径决策应相同")
    finally:
        shutil.rmtree(base, ignore_errors=True)


# ---------------- G04/G05 decide() 同源绑定单元负/正例 ----------------

def test_G04_decide_neg_mismatch():
    from sparse_attn.indexer import dyn_controller as dc
    sys.path.insert(0, REPO)
    # 负例 1：v1 错配几何的直接复现——k_mid 16829 行 vs n_valid=32768
    k_mid = torch.zeros(16829, 2, 64)
    q = torch.zeros(8, 64)
    try:
        dc.decide(q, k_mid, 0, 1, "mavg", k1=K1, k2=K2, bs=BS,
                  n_valid=32768, n_protected=N_PROT,
                  n_prefix=N_PREFIX, n_swa=N_SWA)
        _check(False, "16829 行 k_mid + n_valid=32768 必须 fail-closed，却放行了")
    except ValueError as e:
        _check("E124A-GEOM-MISMATCH" in str(e),
               f"错误信息须含 E124A-GEOM-MISMATCH，得: {e}")
        _check("16829" in str(e) and "32512" in str(e),
               f"错误信息须给出两侧几何值（16829 vs 32512）: {e}")
    # 负例 2：小几何（既有套件 T10 旧入参口径 2048 vs 28）
    try:
        dc.decide(q, torch.zeros(2048, 2, 64), 0, 1, "mavg", k1=32, k2=14,
                  bs=1, n_valid=28, n_protected=0, n_prefix=0, n_swa=0)
        _check(False, "k_mid 2048 行 + n_valid=28 必须 fail-closed，却放行了")
    except ValueError as e:
        _check("E124A-GEOM-MISMATCH" in str(e), f"小几何负例须同标记: {e}")
    # 负例 3：k_mid=None + n_valid>n_protected（空张量谎报几何）
    try:
        dc.decide(q, None, 0, 1, "mavg", k1=K1, k2=K2, bs=BS,
                  n_valid=32768, n_protected=N_PROT,
                  n_prefix=N_PREFIX, n_swa=N_SWA)
        _check(False, "k_mid=None + n_valid=32768 必须 fail-closed")
    except ValueError:
        pass


def test_G05_decide_pos_binding():
    from sparse_attn.indexer import dyn_controller as dc
    # 绑定正例：k_mid 16701 行 == n_valid(16957) - n_protected(256)
    g = torch.Generator().manual_seed(7)
    k_mid = torch.randn(16701, 2, 64, generator=g)
    q = torch.randn(8, 64, generator=g)
    dec = dc.decide(q, k_mid, 0, 1, "mavg", k1=K1, k2=K2, bs=BS,
                    n_valid=S_REAL, n_protected=N_PROT,
                    n_prefix=N_PREFIX, n_swa=N_SWA)
    _check(dec["features"]["n_mid"] == 16701,
           f"绑定正例 n_mid 应 16701，得 {dec['features']['n_mid']}")
    _check(dec["budget"]["Kmid"] == 768, "Kmid 应 min(1024,16957)-256=768")
    # 空 middle 合法路径不受绑定破坏：k_mid 0 行 + n_valid==n_protected
    dec0 = dc.decide(q, torch.zeros(0, 2, 64), 0, 1, "mavg", k1=K1, k2=K2,
                     bs=BS, n_valid=N_PROT, n_protected=N_PROT,
                     n_prefix=64, n_swa=64)
    _check(dec0["features"]["status"] == "empty_middle",
           f"空 middle 合法路径应保留，得 {dec0['features']['status']}")


# ---------------- G06/G07 合成 fixture 篡改负例 ----------------

def _tamper_fixture(base, kind):
    """在合成 trace 上注入几何不一致：kind ∈ {"k_rows", "qpos"}。"""
    tdir = _build_synthetic_trace(base)
    if kind == "k_rows":
        # meta 声明 S=16957，实际 k 张量只有 16000 行（meta 与张量不一致）
        p = os.path.join(tdir, "layer01.pt")
        blob = torch.load(p, map_location="cpu")
        blob["k"] = blob["k"][:16000]
        torch.save(blob, p)
    elif kind == "qpos":
        # 末位 query 因果位置 16954 → 可见 keys 16955 != S=16957
        p = os.path.join(tdir, "layer01.pt")
        blob = torch.load(p, map_location="cpu")
        blob["qpos"] = blob["qpos"] - 2
        torch.save(blob, p)
    return tdir


def test_G06_meta_vs_blob_rows():
    from sparse_attn.indexer import dyn_controller as dc  # noqa: F401 环境预热
    base = tempfile.mkdtemp(prefix="e124g06_")
    try:
        tdir = _tamper_fixture(base, "k_rows")
        rc, out = _run_entry(base, tdir, [])
        _check(rc != 0, f"meta.S 与 blob k 行数不一致必须非零退出，得 rc={rc}")
        _check("E124A-ABORT" in out and "16000" in out and "16957" in out,
               f"错误信息须含双向核对两侧值（16000 vs 16957）:\n{out}")
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_G07_qpos_causal_mismatch():
    base = tempfile.mkdtemp(prefix="e124g07_")
    try:
        tdir = _tamper_fixture(base, "qpos")
        rc, out = _run_entry(base, tdir, [])
        _check(rc != 0, f"qpos 因果位置与 K 长度不一致必须非零退出，得 rc={rc}")
        _check("E124A-ABORT" in out and "qpos" in out,
               f"错误信息须含 qpos 因果不一致说明:\n{out}")
        _check("16955" in out and "16957" in out,
               "错误信息须给出可见 keys(16955) vs n_valid(16957) 两侧值")
    finally:
        shutil.rmtree(base, ignore_errors=True)


# ---------------- G08 已入库 v1 错配记录 = 拒绝负例闭包 ----------------

def test_G08_v1_records_negative_closure():
    """9 条 v1 错配记录逐条核验其错配签名：按新绑定口径（k_mid 行数 ==
    n_valid−n_protected）全部应被拒绝。历史记录保留不动，只验不改。"""
    _check(os.path.exists(V1_JSONL), f"v1 错配记录应已入库: {V1_JSONL}")
    recs = _load_jsonl(V1_JSONL)
    _check(len(recs) == 9, f"v1 应恰 9 条错配决策，得 {len(recs)}")
    n_rejected = 0
    for i, d in enumerate(recs):
        nv = d["protection"]["n_valid"]
        np_ = d["protection"]["n_protected"]
        n_mid = d["features"]["n_mid"]
        expected = max(0, nv - np_)
        # 绑定校验复演：不一致 → 拒绝
        if n_mid != expected:
            n_rejected += 1
        _check(n_mid != expected,
               f"rec{i}: v1 记录应携带错配签名（n_mid={n_mid} != "
               f"n_valid-n_protected={expected}），否则不是负例素材")
        _check(nv == 32768 and n_mid == 16829,
               f"rec{i}: v1 错配几何应为 n_valid=32768/n_mid=16829")
        # 越界证据：预算 near 区间越出 trace 实际序列长 16957
        near = d["budget"]["near_range"]
        _check(near[1] == 32640 and near[1] > S_REAL,
               f"rec{i}: v1 near_range 上界应越出实际序列长（越界证据），"
               f"得 {near}")
        # SWA 尾段泄漏证据：v1 特征 middle 含 128 个 SWA 尾 token
        _check(n_mid == S_REAL - N_PREFIX,
               f"rec{i}: v1 n_mid 应恰为被钳制切片 [128:16957] 的行数 "
               f"{S_REAL - N_PREFIX}（SWA 尾段未切除的泄漏证据）")
    _check(n_rejected == 9, f"9 条 v1 记录应全部被新绑定口径拒绝，"
                            f"得 {n_rejected}/9")


# ---------------- G09 已入库 v2 正例记录 = 同源闭包 + 档位对比 ----------------

def test_G09_v2_records_consistent_closure():
    _check(os.path.exists(V2_JSONL), f"v2 正例记录应已入库: {V2_JSONL}")
    v2 = _load_jsonl(V2_JSONL)
    _check(len(v2) == 9, f"v2 应恰 9 层决策，得 {len(v2)}")
    for i, d in enumerate(v2):
        _assert_record_consistent(d, f"v2[{i}]")
    # v1/v2 档位分布对比：如实记录，不要求档位不变（几何修正后 s 重算）
    v1 = _load_jsonl(V1_JSONL)

    def _dist(recs):
        dd = {}
        for r in recs:
            t = r["profile"]["applied"]
            dd[t] = dd.get(t, 0) + 1
        return dd

    d1, d2 = _dist(v1), _dist(v2)
    changed = [(a["layer_idx"], a["profile"]["applied"], b["profile"]["applied"])
               for a, b in zip(v1, v2)
               if a["profile"]["applied"] != b["profile"]["applied"]]
    print(f"[G09] v1(错配回放)档位分布: {json.dumps(d1, ensure_ascii=False)} -> "
          f"v2(同源修正)档位分布: {json.dumps(d2, ensure_ascii=False)}；"
          f"档位变化 {len(changed)}/9 层: {changed}", flush=True)
    _check(os.path.exists(V2_SUMMARY), f"v2 summary 应已入库: {V2_SUMMARY}")
    summ = json.load(open(V2_SUMMARY))
    _check(summ["geometry"]["n_valid"] == S_REAL
           and summ["geometry"]["n_mid"] == N_MID_OK,
           "v2 summary geometry 段应为修正后同源几何")
    _check(len(summ.get("per_layer_v1_vs_v2", [])) == 9,
           "v2 summary 应记录 9 层 v1/v2 对比（档位变化如实入档）")
    _check(summ["tiers_v2"] == d2, "v2 summary 档位分布应与 jsonl 实测一致")
    # 对比记录本身的一致性：每层 tier_changed 与实测吻合
    for row, (a, b) in zip(summ["per_layer_v1_vs_v2"], zip(v1, v2)):
        _check(row["v1_applied"] == a["profile"]["applied"]
               and row["v2_applied"] == b["profile"]["applied"]
               and row["tier_changed"] == (a["profile"]["applied"]
                                           != b["profile"]["applied"]),
               f"layer {row['layer_idx']} 对比记录与 jsonl 实测不符")


# ---------------- G10 保护集两分量不闭合 → ABORT ----------------

def test_G10_prefix_swa_not_closing():
    base = tempfile.mkdtemp(prefix="e124g10_")
    try:
        tdir, _ = _trace_dir(base)
        rc, out = _run_entry(base, tdir, ["--n-prefix", "100", "--n-swa", "100"])
        _check(rc != 0, f"n_prefix+n_swa(200) != n_protected(256) 必须非零退出，"
                        f"得 rc={rc}")
        _check("E124A-ABORT" in out and "不闭合" in out,
               f"错误信息须说明保护集两分量不闭合:\n{out}")
    finally:
        shutil.rmtree(base, ignore_errors=True)


# ---------------- G11 dry-run 预览：缺省不再静默 32768 ----------------

def test_G11_dryrun_preview_honest():
    argv = [sys.executable]
    if not __debug__:
        argv.append("-O")
    argv += [ENTRY, "--dry-run"]
    p = subprocess.run(argv, capture_output=True, text=True, timeout=300)
    _check(p.returncode == 0, f"dry-run 应成功: {p.stdout + p.stderr}")
    cfg = json.loads(p.stdout)
    _check(cfg["run_config"]["n_valid"] is None,
           f"缺省 --n-valid 时 run_config.n_valid 应为 None（旧版静默 32768 "
           f"已废），得 {cfg['run_config']['n_valid']}")
    for tier in ("P_F", "P_C", "P_N", "fixed_tier"):
        _check(cfg["compiled_preview"][tier].get("reason")
               == "n_valid_unspecified",
               f"{tier} 预览应如实记 n_valid_unspecified，得 "
               f"{cfg['compiled_preview'][tier]}")
    # 显式给值时预览正常编译
    p2 = subprocess.run(argv + ["--n-valid", "32768"],
                        capture_output=True, text=True, timeout=300)
    cfg2 = json.loads(p2.stdout)
    _check(cfg2["run_config"]["n_valid"] == 32768
           and cfg2["compiled_preview"]["P_C"].get("Kmid") == 768,
           "显式 --n-valid 的 dry-run 预览应正常编译")


# ================================================================ main

PLAN = [
    ("G01", test_G01_neg_replay_explicit_nvalid),
    ("G02", test_G02_pos_fallback_meta_s),
    ("G03", test_G03_pos_explicit_consistent),
    ("G04", test_G04_decide_neg_mismatch),
    ("G05", test_G05_decide_pos_binding),
    ("G06", test_G06_meta_vs_blob_rows),
    ("G07", test_G07_qpos_causal_mismatch),
    ("G08", test_G08_v1_records_negative_closure),
    ("G09", test_G09_v2_records_consistent_closure),
    ("G10", test_G10_prefix_swa_not_closing),
    ("G11", test_G11_dryrun_preview_honest),
]


def main():
    """红绿 runner：显式 _check（python -O 兼容）；E124G_ONLY 过滤；
    一次跑完全部用例给红绿全景；FAIL>0 非零退出。"""
    only = os.environ.get("E124G_ONLY", "")
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
    print(f"\nE124A-GEOMETRY-CONSISTENCY RESULT: PASS={n_pass} FAIL={n_fail} "
          f"(total {len(plan)})")
    if failed:
        print(f"FAILED: {failed}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
