#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E124a 结构几何门禁（088）+ seq 身份/seed 契约（089）红绿测试。

背景（GPT 2026-10-11 1330 增量审计，三项 confirmed，主 AI 逐位复现）：
  - TL-E124-STRUCT-GUARD-088：编译器/CLI 未拒绝负 K2、n_protected>n_valid、
    负 prefix/window 等不可能几何，反而 max(0,...) 静默压零返回 ok=true
    （GPT 复现：compile_tier("P_C", k2=-1,...)→True；n_protected=256>
    n_valid=128→True；n_prefix=-1,n_swa=257→True 且 far_range=[-1,24383]）。
    修复 = dyn_controller.validate_geometry 集中实现，compile_tier/
    compile_fixed/CLI 入口共用；非法一律 reason=struct、入口非零退出；
    K2=0 是协议允许值单列 k2_zero 如实记录，不与负值混同。
  - TL-E124-SEQ-IDENTITY-089：seed 契约 sha256('e124a|{seq_id}|{layer_idx}')
    要求每 seq×layer 唯一，dry-run 入口却默认 --seq-id 0，8 个 trace
    目录全部 seq_id=0 → 同层跨序列共享随机投影 seed、decision 身份跨输入
    碰撞，且记录不含 trace 来源。修复 = --seq-id auto（缺省，按 trace
    目录派生稳定唯一 ID）+ decision 记 trace 来源（geometry.trace_source）
    + DecisionLog 追加重复 identity fail-closed 门禁。

用例矩阵（每项均含正例/反例双证）：
  S01 负 K2 反例   四档（P_F/P_C/P_N/fixed）全 reason=struct；正例对照：
                   同几何合法 K2 编译成功且基准数值（Kmid=768/Tn=Tf=384）
                   与修复前逐位一致（数值语义合法几何下不变）。
  S02 保护超 U     n_protected=256 > n_valid=128 → 四档 struct（GPT 复现例）；
                   正例 n_protected<=n_valid 放行。
  S03 负 prefix    n_prefix=-1（闭合 -1+257=256，GPT 复现例）→ struct，
                   不再落 far_range=[-1,...]；负 n_swa 同理；正例放行。
  S04 负 k1/bs     k1<=0、bs<=0 → struct；正例放行。
  S05 k2=0 单列    协议允许值：ok=True、Kmid=0、k2_zero=True 单列记录；
                   与负 K2 的 struct 拒绝不混同（同套断言内对照）。
  S06 分量不闭合   n_prefix+n_swa!=n_protected → struct；只给其一 = 分解
                   不完整 → struct；正例闭合放行。
  S07 CLI dry-run  struct 输入非零退出（--k2 -1 / --n-protected>--n-valid）；
                   合法输入与 k2=0 均 rc=0（诚实预览不误伤）。
  S08 CLI 实跑     struct 输入非零退出（进入 trace 读取前的前置校验，
                   不依赖 trace 存在）；合法输入实跑成功（合成 trace）。
  I01 auto 唯一性  两个不同 trace 目录 auto 模式 → 两个不同 seq_id、
                   同层 seed 不同（sha256('e124a|{id}|{layer}') 实证区分）、
                   投影 R 不同；decision 记录含 trace_source（目录名+meta sha）。
  I02 同目录重跑   同 --out 追加 → 重复 identity（IDENTITY-DUP）非零退出，
                   不产生重复行（聚合消费端双计防护）。
  I03 显式冒用     两个不同目录显式 --seq-id 0 → 同 out 追加被
                   IDENTITY-CLASH 拒（这正是 089 缺陷场景：同层 seed
                   跨输入碰撞，门禁拦截）；且两个 seed 实证相同（碰撞证明）。
  I04 显式保留     单目录显式 --seq-id N 合法（fresh --out）→ rc 0、
                   geometry.seq_id_source=="explicit"。
  I05 DecisionLog  库级 identity 门禁单元：重复 identity ValueError；
                   既有文件损坏 → 拒绝在损坏文件上继续混写。

运行：
  python3 exp/trace/test_e124a_struct_and_identity.py          # 全量
  E124SI_ONLY=S01,I01 python3 exp/trace/...                   # 子集
  python3 -O exp/trace/test_e124a_struct_and_identity.py       # -O 双跑
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

# 合成 trace 几何（小规模即可满足特征合法：m_len=768 → 16 anchor，N/F ≥ 2）
SYN_S = 1024
SYN_LAYERS = [1]


def _check(cond, msg=""):
    """045 纪律：显式判定，非 assert（python -O 不删除）。"""
    if not cond:
        raise SystemExit(f"[TEST-FAIL] {msg}")


def _build_syn_trace(base, name, prompt):
    """合成 trace fixture：S=1024、单层、GQA 8/2、D=64。固定 seed 可复现。
    不同 name/prompt → 不同目录 realpath 与不同 meta.json 字节（089 的
    trace 来源消歧素材）。"""
    tdir = os.path.join(base, name)
    os.makedirs(tdir, exist_ok=True)
    meta = {"S": SYN_S, "n_layers": 36, "prompt": prompt,
            "layers": SYN_LAYERS, "with_v": True,
            "question": f"synthetic {name}", "answer": ""}
    with open(os.path.join(tdir, "meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False)
    g = torch.Generator().manual_seed(20261011 + SYN_S)
    tq = 4
    qpos = torch.arange(SYN_S - tq, SYN_S, dtype=torch.long)
    for layer in SYN_LAYERS:
        k = torch.randn(SYN_S, 2, 64, generator=g)
        q = torch.randn(tq, 8, 64, generator=g)
        v = torch.randn(SYN_S, 2, 64, generator=g)
        torch.save({"k": k, "v": v, "q": q, "qpos": qpos, "S": SYN_S},
                   os.path.join(tdir, f"layer{int(layer):02d}.pt"))
    return tdir


def _run_entry(args, timeout=600):
    """跑入口脚本（与当前解释器同 -O 口径）。返回 CompletedProcess。"""
    argv = [sys.executable]
    if not __debug__:
        argv.append("-O")
    argv += [ENTRY] + list(args)
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)


def _load_jsonl(path):
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


# ---------------- S01 负 K2（GPT 复现例）四档全拒 + 正例数值不变 ----------------

def test_S01_negative_k2():
    from sparse_attn.indexer import dyn_controller as dc
    kw = dict(k1=128, k2=-1, bs=64, n_valid=32768, n_protected=256,
              n_prefix=128, n_swa=128)
    for tier in ("P_F", "P_C", "P_N"):
        ok, info = dc.compile_tier(tier, **kw)
        _check(not ok, f"{tier} 负 K2 必须拒绝（GPT 复现：旧版 ok=True）")
        _check(info.get("reason") == "struct",
               f"{tier} 拒绝原因应为 struct，得 {info.get('reason')}")
    ok, info = dc.compile_fixed("mavg", **kw)
    _check(not ok and info.get("reason") == "struct",
           f"fixed 档负 K2 同拒，得 ok={ok} {info.get('reason')}")
    # 正例对照：同几何合法 K2 编译成功，且基准数值与修复前逐位一致
    ok, info = dc.compile_tier("P_C", k1=128, k2=1024, bs=64, n_valid=32768,
                               n_protected=256, n_prefix=128, n_swa=128)
    _check(ok, f"合法 K2 正例应编译成功: {info}")
    _check(info["Kmid"] == 768, f"Kmid 应 min(1024,32768)-256=768（数值"
                                f"语义合法几何下不变），得 {info['Kmid']}")
    _check(info["Tn"] == 384 and info["Tf"] == 384,
           f"P_C 应 Tn=Tf=384（ρ=1/2），得 {info['Tn']}/{info['Tf']}")


# ---------------- S02 保护数超过 U（GPT 复现例） ----------------

def test_S02_protected_exceeds_valid():
    from sparse_attn.indexer import dyn_controller as dc
    kw = dict(k1=128, k2=1024, bs=64, n_valid=128, n_protected=256,
              n_prefix=128, n_swa=128)
    for tier in ("P_F", "P_C", "P_N"):
        ok, info = dc.compile_tier(tier, **kw)
        _check(not ok and info.get("reason") == "struct",
               f"{tier} n_protected(256)>n_valid(128) 必须 struct 拒"
               f"（GPT 复现：旧版 ok=True 且 Kmid 被压 0），得 {ok} {info}")
    ok, info = dc.compile_fixed("mavg", **kw)
    _check(not ok and info.get("reason") == "struct",
           f"fixed 档保护超 U 同拒，得 {ok} {info.get('reason')}")
    # 正例：n_protected == n_valid（全保护、空 middle 合法边界）放行
    ok, info = dc.compile_tier("P_C", k1=128, k2=1024, bs=64, n_valid=256,
                               n_protected=256, n_prefix=128, n_swa=128)
    _check(ok and info["Kmid"] == 0,
           f"n_protected==n_valid 是合法边界（Kmid=0），得 {ok} {info}")


# ---------------- S03 负 prefix / 负 window（GPT 复现例） ----------------

def test_S03_negative_prefix_window():
    from sparse_attn.indexer import dyn_controller as dc
    # GPT 复现例：n_prefix=-1, n_swa=257（-1+257=256 闭合，旧版 ok=True
    # 且 far_range=[-1, 24383] 落负起点）
    ok, info = dc.compile_tier("P_C", k1=128, k2=1024, bs=64, n_valid=32768,
                               n_protected=256, n_prefix=-1, n_swa=257)
    _check(not ok and info.get("reason") == "struct",
           f"负 prefix 必须拒绝（GPT 复现：旧版 ok=True 且 far_range "
           f"落 [-1, 24383]），得 {ok} {info.get('reason')}")
    _check("far_range" not in info, "拒绝路径不得携带区间摘要（负起点区间）")
    for tier in ("P_F", "P_N"):
        ok, info = dc.compile_tier(tier, k1=128, k2=1024, bs=64, n_valid=32768,
                                   n_protected=256, n_prefix=-1, n_swa=257)
        _check(not ok and info.get("reason") == "struct",
               f"{tier} 负 prefix 同拒（四档一致），得 {ok}")
    ok, info = dc.compile_fixed("mavg", k1=128, k2=1024, bs=64, n_valid=32768,
                                n_protected=256, n_prefix=-1, n_swa=257)
    _check(not ok and info.get("reason") == "struct",
           f"fixed 档负 prefix 同拒，得 {ok}")
    # 负 window（负 SWA 不是合法保护几何）
    ok, info = dc.compile_tier("P_C", k1=128, k2=1024, bs=64, n_valid=32768,
                               n_protected=256, n_prefix=257, n_swa=-1)
    _check(not ok and info.get("reason") == "struct", f"负 n_swa 必须拒，得 {ok}")
    # 正例：合法非负分解放行且区间正确
    ok, info = dc.compile_tier("P_C", k1=128, k2=1024, bs=64, n_valid=32768,
                               n_protected=256, n_prefix=128, n_swa=128)
    _check(ok and info["far_range"] == [128, 128 + info["F_len"]]
           and info["near_range"][0] >= 128,
           f"合法分解区间应从 n_prefix=128 起（无负起点），得 "
           f"{info.get('far_range')}/{info.get('near_range')}")


# ---------------- S04 负 k1 / 非正 bs ----------------

def test_S04_negative_k1_bs():
    from sparse_attn.indexer import dyn_controller as dc
    for bad in (0, -128):
        ok, info = dc.compile_tier("P_C", k1=bad, k2=1024, bs=64,
                                   n_valid=32768, n_protected=256)
        _check(not ok and info.get("reason") == "struct",
               f"k1={bad} 必须拒，得 {ok} {info.get('reason')}")
    ok, info = dc.compile_tier("P_C", k1=128, k2=1024, bs=0,
                               n_valid=32768, n_protected=256)
    _check(not ok and info.get("reason") == "struct", f"bs=0 必须拒，得 {ok}")
    # 正例
    ok, info = dc.compile_tier("P_C", k1=128, k2=1024, bs=64,
                               n_valid=32768, n_protected=256)
    _check(ok, f"合法正例应放行: {info.get('reason')}")


# ---------------- S05 K2=0 协议允许值单列口径 ----------------

def test_S05_k2_zero_distinct():
    from sparse_attn.indexer import dyn_controller as dc
    # k2=0：协议允许值（mid L2 预算恰为零）→ ok=True、Kmid=0、k2_zero 单列
    for tier in ("P_F", "P_C", "P_N"):
        ok, info = dc.compile_tier(tier, k1=128, k2=0, bs=64, n_valid=32768,
                                   n_protected=256, n_prefix=128, n_swa=128)
        _check(ok, f"{tier} k2=0 是协议允许值必须放行（不得与负值混同拒），"
                  f"得 ok={ok} {info.get('reason')} {info.get('detail', '')}")
        _check(info["Kmid"] == 0 and info["Tn"] == 0 and info["Tf"] == 0,
               f"{tier} K2=0 预算应全零，得 Kmid={info['Kmid']}")
        _check(info.get("k2_zero") is True,
               f"{tier} k2=0 应单列 k2_zero=True 如实记录，得 {info.get('k2_zero')}")
    ok, info = dc.compile_fixed("mavg", k1=128, k2=0, bs=64, n_valid=32768,
                                n_protected=256, n_prefix=128, n_swa=128)
    _check(ok and info.get("k2_zero") is True,
           f"fixed 档 k2=0 同样合法单列，得 ok={ok} {info.get('k2_zero')}")
    # 对照：负 K2 仍是 struct（单列口径不豁免负值）
    ok, info = dc.compile_tier("P_C", k1=128, k2=-1, bs=64, n_valid=32768,
                               n_protected=256, n_prefix=128, n_swa=128)
    _check(not ok and info.get("reason") == "struct"
           and "k2_zero" not in info,
           f"负 K2 必须仍是 struct 且不带 k2_zero 标记，得 {ok} {info}")


# ---------------- S06 保护分量闭合 / 成对 ----------------

def test_S06_closure_incomplete():
    from sparse_attn.indexer import dyn_controller as dc
    # 不闭合（G10 同口径在 CLI 层已有；编译器层同样拒绝）
    ok, info = dc.compile_tier("P_C", k1=128, k2=1024, bs=64, n_valid=32768,
                               n_protected=256, n_prefix=100, n_swa=100)
    _check(not ok and info.get("reason") == "struct"
           and "不闭合" in info.get("detail", ""),
           f"分量不闭合必须 struct 拒（区间不可由当前 U 合法构造），得 {info}")
    # 只给其一 = 分解不完整
    ok, info = dc.compile_tier("P_C", k1=128, k2=1024, bs=64, n_valid=32768,
                               n_protected=256, n_prefix=128)
    _check(not ok and info.get("reason") == "struct",
           f"只给 n_prefix 不给 n_swa 必须拒，得 {ok} {info.get('reason')}")
    # 正例：均缺省 = 合法（区间摘要记 None，计数仍准确——既有语义保留）
    ok, info = dc.compile_tier("P_C", k1=128, k2=1024, bs=64, n_valid=32768,
                               n_protected=256)
    _check(ok and info.get("near_range") is None
           and "range_note" in info,
           f"无分解参数应合法放行且区间记 None，得 {ok} {info.get('near_range')}")
    # 正例：闭合分解放行
    ok, info = dc.compile_tier("P_C", k1=128, k2=1024, bs=64, n_valid=32768,
                               n_protected=256, n_prefix=128, n_swa=128)
    _check(ok and info.get("near_range") is not None,
           f"闭合分解应放行并附区间，得 {ok}")


# ---------------- S07 CLI dry-run：struct 非零退出 ----------------

def test_S07_dryrun_struct_exit():
    # 负 K2（带 --n-valid）→ 非零退出 + [E124A-ABORT][STRUCT]
    r = _run_entry(["--dry-run", "--k2", "-1", "--n-valid", "32768"])
    _check(r.returncode != 0,
           f"dry-run 负 K2 必须非零退出，得 rc={r.returncode}")
    _check("E124A-ABORT" in (r.stdout + r.stderr)
           and "STRUCT" in (r.stdout + r.stderr)
           and "k2=-1" in (r.stdout + r.stderr),
           f"dry-run struct 错误须含 [E124A-ABORT][STRUCT] 与参数值:\n"
           f"{(r.stdout + r.stderr)[-500:]}")
    # 保护超 U → 非零退出
    r = _run_entry(["--dry-run", "--n-valid", "128", "--n-protected", "256",
                    "--n-prefix", "128", "--n-swa", "128"])
    _check(r.returncode != 0, f"dry-run 保护超 U 必须非零退出，得 rc={r.returncode}")
    # 负 K2 即使无 --n-valid（前置校验不依赖 U）→ 非零退出
    r = _run_entry(["--dry-run", "--k2", "-1"])
    _check(r.returncode != 0,
           f"dry-run 负 K2 无 --n-valid 也应被前置校验拒绝，得 rc={r.returncode}")
    # 正例 1：合法 dry-run（缺省 --n-valid）→ rc 0（诚实预览不误伤）
    r = _run_entry(["--dry-run"])
    _check(r.returncode == 0, f"合法 dry-run（n_valid 缺省）应 rc 0，得 {r.returncode}")
    # 正例 2：k2=0 dry-run → rc 0 且 k2_zero 单列如实预览
    r = _run_entry(["--dry-run", "--k2", "0", "--n-valid", "32768"])
    _check(r.returncode == 0, f"k2=0 dry-run 应 rc 0（协议允许值），得 {r.returncode}")
    cfg = json.loads(r.stdout)
    _check(cfg["compiled_preview"]["P_C"].get("k2_zero") is True,
           f"k2=0 预览应单列 k2_zero=True，得 {cfg['compiled_preview']['P_C']}")
    # 正例 3：合法显式 --n-valid 预览正常编译
    r = _run_entry(["--dry-run", "--n-valid", "32768"])
    _check(r.returncode == 0
           and json.loads(r.stdout)["compiled_preview"]["P_C"]["Kmid"] == 768,
           f"合法显式 n_valid 预览应正常编译，得 rc={r.returncode}")


# ---------------- S08 CLI 实跑：struct 非零退出（前置，不依赖 trace） ----------------

def test_S08_realrun_struct_exit():
    base = tempfile.mkdtemp(prefix="e124s08_")
    try:
        out = os.path.join(base, "dec.jsonl")
        # 负 K2 实跑：main 前置校验先于 trace 读取——即便 trace-dir 不存在
        # 也应是 struct 拒绝而非「文件不存在」
        r = _run_entry(["--trace-dir", os.path.join(base, "no_such_trace"),
                        "--out", out, "--k2", "-1"])
        _check(r.returncode != 0,
               f"实跑负 K2 必须非零退出，得 rc={r.returncode}")
        _check("E124A-ABORT" in (r.stdout + r.stderr)
               and "STRUCT" in (r.stdout + r.stderr),
               f"实跑 struct 错误须含 [E124A-ABORT][STRUCT]:\n"
               f"{(r.stdout + r.stderr)[-400:]}")
        _check(not os.path.exists(out) or os.path.getsize(out) == 0,
               "struct 拒绝不得落任何决策")
        # 正例：合法输入实跑成功（合成 trace，auto seq-id）
        tdir = _build_syn_trace(base, "syn_ok", "s08")
        out2 = os.path.join(base, "dec_ok.jsonl")
        r = _run_entry(["--trace-dir", tdir, "--out", out2])
        _check(r.returncode == 0, f"合法实跑应成功，rc={r.returncode}:\n"
                                  f"{(r.stdout + r.stderr)[-600:]}")
        recs = _load_jsonl(out2)
        _check(len(recs) == 1, f"应落 1 条决策，得 {len(recs)}")
    finally:
        shutil.rmtree(base, ignore_errors=True)


# ---------------- I01 auto 模式：两个 trace 目录 → 唯一 seq_id/seed ----------------

def test_I01_auto_unique_seq_id():
    from sparse_attn.indexer import dyn_controller as dc
    base = tempfile.mkdtemp(prefix="e124i01_")
    try:
        tA = _build_syn_trace(base, "seq_A", "prompt-A")
        tB = _build_syn_trace(base, "seq_B", "prompt-B")
        outA = os.path.join(base, "decA.jsonl")
        outB = os.path.join(base, "decB.jsonl")
        for tdir, out in ((tA, outA), (tB, outB)):
            r = _run_entry(["--trace-dir", tdir, "--out", out])
            _check(r.returncode == 0,
                   f"auto 模式实跑应成功 {tdir}: {(r.stdout + r.stderr)[-400:]}")
        ra, rb = _load_jsonl(outA)[0], _load_jsonl(outB)[0]
        sA, sB = ra["seq_id"], rb["seq_id"]
        _check(sA != sB, f"两个不同 trace 目录应派生两个不同 seq_id，"
                        f"得 {sA} == {sB}（089 缺陷场景：全目录共享 0）")
        # 同层 seed 实证区分：sha256('e124a|{id}|{layer}') 由 seq_id 派生
        seedA = dc._seq_layer_seed(sA, 1)
        seedB = dc._seq_layer_seed(sB, 1)
        _check(seedA != seedB,
               f"同层不同 seq 的投影 seed 必须不同（seed 契约逐 seq×layer "
               f"唯一），得 {seedA} == {seedB}")
        _check(not torch.equal(dc.make_projection(sA, 1, 64),
                               dc.make_projection(sB, 1, 64)),
               "同层不同 seq 的随机投影 R 必须不同（089 修复目标）")
        # decision 记录含 trace 来源（目录名 + meta 输入 sha256）
        for rec, tdir, tname in ((ra, tA, "seq_A"), (rb, tB, "seq_B")):
            geo = rec["geometry"]
            src = geo.get("trace_source")
            _check(isinstance(src, dict) and src.get("trace_dir_name") == tname,
                   f"trace_source 应含目录名 {tname}，得 {src}")
            _check(isinstance(src.get("trace_meta_sha256"), str)
                   and len(src["trace_meta_sha256"]) == 64,
                   f"trace_source 应含 64 位 meta 输入 sha256，得 {src}")
            _check(geo.get("seq_id") == rec["seq_id"]
                   and str(geo.get("seq_id_source", "")).startswith("auto:"),
                   f"geometry 应如实记录 seq_id 与 auto 派生来源，得 "
                   f"{geo.get('seq_id_source')}")
        # meta sha 消歧实证：两个目录 meta 内容不同 → sha 不同
        _check(ra["geometry"]["trace_source"]["trace_meta_sha256"]
               != rb["geometry"]["trace_source"]["trace_meta_sha256"],
               "不同目录的 meta 输入 sha 应不同（来源消歧字段有区分力）")
        # auto seq_id 稳定性：同目录重跑（fresh out）seq_id 不变
        outA2 = os.path.join(base, "decA_rerun.jsonl")
        r = _run_entry(["--trace-dir", tA, "--out", outA2])
        _check(r.returncode == 0, f"同目录重跑 fresh out 应成功: {r.stderr[-300:]}")
        _check(_load_jsonl(outA2)[0]["seq_id"] == sA,
               "auto seq_id 从 trace 目录派生，跨运行必须稳定")
    finally:
        shutil.rmtree(base, ignore_errors=True)


# ---------------- I02 同目录重跑同 --out → 重复 identity 拒绝 ----------------

def test_I02_rerun_same_out_rejected():
    base = tempfile.mkdtemp(prefix="e124i02_")
    try:
        tdir = _build_syn_trace(base, "syn_rerun", "i02")
        out = os.path.join(base, "dec.jsonl")
        r = _run_entry(["--trace-dir", tdir, "--out", out])
        _check(r.returncode == 0, f"首跑应成功: {(r.stdout + r.stderr)[-300:]}")
        r2 = _run_entry(["--trace-dir", tdir, "--out", out])
        _check(r2.returncode != 0,
               f"同目录重跑同 --out 必须非零退出（重复 identity 门禁），"
               f"得 rc={r2.returncode}")
        _check("E124A-IDENTITY-DUP" in (r2.stdout + r2.stderr),
               f"错误须含 IDENTITY-DUP 标记:\n{(r2.stdout + r2.stderr)[-400:]}")
        recs = _load_jsonl(out)
        _check(len(recs) == 1,
               f"重复追加被拒后文件不得出现重复行（聚合双计防护），得 {len(recs)}")
    finally:
        shutil.rmtree(base, ignore_errors=True)


# ---------------- I03 显式 --seq-id 冒用：跨输入 identity clash 拒绝 ----------------

def test_I03_explicit_seq_id_clash():
    from sparse_attn.indexer import dyn_controller as dc
    base = tempfile.mkdtemp(prefix="e124i03_")
    try:
        tA = _build_syn_trace(base, "clash_A", "i03-a")
        tB = _build_syn_trace(base, "clash_B", "i03-b")
        out = os.path.join(base, "dec.jsonl")
        r = _run_entry(["--trace-dir", tA, "--out", out, "--seq-id", "0"])
        _check(r.returncode == 0, f"单目录显式 seq-id 0 首跑合法: {r.stderr[-200:]}")
        # 碰撞证明：显式同 seq_id 同层 → seed 相同（正是 089 缺陷本体，
        # 门禁必须在混入前拦截）
        _check(dc._seq_layer_seed(0, 1) == dc._seq_layer_seed(0, 1)
               and dc.make_projection(0, 1, 64).equal(
                   dc.make_projection(0, 1, 64)),
               "seed 契约自一致性（同 id 同层 seed 相同）——冒用同 id 即碰撞")
        r2 = _run_entry(["--trace-dir", tB, "--out", out, "--seq-id", "0"])
        _check(r2.returncode != 0,
               f"不同 trace 目录冒用同一显式 seq_id 追加同一 --out 必须"
               f"非零退出，得 rc={r2.returncode}")
        _check("E124A-IDENTITY-CLASH" in (r2.stdout + r2.stderr),
               f"错误须含 IDENTITY-CLASH 标记:\n{(r2.stdout + r2.stderr)[-400:]}")
        _check("clash_A" in (r2.stdout + r2.stderr)
               and "clash_B" in (r2.stdout + r2.stderr),
               "clash 错误须给出两侧 trace 来源（消歧信息可见）")
        recs = _load_jsonl(out)
        _check(len(recs) == 1, f"clash 拒绝后不得混入第二条，得 {len(recs)}")
    finally:
        shutil.rmtree(base, ignore_errors=True)


# ---------------- I04 显式 --seq-id 保留（单目录合法） ----------------

def test_I04_explicit_seq_id_kept():
    base = tempfile.mkdtemp(prefix="e124i04_")
    try:
        tdir = _build_syn_trace(base, "syn_explicit", "i04")
        out = os.path.join(base, "dec.jsonl")
        r = _run_entry(["--trace-dir", tdir, "--out", out, "--seq-id", "7"])
        _check(r.returncode == 0, f"显式 seq-id 7 单目录应合法，rc={r.returncode}:"
                                 f"{(r.stdout + r.stderr)[-300:]}")
        rec = _load_jsonl(out)[0]
        _check(rec["seq_id"] == 7, f"决策应记显式 seq_id=7，得 {rec['seq_id']}")
        _check(rec["geometry"].get("seq_id_source") == "explicit",
               f"显式模式应如实记 seq_id_source=explicit，得 "
               f"{rec['geometry'].get('seq_id_source')}")
        _check("trace_source" in rec["geometry"],
               "显式模式同样记录 trace 来源（身份可回链）")
        # 非法值拒绝
        r = _run_entry(["--trace-dir", tdir, "--out", os.path.join(base, "d2.jsonl"),
                        "--seq-id", "not_an_int"])
        _check(r.returncode != 0, f"--seq-id 非法值应拒绝，得 rc={r.returncode}")
        r = _run_entry(["--trace-dir", tdir, "--out", os.path.join(base, "d3.jsonl"),
                        "--seq-id", "-3"])
        _check(r.returncode != 0, f"--seq-id 负值应拒绝，得 rc={r.returncode}")
    finally:
        shutil.rmtree(base, ignore_errors=True)


# ---------------- I05 DecisionLog 库级 identity 门禁单元 ----------------

def test_I05_decisionlog_gate_unit():
    from sparse_attn.indexer import dyn_controller as dc
    base = tempfile.mkdtemp(prefix="e124i05_")
    try:
        p = os.path.join(base, "dec.jsonl")
        log = dc.DecisionLog(p)
        d1 = {"seq_id": 1, "layer_idx": 1, "phase": "prefill", "state_epoch": 0,
              "trace_source": {"trace_dir_name": "A", "trace_meta_sha256": "x"}}
        log.append(d1)
        log.close()
        # 重复 identity 同来源 → IDENTITY-DUP
        log2 = dc.DecisionLog(p)
        try:
            log2.append(dict(d1))
            _check(False, "重复 identity 追加必须抛 ValueError，却放行了")
        except ValueError as e:
            _check("E124A-IDENTITY-DUP" in str(e), f"DUP 标记: {e}")
        log2.close()
        # 重复 identity 异来源 → IDENTITY-CLASH
        log3 = dc.DecisionLog(p)
        d2 = {"seq_id": 1, "layer_idx": 1, "phase": "prefill", "state_epoch": 0,
              "trace_source": {"trace_dir_name": "B", "trace_meta_sha256": "y"}}
        try:
            log3.append(d2)
            _check(False, "异来源同 identity 追加必须抛 ValueError，却放行了")
        except ValueError as e:
            _check("E124A-IDENTITY-CLASH" in str(e), f"CLASH 标记: {e}")
        # 不同 identity（异层/异 seq）→ 放行
        log3.append({"seq_id": 1, "layer_idx": 2, "phase": "prefill",
                     "state_epoch": 0, "trace_source": d1["trace_source"]})
        log3.append({"seq_id": 2, "layer_idx": 1, "phase": "prefill",
                     "state_epoch": 0, "trace_source": d2["trace_source"]})
        log3.close()
        _check(len(_load_jsonl(p)) == 3, "文件应恰 3 条（1 首跑 + 2 合法异 identity）")
        # 既有文件损坏 → 拒绝在损坏文件上继续混写
        with open(p, "a") as f:
            f.write("{corrupted\n")
        try:
            dc.DecisionLog(p)
            _check(False, "损坏文件上打开追加模式必须抛 ValueError")
        except ValueError as e:
            _check("E124A-IDENTITY-GATE" in str(e), f"GATE 标记: {e}")
    finally:
        shutil.rmtree(base, ignore_errors=True)


# ================================================================ main

PLAN = [
    ("S01", test_S01_negative_k2),
    ("S02", test_S02_protected_exceeds_valid),
    ("S03", test_S03_negative_prefix_window),
    ("S04", test_S04_negative_k1_bs),
    ("S05", test_S05_k2_zero_distinct),
    ("S06", test_S06_closure_incomplete),
    ("S07", test_S07_dryrun_struct_exit),
    ("S08", test_S08_realrun_struct_exit),
    ("I01", test_I01_auto_unique_seq_id),
    ("I02", test_I02_rerun_same_out_rejected),
    ("I03", test_I03_explicit_seq_id_clash),
    ("I04", test_I04_explicit_seq_id_kept),
    ("I05", test_I05_decisionlog_gate_unit),
]


def main():
    """红绿 runner：显式 _check（python -O 兼容）；E124SI_ONLY 过滤；
    一次跑完全部用例给红绿全景；FAIL>0 非零退出。"""
    only = os.environ.get("E124SI_ONLY", "")
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
    print(f"\nE124A-STRUCT-IDENTITY RESULT: PASS={n_pass} FAIL={n_fail} "
          f"(total {len(plan)})")
    if failed:
        print(f"FAILED: {failed}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
