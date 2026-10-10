#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""#203（GPT 2026-10-10 2131 增量复审）076/077/078 三项修复验收套件
（CPU only；python 与 python -O 双跑安全——全部显式判定，无 assert）。

  P1  076 pairwise 单射：GPT 6 配置碰撞矩阵（minmax/avg、cluster/
      sim_greedy 等同路径族）修复前 3 组同名碰撞（6 配置 → 3 名），
      修复后 _h<hash10> 身份段全互异；readable 前缀结构与审计矩阵
      逐组吻合（碰撞组 {c1,c2,c3}/{c4,c5}/{c6}——证明 hash 正是
      分离器）；c1 精确名硬锚点（字段集固定后逐位稳定）。
  P2  076 同配置顺序无关 + 缺省等价：argparse 全缺省显式命名空间 vs
      程序化最小命名空间（getattr 兜底）vs 字段乱序构造 → 同名同
      hash；多次调用稳定（缺省 namespace 与显式默认值 namespace
      必须同 hash——B09 族失配教训的 hash 侧硬约束）。
  P3  076 fail-closed 写入门：目标已存在 + sidecar manifest 异配置
      → SystemExit 拒绝且目标字节不变（不覆盖）；同 manifest →
      幂等放行；目标存在但无 sidecar → 拒绝；新路径 → 放行。
  P4  076 截断保身份段：超长名截断后 _h<hash10> 尾段保留、异配置
      截断名仍互异；非 tli 名回退旧口径。
  N1  077 NUL pointer 单格：.tli_gen 内容 b"gen\x00name\n" → 探针
      STATE=invalid rc=2（修复前 ValueError 裸 traceback rc=1）；
      控制字符 \x01 同口径；进程内 resolve 直接 SystemExit（077）。
  N2  077 audit 坏格夹好格：好格(111) + NUL 坏格(155) + 好格(222)
      → 逐格继续枚举（字典序 222 在坏格后仍 OK）、total=3
      invalid=1 rc=2；好格单格探针 STATE=complete（fixture 自证）。
  D1  078 假干净复现：root 仅含 dangling symlink（GPT 最小复现）
      → coverage_errors=1 rc=2，「零 pointer 产物」注记被抑制
      （修复前 total=0 invalid=0 coverage_errors=0 rc=0）。
  D2  078 好格夹 dangling 别名：非指针形 dangling 链接计 coverage
      error（保留「从有效别名根直接审计」出口），两好格照常 OK。
  D3  078 指针形 dangling：*.tli_gen 名的 dangling 链接走「存在即
      证据」invalid 口径（071① symlink 门禁），不与 coverage 双计。

红态对照：修复前（基点 9fb2643b0）P1-P4 因缺 _h 尾段/gate 函数、
N1/N2 因 ValueError 裸 traceback（rc=1 无 STATE）、D1/D2/D3 因
coverage 缺口（rc=0 假干净）全 FAIL——本套件对主树可作红态复跑
（git stash 后运行即红）。

用法（与 pointer 套件同环境，须带 torch 的解释器）：
  cd <sglang 根> && PYTHONPATH=two-level-attention:. \
    python benchmark/RULER/test_e119_fixes_076_077_078.py
  python -O 同上双跑；E119_ONLY=P1,N1 子集过滤。
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import types

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from benchmark.RULER.yarn_receipt import (  # noqa: E402
    GENERATION_POINTER_SUFFIX, _file_sha256, build_yarn_receipt,
    commit_yarn_generation, resolve_generation_pointer,
    stage_yarn_generation, stage_yarn_receipt)
import sparse_attn.info as INFO  # noqa: E402  （torch 依赖；命名/manifest/gate 单源）

PROBE = os.path.join(REPO, "benchmark", "RULER", "gen_completion_probe.py")

# -O 传播：套件在 python -O 下运行时子进程同口径（探针自身无 assert，
# 此处保证红绿证据的 -O 双跑覆盖子进程分支）
_OFLAG = [] if __debug__ else ["-O"]


def _check(cond, msg=""):
    """065/067 纪律：显式判定，非 assert（python -O 不删除）。"""
    if not cond:
        raise SystemExit(f"[TEST-FAIL] {msg}")


# ================================================================ fixtures

def _gcfg(far_m="minmax", near_m="avg", far_s="4bit", near_s="4bit",
          kmeans=False, **kw):
    """GPT 076 矩阵配置构造（tia_level2_topk=2048/c4，α.25/β.125/γ.625）。

    kw 可覆盖任意字段（P2 缺省等价用）。"""
    base = dict(method="tli", tia_block_size=64, tia_level1_topk=128,
                tia_level2_topk=2048, tia_level2_cmp_ratio=4,
                tli_alpha=0.25, tli_beta=0.125, tli_gamma=0.625,
                tli_enable_kmeans=kmeans, tli_enable_layer_skip=False,
                tli_far_method=far_m, tli_near_method=near_m,
                tli_far_select=far_s, tli_near_select=near_s)
    base.update(kw)
    return types.SimpleNamespace(**base)


def _mk_committed_cell(out_dir, task, t, rows=3):
    """进程内构造完整好格（059+066 指针协议零依赖侧）：不可变 gen 目录
    + v2 同代绑定回执 + 单次原子切指针；构造后自证 resolve+validate
    双过（fixture 不依赖被测修复点）。"""
    os.makedirs(out_dir, exist_ok=True)
    logical = os.path.join(out_dir, f"{task}-stubm-{t}.jsonl")
    attempt = f"{int(time.time() * 1000)}-pid{os.getpid()}-t{t}"
    tmp_pred = stage_yarn_generation(logical, attempt)
    with open(tmp_pred, "w", encoding="utf-8") as f:
        for i in range(rows):
            f.write(json.dumps({"pred": f"stub{i}", "answers": [f"gt{i}"],
                                "length": 32768, "budget": 0},
                               ensure_ascii=False) + "\n")
    receipt = build_yarn_receipt(
        yarn_enabled=False, effective_factor=None, yarn_factor_cli=None,
        rope_scaling=None, context_length=32768, task=task,
        model_path="/tmp/fake-model-e119-076-078", model_config_sha256=None,
        native_mpe=32768,
        generation_params={"max_gen": 128, "max_num": 0, "seed": 42,
                           "method": "tli", "pred_postfix": "_stub", "t": t},
        producer_script_path="benchmark/RULER/test_e119_fixes_076_077_078.py",
        producer_script_sha256=hashlib.sha256(b"fixture").hexdigest(),
        run_id=attempt,
        prediction_basename=os.path.basename(logical),
        prediction_sha256=_file_sha256(tmp_pred),
        prediction_lines=rows)
    tmp_rcp = stage_yarn_receipt(logical, receipt, attempt)
    commit_yarn_generation(tmp_pred, tmp_rcp, logical)
    # fixture 自证：resolve + validate 双过（好格语义成立才可作夹层）
    gen = resolve_generation_pointer(logical)
    _check(gen is not None, f"fixture 自证失败：resolve({logical}) = None")
    from benchmark.RULER.yarn_receipt import validate_committed_generation
    validate_committed_generation(gen["pred_path"], gen["rcp_path"])
    return logical


def _write_bad_pointer(out_dir, basename_no_ext, content_bytes):
    """手写坏指针格（077 类：内容级坏，无 gen 目录）。"""
    os.makedirs(out_dir, exist_ok=True)
    logical = os.path.join(out_dir, basename_no_ext + ".jsonl")
    ptr = logical + GENERATION_POINTER_SUFFIX
    with open(ptr, "wb") as f:
        f.write(content_bytes)
    return logical


def _probe_cli(out_dir, task, max_num):
    argv = [sys.executable, "-u"] + _OFLAG + \
        [PROBE, "--out-dir", out_dir, "--task", task,
         "--max-num", str(max_num)]
    return subprocess.run(argv, capture_output=True, text=True)


def _audit_cli(root):
    argv = [sys.executable, "-u"] + _OFLAG + [PROBE, "--audit-dir", root]
    return subprocess.run(argv, capture_output=True, text=True)


def _audit_summary(stdout):
    """解析 AUDIT RESULT 行 → (total, invalid, coverage_errors)。"""
    for ln in stdout.splitlines():
        if ln.startswith("AUDIT RESULT:"):
            fields = dict(kv.split("=", 1) for kv in
                          ln[len("AUDIT RESULT:"):].split())
            return (int(fields["total"]), int(fields["invalid"]),
                    int(fields["coverage_errors"]))
    return None


# ================================================================ 076

def test_P1_matrix_pairwise_injective():
    """GPT 6 配置矩阵：修复前 3 组同名碰撞（GPT 审计 CPU 复现），修复后
    全互异；readable 前缀组结构吻合（hash 是唯一分离器）+ c1 精确锚点。"""
    cfgs = [
        ("c1 minmax/avg/4bit/4bit", _gcfg("minmax", "avg", "4bit", "4bit")),
        ("c2 avg/avg/4bit/4bit", _gcfg("avg", "avg", "4bit", "4bit")),
        ("c3 minmax/minmax/4bit/4bit", _gcfg("minmax", "minmax", "4bit",
                                             "4bit")),
        ("c4 avg/avg/cluster/4bit", _gcfg("avg", "avg", "cluster", "4bit",
                                          kmeans=True)),
        ("c5 avg/avg/cluster/cluster", _gcfg("avg", "avg", "cluster",
                                             "cluster", kmeans=True)),
        ("c6 avg/avg/sim_greedy/4bit", _gcfg("avg", "avg", "sim_greedy",
                                             "4bit")),
    ]
    names = {}
    for tag, ns in cfgs:
        n = INFO.get_method_name_with_info(ns)
        _check(INFO._TREATMENT_HASH_RE.match(n) is not None,
               f"{tag}: tli 名缺 _h<hash10> 身份尾段：{n!r}")
        names[tag] = n
        print(f"  076-matrix {tag} -> {n}")
    # pairwise 互异（6 选 2 = 15 对全查）
    keys = list(names)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            _check(names[keys[i]] != names[keys[j]],
                   f"076 单射失败：{keys[i]} 与 {keys[j]} 同名 "
                   f"{names[keys[i]]!r}（treatment 矩阵碰撞回归）")
    # readable 前缀组结构与 GPT 审计矩阵逐组吻合（碰撞组即 hash 分离组）
    def _readable(n):
        m = INFO._TREATMENT_HASH_RE.match(n)
        _check(m is not None, f"无法拆 readable 段：{n!r}")
        return m.group("readable")
    r = {k: _readable(v) for k, v in names.items()}
    _check(r["c1 minmax/avg/4bit/4bit"] == r["c2 avg/avg/4bit/4bit"]
           == r["c3 minmax/minmax/4bit/4bit"]
           == "tli_64_128_2048_c4_a0.25_b0.125_g0.625",
           f"c1/c2/c3 应为 GPT 矩阵碰撞组 1（同 readable），实际 {r}")
    _check(r["c4 avg/avg/cluster/4bit"] == r["c5 avg/avg/cluster/cluster"]
           == "tli_64_128_2048_c4_Ba0.25_b0.125_g0.625",
           f"c4/c5 应为 GPT 矩阵碰撞组 2，实际 {r}")
    _check(r["c6 avg/avg/sim_greedy/4bit"]
           == "tli_64_128_2048_c4_a0.25_b0.125_g0.625",
           f"c6 readable 应与审计矩阵一致（组 3 与组 1 同 readable），实际 {r}")
    # c1 精确名硬锚点：_TREATMENT_FIELD_DEFAULTS 固定后逐位稳定；
    # 字段集变更时此锚点必须重算（同步更新 A4/T5 同理）。
    # 079 口径：manifest 新增 tia_enable_async_topk（getattr 缺省
    # False）→ 全部 tli hash 重算，锚点由 he797a70df5 更新。
    # 081 + kimi3 0316 口径（B/D 开关入 manifest + 默认 D′ 掩码内容
    # 身份）：tli_enable_kmeans/tli_enable_layer_skip 入
    # _TREATMENT_FIELD_DEFAULTS（防可读段截断吃掉 B/D 位后 hash 互覆）
    # → 全部 tli hash 重算，锚点由 h046b987e01 更新为 h175d1bcba6
    #（本矩阵 c1-c6 全部 layer_skip=False，manifest D′ 掩码身份恒
    # null，故本锚点只受 kimi3 0316 字段集扩张影响）。
    _check(names["c1 minmax/avg/4bit/4bit"]
           == "tli_64_128_2048_c4_a0.25_b0.125_g0.625_h175d1bcba6",
           f"c1 稳定性锚点失配（字段集被改？须重算锚点）："
           f"{names['c1 minmax/avg/4bit/4bit']!r}")
    return "PASS"


def test_P2_order_independent_and_default_equivalent():
    """同配置三种构造（argparse 显式全缺省 / 程序化最小 ns / 乱序 kw）
    → 同名同 hash；重复调用稳定。缺省 getattr 与 argparse 默认值必须
    同 hash（「显式传默认值 ≠ 不传」是 B09 族失配的 hash 侧硬约束）。"""
    import argparse as _ap
    from sparse_attn.arguments import add_sparse_attn_args
    parser = _ap.ArgumentParser()
    add_sparse_attn_args(parser)
    # argparse 缺省（kmeans/layer_skip 默认 True）须显式对齐 _gcfg 的
    # False——本测试比对的缺省等价是 manifest 字段集，非 readable 段
    ns_argparse = parser.parse_args(
        ["--method", "tli", "--tia_level2_topk", "2048",
         "--tia_level2_cmp_ratio", "4", "--tli_alpha", "0.25",
         "--tli_beta", "0.125", "--tli_gamma", "0.625",
         "--tli_enable_kmeans", "false", "--tli_enable_layer_skip",
         "false"])
    name_argparse = INFO.get_method_name_with_info(ns_argparse)

    ns_min = _gcfg("minmax", "avg", "4bit", "4bit")  # getattr 兜底缺省
    name_min = INFO.get_method_name_with_info(ns_min)
    _check(name_argparse == name_min,
           f"argparse 全缺省 vs 程序化最小 ns 应同名（缺省等价）："
           f"{name_argparse!r} vs {name_min!r}")
    # 与 P1 的 c1（同一配置）也须一致——跨构造路径单射到同一身份
    # （079 口径锚点：async 字段入 manifest 后重算；081+kimi3 0316
    # 口径：B/D 入 _TREATMENT_FIELD_DEFAULTS 后与 P1 c1 同步重算）
    _check(name_min == "tli_64_128_2048_c4_a0.25_b0.125_g0.625_h175d1bcba6",
           f"跨构造路径身份漂移：{name_min!r}")

    # 乱序构造（字段注入顺序不影响 sort_keys 序列化）
    kw = {"tli_far_select": "4bit", "tia_block_size": 64,
          "tli_gamma": 0.625, "tli_near_method": "avg",
          "tia_level2_cmp_ratio": 4, "tli_beta": 0.125,
          "tli_far_method": "minmax", "tia_level1_topk": 128,
          "tli_alpha": 0.25, "tli_near_select": "4bit",
          "tia_level2_topk": 2048, "tli_enable_kmeans": False,
          "tli_enable_layer_skip": False, "method": "tli"}
    ns_shuffle = types.SimpleNamespace(**{k: kw[k] for k in reversed(list(kw))})
    _check(INFO.get_method_name_with_info(ns_shuffle) == name_min,
           "同配置乱序构造应同名（sort_keys 序列化顺序无关）")

    # 重复调用稳定（同一 args 对象多次求名/hash 不漂移）
    _check(INFO.get_method_name_with_info(ns_min) == name_min,
           "同 args 重复求名不稳定")
    h1 = INFO._treatment_hash(ns_min)
    h2 = INFO._treatment_hash(ns_min)
    _check(h1 == h2 and len(h1) == 10, f"hash 重复求值不稳定/长度错：{h1!r}")
    # manifest json 与 hash 单源：sidecar 串可独立复核出同一 hash
    payload = INFO.get_treatment_manifest_json(ns_min)
    _check(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10] == h1,
           "get_treatment_manifest_json 与 _treatment_hash 非同源")
    return "PASS"


def test_P3_gate_fail_closed_no_overwrite(base):
    """076 fail-closed 写入门：异配置目标拒绝运行（不覆盖、字节不变）；
    同配置幂等放行；缺 sidecar 拒绝；新路径放行。"""
    d = os.path.join(base, "p3")
    os.makedirs(d, exist_ok=True)
    out = os.path.join(
        d, "hotpotqa-tli_64_128_2048_c4_a0.25_b0.125_g0.625_"
          "h175d1bcba6-09090909.jsonl")   # 081+kimi3 0316 口径：与 P1 c1 同步重算
    data_bytes = b'{"pred": "arm-A"}\n'
    with open(out, "wb") as f:
        f.write(data_bytes)
    manifest_A = INFO.get_treatment_manifest_json(
        _gcfg("minmax", "avg", "4bit", "4bit"))
    manifest_B = INFO.get_treatment_manifest_json(
        _gcfg("avg", "avg", "4bit", "4bit"))          # 异配置（far_method）
    sidecar = out + INFO.TREATMENT_MANIFEST_SIDECAR_SUFFIX
    with open(sidecar, "w", encoding="utf-8") as f:
        f.write(manifest_A)

    # ③ 异配置 → SystemExit 拒绝 + 目标字节不变（不覆盖）
    refused = False
    try:
        INFO.gate_output_treatment_identity(out, manifest_B)
    except SystemExit as e:
        refused = True
        msg = str(e)
        _check("GATE-FAIL" in msg and "076" in msg,
               f"拒绝消息应带 GATE-FAIL+076：{msg!r}")
    _check(refused, "异配置目标未拒绝（fail loudly 缺失 = 076 回归）")
    with open(out, "rb") as f:
        _check(f.read() == data_bytes, "拒绝路径不得改写目标字节")

    # 同配置 → 幂等放行（SKIP 语义）
    sc = INFO.gate_output_treatment_identity(out, manifest_A)
    _check(sc == sidecar, f"sidecar 路径约定漂移：{sc!r}")

    # ① 目标存在但无 sidecar → 拒绝（provenance 未知）
    out2 = os.path.join(d, "gov_report-tli_x_h0000000000-09090909.jsonl")
    with open(out2, "wb") as f:
        f.write(b"stale\n")
    refused2 = False
    try:
        INFO.gate_output_treatment_identity(out2, manifest_A)
    except SystemExit as e:
        refused2 = "GATE-FAIL" in str(e)
    _check(refused2, "无 sidecar 的既有目标未拒绝（provenance 未知须 fail loudly）")

    # 新路径 → 放行（返回 sidecar 路径）
    out3 = os.path.join(d, "musique-tli_x_h1111111111-09090909.jsonl")
    sc3 = INFO.gate_output_treatment_identity(out3, manifest_A)
    _check(sc3 == out3 + INFO.TREATMENT_MANIFEST_SIDECAR_SUFFIX,
           "新路径应放行并返回 sidecar 约定路径")
    _check(not os.path.exists(out3), "放行不等于写数据（gate 零写）")
    return "PASS"


def test_P4_truncation_keeps_hash():
    """截断保身份段：超长名先截 readable 中段、保 _h 尾段与 -t 后缀；
    异配置截断名仍互异（076 审计：尾部截断后再碰撞的回归位）；
    非 tli 名回退旧口径。"""
    import re as _re
    prefix = "d" * 200
    mn1 = INFO.get_method_name_with_info(_gcfg("minmax", "avg", "4bit",
                                               "4bit"))
    mn2 = INFO.get_method_name_with_info(_gcfg("avg", "avg", "4bit", "4bit"))
    t1 = INFO.truncate_output_name_keep_hash(prefix, mn1, "09090909")
    t2 = INFO.truncate_output_name_keep_hash(prefix, mn2, "09090909")
    print(f"  076-truncate t1 -> {t1!r}")
    print(f"  076-truncate t2 -> {t2!r}")
    _check(t1 != t2, "截断后异配置仍互覆（hash 尾段被吃 = 076 回归）")
    for t, mn in ((t1, mn1), (t2, mn2)):
        m = _re.search(r"_h([0-9a-f]{10})$", mn)
        _check(m is not None, f"method_name 缺 hash 段：{mn!r}")
        _check(t.endswith(f"_h{m.group(1)}-09090909"),
               f"截断名须保 hash 尾段与 -t：{t!r}")
        # 081 + kimi3 0316 追加修复 1：上限 245 → 230——out_fn 落盘拼
        # ".jsonl"（6）+ 写门 sidecar ".tli_manifest.json"（18），245 下
        # sidecar 实名 269 > ext4 255 → OSError 36（kimi3 实测）；
        # 230 = 255 − 6 − 18 − 1。
        _check(len(t) <= 230, f"截断名超 230 上限（sidecar 链超 ext4）：{len(t)}")
        _check(len(t) + len(".jsonl") + len(".tli_manifest.json") <= 255,
               f"sidecar 实名链超 ext4 255：{len(t) + 6 + 18}")
    # readable 中段截断占位存在（前缀保留）
    _check(t1.startswith(prefix + "-") and "..." in t1,
           "截断名应保留 dataset 前缀并以 ... 占位中段")
    # 非 tli 名回退旧口径（旧代码 out_fn[:limit]+"..." 语义，limit=230）
    fb = INFO.truncate_output_name_keep_hash(prefix, "quest_64_128", "0909")
    _check(fb == (prefix + "-quest_64_128-0909")[:230] + "...",
           f"非 tli 名应回退旧截断口径，实际 {fb!r}")
    # split_method_name_hash：可拆回（readable + hash 段）
    r1, h1 = INFO.split_method_name_hash(mn1)
    _check(r1 + h1 == mn1 and h1.startswith("_h"),
           f"split 不可逆：{r1!r} + {h1!r} != {mn1!r}")
    rn, hn = INFO.split_method_name_hash("none")
    _check(rn == "none" and hn == "", "非 tli 名 split 应返回原名/空段")
    return "PASS"


# ================================================================ 077

def test_N1_nul_pointer_single_cell_invalid(base):
    """077：.tli_gen 内容 b"gen\x00name\n"（合法 UTF-8，NUL 穿透名称三查）
    → 单格 STATE=invalid rc=2（修复前 ValueError 裸 traceback rc=1 无四态）；
    控制字符同口径；进程内 resolve 直接 SystemExit（077 前置拒绝，
    任何文件系统调用前）。"""
    for label, raw in (("nul", b"gen\x00name\n"), ("ctrl", b"gen\x01name\n")):
        # 每变体独立子目录（单格探查的 task glob 语义：task=vt 匹配全部
        # vt-*.jsonl 候选，混放会把另一变体卷进同一探查）
        d = os.path.join(base, f"n1_{label}")
        os.makedirs(d, exist_ok=True)
        logical = _write_bad_pointer(d, "vt-stubm-bad", raw)
        # 进程内共享 resolver：077① 前置拒绝 → SystemExit（非 ValueError）
        try:
            resolve_generation_pointer(logical)
            _check(False, f"{label} pointer 应被 077① 前置拒绝")
        except ValueError:
            _check(False, f"{label} pointer ValueError 穿透（077 未修）")
        except SystemExit as e:
            _check("077" in str(e) and "GATE-FAIL" in str(e),
                   f"{label} 拒绝消息应带 GATE-FAIL+077：{e!r}")
        # 单格探针 CLI：STATE=invalid rc=2，无裸 traceback
        r = _probe_cli(d, "vt", 3)
        print(f"  077-{label} probe rc={r.returncode} out={r.stdout.strip()[:160]!r}")
        _check(r.returncode == 2,
               f"{label} 单格应 rc=2（STATE=invalid），实际 rc="
               f"{r.returncode} stderr={r.stderr[-300:]!r}")
        _check(r.stdout.startswith("STATE=invalid"),
               f"{label} 单格应 STATE=invalid：{r.stdout!r}")
        _check("GATE-FAIL" in r.stdout and "077" in r.stdout,
               f"{label} REASON 应含 GATE-FAIL+077：{r.stdout!r}")
        _check("Traceback" not in r.stderr,
               f"{label} 不得留裸 traceback（修复前 rc=1）：{r.stderr[-300:]!r}")
    return "PASS"


def test_N2_audit_nul_between_goods_continue(base):
    """077 audit：坏格夹两好格之间 → 逐格继续（字典序后位好格仍枚举 OK）、
    total=3 invalid=1 coverage_errors=0 rc=2；好格单格探针 STATE=complete
    （fixture 自证 + 修复后好格零扰动）。"""
    d = os.path.join(base, "n2")
    os.makedirs(d, exist_ok=True)
    _mk_committed_cell(d, "vt", "111")
    # 好格单格探针自证（修复后零扰动）
    r0 = _probe_cli(d, "vt", 3)
    _check(r0.returncode == 0 and "STATE=complete" in r0.stdout
           and "N=3" in r0.stdout and "SRC=pointer" in r0.stdout,
           f"好格探针应 complete N=3 SRC=pointer：rc={r0.returncode} "
           f"out={r0.stdout!r}")
    # 坏格（155）夹中间 + 好格（222，字典序在坏格之后）
    _write_bad_pointer(d, "vt-stubm-155", b"gen\x00name\n")
    _mk_committed_cell(d, "vt", "222")
    r = _audit_cli(d)
    print(f"  077-audit rc={r.returncode}")
    for ln in r.stdout.splitlines():
        print(f"    | {ln}")
    _check(r.returncode == 2, f"audit 应 rc=2，实际 {r.returncode} "
                             f"stderr={r.stderr[-300:]!r}")
    summ = _audit_summary(r.stdout)
    _check(summ == (3, 1, 0),
           f"audit 应 total=3 invalid=1 coverage_errors=0，实际 {summ}")
    _check(sum(any("vt-stubm-111" in ln and ln.endswith("OK")
                   for ln in r.stdout.splitlines()) for _ in [0]),
           "好格 111 未被枚举 OK")
    _check(any("vt-stubm-222" in ln and ln.endswith("OK")
               for ln in r.stdout.splitlines()),
           "坏格后的好格 222 未继续枚举（077 逐格继续契约失败）")
    _check(any("vt-stubm-155" in ln and "INVALID" in ln
               for ln in r.stdout.splitlines()),
           "坏格 155 未计 invalid")
    _check("Traceback" not in r.stderr, f"audit 不得留裸 traceback：{r.stderr[-300:]!r}")
    return "PASS"


# ================================================================ 078

def test_D1_dangling_only_fake_clean(base):
    """078 GPT 最小复现：root 仅含 dangling symlink → 修复前
    rc=0 total=0 假干净；修复后 coverage_errors=1 rc=2 且
    「零 pointer 产物」注记被抑制。"""
    d = os.path.join(base, "d1")
    os.makedirs(d, exist_ok=True)
    os.symlink("missing-cell", os.path.join(d, "linked-cell"))
    r = _audit_cli(d)
    print(f"  078-d1 rc={r.returncode} out={r.stdout.strip()[:200]!r}")
    _check(r.returncode == 2,
           f"dangling-only audit 应 rc=2（覆盖缺失 fail-closed），"
           f"实际 rc={r.returncode}")
    summ = _audit_summary(r.stdout)
    _check(summ == (0, 0, 1),
           f"应 total=0 invalid=0 coverage_errors=1，实际 {summ}")
    _check("零 pointer 产物" not in r.stdout,
           "覆盖不完整时不得出现「零 pointer 产物」注记（假干净口径）")
    _check(any("linked-cell" in ln and "TRAVERSAL-COVERAGE-ERROR" in ln
               for ln in r.stdout.splitlines()),
           "dangling symlink 未计 coverage error")
    return "PASS"


def test_D2_dangling_alias_between_goods(base):
    """078 好格夹 dangling 目录别名：非指针形 dangling 计 coverage error
    （保留「从有效别名根直接审计」出口），两好格照常 OK、枚举完整。"""
    d = os.path.join(base, "d2")
    os.makedirs(d, exist_ok=True)
    _mk_committed_cell(d, "vt", "111")
    os.symlink("missing-subtree", os.path.join(d, "m-alias"))
    _mk_committed_cell(d, "vt", "222")
    r = _audit_cli(d)
    print(f"  078-d2 rc={r.returncode}")
    for ln in r.stdout.splitlines():
        print(f"    | {ln}")
    _check(r.returncode == 2, f"audit 应 rc=2，实际 {r.returncode}")
    summ = _audit_summary(r.stdout)
    _check(summ == (2, 0, 1),
           f"应 total=2 invalid=0 coverage_errors=1，实际 {summ}")
    _check(any("m-alias" in ln and "TRAVERSAL-COVERAGE-ERROR" in ln
               and "别名根" in ln
               for ln in r.stdout.splitlines()),
           "dangling 别名未计 coverage error 或缺「别名根」出口提示")
    _check(any("vt-stubm-111" in ln and ln.endswith("OK")
               for ln in r.stdout.splitlines()), "好格 111 未枚举 OK")
    _check(any("vt-stubm-222" in ln and ln.endswith("OK")
               for ln in r.stdout.splitlines()), "好格 222 未枚举 OK")
    return "PASS"


def test_D3_pointer_shaped_dangling_invalid(base):
    """078 口径分界：*.tli_gen 名的 dangling 链接走「存在即证据」
    invalid（071① symlink 门禁），不与 coverage 双计；单格探针同判
    STATE=invalid rc=2。"""
    d = os.path.join(base, "d3")
    os.makedirs(d, exist_ok=True)
    os.symlink("missing-target",
               os.path.join(d, "vt-stubm-333.jsonl" + GENERATION_POINTER_SUFFIX))
    r = _audit_cli(d)
    print(f"  078-d3 rc={r.returncode}")
    for ln in r.stdout.splitlines():
        print(f"    | {ln}")
    _check(r.returncode == 2, f"audit 应 rc=2，实际 {r.returncode}")
    summ = _audit_summary(r.stdout)
    _check(summ == (1, 1, 0),
           f"指针形 dangling 应 total=1 invalid=1 coverage_errors=0"
           f"（不双计），实际 {summ}")
    _check(any("vt-stubm-333" in ln and "INVALID" in ln
               for ln in r.stdout.splitlines()),
           "指针形 dangling 未计 invalid")
    r2 = _probe_cli(d, "vt", 3)
    _check(r2.returncode == 2 and r2.stdout.startswith("STATE=invalid"),
           f"单格探针应 STATE=invalid rc=2：rc={r2.returncode} "
           f"out={r2.stdout!r}")
    return "PASS"


# ================================================================ main

def main():
    """067 纪律：PASS/SKIP/FAIL 三分显式计数；异常 → FAIL 继续跑完其余
    用例（一次运行给全红绿图）；FAIL>0 非零退出。E119_ONLY 过滤。"""
    base = tempfile.mkdtemp(prefix="e119_fixes_076_078_")
    plan = [
        ("P1", test_P1_matrix_pairwise_injective),
        ("P2", test_P2_order_independent_and_default_equivalent),
        ("P3", lambda: test_P3_gate_fail_closed_no_overwrite(base)),
        ("P4", test_P4_truncation_keeps_hash),
        ("N1", lambda: test_N1_nul_pointer_single_cell_invalid(base)),
        ("N2", lambda: test_N2_audit_nul_between_goods_continue(base)),
        ("D1", lambda: test_D1_dangling_only_fake_clean(base)),
        ("D2", lambda: test_D2_dangling_alias_between_goods(base)),
        ("D3", lambda: test_D3_pointer_shaped_dangling_invalid(base)),
    ]
    only = os.environ.get("E119_ONLY", "")
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
    print(f"\nE119-FIXES-076/077/078 RESULT: "
          f"PASS={n_pass} FAIL={n_fail} (total {len(plan)})")
    if failed:
        print(f"FAILED: {failed}")
    return 0 if (n_fail == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
