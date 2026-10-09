#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""#195 红绿测试（GPT 2026-10-10 0428 审计三项）：
  TL-E119-YARN-RECEipt-BINDING-059（P1）回执与预测同代绑定
  TL-E119-YARN-RECEIPT-CLOSURE-060（P2）回执完整配置校验
  TL-E119-YARN-CORRECTION-DISCOVERY-061（P2）correction 机器消费

违反事实（GPT 审计已复现）：
  059  pred_ruler.py 直接截断最终预测路径 + 回执在生成循环之前落最终
       旁挂路径且无 SHA/行数/run_id/完成标记；formal 分别冻结「当前预测
       SHA」与「当前回执 SHA」不做同代交叉验证——改预测+改回执仍 exit 0
       标 producer_receipt（「A 进程的预测配 B 进程的回执」可达）；
  060  validate_producer_receipt 只校验开关与 factor——beta_fast=999 /
       seed=999 / 模型与脚本假 hash 均通过；
  061  三份 128K .manifest.yarn_correction.json 零机器消费者，旧 manifest
       仍公开 yarn_factor=2.0 无 provenance。

用例矩阵：
  059 生产侧（e119_yarn_producer_runner_059.py 真实子进程，056 同款——
  stub 只替换 tokenizer/model/GPU 环境桩，锁/临时 generation/SHA/行数/
  完成回执/单次原子提交全部走生产 main()）：
    P1 success 单跑      v2 回执四件（status/basename/SHA/行数）齐备且
                        与预测字节逐位一致；无 tmp/gen 残留；
    P2 crash-mid        生成中途硬杀（持锁）→ 最终路径保持上一代完整
                        产物（SHA 逐位不变）+ partial 只留临时 generation
                        （不进 {task}-*.jsonl glob）+ 消费侧仍接受旧代；
    P3 crash-between    提交点两步之间硬杀 → 新预测配旧回执（SHA 失配）
                        → 消费侧同代绑定校验 fail-closed；
    P4 success/success  栅栏双进程并发同输出 → 锁串行化，终态为某一完整
                        代际（预测与回执同 run_id/SHA 一致）；
    P5 success/crash-mid 组合 → 终态为 success 方完整代际（同代一致）；
    P6 crash-between 后 success → 失配代际被后继成功提交修复为一致；
    P7 lock 互斥        lock-hold 持锁期间 LOCK_NB 探测必失败，释放后
                        可获锁（056 口径跨进程互斥）。
  059 消费侧（score_ruler_formal 真实子进程）：
    C1 v2 全格正例      → 成功：provenance=producer_receipt、逐格
                        prediction_binding.verified_same_generation、
                        config_fingerprint.effective_config_sha256；
    C2 篡改预测不改回执 → 非零退出不发布（python 与 -O 双跑）；
    C3 篡改回执不改预测 → 非零退出不发布；
    C4 同步篡改         审计 059 原复现（改预测 pred 字段 + 改回执非
                        factor 配置）→ 非零退出不发布；
    C5 v1 全格          → 成功但降级 producer_receipt_v1_partial（只证
                        factor 口径不证同代）；
    C6 v1/v2 混装       → 非零退出（不混合证据代际）；
    C7 v2 status!=complete → 非零退出（中断代际拒收）；
    C8 v2 行数失配      → 非零退出；
    C9 basename 失配    → 进程内 validate_producer_receipt 拒绝。
  060 schema/跨格：
    S1 schema 负例      beta_fast=999 / seed=-1 / 缺键 / 坏 hex / 假枚举 /
                        MPE 失配 / rope_type 错 / status!=complete /
                        basename 失配 → 进程内校验全部拒绝；
    S2 跨格负例         单格 seed=999 / 模型 hash 假 / 脚本 hash 假
                        （factor 全同）→ formal 非零退出；seed 例
                        python 与 -O 双跑；
    S3 分组规则         task/method/t 不进指纹（同配置同 hash）、跨档
                        不比较（context_length 进指纹）；formal 双档
                        （yarn off）正例通过。
  061 纠偏消费：
    R1 三份真实 128K manifest → effective=null + not_effective +
                        target/correction 双 SHA 绑定（不脑补 2.0/4.0）；
    R2 无旁挂透传       → manifest 原值 + manifest_declared_no_provenance；
    R3 target hash 失配 → SystemExit fail-closed；
    R4 128K analyzer 接线：三臂全纠偏 → 汇总通过 +
                        identity_gate.per_arm_yarn_identity 逐臂
                        not_effective；单臂纠偏 → 跨臂身份不一致
                        fail-closed（不静默混装两种口径）。

用法：
  PYTHONPATH=$PWD python3 -m benchmark.RULER.test_e119_yarn_binding_059_060_061
"""
import fcntl
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
TESTDATA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "testdata", "e116e")
RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "e119_yarn_producer_runner_059.py")
# 061 R4：128K fixture 现场重建（与 test_e119_crossarm_identity.py 128k 档
# 同一生成器/调用方式）
GEN_SCRIPT = os.path.join(REPO, "exp", "trace", "testdata", "e119_min",
                          "gen_e119_min_fixture.py")
ANALYZER_128K = os.path.join(REPO, "exp", "trace",
                             "analyze_e119_ruler128k_formal.py")

from benchmark.RULER.score_ruler import TASKS  # noqa: E402
from benchmark.RULER.score_ruler_formal import (  # noqa: E402
    _load_producer_yarn_receipt,
)
from benchmark.RULER.yarn_receipt import (  # noqa: E402
    RECEIPT_V1_VERSION, RECEIPT_VERSION, attempt_lock_path,
    build_yarn_receipt, effective_config_sha256, manifest_correction_path_for,
    producer_receipt_path_for, resolve_manifest_yarn_identity,
    validate_producer_receipt, write_yarn_receipt,
)

NATIVE_MPE = 40960
PASS = 0


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _nlines(path):
    return sum(1 for _ in open(path, "rb"))


def _formal(root, out, data_root=None, min_samples=2, extra=(),
            postfix="_fx", optimized=False):
    argv = [sys.executable] + (["-O"] if optimized else []) + \
        ["-u", "-m", "benchmark.RULER.score_ruler_formal",
         "--root", root, "--pred-postfix", postfix,
         "--data-root", data_root or os.path.join(TESTDATA, "data_root"),
         "--out", out, "--min-samples", str(min_samples)]
    argv += list(extra)
    return subprocess.run(argv, capture_output=True, text=True, cwd=REPO,
                          env={**os.environ, "PYTHONPATH": REPO})


def _products(out):
    return [out, out[:-len(".json")] + ".md",
            out + ".manifest.json", out + ".receipt.json"]


def _copy_native_fixture(base, name):
    dst = os.path.join(base, name)
    shutil.copytree(os.path.join(TESTDATA, "pred_root_native"), dst)
    return dst


def _write_receipts(pred_root, context_length, factor, enabled=True,
                    tasks=None, version=RECEIPT_VERSION):
    """给 root 下每个 pred jsonl 写旁挂生产者回执（默认 v2 同代绑定，
    version=RECEIPT_V1_VERSION 写 v1 历史协议）。context_length=None 时
    从文件所在 L{N} 目录自动推导档位（双档正例用，防止跨档回执错配
    057 的目录档位一致性门禁）。返回 [回执路径]。"""
    paths = []
    for f in sorted(glob.glob(os.path.join(
            pred_root, "L*", "pred_*", "*.jsonl"))):
        if os.path.basename(f).endswith("-merged.jsonl"):
            continue
        task = os.path.basename(f).split("-", 1)[0]
        if tasks is not None and task not in tasks:
            continue
        cl = context_length
        if cl is None:
            cl = int(os.path.basename(
                os.path.dirname(os.path.dirname(f)))[1:])
        if enabled:
            eff, scaling = factor, {
                "rope_type": "yarn", "type": "yarn", "factor": factor,
                "original_max_position_embeddings": NATIVE_MPE,
                "beta_fast": 32, "beta_slow": 1,
            }
        else:
            eff, scaling = None, None
        kw = {}
        if version == RECEIPT_VERSION:
            kw = {"run_id": f"fixture-{os.getpid()}-{len(paths)}",
                  "prediction_basename": os.path.basename(f),
                  "prediction_sha256": _sha(f),
                  "prediction_lines": _nlines(f)}
        rcp = build_yarn_receipt(
            yarn_enabled=enabled, effective_factor=eff, yarn_factor_cli=None,
            rope_scaling=scaling, context_length=cl, task=task,
            model_path="/synthetic/Qwen3-8B", model_config_sha256=None,
            native_mpe=NATIVE_MPE,
            generation_params={"max_gen": 64, "max_num": 500, "seed": 42,
                               "method": "tli", "pred_postfix": "_fx",
                               "t": "01010000"},
            producer_script_path="benchmark/RULER/pred_ruler.py",
            producer_script_sha256="0" * 64,
            receipt_version=version, **kw)
        write_yarn_receipt(f, rcp)
        paths.append(producer_receipt_path_for(f))
    return paths


def _rewrite_receipt(rcp_path, mutate):
    """读-改-写单格回执（负例注入用；保持 JSON 可解析）。"""
    rcp = json.load(open(rcp_path, encoding="utf-8"))
    mutate(rcp)
    json.dump(rcp, open(rcp_path, "w", encoding="utf-8"), indent=1,
              ensure_ascii=False)


def _cell_file(root, task):
    return glob.glob(os.path.join(root, "L*", "pred_fx",
                                  f"{task}-*.jsonl"))[0]


def _cell_receipt(root, task):
    return producer_receipt_path_for(_cell_file(root, task))


# ================================================================ 059 生产侧

def _run_producer(base, name, mode, tag, rows=3, crash_after=1, wait=True,
                  barrier=None, hold=2.0, context_length=32768):
    """起一个真实生产子进程（runner），返回 Popen（wait=False 时）。"""
    out_dir = os.path.join(base, name, "out")
    data_root = os.path.join(base, "shared_data")
    argv = [sys.executable, RUNNER, "--out-dir", out_dir,
            "--data-root", data_root, "--mode", mode,
            "--pred-tag", tag, "--rows", str(rows),
            "--crash-after", str(crash_after), "--hold", str(hold),
            "--context-length", str(context_length)]
    if barrier:
        argv += ["--barrier-dir", barrier[0], "--barrier-count",
                 str(barrier[1])]
    p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, cwd=REPO)
    return (p.communicate()[0], p.returncode) if wait else p


def _producer_products(base, name):
    d = os.path.join(base, name, "out", "L32768", "pred_stub")
    pred = os.path.join(d, "vt-stubm-09090909.jsonl")
    return d, pred, producer_receipt_path_for(pred)


def _assert_consistent_generation(tag, pred_path, rcp_path):
    """终态一致性：回执声明 SHA/行数与预测当前字节逐位一致（同代）。"""
    rcp = json.load(open(rcp_path, encoding="utf-8"))
    assert rcp["receipt_version"] == RECEIPT_VERSION, rcp["receipt_version"]
    assert rcp["status"] == "complete"
    assert rcp["prediction_basename"] == os.path.basename(pred_path)
    assert rcp["prediction_sha256"] == _sha(pred_path), \
        f"{tag}: 回执 SHA 与预测字节不一致（不同代）"
    assert rcp["prediction_lines"] == _nlines(pred_path), \
        f"{tag}: 回执行数与预测不一致"
    # 消费侧入口同口径接受（真实 _load_producer_yarn_receipt，非复制品）
    got = _load_producer_yarn_receipt(
        os.path.dirname(pred_path), os.path.basename(pred_path), "vt", 32768)
    assert got is not None and \
        got["prediction_binding"]["verified_same_generation"] is True, got


def test_P1_success_single(base):
    """P1：success 单跑——v2 同代绑定四件齐备 + 无临时残留 + 消费侧接受。"""
    _run_producer(base, "p1", "success", "A")
    d, pred, rcp = _producer_products(base, "p1")
    assert os.path.isfile(pred) and os.path.isfile(rcp)
    _assert_consistent_generation("P1", pred, rcp)
    # 无临时残留（.gen-/.tmp- 均经 os.replace 提交或不存在）
    assert not glob.glob(os.path.join(d, "*.gen-*")), "gen 临时残留"
    assert not glob.glob(os.path.join(d, "*tmp-*")), "tmp 临时残留"
    # 重复成功提交（不同 tag）→ 终态仍同代一致（重跑覆盖一代完整产物）
    _run_producer(base, "p1", "success", "B")
    _assert_consistent_generation("P1-re", pred, rcp)
    print("P1 PASS  success 单跑：v2 回执 status/basename/SHA/行数与预测"
          "逐位同代一致；无 gen/tmp 残留；重跑后仍同代一致")


def test_P2_crash_mid_preserves_old(base):
    """P2：crash-mid（生成中途持锁硬杀）——最终路径保持上一代完整产物
    （SHA 逐位不变），partial 只留临时 generation 且不进 jsonl glob。"""
    _run_producer(base, "p2", "success", "A")          # 第一代
    d, pred, rcp = _producer_products(base, "p2")
    sha_before = _sha(pred)
    rcp_before = _sha(rcp)
    out, rc = _run_producer(base, "p2", "crash-mid", "B",
                            crash_after=1)             # 中途硬杀
    assert rc == 9, (rc, out[-300:])
    assert _sha(pred) == sha_before, "crash-mid 改写了上一代完整预测"
    assert _sha(rcp) == rcp_before, "crash-mid 改写了上一代回执"
    # partial 只在临时 generation（不以 .jsonl 结尾 → 不进 {task}-*.jsonl glob）
    gens = glob.glob(os.path.join(d, "*.gen-*"))
    assert gens, "crash-mid 应留下临时 generation（partial 证据）"
    assert not glob.glob(os.path.join(d, "vt-*.jsonl.gen-*"[:-7] + "*")) \
        or all(not g.endswith(".jsonl") for g in gens)
    assert all(not g.endswith(".jsonl") for g in gens), \
        "临时 generation 不得以 .jsonl 结尾（会污染 best-file glob）"
    jsonl_glob = glob.glob(os.path.join(d, "vt-*.jsonl"))
    assert jsonl_glob == [pred], ("glob 污染", jsonl_glob)
    # 消费侧仍接受旧代（上一代预测+回执同代自洽）
    _assert_consistent_generation("P2", pred, rcp)
    print("P2 PASS  crash-mid：最终路径 SHA 逐位不变；partial 只留 "
          ".gen- 临时文件（不污染 {task}-*.jsonl glob）；旧代仍可消费")


def test_P3_crash_between_mismatch(base):
    """P3：crash-between（提交点两步之间硬杀）——新预测配旧回执 →
    消费侧同代绑定校验 fail-closed（坏态可检，不静默）。"""
    _run_producer(base, "p3", "success", "A")
    d, pred, rcp = _producer_products(base, "p3")
    out, rc = _run_producer(base, "p3", "crash-between", "C", rows=2)
    assert rc == 9, (rc, out[-300:])
    rcp_obj = json.load(open(rcp, encoding="utf-8"))
    assert rcp_obj["prediction_sha256"] != _sha(pred), \
        "crash-between 后应留下新预测配旧回执的失配态"
    try:
        _load_producer_yarn_receipt(os.path.dirname(pred),
                                     os.path.basename(pred), "vt", 32768)
        raise AssertionError("失配代际未被消费侧拒收（059 绑定失效）")
    except SystemExit as e:
        assert "prediction_sha256" in str(e) and "不同代" in str(e), e
    print("P3 PASS  crash-between：新预测配旧回执 → 消费侧同代绑定校验"
          "fail-closed 拒收")


def test_P4_dual_success(base):
    """P4：success/success 栅栏双进程并发同输出——锁串行化，终态为某一
    完整代际（预测与回执同 run_id/SHA 一致），无混合代际。"""
    barrier = os.path.join(base, "p4_bar")
    pa = _run_producer(base, "p4", "success", "A", wait=False,
                       barrier=(barrier, 2))
    pb = _run_producer(base, "p4", "success", "B", wait=False,
                       barrier=(barrier, 2))
    oa = pa.communicate()[0]
    ra = pa.returncode
    ob = pb.communicate()[0]
    rb = pb.returncode
    assert ra == 0 and rb == 0, (ra, oa[-400:], rb, ob[-400:])
    d, pred, rcp = _producer_products(base, "p4")
    _assert_consistent_generation("P4", pred, rcp)
    rcp_obj = json.load(open(rcp, encoding="utf-8"))
    # 终态属于其中一代（pred 内容与回执 run_id 同代），而非 A 预测配 B 回执
    with open(pred, encoding="utf-8") as f:
        first_tag = json.loads(f.readline())["pred"].split("-")[2]
    assert rcp_obj["run_id"].startswith("20"), rcp_obj["run_id"]
    assert f"stub-pred-{first_tag}-" in open(pred, encoding="utf-8").read()
    print("P4 PASS  success/success 双进程并发：锁串行化提交，终态为单一"
          "完整代际（同 run_id 同代），无混合代际")


def test_P5_success_crash_mix(base):
    """P5：success/crash-mid 并发组合——crash 方不触碰最终路径，终态为
    success 方完整代际。"""
    barrier = os.path.join(base, "p5_bar")
    pa = _run_producer(base, "p5", "success", "A", rows=3, wait=False,
                       barrier=(barrier, 2))
    pb = _run_producer(base, "p5", "crash-mid", "B", rows=3, crash_after=1,
                       wait=False, barrier=(barrier, 2))
    oa = pa.communicate()[0]
    ra = pa.returncode
    ob = pb.communicate()[0]
    rb = pb.returncode
    assert ra == 0, (ra, oa[-400:])
    assert rb == 9, (rb, ob[-400:])
    d, pred, rcp = _producer_products(base, "p5")
    _assert_consistent_generation("P5", pred, rcp)
    print("P5 PASS  success/crash-mid 并发：终态为 success 方完整代际"
          "（crash 方 partial 只留临时 generation）")


def test_P6_crash_between_then_success(base):
    """P6：crash-between 失配态被后继成功提交修复——终态恢复同代一致。"""
    _run_producer(base, "p6", "success", "A")
    _run_producer(base, "p6", "crash-between", "C", rows=2)
    d, pred, rcp = _producer_products(base, "p6")
    rcp_obj = json.load(open(rcp, encoding="utf-8"))
    assert rcp_obj["prediction_sha256"] != _sha(pred), "前置：应处失配态"
    _run_producer(base, "p6", "success", "D")
    _assert_consistent_generation("P6", pred, rcp)
    print("P6 PASS  crash-between 失配态 → 后继 success 提交修复为同代一致")


def test_P7_lock_mutual_exclusion(base):
    """P7：lock-hold 持锁期间 LOCK_NB 非阻塞探测必失败；释放后可获锁
    （056 口径跨进程互斥，崩溃内核自动释放）。"""
    _run_producer(base, "p7", "success", "A")   # 确保输出目录存在
    d, pred, rcp = _producer_products(base, "p7")
    lock_path = attempt_lock_path(pred)
    holder = _run_producer(base, "p7", "lock-hold", "H", wait=False,
                           hold=5.0)
    # 轮询 holder stdout 直到其报告「已获锁」（import torch 耗时抖动大，
    # 固定 sleep 不可靠——读到消息后再探测，保证探测窗口落在持锁期内）
    got_lock_msg = False
    deadline = time.time() + 60
    while time.time() < deadline:
        line = holder.stdout.readline()
        if not line:
            break            # 进程退出/管道关闭
        if "已获锁" in line:
            got_lock_msg = True
            break
    assert got_lock_msg, "holder 未在 60s 内报告已获锁"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        blocked = False
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            blocked = True
        assert blocked, "持锁期间 LOCK_NB 探测不应成功（互斥失效）"
    finally:
        os.close(fd)
    holder.communicate()
    # 释放后可获锁
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
    print("P7 PASS  lock-hold 持锁期 LOCK_NB 探测失败、释放后可获锁"
          "（056 口径跨进程互斥）")


# ================================================================ 059 消费侧

def test_C1_v2_positive(base):
    """C1：11 格 v2 同代回执 → formal 成功，逐格绑定 verified +
    config_fingerprint 冻结进 manifest。"""
    root = _copy_native_fixture(base, "c1_root")
    _write_receipts(root, 32768, 2.0)
    out = os.path.join(base, "c1.json")
    r = _formal(root, out, extra=("--yarn",))
    assert r.returncode == 0 and "DONE" in r.stdout, \
        (r.returncode, r.stdout[-3000:], r.stderr[-2000:])
    mf = json.load(open(out + ".manifest.json"))
    ri = mf["run_identity"]
    assert ri["yarn_factor_provenance"] == "producer_receipt", ri
    pe = ri["producer_evidence"]
    assert pe["protocol"] == RECEIPT_VERSION and \
        pe["same_generation_bound"] is True and \
        pe["config_consistency_enforced"] is True, pe
    for cell in mf["cells"].values():
        for t, ti in cell["tasks"].items():
            rc = ti["producer_yarn_receipt"]
            assert rc["prediction_binding"] is not None and \
                rc["prediction_binding"]["verified_same_generation"] \
                is True, (t, rc)
            assert "effective_config_sha256" in rc["config_fingerprint"]
    print("C1 PASS  11 格 v2 回执 → producer_receipt + same_generation_"
          "bound + 逐格 prediction_binding.verified + 配置指纹冻结")


def _expect_formal_reject(base, tag, root, out, extra=("--yarn",),
                           optimized=False, needle=None):
    r = _formal(root, out, extra=extra, optimized=optimized)
    assert r.returncode != 0, \
        f"{tag}: 仍 exit=0（059/060 门禁失效）：\n{r.stdout[-2000:]}"
    blob = r.stdout + r.stderr
    assert "GATE-FAIL" in blob, (tag, blob[-1500:])
    if needle is not None:
        assert needle in blob, (tag, needle, blob[-1500:])
    assert not any(os.path.exists(p) for p in _products(out)), \
        f"{tag}: 失败仍发布了产物"
    assert glob.glob(out + ".failure-*.json")
    mode = "python -O " if optimized else ""
    print(f"  {tag} PASS  {mode}exit={r.returncode} 不发布；"
          f"拒绝: {blob.strip().splitlines()[-1][:110]}")


def test_C2_tamper_pred_only(base):
    """C2：篡改预测（JSON 行内 pred 字段）不改回执 → fail（python 与 -O）。"""
    for optimized in (False, True):
        root = _copy_native_fixture(
            base, f"c2_root_{'O' if optimized else 'py'}")
        _write_receipts(root, 32768, 2.0)
        tgt = _cell_file(root, "vt")
        rows = [json.loads(l) for l in open(tgt, encoding="utf-8")]
        rows[0]["pred"] = "tampered-pred"
        with open(tgt, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        out = os.path.join(base, "c2.json")
        _expect_formal_reject(base, "C2", root, out, optimized=optimized,
                              needle="不同代")
    print("C2 PASS  篡改预测不改回执 → 非零退出不发布（python 与 -O "
          "双跑，同代绑定门禁非 assert）")


def test_C3_tamper_receipt_only(base):
    """C3：篡改回执（prediction_sha256 换成合法 hex 假值）不改预测 → fail。"""
    root = _copy_native_fixture(base, "c3_root")
    _write_receipts(root, 32768, 2.0)
    _rewrite_receipt(_cell_receipt(root, "vt"),
                     lambda r: r.__setitem__("prediction_sha256", "f" * 64))
    out = os.path.join(base, "c3.json")
    _expect_formal_reject(base, "C3", root, out, needle="不同代")
    print("C3 PASS  篡改回执不改预测 → 非零退出不发布（回执声明 SHA 与"
          "预测字节失配）")


def test_C4_synchronized_tamper(base):
    """C4：审计 059 原复现——同时改预测 pred 字段 + 改回执非 factor 配置
    → 仍 fail（v1 时代可绕过，v2 同代绑定闭合后不可达）。"""
    root = _copy_native_fixture(base, "c4_root")
    _write_receipts(root, 32768, 2.0)
    tgt = _cell_file(root, "vt")
    rows = [json.loads(l) for l in open(tgt, encoding="utf-8")]
    rows[0]["pred"] = "tampered-pred"
    with open(tgt, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    _rewrite_receipt(_cell_receipt(root, "vt"),
                     lambda r: r["generation_params"].__setitem__(
                         "seed", 999))
    out = os.path.join(base, "c4.json")
    _expect_formal_reject(base, "C4", root, out, needle="不同代")
    print("C4 PASS  同步篡改（预测+回执非 factor 配置）→ 非零退出不发布"
          "（审计 059 复现路径闭合）")


def test_C5_v1_downgrade(base):
    """C5：11 格 v1（057 时代协议）→ formal 成功但降级
    producer_receipt_v1_partial——只证 factor 口径不证同代，不冒充。"""
    root = _copy_native_fixture(base, "c5_root")
    _write_receipts(root, 32768, 2.0, version=RECEIPT_V1_VERSION)
    out = os.path.join(base, "c5.json")
    r = _formal(root, out, extra=("--yarn",))
    assert r.returncode == 0 and "DONE" in r.stdout, \
        (r.returncode, r.stdout[-3000:], r.stderr[-2000:])
    mf = json.load(open(out + ".manifest.json"))
    ri = mf["run_identity"]
    assert ri["yarn_factor"] == 2.0 and \
        ri["yarn_factor_provenance"] == "producer_receipt_v1_partial", ri
    pe = ri["producer_evidence"]
    assert pe["protocol"] == RECEIPT_V1_VERSION and \
        pe["same_generation_bound"] is False and \
        "只证 factor 口径" in pe["note"], pe
    for cell in mf["cells"].values():
        for ti in cell["tasks"].values():
            assert ti["producer_yarn_receipt"]["prediction_binding"] is None
    print("C5 PASS  v1 全格 → 成功但 producer_receipt_v1_partial 降级"
          "（只证 factor 口径不证同代；逐格 prediction_binding=null）")


def test_C6_mixed_versions(base):
    """C6：v1/v2 混装 → 非零退出（单次 run_identity 不混合证据代际）。"""
    root = _copy_native_fixture(base, "c6_root")
    _write_receipts(root, 32768, 2.0)
    _write_receipts(root, 32768, 2.0, tasks={"vt"},
                    version=RECEIPT_V1_VERSION)
    out = os.path.join(base, "c6.json")
    _expect_formal_reject(base, "C6", root, out, needle="协议代际混合")
    print("C6 PASS  v1/v2 混装 → 非零退出不发布")


def test_C7_status_not_complete(base):
    """C7：v2 回执 status=partial（中断代际）→ 非零退出。"""
    root = _copy_native_fixture(base, "c7_root")
    _write_receipts(root, 32768, 2.0)
    _rewrite_receipt(_cell_receipt(root, "vt"),
                     lambda r: r.__setitem__("status", "partial"))
    out = os.path.join(base, "c7.json")
    _expect_formal_reject(base, "C7", root, out, needle="complete")
    print("C7 PASS  status=partial（无完成标记）→ 非零退出不发布")


def test_C8_lines_mismatch(base):
    """C8：v2 回执 prediction_lines 与实际行数失配（SHA 正确）→ 非零退出。"""
    root = _copy_native_fixture(base, "c8_root")
    _write_receipts(root, 32768, 2.0)
    _rewrite_receipt(_cell_receipt(root, "vt"),
                     lambda r: r.__setitem__("prediction_lines", 999))
    out = os.path.join(base, "c8.json")
    _expect_formal_reject(base, "C8", root, out, needle="行数")
    print("C8 PASS  prediction_lines 失配 → 非零退出不发布")


def test_C9_basename_mismatch():
    """C9：v2 回执 prediction_basename 与产物文件名失配 → 进程内校验拒绝
    （消费侧在 formal 探查前拒收）。"""
    pred = "/syn/pred_root/L32768/pred_fx/vt-fxm-01010000.jsonl"
    rcp = build_yarn_receipt(
        yarn_enabled=True, effective_factor=2.0, yarn_factor_cli=None,
        rope_scaling={"rope_type": "yarn", "type": "yarn", "factor": 2.0,
                      "original_max_position_embeddings": NATIVE_MPE,
                      "beta_fast": 32, "beta_slow": 1},
        context_length=32768, task="vt", model_path="/synthetic/Qwen3-8B",
        model_config_sha256=None, native_mpe=NATIVE_MPE,
        generation_params={"max_gen": 64, "max_num": 500, "seed": 42,
                           "method": "tli", "pred_postfix": "_fx",
                           "t": "01010000"},
        producer_script_path="benchmark/RULER/pred_ruler.py",
        producer_script_sha256="0" * 64,
        run_id="x", prediction_basename="cwe-fxm-01010000.jsonl",
        prediction_sha256="0" * 64, prediction_lines=2)
    err = validate_producer_receipt(rcp, pred)
    assert err is not None and "prediction_basename" in err, err
    print("C9 PASS  prediction_basename 失配 → 校验拒绝")


# ================================================================ 060

def test_S1_schema_negatives():
    """S1：严格 schema 负例九连——beta_fast=999 / seed=-1 / 缺键 / 坏 hex /
    假枚举 / MPE 失配 / rope_type 错 / 行数非正 / status!=complete。"""
    pred = "/syn/pred_root/L32768/pred_fx/vt-fxm-01010000.jsonl"

    def good():
        return build_yarn_receipt(
            yarn_enabled=True, effective_factor=2.0, yarn_factor_cli=None,
            rope_scaling={"rope_type": "yarn", "type": "yarn",
                          "factor": 2.0,
                          "original_max_position_embeddings": NATIVE_MPE,
                          "beta_fast": 32, "beta_slow": 1},
            context_length=32768, task="vt",
            model_path="/synthetic/Qwen3-8B", model_config_sha256=None,
            native_mpe=NATIVE_MPE,
            generation_params={"max_gen": 64, "max_num": 500, "seed": 42,
                               "method": "tli", "pred_postfix": "_fx",
                               "t": "01010000"},
            producer_script_path="benchmark/RULER/pred_ruler.py",
            producer_script_sha256="0" * 64, run_id="x",
            prediction_basename=os.path.basename(pred),
            prediction_sha256="0" * 64, prediction_lines=2)

    assert validate_producer_receipt(good(), pred) is None  # 正例基线
    cases = []

    def case(name, mutate):
        r = good()
        mutate(r)
        err = validate_producer_receipt(r, pred)
        assert err is not None, f"{name}: 未被拒绝（060 schema 漏洞）"
        cases.append(name)

    case("beta_fast=999",
         lambda r: r["rope_scaling"].__setitem__("beta_fast", 999))
    case("seed=-1",
         lambda r: r["generation_params"].__setitem__("seed", -1))
    case("seed 缺键",
         lambda r: r["generation_params"].pop("seed"))
    case("model hash 坏 hex",
         lambda r: r.__setitem__("model_config_sha256", "xyz"))
    case("脚本 hash 短 hex",
         lambda r: r["producer_script"].__setitem__("sha256", "b" * 63))
    case("source 假枚举",
         lambda r: r.__setitem__("yarn_factor_source", "bogus"))
    case("MPE 失配",
         lambda r: r["rope_scaling"].__setitem__(
             "original_max_position_embeddings", 1024))
    case("rope_type 错",
         lambda r: r["rope_scaling"].__setitem__("rope_type", "linear"))
    case("status!=complete",
         lambda r: r.__setitem__("status", "partial"))
    case("prediction_lines=0",
         lambda r: r.__setitem__("prediction_lines", 0))
    case("未知协议版本",
         lambda r: r.__setitem__("receipt_version", "v3"))
    print(f"S1 PASS  schema 负例 {len(cases)} 连全拒收"
          f"（{'/'.join(cases)}）；正例基线通过")


def test_S2_cross_cell_negatives(base):
    """S2：跨格配置负例——单格 seed=999 / 模型 hash 假 / 脚本 hash 假
    （factor 全同 2.0）→ formal 非零退出；seed 例 python 与 -O 双跑。"""
    # ① seed=999（python 与 -O）
    for optimized in (False, True):
        root = _copy_native_fixture(
            base, f"s2a_root_{'O' if optimized else 'py'}")
        _write_receipts(root, 32768, 2.0)
        _rewrite_receipt(_cell_receipt(root, "vt"),
                         lambda r: r["generation_params"].__setitem__(
                             "seed", 999))
        out = os.path.join(base, "s2a.json")
        _expect_formal_reject(base, "S2-seed", root, out,
                              optimized=optimized, needle="seed")
    # ② 模型 config hash 假（64 位合法 hex，跨格失配）
    root = _copy_native_fixture(base, "s2b_root")
    _write_receipts(root, 32768, 2.0)
    _rewrite_receipt(_cell_receipt(root, "vt"),
                     lambda r: r.__setitem__("model_config_sha256",
                                             "a" * 64))
    out = os.path.join(base, "s2b.json")
    _expect_formal_reject(base, "S2-model-hash", root, out,
                          needle="model_config_sha256")
    # ③ 生产脚本 hash 假
    root = _copy_native_fixture(base, "s2c_root")
    _write_receipts(root, 32768, 2.0)
    _rewrite_receipt(_cell_receipt(root, "vt"),
                     lambda r: r["producer_script"].__setitem__(
                         "sha256", "b" * 64))
    out = os.path.join(base, "s2c.json")
    _expect_formal_reject(base, "S2-script-hash", root, out,
                          needle="producer_script")
    print("S2 PASS  factor 相同但 seed=999 / 模型 hash 假 / 脚本 hash 假"
          "→ 全部非零退出不发布（060 跨格门禁；seed 例 python 与 -O 双跑）")


def test_S3_grouping_rules(base):
    """S3：分组规则——task/method/t 不进指纹（同配置同 hash）；跨档
    （context_length）不比较；formal 双档 yarn off 正例通过。"""
    def receipt_for(task, context_length, seed=42, t="01010000",
                    method="tli"):
        return build_yarn_receipt(
            yarn_enabled=True, effective_factor=2.0, yarn_factor_cli=None,
            rope_scaling={"rope_type": "yarn", "type": "yarn",
                          "factor": 2.0,
                          "original_max_position_embeddings": NATIVE_MPE,
                          "beta_fast": 32, "beta_slow": 1},
            context_length=context_length, task=task,
            model_path="/synthetic/Qwen3-8B", model_config_sha256=None,
            native_mpe=NATIVE_MPE,
            generation_params={"max_gen": 64, "max_num": 500, "seed": seed,
                               "method": method, "pred_postfix": "_fx",
                               "t": t},
            producer_script_path="benchmark/RULER/pred_ruler.py",
            producer_script_sha256="0" * 64, run_id="x",
            prediction_basename=f"{task}-fxm-{t}.jsonl",
            prediction_sha256="0" * 64, prediction_lines=2)

    # 分组自由字段（task/method/t）不进指纹 → 同 hash
    h1 = effective_config_sha256(receipt_for("vt", 32768))
    h2 = effective_config_sha256(receipt_for("cwe", 32768))
    h3 = effective_config_sha256(
        receipt_for("vt", 32768, method="quest", t="09090909"))
    assert h1 == h2 == h3, "task/method/t 不得进配置指纹（分组自由字段）"
    # 跨档不比较（context_length 进指纹 → 跨档 hash 不同属预期）
    h4 = effective_config_sha256(receipt_for("vt", 65536))
    assert h4 != h1
    # 应同字段漂移（seed）→ hash 变化
    h5 = effective_config_sha256(receipt_for("vt", 32768, seed=999))
    assert h5 != h1
    # formal 双档正例：yarn off（factor=None 全局统一）+ 每档内配置
    # 一致 → 按档分组比较通过
    root = _copy_native_fixture(base, "s3_root")
    l64 = os.path.join(root, "L65536")
    shutil.copytree(os.path.join(root, "L32768"), l64)
    data_root = os.path.join(base, "s3_data")
    shutil.copytree(os.path.join(TESTDATA, "data_root", "32768"),
                    os.path.join(data_root, "32768"))
    shutil.copytree(os.path.join(TESTDATA, "data_root", "32768"),
                    os.path.join(data_root, "65536"))
    _write_receipts(root, None, None, enabled=False)
    out = os.path.join(base, "s3.json")
    r = _formal(root, out, data_root=data_root)
    assert r.returncode == 0 and "DONE" in r.stdout, \
        (r.returncode, r.stdout[-3000:], r.stderr[-2000:])
    mf = json.load(open(out + ".manifest.json"))
    assert mf["run_identity"]["yarn_factor_provenance"] == \
        "producer_receipt", mf["run_identity"]["yarn_factor_provenance"]
    print("S3 PASS  task/method/t 不进指纹；跨档不比较；双档 yarn off "
          "formal 正例通过（按 context_length 分组的一致性门禁）")


# ================================================================ 061

def test_R1_real_128k_corrections():
    """R1：三份真实 128K manifest 旁挂纠偏 → 解析入口返回
    effective=null + operator_declared_not_effective + 双 SHA 绑定
    （不得是 2.0，不脑补 4.0）。"""
    results = os.path.join(REPO, "exp", "trace", "results")
    for arm in ("mavg", "fullkv", "aavg"):
        mp = os.path.join(results,
                          f"e119_ruler128k_formal_{arm}.json.manifest.json")
        res = resolve_manifest_yarn_identity(mp)
        assert res["yarn_factor_provenance"] == \
            "operator_declared_not_effective", (arm, res)
        assert res["effective_yarn_factor"] is None, (arm, res)
        assert res["manifest_run_identity_yarn_factor"] == 2.0, (arm, res)
        c = res["correction"]
        assert c is not None and \
            c["correction_version"] == "yarn-identity-057-v1" and \
            c["target_manifest_sha256"] == _sha(mp) and \
            c["correction_sha256"] == _sha(c["correction_path"]), (arm, c)
    print("R1 PASS  三份真实 128K manifest → effective=null + "
          "operator_declared_not_effective + target/correction 双 SHA "
          "绑定（旧 2.0 不再当 effective，不脑补 4.0）")


def test_R2_passthrough_no_correction(base):
    """R2：无旁挂纠偏 → manifest 原值透传（legacy 无 provenance 字段 →
    manifest_declared_no_provenance 如实标注）。"""
    results = os.path.join(REPO, "exp", "trace", "results")
    src = os.path.join(results, "e119_ruler128k_formal_mavg.json"
                                    ".manifest.json")
    dst = os.path.join(base, "r2_manifest.json")
    shutil.copyfile(src, dst)   # 副本同字节；纠偏 sidecar 不随行
    res = resolve_manifest_yarn_identity(dst)
    assert res["correction"] is None and \
        res["effective_yarn_factor"] == 2.0 and \
        res["yarn_factor_provenance"] == "manifest_declared_no_provenance", \
        res
    print("R2 PASS  无旁挂纠偏 → manifest 原值透传 + "
          "manifest_declared_no_provenance 如实标注")


def test_R3_target_hash_mismatch(base):
    """R3：纠偏 target_manifest_sha256 与 manifest 当前字节失配 →
    SystemExit fail-closed（manifest 已被重发布，旧纠偏不可再消费）。"""
    results = os.path.join(REPO, "exp", "trace", "results")
    src_m = os.path.join(results, "e119_ruler128k_formal_mavg.json"
                                     ".manifest.json")
    src_c = manifest_correction_path_for(src_m)
    dst_m = os.path.join(base, "r3_manifest.json")
    shutil.copyfile(src_m, dst_m)
    shutil.copyfile(src_c, manifest_correction_path_for(dst_m))
    # manifest 字节被重发布（追加空白）→ 旧纠偏 target hash 失配
    with open(dst_m, "a", encoding="utf-8") as f:
        f.write("\n")
    try:
        resolve_manifest_yarn_identity(dst_m)
        raise AssertionError("target hash 失配未被拒收（061 解析失效）")
    except SystemExit as e:
        assert "target_manifest_sha256" in str(e), e
    print("R3 PASS  target hash 失配 → SystemExit fail-closed")


def _write_correction(manifest_path, version="yarn-identity-057-v1"):
    """为 manifest 落一份最小合法纠偏 sidecar（target hash 绑定当前字节）。"""
    correction = {
        "correction_version": version,
        "target_manifest": os.path.basename(manifest_path),
        "target_manifest_sha256": _sha(manifest_path),
        "correction": {
            "yarn_factor_status": "operator_declared_not_effective",
            "effective_yarn_factor": None,
        },
    }
    cpath = manifest_correction_path_for(manifest_path)
    json.dump(correction, open(cpath, "w", encoding="utf-8"), indent=1,
              ensure_ascii=False)
    return cpath


def test_R4_analyzer_wiring(base):
    """R4：128K analyzer 纠偏消费接线——
    ① 三臂全纠偏 → 汇总通过 + identity_gate.per_arm_yarn_identity 逐臂
       operator_declared_not_effective（effective=null 进跨臂身份，三臂
       一致）；
    ② 单臂纠偏 → 跨臂身份不一致（null vs 2.0）→ fail-closed 不静默。"""
    fixture = os.path.join(base, "r4_fixture")
    r = subprocess.run([sys.executable, GEN_SCRIPT, fixture,
                        "--tier", "128k"], capture_output=True, text=True)
    assert r.returncode == 0, (r.returncode, r.stdout[-500:], r.stderr[-500:])
    # ① 三臂全纠偏
    b1 = os.path.join(base, "r4_all")
    shutil.copytree(os.path.join(fixture, "results"),
                    os.path.join(b1, "results"))
    shutil.copytree(os.path.join(fixture, "pred_root"),
                    os.path.join(b1, "pred_root"))
    for arm in ("mavg", "fullkv", "aavg"):
        _write_correction(os.path.join(
            b1, "results", f"e119_ruler128k_formal_{arm}.json"
                           ".manifest.json"))
    r1 = subprocess.run(
        [sys.executable, ANALYZER_128K,
         "--results-dir", os.path.join(b1, "results"),
         "--pred-root", os.path.join(b1, "pred_root")],
        capture_output=True, text=True, cwd=REPO)
    assert r1.returncode == 0, (r1.returncode, r1.stdout[-1500:],
                                r1.stderr[-1000:])
    summary = json.load(open(os.path.join(
        b1, "results", "e119_ruler128k_formal_summary.json")))
    per_yarn = summary["identity_gate"]["per_arm_yarn_identity"]
    assert set(per_yarn) == {"mavg", "FullKV", "aavg"}, per_yarn
    for arm, y in per_yarn.items():
        assert y["yarn_factor_provenance"] == \
            "operator_declared_not_effective", (arm, y)
        assert y["effective_yarn_factor"] is None and \
            y["correction"]["target_manifest_sha256"] is not None, (arm, y)
    # 跨臂身份 digest 在三臂同 null 下仍一致（可比性保持）
    # ② 单臂纠偏 → 跨臂身份不一致 fail-closed
    b2 = os.path.join(base, "r4_one")
    shutil.copytree(os.path.join(fixture, "results"),
                    os.path.join(b2, "results"))
    shutil.copytree(os.path.join(fixture, "pred_root"),
                    os.path.join(b2, "pred_root"))
    _write_correction(os.path.join(
        b2, "results", "e119_ruler128k_formal_mavg.json.manifest.json"))
    r2 = subprocess.run(
        [sys.executable, ANALYZER_128K,
         "--results-dir", os.path.join(b2, "results"),
         "--pred-root", os.path.join(b2, "pred_root")],
        capture_output=True, text=True, cwd=REPO)
    assert r2.returncode != 0, "单臂纠改应被跨臂身份门禁拒绝"
    assert "yarn_factor" in (r2.stdout + r2.stderr), \
        (r2.stdout[-800:], r2.stderr[-800:])
    print("R4 PASS  128K analyzer 接线：三臂全纠偏 → 通过 + 逐臂 "
          "not_effective（null 进跨臂身份）；单臂纠偏 → 跨臂身份不一致"
          " fail-closed")


# ================================================================ main

def main():
    global PASS
    base = tempfile.mkdtemp(prefix="e119_binding_195_")
    n = 0
    plan = [
        ("P1", lambda: test_P1_success_single(base)),
        ("P2", lambda: test_P2_crash_mid_preserves_old(base)),
        ("P3", lambda: test_P3_crash_between_mismatch(base)),
        ("P4", lambda: test_P4_dual_success(base)),
        ("P5", lambda: test_P5_success_crash_mix(base)),
        ("P6", lambda: test_P6_crash_between_then_success(base)),
        ("P7", lambda: test_P7_lock_mutual_exclusion(base)),
        ("C1", lambda: test_C1_v2_positive(base)),
        ("C2", lambda: test_C2_tamper_pred_only(base)),
        ("C3", lambda: test_C3_tamper_receipt_only(base)),
        ("C4", lambda: test_C4_synchronized_tamper(base)),
        ("C5", lambda: test_C5_v1_downgrade(base)),
        ("C6", lambda: test_C6_mixed_versions(base)),
        ("C7", lambda: test_C7_status_not_complete(base)),
        ("C8", lambda: test_C8_lines_mismatch(base)),
        ("C9", test_C9_basename_mismatch),
        ("S1", test_S1_schema_negatives),
        ("S2", lambda: test_S2_cross_cell_negatives(base)),
        ("S3", lambda: test_S3_grouping_rules(base)),
        ("R1", test_R1_real_128k_corrections),
        ("R2", lambda: test_R2_passthrough_no_correction(base)),
        ("R3", lambda: test_R3_target_hash_mismatch(base)),
        ("R4", lambda: test_R4_analyzer_wiring(base)),
    ]
    try:
        for name, fn in plan:
            fn()
            n += 1
        PASS = n
    finally:
        shutil.rmtree(base, ignore_errors=True)
    print(f"\nE119-YARN-BINDING-059/060/061 ALL PASS ({PASS}/{len(plan)})")


if __name__ == "__main__":
    main()
