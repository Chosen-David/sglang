#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""#195 红绿测试（GPT 2026-10-10 0428 审计三项）：
  TL-E119-YARN-RECEipt-BINDING-059（P1）回执与预测同代绑定
  TL-E119-YARN-RECEIPT-CLOSURE-060（P2）回执完整配置校验
  TL-E119-YARN-CORRECTION-DISCOVERY-061（P2）correction 机器消费
#196 增量红绿测试（GPT 2026-10-10 0635 二轮复审四项）：
  TL-E119-YARN-SNAPSHOT-RACE-062（P1）staging 快照与源目录验证的时序竞态
  TL-E119-YARN-CORRECTION-SCHEMA-063（P2）纠偏 sidecar schema 太弱
  TL-E119-YARN-CONFIG-PARTIAL-064（P2）config 闭包两处残缺
  TL-E119-YARN-TEST-ORACLE-065（P1）测试断言 -O 失效
#197 增量红绿测试（GPT 2026-10-10 0834 审计）：
  TL-E119-YARN-SAME-BYTES-PROVENANCE-066（P2）三方 SHA 一致 ≠ 运行同代

违反事实（GPT 审计已复现）：
  059  pred_ruler.py 直接截断最终预测路径 + 回执在生成循环之前落最终
       旁挂路径且无 SHA/行数/run_id/完成标记；formal 分别冻结「当前预测
       SHA」与「当前回执 SHA」不做同代交叉验证——改预测+改回执仍 exit 0
       标 producer_receipt（「A 进程的预测配 B 进程的回执」可达）；
  060  validate_producer_receipt 只校验开关与 factor——beta_fast=999 /
       seed=999 / 模型与脚本假 hash 均通过；
  061  三份 128K .manifest.yarn_correction.json 零机器消费者，旧 manifest
       仍公开 yarn_factor=2.0 无 provenance；
  062  formal 先复制源预测到 staging（best-file 仲裁用 staging），之后才
       从可变源目录验证源文件与回执，且无「staging SHA == receipt SHA」
       断言——生产者在复制后、验证前提交 B 代 → formal 评 staging A、
       manifest 标 B 的 source SHA 与回执 verified=true（CPU 复现：
       staged_derived_sha256 != receipt_prediction_sha256 但 freeze 成功）；
  063  纠偏消费者只要求 dict + 正确 target SHA + 任意非空
       correction_version——缺 correction 主体的两字段 sidecar 也被解释
       成有效纠偏；
  064  v2 schema 允许 model_config_sha256=None（「完整模型配置闭包」不
       成立）+ effective_config_sha256 漏掉必需键 generation_params.
       max_num（max_num=1 与 100 同指纹 a09b4ef6...）；
  065  本套件含 62 个 AST assert 节点——python -O 删除全部判定，validator
       被破坏后 -O 仍 exit 0 打印 PASS（门禁失效）；
  066  三方 SHA 一致只证「内容等价」不证「运行同代」：run-A（seed=42/
       max_num=1）与 run-B（seed=99/max_num=100）可产出字节完全相同的
       预测 JSONL——062 门禁全绿，formal 仍把冻结 A 代的 manifest 标上
       B 代的 run_id/seed/config（066 CPU 复现：verified_same_generation=
       true 但身份字段错配）。

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
  062 快照竞态 barrier（进程内真实 freeze_and_stage + 注入回调模拟生产者
    在精确窗口原子提交 B 代）：
    B1 复制后/读回执前提交 B → staging=A、receipt=B → 必须 fail-closed
                        拒绝（不得 verified=true 发布混合代际）；
    B2 回执快照后/源校验前提交 B → 三方一致门禁（staging==receipt==源）
                        必须拒绝；两例均先跑无注入正例（三方一致自洽）。
  066 同字节异代 provenance 锁窗口（#197：formal 与生产者复用同一把
    realpath 输出路径 flock，冻结全程锁内，生产者两步提交不可穿插）：
    B3 same-bytes 互斥  run-A（seed=42/max_num=1）冻结进行中，真子进程
                        在 best-file 仲裁点提交字节全同的 B 代（seed=99/
                        max_num=100，同键 flock）→ 子进程必须被锁阻塞
                        （done 标记在冻结窗口内不出现）→ 冻结身份保持
                        A 代（run-A/seed=42），manifest 记录 freeze_lock
                        审计字段；锁释放后 B 代完整落盘（同代自洽）；
    B4 锁不可获 fail-closed 源目录只读 → 锁获取失败必须 [GATE-FAIL]
                        非零退出拒收（不静默降级）；root/网络 FS 无法
                        模拟只读时显式 SKIP（不虚报）。
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
  064 配置闭包：
    S4 max_num 指纹负例  max_num=1 vs 100 两份合法回执指纹必须不同
                        （修复前同指纹）；单格 max_num=999 → formal
                        跨格门禁非零退出；model_config_sha256=None →
                        manifest config_identity=missing +
                        model_config_closure=false 降级标注，有值 →
                        bound/true（不虚报不漏报）。
  061 纠偏消费：
    R1 三份真实 128K manifest → effective=null + not_effective +
                        target/correction 双 SHA 绑定（不脑补 2.0/4.0）；
    R2 无旁挂透传       → manifest 原值 + manifest_declared_no_provenance；
    R3 target hash 失配 → SystemExit fail-closed；
    R4 128K analyzer 接线：三臂全纠偏 → 汇总通过 +
                        identity_gate.per_arm_yarn_identity 逐臂
                        not_effective；单臂纠偏 → 跨臂身份不一致
                        fail-closed（不静默混装两种口径）；
    R5 纠偏 schema 负例（063）缺 correction 主体 / 未知版本 / original
                        factor 与 manifest 不符 / closed=true / effective
                        非 null / status 语义错 → 全部 fail-closed；
                       完整合法 sidecar（与真实三份同构）仍通过。
  065 测试 oracle 元测试（TOR）：monkeypatch validate_producer_receipt
    恒返回 None（模拟 validator 被破坏）→ 以子进程双跑进程内 oracle 用例
    （S1+C9），普通 Python 与 -O 都必须失败（红）；恢复后双跑都过（绿）。
    同时全套件 assert 已全部显式化为 _check（-O 不删除），-O 下门禁有效。

用法：
  PYTHONPATH=$PWD python3 -m benchmark.RULER.test_e119_yarn_binding_059_060_061
  PYTHONPATH=$PWD python3 -O -m benchmark.RULER.test_e119_yarn_binding_059_060_061
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
from benchmark.RULER import score_ruler_formal as SF  # noqa: E402
from benchmark.RULER.score_ruler_formal import (  # noqa: E402
    _load_producer_yarn_receipt, freeze_and_stage,
)
from benchmark.RULER.yarn_receipt import (  # noqa: E402
    RECEIPT_V1_VERSION, RECEIPT_VERSION, attempt_lock_path,
    build_yarn_receipt, effective_config_sha256, manifest_correction_path_for,
    producer_receipt_path_for, resolve_manifest_yarn_identity,
    validate_producer_receipt, write_yarn_receipt,
)

NATIVE_MPE = 40960
PASS = 0


def _check(cond, msg=""):
    """065（TL-E119-YARN-TEST-ORACLE）：显式判定——非 assert 语句，
    python -O 不删除，验收门禁在 -O 下仍有效。失败 → SystemExit
    （[TEST-FAIL] 前缀，非零退出码）。本套件全部判定（含负例必须拒收、
    终态一致性、manifest 字段核对）统一走本入口；GPT 审计 065 复现：
    validator 被破坏后普通 Python 正确失败而 -O 仍 exit 0 打印 PASS。"""
    if not cond:
        raise SystemExit(f"[TEST-FAIL] {msg}")


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


def _receipt_for(f, context_length, task, factor=2.0, enabled=True,
                version=RECEIPT_VERSION, run_id=None, max_num=500, seed=42):
    """为预测文件 f 的【当前字节】构建 v2 完成回执（062 barrier 的
    B 代提交与 _write_receipts 共用；SHA/行数对当前文件现算——生产者
    提交语义：先改预测字节，再用本函数对新字节出回执）。066 B3：seed
    参数化——run-A（seed=42/max_num=1）与 run-B（seed=99/max_num=100）
    预测字节相同、配置不同。"""
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
        kw = {"run_id": run_id or f"fixture-{os.getpid()}",
              "prediction_basename": os.path.basename(f),
              "prediction_sha256": _sha(f),
              "prediction_lines": _nlines(f)}
    return build_yarn_receipt(
        yarn_enabled=enabled, effective_factor=eff, yarn_factor_cli=None,
        rope_scaling=scaling, context_length=context_length, task=task,
        model_path="/synthetic/Qwen3-8B", model_config_sha256=None,
        native_mpe=NATIVE_MPE,
        generation_params={"max_gen": 64, "max_num": max_num, "seed": seed,
                           "method": "tli", "pred_postfix": "_fx",
                           "t": "01010000"},
        producer_script_path="benchmark/RULER/pred_ruler.py",
        producer_script_sha256="0" * 64,
        receipt_version=version, **kw)


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
        rcp = _receipt_for(f, cl, task, factor=factor, enabled=enabled,
                           version=version,
                           run_id=f"fixture-{os.getpid()}-{len(paths)}")
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
    """终态一致性：回执声明 SHA/行数与预测当前字节逐位一致（同代）。

    062：_load_producer_yarn_receipt 签名重排为（staged, source）——此处
    单文件直查场景 staged==source（同一文件既当评分副本又当回执旁挂
    探查对象），消费侧三方一致语义退化为两方（staged==receipt==源同
    一文件）；staging 路径分離场景由 B1/B2 barrier 用例覆盖。"""
    rcp = json.load(open(rcp_path, encoding="utf-8"))
    _check(rcp["receipt_version"] == RECEIPT_VERSION, rcp["receipt_version"])
    _check(rcp["status"] == "complete")
    _check(rcp["prediction_basename"] == os.path.basename(pred_path))
    _check(rcp["prediction_sha256"] == _sha(pred_path), f"{tag}: 回执 SHA 与预测字节不一致（不同代）")
    _check(rcp["prediction_lines"] == _nlines(pred_path), f"{tag}: 回执行数与预测不一致")
    # 消费侧入口同口径接受（真实 _load_producer_yarn_receipt，非复制品）
    got = _load_producer_yarn_receipt(pred_path, pred_path, "vt", 32768)
    _check(got is not None and \
        got["prediction_binding"]["verified_same_generation"] is True, got)


def test_P1_success_single(base):
    """P1：success 单跑——v2 同代绑定四件齐备 + 无临时残留 + 消费侧接受。"""
    _run_producer(base, "p1", "success", "A")
    d, pred, rcp = _producer_products(base, "p1")
    _check(os.path.isfile(pred) and os.path.isfile(rcp))
    _assert_consistent_generation("P1", pred, rcp)
    # 无临时残留（.gen-/.tmp- 均经 os.replace 提交或不存在）
    _check(not glob.glob(os.path.join(d, "*.gen-*")), "gen 临时残留")
    _check(not glob.glob(os.path.join(d, "*tmp-*")), "tmp 临时残留")
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
    _check(rc == 9, (rc, out[-300:]))
    _check(_sha(pred) == sha_before, "crash-mid 改写了上一代完整预测")
    _check(_sha(rcp) == rcp_before, "crash-mid 改写了上一代回执")
    # partial 只在临时 generation（不以 .jsonl 结尾 → 不进 {task}-*.jsonl glob）
    gens = glob.glob(os.path.join(d, "*.gen-*"))
    _check(gens, "crash-mid 应留下临时 generation（partial 证据）")
    _check(not glob.glob(os.path.join(d, "vt-*.jsonl.gen-*"[:-7] + "*")) \
        or all(not g.endswith(".jsonl") for g in gens))
    _check(all(not g.endswith(".jsonl") for g in gens), "临时 generation 不得以 .jsonl 结尾（会污染 best-file glob）")
    jsonl_glob = glob.glob(os.path.join(d, "vt-*.jsonl"))
    _check(jsonl_glob == [pred], ("glob 污染", jsonl_glob))
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
    _check(rc == 9, (rc, out[-300:]))
    rcp_obj = json.load(open(rcp, encoding="utf-8"))
    _check(rcp_obj["prediction_sha256"] != _sha(pred), "crash-between 后应留下新预测配旧回执的失配态")
    try:
        _load_producer_yarn_receipt(pred, pred, "vt", 32768)
        raise AssertionError("失配代际未被消费侧拒收（059 绑定失效）")
    except SystemExit as e:
        _check("prediction_sha256" in str(e) and "不同代" in str(e), e)
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
    _check(ra == 0 and rb == 0, (ra, oa[-400:], rb, ob[-400:]))
    d, pred, rcp = _producer_products(base, "p4")
    _assert_consistent_generation("P4", pred, rcp)
    rcp_obj = json.load(open(rcp, encoding="utf-8"))
    # 终态属于其中一代（pred 内容与回执 run_id 同代），而非 A 预测配 B 回执
    with open(pred, encoding="utf-8") as f:
        first_tag = json.loads(f.readline())["pred"].split("-")[2]
    _check(rcp_obj["run_id"].startswith("20"), rcp_obj["run_id"])
    _check(f"stub-pred-{first_tag}-" in open(pred, encoding="utf-8").read())
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
    _check(ra == 0, (ra, oa[-400:]))
    _check(rb == 9, (rb, ob[-400:]))
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
    _check(rcp_obj["prediction_sha256"] != _sha(pred), "前置：应处失配态")
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
    _check(got_lock_msg, "holder 未在 60s 内报告已获锁")
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        blocked = False
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            blocked = True
        _check(blocked, "持锁期间 LOCK_NB 探测不应成功（互斥失效）")
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
    _check(r.returncode == 0 and "DONE" in r.stdout, (r.returncode, r.stdout[-3000:], r.stderr[-2000:]))
    mf = json.load(open(out + ".manifest.json"))
    ri = mf["run_identity"]
    _check(ri["yarn_factor_provenance"] == "producer_receipt", ri)
    pe = ri["producer_evidence"]
    _check(pe["protocol"] == RECEIPT_VERSION and \
        pe["same_generation_bound"] is True and \
        pe["config_consistency_enforced"] is True, pe)
    for cell in mf["cells"].values():
        for t, ti in cell["tasks"].items():
            rc = ti["producer_yarn_receipt"]
            _check(rc["prediction_binding"] is not None and \
                rc["prediction_binding"]["verified_same_generation"] \
                is True, (t, rc))
            _check("effective_config_sha256" in rc["config_fingerprint"])
    print("C1 PASS  11 格 v2 回执 → producer_receipt + same_generation_"
          "bound + 逐格 prediction_binding.verified + 配置指纹冻结")


def _expect_formal_reject(base, tag, root, out, extra=("--yarn",),
                           optimized=False, needle=None):
    r = _formal(root, out, extra=extra, optimized=optimized)
    _check(r.returncode != 0, f"{tag}: 仍 exit=0（059/060 门禁失效）：\n{r.stdout[-2000:]}")
    blob = r.stdout + r.stderr
    _check("GATE-FAIL" in blob, (tag, blob[-1500:]))
    if needle is not None:
        _check(needle in blob, (tag, needle, blob[-1500:]))
    _check(not any(os.path.exists(p) for p in _products(out)), f"{tag}: 失败仍发布了产物")
    _check(glob.glob(out + ".failure-*.json"))
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
    _check(r.returncode == 0 and "DONE" in r.stdout, (r.returncode, r.stdout[-3000:], r.stderr[-2000:]))
    mf = json.load(open(out + ".manifest.json"))
    ri = mf["run_identity"]
    _check(ri["yarn_factor"] == 2.0 and \
        ri["yarn_factor_provenance"] == "producer_receipt_v1_partial", ri)
    pe = ri["producer_evidence"]
    _check(pe["protocol"] == RECEIPT_V1_VERSION and \
        pe["same_generation_bound"] is False and \
        "只证 factor 口径" in pe["note"], pe)
    for cell in mf["cells"].values():
        for ti in cell["tasks"].values():
            _check(ti["producer_yarn_receipt"]["prediction_binding"] is None)
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
    _check(err is not None and "prediction_basename" in err, err)
    print("C9 PASS  prediction_basename 失配 → 校验拒绝")


# ================================================================ 062 快照竞态

# 屏障用单任务：取 TASKS[0]——重试 formal 进程带 --expect-tasks 1，
# 期望任务集即 TASKS[:1]，单任务 root 必须恰好保留该任务才可重跑成功
BTASK = TASKS[0]


def _single_task_root(base, name, task=None):
    """单任务 root（062 barrier 用）：native fixture 只保留一个任务的
    文件 → freeze 的 best-file 仲裁唯一（首次 _nlines 调用即该任务的
    仲裁点，候选复制已完成、回执尚未读取——生产者提交注入点唯一且
    精确落在审计 062 的竞态窗口内）。"""
    if task is None:
        task = BTASK
    root = _copy_native_fixture(base, name)
    for d in glob.glob(os.path.join(root, "L*", "pred_*")):
        for f in os.listdir(d):
            if f.endswith(".jsonl") and not f.startswith(task + "-"):
                os.remove(os.path.join(d, f))
    return root


def _commit_generation_B(pred_path, task, context_length=32768):
    """模拟生产者新一代原子提交（062 barrier 注入）：改写预测内容 →
    os.replace 提交预测 → 原子写新回执（write_yarn_receipt 内部
    tmp+os.replace；与 commit_yarn_generation 同语义：回执最后落盘 =
    提交信号）。对提交后的 B 代字节现算 SHA/行数出新回执。"""
    rows = [json.loads(l) for l in open(pred_path, encoding="utf-8")]
    for r in rows:
        r["pred"] = f"gen-B-{r['pred']}"
    tmp = pred_path + ".commitB.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, pred_path)
    write_yarn_receipt(pred_path, _receipt_for(
        pred_path, context_length, task, run_id="gen-B-062"))


def _assert_freeze_three_way_consistent(root, staging, src_pred, tag):
    """062 正例核验（无注入）：freeze 成功且三方一致自洽——staging ==
    回执声明 == 源当前字节 == manifest 记录的 source/derived SHA，
    binding.verified_same_generation=true（GPT 审计 062 修复前正是
    缺这条三方断言才可达 staging=A/receipt=B/verified=true）。"""
    _, cells_info, _, pyarn = freeze_and_stage(
        root, "_fx", 1, True, staging, 2,
        os.path.join(TESTDATA, "data_root"))
    cell = cells_info[f"L32768/fxm"]["tasks"][BTASK]
    rcp = cell["producer_yarn_receipt"]
    b = rcp["prediction_binding"]
    _check(b is not None and b["verified_same_generation"] is True,
           f"{tag}: 正例 binding 缺失或未 verified: {b}")
    # 三方一致：staging == 回执声明 == 源冻结时刻字节；manifest 的
    # source_sha256/derived_sha256 复用同一冻结窗口值
    _check(b["staged_sha256"] == b["source_sha256_at_freeze"] ==
           b["prediction_sha256"], f"{tag}: binding 三方不一致: {b}")
    _check(b["staged_sha256"] == cell["source_sha256"] ==
           cell["derived_sha256"],
           f"{tag}: manifest SHA 与冻结窗口不一致: binding={b} "
           f"cell_src={cell['source_sha256']} "
           f"cell_derived={cell['derived_sha256']}")
    _check(pyarn["cells"][f"L32768/fxm/{BTASK}"] is not None,
           f"{tag}: producer_yarn 缺格")


def test_B1_commit_after_copy_before_receipt(base):
    """B1（062 barrier）：生产者恰在「staging 复制完成后、回执读取前」
    原子提交 B 代（预测+回执）→ freeze_and_stage 必须 fail-closed 拒绝，
    不得出现 staging=A、receipt=B、verified=true（GPT 复现的混合代际
    发布态）。注入点：monkeypatch SF._nlines——首次调用即 best-file
    仲裁（位于候选复制之后、回执读取之前的精确窗口），先执行 B 代
    原子提交再委托原实现；先跑无注入正例（三方一致自洽）作绿基线。"""
    root = _single_task_root(base, "b1_root")
    _write_receipts(root, 32768, 2.0)
    src_pred = _cell_file(root, BTASK)
    data_root = os.path.join(TESTDATA, "data_root")
    # ---- 绿基线：无注入 freeze 成功且三方一致 ----
    _assert_freeze_three_way_consistent(
        root, os.path.join(base, "b1_staging_ok"), src_pred, "B1-baseline")
    # ---- 红：窗口内提交 B 代 → 必须拒收 ----
    real_nlines = SF._nlines
    fired = {"done": False}

    def _nlines_commit_B_then_delegate(path):
        if not fired["done"]:
            fired["done"] = True
            _commit_generation_B(src_pred, BTASK)   # 复制后、回执读取前
        return real_nlines(path)

    SF._nlines = _nlines_commit_B_then_delegate
    try:
        freeze_and_stage(root, "_fx", 1, True,
                         os.path.join(base, "b1_staging"), 2, data_root)
        raise AssertionError("B1: staging=A/receipt=B 混合代际未被拒收"
                            "（062 快照竞态门禁失效）")
    except SystemExit as e:
        blob = str(e)
        _check("prediction_sha256" in blob and "不同代" in blob,
               f"B1: 拒绝原因非同代绑定失配: {blob}")
    finally:
        SF._nlines = real_nlines
    # 终态实况：源与回执已是 B 代（自洽 B+B），staging 内是 A——正式
    # formal 进程（不持注入）再跑应评 B 代并成功（拒绝并重试语义）
    r = _formal(root, os.path.join(base, "b1.json"),
                extra=("--yarn", "--expect-tasks", "1"))
    _check(r.returncode == 0 and "DONE" in r.stdout,
           ("B1-retry", r.returncode, r.stdout[-2000:], r.stderr[-1000:]))
    print("B1 PASS  复制后/读回执前提交 B 代 → freeze fail-closed 拒收"
          "（staging=A/receipt=B 不可达 verified=true）；重试评 B+B 成功")


def test_B2_commit_after_snapshot_before_source_check(base):
    """B2（062 barrier）：生产者在「回执 bytes 快照读取后、源三方校验
    前」原子提交 B 代 → 回执快照与 staging 仍同为 A 代，但源目录已推进
    到 B——三方一致门禁（staging==receipt==源当前字节）必须 fail-closed
    拒绝，不得发布 source_sha256 指向新一代的 manifest。注入点：
    monkeypatch SF.validate_producer_receipt——该调用位于回执快照读取
    之后、staged/源比对之前，首次调用先提交 B 代再委托原实现。"""
    root = _single_task_root(base, "b2_root")
    _write_receipts(root, 32768, 2.0)
    src_pred = _cell_file(root, BTASK)
    data_root = os.path.join(TESTDATA, "data_root")
    _assert_freeze_three_way_consistent(
        root, os.path.join(base, "b2_staging_ok"), src_pred, "B2-baseline")
    real_vpr = SF.validate_producer_receipt
    fired = {"done": False}

    def _vpr_commit_B_then_delegate(receipt, pred_path):
        if not fired["done"]:
            fired["done"] = True
            _commit_generation_B(src_pred, BTASK)   # 快照后、源校验前
        return real_vpr(receipt, pred_path)

    SF.validate_producer_receipt = _vpr_commit_B_then_delegate
    try:
        freeze_and_stage(root, "_fx", 1, True,
                         os.path.join(base, "b2_staging"), 2, data_root)
        raise AssertionError("B2: 源推进后的三方失配未被拒收"
                            "（062 三方一致门禁失效）")
    except SystemExit as e:
        blob = str(e)
        _check("冻结窗口" in blob and "062" in blob,
               f"B2: 拒绝原因非三方一致门禁: {blob}")
    finally:
        SF.validate_producer_receipt = real_vpr
    print("B2 PASS  回执快照后/源校验前提交 B 代 → 三方一致门禁"
          " fail-closed 拒绝（不发布 source_sha256 指向新一代的 manifest）")


# ================================================================ 066

# B3 提交子进程脚本（测试运行时写进临时目录，真子进程执行）：
# 与生产者/formal 同键 acquire_output_lock（真实 flock）——formal 持锁
# 期间阻塞；获锁后按生产提交语义提交 B 代：预测字节保持与 A 完全相同
# （066 审计前提：同字节不同配置），回执最后落盘 = 提交信号。
_B3_COMMIT_SUBPROC = """\
import json, os, sys
repo, src, rcp_json, ready_marker, done_marker = sys.argv[1:6]
sys.path.insert(0, repo)
from benchmark.RULER import yarn_receipt as YR
open(ready_marker, "wb").close()      # 已就绪、即将取锁（父进程据此推进）
fd = YR.acquire_output_lock(src)      # 与生产者/formal 同键 flock
try:
    raw = open(src, "rb").read()      # A 代当前字节
    tmp = src + ".commitB.tmp"
    with open(tmp, "wb") as f:
        f.write(raw)                  # B 代预测字节 == A（066 审计前提）
    os.replace(tmp, src)              # 生产提交语义第一步（预测）
    YR.write_yarn_receipt(src, json.load(open(rcp_json)))   # 第二步（回执）
finally:
    YR.release_output_lock(fd)
open(done_marker, "wb").close()        # 提交完成（阻塞/释放的终点信号）
"""


def test_B3_same_bytes_provenance_locked(base):
    """B3（066 同字节代际来源负例）：run-A（seed=42/max_num=1）与 run-B
    （seed=99/max_num=100）预测字节完全相同、配置不同；在「staging 复制
    与回执读取之间」的精确窗口内用【真子进程】（同键 flock）提交 B 代。

    修复语义（GPT 方案 2 共享锁）：formal 在本格冻结窗口（候选复制 →
    best-file 仲裁 → 回执 bytes 快照 → 三方校验）全程持生产者同键锁 →
    真子进程的提交在锁上阻塞、窗口内不可穿插；formal 冻结 A 代自洽
    （run_id=run-A/seed=42/max_num=1，verified 同 A 代）；锁释放后子
    进程完成 B 代自洽落盘。验收红路径（修复前）：formal 无锁 → 子进程
    即时完成提交 → done marker 在窗口内出现（红）+ 冻结身份被 B 代
    抢注（staging 配 A 字节 + 回执配 B 配置且 verified=true，066 审计
    复现态）。不得出现 staging 配 A 回执配 B 且 verified=true。"""
    root = _single_task_root(base, "b3_root")
    src_pred = _cell_file(root, BTASK)
    data_root = os.path.join(TESTDATA, "data_root")
    # ---- A 代（run-A / seed=42 / max_num=1）提交在位 ----
    write_yarn_receipt(src_pred, _receipt_for(
        src_pred, 32768, BTASK, run_id="run-A-066", max_num=1, seed=42))
    a_sha = _sha(src_pred)
    a_lines = _nlines(src_pred)
    # ---- B 代回执（字节与 A 完全相同、run_id/seed/max_num 不同）----
    rcp_b = _receipt_for(src_pred, 32768, BTASK, run_id="run-B-066",
                         max_num=100, seed=99)
    _check(rcp_b["prediction_sha256"] == a_sha and
           rcp_b["prediction_lines"] == a_lines,
           "B3 前提：B 代回执绑定字段须与 A 代字节完全相同（同字节前提）")
    rcp_b_path = os.path.join(base, "b3_rcp_b.json")
    json.dump(rcp_b, open(rcp_b_path, "w", encoding="utf-8"), indent=1,
              ensure_ascii=False)
    helper = os.path.join(base, "b3_commitB_locked.py")
    with open(helper, "w", encoding="utf-8") as f:
        f.write(_B3_COMMIT_SUBPROC)
    ready = os.path.join(base, "b3_ready.marker")
    done = os.path.join(base, "b3_done.marker")
    # ---- 注入点：首次 _nlines 调用 = best-file 仲裁（候选复制已完成、
    #      回执尚未读取——066 审计的精确穿插窗口；此时 formal（本进程）
    #      已按 066 修复持有生产者同键锁）→ 起真子进程提交 B 代 ----
    real_nlines = SF._nlines
    fired = {"done": False}
    spawned = {}

    def _nlines_spawn_commit_B(path):
        if not fired["done"]:
            fired["done"] = True
            spawned["p"] = subprocess.Popen(
                [sys.executable, helper, REPO, src_pred, rcp_b_path,
                 ready, done],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                env={**os.environ, "PYTHONPATH": REPO})
            deadline = time.time() + 60
            while time.time() < deadline and not os.path.exists(ready):
                time.sleep(0.02)
            _check(os.path.exists(ready), "B3: 提交子进程未在 60s 内就绪")
            # 锁互斥实测：formal 持锁窗口内 B 代提交不可完成（修复前
            # formal 无锁 → 子进程即时完成 → done 出现 → 红）
            time.sleep(0.4)
            _check(not os.path.exists(done),
                   "B3: 锁窗口内 B 代提交完成——formal 未持生产者同键锁"
                   "（066 修复失效，穿插可达）")
        return real_nlines(path)

    SF._nlines = _nlines_spawn_commit_B
    b3_cell = {}
    try:
        _, cells_info, _, _ = freeze_and_stage(
            root, "_fx", 1, True, os.path.join(base, "b3_staging"), 2,
            data_root)
        b3_cell = cells_info[f"L32768/fxm"]["tasks"][BTASK]
    finally:
        SF._nlines = real_nlines
    # ---- 冻结成功：A 代自洽冻结，身份归属未被 B 代抢注 ----
    rcp = b3_cell["producer_yarn_receipt"]
    b = rcp["prediction_binding"]
    _check(b is not None and b["verified_same_generation"] is True,
           f"B3: A 代自洽冻结的 binding 异常: {b}")
    _check(b["run_id"] == "run-A-066",
           f"B3: 冻结身份被 B 代抢注（run_id={b['run_id']!r}）——同字节"
           f"不同配置的穿插不可达才对（066）")
    _check(rcp["config_fingerprint"]["seed"] == 42 and
           rcp["config_fingerprint"]["max_num"] == 1,
           f"B3: 冻结配置指纹须属 A 代: {rcp['config_fingerprint']}")
    _check(b["staged_sha256"] == b["source_sha256_at_freeze"] == a_sha ==
           b["prediction_sha256"],
           f"B3: 冻结窗口三方应全为 A 代字节: {b}")
    _check(b["staged_sha256"] == b3_cell["source_sha256"] ==
           b3_cell["derived_sha256"],
           f"B3: manifest SHA 与冻结窗口不一致: {b3_cell}")
    # ---- 066 锁审计字段（轻量）在位 ----
    fl = b3_cell.get("freeze_lock")
    _check(isinstance(fl, dict) and fl.get("n_files") == 1 and
           isinstance(fl.get("acquire_wait_seconds"), (int, float)) and
           fl["acquire_wait_seconds"] >= 0,
           f"B3: freeze_lock 审计字段异常: {fl}")
    # ---- 释放后子进程获锁 → B 代自洽落盘（阻塞/释放实测的正向终点）----
    p = spawned["p"]
    out, _ = p.communicate(timeout=120)
    _check(p.returncode == 0, f"B3: 提交子进程失败: {out[-500:]}")
    _check(os.path.exists(done), "B3: 锁释放后子进程未完成提交（死锁？）")
    rcp_after = json.load(open(producer_receipt_path_for(src_pred),
                               encoding="utf-8"))
    _check(rcp_after["run_id"] == "run-B-066" and
           rcp_after["prediction_sha256"] == _sha(src_pred),
           f"B3: 释放后 B 代应自洽落盘（预测与回执同代）: {rcp_after}")
    _check(_sha(src_pred) == a_sha, "B3 前提复核：B 代预测字节与 A 完全相同")
    print(f"B3 PASS  同字节不同配置的 B 代真子进程在「复制→读回执」窗口内"
          f"提交 → 同键锁窗口内阻塞不可穿插；formal 冻结身份保持 run-A/"
          f"seed=42/max_num=1 且三方一致；锁释放后 B 代自洽落盘"
          f"（freeze_lock.acquire_wait_seconds={fl['acquire_wait_seconds']}s）")


def test_B4_lock_unavailable_fail_closed(base):
    """B4（066 锁不可用 fail-closed）：冻结窗口输出路径锁获取失败
    （只读源目录，锁文件不可创建）→ freeze_and_stage 必须 fail-closed
    拒收，不得静默降级为无锁冻结。环境无法模拟只读（root 用户/无效
    chmod 的网络 FS）时如实 SKIP 不冒充。"""
    if os.geteuid() == 0:
        print("B4 SKIP  以 root 运行，chmod 无法模拟只读目录——不冒充通过")
        return
    root = _single_task_root(base, "b4_root")
    _write_receipts(root, 32768, 2.0)
    pred_dir = os.path.dirname(_cell_file(root, BTASK))
    os.chmod(pred_dir, 0o555)
    try:
        # 前置探测：chmod 确实使新文件创建失败（否则环境不可模拟 → SKIP）
        probe = os.path.join(pred_dir, ".b4_probe")
        try:
            open(probe, "wb").close()
        except OSError:
            probe_blocked = True
        else:
            os.remove(probe)
            probe_blocked = False
        if not probe_blocked:
            print("B4 SKIP  chmod 0o555 未使目录只读（网络 FS/root-squash"
                  "等）——环境无法模拟锁不可用，不冒充通过")
            return
        try:
            freeze_and_stage(root, "_fx", 1, True,
                             os.path.join(base, "b4_staging"), 2,
                             os.path.join(TESTDATA, "data_root"))
            raise AssertionError("B4: 锁不可用时未 fail-closed 拒收"
                                "（066 静默降级漏洞）")
        except SystemExit as e:
            blob = str(e)
            _check("锁获取失败" in blob and "066" in blob,
                   f"B4: 拒绝原因非锁不可用 fail-closed: {blob}")
    finally:
        os.chmod(pred_dir, 0o755)
    print("B4 PASS  锁不可用（只读源目录）→ fail-closed 拒收"
          "（不静默降级为无锁冻结）")


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

    _check(validate_producer_receipt(good(), pred) is None)
    cases = []

    def case(name, mutate):
        r = good()
        mutate(r)
        err = validate_producer_receipt(r, pred)
        _check(err is not None, f"{name}: 未被拒绝（060 schema 漏洞）")
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
    _check(h1 == h2 == h3, "task/method/t 不得进配置指纹（分组自由字段）")
    # 跨档不比较（context_length 进指纹 → 跨档 hash 不同属预期）
    h4 = effective_config_sha256(receipt_for("vt", 65536))
    _check(h4 != h1)
    # 应同字段漂移（seed）→ hash 变化
    h5 = effective_config_sha256(receipt_for("vt", 32768, seed=999))
    _check(h5 != h1)
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
    _check(r.returncode == 0 and "DONE" in r.stdout, (r.returncode, r.stdout[-3000:], r.stderr[-2000:]))
    mf = json.load(open(out + ".manifest.json"))
    _check(mf["run_identity"]["yarn_factor_provenance"] == \
        "producer_receipt", mf["run_identity"]["yarn_factor_provenance"])
    print("S3 PASS  task/method/t 不进指纹；跨档不比较；双档 yarn off "
          "formal 正例通过（按 context_length 分组的一致性门禁）")


# ================================================================ 064

def test_S4_config_closure_negatives(base):
    """S4（064 配置闭包两处残缺）：
    ① max_num=1 vs 100 两份合法回执 effective_config_sha256 必须不同
      （修复前同指纹 a09b4ef6...——「必需+类型校验但不进指纹」漏洞）；
      同 max_num → 同指纹（sanity）；
    ② formal 跨格负例：单格 max_num=999（其余全同）→ 非零退出
      （060 跨格门禁现收录 max_num）；
    ③ model_config_sha256=None（fixture 默认，远端模型 ID 场景）→
      manifest config_fingerprint.config_identity=missing +
      producer_evidence.model_config_closure=false + 降级 note；
      改为合法 hex（全格一致）→ config_identity=config_json_sha256_bound
      + closure=true（不虚报不漏报，均 python 与 -O 双跑）。"""
    # ① 指纹负例（进程内纯函数；绑定字段用占位值——指纹只依赖配置字段）
    def _cfg_receipt(max_num):
        return build_yarn_receipt(
            yarn_enabled=True, effective_factor=2.0, yarn_factor_cli=None,
            rope_scaling={"rope_type": "yarn", "type": "yarn",
                          "factor": 2.0,
                          "original_max_position_embeddings": NATIVE_MPE,
                          "beta_fast": 32, "beta_slow": 1},
            context_length=32768, task="vt",
            model_path="/synthetic/Qwen3-8B", model_config_sha256=None,
            native_mpe=NATIVE_MPE,
            generation_params={"max_gen": 64, "max_num": max_num, "seed": 42,
                               "method": "tli", "pred_postfix": "_fx",
                               "t": "01010000"},
            producer_script_path="benchmark/RULER/pred_ruler.py",
            producer_script_sha256="0" * 64, run_id="x",
            prediction_basename="vt-fxm-01010000.jsonl",
            prediction_sha256="0" * 64, prediction_lines=2)
    h1 = effective_config_sha256(_cfg_receipt(1))
    h100 = effective_config_sha256(_cfg_receipt(100))
    h1b = effective_config_sha256(_cfg_receipt(1))
    _check(h1 != h100, f"064② 未修复：max_num=1 与 100 同指纹 {h1}")
    _check(h1 == h1b, "同配置应同指纹（max_num 稳定入指纹）")
    # ② formal 跨格负例：单格 max_num=999 → 060 门禁收录 max_num 后必拒
    root = _copy_native_fixture(base, "s4a_root")
    _write_receipts(root, 32768, 2.0)
    _rewrite_receipt(_cell_receipt(root, "vt"),
                     lambda r: r["generation_params"].__setitem__(
                         "max_num", 999))
    out = os.path.join(base, "s4a.json")
    _expect_formal_reject(base, "S4-max_num", root, out, needle="max_num")
    # ③ config_identity=missing 降级标注（None 为 fixture 默认——远端
    #    模型 ID / config.json 不在 model_path 的生产侧真实形态）
    for optimized in (False, True):
        root = _copy_native_fixture(
            base, f"s4b_root_{'O' if optimized else 'py'}")
        _write_receipts(root, 32768, 2.0)   # model_config_sha256=None
        out = os.path.join(base, "s4b.json")
        r = _formal(root, out, extra=("--yarn",), optimized=optimized)
        _check(r.returncode == 0 and "DONE" in r.stdout,
               ("S4-missing", optimized, r.returncode,
                r.stdout[-2000:], r.stderr[-1000:]))
        mf = json.load(open(out + ".manifest.json"))
        pe = mf["run_identity"]["producer_evidence"]
        _check(pe["model_config_closure"] is False, pe)
        _check("model_config_closure_note" in pe and
               "完整模型配置闭包" in pe["model_config_closure_note"], pe)
        for cell in mf["cells"].values():
            for ti in cell["tasks"].values():
                cf = ti["producer_yarn_receipt"]["config_fingerprint"]
                _check(cf["config_identity"] == "missing", cf)
                _check(cf["model_config_sha256"] is None, cf)
    # ④ 有值（合法 hex，全格一致）→ bound + closure=true
    root = _copy_native_fixture(base, "s4c_root")
    _write_receipts(root, 32768, 2.0)
    for f in sorted(glob.glob(os.path.join(root, "L*", "pred_fx",
                                            "*-yarn_receipt.json"))):
        _rewrite_receipt(f, lambda r: r.__setitem__(
            "model_config_sha256", "c" * 64))
    out = os.path.join(base, "s4c.json")
    r = _formal(root, out, extra=("--yarn",))
    _check(r.returncode == 0 and "DONE" in r.stdout,
           ("S4-bound", r.returncode, r.stdout[-2000:], r.stderr[-1000:]))
    mf = json.load(open(out + ".manifest.json"))
    pe = mf["run_identity"]["producer_evidence"]
    _check(pe["model_config_closure"] is True, pe)
    _check("model_config_closure_note" not in pe, pe)
    for cell in mf["cells"].values():
        for ti in cell["tasks"].values():
            cf = ti["producer_yarn_receipt"]["config_fingerprint"]
            _check(cf["config_identity"] == "config_json_sha256_bound", cf)
    print("S4 PASS  max_num=1 vs 100 指纹必异；单格 max_num=999 → formal"
          " 拒收；model_config_sha256=None → config_identity=missing + "
          "closure=false 降级（python 与 -O 双跑），有值 → bound/true")


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
        _check(res["yarn_factor_provenance"] == \
            "operator_declared_not_effective", (arm, res))
        _check(res["effective_yarn_factor"] is None, (arm, res))
        _check(res["manifest_run_identity_yarn_factor"] == 2.0, (arm, res))
        c = res["correction"]
        _check(c is not None and \
            c["correction_version"] == "yarn-identity-057-v1" and \
            c["target_manifest_sha256"] == _sha(mp) and \
            c["correction_sha256"] == _sha(c["correction_path"]), (arm, c))
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
    _check(res["correction"] is None and \
        res["effective_yarn_factor"] == 2.0 and \
        res["yarn_factor_provenance"] == "manifest_declared_no_provenance", res)
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
        _check("target_manifest_sha256" in str(e), e)
    print("R3 PASS  target hash 失配 → SystemExit fail-closed")


def _write_correction(manifest_path, version="yarn-identity-057-v1"):
    """为 manifest 落一份完整合法纠偏 sidecar（target hash 绑定当前字节）。

    063：纠偏 schema 已严格化——sidecar 须含 original_run_identity（与
    manifest run_identity 声明逐位一致）+ correction 语义字段齐备
    （status=not_effective / effective=null / closed=false / 声明值一致），
    与三份既有 128K 生产 sidecar 同构；version 须在已发布枚举内。"""
    mf = json.load(open(manifest_path, encoding="utf-8"))
    ri = mf["run_identity"]
    declared = ri.get("yarn_factor")
    correction = {
        "correction_version": version,
        "target_manifest": os.path.basename(manifest_path),
        "target_manifest_sha256": _sha(manifest_path),
        "original_run_identity": {"yarn": bool(ri.get("yarn")),
                                  "yarn_factor": declared},
        "correction": {
            "yarn_factor_status": "operator_declared_not_effective",
            "operator_declared_yarn_factor": declared,
            "effective_yarn_factor": None,
            "effective_factor_closed": False,
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
    _check(r.returncode == 0, (r.returncode, r.stdout[-500:], r.stderr[-500:]))
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
    _check(r1.returncode == 0, (r1.returncode, r1.stdout[-1500:],
                                r1.stderr[-1000:]))
    summary = json.load(open(os.path.join(
        b1, "results", "e119_ruler128k_formal_summary.json")))
    per_yarn = summary["identity_gate"]["per_arm_yarn_identity"]
    _check(set(per_yarn) == {"mavg", "FullKV", "aavg"}, per_yarn)
    for arm, y in per_yarn.items():
        _check(y["yarn_factor_provenance"] == \
            "operator_declared_not_effective", (arm, y))
        _check(y["effective_yarn_factor"] is None and \
            y["correction"]["target_manifest_sha256"] is not None, (arm, y))
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
    _check(r2.returncode != 0, "单臂纠改应被跨臂身份门禁拒绝")
    _check("yarn_factor" in (r2.stdout + r2.stderr), (r2.stdout[-800:], r2.stderr[-800:]))
    print("R4 PASS  128K analyzer 接线：三臂全纠偏 → 通过 + 逐臂 "
          "not_effective（null 进跨臂身份）；单臂纠偏 → 跨臂身份不一致"
          " fail-closed")


# ================================================================ 063

def test_R5_correction_schema_negatives(base):
    """R5（063 纠偏 sidecar schema 太弱）：缺 correction 主体 / 未知
    correction_version / original factor 与 manifest 不符 /
    effective_factor_closed=true / effective_yarn_factor 非 null /
    yarn_factor_status 语义错 / 缺 original_run_identity → 全部
    SystemExit fail-closed；完整合法 sidecar（与三份真实 128K sidecar
    同构，正例回归由 R1 承载）仍通过。修复前最小两字段 sidecar
    （version=unrecognized-garbage-version）也被解释成有效纠偏。"""
    results = os.path.join(REPO, "exp", "trace", "results")
    src_m = os.path.join(results, "e119_ruler128k_formal_mavg.json"
                                     ".manifest.json")

    def full_sidecar(manifest_path, mutate=None):
        """完整合法 sidecar（真实 128K 生产 sidecar 同构）+ 注入变异。"""
        c = {
            "correction_version": "yarn-identity-057-v1",
            "target_manifest": os.path.basename(manifest_path),
            "target_manifest_sha256": _sha(manifest_path),
            "original_run_identity": {"yarn": True, "yarn_factor": 2.0},
            "correction": {
                "yarn_factor_status": "operator_declared_not_effective",
                "operator_declared_yarn_factor": 2.0,
                "effective_yarn_factor": None,
                "effective_factor_closed": False,
            },
        }
        if mutate:
            mutate(c)
        cpath = manifest_correction_path_for(manifest_path)
        json.dump(c, open(cpath, "w", encoding="utf-8"), indent=1,
                  ensure_ascii=False)
        return cpath

    cases = []

    def case(name, mutate, needle):
        mp = os.path.join(base, f"r5_{len(cases)}_manifest.json")
        shutil.copyfile(src_m, mp)
        full_sidecar(mp, mutate)
        try:
            resolve_manifest_yarn_identity(mp)
            raise AssertionError(f"R5 {name}: 未被拒收（063 schema 漏洞）")
        except SystemExit as e:
            _check(needle in str(e), f"R5 {name}: 拒绝原因不符: {e}")
        cases.append(name)

    # 审计 063 最小复现：缺主体 + 任意非空未知版本 → 修复前静默通过
    case("缺 correction 主体",
         lambda c: c.pop("correction"), "correction 主体")
    case("未知 correction_version",
         lambda c: c.__setitem__("correction_version",
                                 "unrecognized-garbage-version"),
         "未知版本")
    case("original factor 与 manifest 不符",
         lambda c: c["original_run_identity"].__setitem__("yarn_factor",
                                                          4.0),
         "不一致")
    case("effective_factor_closed=true",
         lambda c: c["correction"].__setitem__("effective_factor_closed",
                                               True),
         "effective_factor_closed")
    case("effective_yarn_factor 非 null",
         lambda c: c["correction"].__setitem__("effective_yarn_factor", 4.0),
         "非 null")
    case("yarn_factor_status 语义错",
         lambda c: c["correction"].__setitem__("yarn_factor_status",
                                              "effective"),
         "yarn_factor_status")
    case("缺 original_run_identity",
         lambda c: c.pop("original_run_identity"), "original_run_identity")
    # 正例：完整合法 sidecar → not_effective/null（三份真实 sidecar 的
    # 同构回归由 R1 承载；此处验证新 schema 不误伤合法构造）
    mp = os.path.join(base, "r5_pos_manifest.json")
    shutil.copyfile(src_m, mp)
    full_sidecar(mp)
    res = resolve_manifest_yarn_identity(mp)
    _check(res["effective_yarn_factor"] is None and
           res["yarn_factor_provenance"] ==
           "operator_declared_not_effective" and
           res["correction"]["correction_version"] ==
           "yarn-identity-057-v1", res)
    print(f"R5 PASS  纠偏 schema 负例 {len(cases)} 连全拒收"
          f"（{'/'.join(cases)}）；完整合法 sidecar 正例通过")


# ================================================================ 065

_ORACLE_HARNESS = (
    "import sys, tempfile, shutil\n"
    "sys.path.insert(0, {repo!r})\n"
    "import benchmark.RULER.yarn_receipt as _yr\n"
    "import benchmark.RULER.score_ruler_formal as _sf\n"
    "if {break_validator}:\n"
    "    _broken = lambda receipt, pred_path: None\n"
    "    _yr.validate_producer_receipt = _broken\n"
    "    _sf.validate_producer_receipt = _broken\n"
    "import benchmark.RULER.test_e119_yarn_binding_059_060_061 as T\n"
    "if {break_validator}:\n"
    "    T.validate_producer_receipt = _yr.validate_producer_receipt\n"
    "base = tempfile.mkdtemp(prefix='e119_oracle_')\n"
    "try:\n"
    "    T.test_S1_schema_negatives()\n"
    "    T.test_C9_basename_mismatch()\n"
    "finally:\n"
    "    shutil.rmtree(base, ignore_errors=True)\n"
    "print('ORACLE-HARNESS-END')\n"
)


def test_T_oracle_meta():
    """TOR（065 测试 oracle 元测试）：monkeypatch validate_producer_receipt
    恒返回 None（模拟 validator 被破坏）→ 以子进程跑进程内 oracle 用例
    （S1 schema 负例 + C9 basename 失配），普通 Python 与 -O 都必须
    失败（红）；恢复后双跑都过（绿）。修复前（assert 版）普通 Python
    正确失败而 -O 删除断言 exit 0 打印 PASS——本元测试证明 -O 门禁
    有效性；S1/C9 是直接消费 validate_producer_receipt 的最小 oracle
    集，全套件其余判定同走 _check（AST 扫描零 assert 语句）。"""
    for break_validator, optimized, expect_ok in (
            (True, False, False), (True, True, False),
            (False, False, True), (False, True, True)):
        mode = ("-O " if optimized else "") + \
               ("broken" if break_validator else "intact")
        argv = [sys.executable] + (["-O"] if optimized else []) + \
            ["-c", _ORACLE_HARNESS.format(repo=REPO,
                                          break_validator=break_validator)]
        r = subprocess.run(argv, capture_output=True, text=True, cwd=REPO,
                           env={**os.environ, "PYTHONPATH": REPO})
        blob = r.stdout + r.stderr
        if expect_ok:
            _check(r.returncode == 0 and "ORACLE-HARNESS-END" in blob,
                   f"TOR[{mode}]: 恢复后应通过: rc={r.returncode} {blob[-800:]}")
        else:
            _check(r.returncode != 0 and "未被拒绝" in blob,
                   f"TOR[{mode}]: validator 被破坏后必须失败（-O 门禁失效）"
                   f": rc={r.returncode} {blob[-800:]}")
        print(f"  TOR[{mode}] PASS  rc={r.returncode}（{'绿' if expect_ok else '红'}）")
    print("TOR PASS  oracle 元测试：validator 破坏 → python 与 -O 双红；"
          "恢复 → 双绿（_check 显式判定在 -O 下不失效）")


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
        # #196 增量用例：062 竞态屏障 / 063 schema 负例 / 064 闭包负例 /
        # 065 oracle 元测试
        ("B1", lambda: test_B1_commit_after_copy_before_receipt(base)),
        ("B2", lambda: test_B2_commit_after_snapshot_before_source_check(base)),
        ("S4", lambda: test_S4_config_closure_negatives(base)),
        ("R5", lambda: test_R5_correction_schema_negatives(base)),
        # #197 增量用例：066 同字节异代 provenance 锁窗口
        ("B3", lambda: test_B3_same_bytes_provenance_locked(base)),
        ("B4", lambda: test_B4_lock_unavailable_fail_closed(base)),
        ("TOR", test_T_oracle_meta),
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
