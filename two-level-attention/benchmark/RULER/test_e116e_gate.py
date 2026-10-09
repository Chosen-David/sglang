# E116e 红绿测试（GPT 1033 审计 030-036 七项修复回归）
#   TL-RULER-FORMAL-INCOMPLETE-030（min-samples 只 WARN 不门禁）→ E1
#   TL-RULER-STALE-OUTPUT-031（失败重跑保留旧成功产物）→ E4/E5
#   TL-RULER-INPUT-IDENTITY-032（manifest 不绑 length/源数据/参数）→ E3/E6
#   TL-RULER-OUT-CLOBBER-033（无 .json 后缀 MD 覆盖 JSON）→ E7
#   TL-RULER-CLOSURE-OPT-034（python -O 删除 closure 断言）→ E8
#   TL-RULER-E116D-REPRO-035（D9 硬依赖不入库的真实数据）→ 仓库内 fixture
#     （benchmark/RULER/testdata/e116e/，clean clone 恒可运行）+ D9 显式
#     --with-real-data 开关（缺数据 SKIP 不计门禁）→ E10/D9
#   TL-RULER-LEGACY-MUTATION-036（原地改写原始 pred 文件）→ E9/D9
#   既有套件不回归 → D10（E116c 11/11 + E116d）
# 用法:
#   PYTHONPATH=$PWD python3 -m benchmark.RULER.test_e116e_gate
#   PYTHONPATH=$PWD python3 -m benchmark.RULER.test_e116e_gate --with-real-data
#     （真实数据存在时额外跑三臂 32K 生产回归 59.38/59.99/57.33）
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
TESTDATA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "testdata", "e116e")
CLOSURE_SCRIPT = os.path.join(REPO, "exp", "trace",
                              "analyze_e116d_closure_v2.py")
# 真实数据集成测试（D9）：默认本机生产路径，env 可覆盖（clean clone 模拟）
# 注意目录层级：正式入口 glob root/L*/pred{postfix}——32K 平铺结构的 root
# 是 e109_full_Qwen3-8B（其下 L32768 匹配 L*，aavg/fullkv/mavg 不匹配）；
# 64K/128K 的 arm-first 结构 root 是 {arm}（其下 L65536/L131072 匹配）。
REAL_ROOT = os.environ.get(
    "E116E_REAL_ROOT",
    os.path.join(REPO, "exp/results_ruler/e109_full_Qwen3-8B"))
REAL_DATA_ROOT = os.environ.get(
    "E116E_DATA_ROOT",
    "/home/wangyuanshuo02/sparse-bench/third_party/"
    "KVCache-Factory/data/RULER")

from benchmark.RULER.score_ruler import TASKS  # noqa: E402

PASS = 0


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _products(out):
    """正式入口成功发布的四个产物路径。"""
    return [out, out[:-len(".json")] + ".md",
            out + ".manifest.json", out + ".receipt.json"]


def _formal(root, out, data_root=None, min_samples=None,
            manifest_out=None, extra=(), postfix="_fx"):
    argv = [sys.executable, "-u", "-m", "benchmark.RULER.score_ruler_formal",
            "--root", root, "--pred-postfix", postfix,
            "--data-root", data_root or os.path.join(TESTDATA, "data_root"),
            "--out", out]
    if min_samples is not None:
        argv += ["--min-samples", str(min_samples)]
    if manifest_out is not None:
        argv += ["--manifest-out", manifest_out]
    argv += list(extra)
    return subprocess.run(argv, capture_output=True, text=True, cwd=REPO,
                           env={**os.environ, "PYTHONPATH": REPO})


def _copy_fixture(base, name, src=None):
    """把仓库内 fixture 只读拷贝到临时目录（后续修改不污染入库数据）。"""
    dst = os.path.join(base, name)
    shutil.copytree(src or os.path.join(TESTDATA, "pred_root"), dst)
    return dst


def _fixture_files():
    return sorted(glob.glob(os.path.join(
        TESTDATA, "pred_root", "L32768", "pred_fx", "*.jsonl")))


def test_E1_min_samples_hard_gate(base):
    """E1 负例（030）：11 任务各 1 行 + 默认 min-samples 100 → 非零退出，
    不发布任何产物（含 staging 清理），failure receipt 落盘。"""
    root = _copy_fixture(base, "e1_root")
    # 截断到每任务 1 行
    for f in glob.glob(os.path.join(root, "L32768", "pred_fx", "*.jsonl")):
        lines = open(f, encoding="utf-8").readlines()
        open(f, "w", encoding="utf-8").writelines(lines[:1])
    out = os.path.join(base, "e1.json")
    r = _formal(root, out)
    assert r.returncode != 0, (r.returncode, r.stdout, r.stderr)
    assert "min-samples" in (r.stdout + r.stderr) and \
        "拒绝发布" in (r.stdout + r.stderr), r.stdout[-2000:]
    assert not any(os.path.exists(p) for p in _products(out)), \
        "min-samples 门禁失败仍发布了产物"
    assert not glob.glob(out + ".staging-*"), "staging 未清理（030）"
    assert glob.glob(out + ".failure-*.json"), "failure receipt 未落盘"
    print("E1 PASS  11 任务×1 行 + 默认 min-samples=100 → 非零退出 + "
          "不发布任何产物 + staging 清理 + failure receipt 落盘")


def test_E2_missing_task(base):
    """E2 负例（030/闭包）：10/11 任务（缺 vt）→ 非零退出不发布。"""
    root = _copy_fixture(base, "e2_root")
    os.remove(os.path.join(root, "L32768", "pred_fx", "vt-fxm-01010000.jsonl"))
    out = os.path.join(base, "e2.json")
    r = _formal(root, out, min_samples=2)
    assert r.returncode != 0 and "vt" in (r.stdout + r.stderr), \
        (r.returncode, r.stdout[-2000:], r.stderr[-1000:])
    assert not any(os.path.exists(p) for p in _products(out))
    assert not glob.glob(out + ".staging-*")
    assert glob.glob(out + ".failure-*.json")
    print("E2 PASS  10/11 任务（缺 vt）→ 非零退出 + 不发布 + failure receipt")


def test_E3_positive(base):
    """E3 正例：11 任务全满（fixture 2 行 + --min-samples 2）→ 原子发布，
    四产物完整可解析，receipt/manifest 身份字段齐备。"""
    root = os.path.join(TESTDATA, "pred_root")   # 直接用入库 fixture（只读）
    out = os.path.join(base, "e3.json")
    r = _formal(root, out, min_samples=2,
                extra=("--model-path", "/synthetic/Qwen3-8B", "--yarn",
                       "--extra-param", "tia_level1_topk=1024"))
    assert r.returncode == 0 and "DONE" in r.stdout, \
        (r.returncode, r.stdout[-3000:], r.stderr[-2000:])
    for p in _products(out):
        assert os.path.isfile(p), f"产物缺失: {p}"
    res = json.load(open(out))
    key = "L32768/fxm"
    assert set(res["scores"][key]) == set(TASKS) and \
        all(v == 100.0 for v in res["scores"][key].values()), res["scores"]
    mf = json.load(open(out + ".manifest.json"))
    assert mf["manifest_version"] == 2 and mf["run_id"]
    assert mf["run_identity"]["model_path"] == "/synthetic/Qwen3-8B" and \
        mf["run_identity"]["yarn"] is True and \
        mf["run_identity"]["extra_params"] == {"tia_level1_topk": "1024"}
    assert len(mf["source_data_sha256"]) == 11, "源数据 SHA 绑定缺任务"
    assert all(len(v["sha256"]) == 64
               for v in mf["source_data_sha256"].values())
    assert mf["cells"][key]["identity_mode"] == "legacy-partial" and \
        mf["cells"][key]["length_dir"] == 32768
    assert mf["tasks"]["vt"]["identity_mode"] == "legacy-partial"
    rc = json.load(open(out + ".receipt.json"))
    assert rc["status"] == "success" and rc["run_id"] == mf["run_id"]
    assert rc["avg"][key] == 100.0
    assert rc["legacy_stamp"]["invariant_asserted"] is True
    assert len(rc["scorer"]["sha256"]) == 64 and \
        len(rc["formal"]["sha256"]) == 64
    # 派生目录存在且含补刻副本（staging 成功后改名 .run-*）
    runs = glob.glob(out + ".run-*")
    assert len(runs) == 1, f"版本化派生目录异常: {runs}"
    stamped = glob.glob(os.path.join(
        runs[0], "pred_root", "L32768", "pred_fx", "vt-fxm-01010000.jsonl"))
    assert stamped and "_id" in json.loads(
        open(stamped[0], encoding="utf-8").readline()), "派生副本缺 _id"
    assert not glob.glob(out + ".staging-*"), "成功后 staging 未收编"
    print("E3 PASS  11 任务正例 → 四产物原子发布 + receipt/manifest 身份"
          "（model/yarn/extra-param/源数据 SHA×11/legacy-partial）+ "
          "派生目录补刻副本")


def test_E4_fail_keeps_old_products(base):
    """E4 负例（031）：成功后删任务重跑同一 --out → 失败；旧四产物 SHA
    逐位不变；failure receipt 落盘并声明旧产物属于上一轮成功。"""
    root = _copy_fixture(base, "e4_root")
    out = os.path.join(base, "e4.json")
    r = _formal(root, out, min_samples=2)
    assert r.returncode == 0 and "DONE" in r.stdout, r.stdout[-2000:]
    shas_before = {p: _sha(p) for p in _products(out)}
    # 删 vt 后重跑同一 out
    os.remove(os.path.join(root, "L32768", "pred_fx",
                           "vt-fxm-01010000.jsonl"))
    r = _formal(root, out, min_samples=2)
    assert r.returncode != 0, (r.returncode, r.stdout[-2000:])
    shas_after = {p: _sha(p) for p in _products(out)}
    assert shas_before == shas_after, \
        f"失败重跑改动了旧成功产物: {set(shas_before) ^ set(shas_after)}"
    fails = glob.glob(out + ".failure-*.json")
    assert len(fails) == 1, fails
    fr = json.load(open(fails[0]))
    assert fr["status"] == "failed" and "vt" in fr["error"]
    assert "上一轮成功" in fr["note"], fr["note"]
    assert not glob.glob(out + ".staging-*"), "失败后 staging 未清理"
    print("E4 PASS  成功后删任务重跑同 out → 非零退出 + 旧产物 SHA 逐位"
          "不变 + failure receipt（含上一轮成功声明）+ staging 清理")


def test_E5_scorer_stage_failure(base):
    """E5 负例（031）：scorer 阶段 answers hash 错配（行内 _answers_sha 与
    answers 不一致）重跑 → 失败不发布，旧产物保留。"""
    root = _copy_fixture(base, "e5_root",
                         src=os.path.join(TESTDATA, "pred_root_native"))
    out = os.path.join(base, "e5.json")
    r = _formal(root, out, min_samples=2)
    assert r.returncode == 0 and "DONE" in r.stdout, r.stdout[-2000:]
    shas_before = {p: _sha(p) for p in _products(out)}
    # 篡改源文件行 0 的 answers（保留行内 _answers_sha）→ 行内声明与
    # answers 重算不一致 → scorer 阶段 GATE-FAIL
    tgt = os.path.join(root, "L32768", "pred_fx", "vt-fxm-01010000.jsonl")
    rows = [json.loads(l) for l in open(tgt, encoding="utf-8")]
    rows[0]["answers"] = ["tampered answer"]
    with open(tgt, "w", encoding="utf-8") as f:
        for rr in rows:
            f.write(json.dumps(rr, ensure_ascii=False) + "\n")
    r = _formal(root, out, min_samples=2)
    assert r.returncode != 0, (r.returncode, r.stdout[-2000:])
    assert "answers" in (r.stdout + r.stderr), \
        (r.stdout[-2000:], r.stderr[-1000:])
    assert {p: _sha(p) for p in _products(out)} == shas_before, \
        "scorer 阶段失败改动了旧成功产物"
    assert glob.glob(out + ".failure-*.json"), "failure receipt 未落盘"
    assert not glob.glob(out + ".staging-*")
    print("E5 PASS  scorer 阶段 answers hash 错配重跑 → 非零退出 + 旧产物"
          " SHA 不变 + failure receipt（混合代际状态不可达）")


def test_E6_length_identity(base):
    """E6 负例（032）：① 混入 length=131072 行进 L32768 目录 → 长度身份
    门禁拒；② 同 row index/answers 不同 length 的两 method（32768 vs
    32000，均不越界）→ 跨 method 逐行 length 一致性拒。"""
    # ① 上界门禁
    root = _copy_fixture(base, "e6a_root")
    tgt = os.path.join(root, "L32768", "pred_fx", "vt-fxm-01010000.jsonl")
    rows = [json.loads(l) for l in open(tgt, encoding="utf-8")]
    for rr in rows:
        rr["length"] = 131072
    with open(tgt, "w", encoding="utf-8") as f:
        for rr in rows:
            f.write(json.dumps(rr, ensure_ascii=False) + "\n")
    out = os.path.join(base, "e6a.json")
    r = _formal(root, out, min_samples=2)
    assert r.returncode != 0 and "长度身份门禁" in (r.stdout + r.stderr), \
        (r.returncode, r.stdout[-2000:], r.stderr[-1000:])
    assert not any(os.path.exists(p) for p in _products(out))
    assert not glob.glob(out + ".staging-*")
    # ② 跨 method 逐行 length 一致性（两 method 均在档位内）
    root = _copy_fixture(base, "e6b_root")
    src = os.path.join(root, "L32768", "pred_fx", "vt-fxm-01010000.jsonl")
    rows = [json.loads(l) for l in open(src, encoding="utf-8")]
    alt = [dict(rr, length=32000) for rr in rows]
    with open(os.path.join(root, "L32768", "pred_fx",
                           "vt-fxm2-01020000.jsonl"), "w",
              encoding="utf-8") as f:
        for rr in alt:
            f.write(json.dumps(rr, ensure_ascii=False) + "\n")
    out = os.path.join(base, "e6b.json")
    r = _formal(root, out, min_samples=2)
    assert r.returncode != 0 and \
        "逐行 length 不一致" in (r.stdout + r.stderr), \
        (r.returncode, r.stdout[-2000:], r.stderr[-1000:])
    assert not any(os.path.exists(p) for p in _products(out))
    print("E6 PASS  ①131072 行混入 L32768 目录拒（上界门禁）；②两 method "
          "同 ids/answers 不同 length 拒（跨 method 逐行一致性）")


def test_E7_out_suffix(base):
    """E7 负例（033）：--out 无后缀 / 大写 .JSON / 路径中间含 .json /
    --manifest-out 与 --out 相同 → 全拒且无 staging 残留。"""
    root = os.path.join(TESTDATA, "pred_root")
    bad_outs = [os.path.join(base, "no_suffix"),
                os.path.join(base, "upper.JSON"),
                os.path.join(base, "mid.json", "result"),
                os.path.join(base, "dotjsonl.jsonl")]
    for out in bad_outs:
        r = _formal(root, out, min_samples=2)
        assert r.returncode != 0, (out, r.returncode)
        assert ".json" in (r.stdout + r.stderr), out
        assert not glob.glob(out + ".staging-*"), f"staging 残留: {out}"
        assert not glob.glob(out + ".run-*"), f"run 目录残留: {out}"
    # --manifest-out 与 --out 相同 → 四路径两两不同断言拒
    out = os.path.join(base, "valid.json")
    r = _formal(root, out, min_samples=2, manifest_out=out)
    assert r.returncode != 0 and "两两不同" in (r.stdout + r.stderr), \
        (r.returncode, r.stdout[-1500:])
    assert not glob.glob(out + ".staging-*")
    # 合法后缀路径本身没有被发布（此前的失败路径都不写产物）
    assert not os.path.exists(out)
    print("E7 PASS  --out 无后缀/.JSON/中间含 .json/.jsonl + manifest-out"
          "==out → 全拒 + 无 staging 残留")


def _run_closure(base_dir, out, arms, optimized):
    """以 monkeypatch 方式驱动 closure v2 脚本（可 -O 模式运行）。"""
    code = (
        "import importlib.util\n"
        f"spec = importlib.util.spec_from_file_location("
        f"'closure_v2', {CLOSURE_SCRIPT!r})\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
        f"mod.BASE = {base_dir!r}\n"
        f"mod.OUT = {out!r}\n"
        f"mod.ARMS = {arms!r}\n"
        "mod.main()\n"
    )
    argv = [sys.executable] + (["-O"] if optimized else []) + ["-c", code]
    return subprocess.run(argv, capture_output=True, text=True, cwd=REPO,
                          env={**os.environ, "PYTHONPATH": REPO})


def _mk_closure_fixture(base, name, rows_per_cell):
    """closure v2 的 monkeypatch fixture：{name}/{arm}/{task}-m-ts.jsonl。"""
    base_dir = os.path.join(base, name)
    arms = ["armA", "armB", "armC"]
    for arm in arms:
        os.makedirs(os.path.join(base_dir, arm))
        for task in TASKS:
            with open(os.path.join(base_dir, arm,
                                   f"{task}-m-01010000.jsonl"), "w",
                      encoding="utf-8") as f:
                for i in range(rows_per_cell):
                    f.write(json.dumps({
                        "pred": f"the answer is gt{i}",
                        "answers": [f"gt{i}"],
                        "length": 32768, "budget": 0,
                    }, ensure_ascii=False) + "\n")
    return base_dir, arms


def test_E8_closure_opt_mode(base):
    """E8（034）：closure v2 在普通与 python -O 两种模式下行为一致——
    33×1 负例都非零退出不写结果；33×100 正例都成功且 verdict 一致。"""
    # 负例：33 格各 1 行
    neg_dir, arms = _mk_closure_fixture(base, "closure_neg", 1)
    neg_out = os.path.join(base, "closure_neg.json")
    for optimized in (False, True):
        r = _run_closure(neg_dir, neg_out, arms, optimized)
        assert r.returncode != 0, \
            (optimized, r.returncode, r.stdout[-1500:], r.stderr[-800:])
        assert "CLOSURE-FAIL" in (r.stdout + r.stderr)
        assert not os.path.exists(neg_out), \
            f"python{' -O' if optimized else ''} 负例写了结果文件"
    # 正例：33 格各 100 行
    pos_dir, arms = _mk_closure_fixture(base, "closure_pos", 100)
    pos_out = os.path.join(base, "closure_pos.json")
    for optimized in (False, True):
        r = _run_closure(pos_dir, pos_out, arms, optimized)
        assert r.returncode == 0, \
            (optimized, r.returncode, r.stdout[-2000:], r.stderr[-800:])
        assert os.path.exists(pos_out)
        d = json.load(open(pos_out))
        v = d["verdict"]
        assert v["cells_complete"] is True and \
            v["cross_arm_answers_rowwise_identical"] is True and \
            v["within_file_zero_duplicate"] is True
        assert d["cell_count"]["selected_rows_total"] == 3300
        # 034：描述文本从 verdict 派生（不再硬编码「断言通过」）
        assert "cells_complete=True" in d["cell_count"]["assertion"] and \
            "= 3300" in d["cell_count"]["assertion"]
    print("E8 PASS  closure v2：33×1 负例在普通与 python -O 模式都非零退出"
          "不写结果；33×100 正例两模式都成功且 verdict/派生文本一致")


def test_E9_legacy_source_immutable(base):
    """E9（036）：legacy 源文件哈希补刻前后逐位不变；补刻只发生在派生
    副本且逐字段不变量成立（pred/answers/length/budget）。"""
    before = {f: _sha(f) for f in _fixture_files()}
    root = os.path.join(TESTDATA, "pred_root")
    out = os.path.join(base, "e9.json")
    r = _formal(root, out, min_samples=2)
    assert r.returncode == 0 and "DONE" in r.stdout, r.stdout[-2000:]
    after = {f: _sha(f) for f in _fixture_files()}
    assert before == after, \
        f"入库 fixture 源文件被改动（036 回归）: " \
        f"{[f for f in before if before[f] != after.get(f)]}"
    # 派生副本逐字段不变量 + receipt 的 source→derived 追溯
    rc = json.load(open(out + ".receipt.json"))
    for task, cell in rc["cells"]["L32768/fxm"]["tasks"].items():
        src_row = json.loads(open(cell["source_path"],
                                  encoding="utf-8").readline())
        assert src_row.get("_id") is None, "源文件被原地补刻"
        assert len(cell["source_sha256"]) == 64 and \
            len(cell["derived_sha256"]) == 64
    runs = glob.glob(out + ".run-*")
    stamped = json.loads(open(os.path.join(
        runs[0], "pred_root", "L32768", "pred_fx",
        "cwe-fxm-01010000.jsonl"), encoding="utf-8").readline())
    assert stamped["_id"] == "cwe:0" and stamped["_answers_sha"]
    src_row = json.loads(open(os.path.join(
        TESTDATA, "pred_root", "L32768", "pred_fx",
        "cwe-fxm-01010000.jsonl"), encoding="utf-8").readline())
    for k in ("pred", "answers", "length", "budget"):
        assert stamped[k] == src_row[k], f"派生副本字段 {k} 漂移"
    print("E9 PASS  legacy 源文件哈希前后逐位不变 + 派生副本含 _id 且四"
          "业务字段逐位不变 + receipt 记录 source→derived SHA256")


def test_E10_clean_clone_semantics(base):
    """E10（035）：clean clone 语义——D9 显式 --with-real-data 且真实数据
    缺失（env 指向不存在路径模拟）→ 打印 SKIP 且退出码 0，不计门禁。"""
    code = ("from benchmark.RULER.test_e116e_gate import run_d9\n"
            "import sys\n"
            "sys.exit(run_d9())")
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        cwd=REPO, env={**os.environ, "PYTHONPATH": REPO,
                       "E116E_REAL_ROOT": os.path.join(base, "no_such_dir")})
    assert r.returncode == 0 and "D9 SKIP" in r.stdout, \
        (r.returncode, r.stdout[-1500:], r.stderr[-800:])
    print("E10 PASS 缺真实数据时 D9 显式 SKIP + 退出码 0（不计入程序门禁，"
          "clean clone 可独立复现 E1-E9+D10）")


def run_d9():
    """D9（真实数据集成测试，E116e 起需显式 --with-real-data）：
    三臂 32K 生产数据直接走新正式入口（只读），AVG 与历史逐位对照
    （FULLKV=59.38 / mavg=59.99 / aavg=57.33），生产文件哈希前后不变。
    返回进程退出码（缺数据时打印 SKIP 并返回 0）。"""
    if not (os.path.isdir(os.path.join(REAL_ROOT, "L32768")) and
            os.path.isdir(os.path.join(REAL_DATA_ROOT, "32768"))):
        print("D9 SKIP  无真实 32K 三臂数据/源数据（外部数据集成测试，"
              "不计入程序门禁；需显式 --with-real-data 才会尝试运行）")
        return 0
    base = tempfile.mkdtemp(prefix="e116e_d9_")
    try:
        prod_files = sorted(glob.glob(os.path.join(
            REAL_ROOT, "L32768", "pred_E109_*", "*.jsonl")))
        assert prod_files, f"生产目录零 pred 文件: {REAL_ROOT}"
        before = {f: _sha(f) for f in prod_files}
        avgs = {}
        for tag, postfix in [("FULLKV", "_E109_FULLKV"),
                             ("mavg", "_E109_mavg_a0.25_b0.125_g0.625"),
                             ("aavg", "_E109_aavg_a0_b0_g0")]:
            out = os.path.join(base, f"formal_{tag}.json")
            r = _formal(REAL_ROOT, out, data_root=REAL_DATA_ROOT,
                        postfix=postfix)
            assert r.returncode == 0 and "DONE" in r.stdout, \
                (tag, r.returncode, r.stdout[-3000:], r.stderr[-2000:])
            d = json.load(open(out))
            mkey = [k for k in d["scores"]][0]
            scores = d["scores"][mkey]
            assert len(scores) == 11, (tag, scores.keys())
            avgs[tag] = round(sum(scores.values()) / 11, 2)
            rc = json.load(open(out + ".receipt.json"))
            assert rc["status"] == "success" and len(rc["run_id"]) > 8
            assert len(rc["source_data_sha256"]) == 11
            # E116f（审计建议 5）+ E116j（052）：真实数据只读重跑的
            # generation 闭合——receipt 声明 SHA ↔ 公开镜像 ↔ generation
            # 目录规范文件三方逐位一致 + 发布协议版本字段。v3（E116j）：
            # 公开 receipt = entry receipt（含 gen_md_sha256/
            # gen_receipt_sha256），与 generation 内 receipt.json 是两个
            # 不同文件——公开 receipt 逐位等于 generation 的
            # entry_receipt.json，且两内容哈希与 generation 实际文件闭合
            assert rc["publish_protocol"] == "e116i-generation-v3", \
                (tag, rc.get("publish_protocol"))
            assert rc["result_sha256"] == _sha(out) and \
                rc["manifest_sha256"] == _sha(out + ".manifest.json")
            g = rc["outputs"]["derived_dir"]
            assert os.path.isdir(g)
            assert _sha(os.path.join(g, "result.json")) == _sha(out) and \
                _sha(os.path.join(g, "manifest.json")) == \
                _sha(out + ".manifest.json") and \
                _sha(os.path.join(g, "entry_receipt.json")) == \
                _sha(out + ".receipt.json"), (tag, "generation 不闭合")
            assert rc["gen_md_sha256"] == \
                _sha(os.path.join(g, "result.md")) and \
                rc["gen_receipt_sha256"] == \
                _sha(os.path.join(g, "receipt.json")), \
                (tag, "052 内容绑定不闭合")
        assert avgs == {"FULLKV": 59.38, "mavg": 59.99, "aavg": 57.33}, avgs
        after = {f: _sha(f) for f in prod_files}
        assert before == after, "生产 pred 文件哈希前后不一致（036 回归）"
        print(f"D9 PASS  三臂真实 32K 生产数据直接走新正式入口（只读）："
              f"AVG FULLKV={avgs['FULLKV']} / mavg={avgs['mavg']} / "
              f"aavg={avgs['aavg']}（与历史逐位一致）；生产 pred 文件哈希"
              f"前后不变（{len(prod_files)} 个文件）；receipt↔镜像↔"
              f"generation 三方 SHA 闭合 + 052 内容绑定（publish_protocol="
              f"e116i-generation-v3）")
        return 0
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_D10_no_regression():
    """D10：既有 E116c/E116d 套件不回归。"""
    for mod, needle in [
            ("benchmark.RULER.test_e116c_gate", "E116c ALL PASS (11/11)"),
            ("benchmark.RULER.test_e116d_gate", "E116d ALL PASS")]:
        r = subprocess.run([sys.executable, "-m", mod],
                           capture_output=True, text=True, cwd=REPO,
                           env={**os.environ, "PYTHONPATH": REPO})
        assert r.returncode == 0 and needle in r.stdout, \
            (mod, r.returncode, r.stdout[-3000:], r.stderr[-2000:])
    print("D10 PASS test_e116c_gate.py 11/11 + test_e116d_gate.py 无回归")


def main():
    global PASS
    import argparse
    ap = argparse.ArgumentParser(description="E116e 红绿测试")
    ap.add_argument("--with-real-data", action="store_true",
                    help="额外运行 D9 真实数据集成测试（缺数据时 SKIP）")
    args = ap.parse_args()

    base = tempfile.mkdtemp(prefix="e116e_")
    try:
        test_E1_min_samples_hard_gate(base)
        test_E2_missing_task(base)
        test_E3_positive(base)
        test_E4_fail_keeps_old_products(base)
        test_E5_scorer_stage_failure(base)
        test_E6_length_identity(base)
        test_E7_out_suffix(base)
        test_E8_closure_opt_mode(base)
        test_E9_legacy_source_immutable(base)
        test_E10_clean_clone_semantics(base)
        PASS = 11
        if args.with_real_data:
            rc = run_d9()
            assert rc == 0
            PASS = 12
        test_D10_no_regression()
        PASS += 1
    finally:
        shutil.rmtree(base, ignore_errors=True)
    print(f"\nE116e ALL PASS ({PASS}/{12 + 1}"
          f"{'，D9 SKIP' if PASS == 12 else ''})")


if __name__ == "__main__":
    main()
