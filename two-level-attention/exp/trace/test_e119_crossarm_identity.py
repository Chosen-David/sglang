#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E119 跨臂身份门禁红绿测试（041 + E116h 045/046/047/048①）。

048① 可移植性：本测试不再依赖仓库外的生产 prediction JSONL——改用已入库
的全合成最小 fixture（exp/trace/testdata/e119_min/，由
gen_e119_min_fixture.py 生成：4 任务 × 3 行/任务、三臂 treatment 与
ARM_CONTRACT 逐字段一致、全部 SHA 闭合、legacy receipt），干净检出即可
运行。analyzer 通过 --results-dir/--pred-root 只读指向临时副本，fixture
与生产数据零写入。

用例矩阵：

  P1 正例      fixture 三臂原样 → 汇总通过，arms 数值 40.0/35.0/30.0，
               identity_gate 段 + arm_contract + 逐臂 legacy_protocol 标注，
               结论动态生成（mavg 冠军 +5.00 / aavg 居末 -5.00）；
  N1 _id       篡改单臂 manifest 一个 task 的 ids[0] → 拒绝（exit≠0）；
  N2 answers   篡改单臂 manifest 一个 answers_sha 值 → 拒绝；
  N3 length    篡改单臂 manifest 一个 lengths[k] → 拒绝；
  N4 源数据    篡改单臂 manifest source_data_sha256 一个 SHA → 拒绝；
  N5 模型      篡改单臂 manifest run_identity.model_path（+receipt 同步）
               → 拒绝；
  N6 多 cell   单臂 result JSON 塞入第二个 cell key（+receipt result
               SHA 同步）→ fail-closed 拒绝（next(iter) 静默取首键修复）；
  N7 treatment 交换 mavg/aavg 的 treatment（046②：把 aavg 配置冠到 mavg
               臂名下，receipt 同步闭合）→ arm 契约 fail-closed 拒绝；
  N8 脚本SHA   篡改单臂 formal_script_sha256（047：评分口径公平）→
               fail-closed 拒绝（不再是 warn-only）；
  P2 结论动态  篡改 aavg result 分数使其反超（+receipt SHA 同步）→
               通过，但 verdict.ruler64k_ranking/conclusion 必须跟随新
               排序（aavg 冠军 +10.00），不得残留硬编码 mavg 冠军文本
               （046④）；
  O1 python -O 两类篡改（receipt SHA 不一致 / 跨臂 answers_sha）在
               python -O 下重跑 → 仍非零退出且不覆盖旧 summary
               （045：门禁全部为显式条件 + _fail，assert 零依赖）。

E116h 049（GPT 1531 审计 TL-E119-SUMMARY-ATOMICITY-049）summary 原子发布：
  N9a 写中断   注入 json.dump 写出 '{"partial":' 前缀后抛 OSError →
               非零退出、旧 summary SHA 逐位不变且仍可解析、无 .tmp
               残留（修复前 open(p_out,"w") 直接截断毁掉 last-known-good）；
  N9b 并发发布 双汇总器（一慢速持锁写、一正常）并发写同一 summary →
               两进程均成功，最终文件为某一完整代际（与单跑逐字节
               一致），无混写/截断/临时残留；
  O2 python -O N9a 同款写中断在 python -O 下重跑 → 仍非零退出且旧
               summary 逐位不变（原子发布不依赖 assert）。

所有负例还断言「不覆盖旧 summary」：失败运行前放置哨兵 summary 文件，
失败后内容逐位不变。

用法：
  python3 exp/trace/test_e119_crossarm_identity.py
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

# 049 故障注入 wrapper：importlib 加载 analyzer 模块，按模式 monkeypatch
# json.dump（analyzer 唯一的 json.dump 调用点 = summary 原子发布）。
#   interrupt —— 写出 '{"partial":' 前缀后抛 OSError（049 审计原始反例）
#   slow      —— 持锁慢速写（time.sleep 后真写），驱动并发发布窗口
_WRAPPER = '''
import json, os, sys, importlib.util
MODE, SCRIPT = sys.argv[1], sys.argv[2]
args = sys.argv[3:]
spec = importlib.util.spec_from_file_location("e119_analyzer", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
real_dump = json.dump
if MODE == "interrupt":
    def boom(obj, fh, *a, **k):
        fh.write('{"partial":')
        fh.flush()
        raise OSError("injected-write-interruption")
    json.dump = boom
elif MODE == "slow":
    def slow(obj, fh, *a, **k):
        import time
        time.sleep(float(os.environ.get("E119_SLOW_DUMP", "3")))
        real_dump(obj, fh, *a, **k)
    json.dump = slow
sys.argv = [SCRIPT] + args
mod.main()
'''

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
FIXTURE = os.path.join(REPO, "exp", "trace", "testdata", "e119_min")
GEN_SCRIPT = os.path.join(FIXTURE, "gen_e119_min_fixture.py")
SCRIPT = os.path.join(REPO, "exp", "trace",
                      "analyze_e119_ruler64k_formal.py")
SUMMARY = "e119_ruler64k_formal_summary.json"
# fixture 合成分数（与生产排序方向一致：mavg > FullKV > aavg）
FIXTURE_AVG = {"mavg": 40.0, "FullKV": 35.0, "aavg": 30.0}
SENTINEL = {"sentinel": "old-summary-must-not-be-overwritten"}

PASS = 0


def _sha(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fixture_usable():
    """fixture 完整性探测：receipt 三件套 + 源预测 jsonl 均在位才可用
    （防止 .gitignore 漏白名单或异常检出导致半套 fixture 误判可用）。"""
    probes = [
        os.path.join(FIXTURE, "results",
                     "e119_ruler64k_formal_mavg.json.receipt.json"),
        os.path.join(FIXTURE, "pred_root", "mavg", "L65536", "pred_1024"),
        os.path.join(FIXTURE, "pred_root", "fullkv", "L65536", "pred_1024"),
        os.path.join(FIXTURE, "pred_root", "aavg", "L65536", "pred_1024"),
    ]
    for p in probes:
        if not (os.path.isfile(p) or os.path.isdir(p)):
            return False
    return len(glob.glob(os.path.join(
        probes[1], "*.jsonl"))) == 4


def _ensure_fixture(root_tmp):
    """048①：fixture 源目录——已入库且完整则直接用；缺失（异常检出/
    漏白名单）则用已入库生成器现场重建到临时目录，保证干净路径可跑。"""
    if _fixture_usable():
        return FIXTURE
    dest = os.path.join(root_tmp, "fixture_regen")
    os.makedirs(dest, exist_ok=True)
    r = subprocess.run([sys.executable, GEN_SCRIPT, dest],
                       capture_output=True, text=True)
    assert r.returncode == 0, (r.returncode, r.stdout[-500:], r.stderr[-500:])
    return dest


def _copy_fixture(base, fixture):
    """三臂产物完整副本（results 三件套 + generation 派生目录 + 源预测）。"""
    res = os.path.join(base, "results")
    shutil.copytree(os.path.join(fixture, "results"), res)
    shutil.copytree(os.path.join(fixture, "pred_root"),
                    os.path.join(base, "pred_root"))


def _run(base, opt_o=False):
    """对副本目录跑汇总脚本（--results-dir/--pred-root 只读指向副本）。"""
    cmd = [sys.executable] + (["-O"] if opt_o else []) + [
        SCRIPT, "--results-dir", os.path.join(base, "results"),
        "--pred-root", os.path.join(base, "pred_root")]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)


def _manifest_path(base, arm):
    return os.path.join(base, "results",
                        f"e119_ruler64k_formal_{arm.lower()}.json"
                        ".manifest.json")


def _receipt_path(base, arm):
    return _manifest_path(base, arm).replace(".manifest.json",
                                             ".receipt.json")


def _result_path(base, arm):
    return os.path.join(base, "results",
                        f"e119_ruler64k_formal_{arm.lower()}.json")


def _tamper_manifest(base, arm, mutate, also_receipt_identity=None):
    """单臂篡改 + 同步修补 receipt 闭合（模拟「内部闭合但身份错配」）。

    mutate(manifest) 就地修改 manifest；随后重写 receipt.manifest_sha256
    使 receipt↔manifest SHA 闭合——确保拒绝来自跨臂身份/契约门禁而非
    receipt 闭合检查。also_receipt_identity(receipt) 可同步篡改
    receipt.inputs.run_identity。"""
    mp = _manifest_path(base, arm)
    rp = _receipt_path(base, arm)
    m = json.load(open(mp))
    mutate(m)
    json.dump(m, open(mp, "w"), indent=1, ensure_ascii=False)
    r = json.load(open(rp))
    r["manifest_sha256"] = _sha(mp)
    if also_receipt_identity is not None:
        also_receipt_identity(r)
    json.dump(r, open(rp, "w"), indent=1, ensure_ascii=False)


def _place_sentinel_summary(base):
    p = os.path.join(base, "results", SUMMARY)
    json.dump(SENTINEL, open(p, "w"))
    return p


def _expect_reject(base, tag, opt_o=False, needle=None):
    """负例断言：exit≠0 + 门禁标记 + 旧 summary 未被覆盖。"""
    sp = _place_sentinel_summary(base)
    r = _run(base, opt_o=opt_o)
    assert r.returncode != 0, \
        f"{tag}: 篡改后仍 exit=0（跨臂身份门禁失效）：\n{r.stdout[-800:]}"
    assert ("E119-GATE-FAIL" in r.stderr) or ("跨臂" in r.stderr) or \
           ("cell" in r.stderr), (tag, r.stderr[-800:])
    if needle is not None:
        assert needle in r.stderr, (tag, needle, r.stderr[-900:])
    cur = json.load(open(sp))
    assert cur == SENTINEL, f"{tag}: 失败运行覆盖了旧 summary"
    mode = "python -O " if opt_o else ""
    print(f"  {tag} PASS  {mode}exit={r.returncode}，旧 summary 未覆盖；"
          f"拒绝原因: {r.stderr.strip().splitlines()[-1][:110]}")
    return True


def test_P1_positive(base):
    """P1：fixture 三臂原样 → 通过 + identity_gate/arm 契约 + 数值不变。"""
    r = _run(base)
    assert r.returncode == 0, (r.returncode, r.stdout[-1500:],
                               r.stderr[-1000:])
    s = json.load(open(os.path.join(base, "results", SUMMARY)))
    assert {a: v["avg"] for a, v in s["arms"].items()} == FIXTURE_AVG, \
        s["arms"]
    assert {a: v["legacy_protocol"] for a, v in s["arms"].items()} == \
        {"mavg": True, "FullKV": True, "aavg": True}, s["arms"]
    ig = s["identity_gate"]
    assert ig["protocol"] == "e116g-crossarm-identity-v1"
    assert ig["all_arms_data_identity_identical"] is True
    assert len(ig["common_identity_digest"]) == 64
    assert ig["script_sha_identical"] is True
    # 046②：arm 契约逐字段入 summary，且 per_arm_treatment 与契约相等
    assert ig["arm_contract"]["mavg"] == {
        "far_method": "minmax", "near_method": "avg",
        "alpha": "0.25", "beta": "0.125", "gamma": "0.625"}
    assert ig["arm_contract"]["FullKV"] == {"method": "none"}
    for arm, treat in ig["per_arm_treatment"].items():
        assert treat == ig["arm_contract"][arm], (arm, treat)
    # 046③：legacy receipt 显式标注，不得声称发布锁协议已作用
    for arm, proto in ig["per_arm_receipt_protocol"].items():
        assert proto["legacy_protocol"] is True, (arm, proto)
        assert proto["publish_protocol"] is None, (arm, proto)
    assert "不声称" in ig["protocol_note"]
    # 046④：结论从结构化数值动态生成（fixture 口径 mavg +5.00 冠军）
    assert s["identity"]["samples_per_task"] == 3
    assert s["identity"]["tasks"] == 4
    assert s["verdict"]["ruler64k_ranking"] == \
        "mavg 40.0 > FullKV 35.0 > aavg 30.0"
    assert "mavg（vs FullKV +5.00）居首" in s["verdict"]["conclusion"]
    assert "aavg（-5.00）居末" in s["verdict"]["conclusion"]
    assert "方向一致" in s["verdict"]["conclusion"]
    print("P1 PASS  fixture 三臂 → 汇总通过；arms 40.0/35.0/30.0；"
          "arm 契约逐臂记录且与 treatment 相等；三臂 legacy_protocol "
          "如实标注；结论从结构化 ranking 动态生成")


def test_N1_tamper_id(base):
    """N1（041 建议 4）：单臂一个 _id 篡改 → 拒绝发布。"""
    _tamper_manifest(base, "aavg", lambda m: m["tasks"]
                     ["niah_single_1"]["ids"].__setitem__(0, "tampered:999"))
    _expect_reject(base, "N1(_id)")


def test_N2_tamper_answers_sha(base):
    """N2：单臂一个 answers_sha 篡改 → 拒绝发布。"""
    def mut(m):
        m["tasks"]["niah_single_1"]["answers_sha"]["niah_single_1:0"] = \
            "deadbeefdeadbeef"
    _tamper_manifest(base, "FullKV", mut)
    _expect_reject(base, "N2(answers_sha)")


def test_N3_tamper_length(base):
    """N3：单臂一个逐行 length 篡改 → 拒绝发布。"""
    def mut(m):
        m["tasks"]["vt"]["lengths"][0] = m["tasks"]["vt"]["lengths"][0] + 1
    _tamper_manifest(base, "mavg", mut)
    _expect_reject(base, "N3(lengths)")


def test_N4_tamper_source_data_sha(base):
    """N4：单臂 source_data_sha256 一个 SHA 篡改 → 拒绝发布。"""
    def mut(m):
        key = sorted(m["source_data_sha256"])[0]
        m["source_data_sha256"][key]["sha256"] = "0" * 64
    _tamper_manifest(base, "aavg", mut)
    _expect_reject(base, "N4(source_data_sha256)")


def test_N5_tamper_model_identity(base):
    """N5：单臂模型身份篡改（manifest + receipt 同步）→ 拒绝发布。"""
    def mut(m):
        m["run_identity"]["model_path"] = "/home/other/Qwen3-8B-tampered"
    def mut_rc(r):
        r["inputs"]["run_identity"]["model_path"] = \
            "/home/other/Qwen3-8B-tampered"
    _tamper_manifest(base, "mavg", mut, also_receipt_identity=mut_rc)
    _expect_reject(base, "N5(model_path)")


def test_N6_multi_cell(base):
    """N6（041 建议 4）：单臂 result JSON 多 cell → fail-closed 拒绝
    （修 next(iter(d["n"])) 静默取首键假设 bug）。"""
    jp = _result_path(base, "FullKV")
    rp = _receipt_path(base, "FullKV")
    d = json.load(open(jp))
    key = next(iter(d["n"]))
    fake = key + "_SECOND_L"
    d["n"][fake] = dict(d["n"][key])
    d["scores"][fake] = dict(d["scores"][key])
    json.dump(d, open(jp, "w"), indent=1, ensure_ascii=False)
    r = json.load(open(rp))
    r["result_sha256"] = _sha(jp)
    json.dump(r, open(rp, "w"), indent=1, ensure_ascii=False)
    _expect_reject(base, "N6(multi-cell)")


def test_N7_treatment_swap(base):
    """N7（046②）：交换 treatment——把 aavg 配置（far=avg/α=β=γ=0）
    冠到 mavg 臂名下（manifest + receipt 同步闭合）→ arm 契约 fail-closed
    拒绝。修复前唯一通道是篡改 treatment，此门禁使其不可达。"""
    def mut(m):
        m["run_identity"]["extra_params"] = {
            "far_method": "avg", "near_method": "avg",
            "alpha": "0", "beta": "0", "gamma": "0"}
    def mut_rc(r):
        r["inputs"]["run_identity"]["extra_params"] = {
            "far_method": "avg", "near_method": "avg",
            "alpha": "0", "beta": "0", "gamma": "0"}
    _tamper_manifest(base, "mavg", mut, also_receipt_identity=mut_rc)
    _expect_reject(base, "N7(treatment-swap)", needle="046②")


def test_N8_script_sha_fail_closed(base):
    """N8（047）：单臂 formal_script_sha256 篡改（+receipt 同步闭合）→
    fail-closed 拒绝——评分口径公平从 warn-only 升格为硬门禁（数据身份
    相同不足以证明 A/B/C 评分公平）。"""
    # 注意：fixture 三臂 FORMAL_SHA 本身是 "f"*64，须篡成不同值才构成反例
    def mut(m):
        m["run_identity"]["formal_script_sha256"] = "a" * 64
    def mut_rc(r):
        r["inputs"]["run_identity"]["formal_script_sha256"] = "a" * 64
    _tamper_manifest(base, "aavg", mut, also_receipt_identity=mut_rc)
    _expect_reject(base, "N8(script-sha)", needle="047")


def test_P2_dynamic_conclusion(base):
    """P2（046④）：篡改 aavg 分数反超 mavg（+receipt result SHA 同步）→
    门禁全过（分数属臂私有、receipt 闭合），但 verdict 必须跟随新排序
    aavg 冠军 +10.00——验证结论从结构化数值动态生成，无硬编码残留。"""
    jp = _result_path(base, "aavg")
    rp = _receipt_path(base, "aavg")
    d = json.load(open(jp))
    key = next(iter(d["n"]))
    d["scores"][key] = {t: 45.0 for t in d["scores"][key]}
    json.dump(d, open(jp, "w"), indent=1, ensure_ascii=False)
    r = json.load(open(rp))
    r["result_sha256"] = _sha(jp)
    json.dump(r, open(rp, "w"), indent=1, ensure_ascii=False)
    rr = _run(base)
    assert rr.returncode == 0, (rr.returncode, rr.stdout[-1200:],
                                rr.stderr[-800:])
    s = json.load(open(os.path.join(base, "results", SUMMARY)))
    assert {a: v["avg"] for a, v in s["arms"].items()} == \
        {"aavg": 45.0, "mavg": 40.0, "FullKV": 35.0}, s["arms"]
    assert s["arms"]["aavg"]["delta_vs_fullkv"] == 10.0
    assert s["verdict"]["ruler64k_ranking"] == \
        "aavg 45.0 > mavg 40.0 > FullKV 35.0"
    assert "aavg（vs FullKV +10.00）居首" in s["verdict"]["conclusion"]
    assert "FullKV（+0.00）居末" in s["verdict"]["conclusion"]
    # 排序与 32K 方向相反 → 结论如实写「不一致」，不得仍称 mavg 冠军稳定
    assert "不一致" in s["verdict"]["conclusion"]
    assert "mavg 冠军" not in s["verdict"]["conclusion"]
    print("P2 PASS  分数改动 → verdict/conclusion 逐字跟随新排序"
          "（aavg 冠军 +10.00、32K 方向不一致如实标注），无硬编码残留")


def test_O1_python_opt(base):
    """O1（045）：python -O 双跑负例——两类篡改在优化模式下仍必须
    非零退出且不覆盖旧 summary（门禁全部为显式条件 + _fail，
    assert 零依赖）。"""
    # 子例 a：aavg result JSON 篡改但不修补 receipt → receipt SHA 不一致
    jp = _result_path(base, "aavg")
    d = json.load(open(jp))
    key = next(iter(d["n"]))
    d["scores"][key]["cwe"] = 13.97
    json.dump(d, open(jp, "w"), indent=1, ensure_ascii=False)
    _expect_reject(base, "O1a(python -O, receipt SHA)", opt_o=True)
    # 子例 b：跨臂 answers_sha 篡改 + receipt 闭合 → 跨臂身份门禁
    base_b = os.path.join(os.path.dirname(base), "fx_O1b")
    os.makedirs(base_b)
    _copy_fixture(base_b, _FIXTURE)
    def mut(m):
        m["tasks"]["cwe"]["answers_sha"]["cwe:0"] = "0123456789abcdef"
    _tamper_manifest(base_b, "aavg", mut)
    _expect_reject(base_b, "O1b(python -O, crossarm identity)", opt_o=True)


def _run_wrapper(base, mode, opt_o=False, extra_env=None, popen=False):
    """049 故障注入：经 wrapper 进程跑 analyzer（monkeypatch json.dump）。"""
    wpath = os.path.join(base, f"wrapper_{mode}.py")
    with open(wpath, "w", encoding="utf-8") as f:
        f.write(_WRAPPER)
    cmd = [sys.executable] + (["-O"] if opt_o else []) + [
        wpath, mode, SCRIPT,
        "--results-dir", os.path.join(base, "results"),
        "--pred-root", os.path.join(base, "pred_root")]
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    if popen:
        return subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True,
                                cwd=REPO, env=env)
    return subprocess.run(cmd, capture_output=True, text=True,
                           cwd=REPO, env=env)


def _assert_no_tmp(base):
    res = os.path.join(base, "results")
    left = [f for f in os.listdir(res) if ".tmp-" in f]
    assert not left, f"summary 临时文件残留: {left}"


def test_N9_summary_atomic(base):
    """N9（049）：summary 原子发布红绿——a 写中断毁不了旧 summary、
    b 并发双发布无混写/截断。"""
    sp = _place_sentinel_summary(base)
    old_sha = _sha(sp)
    # -- a：写中断（049 审计原始反例：写前缀后抛 OSError）--
    r = _run_wrapper(base, "interrupt")
    assert r.returncode != 0, ("N9a: 写中断后仍 exit=0", r.stdout[-500:])
    cur = json.load(open(sp))
    assert cur == SENTINEL, "N9a: 写中断毁掉了旧 summary"
    assert _sha(sp) == old_sha, "N9a: 旧 summary 字节被改动"
    _assert_no_tmp(base)
    print("  N9a(写中断) PASS  exit=%d，旧 summary SHA 逐位不变且可解析，"
          "无 .tmp 残留（049: last-known-good 不再被 open(w) 截断）"
          % r.returncode)
    # -- b：并发双发布（慢速持锁写 × 正常写）--
    # 先单跑一次取「完整代际」期望字节
    r_single = _run(base)
    assert r_single.returncode == 0, (r_single.returncode, r_single.stderr[-500:])
    expected = open(sp, "rb").read()
    procs = [
        _run_wrapper(base, "slow",
                     extra_env={"E119_SLOW_DUMP": "4"}, popen=True),
        _run_wrapper(base, "slow",
                     extra_env={"E119_SLOW_DUMP": "0.5"}, popen=True),
    ]
    rcs = [p.wait() for p in procs]
    assert rcs == [0, 0], ("N9b: 并发发布进程失败", rcs,
                          [p.stderr.read()[-300:] for p in procs])
    assert open(sp, "rb").read() == expected, \
        "N9b: 并发发布后 summary 不是任一完整代际（混写/截断）"
    _assert_no_tmp(base)
    print("  N9b(并发发布) PASS  双汇总器并发（4s/0.5s 持锁写）均成功，"
          "最终文件与单跑逐字节一致，无混写/截断/临时残留")


def test_O2_python_opt_summary_atomic(base):
    """O2（049 建议 4）：N9a 同款写中断在 python -O 下重跑 → 仍非零
    退出且旧 summary 逐位不变（原子发布路径不依赖 assert）。"""
    sp = _place_sentinel_summary(base)
    old_sha = _sha(sp)
    r = _run_wrapper(base, "interrupt", opt_o=True)
    assert r.returncode != 0, ("O2: python -O 写中断后仍 exit=0",
                               r.stdout[-500:])
    assert json.load(open(sp)) == SENTINEL, "O2: python -O 写中断毁掉旧 summary"
    assert _sha(sp) == old_sha
    _assert_no_tmp(base)
    print("  O2(python -O 写中断) PASS  exit=%d，旧 summary 逐位不变，"
          "无 .tmp 残留" % r.returncode)


def main():
    global PASS, _FIXTURE
    root_tmp = tempfile.mkdtemp(prefix="e119_ident_")
    try:
        # 048①：fixture 源（已入库优先，异常检出时现场重建）
        _FIXTURE = _ensure_fixture(root_tmp)
        # 每个用例独立新鲜副本：篡改互不累积，拒绝原因可精确归因到
        # 该用例注入的单字段错配（共享副本会让上一轮篡改先触发门禁）
        cases = [
            ("P1", test_P1_positive),
            ("N1", test_N1_tamper_id),
            ("N2", test_N2_tamper_answers_sha),
            ("N3", test_N3_tamper_length),
            ("N4", test_N4_tamper_source_data_sha),
            ("N5", test_N5_tamper_model_identity),
            ("N6", test_N6_multi_cell),
            ("N7", test_N7_treatment_swap),
            ("N8", test_N8_script_sha_fail_closed),
            ("P2", test_P2_dynamic_conclusion),
            ("O1", test_O1_python_opt),
            ("N9", test_N9_summary_atomic),
            ("O2", test_O2_python_opt_summary_atomic),
        ]
        for tag, fn in cases:
            base = os.path.join(root_tmp, f"fx_{tag}")
            os.makedirs(base)
            _copy_fixture(base, _FIXTURE)
            fn(base)
            PASS += 1
    finally:
        shutil.rmtree(root_tmp, ignore_errors=True)
    print(f"\nE119 crossarm identity ALL PASS ({PASS}/13)")


if __name__ == "__main__":
    main()
