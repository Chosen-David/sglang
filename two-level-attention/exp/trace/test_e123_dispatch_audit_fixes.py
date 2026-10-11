# -*- coding: utf-8 -*-
"""E123 审计修复红绿套件（GPT 4303fb9 audit，2026-10-11 修复 agent 交付）。

覆盖三项 confirmed 缺陷的黑盒验证：
  - TL-E123-SCORER-IDENTITY-085：scorer 身份脱节（analyze_e123_trial_verdict.py）
  - TL-E123-RESUME-PREFIX-086：SKIP 前缀 mismatch（run_e123_trial_dispatch.sh）
  - TL-E2E-FAILMASK-008 新入口：失败仍报告完成（同 dispatch 脚本）

驱动方式（subprocess 黑盒，不 import 被测代码、不依赖 /tmp 下任何现有文件）：
  - analyze：subprocess + 环境变量 E123_RAW_DIR（原始数据临时副本）/
    E123_OUT_JSON（verdict 临时出口）；打分依赖 jieba/rouge（Levenshtein 用例
    另加）以 pip install --target 注入临时目录经 PYTHONPATH 传递。
  - dispatch：源文本重绑定 REPO_ROOT/OUTROOT + PATH 注入 fake python
    （固定退出码），CALL_LOG 记录每次预测调用。

红绿双证模式：
  - 默认（绿）：对当前目录脚本跑，全部断言须 PASS。
  - E123_TEST_BASELINE=1（红）：用 git show 从基线 SHA
    （E123_TEST_BASELINE_SHA，默认 7f7092f54 = 修复前最后版本）提取原始两
    脚本到临时目录，跑同一套断言——每个非 green_only 用例至少一条断言失败
    = 红证成立（缺陷在基线可复现）。基线 analyze 对 eval 子进程硬编码
    PYTHONPATH=REPO:/tmp/e117_extra_pkgs（机器既有通道），红模式下打分依赖
    由此通道供给；通道缺失时相应用例如实 SKIP 记录。
  - green_only 用例（正例对照）在红模式跳过——两种实现下行为一致属预期，
    不参与红证。
  - 断言全部走显式 _check()（python -O 下 assert 被删的项目纪律），
    python / python -O 双跑必须全绿。

用法：
  python test_e123_dispatch_audit_fixes.py                 # 绿（修复后）
  python -O test_e123_dispatch_audit_fixes.py              # 绿 + -O
  E123_TEST_BASELINE=1 python test_e123_dispatch_audit_fixes.py   # 红（基线）
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TLA_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))  # two-level-attention
RAW_SRC = os.path.join(TLA_ROOT, "exp", "trace", "results", "e123_trial_raw")
COMMITTED_VERDICT = os.path.join(
    TLA_ROOT, "exp", "trace", "results", "e123_trial_verdict.json")
ARMS = ["mavg", "cavg_g", "cavg_off", "fullkv"]
# 与 dispatch 脚本一致的任务集与行数门（repobench-p=500 其余=200）
TASKS_AND_ROWS = [("qasper", 200), ("hotpotqa", 200), ("gov_report", 200),
                  ("musique", 200), ("repobench-p", 500)]
BASELINE_SHA = os.environ.get("E123_TEST_BASELINE_SHA", "7f7092f54")
BASELINE_MODE = os.environ.get("E123_TEST_BASELINE", "") == "1"
TESTED_DIR = os.environ.get("E123_TEST_SCRIPT_DIR", SCRIPT_DIR)
ANALYZE_TIMEOUT = int(os.environ.get("E123_TEST_ANALYZE_TIMEOUT", "1800"))


def _check(cond, msg, results):
    """显式断言（python -O 安全）：不依赖 assert，结果收集到 results 列表。"""
    ok = bool(cond)
    results.append((ok, msg))
    print(("  [PASS] " if ok else "  [FAIL] ") + msg)
    return ok


def ensure_pkgs(kind):
    """pip install --target 注入打分依赖到缓存目录（多次运行复用）。

    返回 PYTHONPATH 可用目录；pip 源不可用/装不上 → None（调用方如实记录）。
    版本钉住审计环境（jieba 0.42.1 / rouge 1.0.1 / Levenshtein 0.27.5），
    保证与已提交 verdict / 审计独立复算的数值可比。
    """
    cache_root = os.environ.get("E123_TEST_PKGS_CACHE",
                                "/tmp/e123_test_pkgs_cache")
    d = os.path.join(cache_root, kind)
    spec = {"score": ["jieba==0.42.1", "rouge==1.0.1", "six"],
            "lev": ["Levenshtein==0.27.5", "rapidfuzz"]}[kind]
    probe = ("import jieba, rouge" if kind == "score" else "import Levenshtein")
    if os.path.isdir(d):
        r = subprocess.run([sys.executable, "-c", probe],
                           env={**os.environ, "PYTHONPATH": d},
                           capture_output=True)
        if r.returncode == 0:
            return d
    os.makedirs(d, exist_ok=True)
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps",
         "--target", d] + spec, capture_output=True, text=True)
    if r.returncode != 0:
        print("  [WARN] pip install 失败: " + (r.stderr or r.stdout)[-500:])
        return None
    r = subprocess.run([sys.executable, "-c", probe],
                       env={**os.environ, "PYTHONPATH": d},
                       capture_output=True)
    return d if r.returncode == 0 else None


def baseline_channel_ok():
    """红模式前置：基线 analyze 硬编码的 /tmp/e117_extra_pkgs 通道可用性。"""
    r = subprocess.run(
        [sys.executable, "-c", "import jieba, rouge"],
        env={**os.environ, "PYTHONPATH": "/tmp/e117_extra_pkgs"},
        capture_output=True)
    return r.returncode == 0


def prepare_tested_dir(tb):
    """红模式：git show 提取基线两脚本到临时目录；绿模式：当前目录。"""
    if not BASELINE_MODE:
        return SCRIPT_DIR
    d = os.path.join(tb, "baseline_scripts")
    os.makedirs(d)
    for name in ("analyze_e123_trial_verdict.py",
                 "run_e123_trial_dispatch.sh"):
        r = subprocess.run(
            ["git", "-C", TLA_ROOT, "show",
             f"{BASELINE_SHA}:two-level-attention/exp/trace/{name}"],
            capture_output=True)
        if r.returncode != 0:
            sys.exit("[FATAL] 无法提取基线脚本 " + name + ": "
                     + r.stderr.decode(errors="replace"))
        with open(os.path.join(d, name), "wb") as f:
            f.write(r.stdout)
    return d


def run_analyze(raw_dir, out_json, env_extra):
    """黑盒驱动 analyze 脚本（E123_RAW_DIR/E123_OUT_JSON 环境变量入口）。"""
    env = {k: v for k, v in os.environ.items()
           if k not in ("TLI_SCORER_BACKEND", "E123_RAW_DIR", "E123_OUT_JSON")}
    env.update(env_extra)
    env["E123_RAW_DIR"] = raw_dir
    env["E123_OUT_JSON"] = out_json
    script = os.path.join(TESTED_DIR, "analyze_e123_trial_verdict.py")
    return subprocess.run([sys.executable, script], env=env,
                          capture_output=True, text=True,
                          timeout=ANALYZE_TIMEOUT)


def _dump(r, tail=1500):
    print("  ---- stdout 尾部 ----")
    print(r.stdout[-tail:] if r.stdout else "(空)")
    print("  ---- stderr 尾部 ----")
    print(r.stderr[-tail:] if r.stderr else "(空)")


def _copy_raw(ctb, tag):
    raw = os.path.join(ctb, "raw_" + tag)
    shutil.copytree(RAW_SRC, raw)  # 入库原始数据严格只读，测试一律用副本
    return raw


# ----------------------------------------------------------------------------
# 085：scorer 身份（analyze_e123_trial_verdict.py）
# ----------------------------------------------------------------------------

def case_085_default(ctb):
    name = "085-1 默认后端：verdict scorer 身份如实记录 + 非meta字段回归"
    res = []
    pkgs = ensure_pkgs("score")
    if pkgs is None:
        print("  [SKIP] jieba/rouge pip --target 注入失败——用例如实记为跳过")
        return name, res, True
    raw = _copy_raw(ctb, "default")
    out_json = os.path.join(ctb, "verdict_default.json")
    r = run_analyze(raw, out_json, {"PYTHONPATH": pkgs})
    _check(r.returncode == 0,
           f"analyze 退出码 0（实际 {r.returncode}）", res)
    if r.returncode == 0:
        with open(out_json) as f:
            v = json.load(f)
        _check(v.get("meta", {}).get("scorer_backend") == "difflib:stdlib",
               "verdict meta.scorer_backend == 'difflib:stdlib'（metrics.py "
               "SCORER_BACKEND_ID 格式）", res)
        _check("difflib:stdlib" in str(v.get("meta", {}).get("scorer", "")),
               "verdict meta.scorer 携带实际后端标识（不再写死 'difflib 后端'）",
               res)
        for arm in ARMS:
            with open(os.path.join(raw, "pred_" + arm, "result.json")) as f:
                m = (json.load(f).get("_meta") or {})
            _check(m.get("scorer_backend") == "difflib:stdlib",
                   f"{arm} result.json _meta 刷新为 difflib:stdlib", res)
        with open(COMMITTED_VERDICT) as f:
            ref = json.load(f)
        keys = ("per_task", "input_manifest", "verdict",
                "cavg_g_avg_delta", "cavg_g_ci95", "cavg_g_significant",
                "cavg_off_avg_delta", "cavg_off_ci95", "cavg_off_significant",
                "fullkv_avg_delta", "fullkv_ci95", "fullkv_significant")
        _check(all(v.get(k) == ref.get(k) for k in keys),
               "非 meta 字段与已提交 e123_trial_verdict.json 逐位一致"
               "（默认后端重建不改变任何数值）", res)
    else:
        _dump(r)
        _check(False, "analyze rc!=0，身份断言无法执行（前置失败）", res)
    return name, res, False


def case_085_mismatch(ctb):
    name = "085-2 篡改一臂 _meta 后端 → fail-closed 拒绝（非零退出 + 错误含 mismatch）"
    res = []
    pkgs = ensure_pkgs("score")
    if pkgs is None:
        print("  [SKIP] jieba/rouge pip --target 注入失败——用例如实记为跳过")
        return name, res, True
    raw = _copy_raw(ctb, "mismatch")
    # 伪造身份歧义：一臂 _meta 声明 levenshtein，其余三臂 difflib
    rp = os.path.join(raw, "pred_cavg_g", "result.json")
    with open(rp) as f:
        d = json.load(f)
    d["_meta"]["scorer_backend"] = "levenshtein:0.99.9-fake"
    with open(rp, "w") as f:
        json.dump(d, f, ensure_ascii=False, indent=4)
    out_json = os.path.join(ctb, "verdict_mismatch.json")
    r = run_analyze(raw, out_json, {"PYTHONPATH": pkgs})
    _check(r.returncode != 0,
           f"混后端身份输入被拒，非零退出（实际 {r.returncode}）", res)
    # 注意收紧到 stderr + 完整短语：out_json 文件名含 "mismatch" 会污染
    # 子串检查（红模式曾借此假绿）；sys.exit(str) 的错误信息走 stderr
    _check("scorer backend mismatch" in r.stderr,
           "stderr 错误信息含 'scorer backend mismatch'", res)
    _check(not os.path.exists(out_json),
           "篡改身份下 verdict 不落盘（fail-closed 先于任何打分）", res)
    if r.returncode == 0:
        _dump(r)
    return name, res, False


def case_085_levenshtein(ctb):
    name = "085-3 TLI_SCORER_BACKEND=levenshtein 重跑 → verdict 如实记录 levenshtein 身份"
    res = []
    score_pkgs = ensure_pkgs("score")
    if score_pkgs is None:
        print("  [SKIP] jieba/rouge pip --target 注入失败——用例如实记为跳过")
        return name, res, True
    lev_pkgs = ensure_pkgs("lev")
    if lev_pkgs is None:
        # 审计口径：python-Levenshtein 装不上（pip 源不可用）就跳过并如实记录
        print("  [SKIP] python-Levenshtein 装不上——levenshtein 用例如实记为跳过")
        return name, res, True
    raw = _copy_raw(ctb, "lev")
    out_json = os.path.join(ctb, "verdict_lev.json")
    r = run_analyze(raw, out_json, {
        "PYTHONPATH": score_pkgs + ":" + lev_pkgs,
        "TLI_SCORER_BACKEND": "levenshtein"})
    _check(r.returncode == 0,
           f"levenshtein 重跑退出码 0（实际 {r.returncode}）", res)
    if r.returncode == 0:
        with open(out_json) as f:
            v = json.load(f)
        bid = str(v.get("meta", {}).get("scorer_backend", ""))
        _check(bid.startswith("levenshtein:"),
               f"verdict meta.scorer_backend 如实记录 levenshtein（实际 {bid!r}）",
               res)
        _check("levenshtein:" in str(v.get("meta", {}).get("scorer", "")),
               "verdict meta.scorer 携带 levenshtein 身份（含版本）", res)
        arm_bids = {}
        for arm in ARMS:
            with open(os.path.join(raw, "pred_" + arm, "result.json")) as f:
                m = (json.load(f).get("_meta") or {})
            arm_bids[arm] = m.get("scorer_backend", "")
            _check(arm_bids[arm].startswith("levenshtein:"),
                   f"{arm} result.json 实际以 levenshtein 打分（{arm_bids[arm]!r}）",
                   res)
        _check(len(set(arm_bids.values())) == 1,
               "四臂实际打分后端一致", res)
        # 交叉验证（审计 §085 主审+独立复核双重复算表）：版本吻合时 repobench
        # （唯一 code_sim 任务）分数应逐位复现审计数值。注意 analyze 的
        # per_task 只含三个对比臂，mavg 是锚点、其分数嵌在对比臂条目的
        # "mavg" 字段里
        if bid == "levenshtein:0.27.5":
            audit_tbl = {"mavg": 67.36, "cavg_g": 66.17,
                         "cavg_off": 64.48, "fullkv": 66.37}
            for arm, exp in audit_tbl.items():
                if arm == "mavg":
                    got = v["per_task"]["cavg_g/repobench"]["mavg"]
                else:
                    got = v["per_task"][f"{arm}/repobench"]["arm"]
                _check(abs(got - exp) < 0.005,
                       f"{arm} repobench levenshtein={got} 复现审计独立复算值 "
                       f"{exp}", res)
            # 非 code_sim 任务后端无关：与已提交 verdict 逐位一致
            # （条目含 arm/mavg/delta 三字段，整字典比较即覆盖锚点分数）
            with open(COMMITTED_VERDICT) as f:
                ref = json.load(f)
            same = all(
                v["per_task"][f"{a}/{t}"] == ref["per_task"][f"{a}/{t}"]
                for a in ("cavg_g", "cavg_off", "fullkv")
                for t in ("qasper", "hotpotqa", "gov_report", "musique"))
            _check(same, "四个非 code_sim 任务分数与已提交 verdict 逐位一致"
                         "（后端切换只影响 repobench）", res)
    else:
        _dump(r)
        _check(False, "levenshtein 重跑 rc!=0，身份断言无法执行", res)
    return name, res, False


# ----------------------------------------------------------------------------
# 086 / 008：dispatch 脚本（run_e123_trial_dispatch.sh）
# ----------------------------------------------------------------------------

def _write_rows(path, n):
    with open(path, "w") as f:
        f.write("{}\n" * n)  # 占位行，仅喂行数门（黑盒不冒充真实预测）


def _fixture_complete(outroot):
    """四臂 20 格全部完整——repobench 按 producer 命名（split('-')[0]）写入。"""
    for arm in ARMS:
        d = os.path.join(outroot, "pred_" + arm)
        os.makedirs(d, exist_ok=True)
        for t, rows in TASKS_AND_ROWS:
            _write_rows(os.path.join(d, f"{t.split('-')[0]}-fixture-00000001.jsonl"),
                        rows)


def run_dispatch(ctb, fake_exit, fixture=None):
    """黑盒驱动 dispatch：OUTROOT 重绑定 + PATH 注入 fake python。"""
    outroot = os.path.join(ctb, "out")
    os.makedirs(outroot, exist_ok=True)
    if fixture:
        fixture(outroot)
    bin_dir = os.path.join(ctb, "bin")
    os.makedirs(bin_dir, exist_ok=True)
    fake = os.path.join(bin_dir, "python")
    with open(fake, "w") as f:
        f.write('#!/bin/bash\nprintf \'%%s\\n\' "$*" >> "$CALL_LOG"\nexit %d\n'
                % fake_exit)
    os.chmod(fake, 0o755)
    calls = os.path.join(ctb, "calls.log")
    src_path = os.path.join(TESTED_DIR, "run_e123_trial_dispatch.sh")
    with open(src_path) as f:
        src = f.read()
    for old, new in (
            ("REPO_ROOT=/home/wangyuanshuo02/sglang/two-level-attention",
             "REPO_ROOT=" + TLA_ROOT),
            ("OUTROOT=/tmp/e123_trial", "OUTROOT=" + outroot)):
        if old not in src:
            raise RuntimeError(f"dispatch 源缺少待重绑定行: {old}")
        src = src.replace(old, new)
    script = os.path.join(ctb, "dispatch.sh")
    with open(script, "w") as f:
        f.write(src)
    env = {**os.environ, "PATH": bin_dir + ":" + os.environ.get("PATH", ""),
           "CALL_LOG": calls}
    r = subprocess.run(["bash", script], env=env, capture_output=True,
                       text=True, timeout=300)
    # 0 次调用（全 SKIP）时 fake python 从未执行，calls.log 不存在
    lines = []
    if os.path.exists(calls):
        with open(calls) as f:
            lines = [l for l in f.read().splitlines() if l.strip()]
    return r, lines


def _call_tasks(lines):
    out = []
    for l in lines:
        m = re.search(r"--task (\S+)", l)
        out.append(m.group(1) if m else "?")
    return out


def case_086_all_skip(ctb):
    name = "086-1 四臂 20 格全部完整（repobench 按 producer 命名）→ 20 格全 SKIP、0 次预测调用"
    res = []
    r, calls = run_dispatch(ctb, fake_exit=42, fixture=_fixture_complete)
    tasks = _call_tasks(calls)
    _check(r.returncode == 0, f"全部 SKIP 时退出码 0（实际 {r.returncode}）", res)
    _check(len(calls) == 0,
           f"0 次预测调用（实际 {len(calls)} 次: {tasks}）", res)
    _check(r.stdout.count("SKIP") == 20,
           f"20 格全部 SKIP（实际 SKIP 行数 {r.stdout.count('SKIP')}）", res)
    _check("全部结束" in r.stdout, "全部成功才打印「全部结束」", res)
    if res and not res[0][0]:
        print("  ---- stdout 尾部 ----")
        print("\n".join(r.stdout.splitlines()[-15:]))
    return name, res, False


def case_086_ambiguous(ctb):
    name = "086-2 同臂 repobench 双候选（不同时间戳）→ 显式警告且不静默 SKIP"
    res = []

    def fixture(outroot):
        _fixture_complete(outroot)
        # 同臂同前缀第二个候选（身份歧义：081 后 _h 尾段属正常口径，
        # 同臂同任务多文件即歧义，不得静默任选其一）
        _write_rows(os.path.join(outroot, "pred_mavg",
                                "repobench-dup-00000002.jsonl"), 500)

    r, calls = run_dispatch(ctb, fake_exit=42, fixture=fixture)
    tasks = _call_tasks(calls)
    _check("AMBIGUOUS" in r.stdout, "双候选显式警告（AMBIGUOUS）", res)
    _check(tasks == ["repobench-p"],
           f"歧义格不静默 SKIP，重跑 1 次暴露（实际调用 {tasks}）", res)
    _check(r.returncode != 0,
           f"歧义格重跑失败传播到总退出码（实际 {r.returncode}）", res)
    _check("全部结束" not in r.stdout, "存在失败不打印「全部结束」", res)
    return name, res, False


def case_086_incomplete(ctb):
    name = "086-3 单格未完成（mavg/hotpotqa 缺行）→ 只重跑该格"
    res = []

    def fixture(outroot):
        _fixture_complete(outroot)
        _write_rows(os.path.join(outroot, "pred_mavg",
                                 "hotpotqa-fixture-00000001.jsonl"), 100)

    r, calls = run_dispatch(ctb, fake_exit=42, fixture=fixture)
    tasks = _call_tasks(calls)
    _check(tasks == ["hotpotqa"],
           f"只有未完成格重跑（实际调用 {tasks}）", res)
    _check(r.stdout.count("SKIP") == 19,
           f"其余 19 格 SKIP（实际 {r.stdout.count('SKIP')}）", res)
    _check(r.returncode != 0,
           f"重跑失败传播到总退出码（实际 {r.returncode}）", res)
    _check("全部结束" not in r.stdout, "存在失败不打印「全部结束」", res)
    return name, res, False


def case_008_all_fail(ctb):
    name = "008-1 空目录 + fake python 固定失败（exit 42）×20 → 总退出码非零、不打印「全部结束」"
    res = []
    r, calls = run_dispatch(ctb, fake_exit=42)
    tasks = _call_tasks(calls)
    _check(len(calls) == 20, f"20 次预测全部被调用（实际 {len(calls)}）", res)
    _check(r.returncode != 0,
           f"任一失败总脚本非零退出（实际 {r.returncode}）", res)
    _check("全部结束" not in r.stdout, "失败时不打印「全部结束」", res)
    _check("ARM-FAIL" in r.stdout and "PRED-FAIL" in r.stdout,
           "失败摘要可见（PRED-FAIL/ARM-FAIL）", res)
    if res and not res[1][0]:
        print("  ---- stdout 尾部 ----")
        print("\n".join(r.stdout.splitlines()[-15:]))
    return name, res, False


def case_008_positive(ctb):
    name = "008-2 正例对照：fake python 全成功 → 退出码 0 + 打印「全部结束」（green_only）"
    res = []
    r, calls = run_dispatch(ctb, fake_exit=0)
    tasks = _call_tasks(calls)
    _check(len(calls) == 20, f"20 次预测全部被调用（实际 {len(calls)}）", res)
    _check(r.returncode == 0,
           f"全部成功退出码 0（实际 {r.returncode}）", res)
    _check("全部结束" in r.stdout, "全部成功打印「全部结束」", res)
    return name, res, False


def main():
    global TESTED_DIR
    tb = tempfile.mkdtemp(prefix="e123_audit_suite_")
    TESTED_DIR = prepare_tested_dir(tb)
    print("=" * 78)
    print("E123 审计修复红绿套件（085/086/008）")
    print(f"two-level-attention root: {TLA_ROOT}")
    print(f"被测脚本目录: {TESTED_DIR}"
          + (f"（红证模式，基线 SHA {BASELINE_SHA}）" if BASELINE_MODE
             else "（绿模式：修复后脚本）"))
    print("=" * 78)
    if BASELINE_MODE and not baseline_channel_ok():
        print("[WARN] /tmp/e117_extra_pkgs 通道不可用——基线 analyze 的 eval "
              "子进程无法打分，085 红证将退化为「依赖失败」级（如实记录）")

    cases = [
        (case_085_default, False),
        (case_085_mismatch, False),
        (case_085_levenshtein, False),
        (case_086_all_skip, False),
        (case_086_ambiguous, False),
        (case_086_incomplete, False),
        (case_008_all_fail, False),
        (case_008_positive, True),   # green_only：正例对照，红模式跳过
    ]
    summary = []   # (case_name, n_pass, n_fail, skipped, red_confirmed)
    for fn, green_only in cases:
        ctb = tempfile.mkdtemp(prefix="e123_case_", dir=tb)
        print(f"\n===== {fn.__name__} =====")
        if BASELINE_MODE and green_only:
            print("  [SKIP] green_only 正例对照——两种实现下行为一致属预期，"
                  "不参与红证")
            summary.append((fn.__name__, 0, 0, True, True))
            continue
        try:
            name, res, skipped = fn(ctb)
        except Exception as e:  # noqa: BLE001 —— 崩溃也算证据，如实记录
            import traceback
            traceback.print_exc()
            summary.append((fn.__name__ + "（崩溃）", 0, 1, False, True))
            continue
        n_pass = sum(1 for ok, _ in res if ok)
        n_fail = sum(1 for ok, _ in res if not ok)
        red_confirmed = (not skipped) and n_fail > 0
        if BASELINE_MODE:
            print(f"  [RED-CONFIRMED] {name}" if red_confirmed else
                  f"  [RED-MISSING] {name}——基线未复现缺陷（skipped={skipped}）")
        summary.append((name, n_pass, n_fail, skipped, red_confirmed))

    print("\n" + "=" * 78)
    print("汇总")
    print("=" * 78)
    gate_fail = []
    for name, n_pass, n_fail, skipped, red_confirmed in summary:
        if skipped:
            status = "SKIP（依赖不可用，如实记录）"
        elif BASELINE_MODE:
            status = ("红证成立" if red_confirmed else "红证缺失!")
            if not red_confirmed:
                gate_fail.append(name)
        else:
            status = ("PASS" if n_fail == 0 else f"FAIL ({n_fail} 断言)")
            if n_fail:
                gate_fail.append(name)
        print(f"  {status:24s} {name}  [pass={n_pass} fail={n_fail}]")
    print(f"\n工作目录（含临时副本/verdict，供调试）: {tb}")
    if gate_fail:
        print(f"[{'红证缺失' if BASELINE_MODE else '套件失败'}] "
              f"{len(gate_fail)} 个用例未过门: {gate_fail}")
        sys.exit(1)
    print("[OK] " + ("全部用例红证成立（基线复现缺陷）" if BASELINE_MODE
                     else "全部用例绿（修复后行为符合预期）"))
    sys.exit(0)


if __name__ == "__main__":
    main()
