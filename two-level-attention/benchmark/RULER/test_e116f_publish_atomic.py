# E116f 红绿测试（GPT 1228 审计 038/039 两项 P1 修复回归）
#   TL-RULER-PUBLISH-ATOMICITY-038（四个公开文件逐个 os.replace 不是
#       跨文件事务；中途失败留下混合代际「新 JSON + 旧 receipt」；
#       except 只捕 SystemExit，OSError/EXDEV 继续上抛但已覆盖前几个
#       文件）→ T1：对第 1/2/3/4 个公开替换分别注入 OSError，断言旧
#       commit generation 仍完整可读、没有旧 receipt 配新结果、没有
#       success 提交、failure receipt 落盘、无残留；
#   TL-RULER-DERIVED-COMMIT-039（success receipt 公开后才 rename staging
#       成其声明的 derived_dir；rename 失败时成功 receipt 指向不存在
#       的目录）→ T2：对 generation 最终 rename 注入 OSError，断言不会
#       发布 success pointer/receipt、不存在指向缺失 derived_dir 的
#       提交（有旧产物=旧代际逐位保留；首轮=零产物零提交）；
#   审计建议 3（跨设备 --manifest-out 的 EXDEV 复现路径）→ T3：真实
#       跨文件系统 --manifest-out 正例——新协议下镜像走「目标文件系统
#       临时文件 + 原子替换」，EXDEV 不可达，发布成功且 receipt 与跨
#       设备 manifest SHA 闭合；
#   既有 E116e 套件（E1-E12 + D9 SKIP 基线）不回归 → D1。
# ===== E116g（GPT 1326 审计 TL-RULER-CONCURRENT-PUBLISH-040）=====
#   T4：两个子进程同 --out 确定性交错发布（A 慢速安装者注入延迟、B
#       快速后到者）→ fcntl.flock 发布锁串行化，两进程均 rc=0，最终
#       四固定 aliases 属同一 run_id、receipt result/manifest SHA 与
#       固定文件闭合、后到者完全覆盖前者（最终 JSON 与 B 单独发布
#       逐位一致，无混合代际）；
#   T5（红例）：同一交错调度下把 fcntl.flock 补丁成 no-op（模拟修复
#       前无互斥）→「两进程均成功 + 最终代际闭合」不再成立（B 的镜像
#       交错进 A 的安装窗口，A 的锁内 SHA 终验/回滚被破坏或失败）——
#       证明 T4 的绿灯来自发布锁而非侥幸时序；
#   T6：一方发布中持锁安装中途被 SIGKILL 异常退出 → flock 由内核
#       自动释放（无陈旧锁），另一方正常发布成功且最终四 aliases
#       完整属于后者（被杀进程的 .bak/generation 残留属预期——
#       SIGKILL 无法执行清理，不构成混合代际）。
# 用法:
#   PYTHONPATH=$PWD python3 -m benchmark.RULER.test_e116f_publish_atomic
import glob
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
TESTDATA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "testdata", "e116e")

PASS = 0


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _products(out):
    """正式入口成功发布的四个公开产物路径（兼容镜像）。"""
    return [out, out[:-len(".json")] + ".md",
            out + ".manifest.json", out + ".receipt.json"]


def _gen_files(gen_dir):
    """generation 目录内的四个规范文件。"""
    return [os.path.join(gen_dir, n) for n in
            ("result.json", "result.md", "manifest.json", "receipt.json")]


def _formal(root, out, min_samples=2, manifest_out=None):
    argv = [sys.executable, "-u", "-m", "benchmark.RULER.score_ruler_formal",
            "--root", root, "--pred-postfix", "_fx",
            "--data-root", os.path.join(TESTDATA, "data_root"),
            "--out", out, "--min-samples", str(min_samples)]
    if manifest_out is not None:
        argv += ["--manifest-out", manifest_out]
    return subprocess.run(argv, capture_output=True, text=True, cwd=REPO,
                           env={**os.environ, "PYTHONPATH": REPO})


def _formal_inject(root, out, inject_replace=None, inject_rename=False,
                   min_samples=2):
    """带故障注入的正式入口运行（审计建议 4 的 monkeypatch 方式）：
    在子进程内对 os.replace / os.rename 打计数补丁——发布协议里
    os.replace 恰好按 JSON→MD→manifest→receipt 顺序调用四次（兼容
    镜像安装），os.rename 仅调用一次（generation 提交）。"""
    formal_argv = ["--root", root, "--pred-postfix", "_fx",
                   "--data-root", os.path.join(TESTDATA, "data_root"),
                   "--out", out, "--min-samples", str(min_samples)]
    code = (
        "import os, sys\n"
        "sys.argv = ['score_ruler_formal'] + " + repr(formal_argv) + "\n"
        "nth = " + repr(inject_replace) + "\n"
        "inject_rename = " + repr(inject_rename) + "\n"
        "_rp, _rn = os.replace, os.rename\n"
        "_cnt = {'n': 0}\n"
        "def _replace(a, b):\n"
        "    _cnt['n'] += 1\n"
        "    if nth is not None and _cnt['n'] == nth:\n"
        "        raise OSError('INJECTED os.replace #%s' % nth)\n"
        "    return _rp(a, b)\n"
        "os.replace = _replace\n"
        "if inject_rename:\n"
        "    def _rename(a, b):\n"
        "        raise OSError('INJECTED generation rename failure')\n"
        "    os.rename = _rename\n"
        "import benchmark.RULER.score_ruler_formal as m\n"
        "m.main()\n"
    )
    return subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True, cwd=REPO,
                          env={**os.environ, "PYTHONPATH": REPO})


def _copy_fixture(base, name):
    return shutil.copytree(os.path.join(TESTDATA, "pred_root"),
                           os.path.join(base, name))


# ---- E116g（040）：并发发布交错测试辅助 ----

def _formal_bg(root, out):
    """普通（无注入）正式入口后台子进程。"""
    argv = [sys.executable, "-u", "-m", "benchmark.RULER.score_ruler_formal",
            "--root", root, "--pred-postfix", "_fx",
            "--data-root", os.path.join(TESTDATA, "data_root"),
            "--out", out, "--min-samples", "2"]
    return subprocess.Popen(argv, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, cwd=REPO,
                            env={**os.environ, "PYTHONPATH": REPO})


def _formal_inject_bg(root, out, delay=0.0, marker=None, no_lock=False):
    """带发布阶段延迟/锁禁用的正式入口后台子进程（040 交错测试）。

    注入语义：os.replace 先执行真实替换、再写 marker、再 sleep(delay)
    ——即「该镜像已安装 + 发布者停在发布临界区内」，确定性构造审计
    040 的交错窗口；no_lock=True 时把 fcntl.flock 补丁成 no-op（模拟
    修复前无互斥的红例——子进程内先打补丁再 import 正式入口模块，
    模块级 `import fcntl` 取到的是同一被补丁的模块对象）。"""
    formal_argv = ["--root", root, "--pred-postfix", "_fx",
                   "--data-root", os.path.join(TESTDATA, "data_root"),
                   "--out", out, "--min-samples", "2"]
    code = (
        "import os, sys, time\n"
        "sys.argv = ['score_ruler_formal'] + " + repr(formal_argv) + "\n"
        "delay = " + repr(delay) + "\n"
        "marker = " + repr(marker) + "\n"
        "no_lock = " + repr(no_lock) + "\n"
        "if no_lock:\n"
        "    import fcntl\n"
        "    fcntl.flock = lambda *a, **k: None\n"
        "_rp = os.replace\n"
        "_cnt = {'n': 0}\n"
        "def _replace(a, b):\n"
        "    _cnt['n'] += 1\n"
        "    r = _rp(a, b)\n"
        "    if marker is not None:\n"
        "        with open(marker, 'w') as f:\n"
        "            f.write(str(_cnt['n']))\n"
        "    if delay:\n"
        "        time.sleep(delay)\n"
        "    return r\n"
        "os.replace = _replace\n"
        "import benchmark.RULER.score_ruler_formal as m\n"
        "m.main()\n"
    )
    return subprocess.Popen([sys.executable, "-c", code],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, cwd=REPO,
                            env={**os.environ, "PYTHONPATH": REPO})


def _aliases_closed(out):
    """四个固定 aliases 是否同一代际闭合：receipt success + receipt
    result/manifest SHA 与固定文件逐位一致 + manifest.run_id 与
    receipt.run_id 相同 + generation 目录真实存在。混合代际（040 的
    持久病态）在该口径下必表现为 False。"""
    try:
        rc = json.load(open(out + ".receipt.json"))
        mf = json.load(open(out + ".manifest.json"))
    except (OSError, ValueError):
        return False
    if rc.get("status") != "success":
        return False
    try:
        return (rc.get("result_sha256") == _sha(out) and
                rc.get("manifest_sha256") == _sha(out + ".manifest.json")
                and mf.get("run_id") == rc.get("run_id") and
                os.path.isdir(rc["outputs"]["derived_dir"]))
    except OSError:
        return False


def test_T4_concurrent_publish_lock(base):
    """T4（040）：A 慢速安装者（每个镜像安装后停 1s）+ B 快速后到者，
    同 --out 并发 → 发布锁串行化，两进程均成功，最终四 aliases 完整
    属于 B（后到者完全覆盖前者），无混合代际。"""
    root_a = _copy_fixture(base, "t4_root_a")
    root_b = _copy_fixture(base, "t4_root_b")
    _tamper_vt_pred(root_a, "T4-A")
    _tamper_vt_pred(root_b, "T4-B")
    out = os.path.join(base, "t4.json")
    # B 单独发布的参考产物：并发收口的最终代际必须与它逐位一致
    out_b_ref = os.path.join(base, "t4_b_ref.json")
    r = _formal(root_b, out_b_ref)
    assert r.returncode == 0 and "DONE" in r.stdout, \
        (r.returncode, r.stdout[-2000:], r.stderr[-1000:])
    a = _formal_inject_bg(root_a, out, delay=1.0)
    time.sleep(0.3)          # 确保 A 先进入发布临界区（持锁安装中）
    b = _formal_bg(root_b, out)
    a_out, a_err = a.communicate(timeout=180)
    b_out, b_err = b.communicate(timeout=180)
    assert a.returncode == 0 and "DONE" in a_out, \
        (a.returncode, a_out[-1500:], a_err[-800:])
    assert b.returncode == 0 and "DONE" in b_out, \
        (b.returncode, b_out[-1500:], b_err[-800:])
    # 四固定 aliases 同一代际闭合（receipt↔JSON/manifest SHA +
    # manifest.run_id == receipt.run_id + generation 存在）
    assert _aliases_closed(out), \
        "并发发布后固定 aliases 不闭合（混合代际，040 复发）"
    # 后到者完全覆盖前者：最终 JSON 与 B 单独发布逐位一致
    assert _sha(out) == _sha(out_b_ref), \
        "最终 JSON ≠ 后到者 B 的单独发布结果（存在部分代际残留）"
    # 锁文件留存属预期（flock 无持锁状态，文件本身无害可长存）
    assert os.path.isfile(out + ".lock")
    print("T4 PASS  同 --out 双进程确定性交错发布（A 持锁慢速安装、B "
          "阻塞等待）→ 两进程均成功；四固定 aliases 同 run_id 闭合；"
          "最终代际 = 后到者 B 逐位（完全覆盖前者，无混合代际，040 修复）")


def test_T5_no_lock_red(base):
    """T5（040 红例）：同一交错调度 + flock 补丁 no-op（模拟修复前）→
    「两进程均成功 + 最终代际闭合」不成立（本调度下 B 的镜像交错进
    A 的安装窗口：A 的 SHA 终验捕获交错后失败回滚/最终代际被破坏）。
    证明 T4 绿灯来自发布锁本身而非侥幸时序。"""
    root_a = _copy_fixture(base, "t5_root_a")
    root_b = _copy_fixture(base, "t5_root_b")
    _tamper_vt_pred(root_a, "T5-A")
    _tamper_vt_pred(root_b, "T5-B")
    out = os.path.join(base, "t5.json")
    a = _formal_inject_bg(root_a, out, delay=2.0, no_lock=True)
    time.sleep(0.3)
    b = _formal_inject_bg(root_b, out, no_lock=True)
    a_out, a_err = a.communicate(timeout=180)
    b_out, b_err = b.communicate(timeout=180)
    ok = (a.returncode == 0 and b.returncode == 0 and
          _aliases_closed(out))
    assert not ok, (
        f"红例失效：禁用 flock 后同一交错调度仍「双成功+闭合」——"
        f"说明 T4 的交错窗口未真正覆盖临界区（a_rc={a.returncode}, "
        f"b_rc={b.returncode}）：\nA: {a_out[-600:]}\nB: {b_out[-600:]}")
    print(f"T5 PASS  红例：flock 禁用 + 同一交错调度 → 双成功+代际闭合"
          f"被破坏（a_rc={a.returncode}, b_rc={b.returncode}, "
          f"closed={_aliases_closed(out)}）——T4 绿灯确证来自发布锁")


def test_T6_kill_while_locked(base):
    """T6（040）：一方持锁安装中途被 SIGKILL → flock 由内核自动释放
    （陈旧锁天然恢复，无人工清理），另一方正常发布成功且最终四
    aliases 完整属于后者。被杀进程的 .bak/generation 残留属预期
    （SIGKILL 无法执行清理代码），不构成混合代际。"""
    root_a = _copy_fixture(base, "t6_root_a")
    root_b = _copy_fixture(base, "t6_root_b")
    _tamper_vt_pred(root_a, "T6-A")
    _tamper_vt_pred(root_b, "T6-B")
    out = os.path.join(base, "t6.json")
    out_b_ref = os.path.join(base, "t6_b_ref.json")
    r = _formal(root_b, out_b_ref)
    assert r.returncode == 0 and "DONE" in r.stdout, r.stdout[-1500:]
    marker = os.path.join(base, "t6_marker.txt")
    a = _formal_inject_bg(root_a, out, delay=3.0, marker=marker)
    # 轮询 marker=="1"：A 已安装第 1 个镜像（JSON）并停在临界区延迟内
    # ——此刻 A 持有发布锁
    deadline = time.time() + 60
    entered = False
    while time.time() < deadline:
        if (os.path.isfile(marker) and
                open(marker).read().strip() == "1"):
            entered = True
            break
        time.sleep(0.05)
    assert entered, "60s 内未观察到 A 进入发布临界区（marker 未出现）"
    a.kill()                 # SIGKILL：持锁状态下异常退出
    a.wait()
    assert a.returncode != 0, "被杀进程 returncode 应非零"
    b = _formal_bg(root_b, out)
    b_out, b_err = b.communicate(timeout=180)
    assert b.returncode == 0 and "DONE" in b_out, \
        (b.returncode, b_out[-1500:], b_err[-800:])
    assert _aliases_closed(out), \
        "被杀发布者之后另一方的发布结果不闭合（锁未正确释放/混合代际）"
    assert _sha(out) == _sha(out_b_ref), \
        "最终 JSON ≠ 后到者 B 的单独发布结果（被杀进程代际残留）"
    print("T6 PASS  持锁发布中 SIGKILL → 内核自动释放 flock（无陈旧锁）；"
          "另一方正常发布成功，最终四 aliases 完整属于后者（被杀进程的"
          " .bak/generation 残留属预期，不构成混合代际）")


def test_T7_post_publish_io_error(base):
    """T7（kimi3 1404）：发布事务成功提交后、报告阶段发生 OSError
    （stdout broken pipe 等）→ 不得回滚已发布产物、不得删除 generation
    目录；rc=0（发布有效性不受报告型错误影响）；failure receipt 如实
    记录 publish_committed=True。注入方式：monkeypatch _locked_publish
    包装器——真实发布事务完成后立刻抛 OSError，等价于成功路径 print
    遇 broken pipe。"""
    root = _copy_fixture(base, "t7_root")
    out = os.path.join(base, "t7.json")
    formal_argv = ["--root", root, "--pred-postfix", "_fx",
                   "--data-root", os.path.join(TESTDATA, "data_root"),
                   "--out", out, "--min-samples", "2"]
    code = (
        "import os, sys\n"
        "sys.argv = ['score_ruler_formal'] + " + repr(formal_argv) + "\n"
        "import benchmark.RULER.score_ruler_formal as m\n"
        "_lp = m._locked_publish\n"
        "def _lp_raise(*a, **k):\n"
        "    _lp(*a, **k)\n"
        "    raise OSError('INJECTED post-publish broken pipe')\n"
        "m._locked_publish = _lp_raise\n"
        "m.main()\n"
    )
    r = subprocess.run([sys.executable, "-c", code],
                       capture_output=True, text=True, cwd=REPO,
                       env={**os.environ, "PYTHONPATH": REPO})
    # 发布已成功提交 → rc=0（报告型错误不否定发布有效性）
    assert r.returncode == 0, \
        (r.returncode, r.stdout[-1500:], r.stderr[-1500:])
    # 四个公开产物完整在位且同代际闭合
    assert all(os.path.isfile(p) for p in _products(out)), \
        "发布成功后的报告型错误把公开产物删了（kimi3 1404 复发）"
    assert _aliases_closed(out), \
        "发布成功后的报告型错误破坏了代际闭合（kimi3 1404 复发）"
    # generation 目录保留（receipt 的 derived_dir 单指针目标）
    rc = json.load(open(out + ".receipt.json"))
    assert os.path.isdir(rc["outputs"]["derived_dir"]), \
        "generation 目录被 post-publish 错误误删（derived_dir 单指针断裂）"
    # failure receipt 如实记录 committed 语义
    fail = json.load(open(out + ".failure-" + rc["run_id"] + ".json"))
    assert fail["publish_committed"] is True, fail
    assert fail["generation_cleaned"] is False, fail
    assert fail["rollback"]["attempted"] is False, fail
    print("T7 PASS  发布成功后报告阶段 OSError（broken pipe 语义）→ "
          "不回滚公开产物、不删 generation、rc=0；failure receipt 记录 "
          "publish_committed=True（kimi3 1404 修复）")


def _tamper_vt_pred(root, tag):
    """篡改 vt 行 0 的 pred（_id/answers 身份不变）→ 重跑结果可辨。"""
    tgt = os.path.join(root, "L32768", "pred_fx", "vt-fxm-01010000.jsonl")
    rows = [json.loads(l) for l in open(tgt, encoding="utf-8")]
    rows[0]["pred"] = f"tampered pred ({tag})"
    with open(tgt, "w", encoding="utf-8") as f:
        for rr in rows:
            f.write(json.dumps(rr, ensure_ascii=False) + "\n")


def _assert_no_residue(out, tag):
    """无 staging / generation / tmp / bak 残留（排除既有旧产物）。"""
    assert not glob.glob(out + ".staging-*"), f"{tag}: staging 残留"
    for p in _products(out):
        assert not glob.glob(p + ".tmp-*"), f"{tag}: tmp 残留: {p}"
        assert not glob.glob(p + ".bak-*"), f"{tag}: bak 残留: {p}"


def test_T1_replace_injection(base):
    """T1（038）：第 1/2/3/4 个公开替换分别注入 OSError → 非零退出 +
    旧四产物 SHA 逐位还原（回滚生效）+ 旧 receipt 仍配旧结果 + 没有
    success 提交 + 旧 generation 完整可读 + failure receipt 落盘 +
    无任何残留。"""
    root = _copy_fixture(base, "t1_root")
    out = os.path.join(base, "t1.json")
    # 第一轮成功 → 旧 commit generation（四公开镜像 + generation 目录）
    r = _formal(root, out)
    assert r.returncode == 0 and "DONE" in r.stdout, \
        (r.returncode, r.stdout[-2000:], r.stderr[-1000:])
    old = {p: _sha(p) for p in _products(out)}
    old_rc = json.load(open(out + ".receipt.json"))
    old_runs = glob.glob(out + ".run-*")
    assert len(old_runs) == 1, old_runs
    assert old_rc["publish_protocol"] == "e116f-generation-v2"
    assert os.path.isdir(old_rc["outputs"]["derived_dir"])
    # 篡改 vt 行 0 pred → 新旧代际可辨（保证断言「旧 receipt 配新结果」
    # 若发生必被 SHA 比对抓到）
    _tamper_vt_pred(root, "T1")
    out_check = os.path.join(base, "t1_check.json")
    r = _formal(root, out_check)
    assert r.returncode == 0, (r.returncode, r.stdout[-1500:])
    assert _sha(out_check) != old[out], \
        "篡改后重跑结果与旧产物相同——注入测试的新旧代际不可辨，测试无效"
    # 第 1/2/3/4 个公开替换（JSON/MD/manifest/receipt 镜像安装）逐个注入
    for nth in (1, 2, 3, 4):
        r = _formal_inject(root, out, inject_replace=nth)
        assert r.returncode != 0, (nth, r.returncode, r.stdout[-2000:])
        assert f"INJECTED os.replace #{nth}" in (r.stdout + r.stderr), \
            (nth, r.stdout[-1500:], r.stderr[-1000:])
        # 旧 commit generation 仍完整可读：四公开产物 SHA 逐位不变（回滚）
        cur = {p: _sha(p) for p in _products(out)}
        assert cur == old, \
            f"注入 #{nth} 后旧公开产物被改动或未回滚: " \
            f"{[p for p in cur if cur[p] != old[p]]}"
        # 没有 success 提交：receipt 仍是旧 run_id（新代际 receipt 未发布）
        cur_rc = json.load(open(out + ".receipt.json"))
        assert cur_rc["run_id"] == old_rc["run_id"], \
            f"注入 #{nth} 后 receipt 换代（success 提交泄漏）"
        # 没有旧 receipt 配新结果：receipt 声明 SHA 与当前公开文件一致
        assert cur_rc["result_sha256"] == _sha(out), \
            f"注入 #{nth} 后旧 receipt 配新 JSON（混合代际，038 复发）"
        assert cur_rc["manifest_sha256"] == \
            _sha(out + ".manifest.json")
        # 旧 generation 目录规范四件仍完整
        for p in _gen_files(old_runs[0]):
            assert os.path.isfile(p), f"注入 #{nth}: {p} 缺失"
        # failure receipt 落盘（每轮一条）且 error 含注入标记
        fails = sorted(glob.glob(out + ".failure-*.json"))
        assert len(fails) == nth, (nth, fails)
        fr = json.load(open(fails[-1]))
        assert fr["status"] == "failed" and \
            f"INJECTED os.replace #{nth}" in fr["error"]
        # 无新 generation / staging / tmp / bak 残留
        assert glob.glob(out + ".run-*") == old_runs, \
            f"注入 #{nth}: 孤儿 generation 残留"
        _assert_no_residue(out, f"注入 #{nth}")
    print("T1 PASS  第 1/2/3/4 个公开替换分别注入 OSError → 旧 commit "
          "generation 逐位还原 + 旧 receipt 仍配旧结果 + 无 success 提交"
          " + failure receipt + 无残留（038 混合代际不可达）")


def test_T2_rename_injection(base):
    """T2（039）：generation 最终 rename 注入 OSError → 非零退出 +
    不发布任何公开文件（有旧产物=旧代际 SHA 逐位保留；首轮=零产物）+
    不存在指向缺失 derived_dir 的提交 + failure receipt + 无残留。"""
    root = _copy_fixture(base, "t2_root")
    out = os.path.join(base, "t2.json")
    r = _formal(root, out)
    assert r.returncode == 0 and "DONE" in r.stdout, r.stdout[-2000:]
    old = {p: _sha(p) for p in _products(out)}
    old_runs = glob.glob(out + ".run-*")
    assert len(old_runs) == 1
    _tamper_vt_pred(root, "T2")
    # 场景一：已有旧成功产物 → rename 失败时任何公开文件都未被触碰
    r = _formal_inject(root, out, inject_rename=True)
    assert r.returncode != 0, (r.returncode, r.stdout[-2000:])
    assert "INJECTED generation rename failure" in (r.stdout + r.stderr)
    assert {p: _sha(p) for p in _products(out)} == old, \
        "generation rename 失败仍触碰了旧公开产物"
    # 不会发布 success pointer/receipt：receipt 仍是旧 run_id 且其声明
    # 的 derived_dir 真实存在（不存在指向缺失目录的提交）
    cur_rc = json.load(open(out + ".receipt.json"))
    assert cur_rc["status"] == "success"
    assert os.path.isdir(cur_rc["outputs"]["derived_dir"]), \
        "存在指向缺失 derived_dir 的提交（039 复发）"
    assert glob.glob(out + ".run-*") == old_runs, "孤儿 generation 残留"
    fails = glob.glob(out + ".failure-*.json")
    assert len(fails) == 1, fails
    fr = json.load(open(fails[0]))
    assert "INJECTED" in fr["error"] and \
        fr["rollback"]["attempted"] is False
    _assert_no_residue(out, "rename 注入（旧产物场景）")
    # 场景二：首轮（无旧产物）→ rename 失败 → 零产物零提交
    out2 = os.path.join(base, "t2b.json")
    r = _formal_inject(root, out2, inject_rename=True)
    assert r.returncode != 0
    assert not any(os.path.exists(p) for p in _products(out2)), \
        "首轮 rename 失败仍发布了产物"
    assert not glob.glob(out2 + ".run-*"), "首轮 rename 失败残留 generation"
    assert not glob.glob(out2 + ".staging-*")
    assert glob.glob(out2 + ".failure-*.json"), "failure receipt 未落盘"
    _assert_no_residue(out2, "rename 注入（首轮场景）")
    print("T2 PASS  generation rename 注入 OSError → 不发布 success "
          "pointer/receipt + derived_dir 声明全部可闭合（旧产物场景 SHA "
          "逐位保留；首轮场景零产物）+ failure receipt + 无残留（039 "
          "不可达）")


def test_T3_cross_device_manifest(base):
    """T3（审计建议 3 / 038 的 EXDEV 复现路径）：--manifest-out 指向
    与 --out 不同设备（st_dev 不等）的挂载点 → 新协议下镜像安装走
    「目标文件系统临时文件 + 原子替换」，发布成功且 receipt 与跨设备
    manifest SHA 闭合。找不到第二设备时 SKIP。返回 True=PASS /
    False=SKIP。"""
    base_dev = os.stat(base).st_dev
    cands = ["/dev/shm", "/var/tmp", "/tmp", os.path.expanduser("~")]
    other = next((c for c in cands
                  if os.path.isdir(c) and os.stat(c).st_dev != base_dev),
                 None)
    if other is None:
        print("T3 SKIP  未找到与测试临时目录不同 st_dev 的文件系统"
              "（跨设备 --manifest-out 正例无法构造）")
        return False
    root = _copy_fixture(base, "t3_root")
    out = os.path.join(base, "t3.json")
    mout = os.path.join(
        other, f"e116f_t3_{os.getpid()}_{random.randint(1000, 9999)}"
        f".manifest.json")
    try:
        r = _formal(root, out, manifest_out=mout)
        assert r.returncode == 0 and "DONE" in r.stdout, \
            (r.returncode, r.stdout[-2000:], r.stderr[-1500:])
        assert os.path.isfile(mout), "跨设备 manifest 未落盘"
        rc = json.load(open(out + ".receipt.json"))
        # receipt 与跨设备 manifest 的 SHA 闭合（提交信号可闭合验证）
        assert rc["manifest_sha256"] == _sha(mout), \
            "receipt 声明的 manifest SHA 与跨设备 manifest 不一致"
        assert rc["result_sha256"] == _sha(out)
        assert os.path.isdir(rc["outputs"]["derived_dir"])
        assert not glob.glob(mout + ".tmp-*") and \
            not glob.glob(mout + ".bak-*"), "跨设备侧 tmp/bak 残留"
        print(f"T3 PASS  --manifest-out 跨设备（{other}，st_dev "
              f"{os.stat(other).st_dev} vs {base_dev}）→ 发布成功，"
              f"EXDEV 不可达，receipt↔manifest SHA 闭合")
        return True
    finally:
        if os.path.exists(mout):
            os.remove(mout)


def test_D1_no_regression():
    """D1：既有 E116e 套件（E1-E10 + D10 基线，D9 SKIP）不回归。"""
    r = subprocess.run(
        [sys.executable, "-m", "benchmark.RULER.test_e116e_gate"],
        capture_output=True, text=True, cwd=REPO,
        env={**os.environ, "PYTHONPATH": REPO})
    assert r.returncode == 0 and "E116e ALL PASS" in r.stdout, \
        (r.returncode, r.stdout[-3000:], r.stderr[-2000:])
    print("D1 PASS test_e116e_gate.py E1-E10 + D10 无回归（12/13 + D9 SKIP）")


def main():
    global PASS
    base = tempfile.mkdtemp(prefix="e116f_")
    try:
        test_T1_replace_injection(base)
        PASS += 1
        test_T2_rename_injection(base)
        PASS += 1
        t3 = test_T3_cross_device_manifest(base)
        if t3:
            PASS += 1
        test_T4_concurrent_publish_lock(base)
        PASS += 1
        test_T5_no_lock_red(base)
        PASS += 1
        test_T6_kill_while_locked(base)
        PASS += 1
        test_T7_post_publish_io_error(base)
        PASS += 1
        test_D1_no_regression()
        PASS += 1
    finally:
        shutil.rmtree(base, ignore_errors=True)
    total = 8 if t3 else 7
    skip = "" if t3 else "，T3 SKIP（无第二设备）"
    print(f"\nE116f ALL PASS ({PASS}/{total}{skip})")


if __name__ == "__main__":
    main()
