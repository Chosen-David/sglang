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
        test_D1_no_regression()
        PASS += 1
    finally:
        shutil.rmtree(base, ignore_errors=True)
    total = 4 if t3 else 3
    skip = "" if t3 else "，T3 SKIP（无第二设备）"
    print(f"\nE116f ALL PASS ({PASS}/{total}{skip})")


if __name__ == "__main__":
    main()
