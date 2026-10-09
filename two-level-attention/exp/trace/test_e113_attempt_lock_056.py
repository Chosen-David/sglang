#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E113 microbench 056 修复红绿单测：attempt 输出路径跨进程锁（真实双进程）。

对应 GPT 审计 TL-E113-ATTEMPT-RACE-056（2026-10-10_0330，P1）的修复验收。
审计复现：两个同 --out attempt 在发布前都可完成隔离检查，随后交错发布可致
success JSON（B）+ sidecar（B）+ failure.json（A）三公开路径并存，违反
「可见终态唯一且属于本 attempt」协议。修复 = 生命周期 flock（acquire 于
quarantine 之前，release 于终检/失败落盘之后，崩溃由内核自动释放）。

本测试**不允许同进程顺序模拟**——所有并发用例都是 `subprocess.Popen` 起的
真实 python 子进程（e113_attempt_lock_runner_056.py），子进程加载生产
e113_microbench 并调用生产 main() 走真实「隔离 → 执行 → 发布 → 终检」路径
（GPU/张量/实现加载全部 stub，只替换环境桩，不绕过生产状态机；本机不占卡）。

用例：
  T9a 锁原语：attempt_lock_path realpath 规范化（symlink 父目录 / .. 拼写
      → 同一锁文件）；lock-hold 子进程持锁期间父进程 LOCK_EX|LOCK_NB 必须
      失败、子进程退出后必须成功（真实跨进程互斥，非线程/非同进程模拟）；
  T9b success/success、T9c success/failure、T9d failure/success、
      T9e failure/failure 四组合双进程并发（--barrier 对齐进入 main() 的
      时刻 + 桩 bench sleep 拉开 quarantine 与发布的窗口），每组合断言：
      ① 公开入口最多一个有效终态（success 对 out+sidecar 与 failure.json
         互斥、无临时文件残留）；
      ② success 时 JSON/sidecar/attempt 三者同代际（sidecar=JSON 字节 SHA、
         内容 SHA 自校验过、attempt_id 属于两进程之一）；
      ③ 后到者（loser）按隔离协议处理前任终态：可见终态的 superseded_files
         覆盖前任留下的全部可见产物、隔离副本真实存在且 attempt 归属前任 pid、
         隔离记录只涉及三个可见后缀（锁文件/临时文件绝不进隔离）、前任终态
         未被删除；
      ④ 两进程退出码符合各自模式（success→0 / failure→1，_fail 终检=3 视为
         失败）；
  T9f 崩溃注入：success JSON 已原子替换、sidecar 未写时持锁硬杀
      （os._exit(9)，不经 finally 不显式释放）→ ① 内核自动释放 flock（下一
      attempt 正常获锁）；② 孤儿 JSON（无 sidecar 的不一致中间态）被下一
      attempt 按隔离协议改名保留，可见终态恢复「唯一且同代际」。

红测说明：修复前（无锁）双方 quarantine 都看到空目录 → 后到者 superseded
记录为空 → 上述断言 ③ 必然失败（且 failure 组合常伴 3 路径并存/退出码 3）。

环境边界（如实声明）：runner 需要可 import 的 torch（只用 CPU 张量做 gate
桩，GPU 全 stub，不占卡）；无 torch 环境下本测试 import 即退，不冒充通过。
python 与 python -O 双跑安全：全部显式 check，不依赖 assert。
"""
import fcntl
import glob
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "e113_microbench.py")
RUNNER = os.path.join(HERE, "e113_attempt_lock_runner_056.py")

RESULTS = []


def report(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""), flush=True)


def load_prod_module():
    """按文件加载生产 e113_microbench（锁原语/发布校验函数供父进程使用）。"""
    spec = importlib.util.spec_from_file_location(
        f"e113_prod_mod_test_{os.getpid()}_{time.time_ns()}", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def py_cmd():
    """子进程与父进程同优化模式（父进程 -O 时子进程也 -O）。"""
    return [sys.executable] + (["-O"] if not __debug__ else [])


def spawn(out, mode, barrier_dir=None, barrier_count=1, bench_sleep=0.25, hold=2.0):
    cmd = py_cmd() + [RUNNER, "--out", out, "--mode", mode,
                      "--bench-sleep", str(bench_sleep), "--hold", str(hold)]
    if barrier_dir:
        cmd += ["--barrier-dir", barrier_dir, "--barrier-count", str(barrier_count)]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def wait_proc(p, timeout=240):
    """等子进程退出；超时（疑似死锁——锁未释放的回归信号）杀掉并如实报失败。"""
    try:
        out, _ = p.communicate(timeout=timeout)
        return p.returncode, out
    except subprocess.TimeoutExpired:
        p.kill()
        out, _ = p.communicate()
        return None, out + "\n[TEST] 超时被杀（疑似持锁未释放/死锁）"


def attempt_pid(att_id):
    """attempt_id = <ts>-<pid>-<digest8> 的 pid 段。"""
    return int(att_id.split("-")[1])


def audit_visible_state(out, pids, prod):
    """公开入口终态审计：唯一性 + 同代际 + attempt 归属。返回 (ok, info)。"""
    problems = []
    has_out = os.path.exists(out)
    has_side = os.path.exists(out + ".sha256")
    has_fail = os.path.exists(out + ".failure.json")
    # ① 唯一性
    if has_out and has_fail:
        problems.append("success JSON 与 failure.json 三路径并存（056 竞态未消除）")
    if has_out != has_side:
        problems.append("success JSON 与 sidecar 不同代（一有一无）")
    if not has_out and not has_fail:
        problems.append("无任何可见终态")
    for pat in (out + ".tmp-*", out + ".sha256.tmp-*", out + ".failure.json.tmp-*"):
        if glob.glob(pat):
            problems.append(f"临时文件残留：{os.path.basename(pat)}")
    info = {"problems": problems}
    # ② 同代际 + 归属
    if has_out:
        raw = open(out, "rb").read()
        side = open(out + ".sha256", "rb").read()
        if side != hashlib.sha256(raw).hexdigest().encode() + b"\n":
            problems.append("sidecar 与 success JSON 字节不同代")
        d = json.loads(raw)
        att = d.get("meta", {}).get("attempt", {})
        info["kind"] = "success"
        info["attempt_id"] = att.get("attempt_id", "")
        info["superseded"] = att.get("superseded_files", [])
        if not info["attempt_id"]:
            problems.append("success JSON 缺 attempt_id")
        elif attempt_pid(info["attempt_id"]) not in pids:
            problems.append(f"可见终态 attempt={info['attempt_id']} 不属于两进程")
        if not prod.verify_output_sha256(out):
            problems.append("发布内容 SHA 自校验失败")
    elif has_fail:
        with open(out + ".failure.json", encoding="utf-8") as f:
            d = json.load(f)
        att = d.get("attempt", {})
        info["kind"] = "failure"
        info["attempt_id"] = att.get("attempt_id", "")
        info["superseded"] = att.get("superseded_files", [])
        if d.get("status") != "failed":
            problems.append("failure.json 缺 status=failed")
        if not info["attempt_id"]:
            problems.append("failure.json 缺 attempt_id")
        elif attempt_pid(info["attempt_id"]) not in pids:
            problems.append(f"可见终态 attempt={info['attempt_id']} 不属于两进程")
    return (not problems), info


def check_loser_protocol(out, info, pids, mode_of):
    """断言 ③：后到者（可见终态 owner）按隔离协议处理前任终态。

    owner = 可见终态的 attempt 归属进程（= 获得锁较晚、最后发布的那位）；
    前任 = 另一进程。获锁时前任终态已完整就位（056 锁保证），因此 owner 的
    superseded_files 必须覆盖前任留下的全部可见产物；被隔离文件真实存在
    （历史保留不删除）、success JSON 隔离副本可解析且 attempt 归属前任；
    隔离记录只允许涉及三个可见后缀（<out>.attempt.lock 等绝不进隔离）。"""
    problems = []
    owner_pid = attempt_pid(info["attempt_id"])
    other_pid = [p for p in pids if p != owner_pid]
    if not other_pid:
        problems.append("无法识别前任进程（attempt pid 异常）")
        return False, problems
    other_pid = other_pid[0]
    other_mode = mode_of[other_pid]
    expected = {"success": [out, out + ".sha256"],
                "failure": [out + ".failure.json"],
                "crash-between": [out]}[other_mode]
    rec = {r.get("visible_path"): r.get("superseded_as") for r in info["superseded"]}
    for p in expected:
        if p not in rec:
            problems.append(f"前任产物未被隔离记录：{os.path.basename(p)}（superseded={info['superseded']}）")
        elif not os.path.exists(rec[p]):
            problems.append(f"隔离副本不存在（前任终态被删除？）：{os.path.basename(rec[p])}")
    if other_mode == "success" and out in rec and os.path.exists(rec[out]):
        try:
            with open(rec[out], encoding="utf-8") as f:
                d2 = json.load(f)
            att2 = d2.get("meta", {}).get("attempt", {}).get("attempt_id", "")
            if attempt_pid(att2) != other_pid:
                problems.append("前任 success JSON 隔离副本 attempt 归属异常")
        except Exception as e:
            problems.append(f"前任 success JSON 隔离副本不可解析: {e}")
    if other_mode == "failure" and (out + ".failure.json") in rec and os.path.exists(rec[out + ".failure.json"]):
        try:
            with open(rec[out + ".failure.json"], encoding="utf-8") as f:
                d2 = json.load(f)
            if d2.get("status") != "failed":
                problems.append("前任 failure 隔离副本内容异常")
        except Exception as e:
            problems.append(f"前任 failure 隔离副本不可解析: {e}")
    allowed = {out, out + ".sha256", out + ".failure.json"}
    for r in info["superseded"]:
        if r.get("visible_path") not in allowed:
            problems.append(f"隔离记录越界（非可见终态路径）：{r.get('visible_path')}")
    return (not problems), problems


def tail(text, n=6):
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return " | ".join(lines[-n:])


# ================================================================ T9a 锁原语
def t9a_lock_primitives():
    name = "T9a（056 原语）attempt_lock_path realpath 规范化 + 真实跨进程互斥"
    try:
        prod = load_prod_module()
        ok = True
        detail_parts = []
        with tempfile.TemporaryDirectory() as td:
            real_dir = os.path.join(td, "d")
            os.makedirs(real_dir)
            link_dir = os.path.join(td, "dlink")
            os.symlink(real_dir, link_dir)
            p1 = prod.attempt_lock_path(os.path.join(real_dir, "r.json"))
            p2 = prod.attempt_lock_path(os.path.join(link_dir, "r.json"))     # symlink 父目录
            p3 = prod.attempt_lock_path(os.path.join(real_dir, "sub", "..", "r.json"))  # .. 拼写
            want = os.path.join(real_dir, "r.json.attempt.lock")
            if not (p1 == p2 == p3 == want):
                ok = False
                detail_parts.append(f"realpath 漂移: {p1} / {p2} / {p3} != {want}")
            else:
                detail_parts.append("realpath 规范化（symlink/../.. 拼写同锁文件）OK")
            # ---- 真实跨进程互斥：轮询探测子进程获锁（import torch 耗时有抖动，
            #      固定 sleep 不可靠；父进程每次 NB 成功都立即 UN 放行，绝不持有，
            #      否则会把子进程死锁在 flock 上——首轮版本曾因此自锁）----
            out = os.path.join(real_dir, "r.json")
            child = spawn(out, "lock-hold", hold=3.0)
            fd = os.open(prod.attempt_lock_path(out), os.O_CREAT | os.O_RDWR, 0o644)
            blocked_after, t0 = None, time.time()
            while time.time() - t0 < 30:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    # 子进程尚未获锁——立即释放让子进程继续（父进程绝不持锁等待）
                    fcntl.flock(fd, fcntl.LOCK_UN)
                    time.sleep(0.05)
                except OSError:
                    blocked_after = time.time() - t0   # 子进程持锁 → EWOULDBLOCK
                    break
            if blocked_after is None:
                ok = False
                detail_parts.append("30s 内子进程未获锁（或互斥失效）")
            else:
                detail_parts.append(f"子进程持锁后 {blocked_after:.1f}s 起 LOCK_NB 持续被拒（互斥生效）")
            rc, log = wait_proc(child)
            released_ok = False
            t1 = time.time()
            while time.time() - t1 < 10:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)   # 子进程退出后必须可获
                    released_ok = True
                    break
                except OSError:
                    time.sleep(0.1)
            if released_ok:
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
            if rc != 0:
                ok = False
                detail_parts.append(f"lock-hold 子进程 rc={rc}（None=超时，疑似死锁）")
            elif not released_ok:
                ok = False
                detail_parts.append("子进程退出后锁仍被占（释放失效/死锁残留）")
            else:
                detail_parts.append("子进程退出后父进程立即可获（无死锁残留）")
        report(name, ok, "；".join(detail_parts))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ T9b-T9e 四组合并发
def combo_case(name, mode_a, mode_b):
    try:
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "result.json")
            bar = os.path.join(td, "bar")
            pa = spawn(out, mode_a, bar, 2)
            pb = spawn(out, mode_b, bar, 2)
            rc_a, log_a = wait_proc(pa)
            rc_b, log_b = wait_proc(pb)
            pids = [pa.pid, pb.pid]
            mode_of = {pa.pid: mode_a, pb.pid: mode_b}
            prod = load_prod_module()
            problems = []
            # ④ 退出码
            for rc, mode, who in ((rc_a, mode_a, "A"), (rc_b, mode_b, "B")):
                want = {"success": 0, "failure": 1}[mode]
                if rc != want:
                    problems.append(f"进程 {who}({mode}) rc={rc}（期望 {want}；3=终检失败/None=超时）")
            if rc_a is None or rc_b is None:
                problems.append("有进程超时被杀（疑似死锁）")
            # ①② 终态审计
            if rc_a is not None and rc_b is not None:
                ok_vis, info = audit_visible_state(out, pids, prod)
                problems += info["problems"]
                # ③ loser 隔离协议
                if ok_vis:
                    ok_lose, lose_problems = check_loser_protocol(out, info, pids, mode_of)
                    problems += lose_problems
                # 锁文件存在（新产物）但绝不进可见终态/隔离（hasattr 守卫：
                # pre-fix 代码没有锁原语时让红测展示真实协议违规而非 AttributeError）
                if hasattr(prod, "attempt_lock_path"):
                    if not os.path.exists(prod.attempt_lock_path(out)):
                        problems.append("锁文件未创建（锁未生效？）")
            if problems:
                report(name, False, f"{mode_a}/{mode_b}：{'；'.join(problems)}；"
                                    f"日志尾 A:[{tail(log_a)}] B:[{tail(log_b)}]")
            else:
                report(name, True, f"{mode_a}/{mode_b}：可见终态唯一（{info['kind']}，"
                                   f"attempt={info['attempt_id']}）、同代际、前任产物隔离保留、"
                                   f"rc=({rc_a},{rc_b})")
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


def t9b_success_success():
    combo_case("T9b（056）并发 success/success：串行化 + 后到者隔离前任成功对",
               "success", "success")


def t9c_success_failure():
    combo_case("T9c（056）并发 success/failure：三路径并存不可达",
               "success", "failure")


def t9d_failure_success():
    combo_case("T9d（056）并发 failure/success：三路径并存不可达（反序）",
               "failure", "success")


def t9e_failure_failure():
    combo_case("T9e（056）并发 failure/failure：可见 failure 唯一且属最后 attempt",
               "failure", "failure")


# ================================================================ T9f 崩溃注入
def t9f_crash_between():
    name = "T9f（056）JSON/sidecar 两次替换间持锁硬杀 → 内核释放 + 下一 attempt 恢复一致"
    try:
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "result.json")
            pa = spawn(out, "crash-between")
            rc_a, log_a = wait_proc(pa)
            problems = []
            if rc_a != 9:
                problems.append(f"crash 注入进程 rc={rc_a}（期望 9）")
            # 崩溃后中间态：孤儿 success JSON（无 sidecar、无 failure）
            orphan = os.path.exists(out)
            if not orphan:
                problems.append("崩溃点异常：孤儿 JSON 未落盘（注入点漂移）")
            if os.path.exists(out + ".sha256"):
                problems.append("崩溃点异常：sidecar 竟已写入")
            # 下一 attempt 恢复一致（内核已自动释放 flock——否则这里会死锁超时）
            pb = spawn(out, "success")
            rc_b, log_b = wait_proc(pb)
            if rc_b != 0:
                problems.append(f"恢复 attempt rc={rc_b}（期望 0；锁未释放会超时）")
            if rc_b == 0 and orphan:
                prod = load_prod_module()
                ok_vis, info = audit_visible_state(out, [pa.pid, pb.pid], prod)
                problems += info["problems"]
                if ok_vis:
                    ok_lose, lose_problems = check_loser_protocol(
                        out, info, [pa.pid, pb.pid], {pa.pid: "crash-between", pb.pid: "success"})
                    problems += lose_problems
                    # 孤儿 JSON 被隔离且内容可解析、attempt 归属崩溃进程
                    rec = {r.get("visible_path"): r.get("superseded_as")
                           for r in info["superseded"]}
                    if out in rec and os.path.exists(rec[out]):
                        with open(rec[out], encoding="utf-8") as f:
                            d2 = json.load(f)
                        att2 = d2.get("meta", {}).get("attempt", {}).get("attempt_id", "")
                        if attempt_pid(att2) != pa.pid:
                            problems.append("孤儿 JSON 隔离副本归属异常")
                    else:
                        problems.append("孤儿 JSON 未被隔离保留")
            if problems:
                report(name, False, f"{'；'.join(problems)}；日志尾 A:[{tail(log_a)}] B:[{tail(log_b)}]")
            else:
                report(name, True, f"rc=({rc_a},{rc_b})：崩溃后中间态（孤儿 JSON 无 sidecar）"
                                   f"被下一 attempt 隔离保留，可见终态恢复唯一同代际"
                                   f"（attempt={info['attempt_id']}）")
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


if __name__ == "__main__":
    t9a_lock_primitives()
    t9b_success_success()
    t9c_success_failure()
    t9d_failure_success()
    t9e_failure_failure()
    t9f_crash_between()
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print("\n" + "=" * 60)
    print(f"总计 {len(RESULTS)} 项，通过 {len(RESULTS) - n_fail}，失败 {n_fail}")
    sys.exit(1 if n_fail else 0)
