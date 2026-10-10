#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""#199/#200 红绿测试（GPT 2026-10-10 1330/1528 审计）：
  TL-E119-POINTER-SKIP-068（P2）E109 调度器 SKIP 判定看不见指针代
  TL-E119-POINTER-SCORER-069（P3）基础 scorer direct CLI 与指针协议脱节
  TL-E119-PROBE-RECEIPT-VALIDATION-070（P2）完成探针只数行数不校验回执——
     「预测足量 + 回执存在但无效」被判 complete → SKIP，正式汇总却
     fail-closed 拒收：调度与交付的完成定义分裂，坏格永久 SKIP（#200）

违反事实（GPT 审计已 CPU 复现）：
  068  #198 指针协议（066/crash-recovery）提交后只留 {out}.tli_gen 指针 +
       gen 目录内 JSONL，逻辑 .jsonl 不落盘；run_ruler_e109.sh:63 仍
       `ls $OUTDIR/$T-*.jsonl | head -1`——对已提交完整代实测
       SKIP_CONDITION=false（指针在、gen 预测在、调度仍重跑），断点续跑
       失效，重启重跑昂贵 32K/64K/128K GPU 任务。
  069  score_ruler.py 公开 direct CLI 只 glob 直写 JSONL：pointer-only
       root 带 --expect-tasks 时误报 0 方法键；无门禁时 exit 0 写空结果
       {"scores": {}, ...}（SHA 4d493b3e...）冒充正式外观产物。
       pred_ruler.py docstring 还声称由 score_ruler.py 打分（过时）。

修复口径（主 AI advice 回应已定）：
  068  新增零依赖探针 benchmark/RULER/gen_completion_probe.py——复用
       resolve_generation_pointer（不 shell 复制协议解析），best-file
       语义取最大行数（不做 head -1 字典序），pointer 优先于 stale
       legacy（formal 782-807 同口径），损坏/缺件指针 fail-closed
       （STATE=invalid 非零退出，不当需要重跑）。run_ruler_e109.sh
       改调探针：complete→SKIP / partial|missing→跑 / invalid→
       PROBE-FAIL 计失败（协议错误人工介入）。
  069  方案 2（单口径原则，不维护第二套指针解析）：direct CLI 检测到
       root 下任何 .tli_gen → [GATE-FAIL] 非零退出零输出、提示改用
       score_ruler_formal.py；门禁只在 main() 生效（formal import 本
       模块函数不经 main，staging 派生副本无指针文件恒零触发）；
       pred_ruler.py docstring 打分指向改为 formal。

用例矩阵（G = 068 调度决策回归，N = 069；全部真实实跑）：
  G1 pointer-only 足量 → complete/SKIP；不足 → partial/RUN
     （审计复现格：修复前 SKIP_CONDITION=false，修复后必须 SKIP）
  G2 legacy-only 三态 + 目录不存在 → complete/partial/missing（-m 包
     模式调探针，覆盖包导入分支）+ 四态→shell 决策映射
  G3 pointer + stale legacy 同基名 → 指针优先不回退（指针 2 行 vs
     legacy 100 行 → 探针报 partial N=2；回退 legacy 会误报 complete）
  G4 best-file 语义：多候选取最大行数，不做字典序 head -1（历史坑：
     E109 SKIP 首文件 partial 误判）+ pointer/legacy 混合候选仲裁
  G5 损坏指针三负例（指针指空名/路径逃逸/gen 缺回执）→ STATE=invalid
     非零退出，决策 ERROR 不当需要重跑
  G6 crash-before-switch：指针未切 → 仍见旧完整代 complete（断点续跑
     语义）；首跑 crash-mid（无指针）→ missing（partial gen 不被计入）
  G7 锁内并发：生产者持锁提交期间探针只认已提交代（不见在飞 partial），
     提交后只认新提交代
  N1 direct scorer 对 pointer-only root → 非零退出 + [GATE-FAIL] +
     formal 提示 + 零输出文件（-m 与直接脚本两种调用方式；有无
     --expect-tasks 都必须拒）
  N2 legacy-only root 对照 → 正常打分不回归（exit 0、分数正确、
     JSON/MD 落盘）
  N3 run_ruler_e109.sh 接线静态检查（bash -n + 探针调用在位 +
     旧 glob 判定已移除）
  RP 红探针：打桩 _discover_candidates(include_pointers=False) 复刻
     修复前「只看 .jsonl glob」行为 → pointer-only complete 判定必须
     红非 complete——证明 G1 断言确实行使指针解析，不是平凡通过
  G8（070 审计复现格）：真实提交代上破坏 gen 内回执六负例（非 JSON/
     {}/status!=complete/basename 错/SHA 错/行数错）→ 全部
     STATE=invalid 非零退出 + [GATE-FAIL] + PROBE-FAIL（修复前只数行数
     → complete/SKIP 假完成——审计最小复现闭合）；破坏前同代探针必须
     complete/SKIP（共享校验器不误伤合法提交代）；legacy 直写分支不被
     pointer 回执门禁误伤（G2 回归承载）
  A1 --audit-dir 只读预检：全 valid root → rc=0；含破坏 pointer →
     rc=2 + invalid 格清单；零 pointer 目录 → 如实报告 rc=0；全程零写
     （目录内容快照前后逐位一致）

065 纪律：零裸 assert（全部 _check 显式判定，python -O 不失效）；
067 纪律：PASS/SKIP/FAIL 三分显式计数，SKIP>0 不打 ALL PASS。

用法（two-level-attention 仓库根）：
  PYTHONPATH=$PWD python3 -m benchmark.RULER.test_e119_pointer_consumer_068_069
  PYTHONPATH=$PWD python3 -O -m benchmark.RULER.test_e119_pointer_consumer_068_069
  E119_ONLY=G1,RP PYTHONPATH=$PWD python3 -m ...   # 子集过滤
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from benchmark.RULER.yarn_receipt import (  # noqa: E402
    GENERATION_POINTER_SUFFIX, generation_pointer_path,
    resolve_generation_pointer)

RUNNER = os.path.join(REPO, "benchmark", "RULER",
                      "e119_yarn_producer_runner_059.py")
PROBE = os.path.join(REPO, "benchmark", "RULER",
                     "gen_completion_probe.py")
SCORER = os.path.join(REPO, "benchmark", "RULER", "score_ruler.py")
E109_SH = os.path.join(REPO, "benchmark", "RULER", "run_ruler_e109.sh")

# 与 run_ruler_e109.sh 修复后 shell 分支逐位同口径的决策映射
_DECISION = {"complete": "SKIP", "partial": "RUN", "missing": "RUN"}


def _check(cond, msg=""):
    """065 纪律：显式判定，非 assert（python -O 不删除）。失败 →
    SystemExit（[TEST-FAIL] 前缀），main 捕获计 FAIL。"""
    if not cond:
        raise SystemExit(f"[TEST-FAIL] {msg}")


def _check_state(r, state, n=None, src=None, tag=""):
    """断言探针子进程输出四态之一及其 N/SRC 字段。"""
    _check(r.returncode == 0, f"{tag}: 探针不应非零退出（{state} 是合法"
                             f"态）——rc={r.returncode} out={r.stdout[-500:]}")
    line = None
    for ln in r.stdout.splitlines():
        if ln.startswith("STATE="):
            line = ln
            break
    _check(line is not None, f"{tag}: 探针输出缺 STATE= 行: {r.stdout!r}")
    fields = dict(kv.split("=", 1) for kv in line.split()[1:])
    got = line[len("STATE="):].split()[0]
    _check(got == state, f"{tag}: 期望 STATE={state}，得到 {got}"
                         f"（完整输出 {r.stdout!r}）")
    if n is not None:
        _check(fields.get("N") == str(n),
               f"{tag}: 期望 N={n}，得到 {fields.get('N')!r}"
               f"（完整输出 {r.stdout!r}）")
    if src is not None:
        _check(fields.get("SRC") == src,
               f"{tag}: 期望 SRC={src}，得到 {fields.get('SRC')!r}"
               f"（完整输出 {r.stdout!r}）")
    return fields


def _decision(r):
    """shell 分支决策（与 run_ruler_e109.sh 修复后逻辑逐位一致）：
    rc!=0 → PROBE-FAIL（协议错误）；STATE=complete → SKIP；
    partial/missing → RUN。"""
    if r.returncode != 0:
        return "PROBE-FAIL"
    for ln in r.stdout.splitlines():
        if ln.startswith("STATE="):
            return _DECISION.get(ln[len("STATE="):].split()[0], "PROBE-FAIL")
    return "PROBE-FAIL"


def _probe(out_dir, task="vt", max_num=3, module_mode=False):
    """探针子进程。module_mode=True 走 -m 包调用（覆盖包导入分支），
    否则直接脚本调用（覆盖 sys.path 脚本目录导入分支）。"""
    if module_mode:
        argv = [sys.executable, "-u", "-m",
                "benchmark.RULER.gen_completion_probe"]
    else:
        argv = [sys.executable, "-u", PROBE]
    argv += ["--out-dir", out_dir, "--task", task, "--max-num", str(max_num)]
    return subprocess.run(argv, capture_output=True, text=True, cwd=REPO,
                          env={**os.environ, "PYTHONPATH": REPO})


def _run_producer(base, name, mode, tag, rows=3, crash_after=1, hold=2.0,
                  wait=True, max_num=0):
    """起真实生产子进程（e119_yarn_producer_runner_059，stub 只换环境桩，
    锁/gen 目录/v2 回执/单次原子切指针全走生产 main()）。
    wait=True → (stdout, rc)；False → Popen。"""
    out_dir = os.path.join(base, name, "out")
    data_root = os.path.join(base, "shared_data")
    argv = [sys.executable, RUNNER, "--out-dir", out_dir,
            "--data-root", data_root, "--mode", mode,
            "--pred-tag", tag, "--rows", str(rows),
            "--crash-after", str(crash_after), "--hold", str(hold),
            "--max-num", str(max_num)]
    if not wait:
        return subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            cwd=REPO)
    r = subprocess.run(argv, capture_output=True, text=True, cwd=REPO)
    return r.stdout, r.returncode


def _pred_dir(base, name):
    return os.path.join(base, name, "out", "L32768", "pred_stub")


def _logical(base, name):
    return os.path.join(_pred_dir(base, name), "vt-stubm-09090909.jsonl")


def _wait_marker(p, marker, timeout=60.0):
    """读子进程 stdout 直到出现 marker（带超时；用于 lock-hold 确定性
    同步——持锁确认后再起后继生产者）。"""
    t0 = time.time()
    buf = []
    while time.time() - t0 < timeout:
        ln = p.stdout.readline()
        if not ln:
            _check(p.poll() is not None,
                   f"等 marker {marker!r} 超时且子进程未退出")
            break
        buf.append(ln)
        if marker in ln:
            return "".join(buf)
    raise SystemExit(f"[TEST-FAIL] 等待 marker {marker!r} 超时："
                     f"{''.join(buf)[-500:]}")


def _write_legacy(out_dir, basename, rows, answers=("gt0",)):
    """手写 legacy 直写预测文件（旧协议产物模拟；行内容合法 jsonl）。"""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, basename)
    with open(path, "w", encoding="utf-8") as f:
        for i in range(rows):
            f.write(json.dumps({
                "pred": "stub", "answers": list(answers),
                "length": 32768, "budget": 0,
            }, ensure_ascii=False) + "\n")
    return path


# ================================================================ 068 调度决策

def test_G1_pointer_only_complete_partial(base):
    """G1（068 审计复现格）：pointer-only 已提交完整代——修复前
    `ls | head -1` 判定 SKIP_CONDITION=false（重跑），修复后探针必须
    complete/SKIP；行数不足时 partial/RUN。"""
    out, rc = _run_producer(base, "g1", "success", "A", rows=3)
    _check(rc == 0, f"G1: 生产者提交失败 rc={rc}: {out[-300:]}")
    d = _pred_dir(base, "g1")
    _check(not os.path.exists(_logical(base, "g1")),
           "G1 前提：指针协议下逻辑 .jsonl 不得落盘（068 违反事实根源）")
    r = _probe(d, max_num=3)
    _check_state(r, "complete", n=3, src="pointer", tag="G1-complete")
    _check(_decision(r) == "SKIP",
           f"G1: complete 必须映射 SKIP（断点续跑核心修复），"
           f"得到 {_decision(r)}")
    r2 = _probe(d, max_num=5)
    _check_state(r2, "partial", n=3, src="pointer", tag="G1-partial")
    _check(_decision(r2) == "RUN", f"G1: partial 必须映射 RUN")
    print("G1 PASS  pointer-only 足量→complete/SKIP；不足→partial/RUN"
          "（修复前该格 SKIP_CONDITION=false 必重跑——068 违反事实闭合）")


def test_G2_legacy_only_three_states(base):
    """G2：legacy-only 三态 + 目录不存在 + 决策映射（-m 包模式调探针，
    覆盖包导入分支）。"""
    d = os.path.join(base, "g2", "out", "L32768", "pred_stub")
    _write_legacy(d, "vt-stubm-01010000.jsonl", 3)
    r = _probe(d, max_num=3, module_mode=True)
    _check_state(r, "complete", n=3, src="legacy", tag="G2-complete")
    _check(_decision(r) == "SKIP", "G2: legacy complete 必须映射 SKIP")
    r = _probe(d, max_num=5, module_mode=True)
    _check_state(r, "partial", n=3, src="legacy", tag="G2-partial")
    _check(_decision(r) == "RUN", "G2: legacy partial 必须映射 RUN")
    r = _probe(os.path.join(base, "g2", "no_such_dir"), module_mode=True)
    _check_state(r, "missing", n=0, tag="G2-missing-dir")
    _check(_decision(r) == "RUN", "G2: 目录不存在必须 missing/RUN"
                                  "（新格首跑，非协议错误）")
    print("G2 PASS  legacy-only complete/partial/missing 三态 + 目录"
          "不存在 missing + 四态→shell 决策映射（-m 包模式导入分支覆盖）")


def test_G3_pointer_priority_over_stale_legacy(base):
    """G3：pointer + stale legacy 同基名并存 → 指针优先（formal 782-807
    同口径）。指针代 2 行 + stale legacy 100 行 → 探针必须报 partial
    N=2 SRC=pointer——若回退 legacy 会误报 complete N=100（068 修复
    契约：不得把 stale legacy 当当前代）。"""
    out, rc = _run_producer(base, "g3", "success", "A", rows=2)
    _check(rc == 0, f"G3: 生产者提交失败 rc={rc}: {out[-300:]}")
    # 同基名 stale legacy 直写残留（旧协议运行时代产物；新代码不写）
    _write_legacy(_pred_dir(base, "g3"), "vt-stubm-09090909.jsonl", 100)
    d = _pred_dir(base, "g3")
    r = _probe(d, max_num=100)
    _check_state(r, "partial", n=2, src="pointer", tag="G3-priority")
    _check(_decision(r) == "RUN",
           "G3: 指针代 2 行 < 100 → RUN（指针优先语义下正确重跑）")
    r2 = _probe(d, max_num=2)
    _check_state(r2, "complete", n=2, src="pointer", tag="G3-complete")
    _check(_decision(r2) == "SKIP", "G3: 指针代达阈值必须 SKIP")
    print("G3 PASS  pointer + stale legacy 同基名 → 指针优先不回退"
          "（2 行指针代压过 100 行 stale legacy，与 formal 同口径）")


def test_G4_best_file_not_head1(base):
    """G4：best-file 语义——多候选取最大行数，不做字典序 head -1。
    历史坑：E109 SKIP 曾按 head -1 单文件误判（首文件 partial 时完整
    格被重跑）。字典序首文件 1 行、次文件 3 行、指针候选 2 行 →
    探针必须取 3 行 complete；head -1 行为会取 1 行 partial/RUN。"""
    d = os.path.join(base, "g4", "out", "L32768", "pred_stub")
    _write_legacy(d, "vt-stubm-1111.jsonl", 1)     # 字典序首（partial）
    _write_legacy(d, "vt-stubm-9999.jsonl", 3)     # 最大行数
    r = _probe(d, max_num=3)
    _check_state(r, "complete", n=3, src="legacy", tag="G4-best")
    _check("BASE=vt-stubm-9999.jsonl" in r.stdout,
           f"G4: best-file 仲裁应选 9999（3 行），得到 {r.stdout!r}")
    _check(_decision(r) == "SKIP",
           "G4: head -1 行为会取 1111（1 行）→ RUN 重跑；best-file "
           "语义必须 SKIP")
    # 混合通道：加入指针候选（2 行）→ 仍取最大 3 行 legacy
    out, rc = _run_producer(base, "g4", "success", "P", rows=2)
    _check(rc == 0, f"G4: 生产者提交失败 rc={rc}: {out[-300:]}")
    r2 = _probe(d, max_num=5)
    _check_state(r2, "partial", n=3, tag="G4-mixed")
    print("G4 PASS  best-file 语义：多候选取最大行数（1/3/2 行混合 → 3），"
          "不做字典序 head -1（E109 SKIP 历史 trap 不回归）")


def test_G5_invalid_pointer_fail_closed(base):
    """G5：损坏指针三负例 → STATE=invalid 非零退出，决策 ERROR
    （协议错误不当需要重跑处理——run_ruler_e109.sh PROBE-FAIL 分支）。
    全部在真实提交代上注入损坏（先生产者真实提交，再破坏），非手写
    伪指针。"""
    cases = []
    # a) 指针指向不存在的 gen 目录（内容被截断/改写）
    out, rc = _run_producer(base, "g5a", "success", "A", rows=3)
    _check(rc == 0, f"G5a: 生产者提交失败 rc={rc}: {out[-300:]}")
    ptr = generation_pointer_path(_logical(base, "g5a"))
    with open(ptr, "w", encoding="utf-8") as f:
        f.write("no-such-gen-directory\n")
    cases.append(("g5a", "指向缺失目录"))
    # b) 指针内容路径逃逸（非法 generation 目录名）
    out, rc = _run_producer(base, "g5b", "success", "A", rows=3)
    _check(rc == 0, f"G5b: 生产者提交失败 rc={rc}: {out[-300:]}")
    with open(generation_pointer_path(_logical(base, "g5b")), "w",
              encoding="utf-8") as f:
        f.write("../escape\n")
    cases.append(("g5b", "路径逃逸"))
    # c) gen 目录缺完成回执（gen 未就绪的死亡中间态）
    out, rc = _run_producer(base, "g5c", "success", "A", rows=3)
    _check(rc == 0, f"G5c: 生产者提交失败 rc={rc}: {out[-300:]}")
    gi = resolve_generation_pointer(_logical(base, "g5c"))
    _check(gi is not None, "G5c 前提：真实提交代指针可解析")
    os.remove(gi["rcp_path"])
    cases.append(("g5c", "gen 缺回执"))
    for name, why in cases:
        r = _probe(_pred_dir(base, name), max_num=3)
        _check(r.returncode != 0,
               f"{name}（{why}）: 损坏指针必须非零退出，rc={r.returncode}"
               f" out={r.stdout[-400:]}")
        _check("STATE=invalid" in r.stdout,
               f"{name}: 必须显式输出 STATE=invalid: {r.stdout!r}")
        _check("[GATE-FAIL]" in r.stdout,
               f"{name}: 必须 [GATE-FAIL] 原因透传: {r.stdout!r}")
        _check(_decision(r) == "PROBE-FAIL",
               f"{name}: 损坏指针的 shell 决策必须是 PROBE-FAIL"
               f"（不当需要重跑），得到 {_decision(r)}")
    print("G5 PASS  损坏指针三负例（指空目录/路径逃逸/缺回执）→ "
          "STATE=invalid 非零退出 + [GATE-FAIL] 透传 + 决策 ERROR"
          "（协议错误不盲目重跑烧 GPU）")


def test_G6_crash_before_switch_old_gen_visible(base):
    """G6：crash-before-switch（指针未切）→ 探针仍见旧完整代 complete
    （断点续跑语义——066/crash-recovery 崩溃相位在调度侧的消费验收）；
    首跑 crash-mid（无指针）→ missing（partial gen 不被计入）。"""
    out, rc = _run_producer(base, "g6", "success", "A", rows=3)
    _check(rc == 0, f"G6: 第一代提交失败 rc={rc}: {out[-300:]}")
    out, rc = _run_producer(base, "g6", "crash-between", "B", rows=5)
    _check(rc == 9, f"G6: crash-between 须硬杀 rc=9，得到 {rc}")
    r = _probe(_pred_dir(base, "g6"), max_num=3)
    _check_state(r, "complete", n=3, src="pointer", tag="G6-old-gen")
    _check(_decision(r) == "SKIP",
           "G6: 指针未切 → 旧完整代必须 SKIP（断点续跑不重跑已完成格）")
    # 首跑即崩：无指针、gen 残留 partial → missing/RUN（partial 不计入）
    out, rc = _run_producer(base, "g6b", "crash-mid", "B", rows=3,
                           crash_after=1)
    _check(rc == 9, f"G6b: crash-mid 须硬杀 rc=9，得到 {rc}")
    gens = [f for f in os.listdir(_pred_dir(base, "g6b"))
            if ".gen-" in f]
    _check(gens, "G6b 前提：crash-mid 应留下未被引用的 partial gen 目录")
    r2 = _probe(_pred_dir(base, "g6b"), max_num=3)
    _check_state(r2, "missing", n=0, tag="G6b-no-pointer")
    _check(_decision(r2) == "RUN", "G6b: 无指针首跑崩 → missing/RUN")
    print("G6 PASS  crash-before-switch → 旧完整代 complete/SKIP；首跑"
          " crash-mid → missing/RUN（partial gen 不进调度口径）")


def test_G7_lock_concurrent_only_committed(base):
    """G7：锁内并发提交——生产者持锁生成期间探针只认已提交代（不见
    在飞 partial）；提交完成后只认新提交代。指针单次原子切换保证探针
    无锁读取到的恒为完整提交代（partial 只存在于未被引用的 gen 目录）。"""
    out, rc = _run_producer(base, "g7", "success", "A", rows=3)
    _check(rc == 0, f"G7: 第一代提交失败 rc={rc}: {out[-300:]}")
    d = _pred_dir(base, "g7")
    # holder 获生产锁（确定性同步：等到「已获锁」marker 再起后继）
    holder = _run_producer(base, "g7", "lock-hold", "H", hold=12.0,
                           wait=False)
    _wait_marker(holder, "已获锁")
    # B 阻塞在锁上（同输出路径同锁键）；阻塞期探针必须只见已提交的 A
    proc_b = _run_producer(base, "g7", "success", "B", rows=5, wait=False)
    time.sleep(3.0)   # B boot + 取锁阻塞窗口
    r = _probe(d, max_num=3)
    _check_state(r, "complete", n=3, src="pointer", tag="G7-blocked")
    _check("N=3" in r.stdout and "N=5" not in r.stdout,
           f"G7: 持锁期探针不得见 B 代在飞 partial/提前提交"
           f"（只认提交代 A），得到 {r.stdout!r}")
    b_out = proc_b.communicate()[0]
    _check(proc_b.returncode == 0,
           f"G7: B 代须在锁释放后成功提交，rc={proc_b.returncode}:"
           f" {b_out[-300:]}")
    holder.communicate()
    r2 = _probe(d, max_num=5)
    _check_state(r2, "complete", n=5, src="pointer", tag="G7-committed")
    print("G7 PASS  锁内并发：持锁生成期探针只认已提交代 A（N=3 非 B "
          "在飞 5 行）；B 提交后只认新提交代（N=5）")


# ================================================================ 070 回执证据闭包

def _corrupt_committed_gen(base, name, why, raw_bytes=None, mutate=None):
    """070 负例构造：先真实生产提交（v2 回执 + 指针），再在 committed
    generation 上改写 gen 内回执——审计原文的可达触发条件（预测足量 +
    回执存在但无效/错代），非手写伪指针。返回 (pred_dir, why)。

    破坏前先断言探针 complete/SKIP（合法提交代不被共享校验器误伤，
    负例非平凡——负例红证明 complete 判定确实行使回执校验）。"""
    out, rc = _run_producer(base, name, "success", "A", rows=3)
    _check(rc == 0, f"{name}: 生产者提交失败 rc={rc}: {out[-300:]}")
    d = _pred_dir(base, name)
    r0 = _probe(d, max_num=3)
    _check_state(r0, "complete", n=3, src="pointer", tag=f"{name}-pre-green")
    _check(_decision(r0) == "SKIP",
           f"{name}: 破坏前合法提交代必须 complete/SKIP（共享校验器"
           f"不得误伤真实代），得到 {_decision(r0)}")
    gi = resolve_generation_pointer(_logical(base, name))
    _check(gi is not None, f"{name} 前提：真实提交代指针可解析")
    rcp_path = gi["rcp_path"]
    if raw_bytes is not None:
        with open(rcp_path, "wb") as f:
            f.write(raw_bytes)
    else:
        with open(rcp_path, encoding="utf-8") as f:
            rcp = json.load(f)
        mutate(rcp)
        with open(rcp_path, "w", encoding="utf-8") as f:
            json.dump(rcp, f, ensure_ascii=False, indent=1)
    return d, why


def test_G8_receipt_binding_fail_closed(base):
    """G8（070 审计复现格，#200）：「预测足量 + 回执存在但无效」六负例
    ——修复前探针只数行数判 complete → SKIP（调度称完成、正式汇总
    fail-closed 拒收的完成定义分裂）；修复后共享校验器必须全部
    STATE=invalid 非零退出 + [GATE-FAIL] 透传 + 决策 PROBE-FAIL。
    负例清单（GPT 建议 4 全集）：回执非 JSON / 回执 {} / status!=complete
    / prediction_basename 错 / prediction_sha256 错 / prediction_lines 错。"""
    import hashlib
    cases = [
        ("g8a", "回执非 JSON（半写截断）", {"raw_bytes":
            b'{"receipt_version": "producer-yarn-config-v'}),
        ("g8b", "回执 {}（合法 JSON 空对象）", {"raw_bytes": b"{}"}),
        ("g8c", "status != complete", {"mutate":
            lambda r: r.__setitem__("status", "partial")}),
        ("g8d", "prediction_basename 错", {"mutate":
            lambda r: r.__setitem__("prediction_basename",
                                    "cwe-xxx-01010000.jsonl")}),
        ("g8e", "prediction_sha256 错（合法 hex 假值）", {"mutate":
            lambda r: r.__setitem__("prediction_sha256", "f" * 64)}),
        ("g8f", "prediction_lines 错", {"mutate":
            lambda r: r.__setitem__("prediction_lines", 999)}),
    ]
    for name, why, kw in cases:
        d, _ = _corrupt_committed_gen(base, name, why, **kw)
        r = _probe(d, max_num=3)
        _check(r.returncode != 0,
               f"{name}（{why}）: 无效回执必须非零退出，rc={r.returncode}"
               f" out={r.stdout[-400:]}")
        _check("STATE=invalid" in r.stdout,
               f"{name}: 必须显式输出 STATE=invalid: {r.stdout!r}")
        _check("[GATE-FAIL]" in r.stdout,
               f"{name}: 必须 [GATE-FAIL] 原因透传: {r.stdout!r}")
        _check(_decision(r) == "PROBE-FAIL",
               f"{name}（{why}）: 无效回执的 shell 决策必须是 PROBE-FAIL"
               f"（调度与正式交付完成定义统一，不 SKIP 假完成），"
               f"得到 {_decision(r)}")
        # 反证区分：invalid ≠ partial/missing（不触发盲目重跑烧 GPU）
        _check("STATE=partial" not in r.stdout and
               "STATE=missing" not in r.stdout,
               f"{name}: 无效回执不得被解释为需要重跑: {r.stdout!r}")
    # 对照正例：未破坏的真实提交代（同 fixture 生产）仍 complete/SKIP，
    # 且 gen 预测字节与回执逐位一致（共享校验器对合法 v2 代零误伤）
    out, rc = _run_producer(base, "g8ok", "success", "A", rows=3)
    _check(rc == 0, f"g8ok: 生产者提交失败 rc={rc}: {out[-300:]}")
    gi = resolve_generation_pointer(_logical(base, "g8ok"))
    with open(gi["pred_path"], "rb") as f:
        actual_sha = hashlib.sha256(f.read()).hexdigest()
    with open(gi["rcp_path"], encoding="utf-8") as f:
        rcp = json.load(f)
    _check(rcp["prediction_sha256"] == actual_sha and
           rcp["prediction_lines"] == 3,
           "g8ok 前提：真实提交代回执与预测字节逐位一致（负例的绿基线）")
    r = _probe(_pred_dir(base, "g8ok"), max_num=3)
    _check_state(r, "complete", n=3, src="pointer", tag="G8-ok-control")
    _check(_decision(r) == "SKIP", "G8: 合法 v2 提交代必须仍 SKIP")
    print("G8 PASS  提交代回执破坏六负例（非JSON/{}/status/basename/SHA/"
          "行数）→ STATE=invalid + [GATE-FAIL] + PROBE-FAIL；合法 v2 代"
          "零误伤仍 SKIP（070 完成定义分裂闭合：调度=正式交付口径）")


# ================================================================ 069 scorer

def _scorer(root, out, expect_tasks=None, postfix="_stub", direct_script=False,
            min_samples="1"):
    """score_ruler direct CLI 子进程（-m 包模式或直接脚本模式）。"""
    if direct_script:
        argv = [sys.executable, "-u", SCORER]
    else:
        argv = [sys.executable, "-u", "-m", "benchmark.RULER.score_ruler"]
    argv += ["--root", root, "--pred-postfix", postfix, "--out", out,
             "--min-samples", min_samples]
    if expect_tasks is not None:
        argv += ["--expect-tasks", str(expect_tasks)]
    return subprocess.run(argv, capture_output=True, text=True, cwd=REPO,
                          env={**os.environ, "PYTHONPATH": REPO})


def test_N1_direct_scorer_pointer_root_fail_loudly(base):
    """N1（069）：direct scorer 对 pointer-only root → 非零退出 +
    [GATE-FAIL] + formal 提示 + 零输出文件。有无 --expect-tasks、
    -m 与直接脚本两种调用方式都必须拒（修复前无门禁时 exit 0 写空
    {"scores": {}, ...} 冒充正式外观产物）。"""
    out, rc = _run_producer(base, "n1", "success", "A", rows=1)
    _check(rc == 0, f"N1: 生产者提交失败 rc={rc}: {out[-300:]}")
    root = os.path.join(base, "n1", "out")
    out_json = os.path.join(base, "n1", "res.json")
    for tag, expect_tasks, direct in [
            ("无门禁/-m", None, False),
            ("expect-tasks/-m", 1, False),
            ("无门禁/直接脚本", None, True)]:
        r = _scorer(root, out_json, expect_tasks=expect_tasks,
                    direct_script=direct)
        blob = r.stdout + r.stderr
        _check(r.returncode != 0,
               f"N1[{tag}]: pointer-only root 必须非零退出，"
               f"rc={r.returncode} out={blob[-400:]}")
        _check("[GATE-FAIL]" in blob, f"N1[{tag}]: 缺 [GATE-FAIL]: {blob!r}")
        _check("score_ruler_formal" in blob,
               f"N1[{tag}]: 必须提示改用 formal（069 单口径原则）: {blob!r}")
        _check(not os.path.exists(out_json),
               f"N1[{tag}]: 不得写任何输出文件（修复前空结果冒充正式外观）")
        _check(not os.path.exists(out_json[:-len(".json")] + ".md"),
               f"N1[{tag}]: 不得写 MD 输出文件")
    print("N1 PASS  direct scorer pointer-only root：三种调用 × 有无门禁"
          "全拒——非零退出 + [GATE-FAIL] + formal 提示 + JSON/MD 零落盘")


def test_N2_direct_scorer_legacy_root_regression(base):
    """N2（069 对照）：legacy-only root 正常打分不回归——exit 0、分数
    正确（string_match_all=100）、JSON/MD 落盘。"""
    d = os.path.join(base, "n2", "out", "L32768", "pred_stub")
    os.makedirs(d, exist_ok=True)
    import hashlib
    asha = hashlib.sha256(json.dumps(
        ["gt0"], ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    with open(os.path.join(d, "vt-stubm-01010000.jsonl"), "w",
              encoding="utf-8") as f:
        f.write(json.dumps({"pred": "gt0", "answers": ["gt0"],
                            "length": 32768, "budget": 0,
                            "_id": "vt:0", "_answers_sha": asha},
                           ensure_ascii=False) + "\n")
    out_json = os.path.join(base, "n2", "res.json")
    r = _scorer(os.path.join(base, "n2", "out"), out_json)
    _check(r.returncode == 0,
           f"N2: legacy root 不得被门禁误伤，rc={r.returncode}: "
           f"{(r.stdout + r.stderr)[-400:]}")
    _check(os.path.isfile(out_json), "N2: 结果 JSON 必须落盘")
    res = json.load(open(out_json))
    got = res["scores"].get("L32768/stubm", {}).get("vt")
    _check(got == 100.0, f"N2: 分数须 100.0（string_match_all 全命中），"
                         f"得到 {got}")
    _check(res["n"]["L32768/stubm"]["vt"] == 1, "N2: n 须为 1")
    _check(os.path.isfile(out_json[:-len(".json")] + ".md"),
           "N2: MD 表必须落盘")
    print("N2 PASS  legacy-only root 对照：exit 0 + score=100.0 + n=1 + "
          "JSON/MD 落盘（direct CLI 的 legacy-direct 支持边界零回归）")


def test_N3_shell_wiring_static():
    """N3：run_ruler_e109.sh 接线静态检查——bash -n 语法过 + 探针调用
    在位 + 旧 `ls | head -1` glob 判定已移除 + 四态分支文本在位
    （ARM/TASKS/EXTRA 参数签名不变由 bash -n + 原文比对保证）。"""
    r = subprocess.run(["bash", "-n", E109_SH], capture_output=True,
                       text=True)
    _check(r.returncode == 0,
           f"N3: run_ruler_e109.sh 语法错误: {r.stderr}")
    c = open(E109_SH, encoding="utf-8").read()
    _check("gen_completion_probe.py" in c,
           "N3: 调度脚本必须调用 gen_completion_probe.py（068 修复接线）")
    _check("ls $OUTDIR/$T-*.jsonl" not in c,
           "N3: 旧 `ls $T-*.jsonl` legacy-only 判定必须移除"
           "（pointer-only 已完成格不可见——068 违反事实）")
    _check('"$STATE" = "complete"' in c, "N3: complete→SKIP 分支必须存在")
    _check("PROBE-FAIL" in c, "N3: invalid→PROBE-FAIL 分支必须存在")
    # 参数签名不变（未来批次派单路径兼容）
    _check('bash benchmark/RULER/run_ruler_e109.sh <arm> <gpu_id> [max_num] [lens]'
           in c, "N3: 用法/参数签名必须保持不变")
    print("N3 PASS  run_ruler_e109.sh 接线：bash -n 过 + 探针调用在位 + "
          "旧 glob 判定移除 + 四态分支 + 参数签名不变")


# ================================================================ 红探针

def test_RP_red_probe_ignore_pointer(base):
    """RP 红探针：打桩 _discover_candidates(include_pointers=False) 复刻
    修复前 shell「只看 .jsonl glob」行为 → pointer-only complete 判定
    必须红（非 complete）——证明 G1 的 complete/SKIP 断言确实行使指针
    解析，不是平凡通过；恢复后复绿。"""
    out, rc = _run_producer(base, "rp", "success", "A", rows=3)
    _check(rc == 0, f"RP: 生产者提交失败 rc={rc}: {out[-300:]}")
    d = _pred_dir(base, "rp")
    import benchmark.RULER.gen_completion_probe as GP
    info = GP.probe_task(d, "vt", 3)
    _check(info["state"] == "complete" and info["n"] == 3,
           f"RP 前提（绿）：真实探针须 complete N=3，得到 {info}")
    orig = GP._discover_candidates

    def _legacy_only(out_dir, task, include_pointers=True):
        # 复刻修复前 run_ruler_e109.sh 的候选发现：只 glob 直写 .jsonl
        return orig(out_dir, task, include_pointers=False)

    GP._discover_candidates = _legacy_only
    try:
        info2 = GP.probe_task(d, "vt", 3)
        _check(info2["state"] != "complete",
               f"RP 红探针：忽略指针（复刻修复前 shell glob 行为）时 "
               f"pointer-only complete 判定仍为 complete——G1 断言未真正"
               f"行使指针解析（平凡通过风险），得到 {info2}")
        _check(info2["state"] == "missing",
               f"RP: 忽略指针时 pointer-only 格应为 missing（无 legacy "
               f"直写文件），得到 {info2}")
    finally:
        GP._discover_candidates = orig
    info3 = GP.probe_task(d, "vt", 3)
    _check(info3["state"] == "complete",
           f"RP 恢复后必须复绿（complete），得到 {info3}")
    print("RP PASS  红探针：打桩忽略指针（复刻修复前行为）→ pointer-only "
          "complete 判定红（missing）；恢复后复绿——G1 断言确实行使"
          "指针解析")


# ================================================================ 070 只读预检

def _audit(root, module_mode=False):
    """--audit-dir 只读预检子进程（直接脚本 / -m 包两种调用）。"""
    if module_mode:
        argv = [sys.executable, "-u", "-m",
                "benchmark.RULER.gen_completion_probe"]
    else:
        argv = [sys.executable, "-u", PROBE]
    argv += ["--audit-dir", root]
    return subprocess.run(argv, capture_output=True, text=True, cwd=REPO,
                          env={**os.environ, "PYTHONPATH": REPO})


def _tree_snapshot(root):
    """目录全量快照（相对路径 → 内容 SHA256），只读预检的零写验收锚。"""
    import hashlib
    snap = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in sorted(filenames):
            p = os.path.join(dirpath, fn)
            with open(p, "rb") as f:
                snap[os.path.relpath(p, root)] = \
                    hashlib.sha256(f.read()).hexdigest()
    return snap


def test_A1_audit_dir_readonly_prec_check(base):
    """A1（070 建议 5）：--audit-dir 只读预检——遍历目录全部 pointer，
    逐格验证回执绑定。全 valid → rc=0；含破坏 pointer → rc=2 +
    invalid 格清单（定点重跑决策输入）；零 pointer → 如实报告 rc=0；
    全程零写（快照逐位一致，E109 已收口数据零改动纪律）。"""
    # 三个真实提交代 + 一个破坏代（回执 {}），分置不同子目录
    for name in ("a1v1", "a1v2", "a1v3"):
        out, rc = _run_producer(base, name, "success", "A", rows=3)
        _check(rc == 0, f"A1: 生产者提交失败（{name}）rc={rc}: {out[-200:]}")
    d, _ = _corrupt_committed_gen(base, "a1bad", "回执 {}", raw_bytes=b"{}")
    root = os.path.join(base)
    # 破坏代目录里只有它自己的指针；把 a1bad 的 pred 目录整体挪进独立
    # 子树，避免与其他 valid 代混目录（--audit-dir 逐 pointer 独立验证）
    bad_sub = os.path.join(base, "bad_cell", "L32768", "pred_stub")
    os.makedirs(os.path.dirname(bad_sub), exist_ok=True)
    shutil.move(d, bad_sub)
    # 零 pointer 目录（legacy 直写无指针，也在 base 内）先建好，
    # 再拍全量快照——测试自身的建目录动作不得混进只读验收窗口
    empty = os.path.join(base, "a1empty", "out")
    os.makedirs(os.path.join(empty, "L32768", "pred_stub"), exist_ok=True)
    _write_legacy(os.path.join(empty, "L32768", "pred_stub"),
                  "vt-stubm-01010000.jsonl", 3)
    valid_root = os.path.join(base)   # 含 a1v1..a1v3 + bad_cell + a1empty
    snap = _tree_snapshot(valid_root)
    # ① 全量 root：3 valid + 1 invalid → rc=2 + invalid 清单
    r = _audit(valid_root)
    _check(r.returncode == 2,
           f"A1: 含 invalid pointer 的预检必须 rc=2，得到 {r.returncode}"
           f" out={r.stdout[-600:]}")
    _check(r.stdout.count("POINTER") >= 4,
           f"A1: 4 个 pointer 须逐一报告，得到 {r.stdout!r}")
    _check("INVALID" in r.stdout,
           f"A1: 必须输出 invalid 格清单: {r.stdout!r}")
    _check("a1bad" in r.stdout or "bad_cell" in r.stdout,
           f"A1: invalid 清单必须指认破坏格: {r.stdout!r}")
    _check("[GATE-FAIL]" in r.stdout,
           f"A1: invalid 项必须带拒绝原因: {r.stdout!r}")
    # ② -m 包模式同口径
    r_m = _audit(valid_root, module_mode=True)
    _check(r_m.returncode == 2 and "INVALID" in r_m.stdout,
           f"A1: -m 包模式预检同口径，rc={r_m.returncode} "
           f"out={r_m.stdout[-400:]}")
    # ③ valid-only 子树 → rc=0
    r_ok = _audit(os.path.join(base, "a1v1"))
    _check(r_ok.returncode == 0 and "invalid=0" in r_ok.stdout,
           f"A1: 全 valid root 须 rc=0 且 invalid=0，rc={r_ok.returncode}"
           f" out={r_ok.stdout[-400:]}")
    # ④ 零 pointer 目录 → 如实报告 rc=0（不伪造对象；目录在快照前已建）
    r0 = _audit(empty)
    _check(r0.returncode == 0 and "total=0" in r0.stdout,
           f"A1: 零 pointer 目录须如实报告 total=0 rc=0，"
           f"rc={r0.returncode} out={r0.stdout[-400:]}")
    # ⑤ 只读验收：全程零写（快照逐位一致）
    _check(_tree_snapshot(valid_root) == snap,
           "A1: 只读预检不得写/改任何数据文件（E109 已收口数据零改动）")
    print("A1 PASS  --audit-dir 只读预检：4 pointer 逐格验证（3 OK + 1 "
          "invalid → rc=2 + 清单）；valid-only rc=0；零 pointer 如实"
          " total=0；目录快照逐位一致零写")


# ================================================================ main

def main():
    """067 纪律：PASS/SKIP/FAIL 三分显式计数；异常 → FAIL 继续跑完
    其余用例（一次运行给出完整红绿图）；SKIP>0 或 FAIL>0 不打
    ALL PASS 且非零退出。E119_ONLY=NAME1,NAME2 子集过滤。"""
    base = tempfile.mkdtemp(prefix="e119_ptr_consumer_199_")
    plan = [
        ("G1", lambda: test_G1_pointer_only_complete_partial(base)),
        ("G2", lambda: test_G2_legacy_only_three_states(base)),
        ("G3", lambda: test_G3_pointer_priority_over_stale_legacy(base)),
        ("G4", lambda: test_G4_best_file_not_head1(base)),
        ("G5", lambda: test_G5_invalid_pointer_fail_closed(base)),
        ("G6", lambda: test_G6_crash_before_switch_old_gen_visible(base)),
        ("G7", lambda: test_G7_lock_concurrent_only_committed(base)),
        ("G8", lambda: test_G8_receipt_binding_fail_closed(base)),
        ("N1", lambda: test_N1_direct_scorer_pointer_root_fail_loudly(base)),
        ("N2", lambda: test_N2_direct_scorer_legacy_root_regression(base)),
        ("N3", test_N3_shell_wiring_static),
        ("RP", lambda: test_RP_red_probe_ignore_pointer(base)),
        ("A1", lambda: test_A1_audit_dir_readonly_prec_check(base)),
    ]
    only = os.environ.get("E119_ONLY", "")
    if only:
        keep = {x.strip() for x in only.split(",") if x.strip()}
        plan = [p for p in plan if p[0] in keep]
    n_pass = n_skip = n_fail = 0
    failed = []
    try:
        for name, fn in plan:
            try:
                status = fn()
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
                      f"{traceback.format_exc()[-1500:]}", flush=True)
                continue
            if status == "SKIP":
                n_skip += 1
            else:
                n_pass += 1
    finally:
        shutil.rmtree(base, ignore_errors=True)
    print(f"\nE119-POINTER-CONSUMER-068/069 RESULT: "
          f"PASS={n_pass} SKIP={n_skip} FAIL={n_fail} (total {len(plan)})")
    if n_skip == 0 and n_fail == 0:
        print(f"E119-POINTER-CONSUMER-068/069 ALL PASS "
              f"({n_pass}/{len(plan)})")
        return 0
    if failed:
        print(f"FAILED: {failed}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
