#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""068（TL-E119-POINTER-SKIP，GPT 2026-10-10 1330 审计，#199）+
070（TL-E119-PROBE-RECEIPT-VALIDATION，GPT 2026-10-10 1528 审计，#200）：
E109 调度器 SKIP 幂等判定的 generation 指针感知探针（零重依赖 CLI）。

违反事实 068（审计已 CPU 复现）：#198 指针协议（066/crash-recovery）下
生产者 pred_ruler.py 成功提交后只留 {out}.tli_gen 指针 + 不可变
generation 目录内 JSONL，逻辑 {out}.jsonl 刻意不落盘；而
run_ruler_e109.sh 的断点续跑判定仍只 `ls $OUTDIR/$T-*.jsonl | head -1`
——pointer-only 已完成格确定性不可见（SKIP_CONDITION=false），重启会
重跑昂贵的 32K/64K/128K GPU 任务，断点续跑失效。

违反事实 070：探针对 pointer 候选只调 resolve_generation_pointer（明确
不校验回执内容）后数预测行数判 complete——「预测足量 + 回执存在但无效
（半写 JSON/未知 schema/非 complete/basename/SHA/行数失配）」被判
complete → 调度 SKIP，正式汇总却 fail-closed 拒收：调度与交付的完成
定义分裂，坏格永久 SKIP。修复：pointer 完成判定在 resolve 后调用共享
校验器 yarn_receipt.validate_committed_generation（与 formal
score_ruler_formal.py 单口径）——回执 bytes 快照 → JSON → 严格
schema → v2 同代字节绑定（实际 SHA/行数 vs 回执声明逐位比对）任一
失败 → STATE=invalid 非零退出（PROBE-FAIL，人工介入；不盲目重跑）。
注意区分：invalid = 存在即证据但绑定非法（协议错误）；partial/missing
仍是调度重跑信号；legacy 直写分支维持行数语义零改动（不被 pointer
回执门禁误伤）。

本探针是 shell 与指针协议之间的唯一桥（不在 shell 里复制协议解析）：
  - 候选发现双通道（与 score_ruler_formal.py formal 入口 782-807 同口径）：
    ① legacy 直写文件（{task}-*.jsonl glob，E116d 口径排除 -merged 派生物）；
    ② 指针文件（{task}-*.jsonl.tli_gen glob）→ resolve_generation_pointer
    解析 gen 目录内预测。同基名双通道并存 → 指针优先（指针切换 = 唯一
    提交信号，直写残留只可能来自更早旧协议运行）。
  - best-file 语义：多候选取最大行数（并列取时间戳最新，与
    score_ruler._resolve_group 仲裁同口径）。历史坑：旧 shell 的
    `head -1` 字典序取首文件，首文件是 partial 时完整格被误判重跑
    （E109 SKIP 幂等 trap，13:3X 监督轮实锤）——逐候选枚举不做 head -1。
  - fail-closed：指针存在但损坏/指向缺失目录/缺预测/缺回执/回执或
    同代绑定不合法 → resolve_generation_pointer 或
    validate_committed_generation 内部 SystemExit → STATE=invalid +
    非零退出（存在即证据，不回退 stale legacy——协议错误不当「需要
    重跑」处理）。
  - 探针不取生产者输出路径锁：指针单次原子切换（commit_yarn_generation
    单次 os.replace）保证任何读取瞬间解析到的都是完整提交代；partial
    只存在于未被指针引用的 gen 目录（目录名不以 .jsonl 结尾，不进 glob）。
    锁窗口内的换代语义由测试 G7 验收（阻塞期只认已提交代）。

输出（stdout 单行，机器可解析）：
  STATE=complete N=100 BASE=vt-xxx-09090909.jsonl SRC=pointer|legacy
  STATE=partial  N=3   BASE=... SRC=...
  STATE=missing  N=0   BASE=- SRC=-
  STATE=invalid  REASON=[GATE-FAIL] ...   （exit 2）
四态 → run_ruler_e109.sh 调度决策：
  complete（N>=max_num）→ SKIP；partial / missing → 跑；
  invalid → PROBE-FAIL 非零退出（协议错误，非需重跑）。

另附只读预检模式（070 建议 5）：
  python benchmark/RULER/gen_completion_probe.py --audit-dir <root>
遍历 root 下全部 {*.jsonl}.tli_gen 指针，逐格验证回执绑定（共享校验器
同口径），输出 OK/INVALID 清单与汇总；INVALID 存在 → exit 2。全程零写
（E109 已收口数据零改动），供定点重跑决策，不修改任何数据。

用法：
  python benchmark/RULER/gen_completion_probe.py \
      --out-dir exp/results_ruler/e109_full_Qwen3-8B/mavg/L65536/pred_1024 \
      --task niah_single_1 --max-num 100
（或 python -m benchmark.RULER.gen_completion_probe，PYTHONPATH 指仓库根）

本模块刻意零重依赖（不 import torch/transformers/sparse_attn）——与
yarn_receipt.py 共享同一指针解析与回执校验口径，干净检出恒可单测。
"""
import argparse
import glob
import os
import sys

if __package__:
    from benchmark.RULER.yarn_receipt import (  # noqa: E402
        GENERATION_POINTER_SUFFIX, resolve_generation_pointer,
        validate_committed_generation)
else:
    # 直接脚本调用（无包上下文）：脚本目录即 benchmark/RULER
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    from yarn_receipt import (  # noqa: E402
        GENERATION_POINTER_SUFFIX, resolve_generation_pointer,
        validate_committed_generation)

# E116d 口径：merged 规范文件是上轮仲裁的派生物，不参与候选竞争
MERGED_SUFFIX = "-merged.jsonl"

# 四态输出（与 docstring 的 shell 决策契约一致）
STATE_COMPLETE = "complete"
STATE_PARTIAL = "partial"
STATE_MISSING = "missing"
STATE_INVALID = "invalid"


def _count_lines(path):
    """行数计数（与 wc -l / score_ruler._nlines 同语义：逐行迭代）。"""
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def _ts_of(basename):
    """逻辑 basename 的末段时间戳（行数并列时取最新）。

    与 score_ruler._ts_of 同口径：去 .jsonl 后按最后一个 '-' 分段。
    指针候选的 basename 是逻辑名（{task}-{method}-{t}.jsonl），与
    legacy 直写文件同名同构，仲裁规则跨通道一致。"""
    return basename[:-len(".jsonl")].rsplit("-", 1)[-1]


def _discover_candidates(out_dir, task, include_pointers=True):
    """单任务候选发现（legacy 直写 + 指针代双通道）。

    返回 {逻辑 basename: {"logical": 最终逻辑路径, "pointer": bool}}，
    键排序确定（fail-closed 报错顺序稳定）。-merged 派生物两通道一致
    排除（E116d：它是上轮仲裁产物，非原始 run）。
    include_pointers=False 仅供红探针使用（复刻修复前 shell 只看
    .jsonl glob 的行为，验证 pointer-only 判定确实依赖指针解析）。
    目录不存在 → 空候选（missing，不算协议错误）。"""
    cand = {}
    if not os.path.isdir(out_dir):
        return cand
    for f in sorted(glob.glob(os.path.join(out_dir, f"{task}-*.jsonl"))):
        b = os.path.basename(f)
        if b.endswith(MERGED_SUFFIX):
            continue
        cand[b] = {"logical": f, "pointer": False}
    if include_pointers:
        for p in sorted(glob.glob(
                os.path.join(out_dir, f"{task}-*{GENERATION_POINTER_SUFFIX}"))):
            logical = p[: -len(GENERATION_POINTER_SUFFIX)]
            b = os.path.basename(logical)
            if b.endswith(MERGED_SUFFIX):
                continue
            # 指针优先（formal 782-807 同口径）：同基名双通道并存时指针代
            # 覆盖 legacy 直写残留（新代码不写直写文件，残留只属旧协议）
            cand[b] = {"logical": logical, "pointer": True}
    return cand


def probe_task(out_dir, task, max_num):
    """单 (out_dir, task) 格的调度判定核心（纯函数式，供 CLI 与测试共用）。

    返回 dict {"state", "n", "base", "source"}：
      - missing：无任何候选（含目录不存在）→ 调度决策 RUN；
      - complete：best-file 行数 >= max_num → SKIP；
      - partial：0 <= best 行数 < max_num → RUN；
      - invalid：不返回——指针候选损坏/缺件时 resolve_generation_pointer
        或 validate_committed_generation（070：回执内容 + v2 同代绑定）
        的 SystemExit 直接穿透（fail-closed：存在即证据，不回退 stale
        legacy、不把协议错误当需要重跑）。
    best-file 语义：跨候选取 (行数, 时间戳) 最大——不做字典序 head -1
    （历史坑：E109 SKIP 曾按 head -1 单文件误判 partial 首文件）。

    070：pointer 完成判定的证据闭包——resolve 只证明「gen 目录与两文件
    齐备」，不证明「回执与预测同代且合法」。共享校验器
    validate_committed_generation（与 formal 单口径）在数行数之前校验：
    回执 bytes 快照 → JSON → validate_producer_receipt（协议版本 +
    严格 schema + v2 status/basename/SHA 格式/行数格式）→ v2 实际
    SHA/行数 vs 回执声明逐位比对。任一失败 = 该格「预测足量 + 回执
    无效」——正是正式评分必拒收的坏态，调度不得 SKIP（完成定义统一）。
    v1 回执（理论边界）无绑定字段 → 共享校验器跳过字节比对，与 formal
    降级消费同口径（指针协议生产路径只产 v2）。"""
    if not isinstance(max_num, int) or isinstance(max_num, bool) \
            or max_num < 1:
        raise SystemExit(
            f"[GATE-FAIL] max_num={max_num!r} 须为正整数——调度阈值非法，"
            f"fail closed（068）")
    cand = _discover_candidates(out_dir, task)
    if not cand:
        return {"state": STATE_MISSING, "n": 0, "base": None,
                "source": None}
    best = None
    for b in sorted(cand):
        info = cand[b]
        if info["pointer"]:
            # 指针存在即证据：损坏/缺目录/缺预测/缺回执由
            # resolve_generation_pointer 内部 SystemExit fail-closed
            # （066/crash-recovery 口径，python -O 不失效）
            gen = resolve_generation_pointer(info["logical"])
            if gen is None:
                raise SystemExit(
                    f"[GATE-FAIL] {info['logical']}{GENERATION_POINTER_SUFFIX}"
                    f": 指针文件存在但 resolve 返回 None——指针协议解析"
                    f"内部矛盾，fail closed（068）")
            # 070：完成证据闭包——回执内容 + v2 同代绑定必须过共享校验器
            # （与 formal score_ruler_formal 单口径）；失败 SystemExit
            # 穿透 → STATE=invalid/PROBE-FAIL（不把坏回执格当 SKIP 假完成）
            validate_committed_generation(gen["pred_path"], gen["rcp_path"])
            n = _count_lines(gen["pred_path"])
            src = "pointer"
        else:
            # legacy 直写（glob 命中即存在）
            n = _count_lines(info["logical"])
            src = "legacy"
        if best is None or (n, _ts_of(b)) > (best["n"], _ts_of(best["base"])):
            best = {"n": n, "base": b, "source": src}
    state = STATE_COMPLETE if best["n"] >= max_num else STATE_PARTIAL
    return {"state": state, "n": best["n"], "base": best["base"],
            "source": best["source"]}


def audit_directory(root):
    """070 建议 5：只读预检——遍历 root 下全部 generation 指针，逐格
    验证回执绑定（resolve + 共享校验器同口径），输出 OK/INVALID 清单。

    全程零写（不修改任何数据——E109 已收口数据零改动纪律）。
    返回 (total, invalid_count)；invalid 项打印 [GATE-FAIL] 拒绝原因，
    供定点重跑决策。目录不存在 → 如实报零对象（非协议错误）。"""
    if not os.path.isdir(root):
        raise SystemExit(
            f"[GATE-FAIL] --audit-dir {root!r} 不存在或非目录——预检"
            f"目标非法，fail closed（070）")
    pointers = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in sorted(filenames):
            if fn.endswith(GENERATION_POINTER_SUFFIX):
                pointers.append(os.path.join(dirpath, fn))
    invalid = 0
    for ptr in sorted(pointers):
        logical = ptr[: -len(GENERATION_POINTER_SUFFIX)]
        try:
            gen = resolve_generation_pointer(logical)
            if gen is None:
                raise SystemExit(
                    f"[GATE-FAIL] {ptr}: 指针文件存在但 resolve 返回 None"
                    f"——指针协议解析内部矛盾（070 预检）")
            validate_committed_generation(gen["pred_path"], gen["rcp_path"])
        except SystemExit as e:
            invalid += 1
            print(f"POINTER {ptr} INVALID: {e}", flush=True)
            continue
        print(f"POINTER {ptr} OK", flush=True)
    return len(pointers), invalid


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="E109 SKIP 幂等判定探针（pointer-v1 + legacy 双通道，"
                    "best-file 语义，损坏指针/无效回执 fail-closed）+ "
                    "--audit-dir 只读回执绑定预检")
    ap.add_argument("--out-dir",
                    help="pred 输出目录（如 .../L65536/pred_1024）")
    ap.add_argument("--task",
                    help="任务名（glob 前缀，如 niah_single_1）")
    ap.add_argument("--max-num", type=int,
                    help="调度阈值 N（行数 >= N → complete/SKIP）")
    ap.add_argument("--audit-dir",
                    help="只读预检模式：遍历该目录下全部 .tli_gen 指针，"
                         "逐格验证回执绑定（零写；invalid 存在 → exit 2）")
    args = ap.parse_args(argv)
    if args.audit_dir is not None:
        # 只读预检模式（与单格探查互斥）
        if args.out_dir or args.task or args.max_num is not None:
            ap.error("--audit-dir 与 --out-dir/--task/--max-num 互斥")
        total, invalid = audit_directory(args.audit_dir)
        note = "（零 pointer 产物，预检无对象）" if total == 0 else ""
        print(f"AUDIT RESULT: total={total} invalid={invalid}{note}",
              flush=True)
        return 2 if invalid else 0
    if not args.out_dir or not args.task or args.max_num is None:
        ap.error("单格探查须同时提供 --out-dir/--task/--max-num"
                 "（或改用 --audit-dir 只读预检）")
    try:
        info = probe_task(args.out_dir, args.task, args.max_num)
    except SystemExit as e:
        # invalid：非零退出（shell 分支 PROBE-FAIL，不当需要重跑处理）。
        # str(SystemExit(msg)) == msg，GATE-FAIL 原因原样透传
        print(f"STATE={STATE_INVALID} REASON={e}", flush=True)
        return 2
    print(f"STATE={info['state']} N={info['n']} "
          f"BASE={info['base'] or '-'} SRC={info['source'] or '-'}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
