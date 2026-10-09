#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E113 microbench fail-closed 门 GPU 侧注入验收（GPT 审计 050 方案 4 验收红例）。

不修改生产脚本 e113_microbench.py：monkeypatch 其 load_triton_kernel，把真实
Triton kernel 的 greedy_build_triton 包一层，篡改返回的 assignment 一位（置为
不可能的簇索引），模拟「两侧实现不一致」→ 预期三件事全部发生：
  1. 非零退出（SystemExit code=1）；
  2. <out>.failure.json 落盘（含 identity / 输入 hash / gate errors）；
  3. 性能 JSON（<out> 本体）不发布。

用法：CUDA_VISIBLE_DEVICES=<idle> python3 e113_failclosed_inject.py
退出码：0 = 注入被正确拒绝（fail-closed 生效）；1 = 验收失败。
2026-10-10 方案 4 验收实跑 PASS（GPU0，worktree checkout 6bdb7eb3b 前的主树
同源代码；本脚本为该次验收的可复现存档）。
"""
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "e113_microbench.py")
OUTDIR = "/tmp/e113_inject_test"
OUT = os.path.join(OUTDIR, "e113_inject.json")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    # 清场：确保「性能 JSON 不发布」的断言从零开始成立
    for f in (OUT, OUT + ".failure.json", OUT + ".sha256"):
        if os.path.exists(f):
            os.remove(f)

    spec = importlib.util.spec_from_file_location("e113_bench_mod", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    real_load = m.load_triton_kernel

    def patched_load(tri_root):
        TRI = real_load(tri_root)
        real_fn = TRI.greedy_build_triton

        def tampered(*a, **k):
            out = real_fn(*a, **k)
            # 返回序：(sums, cnt, sq, k_live, assign)——与 check_pair 解包一致
            sums, cnt, sq, kl, assign = out
            assign = assign.clone()
            assign.view(-1)[0] = 123456789   # 不可能的簇索引，必 mismatch
            return (sums, cnt, sq, kl, assign)

        TRI.greedy_build_triton = tampered
        return TRI

    m.load_triton_kernel = patched_load

    sys.argv = ["e113_microbench.py", "--out", OUT]
    print("[INJECT] 已注入 assignment 篡改，开始跑 main()（预期首个 case 即 fail-closed）", flush=True)
    try:
        m.main()
        print("[INJECT-RESULT] FAIL：main 正常返回，fail-closed 未生效！")
        return 1
    except SystemExit as e:
        code = e.code
        print(f"[INJECT-RESULT] SystemExit code={code}")
        published = os.path.exists(OUT)
        failure = os.path.exists(OUT + ".failure.json")
        print(f"  性能 JSON 发布? {'是——异常!' if published else '否（符合预期）'}")
        print(f"  failure.json 落盘? {'是' if failure else '否——异常!'}")
        if failure:
            with open(OUT + ".failure.json", encoding="utf-8") as f:
                d = json.load(f)
            need = ["status", "gate", "identity", "failed_case", "errors"]
            missing = [k for k in need if k not in d]
            print(f"  failure.json 必需字段齐全? {'是' if not missing else '否——缺 ' + str(missing)}")
            print(f"  gate errors: {d['errors']}")
        ok = (code == 1) and (not published) and failure
        print(f"[VERDICT] {'PASS：非零退出 + failure.json 落盘 + 性能 JSON 不发布' if ok else 'FAIL'}")
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
