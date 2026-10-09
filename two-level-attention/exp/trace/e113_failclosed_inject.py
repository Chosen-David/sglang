#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E113 microbench fail-closed 门 GPU 侧注入验收（050 方案 4 + 054 状态机版）。

不修改生产脚本 e113_microbench.py：monkeypatch 其 load_triton_kernel，把真实
Triton kernel 的 greedy_build_triton 包一层，篡改返回的 assignment 一位（置为
不可能的簇索引），模拟「两侧实现不一致」。

v3（TL-E113-FAILCLOSED-054 修复后）：**不再预清场**——050 版曾在调用生产入口
前预删 OUT / failure / sidecar 三类文件，恰好绕过了「同路径重跑」这一最危险
状态转换（GPT 审计 2026-10-10_0227 复现方法）。本版反向操作：**预埋**一份假
旧成功 JSON + sidecar 在目标路径上，再注入篡改跑生产 main()，预期生产状态机
五件事全部发生：
  1. 非零退出（SystemExit code=1）；
  2. 旧成功 JSON / sidecar 被生产代码隔离改名为 *.attempt-<sha8>.superseded
     （可见路径无残留成功终态，历史保留不删除）；
  3. <out>.failure.json 落盘且 attempt.attempt_id 非空（可区分代际），
     attempt.superseded_files 记录被隔离文件；
  4. 性能 JSON（<out> 本体）不发布；
  5. failure receipt 含 identity / 输入 hash / gate errors（050 契约不回归）。

用法：CUDA_VISIBLE_DEVICES=<idle> python3 e113_failclosed_inject.py
退出码：0 = 注入被正确拒绝且状态机生效；1 = 验收失败。
2026-10-10 v3 口径复验记录：见本文件 git log（054 修复验收轮）。
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
    # ---- 054：不预清场，反向预埋假旧成功产物（制造「旧成功 → 新失败」真实转换）----
    OLD_DOC = json.dumps({"status": "old-success", "cases": []}, indent=1).encode("utf-8")
    with open(OUT, "wb") as f:
        f.write(OLD_DOC)
    with open(OUT + ".sha256", "wb") as f:
        f.write(b"old-sidecar\n")

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
    print("[INJECT] 已注入 assignment 篡改（目标路径预埋旧成功产物，不预清场），"
          "开始跑 main()（预期首个 case 即 fail-closed + 旧产物被隔离）", flush=True)
    try:
        m.main()
        print("[INJECT-RESULT] FAIL：main 正常返回，fail-closed 未生效！")
        return 1
    except SystemExit as e:
        code = e.code
        print(f"[INJECT-RESULT] SystemExit code={code}")
        published = os.path.exists(OUT)
        failure = os.path.exists(OUT + ".failure.json")
        sidecar_visible = os.path.exists(OUT + ".sha256")
        print(f"  性能 JSON 发布? {'是——异常!' if published else '否（符合预期）'}")
        print(f"  可见 sidecar 残留? {'是——异常!' if sidecar_visible else '否（符合预期）'}")
        print(f"  failure.json 落盘? {'是' if failure else '否——异常!'}")
        ok = (code == 1) and (not published) and failure and (not sidecar_visible)
        # ---- 054 状态机断言：旧产物隔离改名 + failure 属于新 attempt ----
        # 以 receipt 记录为准核验（多轮累积下同内容 seed 产生计数后缀 -1/-2…，
        # glob 顺序不可靠——消费者对账必须走 receipt 的 superseded_files 记录）
        old_quarantined, sup_side_rec, rec_ok = False, None, False
        if failure:
            with open(OUT + ".failure.json", encoding="utf-8") as f:
                d = json.load(f)
            need = ["status", "gate", "identity", "failed_case", "errors"]
            missing = [k for k in need if k not in d]
            print(f"  failure.json 050 必需字段齐全? "
                  f"{'是' if not missing else '否——缺 ' + str(missing)}")
            att = d.get("attempt", {})
            has_att = bool(att.get("attempt_id"))
            sup_rec = {r.get("visible_path"): r.get("superseded_as")
                       for r in att.get("superseded_files", [])}
            sup_out_rec = sup_rec.get(OUT)
            sup_side_rec = sup_rec.get(OUT + ".sha256")
            # receipt 记录的隔离文件必须真实存在且内容逐字节保留（历史不删除）
            old_quarantined = bool(sup_out_rec) and os.path.exists(sup_out_rec) \
                and open(sup_out_rec, "rb").read() == OLD_DOC
            rec_ok = bool(sup_side_rec) and os.path.exists(sup_side_rec)
            # 可见路径残留的任何 *.superseded 都不算发布产物（消费者只按
            # OUT / OUT.sha256 / OUT.failure.json 三个可见名解析终态）
            print(f"  receipt 含 attempt_id? {'是: ' + att.get('attempt_id', '') if has_att else '否——异常!'}")
            print(f"  旧成功 JSON 已隔离改名且内容保留? "
                  f"{'是 -> ' + sup_out_rec if old_quarantined else '否——异常!'}")
            print(f"  旧 sidecar 已隔离改名? {'是 -> ' + sup_side_rec if rec_ok else '否——异常!'}")
            print(f"  gate errors: {d['errors']}")
            ok = ok and (not missing) and has_att and old_quarantined and rec_ok
        else:
            ok = False
        print(f"[VERDICT] {'PASS：非零退出 + failure.json(attempt_id) 落盘 + 旧产物隔离改名 + 性能 JSON 不发布' if ok else 'FAIL'}")
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
