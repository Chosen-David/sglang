#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E113 microbench 054/055 修复红绿单测（CPU-only，stub 生产 main() 全流程）。

对应 GPT 审计 TL-E113-FAILCLOSED-054 / TL-E113-RAW-TIMING-055（2026-10-10_0227）
的修复验收。方法与 GPT 审计复现一致：stub 掉 GPU/张量/实现加载，直接调用
生产 main() 走原始 errs 分支 / 完整成功发布循环——**不复制生产逻辑、不绕过
生产状态机、不预清场**（预埋旧产物制造真实重跑状态转换）：

  T5（054）旧成功 → 新失败：预埋假旧成功 JSON + sidecar，stub tri 篡改
     assignment 进原始 errs 分支，断言：旧成功已被隔离改名
     （*.attempt-<sha8>.superseded，内容逐字节保留）、可见路径无残留成功
     JSON/sidecar、failure.json 属于新 attempt（attempt_id 非空 + 隔离记录
     与实际改名一致）+ 050 契约字段不回归；
  T6（054）旧失败 → 新成功：预埋 stale failure.json，走完整成功发布，
     断言：stale failure 已从可见路径清理（隔离改名保留历史）、成功发布
     完整（attempt_id 匹配 + 内容 SHA 自洽 + 10 case 全齐）；
  T7（055）读回复算：从发布 JSON 的持久化原始整数纳秒样本复算
     median / us_per_token / speedup，逐位一致（µs/token 用 ns→µs 正确换算
     med_ns/(1000*T)——2026-10-09_E113_Nanosecond_Unit_Regression 量纲修复后，
     复算式与生产同物理单位）；样本全为整数、乱序注入的
     样本按执行顺序原样落盘（未被排序）；sidecar 与内容 SHA 自洽；
  T9（055 量纲）us_per_token 独立常量守卫：发布字段的预期值全部硬编码
     （物理正确值，不用生产表达式反算）——python 臂 123456789ns/T=1024 →
     120.56、triton 臂 3800000ns/T=1024 → 3.71（GPT advice 原文示例值）；
     生产若回退旧式 1e6*med/T 会得到 10^11 量级荒谬值，硬编码断言必红。
  T8（055/054 原语）supersede_file：隔离改名 / 不存在返回 None / 同内容
     重复隔离加计数后缀（历史永不覆盖删除）。

python 与 python -O 双跑安全：全部断言用显式 check（不依赖 assert——
assert 在 -O 下会被删除导致测试静默通过）。

用法：python3 test_e113_state_machine_054_055.py   （exp/trace/ 目录下，纯 CPU）
"""
import glob
import importlib.util
import json
import os
import sys
import tempfile
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "e113_microbench.py")

RESULTS = []

# 055 注入样本：python rep=1 单样本；triton rep=3 **故意乱序**——验证执行顺序
# 原样持久化（生产代码若再 sort()，此乱序会暴露为 FAIL）
PY_NS = [123_456_789]
TRI_NS = [4_500_000, 3_100_000, 3_800_000]


def report(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))


# ---------------------------------------------------------------- 生产模块加载
_load_counter = [0]


def load_prod_module():
    """按文件加载生产 e113_microbench（每次唯一模块名，避免 sys.modules 串台）。"""
    _load_counter[0] += 1
    spec = importlib.util.spec_from_file_location(f"e113_bench_mod_sm{_load_counter[0]}", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------- 生产终态桩
def fake_state(tamper):
    """最小贪心终态（CPU 张量，与 check_pair 解包序一致：
    (sums, cnt, sq, k_live, assign)）。tamper=True 篡改 assignment 一位→必 mismatch。

    每次调用以固定 seed 重新生成并 clone——ref/tri 两臂各自拿到内容一致的
    独立副本（成功态过 check_pair，失败态在 assign 上分叉）。"""
    g = torch.Generator().manual_seed(7)
    sums = torch.randn(2, 4, 8, generator=g)
    cnt = torch.rand(2, 4, generator=g) * 4 + 1
    sq = (sums * sums).sum(-1)
    kl = torch.tensor([2, 3], dtype=torch.long)
    assign = torch.stack([torch.randint(0, 2, (16,), generator=g),
                          torch.randint(0, 3, (16,), generator=g)]).long()
    if tamper:
        assign = assign.clone()
        assign.view(-1)[0] = 99999   # 不可能的簇索引，必 mismatch
    return (sums.clone(), cnt.clone(), sq.clone(), kl.clone(), assign.clone())


def run_prod_main(out, fail_mode):
    """stub GPU/张量/实现加载后调用生产 main()。

    返回 (exit_code, 生产模块)：exit_code=None 表示 main 正常返回（成功发布），
    整数表示 SystemExit code（fail-closed=1 / 状态机终检=3）。
    生产状态机（隔离/清理/终检）全程真实执行——本函数只替换环境桩。"""
    m = load_prod_module()

    class _FakeTLI:
        @staticmethod
        def _greedy_cluster_pass_python(*a, **k):
            return fake_state(False)

    class _FakeIDX:
        TLIIndexer = _FakeTLI

    fake_tri = types.SimpleNamespace(
        greedy_build_triton=lambda *a, **k: fake_state(fail_mode))
    fake_x = torch.randn(4, 2, 8)

    def fake_bench(fn, warmup=1, rep=3):
        # 055 样本桩：按生产调用形状（python rep=1 / triton rep=3）返回乱序整数纳秒
        return list(TRI_NS) if rep == 3 else list(PY_NS)

    saved = []

    def patch(obj, name, val):
        saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, val)

    real_zeros = torch.zeros

    def cpu_zeros(*a, **k):
        # 生产 run_ref 里 torch.zeros(..., device="cuda") 在 CPU 桩环境映射到 cpu
        if k.get("device") == "cuda":
            k = dict(k)
            k["device"] = "cpu"
        return real_zeros(*a, **k)

    try:
        # ---- 环境桩（GPU 侧）----
        patch(torch.cuda, "is_available", lambda: True)
        patch(torch.cuda, "synchronize", lambda *a, **k: None)
        patch(torch.cuda, "empty_cache", lambda: None)
        patch(torch.cuda, "get_device_name", lambda *a, **k: "stub-cpu-gpu")
        patch(torch, "zeros", cpu_zeros)
        # ---- 实现加载桩（生产 build_identity/check_pair/发布协议/状态机真实执行）----
        m.load_sparse_attn = lambda *a, **k: _FakeIDX()
        m.load_triton_kernel = lambda *a, **k: fake_tri
        m.gen_structured = lambda *a, **k: fake_x.clone()
        m.gen_singleton = lambda *a, **k: fake_x.clone()
        m.bench = fake_bench

        sys.argv = ["e113_microbench.py", "--out", out]
        try:
            m.main()
            return None, m
        except SystemExit as e:
            return e.code, m
    finally:
        for obj, name, val in reversed(saved):
            setattr(obj, name, val)


# ================================================================ T5 旧成功 → 新失败
def t5_old_success_new_failure():
    name = "T5（054）旧成功 → 新失败：旧产物隔离改名 + failure 属新 attempt"
    try:
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "result.json")
            # 预埋假旧成功产物（不预清场——这正是 054 审计指出的最危险转换）
            old_doc = json.dumps({"status": "old-success", "cases": []},
                                 indent=1).encode("utf-8")
            with open(out, "wb") as f:
                f.write(old_doc)
            with open(out + ".sha256", "wb") as f:
                f.write(b"old-sidecar\n")

            code, m = run_prod_main(out, fail_mode=True)

            ok = code == 1
            # 可见路径无残留成功终态（消费者不可能再把旧成功当本次结果）
            ok = ok and not os.path.exists(out)
            ok = ok and not os.path.exists(out + ".sha256")
            # 旧产物被生产状态机隔离改名，内容逐字节保留（历史不删除）
            sup_out = glob.glob(out + ".attempt-*.superseded")
            sup_side = glob.glob(out + ".sha256.attempt-*.superseded")
            quarantined = bool(sup_out) and open(sup_out[0], "rb").read() == old_doc
            ok = ok and quarantined and len(sup_side) == 1
            # failure.json 落盘且属于新 attempt
            fail_path = out + ".failure.json"
            ok = ok and os.path.exists(fail_path)
            att_id, sup_rec = "", {}
            if os.path.exists(fail_path):
                with open(fail_path, encoding="utf-8") as f:
                    d = json.load(f)
                att = d.get("attempt", {})
                att_id = att.get("attempt_id", "")
                sup_rec = {r.get("visible_path"): r.get("superseded_as")
                           for r in att.get("superseded_files", [])}
                ok = ok and bool(att_id)
                # 隔离记录与实际改名逐一致（receipt 可审计）
                ok = ok and sup_rec.get(out) == (sup_out[0] if sup_out else None)
                ok = ok and sup_rec.get(out + ".sha256") == (sup_side[0] if sup_side else None)
                # 050 契约字段不回归
                ok = ok and all(k in d for k in
                                ("status", "gate", "identity", "failed_case", "errors"))
            report(name, ok, f"exit={code} attempt_id={att_id or '(缺失)'} "
                             f"隔离 {len(sup_rec)} 件，旧成功内容保留={quarantined}")
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ T6 旧失败 → 新成功
def t6_old_failure_new_success():
    name = "T6（054）旧失败 → 新成功：stale failure 清理 + 成功发布完整"
    try:
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "result.json")
            # 预埋 stale failure（不预清场）
            stale = json.dumps({"status": "old-failure"}, indent=1).encode("utf-8")
            with open(out + ".failure.json", "wb") as f:
                f.write(stale)

            code, m = run_prod_main(out, fail_mode=False)

            ok = code is None   # 成功发布正常返回
            ok = ok and os.path.exists(out)
            ok = ok and os.path.exists(out + ".sha256")
            ok = ok and m.verify_output_sha256(out)
            stale_gone = not os.path.exists(out + ".failure.json")
            ok = ok and stale_gone
            att_id, sup_files, n_cases = "", [], -1
            if os.path.exists(out):
                with open(out, encoding="utf-8") as f:
                    d = json.load(f)
                att = d["meta"]["attempt"]
                att_id = att["attempt_id"]
                sup_files = att["superseded_files"]
                n_cases = len(d["cases"])
                ok = ok and bool(att_id)
                # stale failure 被隔离改名保留（历史不删除）且记录在 manifest
                sup_rec = {r["visible_path"]: r["superseded_as"] for r in sup_files}
                stale_as = sup_rec.get(out + ".failure.json")
                ok = ok and stale_as and os.path.exists(stale_as) \
                    and open(stale_as, "rb").read() == stale
                # 成功终态唯一：可见路径只有本 attempt 的成功 JSON
                ok = ok and len(sup_files) == 1
                ok = ok and n_cases == 10
            report(name, ok, f"exit={code} attempt_id={att_id or '(缺失)'} "
                             f"stale failure 可见路径已清={stale_gone} cases={n_cases}")
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ T7 055 读回复算
def t7_readback_recompute():
    name = "T7（055）读回 JSON 复算 median/us_per_token/speedup 逐位一致"
    try:
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "result.json")
            code, m = run_prod_main(out, fail_mode=False)
            ok = code is None
            with open(out, encoding="utf-8") as f:
                d = json.load(f)
            n_bad = 0
            n_cases = len(d["cases"])
            for rec in d["cases"]:
                pys = rec["wall_ns_python_samples"]
                tris = rec["wall_ns_triton_samples"]
                # 样本全为整数纳秒（无舍入）
                if not all(isinstance(s, int) for s in pys + tris):
                    n_bad += 1
                # 执行顺序保留：乱序注入的桩样本原样落盘（未被 sort）
                if pys != PY_NS or tris != TRI_NS:
                    n_bad += 1
                # median 从已持久化原始值派生
                med_py = m.median_int(pys)
                med_tri = m.median_int(tris)
                if med_py != rec["wall_ns_python_median"] or \
                   med_tri != rec["wall_ns_triton_median"]:
                    n_bad += 1
                # us_per_token / speedup 复算逐位一致（µs/token = med_ns/(1000*T)，
                # ns→µs 正确换算；speedup 无量纲不受量纲修复影响）
                if round(med_py / (1000 * rec["T"]), 2) != rec["us_per_token_python"] or \
                   round(med_tri / (1000 * rec["T"]), 2) != rec["us_per_token_triton"]:
                    n_bad += 1
                if round(med_py / med_tri, 1) != rec["speedup"]:
                    n_bad += 1
            ok = ok and n_bad == 0 and n_cases == 10
            # sidecar + 内容 SHA 自洽
            ok = ok and os.path.exists(out + ".sha256") and m.verify_output_sha256(out)
            # manifest 表述：timing 含整数纳秒口径 + 独立重复样本（不再声明 paired 样本）
            timing = d["meta"]["timing"]
            ok = ok and "perf_counter_ns" in timing and "独立重复样本" in timing
            report(name, ok, f"10 case 复算不一致项 {n_bad}，样本=整数纳秒且乱序保留，"
                             f"timing 口径表述正确")
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ T8 隔离原语
def t8_supersede_primitives():
    name = "T8（054 原语）supersede_file：隔离改名 / None / 计数后缀不覆盖"
    try:
        m = load_prod_module()
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "a.json")
            with open(p, "wb") as f:
                f.write(b"hello")
            r1 = m.supersede_file(p)
            ok = (r1 is not None
                  and os.path.basename(r1).startswith("a.json.attempt-")
                  and r1.endswith(".superseded"))
            ok = ok and not os.path.exists(p) and open(r1, "rb").read() == b"hello"
            # 不存在 → None（幂等）
            ok = ok and m.supersede_file(p) is None
            # 同名同内容二次隔离 → 计数后缀防覆盖，历史两份都在
            with open(p, "wb") as f:
                f.write(b"hello")
            r2 = m.supersede_file(p)
            ok = ok and r2 is not None and r2 != r1
            ok = ok and open(r2, "rb").read() == b"hello" and os.path.exists(r1)
            # quarantine_prior_artifacts：三后缀全隔离 + 记录与改名一致
            q_out = os.path.join(td, "q.json")
            for suf in ("", ".sha256", ".failure.json"):
                with open(q_out + suf, "wb") as f:
                    f.write(("old" + suf).encode())
            recs = m.quarantine_prior_artifacts(q_out)
            ok = ok and len(recs) == 3
            ok = ok and all(os.path.exists(r["superseded_as"]) and
                            not os.path.exists(r["visible_path"]) for r in recs)
            report(name, ok, f"r1={os.path.basename(r1) if r1 else None} "
                             f"r2={os.path.basename(r2) if r2 else None} 三后缀隔离 {len(recs)} 件")
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ T9 us_per_token 量纲守卫
def t9_us_token_unit_guard():
    """（2026-10-09_E113_Nanosecond_Unit_Regression）独立常量守卫：发布字段的
    预期值全部硬编码物理正确值，**不用生产表达式反算**——生产若回退旧式
    秒→µs 换算（1e6*med_ns/T）会得到 10^11 量级荒谬值，此处必红。"""
    name = "T9（055 量纲）us_per_token 独立常量守卫（python/triton 两臂）"
    try:
        # 纯算术常量例（GPT advice 原文示例，预期值硬编码）：
        # 1e9 ns / 1000 token = 1000.00 µs/token；5e8 ns / 1000 token = 500.00；
        # 3.8e6 ns / 1024 token = 3.71；123456789 ns / 1024 token = 120.56
        const_bad = 0
        for med_ns, tok, want in ((1_000_000_000, 1000, 1000.00),
                                  (500_000_000, 1000, 500.00),
                                  (3_800_000, 1024, 3.71),
                                  (123_456_789, 1024, 120.56)):
            # 正确换算 ns→µs：med_ns / (1000 * tokens)
            if round(med_ns / (1000 * tok), 2) != want:
                const_bad += 1
        # 生产发布字段：stub main 成功发布后按硬编码预期核验（两臂都覆盖）
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "result.json")
            code, m = run_prod_main(out, fail_mode=False)
            ok = code is None and const_bad == 0
            if os.path.exists(out):
                with open(out, encoding="utf-8") as f:
                    d = json.load(f)
                field_bad = 0
                for rec in d["cases"]:
                    # python 臂：桩样本单值 123456789 ns；triton 臂：桩样本中位 3800000 ns
                    if rec["us_per_token_python"] != round(123_456_789 / (1000 * rec["T"]), 2):
                        field_bad += 1
                    if rec["us_per_token_triton"] != round(3_800_000 / (1000 * rec["T"]), 2):
                        field_bad += 1
                    # T=1024 case 的硬编码字面预期（GPT advice 示例值 3.71）
                    if rec["T"] == 1024:
                        if rec["us_per_token_triton"] != 3.71 or \
                           rec["us_per_token_python"] != 120.56:
                            field_bad += 1
                ok = ok and field_bad == 0
                # 无量纲 speedup 与原始 ns 字段不受量纲修复影响（不回归）
                for rec in d["cases"]:
                    if rec["speedup"] != round(123_456_789 / 3_800_000, 1):
                        ok = False
                sample = d["cases"][0]
                report(name, ok,
                       f"常量例 4/4 {'全过' if const_bad == 0 else '红 ' + str(const_bad)}，"
                       f"发布字段硬编码预期核验 {'全过' if field_bad == 0 else '红 ' + str(field_bad)}，"
                       f"10 case speedup/ns 原始字段无回归；"
                       f"T=1024 实测 python={sample['us_per_token_python']} "
                       f"triton={sample['us_per_token_triton']} µs/token")
            else:
                report(name, False, "发布 JSON 不存在")
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


if __name__ == "__main__":
    t5_old_success_new_failure()
    t6_old_failure_new_success()
    t7_readback_recompute()
    t8_supersede_primitives()
    t9_us_token_unit_guard()
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print("\n" + "=" * 60)
    print(f"总计 {len(RESULTS)} 项，通过 {len(RESULTS) - n_fail}，失败 {n_fail}")
    sys.exit(1 if n_fail else 0)
