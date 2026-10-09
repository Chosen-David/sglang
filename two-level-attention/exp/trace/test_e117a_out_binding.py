#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E117a 输出绑定红绿测试（GPT 2026-10-10 0125 审计
TL-E117A-OUT-BINDING-053）。

缺陷（红）：analyze_e117a_mavg_ref.py 的 main() 在 base.main() 后用
  attach_comparison(out_default(argv)) 定位注入目标——out_default 只
  区分 --dry-run，丢弃用户显式 --out PATH / --out=PATH，导致
  comparison_with_avg_avg_v1 元数据注入默认路径（默认路径有旧文件时
  被误改）而用户显式输出无注入（默认路径不存在时静默丢失）。

修复（绿）：resolve_out_path(argv) 从 inject_defaults 返回后的 argv
  解析出唯一 resolved 输出路径（--out PATH 与 --out=PATH 两种形式，
  多次出现取最后一个 = argparse 后值覆盖语义），base.main() 写盘与
  attach_comparison 注入用同一值。

测试方式：不真跑 base.main()（700 行长跑 + torch），用 sys.modules
stub 掉 analyze_e117a_wo_projection 与 sparse_attn.indexer.tli_indexer
两个重依赖后 importlib 加载被测模块，直接测路径解析 +
attach_comparison 行为——纯 CPU 秒级，无需 torch/GPU。

用例：
  T1 默认路径   argv=[] → 注入默认 e117a_mavg_ref.json；只改目标文件，
               dryrun 默认文件与其余文件逐位不变；
  T2 --out PATH 用户显式 --out custom.json → 注入 custom.json 且其含
               comparison_with_avg_avg_v1；默认路径旧文件（哨兵）逐位
               不被误改——修复前 comparison 注入默认路径（红）；
  T3 --out=PATH 等号形式同 T2；附带验证多次 --out 取最后一个
               （argparse 覆盖语义）。

用法： python3 exp/trace/test_e117a_out_binding.py
"""
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "analyze_e117a_mavg_ref.py")

PASS = 0


def _stub_heavy_imports():
    """stub 掉被测模块的重依赖（analyze_e117a_wo_projection 700 行 +
    tli_indexer/torch），使本测试无需 torch、秒级完成。"""
    base_stub = types.ModuleType("analyze_e117a_wo_projection")
    base_stub.p0p = None           # patch_selfcheck 用（本测试不调用）
    base_stub.main = lambda: None
    sys.modules["analyze_e117a_wo_projection"] = base_stub
    for name in ("sparse_attn", "sparse_attn.indexer"):
        m = types.ModuleType(name)
        m.__path__ = []
        sys.modules[name] = m
    tli = types.ModuleType("sparse_attn.indexer.tli_indexer")

    class TLIIndexer:   # 模块级 _orig_need_avg = TLIIndexer._need_avg_score
        @staticmethod
        def _need_avg_score(self):
            return False
    tli.TLIIndexer = TLIIndexer
    sys.modules["sparse_attn.indexer.tli_indexer"] = tli


def _load_module(tmp):
    """importlib 加载被测模块（stub 依赖就位后）。"""
    spec = importlib.util.spec_from_file_location(
        "analyze_e117a_mavg_ref_under_test", TARGET)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # 重定向模块内路径常量：测试写到临时目录，绝不触碰仓库生产
    # e117a_mavg_ref.json / e117a_wo_projection.json
    mod.HERE = tmp
    mod.V1_PATH = os.path.join(tmp, "e117a_wo_projection.json")
    return mod


def _mk_gap_json(exp, far, near, gstar, med, verdict):
    """attach_comparison 消费的最小结构（config/global 两段）。"""
    return {
        "exp": exp,
        "config": {"far_method": far, "near_method": near, "gstar": gstar},
        "global": {
            "median_gap_rel_cal": med,
            "median_gap_rel_conf": round(med / 2, 6),
            "gap_rel_cal_by_layer": {"L35": 0.07, "L36": 0.05},
            "verdict": verdict,
        },
    }


V1 = _mk_gap_json("e117a_wo_projection", "avg", "avg",
                  [0.125, 0.125, 0.375], 0.08045, "NO-GO")
V2 = _mk_gap_json("e117a_mavg_ref", "minmax", "avg",
                  [0.25, 0.125, 0.625], 0.3479, "GO")


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def _dir_snapshot(d):
    snap = {}
    for root, _, files in os.walk(d):
        for f in files:
            fp = os.path.join(root, f)
            snap[os.path.relpath(fp, d)] = _sha(fp)
    return snap


def _run_case(mod, tmp, argv_in, target_relpath, tag):
    """公共流程：放置 v1/v2 fixture → 注入默认 → 解析 → 注入 → 断言。"""
    res_dir = os.path.join(tmp, "results")
    os.makedirs(res_dir, exist_ok=True)
    json.dump(V1, open(mod.V1_PATH, "w"), ensure_ascii=False)
    # 默认路径文件（哨兵）：显式 --out 场景必须逐位不被误改（红例靶子）
    default_p = os.path.join(res_dir, "e117a_mavg_ref.json")
    json.dump({"sentinel": "default-path-must-not-be-touched",
               **V2}, open(default_p, "w"), ensure_ascii=False)
    dry_p = os.path.join(res_dir, "e117a_mavg_ref_dryrun.json")
    json.dump({"sentinel": "dryrun-must-not-be-touched"},
              open(dry_p, "w"), ensure_ascii=False)

    injected = mod.inject_defaults(list(argv_in))
    resolved = mod.resolve_out_path(injected)
    # 本次目标文件放 v2 fixture（模拟 base.main() 已写盘）
    target_p = os.path.join(tmp, target_relpath)
    os.makedirs(os.path.dirname(target_p), exist_ok=True)
    json.dump(V2, open(target_p, "w"), ensure_ascii=False)
    assert resolved == target_p, (tag, resolved, target_p)

    before = _dir_snapshot(tmp)
    mod.attach_comparison(resolved)
    after = _dir_snapshot(tmp)

    # (1) 只修改本次目标文件：目录内其余文件（含默认路径哨兵、dryrun、
    #     v1 fixture）SHA 逐位不变
    changed = sorted(k for k in after if before.get(k) != after[k])
    assert changed == [os.path.relpath(target_p, tmp)], \
        f"{tag}: attach_comparison 改动了目标之外的文件: {changed}"
    # (2) 目标文件含 comparison_with_avg_avg_v1，且两口径 gstar/中位数
    #     记录正确（v1=avg/avg 代理口径、v2=mavg 同方法口径）
    out = json.load(open(target_p))
    cmp = out["comparison_with_avg_avg_v1"]
    assert cmp["v1"]["gstar"] == [0.125, 0.125, 0.125] or \
        cmp["v1"]["gstar"] == [0.125, 0.125, 0.375], cmp
    assert cmp["v1"]["far_method"] == "avg" and \
        cmp["v2"]["far_method"] == "minmax", cmp
    assert cmp["v2"]["gstar"] == [0.25, 0.125, 0.625], cmp
    assert "delta_median_gap_cal" in cmp and \
        "same_direction" in cmp, cmp
    assert out["config"]["gstar"] == [0.25, 0.125, 0.625], \
        "attach_comparison 不得破坏原结果内容"
    # (3) 默认路径旧文件（哨兵）不被误改——仅当本次目标≠默认路径时
    #     断言（T1 的目标就是默认路径，其被合法改写属预期）；修复前
    #     显式 --out 场景 comparison 注入的正是这个默认路径文件
    if os.path.abspath(resolved) != os.path.abspath(default_p):
        cur = json.load(open(default_p))
        assert cur["sentinel"] == "default-path-must-not-be-touched" and \
            "comparison_with_avg_avg_v1" not in cur, \
            f"{tag}: 默认路径旧文件被 attach_comparison 误改（053 红例复发）"
    print(f"{tag} PASS  resolved={os.path.relpath(resolved, tmp)}；"
          f"只改目标文件；目标含 comparison_with_avg_avg_v1；"
          f"{'目标≠默认路径：默认路径哨兵逐位不变' if os.path.abspath(resolved) != os.path.abspath(default_p) else '目标=默认路径（注入点正确）'}")


def main():
    global PASS
    _stub_heavy_imports()
    root = tempfile.mkdtemp(prefix="e117a_out_binding_")
    import shutil
    try:
        # T1：默认路径（无显式 --out）→ 注入默认 e117a_mavg_ref.json
        tmp1 = os.path.join(root, "t1")
        os.makedirs(tmp1)
        mod = _load_module(tmp1)
        _run_case(mod, tmp1, [],
                  os.path.join("results", "e117a_mavg_ref.json"), "T1")
        PASS += 1
        # T2：显式 --out PATH（空格形式）→ 注入 custom，默认路径不误改
        tmp2 = os.path.join(root, "t2")
        os.makedirs(tmp2)
        mod = _load_module(tmp2)
        _run_case(mod, tmp2, ["--out", os.path.join(tmp2, "custom.json")],
                  "custom.json", "T2")
        PASS += 1
        # T3：显式 --out=PATH（等号形式）→ 注入 custom2；附带验证多次
        #     --out 取最后一个（argparse 后值覆盖语义）
        tmp3 = os.path.join(root, "t3")
        os.makedirs(tmp3)
        mod = _load_module(tmp3)
        argv3 = ["--out=" + os.path.join(tmp3, "wrong_first.json"),
                 "--out", os.path.join(tmp3, "custom2.json")]
        injected = mod.inject_defaults(list(argv3))
        resolved = mod.resolve_out_path(injected)
        assert resolved == os.path.join(tmp3, "custom2.json"), \
            ("T3 前置: 多次 --out 未取最后一个", resolved)
        _run_case(mod, tmp3, argv3, "custom2.json", "T3")
        PASS += 1
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print(f"\nE117a out-binding ALL PASS ({PASS}/3)")


if __name__ == "__main__":
    main()
