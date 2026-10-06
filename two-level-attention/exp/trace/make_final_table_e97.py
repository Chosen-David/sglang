# 终表合成（用户 2026-10-02 终表口径）：method组合｜(α,β,γ)｜精度（trace mass + e2e）｜速度
# 数据源（缺失臂自动跳过并标注 MISSING）：
#   trace mass : e64g_full_grid / e88_partition_verdict / e87_topsigma_merged / e64i_tight
#   e2e 精度   : e71_main_table / e72_screen_verdict / e87_e2e_screen / streamingllm+MoBA 打分
#   速度       : MACs（trace 口径 sel_tok/macs）+ kernel 时延（既有 e64c 数据，引用不重跑）
# 输出：results/final_table_e97.json + final_table_e97.md
import glob
import json
import os

import numpy as np

R = "/home/wangyuanshuo02/two-level-attention/exp/trace/results"
OUT_JSON = f"{R}/final_table_e97.json"
OUT_MD = f"{R}/final_table_e97.md"


def load(fn):
    p = f"{R}/{fn}"
    return json.load(open(p)) if os.path.exists(p) else None


def lb_avg(method, e71):
    """13 任务 LongBench AVG（e71_main_table 口径）"""
    if e71 is None or method not in e71:
        return None
    return e71[method].get("AVG")


def main():
    e71 = load("e71_main_table.json")
    e72 = load("e72_screen_verdict.json")
    e87_e2e = load("e87_e2e_screen.json")
    e87_tr = load("e87_topsigma_merged.json")
    e88 = load("e88_partition_verdict.json")
    e64g = load("e64g_full_grid.json")

    # MoBA / StreamingLLM e2e（hotpotqa+musique screen + 全量目录存在则用）
    def screen2(pred_dir_glob, tasks=("hotpotqa", "musique")):
        vals = {}
        import sys
        sys.path.insert(0, "/home/wangyuanshuo02/two-level-attention")
        for t in tasks:
            fs = sorted(glob.glob(pred_dir_glob.format(task=t)))
            if not fs:
                return None
            from benchmark.LongBench.eval import scorer
            preds, answers, allc = [], [], None
            for line in open(fs[-1]):
                d = json.loads(line)
                preds.append(d["pred"]); answers.append(d["answers"]); allc = d["all_classes"]
            vals[t] = round(scorer(t, preds, answers, allc), 2)
        vals["screen_avg"] = round(float(np.mean([vals[t] for t in tasks])), 2)
        return vals

    moba = screen2("/tmp/e89_moba/pred_moba/{task}-tli_*.jsonl")
    sllm = screen2(
        "/home/wangyuanshuo02/two-level-attention/exp/results_longbench/Qwen3-8B"
        "/pred_kvcf/streamingllm/{task}*-streamingllm-*.jsonl")

    # trace mass 汇总（e64g 16 样本网格取部署臂 + e87/e88 merged）
    def trace_mass(key):
        if e64g is None:
            return None
        vals = [e64g[s][key] for s in e64g if key in e64g[s]]
        return round(float(np.mean(vals)), 4) if vals else None

    # E90/E87c/M10/MoBA 全量（round3-f 收官数据）
    e90 = load("e90_subspace_e2e.json")
    e87c_m = load("e87c_mavg_tail.json")
    e87c_s = load("e87c_signear8_tail.json")
    e67g = load("e67_gate_benefit.json")
    moba_full = load("e89_moba_full.json")

    rows = []

    def add(method, cfg, mass, e2e, speed, note=""):
        rows.append({"method": method, "config": cfg, "trace_mass": mass,
                     "e2e": e2e, "speed": speed, "note": note})

    # ---- 主行：PSI 部署臂（论文主表口径 = E72 mavg β.375, LB 50.54）----
    add("PSI (minmax,avg) 分区", "α.125/β.375/γ.125",
        trace_mass("mavg_a0.125_b0.375"),
        {"longbench_avg": lb_avg("TLI_E72", e71),
         "screen": {"hotpotqa": 54.43, "musique": 34.76,
                    "calib_tail32_L1": {"hotpotqa": 54.83, "musique": 33.22}}},
        "选择链 1.46×@32K→5.09×@128K；MACs/token≈258；336B/token",
        "L1 上界全维 128 口径（L2 细筛两口径数学等价 tail32，E87c 配对校准："
        "L1-tail 臂 hq 54.83/mu 33.22——主表结论对 L1 维度不敏感）")
    add("PSI 分区 (β.25 对照)", "α.125/β.25/γ.125",
        trace_mass("mavg_a0.125_b0.25"),
        {"longbench_avg": lb_avg("TLI_B7", e71),
         "screen": e72.get("mavg(B7s)") if e72 else None},
        "同上")
    add("PSI 单池 (minmax)", "bp128/γ.125",
        trace_mass("mavg_a0.0_b0.0"),
        {"longbench_avg": lb_avg("TLI_C0", e71),
         "screen": None},
        "同上（无分区开销略低）")
    # ---- method 组合消融行（E72 screen + e64g mass）----
    for arm, key, note in [
        ("mavg (minmax,avg)", "mavg_a0.125_b0.375", "E72 冠军 β.375"),
        ("mminmax (minmax,minmax)", "mavg_a0.375_b0.375", None),
        ("aavg (avg,avg)", "aavg_a0.125_b0.25", None),
        ("cavg (cluster,avg)", "cavg_a0.125_b0.375", "ClusterKV-style 代表"),
    ]:
        short = arm.split(" ")[0]
        # E72 键名：mminmax/aavg/cavg 直接匹配；mavg 有 (B7s) 后缀歧义 → 取 E72 冠军臂
        e2e_v = None
        if e72:
            if short == "mavg":
                e2e_v = e72.get("mavg(B7s)") or e72.get("mavg")
            else:
                e2e_v = e72.get(short)
        add(arm, "E72 最优 α/β/γ", trace_mass(key), e2e_v,
            "—" if short == "cavg" else None, note)
    # ---- E87 top-σ 行 ----
    if e87_tr:
        s = e87_tr["summary"]
        tw = s.get("twolvl", {})
        add("两级 baseline (minmax,avg)", "α.125/β.25/γ1", tw.get("cov"),
            {"screen": {"hotpotqa": 54.23, "musique": 32.82}},
            f"macs={tw.get('macs')}, sel={tw.get('sel_tok')}", "trace 重放参照臂")
        for arm in ["near_sigma_8.0", "far_sigma_8.0", "mid_sigma_8.0",
                    "far_sigma_32.0", "mid_sigma_32.0"]:
            v = s.get(arm, {})
            e2e_key = arm.replace(".0", "")      # e2e 侧键名无 .0
            add(f"top-σ {arm}", f"σ={arm.split('_')[-1]}", v.get("cov"),
                {"screen": (e87_e2e or {}).get(e2e_key)},
                f"macs={v.get('macs')}, sel={v.get('sel_tok')}",
                "预算可变质量-预算旋钮")
    # ---- E88 分区收益行 ----
    if e88:
        for b, v in sorted(e88.items(), key=lambda x: int(x[0][1:])):
            add(f"mono vs 分区 @{b}", "mavg 同预算",
                v["delta"], None, None,
                f"mono={v['mono']} part={v['part']} 分区胜 {v['wins']}")
    # ---- baseline 行 ----
    for m, disp in [("FullKV", "FullKV"), ("Quest", "Quest"), ("TIA", "TIA"),
                    ("SnapKV", "SnapKV"), ("H2O", "H2O"), ("PyramidKV", "PyramidKV")]:
        add(disp, "官方配置", None, {"longbench_avg": lb_avg(m, e71)}, None)
    add("StreamingLLM", "budget=1024",
        None, {"longbench_avg": lb_avg("StreamingLLM", e71), "screen": sllm},
        None, "E89 全量 13 任务：静态稀疏崩塌证据")
    add("MoBA (统一harness)", "chunk gate top-16 块",
        None, {"longbench_avg": (moba_full or {}).get("AVG"), "screen": moba},
        None, "E89 复现 13 任务全量：QA/检索落后 PSI 3+/代码任务反超"
              "（repobench 68.13 vs 66.67）——无上界 chunk gate 掉档")
    # ---- E90 子空间 e2e 五臂（round3-a 收官）----
    if e90 and "arms" in e90:
        for arm, v in e90["arms"].items():
            add(f"E90 子空间 {arm}", "twolvl α.125/β.25/γ.125", None,
                {"screen": v}, None,
                "五臂排序（hq）：rope64 55.98 > full128 54.72 > tail32 54.23 > "
                "nope64 52.4 >> random32 46.01 >> highfreq 33.8；崩塌级双口径同向")
    # ---- E87c 维度校准双臂 ----
    if e87c_s and "arms" in e87c_s:
        v = e87c_s["arms"].get("signear8_tail (GPU0)")
        if v:
            add("top-σ near σ8 (L1-tail 校准臂)", "σ=8, L1 上界 tail32", None,
                {"screen": {"hotpotqa": v["hotpotqa"], "musique": v["musique"]}},
                None, "E87c：top-σ 判决不受 L1 维度混杂影响（hq 维度差 −0.04 噪声级）")
    if e87c_m and "arms" in e87c_m:
        v = e87c_m["arms"].get("mavg_tail (GPU1 重跑, α.125/β.375, tail32)")
        if v:
            add("mavg β.375 (L1-tail 校准臂)", "α.125/β.375, L1 上界 tail32", None,
                {"screen": {"hotpotqa": v["hotpotqa"], "musique": v["musique"]}},
                None, "E87c：L1 维度效应在 mavg 臂方向反转（tail +0.40）——"
                      "非可加常数、method×dimension 交互")
    # ---- M10 gate 收益行 ----
    if e67g:
        g = e67g.get("summary", e67g)
        tau005 = (g.get("arms", g)).get("tau0.05", g.get("tau0.05", {}))
        if tau005:
            add("D' gate (τ=0.05 保守臂)", "far 统计跳层", None, None,
                "跳层 14.3%（24/168 层）mass 损失 0.8%、选择链省等比 4.2/29.5ms@8K",
                "上界=离线轮廓 13/36=36.1%；τ0.1 跳 30.9% 但损失 2.3%")

    json.dump(rows, open(OUT_JSON, "w"), indent=1, ensure_ascii=False)
    md = ["| method | config | trace mass | e2e | speed | note |", "|---|---|---|---|---|---|"]
    for r in rows:
        e2e = r["e2e"]
        if isinstance(e2e, dict):
            if "hotpotqa" in e2e or "screen_avg" in e2e:
                e2e_s = f"hq {e2e.get('hotpotqa','?')} / mu {e2e.get('musique','?')}"
            elif "longbench_avg" in e2e and e2e["longbench_avg"] is not None:
                e2e_s = f"LB {e2e['longbench_avg']}"
                if e2e.get("screen"):
                    sc = e2e["screen"]
                    e2e_s += (f" / hq {sc.get('hotpotqa','?')} / mu {sc.get('musique','?')}"
                              if isinstance(sc, dict) else f" / {sc}")
            elif e2e.get("screen"):
                sc = e2e["screen"]
                e2e_s = (f"hq {sc.get('hotpotqa','?')} / mu {sc.get('musique','?')}"
                         if isinstance(sc, dict) else str(sc))
            else:
                e2e_s = "MISSING"
        else:
            e2e_s = str(e2e) if e2e is not None else "-"
        md.append(f"| {r['method']} | {r['config'] or '-'} | {r['trace_mass'] or '-'} | "
                   f"{e2e_s} | {r['speed'] or '-'} | {r['note'] or ''} |")
    open(OUT_MD, "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nsaved {OUT_JSON} / {OUT_MD}")


if __name__ == "__main__":
    main()
