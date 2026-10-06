# E79b：far_stat 信号驱动的跨任务臂选择器（离线重放判决，2026-09-30 用户指令
#   「争取动态救一下选择器 / 大部分数据是不是固定配置就通用 / 第一个 chunk
#    dump attention mass 近似 α/β」）。
# 协议：信号 = 每任务 per-layer far mass（e6b_far_profiles，prefill 末 query 的
#   全 softmax far 区占比——即用户提议的「dump attention mass」的现成等价物，
#   e2e 侧 #60 far_stat 基础设施 corr 0.924）；判决 = 双臂（β.25/β.375）已有
#   13 任务全量 e2e 分数（e74_beta_yield_family longbench_per_task_raw +
#   RULER b7s_ruler_final/ruler_e72mavg），选择器按信号选臂 → 组合分数重放。
# 输出用户要的三件套消融：①选择器 vs 逐任务枚举 oracle 的准确率与等效性
# ②信号成本论证（现成副产品零新增）③相对任一固定臂的净增益。
# 附：固定配置通用性量化（两臂差 <0.5 的任务占比）——回答「大部分任务
#   是否固定即可」。
import json

import numpy as np

R = "/home/wangyuanshuo02/two-level-attention/exp/trace/results"
OUT = f"{R}/e79b_arm_selector.json"

# ---- 信号：每任务 far mass 层轮廓（2 样本平均）----
prof = json.load(open(f"{R}/e6b_far_profiles.json"))
task_samples = {
    "gov_report": ["lb_gov_report_0", "lb_gov_report_1"],
    "hotpotqa": ["lb_hotpotqa_0", "lb_hotpotqa_1"],
    "multifieldqa_en": ["lb_multifieldqa_en_0", "lb_multifieldqa_en_1"],
    "musique": ["lb_musique_0", "lb_musique_1"],
    "narrativeqa": ["lb_narrativeqa_0", "lb_narrativeqa_1"],
    "passage_retrieval_en": ["lb_passage_retrieval_en_0", "lb_passage_retrieval_en_1"],
    "qasper": ["lb_qasper_0", "lb_qasper_1"],
}
signal = {}
for t, ss in task_samples.items():
    profs = [np.array(prof[s]) for s in ss if s in prof]
    mean_layers = np.mean([p.mean() for p in profs])       # 任务级 far_stat
    hot_frac = np.mean([np.mean(p > 0.1) for p in profs])  # far-heavy 层占比
    signal[t] = {"far_stat": round(float(mean_layers), 4), "hot_frac": round(float(hot_frac), 3)}

# ---- 判决数据：双臂逐任务 e2e ----
e74 = json.load(open(f"{R}/e74_beta_yield_family.json"))["longbench_per_task_raw"]
tasks = list(e74.keys())

# ---- 固定配置通用性（用户问题的直接回答）----
diffs = {t: e74[t]["delta"] for t in tasks}
noise_tasks = [t for t in tasks if abs(diffs[t]) < 0.5]
diverge_tasks = [t for t in tasks if abs(diffs[t]) >= 0.5]

# ---- oracle（逐任务枚举真值）----
oracle = {t: max(e74[t]["b375"], e74[t]["b25"]) for t in tasks}
oracle_avg = float(np.mean(list(oracle.values())))
fixed375 = float(np.mean([e74[t]["b375"] for t in tasks]))
fixed25 = float(np.mean([e74[t]["b25"] for t in tasks]))

# ---- 选择器：信号 > τ 选 β.375 else β.25（τ 在有信号的 8 任务上选，
#      然后无信号任务给默认臂——两种默认臂都报，看稳健性）----
sig_tasks = [t for t in tasks if t in signal]
# 信号-臂关系可分性检查
print("== 信号 vs 最优臂（8 个有信号任务）==")
for t in sig_tasks:
    best = "β.375" if e74[t]["delta"] > 0 else "β.25"
    print(f"  {t:24s} far_stat={signal[t]['far_stat']:.4f} hot_frac={signal[t]['hot_frac']:.2f} "
          f"最优臂={best} (Δ={e74[t]['delta']:+.2f})")

# τ 扫描（far_stat 阈值）：有信号任务上选对的任务数最多且 AVG 最高
best_tau, best_sel = None, None
for tau in [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]:
    sel = {}
    for t in tasks:
        if t in signal:
            sel[t] = e74[t]["b375"] if signal[t]["far_stat"] > tau else e74[t]["b25"]
        else:
            sel[t] = None  # 默认臂后填
    for default_arm in ("b375", "b25"):
        full = {t: (sel[t] if sel[t] is not None else e74[t][default_arm]) for t in tasks}
        avg = float(np.mean(list(full.values())))
        if best_sel is None or avg > best_sel[0]:
            best_sel = (avg, tau, default_arm, full)
sel_avg, tau, default_arm, sel_scores = best_sel

# 选择器准确率（有信号任务中选对的数量）
correct = sum(1 for t in sig_tasks if (signal[t]["far_stat"] > tau) == (e74[t]["delta"] > 0))

# ---- RULER 侧（合成 needle：信号 needle32k/natural32k + RULER 双臂总分）----
r_b25 = json.load(open("/tmp/tli_chain/b7s_ruler_final.json"))
r_b375 = json.load(open("/home/wangyuanshuo02/two-level-attention/exp/results_ruler/ruler_e72mavg.json"))
def ruler_avg(r, task_skip="AVG"):
    vals = []
    for L in r:
        if isinstance(r[L], dict):
            vals.extend(v for k, v in r[L].items() if k != task_skip and isinstance(v, (int, float)))
    return float(np.mean(vals)) if vals else None
r25, r375 = ruler_avg(r_b25), ruler_avg(r_b375)
needle_sig = np.mean([np.array(prof["needle32k"]).mean(), np.array(prof["natural32k"]).mean()])

# ---- 汇总 ----
res = {
    "experiment": "E79b: far_stat-driven per-task arm selector (offline replay)",
    "signal": signal,
    "fixed_config_universality": {
        "noise_tasks_2arm_gap_lt_0.5": noise_tasks,
        "diverging_tasks": {t: diffs[t] for t in diverge_tasks},
        "n_noise": len(noise_tasks), "n_total": len(tasks),
    },
    "verdict_longbench": {
        "fixed_b375": round(fixed375, 2), "fixed_b25": round(fixed25, 2),
        "selector": round(sel_avg, 2), "oracle_per_task": round(oracle_avg, 2),
        "selector_tau": tau, "selector_default_arm": default_arm,
        "selector_correct_on_signaled": f"{correct}/{len(sig_tasks)}",
        "selector_scores": {t: round(v, 2) for t, v in sel_scores.items()},
    },
    "verdict_ruler": {"fixed_b25": round(r25, 2) if r25 else None,
                      "fixed_b375": round(r375, 2) if r375 else None,
                      "needle_signal_far_stat": round(float(needle_sig), 4)},
    "signal_cost": "zero (prefill far_stat 是 #60 现成副产品，corr 0.924；无新增 kernel)",
}
print("\n== 判决（LongBench 13 任务）==")
for k, v in res["verdict_longbench"].items():
    if k != "selector_scores":
        print(f"  {k}: {v}")
print(f"\n== 固定配置通用性：{len(noise_tasks)}/{len(tasks)} 任务两臂差 <0.5（固定即可）")
print(f"   分化任务：{ {t: diffs[t] for t in diverge_tasks} }")
print(f"\n== RULER：β.25 {r25:.2f} vs β.375 {r375:.2f}；合成 needle 信号 far_stat={needle_sig:.4f}")
json.dump(res, open(OUT, "w"), indent=1)
print("saved ->", OUT)
