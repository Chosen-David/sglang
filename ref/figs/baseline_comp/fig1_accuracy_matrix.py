#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig1_accuracy_matrix.py
论文对比数据可视化（一）：主表精度总览

各方法 × 5 个 benchmark 的 AVG 分组条形图（Qwen3-8B）。
数据源（正式收口结果文件）：
  LBv1 16 任务  : TLI=e116b_full_lbv1.json(mavg)；FullKV=49.39、Quest=41.35、MInference=47.71、
                  SparQ=48.66、FASA=33.31 均 vendor 协议（exp/results_longbench_vendor/Qwen3-8B）
  LBv2 503     : TLI=e109_full_lbv2_v2.json(mavg 32.01)；vendor 协议（quest 30.22/minf 31.21/
                  sparq 33.40/fasa 11.93/fullkv 32.01）
  RULER 32K    : score_ruler_formal.py 同协议（TLI mavg E116b=59.65/fullkv 59.38/sparq 60.45/
                  minf 49.94/quest 42.85/fasa 24.00[6/11]）
  RULER 64K    : TLI mavg E119=49.42/fullkv 48.54/minf 50.29/quest 26.25；sparq 64K 在飞未出数
  RULER 128K   : TLI mavg E119=47.49/fullkv 46.43/minf 49.53/quest 19.33；sparq 128K 未派

FASA RULER 32K 为 6/11 任务平均；SparQ 64K 在飞、128K 未派（图上留空 + 脚注）。
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/wangyuanshuo02/.claude/skills/publication-figures/scripts")
from style_config import apply_style, get_colors  # noqa: E402
from export_figure import export_figure            # noqa: E402

palette = apply_style("okabe_ito")
# 系统无 Arial，回退 DejaVu Sans（宽度接近 Arial 的 sans-serif）
plt.rcParams["font.family"] = "DejaVu Sans"
COLORS = get_colors("okabe_ito", n=6)

# ---------------------------------------------------------------------------
# 数据（AVG；NaN = 未测 / 在飞）
# ---------------------------------------------------------------------------
METHODS = ["TLI (ours)", "FullKV", "SparQ", "MInference", "Quest", "FASA"]
BENCH = ["LBv1\n(16 tasks)", "LBv2\n(503)", "RULER\n32K", "RULER\n64K", "RULER\n128K"]

data = np.array([
    [48.90, 32.01, 59.65, 49.42, 47.49],   # TLI mavg
    [49.39, 32.01, 59.38, 48.54, 46.43],   # FullKV 锚点
    [48.66, 33.40, 60.45,   np.nan, np.nan],  # SparQ
    [47.71, 31.21, 49.94, 50.29, 49.53],   # MInference
    [41.35, 30.22, 42.85, 26.25, 19.33],   # Quest
    [33.31, 11.93, 24.00,   np.nan, np.nan],  # FASA（RULER32=6/11）
], dtype=float)

# ---------------------------------------------------------------------------
# 图
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(16.5, 7.5), layout="constrained")

x = np.arange(len(BENCH))
n_methods = len(METHODS)
bar_w = 0.13
offsets = np.arange(n_methods) - (n_methods - 1) / 2.0

ax.set_xlim(-0.6, len(BENCH) - 0.4)
ax.set_ylim(0, 70)
ax.set_xticks(x)
ax.set_xticklabels(BENCH, fontsize=20)
ax.set_ylabel("Average accuracy (%)", fontsize=22)
ax.tick_params(axis="y", labelsize=20)
ax.tick_params(axis="both", direction="out", length=7, width=2)

for i, m in enumerate(METHODS):
    vals = data[i]
    color = COLORS[i]
    for j in range(len(BENCH)):
        v = vals[j]
        if np.isnan(v):
            continue
        ax.bar(x[j] + offsets[i] * bar_w, v, width=bar_w, color=color,
               edgecolor="black", linewidth=1.2, zorder=3)

# 数值标注（每条）
for i in range(n_methods):
    for j in range(len(BENCH)):
        v = data[i, j]
        if np.isnan(v):
            continue
        ax.text(x[j] + offsets[i] * bar_w, v + 0.8, f"{v:.1f}",
                ha="center", va="bottom", fontsize=9.5, color="black", zorder=5)

# 缺失标注
for j in [3, 4]:
    if np.isnan(data[2, j]):
        ax.text(x[j] + offsets[2] * bar_w, 2, "n/a", ha="center", va="bottom",
                fontsize=13, color="gray", style="italic")
    if np.isnan(data[5, j]):
        ax.text(x[j] + offsets[5] * bar_w, 2, "n/a", ha="center", va="bottom",
                fontsize=13, color="gray", style="italic")

# FASA RULER32 标注 6/11（条顶上方，避免与数值重叠）
ax.text(x[2] + offsets[5] * bar_w, 24.0 + 3.0, "6/11", ha="center", va="bottom",
        fontsize=10, color="#777777", style="italic")

# 图例（图外右侧，不遮挡数据）
handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[i], edgecolor="black",
                         linewidth=1.2) for i in range(n_methods)]
leg = ax.legend(handles, METHODS, loc="center left", fontsize=16,
                ncol=1, frameon=True, fancybox=False, shadow=False,
                bbox_to_anchor=(1.0, 0.5))
leg.get_frame().set_linewidth(2.5)
leg.get_frame().set_edgecolor("black")
leg.get_frame().set_facecolor("white")

# 全部 spine 可见
for s in ax.spines.values():
    s.set_linewidth(2.0)
    s.set_visible(True)

ax.set_title("", pad=10)

out = "/home/wangyuanshuo02/sglang/ref/figs/baseline_comp/fig1_accuracy_matrix"
export_figure(fig, out)
plt.close(fig)
print("fig1 done ->", out)
