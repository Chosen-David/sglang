#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig2_ruler_heatmap.py
论文对比数据可视化（二）：RULER 32K/64K/128K 逐任务精度热力图

行 = 方法，列 = 11 个 RULER 合成任务（niah_single/multikey/multiquery/multivalue/cwe/fwe/vt）。
数值 = score_ruler_formal.py 分数（Qwen3-8B）。NaN = 未测/在飞（SparQ 64K/128K、FASA 64K/128K）。
FASA 32K 仅 6/11 任务。
数据源：
  TLI mavg 32K=E116b（e116b_ruler_summary.json）/ 64K·128K=E119 重跑
  FullKV/SparQ/MInference/Quest/FASA = e119_*_formal_vendor.json 同协议
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

sys.path.insert(0, "/home/wangyuanshuo02/.claude/skills/publication-figures/scripts")
from style_config import apply_style  # noqa: E402
from export_figure import export_figure  # noqa: E402

palette = apply_style("okabe_ito")
plt.rcParams["font.family"] = "DejaVu Sans"

TASKS = ["s1", "s2", "s3", "mk1", "mk2", "mk3", "mq", "mv", "cwe", "fwe", "vt"]
METHODS = ["TLI (ours)", "FullKV", "SparQ", "MInference", "Quest", "FASA"]
NAN = np.nan

# 逐任务分数（行=方法顺序，列=任务顺序）
data_32k = np.array([
    [100, 100, 98, 98, 84, 8, 24.5, 28.5, 71.3, 11.67, 32.2],   # TLI mavg E116b
    [100, 100, 100, 99, 76, 4, 24.5, 25.0, 83.5, 7.0, 34.2],    # FullKV
    [100, 100, 100, 99, 74, 10, 27.5, 33.0, 77.3, 10.0, 34.2],  # SparQ
    [42, 100, 100, 99, 78, 1, 18.0, 25.0, 16.1, 45.0, 25.2],    # MInference
    [41, 96, 36, 87, 44, 0, 20.0, 32.0, 44.6, 18.0, 52.8],      # Quest
    [0, 78, 4, 53, 9, 0, NAN, NAN, NAN, NAN, NAN],              # FASA 6/11
], dtype=float)

data_64k = np.array([
    [86, 100, 100, 99, 54, 0, 28.5, 31.5, 0.3, 12.33, 32.0],   # TLI mavg E119
    [100, 100, 100, 100, 50, 0, 20.25, 26.25, 0.6, 4.0, 32.8], # FullKV
    [NAN] * 11,
    [75, 100, 100, 100, 52, 1, 25.75, 27.5, 7.7, 9.67, 54.6],  # MInference
    [59, 79, 4, 71, 6, 0, 14.0, 16.25, 0.8, 9.67, 29.0],       # Quest
    [NAN] * 11,
], dtype=float)

data_128k = np.array([
    [100, 100, 99, 99, 50, 1, 21.25, 2.0, 0.7, 26.67, 22.8],  # TLI mavg E119
    [100, 100, 100, 99, 57, 1, 12.25, 2.5, 3.4, 13.0, 22.6],  # FullKV
    [NAN] * 11,
    [19, 100, 86, 99, 79, 0, 25.0, 39.0, 3.4, 36.67, 57.8],   # MInference
    [85, 46, 0, 36, 1, 0, 6.75, 3.5, 0.6, 9.33, 24.4],        # Quest
    [NAN] * 11,
], dtype=float)

# 白 → okabe_ito primary blue 渐变
cmap = mcolors.LinearSegmentedColormap.from_list("tli_blue",
        ["#ffffff", "#cfe0ee", "#0072B2"], N=256)
NAN_COLOR = "#d8d8d8"

def draw_panel(ax, data, title):
    ax.set_title(title, fontsize=22, pad=12)
    ax.imshow(np.ma.masked_invalid(data), cmap=cmap, vmin=0, vmax=100,
              aspect="auto", interpolation="nearest")
    n_m, n_t = data.shape
    ax.set_xticks(np.arange(n_t))
    ax.set_xticklabels(TASKS, fontsize=12, rotation=40, ha="right")
    ax.set_yticks(np.arange(n_m))
    ax.set_yticklabels(METHODS, fontsize=15)
    # 网格线
    ax.set_xticks(np.arange(-0.5, n_t, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_m, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    # 缺失格 → 灰 + n/a
    for i in range(n_m):
        for j in range(n_t):
            v = data[i, j]
            if np.isnan(v):
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                              facecolor=NAN_COLOR, edgecolor="white", linewidth=1.2))
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=9,
                        color="#666666", style="italic")
            else:
                tc = "white" if v > 65 else "#111111"
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=12, color=tc, fontweight="bold")
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_xticks(np.arange(n_t), minor=False)
    # 保留 set_xticklabels 设置的 rotation/ha（不重置为 0）
    ax.spines[:].set_linewidth(1.5)

fig, axes = plt.subplots(1, 3, figsize=(19, 6.2))
draw_panel(axes[0], data_32k, "RULER-32K")
draw_panel(axes[1], data_64k, "RULER-64K")
draw_panel(axes[2], data_128k, "RULER-128K")

# 色标
sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0, 100))
cbar = fig.colorbar(sm, ax=axes, pad=0.02, aspect=30, shrink=0.9)
cbar.set_label("Accuracy (%)", fontsize=22, rotation=270, labelpad=28)
cbar.ax.tick_params(labelsize=18)

# 脚注：任务缩写
fig.text(0.5, 0.01,
         "s=niah_single, mk=niah_multikey, mq=niah_multiquery, mv=niah_multivalue; "
         "SparQ 64K/128K and FASA 64K/128K not measured; FASA 32K covers 6/11 tasks",
         ha="center", fontsize=13, color="#444444")

plt.tight_layout(rect=[0, 0.02, 1, 1])
out = "/home/wangyuanshuo02/sglang/ref/figs/baseline_comp/fig2_ruler_heatmap"
export_figure(fig, out)
plt.close(fig)
print("fig2 done ->", out)
