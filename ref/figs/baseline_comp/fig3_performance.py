#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig3_performance.py
论文对比数据可视化（三）：性能

Panel A: decode ms/step（TLI vs Quest vs dense-Triton，Qwen3-30B-A3B，sglang 集成，bs=1）
         数据源：c3_3arm_e2e.json（两遍法 pass1=prefill / pass2=total / decode=差值，单请求 64 decode step）
         诚实标注：E135 审计判 "reported/unverified"（单次运行、无 EOS/token 计数、无 timestamp），
         入论文前需空卡重测。
Panel B: TLI 索引 kernel 加速（E127 paired，normal 层，H20-3e 同机配对，GPU speedup）
         数据源：e128_tradeoff.json -> e127_paired_baseline_vs_iter8
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/wangyuanshuo02/.claude/skills/publication-figures/scripts")
from style_config import apply_style, get_colors  # noqa: E402
from export_figure import export_figure            # noqa: E402

palette = apply_style("okabe_ito")
plt.rcParams["font.family"] = "DejaVu Sans"
COLORS = get_colors("okabe_ito", n=6)
# TLI=blue, Quest=vermillion, Dense=gray
C_TLI, C_QUEST, C_DENSE = COLORS[0], COLORS[4], COLORS[1]

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(15.5, 6.6), layout="constrained")

# ---------------------------------------------------------------------------
# Panel A: decode ms/step
# ---------------------------------------------------------------------------
S_lens = ["32K", "64K"]
ms = {
    "TLI (ours)":  [140.68, 141.33],
    "Quest":       [108.04, 107.83],
    "Dense":       [44.93, 43.88],
}
colors_a = [C_TLI, C_QUEST, C_DENSE]
x = np.arange(len(S_lens))
bar_w = 0.24
ax_a.set_title("End-to-end decode latency (unverified)", fontsize=20, pad=10)
for i, (m, vals) in enumerate(ms.items()):
    off = (i - 1) * bar_w
    ax_a.bar(x + off, vals, width=bar_w, color=colors_a[i],
             edgecolor="black", linewidth=1.2, label=m, zorder=3)
    for j, v in enumerate(vals):
        ax_a.text(x[j] + off, v + 1.5, f"{v:.0f}", ha="center", va="bottom",
                  fontsize=12, color="black")
ax_a.set_xticks(x)
ax_a.set_xticklabels([f"S={l}" for l in S_lens], fontsize=18)
ax_a.set_ylabel("Decode latency (ms/step)", fontsize=20)
ax_a.set_ylim(0, 160)
ax_a.set_yticks([0, 40, 80, 120, 160])
ax_a.tick_params(axis="y", labelsize=17)
# legend 移到 Panel A 底部外侧横排，避免遮挡柱顶数值标注
leg_a = ax_a.legend(fontsize=15, frameon=True, fancybox=False, shadow=False,
                    loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3,
                    borderaxespad=0.0)
leg_a.get_frame().set_linewidth(2.0)
leg_a.get_frame().set_edgecolor("black")

# ---------------------------------------------------------------------------
# Panel B: TLI kernel speedup vs S
# ---------------------------------------------------------------------------
S_pts = np.array([8192, 16384, 32768, 65536])
speedup = np.array([1.159, 1.882, 3.065, 5.468])
ax_b.set_title("TLI index kernel speedup", fontsize=20, pad=10)
ax_b.plot(S_pts, speedup, "o-", color=C_TLI, linewidth=3.5,
          markersize=12, markeredgecolor="white", markeredgewidth=2.5,
          label="GPU speedup vs baseline", zorder=5)
for xi, yi in zip(S_pts, speedup):
    ax_b.text(xi, yi + 0.25, f"{yi:.2f}×", ha="center", va="bottom",
              fontsize=15, color=C_TLI)
ax_b.set_xscale("log", base=2)
ax_b.set_xticks(S_pts)
ax_b.set_xticklabels(["8K", "16K", "32K", "64K"], fontsize=17)
ax_b.set_xlabel("Context length", fontsize=20)
ax_b.set_ylabel("Speedup (×)", fontsize=20)
ax_b.set_ylim(0.5, 6.8)
ax_b.tick_params(axis="y", labelsize=17)
leg_b = ax_b.legend(fontsize=15, frameon=True, fancybox=False, shadow=False,
                    loc="upper left")
leg_b.get_frame().set_linewidth(2.0)
leg_b.get_frame().set_edgecolor("black")
# 注记：iter8 后 64K normal 每步 GPU 1.52ms
# 纯文字放曲线之上空区（x≈55K, y≈6.3），避开左上 legend 与数据点标注；无箭头避免 lint 碰撞
ax_b.text(55000, 6.3, "64K: 8.29→1.52 ms/step",
          ha="center", fontsize=14, color="#444444")

for a in (ax_a, ax_b):
    for s in a.spines.values():
        s.set_linewidth(2.0)
        s.set_visible(True)

fig.text(0.5, 0.01,
         "Panel A: c3_3arm_e2e.json, Qwen3-30B-A3B, bs=1, sglang; E135 audit: unverified, "
         "re-measure on empty GPUs. Panel B: e128_tradeoff.json, E127 paired, H20-3e, normal layer",
         ha="center", fontsize=13, color="#444444")

out = "/home/wangyuanshuo02/sglang/ref/figs/baseline_comp/fig3_performance"
export_figure(fig, out)
plt.close(fig)
print("fig3 done ->", out)
