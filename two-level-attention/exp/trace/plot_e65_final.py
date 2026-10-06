# E65 降维探索终版可视化（用户要求：降维前 vs 不同降维方法直观对比，美观）
# 输出：/home/wangyuanshuo02/sglang/ref/figs/e65_dim_reduction.{png,pdf}
# 口径：7 泛化样本（校准集 hotpotqa_0 不计入）mean ± std
#       trunc_d32 = 降维前基线（nope 32 维直取，无任何投影）
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = "/home/wangyuanshuo02/two-level-attention/exp/trace/results"
OUT = "/home/wangyuanshuo02/sglang/ref/figs"
e65 = json.load(open(f"{R}/e65_dim_reduction.json"))
e65b = json.load(open(f"{R}/e65b_sup_proj.json"))
SAMPLES = list(e65b.keys())  # 7 泛化样本
DIMS = [4, 8, 16, 32]

# (标签, 数据源, 颜色, 线型, 标记)——trunc 是基线族用实心蓝，监督投影暖色，无监督冷色系
METHODS = [
    ("trunc (tail-cut)", e65, "tab:blue", "-", "^"),
    ("sup_wsvd", e65b, "tab:red", "-", "o"),
    ("sup_grad", e65b, "tab:orange", "-", "s"),
    ("pca (full-dim)", e65, "tab:green", "--", "D"),
    ("pca (in-subspace)", e65, "tab:cyan", "--", "v"),
    ("jl (full-dim)", e65, "tab:purple", ":", "P"),
    ("jl (in-subspace)", e65, "tab:pink", ":", "X"),
]
PANELS = [
    ("A_far_coarse", "(A) far-coarse score reduction\n(fine fixed at nope-32)", "far mass capture"),
    ("C_fine", "(C) fine (refine) score reduction\n(coarse fixed at nope-32)", "far mass capture"),
    ("D_full", "(D) full pipeline, same d\n(coarse + fine)", "far mass capture"),
    ("B_near_coarse", "(B) near-coarse score reduction\n(near-zone mass)", "near mass capture"),
]


def agg(prefix, data):
    """7 样本 per-key 数组 → {d: (mean, std)}"""
    per = {}
    for name in SAMPLES:
        if name not in data:
            continue
        for k, v in data[name].items():
            if k.startswith(prefix):
                d = int(k.rsplit("_d", 1)[1])
                per.setdefault(d, []).append(v)
    return {d: (float(np.mean(v)), float(np.std(v))) for d, v in per.items()}


plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False,
                     "axes.spines.right": False})
fig, axes = plt.subplots(2, 2, figsize=(12.5, 9))

table = {}  # 供表格导出
for ax, (arm, title, ylab) in zip(axes.flat, PANELS):
    # 降维前基线（nope 32 维直取 = trunc_d32）
    base = agg(f"{arm}_trunc_", e65).get(32)
    if base:
        ax.axhline(base[0], color="k", ls=(0, (6, 3)), lw=1.6, alpha=0.75, zorder=1)
        ax.text(0.99, base[0] + 0.025, "nope-32 original (no reduction)",
                transform=ax.get_yaxis_transform(), ha="right",
                fontsize=8.5, color="k", alpha=0.85)
    for label, data, color, ls, marker in METHODS:
        a = agg(f"{arm}_{label.split(' (')[0]}_", data)
        if not a:
            continue
        xs = sorted(a)
        mu = [a[d][0] for d in xs]
        sd = [a[d][1] for d in xs]
        is_base = label.startswith("trunc")
        ax.plot(xs, mu, color=color, ls=ls, marker=marker, ms=5.5,
                lw=2.0 if is_base else 1.6, label=label, zorder=3 if is_base else 2)
        ax.fill_between(xs, [m - s for m, s in zip(mu, sd)],
                        [m + s for m, s in zip(mu, sd)], color=color, alpha=0.12, lw=0)
        table[(arm, label)] = {d: round(a[d][0], 3) for d in xs}
    ax.set_xscale("log", base=2)
    ax.set_xticks(DIMS)
    ax.set_xticklabels([str(d) for d in DIMS])
    ax.set_xlim(3.4, 38)
    ax.set_ylim(0, 1.02)
    ax.set_title(title, fontsize=10.5)
    ax.set_ylabel(ylab, fontsize=9)
    ax.grid(alpha=0.25, lw=0.6)
    ax.axhline(1.0, color="gray", lw=0.6, alpha=0.4)

axes[0, 0].legend(fontsize=8, loc="upper left", frameon=False, ncol=1)
fig.suptitle("E65 dimension-reduction on the nope-32 subspace (Qwen3-8B, mean$\\pm$std over 7 held-out samples; calibrated on hotpotqa_0)",
             fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.955))
fig.savefig(f"{OUT}/e65_dim_reduction.png", dpi=170)
fig.savefig(f"{OUT}/e65_dim_reduction.pdf")
print("saved:", OUT)

# ---- markdown 表格导出 ----
lines = ["# E65 dimension-reduction ablation (mean of 7 held-out samples)", "",
         "A=far coarse / B=near coarse / C=fine / D=full pipeline; values = far(near) mass capture; trunc_d32 = no-reduction baseline", ""]
for arm, title, _ in PANELS:
    lines.append(f"## {title.splitlines()[0]}")
    lines.append("| 方法 | d=4 | d=8 | d=16 | d=32 |")
    lines.append("|---|---|---|---|---|")
    for label, data, *_ in METHODS:
        row = table.get((arm, label), {})
        if row:
            lines.append("| " + label + " | " + " | ".join(
                f"**{row[d]:.3f}**" if (label.startswith("trunc") and d == 32) else f"{row.get(d, float('nan')):.3f}"
                for d in DIMS) + " |")
    lines.append("")
open(f"{OUT}/e65_dim_reduction_table.md", "w").write("\n".join(lines) + "\n")
print("table saved")
