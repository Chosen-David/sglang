# fig9：e2e 诚实边界三联（PAPER_DRAFT §6.4 图形化）
# (a) decode S 收窄链（bs=16，CUDA graph，N=256，ignore_eos 干净口径 §8b-25）：
#     S≈7.5K（实测 avg，历史标签 9.9K 按中文比例错算）慢 3.53×（63.5/18.0）
#     → 30K 慢 1.61×（65.0/40.4）；误差棒 = 多轮 spread（tli 63.5±0.4 稳态、
#     triton 40.4±1.6）
# (b) prefill 30K 双档：M8 787/1565s → M10 373/741s（2.11× 两档一致）vs triton 58/115s
# (c) 翻转点外推（S 标签修正后）：triton 两点（7.5K,18.0）（30K,40.4）拟合
#     b=1.005ms/K、a=10.4ms；tli（7.5K,63.5）（30K,65.0）b=0.067ms/K；
#     拟合线交叉 ~56K；tli 平台带 63.5-65.0 → 交叉区间 54–57K
# 数据源：tli_e2e_variance_results.json（ignore_eos + ktok=257 校验 + 4 轮分布）
#         / tli_m8_e2e_long_results.json（prefill 档）
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
C = {"gray": "#9e9e9e", "blue": "#1f77b4", "green": "#2ca02c",
     "orange": "#ff7f0e", "red": "#d62728", "purple": "#9467bd"}

fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.7))

# ---- (a) decode S 收窄链（bs=16，CUDA graph，干净口径；40K 点 mem0.85+chunk2048）----
ax = axes[0]
S_lbl = ["S=7.5K", "S=30K", "S=40K"]
tli = [63.5, 65.0, 64.9]
tli_err = [0.4, 1.1, 2.4]
tri = [18.0, 40.4, 64.8]
tri_err = [0.0, 1.6, 1.3]
x = np.arange(3); w = 0.35
ax.bar(x - w / 2, tri, w, yerr=tri_err, capsize=3,
       label="triton dense", color=C["orange"])
ax.bar(x + w / 2, tli, w, yerr=tli_err, capsize=3,
       label="TLI (M8 kernels)", color=C["blue"])
for i in range(3):
    r = tli[i] / tri[i]
    ax.text(i, max(tli[i], tri[i]) + 5, f"{r:.2f}x slower", ha="center",
            fontsize=8, fontweight="bold", color=C["red"] if r > 2 else "#8a6d00")
ax.set_xticks(x); ax.set_xticklabels(S_lbl, fontsize=8.5)
ax.set_ylabel("decode step (ms, bs=16, graph, N=256)")
ax.set_ylim(0, 82)
ax.legend(loc="upper left", fontsize=7.5, framealpha=0.9)
ax.set_title("(a) honest boundary: gap narrows\n3.53x -> 1.61x -> 1.00x (parity at 40K)", fontsize=8.5)

# ---- (b) prefill 30K 批分双档（M8 → M10 kernel 化）----
ax = axes[1]
bs_lbl = ["bs=8", "bs=16"]
m8 = [787, 1565]
m10 = [373, 741]
tri_p = [58, 115]
x = np.arange(2); w = 0.26
ax.bar(x - w, m8, w, label="TLI M8 (eager select)", color=C["gray"])
ax.bar(x, m10, w, label="TLI M10 (kernel select)", color=C["green"])
ax.bar(x + w, tri_p, w, label="triton dense", color=C["orange"])
for i in range(2):
    ax.text(i - w, m8[i] + 25, f"{m8[i]:.0f}s", ha="center", fontsize=7)
    ax.text(i, m10[i] + 25, f"{m10[i]:.0f}s", ha="center", fontsize=7, color=C["green"])
    ax.text(i + w, tri_p[i] + 25, f"{tri_p[i]:.0f}s", ha="center", fontsize=7)
    ax.annotate("", xy=(i, m10[i] * 0.92), xytext=(i - w, m8[i] * 0.92),
                arrowprops=dict(arrowstyle="->", color=C["green"], lw=1.0))
ax.text(0.5, 1720, "2.11x both tiers (linear bs*S term removed)",
        ha="center", fontsize=8, color=C["green"], fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(bs_lbl, fontsize=9)
ax.set_ylabel("prefill time (s, S=30K)")
ax.set_ylim(0, 1900)
ax.legend(loc="upper left", fontsize=7.5, framealpha=0.9)
ax.set_title("(b) prefill: M8 -> M10 kernelization\n2.11x at both batch tiers", fontsize=8.5)

# ---- (c) 三点实测交叉（bs=16，7.5K/30K/40K；40K 点 mem0.85+chunk2048）----
ax = axes[2]
S_pts = np.array([7.5, 30.0, 40.0])
tri_pts = np.array([18.0, 40.4, 64.8])
tli_pts = np.array([63.5, 65.0, 64.9])
b_t, a_t = np.polyfit(S_pts, tri_pts, 1)  # 3pt: b~1.365, a~5.8
b_l, a_l = np.polyfit(S_pts, tli_pts, 1)  # 3pt: b~0.047（平台）
Sx = np.linspace(0, 128, 200)
fit_tri = a_t + b_t * Sx
fit_tli = a_l + b_l * Sx
cross = (a_l - a_t) / (b_t - b_l)  # ~43.6
meas = (Sx >= 0) & (Sx <= 40)
extr = (Sx >= 40)
ax.plot(Sx[meas], fit_tri[meas], color=C["orange"], lw=1.8, label=f"triton fit (b={b_t:.2f} ms/K)")
ax.plot(Sx[extr], fit_tri[extr], color=C["orange"], lw=1.4, ls="--", alpha=0.75)
ax.plot(Sx[meas], fit_tli[meas], color=C["blue"], lw=1.8, label=f"TLI fit (b={b_l:.2f} ms/K)")
ax.plot(Sx[extr], fit_tli[extr], color=C["blue"], lw=1.4, ls="--", alpha=0.75)
# 实测交叉区间：tli 平台 63.5-67.3 × triton 3pt 拟合 → 42-45K
ax.axvspan(42, 45, color=C["green"], alpha=0.13)
ax.text(43.5, 20, "crossover\n~42-45K\n(3-pt measured,\n40K at parity)", ha="center",
        fontsize=7.5, color=C["green"], fontweight="bold")
ax.scatter(S_pts, tri_pts, color=C["orange"], zorder=5, s=28)
ax.scatter(S_pts, tli_pts, color=C["blue"], zorder=5, s=28)
ax.annotate("(7.5K, 18.0)", xy=(7.5, 18.0), xytext=(14, 6), fontsize=6.5,
            color=C["orange"], arrowprops=dict(arrowstyle="->", color=C["orange"], lw=0.6))
ax.annotate("(30K, 40.4)", xy=(30, 40.4), xytext=(20, 26), fontsize=6.5,
            color=C["orange"], arrowprops=dict(arrowstyle="->", color=C["orange"], lw=0.6))
ax.annotate("(40K, 64.8)", xy=(40, 64.8), xytext=(48, 50), fontsize=6.5,
            color=C["orange"], arrowprops=dict(arrowstyle="->", color=C["orange"], lw=0.6))
ax.annotate("(7.5K, 63.5)", xy=(7.5, 63.5), xytext=(2, 78), fontsize=6.5, color=C["blue"],
            arrowprops=dict(arrowstyle="->", color=C["blue"], lw=0.6))
ax.annotate("(30K, 65.0)", xy=(30, 65.0), xytext=(14, 90), fontsize=6.5, color=C["blue"],
            arrowprops=dict(arrowstyle="->", color=C["blue"], lw=0.6))
ax.annotate("(40K, 64.9)", xy=(40, 64.9), xytext=(26, 84), fontsize=6.5, color=C["blue"],
            arrowprops=dict(arrowstyle="->", color=C["blue"], lw=0.6))
ax.set_xlabel("context length S (K tokens)")
ax.set_ylabel("decode step (ms, bs=16)")
ax.set_xlim(0, 128); ax.set_ylim(0, 100)
ax.legend(loc="lower right", fontsize=7)
ax.set_title("(c) crossover: 3-pt measured ~42-45K\nTLI ~S-invariant vs triton linear in S", fontsize=8.5)

fig.suptitle("End-to-end honest boundaries: gap narrows with S (3.53x -> 1.61x -> 1.00x parity at 40K); crossover ~42-45K measured; prefill kernelization 2.11x",
             fontsize=9, y=1.05, color="#444444")
fig.tight_layout()
fig.savefig(f"{OUT}/fig9_e2e_boundaries.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/fig9_e2e_boundaries.png", dpi=180, bbox_inches="tight")
print(f"saved fig9 (crossover fit: {cross:.1f}K)")
