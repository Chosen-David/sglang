# fig10：M8 kernel 组合阶梯 + 三方 select 微基准 + decode e2e 轨迹
# 数据源：TWO_LEVEL_PAPER_REPORT.md §8b-5/§8b-19/§8b-20 主表、PAPER_DRAFT.md §6.3/§6.4
# 口径注记：select 阶梯 E/F 两步绝对值跨轮环境漂移 ±7%（§8b-20），
#   F 用三档同进程自洽口径（eager 23.19 / off 1.93 / on 1.55）
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
C = {"gray": "#9e9e9e", "blue": "#1f77b4", "green": "#2ca02c",
     "orange": "#ff7f0e", "red": "#d62728", "purple": "#9467bd"}

fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))

# ---- (a) select kernel 组合阶梯（bs=32/131K 全函数）----
ax = axes[0]
labels = ["eager\n(P1-P7)", "+A+B kernels\n(fused score+compact)", "+C+D\n(dual-pool direct\n+row direct)",
          "+E near-pool\ncompact write", "+F CHUNK\nshape tuning"]
vals = [23.25, 2.11, 1.80, 1.46, 1.55]
colors = [C["gray"], C["blue"], C["blue"], C["green"], C["green"]]
bars = ax.bar(range(5), vals, color=colors, width=0.62)
ax.set_yscale("log")
ax.set_ylim(1, 40)
for i, v in enumerate(vals):
    sp = 23.25 / v
    ax.text(i, v * 1.12, f"{v:.2f} ms\n({sp:.0f}x)", ha="center", fontsize=7.5,
            fontweight="bold" if i >= 3 else "normal",
            color=C["green"] if i >= 3 else "#333333")
ax.set_xticks(range(5)); ax.set_xticklabels(labels, fontsize=7)
ax.set_ylabel("select_decode_batched (ms, log)")
ax.set_title("(a) kernel composition ladder\n(bs=32, S=131K; F = same-run 3-tier: 23.19->1.55 = 15x)", fontsize=8.5)
# E→F 的 1.46→1.55 注记：跨轮环境漂移 ±7%，kernel 级 dual 0.49→0.39
ax.annotate("E-to-F: cross-run drift +/-7%\n(dual kernel 0.49->0.39ms, 22%)",
            xy=(4, 1.55), xytext=(1.6, 6), fontsize=7, color="#888888",
            arrowprops=dict(arrowstyle="->", color="#aaaaaa", lw=0.7))

# ---- (b) 三方 select 微基准（单 token per-layer-call，官方 kernel 原样）----
ax = axes[1]
S = ["S=10K", "S=40K", "S=131K"]
quest = [0.066, 0.077, 0.107]
dsa = [0.476, 0.490, 0.503]
tli = [0.604, 0.658, 0.787]
x = np.arange(3); w = 0.26
ax.bar(x - w, quest, w, label="Quest (official raft)", color=C["orange"])
ax.bar(x, dsa, w, label="DSA (official tilelang fp8)", color=C["purple"])
ax.bar(x + w, tli, w, label="TLI fused L1 (single-token)", color=C["blue"])
for i in range(3):
    ax.text(i + w, tli[i] + 0.012, f"{tli[i]:.2f}", ha="center", fontsize=7)
    ax.text(i - w, quest[i] + 0.012, f"{quest[i]:.2f}", ha="center", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(S)
ax.set_ylabel("select latency (ms/layer, single token)")
ax.set_ylim(0, 1.0)
ax.set_title("(b) three-way microbench (official kernels)\n+ TLI batch: 1.55ms/32req -> 0.048ms/req/layer-equiv", fontsize=8.5)
ax.legend(loc="upper left", fontsize=6.5, framealpha=0.9)

# ---- (c) decode e2e 轨迹（bs=32, S=9.9K, CUDA graph）----
ax = axes[2]
stages = ["M3 eager\nprototype", "M4 batch\nselect", "M5 +CUDA\ngraph", "M8\n+kernels"]
ms = [1236.8, 326.4, 188.6, 99.0]
tps = [25.9, 169.7, 169.7 * 188.6 / 188.6, 323.0]  # tok/s 标注用真实值
bars = ax.bar(range(4), ms, color=[C["gray"], C["blue"], C["blue"], C["green"]], width=0.58)
ax.set_yscale("log")
ax.set_ylim(50, 3000)
for i, v in enumerate(ms):
    sp = 1236.8 / v
    ax.text(i, v * 1.15, f"{v:.0f} ms\n({sp:.1f}x)", ha="center", fontsize=7.5,
            fontweight="bold" if i == 3 else "normal",
            color=C["green"] if i == 3 else "#333333")
ax.text(3, 60, "323 tok/s", ha="center", fontsize=7.5, color=C["green"], fontweight="bold")
ax.set_xticks(range(4)); ax.set_xticklabels(stages, fontsize=7.5)
ax.set_ylabel("decode step time (ms, log)")
ax.set_title("(c) decode trajectory\n(bs=32, S=9.9K, narrativeqa, CUDA graph)", fontsize=8.5)

fig.suptitle("M8: batched kernel composition - select 15x (same-run 3-tier), parity = torch.equal x3 scenarios",
             fontsize=9, y=1.05, color="#444444")
fig.tight_layout()
fig.savefig(f"{OUT}/fig10_m8_kernels.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/fig10_m8_kernels.png", dpi=180, bbox_inches="tight")
print("saved fig10")
