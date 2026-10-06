# Fig 8: M3 系统集成实测（sglang tli backend，Qwen3-8B 真实上下文，H20）
# 数据源：tli_batch_decode_results.json / tli_throughput_results.json（e2e 真实 narrativeqa）
# + select 微基准（L1+L2 双 kernel vs eager）+ decode 归因（TLI_PROFILE_TIMING）
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = "/home/wangyuanshuo02/sglang/"
OUT = "/home/wangyuanshuo02/two-level-attention/exp/figures"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 150, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
})
C = {"blue": "#2f6f9f", "red": "#c1443c", "green": "#3a7d44", "orange": "#d98e32",
     "purple": "#7b5aa6", "gray": "#8a8a8a", "teal": "#2a9d8f"}

d = json.load(open(R + "tli_batch_decode_results.json"))
bs = [r["bs"] for r in d["triton"]]
tr_ms = [r["step_ms"] for r in d["triton"]]
tli_ms = [r["step_ms"] for r in d["tli"]]

fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.0))

# ---- (a) 高并发 decode 曲线 ----
ax = axes[0]
ax.plot(bs, tr_ms, "o-", color=C["blue"], label="triton (dense)")
ax.plot(bs, tli_ms, "s-", color=C["red"], label="tli (eager per-req loop)")
ax.set_xscale("log", base=2); ax.set_yscale("log", base=10)
ax.set_xticks(bs); ax.set_xticklabels([str(b) for b in bs])
ax.set_xlabel("batch size (S ≈ 9.9K tok, Qwen3-8B)")
ax.set_ylabel("decode step latency (ms)")
ax.set_title("(a) M3-c: decode vs batch size (H20)")
ax.legend(loc="upper left", framealpha=0.9)
# 标注线性放大斜率
ax.annotate("linear ×bs\n(Python per-req loop)", xy=(16, 666), xytext=(3.2, 400),
            fontsize=7.5, color=C["red"], arrowprops=dict(arrowstyle="->", color=C["red"], lw=0.8))
ax.annotate("3.2× total\n(batched)", xy=(32, 32.4), xytext=(6, 14),
            fontsize=7.5, color=C["blue"], arrowprops=dict(arrowstyle="->", color=C["blue"], lw=0.8))

# ---- (b) e2e decode 优化轨迹（bs=1, S≈9.9K token，墙钟口径，全部真实实测）----
# M3-a: eager 81 / M3-b: +_sparse_attn 向量化 + L1 kernel 56 / M3-c: +L2 kernel 51.4
ax = axes[1]
stages = ["eager\n(M3-a)", "+vec attn\n+L1 kernel", "+L2 kernel\n(M3-c)"]
steps = [81.0, 56.0, 51.4]
x = np.arange(3)
bars = ax.bar(x, steps, 0.55, color=[C["gray"], C["teal"], C["green"]])
for i, v in enumerate(steps):
    ax.text(i, v + 1.5, f"{v:.0f} ms", ha="center", fontsize=9, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(stages)
ax.set_ylabel("decode step latency (ms, wall clock)")
ax.set_ylim(0, 95)
ax.set_title("(b) M3: decode step trajectory (bs=1)")
# 归因注释（TLI_PROFILE_TIMING 同口径对比：select 46 → 28.7 ms）
ax.annotate("select 46→28.7 ms\n(same-methodology attribution)", xy=(2, 51.4),
            xytext=(0.7, 75), fontsize=7.5, color="#444444",
            arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8))

# ---- (c) select 微基准（两级 kernel 化）----
ax = axes[2]
S_list = [9891, 131072]
eager = [0.527, 0.658]
both = [0.392, 0.623]
x = np.arange(2)
w = 0.35
ax.bar(x - w / 2, eager, w, label="eager PyTorch", color=C["gray"])
ax.bar(x + w / 2, both, w, label="L1+L2 fused kernels", color=C["green"])
for i, (e, b) in enumerate(zip(eager, both)):
    ax.text(i + w / 2, b + 0.01, f"{e / b:.2f}×", ha="center", fontsize=8,
            fontweight="bold", color=C["green"])
ax.set_xticks(x); ax.set_xticklabels(["S=9.9K", "S=131K"])
ax.set_ylabel("select latency (ms)")
ax.set_ylim(0, 0.78)
ax.set_title("(c) two-level select: fused vs eager")
ax.legend(loc="upper left", framealpha=0.9)

fig.suptitle("M3: sglang integration (real Qwen3-8B, narrativeqa contexts; kernel-vs-eager parity = exact)",
             fontsize=9, y=1.04, color="#444444")
fig.savefig(f"{OUT}/fig8_m3_system.pdf")
fig.savefig(f"{OUT}/fig8_m3_system.png")
plt.close(fig)
print("saved fig8")
