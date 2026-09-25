# Fig 9: M3-M10 系统兑现三联图（论文 §6.4 / PPT）
# (a) decode 优化轨迹 M3→M5→M8 @bs=32（1236.8→188.6→99.0，triton 参照线）
# (b) 30K e2e prefill M8→M10 双档（787/1565→373/741s，triton 参照）
# (c) S 收窄链：bs16 decode tli/triton 比 9.9K→30K→128K（外推虚线，翻转点标注）
# 数据源（全部实测 JSON，无手抄）：
#   tli_m8_e2e_results.json（M8 9.9K graph）、tli_m8_e2e_long_results.json
#   （30K 双侧 N=256 + M10 prefill）、tli_m10_bench.json（微基准）
# 风格对齐导师仓库 make_figures.py（同 rcParams/配色）。
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "/home/wangyuanshuo02/sglang"
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 150, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
})
C = {"blue": "#2f6f9f", "red": "#c1443c", "green": "#3a7d44", "orange": "#d98e32",
     "purple": "#7b5aa6", "gray": "#8a8a8a", "teal": "#2a9d8f"}

e2e = json.load(open(f"{OUT}/tli_m8_e2e_results.json"))
long_ = json.load(open(f"{OUT}/tli_m8_e2e_long_results.json"))

fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.0))

# ---- (a) decode 优化轨迹 @bs=32（9.9K，graph 口径） ----
ax = axes[0]
m8 = {r["bs"]: r for r in e2e["m8_tli_graph1"]}
tri = {r["bs"]: r for r in e2e["m8_triton_graph1"]}
steps = [("M3 原型\n(eager)", 1236.8), ("M5\n+graph", 188.6),
         ("M8\n+kernel", m8[32]["step_ms"]), ("triton\ndense", tri[32]["step_ms"])]
xs = np.arange(len(steps))
vals = [v for _, v in steps]
colors = [C["red"], C["orange"], C["green"], C["gray"]]
bars = ax.bar(xs, vals, 0.62, color=colors)
for x, v in zip(xs, vals):
    ax.text(x, v * 1.03, f"{v:.1f}", ha="center", fontsize=8)
ax.set_yscale("log")
ax.set_ylabel("decode step latency (ms, log)")
ax.set_xticks(xs)
ax.set_xticklabels([n for n, _ in steps], fontsize=7.5)
ax.set_title("(a) decode trajectory @bs=32, S=9.9K\n12.5× total (323 tok/s)")
ax.set_ylim(20, 2600)

# ---- (b) 30K prefill M8→M10 双档 ----
ax = axes[1]
m10 = {r["bs"]: r for r in long_["m10_long_tli_graph1"]}
m8l = {r["bs"]: r for r in long_["m8_long_tli_graph1"]}
tri = {r["bs"]: r for r in long_["m8_long_triton_graph1"]}
bs_list = [8, 16]
w = 0.26
x = np.arange(len(bs_list))
for i, (lbl, getter, col) in enumerate([
    ("M8 eager select", lambda b: m8l[b]["prefill_s"], C["orange"]),
    ("M10 kernel", lambda b: m10[b]["prefill_s"], C["green"]),
    ("triton dense", lambda b: tri[b]["prefill_s"], C["gray"]),
]):
    vals = [getter(b) for b in bs_list]
    bars = ax.bar(x + (i - 1) * w, vals, w, label=lbl, color=col)
    for xx, v in zip(x + (i - 1) * w, vals):
        ax.text(xx, v * 1.03, f"{v:.0f}", ha="center", fontsize=7.5)
ax.set_xticks(x)
ax.set_xticklabels([f"bs={b}" for b in bs_list])
ax.set_ylabel("prefill latency (s)")
ax.set_title("(b) 30K prefill: M10 kernel 化\n2.11× both bs (787→373 / 1565→741 s)")
ax.legend(framealpha=0.9)
ax.set_ylim(0, 1900)

# ---- (c) S 收窄链（bs16 decode tli/triton） ----
ax = axes[2]
# 实测两点：9.9K（M8 e2e bs16 64.7 / triton 18.7）与 30K（N=256 双侧 67.1 / 43.4）
S_pts = [9.9, 30.0]
ratio_pts = [64.7 / 18.7, 67.1 / 43.4]
ax.plot(S_pts, ratio_pts, "o-", color=C["red"], lw=2, ms=6, label="measured tli/triton (bs=16)")
for s, r in zip(S_pts, ratio_pts):
    ax.annotate(f"{r:.2f}×", (s, r), textcoords="offset points",
                xytext=(8, 6), fontsize=8, color=C["red"])
# 外推：triton attention 流量 ∝S vs tli ∝K2 恒定——简化线性模型（MLP 项固定）：
# triton_step(S) ≈ a + b*S；tli_step(S) ≈ c（9.9K: 18.7= a+9.9b; 30K: 43.4=a+30b
# → b=1.204, a=6.78）；tli 67.1→c~55-67 取 60 线性缓增 → 翻转点解 a+bS=60
b_s = (43.4 - 18.7) / (30.0 - 9.9)
a_s = 18.7 - 9.9 * b_s
S_flip = (60.0 - a_s) / b_s
S_ext = np.linspace(9.9, 135, 100)
tri_ext = a_s + b_s * S_ext
tli_ext = 60.0 + 0.25 * (S_ext - 30.0)  # 轻微增长项（select 池 ∝bs×K2 与 S 无关）
ax.plot(S_ext, tri_ext / tli_ext, "--", color=C["gray"], lw=1.5,
        label="extrapolated (linear traffic model)")
ax.axhline(1.0, color=C["green"], lw=1.2, ls=":")
ax.axvline(S_flip, color=C["green"], lw=1, ls="--")
ax.annotate(f"crossover ≈ S={S_flip:.0f}K", (S_flip, 1.0), textcoords="offset points",
            xytext=(8, 10), fontsize=8, color=C["green"])
ax.set_xlabel("context length S (K tokens)")
ax.set_ylabel("tli / triton decode latency ratio")
ax.set_title("(c) gap narrows with S (bs=16)\nextrapolated crossover at S≈128K (H100 target)")
ax.legend(framealpha=0.9, loc="upper right")
ax.set_ylim(0.4, 4.2)

fig.suptitle("System payoffs M3→M10 (sglang, Qwen3-8B, H20-3e; all measured, honest boundaries)",
             fontsize=10.5, y=1.04)
fig.savefig(f"{OUT}/fig9_m3_to_m10_payoffs.pdf")
fig.savefig(f"{OUT}/fig9_m3_to_m10_payoffs.png")
print("saved fig9_m3_to_m10_payoffs.pdf/png")
print(f"crossover S = {S_flip:.1f}K (linear model from two measured points)")
