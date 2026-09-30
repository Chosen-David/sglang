# Fig: E72 五臂 trace mass vs e2e F1 排名系统性反转（口径鸿沟方法论，贡献 #4）
# house style: okabe_ito 色盲友好色板 + Arial + 无多余网格 + PDF/PNG 双格式
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/wangyuanshuo02/two-level-attention/exp/figures"
d = json.load(open("/home/wangyuanshuo02/two-level-attention/exp/trace/results/e72_screen_verdict.json"))

# okabe_ito 色板（色盲友好）
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
      "red": "#D55E00", "purple": "#CC79A7", "gray": "#7F7F7F"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight",
})

arms = ["mavg", "aavg", "mminmax", "mavg(B7s)", "cavg"]
disp = {"mavg": "mavg\n(minmax,avg)", "aavg": "aavg\n(avg,avg)",
        "mminmax": "mminmax\n(minmax,minmax)", "mavg(B7s)": "mavg B7s\nβ=0.25 ref", "cavg": "cavg\n(cluster,avg)"}
trace_mass = [d[a]["trace_mass"] for a in arms]
e2e = [d[a]["screen_avg"] for a in arms]
# e2e 全量终值（mavg 50.54 冠军）
e2e_full = {"mavg": 50.54, "aavg": 43.77, "mminmax": 43.70, "mavg(B7s)": 50.41, "cavg": 42.87}

fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), sharey=False)

# ---- panel A: trace mass vs e2e F1 散点（排名反转主图）----
ax = axes[0]
# 各臂 trace 排名（降序名次）与 e2e 排名
tr_rank = {a: i for i, a in enumerate(sorted(arms, key=lambda x: -d[x]["trace_mass"]))}
e2e_rank = {a: i for i, a in enumerate(sorted(arms, key=lambda x: -e2e_full[x]))}
for a in arms:
    x, y = d[a]["trace_mass"], e2e_full[a]
    hot = a == "mavg"   # trace 第 4 → e2e 冠军，最 striking
    ax.scatter(x, y, s=110 if hot else 60,
               color=OI["red"] if hot else OI["blue"], zorder=3,
               edgecolors="white", linewidths=1.2)
    dy = {"mavg": 0.35, "aavg": 0.4, "mminmax": -0.75, "mavg(B7s)": 0.4, "cavg": -0.75}[a]
    ax.annotate(f"{a}\ntrace#{tr_rank[a]+1}→e2e#{e2e_rank[a]+1}",
                (x, y), xytext=(6, dy * 12), textcoords="offset points",
                fontsize=7.2, color="#333333")
ax.set_xlabel("trace replay mass (offline proxy)")
ax.set_ylabel("LongBench e2e F1 (13-task full)")
ax.set_title("(a) Offline proxy vs e2e: rank inversion", fontsize=9.5, fontweight="bold")
ax.set_xlim(0.78, 0.95)
ax.set_ylim(40, 55)

# ---- panel B: 双排名 slope 图（五臂全反转一眼可见）----
ax = axes[1]
xs = {"trace": 0, "e2e": 1}
for a in arms:
    y0 = -tr_rank[a]           # trace 排名（1 在上）
    y1 = -e2e_rank[a]
    inv = (y1 - y0) != 0
    ax.plot([0, 1], [y0, y1], color=OI["red"] if inv else OI["gray"],
            lw=2.2 if a == "mavg" else 1.3, alpha=0.9, zorder=2)
    ax.scatter([0, 1], [y0, y1], s=42, color=OI["red"] if inv else OI["gray"],
               zorder=3, edgecolors="white", linewidths=0.8)
    ax.annotate(a, (0, y0), xytext=(-8, 0), textcoords="offset points",
                ha="right", va="center", fontsize=7.8, color="#333333")
    ax.annotate(f"{e2e_full[a]:.2f}", (1, y1), xytext=(8, 0), textcoords="offset points",
                ha="left", va="center", fontsize=7.2, color="#333333")
ax.set_xlim(-0.45, 1.35)
ax.set_ylim(-4.6, 0.6)
ax.set_xticks([0, 1])
ax.set_xticklabels(["trace mass rank", "e2e F1 rank"], fontsize=8.5)
ax.set_yticks([])
ax.set_title("(b) All five arms re-ranked", fontsize=9.5, fontweight="bold")
for s in ("left", "bottom"):
    ax.spines[s].set_visible(False)

fig.suptitle("Metric-gap: offline trace-mass ranking cannot replace e2e verdict (E72)",
             fontsize=10, fontweight="bold", y=1.04)
fig.savefig(f"{OUT}/e72_rank_inversion.pdf")
fig.savefig(f"{OUT}/e72_rank_inversion.png")
print("saved e72_rank_inversion")
