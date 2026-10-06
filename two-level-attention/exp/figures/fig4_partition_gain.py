# Fig 4 (round3 新增): 分区收益双 panel
# (a) RULER 长度梯度：FullKV / PSI-单池(C0) / PSI(B7s 分区) 三长度 AVG 折线
#     —— 支撑「分区 far 检索收益区 = 超长上下文」(长度梯度 −0.14 → −0.25 → +0.98)
# (b) LongBench per-task Δ 条形：PSI(50.54, E72 mavg) vs PSI-单池(C0 50.16)
#     —— 支撑「13 任务互有胜负、musique +5.02 主导」的诚实分解
# 数据快照（落袋 JSON，勿手改数字）：
#   exp/results_ruler/ruler_c0.json         单池 C0 分长度 per-task
#   exp/results_ruler/ruler_b7s.json        分区 B7s 分长度 per-task
#   exp/results_ruler/ruler_table_final.json none/quest/tia 分长度 per-task
#   exp/trace/results/e71_main_table.json   LongBench 13 臂 × 13 任务
# house style: okabe_ito、白底、全英文、矢量 PDF+PNG
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RR = "/home/wangyuanshuo02/two-level-attention/exp/results_ruler"
LB = "/home/wangyuanshuo02/two-level-attention/exp/trace/results"
OUTS = ["/home/wangyuanshuo02/sglang/paper/figures",
        "/home/wangyuanshuo02/two-level-attention/exp/figures"]

OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
      "red": "#D55E00", "purple": "#CC79A7", "gray": "#7F7F7F", "sky": "#56B4E9"}
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "figure.dpi": 200, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})

# ---------- 数据 ----------
c0 = json.load(open(f"{RR}/ruler_c0.json"))
b7s = json.load(open(f"{RR}/ruler_b7s.json"))
fin = json.load(open(f"{RR}/ruler_table_final.json"))
e71 = json.load(open(f"{LB}/e71_main_table.json"))

LENS = ["L4096", "L8192", "L16384"]
def avg(d, key):
    v = d[f"{key}"]
    return float(np.mean([v[t] for t in v if t != "AVG"]))
full_kv = [avg(fin, f"{L}/none") for L in LENS]
psi_c0 = [avg(c0, f"{L}/tli_64_128_1024_c4_A") for L in LENS]
psi_b7s = [avg(b7s, f"{L}/tli_64_128_1024_c4_A") for L in LENS]

# LongBench per-task Δ (PSI = E72 mavg 头条臂 50.54；单池 = C0 50.16)
tasks = [t for t in e71["TLI_C0"].keys() if t != "AVG"]
delta = np.array([e71["TLI_E72"][t] - e71["TLI_C0"][t] for t in tasks])

# ---------- 画布 ----------
fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.6), gridspec_kw={"width_ratios": [1, 1.35]})

# (a) RULER 长度梯度
ax = axes[0]
x = np.arange(3)
ax.plot(x, full_kv, "s-", color=OI["gray"], lw=1.6, ms=5, label="FullKV")
ax.plot(x, psi_c0, "^--", color=OI["orange"], lw=1.6, ms=5.5, label="PSI single-pool")
ax.plot(x, psi_b7s, "o-", color=OI["blue"], lw=1.9, ms=5.5, label="PSI partitioned")
for i, (a, b) in enumerate(zip(psi_c0, psi_b7s)):
    ax.annotate(f"{b - a:+.2f}", (i, (a + b) / 2), textcoords="offset points",
                xytext=(14, 0), fontsize=7.4, ha="left",
                color=OI["blue"] if b >= a else OI["orange"])
ax.set_xticks(x)
ax.set_xticklabels(["4K", "8K", "16K"], fontsize=8.5)
ax.set_xlabel("RULER context length", fontsize=8.5)
ax.set_ylabel("11-task average score", fontsize=8.5)
ax.set_ylim(80, 93.5)
ax.set_title("(a) partition gain grows with length\nfar-retrieval benefit zone = 16K",
             fontsize=8.6)
ax.legend(fontsize=7.2, frameon=False, loc="lower left")
ax.tick_params(labelsize=7.5)

# (b) LongBench per-task Δ
ax = axes[1]
order = np.argsort(delta)[::-1]
dl = delta[order]
tl = [tasks[i] for i in order]
colors = [OI["blue"] if v >= 0 else OI["orange"] for v in dl]
ax.bar(np.arange(len(dl)), dl, 0.62, color=colors)
ax.axhline(0, color="#555555", lw=0.8)
for i, v in enumerate(dl):
    if abs(v) >= 1.0:  # 只标显著项，避免拥挤
        ax.annotate(f"{v:+.2f}", (i, v), textcoords="offset points",
                    xytext=(0, 5 if v >= 0 else -10), fontsize=6.6,
                    ha="center", color="#333333")
ax.set_xticks(np.arange(len(tl)))
ax.set_xticklabels([t[:11] for t in tl], rotation=42, ha="right", fontsize=6.4)
ax.set_ylabel("PSI $-$ single-pool (per task)", fontsize=8.5)
ax.set_title("(b) LongBench per-task deltas\n13 tasks, aggregate +0.38, musique +5.02 dominates",
             fontsize=8.6)
ax.tick_params(labelsize=7.5)

fig.tight_layout(w_pad=2.4)
for out in OUTS:
    fig.savefig(f"{out}/fig4_partition_gain.pdf")
    fig.savefig(f"{out}/fig4_partition_gain.png")
print("saved fig4_partition_gain")
print("RULER lens:", dict(zip(LENS, zip(full_kv, psi_c0, psi_b7s))))
print("LB delta sorted:", {t: round(v, 2) for t, v in zip(tl, dl)})
