# E72 快筛五臂判决可视化：trace mass vs e2e F1 排名反转（口径鸿沟核心证据图）
# 输出：/home/wangyuanshuo02/sglang/ref/figs/e72_trace_vs_e2e.{png,pdf}
# 左 panel：trace mass（E64j，B_TOK=2048 宽预算）条形图；右 panel：e2e F1
# 快筛（hotpotqa/musique 均值，严格口径 K2=1024）——同色配对显示排名反转。
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (臂名, trace mass, e2e hotpotqa, e2e musique, 颜色)
ARMS = [
    ("mminmax\n(minmax,minmax)", 0.9116, 53.27, 34.14, "tab:blue"),
    ("cavg\n(cluster,avg)",      0.9033, 53.43, 32.31, "tab:cyan"),
    ("mavg\n(minmax,avg)",       0.9012, 54.43, 34.76, "tab:green"),
    ("aavg\n(avg,avg)",          0.8036, 54.72, 32.82, "tab:orange"),
    ("mavg(B7s)\nβ=0.25 ref",    0.9166, 54.23, 32.82, "tab:gray"),  # mono trace 口径
]
# e2e 均值（快筛 screen_avg）
E2E = {a[0]: round((a[2] + a[3]) / 2, 2) for a in ARMS}
# trace 排名（含 mono 参照 0.9166 排第 1）
TRACE_RANK = {"mminmax\n(minmax,minmax)": 2, "cavg\n(cluster,avg)": 3,
              "mavg\n(minmax,avg)": 4, "aavg\n(avg,avg)": 5, "mavg(B7s)\nβ=0.25 ref": 1}
E2E_RANK = {k: r for r, (k, _) in enumerate(
    sorted(E2E.items(), key=lambda x: -x[1]), 1)}

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
names = [a[0] for a in ARMS]
# 左：trace mass
vals = [a[1] for a in ARMS]
bars = axes[0].bar(range(len(ARMS)), vals,
                   color=[a[4] for a in ARMS], alpha=0.85, edgecolor="black", linewidth=0.6)
for i, (v, a) in enumerate(zip(vals, ARMS)):
    axes[0].text(i, v + 0.004, f"{v:.4f}\n(rank {TRACE_RANK[a[0]]})",
                 ha="center", va="bottom", fontsize=9)
axes[0].set_ylim(0.78, 0.95)
axes[0].set_ylabel("trace mass coverage (E64j, B_TOK=2048)")
axes[0].set_title("Trace-mass ranking (wide budget, no-sink-dual-count)")
axes[0].set_xticks(range(len(ARMS))); axes[0].set_xticklabels(names, fontsize=8.5)

# 右：e2e F1 快筛均值
vals2 = [E2E[a[0]] for a in ARMS]
bars2 = axes[1].bar(range(len(ARMS)), vals2,
                    color=[a[4] for a in ARMS], alpha=0.85, edgecolor="black", linewidth=0.6)
for i, (v, a) in enumerate(zip(vals2, ARMS)):
    axes[1].text(i, v + 0.06, f"{v:.2f}\n(rank {E2E_RANK[a[0]]})",
                 ha="center", va="bottom", fontsize=9)
axes[1].set_ylim(42.0, 45.6)
axes[1].set_ylabel("e2e F1 screen avg (hotpotqa + musique)")
axes[1].set_title("e2e F1 ranking (strict budget K2=1024, sink/swa granted)")
axes[1].set_xticks(range(len(ARMS))); axes[1].set_xticklabels(names, fontsize=8.5)

fig.suptitle("E72: trace-mass ranking inverts under e2e F1 (budget-caliber gap)",
             fontsize=13, y=0.99)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"/home/wangyuanshuo02/sglang/ref/figs/e72_trace_vs_e2e.{ext}", dpi=200,
                bbox_inches="tight")
print("saved -> /home/wangyuanshuo02/sglang/ref/figs/e72_trace_vs_e2e.{png,pdf}")
# 打印配对排名表
print("\narm                trace_rank  e2e_rank  delta")
for a in ARMS:
    n = a[0]
    print(f"{n.replace(chr(10), ' '):24s} {TRACE_RANK[n]:>5d} {E2E_RANK[n]:>8d} "
          f"{E2E_RANK[n] - TRACE_RANK[n]:>6d}")
