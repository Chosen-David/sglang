# E64g α×β 9×9 全网格可视化（用户要求：9*9 数据 + 合适的可视化图找最优配比）
# 输出：figs/e64g_ab_grid.{png,pdf}（3 方法族热力图 + 无意义角点标注 + 冠军标记）
#       + e64g_ab_grid_table.md（16 样本均值表 + 各数据集最优配比）
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = "/home/wangyuanshuo02/two-level-attention/exp/trace/results"
OUT = "/home/wangyuanshuo02/sglang/ref/figs"
d = json.load(open(f"{R}/e64g_full_grid.json"))
GRID = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
METHODS = [("mavg", "mavg (far=minmax, near=avg)"),
           ("cavg", "cavg (far=cluster, near=avg)"),
           ("aavg", "aavg (far=avg,   near=avg)")]
MEANINGLESS = {(1.0, 0.0), (0.0, 1.0)}   # 用户定义：负责区无选择权利


def grid_of(meth):
    g = np.full((9, 9), np.nan)
    for name, rec in d.items():
        for a_i, a in enumerate(GRID):
            for b_i, b in enumerate(GRID):
                k = f"{meth}_a{a}_b{b}"
                if k in rec:
                    g[a_i, b_i] = np.nansum([g[a_i, b_i], rec[k]], axis=0) if not np.isnan(g[a_i, b_i]) else rec[k]
    # 除以样本数
    cnt = len(d)
    return g / cnt


# 正确聚合：逐样本累加再除
def grid_of2(meth):
    g = np.zeros((9, 9)); n = np.zeros((9, 9))
    for name, rec in d.items():
        for a_i, a in enumerate(GRID):
            for b_i, b in enumerate(GRID):
                k = f"{meth}_a{a}_b{b}"
                if k in rec:
                    g[a_i, b_i] += rec[k]; n[a_i, b_i] += 1
    return np.where(n > 0, g / np.maximum(n, 1), np.nan)


plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6))
table = {}
for ax, (meth, title) in zip(axes, METHODS):
    g = grid_of2(meth)
    # 无意义角点置 NaN（视觉上区分）
    for (a, b) in MEANINGLESS:
        g[GRID.index(a), GRID.index(b)] = np.nan
    im = ax.imshow(g.T, origin="lower", cmap="viridis", vmin=0.55, vmax=0.90,
                   extent=(-0.06, 1.06, -0.06, 1.06), aspect="auto")
    # 数值标注
    for a_i in range(9):
        for b_i in range(9):
            if not np.isnan(g[a_i, b_i]):
                ax.text(GRID[a_i], GRID[b_i], f"{g[a_i, b_i]:.3f}"[:5],
                        ha="center", va="center", fontsize=6.2,
                        color="white" if g[a_i, b_i] < 0.78 else "black")
    # 冠军标记
    valid = np.where(~np.isnan(g), g, -np.inf)
    ai, bi = np.unravel_index(valid.argmax(), g.shape)
    ax.plot(GRID[ai], GRID[bi], marker="*", ms=18, color="red", mec="white", mew=1.2)
    ax.set_title(f"{title}\nbest α={GRID[ai]:.3f} β={GRID[bi]:.3f}  cov={g[ai, bi]:.4f}", fontsize=10)
    ax.set_xlabel("α (near fraction of mid)")
    ax.set_ylabel("β (near share of page budget)")
    ax.set_xticks(GRID); ax.set_xticklabels([f"{x:g}" for x in GRID], fontsize=7, rotation=45)
    ax.set_yticks(GRID); ax.set_yticklabels([f"{x:g}" for x in GRID], fontsize=7)
    # 无意义角点文字
    ax.text(1.0, 0.0, "∅", ha="center", va="center", fontsize=20, color="red", alpha=0.85)
    ax.text(0.0, 1.0, "∅", ha="center", va="center", fontsize=20, color="red", alpha=0.85)
    plt.colorbar(im, ax=ax, fraction=0.046)
    table[meth] = g

fig.suptitle("E64g α×β full grid (9×9, step 0.125) — mass coverage, mean over 16 samples; "
             "ab(1,1)=all-near, ab(0,0)=all-far, ∅=meaningless corners (owner without selection right)",
             fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(f"{OUT}/e64g_ab_grid.png", dpi=170)
fig.savefig(f"{OUT}/e64g_ab_grid.pdf")
print("saved:", OUT)

# ---- markdown 表 ----
lines = ["# E64g α×β 9×9 full grid (mean over 16 samples)", "",
         "rows=α (near fraction), cols=β (near budget share); ∅=meaningless corners", ""]
for meth, _ in METHODS:
    g = table[meth]
    lines.append(f"## {meth}")
    lines.append("| α\\β | " + " | ".join(f"{b:g}" for b in GRID) + " |")
    lines.append("|---|" + "---|" * 9)
    for a_i, a in enumerate(GRID):
        row = []
        for b_i, b in enumerate(GRID):
            v = g[a_i, b_i]
            row.append("∅" if np.isnan(v) else f"{v:.3f}")
        lines.append(f"| {a:g} | " + " | ".join(row) + " |")
    valid = np.where(~np.isnan(g), g, -np.inf)
    ai, bi = np.unravel_index(valid.argmax(), g.shape)
    lines.append(f"\n**best: α={GRID[ai]:g}, β={GRID[bi]:g}, cov={g[ai, bi]:.4f}**\n")
# 各数据集最优配比（Top-5 样本展示 + 全量附录）
lines.append("## per-dataset best (α,β) per method")
lines.append("| 样本 | mavg best | cavg best | aavg best |")
lines.append("|---|---|---|---|")
for name in d:
    row = []
    for meth, _ in METHODS:
        best, ba, bb = -1, None, None
        for a in GRID:
            for b in GRID:
                if (a, b) in MEANINGLESS:
                    continue
                v = d[name].get(f"{meth}_a{a}_b{b}", -1)
                if v > best:
                    best, ba, bb = v, a, b
        row.append(f"({ba:g},{bb:g})={best:.3f}")
    lines.append(f"| {name} | " + " | ".join(row) + " |")
open(f"{OUT}/e64g_ab_grid_table.md", "w").write("\n".join(lines) + "\n")
print("table saved")
# 快速终端摘要
for meth, _ in METHODS:
    g = table[meth]
    valid = np.where(~np.isnan(g), g, -np.inf)
    ai, bi = np.unravel_index(valid.argmax(), g.shape)
    print(f"{meth}: best a={GRID[ai]:g} b={GRID[bi]:g} cov={g[ai, bi]:.4f}")
