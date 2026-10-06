# E64g 可视化：α×β 全网格（9×9，步长 0.125，含角点）
# 图1：3 method 均值热图（16 样本）
# 图2：mavg 每数据集热图（4×4 panel）——用户要的「不同数据集下的精度数据」
# 图3：各数据集最优臂（内部 vs 边界）对比条形
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = "/home/wangyuanshuo02/two-level-attention/exp/trace/results"
F = "/home/wangyuanshuo02/two-level-attention/exp/figures"
d = json.load(open(f"{R}/e64g_full_grid.json"))
GRID = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
METHODS = ["mavg", "cavg", "aavg"]


def grid_of(meth, sample=None):
    src = {sample: d[sample]} if sample else d
    g = np.zeros((9, 9))
    for i, a in enumerate(GRID):
        for j, b in enumerate(GRID):
            k = f"{meth}_a{a}_b{b}"
            vals = [src[s][k] for s in src if k in src[s]]
            g[i, j] = sum(vals) / len(vals) if vals else np.nan
    return g


# 图1：3 method 均值热图
fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
for ax, meth in zip(axes, METHODS):
    g = grid_of(meth)
    im = ax.imshow(g, cmap="viridis", vmin=0.66, vmax=0.90, aspect="auto")
    for i in range(9):
        for j in range(9):
            ax.text(j, i, f"{g[i,j]:.3f}", ha="center", va="center", fontsize=5.5,
                    color="w" if g[i, j] < 0.8 else "k")
    ax.set_xticks(range(9), [f"{b:g}" for b in GRID], fontsize=7)
    ax.set_yticks(range(9), [f"{a:g}" for a in GRID], fontsize=7)
    ax.set_xlabel("beta (near budget share)")
    ax.set_ylabel("alpha (near region share)")
    ax.set_title(f"{meth} (mean, n={len(d)})", fontsize=11)
fig.colorbar(im, ax=axes, shrink=0.85, label="mass coverage")
fig.suptitle("E64g full alpha-beta grid: corners (0,0)=all-far / (1,1)=all-near / (1,0)&(0,1)=meaningless (sink+swa only)", fontsize=10)
fig.savefig(f"{F}/e64g_grid_heatmap.png", dpi=150, bbox_inches="tight")
fig.savefig(f"{F}/e64g_grid_heatmap.pdf", bbox_inches="tight")
plt.close(fig)

# 图2：mavg 每数据集热图（4×4）
names = sorted(d.keys())
fig, axes = plt.subplots(4, 4, figsize=(15, 15))
for ax, name in zip(axes.flat, names):
    g = grid_of("mavg", name)
    ax.imshow(g, cmap="viridis", vmin=0.5, vmax=0.95, aspect="auto")
    ax.set_xticks(range(9), [f"{b:g}" for b in GRID], fontsize=5)
    ax.set_yticks(range(9), [f"{a:g}" for a in GRID], fontsize=5)
    ax.set_title(name, fontsize=8)
fig.suptitle("E64g mavg per-dataset alpha-beta grid (16 samples)", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(f"{F}/e64g_per_dataset_mavg.png", dpi=130)
fig.savefig(f"{F}/e64g_per_dataset_mavg.pdf")
plt.close(fig)

# 图3：各数据集最优臂（内部 vs α=0 边界）
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(names))
bnd = [d[s]["mavg_a0.0_b0.0"] for s in names]      # α=0 边界（全 far）
intr = []
for s in names:
    best = max((v, k) for k, v in d[s].items() if k.startswith("mavg_") and 0 < float(k.split("_a")[1].split("_b")[0]) < 1)
    intr.append(best[0])
ax.bar(x - 0.18, bnd, 0.36, label="boundary alpha=0 (all-far)")
ax.bar(x + 0.18, intr, 0.36, label="best interior arm (0<alpha<1)")
ax.set_xticks(x, [n.replace("lb_", "").replace("_en", "") for n in names], rotation=40, ha="right", fontsize=8)
ax.set_ylabel("mass coverage")
ax.set_ylim(0.7, 1.0)
ax.legend()
ax.set_title("E64g: per-dataset boundary (all-far) vs best interior split (mavg)", fontsize=10)
fig.tight_layout()
fig.savefig(f"{F}/e64g_boundary_vs_interior.png", dpi=150)
fig.savefig(f"{F}/e64g_boundary_vs_interior.pdf")
plt.close(fig)

# 文本表：每数据集最优内部臂 + 边界
print("sample | boundary(0,0) | best-interior arm")
for s in names:
    best = max((v, k) for k, v in d[s].items() if k.startswith("mavg_") and 0 < float(k.split("_a")[1].split("_b")[0]) < 1)
    print(f"  {s:28s} {d[s]['mavg_a0.0_b0.0']:.4f} | {best[1]} {best[0]:.4f} (Δ={best[0]-d[s]['mavg_a0.0_b0.0']:+.4f})")
print("saved e64g_{grid_heatmap, per_dataset_mavg, boundary_vs_interior}")
