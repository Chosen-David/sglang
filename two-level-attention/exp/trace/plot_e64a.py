# E64a 可视化：method×alpha×beta 网格消融（用户指定图型）
# 图1：α-β 网格热图（每 method 一 panel，格=臂 cov，均值跨数据集）
# 图2：横轴 alpha 纵轴 cov，线=方法（均值）+ 各数据集细线（浅色）
# 图3：横轴 beta 纵轴 cov，线=方法（均值）
# 图4：mono/tli_anchor 对照水平线 + 三 method 最优臂
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = "/home/wangyuanshuo02/two-level-attention/exp/trace/results"
F = "/home/wangyuanshuo02/two-level-attention/exp/figures"
d = json.load(open(f"{R}/e64a_ab_grid.json"))
METHODS = ["cavg", "mavg", "aavg"]
ALPHAS = [0.125, 0.25, 0.5, 0.75]
BETAS = [0.125, 0.25, 0.5, 0.75]
MCOL = {"cavg": "tab:orange", "mavg": "tab:blue", "aavg": "tab:green"}

def arm_mean(meth, a, b):
    vals = [d[s][f"{meth}_a{a}_b{b}"] for s in d if f"{meth}_a{a}_b{b}" in d[s]]
    return sum(vals) / len(vals) if vals else None

# ---- 图1：3 panel 热图 ----
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, meth in zip(axes, METHODS):
    grid = np.array([[arm_mean(meth, a, b) or np.nan for b in BETAS] for a in ALPHAS])
    im = ax.imshow(grid, cmap="viridis", vmin=0.6, vmax=0.9, aspect="auto")
    for i in range(len(ALPHAS)):
        for j in range(len(BETAS)):
            ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", fontsize=8,
                    color="w" if grid[i, j] < 0.8 else "k")
    ax.set_xticks(range(len(BETAS)), [str(b) for b in BETAS])
    ax.set_yticks(range(len(ALPHAS)), [str(a) for a in ALPHAS])
    ax.set_xlabel("beta (near page budget share)")
    ax.set_ylabel("alpha (near region share)")
    ax.set_title(meth, fontsize=11)
mono = sum(d[s]["mono"] for s in d) / len(d)
tlia = sum(d[s]["tli_anchor"] for s in d) / len(d)
fig.colorbar(im, ax=axes, shrink=0.85, label="mass coverage")
fig.suptitle(f"E64a alpha-beta grid (mean over {len(d)} samples; mono={mono:.4f}, tli_anchor={tlia:.4f})", fontsize=12)
fig.savefig(f"{F}/e64a_grid_heatmap.png", dpi=150, bbox_inches="tight")
fig.savefig(f"{F}/e64a_grid_heatmap.pdf", bbox_inches="tight")
plt.close(fig)

# ---- 图2：alpha 曲线（线=方法均值 + 数据集细线）----
fig, ax = plt.subplots(figsize=(7.5, 5.5))
for meth in METHODS:
    ys = [arm_mean(meth, a, BETAS[1]) for a in ALPHAS]  # beta=0.25 固定
    ax.plot(ALPHAS, ys, color=MCOL[meth], marker="o", lw=2.5, label=f"{meth} (beta=0.25)")
    for s in list(d)[:6]:
        ys_s = [d[s].get(f"{meth}_a{a}_b0.25") for a in ALPHAS]
        ax.plot(ALPHAS, ys_s, color=MCOL[meth], alpha=0.18, lw=0.8)
ax.axhline(mono, color="k", ls="--", lw=1.2, label=f"mono (alpha=0)")
ax.axhline(tlia, color="gray", ls=":", lw=1.2, label="tli_anchor")
ax.set_xlabel("alpha (near region share of mid)")
ax.set_ylabel("mass coverage")
ax.grid(alpha=0.3)
ax.legend(fontsize=9)
ax.set_title("E64a: coverage vs alpha (thick=mean over samples, thin=per-sample; beta=0.25)", fontsize=10)
fig.tight_layout()
fig.savefig(f"{F}/e64a_alpha_curves.png", dpi=150)
fig.savefig(f"{F}/e64a_alpha_curves.pdf")
plt.close(fig)

# ---- 图3：beta 曲线 ----
fig, ax = plt.subplots(figsize=(7.5, 5.5))
for meth in METHODS:
    ys = [arm_mean(meth, ALPHAS[0], b) for b in BETAS]  # alpha=0.125 固定
    ax.plot(BETAS, ys, color=MCOL[meth], marker="s", lw=2.5, label=f"{meth} (alpha=0.125)")
ax.axhline(mono, color="k", ls="--", lw=1.2, label="mono (alpha=0)")
ax.set_xlabel("beta (near page budget share)")
ax.set_ylabel("mass coverage")
ax.grid(alpha=0.3)
ax.legend(fontsize=9)
ax.set_title("E64a: coverage vs beta (alpha=0.125)", fontsize=10)
fig.tight_layout()
fig.savefig(f"{F}/e64a_beta_curves.png", dpi=150)
fig.savefig(f"{F}/e64a_beta_curves.pdf")
plt.close(fig)
print("saved e64a_{grid_heatmap, alpha_curves, beta_curves}.{png,pdf}")
print(f"mono={mono:.4f} tli_anchor={tlia:.4f}")
for meth in METHODS:
    best = max((arm_mean(meth, a, b), a, b) for a in ALPHAS for b in BETAS)
    print(f"{meth} best: a={best[1]} b={best[2]} cov={best[0]:.4f}")
