# E65 可视化：降维方法族消融图（4 panel：A far 粗筛 / C 细筛 / D 全链 / B near）
# 横轴 d ∈ {4,8,16,32}，纵轴 far/near mass 捕获率（7 泛化样本均值），线=方法
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/home/wangyuanshuo02/two-level-attention/exp/trace/results"
e65 = json.load(open(f"{R}/e65_dim_reduction.json"))
e65b = json.load(open(f"{R}/e65b_sup_proj.json"))
SAMPLES = list(e65b.keys())          # 7 泛化样本（不含校准集 hotpotqa_0）


def agg(data, prefix):
    out = {}
    for name in SAMPLES:
        if name not in data:
            continue
        for k, v in data[name].items():
            if k.startswith(prefix):
                out.setdefault(k, []).append(v)
    return {k: sum(v) / len(v) for k, v in out.items()}


# 方法 → (数据源, 臂前缀, 颜色, 标记)
METHODS = [
    ("sup_wsvd", e65b, "tab:red", "o"),
    ("sup_grad", e65b, "tab:orange", "s"),
    ("trunc", e65, "tab:blue", "^"),
    ("pca_sub", e65, "tab:green", "D"),
    ("jl_sub", e65, "tab:purple", "v"),
]
DIMS = [4, 8, 16, 32]
PANELS = [("A_far_coarse", "A: far coarse (fine fixed at d32)"),
          ("C_fine", "C: fine (coarse fixed at d32)"),
          ("D_full", "D: full pipeline (coarse+fine same d)"),
          ("B_near_coarse", "B: near coarse")]

fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
for ax, (arm, title) in zip(axes.flat, PANELS):
    for meth, data, color, marker in METHODS:
        a = agg(data, f"{arm}_{meth}_")
        ys = [a.get(f"{arm}_{meth}_d{d}") for d in DIMS]
        xs = [d for d, y in zip(DIMS, ys) if y is not None]
        yy = [y for y in ys if y is not None]
        ax.plot(xs, yy, color=color, marker=marker, label=meth, lw=1.8)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(DIMS)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.0)
axes[0, 0].set_ylabel("far mass capture")
axes[1, 0].set_ylabel("mass capture")
axes[1, 0].set_xlabel("feature dim d")
axes[1, 1].set_xlabel("feature dim d")
axes[0, 0].legend(fontsize=8, loc="lower right")
fig.suptitle("E65/E65b dimension-reduction ablation (Qwen3-8B, mean of 7 held-out samples; calib=hotpotqa_0)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(f"{R}/../../figures/e65_dim_reduction.png", dpi=150)
fig.savefig(f"{R}/../../figures/e65_dim_reduction.pdf")
print("saved e65_dim_reduction.{png,pdf}")
