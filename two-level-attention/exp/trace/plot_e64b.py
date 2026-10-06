# E64b 可视化：bp × gamma 细扫（冠军法 mavg 为主 + cavg 对照）
# 左图：cov vs bp（线=gamma，虚线=mono 同 bp 对照）；右图：cov vs gamma（分组=bp）
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/home/wangyuanshuo02/two-level-attention/exp/trace/results"
F = "/home/wangyuanshuo02/two-level-attention/exp/figures"
d = json.load(open(f"{R}/e64b_bp_gamma.json"))
BPS = [16, 32, 64, 128, 256]
GAMMAS = [0.5, 0.75, 1.0]
GCOL = {0.5: "tab:red", 0.75: "tab:orange", 1.0: "tab:blue"}


def arm(k):
    vals = [d[s][k] for s in d if k in d[s]]
    return sum(vals) / len(vals) if vals else None


fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
# 左：cov vs bp（mavg 各 gamma + mono）
for g in GAMMAS:
    ys = [arm(f"mavg_bp{bp}_g{g}") for bp in BPS]
    axes[0].plot(BPS, ys, marker="o", color=GCOL[g], lw=2, label=f"mavg gamma={g}")
mono_ys = [arm(f"mono_bp{bp}") for bp in BPS]
axes[0].plot(BPS, mono_ys, marker="^", color="k", ls="--", lw=1.8, label="mono (single pool)")
axes[0].set_xscale("log", base=2)
axes[0].set_xticks(BPS, [str(b) for b in BPS])
axes[0].set_xlabel("budget_page_topk (bp)")
axes[0].set_ylabel("mass coverage")
axes[0].grid(alpha=0.3)
axes[0].legend(fontsize=9)
axes[0].set_title("E64b: coverage vs page budget (mavg, alpha=0.125 beta=0.25)", fontsize=10)
# 右：cov vs gamma（分 bp 组，mavg 实线 cavg 虚线）
for i, bp in enumerate(BPS):
    xoff = [g + (i - 2) * 0.02 for g in GAMMAS]
    ys = [arm(f"mavg_bp{bp}_g{g}") for g in GAMMAS]
    axes[1].plot(xoff, ys, marker="o", lw=1.8, label=f"mavg bp={bp}")
ys = [arm(f"cavg_bp128_g{g}") for g in GAMMAS]
axes[1].plot([g + 0.05 for g in GAMMAS], ys, marker="s", ls=":", color="gray", lw=1.5, label="cavg bp=128")
axes[1].set_xticks(GAMMAS)
axes[1].set_xlabel("gamma (near fine-selection discount)")
axes[1].set_ylabel("mass coverage")
axes[1].grid(alpha=0.3)
axes[1].legend(fontsize=8)
axes[1].set_title("E64b: coverage vs gamma (gamma effect grows with bp)", fontsize=10)
fig.tight_layout()
fig.savefig(f"{F}/e64b_bp_gamma.png", dpi=150)
fig.savefig(f"{F}/e64b_bp_gamma.pdf")
print("saved e64b_bp_gamma.{png,pdf}")
