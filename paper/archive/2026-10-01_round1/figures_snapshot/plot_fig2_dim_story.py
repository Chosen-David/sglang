# Fig2: 子空间与降维故事（终版规划图，贡献 #1 证据）
# panel A: 旋转对三判决（E85b）—— 配对数饱和 / 频率单调 / 配对完整性
#   （重画：旧版画 E73 两段互补 bar，与论文 caption 承诺的三判决不符）
# panel B: 8 样本逐点 paired —— tail32 vs MLA 式共享 SVD d16 追平、per-head PCA 落后（E76）
# panel C: 降维方法族均值 bar（tail32 / shared d8/d16/d32 / per-head d16 / UMAP d16 n=2 / oracle）
# house style: okabe_ito 色盲友好 + 无多余网格 + PDF/PNG 双格式 + 图内英文 + 无内部代号
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTS = ["/home/wangyuanshuo02/sglang/paper/figures",
        "/home/wangyuanshuo02/two-level-attention/exp/figures"]
R = "/home/wangyuanshuo02/two-level-attention/exp/trace/results"
e85b = json.load(open(f"{R}/e85b_pair_mechanism.json"))["mean"]
e76 = json.load(open(f"{R}/e76_nonlinear_reduce.json"))["quality"]
e73b = json.load(open(f"{R}/e73b_seg_complementarity.json"))

OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
      "red": "#D55E00", "purple": "#CC79A7", "gray": "#7F7F7F"}
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight",
})

samples = [s for s in e76 if s in e73b]
disp = {"lb_gov_report_0": "gov", "lb_hotpotqa_0": "hotpot", "lb_musique_0": "musique",
        "lb_narrativeqa_0": "narrqa", "lb_passage_retrieval_en_0": "passage",
        "lb_qasper_0": "qasper", "needle32k": "needle", "natural32k": "natural"}

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))

# ---- panel A: 旋转对三判决 ----
ax = axes[0]
m = e85b
bars = [
    # (label, value, color) —— 三组：配对数饱和 / 频率单调 / 配对完整性
    ("8", m["pair8"], "#a6d8c8"),
    ("16", m["pair16"], OI["green"]),
    ("24", m["pair24"], "#a6d8c8"),
    ("32", m["pair32"], "#a6d8c8"),
    ("low", m["pair16"], OI["blue"]),
    ("mid", m["pair16_midfreq"], OI["orange"]),
    ("high", m["pair16_hifreq"], OI["red"]),
    ("aligned", m["pair16"], OI["green"]),
    ("mis-\naligned", m["mismatch_lo_mid"], OI["purple"]),
]
xs = np.arange(len(bars))
vals = [b[1] for b in bars]
cols = [b[2] for b in bars]
ax.bar(xs, vals, color=cols, width=0.62, edgecolor="white")
for x, v in zip(xs, vals):
    ax.text(x, v + 0.015, f"{v:.3f}", ha="center", fontsize=6.8)
ax.set_xticks(xs)
ax.set_xticklabels([b[0] for b in bars], fontsize=7.0)
# 三组分隔与组标签
for gx in (3.5, 6.5):
    ax.axvline(gx, color="#cccccc", lw=0.8, ls="--")
ax.text(1.5, 1.06, "pair-count saturation", ha="center", fontsize=7.2, color="#444444")
ax.text(5.0, 1.06, "rotation frequency", ha="center", fontsize=7.2, color="#444444")
ax.text(7.5, 1.06, "pairing integrity", ha="center", fontsize=7.2, color="#444444")
ax.set_ylim(0, 1.14)
ax.set_ylabel("far-region mass capture")
ax.set_xlabel("16-pair selection, by count / frequency / alignment", fontsize=7.2, labelpad=1)
ax.set_title("(a) tail32 = 16 lowest-frequency rotation pairs", fontsize=9.5, fontweight="bold")

# ---- panel B: 8 样本逐点追平 ----
ax = axes[1]
x = np.arange(len(samples))
t32 = [e76[s]["C_tail32_ref"] for s in samples]
sh16 = [e76[s]["C_pca_shared_d16"] for s in samples]
ph16 = [e76[s]["C_pca_perhead_d16"] for s in samples]
ax.plot(x, t32, "o", ms=6, color=OI["green"], label="tail32 (default)", zorder=3)
ax.plot(x, sh16, "x", ms=7, mew=2, color=OI["blue"], label="shared SVD d16 (MLA-style)", zorder=3)
ax.plot(x, ph16, "^", ms=6, color=OI["orange"], label="per-head PCA d16", zorder=3)
ax.set_xticks(x)
ax.set_xticklabels([disp.get(s, s[:8]) for s in samples], rotation=38, ha="right", fontsize=7.5)
ax.set_ylabel("far-region mass capture")
ax.set_ylim(0.55, 0.95)
ax.legend(fontsize=7, frameon=False, loc="lower right")
ax.set_title("(b) Shared SVD basis matches tail32 on 8/8", fontsize=9.5, fontweight="bold")

# ---- panel C: 方法族均值 ----
ax = axes[2]
arms = ["C_tail32_ref", "C_pca_shared_d16", "C_pca_shared_d8", "C_pca_perhead_d16", "C_umap_d16", "oracle128"]
labels = ["tail32\n(d=32)", "shared\nSVD d16", "shared\nSVD d8", "per-head\nPCA d16", "UMAP\nd16 (n=2)", "oracle\n(d=128)"]
means = [np.mean([e76[s][a] for s in samples if a in e76[s]]) for a in arms]
errs = [np.std([e76[s][a] for s in samples if a in e76[s]]) for a in arms]
cols = [OI["green"], OI["blue"], "#56B4E9", OI["orange"], OI["purple"], OI["gray"]]
ax.bar(range(len(arms)), means, yerr=errs, color=cols, width=0.62,
       edgecolor="white", capsize=3, error_kw={"lw": 1})
for i, (v, n) in enumerate(zip(means, [8, 8, 8, 8, 2, 8])):
    ax.text(i, v + 0.035, f"{v:.3f}", ha="center", fontsize=7.8)
ax.axhline(means[0], color=OI["green"], ls=":", lw=1, alpha=0.6)
ax.set_xticks(range(len(arms)))
ax.set_xticklabels(labels, fontsize=7.3)
ax.set_ylim(0, 1.12)
ax.set_ylabel("far-region mass capture (mean±std)")
ax.set_title("(c) Linear > nonlinear; shared > per-head", fontsize=9.5, fontweight="bold")

fig.suptitle("Dimensionality-reduction freedom comes from position priors, not learning",
             fontsize=10, fontweight="bold", y=1.03)
for out in OUTS:
    fig.savefig(f"{out}/fig2_dim_reduction_story.pdf")
    fig.savefig(f"{out}/fig2_dim_reduction_story.png")
print("saved fig2_dim_reduction_story (panel a redrawn)")
