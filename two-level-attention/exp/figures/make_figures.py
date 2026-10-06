# 论文级实验图绘制（TLI 论文 Figure 套件）
# 数据源：two-level-attention/exp/trace/results/*.json（全部真实 trace 实测）
# 输出：/home/wangyuanshuo02/two-level-attention/exp/figures/*.pdf + *.png
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = "/home/wangyuanshuo02/two-level-attention/exp/trace/results/"
OUT = "/home/wangyuanshuo02/two-level-attention/exp/figures"
os.makedirs(OUT, exist_ok=True)

# 论文风格
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 150, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
})
C = {"blue": "#2f6f9f", "red": "#c1443c", "green": "#3a7d44", "orange": "#d98e32",
     "purple": "#7b5aa6", "gray": "#8a8a8a", "teal": "#2a9d8f"}

TRACE_SHORT = {
    "lb_gov_report": "gov", "lb_hotpotqa": "hpqa", "lb_narrativeqa": "nqa",
    "lb_passage_retrieval_en": "pse", "needle32k": "needle", "natural32k": "natural",
}


def tname(k):
    for pre, s in TRACE_SHORT.items():
        if k.startswith(pre):
            return s + ("0" if k.endswith("_0") else "1")
    return k[:6]


# ================= Fig 1: H1 位置质量三层分解（10 trace 堆叠 bar） =================
def fig_h1():
    d = json.load(open(R + "h1_full_decomposition.json"))
    names = sorted([k for k in d if "sink_mean" in d[k]], key=tname)
    sink = np.array([d[k]["sink_mean"] for k in names])
    near = np.array([d[k]["near_mean"] for k in names])
    far = np.array([d[k]["far_mean"] for k in names])
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    x = np.arange(len(names))
    ax.bar(x, sink, 0.62, label="sink (first 64 tok)", color=C["blue"])
    ax.bar(x, near, 0.62, bottom=sink, label="near (last 2048 tok)", color=C["teal"])
    ax.bar(x, sink + near, 0.62, bottom=far * 0 + sink + near, label="far (middle)", color=C["orange"])
    ax.set_xticks(x)
    ax.set_xticklabels([tname(k) for k in names], rotation=45, ha="right")
    ax.set_ylabel("attention mass fraction")
    ax.set_title("H1: position-dependent mass decomposition (Qwen3-8B, 10 traces)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(0, 1.05)
    fig.savefig(f"{OUT}/fig1_h1_decomposition.pdf")
    fig.savefig(f"{OUT}/fig1_h1_decomposition.png")
    plt.close(fig)


# ================= Fig 2: E3 子空间 recall（A 创新点） =================
def fig_e3():
    d = json.load(open(R + "e3_subspace_recall_v2.json"))
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))
    for ax, trace in zip(axes, ["natural32k", "needle32k"]):
        cfgs = ["blk32_d32_lowfreq", "blk32_d32_random", "blk32_d32_highfreq", "blk32_d128_lowfreq"]
        labels = ["low-freq\nd'=32", "random\nd'=32", "high-freq\nd'=32", "low-freq\nd'=128"]
        cols = [C["blue"], C["gray"], C["red"], C["green"]]
        mass = [d[trace][c]["mass_recall"] for c in cfgs]
        entry = [d[trace][c]["entry_recall"] for c in cfgs]
        x = np.arange(4)
        ax.bar(x - 0.19, entry, 0.36, label="entry recall", color=cols, alpha=0.55)
        ax.bar(x + 0.19, mass, 0.36, label="mass recall", color=cols)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=28, ha="right")
        ax.set_title(f"{tname(trace)}")
        ax.set_ylim(0, 0.85)
        ax.set_ylabel("recall vs full-dim")
    axes[0].legend(loc="upper right")
    fig.suptitle("E3: subspace selection for L1 block scores (A)", y=1.04)
    fig.savefig(f"{OUT}/fig2_e3_subspace.pdf")
    fig.savefig(f"{OUT}/fig2_e3_subspace.png")
    plt.close(fig)


# ================= Fig 3: E4c 严格预算策略对比（B 重定位依据） =================
def fig_e4c():
    d = json.load(open(R + "e4c_strict_budget.json"))
    names = sorted(d.keys(), key=tname)
    strats = ["minmax_blk", "km_blk", "km_tok", "oracle"]
    labels = ["minmax block\n(TIA L1)", "kmeans block\n(scatter-amax)", "kmeans token\n(cluster score)", "oracle\n(exact topk)"]
    cols = [C["blue"], C["red"], C["orange"], C["green"]]
    budgets = ["512", "1024", "2048"]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.7), sharey=True)
    for ax, b in zip(axes, budgets):
        x = np.arange(4)
        vals = []
        for s in strats:
            v = [d[n][s][b] for n in names if d[n][s][b] is not None]
            vals.append(np.mean(v))
        bars = ax.bar(x, vals, 0.6, color=cols)
        for r, v in zip(bars, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6.5, rotation=0)
        ax.set_title(f"budget = {b} tok/head")
        ax.set_ylim(0, 1.08)
    axes[0].set_ylabel("far-region mass captured")
    fig.suptitle("E4c: far-region candidate quality under strict token budget (10 traces × 36 layers)", y=1.04)
    fig.savefig(f"{OUT}/fig3_e4c_strict_budget.pdf")
    fig.savefig(f"{OUT}/fig3_e4c_strict_budget.png")
    plt.close(fig)


# ================= Fig 4: E6 层跳过掩码 + far mass 轮廓（D'） =================
def fig_e6():
    d = json.load(open(R + "e6_layer_skip.json"))
    mask = json.load(open(R + "tli_layer_skip_mask.json"))
    per = d["per_prompt"]
    fig, ax = plt.subplots(figsize=(4.8, 2.6))
    layers = np.arange(36)
    # 每层跨 prompt 平均 far mass
    avg = np.zeros(36)
    cnt = np.zeros(36)
    for trace, info in per.items():
        prof = info.get("layer_profile")
        if prof is None:
            continue
        for i, v in enumerate(prof):
            if v is not None:
                avg[i] += v
                cnt[i] += 1
    avg = avg / np.maximum(cnt, 1)
    ax.bar(layers, avg, 0.7, color=np.where(np.isin(layers, mask["skip"]), C["red"], C["blue"]))
    ax.axhline(mask["threshold"], color=C["gray"], linestyle="--", linewidth=1)
    ax.text(35.5, mask["threshold"] + 0.004, f"τ = {mask['threshold']}", ha="right", fontsize=7, color=C["gray"])
    ax.set_xlabel("layer index")
    ax.set_ylabel("far-region mass (avg over prompts)")
    ax.set_title("E6/D': static layer-skip mask (red = skip far retrieval, 13/36)")
    fig.savefig(f"{OUT}/fig4_e6_layer_skip.pdf")
    fig.savefig(f"{OUT}/fig4_e6_layer_skip.png")
    plt.close(fig)


# ================= Fig 5: E8-1 索引开销 + E8-2 kernel 原型 =================
def fig_e8():
    d = json.load(open(R + "e8_1_index_flop.json"))
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))
    # 左：E8-1 FLOP 分解
    skip = np.array([s["speedup"] for s in d if s["skip"]])
    noskip = np.array([s["speedup"] for s in d if not s["skip"]])
    cats = ["skip layers\n(13/36)", "non-skip\nlayers", "overall\nmean"]
    vals = [skip.mean(), noskip.mean(), (skip.sum() + noskip.sum()) / (len(skip) + len(noskip))]
    bars = axes[0].bar(cats, vals, 0.55, color=[C["red"], C["blue"], C["gray"]])
    for r, v in zip(bars, vals):
        axes[0].text(r.get_x() + r.get_width() / 2, v + 0.06, f"{v:.2f}×", ha="center", fontsize=8)
    axes[0].set_ylabel("indexer FLOP speedup vs TIA")
    axes[0].set_ylim(0, 5.6)
    axes[0].set_title("E8-1: index computation (A + D')")
    # 右：E8-2 kernel 延迟
    scen = ["L1 skip\n(D' far-removed)", "L1 non-skip\n(full blocks)", "L2 cascade\n(partition topk)"]
    eager = [0.123, 0.123, 1.897]
    tri = [0.034, 0.075, 1.165]
    x = np.arange(3)
    axes[1].bar(x - 0.17, eager, 0.34, label="eager PyTorch", color=C["gray"])
    axes[1].bar(x + 0.17, tri, 0.34, label="Triton fused", color=C["teal"])
    for i in range(3):
        axes[1].text(x[i] - 0.17, eager[i] + 0.02, f"{eager[i]:.2f}", ha="center", fontsize=7)
        axes[1].text(x[i] + 0.17, tri[i] + 0.02, f"{tri[i]:.2f}", ha="center", fontsize=7)
        axes[1].text(x[i] + 0.17, tri[i] + 0.13, f"{eager[i]/tri[i]:.2f}×", ha="center", fontsize=8, color=C["red"])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(scen, fontsize=7.5)
    axes[1].set_ylabel("latency (ms)")
    axes[1].set_title("E8-2: fused cascade kernels (S=128K)")
    axes[1].legend()
    fig.savefig(f"{OUT}/fig5_e8_speedup.pdf")
    fig.savefig(f"{OUT}/fig5_e8_speedup.png")
    plt.close(fig)


# ================= Fig 6: TLI vs TIA 逐层 mass 覆盖（E5b debug 数据） =================
def fig_tli_layers():
    # 从 debug log 提取（A+B+D' 全开 vs TIA）
    import re
    log = open("/tmp/tli_e5b_debug.log").read()
    sec = log.split("=== A+B+D' (far_blocks=16) ===")[1].split("=== A only ===")[0]
    rows = re.findall(r"L(\d+) far=([\d.]+) TIA=([\d.]+) TLI=([\d.]+)", sec)
    if not rows:
        print("fig6: debug log 不可用，跳过")
        return
    layers = np.array([int(r[0]) for r in rows])
    far = np.array([float(r[1]) for r in rows])
    tia = np.array([float(r[2]) for r in rows])
    tli = np.array([float(r[3]) for r in rows])
    mask = json.load(open(R + "tli_layer_skip_mask.json"))["skip"]
    fig, ax = plt.subplots(figsize=(5.2, 2.7))
    ax.plot(layers, tia, "-o", ms=3, color=C["gray"], label="TIA (baseline)")
    ax.plot(layers, tli, "-o", ms=3, color=C["blue"], label="TLI (A+B'+D')")
    for l in mask:
        if l in layers:
            ax.axvspan(l - 0.4, l + 0.4, color=C["red"], alpha=0.12)
    ax2 = ax.twinx()
    ax2.bar(layers, far, 0.55, color=C["orange"], alpha=0.35, label="far mass (right)")
    ax2.set_ylabel("far-region mass", color=C["orange"])
    ax2.tick_params(axis="y", colors=C["orange"])
    ax2.grid(False)
    ax.set_xlabel("layer")
    ax.set_ylabel("total mass coverage")
    ax.set_ylim(0.994, 1.0005)
    ax.legend(loc="lower right")
    ax.set_title("TLI vs TIA per-layer mass coverage (hotpotqa trace; red = D' skip)")
    fig.savefig(f"{OUT}/fig6_tli_vs_tia_layers.pdf")
    fig.savefig(f"{OUT}/fig6_tli_vs_tia_layers.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_h1()
    fig_e3()
    fig_e4c()
    fig_e6()
    fig_e8()
    fig_tli_layers()
    print("figures saved to", OUT)
    print(os.listdir(OUT))
