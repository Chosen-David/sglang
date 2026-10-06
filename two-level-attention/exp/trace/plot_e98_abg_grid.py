# -*- coding: utf-8 -*-
# E98 α×β×γ 三维全网格 publication 图（3 张）
# 数据源：exp/trace/results/e98_abg_full_grid.json（16 样本、5 method 组合、3600 有效臂）
# 输出：sglang/ref/figs/fig8_e98_abg_heatmap / fig9_e98_gamma_sweep / fig10_e98_mono_vs_partition
#
# 臂语义（对照 analyze_e98_abg_full_grid.py / 论文indexer.md）：
#   α = near 区长度占 mid 比例；β = near 页池占 BP=64 比例；γ = near 细筛 token 折扣
#   角点退化：α=0 或 β=0 → far-only 单池；α=1 或 β=1 → near-only 单池（角点臂 = 单池臂）
#   约束过滤（灰格）：near_L < nb_near*BS（near 区装不下页池）或 far_budget > far_L
#   部分覆盖（白点）：该格仅部分样本满足约束（均值在该样本子集上计算）
#
# 字体：本机无宋体/SimSun，采用 Noto Serif CJK SC（思源宋体，宋体风格衬线），
#   来源 mplfonts wheel（内网 PyPI），已安装到 ~/.local/share/fonts/cjk/。
#
# 用法：PYTHONPATH=/home/wangyuanshuo02/.local/pylibs python3 plot_e98_abg_grid.py
#   e2e 数据落地后：python3 plot_e98_abg_grid.py --e2e <e2e.json>（add_e2e_overlay 目前占位）
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------------------
# 风格：publication-figures skill（okabe_ito 色板 + 线宽/字号规范）
# ---------------------------------------------------------------------------
SKILL_SCRIPTS = "/home/wangyuanshuo02/.claude/skills/publication-figures/scripts"
sys.path.insert(0, SKILL_SCRIPTS)
import style_config as sc  # noqa: E402
from figure_lint import lint_figure, summarize  # noqa: E402

PALETTE = sc.apply_style("okabe_ito")

# 中文字体注册（思源宋体 = 宋体风格衬线）
FONT_DIR = "/home/wangyuanshuo02/.local/share/fonts/cjk"
for _f in ("NotoSerifCJKsc-Regular.otf", "NotoSansCJKsc-Regular.otf"):
    _p = os.path.join(FONT_DIR, _f)
    if os.path.exists(_p):
        fm.fontManager.addfont(_p)
CJK_FONT = "Noto Serif CJK SC"
plt.rcParams["font.family"] = [CJK_FONT, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["hatch.linewidth"] = 1.2

# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
DATA = "/home/wangyuanshuo02/two-level-attention/exp/trace/results/e98_abg_full_grid.json"
OUT_DIR = "/home/wangyuanshuo02/sglang/ref/figs"
GRID = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
GAMMA_SLICE = 0.75          # 热图切片（任务指定，接近各组合 top 臂的 γ）
COMBOS = ["mavg", "mminmax", "aavg", "cavg", "ccluster"]
COMBO_DESC = {
    "mavg": "mavg (minmax, avg)",
    "mminmax": "mminmax (minmax, minmax)",
    "aavg": "aavg (avg, avg)",
    "cavg": "cavg (cluster, avg)",
    "ccluster": "ccluster (cluster, cluster)",
}
COMBO_COLOR = {
    "mavg": "#0072B2", "mminmax": "#E69F00", "aavg": "#009E73",
    "cavg": "#56B4E9", "ccluster": "#CC79A7",
}
COMBO_MARKER = {"mavg": "o", "mminmax": "s", "aavg": "^", "cavg": "D", "ccluster": "v"}
TICKLAB = ["0", ".125", ".25", ".375", ".5", ".625", ".75", ".875", "1"]
INVALID_GRAY = "#CCCCCC"

D = json.load(open(DATA))
MEAN, PS = D["mean"], D["per_sample"]
N_SAMP = D["n_samples"]


def arm(combo, a, b, g):
    return f"{combo}_a{a}_b{b}_g{g}"


def mass(combo, a, b, g):
    return MEAN.get(arm(combo, a, b, g))


def coverage(combo, a, b, g):
    k = arm(combo, a, b, g)
    return sum(1 for s in PS.values() if k in s)


def is_partition(a, b):
    """真分区臂：near/far 两池都激活（0<α<1 且 0<β<1）；角点 = 单池退化臂"""
    return 0.0 < a < 1.0 and 0.0 < b < 1.0


def best_partition_ab(combo):
    """该组合全 γ 网格上的最优真分区 (α,β)（每 (α,β) 取 9 个 γ 的最大值）"""
    best, best_v = None, -1.0
    for a in GRID:
        for b in GRID:
            if not is_partition(a, b):
                continue
            vals = [mass(combo, a, b, g) for g in GRID]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            v = max(vals)
            if v > best_v:
                best_v, best = v, (a, b)
    return best, best_v


# ---------------------------------------------------------------------------
# e2e overlay 预留接口（e2e 精度数据落地后补实现）
# ---------------------------------------------------------------------------
def add_e2e_overlay(ax, e2e_json_path):
    """在 mass 图上叠加 e2e 精度数据（双口径铁律：mass + e2e 都要有）。

    预期 e2e_json 格式：{"<臂名>": {"avg_f1": ..., "n_tasks": ...}, ...}
    落地后在此实现：fig9 画副轴散点 / fig10 加 e2e 列。当前占位。
    """
    pass


# ---------------------------------------------------------------------------
# 图 1：α×β 热图（γ=0.75 切片，5 组合 + 共享色条）
# ---------------------------------------------------------------------------
def build_slice_matrix(combo, gamma):
    """返回 (9×9 矩阵, 覆盖数矩阵)，NaN = 约束过滤的非法格"""
    M = np.full((9, 9), np.nan)
    C = np.zeros((9, 9), int)
    for i, a in enumerate(GRID):
        for j, b in enumerate(GRID):
            v = mass(combo, a, b, gamma)
            if v is not None:
                M[j, i] = v          # 行=β，列=α
            C[j, i] = coverage(combo, a, b, gamma)
    return M, C


def fig1_heatmaps():
    mats = {c: build_slice_matrix(c, GAMMA_SLICE) for c in COMBOS}
    # 统一色条范围：4 个 e2e 组合 valid 臂 min/max（任务口径；ccluster 范围在其内）
    e2e_vals = np.concatenate([mats[c][0].ravel() for c in COMBOS[:4]])
    e2e_vals = e2e_vals[~np.isnan(e2e_vals)]
    vmin, vmax = float(e2e_vals.min()), float(e2e_vals.max())
    vmin = np.floor(vmin * 20) / 20  # 0.65
    vmax = np.ceil(vmax * 20) / 20   # 0.90

    fig = plt.figure(figsize=(18, 12.8))
    # 5 个热图占 2×3 网格的前 5 格；细色条手工放置在最右侧、跨两行
    gs = GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.34,
                  left=0.075, right=0.862, top=0.875, bottom=0.07)
    ext = [-0.0625, 1.0625, -0.0625, 1.0625]
    letters = "ABCDE"
    readings = {}

    im = None
    for pi, combo in enumerate(COMBOS):
        ax = fig.add_subplot(gs[pi // 3, pi % 3])
        M, C = mats[combo]
        im = ax.imshow(M, origin="lower", extent=ext, cmap="viridis",
                       vmin=vmin, vmax=vmax, aspect="equal",
                       interpolation="nearest", zorder=1)
        # 非法配置格：灰底 + 白色对角影线
        for i, a in enumerate(GRID):
            for j, b in enumerate(GRID):
                if np.isnan(M[j, i]):
                    ax.add_patch(Rectangle((a - 0.0625, b - 0.0625), 0.125, 0.125,
                                           facecolor=INVALID_GRAY, edgecolor="white",
                                           hatch="///", lw=1.0, zorder=3))
        # 部分样本覆盖格：白点标记（n<16）
        for i, a in enumerate(GRID):
            for j, b in enumerate(GRID):
                if not np.isnan(M[j, i]) and 0 < C[j, i] < N_SAMP:
                    ax.plot(a, b, "o", ms=10, mfc="white", mec="black",
                            mew=2.0, zorder=6)
        # 真分区区（0<α,β<1）：黑色虚线框（角点行/列 = 单池退化臂）
        ax.add_patch(Rectangle((0.0625, 0.0625), 0.875, 0.875, fill=False,
                               ls="--", ec="black", lw=2.0, zorder=5))
        # 该切片真分区最优格：白星
        interior = M[1:8, 1:8]
        if np.any(~np.isnan(interior)):
            jj, ii = np.unravel_index(np.nanargmax(interior), interior.shape)
            ia, ib = GRID[ii + 1], GRID[jj + 1]
            ax.plot(ia, ib, "*", ms=26, mfc="white", mec="black", mew=1.8, zorder=7)
            tmax = M[jj + 1, ii + 1]
        else:
            tmax = float("nan")
        readings[combo] = (tmax, ia, ib)

        ax.set_xticks(GRID)
        ax.set_yticks(GRID)
        ax.set_xticklabels(TICKLAB, fontsize=18)
        ax.set_yticklabels(TICKLAB, fontsize=18)
        # 轴标签只标外圈（底行标 α、左列标 β），列/行语义共享
        if pi // 3 == 1:
            ax.set_xlabel("α（near 区长度比例）", fontsize=sc.FONT_SIZES["axis_label"])
        if pi % 3 == 0:
            ax.set_ylabel("β（near 页池比例）", fontsize=sc.FONT_SIZES["axis_label"])
        suffix = "（仅 mass）" if combo == "ccluster" else ""
        ax.set_title(f"{COMBO_DESC[combo]}{suffix}\n"
                     f"分区最优 {tmax:.3f} @ (α{ia}, β{ib})",
                     fontsize=20, pad=8)
        # 面板字母（白字黑描边，任何底色可见）
        ax.text(0.03, 0.97, letters[pi], transform=ax.transAxes, fontsize=32,
                fontweight="bold", va="top", ha="left", color="white", zorder=10,
                path_effects=[pe.withStroke(linewidth=3.5, foreground="black")])
        ax.tick_params(direction="out")

    # 共享细色条（跨两行，最右侧）
    cax = fig.add_axes([0.882, 0.07, 0.015, 0.805])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("softmax mass coverage（16 样本均值）",
                   fontsize=sc.FONT_SIZES["colorbar"], rotation=270, labelpad=52)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_ticks(np.arange(0.65, 0.905, 0.05))

    fig.suptitle(f"E98 α×β 网格 attention mass coverage（γ={GAMMA_SLICE} 切片，"
                 f"B_TOK=2048，页池 BP=64）", fontsize=23, y=0.975)
    fig.text(0.07, 0.015,
             "灰格（白斜线）= 约束过滤的非法配置（near_L < near_bp×BS 或 far 预算 > far_L）；白点 = 仅部分样本满足该格约束（6–14/16）\n"
             "黑色虚线框 = 真分区区（0<α,β<1），框外角点行/列为单池退化臂；白星 = 该切片分区最优。",
             fontsize=15, ha="left", va="bottom")
    return fig, readings


# ---------------------------------------------------------------------------
# 图 2：γ 扫描曲线（各组合最优 (α,β) 处）
# ---------------------------------------------------------------------------
def fig2_gamma_sweep():
    fig, ax = plt.subplots(figsize=(16, 9.5))
    fig.subplots_adjust(left=0.09, right=0.652, top=0.90, bottom=0.15)
    flat_lo, flat_hi = 0.625, 0.875
    readings = {}

    # mavg 平坦区底纹（服务「参数不需精调 = 部署简单」卖点）
    ax.axvspan(flat_lo, flat_hi, color="#999999", alpha=0.18, zorder=0)

    for combo in COMBOS:
        (a, b), _ = best_partition_ab(combo)
        xs, ys, ns = [], [], []
        for g in GRID:
            v = mass(combo, a, b, g)
            if v is None:
                continue
            xs.append(g)
            ys.append(v)
            ns.append(coverage(combo, a, b, g))
        ls = "--" if combo == "ccluster" else "-"   # ccluster 无 e2e 路径 → 虚线
        ax.plot(xs, ys, ls, color=COMBO_COLOR[combo], lw=3.5,
                marker=COMBO_MARKER[combo], ms=12, mfc=COMBO_COLOR[combo],
                mec="white", mew=2.5, zorder=5,
                label=f"{combo} @ (α={a}, β={b})"
                      f"{'（仅 mass）' if combo == 'ccluster' else ''}")
        # 部分覆盖点（n<16）：开圈标记（均值在该样本子集上）
        px = [x for x, n in zip(xs, ns) if n < N_SAMP]
        py = [y for x, y, n in zip(xs, ys, ns) if n < N_SAMP]
        if px:
            ax.plot(px, py, "o", ms=15, mfc="white", mec=COMBO_COLOR[combo],
                    mew=2.5, ls="none", zorder=6)
        readings[combo] = {"ab": (a, b), "gamma": xs, "mass": ys, "n": ns}

    # 平坦区注记（mavg：γ∈[0.625,0.875] 极差 < 0.001）——置于底纹区正上方，无需箭头
    mv = readings["mavg"]
    flat_vals = [y for x, y in zip(mv["gamma"], mv["mass"]) if flat_lo <= x <= flat_hi]
    fr = max(flat_vals) - min(flat_vals)
    ax.text(0.70, 0.913, f"mavg 平坦区 γ∈[0.625, 0.875]\n极差 {fr:.4f} (< 0.001)",
            fontsize=18, ha="center", va="bottom", zorder=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=COMBO_COLOR["mavg"], linewidth=2.0, alpha=0.95))

    # 部署参照点：mavg 曲线上 γ=0.125（部署臂取保守 γ）
    dep_y = mass("mavg", 0.125, 0.25, 0.125)
    flat_mean = float(np.mean(flat_vals))
    ax.plot([0.125], [dep_y], "s", ms=15, mfc="white", mec=COMBO_COLOR["mavg"],
            mew=2.5, ls="none", zorder=7)
    ax.annotate(f"部署臂取 γ=0.125（保守）\n较平坦区均值 {dep_y - flat_mean:+.4f}",
                xy=(0.125, dep_y), xytext=(0.02, 0.913),
                fontsize=18, ha="left", va="bottom",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="#555555", linewidth=2.0, alpha=0.95),
                arrowprops=dict(arrowstyle="->", color="#555555", lw=2.5,
                                shrinkA=20, shrinkB=4))

    ax.set_xlabel("γ（near 细筛 token 折扣，nt_near = nb_near·BS·γ）",
                  fontsize=sc.FONT_SIZES["axis_label"])
    ax.set_ylabel("softmax mass coverage（均值）", fontsize=sc.FONT_SIZES["axis_label"])
    ax.set_xticks(GRID)
    ax.set_xticklabels(TICKLAB, fontsize=sc.FONT_SIZES["tick_label"])
    ax.set_yticks(np.arange(0.65, 0.96, 0.05))
    ax.tick_params(axis="y", labelsize=sc.FONT_SIZES["tick_label"])
    ax.set_xlim(-0.05, 1.08)
    ax.set_ylim(0.63, 0.97)
    ax.set_title("E98 γ 扫描：各 method 组合最优 (α,β) 处 mass–γ 曲线（B_TOK=2048）",
                 fontsize=23, pad=14)
    leg = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=17,
                    frameon=True, fancybox=False, shadow=False,
                    borderaxespad=0)
    leg.get_frame().set_linewidth(2.5)
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(0.95)
    fig.text(0.09, 0.015,
             "method 组合（far, near）：mavg=(minmax,avg)、mminmax=(minmax,minmax)、aavg=(avg,avg)、cavg=(cluster,avg)、ccluster=(cluster,cluster)\n"
             "开圈 = 该 γ 点仅部分样本满足 far 预算约束（mminmax/aavg 在 (α.875,β.875) 处 n=8–14/16，γ≥0.5 平台区为全 16 样本）；ccluster 虚线 = 无 e2e 实现路径",
             fontsize=15, ha="left", va="bottom")
    return fig, readings


# ---------------------------------------------------------------------------
# 图 3：单池角点 vs 各组合分区最优 vs 部署参照臂
# ---------------------------------------------------------------------------
def fig3_mono_vs_partition():
    mono_v = mass("mavg", 0.0, 0.0, 0.0)             # 0.8979（minmax 单池角点）
    dep_v = mass("mavg", 0.125, 0.375, 0.125)        # 0.8728（部署参照臂）
    rows = []
    rows.append(("单池角点 (minmax top-k, α=β=0)", mono_v, "#999999", ""))
    for combo in COMBOS:
        (a, b), _ = best_partition_ab(combo)
        # 该组合最优臂（含最优 γ）
        best_k, best_v = None, -1
        for g in GRID:
            v = mass(combo, a, b, g)
            if v is not None and v > best_v:
                best_v, best_k = v, g
        hatch = "//" if combo == "ccluster" else ""
        suffix = " ※仅 mass" if combo == "ccluster" else ""
        rows.append((f"{combo} 分区最优 (α{a} β{b} γ{best_k}){suffix}",
                     best_v, COMBO_COLOR[combo], hatch))
    rows.append(("部署参照臂 mavg (α.125 β.375 γ.125)", dep_v, "#D55E00", ""))

    # 水平条形图（类别标签长，横排可读性最好）
    fig, ax = plt.subplots(figsize=(15.5, 9))
    fig.subplots_adjust(left=0.36, right=0.965, top=0.90, bottom=0.13)
    ys = np.arange(len(rows))
    for i, (lab, v, c, h) in enumerate(rows):
        ax.barh(i, v, height=0.62, color=c, edgecolor="black", lw=2.0,
                hatch=h, zorder=5)
        ax.text(v + 0.0015, i, f"{v:.4f}", ha="left", va="center",
                fontsize=18, zorder=8)
    # 单池角点参考线（竖直虚线）+ 顶部标签（y 负方向 = invert 后的图顶）
    ax.axvline(mono_v, color="#333333", ls="--", lw=3.0, zorder=4)
    ax.text(mono_v, -0.68, "单池角点 0.8979", ha="center", va="center",
            fontsize=17, color="#333333")

    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=17)
    ax.set_ylim(len(rows) - 0.5 + 0.05, -0.95)   # invert：单池角点置顶，顶部留标签空间
    ax.set_xlabel("softmax mass coverage（B_TOK=2048 宽松预算口径，16 样本均值）",
                  fontsize=sc.FONT_SIZES["axis_label"])
    ax.set_xlim(0.74, 0.925)
    ax.set_xticks(np.arange(0.74, 0.925, 0.02))
    ax.set_title("E98 单池角点 vs 各组合分区最优 vs 部署参照臂",
                 fontsize=23, pad=14)
    ax.tick_params(axis="x", labelsize=sc.FONT_SIZES["tick_label"])
    fig.text(0.345, 0.015,
             "口径注：宽松预算（B_TOK=2048）mass 口径下单池 top-k 略高于分区（+0.002 vs mminmax，+0.016 vs mavg，+0.025 vs 部署臂）\n"
             "E88 已判紧预算下分区赢，两口径并存；ccluster（斜线柱）仅 mass 数据，无 e2e 路径；x 轴自 0.74 截断以放大差异。",
             fontsize=15, ha="left", va="bottom")
    return fig, rows


# ---------------------------------------------------------------------------
# 导出 + lint + 空白检查
# ---------------------------------------------------------------------------
def export(fig, name, lint_name):
    pdf = os.path.join(OUT_DIR, f"{name}.pdf")
    png = os.path.join(OUT_DIR, f"{name}.png")
    rep = lint_figure(fig)
    print(f"[lint:{lint_name}] {summarize(rep)}")
    fig.savefig(pdf, facecolor="white")
    fig.savefig(png, facecolor="white", dpi=300)
    plt.close(fig)
    # PNG 非空白检查（PIL 极值差）
    from PIL import Image
    im = Image.open(png).convert("L")
    extrema = im.getextrema()
    ok = extrema[1] - extrema[0] > 50
    print(f"[check:{lint_name}] PNG {im.size}, 灰度极值 {extrema}, "
          f"{'OK 非空白' if ok else '疑似空白!'}, "
          f"{os.path.getsize(png)//1024} KB / {os.path.getsize(pdf)//1024} KB")
    return rep


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    e2e_path = None
    if "--e2e" in sys.argv:
        e2e_path = sys.argv[sys.argv.index("--e2e") + 1]

    # ---- 图 1
    fig, readings = fig1_heatmaps()
    if e2e_path:
        add_e2e_overlay(fig, e2e_path)
    export(fig, "fig8_e98_abg_heatmap", "fig8")
    print("[fig8] 切片分区最优:", {c: (round(v, 4), a, b) for c, (v, a, b) in readings.items()})

    # ---- 图 2
    fig, r2 = fig2_gamma_sweep()
    if e2e_path:
        add_e2e_overlay(fig.axes[0], e2e_path)
    export(fig, "fig9_e98_gamma_sweep", "fig9")
    mv = r2["mavg"]
    flat = [y for x, y in zip(mv["gamma"], mv["mass"]) if 0.625 <= x <= 0.875]
    print(f"[fig9] mavg @ {mv['ab']}: γ 平坦区 [0.625,0.875] 极差 "
          f"{max(flat)-min(flat):.5f}；部署 γ=0.125 较平坦区 "
          f"{mass('mavg', 0.125, 0.25, 0.125) - float(np.mean(flat)):+.4f}")
    for c in COMBOS:
        print(f"[fig9] {c} best ab = {r2[c]['ab']}, curve = "
              f"{[round(y, 4) for y in r2[c]['mass']]}, n = {r2[c]['n']}")

    # ---- 图 3
    fig, bars = fig3_mono_vs_partition()
    if e2e_path:
        add_e2e_overlay(fig.axes[0], e2e_path)
    export(fig, "fig10_e98_mono_vs_partition", "fig10")
    print("[fig10] bars:", [(b[0].replace(chr(10), ' | '), round(b[1], 4)) for b in bars])


if __name__ == "__main__":
    main()
