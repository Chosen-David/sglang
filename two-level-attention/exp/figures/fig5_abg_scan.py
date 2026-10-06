# Fig 5 (round3 新增, 用户 10-02 指令): α/β/γ 扫描与各 method 组合最优配比
# (a) 3 method 组合 (mavg/cavg/aavg, 即 (far,near) = (minmax,avg)/(cluster,avg)/(avg,avg))
#     × 9α × 9β 网格热力图（16 样本均值）
#     —— 合法域过滤: near_L ≥ nb_near·page_size ≥ nt_near
#        nb_near = BP·β (BP=64 块池), near_L = α·mid_len, mid_len = S−sink−swa
#        非法格点(β·4096 > α·mid_len_median)灰化 hatch, 代码内 topk min() 截断保护
#        标各 method 最优格点(★)与固定臂 α.125/β.25(○)
# (b) γ 扫描: e64b_bp_gamma bp128, 3 method × γ∈{0.5,0.75,1.0} (trace 重放口径)
#     + e2e γ 等比缩放修正点注记 (γ=0.125 @ K2=1024)
# (c) 各 method 最优组合 vs 固定臂差距 bar —— oracle−固定 ≤0.01 平坦性证据
# (d) [E98 e2e 落袋后新增] mass vs e2e 散点（E98 e2e 网格 12 臂 + 参照臂）：
#     核心信息 = mass→e2e 排序反转 —— mass 冠军 mminmax_a.875_b.875_g.5 (0.896)
#     e2e 仅 43.90（第 2 档，与 aavg 并列）；e2e 冠军 mavg_a.125_b.375_g.625
#     (mass 0.8801, 有效配置 mass 第 5) e2e 45.08。散点按组合着色，★=e2e 冠军，
#     灰 ○=部署参照臂 (γ=0.125, 44.59)；并给出 12 臂 Spearman 秩相关。
#     热力图上叠加 e2e 实测臂格点环标（open ring；α/β 语义两口径一致可叠标，
#     mminmax 的 e2e 臂在 α,β≥0.625、无对应热图，只在 panel (d) 出现）
# 数据快照: e64g_full_grid.json / e64b_bp_gamma.json / e64i_tight.json / 样本 S (/tmp/e64_sample_S.json)
#           e98_e2e_grid.json (e2e 13 数据臂) / e98_best_election.json (选举判决)
# 口径注: mass = BP=64, B_TOK=2048, 16 样本 trace 均值;
#         e2e  = K1=128, K2=1024, LongBench hotpotqa+musique (各 n=200), avg=(hq+mu)/2
# house style: okabe_ito、白底、全英文、矢量 PDF+PNG
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

RES = "/home/wangyuanshuo02/two-level-attention/exp/trace/results"
OUTS = ["/home/wangyuanshuo02/sglang/paper/figures",
        "/home/wangyuanshuo02/two-level-attention/exp/figures"]

OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
      "red": "#D55E00", "purple": "#CC79A7", "gray": "#7F7F7F", "sky": "#56B4E9"}
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "figure.dpi": 200, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})

METHODS = ["mavg", "cavg", "aavg"]
MNAME = {"mavg": "minmax+avg", "cavg": "cluster+avg", "aavg": "avg+avg"}
GRID = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
SINK, SWA, BS, BP = 128, 1024, 64, 64

# ---------- 数据 ----------
g = json.load(open(f"{RES}/e64g_full_grid.json"))
samples = sorted(g.keys())
S_map = json.load(open("/tmp/e64_sample_S.json"))
mid_len = np.array([S_map[s] - SINK - SWA for s in samples])
mid_med = float(np.median(mid_len))

# 3 method × 9×9 网格均值
grid = {m: np.full((9, 9), np.nan) for m in METHODS}
for m in METHODS:
    for i, a in enumerate(GRID):
        for j, b in enumerate(GRID):
            key = f"{m}_a{a}_b{b}"
            vals = [g[s][key] for s in samples if key in g[s]]
            if vals:
                grid[m][i, j] = float(np.mean(vals))

# 合法域: near_L = α·mid_len ≥ nb_near·BS = BP·β·BS（中位样本口径）
def legal(a, b):
    if a == 0 or b == 0:
        return True  # 边界角点语义已由脚本显式处理
    return BP * b * BS <= a * mid_med

# γ 数据 (bp=128)
gb = json.load(open(f"{RES}/e64b_bp_gamma.json"))
gsamples = sorted(gb.keys())
GAMMAS = [0.5, 0.75, 1.0]
gcurve = {m: [] for m in METHODS}
for m in METHODS:
    for gm in GAMMAS:
        key = f"{m}_bp128_g{gm}"
        gcurve[m].append(float(np.mean([gb[s][key] for s in gsamples if key in gb[s]])))

# ---------- 最优组合汇总 ----------
fixed_key = "a0.125_b0.25"
summary = {}
for m in METHODS:
    best = np.unravel_index(np.nanargmax(grid[m]), grid[m].shape)
    best_a, best_b = GRID[best[0]], GRID[best[1]]
    best_v = grid[m][best[0], best[1]]
    fixed_v = grid[m][GRID.index(0.125), GRID.index(0.25)]
    summary[m] = dict(best_a=best_a, best_b=best_b, best_v=best_v,
                      fixed_v=fixed_v, gap=best_v - fixed_v)
    print(f"{m}: best (a={best_a}, b={best_b}) = {best_v:.4f}; "
          f"fixed (0.125, 0.25) = {fixed_v:.4f}; gap = {best_v - fixed_v:+.4f}")

# e64i_tight: 9 组合（α.125/β.25 固定）紧预算
ti = json.load(open(f"{RES}/e64i_tight.json"))
tisamples = sorted(ti.keys())
combo = {}
for c in ["mavg+mavg", "mavg+cavg", "mavg+aavg", "cavg+mavg", "cavg+cavg",
           "cavg+aavg", "aavg+mavg", "aavg+cavg", "aavg+aavg"]:
    key = f"{c}_a0.125_b0.25"
    combo[c] = float(np.mean([ti[s][key] for s in tisamples if key in ti[s]]))
best_combo = max(combo, key=combo.get)
print("tight 9-combo:", {k: round(v, 4) for k, v in combo.items()})
print("best combo:", best_combo, combo[best_combo])
json.dump({"summary": summary, "combo_tight": combo,
           "mid_len_median": mid_med,
           "grid": {m: grid[m].tolist() for m in METHODS}},
          open(f"{RES}/e64k_abg_scan_summary.json", "w"), indent=1)

# ---------- E98 e2e 数据（mass→e2e 排序反转, panel (d)） ----------
E2E = json.load(open(f"{RES}/e98_e2e_grid.json"))
ELEC = json.load(open(f"{RES}/e98_best_election.json"))
E2E_COMBO = ["mavg", "mminmax", "aavg", "cavg"]
E2E_COLOR = {"mavg": OI["blue"], "mminmax": OI["orange"],
             "aavg": OI["green"], "cavg": OI["sky"]}

# 解析 13 个数据臂（12 有效臂 + 1 参照臂 γ=0.125）
arms = []
for k, v in E2E.items():
    if not isinstance(v, dict) or "mass" not in v:
        continue  # note / v1_fix / protocol 等元信息键
    mm = re.match(r"(\w+)_a([\d.]+)_b([\d.]+)_g([\d.]+)", k)
    arms.append(dict(tag=k, combo=mm.group(1), a=float(mm.group(2)),
                     b=float(mm.group(3)), g=float(mm.group(4)),
                     mass=float(v["mass"]), hq=float(v["hotpotqa"]),
                     mu=float(v["musique"]),
                     avg=(float(v["hotpotqa"]) + float(v["musique"])) / 2.0,
                     ref=bool(v.get("ref", False))))
elig = [a for a in arms if not a["ref"]]
ref_arm = [a for a in arms if a["ref"]][0]
champ = ELEC["best"]
mass_champ = ELEC["mass_vs_e2e"]["mass_best"]

# 有效配置去重（γ 截断坍缩: mavg β.25 的 g0.75/g0.375 e2e 同分同配置）
seen, dedup = set(), []
for a in sorted(elig, key=lambda x: -x["mass"]):
    kk = (a["combo"], a["a"], a["b"], a["hq"], a["mu"])
    if kk not in seen:
        seen.add(kk)
        dedup.append(a)
# tie 感知名次（修改竞赛排名: 并列块内取最后一位, 与 fig11 的 1,3,3,4 规则一致）:
# rank = 有效配置中分值不低于自身的臂数（含自身）。mminmax 最优臂 43.90 与 aavg
# 最优臂 43.90 的 hq/mu 逐位相同, 朴素 argsort 在并列内的名次是任意序（读者报告 C1）
champ_mass_rank = sum(1 for a in dedup
                      if round(a["mass"], 4) >= round(champ["mass"], 4))
mass_champ_e2e_rank = sum(1 for a in dedup
                          if round(a["avg"], 2) >= round(mass_champ["avg"], 2))
rho, _ = spearmanr([a["mass"] for a in elig], [a["avg"] for a in elig])
print(f"[E98 e2e] {len(elig)} eligible arms + 1 ref; Spearman rho = {rho:.3f}")
print(f"[E98 e2e] e2e champion {champ['tag']}: mass {champ['mass']:.4f} "
      f"(rank #{champ_mass_rank} of {len(dedup)} effective cfgs) -> e2e {champ['avg']:.2f}")
print(f"[E98 e2e] mass champion {mass_champ['tag']}: mass {mass_champ['mass']:.4f} "
      f"-> e2e {mass_champ['avg']:.2f} (rank #{mass_champ_e2e_rank})")
print(f"[E98 e2e] ref arm {ref_arm['tag']}: e2e {ref_arm['avg']:.2f}; "
      f"best - ref = {champ['avg'] - ref_arm['avg']:+.2f}")

# ---------- 画布 ----------
# 版式修复 (glyph 重叠 16 对): 图加高 + bottom 抬高, 给底部脚注让出
# xlabel 以下的独立 y 带; 脚注两行拉开行距 (>字符 bbox 高 11.5pt)
fig = plt.figure(figsize=(17.2, 4.9))
gs = fig.add_gridspec(1, 6, width_ratios=[1, 1, 1, 1.05, 1.05, 1.62],
                      wspace=0.6, bottom=0.21, top=0.87)

# (a) 三 method 热力图
vmin = min(np.nanmin(grid[m]) for m in METHODS)
vmax = max(np.nanmax(grid[m]) for m in METHODS)
for k, m in enumerate(METHODS):
    ax = fig.add_subplot(gs[0, k])
    Z = grid[m].T  # 行=β, 列=α
    im = ax.pcolormesh(np.array(GRID), np.array(GRID), Z,
                       cmap="viridis", vmin=vmin, vmax=vmax, shading="nearest")
    # 非法域灰化
    for i, a in enumerate(GRID):
        for j, b in enumerate(GRID):
            if not legal(a, b):
                ax.add_patch(plt.Rectangle((a - 0.0625, b - 0.0625), 0.125, 0.125,
                                           fc="none", ec="white", lw=1.2, hatch="///",
                                           alpha=0.85, zorder=3))
    # 最优点 ★ 与固定臂 ○
    bi = np.unravel_index(np.nanargmax(grid[m]), grid[m].shape)
    ax.plot(GRID[bi[0]], GRID[bi[1]], "*", ms=13, color=OI["red"],
            mec="white", mew=0.7, zorder=5)
    ax.plot(0.125, 0.25, "o", ms=7, color=OI["orange"], mec="white", mew=0.7, zorder=5)
    # E98 e2e 实测臂格点环标（open ring; α/β 语义两口径一致, mminmax 无热图见 panel d）
    for a_ in arms:
        if a_["combo"] == m:
            ax.plot(a_["a"], a_["b"], "o", ms=11, mfc="none", mec="black",
                    mew=1.2, zorder=6)
    ax.set_xlabel(r"$\alpha$ (near length share)", fontsize=8)
    if k == 0:
        ax.set_ylabel(r"$\beta$ (near page share)", fontsize=8)
    ax.set_title(f"{MNAME[m]}\nbest {summary[m]['best_v']:.3f} @ "
                 f"({summary[m]['best_a']:.3f}, {summary[m]['best_b']:.3f})",
                 fontsize=8.2)
    ax.tick_params(labelsize=7)
# 图例（在第一个子图下加文字说明）
# 两行 y 拉开 0.034 (≈12pt > 字符 bbox 高 11.5pt): 原行距 4.9pt 导致跨行互叠;
# 脚注整体已由 bottom=0.21 抬离 xlabel 带
fig.text(0.055, 0.048, "★ per-method best    ● deployed (α=0.125, β=0.25)    "
         "○ ring = grid point of the 13 e2e-measured arms (mminmax e2e arms at α,β ≥ 0.625 → panel (d))    "
         "hatched: near budget > near length (auto-clipped by topk)",
         fontsize=6.8, color="#444444")
fig.text(0.055, 0.014, "calibers: mass = BP=64, B_TOK=2048, 16-sample trace mean; "
         "e2e (panel d) = K₁=128, K₂=1024, LongBench hotpotqa+musique (n=200 per task), "
         "avg = (hq + mu) / 2",
         fontsize=6.8, color="#444444")

# (b) γ 扫描
axb = fig.add_subplot(gs[0, 3])
xg = np.arange(len(GAMMAS))
for m, c in zip(METHODS, [OI["blue"], OI["purple"], OI["gray"]]):
    axb.plot(xg, gcurve[m], "o-", color=c, lw=1.6, ms=4.5, label=MNAME[m])
    for xv, yv in zip(xg, gcurve[m]):
        axb.annotate(f"{yv:.3f}", (xv, yv), textcoords="offset points",
                     xytext=(0, 6), fontsize=6.2, ha="center", color="#555555")
axb.set_xticks(xg)
axb.set_xticklabels([f"γ={g}" for g in GAMMAS], fontsize=7.5)
axb.set_xlabel("near token discount γ (bp=128, trace)", fontsize=7.8)
axb.set_ylabel("mass coverage", fontsize=7.8)
axb.set_ylim(0.80, 0.95)
axb.set_title("(b) γ scan: flat in [0.5, 1.0]\ne2e needs γ=0.125 at K₂=1024", fontsize=8.2)
axb.legend(fontsize=6.6, frameon=False, loc="lower right")
axb.tick_params(labelsize=7)

# (c) 最优 vs 固定臂
axc = fig.add_subplot(gs[0, 4])
ms = METHODS
gaps = [summary[m]["gap"] for m in ms]
bars = axc.bar(np.arange(3), gaps, 0.55,
               color=[OI["blue"], OI["purple"], OI["gray"]])
for i, (m, gp) in enumerate(zip(ms, gaps)):
    axc.annotate(f"{gp:+.4f}", (i, gp), textcoords="offset points",
                 xytext=(0, 6 if gp >= 0 else -10), fontsize=7.0, ha="center")
axc.axhline(0, color="#555555", lw=0.8)
axc.set_xticks(np.arange(3))
# 旋转 18°: 三个方法名横排宽度 > 刻度间距, 曾水平互叠 ov=1.0 (本次修复)
axc.set_xticklabels([MNAME[m] for m in ms], fontsize=7.5,
                    rotation=18, ha="right")
axc.set_ylabel("oracle best − deployed", fontsize=7.8)
axc.set_title("(c) per-combo best vs deployed\nbest sits at (0,0): pure far-method", fontsize=8.2)
axc.tick_params(labelsize=7)

# (d) [E98] mass vs e2e 散点 —— 排序反转
# 点位横轴分布: aavg 最左 (mass≈0.795) / cavg 0.877-0.879 / mavg 0.880-0.882 /
# mminmax 最右 0.895-0.896 —— 左上/左下/中心三块空白区放注记与图例
axd = fig.add_subplot(gs[0, 5])
for c in E2E_COMBO:
    xs = [a["mass"] for a in elig if a["combo"] == c]
    ys = [a["avg"] for a in elig if a["combo"] == c]
    axd.plot(xs, ys, "o", ms=6.5, color=E2E_COLOR[c], mec="white", mew=0.7,
             ls="none", label=c, zorder=5)
# 部署参照臂（灰开圈）+ 参考线
axd.plot([ref_arm["mass"]], [ref_arm["avg"]], "o", ms=8, mfc="none",
         mec=OI["gray"], mew=1.4, ls="none", label="ref (γ=0.125)", zorder=4)
axd.axhline(ref_arm["avg"], color=OI["gray"], lw=0.8, ls=":", zorder=2)
axd.annotate(f"ref {ref_arm['avg']:.2f}", (0.792, ref_arm["avg"]),
             fontsize=6.2, color=OI["gray"], va="bottom", ha="left")
# e2e 冠军 ★ + 注记（mass 有效配置第 5, 左上空白区）
chx, chy = champ["mass"], champ["avg"]
axd.plot([chx], [chy], "*", ms=13, color=OI["red"], mec="white", mew=0.8,
         zorder=7)
axd.annotate(f"e2e best {chy:.2f}\nmavg (α.125, β.375, γ.625)\n"
             f"mass #{champ_mass_rank} of {len(dedup)} cfgs",
             xy=(chx, chy), xytext=(0.795, 45.52), fontsize=6.4, ha="left",
             va="top", color=OI["red"],
             arrowprops=dict(arrowstyle="->", color=OI["red"], lw=0.9,
                             shrinkA=4, shrinkB=8))
# mass 冠军注记（e2e 第 3 档与 aavg 并列; 放 mminmax 点群左侧中央空白走廊,
# 短水平箭头指向右侧 mass 冠军点, 避开 cavg/mavg 点群）
mcx, mcy = mass_champ["mass"], mass_champ["avg"]
axd.annotate(f"mass best {mcx:.3f}\nmminmax (α.875, β.875, γ.5)\n"
             f"→ e2e {mcy:.2f}, rank #{mass_champ_e2e_rank} (tie aavg)",
             xy=(mcx, mcy), xytext=(0.884, 43.9), fontsize=6.4, ha="right",
             va="center", color=E2E_COLOR["mminmax"],
             arrowprops=dict(arrowstyle="->", color=E2E_COLOR["mminmax"], lw=0.9,
                             shrinkA=3, shrinkB=7))
axd.set_xlabel("mass coverage\n(BP=64, B_TOK=2048, 16-sample mean)", fontsize=7.4)
axd.set_ylabel("e2e avg (K₁=128, K₂=1024,\nhq + mu, n=200 per task)", fontsize=7.4)
axd.set_xlim(0.788, 0.905)
axd.set_ylim(42.4, 45.6)
axd.set_title(f"(d) mass → e2e rank reversal\nSpearman ρ = {rho:.2f} (12 arms)",
              fontsize=8.2)
# 组合短名图例（全名见 panel (a)/脚注）; 左下空白区（aavg 点群 y≥0.37 之上无冲突）
axd.legend(fontsize=6.4, frameon=False, loc="lower left",
           handletextpad=0.3, borderaxespad=0.2)
axd.tick_params(labelsize=7)
# colorbar 移到整图最右 (原位在热图组与 panel (b) 之间的走廊, 其标签与
# panel (b) 的 ylabel 旋转文字互叠 ov=0.75 —— 本次修复移到 panel (d) 右侧空白)
cb = fig.colorbar(im, ax=fig.axes, pad=0.015, fraction=0.025)
cb.set_label("mass coverage (16 samples)", fontsize=7.5)
cb.ax.tick_params(labelsize=6.5)

for out in OUTS:
    fig.savefig(f"{out}/fig5_abg_scan.pdf")
    fig.savefig(f"{out}/fig5_abg_scan.png")
print("saved fig5_abg_scan")
