# Fig 11 (E98 e2e 落袋后新增): 四 method 组合 mass 最优 vs e2e 最优 双口径对照
# (a) mass 口径各组合最优（E98 全网格真分区臂, BP=64/B_TOK=2048, 16 样本均值）
#     —— 排序 mminmax 0.896 > mavg 0.882 > cavg 0.879 > aavg 0.796
# (b) e2e 口径各组合最优（E98 e2e 网格, K1=128/K2=1024, LongBench hq+mu n=200）
#     —— 排序 mavg 45.08 > mminmax 43.90 ≈ aavg 43.90 > cavg 43.58
# 核心信息 = mass→e2e 排序反转：mminmax mass 冠军跌至 e2e 第 3 档（与 aavg 并列），
# mavg（mass 第 2）升为 e2e 冠军（avg 45.08 与 hq 55.44 均为 13 臂最高;
# mu 34.71 仅次于 β=0.25 臂的 35.57, 该臂 avg 44.86 更低）；
# e2e 排序 mavg > mminmax ≈ aavg > cavg 与主表 method 排序一致。
# 两 panel 共用同一 x 顺序（按 mass 降序），名次标签 #1-#4 置于柱顶直观显示反转。
# ccluster 无 e2e 实现路径（仅 mass），如实排除并在脚注注明。
# 数据快照（落袋 JSON, 勿手改数字）:
#   exp/trace/results/e98_abg_full_grid.json  mass 全网格 (per_sample + mean)
#   exp/trace/results/e98_e2e_grid.json       e2e 13 数据臂 (12 有效 + 1 参照)
#   exp/trace/results/e98_best_election.json  选举判决 (best/ref/rank)
# house style: okabe_ito、白底、全英文、矢量 PDF+PNG、axes.spines top/right off
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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

# ---------- 数据 ----------
GRID = json.load(open(f"{RES}/e98_abg_full_grid.json"))["mean"]
E2E = json.load(open(f"{RES}/e98_e2e_grid.json"))
ELEC = json.load(open(f"{RES}/e98_best_election.json"))

COMBOS = ["mavg", "mminmax", "aavg", "cavg"]
MNAME = {"mavg": "minmax+avg", "mminmax": "minmax+minmax",
         "aavg": "avg+avg", "cavg": "cluster+avg"}
COLOR = {"mavg": OI["blue"], "mminmax": OI["orange"],
         "aavg": OI["green"], "cavg": OI["sky"]}

# mass 口径各组合最优真分区臂（0<α,β<1, 角点=单池退化臂不算分区）
def mass_best(combo):
    best = (None, -1.0)
    for k, v in GRID.items():
        mm = re.match(rf"{combo}_a([\d.]+)_b([\d.]+)_g([\d.]+)$", k)
        if not mm:
            continue
        a, b = float(mm.group(1)), float(mm.group(2))
        if not (0.0 < a < 1.0 and 0.0 < b < 1.0):
            continue
        if v > best[1]:
            best = ((a, b, float(mm.group(3))), v)
    return best

# e2e 口径各组合最优臂（12 有效臂, 参照臂不计入组合冠军）
e2e_arms = []
for k, v in E2E.items():
    if not isinstance(v, dict) or "mass" not in v:
        continue
    mm = re.match(r"(\w+)_a([\d.]+)_b([\d.]+)_g([\d.]+)", k)
    e2e_arms.append(dict(combo=mm.group(1), a=float(mm.group(2)),
                         b=float(mm.group(3)), g=float(mm.group(4)),
                         mass=float(v["mass"]), hq=float(v["hotpotqa"]),
                         mu=float(v["musique"]),
                         avg=(float(v["hotpotqa"]) + float(v["musique"])) / 2.0,
                         ref=bool(v.get("ref", False))))
ref_arm = next(a for a in e2e_arms if a["ref"])

rows = {}
for c in COMBOS:
    (a, b, g), mv = mass_best(c)
    ea = max((x for x in e2e_arms if x["combo"] == c and not x["ref"]),
             key=lambda x: x["avg"])
    rows[c] = dict(mass_cfg=(a, b, g), mass=mv, e2e_cfg=(ea["a"], ea["b"], ea["g"]),
                   e2e=ea["avg"], hq=ea["hq"], mu=ea["mu"])
    print(f"{c}: mass best {mv:.4f} @ (α{a}, β{b}, γ{g}) | "
          f"e2e best {ea['avg']:.2f} @ (α{ea['a']}, β{ea['b']}, γ{ea['g']}) "
          f"(hq {ea['hq']:.2f}, mu {ea['mu']:.2f})")

# x 顺序 = 按 mass 降序（panel (a) 从左到右名次 #1..#4; panel (b) 同序看反转）
order = sorted(COMBOS, key=lambda c: -rows[c]["mass"])

# 名次计算: tie 感知的修改竞赛排名（并列块内取最后一位, 1,3,3,4）——
# 朴素 argsort 会把 43.90 精确并列的 (minmax,minmax)/(avg,avg) 误标 #2/#3（读者报告 C1）。
# 规则: rank = 值不低于自身的组合数（含自身）。验算: mavg 45.08 → #1;
# mminmax 43.90 = aavg 43.90（hq/mu 逐位相同 54.4/33.4）→ 双 #3; cavg 43.58 → #4。
# 注意比较精度取 4 位小数: mass 侧 0.8822/0.879 在 2 位舍入下会假并列。
def mcomp_rank(values):
    return {c: sum(1 for d in values if round(values[d], 4) >= round(values[c], 4))
            for c in values}

mass_rank = mcomp_rank({c: rows[c]["mass"] for c in COMBOS})
e2e_rank = mcomp_rank({c: rows[c]["e2e"] for c in COMBOS})
print("mass order:", order, "| mass ranks:", mass_rank, "| e2e ranks:", e2e_rank)

# ---------- 画布 ----------
fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.9),
                         gridspec_kw={"width_ratios": [1, 1], "wspace": 0.30})
x = np.arange(len(order))

# (a) mass 口径各组合最优
ax = axes[0]
mv = [rows[c]["mass"] for c in order]
bars = ax.bar(x, mv, 0.58, color=[COLOR[c] for c in order],
              edgecolor="white", lw=0.6)
for i, c in enumerate(order):
    ax.annotate(f"{mv[i]:.4f}", (i, mv[i]), textcoords="offset points",
                xytext=(0, 5), fontsize=7.4, ha="center")
    ax.annotate(f"#{mass_rank[c]}", (i, mv[i]), textcoords="offset points",
                xytext=(0, 15), fontsize=7.6, ha="center", fontweight="bold",
                color="#333333")
# 最优臂配置作为 xticklabel 第二行（柱宽装不下横排文字, 放轴下深色小字）
def cfg_str(cfg):
    # 去前导零缩写: 0.125→.125, 配置行宽度控制在 xtick 间距内
    return f"α{cfg[0]:g} β{cfg[1]:g} γ{cfg[2]:g}".replace("α0.", "α.").replace("β0.", "β.").replace("γ0.", "γ.")

lab_a = [f"{MNAME[c]}\n{cfg_str(rows[c]['mass_cfg'])}" for c in order]
ax.set_xticks(x)
ax.set_xticklabels(lab_a, fontsize=6.4)
ax.set_ylabel("mass coverage (BP=64, B_TOK=2048,\n16-sample mean)", fontsize=7.8)
ax.set_ylim(0.775, 0.912)
ax.set_title("(a) mass caliber: per-combo best\nmminmax is the mass champion",
             fontsize=8.4)
ax.tick_params(labelsize=6.4)

# (b) e2e 口径各组合最优（同 x 顺序, 高度反转 + hq/mu 双任务刻度在柱右侧）
ax = axes[1]
ev = [rows[c]["e2e"] for c in order]
edge_c = [OI["red"] if c == "mavg" else "white" for c in order]
edge_w = [1.8 if c == "mavg" else 0.6 for c in order]
ax.bar(x, ev, 0.58, color=[COLOR[c] for c in order],
       edgecolor=edge_c, lw=edge_w)
for i, c in enumerate(order):
    ax.annotate(f"{ev[i]:.2f}", (i, ev[i]), textcoords="offset points",
                xytext=(0, 5), fontsize=7.4, ha="center")
    ax.annotate(f"#{e2e_rank[c]}", (i, ev[i]), textcoords="offset points",
                xytext=(0, 15), fontsize=7.6, ha="center", fontweight="bold",
                color=OI["red"] if c == "mavg" else "#333333")
    # hq / mu 双任务值（白字写进柱内; mavg 平均分与 hq 均为 13 臂最高,
    # mu 34.71 仅次于 β=0.25 臂的 35.57——图内注记须与正文 M1 口径一致）
    # （hq≈54-55 超出 avg 轴范围, 不能画成刻度线, 直接标注数值）
    ax.annotate(f"hq {rows[c]['hq']:.1f}\nmu {rows[c]['mu']:.1f}", (i, ev[i]),
                textcoords="offset points", xytext=(0, -22), fontsize=6.2,
                ha="center", va="top", color="white", zorder=6,
                fontweight="bold" if c == "mavg" else "normal")
# 部署参照臂参考线（mavg α.125/β.375/γ.125, e2e 44.59）
# 注记放轴右侧外空白区（紧贴虚线右端）: 轴内任意位置都会撞柱顶标签——
# mavg 柱 45.08 全场最高占据顶部, 其余柱顶有名次/数值标签（glyph 重叠 0.58/0.64, 本次修复）
ax.axhline(ref_arm["avg"], color=OI["gray"], lw=1.0, ls=":", zorder=2)
y_frac = (ref_arm["avg"] - 31.5) / (47.1 - 31.5)
ax.annotate(f"deployed ref {ref_arm['avg']:.2f} (γ=0.125)",
            xy=(1.012, y_frac), xycoords="axes fraction",
            fontsize=6.4, color=OI["gray"], ha="left", va="center")
ax.set_xticks(x)
lab_b = [f"{MNAME[c]}\n{cfg_str(rows[c]['e2e_cfg'])}" for c in order]
ax.set_xticklabels(lab_b, fontsize=6.4)
ax.set_ylabel("e2e avg (K₁=128, K₂=1024,\nhq + mu, n=200 per task)", fontsize=7.8)
ax.set_ylim(31.5, 47.1)
ax.set_title("(b) e2e caliber: per-combo best\nmavg best avg & hq; mu 2nd by 0.86",
             fontsize=8.4)
ax.tick_params(labelsize=6.4)

# 双口径脚注（铁律: 口径差异必须写明）
fig.text(0.035, 0.015,
         "dual-caliber note: mass = softmax mass coverage (BP=64, B_TOK=2048, 16-sample trace mean); "
         "e2e = LongBench hotpotqa + musique (n=200 per task), avg = (hq + mu)/2, K₁=128, K₂=1024.\n"
         "same x-order as (a) (sorted by mass); rank labels show the reversal: "
         "mminmax #1→#3 (tie aavg), mavg #2→#1; e2e ordering mavg > mminmax ≈ aavg > cavg "
         "matches the main-table method ordering.\n"
         "in-bar white text = hotpotqa / musique scores of the per-combo e2e-best arm "
         "(hq ≈ 54–55 exceeds the avg axis range, shown as labels); "
         "second x-label line = best-arm (α, β, γ) under each caliber "
         "(mavg and cavg e2e-best arms differ from their mass-best arms); "
         "ccluster excluded (no e2e implementation path, mass-only); "
         "ref arm = mavg (α.125, β.375, γ.125).",
         fontsize=6.6, color="#444444", va="bottom")

fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.27,
                    wspace=0.30)
for out in OUTS:
    fig.savefig(f"{out}/fig11_e98_mass_vs_e2e.pdf")
    fig.savefig(f"{out}/fig11_e98_mass_vs_e2e.png")
print("saved fig11_e98_mass_vs_e2e")
