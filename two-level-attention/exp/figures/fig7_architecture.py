# Fig 7: PSI 架构图 v5 —— 顶会级重做（响应 10-06 用户批评「完全没有顶会水平」+ 25 对叠字）
#
# v5 设计依据（范文程序化分析 brief）：
#   范文 A（MoBA Fig1）：双 panel 白底大框；内部小圆角盒低饱和 4 色（紫/靛/绿/灰）+
#     极浅 pastel 底；文字全黑；图内文字极少且以路径导出（不放长句）。
#   范文 B（DSv3.2 Fig1）：命名模块盒（"Top-k Selector"/"Lightning Indexer"）11pt 粗体标签，
#     张量用数学符号 9.6pt + 7pt 下标，小注 6-8pt；配色 = 绿色系 4 阶明度 + 黑描边为主；
#     全图仅 2 种文字颜色。**核心规律：顶会架构图不放长句注释，用「模块名 + 数学符号 + 短语」**。
#
# v5 vs v4 的结构变化：
#   ① 视觉主链一条：KV cache → L1 块上界粗筛 → L2 双池精筛 → sparse attention（左→右粗箭头）
#     query 从顶部进（tail-32 打分 L1/L2），full-128 q 从右上绕入输出——索引与注意力维度解耦可视化
#   ② 字号四级收拢：11.5（图题）/ 9.6（模块名粗体）/ 7.5（公式与池标签）/ 6.8（脚注短语）
#     —— v4 的 6.8/6.9/7.0/7.2/7.4/8.0/9.6 七级混乱全部合并
#   ③ 长句注释全部砍成 ≤20 字符短语（v4 根因：42 字符单行标签横跨面板压过邻区标签）
#   ④ 配色收拢 4 主色（蓝 far/L1、橙 near、绿 SWA/输出、红 sink/gate）+ 对应 pastel 底，
#     面板描边统一灰 #97A1AA（v4 每面板不同色边框，视觉噪音）
#   ⑤ sink/SWA 强制通道改为输出面板内红色 sink 点 + 绿色 SWA 块 + 脚注「forced in」
#     （v4 的底部大弧线绕行箭头穿 gate 面板，是叠字与视觉混乱源之一）
#   ⑥ \max_{j∈I_i} 下标改写为 ∀j∈I_i 行内式（v4 检测器 0.48 假阳性来源）
# house style: okabe_ito 色盲友好、白底、全英文、矢量 PDF+PNG
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUTS = ["/home/wangyuanshuo02/sglang/paper/figures",
        "/home/wangyuanshuo02/two-level-attention/exp/figures"]

OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
      "red": "#D55E00", "gray": "#666666"}
EDGE = "#97A1AA"          # 面板描边统一灰
CHAIN = "#2F3A44"         # 主链箭头统一深灰蓝（顶会风格：主链不彩色）
FS_TITLE, FS_PANEL, FS_ANNOT, FS_NOTE = 11.5, 9.6, 7.5, 6.8

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "figure.dpi": 200, "savefig.bbox": "tight",
})

fig, ax = plt.subplots(figsize=(13.0, 6.3))
ax.set_xlim(0, 130)
ax.set_ylim(0, 63)
ax.axis("off")


def panel(x, y, w, h, title, fc="#ffffff"):
    """统一面板：pastel 底 + 灰描边 + 顶部粗体模块名（字号四级之第二级）"""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4",
                                fc=fc, ec=EDGE, lw=1.1, zorder=2))
    ax.text(x + w / 2, y + h - 2.0, title, ha="center", va="center",
            fontsize=FS_PANEL, fontweight="bold", color="#2A2A2A", zorder=6)


def arrow(x1, y1, x2, y2, color=CHAIN, lw=2.4, ls="-", rad=0.0, ms=17):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=ms, color=color, lw=lw,
                                 linestyle=ls, zorder=5,
                                 connectionstyle=f"arc3,rad={rad}"))


def note(x, y, s, color="#555555", fs=FS_NOTE, ha="center", style="normal",
         weight="normal"):
    ax.text(x, y, s, ha=ha, va="center", fontsize=fs, color=color,
            style=style, fontweight=weight, zorder=6)


# ================= 顶部：query 维度条（tail-32 高亮） =================
panel(34, 48.5, 70, 11, "decode query  $q\\in\\mathbb{R}^{H\\times 128}$"
      "  ·  GQA (4 q-heads / KV head)", fc="#F7F9FB")
cell_w, x0 = 0.469, 39.0
for i in range(96):                                    # 前 96 维：灰
    ax.add_patch(Rectangle((x0 + i * cell_w, 51.2), cell_w * 0.82, 2.4,
                           fc="#CFD6DD", ec="none", zorder=4))
for i in range(32):                                    # 尾 32 维：蓝（16 个最低频 RoPE 旋转对）
    ax.add_patch(Rectangle((x0 + (96 + i) * cell_w, 51.2), cell_w * 0.82, 2.4,
                           fc=OI["blue"], ec="none", zorder=4))
note(x0 + 48 * cell_w, 49.7, "first 96 dims", color="#8A939B")
note(x0 + 112 * cell_w, 49.7, "tail 32 · 16 lowest-freq RoPE pairs",
     color=OI["blue"], weight="bold")

# ================= 主链行：四个面板（y 19–46） =================
PY, PH = 19, 27

# ---- 面板 1：KV cache 四分区时间轴 ----
panel(2, PY, 27, PH, "KV cache   $[S, H_{kv}, 128]$", fc="#FCF8F1")
nb, bw, tx0, by, bh = 22, 1.0, 4.5, 34.8, 4.0
groups = [(0, 1, "#F5C9C2", OI["red"],   "sink",  5.0),
          (1, 14, "#D9E6F2", "#9FB3C2",  "far",   12.0),
          (14, 18, "#F7E4C4", OI["orange"], "near", 20.5),
          (18, 22, "#C4E5D7", OI["green"], "SWA",  24.5)]
for i0, i1, fc, ec, lab, cx in groups:
    for i in range(i0, i1):
        ax.add_patch(Rectangle((tx0 + i * bw, by), bw * 0.86, bh,
                               fc=fc, ec=ec, lw=0.7, zorder=4))
    note(cx, 40.6, lab, color=ec, fs=FS_ANNOT, weight="bold")
note(12.0, 32.8, "retrieval", color=OI["blue"])
note(20.5, 32.8, "recency", color=OI["orange"])
note(15.5, 29.9, "64-token blocks · per-block min / max")
note(15.5, 27.7, "on d′ = 32 · 4-bit (40 B / token-head)")
note(15.5, 25.3, "incremental O(1) / token", style="italic")

# ---- 面板 2：L1 块级 minmax 上界粗筛 ----
panel(36, PY, 27, PH, "L1 · upper-bound screening", fc="#F2F6FA")
ax.text(49.5, 40.9,
        r"$s_{blk}(q,i)=\sum_{d=1}^{32}\max(q_d\,k^{min}_{i,d},\; q_d\,k^{max}_{i,d})$",
        ha="center", va="center", fontsize=7.2, color="#222222", zorder=6)
note(49.5, 37.8, r"$s_{blk}\ \geq\ \langle q,k_j\rangle\ \ \forall j\in I_i$  ·  no-miss")
vals = [7.5, 6.9, 6.3, 5.8, 5.3, 4.9, 3.4, 3.1, 2.8, 2.5, 2.2,
        1.9, 1.6, 1.4, 1.2, 1.0, 0.85, 0.7]
for i, v in enumerate(vals):
    c = OI["blue"] if i < 6 else "#C6CFD8"
    ax.add_patch(Rectangle((38 + i * 1.3, 25.2), 1.0, v, fc=c, ec="none", zorder=4))
ax.plot([37.6, 61.6], [29.8, 29.8], color=OI["gray"], lw=0.9, ls="--", zorder=5)
note(61.3, 30.7, "$K_1$ cut", ha="right")
note(49.5, 22.9, "top-$K_1$ = 128 blocks", fs=FS_ANNOT, color="#333333")
note(49.5, 20.7, "scored with tail-32 $q$ only")

# ---- 面板 3：L2 双池独立配额精筛 ----
panel(70, PY, 27, PH, "L2 · two-pool fine scoring", fc="#F8F4FB")
note(76.75, 40.6, "far pool (F)", color=OI["blue"], fs=FS_ANNOT, weight="bold")
note(89.75, 40.6, "near pool (γ)", color=OI["orange"], fs=FS_ANNOT, weight="bold")
note(76.75, 38.3, "saturates 128–256", color=OI["blue"])
note(89.75, 38.3, "kept by recency", color=OI["orange"])
ax.plot([83.2, 83.2], [23.8, 34.2], color=OI["gray"], lw=0.9, ls=":", zorder=3)
far_h = [7.0, 6.3, 5.7, 5.1, 2.0, 1.7, 1.4, 1.2, 1.0, 0.85, 0.7, 0.6, 0.5, 0.45]
for i, v in enumerate(far_h):                          # far 池：稀疏少数高分 + 多数低分
    c = OI["blue"] if i < 4 else "#C9DAE9"
    ax.add_patch(Rectangle((71.6 + i * 0.75, 25.2), 0.55, v, fc=c, ec="none", zorder=4))
near_h = [2.2, 2.7, 3.2, 3.8, 4.4, 5.0, 5.6, 6.2, 6.8, 7.2, 7.6]
for i, v in enumerate(near_h):                         # near 池：recency 递增高
    c = OI["orange"] if i >= 6 else "#EDDFC3"
    ax.add_patch(Rectangle((84.4 + i * 0.75, 25.2), 0.55, v, fc=c, ec="none", zorder=4))
note(83.5, 22.9, "token-level 4-bit · separate quotas", fs=FS_ANNOT, color="#333333")
note(83.5, 20.7, "near cannot crowd out far")

# ---- 面板 4：稀疏注意力输出（时间轴） ----
panel(104, PY, 24, PH, "sparse attention", fc="#F1F7F4")
ax.text(116, 40.9, r"$o=\mathrm{Softmax}(q\,K[I]^{\top}/\sqrt{d})\,V[I]$",
        ha="center", va="center", fontsize=7.2, color="#222222", zorder=6)
note(116, 38.4, "budget $K_2$ = 1024 tokens", fs=FS_ANNOT, color="#333333")
oy = 31.5
ax.plot([106.8, 125.2], [oy, oy], color="#9FB3C2", lw=1.0, zorder=3)
ax.plot([107.8], [oy], "o", ms=5.5, color=OI["red"], zorder=6)             # sink 强制
for xt in [109.3, 110.1, 111.2, 112.4, 113.3, 114.6]:                      # far 池选中
    ax.plot([xt], [oy], "|", ms=6.0, mew=1.6, color=OI["blue"], zorder=6)
for i in range(10):                                                        # near 池选中
    ax.plot([115.6 + i * 0.44], [oy], "|", ms=6.0, mew=1.6,
            color=OI["orange"], zorder=6)
ax.add_patch(Rectangle((121.0, oy - 0.85), 3.5, 1.7, fc="#C4E5D7",          # SWA 强制
                       ec=OI["green"], lw=0.8, zorder=5))
for cx, lab, col in [(107.8, "sink", OI["red"]), (111.9, "far", OI["blue"]),
                     (117.7, "near", OI["orange"]), (122.75, "SWA", OI["green"])]:
    note(cx, 29.3, lab, color=col)
note(116, 25.6, "sink + SWA forced in", fs=FS_ANNOT, color="#333333")
note(116, 23.4, "fused kernel · causal by construction")

# ================= 主链箭头 + gap 短标签 =================
for xa, xb, lab, lx in [(29.5, 35.5, "min/max", 32.5),
                        (63.5, 69.5, "top-$K_1$", 66.5),
                        (97.5, 103.5, "selected", 100.5)]:
    arrow(xa, 32.5, xb, 32.5, lw=2.4)
    note(lx, 34.6, lab, color=CHAIN)

# query 进入两级索引（tail-32 打分）；full-128 q 绕右上进入最终注意力
arrow(50.0, 48.5, 49.5, 46.5, color=OI["gray"], lw=1.6, ms=13)
arrow(83.0, 48.5, 83.5, 46.5, color=OI["gray"], lw=1.6, ms=13)
arrow(104.0, 51.5, 117.0, 46.5, color=OI["gray"], lw=1.5, ms=13, rad=-0.12)
note(112.0, 51.9, "full 128-dim $q$", color=OI["gray"], style="italic")

# ================= 底部：感知型 gate（虚线红箭头上探 L2 far 池） =================
panel(36, 3.0, 58, 11.5, "per-request gate · safety switch", fc="#FCF2EF")
note(65, 10.0, "prefill far-stat predicts decode far mass (corr 0.92)",
     color="#55332F", fs=7.2)
note(65, 7.8, "layers with negligible far mass → skip far retrieval",
     color="#55332F", fs=7.2)
note(65, 5.6, "one scalar / layer · zero extra kernel", color="#55332F", fs=7.2)
arrow(78, 14.9, 78, 18.6, color=OI["red"], lw=1.4, ls="--", ms=12)
note(79.6, 16.7, "skip far pool", color=OI["red"], style="italic", ha="left")

# ================= 图题 =================
ax.set_title("PSI: Position-Stable Indexing — training-free upper-bound block screening "
             "with far/near budget partitioning",
             fontsize=FS_TITLE, fontweight="bold", pad=10)

for out in OUTS:
    for name in ("fig7_tli_architecture", "fig7_psi_architecture"):
        fig.savefig(f"{out}/{name}.pdf")
        fig.savefig(f"{out}/{name}.png")
print("saved fig7 v5 (top-conf redesign: 4-tier type ladder, single main chain, "
      "short phrases only)")

# ================= 自检（data 坐标）：文字间距 / 面板越界 / 最小字号 =================
fig.canvas.draw()
inv = ax.transData.inverted()
ren = fig.canvas.get_renderer()
PANELS = [(34, 48.5, 70, 11), (2, PY, 27, PH), (36, PY, 27, PH),
          (70, PY, 27, PH), (104, PY, 24, PH), (36, 3.0, 58, 11.5)]
PAD = 0.55
problems = []
boxes = []
for t in ax.texts:
    bb = t.get_window_extent(renderer=ren)
    x0, y0 = inv.transform((bb.x0, bb.y0))
    x1, y1 = inv.transform((bb.x1, bb.y1))
    boxes.append((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1),
                  t.get_text()[:24], t.get_fontsize()))
for i in range(len(boxes)):
    for j in range(i + 1, len(boxes)):
        a, b = boxes[i], boxes[j]
        ox = min(a[2], b[2]) - max(a[0], b[0])
        oy = min(a[3], b[3]) - max(a[1], b[1])
        if ox > 0.05 and oy > 0.05:
            problems.append(f"TEXT-TEXT overlap: {a[4]!r} x {b[4]!r} "
                            f"(ox={ox:.2f}, oy={oy:.2f})")
for x0, y0, x1, y1, txt, fs in boxes:
    if fs < 6.0:
        problems.append(f"FONT<6pt: {txt!r} fs={fs}")
    if x0 < -0.3 or x1 > 130.3 or y0 < -0.3 or y1 > 63.3:
        problems.append(f"OUT-OF-FIGURE: {txt!r} bbox=({x0:.1f},{y0:.1f},{x1:.1f},{y1:.1f})")
    for px, py, pw, ph in PANELS:      # 部分跨面板边界的文字（gap 标签误入面板）
        ix = min(x1, px + pw) - max(x0, px)
        iy = min(y1, py + ph) - max(y0, py)
        if ix > 0.05 and iy > 0.05 and not (x0 > px - PAD and x1 < px + pw + PAD
                                            and y0 > py - PAD and y1 < py + ph + PAD):
            problems.append(f"CROSSES-PANEL-EDGE: {txt!r} bbox=({x0:.1f},{y0:.1f},"
                            f"{x1:.1f},{y1:.1f}) vs panel=({px},{py},{pw},{ph})")
if problems:
    print("SELF-CHECK FAIL (%d):" % len(problems))
    for p in problems:
        print("  " + p)
    raise SystemExit(1)
print("self-check OK: %d texts, 0 overlaps / 0 edge-crossings / min fs=%.1f"
      % (len(boxes), min(b[5] for b in boxes)))
