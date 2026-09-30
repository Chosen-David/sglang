# Fig 7: TLI 架构图（论文级，matplotlib 矢量绘制，重画版）
# 修正三处：① q 箭头接 L1/L2 打分（原错接 D'）② 主链 L1→L2→分区（原 L1 跳过 L2）
# ③ D' 画已废弃静态层掩码 → 改为论文现行设计：per-request far-stat 感知 gate（安全开关）
# 并清除全部内部代号（E3/E4c/E5b/E6/E8-2/B'/D' 标签等）
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUTS = ["/home/wangyuanshuo02/sglang/paper/figures",
        "/home/wangyuanshuo02/two-level-attention/exp/figures"]
plt.rcParams.update({"font.size": 9, "figure.dpi": 200, "savefig.bbox": "tight"})
C = {"blue": "#2f6f9f", "red": "#c1443c", "green": "#3a7d44", "orange": "#d98e32",
     "purple": "#7b5aa6", "gray": "#8a8a8a", "teal": "#2a9d8f", "light": "#eef3f8"}

fig, ax = plt.subplots(figsize=(9.6, 4.6))
ax.set_xlim(0, 104)
ax.set_ylim(0, 50)
ax.axis("off")


def box(x, y, w, h, title, lines, fc, ec, title_fs=9.5, fs=7.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35",
                                fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h - 2.7, title, ha="center", va="center",
            fontsize=title_fs, fontweight="bold", color=ec)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 5.6 - i * 2.6, ln, ha="center", va="center",
                fontsize=fs, color="#333333")


def arrow(x1, y1, x2, y2, color="#555555", style="-|>", lw=1.5, ls="-",
          connectionstyle="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=14, color=color, lw=lw,
                                 linestyle=ls, connectionstyle=connectionstyle))


# ============ 顶部：查询 ============
box(38, 43, 28, 6.5, "Query (per decode step)",
    ["q ∈ R^{H×128},  GQA: H = 4·H_kv"], C["light"], C["gray"])

# ============ 主链三段（左→右） ============
# ① 输入与索引
box(2, 21, 24, 17, "KV Cache + Block Index",
    ["[S, H_kv, 128] bf16, S up to 131K",
     "64-token blocks",
     "min/max bounds on the",
     "position-stable tail subspace",
     "d' = 32, 4bit (40B/token-head)"],
    "#f5ede1", C["orange"])

# ② L1 粗筛
box(30, 21, 26, 17, "L1: Upper-Bound Coarse",
    ["s_blk = Σ max(q_d k^min, q_d k^max)",
     "≥ max ⟨q, k_j⟩ over block",
     "no-miss: high-scoring block",
     "always survives",
     "top-K1 = 128 blocks"],
    "#e8eff6", C["blue"], fs=7.0)

# ③ L2 分区精筛
box(59, 21, 22, 17, "L2: Partitioned Fine",
    ["token-level 4bit scoring",
     "far pool: quota F ≥ 64",
     "(saturates 128–256)",
     "near pool: γ·near blocks",
     "anti-eviction under GQA"],
    "#f2ecf7", C["purple"])

# ④ 输出
box(84, 21, 18, 17, "Sparse Attention",
    ["K2 = 1024 tokens",
     "sink + sliding window",
     "forced, no budget",
     "fused kernel,",
     "causal by construction"],
    "#efe9e2", C["teal"])

# ============ 箭头 ============
arrow(52, 43, 42, 38, color=C["gray"])              # q → L1（块上界打分）
arrow(52, 43, 69, 38, color=C["gray"])              # q → L2（token 精筛打分）
arrow(26, 29.5, 30, 29.5, color=C["blue"])          # 块索引 → L1
arrow(56, 29.5, 59, 29.5, color=C["blue"])          # L1 → L2（选中块的候选 token）
arrow(81, 29.5, 84, 29.5, color=C["teal"])          # L2 → 稀疏注意力
# sink/swa 直接进输出的强制通道（不占分区预算，从主链下方绕行）
arrow(14, 21, 90, 21, color=C["teal"], ls="--", lw=1.2,
      connectionstyle="arc3,rad=-0.15")

# ============ 感知 gate（安全开关，虚线控制，置于左下） ============
box(2, 3, 46, 11, "Per-request Gate (safety switch)",
    ["far-stat signal from prefill correlates 0.924",
     "with decode far mass; on layers with negligible",
     "far mass, skip far retrieval — blocks the reverse error"],
    "#f7f7f7", C["red"], title_fs=8.8, fs=7.0)
arrow(42, 14, 66, 21, color=C["red"], ls="--", lw=1.2)   # gate 控制 L2 far 池

# 强制通道标签（置于弧线下方右侧，避开 gate 框）
ax.text(76, 15.5, "sink / sliding window forced path\n(no partitioned budget)",
        fontsize=7.0, color=C["teal"], ha="center", style="italic")

ax.set_title("TLI: training-free two-level sparse attention indexer",
             fontsize=11, fontweight="bold", pad=12)

for out in OUTS:
    fig.savefig(f"{out}/fig7_tli_architecture.pdf")
    fig.savefig(f"{out}/fig7_tli_architecture.png")
print("saved fig7 (redrawn)")
