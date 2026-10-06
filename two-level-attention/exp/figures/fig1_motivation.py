# Fig 1 (论文 Fig.1 动机图): 三个观察 → 三个设计蕴含
# (a) 位置三层台阶: sink/near/far mass 分布 → 保留区 + 剩余预算分配
# (b) GQA far-mass 单 kv-head 独占 → kv-head 级选择 + 独立 far 配额
# (c) far 预算 128-1024 饱和曲线 → 紧预算下的预算分配是主要矛盾
# house style: okabe_ito, 白底, 全英文, 矢量 PDF+PNG
import json
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
h1 = json.load(open(f"{RES}/h1_full_decomposition.json"))
samples = [k for k in h1.keys() if k.startswith("lb_")][:8]

# (a) 每样本 sink/near/far mean
sink = np.array([h1[s]["sink_mean"] for s in samples])
near = np.array([h1[s]["near_mean"] for s in samples])
far = np.array([h1[s]["far_mean"] for s in samples])
short = [s.replace("lb_", "").replace("_en", "")[:14] for s in samples]

# (b) far-heavy 样本的 per-kv-head far mass 分布
fp_samples = [(s, np.array(h1[s]["far_profile"])) for s in samples
              if h1[s].get("far_max", 0) > 0.3][:3]
n_heads = min(len(p) for _, p in fp_samples)

# (c) far 预算饱和
sat = json.load(open(f"{RES}/e5b_far_tokens_sensitivity.json"))
budgets = sorted(int(k) for k in sat.keys())
sat_mean = np.array([np.mean(sat[str(b)]) for b in budgets])

# ---------- 画布 ----------
fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.4))

# (a) 三层台阶
ax = axes[0]
x = np.arange(len(samples))
w = 0.27
ax.bar(x - w, sink, w, color=OI["red"], label="sink")
ax.bar(x, near, w, color=OI["orange"], label="near (SWA)")
ax.bar(x + w, far, w, color=OI["blue"], label="far (retrieval)")
ax.axhline(sink.mean(), color=OI["red"], ls=":", lw=0.9, alpha=0.7)
ax.axhline(near.mean(), color=OI["orange"], ls=":", lw=0.9, alpha=0.7)
ax.axhline(far.mean(), color=OI["blue"], ls=":", lw=0.9, alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(short, rotation=38, ha="right", fontsize=6.4)
ax.set_ylabel("attention mass share", fontsize=8)
ax.set_title("(a) three-tier position profile\nsink / near / far mass per sample", fontsize=8.6)
ax.legend(fontsize=6.8, frameon=False, loc="upper right")
ax.tick_params(labelsize=7)

# (b) GQA far-mass 头部集中
ax = axes[1]
xm = np.arange(n_heads)
for i, (s, p) in enumerate(fp_samples):
    pp = p[:n_heads] / p[:n_heads].sum()
    ax.bar(xm + (i - 1) * 0.26, pp, 0.24,
           color=[OI["blue"], OI["sky"], OI["gray"]][i],
           label=s.replace("lb_", "")[:16], alpha=0.9 if i == 0 else 0.65)
top_share = max((p[:n_heads] / p[:n_heads].sum()).max() for _, p in fp_samples)
ax.annotate(f"one kv-head holds {top_share:.0%}\nof all far mass",
            xy=(np.argmax(fp_samples[0][1][:n_heads] / fp_samples[0][1][:n_heads].sum()),
                top_share), xytext=(n_heads * 0.42, top_share * 0.92),
            fontsize=7.2, color="#222222",
            arrowprops=dict(arrowstyle="->", color="#555555", lw=0.9))
ax.set_xticks(xm)
# 稀疏化刻度标签（每 4 个标一个，刻度位置/柱形不动）: 36 个 h 标签过密,
# 相邻标签互叠约半字符宽（检测器 25 对，本次修复）
ax.set_xticklabels([f"h{i}" if i % 4 == 0 else "" for i in range(n_heads)],
                   fontsize=7)
ax.set_xlabel("kv-head (GQA group)", fontsize=8)
ax.set_ylabel("far-mass share within sample", fontsize=8)
ax.set_title("(b) far mass concentrates\non few kv-heads (GQA)", fontsize=8.6)
ax.legend(fontsize=6.4, frameon=False, loc="upper right")
ax.tick_params(labelsize=7)

# (c) far 预算饱和
ax = axes[2]
ax.plot(budgets, sat_mean, "o-", color=OI["blue"], lw=1.6, ms=4.5)
for b, v in zip(budgets, sat_mean):
    ax.annotate(f"{v:.3f}", (b, v), textcoords="offset points",
                xytext=(0, 6), fontsize=6.4, ha="center", color="#555555")
ax.set_xscale("log", base=2)
ax.set_xticks(budgets)
ax.set_xticklabels([str(b) for b in budgets], fontsize=7)
ax.set_xlabel("far token quota $F$ (per kv-head)", fontsize=8)
ax.set_ylabel("far mass coverage", fontsize=8)
ax.set_ylim(0.94, 1.005)
ax.set_title("(c) far quota saturates at 128\u2013256\n\u2192 allocation, not budget size, is the lever", fontsize=8.6)
ax.tick_params(labelsize=7)

fig.tight_layout(w_pad=2.2)
for out in OUTS:
    fig.savefig(f"{out}/fig1_motivation.pdf")
    fig.savefig(f"{out}/fig1_motivation.png")
print("saved fig1_motivation (3-panel observations)")
