# E64 帕累托图：质量（mass coverage，E64b 真实 trace）vs 成本（选择阶段 GPU ms，E64c-1/c-2 成本模型外推）
# 成本口径（合成 microbench，单层单请求 S=64K，H20）：
#   块 build：minmax ~0.011ms(d32) / avg ~0.005ms
#   块打分：minmax 0.082ms / avg 0.004ms
#   页 topk：0.02ms（块数 1024 量级）
#   细筛 gather+bmm：0.026ms(d4)/0.071ms(d32)/0.101ms(d128) × (bp/128 比例外推 gather 部分)
#   token topk：0.06ms
#   贪心 build（cavg/hybrid）：12s/层在线不可行→增量摊销（E7），此处不计入（诚实标注：prefill 一次性）
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/home/wangyuanshuo02/two-level-attention/exp/trace/results"
F = "/home/wangyuanshuo02/two-level-attention/exp/figures"
d = json.load(open(f"{R}/e64b_bp_gamma.json"))


def arm(k):
    vals = [d[s][k] for s in d if k in d[s]]
    return sum(vals) / len(vals) if vals else None


# 细筛成本外推：gather 部分 ∝ bp，bmm 部分 ∝ bp×d
def cost_ms(bp, d, gam):
    pg = {4: 0.026, 32: 0.071, 128: 0.101}          # A_gather_bmm @ bp=128 页池（8192 token）
    go = {4: 0.009, 32: 0.051, 128: 0.050}           # gather only @ bp=128
    scale = (bp * 64) / 8192
    gemv = (pg[d] - go[d]) * scale + go[d] * scale   # 两部分都 ∝ token 数
    return 0.011 + 0.082 + 0.02 + gemv + 0.06        # build+块分+页topk+细筛+token topk


PTS = []
for bp in (16, 32, 64, 128, 256):
    for g in (0.5, 0.75, 1.0):
        cov = arm(f"mavg_bp{bp}_g{g}")
        if cov:
            PTS.append((cost_ms(bp, 32, g), cov, f"mavg bp={bp} g={g}", "tab:blue", "o", 32))
    cov = arm(f"mono_bp{bp}")
    PTS.append((cost_ms(bp, 32, 1), cov, f"mono bp={bp}", "k", "^", 32))
# 降维臂：bp=128 d=4（sup_wsvd 降维后的细筛成本）
cov = arm("mavg_bp128_g0.5")
PTS.append((cost_ms(128, 4, 0.5), cov, "mavg bp=128 g=0.5 d=4 (sup_wsvd)", "tab:red", "*", 4))

fig, ax = plt.subplots(figsize=(8, 5.8))
for c, v, lab, col, mk, dd in PTS:
    size = 60 if dd == 32 else 200
    ax.scatter(c, v, color=col, marker=mk, s=size, zorder=3 if dd == 4 else 2)
    if bp_lab := [p for p in (128, 256) if f"bp={p}" in lab]:
        ax.annotate(lab, (c, v), textcoords="offset points", xytext=(6, 4), fontsize=7)
ax.set_xlabel("selection-stage GPU cost (ms, synthetic microbench, S=64K, single layer/request)")
ax.set_ylabel("mass coverage (real trace, 16 samples)")
ax.grid(alpha=0.3)
ax.set_title("E64 Pareto: quality vs selection cost (cost = synthetic microbench, annotated)", fontsize=10)
fig.tight_layout()
fig.savefig(f"{F}/e64_pareto.png", dpi=150)
fig.savefig(f"{F}/e64_pareto.pdf")
print("saved e64_pareto.{png,pdf}")
for c, v, lab, *_ in sorted(PTS):
    print(f"  {lab:38s} cost={c:.3f}ms cov={v:.4f}")
