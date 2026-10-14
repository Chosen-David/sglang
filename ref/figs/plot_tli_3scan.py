#!/usr/bin/env python3
"""TLI 3D abg full scan — 6 complementary high-dimensional views."""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

SRC = "/home/wangyuanshuo02/two-level-attention/exp/trace/results/e98_abg_full_grid.json"
OUT_DIR = "/home/wangyuanshuo02/sglang/ref/figs"

# ──────────────────────────────────────────
# 1. parse
# ──────────────────────────────────────────
raw = json.load(open(SRC))
tasks = sorted(raw["per_sample"].keys())
n_tasks = len(tasks)

grid = {}
for task in tasks:
    for k, v in raw["per_sample"][task].items():
        if not k.startswith("mavg_"):
            continue
        parts = k[5:].split("_")
        a, b, g = float(parts[0][1:]), float(parts[1][1:]), float(parts[2][1:])
        grid.setdefault((a, b, g), []).append(v)

mean_grid = {k: np.mean(v) for k, v in grid.items()}
a_vals = sorted({k[0] for k in mean_grid})
b_vals = sorted({k[1] for k in mean_grid})
g_vals = sorted({k[2] for k in mean_grid})
n_cfg = len(mean_grid)
print(f"[info] tasks={n_tasks}, a={len(a_vals)}, b={len(b_vals)}, g={len(g_vals)}, total={n_cfg}")

vmin = min(mean_grid.values())
vmax = max(mean_grid.values())
print(f"[info] mass range: {vmin:.4f} -- {vmax:.4f}")

def get_mass(a, b, g):
    return mean_grid.get((a, b, g), np.nan)

# ──────────────────────────────────────────
# Helper: aggregate
# ──────────────────────────────────────────
def agg_mean(fixed=None):
    """Given a dict of one fixed axis, return mean over the other two."""
    ss = []
    for (a, b, g), m in mean_grid.items():
        if fixed:
            axis, val = fixed
            if (axis == "a" and a != val) or (axis == "b" and b != val) or (axis == "g" and g != val):
                continue
        ss.append(m)
    return np.mean(ss) if ss else np.nan


# ──────────────────────────────────────────
# FIGURE 1 — Marginal waterfall (independent impact per parameter)
# ──────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

# (1A) alpha sweep — each curve is one gamma, averaged over beta
for gi, g in enumerate(g_vals):
    ys = np.array([np.nanmean([get_mass(a, b, g) for b in b_vals])
                   for a in a_vals])
    c = plt.cm.turbo(gi / max(1, len(g_vals) - 1))
    lab = f"g={g}" if gi % max(1, len(g_vals) // 5) == 0 or gi == len(g_vals) - 1 else None
    axes[0].plot(a_vals, ys, "-o", ms=3, color=c, label=lab, lw=1.2)
axes[0].set_xlabel("alpha (near budget share)")
axes[0].set_ylabel("mass coverage (mean over other axes)")
axes[0].set_title(r"alpha sweep | each curve = one gamma, avg-ed over beta")
axes[0].legend(fontsize=7, ncol=2, loc="lower right", framealpha=0.9)
axes[0].grid(lw=0.3); axes[0].set_ylim(vmin - 0.01, vmax + 0.005)

# (1B) beta sweep — each curve is one gamma, averaged over alpha
for gi, g in enumerate(g_vals):
    ys = np.array([np.nanmean([get_mass(a, b, g) for a in a_vals])
                   for b in b_vals])
    c = plt.cm.turbo(gi / max(1, len(g_vals) - 1))
    lab = f"g={g}" if gi % max(1, len(g_vals) // 5) == 0 or gi == len(g_vals) - 1 else None
    axes[1].plot(b_vals, ys, "-o", ms=3, color=c, label=lab, lw=1.2)
axes[1].set_xlabel("beta (far budget share)")
axes[1].set_ylabel("mass coverage (mean over other axes)")
axes[1].set_title(r"beta sweep | each curve = one gamma, avg-ed over alpha")
axes[1].legend(fontsize=7, ncol=2, loc="lower right", framealpha=0.9)
axes[1].grid(lw=0.3); axes[1].set_ylim(vmin - 0.01, vmax + 0.005)

# (1C) gamma sweep — each curve is one alpha, averaged over beta
for ai, a in enumerate(a_vals):
    ys = np.array([np.nanmean([get_mass(a, b, g) for b in b_vals])
                   for g in g_vals])
    c = plt.cm.turbo(ai / max(1, len(a_vals) - 1))
    lab = f"a={a}" if ai % max(1, len(a_vals) // 5) == 0 or ai == len(a_vals) - 1 else None
    axes[2].plot(g_vals, ys, "-o", ms=3, color=c, label=lab, lw=1.2)
axes[2].set_xlabel("gamma (near/far split)")
axes[2].set_ylabel("mass coverage (mean over other axes)")
axes[2].set_title(r"gamma sweep | each curve = one alpha, avg-ed over beta")
axes[2].legend(fontsize=7, ncol=2, loc="lower right", framealpha=0.9)
axes[2].grid(lw=0.3); axes[2].set_ylim(vmin - 0.01, vmax + 0.005)

fig.suptitle(f"TLI alpha-beta-gamma 3D scan (n={n_tasks} tasks, {n_cfg} configs)", fontsize=12)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT_DIR}/fig1_abg_marginal_sweeps.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig)
print("[ok] fig1 saved")

# ──────────────────────────────────────────
# FIGURE 2 — Best-up surface: alpha x beta (max over gamma) + optimal-gamma contour
# ──────────────────────────────────────────
best_mass = np.full((len(a_vals), len(b_vals)), np.nan)
best_gamma = np.full((len(a_vals), len(b_vals)), np.nan)
for i, a in enumerate(a_vals):
    for j, b in enumerate(b_vals):
        cands = [(get_mass(a, b, g), g) for g in g_vals]
        cands = [(m, g) for m, g in cands if not np.isnan(m)]
        if cands:
            m, g = max(cands, key=lambda x: x[0])
            best_mass[i, j] = m
            best_gamma[i, j] = g

fig, ax = plt.subplots(1, 1, figsize=(9, 7))
im = ax.pcolormesh(b_vals, a_vals, best_mass, shading="auto", cmap="turbo",
                   vmin=vmin, vmax=vmax)
CS = ax.contour(b_vals, a_vals, best_gamma, levels=g_vals,
                colors="white", linewidths=0.8, alpha=0.7)
ax.clabel(CS, fontsize=7, fmt="g=%.3f", colors="white")
ij = np.unravel_index(np.nanargmax(best_mass), best_mass.shape)
ax.plot(b_vals[ij[1]], a_vals[ij[0]], "*", c="white", ms=15,
        markeredgecolor="black", markeredgewidth=1.5)
ax.annotate(f"best={best_mass[ij]:.4f}", (b_vals[ij[1]], a_vals[ij[0]]),
            textcoords="offset points", xytext=(10, 5),
            fontsize=9, color="white", fontweight="bold")
ax.set_xlabel("beta (far budget share)")
ax.set_ylabel("alpha (near budget share)")
ax.set_title(r"best mass (max over gamma) | white iso-lines = optimal gamma at each (alpha,beta)")
ax.set_aspect("equal")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mass coverage")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT_DIR}/fig2_ab_best_maxgamma.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig)
print("[ok] fig2 saved")

# ──────────────────────────────────────────
# FIGURE 3 — Gamma-slice heatmaps (4 panels, uniform color scale)
# ──────────────────────────────────────────
n_slices = 4
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()
lev = np.linspace(vmin, vmax, 16)
for idx, gi in enumerate(np.linspace(0, len(g_vals)-1, n_slices, dtype=int)):
    ax = axes[idx]
    g = g_vals[gi]
    M = np.array([[get_mass(a, b, g) for b in b_vals] for a in a_vals])
    cf = ax.contourf(b_vals, a_vals, M, levels=lev, cmap="turbo", extend="both")
    ij = np.unravel_index(np.nanargmax(M), M.shape)
    ax.plot(b_vals[ij[1]], a_vals[ij[0]], "*", c="white", ms=15,
            markeredgecolor="black", markeredgewidth=1)
    ax.annotate(f"best={M[ij]:.4f}", (b_vals[ij[1]], a_vals[ij[0]]),
                textcoords="offset points", xytext=(10, 5),
                fontsize=9, color="white", fontweight="bold")
    ax.set_xlabel("beta")
    ax.set_ylabel("alpha")
    ax.set_title(f"gamma = {g:.3f}")
    fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.03)

fig.suptitle("TLI 3D scan — gamma slices (uniform color)", fontsize=13)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT_DIR}/fig3_gamma_slice_heatmaps.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig)
print("[ok] fig3 saved")

# ──────────────────────────────────────────
# FIGURE 4 — 3D scatter (alpha,beta,gamma) colored by mass, marker=gamma
# ──────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection="3d")
marker_cycle = ["o", "^", "s", "D", "v", "p", "*", "h"]
for gi, g in enumerate(g_vals):
    xs, ys, zs, cs = [], [], [], []
    for (a, b, gg), m in mean_grid.items():
        if gg == g:
            xs.append(a); ys.append(b); zs.append(g); cs.append(m)
    ax.scatter(xs, ys, zs, c=cs, cmap="turbo", s=50,
               marker=marker_cycle[gi % len(marker_cycle)],
               vmin=vmin, vmax=vmax, edgecolors="k", linewidths=0.3,
               label=(f"g={g}" if gi % max(1, len(g_vals) // 5) == 0 else None))

cbar = fig.colorbar(ax.collections[0], ax=ax, fraction=0.03, pad=0.08, label="mass coverage")
ax.set_xlabel("alpha")
ax.set_ylabel("beta")
ax.set_zlabel("gamma")
ax.set_title("TLI 3D scatter | color=perf, marker=gamma-slice")
ax.view_init(elev=25, azim=-60)
ax.legend(loc="upper left", fontsize=7, ncol=2, framealpha=0.9)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT_DIR}/fig4_3d_scatter.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig)
print("[ok] fig4 saved")

# ────────────────────────────────────────--
# FIGURE 5 — Pairwise 2D marginal projections (average over third axis)
# ────────────────────────────────────────--
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# alpha-beta (avg gamma)
M_ab = np.array([[np.nanmean([get_mass(a, b, g) for g in g_vals])
                   for b in b_vals] for a in a_vals])
im = axes[0].pcolormesh(b_vals, a_vals, M_ab, shading="auto", cmap="turbo",
                         vmin=vmin, vmax=vmax)
axes[0].set_xlabel("beta"); axes[0].set_ylabel("alpha")
axes[0].set_title("alpha-beta  (avg over gamma)")
axes[0].set_aspect("equal")
fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.03)

# alpha-gamma (avg beta)
M_ag = np.array([[np.nanmean([get_mass(a, b, g) for b in b_vals])
                   for g in g_vals] for a in a_vals])
im = axes[1].pcolormesh(g_vals, a_vals, M_ag, shading="auto", cmap="turbo",
                         vmin=vmin, vmax=vmax)
axes[1].set_xlabel("gamma"); axes[1].set_ylabel("alpha")
axes[1].set_title("alpha-gamma  (avg over beta)")
axes[1].set_aspect("equal")
fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.03)

# beta-gamma (avg alpha)
M_bg = np.array([[np.nanmean([get_mass(a, b, g) for a in a_vals])
                   for g in g_vals] for b in b_vals])
im = axes[2].pcolormesh(g_vals, b_vals, M_bg, shading="auto", cmap="turbo",
                         vmin=vmin, vmax=vmax)
axes[2].set_xlabel("gamma"); axes[2].set_ylabel("beta")
axes[2].set_title("beta-gamma  (avg over alpha)")
axes[2].set_aspect("equal")
fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.03)

fig.suptitle("Pairwise marginal 2D projections (each averages over the remaining axis)", fontsize=12)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT_DIR}/fig5_2d_marginal_projections.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig)
print("[ok] fig5 saved")

# ──────────────────────────────────────────
# FIGURE 6 — Multi-path 3D interaction scan
#     Show how mass varies along different paths through (alpha,beta,gamma) space.
#     Each path is a different "direction" in the 3D grid.
# ──────────────────────────────────────────
print(f"[info] global best = {max(mean_grid, key=mean_grid.get)}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

# --- (6A) Single-axis sweeps from origin ---
ax = axes[0]
lab_b, lab_g = "vary alpha (b=0,g=0)", None
ys_a = [get_mass(a, 0.0, 0.0) for a in a_vals]
ys_b = [get_mass(0.0, b, 0.0) for b in b_vals]
ys_g = [get_mass(0.0, 0.0, g) for g in g_vals]

# align x-axis: use normalized parameter value [0, 1] for fair comparison
a_norm = list(a_vals)
b_norm = list(b_vals)
g_norm = list(g_vals)
ax.plot(a_norm, ys_a, "D-", color="C0", lw=2, ms=5, label="vary alpha (b=0, g=0)")
ax.plot(b_norm, ys_b, "s-", color="C1", lw=2, ms=5, label="vary beta  (a=0, g=0)")
ax.plot(g_norm, ys_g, "^-", color="C2", lw=2, ms=5, label="vary gamma (a=0, b=0)")
ax.axhline(get_mass(0.0, 0.0, 0.0), color="gray", lw=1, ls=":", label="baseline (0,0,0)")
ax.set_xlabel("parameter value")
ax.set_ylabel("mass coverage")
ax.set_title("6A  single-axis from origin")
ax.legend(fontsize=8); ax.grid(lw=0.3)
ax.set_ylim(vmin - 0.01, vmax + 0.005)

# --- (6B) Cumulative activation: baseline -> turn on each axis one by one ---
ax = axes[1]
baseline = get_mass(0.0, 0.0, 0.0)
# step 1: best alpha given b=0,g=0
a1 = a_vals[np.nanargmax(ys_a)]
m1 = get_mass(a1, 0.0, 0.0)
# step 2: best beta given a=a1,g=0
ys_b2 = [get_mass(a1, b, 0.0) for b in b_vals]
b1 = b_vals[np.nanargmax(ys_b2)]
m2 = get_mass(a1, b1, 0.0)
# step 3: best gamma given a=a1,b=b1
ys_g2 = [get_mass(a1, b1, g) for g in g_vals]
g1 = g_vals[np.nanargmax(ys_g2)]
m3 = get_mass(a1, b1, g1)

# Also show optimal-greedy vs brute force
brute = max(mean_grid.values())

stages = ["baseline\n(0,0,0)", f"α={a1}\n(step 1)", f"α={a1},β={b1}\n(step 2)", f"α={a1},β={b1}\ng={g1} (full)"]
masses = [baseline, m1, m2, m3]
gain = [0.0, m1 - baseline, m2 - m1, m3 - m2]

# bar = total, overlay incremental gain
x = np.arange(len(stages))
bars = ax.bar(x, masses, color=["lightgray", "#90caf9", "#42a5f5", "#1565c0"], edgecolor="k")
# annotate with delta
for i, (m, g) in enumerate(zip(masses, gain)):
    ax.text(i, m + 0.001, f"{m:.4f}\n(+{g:.4f})", ha="center", va="bottom", fontsize=9)
ax.axhline(brute, color="red", ls="--", lw=1, label=f"global best {brute:.4f}")
ax.set_xticks(x); ax.set_xticklabels(stages, fontsize=9)
ax.set_ylabel("mass coverage")
ax.set_title("6B  cumulative activation waterfall")
ax.legend(fontsize=9); ax.grid(lw=0.3, axis="y")
ax.set_ylim(baseline - 0.01, vmax + 0.01)

# --- (6C) Diagonal path: a=b=g (isotropic turn-on) ---
ax = axes[2]
diag_vals = sorted(set(a_vals) & set(b_vals) & set(g_vals))
diag_mass = [get_mass(v, v, v) for v in diag_vals]
ax.plot(diag_vals, diag_mass, "o-", color="purple", lw=2, ms=7, label="α=β=γ (isotropic)")
# Also show: vary 2 of 3 together
diag_ag = [get_mass(v, 0.0, v) for v in diag_vals]
ax.plot(diag_vals, diag_ag, "s--", color="C0", lw=1.5, ms=5, label="α=γ, β=0")
diag_bg = [get_mass(0.0, v, v) for v in diag_vals]
ax.plot(diag_vals, diag_bg, "^--", color="C1", lw=1.5, ms=5, label="β=γ, α=0")
diag_ab = [get_mass(v, v, 0.0) for v in diag_vals]
ax.plot(diag_vals, diag_ab, "D--", color="C2", lw=1.5, ms=5, label="α=β, γ=0")

ax.axhline(baseline, color="gray", lw=1, ls=":", label="baseline")
ax.set_xlabel("parameter value")
ax.set_ylabel("mass coverage")
ax.set_title("6C  multi-axis cooperative sweeps (2 or 3 vars tied)")
ax.legend(fontsize=8); ax.grid(lw=0.3)
ax.set_ylim(vmin - 0.01, vmax + 0.005)

# --- (6D) Contour of ALL configurations overlaid (alpha-beta, gamma as overlay color) ---
ax = axes[3]
# scatter all (a,b) points, raw color from gamma (not mass) -> see gamma distribution
cmap_g = plt.cm.viridis
all_a, all_b, all_g, all_m = [], [], [], []
for (a, b, g), m in mean_grid.items():
    all_a.append(a); all_b.append(b); all_g.append(g); all_m.append(m)
sc = ax.scatter(all_a, all_b, c=all_m, cmap="turbo", s=15, vmin=vmin, vmax=vmax)
fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03, label="mass coverage")
# convex hull of the (a,b) space
ax.plot([0, a_vals[-1]], [0, 0], "k_", lw=0.5)
ax.plot([0, a_vals[-1]], [0, b_vals[-1]], "k--", lw=0.5, alpha=0.4)
ax.set_xlabel("alpha"); ax.set_ylabel("beta")
ax.set_title("6D  all (alpha,beta) points colored by mass (gamma varies)")
ax.set_aspect("equal"); ax.grid(lw=0.3)

fig.suptitle("TLI 3D interaction — combined effects along different paths through parameter space", fontsize=13)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT_DIR}/fig6_combined_interaction_paths.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig)
print("[ok] fig6 saved")

print(f"[done] 6 figures written to {OUT_DIR}")
