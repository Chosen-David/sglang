# 心方案 H2-B：真实 trace 候选复用偏斜统计（用户指示「作图统计 k_avg 和 q 的选中01图」）
# 对每个 (task, layer)：取 trace 全部真实 q（位置 >4096 才有意义），逐 q 跑
# 生产版 L1 块选择（select() 的 L1 部分：子空间区间算术 + 因果 mask + top-K1 +
# 滑窗强制块）→ blk_onehot [Nq, nblk] 01 矩阵，统计：
#   [1] 01 矩阵位图（queries 按位置排序 × pages，near/far 边界线）
#   [2] f_j 激活频率直方图（page 被多少 q 选中）——长尾 = hot few + cold many？
#   [3] tile density ρ 分布（16×16 tile）——大量工作 ρ>0.5 才值得 TC
#   [4] work_hot 曲线：top-x% page 承载多少 q-k pairs
#   [5] near/far 分区拆开（用户预期：far 语义复用更高？）
import json
import sys

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from sglang.srt.layers.attention.tli.config import TLIProfile
from sglang.srt.layers.attention.tli.indexer import TLIIndexer

dev = "cuda:0"
TRACES = [
    ("hotpotqa", "/tmp/trace/qwen3-8b/lb_hotpotqa_0"),
    ("musique", "/tmp/trace/qwen3-8b/lb_musique_0"),
    ("gov_report", "/tmp/trace/qwen3-8b/lb_gov_report_0"),
]
LAYERS = [3, 5, 10, 20, 33]
FIGDIR = "/home/wangyuanshuo02/sglang/figures_m8"
import os

os.makedirs(FIGDIR, exist_ok=True)
prof = TLIProfile()
out = {"rows": [], "config": {"K1": prof.k1_blocks, "bs": prof.block_size}}

print(f"{'task':>12s} {'L':>3s} {'Nq':>4s} {'nblk':>5s} | {'far页面':>6s} "
      f"{'f_j>8占比':>8s} {'ρ>0.5占比':>8s} {'work@20%pg':>9s} {'far work@20%':>10s}")
for tname, TDIR in TRACES:
    mats = {}
    for layer in LAYERS:
        d = torch.load(f"{TDIR}/layer{layer:02d}.pt", map_location=dev)
        k_real = d["k"].float()
        q_real = d["q"].float()
        qpos = d["qpos"].cuda()
        S, Hkv, D = k_real.shape
        H = q_real.shape[1]
        G = H // Hkv
        idxer = TLIIndexer(prof, head_dim=D).to(dev)
        index = idxer.build_block_index(k_real[:S])
        nblk = index["nblk"]
        kmin, kmax = index["kmin"][:nblk], index["kmax"][:nblk]

        # 真实 q 行（t>4096 保证 L1 有区分度）
        order = qpos.argsort()
        valid = qpos[order] > 4096
        qi_list = order[valid]
        Nq = len(qi_list)
        onehot = torch.zeros(Nq, nblk, dtype=torch.bool, device=dev)
        blk_end = (torch.arange(nblk, device=dev) + 1) * prof.block_size - 1
        for i, qi in enumerate(qi_list.tolist()):
            t = int(qpos[qi])
            q_t = q_real[qi : qi + 1]
            qs = q_t[..., idxer.idx1]
            qg = qs.clamp(min=0).reshape(1, Hkv, G, prof.coarse_dim)
            qn = qs.clamp(max=0).reshape(1, Hkv, G, prof.coarse_dim)
            sc1 = (torch.einsum("bhgd,nhd->bhgn", qg, kmax)
                   + torch.einsum("bhgd,nhd->bhgn", qn, kmin)).sum(-2)
            sc1 = sc1.masked_fill(blk_end.view(1, 1, -1) > t, float("-inf"))
            K1 = min(prof.k1_blocks, nblk)
            cand = torch.topk(sc1, K1, dim=-1).indices[0]
            # 滑窗强制块（head 无关）
            last_blk = t // prof.block_size
            f_blks = torch.arange(max(0, last_blk - prof.sliding_blocks + 1), last_blk + 1, device=dev)
            row = torch.zeros(nblk, dtype=torch.bool, device=dev)
            row.scatter_(0, torch.cat([cand.reshape(-1), f_blks]), True)
            onehot[i] = row
        M = onehot.float().cpu().numpy()  # [Nq, nblk]

        # 统计
        fj = M.sum(0)  # page 激活频率
        far_hi_blk = max(2, (S - 2048) // prof.block_size)
        is_far = np.zeros(nblk, dtype=bool)
        is_far[2:far_hi_blk] = True
        fj_far = fj[is_far]
        # tile density（16×16）
        T = 16
        NqT, nbT = M.shape[0] // T, M.shape[1] // T
        if NqT > 0 and nbT > 0:
            Mt = M[: NqT * T, : nbT * T].reshape(NqT, T, nbT, T).mean(axis=(1, 3))
            rho = Mt.ravel()
            rho_gt_half = float((rho > 0.5).mean())
        else:
            rho, rho_gt_half = np.array([]), 0.0
        # work_hot（全部 + far only）
        def work_at(f, frac):
            fs = np.sort(f)[::-1].astype(np.float64)
            k = max(1, int(len(fs) * frac))
            return float(fs[:k].sum() / max(fs.sum(), 1))

        w20 = work_at(fj, 0.2)
        w20_far = work_at(fj_far, 0.2)
        row = {
            "task": tname, "layer": layer, "Nq": Nq, "nblk": nblk,
            "pages_far": int(is_far.sum()),
            "fj_mean": round(float(fj.mean()), 2),
            "fj_max": int(fj.max()),
            "hot_share_f_gt8": round(float((fj > 8).mean()), 4),
            "hot_share_f_gt8_far": round(float((fj_far > 8).mean()), 4),
            "rho_gt05_share": round(rho_gt_half, 4),
            "rho_mean": round(float(rho.mean()) if len(rho) else 0, 4),
            "work_at_20pct_pages": round(w20, 4),
            "work_at_20pct_pages_far": round(w20_far, 4),
        }
        out["rows"].append(row)
        print(f"{tname:>12s} L{layer:02d} {Nq:4d} {nblk:5d} | {row['pages_far']:6d} "
              f"{row['hot_share_f_gt8']:8.3f} {rho_gt_half:8.3f} {w20:9.3f} {w20_far:10.3f}")
        mats[layer] = (M, is_far, qpos[qi_list].cpu().numpy())
        del d, k_real, q_real, index
        torch.cuda.empty_cache()

    # ---- [1] 01 矩阵位图（选 L10 代表层）----
    M, is_far, tpos = mats[10]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.imshow(M, aspect="auto", cmap="Greys", interpolation="nearest")
    far_hi_blk = np.where(is_far)[0][-1] + 1 if is_far.any() else 0
    ax.axvline(far_hi_blk, color="tab:red", lw=1.5, ls="--", label="far/near boundary")
    ax.axvline(2, color="tab:blue", lw=1.5, ls="--", label="sink boundary")
    ax.set_xlabel("page (block) id")
    ax.set_ylabel("query (sorted by position)")
    ax.set_title(f"{tname} L10: Q×K_avg selection 01 matrix (K1={prof.k1_blocks})")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/fig_reuse_matrix_{tname}.png", dpi=150)
    plt.close(fig)

    # ---- [2] f_j 直方图 + [4] work_hot 曲线（合并双 y）----
    fj = M.sum(0)
    fj_far = fj[is_far]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.hist(fj_far, bins=64, color="tab:red", alpha=0.7, label="far pages")
    a1.hist(fj[~is_far], bins=64, color="tab:blue", alpha=0.7, label="near pages")
    a1.set_yscale("log")
    a1.set_xlabel("activation frequency $f_j$ (#queries selecting page)")
    a1.set_ylabel("#pages (log)")
    a1.set_title(f"{tname} L10: page activation frequency")
    a1.legend(fontsize=8)

    def cum_curve(f):
        fs = np.sort(f)[::-1].astype(np.float64)
        x = np.arange(1, len(fs) + 1) / len(fs)
        return x, np.cumsum(fs) / fs.sum()

    x, y = cum_curve(fj)
    xf, yf = cum_curve(fj_far)
    a2.plot(x * 100, y * 100, label="all pages")
    a2.plot(xf * 100, yf * 100, label="far pages", color="tab:red")
    a2.plot([0, 100], [0, 100], "k:", lw=1, label="uniform")
    a2.set_xlabel("top x% pages (by $f_j$)")
    a2.set_ylabel("% of q-k pairs")
    a2.set_title(f"{tname} L10: work concentration (hot pages)")
    a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/fig_reuse_freq_{tname}.png", dpi=150)
    plt.close(fig)

json.dump(out, open("/home/wangyuanshuo02/sglang/tli_reuse_stats.json", "w"), indent=1)
print(f"\nsaved: {FIGDIR}/fig_reuse_matrix_*.png, fig_reuse_freq_*.png, tli_reuse_stats.json")

# ---- 汇总 ----
rows = out["rows"]
for key, lab in [("hot_share_f_gt8", "f_j>8 page 占比"), ("rho_gt05_share", "ρ>0.5 tile 占比"),
                 ("work_at_20pct_pages", "work@top20%pages"), ("work_at_20pct_pages_far", "work@top20%(far)")]:
    v = sum(r[key] for r in rows) / len(rows)
    print(f"{lab:>20s}: {v:.4f}")
