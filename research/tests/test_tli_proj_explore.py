# 投影降维深入探索（用户指示「再探索探索」：noPE 维度对照 / 校准集大小 /
# 跨层共享基 / 4bit 量化 PCA / PCA 基可解释性）
#
# 用户假说：① 其他子集（如 noPE/无旋转维）也许也能压；② 降维投影打分提速，
# 核心瓶颈在打分。Qwen3 rotate_half 全 128 维都旋转、无原生 noPE 维，但低频尾维
# 是「近似 noPE」（旋转慢、位置噪声小）——对照实验量化「子集选择」的全部选项：
#   select-tail（当前，低频尾维 r 个）/ select-head（高频头维）/ select-even（偶数维）
#   / pca（SVD top-r）/ pca-small（校准集只有 2048 token）/ pca-xlayer（跨层共享基）
#   / pca-4bit（投影后 4bit 量化，存储口径 16B/token-head @ r=16）
# 可解释性：PCA top-r 子空间 vs 低频尾维子空间的主角度（PCA 是否重发现低频子空间）。
# 口径：far recall vs dense 全维 far top-256 oracle（隔离口径，fp32 除 pca-4bit）。
import sys

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
import json

import torch

from sglang.srt.layers.attention.tli.indexer import quant4

dev = "cuda:0"
TRACES = [
    ("hotpotqa", "/tmp/trace/qwen3-8b/lb_hotpotqa_0"),
    ("musique", "/tmp/trace/qwen3-8b/lb_musique_0"),
    ("gov_report", "/tmp/trace/qwen3-8b/lb_gov_report_0"),
]
LAYERS = [3, 5, 10, 20, 33]
R = 16  # 主档：打分维度砍到 16（vs 当前 32）——速度卖点档
OUT = "/home/wangyuanshuo02/sglang/tli_proj_explore.json"

# 跨层共享基：hotpotqa L10 far 区 K 的 per-head top-r 主成分
SRC = "/tmp/trace/qwen3-8b/lb_hotpotqa_0"
d = torch.load(f"{SRC}/layer10.pt", map_location="cpu")
k_src = d["k"].float()
S_src, Hkv, D = k_src.shape
far_src = k_src[128 : S_src - 2048].reshape(-1, Hkv, D)
vb = []
for h in range(Hkv):
    _, _, Vt = torch.linalg.svd(far_src[:, h], full_matrices=False)
    vb.append(Vt[:R].T)
V_shared = torch.stack(vb).to(dev)  # [Hkv, D, r]
del d, k_src, far_src
torch.cuda.empty_cache()

# 可解释性：PCA 子空间 vs 低频尾维子空间的主角度（用共享基，逐 head）
tail_idx = list(range(64 - R // 2, 64)) + list(range(128 - R // 2, 128))
E_tail = torch.zeros(D, R, device=dev)
E_tail[tail_idx, torch.arange(R)] = 1.0
angles_all = []
for h in range(Hkv):
    Vh = V_shared[h]  # [D, r]
    # 主角度 = arccos(奇异值) of Vh^T @ E_tail
    sv = torch.linalg.svdvals(Vh.T @ E_tail)
    angles_all.append(torch.rad2deg(torch.arccos(sv.clamp(max=1.0))))
angles = torch.stack(angles_all)  # [Hkv, r]
print(f"[可解释性] PCA{R} vs 尾维{R} 主角度（度）: mean {angles.mean():.1f} "
      f"min {angles.min():.1f} max {angles.max():.1f}（0°=重合，90°=正交）")

rows = []
print(f"{'task':>12s} {'L':>3s} {'t':>6s} | {'tail':>6s} {'head':>6s} {'even':>6s} "
      f"{'pca':>6s} {'pcaS':>6s} {'pcaXL':>6s} {'pca4b':>6s} {'full':>6s}")
for tname, TDIR in TRACES:
    for layer in LAYERS:
        d = torch.load(f"{TDIR}/layer{layer:02d}.pt", map_location=dev)
        k_real = d["k"].float()
        q_real = d["q"].float()
        qpos = d["qpos"].cuda()
        S, Hkv, D = k_real.shape
        H = q_real.shape[1]
        G = H // Hkv
        for t in [S // 2, S - 1]:
            in_range = torch.nonzero(qpos <= t).squeeze(1)
            qi = int(in_range[-1])
            q_t = q_real[qi : qi + 1]
            qg = q_t.reshape(1, Hkv, G, D)
            s_full = (
                torch.einsum("bhgd,chd->bhgc", qg, k_real[: t + 1]).sum(-2)
                * (D**-0.5)
            )
            far_lo, near_len, tok_budget, sw, far_tok = 128, 2048, 1024, 128, 256
            far_hi = max(far_lo, t + 1 - near_len)
            k2_far = min(far_tok, max(0, far_hi - far_lo), max(0, tok_budget - (sw + far_lo)))
            if k2_far <= 0:
                continue
            oracle = torch.topk(s_full[0, :, far_lo:far_hi], k2_far, dim=-1).indices + far_lo
            oh = torch.zeros(Hkv, t + 1, device=dev)
            oh.scatter_(1, oracle, 1.0)
            q_h = qg.sum(2)[0]  # [Hkv, D]
            kfar = k_real[far_lo:far_hi]  # [L, Hkv, D]
            L_far = kfar.shape[0]

            def recall_with(mat_h: torch.Tensor, quant: bool = False) -> float:
                """mat_h: [Hkv, D, r] 投影基。返回 far recall（4bit 可选）。"""
                q_p = torch.einsum("hdr,hd->hr", mat_h, q_h)
                k_p = torch.einsum("hdr,thd->thr", mat_h, kfar)
                if quant:
                    k_p = quant4(k_p)
                sc = torch.einsum("hr,thr->ht", q_p, k_p)
                sel = torch.topk(sc, k2_far, dim=-1).indices + far_lo
                m = torch.zeros(Hkv, t + 1, device=dev)
                m.scatter_(1, sel, 1.0)
                return ((oh * m).sum(1) / k2_far).mean().item()

            def sel_idx(idx: list) -> float:
                it = torch.tensor(idx, device=dev)
                sc = torch.einsum("hd,thd->ht", q_h[:, it], kfar[:, :, it])
                sel = torch.topk(sc, k2_far, dim=-1).indices + far_lo
                m = torch.zeros(Hkv, t + 1, device=dev)
                m.scatter_(1, sel, 1.0)
                return ((oh * m).sum(1) / k2_far).mean().item()

            # 同源 PCA 基（全 far 区）
            vb = []
            for h in range(Hkv):
                _, _, Vt = torch.linalg.svd(kfar[:, h], full_matrices=False)
                vb.append(Vt[:R].T)
            V_same = torch.stack(vb)
            # 小校准集 PCA（2048 token 采样）
            g = torch.Generator(device="cpu").manual_seed(1)
            sel_tok = torch.randperm(L_far, generator=g)[: min(2048, L_far)]
            vb = []
            for h in range(Hkv):
                _, _, Vt = torch.linalg.svd(kfar[sel_tok, h].cpu(), full_matrices=False)
                vb.append(Vt[:R].T)
            V_small = torch.stack(vb).to(dev)

            rec = {
                "tail": sel_idx(list(range(64 - R // 2, 64)) + list(range(128 - R // 2, 128))),
                "head": sel_idx(list(range(R // 2)) + list(range(64, 64 + R // 2))),
                "even": sel_idx(list(range(0, 128, 128 // R))),
                "pca": recall_with(V_same),
                "pca_small": recall_with(V_small),
                "pca_xlayer": recall_with(V_shared),
                "pca_4bit": recall_with(V_same, quant=True),
                "full128": sel_idx(list(range(128))),
            }
            rows.append({"task": tname, "layer": layer, "t": t, "r": R,
                         **{k: round(v, 4) for k, v in rec.items()}})
            print(f"{tname:>12s} L{layer:02d} {t:6d} | " +
                  " ".join(f"{rec[k]:6.4f}" for k in
                           ["tail", "head", "even", "pca", "pca_small", "pca_xlayer", "pca_4bit", "full128"]))
        del d, k_real, q_real
        torch.cuda.empty_cache()

json.dump(rows, open(OUT, "w"), indent=1)
print(f"\nsaved -> {OUT}（{len(rows)} 行，r={R}）")
print("\n==== 汇总（far recall mean）====")
for k in ["tail", "head", "even", "pca", "pca_small", "pca_xlayer", "pca_4bit", "full128"]:
    v = sum(r[k] for r in rows) / len(rows)
    print(f"{k:>10s}: {v:.4f}")
