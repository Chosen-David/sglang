# L2 细筛「降维投影 vs 维度选择」对比（用户建议：用降维方法打分，压缩后维度上打分提速）
#
# 背景：test_tli_dim_sweep2 实测维度选择（tail dims）δ=16→8 far recall 0.557→0.378
# （病因=维度不够，非 4bit 量化）。用户指出：聚类路线（另一项目）128→32 压缩质量不掉，
# 且 DSA 正是用训练投影（weights_proj）把索引打分压到 128 维——TLI 能否用
# training-free 投影（PCA 离线校准 / 随机投影 JL）替代「取 RoPE 尾维」？
#
# 方法（同 dim_sweep2 隔离口径：far 区直接打分，不经 L1，fp32 精度分离维度效应）：
#   select-r  : 取每半尾维 r/2（当前实现，r=2δ）
#   pca-r     : 对 far 区 K 做 per-kv-head SVD 取 top-r 主成分（离线校准口径：
#               用同一 trace 同层同 t 的 K 估计——是校准上限；跨任务迁移单独报）
#   random-r  : 固定 seed 高斯随机投影（JL 引理，免校准）
#   pca-cross : PCA 基取自 hotpotqa，打到其他任务（迁移性）
# 打分 FLOP = r MAC/token-head（vs 全维 128）。
# 口径：far recall vs dense 全维 far top-k2_far（per-head mean/min）。
import sys

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
import json

import torch

dev = "cuda:0"
TRACES = [
    ("hotpotqa", "/tmp/trace/qwen3-8b/lb_hotpotqa_0"),
    ("musique", "/tmp/trace/qwen3-8b/lb_musique_0"),
    ("gov_report", "/tmp/trace/qwen3-8b/lb_gov_report_0"),
    ("narrativeqa", "/tmp/trace/qwen3-8b/lb_narrativeqa_0"),
    ("passage", "/tmp/trace/qwen3-8b/lb_passage_retrieval_en_0"),
]
LAYERS = [3, 5, 10, 20, 33]
RANKS = [16, 32]
OUT = "/home/wangyuanshuo02/sglang/tli_proj_sweep.json"

# 跨任务 PCA 基：hotpotqa 每层 far 区 K 的 per-head top-r 主成分
CROSS_SRC = "/tmp/trace/qwen3-8b/lb_hotpotqa_0"
cross_basis = {}  # (layer, r) -> [Hkv, D, r]
for layer in LAYERS:
    d = torch.load(f"{CROSS_SRC}/layer{layer:02d}.pt", map_location="cpu")
    k = d["k"].float()  # [S, Hkv, D]
    S, Hkv, D = k.shape
    far = k[128 : S - 2048].reshape(-1, Hkv, D)  # [L, Hkv, D]
    for r in RANKS:
        vb = []
        for h in range(Hkv):
            U, Sv, Vt = torch.linalg.svd(far[:, h].cpu(), full_matrices=False)
            vb.append(Vt[:r].T)  # [D, r]
        cross_basis[(layer, r)] = torch.stack(vb).to(dev)  # [Hkv, D, r]
    del d, k, far
torch.cuda.empty_cache()

rows = []
print(f"{'task':>12s} {'L':>3s} {'t':>6s} | {'select16':>9s} {'pca16':>7s} {'rand16':>7s} {'xpcA16':>7s} | {'select32':>9s} {'pca32':>7s} {'rand32':>7s} {'xpcA32':>7s}")
for tname, TDIR in TRACES:
    for layer in LAYERS:
        d = torch.load(f"{TDIR}/layer{layer:02d}.pt", map_location=dev)
        k_real = d["k"].float()
        q_real = d["q"].float()
        qpos = d["qpos"].cuda()
        S, Hkv, D = k_real.shape
        H = q_real.shape[1]
        G = H // Hkv
        for t in [S // 4, S // 2, S - 1]:
            in_range = torch.nonzero(qpos <= t).squeeze(1)
            qi = int(in_range[-1])
            q_t = q_real[qi : qi + 1]
            qg = q_t.reshape(1, Hkv, G, D)
            s_full = (
                torch.einsum("bhgd,chd->bhgc", qg, k_real[: t + 1]).sum(-2)
                * (D**-0.5)
            )
            p_ = torch.softmax(s_full, dim=-1)[0]
            far_lo, near_len, tok_budget, sw, far_tok = 128, 2048, 1024, 128, 256
            far_hi = max(far_lo, t + 1 - near_len)
            k2_far = min(far_tok, max(0, far_hi - far_lo), max(0, tok_budget - (sw + far_lo)))
            if k2_far <= 0:
                continue
            oracle = torch.topk(s_full[0, :, far_lo:far_hi], k2_far, dim=-1).indices + far_lo
            oh = torch.zeros(Hkv, t + 1, device=dev)
            oh.scatter_(1, oracle, 1.0)
            q_h = qg.sum(2)[0]  # [Hkv, D]（GQA 求和口径，与 select 的 q2 一致）
            kfar = k_real[far_lo:far_hi]  # [L, Hkv, D]
            L_far = kfar.shape[0]

            recs = {}
            for r in RANKS:
                # select-r（尾维选择）
                idx = list(range(64 - r // 2, 64)) + list(range(128 - r // 2, 128))
                it = torch.tensor(idx, device=dev)
                sc = torch.einsum("hd,thd->ht", q_h[:, it], kfar[:, :, it])
                sel = torch.topk(sc, k2_far, dim=-1).indices + far_lo
                m = torch.zeros(Hkv, t + 1, device=dev)
                m.scatter_(1, sel, 1.0)
                recs[f"select{r}"] = ((oh * m).sum(1) / k2_far).mean().item()
                # pca-r（同 trace 同层 far 区 K 的 SVD——校准同分布上限）
                vb = []
                for h in range(Hkv):
                    _, _, Vt = torch.linalg.svd(kfar[:, h], full_matrices=False)
                    vb.append(Vt[:r].T)
                V = torch.stack(vb)  # [Hkv, D, r]
                q_p = torch.einsum("hdr,hd->hr", V, q_h)  # [Hkv, r]
                k_p = torch.einsum("hdr,thd->thr", V, kfar)  # [L, Hkv, r]
                sc = torch.einsum("hr,thr->ht", q_p, k_p)
                sel = torch.topk(sc, k2_far, dim=-1).indices + far_lo
                m = torch.zeros(Hkv, t + 1, device=dev)
                m.scatter_(1, sel, 1.0)
                recs[f"pca{r}"] = ((oh * m).sum(1) / k2_far).mean().item()
                # random-r（固定 seed 高斯）
                g = torch.Generator(device="cpu").manual_seed(0)
                Rm = torch.randn(Hkv, D, r, generator=g).to(dev) / (r**0.5)
                q_p = torch.einsum("hdr,hd->hr", Rm, q_h)
                k_p = torch.einsum("hdr,thd->thr", Rm, kfar)
                sc = torch.einsum("hr,thr->ht", q_p, k_p)
                sel = torch.topk(sc, k2_far, dim=-1).indices + far_lo
                m = torch.zeros(Hkv, t + 1, device=dev)
                m.scatter_(1, sel, 1.0)
                recs[f"rand{r}"] = ((oh * m).sum(1) / k2_far).mean().item()
                # pca-cross（hotpotqa 基）
                Vx = cross_basis[(layer, r)]
                q_p = torch.einsum("hdr,hd->hr", Vx, q_h)
                k_p = torch.einsum("hdr,thd->thr", Vx, kfar)
                sc = torch.einsum("hr,thr->ht", q_p, k_p)
                sel = torch.topk(sc, k2_far, dim=-1).indices + far_lo
                m = torch.zeros(Hkv, t + 1, device=dev)
                m.scatter_(1, sel, 1.0)
                recs[f"xpca{r}"] = ((oh * m).sum(1) / k2_far).mean().item()

            rows.append({"task": tname, "layer": layer, "t": t, **{k: round(v, 4) for k, v in recs.items()}})
            print(f"{tname:>12s} L{layer:02d} {t:6d} | "
                  f"{recs['select16']:9.4f} {recs['pca16']:7.4f} {recs['rand16']:7.4f} {recs['xpca16']:7.4f} | "
                  f"{recs['select32']:9.4f} {recs['pca32']:7.4f} {recs['rand32']:7.4f} {recs['xpca32']:7.4f}")
        del d, k_real, q_real
        torch.cuda.empty_cache()

json.dump(rows, open(OUT, "w"), indent=1)
print(f"\nsaved -> {OUT}（{len(rows)} 行）")
print("\n==== 汇总（far recall mean，跨任务×层×t）====")
for k in ["select16", "pca16", "rand16", "xpca16", "select32", "pca32", "rand32", "xpca32"]:
    v = sum(r[k] for r in rows) / len(rows)
    print(f"{k:>9s}: {v:.4f}")
print("参考：select16=当前实现（2δ=32）；select32≈δ16 全维 32=full 上界；打分 FLOP ∝ r")
