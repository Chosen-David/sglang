# E65b：自训练降维矩阵（用户指令：自训练的降维矩阵怎么样对比、哪种 method×降维法最优）
# 两种自训练形态（都在 NoPE 尾维 32 子空间内训练，与 trunc/pca_sub 同口径）：
#   sup_wsvd：注意力概率加权 PCA（闭式）——最小化 score 保持 MSE ||q·k − qWWᵀk||² 的
#             加权最优解 = 加权协方差 C=Σ_t w_t·k_t k_tᵀ 的 top-d 特征向量；
#             与 pca_sub（无权 PCA，保持 K 方差）形成「任务对齐 vs 方差对齐」对照。
#   sup_grad：自由矩阵（Adam 梯度训练，目标 = score 回归 MSE），非正交、可学尺度。
# 校准集 = lb_hotpotqa_0（M9 哲学：离线校准一次、跨任务只读）；评估 = 其余 7 样本（泛化口径）。
# 臂：A far 粗筛 / C 细筛 / D 全链，d ∈ {4,8,16}，与 E65 主表的 trunc / pca_sub 对比。
import json
import os

import torch

import analyze_e65_dim_reduce as base   # 复用 blk_minmax/blk_avg/select_region/常量

TRACE = base.TRACE
OUT = "/home/wangyuanshuo02/two-level-attention/exp/trace/results/e65b_sup_proj.json"
CAL = "lb_hotpotqa_0"
D2I = base.D2I
SINK, SWA, NEAR_BAND = base.SINK, base.SWA, base.NEAR_BAND
BS, N_PAGES, BUD_TOK, TAIL_N = base.BS, base.N_PAGES, base.BUD_TOK, base.TAIL_N
DIMS = [4, 8, 16]
NQ_CAL = 8          # 校准用末 8 个 query（加权信号更多）


def calibrate(lf, device):
    """每层：返回 (basis_sup [Hkv,32,32] 加权PCA, W_grad [Hkv,32,32] 梯度训练)。"""
    d = torch.load(lf, map_location="cpu", weights_only=False)
    k, q, qpos, S = d["k"].float(), d["q"].float(), d["qpos"], d["S"]
    Hkv, H, D = k.shape[1], q.shape[1], k.shape[-1]
    G = H // Hkv
    t = int(qpos[-1])
    mid_hi = t + 1 - SWA
    far_lo, far_hi = SINK, mid_hi - NEAR_BAND
    if far_hi - far_lo < 8192:
        return None
    idx = torch.tensor(D2I)
    k32 = k[..., idx]                                   # [S,Hkv,32]
    # 加权协方差（按 Hkv 累积，权重 = 末 NQ_CAL 个 query 的 far 区 softmax 概率）
    C = torch.zeros(Hkv, 32, 32)
    for ri in range(NQ_CAL):
        q_head = q[-NQ_CAL + ri].reshape(Hkv, G, D).sum(1)
        s = torch.einsum("hd,shd->hs", q_head, k) * (D ** -0.5)
        s = s.masked_fill(torch.arange(S).view(1, -1) > int(qpos[-NQ_CAL + ri]), float("-inf"))
        p = torch.softmax(s, dim=-1)[:, far_lo:far_hi]  # [Hkv, Tfar]
        kf = k32[far_lo:far_hi]                          # [Tfar,Hkv,32]
        C += torch.einsum("ht,thd,the->hde", p, kf, kf)  # Σ_t w_t k kᵀ
    basis = []
    for h in range(Hkv):
        eig, vec = torch.linalg.eigh(C[h])
        basis.append(vec[:, torch.argsort(eig, descending=True)])   # [32,32] 列=主方向
    basis = torch.stack(basis)                           # [Hkv,32,32]
    # 梯度训练自由矩阵：按 dim 各训一个 [Hkv,32,dim]，score 回归目标=32 维子空间分
    q_t = q[-1].reshape(Hkv, G, D).sum(1)[..., idx]      # [Hkv,32]（训练 query）
    s_tgt_sub = torch.einsum("hd,shd->hs", q_t, k32)     # 32 维子空间分（回归目标）
    Wd = {}
    for dim in DIMS:
        W = torch.randn(Hkv, 32, dim, generator=torch.Generator().manual_seed(0)) * 0.05
        W.requires_grad_(True)
        opt = torch.optim.Adam([W], lr=1e-2)
        for it in range(300):
            qp = torch.einsum("hd,hde->he", q_t, W); kp = torch.einsum("shd,hde->she", k32, W)
            s_hat = torch.einsum("hd,shd->hs", qp, kp)
            loss = ((s_hat - s_tgt_sub) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        Wd[dim] = W.detach()
    return basis.detach(), Wd


def eval_layer(lf, device, basis, Wg):
    d = torch.load(lf, map_location="cpu", weights_only=False)
    k, q, qpos, S = d["k"].float(), d["q"].float(), d["qpos"], d["S"]
    Hkv, H, D = k.shape[1], q.shape[1], k.shape[-1]
    G = H // Hkv
    res = {}
    t = int(qpos[-1])
    mid_hi = t + 1 - SWA
    far_lo, far_hi = SINK, mid_hi - NEAR_BAND
    near_lo, near_hi = mid_hi - NEAR_BAND, mid_hi
    if far_hi - far_lo < 8192:
        return None
    idx = torch.tensor(D2I)
    k32 = k[..., idx]
    for ri in range(TAIL_N):
        q_head = q[-TAIL_N + ri].reshape(Hkv, G, D).sum(1)
        s = torch.einsum("hd,shd->hs", q_head, k) * (D ** -0.5)
        s = s.masked_fill(torch.arange(S).view(1, -1) > int(qpos[-TAIL_N + ri]), float("-inf"))
        p = torch.softmax(s, dim=-1)
        far_h = p[:, far_lo:far_hi].sum(-1)
        near_h = p[:, near_lo:near_hi].sum(-1)
        tot_f, tot_n = float(far_h.sum()), float(near_h.sum())
        if tot_f < 1e-6 or tot_n < 1e-6:
            continue
        q32 = q_head[..., idx]
        k_t32_far, k_t32_near = k32[far_lo:far_hi], k32[near_lo:near_hi]
        for dim in DIMS:
            for meth in ("sup_wsvd", "sup_grad", "pca_sub", "trunc"):
                # 特征构造（全部 32→d，除 trunc 是坐标截取）
                if meth == "sup_wsvd":
                    kf = torch.einsum("shd,hde->she", k32, basis[:, :, :dim])
                    qf = torch.einsum("hd,hde->he", q32, basis[:, :, :dim])
                elif meth == "sup_grad":
                    kf = torch.einsum("shd,hde->she", k32, Wg[dim])
                    qf = torch.einsum("hd,hde->he", q32, Wg[dim])
                elif meth == "pca_sub":
                    b = []
                    for h in range(Hkv):
                        _, _, Vt = torch.linalg.svd(k32[far_lo:far_hi, h], full_matrices=False)
                        b.append(Vt[:32].T)
                    b = torch.stack(b)
                    kf = torch.einsum("shd,hde->she", k32, b[:, :, :dim])
                    qf = torch.einsum("hd,hde->he", q32, b[:, :, :dim])
                else:  # trunc（子空间内前 d 维坐标截取）
                    it2 = torch.arange(dim)
                    kf, qf = k32[..., it2], q32[..., it2]
                kf_far = kf[far_lo:far_hi]
                toks = base.select_region(kf_far, qf, base.blk_minmax, k_t32_far, q32)
                cap = sum(float(p[h, toks[h] + far_lo].sum()) for h in range(Hkv))
                res.setdefault(f"A_far_coarse_{meth}_d{dim}", []).append(cap / tot_f)
                toks = base.select_region(kf[near_lo:near_hi], qf, base.blk_avg, k_t32_near, q32)
                cap = sum(float(p[h, toks[h] + near_lo].sum()) for h in range(Hkv))
                res.setdefault(f"B_near_coarse_{meth}_d{dim}", []).append(cap / tot_n)
                toks = base.select_region(k_t32_far, q32, base.blk_minmax, kf_far, qf)
                cap = sum(float(p[h, toks[h] + far_lo].sum()) for h in range(Hkv))
                res.setdefault(f"C_fine_{meth}_d{dim}", []).append(cap / tot_f)
                toks = base.select_region(kf_far, qf, base.blk_minmax, kf_far, qf)
                cap = sum(float(p[h, toks[h] + far_lo].sum()) for h in range(Hkv))
                res.setdefault(f"D_full_{meth}_d{dim}", []).append(cap / tot_f)
    return res


def main():
    torch.set_num_threads(21)
    # 校准（hotpotqa_0）
    cal_res = {}
    n_layers = json.load(open(f"{TRACE}/{CAL}/meta.json"))["n_layers"]
    layers = list(range(0, n_layers, max(1, n_layers // 8)))
    for li in layers:
        out = calibrate(f"{TRACE}/{CAL}/layer{li:02d}.pt", "cpu")
        if out is None:
            continue
        basis, Wg = out
        cal_res[li] = (basis, Wg)
    # 评估（其余 7 样本，跨任务泛化）
    results = {}
    for name in base.SAMPLES:
        if name == CAL or not os.path.isfile(f"{TRACE}/{name}/meta.json"):
            continue
        n_layers = json.load(open(f"{TRACE}/{name}/meta.json"))["n_layers"]
        agg = {}
        for li in range(0, n_layers, max(1, n_layers // 8)):
            li_cal = min(cal_res, key=lambda x: abs(x - li)) if cal_res else None
            if li_cal is None:
                continue
            basis, Wg = cal_res[li_cal]
            r = eval_layer(f"{TRACE}/{name}/layer{li:02d}.pt", "cpu", basis, Wg)
            if not r:
                continue
            for k2, v in r.items():
                agg.setdefault(k2, []).extend(v)
        rec = {k2: round(sum(v) / len(v), 4) for k2, v in agg.items()}
        results[name] = rec
        print(f"[{name}] " + " ".join(f"{k2}={v}" for k2, v in sorted(rec.items())), flush=True)
    json.dump(results, open(OUT, "w"), indent=1)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
