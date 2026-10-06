# E64f：最终配置实测——bp=128 + (α=0.125, β=0.25, γ=0.5) + sup_wsvd 降维 d∈{4,8}（§8b-39 帕累托点的诚实复验）
# 背景：帕累托图把 E64b 的 d32 质量数字与 d4 成本外推拼在一起——须在同一协议下实测降维后的真实质量。
# 协议与 E64b 完全一致（16 样本×12 层、真实全维 softmax 行级 mass coverage、B_TOK=2048），
# 差异仅在特征：k32（NoPE 尾维 32）经 sup_wsvd 基投影到 d 维（粗筛 minmax + 细筛精确分都用投影特征）。
# sup_wsvd 基：hotpotqa_0 离线校准（末 8 query far 区注意力加权协方差 top-d 特征向量，M9/E65b 哲学），
# 每个被评估层校准同层基（12 层一一对应，无跨层映射误差）。
# 臂：mavg_bp128_g0.5_d{4,8,32}（d32=控制组应≈E64b 0.9184）+ mono_bp128_d{4,8,32}。
import json
import os

import torch
import torch.nn.functional as F

import analyze_e64a_ab_grid as base

TRACE = base.TRACE
OUT = "/home/wangyuanshuo02/two-level-attention/exp/trace/results/e64f_final_config.json"
D2I, BS, SINK, SWA, TAIL_N = base.D2I, base.BS, base.SINK, base.SWA, base.TAIL_N
B_TOK = 2048
BP = 128
ALPHA, BETA, GAMMA = 0.125, 0.25, 0.5
DIMS = [4, 8, 32]
CAL = "lb_hotpotqa_0"
NQ_CAL = 8
NEAR_BLK = "avg"   # near 粗筛=avg（E64 冠军协议）


def calibrate_basis(lf):
    """返回 sup_wsvd 基 [Hkv,32,32]（列=按特征值降序的主方向）。"""
    d = torch.load(lf, map_location="cpu", weights_only=False)
    k, q, qpos, S = d["k"].float(), d["q"].float(), d["qpos"], d["S"]
    Hkv, H, D = k.shape[1], q.shape[1], k.shape[-1]
    G = H // Hkv
    t = int(qpos[-1])
    mid_hi = t + 1 - SWA
    far_lo, far_hi = SINK, mid_hi - 4096
    if far_hi - far_lo < 8192:
        return None
    idx = torch.tensor(D2I)
    k32 = k[..., idx]
    C = torch.zeros(Hkv, 32, 32)
    for ri in range(NQ_CAL):
        q_head = q[-NQ_CAL + ri].reshape(Hkv, G, D).sum(1)
        s = torch.einsum("hd,shd->hs", q_head, k) * (D ** -0.5)
        s = s.masked_fill(torch.arange(S).view(1, -1) > int(qpos[-NQ_CAL + ri]), float("-inf"))
        p = torch.softmax(s, dim=-1)[:, far_lo:far_hi]
        kf = k32[far_lo:far_hi]
        C += torch.einsum("ht,thd,the->hde", p, kf, kf)
    basis = []
    for h in range(Hkv):
        eig, vec = torch.linalg.eigh(C[h])
        basis.append(vec[:, torch.argsort(eig, descending=True)])
    return torch.stack(basis)


def eval_layer(lf, device, basis):
    d = torch.load(lf, map_location="cpu", weights_only=False)
    k, q, qpos, S = d["k"].to(device).float(), d["q"].to(device).float(), d["qpos"], d["S"]
    Hkv, H, D = k.shape[1], q.shape[1], k.shape[-1]
    G = H // Hkv
    pos = torch.arange(S, device=device)
    res = {}
    mid_hi_last = int(qpos[-1]) + 1 - SWA
    idx = torch.tensor(D2I, device=device)
    k32 = k[..., idx]                                        # [S,Hkv,32]
    basis = basis.to(device)                                 # [Hkv,32,32]
    # 投影特征（每层每 dim 一次）：kf_d = k32 @ basis[:,:,:d]
    feats = {}
    for ddim in DIMS:
        kf = torch.einsum("shd,hde->she", k32, basis[:, :, :ddim])   # [S,Hkv,d]
        feats[ddim] = kf
    for ri in range(TAIL_N):
        t_r = int(qpos[-TAIL_N + ri])
        mid_hi = t_r + 1 - SWA
        mid_len = mid_hi - SINK
        if mid_len < 4096:
            continue
        qg = q[-TAIL_N + ri].reshape(Hkv, G, D)
        k4 = k[:, :, None, :].expand(S, Hkv, G, D)
        s_full = torch.einsum("hgd,shgd->hgs", qg, k4) * (D ** -0.5)
        s_full = s_full.masked_fill((pos > t_r).view(1, 1, -1), float("-inf"))
        p_full = torch.softmax(s_full, dim=-1)

        def cov_mass(cand):
            return float((p_full * cand.unsqueeze(1)).sum(-1).mean())

        q32 = q[-TAIL_N + ri][..., idx].reshape(Hkv, G, 32).sum(1)   # [Hkv,32]
        qf = {ddim: torch.einsum("hd,hde->he", q32, basis[:, :, :ddim]) for ddim in DIMS}
        sink_c = (pos < SINK).view(1, -1)
        swa_c = ((pos >= t_r + 1 - SWA) & (pos <= t_r)).view(1, -1)
        near_L = int(ALPHA * mid_len)
        near_lo_off, near_hi_off = mid_len - near_L, mid_len
        nb_near = max(1, int(round(BP * BETA)))
        nb_far = max(1, BP - nb_near)
        nt_near = int(nb_near * BS * GAMMA)
        nt_far = max(64, B_TOK - nt_near)
        for ddim in DIMS:
            kf = feats[ddim]
            q_d = qf[ddim]
            nblk = (S + BS - 1) // BS
            kk = F.pad(kf, (0, 0, 0, 0, 0, nblk * BS - S))
            kc = kk.reshape(nblk, BS, Hkv, ddim)
            kmin, kmax, kavg = kc.amin(1), kc.amax(1), kc.mean(1)
            sc_mm = (torch.einsum("hd,nhd->hn", q_d.clamp(min=0), kmax) +
                     torch.einsum("hd,nhd->hn", q_d.clamp(max=0), kmin))
            sc_av = torch.einsum("hd,nhd->hn", q_d, kavg)
            sc_tok = torch.einsum("hd,shd->hs", q_d, kf)             # 投影特征精确 token 分
            sc_mm_mid = sc_mm[:, SINK // BS:]
            sc_av_mid = sc_av[:, SINK // BS:]

            def select_sub(lo_off, hi_off, sc_blk, tok_score, n_pages, n_tokens):
                nblk_mid = sc_blk.shape[1]
                rng = torch.arange(nblk_mid, device=device)
                s1 = sc_blk.masked_fill(~(((rng * BS) < hi_off) & ((rng * BS + BS - 1) >= lo_off)).view(1, -1), float("-inf"))
                ib = torch.topk(s1, min(n_pages, nblk_mid), dim=-1).indices
                tok = (ib.unsqueeze(-1) * BS + torch.arange(BS, device=device).view(1, 1, BS)).reshape(Hkv, -1).clamp(max=mid_len - 1)
                pool = torch.zeros(Hkv, mid_len, dtype=torch.bool, device=device)
                pool.scatter_(1, tok, True)
                pool &= ((torch.arange(mid_len, device=device) >= lo_off) &
                         (torch.arange(mid_len, device=device) < hi_off)).view(1, -1)
                ts = tok_score.masked_fill(~pool, float("-inf"))
                it = torch.topk(ts, min(n_tokens, mid_len), dim=-1).indices
                cand = torch.zeros(Hkv, S, dtype=torch.bool, device=device)
                cand.scatter_(1, it + SINK, True)
                return cand

            # mavg 双分区（冠军协议）
            cand = torch.zeros(Hkv, S, dtype=torch.bool, device=device)
            cand |= sink_c | swa_c
            cand |= select_sub(near_lo_off, near_hi_off, sc_av_mid, sc_tok[:, SINK:mid_hi], nb_near, nt_near)
            cand |= select_sub(0, near_lo_off, sc_mm_mid, sc_tok[:, SINK:mid_hi], nb_far, nt_far)
            res.setdefault(f"mavg_bp{BP}_g{GAMMA}_d{ddim}", []).append(cov_mass(cand))
            # mono 同 bp（单池）
            cand = torch.zeros(Hkv, S, dtype=torch.bool, device=device)
            cand |= sink_c | swa_c
            cand |= select_sub(0, mid_len, sc_mm_mid, sc_tok[:, SINK:mid_hi], BP, B_TOK)
            res.setdefault(f"mono_bp{BP}_d{ddim}", []).append(cov_mass(cand))
    del k, q, d
    torch.cuda.empty_cache()
    return res


def main():
    device = "cuda:" + os.environ.get("E64F_GPU", "0")
    # 校准（hotpotqa_0，12 个被评估层一一对应）
    n_layers_cal = json.load(open(f"{TRACE}/{CAL}/meta.json"))["n_layers"]
    layers_cal = list(range(0, n_layers_cal, max(1, n_layers_cal // 12)))
    basis_map = {}
    for li in layers_cal:
        b = calibrate_basis(f"{TRACE}/{CAL}/layer{li:02d}.pt")
        if b is not None:
            basis_map[li] = b
    print(f"calibrated {len(basis_map)} layers on {CAL}", flush=True)
    names = sorted(n for n in os.listdir(TRACE)
                   if os.path.isfile(os.path.join(TRACE, n, "meta.json")))
    results = {}
    for name in names:
        n_layers = json.load(open(f"{TRACE}/{name}/meta.json"))["n_layers"]
        agg = {}
        for li in range(0, n_layers, max(1, n_layers // 12)):
            li_cal = min(basis_map, key=lambda x: abs(x - li)) if basis_map else None
            if li_cal is None:
                continue
            r = eval_layer(f"{TRACE}/{name}/layer{li:02d}.pt", device, basis_map[li_cal])
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
