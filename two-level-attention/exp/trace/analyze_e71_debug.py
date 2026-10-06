# E71 诊断：B0 崩坏（F1 13.51）根因二分——transformers 侧 B 路径 = 投影(d8) + 4bit量化细筛 + α/β/γ 分区
# E64f 基准（连续投影细筛 + 分区）= 0.9102；逐项替换找失真源：
#   1 proj_cont   : E64f 原样复现（基准）
#   2 proj_quant  : 细筛分数 q_p @ quant4(kp)（transformers 现状：投影特征过 4bit 量化）
#   3 part_noproj : 分区 + d'=32 连续（无投影）→ 测分区独立影响
#   4 noproj_nopart: 单池 d'=32（= C 配置形态，C 冒烟 54.93 良好）
import json
import os

import torch
import torch.nn.functional as F

import analyze_e64a_ab_grid as base

TRACE = base.TRACE
D2I, BS, SINK, SWA, TAIL_N = base.D2I, base.BS, base.SINK, base.SWA, base.TAIL_N
B_TOK, BP = 2048, 128
ALPHA, BETA, GAMMA = 0.125, 0.25, 0.5
CAL = "lb_hotpotqa_0"
NQ_CAL = 8


def quant4(x):
    """逐 token 最后一维 minmax 4bit 量化（TIAIndexer.min_max_per_token_quant 同语义）。"""
    mx = x.amax(dim=-1, keepdim=True)
    mn = x.amin(dim=-1, keepdim=True)
    s = (mx - mn).clamp(min=1e-9) / 15
    return torch.clamp(torch.round((x - mn) / s), 0, 15) * s + mn


def calibrate_basis(lf):
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
    return torch.stack(basis)  # [Hkv,32,32]


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
    basis = basis.to(device)
    kf8 = torch.einsum("shd,hde->she", k32, basis[:, :, :8])  # [S,Hkv,8] 连续投影
    kf8q = quant4(kf8)                                        # 4bit 量化投影
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

        q32 = q[-TAIL_N + ri][..., idx].reshape(Hkv, G, 32).sum(1)  # [Hkv,32] group-sum
        q8 = torch.einsum("hd,hde->he", q32, basis[:, :, :8])       # [Hkv,8]
        sink_c = (pos < SINK).view(1, -1)
        swa_c = ((pos >= t_r + 1 - SWA) & (pos <= t_r)).view(1, -1)
        near_L = int(ALPHA * mid_len)
        near_lo_off, near_hi_off = mid_len - near_L, mid_len
        nb_near = max(1, int(round(BP * BETA)))
        nb_far = max(1, BP - nb_near)
        nt_near = int(nb_near * BS * GAMMA)
        nt_far = max(64, B_TOK - nt_near)

        def make_blk_feats(feat, q_f):
            nblk = (S + BS - 1) // BS
            kk = F.pad(feat, (0, 0, 0, 0, 0, nblk * BS - S))
            kc = kk.reshape(nblk, BS, Hkv, feat.shape[-1])
            kmin, kmax, kavg = kc.amin(1), kc.amax(1), kc.mean(1)
            sc_mm = (torch.einsum("hd,nhd->hn", q_f.clamp(min=0), kmax) +
                     torch.einsum("hd,nhd->hn", q_f.clamp(max=0), kmin))
            sc_av = torch.einsum("hd,nhd->hn", q_f, kavg)
            sc_tok = torch.einsum("hd,shd->hs", q_f, feat)
            return sc_mm[:, SINK // BS:], sc_av[:, SINK // BS:], sc_tok[:, SINK:mid_hi]

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

        # 四臂
        # 1 proj_cont（E64f 基准）：连续投影粗筛+细筛
        mm8, av8, tok8 = make_blk_feats(kf8, q8)
        cand = torch.zeros(Hkv, S, dtype=torch.bool, device=device)
        cand |= sink_c | swa_c
        cand |= select_sub(near_lo_off, near_hi_off, av8, tok8, nb_near, nt_near)
        cand |= select_sub(0, near_lo_off, mm8, tok8, nb_far, nt_far)
        res.setdefault("proj_cont", []).append(cov_mass(cand))
        # 2 proj_quant：细筛分数用量化投影特征（transformers 现状）；粗筛仍连续
        _, _, tok8q = make_blk_feats(kf8q, q8)
        cand = torch.zeros(Hkv, S, dtype=torch.bool, device=device)
        cand |= sink_c | swa_c
        cand |= select_sub(near_lo_off, near_hi_off, av8, tok8q, nb_near, nt_near)
        cand |= select_sub(0, near_lo_off, mm8, tok8q, nb_far, nt_far)
        res.setdefault("proj_quant", []).append(cov_mass(cand))
        # 3 part_noproj：d'=32 连续 + 分区
        mm32, av32, tok32 = make_blk_feats(k32, q32)
        cand = torch.zeros(Hkv, S, dtype=torch.bool, device=device)
        cand |= sink_c | swa_c
        cand |= select_sub(near_lo_off, near_hi_off, av32, tok32, nb_near, nt_near)
        cand |= select_sub(0, near_lo_off, mm32, tok32, nb_far, nt_far)
        res.setdefault("part_noproj", []).append(cov_mass(cand))
        # 4 noproj_nopart（C 形态）：d'=32 单池
        cand = torch.zeros(Hkv, S, dtype=torch.bool, device=device)
        cand |= sink_c | swa_c
        cand |= select_sub(0, mid_len, mm32, tok32, BP, B_TOK)
        res.setdefault("noproj_nopart", []).append(cov_mass(cand))
    del k, q, d
    torch.cuda.empty_cache()
    return res


def main():
    device = "cuda:0"
    n_layers_cal = json.load(open(f"{TRACE}/{CAL}/meta.json"))["n_layers"]
    layers_cal = list(range(0, n_layers_cal, max(1, n_layers_cal // 12)))
    basis_map = {}
    for li in layers_cal:
        b = calibrate_basis(f"{TRACE}/{CAL}/layer{li:02d}.pt")
        if b is not None:
            basis_map[li] = b
    names = ["lb_hotpotqa_0"]
    for name in names:
        n_layers = json.load(open(f"{TRACE}/{name}/meta.json"))["n_layers"]
        agg = {}
        for li in range(0, n_layers, max(1, n_layers // 12)):
            li_cal = min(basis_map, key=lambda x: abs(x - li))
            r = eval_layer(f"{TRACE}/{name}/layer{li:02d}.pt", device, basis_map[li_cal])
            for k2, v in r.items():
                agg.setdefault(k2, []).extend(v)
        print(name, {k2: round(sum(v) / len(v), 4) for k2, v in agg.items()})


if __name__ == "__main__":
    main()
