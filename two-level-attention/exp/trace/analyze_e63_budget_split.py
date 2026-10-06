# #63 near/far 预算配比与同 method 双区消融（离线 trace 重放）
# 用户框架：
#   alpha = near 长度占比；beta = near 预算占比（beta>alpha = 预算让渡近端）
#   gamma = token 级细筛配比（near_budget_token = near_page_topk*page_size*gamma）
#   同 method 双区：(m_far, m_near) 组合，m ∈ {minmax, avg, max}
# 两块实验：
#   A. beta 预算让渡（现行滑窗形态）：固定总预算 B=near_len+far_tokens=2304，
#      扫 near_len ∈ {512,1024,1536,2048}，far_tokens=B-near_len，
#      K1 按 far 预算比例（4× 超选，clamp [8,128] 块）——含「让渡给 near」与
#      「砍 far」两方向在同一曲线上。度量 = 总 mass coverage（sink+far 全覆盖）。
#   B. near 页级形态（新形态，现行 near 是滑窗全保留）：
#      near 区块 topk（near_page_topk 页）→ 池内 token 分数选
#      near_page_topk*64*gamma 个 token；far 侧照旧 K1=128 块 + far_tokens=256
#      细筛。扫 (m_far, m_near) method 组合 + gamma ∈ {0.5, 1.0}。
# 口径：真实全维 softmax（causal）行级 mass coverage，mean over (Hkv,G,行)；
# far 细筛离线代理 = 子空间精确分数 top-far_tokens（E4c 已证 ≈ oracle）。
import json
import os

import torch

TRACE_DIRS = ["/tmp/trace/qwen3-8b", "/tmp/trace/qwen3-30b"]
BS = 64
D_COARSE = 32
NEAR_LEN = 2048
SINK = 128
TAIL_N = 4
K1_FAR = 128
FAR_TOKENS = 256
# A 块：固定总预算扫描
B_TOTAL = 2304
A_GRID = [512, 1024, 1536, 2048]
# B 块：near 页级形态
B_NEAR_PAGES = [16, 32]
B_GAMMAS = [0.5, 1.0]
B_METHODS = ["minmax", "avg", "max"]
OUT = "/home/wangyuanshuo02/two-level-attention/exp/trace/results/e63_budget_split.json"


def l1_scores_three(kc, q):
    """kc: [nblk, Hkv, bs, d']；q: [Hkv, G, d']。
    返回 {minmax, avg, max}: [nblk, Hkv, G]。"""
    kmax = kc.amax(2)
    kmin = kc.amin(2)
    kavg = kc.mean(2)
    qg = q.clamp(min=0)
    qn = q.clamp(max=0)
    mm = torch.einsum("hgd,mhd->mhg", qg, kmax) + torch.einsum("mhd,hgd->mhg", kmin, qn)
    av = torch.einsum("hgd,mhd->mhg", q, kavg)
    # max：乐观上界（不分符号直接取元素 max 的点积，语义近 minmax 但无正负拆分）
    kabsmax = torch.where(kmax.abs() > kmin.abs(), kmax, kmin)
    mx = torch.einsum("hgd,mhd->mhg", q, kabsmax)
    return {"minmax": mm, "avg": av, "max": mx}


def layer_stats(tr, layer_id, device):
    t = torch.load(tr + f"/layer{layer_id:02d}.pt", map_location="cpu", weights_only=False)
    k = t["k"].to(device).float()
    q = t["q"].to(device).float()
    qpos = t["qpos"]
    S = t["S"]
    H, Hkv, D = q.shape[1], k.shape[1], k.shape[2]
    G = H // Hkv
    half = D // 2
    idx1 = list(range(half - 16, half)) + list(range(D - 16, D))
    nblk = (S + BS - 1) // BS
    pad = nblk * BS - S
    k_p = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, pad))
    kc = k_p[..., idx1].reshape(nblk, BS, Hkv, D_COARSE).permute(0, 2, 1, 3)  # [nblk,Hkv,bs,d']
    # 子空间 token 级展开（pad 段会被因果/分区 mask 剔除）
    k_tok = k_p[..., idx1]  # [S_pad, Hkv, d']
    k_e = k.repeat_interleave(G, dim=1)
    pos = torch.arange(S, device=device)
    blk_end = (torch.arange(nblk, device=device) + 1) * BS - 1
    res = {}
    for ri in range(TAIL_N):
        t_r = int(qpos[-TAIL_N + ri])
        if t_r < NEAR_LEN + SINK + FAR_TOKENS + 512:
            continue
        sc_full = torch.einsum("hd,shd->hs", q[-TAIL_N + ri], k_e) * (D**-0.5)
        m = pos <= t_r
        sc_full = sc_full.masked_fill(~m.unsqueeze(0), float("-inf"))
        p_full = torch.softmax(sc_full, dim=-1).view(Hkv, G, S)
        q_sub = q[-TAIL_N + ri][..., idx1].reshape(Hkv, G, D_COARSE)
        # token 级子空间分数 [Hkv, G, S_pad]
        sc_tok = torch.einsum("hgd,shd->hgs", q_sub, k_tok)

        def cov_mass(cand):
            """cand: [Hkv, G, S] bool（pad 段视为未选）→ 行级 mass 均值。"""
            cm = (p_full * cand[..., :S]).sum(-1)
            return float(cm.mean())

        def select_region(lo, hi, method, n_pages, n_tokens):
            """区 [lo,hi) 内：method 块打分 top n_pages 页 → 池内 token 分数 top n_tokens。
            返回 cand [Hkv,G,S] bool。"""
            in_reg = (pos >= lo) & (pos < hi)
            s1 = l1_scores_three(kc, q_sub)[method]  # [nblk,Hkv,G]
            s1 = s1.masked_fill((blk_end.view(-1, 1, 1) > t_r), float("-inf"))
            # 只在区内的块参与（块与区近似对齐：块中心判定）
            blk_lo = torch.arange(nblk, device=device) * BS
            blk_in = (blk_lo < hi) & (blk_end >= lo)
            s1 = s1.masked_fill(~blk_in.view(-1, 1, 1), float("-inf"))
            s1p = s1.permute(1, 2, 0)  # [Hkv,G,nblk]
            ib = torch.topk(s1p, min(n_pages, nblk), dim=-1).indices
            cand = torch.zeros(Hkv, G, S, dtype=torch.bool, device=device)
            blk_off = torch.arange(BS, device=device)
            tok = (ib.unsqueeze(-1) * BS + blk_off.view(1, 1, 1, BS)).clamp(max=S - 1)
            pool = torch.zeros(Hkv, G, S, dtype=torch.bool, device=device)
            pool.scatter_(2, tok.reshape(Hkv, G, -1), True)
            pool &= in_reg.view(1, 1, S)
            # 池内 token 细筛：分数 top n_tokens（池外 -inf）
            sc_pool = sc_tok[..., :S].masked_fill(~pool, float("-inf"))
            # 因果保护（区内块可能跨界）
            sc_pool = sc_pool.masked_fill((pos > t_r).view(1, 1, -1), float("-inf"))
            it = torch.topk(sc_pool, min(n_tokens, S), dim=-1).indices
            cand.scatter_(2, it, True)
            return cand

        sink_c = pos < SINK
        # ---- A 块：固定总预算 B_TOTAL 扫 near_len（滑窗全保留 + far 块选择）----
        for nl in A_GRID:
            ft = B_TOTAL - nl
            if ft < 128:
                continue
            k1 = max(8, min(K1_FAR, (ft * 4) // BS))
            far_hi = max(SINK, t_r + 1 - nl)
            cand = torch.zeros(Hkv, G, S, dtype=torch.bool, device=device)
            cand |= sink_c.view(1, 1, S)
            cand |= (pos >= t_r + 1 - nl).view(1, 1, S) & (pos <= t_r).view(1, 1, S)
            cand |= select_region(SINK, far_hi, "minmax", k1, ft)
            res.setdefault(f"A_nl{nl}_ft{ft}_k1{k1}", []).append(cov_mass(cand))
        # ---- B 块：near 页级形态，method 组合 × gamma × 页数 ----
        far_hi = t_r + 1 - NEAR_LEN
        for m_far, m_near in [(a, b) for a in B_METHODS for b in B_METHODS]:
            for np_ in B_NEAR_PAGES:
                for g in B_GAMMAS:
                    nt = int(np_ * BS * g)
                    cand = torch.zeros(Hkv, G, S, dtype=torch.bool, device=device)
                    cand |= sink_c.view(1, 1, S)
                    cand |= select_region(SINK, far_hi, m_far, K1_FAR, FAR_TOKENS)
                    cand |= select_region(far_hi, t_r + 1, m_near, np_, nt)
                    res.setdefault(f"B_{m_far}-{m_near}_np{np_}_g{g}", []).append(cov_mass(cand))
        # 参考基线：现行滑窗形态（near 全保留）同口径
        cand = torch.zeros(Hkv, G, S, dtype=torch.bool, device=device)
        cand |= sink_c.view(1, 1, S)
        cand |= (pos >= t_r + 1 - NEAR_LEN).view(1, 1, S) & (pos <= t_r).view(1, 1, S)
        cand |= select_region(SINK, far_hi, "minmax", K1_FAR, FAR_TOKENS)
        res.setdefault("BASE_slide2048_far256", []).append(cov_mass(cand))
        # 上界：全保留
        res.setdefault("UPPER_full", []).append(float(p_full.sum(-1).mean()))
    return res


def main():
    import os
    device = "cuda:0"
    out = {}
    for tdir in TRACE_DIRS:
        model = os.path.basename(tdir)
        if not os.path.isdir(tdir):
            continue
        for name in sorted(os.listdir(tdir)):
            tr = os.path.join(tdir, name)
            if not os.path.isfile(os.path.join(tr, "meta.json")):
                continue
            n_layers = json.load(open(tr + "/meta.json"))["n_layers"]
            agg = {}
            for li in range(0, n_layers, max(1, n_layers // 12)):
                r = layer_stats(tr, li, device)
                for k2, v in r.items():
                    agg.setdefault(k2, []).extend(v)
            rec = {k2: round(sum(v) / len(v), 4) for k2, v in agg.items()}
            out[f"{model}/{name}"] = rec
            print(f"[{model}/{name}] " + " ".join(f"{k2}={v}" for k2, v in sorted(rec.items())), flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
