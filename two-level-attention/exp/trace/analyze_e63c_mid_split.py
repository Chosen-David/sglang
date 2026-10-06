# #63b 补充消融：mid 区双分区（用户 proposal 原设计）vs mid 单池（TLI 现实现）
# 用户框架：L 中 sink_L + swa_L 保护；mid_L = L - sink_L - swa_L 交 indexer；
#   mid 内再分 near 子区（alpha·mid_L）与 far 子区，各自独立「粗筛+细筛」：
#   near 拿 beta 份额页池（budget_page_topk·beta 块）→ gamma 折扣 token 细筛
#   far 拿剩余页池与 token 预算。
# 对照臂（同总预算 B = swa + mid 选择 token = 2304、同总页池 128 块、swa=1024）：
#   mono        : mid 单池 128 块 → 1280 tok（TLI 简化版，alpha=0）
#   split_55_5  : alpha=.5 beta=.5 gamma=.5 → near 64 块→640 tok / far 64 块→640 tok
#   split_52_5  : alpha=.5 beta=.25 gamma=.5 → near 32 块→320 / far 96 块→960（预算向 far 倾斜）
#   split_25_25 : alpha=.25 beta=.25 gamma=.5 → 更短 near 子区
#   tli_anchor  : swa 实际 2048 全保留（mid 近端免打分）+ far 单池 128 块→256 tok（TLI 生产形态）
# 口径：真实全维 softmax（causal）行级 mass coverage（sink+swa+mid 全覆盖）；18 样本×12 层×末 4 query。
import json
import os

import torch

TRACE_DIRS = ["/tmp/trace/qwen3-8b", "/tmp/trace/qwen3-30b"]
BS = 64
D_COARSE = 32
SINK = 128
TAIL_N = 4
B_TOTAL = 2304          # swa + mid 选择 token 总预算
SWA = 1024              # proposal 保护段滑窗（区别于 TLI 的 2048）
PAGE_POOL = 128         # 总页池 budget_page_topk
OUT = "/home/wangyuanshuo02/two-level-attention/exp/trace/results/e63c_mid_split.json"


def l1_minmax(kc, q_sub):
    kmax = kc.amax(2)
    kmin = kc.amin(2)
    qg = q_sub.clamp(min=0)
    qn = q_sub.clamp(max=0)
    return torch.einsum("hgd,mhd->mhg", qg, kmax) + torch.einsum("mhd,hgd->mhg", kmin, qn)


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
    kc = k_p[..., idx1].reshape(nblk, BS, Hkv, D_COARSE).permute(0, 2, 1, 3)
    k_tok = k_p[..., idx1]
    k_e = k.repeat_interleave(G, dim=1)
    pos = torch.arange(S, device=device)
    blk_end = (torch.arange(nblk, device=device) + 1) * BS - 1
    blk_lo = torch.arange(nblk, device=device) * BS
    res = {}
    for ri in range(TAIL_N):
        t_r = int(qpos[-TAIL_N + ri])
        if t_r < SWA + SINK + B_TOTAL + 512:
            continue
        sc_full = torch.einsum("hd,shd->hs", q[-TAIL_N + ri], k_e) * (D ** -0.5)
        sc_full = sc_full.masked_fill((pos > t_r).unsqueeze(0), float("-inf"))
        p_full = torch.softmax(sc_full, dim=-1).view(Hkv, G, S)
        q_sub = q[-TAIL_N + ri][..., idx1].reshape(Hkv, G, D_COARSE)
        sc_tok = torch.einsum("hgd,shd->hgs", q_sub, k_tok)

        def cov_mass(cand):
            return float((p_full * cand[..., :S]).sum(-1).mean())

        def select_region(lo, hi, n_pages, n_tokens):
            """区 [lo,hi) 内 minmax 块 top n_pages 页 → 池内 token 分数 top n_tokens。"""
            s1 = l1_minmax(kc, q_sub)  # [nblk,Hkv,G]
            s1 = s1.masked_fill((blk_end.view(-1, 1, 1) > t_r), float("-inf"))
            blk_in = (blk_lo < hi) & (blk_end >= lo)
            s1 = s1.masked_fill(~blk_in.view(-1, 1, 1), float("-inf"))
            ib = torch.topk(s1.permute(1, 2, 0), min(n_pages, nblk), dim=-1).indices
            blk_off = torch.arange(BS, device=device)
            tok = (ib.unsqueeze(-1) * BS + blk_off.view(1, 1, 1, BS)).clamp(max=S - 1)
            pool = torch.zeros(Hkv, G, S, dtype=torch.bool, device=device)
            pool.scatter_(2, tok.reshape(Hkv, G, -1), True)
            pool &= ((pos >= lo) & (pos < hi)).view(1, 1, S)
            sc_pool = sc_tok[..., :S].masked_fill(~pool, float("-inf"))
            sc_pool = sc_pool.masked_fill((pos > t_r).view(1, 1, -1), float("-inf"))
            it = torch.topk(sc_pool, min(n_tokens, S), dim=-1).indices
            cand = torch.zeros(Hkv, G, S, dtype=torch.bool, device=device)
            cand.scatter_(2, it, True)
            return cand

        sink_c = pos < SINK
        swa_c = (pos >= t_r + 1 - SWA) & (pos <= t_r)
        mid_hi = t_r + 1 - SWA               # mid 区上界（swa 之下）
        mid_budget = B_TOTAL - SWA           # 1280
        # ---- mono：mid 单池（alpha=0，TLI 简化版）----
        cand = torch.zeros(Hkv, G, S, dtype=torch.bool, device=device)
        cand |= sink_c.view(1, 1, S) | swa_c.view(1, 1, S)
        cand |= select_region(SINK, mid_hi, PAGE_POOL, mid_budget)
        res.setdefault("mono", []).append(cov_mass(cand))
        # ---- tli_anchor：mid 近端 1024 免打分全保留 + far 单池 128 块→256 ----
        keep_c = (pos >= mid_hi - 1024) & (pos < mid_hi)
        cand = torch.zeros(Hkv, G, S, dtype=torch.bool, device=device)
        cand |= sink_c.view(1, 1, S) | swa_c.view(1, 1, S) | keep_c.view(1, 1, S)
        cand |= select_region(SINK, mid_hi - 1024, PAGE_POOL, 256)
        res.setdefault("tli_anchor", []).append(cov_mass(cand))
        # ---- 双分区臂（用户 proposal：near 子区独立两级 + beta 独立页池）----
        for tag, a_frac, np_near, nt_near in [
            ("split_a50_b50_g5", 0.5, 64, 640),
            ("split_a50_b25_g5", 0.5, 32, 320),
            ("split_a25_b25_g5", 0.25, 32, 320),
        ]:
            near_L = int(a_frac * (mid_hi - SINK))
            near_lo = mid_hi - near_L
            cand = torch.zeros(Hkv, G, S, dtype=torch.bool, device=device)
            cand |= sink_c.view(1, 1, S) | swa_c.view(1, 1, S)
            cand |= select_region(near_lo, mid_hi, np_near, nt_near)          # near 子区独立粗+细
            cand |= select_region(SINK, near_lo, PAGE_POOL - np_near, mid_budget - nt_near)  # far 子区
            res.setdefault(tag, []).append(cov_mass(cand))
        res.setdefault("UPPER_full", []).append(1.0)
    return res


def main():
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
                for k2, v in layer_stats(tr, li, device).items():
                    agg.setdefault(k2, []).extend(v)
            out[f"{model}/{name}"] = {k2: round(sum(v) / len(v), 4) for k2, v in agg.items()}
            print(f"[{model}/{name}] " + " ".join(f"{k2}={v}" for k2, v in sorted(out[f'{model}/{name}'].items())), flush=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
