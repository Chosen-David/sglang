# #59 L1 块表示消融（离线，trace 重放）：minmax 上界 vs 块均值 avg vs
# 块中心范数点积（norm-dot）三种粗筛打分，mass recall（top-K1 块 vs
# 全维真实分布的 top token mass 覆盖）对比。用 8B/30B trace per-layer 重放。
# 口径：与 E3/E4c 一致（真实 K 全维打分做 oracle 分布，块选择为候选，
# 测候选覆盖的 mass）。
import json
import os

import torch

TRACE_DIRS = ["/tmp/trace/qwen3-8b", "/tmp/trace/qwen3-30b"]
K1 = 128  # k1_blocks
BS = 64  # block_size
D_COARSE = 32
NEAR_LEN = 2048
TAIL_N = 4  # 末尾行数
OUT = "/home/wangyuanshuo02/two-level-attention/exp/trace/results/e59_l1_ablation.json"


def l1_scores(kc, q):
    """三种块粗筛打分。kc: [nblk, Hkv, bs, d']；q: [Hkv, G, d']。
    返回 dict[str, [Hkv, G, nblk]]。"""
    kmax = kc.amax(2)  # [nblk, Hkv, d']
    kmin = kc.amin(2)
    kavg = kc.mean(2)
    # minmax 上界：正维取 max、负维取 min（现行实现语义）
    qg = q.clamp(min=0)
    qn = q.clamp(max=0)
    mm = torch.einsum("hgd,mhd->mhg", qg, kmax) + torch.einsum("mhd,hgd->mhg", kmin, qn)
    # avg：均值点积
    av = torch.einsum("hgd,mhd->mhg", q, kavg)
    return {"minmax": mm, "avg": av}


def layer_stats(tr, layer_id, device):
    t = torch.load(tr + f"/layer{layer_id:02d}.pt", map_location="cpu", weights_only=False)
    k = t["k"].to(device).float()
    q = t["q"].to(device).float()
    qpos = t["qpos"]
    S = t["S"]
    H, Hkv, D = q.shape[1], k.shape[1], k.shape[2]
    G = H // Hkv
    half = D // 2
    # E3 子空间：rotate_half 低频尾维（每半 16 维）
    idx1 = list(range(half - 16, half)) + list(range(D - 16, D))
    nblk = (S + BS - 1) // BS
    pad = nblk * BS - S
    k_p = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, pad))
    kc = k_p[..., idx1].reshape(nblk, BS, Hkv, D_COARSE).permute(0, 2, 1, 3)  # [nblk, Hkv, bs, d']
    # nblk 已在 pad 段计算
    k_e = k.repeat_interleave(G, dim=1)  # [S, H, D]
    pos = torch.arange(S, device=device)
    res = {}
    for ri in range(TAIL_N):
        t_r = int(qpos[-TAIL_N + ri])
        if t_r < NEAR_LEN + 128:
            continue
        # oracle 分布：全维真实打分
        sc_full = torch.einsum("hd,shd->hs", q[-TAIL_N + ri], k_e) * (D**-0.5)
        m = pos <= t_r
        sc_full = sc_full.masked_fill(~m.unsqueeze(0), float("-inf"))
        p_full = torch.softmax(sc_full, dim=-1).view(Hkv, G, S)
        # far 区 oracle mass（top-1024）
        far_hi = max(128, t_r + 1 - NEAR_LEN)
        in_far = (pos >= 128) & (pos < far_hi)
        q_sub = q[-TAIL_N + ri][..., idx1].reshape(Hkv, G, D_COARSE)
        sc1 = l1_scores(kc, q_sub)
        for name, s1 in sc1.items():
            # s1: [nblk, Hkv, G]（einsum mhg 输出顺序）
            blk_end = (torch.arange(nblk, device=device) + 1) * BS - 1
            s1 = s1.masked_fill((blk_end.view(-1, 1, 1) > t_r), float("-inf"))
            s1 = s1.permute(1, 2, 0)  # [Hkv, G, nblk]
            i_blk = torch.topk(s1, min(K1, nblk), dim=-1).indices  # [Hkv, G, K1]
            # 候选 token mask（i_blk: [Hkv, G, K1]）
            cand = torch.zeros(Hkv, G, S, dtype=torch.bool, device=device)
            blk_off = torch.arange(BS, device=device)
            tok = (i_blk.unsqueeze(-1) * BS + blk_off.view(1, 1, 1, BS)).clamp(max=S - 1)  # [Hkv, G, K1, BS]
            cand.scatter_(2, tok.reshape(Hkv, G, -1), True)
            cand &= in_far.view(1, 1, S)
            # far mass recall：候选覆盖的 far mass / far 区 top-1024 mass
            pf = p_full * in_far.view(1, 1, S)
            srt = torch.sort(pf.flatten(2), descending=True).values
            denom = srt[:, :, :1024].sum(-1).clamp(min=1e-9)  # [Hkv, G]
            num = (pf * cand).sum(-1)
            rec = (num / denom).mean()
            res.setdefault(name, []).append(float(rec))
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
                r = layer_stats(tr, li, device)
                for k2, v in r.items():
                    agg.setdefault(k2, []).extend(v)
            rec = {k2: round(sum(v) / len(v), 4) for k2, v in agg.items()}
            out[f"{model}/{name}"] = rec
            print(f"[{model}/{name}] " + " ".join(f"{k2}={v}" for k2, v in rec.items()), flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
