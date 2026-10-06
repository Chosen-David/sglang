# E73：near/far 全链 nope 维度 + nope 内压维扫描（2026-09-30 用户指令：
#   「near 侧可以全部 nope 维度先试试——粗筛+细筛全在 nope 维度，然后进一步看能不能压维度。
#    都看看能不能压缩试试」）。
# 与 E65 的区别：E65 的降维全部以尾维 32（tail）为锚（trunc 取 D2I 前 d 维）；
#   E73 把特征源换成 **nope 整段 64 维（后 64 非旋转维，range(64,128)）**，
#   粗筛（minmax/avg 块分）+ 细筛（token 打分）全链用 nope 段，再在段内压维：
#   trunc_nope：nope 段内取连续尾 d 维（d=64 即整段）
#   pca_nope  ：nope 段内 PCA top-d（basis 每 (layer,kv-head) far 区现场 SVD）
#   jl_nope   ：nope 段内固定种子高斯投影（JL 对照）
# 测量组（far/near 分测，每组「粗筛+细筛」都用同一特征——全链口径）：
#   F：far 全链 nope（粗筛 minmax + 细筛均 nope 特征）→ far mass 捕获
#   N：near 全链 nope（粗筛 avg + 细筛均 nope 特征）→ near mass 捕获
# 对照：tail32（现主表口径=尾维32全链）、nope64（整段不压）、oracle128（全维 token 级 top512）。
# 维度扫描 d ∈ {4,8,16,32,64}；口径：真实全维 softmax per-head far/near mass 加权（E4c 同款）。
# 另：MLA qk 下投影矩阵——Qwen3-8B 为纯 GQA 满秩 q/k_proj 无低秩结构，本实验不适用（记录）。
import json
import os

import torch

TRACE = "/tmp/trace/qwen3-8b"
OUT = "/home/wangyuanshuo02/two-level-attention/exp/trace/results/e73_nope_reduction.json"
NOPE = list(range(64, 128))          # nope 整段：后 64 非旋转维
TAIL32 = list(range(48, 64)) + list(range(112, 128))   # 尾维 32（现口径基准）
SINK, SWA, NEAR_BAND = 128, 1024, 4096
DIMS = [4, 8, 16, 32, 64]
BUD_TOK = 512
N_PAGES = 16
BS = 64
TAIL_N = 2
SAMPLES = ["lb_gov_report_0", "lb_hotpotqa_0", "lb_musique_0", "lb_narrativeqa_0",
            "lb_passage_retrieval_en_0", "lb_qasper_0", "needle32k", "natural32k"]

jl64 = None  # [64,64] nope 段 JL 投影，main() 初始化


def feats_nope(k, q_head, dim, method, basis=None):
    """k [S,Hkv,128] / q_head [Hkv,128] → nope 段特征 [S,Hkv,d]/[Hkv,d]。
    trunc_nope：nope 段内取后 d 维（与 tail 口径同向：低索引侧为段尾）；
    pca_nope：nope 段 [S,64] SVD top-d 投影；jl_nope：段内高斯投影前 d 列。"""
    idx = torch.tensor(NOPE)
    kn, qn = k[..., idx], q_head[..., idx]          # [S,Hkv,64] / [Hkv,64]
    if method == "trunc_nope":
        sub = torch.tensor(NOPE[-dim:])             # 段内取「后 d 维」（112..128 方向）
        return k[..., sub], q_head[..., sub]
    if method == "pca_nope":
        kf, qf = [], []
        for h in range(k.shape[1]):
            b = basis[h][:, :dim]                   # [64, d]
            kf.append(kn[:, h] @ b); qf.append(qn[h] @ b)
        return torch.stack(kf, 1), torch.stack(qf, 0)
    if method == "jl_nope":
        g = jl64[:, :dim]
        return kn @ g, qn @ g
    raise ValueError(method)


def feats_tail(k, q_head):
    idx = torch.tensor(TAIL32)
    return k[..., idx], q_head[..., idx]


def blk_minmax(kf, qf, nblk):
    S, Hkv, d = kf.shape
    kk = torch.nn.functional.pad(kf, (0, 0, 0, 0, 0, nblk * BS - S))
    kc = kk.reshape(nblk, BS, Hkv, d)
    kmin, kmax = kc.amin(1), kc.amax(1)
    return (torch.einsum("hd,nhd->hn", qf.clamp(min=0), kmax) +
            torch.einsum("hd,nhd->hn", qf.clamp(max=0), kmin))


def blk_avg(kf, qf, nblk):
    S, Hkv, d = kf.shape
    kk = torch.nn.functional.pad(kf, (0, 0, 0, 0, 0, nblk * BS - S))
    kavg = kk.reshape(nblk, BS, Hkv, d).mean(1)
    return torch.einsum("hd,nhd->hn", qf, kavg)


def select_region(coarse_kf, q_coarse, blk_fn, fine_kf, q_fine):
    """粗筛 top N_PAGES 块 → 池内 fine 特征 token 分 top BUD_TOK。返回区内相对索引 [Hkv,512]。"""
    T, Hkv, _ = coarse_kf.shape
    nblk = (T + BS - 1) // BS
    sc = blk_fn(coarse_kf, q_coarse, nblk)
    ib = torch.topk(sc, min(N_PAGES, nblk), dim=-1).indices
    pool = torch.zeros(Hkv, T, dtype=torch.bool)
    tok = (ib.unsqueeze(-1) * BS + torch.arange(BS).view(1, 1, BS)).reshape(Hkv, -1).clamp(max=T - 1)
    pool.scatter_(1, tok, True)
    ts = torch.einsum("hd,shd->hs", q_fine, fine_kf).masked_fill(~pool, float("-inf"))
    it = torch.topk(ts, min(BUD_TOK, T), dim=-1).indices
    return it


def eval_layer(lf):
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
        del k, q, d
        return None
    # nope 段 PCA basis（每 kv-head far 区现场 SVD——离线校准哲学同 E65/M9）
    idx_nope = torch.tensor(NOPE)
    basis = []
    for h in range(Hkv):
        _, _, Vt = torch.linalg.svd(k[far_lo:far_hi, h][..., idx_nope], full_matrices=False)
        basis.append(Vt[:64].T)                     # [64, 64]
    for ri in range(TAIL_N):
        q_t = q[-TAIL_N + ri]
        q_head = q_t.reshape(Hkv, G, D).sum(1)      # [Hkv,128]（E4c 口径）
        s = torch.einsum("hd,shd->hs", q_head, k) * (D ** -0.5)
        s = s.masked_fill(torch.arange(S).view(1, -1) > int(qpos[-TAIL_N + ri]), float("-inf"))
        p = torch.softmax(s, dim=-1)
        far_h = p[:, far_lo:far_hi].sum(-1)
        near_h = p[:, near_lo:near_hi].sum(-1)
        tot_f, tot_n = float(far_h.sum()), float(near_h.sum())
        if tot_f < 1e-6 or tot_n < 1e-6:
            continue
        k_far, k_near = k[far_lo:far_hi], k[near_lo:near_hi]
        # 基准 1：tail32 全链（现主表口径：粗筛+细筛均尾维32）
        kt, qt = feats_tail(k, q_head)
        toks = select_region(kt[far_lo:far_hi], qt, blk_minmax, kt[far_lo:far_hi], qt)
        cap = sum(float(p[h, toks[h] + far_lo].sum()) for h in range(Hkv))
        res.setdefault("F_tail32_ref", []).append(cap / tot_f)
        toks = select_region(kt[near_lo:near_hi], qt, blk_avg, kt[near_lo:near_hi], qt)
        cap = sum(float(p[h, toks[h] + near_lo].sum()) for h in range(Hkv))
        res.setdefault("N_tail32_ref", []).append(cap / tot_n)
        # E73 主体：nope 全链 + 段内压维（三法 × d 梯度）
        for dim in DIMS:
            for meth in ("trunc_nope", "pca_nope", "jl_nope"):
                kf, qf = feats_nope(k, q_head, dim, meth, basis)
                # F far 全链 nope（minmax 粗筛 + nope 细筛）
                toks = select_region(kf[far_lo:far_hi], qf, blk_minmax, kf[far_lo:far_hi], qf)
                cap = sum(float(p[h, toks[h] + far_lo].sum()) for h in range(Hkv))
                res.setdefault(f"F_nope_{meth}_d{dim}", []).append(cap / tot_f)
                # N near 全链 nope（avg 粗筛 + nope 细筛）
                toks = select_region(kf[near_lo:near_hi], qf, blk_avg, kf[near_lo:near_hi], qf)
                cap = sum(float(p[h, toks[h] + near_lo].sum()) for h in range(Hkv))
                res.setdefault(f"N_nope_{meth}_d{dim}", []).append(cap / tot_n)
        # 基准 2：oracle128（far 全维 token 级 top512）
        s_far = s[:, far_lo:far_hi]
        order = torch.argsort(s_far, dim=-1, descending=True)
        cap = sum(float(p[h, order[h][:BUD_TOK] + far_lo].sum()) for h in range(Hkv))
        res.setdefault("F_oracle128", []).append(cap / tot_f)
        order_n = torch.argsort(s[:, near_lo:near_hi], dim=-1, descending=True)
        cap = sum(float(p[h, order_n[h][:BUD_TOK] + near_lo].sum()) for h in range(Hkv))
        res.setdefault("N_oracle128", []).append(cap / tot_n)
    del k, q, d
    return res


def main():
    torch.set_num_threads(21)
    global jl64
    g = torch.Generator().manual_seed(0)
    jl64 = torch.randn(64, 64, generator=g)         # nope 段 JL 投影
    results = {}
    for name in SAMPLES:
        meta = f"{TRACE}/{name}/meta.json"
        if not os.path.isfile(meta):
            continue
        n_layers = json.load(open(meta))["n_layers"]
        agg = {}
        for li in range(0, n_layers, max(1, n_layers // 8)):
            r = eval_layer(f"{TRACE}/{name}/layer{li:02d}.pt")
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
