# E65：降维探索（CPU 版，纯精度实验——用户指令：near_method 能不能降维、far_method 能不能降维分开测；
#   粗筛 PCA/截维/其他降维法；细筛也降维=性能优化重要一步）。
# 维度来源三法：
#   trunc：低频尾维直接截取前 d 维（NoPE 代理的子集）
#   pca  ：每 (layer, kv-head) far 区 SVD top-r 主成分投影（复用 M9 校准哲学，basis 现场算）
#   jl   ：固定种子高斯随机投影 128→d（JL 引理对照）
# 四组测量（far/near 分测、粗筛/细筛分测，每组的"其余部分"固定在尾维 32 不降）：
#   A far 粗筛降维：far 区 [SINK, mid_hi-4096) 用 d 维特征建 minmax 块分 → top16 块 → 尾维32 token 细筛 top512 → far 区 mass 捕获
#   B near 粗筛降维：near 带 [mid_hi-4096, mid_hi) 用 d 维特征建 avg 块分 → top16 块 → 尾维32 细筛 top512 → near 带 mass 捕获
#   C 细筛降维：粗筛固定尾维32 minmax top16 块 → d 维特征 token 打分 top512 → far 区捕获
#   D 全链降维（粗+细同维）：粗筛 d 维 + 细筛 d 维 → far 区捕获
# 维度扫描 d ∈ {4,8,16,32}（32=基准），另报 128 全维 oracle。
# 口径：真实全维 softmax 的 per-head far mass 加权捕获（E4c 同款）；8 样本×8 层×末 2 query，CPU。
import json
import os

import torch

TRACE = "/tmp/trace/qwen3-8b"
OUT = "/home/wangyuanshuo02/two-level-attention/exp/trace/results/e65_dim_reduction.json"
D2I = list(range(48, 64)) + list(range(112, 128))   # 尾维 32（顺序：前 16 低频 + 后 16）
SINK, SWA, NEAR_BAND = 128, 1024, 4096
DIMS = [4, 8, 16, 32]
BUD_TOK = 512
N_PAGES = 16
BS = 64
TAIL_N = 2
SAMPLES = ["lb_gov_report_0", "lb_hotpotqa_0", "lb_musique_0", "lb_narrativeqa_0",
            "lb_passage_retrieval_en_0", "lb_qasper_0", "needle32k", "natural32k"]


jl32 = None  # [32,32] 子空间 JL 投影，main() 里初始化


def feats(k, q, dim, method, basis=None, jl=None):
    """k [S,Hkv,128], q [Hkv,128]（已 G 聚合）→ 特征 [S,Hkv,d]/[Hkv,d]。
    trunc：尾维前 d 维；pca：全维投影 basis[h][:,:d]；jl：全维高斯投影。"""
    if method == "trunc":
        idx = torch.tensor(D2I[:dim])
        return k[..., idx], q[..., idx]
    if method == "pca":
        kf, qf = [], []
        for h in range(k.shape[1]):
            b = basis[h][:, :dim]                       # [128, d]
            kf.append(k[:, h] @ b); qf.append(q[h] @ b)
        return torch.stack(kf, 1), torch.stack(qf, 0)
    if method == "jl":
        g = jl[:, :dim]
        return k @ g, q @ g
    if method == "pca_sub":          # NoPE 子空间（尾维32）内 PCA 降维
        idx = torch.tensor(D2I)
        k32, q32 = k[..., idx], q[..., idx]
        kf, qf = [], []
        for h in range(k.shape[1]):
            b = basis[h][:, :dim]                   # [32, d]
            kf.append(k32[:, h] @ b); qf.append(q32[h] @ b)
        return torch.stack(kf, 1), torch.stack(qf, 0)
    if method == "jl_sub":           # NoPE 子空间内 JL 随机投影
        idx = torch.tensor(D2I)
        g = jl32[:, :dim]
        return k[..., idx] @ g, q[..., idx] @ g
    raise ValueError(method)


def blk_minmax(kf, qf, nblk):
    """kf [S,Hkv,d] → pad 分块 → minmax 上界块分 [Hkv,nblk]。"""
    S, Hkv, d = kf.shape
    kk = torch.nn.functional.pad(kf, (0, 0, 0, 0, 0, nblk * BS - S))
    kc = kk.reshape(nblk, BS, Hkv, d)
    kmin, kmax = kc.amin(1), kc.amax(1)                 # [nblk,Hkv,d]
    return (torch.einsum("hd,nhd->hn", qf.clamp(min=0), kmax) +
            torch.einsum("hd,nhd->hn", qf.clamp(max=0), kmin))


def blk_avg(kf, qf, nblk):
    S, Hkv, d = kf.shape
    kk = torch.nn.functional.pad(kf, (0, 0, 0, 0, 0, nblk * BS - S))
    kavg = kk.reshape(nblk, BS, Hkv, d).mean(1)         # [nblk,Hkv,d]
    return torch.einsum("hd,nhd->hn", qf, kavg)


def select_region(coarse_kf, q_coarse, blk_fn, fine_kf, q_fine):
    """区特征 coarse_kf [T,Hkv,dc] / fine_kf [T,Hkv,df]（同区同长）。
    粗筛 top N_PAGES 块 → 池内 fine 特征 token 分 top BUD_TOK。返回区内相对索引 [Hkv,512]。"""
    T, Hkv, _ = coarse_kf.shape
    nblk = (T + BS - 1) // BS
    sc = blk_fn(coarse_kf, q_coarse, nblk)               # [Hkv, nblk]
    ib = torch.topk(sc, min(N_PAGES, nblk), dim=-1).indices
    pool = torch.zeros(Hkv, T, dtype=torch.bool)
    tok = (ib.unsqueeze(-1) * BS + torch.arange(BS).view(1, 1, BS)).reshape(Hkv, -1).clamp(max=T - 1)
    pool.scatter_(1, tok, True)
    ts = torch.einsum("hd,shd->hs", q_fine, fine_kf).masked_fill(~pool, float("-inf"))
    it = torch.topk(ts, min(BUD_TOK, T), dim=-1).indices
    return it


def eval_layer(lf, jl):
    d = torch.load(lf, map_location="cpu", weights_only=False)
    k, q, qpos, S = d["k"].float(), d["q"].float(), d["qpos"], d["S"]
    Hkv, H, D = k.shape[1], q.shape[1], k.shape[-1]
    G = H // Hkv
    res = {}
    # PCA basis（每层现场校准：far 区 SVD top-32，取前 d 用）——离线校准哲学同 M9
    t = int(qpos[-1])
    mid_hi = t + 1 - SWA
    far_lo, far_hi = SINK, mid_hi - NEAR_BAND
    near_lo, near_hi = mid_hi - NEAR_BAND, mid_hi
    if far_hi - far_lo < 8192:
        del k, q, d
        return None
    basis, basis_sub = [], []
    idx_all = torch.tensor(D2I)
    for h in range(Hkv):
        _, _, Vt = torch.linalg.svd(k[far_lo:far_hi, h], full_matrices=False)
        basis.append(Vt[:32].T)                          # [128, 32] 全维 PCA
        _, _, Vt2 = torch.linalg.svd(k[far_lo:far_hi, h][..., idx_all], full_matrices=False)
        basis_sub.append(Vt2[:32].T)                     # [32, 32] NoPE 子空间 PCA
    for ri in range(TAIL_N):
        q_t = q[-TAIL_N + ri]                            # [H,128]
        q_head = q_t.reshape(Hkv, G, D).sum(1)           # [Hkv,128]（E4c 口径）
        # 真值分布（per Hkv，G 求和后 softmax——E4c 同款）
        s = torch.einsum("hd,shd->hs", q_head, k) * (D ** -0.5)
        s = s.masked_fill(torch.arange(S).view(1, -1) > int(qpos[-TAIL_N + ri]), float("-inf"))
        p = torch.softmax(s, dim=-1)                     # [Hkv, S]
        far_h = p[:, far_lo:far_hi].sum(-1)
        near_h = p[:, near_lo:near_hi].sum(-1)
        tot_f, tot_n = float(far_h.sum()), float(near_h.sum())
        if tot_f < 1e-6 or tot_n < 1e-6:
            continue
        idx32 = torch.tensor(D2I)
        k_t32 = k[..., idx32]                            # [S,Hkv,32] 细筛基准特征
        q_t32 = q_head[..., idx32]
        for dim in DIMS:
            for meth in ("trunc", "pca", "jl", "pca_sub", "jl_sub"):
                kf, qf = feats(k, q_head, dim, meth, basis_sub if meth == "pca_sub" else basis, jl)
                kf_far, kf_near = kf[far_lo:far_hi], kf[near_lo:near_hi]
                # A far 粗筛降维（细筛固定尾维32 minmax→t32）
                toks = select_region(kf_far, qf, blk_minmax, k_t32[far_lo:far_hi], q_t32)
                cap = sum(float(p[h, toks[h] + far_lo].sum()) for h in range(Hkv))
                res.setdefault(f"A_far_coarse_{meth}_d{dim}", []).append(cap / tot_f)
                # B near 粗筛降维（avg 块）
                toks = select_region(kf_near, qf, blk_avg, k_t32[near_lo:near_hi], q_t32)
                cap = sum(float(p[h, toks[h] + near_lo].sum()) for h in range(Hkv))
                res.setdefault(f"B_near_coarse_{meth}_d{dim}", []).append(cap / tot_n)
                # C 细筛降维（粗筛固定尾维32 minmax）
                toks = select_region(k_t32[far_lo:far_hi], q_t32, blk_minmax, kf_far, qf)
                cap = sum(float(p[h, toks[h] + far_lo].sum()) for h in range(Hkv))
                res.setdefault(f"C_fine_{meth}_d{dim}", []).append(cap / tot_f)
                # D 全链降维（粗+细同 d 维）
                toks = select_region(kf_far, qf, blk_minmax, kf_far, qf)
                cap = sum(float(p[h, toks[h] + far_lo].sum()) for h in range(Hkv))
                res.setdefault(f"D_full_{meth}_d{dim}", []).append(cap / tot_f)
        # 基准：全维 oracle（far 区 token 级 top512，128 维精确分）
        s_far = s[:, far_lo:far_hi]
        order = torch.argsort(s_far, dim=-1, descending=True)
        cap = sum(float(p[h, order[h][:BUD_TOK] + far_lo].sum()) for h in range(Hkv))
        res.setdefault("oracle128", []).append(cap / tot_f)
    del k, q, d
    return res


def main():
    torch.set_num_threads(21)
    g = torch.Generator().manual_seed(0)
    global jl32
    jl = torch.randn(128, 32, generator=g)               # JL 投影（列子集取前 d）
    jl32 = torch.randn(32, 32, generator=g)              # 子空间 JL
    results = {}
    for name in SAMPLES:
        meta = f"{TRACE}/{name}/meta.json"
        if not os.path.isfile(meta):
            continue
        n_layers = json.load(open(meta))["n_layers"]
        agg = {}
        for li in range(0, n_layers, max(1, n_layers // 8)):
            r = eval_layer(f"{TRACE}/{name}/layer{li:02d}.pt", jl)
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
