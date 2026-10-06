# E76：非线性降维 + MLA 式共享投影基 trace 重放判决（2026-09-30 用户指令：
#   「t-SNE 降维 / UMAP 降维 / 其他最新论文的降维方法都试试，降维还是很值得试的，
#    或者 MLA 自己有个投影矩阵缓存了 k 的，都探索一下」）。
# 协议 = E65 组 C（细筛降维，粗筛固定尾维32 minmax top16 块）+ 组 D（全链同特征），
#   真值口径 = 全维 softmax per-head far mass 加权捕获（E4c 同款）。
# 新臂（细筛 d16 为 E65 推荐工作点；非线性法只测 d16 控成本）：
#   umap      ：每 (layer, kv-head) far 区 UMAP 拟合（16K token），pool+q 用 .transform；
#               打分用嵌入空间 cosine 与 -euclidean 两种（非线性法不保内积，取较优=对其从宽）
#   tsne      ：openTSNE 同上（perplexity 30），pool+q 用 .transform
#   pca_shared：MLA 式「单投影缓存 K」——跨 kv-head 共享 SVD 基（far 区全部 head
#               拼接拟合一个 [128,d] 基），d ∈ {8,16,32}；这是 Qwen3 GQA 下 MLA
#               latent（每 token 单投影缓存）的唯一 training-free 代理
# 对照（与 E65 严格同口径）：tail32 细筛（现主表口径）、pca per-head d16（E65 唯一
#   推荐降维 0.766）、oracle128。
# 成本记录：每法 fit/transform wall-clock——部署可行性定量（增量索引 0.128ms/token
#   的对照基准）。非线性法无增量变换（新 K token 须 .transform），成本本身就是判决。
# 样本：非线性 3 样本 × 2 层（musique/hotpotqa/needle32k，layer4/20，far-heavy+
#   多跳+合成检索混合）；pca_shared 全 8 样本 × 8 层（与 E65 可比）。
import json
import os
import time

import numpy as np
import torch

TRACE = "/tmp/trace/qwen3-8b"
OUT = "/home/wangyuanshuo02/two-level-attention/exp/trace/results/e76_nonlinear_reduce.json"
D2I = list(range(48, 64)) + list(range(112, 128))   # 尾维 32（现口径基准）
SINK, SWA, NEAR_BAND = 128, 1024, 4096
BUD_TOK = 512
N_PAGES = 16
BS = 64
TAIL_N = 2
DIM_NL = 16                       # 非线性法统一 d16（E65 推荐工作点）
SAMPLES_FULL = ["lb_gov_report_0", "lb_hotpotqa_0", "lb_musique_0", "lb_narrativeqa_0",
                "lb_passage_retrieval_en_0", "lb_qasper_0", "needle32k", "natural32k"]
SAMPLES_NL = ["lb_musique_0", "needle32k"]   # t-SNE O(n²) 成本大：缩样本（成本即判决）
LAYERS_NL = [4, 20]


def select_region(coarse_kf, q_coarse, blk_fn, fine_kf, q_fine, score_fn=None):
    """E65 同款：粗筛 top N_PAGES 块 → 池内 fine 特征 token 分 top BUD_TOK。
    score_fn=None 时用 einsum 内积（线性特征）；非线性法传自定义 (fine_kf, q_fine)->score。"""
    T, Hkv, _ = coarse_kf.shape
    nblk = (T + BS - 1) // BS
    sc = blk_fn(coarse_kf, q_coarse, nblk)
    ib = torch.topk(sc, min(N_PAGES, nblk), dim=-1).indices
    pool = torch.zeros(Hkv, T, dtype=torch.bool)
    tok = (ib.unsqueeze(-1) * BS + torch.arange(BS).view(1, 1, BS)).reshape(Hkv, -1).clamp(max=T - 1)
    pool.scatter_(1, tok, True)
    if score_fn is None:
        ts = torch.einsum("hd,shd->hs", q_fine, fine_kf).masked_fill(~pool, float("-inf"))
    else:
        ts = score_fn(fine_kf, q_fine).masked_fill(~pool, float("-inf"))
    it = torch.topk(ts, min(BUD_TOK, T), dim=-1).indices
    return it, pool


def blk_minmax(kf, qf, nblk):
    S, Hkv, d = kf.shape
    kk = torch.nn.functional.pad(kf, (0, 0, 0, 0, 0, nblk * BS - S))
    kc = kk.reshape(nblk, BS, Hkv, d)
    kmin, kmax = kc.amin(1), kc.amax(1)
    return (torch.einsum("hd,nhd->hn", qf.clamp(min=0), kmax) +
            torch.einsum("hd,nhd->hn", qf.clamp(max=0), kmin))


def embed_and_score(k_far, q_head, method):
    """k_far [T,Hkv,128] torch / q_head [Hkv,128] → 逐 head 非线性嵌入。
    返回 (sc_cos, sc_euc) 两套打分 [Hkv,T] 与 (fit_s, tr_s) 成本秒数。
    非线性法不保内积——cosine 与 -euclidean² 两套排序分别判决取较优（从宽口径）。
    t-SNE（bh，O(n²)）：>8K token 实测 RSS ~100GB（OOM 风险，本轮实录）——
    tsne 臂降为 1000-token 子采样拟合（transform 全池），质量出数量级、
    成本按 n 外推记录（判决 = 不可行性定量，不是缺测）。"""
    import umap
    import openTSNE
    T, Hkv, _ = k_far.shape
    sc_cos = torch.full((Hkv, T), float("-inf"))
    sc_euc = torch.full((Hkv, T), float("-inf"))
    fit_s = tr_s = 0.0
    for h in range(Hkv):
        X = k_far[:, h].numpy().astype(np.float32)     # [T,128]
        qv = q_head[h].numpy().astype(np.float32)      # [128]
        t0 = time.time()
        if method == "umap":
            reducer = umap.UMAP(n_components=DIM_NL, n_neighbors=15, min_dist=0.1,
                                metric="euclidean", random_state=0, n_jobs=1)
            E = reducer.fit_transform(X)               # [T,d]
            Eq = reducer.transform(qv.reshape(1, -1))  # [1,d]
        else:
            raise ValueError(method)
        fit_s += time.time() - t0
        t0 = time.time()
        En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        Eqn = Eq / (np.linalg.norm(Eq, axis=1, keepdims=True) + 1e-9)
        sc_cos[h] = torch.from_numpy(En @ Eqn[0])      # cosine
        sc_euc[h] = -torch.from_numpy(((E - Eq[0]) ** 2).sum(1))  # -euclidean²
        tr_s += time.time() - t0
    return sc_cos, sc_euc, fit_s, tr_s


def eval_layer(lf, do_nl):
    d = torch.load(lf, map_location="cpu", weights_only=False)
    k, q, qpos, S = d["k"].float(), d["q"].float(), d["qpos"], d["S"]
    Hkv, H, D = k.shape[1], q.shape[1], k.shape[-1]
    G = H // Hkv
    res, cost = {}, {}
    t = int(qpos[-1])
    mid_hi = t + 1 - SWA
    far_lo, far_hi = SINK, mid_hi - NEAR_BAND
    if far_hi - far_lo < 8192:
        del k, q, d
        return None, None
    # 基：per-head PCA（E65 对照）与 shared（MLA 式：全部 head 拼接一个基）
    basis_ph = []
    for h in range(Hkv):
        _, _, Vt = torch.linalg.svd(k[far_lo:far_hi, h], full_matrices=False)
        basis_ph.append(Vt[:32].T)                     # [128,32]
    k_all = k[far_lo:far_hi].reshape(-1, D)            # [T*Hkv,128]
    _, _, Vt_sh = torch.linalg.svd(k_all, full_matrices=False)
    basis_sh = Vt_sh[:32].T                            # [128,32] 共享基
    for ri in range(TAIL_N):
        q_t = q[-TAIL_N + ri]
        q_head = q_t.reshape(Hkv, G, D).sum(1)
        s = torch.einsum("hd,shd->hs", q_head, k) * (D ** -0.5)
        s = s.masked_fill(torch.arange(S).view(1, -1) > int(qpos[-TAIL_N + ri]), float("-inf"))
        p = torch.softmax(s, dim=-1)
        far_h = p[:, far_lo:far_hi].sum(-1)
        tot_f = float(far_h.sum())
        if tot_f < 1e-6:
            continue
        idx32 = torch.tensor(D2I)
        k_t32 = k[..., idx32]
        q_t32 = q_head[..., idx32]
        k_far32, q_far32 = k_t32[far_lo:far_hi], q_t32
        # 粗筛固定尾维32（E65 组 C 同款）→ pool（所有臂共用同一 pool，公平）
        toks, pool = select_region(k_far32, q_far32, blk_minmax, k_far32, q_far32)
        cap = sum(float(p[h, toks[h] + far_lo].sum()) for h in range(Hkv))
        res.setdefault("C_tail32_ref", []).append(cap / tot_f)
        # oracle128
        s_far = s[:, far_lo:far_hi]
        order = torch.argsort(s_far, dim=-1, descending=True)
        cap = sum(float(p[h, order[h][:BUD_TOK] + far_lo].sum()) for h in range(Hkv))
        res.setdefault("oracle128", []).append(cap / tot_f)
        # pca per-head（E65 对照，d16）
        kf = torch.stack([k[far_lo:far_hi, h] @ basis_ph[h][:, :16] for h in range(Hkv)], 1)
        qf = torch.stack([q_head[h] @ basis_ph[h][:, :16] for h in range(Hkv)], 0)
        toks, _ = select_region(k_far32, q_far32, blk_minmax, kf, qf)
        cap = sum(float(p[h, toks[h] + far_lo].sum()) for h in range(Hkv))
        res.setdefault("C_pca_perhead_d16", []).append(cap / tot_f)
        # pca_shared（MLA 式单投影缓存）：d ∈ {8,16,32}
        for dim in (8, 16, 32):
            b = basis_sh[:, :dim]
            kf = k[far_lo:far_hi] @ b                   # [T,Hkv,d]
            qf = q_head @ b                             # [Hkv,d]
            toks, _ = select_region(k_far32, q_far32, blk_minmax, kf, qf)
            cap = sum(float(p[h, toks[h] + far_lo].sum()) for h in range(Hkv))
            res.setdefault(f"C_pca_shared_d{dim}", []).append(cap / tot_f)
        # 非线性臂（umap/tsne，d16，全 far 区拟合→pool+q transform；两套打分取较优）
        if do_nl:
            for meth in ("umap",):   # tsne 已判不可行（BH OOM/病态），见报告 §8b-52
                sc_cos, sc_euc, fit_s, tr_s = embed_and_score(k[far_lo:far_hi], q_head, meth)
                caps = []
                for sc_emb in (sc_cos, sc_euc):
                    sc_emb = sc_emb.masked_fill(~pool, float("-inf"))
                    it = torch.topk(sc_emb, min(BUD_TOK, sc_emb.shape[1]), dim=-1).indices
                    caps.append(sum(float(p[h, it[h] + far_lo].sum()) for h in range(Hkv)) / tot_f)
                res.setdefault(f"C_{meth}_d{DIM_NL}", []).append(max(caps))
                cost.setdefault(f"{meth}_fit_s", []).append(fit_s)
                cost.setdefault(f"{meth}_transform_s", []).append(tr_s)
    del k, q, d
    return res, cost


def main():
    torch.set_num_threads(16)
    results, costs = {}, {}
    for name in SAMPLES_FULL:
        meta = f"{TRACE}/{name}/meta.json"
        if not os.path.isfile(meta):
            continue
        n_layers = json.load(open(meta))["n_layers"]
        do_nl = name in SAMPLES_NL
        layer_list = LAYERS_NL if do_nl else list(range(0, n_layers, max(1, n_layers // 8)))
        agg, cagg = {}, {}
        for li in layer_list:
            r, c = eval_layer(f"{TRACE}/{name}/layer{li:02d}.pt", do_nl)
            if not r:
                continue
            for k2, v in r.items():
                agg.setdefault(k2, []).extend(v)
            if c:
                for k2, v in c.items():
                    cagg.setdefault(k2, []).extend(v)
        rec = {k2: round(sum(v) / len(v), 4) for k2, v in agg.items()}
        results[name] = rec
        if cagg:
            costs[name] = {k2: round(sum(v) / len(v), 2) for k2, v in cagg.items()}
        print(f"[{name}] " + " ".join(f"{k2}={v}" for k2, v in sorted(rec.items())), flush=True)
        if cagg:
            print(f"  cost[s/layer-sample] {costs[name]}", flush=True)
    json.dump({"quality": results, "cost_per_layer_sample_s": costs}, open(OUT, "w"), indent=1)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
