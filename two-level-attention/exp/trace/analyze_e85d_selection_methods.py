# E85d：子空间选取方法对比——SparQ 式逐 query |q| 幅值选取 vs K 方差选取
# vs 旋转对频率先验（tail32）vs pair 感知自适应。
# 动机（用户 2026-09-30 指令）：「想法类似 sparQ 或者其他论文做法那种，真的
# 找到合适的维度作为子空间……你的选取方法是什么」。
# SparQ Attention（Likhomanov et al.）的选取 = 每 query 取 |q| 最大的 r 维做
# 部分点积近似（其 Σ|q_T| 归一化是 per-query 常数，不影响 key 排序，故本
# 重放的 partial-dot 排序与 SparQ 等价）。K 方差选取 = 静态数据驱动（K 统计
# top 方差维）。判决问题：数据驱动选取（query 自适应或 K 统计）能否打过
# 旋转对频率先验；pair 感知自适应是否是二者的正确结合方式。
# 口径与 E85a/b/c 完全一致。
import json
import os

import torch

TRACE = "/tmp/trace/qwen3-8b"
OUT = "/home/wangyuanshuo02/two-level-attention/exp/trace/results/e85d_selection_methods.json"
SINK, SWA, NEAR_BAND = 128, 1024, 4096
BUD_TOK = 512
N_PAGES = 16
BS = 64
TAIL_N = 2
SAMPLES = ["lb_gov_report_0", "lb_hotpotqa_0", "lb_musique_0", "lb_narrativeqa_0",
           "lb_passage_retrieval_en_0", "lb_qasper_0", "needle32k", "natural32k"]


def select_region_fixed(coarse_kf, q_coarse, fine_kf, q_fine):
    """固定子空间 minmax 粗筛+细筛（E73b 同款，全 head 同 idx）"""
    T, Hkv, _ = coarse_kf.shape
    nblk = (T + BS - 1) // BS
    kk = torch.nn.functional.pad(coarse_kf, (0, 0, 0, 0, 0, nblk * BS - T))
    kc = kk.reshape(nblk, BS, Hkv, -1)
    kmin, kmax = kc.amin(1), kc.amax(1)
    sc = (torch.einsum("hd,nhd->hn", q_coarse.clamp(min=0), kmax) +
          torch.einsum("hd,nhd->hn", q_coarse.clamp(max=0), kmin))
    ib = torch.topk(sc, min(N_PAGES, nblk), dim=-1).indices
    tok = (ib.unsqueeze(-1) * BS + torch.arange(BS).view(1, 1, BS)).reshape(Hkv, -1).clamp(max=T - 1)
    pool = torch.zeros(Hkv, T, dtype=torch.bool)
    pool.scatter_(1, tok, True)
    ts = torch.einsum("hd,shd->hs", q_fine, fine_kf).masked_fill(~pool, float("-inf"))
    it = torch.topk(ts, min(BUD_TOK, T), dim=-1).indices
    return it


def select_region_perhead(coarse_k, q_head_rows, idx_per_head):
    """per-head 子空间（query 自适应臂用）：逐 head 用自己的 idx 跑 minmax+细筛。
    coarse_k: [T, Hkv, 128]；q_head_rows: [Hkv, 128]；idx_per_head: [Hkv, r]"""
    T, Hkv, _ = coarse_k.shape
    outs = []
    nblk = (T + BS - 1) // BS
    kk = torch.nn.functional.pad(coarse_k, (0, 0, 0, 0, 0, nblk * BS - T))
    kc = kk.reshape(nblk, BS, Hkv, -1)
    for h in range(Hkv):
        idx = idx_per_head[h]
        kmin = kc[:, :, h, :].amin(1)[:, idx]   # [nblk, r]
        kmax = kc[:, :, h, :].amax(1)[:, idx]
        qc = q_head_rows[h][idx]
        sc = (qc.clamp(min=0) @ kmax.T + qc.clamp(max=0) @ kmin.T)  # [nblk]
        ib = torch.topk(sc, min(N_PAGES, nblk)).indices
        tok = (ib.unsqueeze(-1) * BS + torch.arange(BS)).reshape(-1).clamp(max=T - 1)
        pool = torch.zeros(T, dtype=torch.bool)
        pool[tok] = True
        ts = (coarse_k[:, h, :][:, idx] @ qc).masked_fill(~pool, float("-inf"))
        outs.append(torch.topk(ts, min(BUD_TOK, T)).indices)
    return torch.stack(outs)  # [Hkv, BUD]


def eval_layer(lf):
    """本实验 arm 全部在函数内构造（query/K 统计自适应），返回 {arm: [recall]}"""
    d = torch.load(lf, map_location="cpu", weights_only=False)
    k, q, qpos, S = d["k"].float(), d["q"].float(), d["qpos"], d["S"]
    Hkv, H, D = k.shape[1], q.shape[1], q.shape[-1]
    G = H // Hkv
    res = {}
    t = int(qpos[-1])
    mid_hi = t + 1 - SWA
    far_lo, far_hi = SINK, mid_hi - NEAR_BAND
    if far_hi - far_lo < 8192:
        del k, q, d
        return None
    kfar = k[far_lo:far_hi]  # [T, Hkv, 128]
    # K 方差静态臂的 idx（far 区 K 统计）
    kvar_idx = torch.topk(kfar.var(dim=(0, 1)), 32).indices.sort().values
    for ri in range(TAIL_N):
        q_t = q[-TAIL_N + ri]
        q_head = q_t.reshape(Hkv, G, D).sum(1)  # [Hkv, 128]
        s = torch.einsum("hd,shd->hs", q_head, k) * (D ** -0.5)
        s = s.masked_fill(torch.arange(S).view(1, -1) > int(qpos[-TAIL_N + ri]), float("-inf"))
        p = torch.softmax(s, dim=-1)
        far_h = p[:, far_lo:far_hi].sum(-1)
        tot_f = float(far_h.sum())
        if tot_f < 1e-6:
            continue
        # ---- arm 构造 ----
        tail32 = list(range(48, 64)) + list(range(112, 128))
        # SparQ 式：per-head |q| top-r（破坏配对结构的数据驱动选取）
        sparq_idx = torch.topk(q_head.abs(), 32, dim=-1).indices  # [Hkv, 32]
        # pair 感知自适应：按 |q_j|+|q_{j+64}| 排序取 top-16 完整对
        pair_mag = q_head[:, :64].abs() + q_head[:, 64:].abs()   # [Hkv, 64]
        pair_top = torch.topk(pair_mag, 16, dim=-1).indices      # [Hkv, 16]（频率 j）
        pair_adapt_idx = torch.cat([pair_top, pair_top + 64], dim=-1)  # [Hkv, 32]
        arms = {
            "tail32": ("fixed", tail32),
            "kvar32": ("fixed", kvar_idx.tolist()),
            "sparq_q32": ("perhead", sparq_idx),
            "pair_adapt16": ("perhead", pair_adapt_idx),
        }
        for seg, (mode, idx) in arms.items():
            if mode == "fixed":
                idx_t = torch.tensor(idx)
                it = select_region_fixed(kfar[..., idx_t], q_head[..., idx_t],
                                         kfar[..., idx_t], q_head[..., idx_t])
            else:
                it = select_region_perhead(kfar, q_head, idx)
            cap = sum(float(p[h, it[h] + far_lo].sum()) for h in range(Hkv))
            res.setdefault(seg, []).append(cap / tot_f)
    del k, q, d
    return res


def main():
    torch.set_num_threads(16)
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
        print(f"[{name}] tail32={rec.get('tail32')} kvar={rec.get('kvar32')} "
              f"sparq={rec.get('sparq_q32')} pairadapt={rec.get('pair_adapt16')}", flush=True)

    valid = [v for v in results.values() if v.get("tail32") is not None]
    mean = {k2: round(sum(v[k2] for v in valid) / len(valid), 4) for k2 in valid[0]}
    results["MEAN8"] = mean

    print("\n== 子空间选取方法对比（MEAN8）==")
    print(f"  tail32（旋转对频率先验）: {mean['tail32']:.4f}")
    print(f"  kvar32（K 方差静态）:     {mean['kvar32']:.4f}")
    print(f"  sparq_q32（|q| top-32）:  {mean['sparq_q32']:.4f}")
    print(f"  pair_adapt16（pair 自适应）: {mean['pair_adapt16']:.4f}")

    json.dump({"mean": mean, "per_sample": results}, open(OUT, "w"), indent=1)
    print("\nsaved ->", OUT)


if __name__ == "__main__":
    main()
