# E85e：pair 自适应选取的静态化判决——per-layer 固定 pair（prefill 末段 q
# 统计）能否保住 query 自适应增益（E85d pair_adapt16 0.865 vs tail32 0.811）。
# 选取信号 = 该层尾 256 q 的平均 |q|（模拟 prefill 末段统计，与 D' gate 同构），
# 按 pair 幅值 |q̄_j|+|q̄_{j+64}| 取 top-16 完整对——per-layer 固定，query 到达前
# 可定，与增量块索引兼容（每层一套 32 维索引）。
# 判决：①保住（≈0.86）→ 可索引化的真增益，论文升级；②保不住（≈tail32 或
# 更差）→ q 统计对 decode 漂移脆弱（E66/E71-B 同构），频率先验坐实可行域最优。
# 口径与 E85a-d 完全一致。
import json
import os

import torch

TRACE = "/tmp/trace/qwen3-8b"
OUT = "/home/wangyuanshuo02/two-level-attention/exp/trace/results/e85e_static_pair.json"
SINK, SWA, NEAR_BAND = 128, 1024, 4096
BUD_TOK = 512
N_PAGES = 16
BS = 64
TAIL_N = 2
SAMPLES = ["lb_gov_report_0", "lb_hotpotqa_0", "lb_musique_0", "lb_narrativeqa_0",
           "lb_passage_retrieval_en_0", "lb_qasper_0", "needle32k", "natural32k"]


def select_region_fixed(coarse_kf, q_coarse, fine_kf, q_fine):
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


def eval_layer(lf):
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
    kfar = k[far_lo:far_hi]
    # per-layer 静态 pair：尾 256 q 全体（head 展平）平均 |q| 的 pair 幅值 top-16
    qmean = q.reshape(-1, D).abs().mean(0)          # [128]
    pair_mag = qmean[:64] + qmean[64:]              # [64]
    top_pairs = torch.topk(pair_mag, 16).indices    # 频率 j
    static_pair = torch.cat([top_pairs, top_pairs + 64]).sort().values.tolist()
    tail32 = list(range(48, 64)) + list(range(112, 128))
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
        for seg, idx in {"tail32": tail32, "static_pair16": static_pair}.items():
            idx_t = torch.tensor(idx)
            it = select_region_fixed(kfar[..., idx_t], q_head[..., idx_t],
                                     kfar[..., idx_t], q_head[..., idx_t])
            cap = sum(float(p[h, it[h] + far_lo].sum()) for h in range(Hkv))
            res.setdefault(seg, []).append(cap / tot_f)
        # 与选 pair 时的频率集合对照（诊断：静态化选出的对在哪）
        res.setdefault("static_pairs_used", []).append(sorted(top_pairs.tolist()))
    del k, q, d
    return res


def main():
    torch.set_num_threads(16)
    results = {}
    pairs_log = {}
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
            pairs_log[f"{name}_L{li}"] = r.pop("static_pairs_used")[0] if "static_pairs_used" in r else None
            for k2, v in r.items():
                agg.setdefault(k2, []).extend(v)
        rec = {k2: round(sum(v) / len(v), 4) for k2, v in agg.items()}
        results[name] = rec
        print(f"[{name}] tail32={rec.get('tail32')} static_pair={rec.get('static_pair16')}", flush=True)

    valid = [v for v in results.values() if v.get("tail32") is not None]
    mean = {k2: round(sum(v[k2] for v in valid) / len(valid), 4) for k2 in valid[0]}
    results["MEAN8"] = mean

    print("\n== per-layer 静态 pair 判决（MEAN8）==")
    print(f"  tail32（频率先验）:      {mean['tail32']:.4f}")
    print(f"  static_pair16（q 统计）: {mean['static_pair16']:.4f}")
    print(f"  （对照 query 自适应 pair_adapt16 = 0.8652）")

    json.dump({"mean": mean, "per_sample": results, "static_pairs_log": pairs_log},
              open(OUT, "w"), indent=1)
    print("\nsaved ->", OUT)


if __name__ == "__main__":
    main()
