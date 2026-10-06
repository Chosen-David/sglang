# E85c：完整旋转对机理的跨模型验证——Qwen3-32B（64 层）。
# E85a/E85b 在 8B 上四判决获证（频率单调 / 配对完整性 / 对数饱和 / 数据驱动
# 败给先验）。若「16 个最低频完整旋转对」是 RoPE 数学先验而非 8B 巧合，
# 则 32B 上同样成立（32B head_dim 同为 128、同 rotate_half）。
# 臂集（浓缩版，只留判决所需）：tail32（16 最低频对）/ pair8 / pair24 /
# 中频 pair16 / 错位配对 / rope 半边 16 / 后半 16（旧 nope 叙事）/
# 全 128 维参照。7 任务 × 2 样本 × 8 层采样。
# 口径与 E85a/E85b 完全一致。
import json
import os

import torch

TRACE = "/tmp/trace/qwen3-32b"
OUT = "/home/wangyuanshuo02/two-level-attention/exp/trace/results/e85c_pair_32b.json"
SINK, SWA, NEAR_BAND = 128, 1024, 4096
BUD_TOK = 512
N_PAGES = 16
BS = 64
TAIL_N = 2
SAMPLES = [f"lb_{t}_{i}" for t in ["gov_report", "hotpotqa", "multifieldqa_en", "musique",
                                    "narrativeqa", "passage_retrieval_en", "qasper"]
           for i in [0, 1]]


def select_region(coarse_kf, q_coarse, fine_kf, q_fine):
    """子空间 minmax 粗筛 + 细筛 topk——E73b 同款"""
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


def eval_layer(lf, arms):
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
        for seg, idx in arms.items():
            idx_t = torch.tensor(idx)
            kf = k[far_lo:far_hi][..., idx_t]
            qf = q_head[..., idx_t]
            it = select_region(kf, qf, kf, qf)
            cap = sum(float(p[h, it[h] + far_lo].sum()) for h in range(Hkv))
            res.setdefault(seg, []).append(cap / tot_f)
    del k, q, d
    return res


def main():
    torch.set_num_threads(16)
    arms = {
        # 16 个最低频完整旋转对（tail32）
        "tail32": list(range(48, 64)) + list(range(112, 128)),
        "pair8": list(range(56, 64)) + list(range(120, 128)),
        "pair24": list(range(40, 64)) + list(range(104, 128)),
        # 中频 16 对（频率对照）
        "pair16_midfreq": list(range(32, 48)) + list(range(96, 112)),
        # 错位配对（配对完整性判决）
        "mismatch_lo_mid": list(range(48, 64)) + list(range(96, 112)),
        # 单独前半 16 / 后半 16（旧「两段互补」叙事的 32B 复验）
        "front16": list(range(48, 64)),
        "back16": list(range(112, 128)),
    }

    results = {}
    for name in SAMPLES:
        meta = f"{TRACE}/{name}/meta.json"
        if not os.path.isfile(meta):
            continue
        n_layers = json.load(open(meta))["n_layers"]
        agg = {}
        for li in range(0, n_layers, max(1, n_layers // 8)):
            r = eval_layer(f"{TRACE}/{name}/layer{li:02d}.pt", arms)
            if not r:
                continue
            for k2, v in r.items():
                agg.setdefault(k2, []).extend(v)
        rec = {k2: round(sum(v) / len(v), 4) for k2, v in agg.items()}
        results[name] = rec
        print(f"[{name}] tail32={rec.get('tail32')} mid={rec.get('pair16_midfreq')} "
              f"mismatch={rec.get('mismatch_lo_mid')} f16={rec.get('front16')} b16={rec.get('back16')}",
              flush=True)

    valid = [v for v in results.values() if v.get("tail32") is not None]
    mean = {k2: round(sum(v[k2] for v in valid) / len(valid), 4)
            for k2 in valid[0]}
    results["MEAN"] = mean
    print(f"\n（multifieldqa_en 两样本 far 区为空已跳过，有效 {len(valid)}/14 样本）")

    print("\n== Qwen3-32B 完整旋转对机理复验（7 任务×2 样本）==")
    print(f"  pair8:  {mean['pair8']:.4f}")
    print(f"  tail32(16对): {mean['tail32']:.4f}")
    print(f"  pair24: {mean['pair24']:.4f}")
    print(f"  中频16对:     {mean['pair16_midfreq']:.4f}")
    print(f"  错位配对:     {mean['mismatch_lo_mid']:.4f}")
    print(f"  单独前半16:   {mean['front16']:.4f}")
    print(f"  单独后半16:   {mean['back16']:.4f}")

    json.dump({"mean": mean, "per_sample": results}, open(OUT, "w"), indent=1)
    print("\nsaved ->", OUT)


if __name__ == "__main__":
    main()
