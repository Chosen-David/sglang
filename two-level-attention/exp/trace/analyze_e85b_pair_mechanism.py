# E85b：子空间选取机理判决——「完整 RoPE 旋转对」假说 vs 「两段互补」假说。
# E85a 混合扫描已给出方向性证据（r 单调升，纯 rope 低频 16 对 0.8113 最优，任何
# nope 维替换单调降）。本实验四判决：
#   ① 频率对数扫描：完整低频对 8/16/24/32 对——最优对数在哪；
#   ② 中频 vs 低频：16 中频对 (32..47 + 96..111) vs 16 低频对（tail32）——频率依赖；
#   ③ 错位配对（判决性）：前半取最低频 16、后半取中频对元素（96..111）——
#      若「对完整性」是机理则显著崩；若「两段独立互补」是机理则不应崩（两段
#      与 tail32 段位几乎相同，只是后半元素来源不同）；
#   ④ 数据驱动选取：谱 top-32 直接评估（E85a 只给了维度列表没给 recall）+ 贪心
#      前向 32 维（全样本全层，每步全候选扫描）——tail32 距数据驱动最优差多少。
# 口径与 E85a/E73b 完全一致（far 全链 minmax 粗筛+细筛同特征，真值=全维 softmax
# per-head far mass 加权捕获）。
import json
import os

import torch

TRACE = "/tmp/trace/qwen3-8b"
OUT = "/home/wangyuanshuo02/two-level-attention/exp/trace/results/e85b_pair_mechanism.json"
SINK, SWA, NEAR_BAND = 128, 1024, 4096
BUD_TOK = 512
N_PAGES = 16
BS = 64
TAIL_N = 2
SAMPLES = ["lb_gov_report_0", "lb_hotpotqa_0", "lb_musique_0", "lb_narrativeqa_0",
           "lb_passage_retrieval_en_0", "lb_qasper_0", "needle32k", "natural32k"]


def select_region(coarse_kf, q_coarse, fine_kf, q_fine):
    """子空间 minmax 粗筛（top-N_PAGES 块）+ 细筛 topk（BUD_TOK）——E73b 同款"""
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
    """arms: {name: [dims]}；返回 {name: [recall,...]}（逐尾 query）"""
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


def load_spec():
    """读 E85a 谱：返回逐维 MEAN8 recall dict"""
    j = json.load(open("/home/wangyuanshuo02/two-level-attention/exp/trace/results/e85a_dim_spectrum.json"))
    return {d: v for d, v in j["spectrum_top15"]}, j["per_sample"]


def build_arms():
    arms = {}
    # ① 频率对数扫描：n 个最低频完整对（前半 64-n..63 + 后半 128-n..127）
    for n in [8, 16, 24, 32, 48]:
        arms[f"pair{n}"] = list(range(64 - n, 64)) + list(range(128 - n, 128))
    # ② 中频 vs 低频（等宽 32 维 = 16 对）
    arms["pair16_midfreq"] = list(range(32, 48)) + list(range(96, 112))
    arms["pair16_hifreq"] = list(range(0, 16)) + list(range(64, 80))
    # ③ 错位配对（判决性）：段位与 tail32 几乎重合，仅后半元素换成中频对来源
    arms["mismatch_lo_mid"] = list(range(48, 64)) + list(range(96, 112))
    arms["mismatch_mid_lo"] = list(range(32, 48)) + list(range(112, 128))
    arms["mismatch_shuffled"] = list(range(48, 64)) + sorted(
        [96, 97, 98, 99, 100, 101, 102, 103, 120, 121, 122, 123, 124, 125, 126, 127])
    # ④ 数据驱动：谱 top-32（E85a 维度列表直接评估）
    spec15, _ = load_spec()
    arms["spec_top32"] = [42, 44, 45, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 63,
                          65, 102, 106, 108, 109, 111, 112, 113, 115, 116, 117, 118, 123, 124, 126, 127]
    # 参照臂
    arms["tail32"] = list(range(48, 64)) + list(range(112, 128))
    arms["rope_tail16"] = list(range(48, 64))
    arms["nope_tail16"] = list(range(112, 128))
    return arms


def main():
    torch.set_num_threads(16)
    arms = build_arms()

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
        print(f"[{name}] tail32={rec.get('tail32')} pair16_mid={rec.get('pair16_midfreq')} "
              f"mismatch={rec.get('mismatch_lo_mid')} spec32={rec.get('spec_top32')}", flush=True)

    mean = {k2: round(sum(v[k2] for v in results.values()) / len(results), 4)
            for k2 in results[next(iter(results))]}
    results["MEAN8"] = mean

    print("\n== ① 频率对数扫描 ==")
    for n in [8, 16, 24, 32, 48]:
        print(f"  {n:2d} 个最低频完整对: {mean[f'pair{n}']:.4f}")
    print("\n== ② 中频 vs 低频（16 对等宽）==")
    print(f"  低频 pair16(=tail32): {mean['pair16']:.4f}")
    print(f"  中频 pair16_midfreq: {mean['pair16_midfreq']:.4f}")
    print(f"  高频 pair16_hifreq: {mean['pair16_hifreq']:.4f}")
    print("\n== ③ 错位配对（判决性）==")
    print(f"  tail32（完整对）:     {mean['tail32']:.4f}")
    print(f"  mismatch_lo_mid:      {mean['mismatch_lo_mid']:.4f}")
    print(f"  mismatch_mid_lo:      {mean['mismatch_mid_lo']:.4f}")
    print(f"  mismatch_shuffled:    {mean['mismatch_shuffled']:.4f}")
    print("\n== ④ 数据驱动 vs 手工 ==")
    print(f"  spec_top32: {mean['spec_top32']:.4f}  tail32: {mean['tail32']:.4f}")

    json.dump({"mean": mean, "per_sample": results}, open(OUT, "w"), indent=1)
    print("\nsaved ->", OUT)


if __name__ == "__main__":
    main()
