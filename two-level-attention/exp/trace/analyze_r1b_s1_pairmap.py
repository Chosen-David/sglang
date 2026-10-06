# R1b-S1：pair-map 布局等价性（审稿 S1「按频率对定义子空间，不能按固定地址定义」回应）
# 两个子任务：
#   S1a 布局等价性/选取规则对比：
#     (1) 复现 E85b 三重判决口径（频率单调 / 配对完整性 / 16 对饱和），预算口径
#         与 E85b 完全一致（N_PAGES=16, BUD_TOK=512, BS=64, far 区 minmax 两级）；
#     (2) 「频率排序前 16 对」的不同选取规则对比：频率先验 tail32 vs SparQ 式 |q|
#         幅值（E85d sparq_q32）vs pair 感知自适应（pair_adapt16）vs per-layer 静态
#         pair（E85e）vs K 方差（kvar32）vs 随机 16 完整对（负对照）；
#     (3) 坐标置换等价性（reviewer S1 最小验证）：对 Q/K 同步置换坐标与选维集合，
#         split-half→adjacent 重排与随机置换两臂，同语义子空间的选择质量应与 tail32
#         等价（浮点求和顺序差异 ≤1e-6 量级）——布局无关接口的正确性检查。
#   S1b pair-map 落盘：exp/trace/results/r1b_pair_map.json，逐对 (j, j+64) 的
#         维度索引与 inv_freq（Qwen3-8B rope_theta=1e6，128 维全 RoPE）。
# 真值口径 = 全维 softmax per-head far mass 加权捕获（E85b 同款）。
# GPU：CUDA_VISIBLE_DEVICES=1（不占 GPU0 的 poolavg 实验）。
import json
import os

import torch

TRACE = os.environ.get("R1B_TRACE", "/home/wangyuanshuo02/.archive/trace-dumps/qwen3-8b")
OUT_S1 = "/home/wangyuanshuo02/sglang/two-level-attention/exp/trace/results/r1b_s1_layout_equiv.json"
OUT_MAP = "/home/wangyuanshuo02/sglang/two-level-attention/exp/trace/results/r1b_pair_map.json"
SINK, SWA, NEAR_BAND = 128, 1024, 4096
BUD_TOK = 512          # E85b 口径
N_PAGES = 16           # E85b 口径
BS = 64
TAIL_N = 2
SAMPLES = ["lb_gov_report_0", "lb_hotpotqa_0", "lb_musique_0", "lb_narrativeqa_0",
           "lb_passage_retrieval_en_0", "lb_qasper_0", "needle32k", "natural32k"]
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
ROPE_THETA = 1000000.0   # Qwen3-8B config.json
HEAD_DIM = 128

# ---- E85b 历史参照（复现对照用，来自 e85b/e85d/e85e 结果 JSON 的 MEAN8） ----
REF = {
    "pair8": 0.4731, "pair16": 0.8113, "pair24": 0.8133, "pair32": 0.8022, "pair48": 0.8529,
    "pair16_midfreq": 0.3603, "pair16_hifreq": 0.004,
    "mismatch_lo_mid": 0.5672, "mismatch_mid_lo": 0.3419, "mismatch_shuffled": 0.6867,
    "spec_top32": 0.7585, "tail32": 0.8113, "rope_tail16": 0.5759, "nope_tail16": 0.387,
    "sparq_q32": 0.8482, "pair_adapt16": 0.8652, "static_pair16": 0.8493, "kvar32": 0.7493,
}


def build_pair_map():
    """S1b：Qwen3 rotate_half(split-half, 非交错) 布局的权威 pair-map。
    旋转对 j = (j, j+64)，j=0..63；inv_freq_j = theta^(-2j/128)，j 越大频率越低。
    tail32 = 对 48..63 = 16 个最低频完整对。"""
    inv = [ROPE_THETA ** (-2.0 * j / HEAD_DIM) for j in range(64)]
    pairs = [{"pair_id": j, "dims": [j, j + 64],
              "inv_freq": inv[j],
              "period_tokens": 6.283185307179586 / inv[j]}
             for j in range(64)]
    pmap = {
        "model": "Qwen3-8B",
        "head_dim": HEAD_DIM, "rope_theta": ROPE_THETA,
        "rotary_layout": "split_half (HF rotate_half, non-interleaved)",
        "pairing_rule": "pair j = (j, j+64), j = 0..63; inv_freq_j = theta^(-2j/128)",
        "frequency_ordering": "inv_freq 单调递减于 j：j=0 最高频，j=63 最低频",
        "note": "Qwen3 全 128 维参与 RoPE；[112:128] 不是 NoPE 维，而是旋转对 (j, j+64) 的第二元素",
        "pairs_by_pair_id": pairs,
        "pairs_by_low_to_high_freq": [p["pair_id"] for p in sorted(pairs, key=lambda x: -x["inv_freq"])],
        "subspaces": {
            "tail32": {"pairs": list(range(48, 64)),
                       "dims": list(range(48, 64)) + list(range(112, 128)),
                       "semantics": "16 个最低频完整旋转对（频率排序前 16 对）"},
            "midfreq16_pairs": {"pairs": list(range(32, 48)),
                                "dims": list(range(32, 48)) + list(range(96, 112)),
                                "semantics": "第 17-32 低频完整对（中频对照）"},
            "hifreq16_pairs": {"pairs": list(range(0, 16)),
                               "dims": list(range(0, 16)) + list(range(64, 80)),
                               "semantics": "最高频 16 完整对（高频对照）"},
        },
        "mismatch_arms": {
            "mismatch_lo_mid": {"dims": list(range(48, 64)) + list(range(96, 112)),
                                "semantics": "前半最低频 16 维 + 后半中频对元素（破坏配对完整性）"},
            "mismatch_mid_lo": {"dims": list(range(32, 48)) + list(range(112, 128)),
                                "semantics": "前半中频 16 维 + 后半最低频对元素（破坏配对完整性）"},
        },
    }
    return pmap


def select_region_fixed(coarse_kf, q_coarse, fine_kf, q_fine):
    """固定子空间 minmax 粗筛（top-N_PAGES 块）+ 细筛 topk（BUD_TOK）——E85b 同款"""
    T, Hkv, _ = coarse_kf.shape
    nblk = (T + BS - 1) // BS
    kk = torch.nn.functional.pad(coarse_kf, (0, 0, 0, 0, 0, nblk * BS - T))
    kc = kk.reshape(nblk, BS, Hkv, -1)
    kmin, kmax = kc.amin(1), kc.amax(1)
    sc = (torch.einsum("hd,nhd->hn", q_coarse.clamp(min=0), kmax) +
          torch.einsum("hd,nhd->hn", q_coarse.clamp(max=0), kmin))
    ib = torch.topk(sc, min(N_PAGES, nblk), dim=-1).indices
    tok = (ib.unsqueeze(-1) * BS + torch.arange(BS, device=coarse_kf.device).view(1, 1, BS)).reshape(Hkv, -1).clamp(max=T - 1)
    pool = torch.zeros(Hkv, T, dtype=torch.bool, device=coarse_kf.device)
    pool.scatter_(1, tok, True)
    ts = torch.einsum("hd,shd->hs", q_fine, fine_kf).masked_fill(~pool, float("-inf"))
    it = torch.topk(ts, min(BUD_TOK, T), dim=-1).indices
    return it, pool


def select_region_perhead(coarse_k, q_head_rows, idx_per_head):
    """per-head 子空间（query 自适应臂）：逐 head 用自己的 idx 跑 minmax+细筛（E85d 同款）"""
    T, Hkv, _ = coarse_k.shape
    outs = []
    nblk = (T + BS - 1) // BS
    kk = torch.nn.functional.pad(coarse_k, (0, 0, 0, 0, 0, nblk * BS - T))
    kc = kk.reshape(nblk, BS, Hkv, -1)
    for h in range(Hkv):
        idx = idx_per_head[h]
        kmin = kc[:, :, h, :].amin(1)[:, idx]
        kmax = kc[:, :, h, :].amax(1)[:, idx]
        qc = q_head_rows[h][idx]
        sc = (qc.clamp(min=0) @ kmax.T + qc.clamp(max=0) @ kmin.T)
        ib = torch.topk(sc, min(N_PAGES, nblk)).indices
        tok = (ib.unsqueeze(-1) * BS + torch.arange(BS, device=coarse_k.device)).reshape(-1).clamp(max=T - 1)
        pool = torch.zeros(T, dtype=torch.bool, device=coarse_k.device)
        pool[tok] = True
        ts = (coarse_k[:, h, :][:, idx] @ qc).masked_fill(~pool, float("-inf"))
        outs.append(torch.topk(ts, min(BUD_TOK, T)).indices)
    return torch.stack(outs)


def eval_layer(lf, arms_fixed, perm_random, perm_adjacent):
    """返回 {arm: [recall,...]}（逐尾 query）+ 置换等价性诊断"""
    d = torch.load(lf, map_location="cpu", weights_only=False)
    k, q, qpos, S = d["k"].to(DEV).float(), d["q"].to(DEV).float(), d["qpos"], d["S"]
    Hkv, H, D = k.shape[1], q.shape[1], q.shape[-1]
    G = H // Hkv
    res = {}
    perm_diag = []
    t = int(qpos[-1])
    mid_hi = t + 1 - SWA
    far_lo, far_hi = SINK, mid_hi - NEAR_BAND
    if far_hi - far_lo < 8192:
        del k, q, d
        return None, None
    kfar = k[far_lo:far_hi]
    # per-layer 静态臂的 idx（prefill 末段 q 统计，E85e 同款：尾 272 q 全体平均 |q|）
    qmean = q.reshape(-1, D).abs().mean(0)
    pair_mag = qmean[:64] + qmean[64:]
    top_pairs = torch.topk(pair_mag, 16).indices
    static_pair = torch.cat([top_pairs, top_pairs + 64]).sort().values.tolist()
    # K 方差静态臂（E85d 同款：far 区 K 统计 top 方差 32 维）
    kvar_idx = torch.topk(kfar.var(dim=(0, 1)), 32).indices.sort().values.tolist()
    for ri in range(TAIL_N):
        q_t = q[-TAIL_N + ri]
        q_head = q_t.reshape(Hkv, G, D).sum(1)
        s = torch.einsum("hd,shd->hs", q_head, k) * (D ** -0.5)
        s = s.masked_fill(torch.arange(S, device=DEV).view(1, -1) > int(qpos[-TAIL_N + ri]), float("-inf"))
        p = torch.softmax(s, dim=-1)
        far_h = p[:, far_lo:far_hi].sum(-1)
        tot_f = float(far_h.sum())
        if tot_f < 1e-6:
            continue
        # ---- 固定子空间臂（含 per-layer 统计臂）----
        fixed = dict(arms_fixed)
        fixed["static_pair16"] = static_pair
        fixed["kvar32"] = kvar_idx
        sel_sets = {}
        for seg, idx in fixed.items():
            idx_t = torch.tensor(idx, device=DEV)
            it, _ = select_region_fixed(kfar[..., idx_t], q_head[..., idx_t],
                                        kfar[..., idx_t], q_head[..., idx_t])
            cap = sum(float(p[h, it[h] + far_lo].sum()) for h in range(Hkv))
            res.setdefault(seg, []).append(cap / tot_f)
            if seg == "tail32":
                sel_sets["tail32"] = it  # 相对 far 区坐标（与 it_p 同口径）
        # ---- per-head 自适应臂（E85d 同款）----
        sparq_idx = torch.topk(q_head.abs(), 32, dim=-1).indices
        pm = q_head[:, :64].abs() + q_head[:, 64:].abs()
        pair_top = torch.topk(pm, 16, dim=-1).indices
        pair_adapt_idx = torch.cat([pair_top, pair_top + 64], dim=-1)
        for seg, idx in {"sparq_q32": sparq_idx, "pair_adapt16": pair_adapt_idx}.items():
            it = select_region_perhead(kfar, q_head, idx)
            cap = sum(float(p[h, it[h] + far_lo].sum()) for h in range(Hkv))
            res.setdefault(seg, []).append(cap / tot_f)
        # ---- 坐标置换等价性（reviewer S1 最小验证）----
        # 对 Q/K 同步置换 128 维坐标，选维集合同步映射：同语义子空间应给出等价选择。
        # split-half→adjacent 重排：new[2j]=old[j], new[2j+1]=old[j+64]（布局迁移）
        # 随机置换：seed 固定，任意正交坐标重排（此处为置换=正交矩阵特例）
        tail32_idx = torch.tensor(arms_fixed["tail32"], device=DEV)
        for pname, perm in [("perm_adjacent", perm_adjacent), ("perm_random", perm_random)]:
            kfar_p = kfar[..., perm]
            q_head_p = q_head[..., perm]
            # tail32 语义集合映射到置换后坐标：new_dim i 对应 old_dim perm[i]
            # 选 old tail32 维 → 新坐标位置 = {i : perm[i] ∈ tail32}
            inv_perm = torch.empty_like(perm)
            inv_perm[perm] = torch.arange(perm.numel(), device=DEV)
            new_idx = inv_perm[tail32_idx]
            it_p, _ = select_region_fixed(kfar_p[..., new_idx], q_head_p[..., new_idx],
                                          kfar_p[..., new_idx], q_head_p[..., new_idx])
            cap_p = sum(float(p[h, it_p[h] + far_lo].sum()) for h in range(Hkv))
            rec_p = cap_p / tot_f
            # 选择集合重合率（token 级）
            ov = sum(int((set(it_p[h].tolist()) & set(sel_sets["tail32"][h].tolist())).__len__())
                     for h in range(Hkv)) / (Hkv * BUD_TOK)
            perm_diag.append({"arm": pname, "recall": rec_p,
                              "d_recall_vs_tail32": rec_p - res["tail32"][-1],
                              "sel_overlap": round(ov, 6)})
    del k, q, d
    torch.cuda.empty_cache() if DEV != "cpu" else None
    return res, perm_diag


def build_arms():
    arms = {}
    # ① 频率对数扫描（E85b ①）
    for n in [8, 16, 24, 32, 48]:
        arms[f"pair{n}"] = list(range(64 - n, 64)) + list(range(128 - n, 128))
    # ② 频率依赖（E85b ②）
    arms["pair16_midfreq"] = list(range(32, 48)) + list(range(96, 112))
    arms["pair16_hifreq"] = list(range(0, 16)) + list(range(64, 80))
    # ③ 错位配对（E85b ③，判决性）
    arms["mismatch_lo_mid"] = list(range(48, 64)) + list(range(96, 112))
    arms["mismatch_mid_lo"] = list(range(32, 48)) + list(range(112, 128))
    arms["mismatch_shuffled"] = list(range(48, 64)) + sorted(
        [96, 97, 98, 99, 100, 101, 102, 103, 120, 121, 122, 123, 124, 125, 126, 127])
    # ④ 数据驱动（E85b ④）
    arms["spec_top32"] = [42, 44, 45, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 63,
                          65, 102, 106, 108, 109, 111, 112, 113, 115, 116, 117, 118, 123, 124, 126, 127]
    # 参照臂（E85b）
    arms["tail32"] = list(range(48, 64)) + list(range(112, 128))
    arms["rope_tail16"] = list(range(48, 64))
    arms["nope_tail16"] = list(range(112, 128))
    # 布局选取规则负对照：随机 16 完整对（seed 固定，同「完整对」布局结构、错误选取规则）
    g = torch.Generator().manual_seed(20261007)
    rp = torch.randperm(64, generator=g)[:16].sort().values
    arms["random16pairs"] = (rp.tolist() + (rp + 64).tolist())
    return arms


def main():
    torch.set_grad_enabled(False)
    pmap = build_pair_map()
    json.dump(pmap, open(OUT_MAP, "w"), indent=1)
    print(f"[S1b] pair-map saved -> {OUT_MAP}")

    arms = build_arms()
    g = torch.Generator().manual_seed(20261007)
    perm_random = torch.randperm(128, generator=g).to(DEV)
    # split-half → adjacent 布局重排：new[2j]=old[j], new[2j+1]=old[j+64]
    # split-half → adjacent 布局重排：新坐标 i 取旧坐标 perm[i]；
    # adjacent 布局下对 j 占 new (2j, 2j+1)，split-half 下占 old (j, j+64)
    # → perm[2j]=j, perm[2j+1]=j+64
    perm_adjacent = torch.tensor(sum([[j, j + 64] for j in range(64)], [])).to(DEV)

    results = {}
    perm_all = []
    for name in SAMPLES:
        meta = f"{TRACE}/{name}/meta.json"
        if not os.path.isfile(meta):
            continue
        n_layers = json.load(open(meta))["n_layers"]
        agg = {}
        for li in range(0, n_layers, max(1, n_layers // 8)):
            r, pd_ = eval_layer(f"{TRACE}/{name}/layer{li:02d}.pt", arms, perm_random, perm_adjacent)
            if not r:
                continue
            if pd_:
                perm_all.extend(pd_)
            for k2, v in r.items():
                agg.setdefault(k2, []).extend(v)
        rec = {k2: round(sum(v) / len(v), 4) for k2, v in agg.items()}
        results[name] = rec
        print(f"[{name}] tail32={rec.get('tail32')} mid={rec.get('pair16_midfreq')} "
              f"mismatch={rec.get('mismatch_lo_mid')} pair8={rec.get('pair8')} "
              f"sparq={rec.get('sparq_q32')} pairadapt={rec.get('pair_adapt16')}", flush=True)

    valid = [v for v in results.values() if v.get("tail32") is not None]
    mean = {k2: round(sum(v[k2] for v in valid) / len(valid), 4) for k2 in valid[0]}
    results["MEAN8"] = mean

    # 与 E85b/d/e 历史数字对照（复现偏差）
    repro = {k: {"r1b": mean.get(k), "ref": REF.get(k),
                 "delta": (round(mean[k] - REF[k], 4) if k in REF and k in mean else None)}
             for k in mean if k in REF}

    print("\n== ① 频率单调（16 对等宽）==")
    print(f"  低频 pair16(=tail32): {mean['pair16']:.4f}  (E85b 0.8113)")
    print(f"  中频 pair16_midfreq:  {mean['pair16_midfreq']:.4f}  (E85b 0.3603)")
    print(f"  高频 pair16_hifreq:   {mean['pair16_hifreq']:.4f}  (E85b 0.0040)")
    print("== ② 配对完整性（错位配对崩塌）==")
    print(f"  tail32 完整对:        {mean['tail32']:.4f}")
    print(f"  mismatch_lo_mid:      {mean['mismatch_lo_mid']:.4f}  (E85b 0.5672)")
    print(f"  mismatch_mid_lo:      {mean['mismatch_mid_lo']:.4f}  (E85b 0.3419)")
    print("== ③ 频率对数扫描（16 对饱和）==")
    for n in [8, 16, 24, 32, 48]:
        print(f"  {n:2d} 个最低频完整对: {mean[f'pair{n}']:.4f}")
    print("== ④ 选取规则对比（同为 32 标量维）==")
    print(f"  tail32 频率先验:      {mean['tail32']:.4f}")
    print(f"  random16pairs:        {mean['random16pairs']:.4f}")
    print(f"  kvar32 K 方差:        {mean['kvar32']:.4f}  (E85d 0.7493)")
    print(f"  spec_top32 谱 top:    {mean['spec_top32']:.4f}  (E85b 0.7585)")
    print(f"  sparq_q32 |q| 幅值:   {mean['sparq_q32']:.4f}  (E85d 0.8482)")
    print(f"  static_pair16 静态:   {mean['static_pair16']:.4f}  (E85e 0.8493)")
    print(f"  pair_adapt16 自适应:  {mean['pair_adapt16']:.4f}  (E85d 0.8652)")
    if perm_all:
        import statistics
        for a in ["perm_adjacent", "perm_random"]:
            ds = [x["d_recall_vs_tail32"] for x in perm_all if x["arm"] == a]
            ovs = [x["sel_overlap"] for x in perm_all if x["arm"] == a]
            if ds:
                print(f"== ⑤ 置换等价性 {a}: max|Δrecall|={max(abs(x) for x in ds):.2e}, "
                      f"选择集合重合率均值={statistics.mean(ovs):.4f} ==")

    out = {
        "config": {"trace": TRACE, "budget_tok": BUD_TOK, "n_pages": N_PAGES, "bs": BS,
                   "sink": SINK, "swa": SWA, "near_band": NEAR_BAND, "tail_n": TAIL_N,
                   "samples": SAMPLES, "layers_per_sample": 8, "device": DEV,
                   "口径": "far 区 R_far 全链 minmax 粗筛+细筛同特征，真值=全维 softmax per-head far mass 加权捕获（E85b 同款）"},
        "mean": mean, "per_sample": results,
        "reproduction_vs_e85": repro,
        "perm_equivalence": perm_all,
    }
    json.dump(out, open(OUT_S1, "w"), indent=1)
    print(f"\n[S1a] saved -> {OUT_S1}")


if __name__ == "__main__":
    main()
