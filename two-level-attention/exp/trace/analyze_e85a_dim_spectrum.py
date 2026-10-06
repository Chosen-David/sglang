# E85a：子空间选取方法学探索——第一部分（用户 2026-09-30 指令：不是恰好选了
# nope/tail 然后 work，而是「为什么 work / 能不能选别的维度 / 选取方法是什么」）。
# 三个判决：
#   ① 逐维重要性谱：全 128 维单维 far 全链 recall——tail32 是否就是谱的 top-32，
#      谱形态（连续衰减 vs 两簇）是「位置先验」解释的直接证据；
#   ② rope/nope 混合比例扫描：rope 低频尾 r 维 + nope 尾 32-r 维，扫 r∈{0..32}——
#      最优混合比是否 16/16（E73b 两段互补的定量细化）；
#   ③ nope 段内结构：nope 尾16 vs nope 谱 top16 vs nope 随机16——段内是否有结构。
# 口径与 E73b 完全一致（far 全链 minmax 粗筛+细筛同特征，真值=全维 softmax
# per-head far mass 加权捕获），保证与 0.576/0.387/0.811 可直接对比。
import json
import os

import torch

TRACE = "/tmp/trace/qwen3-8b"
OUT = "/home/wangyuanshuo02/two-level-attention/exp/trace/results/e85a_dim_spectrum.json"
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


def main():
    torch.set_num_threads(16)

    # ---- arm 构造 ----
    arms = {}
    # ① 逐维谱：128 个单维臂
    for dim in range(128):
        arms[f"d{dim}"] = [dim]
    # ② 混合比例扫描（rope 低频尾按频率升序取 r 维 + nope 尾取 32-r 维）
    #    rope 维 0..63（rotate_half 前后两半各 32，低频=高索引维 48..63 及 112..127）
    #    Qwen3 rotate_half: 前半 [0..63] 低频尾=48..63；后半 [64..127] 低频尾=112..127。
    #    「rope 低频 r 维」= 两半各取 r/2（r 为偶数）；nope 尾 = 96..127 取 32-r 的尾部。
    for r in [0, 4, 8, 12, 16, 20, 24, 28, 32]:
        if r == 0:
            mix = list(range(96, 128))          # nope 全 32
        elif r == 32:
            mix = list(range(48, 64)) + list(range(112, 128))  # rope 低频全 32 = tail32
        else:
            half = r // 2
            mix = (list(range(64 - half, 64)) + list(range(128 - half, 128))
                   + list(range(96, 96 + (32 - r))))
        arms[f"mix_r{r}"] = sorted(mix)
    # ③ nope 段内结构（nope=64..127；谱 top16 在跑完①后补第二遍，先放尾16+随机16）
    arms["nope_tail16"] = list(range(112, 128))
    rng = torch.Generator().manual_seed(42)
    arms["nope_rand16"] = sorted(torch.randperm(64, generator=rng)[:16].add(64).tolist())
    # 参照臂（与 E73b 数字直接对比校验）
    arms["rope_tail16"] = list(range(48, 64))
    arms["tail32"] = list(range(48, 64)) + list(range(112, 128))
    arms["random32"] = [17, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83,
                        89, 97, 101, 103, 107, 109, 113, 127, 3, 7, 11, 13, 19, 2, 5, 1]

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
        # 冒烟参照
        print(f"[{name}] tail32={rec.get('tail32')} rope16={rec.get('rope_tail16')} "
              f"nope16={rec.get('nope_tail16')} mix16/16={rec.get('mix_r16')}", flush=True)

    mean = {k2: round(sum(v[k2] for v in results.values()) / len(results), 4)
            for k2 in results[next(iter(results))]}
    results["MEAN8"] = mean

    # ---- 谱分析与排序 ----
    spec = sorted(((int(k[1:]), v) for k, v in mean.items() if k.startswith("d") and k[1:].isdigit()),
                  key=lambda x: -x[1])
    top32_by_spec = sorted(d for d, _ in spec[:32])
    tail32 = set(range(48, 64)) | set(range(112, 128))
    overlap = len(set(top32_by_spec) & tail32)
    print("\n== 逐维重要性谱（MEAN8，top-15）==")
    for d, v in spec[:15]:
        seg = "rope低频" if d in range(48, 64) else ("rope高频" if d < 64 else ("nope尾" if d >= 112 else "nope中"))
        print(f"  dim{d:3d} ({seg})  recall={v:.4f}")
    print(f"\n谱 top-32 与 tail32 的交集: {overlap}/32")
    print(f"谱 top-32: {top32_by_spec}")
    print("\n== 混合比例扫描 ==")
    for r in [0, 4, 8, 12, 16, 20, 24, 28, 32]:
        print(f"  rope r={r:2d}: recall={mean[f'mix_r{r}']:.4f}")
    print("\n== nope 段内结构 ==")
    print(f"  tail16={mean['nope_tail16']:.4f} rand16={mean['nope_rand16']:.4f}")

    summary = {
        "top32_by_spectrum": top32_by_spec,
        "spectrum_tail32_overlap": overlap,
        "spectrum_top15": [(d, v) for d, v in spec[:15]],
        "mix_curve": {f"r{r}": mean[f"mix_r{r}"] for r in [0, 4, 8, 12, 16, 20, 24, 28, 32]},
        "nope_structure": {"tail16": mean["nope_tail16"], "rand16": mean["nope_rand16"]},
        "reference": {k: mean[k] for k in ["rope_tail16", "tail32", "random32"]},
        "per_sample": results,
    }
    json.dump(summary, open(OUT, "w"), indent=1)
    print("\nsaved ->", OUT)


if __name__ == "__main__":
    main()
