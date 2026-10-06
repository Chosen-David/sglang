# E79a：per-layer 最佳 α/β oracle 上限（2026-09-30 用户指令：
#   「每一层单独算最佳 alpha beta，prefill 进去时快速得到每层大致最佳配比
#    以及哪些 far 层能不能跳过……预估器要证明和真实最佳精度差不多，
#    且几乎不影响性能」）。
# 第一道闸门：先量化「每层独立选最优 (α,β)」相对「全层固定最优」的 trace 上限。
#   E75 已证 per-task oracle 仅 +0.08（噪声级）；本实验把粒度细化到层。
# 协议：E64a 框架（mavg method = far minmax / near avg），紧预算 e2e 口径
#   B_TOK=1024、γ=0.25（E72 教训：宽预算 2048 与 e2e 结论反转，须贴部署口径）。
# 网格：α ∈ {0.125,0.25,0.375,0.5}（near 子区长度份额）
#       β ∈ {0.125,0.25,0.375,0.5,0.75}（near 页池份额）。
# 每层单独记录全部臂 cov + 该层 far mass（mid 区 mass 占比 = 预估器候选信号，
#   prefill 时可免费获得）。输出三判决量：
#   fixed_best（全层统一最优配置）vs per_layer_oracle（每层独立 argmax）
#   → 上限 Δ；以及每层最优 (α,β) 与 far mass 的关系（预估器可行性依据）。
import json
import os

import torch
import torch.nn.functional as F

TRACE = "/tmp/trace/qwen3-8b"
OUT = "/home/wangyuanshuo02/two-level-attention/exp/trace/results/e79a_perlayer_oracle.json"
D2I = list(range(48, 64)) + list(range(112, 128))
BS = 64
SINK = 128
SWA = 1024
TAIL_N = 4
BP = 64
B_TOK = 1024          # 紧预算（e2e 部署口径，E72 教训）
GAMMA = 0.25
ALPHAS = [0.125, 0.25, 0.375, 0.5]
BETAS = [0.125, 0.25, 0.375, 0.5, 0.75]
SAMPLES = ["lb_gov_report_0", "lb_hotpotqa_0", "lb_musique_0", "lb_narrativeqa_0",
           "lb_passage_retrieval_en_0", "lb_qasper_0", "needle32k", "natural32k"]


def eval_layer(lf, device):
    d = torch.load(lf, map_location="cpu", weights_only=False)
    k, q, qpos, S = d["k"].to(device).float(), d["q"].to(device).float(), d["qpos"], d["S"]
    Hkv, H, D = k.shape[1], q.shape[1], k.shape[-1]
    G = H // Hkv
    pos = torch.arange(S, device=device)
    res = {}
    for ri in range(TAIL_N):
        t_r = int(qpos[-TAIL_N + ri])
        mid_hi = t_r + 1 - SWA
        mid_len = mid_hi - SINK
        if mid_len < 4096:
            continue
        # 真值分布（全维）
        qg = q[-TAIL_N + ri].reshape(Hkv, G, D)
        k4 = k[:, :, None, :].expand(S, Hkv, G, D)
        s_full = torch.einsum("hgd,shgd->hgs", qg, k4) * (D ** -0.5)
        s_full = s_full.masked_fill((pos > t_r).view(1, 1, -1), float("-inf"))
        p_full = torch.softmax(s_full, dim=-1)

        def cov_mass(cand):
            return float((p_full * cand.unsqueeze(1)).sum(-1).mean())

        # far mass（mid 区占比）= 预估器候选信号（prefill 免费可得）
        mid_mask = ((pos >= SINK) & (pos < mid_hi)).view(1, 1, -1)
        far_mass = float((p_full * mid_mask).sum(-1).mean())
        # 共享打分量（mavg：far 粗筛 minmax / near 粗筛 avg，细筛均 token 子空间分）
        idx = torch.tensor(D2I, device=device)
        ksub = k[..., idx]
        qsub = q[-TAIL_N + ri][..., idx].reshape(Hkv, G, 32).sum(1)
        nblk = (S + BS - 1) // BS
        kk = F.pad(ksub, (0, 0, 0, 0, 0, nblk * BS - S))
        kc = kk.reshape(nblk, BS, Hkv, 32)
        kmin, kmax, kavg = kc.amin(1), kc.amax(1), kc.mean(1)
        sc_mm = (torch.einsum("hd,nhd->hn", qsub.clamp(min=0), kmax) +
                 torch.einsum("hd,nhd->hn", qsub.clamp(max=0), kmin))
        sc_av = torch.einsum("hd,nhd->hn", qsub, kavg)
        sc_tok = torch.einsum("hgd,shd->hgs", q[-TAIL_N + ri][..., idx].reshape(Hkv, G, 32), ksub).sum(1)
        sc_blk = {"mm": sc_mm[:, SINK // BS:], "av": sc_av[:, SINK // BS:]}
        tok_score = sc_tok[:, SINK:mid_hi]
        sink_c = (pos < SINK).view(1, -1)
        swa_c = ((pos >= t_r + 1 - SWA) & (pos <= t_r)).view(1, -1)

        def select_sub(lo_off, hi_off, blk_key, n_pages, n_tokens):
            s1 = sc_blk[blk_key].masked_fill(
                ~(((torch.arange(sc_blk[blk_key].shape[1], device=device) * BS + SINK) < SINK + hi_off) &
                  ((torch.arange(sc_blk[blk_key].shape[1], device=device) * BS + BS - 1 + SINK) >= SINK + lo_off)).view(1, -1),
                float("-inf"))
            ib = torch.topk(s1, min(n_pages, s1.shape[-1]), dim=-1).indices
            tok = (ib.unsqueeze(-1) * BS + torch.arange(BS, device=device).view(1, 1, BS)).reshape(Hkv, -1).clamp(max=mid_len - 1)
            pool = torch.zeros(Hkv, mid_len, dtype=torch.bool, device=device)
            pool.scatter_(1, tok, True)
            pool &= ((torch.arange(mid_len, device=device) >= lo_off) &
                     (torch.arange(mid_len, device=device) < hi_off)).view(1, -1)
            ts = tok_score.masked_fill(~pool, float("-inf"))
            it = torch.topk(ts, min(n_tokens, mid_len), dim=-1).indices
            cand = torch.zeros(Hkv, S, dtype=torch.bool, device=device)
            cand.scatter_(1, it + SINK, True)
            return cand

        # ---- 网格臂（20 配置/行）----
        for a in ALPHAS:
            near_L = int(a * mid_len)
            near_lo_off = mid_len - near_L
            for b in BETAS:
                nb_near = max(1, int(round(BP * b)))
                nb_far = max(1, BP - nb_near)
                nt_near = int(nb_near * BS * GAMMA)
                nt_far = max(64, B_TOK - nt_near)
                cand = torch.zeros(Hkv, S, dtype=torch.bool, device=device)
                cand |= sink_c | swa_c
                cand |= select_sub(near_lo_off, mid_len, "av", nb_near, nt_near)   # near=avg
                cand |= select_sub(0, near_lo_off, "mm", nb_far, nt_far)           # far=minmax
                res.setdefault(f"a{a}_b{b}", []).append(cov_mass(cand))
        # mono 对照（α=0 全 mid 单池 minmax）
        cand = torch.zeros(Hkv, S, dtype=torch.bool, device=device)
        cand |= sink_c | swa_c
        cand |= select_sub(0, mid_len, "mm", BP, B_TOK)
        res.setdefault("mono", []).append(cov_mass(cand))
        res.setdefault("far_mass", []).append(far_mass)
    del k, q, d, ksub
    torch.cuda.empty_cache()
    return res


def main():
    device = "cuda:0"
    results = {}
    for name in SAMPLES:
        meta = f"{TRACE}/{name}/meta.json"
        if not os.path.isfile(meta):
            continue
        n_layers = json.load(open(meta))["n_layers"]
        per_layer = {}
        for li in range(0, n_layers, max(1, n_layers // 12)):
            r = eval_layer(f"{TRACE}/{name}/layer{li:02d}.pt", device)
            if not r:
                continue
            rec = {k2: round(sum(v) / len(v), 4) for k2, v in r.items() if k2 != "far_mass"}
            rec["far_mass"] = round(sum(r["far_mass"]) / len(r["far_mass"]), 4)
            per_layer[li] = rec
        # 三判决量
        cfgs = [k2 for k2 in per_layer[list(per_layer)[0]] if k2.startswith("a") or k2 == "mono"]
        # fixed_best：全层统一最优配置（跨层平均 cov 最大）
        fixed_scores = {c: sum(per_layer[li][c] for li in per_layer) / len(per_layer) for c in cfgs}
        fixed_best_cfg = max(fixed_scores, key=fixed_scores.get)
        fixed_best = fixed_scores[fixed_best_cfg]
        # per_layer_oracle：每层独立 argmax（含 mono 作为逐层可选项——预估器可输出「该层用 mono」）
        oracle_vals, oracle_cfgs, fms = [], [], []
        for li in per_layer:
            best_c = max(cfgs, key=lambda c: per_layer[li][c])
            oracle_vals.append(per_layer[li][best_c])
            oracle_cfgs.append(best_c)
            fms.append(per_layer[li]["far_mass"])
        per_layer_oracle = sum(oracle_vals) / len(oracle_vals)
        results[name] = {
            "per_layer": {str(li): per_layer[li] for li in per_layer},
            "fixed_best_cfg": fixed_best_cfg,
            "fixed_best": round(fixed_best, 4),
            "per_layer_oracle": round(per_layer_oracle, 4),
            "oracle_gain": round(per_layer_oracle - fixed_best, 4),
            "oracle_cfg_by_layer": {str(li): c for li, c in zip(per_layer, oracle_cfgs)},
            "far_mass_by_layer": {str(li): f for li, f in zip(per_layer, fms)},
        }
        print(f"[{name}] fixed({fixed_best_cfg})={fixed_best:.4f} "
              f"perlayer_oracle={per_layer_oracle:.4f} gain={per_layer_oracle - fixed_best:+.4f}", flush=True)
        print(f"  oracle_cfg: {results[name]['oracle_cfg_by_layer']}", flush=True)
    # 总判决
    gains = [v["oracle_gain"] for v in results.values()]
    print(f"\n=== E79a 总判决：per-layer oracle 增益均值 {sum(gains)/len(gains):+.4f} "
          f"(min {min(gains):+.4f} / max {max(gains):+.4f}) ===", flush=True)
    json.dump(results, open(OUT, "w"), indent=1)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
