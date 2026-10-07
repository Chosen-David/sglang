# E114（R2a-D1）：F0 vs F1 等宽 32 实标量判别 + value-aware 三损失诊断
# 来源：research/docs/子空间设计改进建议_by_gpt (1).md §15.4/§18.7/§19.4/§19.6
#   F0 = 16 原完整旋转对（= E76 tail32 基准同构，48..63 ∪ 112..127）
#   F1 = 8 关键 FC（H 冻结最低频对）+ 16 实维 H 正交补 Uᵀ 压缩（式 18.1）
#        关键 S = 最低频 8 对 (j,j+64) j∈{56..63}；H = 其余 48 对
#        U = H 区共享 SVD top-8 复用 E76 basis_sh 资产（MLA 式跨 head 共享）
# 判别门（文档 §15.4 停止规则预设）：F1 mass recall 不胜 F0 → 停，负结果如实落袋
# §19.4 三损失：L_score = mean||e||²（残差 logit）；
#   L_sens = mean||W_O·J_s·e||²（式 19.2 微分 α_j(v_j−y)）——本 trace 无 v/W_O，
#   降级为 value-agnostic 代理：||J_s·e||² 用 α_j 加权（v 项留待 D2 teacher-forced）
# 协议：E65/E76 组 C 同款——粗筛固定尾维 32 minmax top16 块，池内细筛各臂特征
#   top BUD_TOK=512；真值 = 全维 softmax per-head far mass 加权捕获（E4c 同款）
import json
import time

import numpy as np
import torch

TRACE = "/tmp/trace/qwen3-8b"
OUT = "/home/wangyuanshuo02/two-level-attention/exp/trace/results/e114_d1_f0_f1.json"
D2I = list(range(48, 64)) + list(range(112, 128))   # F0 尾维 32（16 完整对）
S = {j: (j, j + 64) for j in range(64)}             # 完整旋转对 (j, j+64)
# F1 关键 FC：最低频 8 对 = 尾维中最低频的 8 个完整对（j=56..63）
KEY_J = list(range(56, 64))
KEY_IDX = sorted([j for jd in KEY_J for j in S[jd]])          # 16 实标量
H_IDX = sorted(set(range(128)) - set(KEY_IDX))                # 112 实标量（48 对的 H 补空间）
SINK, SWA, NEAR_BAND = 128, 1024, 4096
BUD_TOKS = [64, 128, 256]   # 收紧预算恢复判别力（512 时 mass 饱和 oracle=1.0）
N_PAGES, BS, TAIL_N = 16, 64, 2
SAMPLES = ["lb_gov_report_0", "lb_hotpotqa_0", "lb_musique_0", "lb_narrativeqa_0",
           "lb_passage_retrieval_en_0", "lb_qasper_0", "needle32k", "natural32k"]
LAYERS = [4, 12, 20, 28]


def select_region(coarse_kf, q_coarse, blk_fn, fine_kf, q_fine, score_fn=None, bud_tok=None):
    """E76 原版：粗筛 top N_PAGES 块 → 池内 fine 特征 token top BUD_TOK。"""
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
    it = torch.topk(ts, min(bud_tok, T), dim=-1).indices
    return it, pool


def blk_minmax(kf, qf, nblk):
    S, Hkv, d = kf.shape
    kk = torch.nn.functional.pad(kf, (0, 0, 0, 0, 0, nblk * BS - S))
    kc = kk.reshape(nblk, BS, Hkv, d)
    kmin, kmax = kc.amin(1), kc.amax(1)
    return (torch.einsum("hd,nhd->hn", qf.clamp(min=0), kmax) +
            torch.einsum("hd,nhd->hn", qf.clamp(max=0), kmin))


def eval_layer(lf, BUD_TOK):
    d = torch.load(lf, map_location="cpu", weights_only=False)
    k, q, qpos, S_ = d["k"].float(), d["q"].float(), d["qpos"], d["S"]
    Hkv, H, D = k.shape[1], q.shape[1], k.shape[-1]
    G = H // Hkv
    res = {}
    t = int(qpos[-1])
    mid_hi = t + 1 - SWA
    far_lo, far_hi = SINK, mid_hi - NEAR_BAND
    if far_hi - far_lo < 8192:
        del k, q, d
        return None
    # F1 共享基：H 补空间（48 对 112 维）跨 kv-head 共享 SVD top-8 → 16 实维
    # （复用 E76 basis_sh 逻辑，但只在 H_IDX 列上拟合）
    k_all_H = k[far_lo:far_hi][:, :, H_IDX].reshape(-1, len(H_IDX))
    _, _, Vt_H = torch.linalg.svd(k_all_H, full_matrices=False)
    U_H = Vt_H[:8].T                                       # [112, 8] → Uᵀ 16 实维
    for ri in range(TAIL_N):
        q_t = q[-TAIL_N + ri]
        q_head = q_t.reshape(Hkv, G, D).sum(1)             # [Hkv,128] GQA 对齐
        s = torch.einsum("hd,shd->hs", q_head, k) * (D ** -0.5)
        s = s.masked_fill(torch.arange(S_).view(1, -1) > int(qpos[-TAIL_N + ri]), float("-inf"))
        p = torch.softmax(s, dim=-1)
        far_h = p[:, far_lo:far_hi].sum(-1)
        tot_f = float(far_h.sum())
        if tot_f < 1e-6:
            continue
        # 粗筛固定尾维 32（所有臂共用同一 pool —— 公平）
        idx32 = torch.tensor(D2I)
        k_far32, q_far32 = k[far_lo:far_hi][:, :, idx32], q_head[:, idx32]
        toks_f0, pool = None, None
        # —— F0：16 原完整对（= tail32 基准同构）——
        toks, pool = select_region(k_far32, q_far32, blk_minmax, k_far32, q_far32, bud_tok=BUD_TOK)
        cap = sum(float(p[h, toks[h] + far_lo].sum()) for h in range(Hkv))
        res.setdefault(f"C_F0_16pairs_b{BUD_TOK}", []).append(cap / tot_f)
        toks_f0 = toks
        # —— F1：8 关键 FC + H 补空间 Uᵀ 16 实维（式 18.1）——
        k_S = k[far_lo:far_hi][:, :, KEY_IDX]              # [T,Hkv,16]
        q_S = q_head[:, KEY_IDX]                            # [Hkv,16]
        k_H = k[far_lo:far_hi][:, :, H_IDX] @ U_H           # [T,Hkv,8]（8 实维 = 4 对）
        q_Hc = q_head[:, H_IDX] @ U_H                       # [Hkv,8]
        kf1 = torch.cat([k_S, k_H], dim=-1)                 # 16+8 = 24 实标量（<32 等宽内）
        qf1 = torch.cat([q_S, q_Hc], dim=-1)
        toks, _ = select_region(k_far32, q_far32, blk_minmax, kf1, qf1, bud_tok=BUD_TOK)
        cap = sum(float(p[h, toks[h] + far_lo].sum()) for h in range(Hkv))
        res.setdefault(f"C_F1_8fcH8_b{BUD_TOK}", []).append(cap / tot_f)
        # —— F1w：等宽 32 实标量版（8 关键 FC 16 + H 压缩 16 实维 = U top-16）——
        U_H16 = Vt_H[:16].T
        k_H16 = k[far_lo:far_hi][:, :, H_IDX] @ U_H16
        q_H16 = q_head[:, H_IDX] @ U_H16
        kf1w = torch.cat([k_S, k_H16], dim=-1)              # 16+16 = 32 实标量
        qf1w = torch.cat([q_S, q_H16], dim=-1)
        toks, _ = select_region(k_far32, q_far32, blk_minmax, kf1w, qf1w, bud_tok=BUD_TOK)
        cap = sum(float(p[h, toks[h] + far_lo].sum()) for h in range(Hkv))
        res.setdefault(f"C_F1w_8fcH16_eq32_b{BUD_TOK}", []).append(cap / tot_f)
        # —— oracle128 ——
        s_far = s[:, far_lo:far_hi]
        order = torch.argsort(s_far, dim=-1, descending=True)
        cap = sum(float(p[h, order[h][:BUD_TOK] + far_lo].sum()) for h in range(Hkv))
        res.setdefault(f"oracle128_b{BUD_TOK}", []).append(cap / tot_f)
        # —— §19.4 诊断：残差 logit e 与 value-aware 一阶敏感（α 加权代理）——
        # e_j = s_j − ŝ_j（F1w 代理分数 vs 全维真值分数），池内计算
        s_true = s_far                                        # [Hkv,T_far]
        # F1w 代理分（未 softmax 的线性分，含 scale）
        s_hat = torch.einsum("hd,thd->ht", qf1w, kf1w) * (D ** -0.5)
        e = (s_true - s_hat)
        res.setdefault("L_score_sq", []).append(float((e ** 2).mean()))
        # J_s 一阶敏感的 value-agnostic 代理：Σ_j α_j·|e_j|（式 19.2 去 v_j−y 项的模长上界）
        alpha_far = p[:, far_lo:far_hi]
        res.setdefault("L_sens_alpha_abs_e", []).append(float((alpha_far * e.abs()).sum(-1).mean()))
        # mass 分歧度：F0 与 F1w 选中集合的重合率（§19.6 表：token CA）
        t0 = torch.stack([torch.tensor(sorted(x.tolist())) for x in toks_f0])
        t1 = torch.stack([torch.tensor(sorted(x.tolist())) for x in toks])
        inter = (t0[:, None, :] == t1[None, :, :]).any(-1).float() if False else None
        ca = []
        for h in range(Hkv):
            a, b = set(toks_f0[h].tolist()), set(toks[h].tolist())
            ca.append(len(a & b) / len(a))
        res.setdefault("CA_F0_F1w", []).append(float(np.mean(ca)))
    del k, q, d
    return res


def main():
    t0 = time.time()
    all_res = {}
    for bt in BUD_TOKS:
        for sm in SAMPLES:
            for ly in LAYERS:
                lf = f"{TRACE}/{sm}/layer{ly:02d}.pt"
                try:
                    r = eval_layer(lf, bt)
                except FileNotFoundError:
                    continue
                if r is None:
                    continue
                for k, v in r.items():
                    all_res.setdefault(k, []).extend(v)
                print(f"{sm} L{ly} b{bt} done", flush=True)
    # 汇总
    summary = {k: dict(mean=round(float(np.mean(v)), 4), n=len(v)) for k, v in all_res.items()}
    verdicts = []
    for bt in BUD_TOKS:
        f0 = summary.get(f"C_F0_16pairs_b{bt}", {}).get("mean")
        f1w = summary.get(f"C_F1w_8fcH16_eq32_b{bt}", {}).get("mean")
        oc = summary.get(f"oracle128_b{bt}", {}).get("mean")
        verdicts.append(f"b{bt}: F0={f0} F1w={f1w} oracle={oc} — " +
                        ("F1w>F0" if f1w and f0 and f1w > f0 else ("tie" if f1w == f0 else "F0 wins")))
    f0, f1, f1w = None, None, None
    verdict = {
        "experiment": "E114 (R2a-D1): F0 vs F1 equal-width-32 real-scalar discrimination",
        "source_doc": "research/docs/子空间设计改进建议_by_gpt (1).md §15.4/§18.7/§19.4/§19.6",
        "protocol": "E65 group-C: coarse fixed tail32 minmax top16 blocks, fine per-arm feature top512, "
                    "oracle = full-dim softmax per-head far mass capture",
        "arms": {
            "F0": "16 original complete rotation pairs (tail32 isomorphic, 48..63+112..127)",
            "F1": "8 key FCs (lowest-freq pairs j=56..63, 16 scalars) + H-orthogonal-complement "
                  "shared-SVD top-8 (16 scalars real dims) = 24 scalars",
            "F1w": "8 key FCs (16) + H shared-SVD top-16 (16) = 32 scalars equal-width"},
        "summary": summary,
        "discrimination_gate": "F1/F1w mass recall must BEAT F0 to proceed to D2 teacher-forced (doc §15.4)",
    }
    verdict["per_budget_verdict"] = verdicts
    verdict["verdict"] = ("saturation analysis: b512 oracle=1.0 (mass locked by coarse pool); "
                          "discrimination at b64/128/256 — " + ("ANY F1w>F0 → D2 eligible"
                          if any("F1w>F0" in v for v in verdicts) else "F1 never beats F0 → STOP per §15.4"))
    json.dump(verdict, open(OUT, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(verdict["verdict"], ensure_ascii=False))
    print("elapsed", round(time.time() - t0, 1), "s →", OUT)


if __name__ == "__main__":
    main()
