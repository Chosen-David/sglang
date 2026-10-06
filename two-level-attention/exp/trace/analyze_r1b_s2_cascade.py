# R1b-S2：级联损失分解（trace 重放）——审稿 S2「区分完整分数、代理分数和级联损失」回应。
# 把 tail32 子空间索引在 far 区 R_far 的 total 质量损失分解为三段：
#   ① 子空间降维损失 = full128_oracle − tail32_oracle（不两级，仅换打分子空间）
#   ② L1 块粗筛损失 = true-top（全维 top-B_TOK）落在 L1 块池外的 mass（不可恢复）
#   ③ L2 token 细筛损失 = true-top 在块池内、但被 L2 细筛 topk 丢掉的 mass
# 四个参照点（两档预算 B_TOK=2048 宽 / 1024 紧，块池 BP=64 块 × BS=64 = 4096 token 候选池）：
#   full128_oracle（不降维不两级）/ full128+两级 / tail32_oracle（降维不两级）/ tail32+两级（PSI replay 路径）
# 附加诊断：
#   psi_mixed（L1 full128 粗筛 + L2 tail32 细筛）= 论文主质量配置（L1 full128 + L2 量化低频 32 维，
#     此处为 fp 无量化 replay 逼近）；
#   oracle-L1（按 true-top mass 选块）把 L1 损失再拆为「块预算限制」vs「minmax 近似」两段；
#   spurious_gain（选中但不在 true-top 的 mass）使分解严格可加：
#     mass(true_top) = retained + L1_loss + L2_loss（逐 query 精确恒等式）
#     loss_total(recall 差) = L1_n + L2_n − spurious_n（精确恒等式，n = /tot_f 归一）
# 口径对齐 E85/E73：far 区全链 minmax 粗筛+细筛同特征，真值=全维 softmax per-head far
# mass 加权捕获；GQA group-sum q；GPU1。
import json
import os

import torch

TRACE = os.environ.get("R1B_TRACE", "/home/wangyuanshuo02/.archive/trace-dumps/qwen3-8b")
OUT = "/home/wangyuanshuo02/sglang/two-level-attention/exp/trace/results/r1b_cascade_decomposition.json"
SINK, SWA, NEAR_BAND = 128, 1024, 4096
BS = 64
BP = 64                    # L1 块池 = 64 块（e64b 术语 BP），× BS=64 = 4096 token 候选池
BUDS = [2048, 1024]        # L2 总 token 预算：宽口径 / 紧口径
TAIL_N = 2
SAMPLES = ["lb_gov_report_0", "lb_hotpotqa_0", "lb_musique_0", "lb_narrativeqa_0",
           "lb_passage_retrieval_en_0", "lb_qasper_0", "needle32k", "natural32k"]
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
TAIL32 = list(range(48, 64)) + list(range(112, 128))   # 16 个最低频完整旋转对（R1b pair-map）


def block_minmax_mask(coarse_kf, q_coarse, n_pages):
    """子空间 minmax 块粗筛：top-n_pages 块 → 块 bool mask [Hkv, nblk]（E85b 同款）"""
    T, Hkv, _ = coarse_kf.shape
    nblk = (T + BS - 1) // BS
    kk = torch.nn.functional.pad(coarse_kf, (0, 0, 0, 0, 0, nblk * BS - T))
    kc = kk.reshape(nblk, BS, Hkv, -1)
    kmin, kmax = kc.amin(1), kc.amax(1)
    sc = (torch.einsum("hd,nhd->hn", q_coarse.clamp(min=0), kmax) +
          torch.einsum("hd,nhd->hn", q_coarse.clamp(max=0), kmin))
    bmask = torch.zeros(Hkv, nblk, dtype=torch.bool, device=coarse_kf.device)
    bmask.scatter_(1, torch.topk(sc, min(n_pages, nblk), dim=-1).indices, True)
    return bmask


def pool_token_mask(bmask, T):
    """块池 mask [Hkv, nblk] → token 池 bool mask [Hkv, T]（padding 尾块截断到 T）"""
    Hkv, nblk = bmask.shape
    return bmask.unsqueeze(-1).expand(Hkv, nblk, BS).reshape(Hkv, nblk * BS)[:, :T]


def eval_layer(lf):
    """返回 [{bud: row}]（每个尾 query 一项；row 含四参照点 + 归一分解）"""
    d = torch.load(lf, map_location="cpu", weights_only=False)
    k, q, qpos, S = d["k"].to(DEV).float(), d["q"].to(DEV).float(), d["qpos"], d["S"]
    Hkv, H, D = k.shape[1], q.shape[1], q.shape[-1]
    G = H // Hkv
    rows_out = []
    t = int(qpos[-1])
    mid_hi = t + 1 - SWA
    far_lo, far_hi = SINK, mid_hi - NEAR_BAND
    if far_hi - far_lo < 8192:
        del k, q, d
        return None
    kfar = k[far_lo:far_hi]                       # [T, Hkv, 128]
    T = far_hi - far_lo
    nblk = (T + BS - 1) // BS
    idx_t = torch.tensor(TAIL32, device=DEV)
    kfar_tail = kfar[..., idx_t]
    for ri in range(TAIL_N):
        q_t = q[-TAIL_N + ri]
        q_head = q_t.reshape(Hkv, G, D).sum(1)    # [Hkv, 128] GQA group-sum
        q_tail = q_head[..., idx_t]               # [Hkv, 32]
        s = torch.einsum("hd,shd->hs", q_head, k) * (D ** -0.5)
        s = s.masked_fill(torch.arange(S, device=DEV).view(1, -1) > int(qpos[-TAIL_N + ri]), float("-inf"))
        p = torch.softmax(s, dim=-1)              # [Hkv, S] 真值分布
        pf = p[:, far_lo:far_hi]                  # [Hkv, T] far 区真值 mass
        tot_f = float(pf.sum())
        if tot_f < 1e-6:
            continue
        # 各打分子空间分数（L1 粗筛与 L2 细筛同特征，E85b 口径）
        s_full = torch.einsum("hd,shd->hs", q_head, kfar)
        s_tail = torch.einsum("hd,shd->hs", q_tail, kfar_tail)
        # L1 块池（minmax 粗筛）
        pool_full = pool_token_mask(block_minmax_mask(kfar, q_head, BP), T)      # [Hkv, T]
        pool_tail = pool_token_mask(block_minmax_mask(kfar_tail, q_tail, BP), T)
        per_bud = {}
        for bud in BUDS:
            kk = min(bud, T)
            it_full_or = torch.topk(s_full, kk, dim=-1).indices
            it_tail_or = torch.topk(s_tail, kk, dim=-1).indices

            def two_level(pool, s_fine):
                ts = s_fine.masked_fill(~pool, float("-inf"))
                return torch.topk(ts, kk, dim=-1).indices

            it_full_2lvl = two_level(pool_full, s_full)
            it_tail_2lvl = two_level(pool_tail, s_tail)
            it_mixed = two_level(pool_full, s_tail)   # 主配置 L1 full128 + L2 tail32

            def rec(it):
                return float(sum(pf[h, it[h]].sum() for h in range(Hkv))) / tot_f

            r = {
                "full128_oracle": rec(it_full_or),
                "tail32_oracle": rec(it_tail_or),
                "full128_2lvl": rec(it_full_2lvl),
                "tail32_2lvl": rec(it_tail_2lvl),
                "psi_mixed_L1full_L2tail": rec(it_mixed),
            }
            # ---- 三段分解（true-top = 全维 top-bud）----
            true_top = torch.zeros(Hkv, T, dtype=torch.bool, device=DEV)
            true_top.scatter_(1, it_full_or, True)
            sel = torch.zeros(Hkv, T, dtype=torch.bool, device=DEV)
            sel.scatter_(1, it_tail_2lvl, True)
            m_top = float((pf * true_top).sum())                       # = full128_oracle × tot_f
            l1_loss = float((pf * (true_top & ~pool_tail)).sum())
            l2_loss = float((pf * (true_top & pool_tail & ~sel)).sum())
            retained = float((pf * (true_top & sel)).sum())
            spurious = float((pf * (sel & ~true_top)).sum())
            # oracle-L1：按 true-top mass 选 BP 块（retained 的块池上界），拆 L1 损失
            blk_top_mass = torch.zeros(Hkv, nblk, device=DEV)
            blk_id = (torch.arange(T, device=DEV) // BS).unsqueeze(0).expand(Hkv, -1)
            blk_top_mass.scatter_add_(1, blk_id, pf * true_top)
            pool_or = pool_token_mask(torch.zeros(Hkv, nblk, dtype=torch.bool, device=DEV)
                                      .scatter_(1, torch.topk(blk_top_mass, min(BP, nblk), dim=-1).indices, True), T)
            l1_budget_loss = float((pf * (true_top & ~pool_or)).sum())              # BP 块预算本身不够
            l1_minmax_loss = float((pf * (true_top & pool_or & ~pool_tail)).sum())   # minmax 选块选错
            nrm = lambda x: x / tot_f
            r["decomp"] = {
                "L1_block_coarse_loss": round(nrm(l1_loss), 6),
                "L2_token_fine_loss": round(nrm(l2_loss), 6),
                "spurious_gain": round(nrm(spurious), 6),
                "retained": round(nrm(retained), 6),
                "L1_budget_restriction_loss": round(nrm(l1_budget_loss), 6),
                "L1_minmax_approx_loss": round(nrm(l1_minmax_loss), 6),
                # 恒等式自检（浮点级）
                "identity_raw": abs(m_top - retained - l1_loss - l2_loss),
                "identity_recall": abs((l1_loss + l2_loss - spurious) / tot_f
                                       - (r["full128_oracle"] - r["tail32_2lvl"])),
            }
            per_bud[str(bud)] = {k2: (round(v, 6) if isinstance(v, float) else v)
                                 for k2, v in r.items()}
        rows_out.append(per_bud)
    del k, q, d, kfar, kfar_tail
    if DEV != "cpu":
        torch.cuda.empty_cache()
    return rows_out


def agg_rows(rows, key):
    vals = [r[key] for r in rows]
    return round(sum(vals) / len(vals), 4) if vals else None


def main():
    torch.set_grad_enabled(False)
    results = {}
    for name in SAMPLES:
        meta = f"{TRACE}/{name}/meta.json"
        if not os.path.isfile(meta):
            continue
        n_layers = json.load(open(meta))["n_layers"]
        layers = {}
        all_rows = {str(b): [] for b in BUDS}
        for li in range(0, n_layers, max(1, n_layers // 8)):
            rows = eval_layer(f"{TRACE}/{name}/layer{li:02d}.pt")
            if not rows:
                continue
            lrow = {}
            for bud in [str(b) for b in BUDS]:
                brows = [r[bud] for r in rows]
                if not brows:
                    continue
                all_rows[bud].extend(brows)
                fp = {k2: agg_rows(brows, k2) for k2 in brows[0] if k2 != "decomp"}
                dc = {k2: round(sum(x["decomp"][k2] for x in brows) / len(brows), 6)
                      for k2 in brows[0]["decomp"]}
                lrow[bud] = {"four_points": fp, "decomp": dc}
            layers[f"L{li:02d}"] = lrow
        samp = {"layers": layers}
        for bud in [str(b) for b in BUDS]:
            brows = all_rows[bud]
            if brows:
                samp[bud] = {
                    "four_points": {k2: agg_rows(brows, k2) for k2 in brows[0] if k2 != "decomp"},
                    "decomp": {k2: round(sum(x["decomp"][k2] for x in brows) / len(brows), 6)
                               for k2 in brows[0]["decomp"]},
                    "n_obs": len(brows),
                }
        results[name] = samp
        for bud in [str(b) for b in BUDS]:
            fp = samp.get(bud, {}).get("four_points")
            if fp:
                dc = samp[bud]["decomp"]
                print(f"[{name} B={bud}] full_or={fp['full128_oracle']} tail_or={fp['tail32_oracle']} "
                      f"full_2lvl={fp['full128_2lvl']} tail_2lvl={fp['tail32_2lvl']} "
                      f"mixed={fp['psi_mixed_L1full_L2tail']} "
                      f"L1={dc['L1_block_coarse_loss']} L2={dc['L2_token_fine_loss']} "
                      f"spur={dc['spurious_gain']}", flush=True)

    summary = {}
    for bud in [str(b) for b in BUDS]:
        samps = [v[bud] for v in results.values() if bud in v]
        if not samps:
            continue
        summary[bud] = {
            "four_points": {k2: round(sum(s["four_points"][k2] for s in samps) / len(samps), 4)
                            for k2 in samps[0]["four_points"]},
            "decomp": {k2: round(sum(s["decomp"][k2] for s in samps) / len(samps), 6)
                       for k2 in samps[0]["decomp"]},
            "n_samples": len(samps),
        }
        fp, dm = summary[bud]["four_points"], summary[bud]["decomp"]
        print(f"\n== B_TOK={bud}（{len(samps)} 样本均值）==")
        print(f"  四参照点: full_oracle={fp['full128_oracle']}  full_2lvl={fp['full128_2lvl']}  "
              f"tail_oracle={fp['tail32_oracle']}  tail_2lvl={fp['tail32_2lvl']}  "
              f"mixed(L1full+L2tail)={fp['psi_mixed_L1full_L2tail']}")
        print(f"  差分: 降维={round(fp['full128_oracle']-fp['tail32_oracle'],4)}  "
              f"两级@tail={round(fp['tail32_oracle']-fp['tail32_2lvl'],4)}  "
              f"两级@full={round(fp['full128_oracle']-fp['full128_2lvl'],4)}  "
              f"总={round(fp['full128_oracle']-fp['tail32_2lvl'],4)}")
        print(f"  true-top 分解(归一): L1={dm['L1_block_coarse_loss']}  L2={dm['L2_token_fine_loss']}  "
              f"spurious={dm['spurious_gain']}  (L1再拆: 预算={dm['L1_budget_restriction_loss']} "
              f"minmax={dm['L1_minmax_approx_loss']})")

    out = {
        "config": {"trace": TRACE, "bs": BS, "bp_blocks": BP, "pool_tokens": BP * BS,
                   "budgets_tok": BUDS, "sink": SINK, "swa": SWA, "near_band": NEAR_BAND,
                   "tail_n": TAIL_N, "tail32_dims": TAIL32,
                   "samples": SAMPLES, "layers_per_sample": 8, "device": DEV,
                   "ground_truth": "全维 softmax per-head far mass 加权捕获；GQA group-sum q（E85/E73 同款）",
                   "arms": "L1 minmax 块粗筛（BP=64 块）+ L2 细筛 topk（B_TOK）；粗筛细筛同特征（E85b 口径）",
                   "decomp_identity": "loss_total = L1_n + L2_n − spurious_n（逐 query 精确恒等）；"
                                      "mass(true_top) = retained + L1 + L2（原始 mass 恒等）",
                   "note": "decomp 均已按 tot_f 归一，与 recall 差分同量纲；identity_* 为浮点自检残差"},
        "summary": summary, "per_sample": results,
    }
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
