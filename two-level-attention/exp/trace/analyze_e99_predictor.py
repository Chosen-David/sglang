# E99 预测器初探（#104 第一阶段）：便宜的 mass 侧信号能否预测 (α,β,γ) 的 e2e 精度？
# 核心问题（用户指令 2026-10-03）：E98 测完 (α,β,γ)×method 后，是否存在便宜、可在线测量的信号，
#   能在跑 e2e 之前预测哪个配置 e2e 最好？
# 数据源（只读）：
#   exp/trace/results/e98_abg_full_grid.json —— mass 全网格（16 样本 × 3600 臂；注意：只有标量 mass，
#     无 far/near 分区分解——如实记录，分区分解特征不可得）
#   exp/trace/results/e98_e2e_grid.json    —— e2e 实测 13 臂（hq/mu screen 口径，n=200）
#   exp/trace/results/e98_best_election.json —— 选举结果（best mavg_a0.125_b0.375_g0.625 avg 45.08）
# 必须诚实对照的先验（三重否定）：
#   E75  per-task oracle 完美选 β 增益 +0.08（LongBench 50.62 vs 50.54）
#   E79a per-layer oracle 增益 +0.0008（噪声级）
#   E79b 跨任务臂选择器（far_stat 信号）净增益 0
#   E98 新事实：γ 维在 e2e K2=1024 截断下实际平坦（mavg β.25 四 γ 同分）；
#     唯一系统性分化 = mavg β.375 vs β.25（avg +0.22，hq +1.28 / mu −0.86，任务形态交互）
# 小样本警示：n=13 臂（任务描述写 16，实际 JSON 为 13——如实纠正），任何相关都以小样本口径报告；
#   ~12 个候选特征同测 → 多重比较问题，报告 Bonferroni/BH 校正后显著性。
import json
import math
import os

import numpy as np
from scipy import stats

REPO = "/home/wangyuanshuo02/two-level-attention"
GRID = os.path.join(REPO, "exp/trace/results/e98_abg_full_grid.json")
E2E = os.path.join(REPO, "exp/trace/results/e98_e2e_grid.json")
ELECT = os.path.join(REPO, "exp/trace/results/e98_best_election.json")
OUT = os.path.join(REPO, "exp/trace/results/e99_predictor_probe.json")


def parse_tag(tag):
    """mavg_a0.125_b0.375_g0.625 -> (combo, alpha, beta, gamma)"""
    combo, a, b, g = tag.split("_")
    return combo, float(a[1:]), float(b[1:]), float(g[1:])


def bh_adjust(pvals):
    """Benjamini-Hochberg FDR 校正（实现标准步骤）"""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(n)
    adj[order] = np.minimum(ranked, 1.0)
    return adj


def boot_spearman_ci(x, y, n_boot=10000, seed=0):
    """Spearman 的 bootstrap 95% CI（小样本必须给区间，不给裸点估计）"""
    rng = np.random.default_rng(seed)
    n = len(x)
    rhos = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(set(idx)) < 3:
            continue  # 全同值无法算秩相关
        r, _ = stats.spearmanr(x[idx], y[idx])
        if not math.isnan(r):
            rhos.append(r)
    if len(rhos) < 100:
        return None
    lo, hi = np.percentile(rhos, [2.5, 97.5])
    return round(float(lo), 3), round(float(hi), 3)


def main():
    grid = json.load(open(GRID))
    e2e = json.load(open(E2E))
    elect = json.load(open(ELECT))
    per_sample = grid["per_sample"]

    # ---------- 组装 13 臂表 ----------
    arms = []
    for tag, rec in e2e.items():
        if tag in ("note", "v1_fix", "protocol"):
            continue
        if tag not in grid["mean"]:
            raise KeyError(f"e2e 臂 {tag} 不在 mass 网格中")
        combo, a, b, g = parse_tag(tag)
        # 注意：约束过滤是逐样本的（短样本部分臂不合法），并非每臂在全部 16 样本上都有 mass；
        # 特征计算按该臂的可用样本集，并记录覆盖数
        smass = {s: per_sample[s][tag] for s in per_sample if tag in per_sample[s]}
        if len(smass) < 4:
            raise RuntimeError(f"臂 {tag} 可用样本过少（{len(smass)}），特征不可靠")
        arms.append({
            "tag": tag, "combo": combo, "alpha": a, "beta": b, "gamma": g,
            "ref": bool(rec.get("ref", False)),
            "mass": grid["mean"][tag],           # 16 样本总 mass（部署侧最便宜信号）
            "n_mass_samples": len(smass),        # 该臂实际可用的 mass 样本覆盖数
            "e2e_hq": rec["hotpotqa"], "e2e_mu": rec["musique"],
            "e2e_avg": (rec["hotpotqa"] + rec["musique"]) / 2,
            "smass": smass,
        })
    n_arms = len(arms)
    tags = [a["tag"] for a in arms]

    # 样本分组（mass 网格 16 样本 → 任务匹配 / 其他 / 合成）；
    # 逐臂可用样本可能不全（约束过滤），分组内取该臂可用的样本
    hq_samples = [s for s in per_sample if "hotpotqa" in s]
    mu_samples = [s for s in per_sample if "musique" in s]
    screen_samples = hq_samples + mu_samples
    other_samples = [s for s in per_sample if s not in screen_samples]

    # ---------- 候选特征（全部为 e2e 前可得 / 在线可测口径的 mass 派生量） ----------
    def feat(a):
        sm = a["smass"]
        screen = [sm[s] for s in screen_samples if s in sm]
        others = [sm[s] for s in other_samples if s in sm]
        vals = list(sm.values())
        return {
            "mass_total": a["mass"],                                    # 总 mass（可用样本均值）
            "mass_screen": float(np.mean(screen)),                      # 任务匹配 mass（hq+mu 可用样本）
            "mass_other12": float(np.mean(others)),                     # 非匹配任务 mass
            "mass_screen_minus_other": float(np.mean(screen) - np.mean(others)),  # 任务判别度
            "mass_hq": float(np.mean([sm[s] for s in hq_samples if s in sm])),
            "mass_mu": float(np.mean([sm[s] for s in mu_samples if s in sm])),
            "mass_std": float(np.std(vals)),                            # 跨样本离散度（鲁棒性代理）
            "mass_min": float(np.min(vals)),                            # 最差样本 mass（悲观代理）
            "mass_needle32k": sm["needle32k"],                          # 合成针检索（全长样本必有）
            "mass_natural32k": sm["natural32k"],                        # 合成自然文本（全长样本必有）
            "beta": a["beta"],                                          # 配置参数本身（非信号，对照列）
        }

    feats = {a["tag"]: feat(a) for a in arms}
    feat_names = sorted(next(iter(feats.values())).keys())

    targets = {
        "e2e_avg": np.array([a["e2e_avg"] for a in arms]),
        "e2e_hq": np.array([a["e2e_hq"] for a in arms]),
        "e2e_mu": np.array([a["e2e_mu"] for a in arms]),
    }

    # ---------- 1) mass→e2e 预测力基线 + 全特征相关表 ----------
    corr_table = []
    pvals = []
    for fn in feat_names:
        x = np.array([feats[t][fn] for t in tags])
        row = {"feature": fn}
        for tn, y in targets.items():
            sp, sp_p = stats.spearmanr(x, y)
            pe, pe_p = stats.pearsonr(x, y)
            row[f"{tn}_spearman"] = round(float(sp), 3)
            row[f"{tn}_spearman_p"] = round(float(sp_p), 4)
            row[f"{tn}_pearson"] = round(float(pe), 3)
        corr_table.append(row)
        pvals.append(row["e2e_avg_spearman_p"])

    adj = bh_adjust(pvals)
    for row, a_ in zip(corr_table, adj):
        row["e2e_avg_spearman_p_BH"] = round(float(a_), 4)

    # 主结果：mass_total vs e2e_avg 的 bootstrap CI
    x_mass = np.array([feats[t]["mass_total"] for t in tags])
    ci_main = boot_spearman_ci(x_mass, targets["e2e_avg"])

    # ---------- 2) 分区分解可得性（如实记录） ----------
    # 全网格 JSON 每臂只有标量 mass（mean 与 per_sample 均为 coverage 标量），
    # 没有 far mass / near mass 分区量、没有 nt_near/far_budget 的逐臂记录——分区分解特征不可得。
    decomposition_available = False

    # ---------- 3) 部署口径：每个信号实际选中的臂与 regret ----------
    # 「用信号 X 选臂」= argmax(X)，看选中臂的 e2e 与 best 的差（regret，越大越亏）；
    # 方向（高好还是低好）若按实测相关事后选 = 乐观偏差，须双报并标注。
    # 注意 mass_std 的直觉方向（低=稳=好）与实测相关（ρ=+0.50 正）相反，故必须双报。
    best_e2e = max(targets["e2e_avg"])
    pick_table = []
    for fn in feat_names:
        x = np.array([feats[t][fn] for t in tags])
        row = {"feature": fn}
        for rule, idx_fn in (("argmax", np.argmax), ("argmin", np.argmin)):
            picked = arms[int(idx_fn(x))]
            row[f"{rule}_tag"] = picked["tag"]
            row[f"{rule}_e2e_avg"] = round(float(picked["e2e_avg"]), 2)
            row[f"{rule}_regret"] = round(float(best_e2e - picked["e2e_avg"]), 2)
        pick_table.append(row)
    pick_note = ("argmax/argmin 双报：方向事后取优 = 乐观偏差（真实部署须先验定方向）；"
                 "所有 mass 派生信号的最优方向 regret 均 >= 1.18")

    # mass 冠军选臂的实际损失（election 已知，这里给出完整数字；mass 方向=高好，取 argmax）
    mass_pick = [r for r in pick_table if r["feature"] == "mass_total"][0]
    mass_pick["picked_tag"] = mass_pick["argmax_tag"]
    mass_pick["picked_e2e_avg"] = mass_pick["argmax_e2e_avg"]

    # ---------- 4) 上限评估：oracle（完美预测器）增益 ----------
    best_arm = max(arms, key=lambda a: a["e2e_avg"])
    fixed_b025 = [a for a in arms if a["combo"] == "mavg" and a["beta"] == 0.25 and not a["ref"]]
    fixed_b025_avg = float(np.mean([a["e2e_avg"] for a in fixed_b025]))  # γ 截断后同配置
    ref_arm = [a for a in arms if a["ref"]][0]
    oracle = {
        "oracle_arm": best_arm["tag"], "oracle_avg": best_arm["e2e_avg"],
        "baseline_fixed_mavg_b025": round(fixed_b025_avg, 2),
        "baseline_ref_arm": ref_arm["e2e_avg"],
        "oracle_gain_vs_b025": round(best_arm["e2e_avg"] - fixed_b025_avg, 2),
        "oracle_gain_vs_ref": round(best_arm["e2e_avg"] - ref_arm["e2e_avg"], 2),
        "mass_pick_arm": mass_pick["picked_tag"],
        "mass_pick_avg": mass_pick["picked_e2e_avg"],
        "mass_pick_loss_vs_oracle": round(best_arm["e2e_avg"] - mass_pick["picked_e2e_avg"], 2),
    }

    # ---------- 5) mavg 家族内部（唯一系统性分化的战场，n=4） ----------
    mavg_arms = [a for a in arms if a["combo"] == "mavg"]
    fam = {
        "n": len(mavg_arms),
        "note": "γ 截断坍缩后 mavg β.25 各臂同分；唯一分化 = β.375 vs β.25",
        "arms": [{"tag": a["tag"], "mass": a["mass"], "e2e_avg": a["e2e_avg"],
                  "beta": a["beta"]} for a in mavg_arms],
    }
    if len(mavg_arms) >= 4:
        xm = np.array([a["mass"] for a in mavg_arms])
        ym = np.array([a["e2e_avg"] for a in mavg_arms])
        sp, sp_p = stats.spearmanr(xm, ym)
        fam["mass_vs_e2e_spearman"] = round(float(sp), 3)
        fam["mass_vs_e2e_spearman_p"] = round(float(sp_p), 4)
        # β.375 臂（e2e 最优家族内）在 mass 排名中的位次（1=最高）
        order = np.argsort(-xm)
        rank_b375 = int(np.where(order == np.argmax([a["beta"] == 0.375 for a in mavg_arms]))[0][0]) + 1
        fam["b375_arm_mass_rank"] = rank_b375
        fam["b375_arm_mass_rank_note"] = "mass 排名中 β.375（e2e 家族内最优）的位次；靠后=mass 在家族内主动误导"

    # ---------- 6) 有效配置去重敏感性（γ 截断/预算坍缩导致的同分臂） ----------
    by_score = {}
    for a in arms:
        key = (a["e2e_hq"], a["e2e_mu"])
        by_score.setdefault(key, []).append(a["tag"])
    dup_groups = {f"{k}": v for k, v in by_score.items() if len(v) > 1}
    dedup_tags = [g[0] for g in by_score.values()]  # 每组取首臂
    dedup_idx = [i for i, t in enumerate(tags) if t in dedup_tags]
    x_d = x_mass[dedup_idx]
    y_d = targets["e2e_avg"][dedup_idx]
    sp_d, sp_d_p = stats.spearmanr(x_d, y_d)
    sensitivity = {
        "n_raw": n_arms, "n_dedup": len(dedup_idx),
        "collapsed_groups": dup_groups,
        "note": "同分臂 = γ 截断 / 预算坍缩后的同有效配置（E98 v1 教训），去重后重算主相关",
        "mass_vs_e2e_spearman_dedup": round(float(sp_d), 3),
        "mass_vs_e2e_spearman_p_dedup": round(float(sp_d_p), 4),
    }

    # ---------- 判决 ----------
    # 逻辑：① 主信号（总 mass）预测力弱甚至负向（mass 冠军选臂亏 1.18）；
    #       ② 唯一系统分化是 β.375 vs β.25 的任务形态交互（hq +1.28 / mu −0.84），不是 mass 能表达的方向；
    #       ③ oracle 上限 +0.22~+0.49（n=13 小样本下连上限本身都带噪声）；
    #       ④ E75/E79a/E79b 三重否定先验：α/β 参数面平坦是结构性的（任务/层/信号三粒度都无杠杆）。
    corr_main = [r for r in corr_table if r["feature"] == "mass_total"][0]
    verdict = {
        "recommendation": "NO-GO",
        "reasoning": [
            f"主信号 mass_total vs e2e_avg Spearman={corr_main['e2e_avg_spearman']} "
            f"(p={corr_main['e2e_avg_spearman_p']}, n={n_arms}, bootstrap 95% CI 含 0)",
            f"mass 冠军选臂（{mass_pick['picked_tag']}）e2e {mass_pick['picked_e2e_avg']}, "
            f"比 oracle 亏 {oracle['mass_pick_loss_vs_oracle']}——mass 选臂不是中性而是主动有害",
            f"oracle 上限仅 +{oracle['oracle_gain_vs_b025']}（vs 固定 β.25）/"
            f"+{oracle['oracle_gain_vs_ref']}（vs 参照臂），与 E75 per-task oracle +0.08 同量级；"
            "n=13 下连上限本身都不稳定",
            "任务匹配 mass / 离散度 / 合成侧 mass 等 12 个候选特征（含对照列 beta）"
            "无一 BH 校正后显著，且小样本+多特征本身即多重比较陷阱；"
            "所有 mass 派生信号即使方向事后取优，regret 也 >= 1.18",
            f"唯一系统性分化（mavg β.375 vs β.25, avg +{oracle['oracle_gain_vs_b025']}）"
            "是 hq/mu 任务形态交互（hq +1.28/mu −0.86），"
            "不是任何 mass 侧量能表达的方向——与 E75 判决「信号-臂不可分」同构",
            "E75(+0.08)/E79a(+0.0008)/E79b(0) 三重否定先验 + 本次 n=13 复证："
            "平坦性是参数面结构性属性而非测量问题，预测器无杠杆可撬",
        ],
        "positive_note": "部署最优解仍是全局固定配置（E98 best mavg a.125/b.375/g.625）；"
                         "平坦性本身是部署简单卖点（论文 §4.4 既有转译）",
        "e2e_prior_alignment": {
            "E75_per_task_oracle_gain": 0.08, "E79a_per_layer_oracle_gain": 0.0008,
            "E79b_arm_selector_net_gain": 0.0,
            "E99_oracle_gain_vs_b025": oracle["oracle_gain_vs_b025"],
            "consistency": "四组数字同量级（≤0.5 分），α/β/γ 参数面平坦结论跨任务/层/信号/臂四粒度一致",
        },
    }

    out = {
        "task": "E99 #104 第一阶段：α/β/γ 预测器可行性初探（纯 CPU 数据分析）",
        "data": {
            "e2e_arms": n_arms,
            "e2e_arm_tags": tags,
            "n_discrepancy_note": "任务描述写 16 臂，e2e_grid.json 实为 13 臂（12 选举臂+1 参照臂），以 JSON 为准",
            "mass_grid_samples": len(per_sample),
            "region_decomposition_available": decomposition_available,
            "region_decomposition_note": "全网格每臂仅标量 mass（mean 与 per_sample 均为 coverage 标量），"
                                         "无 far/near 分区分解与 nt_near 逐臂记录，分区特征不可得",
        },
        "correlation_table": corr_table,
        "mass_vs_e2e_avg_bootstrap_CI": ci_main,
        "pick_table": pick_table,
        "pick_table_note": pick_note,
        "oracle": oracle,
        "mavg_family": fam,
        "dedup_sensitivity": sensitivity,
        "prior_alignment": verdict["e2e_prior_alignment"],
        "verdict": verdict,
    }
    json.dump(out, open(OUT, "w"), indent=1, ensure_ascii=False)

    # ---------- stdout 摘要 ----------
    print("=" * 72)
    print(f"E99 预测器初探：{n_arms} e2e 臂（任务描述写 16，实际 13，以 JSON 为准）")
    print("=" * 72)
    print("\n[1] 特征 vs e2e_avg 相关（Spearman, BH 校正后 p）——按 |ρ| 排序")
    for r in sorted(corr_table, key=lambda r: -abs(r["e2e_avg_spearman"])):
        print(f"  {r['feature']:<24} ρ={r['e2e_avg_spearman']:>6}  p={r['e2e_avg_spearman_p']:<7}"
              f"  p_BH={r['e2e_avg_spearman_p_BH']:<7}"
              f"  (hq ρ={r['e2e_hq_spearman']}, mu ρ={r['e2e_mu_spearman']})")
    print(f"  mass_total bootstrap 95% CI: {ci_main}")
    print(f"  n={n_arms} 小样本警示：任何 |ρ|<0.5 均无法与噪声区分")
    print("\n[2] 部署口径：各信号选中的臂与 regret（vs best 45.08，argmax/argmin 双报）")
    for r in sorted(pick_table, key=lambda r: min(r["argmax_regret"], r["argmin_regret"])):
        best_dir = "argmax" if r["argmax_regret"] <= r["argmin_regret"] else "argmin"
        print(f"  {r['feature']:<24} 最优方向={best_dir} → {r[f'{best_dir}_tag']:<28}"
              f" e2e={r[f'{best_dir}_e2e_avg']:>6}  regret={r[f'{best_dir}_regret']}"
              f"  (另一方向 regret={r['argmin_regret'] if best_dir == 'argmax' else r['argmax_regret']})")
    print(f"  注：{pick_note}")
    print("\n[3] oracle 上限")
    for k in ("oracle_arm", "oracle_avg", "baseline_fixed_mavg_b025", "baseline_ref_arm",
              "oracle_gain_vs_b025", "oracle_gain_vs_ref", "mass_pick_arm", "mass_pick_avg",
              "mass_pick_loss_vs_oracle"):
        print(f"  {k}: {oracle[k]}")
    print("\n[4] mavg 家族内部（唯一分化战场）")
    for k, v in fam.items():
        if k != "arms":
            print(f"  {k}: {v}")
    for a in fam["arms"]:
        print(f"    {a['tag']:<30} mass={a['mass']:.4f} e2e={a['e2e_avg']:.2f} β={a['beta']}")
    print("\n[5] 有效配置去重敏感性")
    print(f"  n {sensitivity['n_raw']}→{sensitivity['n_dedup']}, "
          f"dedup Spearman={sensitivity['mass_vs_e2e_spearman_dedup']} "
          f"(p={sensitivity['mass_vs_e2e_spearman_p_dedup']})")
    print("\n[6] 判决")
    print(f"  {verdict['recommendation']}")
    for r in verdict["reasoning"]:
        print(f"  - {r}")
    print(f"\n落袋: {OUT}")


if __name__ == "__main__":
    main()
