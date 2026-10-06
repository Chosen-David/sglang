# E98 best method + best (α,β,γ) 选举（用户最高指令：只认 e2e 实测，mass 仅为对照）
# 输入：exp/trace/results/e98_e2e_grid.json（v2 网格全落袋后运行）
# 输出：exp/trace/results/e98_best_election.json（选举判决 + 诚实口径注）
# 选举准则（与论文indexer.md 对齐）：
#   1. screen 双任务均值为主排序键（hq+mu 等权）
#   2. 双任务都非 None 才有选举资格（单任务臂只能参考）
#   3. mass 排序与 e2e 排序的差异如实报告（已四度反转，不许用 mass 冠军冒充 e2e 冠军）
#   4. 参照臂（当前部署 mavg α.125/β.375/γ.125）成对对照报告增益
#   5. ccluster 只有 mass 数据无 e2e 路径——如实注明不参与选举
import json

R = "/home/wangyuanshuo02/two-level-attention/exp/trace/results"
d = json.load(open(f"{R}/e98_e2e_grid.json"))
runs = json.load(open("/tmp/e98_e2e_runs_v2.json"))

rows = []
for r in runs:
    tag = f"{r['combo']}_a{r['a']}_b{r['b']}_g{r['g']}"
    e = d.get(tag, {})
    hq, mu = e.get("hotpotqa"), e.get("musique")
    if hq is None or mu is None:
        rows.append({"tag": tag, "mass": r.get("mass"), "hq": hq, "mu": mu,
                     "avg": None, "eligible": False, "note": "单任务缺失，仅参考"})
        continue
    rows.append({"tag": tag, "mass": r.get("mass"), "hq": hq, "mu": mu,
                 "avg": round((hq + mu) / 2, 2), "eligible": True,
                 "ref": bool(r.get("ref"))})

eligible = [r for r in rows if r["eligible"] and not r.get("ref")]
ranking = sorted(eligible, key=lambda x: -x["avg"])
best = ranking[0] if ranking else None

out = {
    "note": "E98 best 选举：只认 e2e 实测（screen hq+mu 等权均值）；mass 为对照列",
    "ccluster": "无 e2e 实现路径（near 侧 cluster 未实现），仅 mass 数据，不参与选举",
    "ranking": ranking,
    "best": best,
    "mass_vs_e2e": None,
    "ref_arm": next((r for r in rows if r.get("ref")), None),
}

# mass 冠军 vs e2e 冠军对照（如实报告反转）
if best:
    mass_rank = sorted(eligible, key=lambda x: -(x["mass"] or 0))
    out["mass_vs_e2e"] = {
        "e2e_best": {"tag": best["tag"], "avg": best["avg"], "mass": best["mass"]},
        "mass_best": {"tag": mass_rank[0]["tag"], "mass": mass_rank[0]["mass"],
                      "avg": mass_rank[0]["avg"]},
        "reversed": mass_rank[0]["tag"] != best["tag"],
    }
    # 参照臂增益
    if out["ref_arm"] and out["ref_arm"]["avg"] is not None:
        out["ref_gain"] = {
            "best_avg": best["avg"], "ref_avg": out["ref_arm"]["avg"],
            "delta": round(best["avg"] - out["ref_arm"]["avg"], 2)}

json.dump(out, open(f"{R}/e98_best_election.json", "w"), indent=1, ensure_ascii=False)
print("=== E98 e2e 排名（hq+mu 均值降序）===")
for r in ranking:
    print(f"  {r['tag']}: avg={r['avg']} hq={r['hq']} mu={r['mu']} mass={r['mass']}")
if out["ref_arm"] and out["ref_arm"]["avg"] is not None:
    print(f"参照臂 {out['ref_arm']['tag']}: avg={out['ref_arm']['avg']}")
if out["mass_vs_e2e"]:
    print(f"mass 冠军 {out['mass_vs_e2e']['mass_best']['tag']} vs e2e 冠军 {out['mass_vs_e2e']['e2e_best']['tag']}"
          f" 反转={out['mass_vs_e2e']['reversed']}")
print(f"\nBEST: {best['tag'] if best else 'N/A'}")
print("saved e98_best_election.json")
