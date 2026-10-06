# 终表生成（用户指定格式）：method组合 | 最佳αβ | 数据集与精度 | far跳层 | 速率
import json, os, re

CD = "/tmp/tli_chain"
ROOT = "/home/wangyuanshuo02/two-level-attention"
meta = json.load(open(f"{CD}/screen_arms.json"))
scores = json.load(open(f"{CD}/screen_scores.json"))
best = open(f"{CD}/best_arm.txt").read().strip()

# 从 screen.log + full0/full1.log 提取每臂每任务耗时（tqdm 末行 elapsed）
def parse_speed():
    sp = {}
    arm = None
    import glob as _g
    logs = [f"{CD}/screen.log", f"{CD}/full0.log", f"{CD}/full1.log"]
    for logf in logs:
        if not os.path.exists(logf):
            continue
        arm = None
        for line in open(logf):
            m = re.search(r"\[screen (\w+) a=([\d.]+) b=([\d.]+)", line)
            if m:
                arm = m.group(1)
            # 兼容单卡 [full mavg] 与双卡 [full0/1 mavg] 前缀
            m2 = re.search(r"\[full0?1? (\w+) (\S+)", line)
            if m2:
                arm = m2.group(1)
            m3 = re.search(r"(\d+)/\d+ \[(\d+):(\d+)<", line)
            if m3 and arm:
                # 任务级累计难归属，取臂级平均 s/it 近似
                sp.setdefault(arm, []).append(float(m3.group(2)) * 60 + float(m3.group(3)))
    return {a: round(sum(v) / len(v) / 60, 1) for a, v in sp.items()}  # 分钟/样本均值

speed = parse_speed()
NAME = {"mavg": "mavg=(minmax,avg)", "mminmax": "mminmax=(minmax,minmax)",
        "aavg": "aavg=(avg,avg)", "cavg": "cavg=(cluster,avg)",
        "mavg(B7s)": "mavg=(minmax,avg)[B7s]", "ccluster": "ccluster=(cluster,cluster)"}
TRACE_ONLY = {"ccluster": "trace-only：e2e 未参数化 near-cluster"}

rows = []
for arm, sc in sorted(scores.items(), key=lambda x: -(x[1].get("screen_avg") or -1)):
    core = arm.replace("(B7s)", "")
    if arm == "mavg(B7s)":
        a, b = 0.125, 0.25  # B7s 参照臂实际 β=0.25（勿用核心臂 mavg 的 β=0.375）
    else:
        a = meta.get(core, {}).get("alpha", 0.125)
        b = meta.get(core, {}).get("beta", 0.25)
    acc = f"hotpotqa {sc.get('hotpotqa')} / musique {sc.get('musique')}"
    if arm == best:
        try:
            full = json.load(open(f"{CD}/full_scores.json"))
            acc += f"；13任务全量 AVG {full.get('AVG')}"
        except FileNotFoundError:
            acc += "（全量待打分）"
    note = TRACE_ONLY.get(core, "")
    rows.append((NAME.get(arm, arm), f"α={a} β={b} γ=0.125", acc,
                 "未启用（E5b：静态跨任务掉分→需 per-task 校准；E6 静态版可跳 13/36 层 far 损失<0.3%）",
                 f"~{speed.get(core, '-')} min/样本(e2e 含 prefill)" + (f" [{note}]" if note else "")))

md = ["# E72 五组合 e2e 真实精度对比（Qwen3-8B LongBench, F1 口径）", "",
      "| method 组合 | 最佳 α/β | 数据集精度 | far 跳层 | 速率 |",
      "|---|---|---|---|---|"]
for r in rows:
    md.append("| " + " | ".join(r) + " |")
md += ["", f"**best arm: {best}**", "",
       "- trace 口径（E64j mass，B_TOK=2048 宽预算）：mono mavg 0.9166 > mminmax 0.9116 >"
       " ccluster 0.9096 > cavg 0.9033 > mavg 0.9012 > aavg 0.8036——与 e2e F1 排名不一致，"
       "快筛判决以 e2e 为准（trace mass 排名不可替代 e2e 判决）",
       "- kernel 速度资产（与 method 无关的公共开销）：L1 fused 3.6×/L2 级联 1.63×（E8-2）",
       "- ccluster 仅 trace 评估；如 trace 显著最优需补 e2e near-cluster 参数化"]
out = "\n".join(md)
open(f"{CD}/final_table.md", "w").write(out)
open(f"{ROOT}/exp/trace/results/e72_combo_table.md", "w").write(out)
print(out)
