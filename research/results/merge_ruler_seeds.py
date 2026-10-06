# RULER 双 seed 合并分析（seed1=1234 + seed2=5678，各 n=20 → 合并 n=40）
# 支持 RULER 四任务 + QA 两任务（qa1/qa2，结果在独立 JSON）。
# 用法：python merge_ruler_seeds.py
import json

RES = "/home/wangyuanshuo02/sglang/tli_ruler_results.json"
RES_QA = "/home/wangyuanshuo02/sglang/tli_ruler_qa_results.json"

sources = []
for path, tags in (
    (RES, {
        1234: {"triton": "ruler_triton", "tli": "ruler_tli"},
        5678: {"triton": "ruler_triton_seed2", "tli": "ruler_tli_seed2"},
    }),
    (RES_QA, {
        1234: {"triton": "qa_triton", "tli": "qa_tli"},
        5678: {"triton": "qa_triton_seed2", "tli": "qa_tli_seed2"},
    }),
):
    try:
        d = json.load(open(path))
    except FileNotFoundError:
        continue
    sources.append((path.split("/")[-1], d, tags))

if not sources:
    raise SystemExit("no result files found")

# 检查哪些 tag 已存在
avail = {}
for fname, d, tags in sources:
    for s, m_tags in tags.items():
        for m, t in m_tags.items():
            avail.setdefault(fname, {}).setdefault(s, {})[m] = t in d
print("tag availability:", avail)

merged = {}  # task -> {method -> [scores...]}
for fname, d, tags in sources:
    for seed, m_tags in tags.items():
        for method, tag in m_tags.items():
            if tag not in d:
                continue
            for task, tr in d[tag].items():
                if tag == "qa_dry":  # dry run 轮不计入
                    continue
                merged.setdefault(task, {}).setdefault(method, [])
                merged[task][method].append((tr["score"], tr["n"]))

print(f"\n{'task':<22} {'FullKV(s1,s2)':<16} {'TLI(s1,s2)':<12} gap_pooled")
rows = []
for task, methods in merged.items():
    fk = methods.get("triton", [])
    tl = methods.get("tli", [])
    fk_str = ",".join(f"{s:.2f}" for s, _ in fk) or "—"
    tl_str = ",".join(f"{s:.2f}" for s, _ in tl) or "—"
    # pooled 加权均值（各 seed n 相同则简单平均）
    if fk and tl:
        fk_p = sum(s * n for s, n in fk) / sum(n for _, n in fk)
        tl_p = sum(s * n for s, n in tl) / sum(n for _, n in tl)
        gap = tl_p - fk_p
        rows.append((task, fk_p, tl_p, gap))
        print(f"{task:<22} {fk_str:<16} {tl_str:<12} {gap:+.3f}  "
              f"(pooled {fk_p:.3f} vs {tl_p:.3f})")
    else:
        print(f"{task:<22} {fk_str:<16} {tl_str:<12} (incomplete)")

if rows:
    # 分组均值：RULER 四任务 vs QA 两任务（QA 未全时自动跳过其行）
    groups = {
        "RULER 四任务": rows[:4],
        "QA 两任务": rows[4:6],
    }
    for gname, grows in groups.items():
        grows = [r for r in grows if r]
        if not grows:
            continue
        avg_gap = sum(r[3] for r in grows) / len(grows)
        avg_fk = sum(r[1] for r in grows) / len(grows)
        avg_tl = sum(r[2] for r in grows) / len(grows)
        print(f"\n{gname}均值 pooled: FullKV {avg_fk:.3f} / TLI {avg_tl:.3f} / gap {avg_gap:+.3f}")
    print("\n（RULER 四任务单 seed gap −0.21；双 seed 合并后 gap 波动 = 任务×seed 方差）")
