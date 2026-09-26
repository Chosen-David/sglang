# RULER 双 seed 合并分析（seed1=1234 + seed2=5678，各 n=20 → 合并 n=40）
# 用法：python merge_ruler_seeds.py
import json

RES = "/home/wangyuanshuo02/sglang/tli_ruler_results.json"

d = json.load(open(RES))
seeds = {
    1234: {"triton": "ruler_triton", "tli": "ruler_tli"},
    5678: {"triton": "ruler_triton_seed2", "tli": "ruler_tli_seed2"},
}

# 检查哪些 tag 已存在
avail = {s: {m: (t in d) for m, t in tags.items()} for s, tags in seeds.items()}
print("tag availability:", avail)

merged = {}  # task -> {method -> [scores...]}
for seed, tags in seeds.items():
    for method, tag in tags.items():
        if tag not in d:
            continue
        for task, tr in d[tag].items():
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
    avg_gap = sum(r[3] for r in rows) / len(rows)
    avg_fk = sum(r[1] for r in rows) / len(rows)
    avg_tl = sum(r[2] for r in rows) / len(rows)
    print(f"\n均值 pooled: FullKV {avg_fk:.3f} / TLI {avg_tl:.3f} / gap {avg_gap:+.3f}")
    # 与单 seed 对比稳定性
    print("（对照单 seed gap −0.21；双 seed 合并后 gap 波动 = 任务×seed 方差）")
