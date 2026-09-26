# RULER QA 失败模式分析：对比 FullKV vs TLI 样本级命中（哪些样本 TLI 丢/共同丢），
# 打印失败样本的答案与输出前缀，供报告失败模式诊断。
import json

RES_QA = "/home/wangyuanshuo02/sglang/tli_ruler_qa_results.json"
d = json.load(open(RES_QA))

PAIRS = [
    ("qa_triton", "qa_tli", "seed1"),
    ("qa_triton_seed2", "qa_tli_seed2", "seed2"),
]

for fk_tag, tli_tag, sname in PAIRS:
    if fk_tag not in d or tli_tag not in d:
        print(f"--- {sname}: incomplete ({fk_tag in d}, {tli_tag in d})")
        continue
    print(f"=== {sname}: {fk_tag} vs {tli_tag}")
    for task in ("qa1", "qa2"):
        if task not in d[fk_tag] or task not in d[tli_tag]:
            continue
        fk_rows = d[fk_tag][task]["samples"]
        tli_rows = d[tli_tag][task]["samples"]
        n = min(len(fk_rows), len(tli_rows))
        both_ok = fk_only = tli_only = both_bad = 0
        fails = []
        for i in range(n):
            f, t = fk_rows[i], tli_rows[i]
            if f["ok"] and t["ok"]:
                both_ok += 1
            elif f["ok"] and not t["ok"]:
                fk_only += 1
                fails.append(("TLI丢", i, f, t))
            elif not f["ok"] and t["ok"]:
                tli_only += 1
                fails.append(("TLI独中", i, f, t))
            else:
                both_bad += 1
                fails.append(("共同丢", i, f, t))
        print(f"\n[{task}] n={n}: both_ok {both_ok} / FullKV独中 {fk_only} / "
              f"TLI独中 {tli_only} / 共同丢 {both_bad}")
        for kind, i, f, t in fails[:6]:
            print(f"  ({kind} #{i}) ans={f['ans'][:2]}")
            print(f"    FullKV out: {f['out'][:70]!r}")
            print(f"    TLI   out: {t['out'][:70]!r}")
    print()
