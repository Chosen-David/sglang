# E64h：预算桥接实验——回答「热力图上 near+far 打不过 baseline（α=0 单池）」
# 口径错位假说：E64g 网格 B_TOK=2048（预算充裕，near 区大部分被 mono 池覆盖，
# 分区无价值）vs e2e K2=1024（预算紧张，near token 被 far 高分挤出，分区
# 「预算保护」价值显现——B7 hotpotqa 55.41 > C0 54.93 的机制）。
# 桥接：B_TOK=768 + GAMMA=0.25 精确匹配 e2e B7 严格口径语义
# （nt_near = nb_near·64·γ = 16·64·0.25 = 256，far = 512，与 K2_mid=768 一致）。
# 若 768 预算下分区臂反超 mono → 口径错位假说证实，热力图结论需加预算条件。
# CPU 跑（复用 e64g eval_layer，monkeypatch 模块常量）。
import json
import os

import torch

import analyze_e64g_full_grid as g

TRACE = g.TRACE
OUT = "/home/wangyuanshuo02/two-level-attention/exp/trace/results/e64h_budget768.json"
g.B_TOK = 768      # e2e 严格口径 mid 预算（K2=1024 − sink128 − swa128）
g.GAMMA = 0.25     # e2e B7 γ
g.GRID = [0.0, 0.125, 0.25]


def main():
    torch.set_num_threads(10)
    names = sorted(n for n in os.listdir(TRACE)
                   if os.path.isfile(os.path.join(TRACE, n, "meta.json")))
    results = {}
    for name in names:
        n_layers = json.load(open(f"{TRACE}/{name}/meta.json"))["n_layers"]
        agg = {}
        for li in range(0, n_layers, max(1, n_layers // 8)):
            r = g.eval_layer(f"{TRACE}/{name}/layer{li:02d}.pt", "cpu")
            if not r:
                continue
            for k2, v in r.items():
                agg.setdefault(k2, []).extend(v)
        rec = {k2: round(sum(v) / len(v), 4) for k2, v in agg.items()}
        results[name] = rec
        print(f"[{name}] " + " ".join(f"{k2}={v}" for k2, v in sorted(rec.items())), flush=True)
        json.dump(results, open(OUT, "w"), indent=1)
    # 总平均
    keys = sorted({k for rec in results.values() for k in rec})
    avg = {k: round(sum(rec[k] for rec in results.values() if k in rec)
                    / sum(1 for rec in results.values() if k in rec), 4) for k in keys}
    results["AVG"] = avg
    json.dump(results, open(OUT, "w"), indent=1)
    print("AVG:", json.dumps(avg))
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
