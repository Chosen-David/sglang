# E64i-tight：E64i 同/异 method 分区对照的紧预算复测（e2e 严格口径语义）
# E64h 已证明预算口径可翻转 mono/分区结论（宽 2048 vs 紧 768）→
#   用户问题「不同 method 分区是否打得过同 method 分区」必须在两种预算下都答。
# 协议：monkeypatch 复用 e64i 全部逻辑，仅 B_TOK=768（K2=1024−sink128−swa128
#   的 mid 预算）、GAMMA=0.25（e2e B7 γ：nt_near=16·64·0.25=256）。
import json
import os

import torch

import analyze_e64i_same_method as g

g.B_TOK = 768
g.GAMMA = 0.25
g.OUT = "/home/wangyuanshuo02/two-level-attention/exp/trace/results/e64i_tight.json"


def main():
    torch.set_num_threads(10)
    names = sorted(n for n in os.listdir(g.TRACE)
                   if os.path.isfile(os.path.join(g.TRACE, n, "meta.json")))
    results = {}
    for name in names:
        n_layers = json.load(open(f"{g.TRACE}/{name}/meta.json"))["n_layers"]
        agg = {}
        for li in range(0, n_layers, max(1, n_layers // 8)):
            r = g.eval_layer(f"{g.TRACE}/{name}/layer{li:02d}.pt")
            if not r:
                continue
            for k2, v in r.items():
                agg.setdefault(k2, []).extend(v)
        rec = {k2: round(sum(v) / len(v), 4) for k2, v in agg.items()}
        results[name] = rec
        print(f"[{name}] " + " ".join(f"{k2}={v}" for k2, v in sorted(rec.items())), flush=True)
        json.dump(results, open(g.OUT, "w"), indent=1)
    keys = sorted({k for rec in results.values() for k in rec})
    avg = {k: round(sum(rec[k] for rec in results.values() if k in rec)
                    / sum(1 for rec in results.values() if k in rec), 4) for k in keys}
    results["AVG"] = avg
    json.dump(results, open(g.OUT, "w"), indent=1)
    print("\nAVG:")
    print(json.dumps(avg, indent=1))
    print("saved ->", g.OUT)


if __name__ == "__main__":
    main()
