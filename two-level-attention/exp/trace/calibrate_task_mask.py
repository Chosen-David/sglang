# D' per-task 校准：给定任务名，用其专属 trace（lb_<task>_{0,1}）生成层跳过掩码
# 用法：python calibrate_task_mask.py musique
# 输出：exp/trace/results/tli_layer_skip_mask_<task>.json
# 语义与 tli_layer_skip_mask.json 一致（τ=0.02，平均轮廓），仅 trace 范围限定为该任务
import os
import sys
import json
import glob
import statistics as st
import torch

sys.path.insert(0, os.path.dirname(__file__))
from analyze_e6 import layer_stats  # 复用同一 far-mass 计算（per-head 平均口径）

TRACE = "/tmp/trace/qwen3-8b"
OUT = "/home/wangyuanshuo02/two-level-attention/exp/trace/results"
THRESH = 0.02


def main():
    task = sys.argv[1]
    pdirs = [f"{TRACE}/lb_{task}_{i}" for i in range(2)]
    pdirs = [p for p in pdirs if os.path.exists(f"{p}/meta.json")]
    assert pdirs, f"no trace for {task}: {pdirs}"
    profiles = []
    for pdir in pdirs:
        rows = [layer_stats(lf) for lf in sorted(glob.glob(f"{pdir}/layer*.pt"))]
        profiles.append([f for f, _ in rows])
        torch.cuda.empty_cache()
    n = min(len(p) for p in profiles)
    avg = [st.mean([p[i] for p in profiles]) for i in range(n)]
    skip = [i for i in range(n) if avg[i] < THRESH]
    far_total = sum(avg)
    far_skipped = sum(avg[i] for i in skip)
    out = {
        "skip": skip, "threshold": THRESH, "task": task,
        "n_layers": n, "n_traces": len(profiles),
        "far_mass_total": far_total, "far_mass_skipped": far_skipped,
    }
    path = f"{OUT}/tli_layer_skip_mask_{task}.json"
    json.dump(out, open(path, "w"), indent=1)
    print(f"{task}: skip {len(skip)}/{n} layers, far mass skipped {far_skipped:.4f}/{far_total:.4f}")
    print("skip:", skip)
    print("saved", path)


if __name__ == "__main__":
    main()
