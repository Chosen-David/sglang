# 生成 e2e 五组合快筛脚本：每臂 hotpotqa+musique（判别力最强对：far 检索 + 最高 F1）
# mavg 若 E64j 最佳=α.125/β.25 则复用 B7s 已有输出（不重跑）；ccluster e2e 无 near-cluster 参数化，trace-only 报告
import json

best = json.load(open("/tmp/tli_chain/e64j_best.json"))
TS = "0929e72"
MODEL = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B"
GAMMA = 0.125

# trace 组合名 → (e2e far_method/far_select, near_method, 额外 flags)
ARMS = {
    "mminmax": ("--tli_far_method minmax --tli_near_method minmax", ""),
    "mavg":    ("--tli_far_method minmax --tli_near_method avg", ""),
    "aavg":    ("--tli_far_method avg --tli_near_method avg", ""),
    "cavg":    ("--tli_far_method minmax --tli_near_method avg",
                "--tli_enable_kmeans true --tli_far_select cluster"),
}
# trace PAIRS 键 → 本臂 αβ 来源
SRC = {"mminmax": "mavg+mavg", "mavg": "mavg+aavg", "aavg": "aavg+aavg", "cavg": "cavg+aavg"}

lines = ["#!/bin/bash",
         "cd /home/wangyuanshuo02/two-level-attention",
         "set -e"]
meta = {}
for arm, (mflags, extra) in ARMS.items():
    src = SRC[arm]
    if src not in best:
        continue
    a, b = best[src]["alpha"], best[src]["beta"]
    meta[arm] = {"alpha": a, "beta": b, "src": src, "trace_mass": best[src]["mass"]}
    if arm == "mavg" and (a, b) == (0.125, 0.25):
        meta[arm]["reuse_b7s"] = True
        continue  # 复用 pred_b7
    lines.append(f"echo '=== [screen {arm} a={a} b={b} $(date +%H:%M:%S)] ==='")
    for task in ("hotpotqa", "musique"):
        lines.append(
            f"CUDA_VISIBLE_DEVICES=0 python -u -m benchmark.LongBench.pred "
            f"--model Qwen3-8B --model_path {MODEL} --task {task} --method tli --e 0 --t {TS} "
            f"--dataset-path /home/wangyuanshuo02/datasets/LongBench/data "
            f"--config-path benchmark/LongBench/config "
            f"--output-dir exp/results_longbench/Qwen3-8B "
            f"--tia_level1_topk 128 --tia_level2_topk 1024 --tia_level2_cmp_ratio 4 "
            f"--tli_enable_layer_skip false --tli_alpha {a} --tli_beta {b} --tli_gamma {GAMMA} "
            f"{mflags} {extra} --pred_postfix _e72_{arm} 2>&1 | tail -2")
lines.append('echo "SCREEN ALL DONE"')
open("/tmp/tli_chain/run_screen.sh", "w").write("\n".join(lines) + "\n")
json.dump(meta, open("/tmp/tli_chain/screen_arms.json", "w"), indent=1)
print("arms:", {k: (v["alpha"], v["beta"], v.get("reuse_b7s", False)) for k, v in meta.items()})
