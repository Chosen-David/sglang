# 生成 best 臂 12 任务全量脚本；若 best=mavg(B7s) 则直接复用 B7s 输出免重跑
import json

meta = json.load(open("/tmp/tli_chain/screen_arms.json"))
best = open("/tmp/tli_chain/best_arm.txt").read().strip()
MODEL = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B"
GAMMA = 0.125
TASKS = ["hotpotqa", "2wikimqa", "musique", "passage_retrieval_en", "qasper",
         "multifieldqa_en", "gov_report", "qmsum", "multi_news", "narrativeqa",
         "triviaqa", "lcc", "repobench-p"]
ARM_FLAGS = {
    "mminmax": "--tli_far_method minmax --tli_near_method minmax",
    "mavg": "--tli_far_method minmax --tli_near_method avg",
    "aavg": "--tli_far_method avg --tli_near_method avg",
    "cavg": "--tli_far_method minmax --tli_near_method avg --tli_enable_kmeans true --tli_far_select cluster",
}

core = best.replace("(B7s)", "")
if best == "mavg(B7s)":
    # B7s 已有 12 任务全量输出 → 直接标记完成
    open("/tmp/tli_chain/FULL_DONE", "w").write("reuse B7s\n")
    print("best=mavg(B7s)：复用 B7s 全量输出，免重跑")
else:
    a, b = meta[core]["alpha"], meta[core]["beta"]
    TS = "0929e72f"
    # 双卡拆分（按 B7s 实测耗时均衡，~10h → ~5h）：
    #   GPU0 重载：narrativeqa(2.5h)+gov_report(1.7h)+qmsum+qasper+multifield+hotpotqa
    #   GPU1 重载：multi_news(2.4h)+triviaqa+lcc+repobench-p+musique+2wikimqa+passage
    SPLIT = {
        0: ["narrativeqa", "gov_report", "qmsum", "qasper", "multifieldqa_en", "hotpotqa"],
        1: ["multi_news", "triviaqa", "lcc", "repobench-p", "musique", "2wikimqa", "passage_retrieval_en"],
    }
    for gpu, tasks in SPLIT.items():
        lines = ["#!/bin/bash", "cd /home/wangyuanshuo02/two-level-attention"]
        for t in tasks:
            lines.append(
                f"echo '=== [full{gpu} {core} {t} $(date +%H:%M:%S)] ==='\n"
                f"CUDA_VISIBLE_DEVICES={gpu} python -u -m benchmark.LongBench.pred "
                f"--model Qwen3-8B --model_path {MODEL} --task {t} --method tli --e 0 --t {TS} "
                f"--dataset-path /home/wangyuanshuo02/datasets/LongBench/data "
                f"--config-path benchmark/LongBench/config "
                f"--output-dir exp/results_longbench/Qwen3-8B "
                f"--tia_level1_topk 128 --tia_level2_topk 1024 --tia_level2_cmp_ratio 4 "
                f"--tli_enable_layer_skip false --tli_alpha {a} --tli_beta {b} --tli_gamma {GAMMA} "
                f"{ARM_FLAGS[core]} --pred_postfix _e72f_{core} 2>&1 | tail -2")
        lines.append(f'echo "FULL GPU{gpu} DONE"')
        open(f"/tmp/tli_chain/run_full_gpu{gpu}.sh", "w").write("\n".join(lines) + "\n")
    print(f"best={core} a={a} b={b}：双卡全量脚本已生成（GPU0 {len(SPLIT[0])} + GPU1 {len(SPLIT[1])} 任务）")
