"""E103 kv-head 共享消融：打分 → 落袋 exp/trace/results/e103_kvhead_ablation.json

臂定义（全部 mavg α=0.125 β=0.375 γ=0.625 K1=128 cmp_ratio=4，Qwen3-8B，GQA G=4）：
  A 共享（论文 §2.2 口径，不重跑）：K2=1024，mask [Hkv,T] 组内广播
  B per-q-head 同 K2：K2=1024，每 q-head 独立 1024（索引量 ×4，计算量持平）
  C per-q-head 同总量：K2=256，32×256=8192 token·head = 共享臂 8×1024
    （结构事实：K2=256 时 K2_mid = 256 − sink 128 − swa 128 = 0，
     mid 预算被固定区吃满——共享摊薄固定开销的优势本身就是消融对象）

baseline 取 e98_full_13tasks.json（A 臂与 E98 best 选举臂同配置）。
"""
import glob
import json
import sys

sys.path.insert(0, "/home/wangyuanshuo02/two-level-attention")
from benchmark.LongBench.eval import scorer

TASKS = ["hotpotqa", "musique"]
RES = "/home/wangyuanshuo02/two-level-attention/exp/trace/results"

# ---- A 臂 baseline（E98 best 选举臂，同配置不重跑）----
e98 = json.load(open(f"{RES}/e98_full_13tasks.json"))
A = {"hotpotqa": e98["tasks"]["hotpotqa"], "musique": e98["tasks"]["musique"]}

def score_arm(postfix):
    out = {}
    for task in TASKS:
        fs = sorted(glob.glob(f"/tmp/e103_perqhead/pred{postfix}/{task}-tli_*.jsonl"))
        fs = [f for f in fs if sum(1 for _ in open(f)) >= 200]
        if not fs:
            out[task] = None
            continue
        preds, answers, allc = [], [], None
        for line in open(fs[-1]):
            j = json.loads(line)
            preds.append(j["pred"])
            answers.append(j["answers"])
            allc = j["all_classes"]
        out[task] = round(scorer(task, preds, answers, allc), 2)
    return out

B = score_arm("_B")
C = score_arm("_C")
print("A(共享):", A)
print("B(per-q-head K2=1024):", B)
print("C(per-q-head K2=256):", C)

verdict_b = {t: (round(B[t] - A[t], 2) if B[t] is not None else None) for t in TASKS}
verdict_c = {t: (round(C[t] - A[t], 2) if C[t] is not None else None) for t in TASKS}

out = {
    "note": "E103 kv-head 共享消融（审稿 MAJOR：§2.2 GQA 组内共享 vs per-q-head 独立选择）",
    "protocol": {
        "model": "Qwen3-8B", "gqa": "32 q-head -> 8 kv-head, group_size G=4",
        "config": "mavg (far=minmax, near=avg), alpha=0.125 beta=0.375 gamma=0.625, "
                  "K1=128, cmp_ratio=4, subspace=full, kmeans/layer_skip off",
        "arms": {
            "A_shared": "K2=1024, mask [Hkv,T] 组内广播（论文 §2.2 口径，不重跑，"
                        "取 e98_full_13tasks.json best 选举臂同配置数字）",
            "B_per_q_head_same_K2": "K2=1024, --tli_per_q_head true（每 q-head 独立 "
                                    "1024；索引量 ×4、计算量与 A 持平——测共享的质量损失）",
            "C_per_q_head_same_total": "K2=256, --tli_per_q_head true（32x256=8192 "
                                       "token-head = 共享臂 8x1024 索引条目持平——同预算公平对照）",
        },
        "caveat": "臂 C 结构事实：K2=256 时 K2_mid = 256 - sink(128) - swa(128) = 0，"
                  "mid 预算被固定区吃满（sink/swa per-head 化后固定开销 ×G）；"
                  "共享口径把固定区摊薄到 kv-head 级（8x256 vs 32x256），"
                  "该摊薄优势是共享结构收益的一部分，如实计入对照语义",
        "tasks": "hotpotqa / musique, LongBench n=200",
    },
    "A_shared": A,
    "B_per_q_head_same_K2": B,
    "C_per_q_head_same_total": C,
    "delta_B_minus_A": verdict_b,
    "delta_C_minus_A": verdict_c,
}
json.dump(out, open(f"{RES}/e103_kvhead_ablation.json", "w"), indent=1, ensure_ascii=False)
print(f"saved {RES}/e103_kvhead_ablation.json")
