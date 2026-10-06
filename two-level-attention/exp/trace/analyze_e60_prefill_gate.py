# #60 D' 升级离线验证：prefill 动态 gate 可行性。
# 科学问题：prefill 阶段观测的 per-layer far mass 能否预测 decode 阶段的
# per-layer far mass（跨层 corr）？若 corr 高 → 「prefill 测层、decode 跳 far」
# 无泛化假设，规避 E5b 离线 gate 跨任务失败（musique/qasper 掉 4.8-5.9 分）。
#
# trace 结构：k [S, Hkv, 128] bf16 + q [272, H, 128] + qpos [272]（均匀中段
# 采样 ~1045 间隔 + 末尾 5 个连续位置 = decode 视角近似）。
# far 区定义与 TLI 运行时一致：[far_lo=128, t+1-near_len=2048)。
# 口径：per-layer far mass = 逐 q-head far mass 的平均（跳层决策粒度）。
# prefill 观测三口径：中段均值 / 中段 max / 中段 P95；
# decode 观测：末 5 连续位置均值。另报「max 口径跳层 → decode 漏检率」。
import json
import os
import sys

import torch

FAR_LO = 128       # sink_blocks(2) * block_size(64)
NEAR_LEN = 2048
TAIL_N = 5         # 末尾连续位置数 = decode 视角
TRACE_DIRS = [
    "/tmp/trace/qwen3-8b",
    "/tmp/trace/qwen3-32b",
]
OUT = "/home/wangyuanshuo02/two-level-attention/exp/trace/results/e60_prefill_dynamic_gate.json"


def layer_far_mass(tr, layer_id, device):
    t = torch.load(tr + f"/layer{layer_id:02d}.pt", map_location="cpu", weights_only=False)
    k = t["k"].to(device).float()          # [S, Hkv, D]
    q = t["q"].to(device).float()          # [Nq, H, D]
    qpos = t["qpos"]
    S = t["S"]
    H, Hkv, D = q.shape[1], k.shape[1], k.shape[2]
    G = H // Hkv
    # GQA：q head i 对 kv head i//G —— k 扩到 q head 口径（head 顺序
    # 与 rotate_half 布局一致：i = h*G+g 归 kv head h）
    k_e = k.repeat_interleave(G, dim=1)                      # [S, H, D]
    # 逐 q-pos far mass：softmax 分布在 [far_lo, far_hi) 的质量和
    # （逐 q-head 算，最后平均）
    scores = torch.einsum("nhd,shd->nhs", q, k_e) * (D**-0.5)  # [Nq, H, S]
    pos = torch.arange(S, device=device)
    qpos_d = qpos.to(device)
    # 因果 mask：pos <= qpos
    causal = pos.view(1, 1, S) <= qpos_d.view(-1, 1, 1)
    scores = scores.masked_fill(~causal, float("-inf"))
    probs = torch.softmax(scores, dim=-1)                    # [Nq, H, S]
    far_hi = (qpos_d + 1 - NEAR_LEN).clamp(min=FAR_LO)
    in_far = (pos.view(1, 1, S) >= FAR_LO) & (pos.view(1, 1, S) < far_hi.view(-1, 1, 1))
    fm = (probs * in_far).sum(-1)                            # [Nq, H] per q-head far mass
    return fm.mean(-1).cpu(), qpos                           # [Nq] 层平均 far mass


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    results = {}
    for tdir in TRACE_DIRS:
        if not os.path.isdir(tdir):
            continue
        model = os.path.basename(tdir)
        for name in sorted(os.listdir(tdir)):
            tr = os.path.join(tdir, name)
            meta_f = os.path.join(tr, "meta.json")
            if not os.path.isfile(meta_f):
                continue
            meta = json.load(open(meta_f))
            n_layers = meta["n_layers"]
            # 末尾 5 连续位置索引（qpos 升序，最后 TAIL_N 个）
            t0 = torch.load(tr + "/layer00.pt", map_location="cpu", weights_only=False)
            qpos = t0["qpos"]
            S = t0["S"]
            tail_idx = set(qpos[-TAIL_N:].tolist())
            mid_mask = torch.tensor(
                [int(p) not in tail_idx and int(p) < S - NEAR_LEN for p in qpos]
            )
            pre_mean, pre_max, pre_p95, dec = [], [], [], []
            for li in range(n_layers):
                fm, _ = layer_far_mass(tr, li, device)
                mid = fm[mid_mask]
                tail = fm[~mid_mask]
                pre_mean.append(float(mid.mean()))
                pre_max.append(float(mid.max()))
                pre_p95.append(float(torch.quantile(mid, 0.95)))
                dec.append(float(tail.mean()))
            def corr(a, b):
                ta, tb = torch.tensor(a), torch.tensor(b)
                return float(torch.corrcoef(torch.stack([ta, tb]))[0, 1])
            rec = {
                "S": S, "n_layers": n_layers,
                "corr_mean": round(corr(pre_mean, dec), 4),
                "corr_max": round(corr(pre_max, dec), 4),
                "corr_p95": round(corr(pre_p95, dec), 4),
                "dec_far_top": sorted(range(n_layers), key=lambda i: -dec[i])[:8],
            }
            # 跳层漏检率：max 口径 TH=0.01 跳过的层中，decode far mass > 0.05 的
            TH_SKIP, TH_MISS = 0.01, 0.05
            skip = [i for i in range(n_layers) if pre_max[i] < TH_SKIP]
            missed = [i for i in skip if dec[i] > TH_MISS]
            rec["n_skip"] = len(skip)
            rec["n_missed"] = len(missed)
            results[f"{model}/{name}"] = rec
            print(f"[{model}/{name}] S={S} corr(mean/max/p95)="
                  f"{rec['corr_mean']}/{rec['corr_max']}/{rec['corr_p95']} "
                  f"skip={len(skip)} missed={len(missed)}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    old = {}
    if os.path.exists(OUT):
        old = json.load(open(OUT))
    old.update(results)
    json.dump(old, open(OUT, "w"), indent=1)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
