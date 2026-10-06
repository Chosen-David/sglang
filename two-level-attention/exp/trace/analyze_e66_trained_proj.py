# E66：训练投影矩阵——跨数据集泛化（用户指令：粗筛细筛都引入训练后降维，
#   训练出的投影在各个数据集上都可行，先调研再设计再写代码）
#
# ---- 调研结论（本地论文仓库 /tmp/papers + sglang/INDEXER_RESEARCH.md）----
# | 方法 | 出处 | 投影训练思想 | 本文借鉴点 |
# | DSA indexer | DeepSeek V3.2 | wq/wk 投影 + head gate，KL(attention‖softmax(I)) 蒸馏 | 分布对齐目标（KL）直接优化选择质量，优于 score 回归 MSE（主推臂） |
# | NSA | arXiv 2502.11089 | gate 分支在预训练内多任务联合 | 多样本混合训练（修 E65b 单样本单 q 短板） |
# | Linformer | arXiv 2006.04768 | 学习投影 E 使注意力低秩化 | 「学投影保注意力」先例（其压序列轴，本文压特征轴 32→d） |
# | ITQ | CVPR 2011 监督哈希 | 学旋转保检索序 | 监督投影保排序的经典形态（对照 sup_wsvd 闭式臂） |
# | SnapKV | NeurIPS 2024 | training-free 校准集统计 | M9 离线校准哲学同源：不动模型权重，只训插入式投影 |
#
# ---- 设计 ----
# 可学参数：P ∈ [Hkv, 32, d]，每层每 kv-head 独立（部署形态同 E65b）；
#           粗筛/细筛共享同一 P（打分同源 q·k，KL 目标天然对齐 token 级分数，
#           minmax/avg 块分是 token 分的上界/均值，投影可迁移）。
# 训练臂（三臂对比，隔离「目标函数」因子）：
#   kl_distill（DSA 式，主推）：KL(softmax(s32) ‖ softmax(s_d))，温度 1，
#     多任务多 q 联合（7 训练样本 × 末 4 q × far 区）
#   mse_multi：E65b sup_grad 升级——score 回归 MSE，从单样本单 q 扩多任务多 q
#   wsvd_multi：E65b sup_wsvd 升级——混合加权 PCA 闭式（多任务协方差求和）
# 泛化协议：LOTO leave-one-task-out 8 折——第 T 折训练时排除任务 T 的全部样本，
#   在 T 上评估 → 直接回答「训练出的投影在各数据集上是否可行」。
# 评估：A far 粗筛 / B near 粗筛 / C 细筛 / D 全链，d ∈ {4,8,16}；
#   口径与 E65/E65b 完全一致（尾维32 子空间内降维，per-head far mass 捕获）。
# 成功线：LOTO D_full d8 ≥ 0.78（trunc_d32 基线 0.804 的 97%），单任务 ≥ 0.70 不崩。
# 判决链：离线泛化成立 → GPU e2e 对拍判 decode q 漂移（E71-B 框架）；
#         离线不成立 → 「离线校准投影」路线整体 No-Go，与 E71-B 合并为 negative result。
import json
import os

import torch

import analyze_e65_dim_reduce as base

TRACE = base.TRACE
OUT = "/home/wangyuanshuo02/two-level-attention/exp/trace/results/e66_trained_proj.json"
D2I = base.D2I
SINK, SWA, NEAR_BAND = base.SINK, base.SWA, base.NEAR_BAND
DIMS = [4, 8, 16]
NQ_TRAIN = 4          # 每训练样本取末 4 个 q
N_ITER = 200          # 自适应温度下 ~100 iter 收敛（干跑 loss 0.0062@75）
MB = 2048             # DSA 式 mini-batch：每 iter 每 (样本,q) 采样 far token 数
LR = 3e-3


def load_far(lf):
    """载入一层 trace → far/near 区 k32 与末尾 q 列表（GQA 聚合）。"""
    d = torch.load(lf, map_location="cpu", weights_only=False)
    k, q, qpos, S = d["k"].float(), d["q"].float(), d["qpos"], d["S"]
    Hkv, H, D = k.shape[1], q.shape[1], k.shape[-1]
    G = H // Hkv
    t = int(qpos[-1])
    mid_hi = t + 1 - SWA
    far_lo, far_hi = SINK, mid_hi - NEAR_BAND
    near_lo, near_hi = mid_hi - NEAR_BAND, mid_hi
    if far_hi - far_lo < 8192:
        return None
    idx = torch.tensor(D2I)
    k32 = k[..., idx]                                   # [S,Hkv,32]
    qs = []
    for ri in range(max(NQ_TRAIN, base.TAIL_N)):
        q_head = q[-max(NQ_TRAIN, base.TAIL_N) + ri].reshape(Hkv, G, D).sum(1)
        qs.append(q_head[..., idx])                     # [Hkv,32]
    return dict(k32=k32, qs=qs, qpos=qpos, S=S, Hkv=Hkv,
                far_lo=far_lo, far_hi=far_hi, near_lo=near_lo, near_hi=near_hi)


def train_proj_layer(train_data, dim, method):
    """per (layer)：多任务联合训练 P [Hkv,32,dim]。method ∈ kl/mse/wsvd。"""
    if method == "wsvd":
        # 混合加权 PCA 闭式：C = Σ_样本 Σ_q Σ_far w_t·k kᵀ（w = far 区 softmax 概率）
        C = None
        for td in train_data:
            k32, Hkv = td["k32"], td["Hkv"]
            kf = k32[td["far_lo"]:td["far_hi"]]
            if C is None:
                C = torch.zeros(Hkv, 32, 32)
            for qi, q32 in enumerate(td["qs"][:NQ_TRAIN]):
                t = int(td["qpos"][-NQ_TRAIN + qi])
                s = torch.einsum("hd,shd->hs", q32, kf)
                s = s.masked_fill(torch.arange(kf.shape[0]).view(1, -1) > t - td["far_lo"],
                                  float("-inf"))
                p = torch.softmax(s * (128 ** -0.5), dim=-1)   # 与 E65b sup_wsvd 完全同温
                C += torch.einsum("ht,thd,the->hde", p, kf, kf)
        basis = []
        for h in range(C.shape[0]):
            eig, vec = torch.linalg.eigh(C[h])
            basis.append(vec[:, torch.argsort(eig, descending=True)])
        return torch.stack(basis)[:, :, :dim]           # [Hkv,32,dim]
    # 梯度训练（kl / mse 共用骨架，目标不同）
    # 速度修正（2026-09-29 监督轮发现）：全量 far 区×28 (样本,q) 对每 iter 太慢
    # （12 分钟仅 1 层 1 臂）→ DSA 式 mini-batch：每 iter 每 (样本,q) 随机采
    # MB 个 far token（覆盖无偏 + SGD 噪声正则）；iter 400→200（自适应温度
    # 下 ~100 iter 已收敛，干跑 loss 0.0062@75）
    Hkv = train_data[0]["Hkv"]
    W = torch.randn(Hkv, 32, dim, generator=torch.Generator().manual_seed(0)) * 0.05
    W.requires_grad_(True)
    opt = torch.optim.Adam([W], lr=LR)
    gen = torch.Generator().manual_seed(123)
    for it in range(N_ITER):
        loss = 0.0
        for td in train_data:
            kf_full = td["k32"][td["far_lo"]:td["far_hi"]]   # [Tfar,Hkv,32]
            Tfar = kf_full.shape[0]
            sel = torch.randint(0, Tfar, (min(MB, Tfar),), generator=gen)
            kf = kf_full[sel]
            for qi, q32 in enumerate(td["qs"][:NQ_TRAIN]):
                t = int(td["qpos"][-NQ_TRAIN + qi])
                # 采样 token 的原始 far 区偏移 = sel；因果重判 sel > t - far_lo 即不可见
                # （far 上界 < t-SWA-NEAR_BAND，仅早 q 部分不可见）
                causal = (sel.view(1, -1) > t - td["far_lo"])
                s32 = torch.einsum("hd,shd->hs", q32, kf).masked_fill(causal, float("-inf"))
                qp = torch.einsum("hd,hde->he", q32, W)
                kp = torch.einsum("shd,hde->she", kf, W)
                sd = torch.einsum("he,she->hs", qp, kp).masked_fill(causal, float("-inf"))
                if method == "kl":
                    # DSA 式分布蒸馏：目标=32 维子空间分数分布，预测=低维分数分布。
                    # 干跑发现的关键修正：原始点积分数量级 ±300（std~243），温度 1 的
                    # softmax 完全饱和成 argmax 监督（loss 假性收敛但 top64 IoU 仅 0.141）。
                    # → 自适应温度 = per-head 分数 std（尺度校准，分布覆盖 top-k 段，
                    #   与 DSA 训练后 indexer 分数尺度合理的实践对齐）。
                    with torch.no_grad():
                        temp = s32.std(dim=-1, keepdim=True).clamp_min(1.0)   # [Hkv,1]
                        p32 = torch.softmax(s32 / temp, dim=-1)
                    logpd = torch.log_softmax(sd / temp, dim=-1)
                    loss = loss + (p32 * (p32.clamp_min(1e-9).log() - logpd)).sum(-1).mean()
                else:  # mse（E65b sup_grad 多样本升级）
                    loss = loss + ((sd - s32) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return W.detach()


def eval_layer(lf, Ws):
    """held-out 样本评估：A/B/C/D 四臂 × {kl,mse,wsvd}×d（复用 E65b 口径）。"""
    td = load_far(lf)
    if td is None:
        return None
    k32, qs, Hkv = td["k32"], td["qs"], td["Hkv"]
    far_lo, far_hi = td["far_lo"], td["far_hi"]
    near_lo, near_hi = td["near_lo"], td["near_hi"]
    # 全维（128）softmax 参考分布——mass 捕获分母（E4c per-head 口径）
    d0 = torch.load(lf, map_location="cpu", weights_only=False)
    k_full = d0["k"].float()
    G = d0["q"].shape[1] // Hkv
    res = {}
    for ri in range(base.TAIL_N):
        q_full = d0["q"][-base.TAIL_N + ri].reshape(Hkv, G, 128).sum(1).float()
        t = int(td["qpos"][-base.TAIL_N + ri])
        s = torch.einsum("hd,shd->hs", q_full, k_full) * (128 ** -0.5)
        s = s.masked_fill(torch.arange(td["S"]).view(1, -1) > t, float("-inf"))
        p = torch.softmax(s, dim=-1)
        far_h = p[:, far_lo:far_hi].sum(-1)
        near_h = p[:, near_lo:near_hi].sum(-1)
        tot_f, tot_n = float(far_h.sum()), float(near_h.sum())
        if tot_f < 1e-6 or tot_n < 1e-6:
            continue
        q32 = qs[ri]
        k_t32_far, k_t32_near = k32[far_lo:far_hi], k32[near_lo:near_hi]
        for meth, Wd in Ws.items():
            for dim in DIMS:
                W = Wd[dim]
                kf_far = torch.einsum("shd,hde->she", k_t32_far, W)
                qf = torch.einsum("hd,hde->he", q32, W)
                kf_near = torch.einsum("shd,hde->she", k_t32_near, W)
                toks = base.select_region(kf_far, qf, base.blk_minmax, k_t32_far, q32)
                cap = sum(float(p[h, toks[h] + far_lo].sum()) for h in range(Hkv))
                res.setdefault(f"A_far_coarse_{meth}_d{dim}", []).append(cap / tot_f)
                toks = base.select_region(kf_near, qf, base.blk_avg, k_t32_near, q32)
                cap = sum(float(p[h, toks[h] + near_lo].sum()) for h in range(Hkv))
                res.setdefault(f"B_near_coarse_{meth}_d{dim}", []).append(cap / tot_n)
                toks = base.select_region(k_t32_far, q32, base.blk_minmax, kf_far, qf)
                cap = sum(float(p[h, toks[h] + far_lo].sum()) for h in range(Hkv))
                res.setdefault(f"C_fine_{meth}_d{dim}", []).append(cap / tot_f)
                toks = base.select_region(kf_far, qf, base.blk_minmax, kf_far, qf)
                cap = sum(float(p[h, toks[h] + far_lo].sum()) for h in range(Hkv))
                res.setdefault(f"D_full_{meth}_d{dim}", []).append(cap / tot_f)
    return res


def main():
    torch.set_num_threads(10)
    SAMPLES = base.SAMPLES
    layers_of = {}
    for name in SAMPLES:
        if os.path.isfile(f"{TRACE}/{name}/meta.json"):
            layers_of[name] = list(range(0, 36, 5))    # 8 层抽样（与 E65b 一致）
    results = {}
    for fold, held in enumerate(SAMPLES):              # LOTO 8 折
        if held not in layers_of:
            continue
        train_names = [n for n in layers_of if n != held]
        print(f"=== LOTO fold {fold}: held-out={held}, train={len(train_names)} ===", flush=True)
        fold_res = {}
        for li in layers_of[held]:
            train_data, eval_layer_file = [], f"{TRACE}/{held}/layer{li:02d}.pt"
            for tn in train_names:
                td = load_far(f"{TRACE}/{tn}/layer{li:02d}.pt")
                if td is not None:
                    train_data.append(td)
            if not train_data:
                continue
            Ws = {}
            for meth in ("kl", "mse", "wsvd"):
                Wd = {}
                for dim in DIMS:
                    Wd[dim] = train_proj_layer(train_data, dim, meth)
                Ws[meth] = Wd
                print(f"  layer{li:02d} {meth}: " + " ".join(
                    f"d{dim}=" + "|".join(f"{Wd[dim][h][:, 0].norm():.2f}" for h in range(0, 1))
                    for dim in DIMS), flush=True)
            r = eval_layer(eval_layer_file, Ws)
            if not r:
                continue
            for k2, v in r.items():
                fold_res.setdefault(k2, []).extend(v)
        rec = {k2: round(sum(v) / len(v), 4) for k2, v in fold_res.items()}
        results[held] = rec
        print(f"[{held}] " + " ".join(f"{k2}={v}" for k2, v in sorted(rec.items())), flush=True)
        json.dump(results, open(OUT, "w"), indent=1)
    # LOTO 总平均（跨数据集泛化主指标）
    keys = sorted({k for rec in results.values() for k in rec})
    avg = {k: round(sum(rec[k] for rec in results.values() if k in rec)
                    / sum(1 for rec in results.values() if k in rec), 4) for k in keys}
    results["LOTO_AVG"] = avg
    json.dump(results, open(OUT, "w"), indent=1)
    print("LOTO_AVG:", json.dumps(avg))
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
