# E87：top-σ 混合方法扫描（用户 2026-10-02 指令，论文indexer.md §概述）
# 语义（用户定义）：far 或 near 区域不再两级筛选，直接细筛——
#   该区 token 的细筛得分（32 维子空间 qk 分）≥ sink token 的 qk 得分 max − σ 即选中。
# 臂（均含 sink/swa 强制保送，与 e64g 同协议）：
#   twolvl     : far=minmax 两级 + near=avg 两级（mavg 组合 α.125/β.25 基线，同批重算）
#   far_sigma  : far=top-σ, near=两级（混合）
#   near_sigma : far=两级, near=top-σ（混合）
#   mid_sigma  : 整个 mid 不分区直接 top-σ（far+near 同阈值 → 数学上=双 top-σ）
# 精度：真实全维 softmax 行级 mass coverage（e64g 同口径，可横向比）。
# 速度：成本模型 MACs（每 kv-head 每行乘加数）+ 平均选中 token 数（top-σ 预算可变）。
# σ 网格：{1,2,4,8,16,32}（32 维子空间点积量级），跑完选优。
# 双 GPU 分片：E87_SHARD=0/1 各跑 8 样本。
import json
import os

import torch
import torch.nn.functional as F

import analyze_e64a_ab_grid as base

TRACE = base.TRACE
D2I, BS, SINK, SWA, TAIL_N = base.D2I, base.BS, base.SINK, base.SWA, base.TAIL_N
B_TOK = 2048
BP = 64
GAMMA = 1.0
ALPHA, BETA = 0.125, 0.25          # mavg 部署臂（e64g 同款）
SIGMAS = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
OUT = "/home/wangyuanshuo02/two-level-attention/exp/trace/results/e87_topsigma_s{}.json"


def eval_layer(lf, device):
    d = torch.load(lf, map_location="cpu", weights_only=False)
    k, q, qpos, S = d["k"].to(device).float(), d["q"].to(device).float(), d["qpos"], d["S"]
    Hkv, H, D = k.shape[1], q.shape[1], k.shape[-1]
    G = H // Hkv
    pos = torch.arange(S, device=device)
    res = {}
    cost = {}   # 臂 -> [macs, avg_sel_tokens]
    mid_hi_last = int(qpos[-1]) + 1 - SWA
    idx = torch.tensor(D2I, device=device)
    ksub_layer = k[..., idx]
    for ri in range(TAIL_N):
        t_r = int(qpos[-TAIL_N + ri])
        mid_hi = t_r + 1 - SWA
        mid_len = mid_hi - SINK
        if mid_len < 4096:
            continue
        qg = q[-TAIL_N + ri].reshape(Hkv, G, D)
        k4 = k[:, :, None, :].expand(S, Hkv, G, D)
        s_full = torch.einsum("hgd,shgd->hgs", qg, k4) * (D ** -0.5)
        s_full = s_full.masked_fill((pos > t_r).view(1, 1, -1), float("-inf"))
        p_full = torch.softmax(s_full, dim=-1)

        def cov_mass(cand):
            return float((p_full * cand.unsqueeze(1)).sum(-1).mean())

        # ---- 共享打分（一次计算，所有臂复用） ----
        ksub = ksub_layer
        qsub = q[-TAIL_N + ri][..., idx].reshape(Hkv, G, 32).sum(1)
        nblk = (S + BS - 1) // BS
        kk = F.pad(ksub, (0, 0, 0, 0, 0, nblk * BS - S))
        kc = kk.reshape(nblk, BS, Hkv, 32)
        kmin, kmax, kavg = kc.amin(1), kc.amax(1), kc.mean(1)
        sc_mm = (torch.einsum("hd,nhd->hn", qsub.clamp(min=0), kmax) +
                 torch.einsum("hd,nhd->hn", qsub.clamp(max=0), kmin))
        sc_av = torch.einsum("hd,nhd->hn", qsub, kavg)
        # 32 维子空间 token 级细筛分（top-σ 的判据分数）
        sc_tok = torch.einsum("hgd,shd->hgs", q[-TAIL_N + ri][..., idx].reshape(Hkv, G, 32), ksub).sum(1)
        sink_c = (pos < SINK).view(1, -1)
        swa_c = ((pos >= t_r + 1 - SWA) & (pos <= t_r)).view(1, -1)

        # 两级选择子（e64g 同款：块 topk → token topk，区域裁剪）
        def select_sub(lo_off, hi_off, method, n_pages, n_tokens):
            s1 = {"mavg": sc_mm[:, SINK // BS:], "aavg": sc_av[:, SINK // BS:]}[method]
            nblk_mid = (mid_len + BS - 1) // BS
            blk_lo, blk_hi = lo_off // BS, (hi_off + BS - 1) // BS
            m = ((torch.arange(nblk_mid, device=device) >= blk_lo) &
                 (torch.arange(nblk_mid, device=device) < blk_hi)).view(1, -1)
            s1 = s1[:, :nblk_mid].masked_fill(~m, float("-inf"))
            ib = torch.topk(s1, min(n_pages, s1.shape[-1]), dim=-1).indices
            tok = (ib.unsqueeze(-1) * BS + torch.arange(BS, device=device).view(1, 1, BS)).reshape(Hkv, -1).clamp(max=mid_len - 1)
            pool = torch.zeros(Hkv, mid_len, dtype=torch.bool, device=device)
            pool.scatter_(1, tok, True)
            pool &= ((torch.arange(mid_len, device=device) >= lo_off) &
                     (torch.arange(mid_len, device=device) < hi_off)).view(1, -1)
            ts = sc_tok[:, SINK:SINK + mid_len].masked_fill(~pool, float("-inf"))
            it = torch.topk(ts, min(n_tokens, mid_len), dim=-1).indices
            cand = torch.zeros(Hkv, S, dtype=torch.bool, device=device)
            cand.scatter_(1, it + SINK, True)
            return cand, int(it.shape[-1])

        # top-σ 选择子：区域 [lo_off, hi_off) 内 sc_tok ≥ sink_max − σ
        def select_sigma(lo_off, hi_off, sigma):
            sc_reg = sc_tok[:, SINK + lo_off: SINK + hi_off]
            sink_max = sc_tok[:, :SINK].max(dim=-1).values          # [Hkv]
            keep = sc_reg >= (sink_max - sigma).unsqueeze(1)
            cand = torch.zeros(Hkv, S, dtype=torch.bool, device=device)
            pos_idx = torch.arange(lo_off, hi_off, device=device).view(1, -1).expand(Hkv, -1)
            cand.scatter_(1, (pos_idx * keep).masked_fill(~keep, 0) + SINK, keep)
            return cand, int(keep.sum(dim=-1).float().mean().item())

        # ---- 预算结构（mavg 部署臂） ----
        near_L = int(ALPHA * mid_len)
        nb_near = int(round(BP * BETA))
        nb_far = BP - nb_near
        nt_near = min(int(nb_near * BS * GAMMA), B_TOK)
        nt_far = max(64, B_TOK - nt_near)
        near_lo_off, near_hi_off = mid_len - near_L, mid_len
        far_hi_off = near_lo_off

        # ---- 臂 1：两级 baseline ----
        c_f, n_f = select_sub(0, far_hi_off, "mavg", nb_far, nt_far)
        c_n, n_n = select_sub(near_lo_off, near_hi_off, "aavg", nb_near, nt_near)
        cand = c_f | c_n | sink_c | swa_c
        res.setdefault("twolvl", []).append(cov_mass(cand))
        cost.setdefault("twolvl", []).append(
            [nblk * 32 + (n_f + n_n) * 32, float(c_f.sum(-1).float().mean() + c_n.sum(-1).float().mean())])

        # ---- 臂 2-4：σ 混合/纯 ----
        for sg in SIGMAS:
            c_fs, n_fs = select_sigma(0, far_hi_off, sg)            # far top-σ
            c_ns, n_ns = select_sigma(near_lo_off, near_hi_off, sg)  # near top-σ
            res.setdefault(f"far_sigma_{sg}", []).append(cov_mass(c_fs | c_n | sink_c | swa_c))
            cost.setdefault(f"far_sigma_{sg}", []).append(
                [far_hi_off * 32 + n_n * 32, float(c_fs.sum(-1).float().mean() + c_n.sum(-1).float().mean())])
            res.setdefault(f"near_sigma_{sg}", []).append(cov_mass(c_f | c_ns | sink_c | swa_c))
            cost.setdefault(f"near_sigma_{sg}", []).append(
                [nblk * 32 + n_f * 32 + near_L * 32, float(c_f.sum(-1).float().mean() + c_ns.sum(-1).float().mean())])
            res.setdefault(f"mid_sigma_{sg}", []).append(cov_mass(c_fs | c_ns | sink_c | swa_c))
            cost.setdefault(f"mid_sigma_{sg}", []).append(
                [mid_len * 32, float(c_fs.sum(-1).float().mean() + c_ns.sum(-1).float().mean())])
    del k, q, d, ksub
    torch.cuda.empty_cache()
    return res, cost


def main():
    shard = os.environ.get("E87_SHARD", "0")
    device = "cuda:0"
    names = sorted(n for n in os.listdir(TRACE)
                   if os.path.isfile(os.path.join(TRACE, n, "meta.json")))
    names = [n for i, n in enumerate(names) if i % 2 == int(shard)]
    results, costs = {}, {}
    for name in names:
        n_layers = json.load(open(f"{TRACE}/{name}/meta.json"))["n_layers"]
        agg, agg_c = {}, {}
        for li in range(0, n_layers, max(1, n_layers // 12)):
            out = eval_layer(f"{TRACE}/{name}/layer{li:02d}.pt", device)
            if not out:
                continue
            r, c = out
            for k2, v in r.items():
                agg.setdefault(k2, []).extend(v)
                for vv in c[k2]:
                    agg_c.setdefault(k2, []).append(vv)
        rec = {k2: round(sum(v) / len(v), 4) for k2, v in agg.items()}
        # cost: [mean_macs, mean_sel_tokens]
        rec_cost = {k2: [round(float(sum(x[0] for x in agg_c[k2]) / len(agg_c[k2])), 1),
                         round(float(sum(x[1] for x in agg_c[k2]) / len(agg_c[k2])), 1)]
                    for k2 in agg_c}
        results[name] = rec
        costs[name] = rec_cost
        print(f"[shard{shard} {name}] done {len(rec)} arms", flush=True)
    json.dump({"cov": results, "cost": costs},
              open(OUT.format(shard), "w"), indent=1)
    print("saved ->", OUT.format(shard))


if __name__ == "__main__":
    main()
