# E98：α×β×γ 三维全组合网格（用户指令 2026-10-03：三个旋钮全面扫描，先不管是否生效，要全面数据）
# 对照 /home/wangyuanshuo02/sglang/论文indexer.md 权威定义：
#   near_budget_token = near_budget_page_topk * page_size * gama（γ 折扣 near 细筛 token）
#   far_budget_token  = budget_token − near_budget_token（far 拿剩余）
#   扫描须去除明显不合理情况：near_L >= near_bp*BS >= near_token；far_budget <= far_L；
#   ab(1,0)/(0,1) 角点无意义（do_near/do_far 守卫，E64g 同款）
# method 组合（far, near）五组（论文indexer.md 列表全量）：
#   mavg=(minmax,avg)  mminmax=(minmax,minmax)  aavg=(avg,avg)
#   cavg=(cluster,avg)  ccluster=(cluster,cluster)
#   （E64g 只扫 far 侧 3 method + near 恒 avg；本实验 near 侧 method 参数化补齐）
# 口径：真实全维 softmax 行级 mass coverage（与 E64g 同）；γ 维 E64g 固定 1、E64b 只在冠军 bp 扫过，
#   本实验首次三维全扫。CPU 采集（论文indexer.md：mass 图 CPU 即可），多进程分样本。
import json
import os

import torch
import torch.nn.functional as F

import analyze_e64a_ab_grid as base

TRACE = base.TRACE
OUT = "/home/wangyuanshuo02/two-level-attention/exp/trace/results/e98_abg_full_grid.json"
D2I, BS, SINK, SWA, TAIL_N = base.D2I, base.BS, base.SINK, base.SWA, base.TAIL_N
B_TOK = 2048
BP = 64                     # 页池总量（E64a/g 同协议连续性）
GRID = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
# (combo 名, far method, near method)——method 键：mavg=minmax 粗筛 / aavg=avg 粗筛 / cavg=cluster
COMBOS = [
    ("mavg", "mavg", "aavg"),
    ("mminmax", "mavg", "mavg"),
    ("aavg", "aavg", "aavg"),
    ("cavg", "cavg", "aavg"),
    ("ccluster", "cavg", "cavg"),
]


def eval_layer(lf, device):
    d = torch.load(lf, map_location="cpu", weights_only=False)
    k, q, qpos, S = d["k"].to(device).float(), d["q"].to(device).float(), d["qpos"], d["S"]
    Hkv, H, D = k.shape[1], q.shape[1], k.shape[-1]
    G = H // Hkv
    pos = torch.arange(S, device=device)
    res = {}
    mid_hi_last = int(qpos[-1]) + 1 - SWA
    idx = torch.tensor(D2I, device=device)
    ksub_layer = k[..., idx]
    cent_layer, assign_mid_layer, _ = base.greedy_cluster_assign(ksub_layer[SINK:mid_hi_last])
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

        ksub = ksub_layer
        qsub = q[-TAIL_N + ri][..., idx].reshape(Hkv, G, 32).sum(1)
        nblk = (S + BS - 1) // BS
        kk = F.pad(ksub, (0, 0, 0, 0, 0, nblk * BS - S))
        kc = kk.reshape(nblk, BS, Hkv, 32)
        kmin, kmax, kavg = kc.amin(1), kc.amax(1), kc.mean(1)
        sc_mm = (torch.einsum("hd,nhd->hn", qsub.clamp(min=0), kmax) +
                 torch.einsum("hd,nhd->hn", qsub.clamp(max=0), kmin))
        sc_av = torch.einsum("hd,nhd->hn", qsub, kavg)
        sc_tok = torch.einsum("hgd,shd->hgs", q[-TAIL_N + ri][..., idx].reshape(Hkv, G, 32), ksub).sum(1)
        assign_mid = assign_mid_layer[:, :mid_len]
        cs = torch.einsum("hd,hkd->hk", qsub, cent_layer)
        tok_cl = cs.gather(1, assign_mid)
        blk_ids_mid = (torch.arange(mid_len, device=device) // BS).unsqueeze(0).expand(Hkv, -1)
        sc_cl_blk = torch.full((Hkv, (mid_len + BS - 1) // BS), float("-inf"), device=device)
        sc_cl_blk.scatter_reduce_(1, blk_ids_mid, tok_cl, reduce="amax", include_self=False)
        sc_blk = {"mavg": sc_mm[:, SINK // BS:], "aavg": sc_av[:, SINK // BS:], "cavg": sc_cl_blk}
        tok_score = {"mavg": sc_tok[:, SINK:mid_hi], "aavg": sc_tok[:, SINK:mid_hi], "cavg": tok_cl}
        sink_c = (pos < SINK).view(1, -1)
        swa_c = ((pos >= t_r + 1 - SWA) & (pos <= t_r)).view(1, -1)

        def select_sub(lo_off, hi_off, method, n_pages, n_tokens):
            if n_pages <= 0 or n_tokens <= 0:
                return torch.zeros(Hkv, S, dtype=torch.bool, device=device)
            s1 = sc_blk[method].masked_fill(
                ~(((torch.arange(sc_blk[method].shape[1], device=device) * BS + SINK) < SINK + hi_off) &
                  ((torch.arange(sc_blk[method].shape[1], device=device) * BS + BS - 1 + SINK) >= SINK + lo_off)).view(1, -1),
                float("-inf"))
            nblk_sub = s1.shape[-1]
            ib = torch.topk(s1, min(n_pages, nblk_sub), dim=-1).indices
            tok = (ib.unsqueeze(-1) * BS + torch.arange(BS, device=device).view(1, 1, BS)).reshape(Hkv, -1).clamp(max=mid_len - 1)
            pool = torch.zeros(Hkv, mid_len, dtype=torch.bool, device=device)
            pool.scatter_(1, tok, True)
            pool &= ((torch.arange(mid_len, device=device) >= lo_off) &
                     (torch.arange(mid_len, device=device) < hi_off)).view(1, -1)
            ts = tok_score[method].masked_fill(~pool, float("-inf"))
            it = torch.topk(ts, min(n_tokens, mid_len), dim=-1).indices
            cand = torch.zeros(Hkv, S, dtype=torch.bool, device=device)
            cand.scatter_(1, it + SINK, True)
            return cand

        for combo, fm, nm in COMBOS:
            for a in GRID:
                near_L = int(a * mid_len)
                far_L = mid_len - near_L
                near_lo_off, near_hi_off = mid_len - near_L, mid_len
                for b in GRID:
                    # 角点守卫（用户语义）：α=0/β=0 → near 空；α=1/β=1 → far 空
                    do_near = (a > 0) and (b > 0)
                    do_far = (a < 1) and (b < 1)
                    nb_near = int(round(BP * b)) if do_near else 0
                    # 约束过滤（论文indexer.md）：near_L >= near_bp*BS（near 区装得下页池）
                    if do_near and near_L < nb_near * BS:
                        continue
                    for g in GRID:
                        # γ 语义：nt_near = nb_near·BS·γ（≤ nb_near·BS 自动满足）
                        nt_near = min(int(nb_near * BS * g), B_TOK) if do_near else 0
                        nt_far = max(64, B_TOK - nt_near) if do_far else 0
                        # 约束过滤：far_budget <= far_L（far 区装得下 far token 预算）
                        if do_far and nt_far > far_L:
                            continue
                        cand = torch.zeros(Hkv, S, dtype=torch.bool, device=device)
                        cand |= sink_c | swa_c
                        if do_near:
                            cand |= select_sub(near_lo_off, near_hi_off, nm, nb_near, nt_near)
                        if do_far:
                            cand |= select_sub(0, near_lo_off, fm, BP - nb_near, nt_far)
                        res.setdefault(f"{combo}_a{a}_b{b}_g{g}", []).append(cov_mass(cand))
    del k, q, d, ksub
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return res


def run_sample(name, device):
    n_layers = json.load(open(f"{TRACE}/{name}/meta.json"))["n_layers"]
    agg = {}
    for li in range(0, n_layers, max(1, n_layers // 12)):
        r = eval_layer(f"{TRACE}/{name}/layer{li:02d}.pt", device)
        if not r:
            continue
        for k2, v in r.items():
            agg.setdefault(k2, []).extend(v)
    return {k2: sum(v) / len(v) for k2, v in agg.items()}


if __name__ == "__main__":
    import multiprocessing as mp

    proc_id = int(os.environ.get("E98_PROC", "0"))
    n_proc = int(os.environ.get("E98_NPROC", "16"))
    device = os.environ.get("E98_DEVICE", "cpu")
    torch.set_num_threads(1)
    names = sorted(n for n in os.listdir(TRACE)
                   if os.path.isfile(os.path.join(TRACE, n, "meta.json")))
    mine = [n for i, n in enumerate(names) if i % n_proc == proc_id]
    print(f"[E98 proc{proc_id}/{n_proc}] device={device} samples={mine}", flush=True)
    out = {}
    for name in mine:
        out[name] = run_sample(name, device)
        print(f"[E98 proc{proc_id}] done {name} arms={len(out[name])}", flush=True)
    json.dump(out, open(f"/tmp/e98_shard_{proc_id}.json", "w"))
    print(f"[E98 proc{proc_id}] shard saved", flush=True)
