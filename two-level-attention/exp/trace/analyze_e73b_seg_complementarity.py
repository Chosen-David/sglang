# E73b：tail32 两段互补性机制分解数据落袋（为终版 Fig2 降维故事图提供可回溯 JSON）
# 报告 §8b-48 已有数字（0.5759/0.3826/拼接 0.7958）但当时未存 JSON——
# 本脚本用 E76 同框架重导出：far 全链 minmax（粗筛+细筛同特征），
# D2I 三变体：rope 低频尾16（48-64）/ nope 尾16（112-128）/ tail32 拼接。
# 真值口径 = 全维 softmax per-head far mass 加权捕获（E4c 同款）。
import json
import os

import torch

TRACE = "/tmp/trace/qwen3-8b"
OUT = "/home/wangyuanshuo02/two-level-attention/exp/trace/results/e73b_seg_complementarity.json"
SINK, SWA, NEAR_BAND = 128, 1024, 4096
BUD_TOK = 512
N_PAGES = 16
BS = 64
TAIL_N = 2
SEGS = {
    "rope_tail16": list(range(48, 64)),
    "nope_tail16": list(range(112, 128)),
    "tail32": list(range(48, 64)) + list(range(112, 128)),
    "random32": [17, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83,
                 89, 97, 101, 103, 107, 109, 113, 127, 3, 7, 11, 13, 19, 2, 5, 1],
}
SAMPLES = ["lb_gov_report_0", "lb_hotpotqa_0", "lb_musique_0", "lb_narrativeqa_0",
           "lb_passage_retrieval_en_0", "lb_qasper_0", "needle32k", "natural32k"]


def select_region(coarse_kf, q_coarse, fine_kf, q_fine):
    T, Hkv, _ = coarse_kf.shape
    nblk = (T + BS - 1) // BS
    kk = torch.nn.functional.pad(coarse_kf, (0, 0, 0, 0, 0, nblk * BS - T))
    kc = kk.reshape(nblk, BS, Hkv, -1)
    kmin, kmax = kc.amin(1), kc.amax(1)
    sc = (torch.einsum("hd,nhd->hn", q_coarse.clamp(min=0), kmax) +
          torch.einsum("hd,nhd->hn", q_coarse.clamp(max=0), kmin))
    ib = torch.topk(sc, min(N_PAGES, nblk), dim=-1).indices
    tok = (ib.unsqueeze(-1) * BS + torch.arange(BS).view(1, 1, BS)).reshape(Hkv, -1).clamp(max=T - 1)
    pool = torch.zeros(Hkv, T, dtype=torch.bool)
    pool.scatter_(1, tok, True)
    ts = torch.einsum("hd,shd->hs", q_fine, fine_kf).masked_fill(~pool, float("-inf"))
    it = torch.topk(ts, min(BUD_TOK, T), dim=-1).indices
    return it


def eval_layer(lf):
    d = torch.load(lf, map_location="cpu", weights_only=False)
    k, q, qpos, S = d["k"].float(), d["q"].float(), d["qpos"], d["S"]
    Hkv, H, D = k.shape[1], q.shape[1], k.shape[-1]
    G = H // Hkv
    res = {}
    t = int(qpos[-1])
    mid_hi = t + 1 - SWA
    far_lo, far_hi = SINK, mid_hi - NEAR_BAND
    if far_hi - far_lo < 8192:
        del k, q, d
        return None
    for ri in range(TAIL_N):
        q_t = q[-TAIL_N + ri]
        q_head = q_t.reshape(Hkv, G, D).sum(1)
        s = torch.einsum("hd,shd->hs", q_head, k) * (D ** -0.5)
        s = s.masked_fill(torch.arange(S).view(1, -1) > int(qpos[-TAIL_N + ri]), float("-inf"))
        p = torch.softmax(s, dim=-1)
        far_h = p[:, far_lo:far_hi].sum(-1)
        tot_f = float(far_h.sum())
        if tot_f < 1e-6:
            continue
        for seg, idx in SEGS.items():
            idx_t = torch.tensor(idx)
            kf = k[far_lo:far_hi][..., idx_t]
            qf = q_head[..., idx_t]
            it = select_region(kf, qf, kf, qf)
            cap = sum(float(p[h, it[h] + far_lo].sum()) for h in range(Hkv))
            res.setdefault(seg, []).append(cap / tot_f)
        # oracle128
        s_far = s[:, far_lo:far_hi]
        order = torch.argsort(s_far, dim=-1, descending=True)
        cap = sum(float(p[h, order[h][:BUD_TOK] + far_lo].sum()) for h in range(Hkv))
        res.setdefault("oracle128", []).append(cap / tot_f)
    del k, q, d
    return res


def main():
    torch.set_num_threads(16)
    results = {}
    for name in SAMPLES:
        meta = f"{TRACE}/{name}/meta.json"
        if not os.path.isfile(meta):
            continue
        n_layers = json.load(open(meta))["n_layers"]
        agg = {}
        for li in range(0, n_layers, max(1, n_layers // 8)):
            r = eval_layer(f"{TRACE}/{name}/layer{li:02d}.pt")
            if not r:
                continue
            for k2, v in r.items():
                agg.setdefault(k2, []).extend(v)
        rec = {k2: round(sum(v) / len(v), 4) for k2, v in agg.items()}
        results[name] = rec
        print(f"[{name}] " + " ".join(f"{k2}={v}" for k2, v in sorted(rec.items())), flush=True)
    mean = {k2: round(sum(v[k2] for v in results.values()) / len(results), 4) for k2 in results[next(iter(results))]}
    results["MEAN8"] = mean
    print(f"[MEAN8] {mean}")
    json.dump(results, open(OUT, "w"), indent=1)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
