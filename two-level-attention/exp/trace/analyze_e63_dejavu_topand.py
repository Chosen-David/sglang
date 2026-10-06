# #63 DeJAVU 式 far 预测 + top& 短路信号（离线 trace 重放，8B/30B）
# 用户框架两项：
#   1. DeJAVU 式 far 预测：prefill 期（早期行）的 far top-K 选择能否预测
#      decode 期（末尾行）的 far top-K？度量 = 早期行 far top-256 集合对
#      末尾行 far oracle mass 的覆盖（IoU + mass recall）。DeJAVU 原文用
#      LSH/预测器，这里用最朴素口径定上界（直接拿早期真实选择当预测）。
#   2. top& 短路：sink 区（前 128 token）mass 高的 (layer,head) 其 far
#      mass 是否低（负相关 → sink 高时可短路 far）？度量 = 行级
#      sink_mass 与 far_mass 的 Pearson corr（per layer 聚合）。
# 口径：真实全维 softmax（causal）；far 区 = [128, t+1-near_len)，
# near_len=2048；末尾 TAIL_N 行 vs 早期行（t_r 的 1/2 处附近）。
import json
import os

import torch

TRACE_DIRS = ["/tmp/trace/qwen3-8b", "/tmp/trace/qwen3-30b"]
NEAR_LEN = 2048
SINK = 128
TAIL_N = 4
FAR_TOPK = 256
OUT = "/home/wangyuanshuo02/two-level-attention/exp/trace/results/e63_dejavu_topand.json"


def layer_stats(tr, layer_id, device):
    t = torch.load(tr + f"/layer{layer_id:02d}.pt", map_location="cpu", weights_only=False)
    k = t["k"].to(device).float()
    q = t["q"].to(device).float()
    qpos = t["qpos"]
    S = t["S"]
    H, Hkv, D = q.shape[1], k.shape[1], k.shape[2]
    G = H // Hkv
    k_e = k.repeat_interleave(G, dim=1)
    pos = torch.arange(S, device=device)

    def row_distr(qrow, t_row):
        sc = torch.einsum("hd,shd->hs", qrow, k_e) * (D**-0.5)
        sc = sc.masked_fill((pos > t_row).unsqueeze(0), float("-inf"))
        return torch.softmax(sc, dim=-1).view(Hkv, G, S)

    res = {}
    for ri in range(TAIL_N):
        t_r = int(qpos[-TAIL_N + ri])
        if t_r < NEAR_LEN + SINK + 1024:
            continue
        # 末尾行（decode 代理）
        p_t = row_distr(q[-TAIL_N + ri], t_r)
        far_hi = t_r + 1 - NEAR_LEN
        in_far = (pos >= SINK) & (pos < far_hi)
        pf = p_t * in_far.view(1, 1, S)
        sink_m = p_t[..., :SINK].sum(-1)  # [Hkv,G]
        far_m = pf.sum(-1)
        res.setdefault("sink_mass", []).extend(sink_m.flatten().tolist())
        res.setdefault("far_mass", []).extend(far_m.flatten().tolist())
        # far oracle top-256（末尾行）
        it_t = torch.topk(pf, FAR_TOPK, dim=-1).indices
        # 早期行（prefill 中段，t' ≈ t_r/2 附近取一条可用的）
        early_idx = None
        half = t_r // 2
        best = None
        for j in range(len(qpos)):
            if abs(int(qpos[j]) - half) < 2048 and int(qpos[j]) > NEAR_LEN + SINK + 1024:
                if best is None or abs(int(qpos[j]) - half) < best[0]:
                    best = (abs(int(qpos[j]) - half), j)
        if best is None:
            continue
        early_idx = best[1]
        t_e = int(qpos[early_idx])
        p_e = row_distr(q[early_idx], t_e)
        in_far_e = (pos >= SINK) & (pos < t_e + 1 - NEAR_LEN)
        pf_e = p_e * in_far_e.view(1, 1, S)
        it_e = torch.topk(pf_e, FAR_TOPK, dim=-1).indices
        # DeJAVU 式预测：早期 far top-256 集合覆盖末尾行 far oracle 的 mass
        pred = torch.zeros_like(pf, dtype=torch.bool)
        pred.scatter_(2, it_e, True)
        denom = torch.sort(pf, descending=True).values[..., :FAR_TOPK].sum(-1).clamp(min=1e-12)
        rec = (pf * pred).sum(-1) / denom
        res.setdefault("dejavu_recall", []).append(float(rec.mean()))
        # IoU（同口径集合）
        cur = torch.zeros_like(pf, dtype=torch.bool)
        cur.scatter_(2, it_t, True)
        inter = (pred & cur).sum(-1).float()
        union = (pred | cur).sum(-1).float().clamp(min=1)
        res.setdefault("dejavu_iou", []).append(float((inter / union).mean()))
    return res


def main():
    device = "cuda:0"
    out = {}
    for tdir in TRACE_DIRS:
        model = os.path.basename(tdir)
        if not os.path.isdir(tdir):
            continue
        for name in sorted(os.listdir(tdir)):
            tr = os.path.join(tdir, name)
            if not os.path.isfile(os.path.join(tr, "meta.json")):
                continue
            n_layers = json.load(open(tr + "/meta.json"))["n_layers"]
            agg = {}
            for li in range(0, n_layers, max(1, n_layers // 12)):
                r = layer_stats(tr, li, device)
                for k2, v in r.items():
                    agg.setdefault(k2, []).extend(v)
            sm = torch.tensor(agg["sink_mass"])
            fm = torch.tensor(agg["far_mass"])
            corr = float(torch.corrcoef(torch.stack([sm, fm]))[0, 1]) if len(sm) > 2 else float("nan")
            # top& 条件分布：sink_mass 十分位分桶 → far_mass 条件均值
            # （判据形态：sink 高分位行的 far 平均质量，若足够低则短路可行）
            qs = torch.quantile(sm, torch.linspace(0, 1, 11))
            bucket = torch.bucketize(sm, qs[1:-1])  # 0..9
            cond = [round(float(fm[bucket == b].mean()), 4) if (bucket == b).any() else None
                    for b in range(10)]
            rec = {
                "dejavu_recall": round(sum(agg["dejavu_recall"]) / len(agg["dejavu_recall"]), 4),
                "dejavu_iou": round(sum(agg["dejavu_iou"]) / len(agg["dejavu_iou"]), 4),
                "sink_far_corr": round(corr, 4),
                "sink_mass_mean": round(float(sm.mean()), 4),
                "far_mass_mean": round(float(fm.mean()), 4),
                "far_mass_by_sink_decile": cond,
                "far_dec10_vs_dec01_ratio": round(
                    cond[9] / cond[0], 3) if cond[0] and cond[0] > 0 else None,
            }
            out[f"{model}/{name}"] = rec
            print(f"[{model}/{name}] {rec}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
