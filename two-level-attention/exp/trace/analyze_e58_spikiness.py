# #58 30B 崩坏根因的尖峰假说离线验证：
# 30B (G=8) vs 8B (G=4) 的 per-(layer, kv-head, q-head) 注意力分布形态：
# top-1/top-8/top-1024 mass 占比（causal softmax over [0, t]）。
# 假说：30B 的远端分布尖峰化（top-8 mass 高）→ 均匀采样丢尖峰即崩；
# 8B 的 mass 集中于近端/sink（top-1024 φ≈0.99）→ 低预算稳。
# 顺带测：均匀 K 采样的期望 mass 捕获（1024/8192）——对应 e2e 预算扫描。
import json
import os

import torch

FAR_LO = 128
NEAR_LEN = 2048
TAIL_N = 8  # 末尾连续位置（近似 decode 视角；这里用末尾行测 prefill 末段）
KS = [1, 8, 64, 1024]
SAMPLE_KS = [1024, 4096, 8192]
TRACE_DIRS = ["/tmp/trace/qwen3-8b", "/tmp/trace/qwen3-30b"]
OUT = "/home/wangyuanshuo02/two-level-attention/exp/trace/results/e58_spikiness.json"


def layer_stats(tr, layer_id, device):
    t = torch.load(tr + f"/layer{layer_id:02d}.pt", map_location="cpu", weights_only=False)
    k = t["k"].to(device).float()
    q = t["q"].to(device).float()
    qpos = t["qpos"]
    S = t["S"]
    H, Hkv, D = q.shape[1], k.shape[1], k.shape[2]
    G = H // Hkv
    k_e = k.repeat_interleave(G, dim=1)  # [S, H, D]
    # 末尾 TAIL_N 个位置（prefill 末段行）
    sc = torch.einsum("nhd,shd->nhs", q[-TAIL_N:], k_e) * (D**-0.5)
    pos = torch.arange(S, device=device)
    res = {}
    for ri in range(TAIL_N):
        t_r = int(qpos[-TAIL_N + ri])
        m = pos <= t_r
        sc_r = sc[ri].masked_fill(~m.unsqueeze(0), float("-inf"))  # [H, S]
        p = torch.softmax(sc_r, dim=-1)  # [H, S]
        pg = p.view(Hkv, G, S)
        far_hi = max(FAR_LO, t_r + 1 - NEAR_LEN)
        in_far = (pos >= FAR_LO) & (pos < far_hi)
        for h in range(Hkv):
            for g in range(G):
                ph = pg[h, g]
                srt = torch.sort(ph, descending=True).values
                for K in KS:
                    key = f"top{K}"
                    res.setdefault(key, []).append(float(srt[: min(K, S)].sum()))
                # far 区内的尖峰（丢近端/sink 后）
                p_far = ph[in_far]
                if p_far.numel() > 16:
                    srt_f = torch.sort(p_far, descending=True).values
                    res.setdefault("far_top8", []).append(float(srt_f[:8].sum()))
                # 均匀采样捕获
                for SK in SAMPLE_KS:
                    kk = min(SK, t_r + 1)
                    idx = torch.randint(0, t_r + 1, (kk,), device=device)
                    res.setdefault(f"unif{SK}", []).append(float(ph[idx].sum()))
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
            for li in range(0, n_layers, max(1, n_layers // 12)):  # 12 层采样
                r = layer_stats(tr, li, device)
                for k2, v in r.items():
                    agg.setdefault(k2, []).extend(v)
            rec = {k2: round(sum(v) / len(v), 4) for k2, v in agg.items()}
            out[f"{model}/{name}"] = rec
            print(f"[{model}/{name}] " + " ".join(f"{k2}={v}" for k2, v in rec.items()), flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
