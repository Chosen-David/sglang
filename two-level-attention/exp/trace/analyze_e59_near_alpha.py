# #59 near/far 配比消融（离线 trace 重放）：near_len ∈ {1024, 2048, 4096}
# 扫描 far 区 top-1024 mass 中「远于 near_len 的质量」——即近端截断点
# 变化时 far 区（[128, t+1-near_len)）承载的 mass 变化。
# 用户假说：固定 near_len=2048 太单一，应 alpha 比例控制。
# 口径：真实全维分布 softmax（causal），far mass 占比 per-(layer,kv-head,
# q-head) 末尾行平均。near_len 越大 far 区越窄、far mass 越小 →
# 索引负担越小；near_len 越小 far 区越宽 → 需要更准的 far 选择。
# 同时测 far 区内 top-256 的 mass（far_tokens 预算口径）。
import json
import os

import torch

NEAR_LENS = [1024, 2048, 4096]
FAR_LO = 128
TAIL_N = 8
TRACE_DIRS = ["/tmp/trace/qwen3-8b", "/tmp/trace/qwen3-30b"]
OUT = "/home/wangyuanshuo02/two-level-attention/exp/trace/results/e59_near_alpha.json"


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
    res = {}
    for ri in range(TAIL_N):
        t_r = int(qpos[-TAIL_N + ri])
        if t_r < 4096 + FAR_LO:
            continue
        sc = torch.einsum("hd,shd->hs", q[-TAIL_N + ri], k_e) * (D**-0.5)
        m = pos <= t_r
        sc = sc.masked_fill(~m.unsqueeze(0), float("-inf"))
        p = torch.softmax(sc, dim=-1).view(Hkv, G, S)
        for nl in NEAR_LENS:
            far_hi = max(FAR_LO, t_r + 1 - nl)
            in_far = (pos >= FAR_LO) & (pos < far_hi)
            pf = p * in_far.view(1, 1, S)
            total_far = pf.sum(-1)  # [Hkv, G]
            srt = torch.sort(pf.flatten(2), descending=True).values
            top256 = srt[:, :, :256].sum(-1)
            key = f"nl{nl}"
            res.setdefault(f"{key}_farmass", []).append(float(total_far.mean()))
            # far top-256 占 far 总 mass（选择质量上界：预算 256 时最优可达）
            res.setdefault(f"{key}_top256frac", []).append(
                float((top256 / total_far.clamp(min=1e-12)).mean()))
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
            rec = {k2: round(sum(v) / len(v), 4) for k2, v in agg.items()}
            out[f"{model}/{name}"] = rec
            print(f"[{model}/{name}] " + " ".join(f"{k2}={v}" for k2, v in sorted(rec.items())), flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
