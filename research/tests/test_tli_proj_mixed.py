# PCA 与低频尾维混合打分（两机制近正交 73°，测互补性）：score = q_pca·k_pca + q_tail·k_tail
# 对照：pca32 / tail32 / mixed32（16+16）/ mixed16（8+8）。同隔离口径。
import sys

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
import json

import torch

dev = "cuda:0"
TRACES = [
    ("hotpotqa", "/tmp/trace/qwen3-8b/lb_hotpotqa_0"),
    ("musique", "/tmp/trace/qwen3-8b/lb_musique_0"),
    ("gov_report", "/tmp/trace/qwen3-8b/lb_gov_report_0"),
]
LAYERS = [3, 5, 10, 20, 33]


def tail_basis(r):
    idx = list(range(64 - r // 2, 64)) + list(range(128 - r // 2, 128))
    E = torch.zeros(128, r, device=dev)
    E[idx, torch.arange(r)] = 1.0
    return E


rows = []
for tname, TDIR in TRACES:
    for layer in LAYERS:
        d = torch.load(f"{TDIR}/layer{layer:02d}.pt", map_location=dev)
        k_real = d["k"].float()
        q_real = d["q"].float()
        qpos = d["qpos"].cuda()
        S, Hkv, D = k_real.shape
        H = q_real.shape[1]
        G = H // Hkv
        for t in [S // 2, S - 1]:
            in_range = torch.nonzero(qpos <= t).squeeze(1)
            qi = int(in_range[-1])
            q_t = q_real[qi : qi + 1]
            qg = q_t.reshape(1, Hkv, G, D)
            s_full = (
                torch.einsum("bhgd,chd->bhgc", qg, k_real[: t + 1]).sum(-2)
                * (D**-0.5)
            )
            far_lo, near_len, tok_budget, sw, far_tok = 128, 2048, 1024, 128, 256
            far_hi = max(far_lo, t + 1 - near_len)
            k2_far = min(far_tok, max(0, far_hi - far_lo), max(0, tok_budget - (sw + far_lo)))
            if k2_far <= 0:
                continue
            oracle = torch.topk(s_full[0, :, far_lo:far_hi], k2_far, dim=-1).indices + far_lo
            oh = torch.zeros(Hkv, t + 1, device=dev)
            oh.scatter_(1, oracle, 1.0)
            q_h = qg.sum(2)[0]
            kfar = k_real[far_lo:far_hi]

            # 同源 PCA 基（全 far 区）
            vb = []
            for h in range(Hkv):
                _, _, Vt = torch.linalg.svd(kfar[:, h], full_matrices=False)
                vb.append(Vt[:32].T)  # 取到 32 备用
            V = torch.stack(vb)  # [Hkv, D, 32]

            def score_proj(Vh, rt):
                q_p = torch.einsum("hdr,hd->hr", Vh[:, :, :rt], q_h)
                k_p = torch.einsum("hdr,thd->thr", Vh[:, :, :rt], kfar)
                return torch.einsum("hr,thr->ht", q_p, k_p)

            def score_tail(rt):
                E = tail_basis(rt)
                q_p = torch.einsum("hdr,hd->hr", E.unsqueeze(0).expand(Hkv, -1, -1), q_h)
                k_p = torch.einsum("hdr,thd->thr", E.unsqueeze(0).expand(Hkv, -1, -1), kfar)
                return torch.einsum("hr,thr->ht", q_p, k_p)

            def rec_of(sc):
                sel = torch.topk(sc, k2_far, dim=-1).indices + far_lo
                m = torch.zeros(Hkv, t + 1, device=dev)
                m.scatter_(1, sel, 1.0)
                return ((oh * m).sum(1) / k2_far).mean().item()

            rec = {
                "pca32": rec_of(score_proj(V, 32)),
                "tail32": rec_of(score_tail(32)),
                "mixed32": rec_of(score_proj(V, 16) + score_tail(16)),
                "mixed16": rec_of(score_proj(V, 8) + score_tail(8)),
            }
            rows.append({"task": tname, "layer": layer, "t": t, **{k: round(v, 4) for k, v in rec.items()}})
            print(f"{tname:>12s} L{layer:02d} {t:6d} | " +
                  " ".join(f"{k}={rec[k]:.4f}" for k in ["pca32", "tail32", "mixed32", "mixed16"]))
        del d, k_real, q_real
        torch.cuda.empty_cache()

json.dump(rows, open("/home/wangyuanshuo02/sglang/tli_proj_mixed.json", "w"), indent=1)
print("\n==== 汇总（far recall mean）====")
for k in ["pca32", "tail32", "mixed32", "mixed16"]:
    print(f"{k:>8s}: {sum(r[k] for r in rows) / len(rows):.4f}")
