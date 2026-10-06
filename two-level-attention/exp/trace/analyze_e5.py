# E5: 附加创新点实测——
#   (1) 跨层 top-k 复用（IndexCache 式）：相邻层 top-1024 IoU + 跳层复用衰减
#   (2) 增量 topk（decode 步间漂移）：相邻 decode 步 top-1024 的新增率
#   (3) 层异质性跨 prompt 稳定性：far 质量轮廓的相关性（离线 profiling 可行性）
import os
import torch
import torch.nn.functional as F
import glob
import json
import statistics as st

TRACE = "/tmp/trace/qwen3-8b"
OUT = "/home/wangyuanshuo02/two-level-attention/exp/trace/results"


def layer_topk(k, q, t, topk=1024):
    """每 kv head 的 dense top-k 位置集合。k:[S,Hkv,D] q:[n,H,D]"""
    Hkv, H, D = k.shape[1], q.shape[1], k.shape[-1]
    G = H // Hkv
    qg = q.reshape(-1, Hkv, G, D)
    s = torch.einsum("bhgd,chd->bhgc", qg, k).sum(-2) * (D ** -0.5)
    s = s.masked_fill(torch.arange(k.shape[0], device="cuda").view(1, 1, -1) > t.view(-1, 1, 1),
                      float("-inf"))
    return torch.topk(s, topk, dim=-1).indices  # [n,Hkv,topk]


def main():
    results = {}
    far_profiles = {}
    for pdir in sorted(glob.glob(f"{TRACE}/*")):
        name = os.path.basename(pdir)
        if not name.startswith(("needle", "natural")):
            continue
        tops = []          # 每层最后 q 位置的 top-k
        far_mass_profile = []
        tail_data = None   # 最后 64 个 tail q（增量分析）
        for lf in sorted(glob.glob(f"{pdir}/layer*.pt")):
            d = torch.load(lf, map_location="cuda:0")
            k, q, qpos, S = d["k"].float(), d["q"].float(), d["qpos"].cuda(), d["S"]
            t_last = qpos[-1].item()
            tops.append(layer_topk(k, q[-1:], torch.tensor([t_last], device="cuda"))[0])  # [Hkv,K]
            # far 质量轮廓
            Hkv, H, D = k.shape[1], q.shape[1], k.shape[-1]
            G = H // Hkv
            qg = q[-1:].reshape(1, Hkv, G, D)
            s = torch.einsum("bhgd,chd->bhgc", qg, k).sum(-2) * (D ** -0.5)
            s = s.masked_fill(torch.arange(S, device="cuda").view(1, 1, S) > t_last, float("-inf"))
            p = torch.softmax(s, dim=-1)
            pm = p.mean(dim=(0, 1))
            far_mass_profile.append(pm[64:t_last - 2048].sum().item())
            if tail_data is None:
                tail_q = q[-64:]                        # [64,H,D] 连续 decode 步
                tail_pos = qpos[-64:]
                tail_data = (k, tail_q, tail_pos)
            else:
                del k
            if len(tops) > 1:
                del d
            torch.cuda.empty_cache()
        # (1) 跨层 IoU
        iou_adj, iou_skip = [], {2: [], 4: [], 8: []}
        for i in range(len(tops) - 1):
            a = set(tops[i].flatten().tolist())
            b = set(tops[i + 1].flatten().tolist())
            iou_adj.append(len(a & b) / len(a | b))
            for skip in iou_skip:
                if i + skip < len(tops):
                    c = set(tops[i + skip].flatten().tolist())
                    iou_skip[skip].append(len(a & c) / len(a | c))
        # (2) 增量 topk：相邻 decode 步新增率
        k, tail_q, tail_pos = tail_data
        tail_tops = layer_topk(k, tail_q, tail_pos)    # [64,Hkv,K]
        churn = []
        for i in range(tail_tops.shape[0] - 1):
            a = set(tail_tops[i].flatten().tolist())
            b = set(tail_tops[i + 1].flatten().tolist())
            churn.append(len(b - a) / len(b))          # 新进入 top-k 的比例
        results[name] = {
            "iou_adjacent": st.mean(iou_adj),
            "iou_skip": {str(s2): st.mean(v) for s2, v in iou_skip.items()},
            "decode_churn_adjacent": st.mean(churn),
            "decode_churn_max": max(churn),
        }
        far_profiles[name] = far_mass_profile
        print(f"=== {name} ===")
        print(f"  跨层 IoU(adj)={st.mean(iou_adj):.3f}  skip2={st.mean(iou_skip[2]):.3f} "
              f"skip4={st.mean(iou_skip[4]):.3f}  skip8={st.mean(iou_skip[8]):.3f}")
        print(f"  decode 步间 churn={st.mean(churn):.4f} max={max(churn):.3f}")
        del tail_data, tops, k
        torch.cuda.empty_cache()
    # (3) 层轮廓跨 prompt 相关性
    if len(far_profiles) == 2:
        a, b = list(far_profiles.values())
        n = min(len(a), len(b))
        import math
        ma, mb = sum(a) / n, sum(b) / n
        cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
        va = sum((a[i] - ma) ** 2 for i in range(n))
        vb = sum((b[i] - mb) ** 2 for i in range(n))
        corr = cov / math.sqrt(va * vb) if va > 0 and vb > 0 else 0.0
        results["far_profile_corr_needle_vs_natural"] = corr
        print(f"  far 质量层轮廓相关性(needle vs natural): {corr:.3f}")
    json.dump(results, open(f"{OUT}/e5_reuse_churn.json", "w"), indent=1)

if __name__ == "__main__":
    main()
