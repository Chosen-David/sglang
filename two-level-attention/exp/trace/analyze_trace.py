# E2/E3/E4/E5 trace 分析：H1 位置偏斜 + 子空间 recall + 代表质量 + 附加创新点
# 输入：collect_trace.py 产出的 /tmp/trace/qwen3-8b/<prompt>/layer*.pt
# 全部在真实 Qwen3-8B trace 上计算，替换 proposal 的 synthetic 数据
import os
import json
import glob
import torch
import torch.nn.functional as F

TRACE_DIR = "/tmp/trace/qwen3-8b"
OUT_DIR = "/home/wangyuanshuo02/two-level-attention/exp/trace/results"
TOPK = 1024          # 与导师 TIA level2_topk 对齐
BLOCK = 64           # TIA block size
SLIDING = 128        # TIA sliding window
NEAR_TOKENS = 2048   # 创新点 B 近端区域定义（proposal slide 10 近端小窗）


def load_layer(path):
    d = torch.load(path, map_location="cuda:0")
    return d["k"].bfloat16(), d["q"].bfloat16(), d["qpos"], d["S"]


def dense_topk(k, q, topk=TOPK):
    """真实 dense top-k（ground truth）。k:[S,Hkv,D] q:[nq,H,D] -> 每层每 head 的 top-k 位置
    返回 gt: [nq, H, topk] long（token 位置），score 未归一化"""
    S, Hkv, D = k.shape
    nq, H, _ = q.shape
    G = H // Hkv
    # q: [nq, H, D] -> [nq, Hkv, G, D]
    qg = q.view(nq, Hkv, G, D)
    # k: [S, Hkv, D] -> [1, Hkv, D, S]
    kt = k.permute(1, 2, 0).unsqueeze(0)
    # 分块算 score 防 OOM：score[nq, Hkv, S] = sum_g q·k
    scores = torch.zeros(nq, Hkv, S, device="cuda", dtype=torch.float32)
    for i in range(0, nq, 32):
        qi = qg[i:i + 32].float()                     # [b, Hkv, G, D]
        s = torch.einsum("bhgd,chd->bhgc", qi, k.float())  # [b,Hkv,G,S]
        s = s.sum(-2)                                  # group sum（kernel 语义）
        # causal mask: q 位置 qpos
        qpos_i = q[i:i + 32].new_empty(0)  # placeholder
        scores[i:i + 32] = s
    return scores


def causal_topk_positions(scores, qpos, topk=TOPK):
    """scores: [nq, Hkv, S]，按 q 位置 t 做因果 mask 后 top-k
    返回 gt_pos [nq, Hkv, topk]"""
    nq, Hkv, S = scores.shape
    idx = torch.arange(S, device="cuda").view(1, 1, S)
    valid = idx <= qpos.view(-1, 1, 1)                 # [nq,1,S] broadcast
    scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))
    k = min(topk, S)
    return torch.topk(scores, k, dim=-1).indices       # [nq, Hkv, k]


def analyze_h1():
    """H1: 位置偏斜——dense top-k 的距离直方图（最后 1 个 q 位置，decode 视角）"""
    results = {}
    for pdir in sorted(glob.glob(f"{TRACE_DIR}/*")):
        name = os.path.basename(pdir)
        meta = json.load(open(f"{pdir}/meta.json"))
        hist_all = torch.zeros(17, dtype=torch.float64)  # log2 距离桶 0..16+
        near_frac_layers = []
        for lf in sorted(glob.glob(f"{pdir}/layer*.pt")):
            layer = int(lf.split("layer")[-1][:2])
            k, q, qpos, S = load_layer(lf)
            qpos = qpos.cuda()
            # 只看最后一个 q 位置（decode 视角）
            t = qpos[-1].item()
            qt = q[-1:].float()                         # [1,H,D]
            Hkv, H = k.shape[1], q.shape[1]
            G = H // Hkv
            qg = qt.view(1, Hkv, G, D := k.shape[-1])
            s = torch.einsum("bhgd,chd->bhgc", qg, k.float()).sum(-2)  # [1,Hkv,S]
            s = s.masked_fill(torch.arange(S, device="cuda").view(1, 1, S) > t, float("-inf"))
            gt = torch.topk(s, TOPK, dim=-1).indices    # [1,Hkv,K]
            dist = (t - gt).float()                     # [1,Hkv,K]
            # 近端覆盖率（<SLIDING 与 < NEAR）
            near_frac_layers.append(((dist < NEAR_TOKENS).float().mean().item()))
            # log2 直方图
            b = torch.clamp(torch.log2(dist.clamp(min=1)).long(), max=16)
            hist_all += torch.bincount(b.flatten().cpu(), minlength=17).double()
            del k, q
            torch.cuda.empty_cache()
        results[name] = {
            "hist_log2": hist_all.tolist(),
            "near2048_frac_mean": sum(near_frac_layers) / len(near_frac_layers),
            "near2048_frac_min": min(near_frac_layers),
            "near2048_frac_per_layer": near_frac_layers,
        }
        print(f"[H1:{name}] top{TOPK} 中 <2048 距离占比: mean={results[name]['near2048_frac_mean']:.3f} "
              f"min={results[name]['near2048_frac_min']:.3f}")
    json.dump(results, open(f"{OUT_DIR}/h1_position_skew.json", "w"), indent=1)
    return results


def qwen_subspace_idx(d_prime, D=128):
    """Qwen3 rotate_half 布局的低频尾部维度（与导师 TIA 相同规则）"""
    assert D == 128 and d_prime <= D and d_prime % 2 == 0
    d = d_prime // 2
    return list(range(64 - d, 64)) + list(range(128 - d, 128))


def level1_coarse_scores(k, q, dim_idx, block=BLOCK):
    """子空间 min/max 块上界粗筛。k:[S,Hkv,D] q:[nq,H,D]
    返回 block 分数 [nq, Hkv, nblocks]（group sum 语义，同 kernel）"""
    S, Hkv, D = k.shape
    nq, H, _ = q.shape
    G = H // Hkv
    nblk = S // block
    idx = torch.tensor(dim_idx, device="cuda")
    ksub = k[..., idx]                                   # [S,Hkv,d']
    pad = (nblk * block - S)
    if pad:
        ksub = F.pad(ksub, (0, 0, 0, 0, 0, pad), value=0.0)
    kc = ksub.view(nblk, block, Hkv, -1)
    kmin, kmax = kc.amin(1), kc.amax(1)                  # [nblk,Hkv,d']
    qsub = q[..., idx].float()                           # [nq,H,d']
    qpos_ = qsub.clamp(min=0)
    qneg_ = qsub.clamp(max=0)
    qg_pos = qpos_.view(nq, Hkv, G, -1)
    qg_neg = qneg_.view(nq, Hkv, G, -1)
    sc = (torch.einsum("bhgd,nhd->bhgn", qg_pos, kmax.float()) +
          torch.einsum("bhgd,nhd->bhgn", qg_neg, kmin.float())).sum(-2)  # [nq,Hkv,nblk]
    return sc


def analyze_h2():
    """H2/E3: 子空间 d' 扫描——Level-1 块粗筛 4x 候选下的 token Recall@K
    口径：gt = dense top-1024（group-sum 分数，同 kernel 语义）；
    候选 = 子空间块分数 top-K1 块展开（K1*64 = 4*1024 token）
    报告 overall / remote(>NEAR) / worst-layer P5"""
    CAND = 4 * TOPK
    K1 = CAND // BLOCK
    dprimes = [32, 48, 64, 96, 128]
    results = {}
    for pdir in sorted(glob.glob(f"{TRACE_DIR}/*")):
        name = os.path.basename(pdir)
        meta = json.load(open(f"{pdir}/meta.json"))
        S_all = meta["S"]
        nblk = S_all // BLOCK
        per_layer = {dp: [] for dp in dprimes}
        remote_per_layer = {dp: [] for dp in dprimes}
        for lf in sorted(glob.glob(f"{pdir}/layer*.pt")):
            layer = int(lf.split("layer")[-1][:2])
            k, q, qpos, S = load_layer(lf)
            qpos = qpos.cuda()
            t = qpos[-1].item()
            Hkv, H, D = k.shape[1], q.shape[1], k.shape[-1]
            # ground truth: dense group-sum 分数 top-1024
            qt = q[-1:].float()
            G = H // Hkv
            qg = qt.view(1, Hkv, G, D)
            s = torch.einsum("bhgd,chd->bhgc", qg, k.float()).sum(-2)
            s = s.masked_fill(torch.arange(S, device="cuda").view(1, 1, S) > t, float("-inf"))
            gt = torch.topk(s, TOPK, dim=-1).indices      # [1,Hkv,K]
            gt_set = [set(gt[0, h].tolist()) for h in range(Hkv)]
            gt_remote = [set(p for p in gt_set[h] if t - p >= NEAR_TOKENS) for h in range(Hkv)]
            # 各 d' 的粗筛
            for dp in dprimes:
                sc = level1_coarse_scores(k, q[-1:], qwen_subspace_idx(dp))  # [1,Hkv,nblk]
                # 块因果：块末 token 位置 <= t
                blk_end = (torch.arange(nblk, device="cuda") + 1) * BLOCK - 1
                sc = sc.masked_fill(blk_end.view(1, 1, -1) > t, float("-inf"))
                cand_blk = torch.topk(sc, min(K1, nblk), dim=-1).indices     # [1,Hkv,K1]
                recall_h, remote_recall_h = [], []
                for h in range(Hkv):
                    cand = set()
                    for b in cand_blk[0, h].tolist():
                        cand.update(range(b * BLOCK, min((b + 1) * BLOCK, S)))
                    recall_h.append(len(cand & gt_set[h]) / TOPK)
                    if gt_remote[h]:
                        remote_recall_h.append(len(cand & gt_remote[h]) / len(gt_remote[h]))
                per_layer[dp].append(sum(recall_h) / len(recall_h))
                remote_per_layer[dp].append(
                    sum(remote_recall_h) / len(remote_recall_h) if remote_recall_h else float("nan"))
            del k, q
            torch.cuda.empty_cache()
        results[name] = {}
        for dp in dprimes:
            ls = per_layer[dp]
            rs = [x for x in remote_per_layer[dp] if x == x]
            results[name][dp] = {
                "recall_mean": sum(ls) / len(ls),
                "recall_p5": sorted(ls)[max(0, int(0.05 * len(ls)))],
                "remote_recall_mean": sum(rs) / len(rs) if rs else None,
                "hbm_ratio": dp / 128,
            }
            print(f"[H2:{name}] d'={dp}: recall={results[name][dp]['recall_mean']:.4f} "
                  f"p5={results[name][dp]['recall_p5']:.4f} remote={results[name][dp]['remote_recall_mean']}")
    json.dump(results, open(f"{OUT_DIR}/h2_subspace_recall.json", "w"), indent=1)
    return results


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "h1"
    if which == "h1":
        analyze_h1()
    elif which == "h2":
        analyze_h2()
