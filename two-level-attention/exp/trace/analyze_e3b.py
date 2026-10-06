# E3b: 完整两级 pipeline（L1 块上界 + L2 4bit 部分维 token 精筛）的概率质量覆盖率
# 对比 dense top-1024 的覆盖上界，解释「TIA 精度为何不掉」+ 定位两级各自的损失
import torch
import torch.nn.functional as F
import glob
import json
import statistics as st

TRACE = "/tmp/trace/qwen3-8b"
OUT = "/home/wangyuanshuo02/two-level-attention/exp/trace/results"


def quant4(x):
    mx = x.amax(-1, keepdim=True)
    mn = x.amin(-1, keepdim=True)
    sc = (mx - mn).clamp(min=1e-9) / 15
    return torch.clamp(torch.round((x - mn) / sc), 0, 15) * sc + mn


def two_level_pipeline(k, q1, t, S, K1=128, dp=32, delta=16, K2=1024):
    """k:[S,Hkv,D](fp32) q1:[1,H,D] -> sel [Hkv,K2] token 位置"""
    Hkv, D = k.shape[1], k.shape[2]
    H = q1.shape[1]
    G = H // Hkv
    nblk = (S + 63) // 64
    if nblk * 64 > S:
        k = F.pad(k, (0, 0, 0, 0, 0, nblk * 64 - S))
    # ---- L1: 子空间块上界
    idx1 = torch.tensor(list(range(64 - dp // 2, 64)) + list(range(128 - dp // 2, 128)), device=k.device)
    kc = k[..., idx1].reshape(nblk, 64, Hkv, dp)
    kmin, kmax = kc.amin(1), kc.amax(1)                       # [nblk,Hkv,dp]
    qs = q1[..., idx1]
    qg = qs.clamp(min=0).reshape(1, Hkv, G, dp)
    qn = qs.clamp(max=0).reshape(1, Hkv, G, dp)
    assert qg.shape == (1, Hkv, G, dp), qg.shape
    sc1 = (torch.einsum("bhgd,nhd->bhgn", qg, kmax) +
           torch.einsum("bhgd,nhd->bhgn", qn, kmin)).sum(-2)  # [1,Hkv,nblk]
    blk_end = (torch.arange(nblk, device=k.device) + 1) * 64 - 1
    sc1 = sc1.masked_fill(blk_end.view(1, 1, -1) > t, float("-inf"))
    # TIA 语义：滑窗块强制入选（不参与 L1 竞争），与 kernel 的 sliding_blocks=3 一致
    sw_blks = 3
    last_blk = t // 64
    force_blks = torch.arange(max(0, last_blk - sw_blks + 1), last_blk + 1, device=k.device)
    cand_blk = torch.topk(sc1, K1, dim=-1).indices[0]          # [Hkv,K1]（不含滑窗）
    # 滑窗块并入候选（去重）
    all_blks = torch.cat([cand_blk, force_blks.unsqueeze(0).expand(Hkv, -1)], dim=1)
    cand_blk = torch.unique(all_blks, dim=1) if False else all_blks  # 保留重复无碍（mask 化）
    # ---- L2: 4bit 量化部分维 token 精筛（TIA: delta=每半取的维数，总维度 = 2*delta）
    idx2 = torch.tensor(list(range(64 - delta, 64)) + list(range(128 - delta, 128)), device=k.device)
    nd2 = 2 * delta
    kq_sub = quant4(k[..., idx2])                              # [nblk*64,Hkv,nd2]
    q2 = q1[..., idx2].reshape(1, Hkv, G, nd2).sum(2)          # [1,Hkv,nd2]（group 求和）
    blk_onehot = torch.zeros(Hkv, nblk, dtype=torch.bool, device=k.device)
    blk_onehot.scatter_(1, cand_blk, True)
    sel_mask = blk_onehot.any(0).repeat_interleave(64)[:S]
    cand_pos = torch.nonzero(sel_mask).squeeze(1)
    kq_h = kq_sub[:S][cand_pos]                                # [Tc,Hkv,delta]
    s2 = torch.einsum("hd,thd->ht", q2[0], kq_h)               # [Hkv,Tc]
    fine = torch.full((Hkv, S), float("-inf"), device=k.device)
    fine[:, cand_pos] = s2
    fine = fine.masked_fill(torch.arange(S, device=k.device).view(1, S) > t, float("-inf"))
    # TIA 语义：最后 sliding_window=128 token 强制入选（p[..., -128:] = 1.0 → 分数 +inf）
    forced = torch.arange(max(0, t - 127), t + 1, device=k.device)
    fine[:, forced] = float("inf")
    return torch.topk(fine, K2, dim=-1).indices                # [Hkv,K2]


def main():
    import glob as _glob
    import os as _os
    results = {}
    for pdir in sorted(_glob.glob(f"{TRACE}/*")):
        name = _os.path.basename(pdir)
        if not _os.path.exists(f"{pdir}/meta.json"):
            continue
        dense_cov, pipe_cov, pipe_vs_dense = [], [], []
        for lf in sorted(glob.glob(f"{TRACE}/{name}/layer*.pt")):
            d = torch.load(lf, map_location="cuda:0")
            k, q, qpos, S = d["k"].float(), d["q"].float(), d["qpos"].cuda(), d["S"]
            t = qpos[-1].item()
            Hkv, H, D = k.shape[1], q.shape[1], k.shape[-1]
            G = H // Hkv
            qg = q[-1:].reshape(1, Hkv, G, D)
            s = torch.einsum("bhgd,chd->bhgc", qg, k).sum(-2) * (D ** -0.5)
            s = s.masked_fill(torch.arange(S, device="cuda").view(1, 1, S) > t, float("-inf"))
            p = torch.softmax(s, dim=-1)
            total = p.sum().item()
            gt = torch.topk(s, 1024, dim=-1).indices[0]
            dense_mass = p[0].gather(-1, gt).sum().item()
            dense_cov.append(dense_mass / total)
            sel = two_level_pipeline(k, q[-1:], t, S)
            pipe_mass = p[0].gather(-1, sel).sum().item()
            pipe_cov.append(pipe_mass / total)
            pipe_vs_dense.append(pipe_mass / dense_mass)       # 相对 dense top1024 的保持率
            del k, q, s, p
            torch.cuda.empty_cache()
        results[name] = {
            "dense_top1024_cov": {"mean": st.mean(dense_cov), "std": st.stdev(dense_cov)},
            "pipeline_cov": {"mean": st.mean(pipe_cov), "std": st.stdev(pipe_cov)},
            "pipeline_vs_dense": {"mean": st.mean(pipe_vs_dense), "std": st.stdev(pipe_vs_dense)},
        }
        print(f"{name}: dense top1024 覆盖 {st.mean(dense_cov):.4f} | 两级 pipeline 覆盖 "
              f"{st.mean(pipe_cov):.4f} | pipeline/dense = {st.mean(pipe_vs_dense):.4f}")
    json.dump(results, open(f"{OUT}/e3b_pipeline_mass_cov.json", "w"), indent=1)


if __name__ == "__main__":
    main()
