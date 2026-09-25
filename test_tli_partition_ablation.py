# B' 分区时序消融 + far survival curve（GPT 批判采纳：量化 far token 到底在 L1 被丢
# 还是在 L2 终选被 near 挤掉）
#
# 概念（用户定调）：粗筛 L1 = 选哪些 block/page；细筛 L2 = 选中 block 内选 token 及数量。
# 三版本对比（其余全同，只改「第一次剪枝时 far 有无保底预算」）：
#   global : L1 全局 top-K1 → L2 全局 top-K2（TIA 原语义，无分区）
#   late   : L1 全局 top-K1 → L2 far/near 分区 topk（当前 B' 实现）
#   hier   : L1 far/near 分区 topk（far 保底 K1_far 块）→ L2 分区 topk（完整 hierarchical）
# survival curve：dense 全维 far oracle token 在 L1 候选池 / L2 终选后的存活率。
# 口径：per-head far recall（GQA far mass 跨 head 极不均）+ 行级剩余 mass coverage。
# 数据：5 任务 trace × 5 层 × 2 个 t，Qwen3-8B 真实权重。
import sys

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
import json

import torch

from sglang.srt.layers.attention.tli.config import TLIProfile
from sglang.srt.layers.attention.tli.indexer import TLIIndexer, kq_unpack

dev = "cuda:0"
TRACES = [
    ("hotpotqa", "/tmp/trace/qwen3-8b/lb_hotpotqa_0"),
    ("musique", "/tmp/trace/qwen3-8b/lb_musique_0"),
    ("gov_report", "/tmp/trace/qwen3-8b/lb_gov_report_0"),
    ("narrativeqa", "/tmp/trace/qwen3-8b/lb_narrativeqa_0"),
    ("passage", "/tmp/trace/qwen3-8b/lb_passage_retrieval_en_0"),
]
LAYERS = [3, 5, 10, 20, 33]
OUT = "/home/wangyuanshuo02/sglang/tli_partition_ablation.json"


def run_mode(index, k_real, q_t, t, prof, idxer, mode):
    """复刻 select 内部（eager 路径），mode 控制两级分区时序。返回
    (sel, l1_pool_mask, cand_pos)，l1_pool_mask 用于 survival 统计。"""
    p = prof
    S = index["S"]
    Hkv = index["kmin"].shape[1]
    H = q_t.shape[1]
    G = H // Hkv
    device = q_t.device
    nblk = index["nblk"]
    kmin, kmax = index["kmin"][:nblk], index["kmax"][:nblk]
    bs = p.block_size
    K1 = min(p.k1_blocks, nblk)
    last_blk = t // bs
    force_blks = torch.arange(
        max(0, last_blk - p.sliding_blocks + 1), last_blk + 1, device=device
    )
    # far 区边界（token 级）
    far_lo = p.sink_blocks * bs
    far_hi = max(far_lo, t + 1 - p.near_len)
    near_floor = p.sliding_window + far_lo
    far_cap = max(0, p.token_budget - near_floor)
    k2_far = min(p.far_tokens, max(0, far_hi - far_lo), far_cap)
    k2_near = max(0, p.token_budget - k2_far)

    # ---- L1 打分（全局，与 select 完全同式）----
    qs = q_t[..., idxer.idx1]
    qg = qs.clamp(min=0).reshape(1, Hkv, G, p.coarse_dim)
    qn = qs.clamp(max=0).reshape(1, Hkv, G, p.coarse_dim)
    sc1 = (
        torch.einsum("bhgd,nhd->bhgn", qg, kmax)
        + torch.einsum("bhgd,nhd->bhgn", qn, kmin)
    ).sum(-2)[0]  # [Hkv, nblk]
    blk_end = (torch.arange(nblk, device=device) + 1) * bs - 1
    sc1 = sc1.masked_fill(blk_end.view(1, -1) > t, float("-inf"))

    far_hi_blk = far_hi // bs  # far 区专属块（不含与 near 带交叠的尾块）
    far_blks = torch.arange(p.sink_blocks, far_hi_blk, device=device)

    if mode == "hier":
        # L1 分区：far 保底 K1_far 块（far 预算 4 块 ×4 安全余量，≥8），
        # 其余名额给 near+sink 区
        K1_far = max(8, min(len(far_blks), (k2_far * 4 + bs - 1) // bs))
        K1_near = max(0, K1 - K1_far)
        sc_far = sc1.clone()
        sc_far[:, : p.sink_blocks] = float("-inf")
        sc_far[:, far_hi_blk:] = float("-inf")
        sc_near = sc1.clone()
        sc_near[:, p.sink_blocks:far_hi_blk] = float("-inf")
        i_far = torch.topk(sc_far, K1_far, dim=-1).indices
        i_near = (
            torch.topk(sc_near, K1_near, dim=-1).indices
            if K1_near > 0
            else torch.zeros(Hkv, 0, dtype=torch.long, device=device)
        )
        cand_blk = torch.cat([i_far, i_near], dim=1)
    else:  # global / late：L1 全局 top-K1
        cand_blk = torch.topk(sc1, K1, dim=-1).indices
    cand_blk = torch.cat(
        [cand_blk, force_blks.unsqueeze(0).expand(Hkv, -1)], dim=1
    )
    blk_onehot = torch.zeros(Hkv, nblk, dtype=torch.bool, device=device)
    blk_onehot.scatter_(1, cand_blk, True)
    pool_mask = blk_onehot.any(0).repeat_interleave(bs)[:S]  # 跨 head 并集池

    # ---- L2 精筛（与 select 同式：kq 4bit 全 S gather）----
    nd2 = 2 * p.delta
    q2 = q_t[..., idxer.idx2].reshape(1, Hkv, G, nd2).sum(2)[0]
    cand_pos = torch.nonzero(pool_mask).squeeze(1)
    kq_h = kq_unpack(
        index["kq_q"][cand_pos], index["kq_sc"][cand_pos], index["kq_mn"][cand_pos]
    )
    s2 = torch.einsum("hd,thd->ht", q2, kq_h)
    fine = torch.full((Hkv, S), float("-inf"), device=device)
    fine[:, cand_pos] = s2
    fine = fine.masked_fill(
        torch.arange(S, device=device).view(1, S) > t, float("-inf")
    )
    forced = torch.arange(max(0, t - p.sliding_window + 1), t + 1, device=device)
    fine[:, forced] = float("inf")

    if mode == "global" or k2_far <= 0 or far_hi <= far_lo:
        return torch.topk(fine, p.token_budget, dim=-1).indices, pool_mask
    # late / hier：L2 far/near 分区（与 select 同式）
    far_f = fine[:, far_lo:far_hi]
    i_f = torch.topk(far_f, k2_far, dim=-1).indices + far_lo
    near_f = fine.clone()
    near_f[:, far_lo:far_hi] = float("-inf")
    i_n = torch.topk(near_f, min(k2_near, S), dim=-1).indices
    return torch.cat([i_f, i_n], dim=-1), pool_mask


rows = []
print(f"{'task':>12s} {'L':>3s} {'t':>6s} | {'mode':>6s} {'far_rec':>8s} {'L1存活':>7s} {'cov总':>7s} {'cov剩':>8s}")
for tname, TDIR in TRACES:
    for layer in LAYERS:
        import os

        f = f"{TDIR}/layer{layer:02d}.pt"
        if not os.path.exists(f):
            continue
        d = torch.load(f, map_location=dev)
        k_real = d["k"].float()
        q_real = d["q"].float()
        qpos = d["qpos"].cuda()
        S, Hkv, D = k_real.shape
        H = q_real.shape[1]
        G = H // Hkv
        prof = TLIProfile()
        idxer = TLIIndexer(prof, head_dim=D).to(dev)
        idx = idxer.build_block_index(k_real)
        for t in [S // 2, S - 1]:
            in_range = torch.nonzero(qpos <= t).squeeze(1)
            qi = int(in_range[-1])
            q_t = q_real[qi : qi + 1]
            # dense oracle（全维）
            qg = q_t.reshape(1, Hkv, G, D)
            s_full = (
                torch.einsum("bhgd,chd->bhgc", qg, k_real[: t + 1]).sum(-2)
                * (D**-0.5)
            )
            p_ = torch.softmax(s_full, dim=-1)[0]
            far_lo = prof.sink_blocks * prof.block_size
            far_hi = max(far_lo, t + 1 - prof.near_len)
            near_floor = prof.sliding_window + far_lo
            far_cap = max(0, prof.token_budget - near_floor)
            k2_far = min(prof.far_tokens, max(0, far_hi - far_lo), far_cap)
            if k2_far <= 0:
                continue
            oracle_far = torch.topk(s_full[0, :, far_lo:far_hi], k2_far, dim=-1).indices + far_lo
            oh = torch.zeros(Hkv, t + 1, device=dev)
            oh.scatter_(1, oracle_far, 1.0)
            sink_hi = prof.sink_blocks * prof.block_size
            sw_lo = max(0, t - prof.sliding_window + 1)
            contested = (
                (torch.arange(t + 1, device=dev) >= sink_hi)
                & (torch.arange(t + 1, device=dev) < sw_lo)
            )
            denom = p_[:, contested].sum().item()

            for mode in ["global", "late", "hier"]:
                sel, pool_mask = run_mode(idx, k_real, q_t, t, prof, idxer, mode)
                selm = torch.zeros(Hkv, t + 1, device=dev)
                selm.scatter_(1, sel.clamp(max=t), 1.0)
                # far recall：终选中 far 区 token ∩ oracle（/k2_far，per-head）
                in_far = (torch.arange(t + 1, device=dev) >= far_lo) & (
                    torch.arange(t + 1, device=dev) < far_hi
                )
                rec = (oh * selm * in_far.view(1, -1)).sum(1) / k2_far
                # far 名额数（global 下 far 可占 >k2_far 名额——rec 口径混淆来源）
                far_slots = (selm * in_far.view(1, -1)).sum(1)  # [Hkv]
                # far mass 覆盖：选中 far 位 mass / far 区总 mass（与名额无关的质量口径）
                far_mass_denom = p_[:, in_far[: t + 1]].sum(1) + 1e-12
                far_mass_cov = (p_ * selm * in_far.view(1, -1)).sum(1) / far_mass_denom
                # survival：oracle far token 在 L1 候选池中的比例（块级池展开）
                oh_pool = oh * pool_mask[: t + 1].view(1, -1).float()
                surv = oh_pool.sum(1) / k2_far
                cov = (p_ * selm).sum().item() / p_.sum().item()
                cov_res = (p_ * selm * contested.view(1, -1)).sum().item() / denom
                rows.append({
                    "task": tname, "layer": layer, "t": t, "mode": mode,
                    "far_recall_mean": round(rec.mean().item(), 4),
                    "far_recall_min": round(rec.min().item(), 4),
                    "far_slots_mean": round(far_slots.mean().item(), 1),
                    "far_mass_cov_mean": round(far_mass_cov.mean().item(), 4),
                    "far_mass_cov_min": round(far_mass_cov.min().item(), 4),
                    "survive_l1_mean": round(surv.mean().item(), 4),
                    "survive_l1_min": round(surv.min().item(), 4),
                    "cov_total": round(cov, 5),
                    "cov_residual": round(cov_res, 5),
                })
                print(f"{tname:>12s} L{layer:02d} {t:6d} | {mode:>6s} "
                      f"rec {rec.mean().item():7.4f} slots {far_slots.mean().item():6.1f} "
                      f"masscov {far_mass_cov.mean().item():7.4f} L1存 {surv.mean().item():6.4f} "
                      f"cov剩 {cov_res:8.5f}")
        del d, k_real, q_real, idx
        torch.cuda.empty_cache()

json.dump(rows, open(OUT, "w"), indent=1)
print(f"\nsaved -> {OUT}（{len(rows)} 行）")

print("\n==== 汇总（跨任务×层×t 聚合）====")
for mode in ["global", "late", "hier"]:
    rs = [r for r in rows if r["mode"] == mode]
    n = len(rs)
    if not n:
        continue
    m = lambda k: sum(r[k] for r in rs) / n
    mn = lambda k: min(r[k] for r in rs)
    print(f"{mode:>6s} ({n} 行): far_rec {m('far_recall_mean'):.4f} | far名额 "
          f"{m('far_slots_mean'):.1f} | far mass cov {m('far_mass_cov_mean'):.4f} "
          f"(min {mn('far_mass_cov_min'):.4f}) | L1 存活 {m('survive_l1_mean'):.4f} "
          f"| cov剩 mean {m('cov_residual'):.5f} min {mn('cov_residual'):.5f}")
