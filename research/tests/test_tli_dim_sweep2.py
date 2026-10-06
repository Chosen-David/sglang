# L2 细筛维度可压性扩展测试（用户指示：v1「细筛不能压维度」结论样本量不足需进一步验证）
#
# v1 缺口（test_tli_dim_sweep.py）：单 trace（hotpotqa）× 5 层 × 2 个 t × 5 配置，
# δ=16→8 的 cov 下降可能只是个别 far-heavy 层的局部现象。
# 本轮扩展：
#   ① 5 任务 trace（hotpotqa/musique/gov_report/narrativeqa/passage_retrieval）
#      × 5 层（L03/L05 far-heavy + L10/L20/L33 典型）× 3 个 t（S/4, S/2, S-1）
#   ② δ 细扫 {16, 12, 8, 4}
#   ③ 机制分离对照（关键）：同样 2δ 维下 fp32 精筛 vs 4bit 量化精筛，
#      far 区直接打分（不经过 L1）——隔离「维度不够」与「低维下 4bit 量化
#      噪声占比放大」两种病因。若 fp-δ8 不掉而 4bit-δ8 崩 → 病因是量化精度
#      （换 8bit/int6 存储可救，存储账要重算）；若两者同崩 → 维度本身不够。
#   ④ 全链路口径（idxer.select 真实 pipeline）与隔离口径并报；far recall
#      对照 oracle = dense 全维 far 区 top-k2_far。
# 口径：per-head far recall（GQA far mass 跨 head 极不均，E4c 教训）+ 行级
# 剩余 mass coverage（竞争区 = 去 sink + 滑窗）。
import sys

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
import json
import os

import torch

from sglang.srt.layers.attention.tli.config import TLIProfile
from sglang.srt.layers.attention.tli.indexer import TLIIndexer, quant4

dev = "cuda:0"
TRACES = [
    ("hotpotqa", "/tmp/trace/qwen3-8b/lb_hotpotqa_0"),
    ("musique", "/tmp/trace/qwen3-8b/lb_musique_0"),
    ("gov_report", "/tmp/trace/qwen3-8b/lb_gov_report_0"),
    ("narrativeqa", "/tmp/trace/qwen3-8b/lb_narrativeqa_0"),
    ("passage", "/tmp/trace/qwen3-8b/lb_passage_retrieval_en_0"),
]
LAYERS = [3, 5, 10, 20, 33]
DELTAS = [16, 12, 8, 4]
OUT = "/home/wangyuanshuo02/sglang/tli_dim_sweep2.json"

rows = []
for tname, TDIR in TRACES:
    for layer in LAYERS:
        f = f"{TDIR}/layer{layer:02d}.pt"
        if not os.path.exists(f):
            print(f"[skip] {f} 不存在")
            continue
        d = torch.load(f, map_location=dev)
        k_real = d["k"].float()  # [S, Hkv, D]
        q_real = d["q"].float()  # [nq, H, D]
        qpos = d["qpos"].cuda()
        S, Hkv, D = k_real.shape
        H = q_real.shape[1]
        G = H // Hkv
        for t in [S // 4, S // 2, S - 1]:
            in_range = torch.nonzero(qpos <= t).squeeze(1)
            qi = int(in_range[-1])
            q_t = q_real[qi : qi + 1]
            # dense 全维分数与分布（oracle 基准）
            qg = q_t.reshape(1, Hkv, G, D)
            s_full = (
                torch.einsum("bhgd,chd->bhgc", qg, k_real[: t + 1]).sum(-2)
                * (D**-0.5)
            )
            p_ = torch.softmax(s_full, dim=-1)[0]  # [Hkv, t+1]
            # far 区边界（B' 默认参数）
            prof0 = TLIProfile()
            far_lo = prof0.sink_blocks * prof0.block_size
            far_hi = max(far_lo, t + 1 - prof0.near_len)
            near_floor = prof0.sliding_window + far_lo
            far_cap = max(0, prof0.token_budget - near_floor)
            k2_far = min(prof0.far_tokens, far_tok_hi_len := max(0, far_hi - far_lo), far_cap)
            if k2_far <= 0:
                continue
            # oracle far token（dense 全维 far 区 top-k2_far，per-head）
            far_s = s_full[0, :, far_lo:far_hi]  # [Hkv, L_far]
            oracle_far = torch.topk(far_s, k2_far, dim=-1).indices + far_lo  # [Hkv, k2far]
            oh = torch.zeros(Hkv, t + 1, device=dev)
            oh.scatter_(1, oracle_far, 1.0)

            for delta in DELTAS:
                # ---- ③ 隔离口径：far 区直接打分（不经 L1）----
                idx2 = list(range(64 - delta, 64)) + list(range(128 - delta, 128))
                idx2_t = torch.tensor(idx2, device=dev)
                k2d = k_real[..., idx2_t]  # [S, Hkv, 2δ]
                q2d = q_t[..., idx2_t].reshape(1, Hkv, G, 2 * delta).sum(2)[0]  # [Hkv, 2δ]
                kfar = k2d[far_lo:far_hi]  # [L_far, Hkv, 2δ]
                # fp32 精筛
                sc_fp = torch.einsum("hd,thd->ht", q2d, kfar)  # [Hkv, L_far]
                sel_fp = torch.topk(sc_fp, k2_far, dim=-1).indices + far_lo
                oh_fp = torch.zeros(Hkv, t + 1, device=dev)
                oh_fp.scatter_(1, sel_fp, 1.0)
                rec_fp = (oh * oh_fp).sum(1) / k2_far  # per-head recall
                # 4bit 精筛（与生产 kq 同量化：per-token-head per-dim 4bit）
                kfar_q = quant4(kfar)  # [L_far, Hkv, 2δ] 格点值 fp32
                sc_q = torch.einsum("hd,thd->ht", q2d, kfar_q)
                sel_q4 = torch.topk(sc_q, k2_far, dim=-1).indices + far_lo
                oh_q4 = torch.zeros(Hkv, t + 1, device=dev)
                oh_q4.scatter_(1, sel_q4, 1.0)
                rec_q4 = (oh * oh_q4).sum(1) / k2_far

                # ---- ④ 全链路口径（真实 pipeline，δ 经 env 生效）----
                os.environ["SGLANG_TLI_DELTA"] = str(delta)
                os.environ["SGLANG_TLI_COARSE_DIM"] = "32"
                prof = TLIProfile()
                idxer = TLIIndexer(prof, head_dim=D).to(dev)
                idx = idxer.build_block_index(k_real)
                sel = idxer.select(idx, q_t, t)  # [Hkv, K2]
                selm = torch.zeros(Hkv, t + 1, device=dev)
                selm.scatter_(1, sel.clamp(max=t), 1.0)
                # far recall（选中位落在 far 区的部分 vs oracle）
                in_far = (torch.arange(t + 1, device=dev) >= far_lo) & (
                    torch.arange(t + 1, device=dev) < far_hi
                )
                in_far_sel = selm * in_far.view(1, -1)  # [Hkv, t+1]（bool→乘法掩码）
                rec_pipe = (oh * in_far_sel).sum(1) / k2_far
                # 剩余 mass coverage（竞争区 = 去 sink + 滑窗强制位）
                cov = (p_ * selm).sum().item() / p_.sum().item()
                sink_hi = prof.sink_blocks * prof.block_size
                sw_lo = max(0, t - prof.sliding_window + 1)
                contested = (
                    (torch.arange(t + 1, device=dev) >= sink_hi)
                    & (torch.arange(t + 1, device=dev) < sw_lo)
                )
                denom = p_[:, contested].sum().item()
                cov_res = (
                    (p_ * selm * contested.view(1, -1)).sum().item() / denom
                )
                rows.append({
                    "task": tname, "layer": layer, "t": t, "delta": delta,
                    "far_recall_fp_mean": round(rec_fp.mean().item(), 4),
                    "far_recall_fp_min": round(rec_fp.min().item(), 4),
                    "far_recall_4bit_mean": round(rec_q4.mean().item(), 4),
                    "far_recall_4bit_min": round(rec_q4.min().item(), 4),
                    "far_recall_pipeline_mean": round(rec_pipe.mean().item(), 4),
                    "far_recall_pipeline_min": round(rec_pipe.min().item(), 4),
                    "cov_total": round(cov, 5),
                    "cov_residual": round(cov_res, 5),
                    "kq_bytes_per_token_head": 2 * delta + 8,
                })
                print(f"{tname:14s} L{layer:02d} t={t:6d} δ={delta:2d} | "
                      f"far_rec fp {rec_fp.mean().item():.4f}/{rec_fp.min().item():.4f} "
                      f"4bit {rec_q4.mean().item():.4f}/{rec_q4.min().item():.4f} "
                      f"pipe {rec_pipe.mean().item():.4f} | cov剩 {cov_res:.5f}")
        del d, k_real, q_real
        torch.cuda.empty_cache()

json.dump(rows, open(OUT, "w"), indent=1)
print(f"\nsaved -> {OUT}（{len(rows)} 行）")

# 汇总
print("\n==== 汇总（跨任务×层×t 聚合）====")
for delta in DELTAS:
    rs = [r for r in rows if r["delta"] == delta]
    n = len(rs)
    if not n:
        continue
    def m(k):
        return sum(r[k] for r in rs) / n
    def mn(k):
        return min(r[k] for r in rs)
    print(f"δ={delta:2d} ({n} 行): far_rec fp {m('far_recall_fp_mean'):.4f} "
          f"(min {mn('far_recall_fp_min'):.4f}) | 4bit {m('far_recall_4bit_mean'):.4f} "
          f"(min {mn('far_recall_4bit_min'):.4f}) | pipe {m('far_recall_pipeline_mean'):.4f} "
          f"| cov剩 mean {m('cov_residual'):.5f} min {mn('cov_residual'):.5f} "
          f"| kq {2*delta+8}B/token-head")
