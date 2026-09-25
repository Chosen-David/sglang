# M9 补充：PCA16 pipeline vs 隔离（fp32/4bit）差距分解（30 行同口径）
import sys

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
import torch

from sglang.srt.layers.attention.tli.config import TLIProfile
from sglang.srt.layers.attention.tli.indexer import TLIIndexer, quant4, quant4_pack, kq_unpack

dev = "cuda:0"
B = torch.load("/home/wangyuanshuo02/sglang/tli_pca_basis_r16.pt", map_location=dev)
TRACES = [
    ("hotpotqa", "/tmp/trace/qwen3-8b/lb_hotpotqa_0"),
    ("musique", "/tmp/trace/qwen3-8b/lb_musique_0"),
    ("gov_report", "/tmp/trace/qwen3-8b/lb_gov_report_0"),
]
LAYERS = [3, 5, 10, 20, 33]
prof = TLIProfile()
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
        V = B[layer]
        r = V.shape[-1]
        idxer = TLIIndexer(prof, head_dim=128, basis=V).to(dev)
        for t in [S // 2, S - 1]:
            in_range = torch.nonzero(qpos <= t).squeeze(1)
            qi = int(in_range[-1])
            q_t = q_real[qi : qi + 1]
            qg = q_t.reshape(1, Hkv, G, D)
            s_full = torch.einsum("bhgd,chd->bhgc", qg, k_real[: t + 1]).sum(-2)
            far_lo, near_len = 128, 2048
            far_hi = max(far_lo, t + 1 - near_len)
            k2_far = min(256, max(0, far_hi - far_lo))
            if k2_far <= 0:
                continue
            oracle = torch.topk(s_full[0, :, far_lo:far_hi], k2_far, dim=-1).indices + far_lo
            oh = torch.zeros(Hkv, t + 1, device=dev)
            oh.scatter_(1, oracle, 1.0)
            q_h = qg.sum(2)[0]
            kfar = k_real[far_lo:far_hi]

            def rec_of(sc):
                sel = torch.topk(sc, k2_far, dim=-1).indices + far_lo
                m = torch.zeros(Hkv, t + 1, device=dev)
                m.scatter_(1, sel, 1.0)
                return ((oh * m).sum(1) / k2_far).mean().item()

            # 隔离 fp32 / 隔离 4bit（quant4_pack）
            q_p = torch.einsum("hdr,hd->hr", V, q_h)
            k_p = torch.einsum("hdr,thd->thr", V, kfar)
            g, s_, m_ = quant4_pack(k_p)
            iso_fp = rec_of(torch.einsum("hr,thr->ht", q_p, k_p))
            iso_4b = rec_of(torch.einsum("hr,thr->ht", q_p, kq_unpack(g, s_, m_)))
            # pipeline
            index = idxer.build_block_index(k_real[: t + 1])
            sel = idxer.select(index, q_t, t)
            m2 = torch.zeros(Hkv, t + 1, device=dev)
            m2.scatter_(1, sel[:, :k2_far], 1.0)
            pipe = ((oh * m2).sum(1) / k2_far).mean().item()
            rows.append((tname, layer, t, iso_fp, iso_4b, pipe))
            print(f"{tname:>12s} L{layer:02d} {t:6d} | iso_fp {iso_fp:.4f} iso_4b {iso_4b:.4f} pipe {pipe:.4f}")
        del d, k_real, q_real
        torch.cuda.empty_cache()

import statistics as st
for i, name in [(3, "iso_fp"), (4, "iso_4b"), (5, "pipe")]:
    print(f"{name}: {st.mean(r[i] for r in rows):.4f}")
