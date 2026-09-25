# M9 debug：逐环节定位 pca16 pipeline far recall 崩溃（0.126 vs 预期 ~0.53）
import sys

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
import torch

from sglang.srt.layers.attention.tli.config import TLIProfile
from sglang.srt.layers.attention.tli.indexer import TLIIndexer, quant4, quant4_pack, kq_unpack

dev = "cuda:0"
B = torch.load("/home/wangyuanshuo02/sglang/tli_pca_basis_r16.pt", map_location=dev)
d = torch.load("/tmp/trace/qwen3-8b/lb_hotpotqa_0/layer03.pt", map_location=dev)
k_real = d["k"].float()
q_real = d["q"].float()
qpos = d["qpos"].cuda()
S, Hkv, D = k_real.shape
H = q_real.shape[1]
G = H // Hkv
t = S - 1

in_range = torch.nonzero(qpos <= t).squeeze(1)
qi = int(in_range[-1])
q_t = q_real[qi : qi + 1]
qg = q_t.reshape(1, Hkv, G, D)
s_full = torch.einsum("bhgd,chd->bhgc", qg, k_real[: t + 1]).sum(-2)
far_lo, near_len = 128, 2048
far_hi = max(far_lo, t + 1 - near_len)
k2_far = 256
oracle = torch.topk(s_full[0, :, far_lo:far_hi], k2_far, dim=-1).indices + far_lo
oh = torch.zeros(Hkv, t + 1, device=dev)
oh.scatter_(1, oracle, 1.0)
kfar = k_real[far_lo:far_hi]
V = B[3]  # [Hkv, D, r]
r = V.shape[-1]
q_h = qg.sum(2)[0]  # [Hkv, D] GQA-sum

# (a) 验证公式：隔离口径 fp32
q_p = torch.einsum("hdr,hd->hr", V, q_h)
k_p = torch.einsum("hdr,thd->thr", V, kfar)
sc = torch.einsum("hr,thr->ht", q_p, k_p)
sel = torch.topk(sc, k2_far, dim=-1).indices + far_lo
m = torch.zeros(Hkv, t + 1, device=dev)
m.scatter_(1, sel, 1.0)
rec_a = ((oh * m).sum(1) / k2_far).mean().item()
print(f"(a) 隔离 fp32（验证公式）: {rec_a:.4f}")

# (b) 隔离 + kq 4bit（quant4_pack 路径）
gq, gsc, gmn = quant4_pack(k_p)
k_pq = kq_unpack(gq, gsc, gmn)
sc = torch.einsum("hr,thr->ht", q_p, k_pq)
sel = torch.topk(sc, k2_far, dim=-1).indices + far_lo
m.zero_()
m.scatter_(1, sel, 1.0)
rec_b = ((oh * m).sum(1) / k2_far).mean().item()
print(f"(b) 隔离 + quant4_pack(k_p): {rec_b:.4f}")

# (c) kq 写入路径核对：_k_refine == kfar @ V 逐位？
idxer = TLIIndexer(TLIProfile(), head_dim=128, basis=V).to(dev)
kr = idxer._k_refine(kfar)
print(f"(c) _k_refine vs kfar@V: max diff {(kr - k_p).abs().max().item():.3e}")

# (d) q 路径核对：_q_refine+Gsum == V^T q_h 逐位？
q2 = idxer._q_refine(q_t).reshape(1, Hkv, G, r).sum(2)[0]
print(f"(d) _q_refine Gsum vs V^T q_h: max diff {(q2 - q_p).abs().max().item():.3e}")

# (e) 完整 pipeline select 的 far recall + 把 pipeline 的 L2 分数和隔离分数对齐检查
prof = TLIProfile()
index = idxer.build_block_index(k_real[: t + 1])
sel_p = idxer.select(index, q_t, t)
m.zero_()
m.scatter_(1, sel_p[:, :k2_far], 1.0)
rec_e = ((oh * m).sum(1) / k2_far).mean().item()
print(f"(e) pipeline select: {rec_e:.4f}")

# (f) 用同 index 复算 L2 分数（绕开 select 内部），cand = oracle far 所在块全展开
blk = oracle // 64
onehot = torch.zeros(Hkv, index["nblk"], dtype=torch.bool, device=dev)
onehot.scatter_(1, blk, True)
sel_mask = onehot.repeat_interleave(64)[: t + 1]
cand_pos = torch.nonzero(sel_mask).squeeze(1)
kq_h = kq_unpack(index["kq_q"][cand_pos], index["kq_sc"][cand_pos], index["kq_mn"][cand_pos])
s2 = torch.einsum("hd,thd->ht", q2, kq_h)
fine = torch.full((Hkv, t + 1), float("-inf"), device=dev)
fine[:, cand_pos] = s2
i_f = torch.topk(fine[:, far_lo:far_hi], k2_far, dim=-1).indices + far_lo
m.zero_()
m.scatter_(1, i_f, 1.0)
print(f"(f) 手动 L2（oracle 块展开 + index kq）: {((oh * m).sum(1) / k2_far).mean().item():.4f}")

# (g) 手动 L2 但 kq 用 (b) 的隔离表（不含 pipeline 写入）
fine2 = torch.full((Hkv, t + 1), float("-inf"), device=dev)
pos_far = torch.arange(far_lo, far_hi, device=dev)
fine2[:, pos_far] = sc if False else torch.einsum("hr,thr->ht", q_p, k_pq)
i_f = torch.topk(fine2[:, far_lo:far_hi], k2_far, dim=-1).indices + far_lo
m.zero_()
m.scatter_(1, i_f, 1.0)
print(f"(g) 隔离表 + oracle 块限制: {((oh * m).sum(1) / k2_far).mean().item():.4f}")
