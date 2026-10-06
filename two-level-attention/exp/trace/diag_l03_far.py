# L03 诊断：远端簇分数 topk 的 mass 捕获 vs oracle（真实 q·k topk）
import sys
import argparse
import torch

sys.path.insert(0, "/home/wangyuanshuo02/two-level-attention")

parser = argparse.ArgumentParser()
from sparse_attn.arguments import add_sparse_attn_args
add_sparse_attn_args(parser)
args = parser.parse_args([
    "--method", "tli", "--tia_level2_topk", "1024", "--tia_level2_cmp_ratio", "4",
])

from sparse_attn.indexer import indexer_type_dict
TLI = indexer_type_dict["tli"]

lf = "/tmp/trace/qwen3-8b/lb_hotpotqa_0/layer03.pt"
d = torch.load(lf, map_location="cuda:0")
k, q, qpos, S = d["k"].float(), d["q"].float(), d["qpos"].cuda(), d["S"]
t = qpos[-1].item()
Hkv, H, D = k.shape[1], q.shape[1], k.shape[-1]
G = H // Hkv

# 真实分布（与 debug 脚本一致：group 求和）
qg = q[-1:].reshape(1, Hkv, G, D)
s = torch.einsum("bhgd,chd->bhgc", qg, k).sum(-2) * (D ** -0.5)
s = s.masked_fill(torch.arange(S, device="cuda").view(1, 1, S) > t, float("-inf"))
p = torch.softmax(s, dim=-1)[0]  # [Hkv, S]
total = p.sum().item()

far_lo, far_hi = 128, S - 2048
pm = p.mean(0)
far_mass_total = pm[far_lo:far_hi].sum().item()
near_mass = pm[:far_lo].sum().item() + pm[far_hi:].sum().item()
print(f"S={S} far区=[{far_lo},{far_hi}) far_mass={far_mass_total/total:.4f} near_mass={near_mass/total:.4f}")

# ---- oracle：真实 q·k 分数 far 区 top-512 ----
K2 = 512
true_far_score = s[0][:, far_lo:far_hi]  # [Hkv, Tfar]
i_oracle = torch.topk(true_far_score, K2, dim=-1).indices + far_lo
m_oracle = p.gather(1, i_oracle).sum().item() / total
print(f"oracle far top-{K2}: 总mass={m_oracle:.4f} far捕获={ (m_oracle - near_mass/total) / (far_mass_total/total):.3f}")

# ---- TLI 簇分数 top-512 ----
idx = TLI(args)
idx.layer_idx = 3
k_in = k.unsqueeze(0).to(torch.bfloat16)
q_in = q[-1:].reshape(1, 1, H, D).to(torch.bfloat16)
cu = torch.tensor([0, S], device="cuda", dtype=torch.int32)
mask, _ = idx.prepare_mask(q_in, torch.tensor([t], device="cuda"), k_in, cu)

far_tok_score = idx._far_token_score(idx._last_q)  # [Hkv, Tfar]
i_c = torch.topk(far_tok_score, K2, dim=-1).indices + idx._km_far_lo
m_c = p.gather(1, i_c).sum().item() / total
print(f"cluster far top-{K2}: 总mass={m_c:.4f} far捕获={ (m_c - near_mass/total) / (far_mass_total/total):.3f}")

# TLI 实际 mask 覆盖
m_tli = p[mask[0, 0]].sum().item() / total
print(f"TLI mask 总mass={m_tli:.4f}")

# 诊断1：簇分数与真实分数的 rank 相关性
corr = torch.corrcoef(torch.stack([
    far_tok_score.mean(0), true_far_score.mean(0)
]))[0, 1].item()
print(f"簇分数 vs 真实分 token 级 corr={corr:.3f}")

# 诊断2：top-512 重叠率
ov = len(set(i_c[0].tolist()) & set(i_oracle[0].tolist()))
print(f"Hkv0 top-{K2} 重叠: {ov}/{K2}")

# 诊断3：q 用 group-sum 但没乘 softmax_scale —— 检查量纲
print(f"簇分数范围: [{far_tok_score.min():.2f}, {far_tok_score.max():.2f}], 真实: [{true_far_score.min():.2f}, {true_far_score.max():.2f}]")

# 诊断4：kmeans 是对 subspace d'=32 聚的；检查全维 vs 子空间聚类质量差
idx.clear()
