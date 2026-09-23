# sglang tli M2 验证：B' 分区 + D' 截断算法正确性 + fused L1 kernel 对拍
# 数据：/tmp/trace/qwen3-8b 真实 trace（Qwen3-8B 权重采集）
import sys

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
import torch

from sglang.srt.layers.attention.tli.config import TLIProfile
from sglang.srt.layers.attention.tli.indexer import TLIIndexer
from sglang.srt.layers.attention.tli.kernels import tli_l1_topk

lf = "/tmp/trace/qwen3-8b/lb_hotpotqa_0/layer03.pt"
d = torch.load(lf, map_location="cuda:0")
k, q, qpos, S = d["k"].float(), d["q"].float(), d["qpos"].cuda(), d["S"]
t = qpos[-1].item()
Hkv, H, D = k.shape[1], q.shape[1], k.shape[-1]
G = H // Hkv

# 真实分布
qg = q[-1:].reshape(1, Hkv, G, D)
s = torch.einsum("bhgd,chd->bhgc", qg, k).sum(-2) * (D**-0.5)
s = s.masked_fill(torch.arange(S, device="cuda:0").view(1, 1, S) > t, float("-inf"))
p = torch.softmax(s, dim=-1)[0]
total = p.sum().item()

# ---- 1. 算法：非跳层（B' 分区）----
prof = TLIProfile()
idx = TLIIndexer(prof, head_dim=D).to("cuda:0")
index = idx.build_block_index(k)
sel = idx.select(index, q[-1:].reshape(1, H, D).cuda(), t)
cov = p.gather(1, sel).sum().item() / total
print(f"[非跳层 B' 分区] 选中 {sel.shape[-1]} tok, mass cov = {cov:.5f}")
assert cov > 0.99, f"mass coverage {cov} < 0.99"

# ---- 2. 算法：跳层（D' 截断）----
idx.skip_far = True
sel2 = idx.select(index, q[-1:].reshape(1, H, D).cuda(), t)
cov2 = p.gather(1, sel2).sum().item() / total
print(f"[跳层 D'] 选中 {sel2.shape[-1]} tok, mass cov = {cov2:.5f}")
# L03 是 far-heavy 层（far mass 0.155），跳层应损失 far（这是预期语义）
print(f"  （L03 far-heavy：D' 跳层预期保留 ~0.84，实测 {cov2:.4f}）")
idx.skip_far = False

# ---- 3. fused L1 kernel 对拍（非跳层）----
q_sub = q[-1:, ..., idx.idx1].reshape(H, -1).contiguous()  # [H, d']
kmin, kmax = index["kmin"].contiguous(), index["kmax"].contiguous()
nblk = index["nblk"]
last_blk = t // prof.block_size
out = tli_l1_topk(q_sub, kmin, kmax, prof.k1_blocks, 2, 2, last_blk)
tri_ids = torch.nonzero(out[0]).squeeze(1)
# eager 参考（同语义：无剔除 + 当前块强制）
qg_sub = q_sub.reshape(Hkv, G, -1).sum(1)
sc = (torch.einsum("hd,nhd->hn", qg_sub.clamp(min=0), kmax)
      + torch.einsum("hd,nhd->hn", qg_sub.clamp(max=0), kmin))[0]
eager_ids = torch.topk(sc, prof.k1_blocks).indices
eager_ids = torch.cat([eager_ids, torch.tensor([last_blk], device="cuda:0")])
inter = len(set(tri_ids.tolist()) & set(eager_ids.tolist()))
print(f"[fused L1 kernel] triton 块数={len(tri_ids)} (K1={prof.k1_blocks}+当前块), "
      f"与 eager id 交集 = {inter}/{len(eager_ids)}")
assert inter >= len(eager_ids) - 3, "fused kernel 块选择与 eager 偏差过大"

print("\nM2 验证 PASS")
