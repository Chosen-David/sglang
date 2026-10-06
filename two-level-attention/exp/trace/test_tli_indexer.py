# tli 单测对拍：TLIIndexer（sglang 集成版）vs e3b 参考口径
# 期望：mass coverage ≈ e3b 的 0.9967–1.0000；D' 跳过层预期 far 缺失但 sink/near 保留
import sys
import torch
import glob

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
from sglang.srt.layers.attention.tli import TLIIndexer, TLIProfile

TRACE = "/tmp/trace/qwen3-8b/needle32k"
PROFILE = TLIProfile()
PROFILE.dense_threshold = 0  # 测试强制走两级
indexer = TLIIndexer(PROFILE, head_dim=128).to("cuda")

covs, far_c = [], []
for lf in sorted(glob.glob(f"{TRACE}/layer*.pt"))[:8]:
    d = torch.load(lf, map_location="cuda:0")
    k, q, qpos, S = d["k"].float(), d["q"].float(), d["qpos"].cuda(), d["S"]
    t = qpos[-1].item()
    Hkv, H, D = k.shape[1], q.shape[1], k.shape[-1]
    G = H // Hkv
    # dense 参考
    qg = q[-1:].reshape(1, Hkv, G, D)
    s = torch.einsum("bhgd,chd->bhgc", qg, k).sum(-2) * (D ** -0.5)
    s = s.masked_fill(torch.arange(S, device="cuda").view(1, 1, S) > t, float("-inf"))
    p = torch.softmax(s, dim=-1)
    total = p.sum().item()
    # TLIIndexer
    index = indexer.build_block_index(k)
    sel = indexer.select(index, q[-1:].float(), t)  # [Hkv, K2]
    mass = p[0].gather(-1, sel).sum().item()
    covs.append(mass / total)
    # far 部分（64 到 t-2048）覆盖
    pm = p.mean(dim=(0, 1))
    far_mask = (sel >= 64) & (sel < t - 2048)
    far_c.append(pm[sel[far_mask]].sum().item() / max(1e-9, pm[64:t - 2048].sum().item()))
    del k, q, s, p, index
    torch.cuda.empty_cache()

import statistics as st
print(f"TLIIndexer mass coverage: {st.mean(covs):.4f} (min {min(covs):.4f})")
print(f"far mass 覆盖: {st.mean(far_c):.3f}")
assert st.mean(covs) > 0.99, "mass coverage 应 ≥0.99（对齐 e3b 实测）"

# D' 测试：skip_far=True 后 far 应被砍掉，near/sink 保留
indexer.skip_far = True
lf = sorted(glob.glob(f"{TRACE}/layer*.pt"))[0]
d = torch.load(lf, map_location="cuda:0")
k, q, qpos, S = d["k"].float(), d["q"].float(), d["qpos"].cuda(), d["S"]
t = qpos[-1].item()
Hkv, H, D = k.shape[1], q.shape[1], k.shape[-1]
G = H // Hkv
qg = q[-1:].reshape(1, Hkv, G, D)
s = torch.einsum("bhgd,chd->bhgc", qg, k).sum(-2) * (D ** -0.5)
s = s.masked_fill(torch.arange(S, device="cuda").view(1, 1, S) > t, float("-inf"))
p = torch.softmax(s, dim=-1)
pm = p.mean(dim=(0, 1))
index = indexer.build_block_index(k)
sel = indexer.select(index, q[-1:].float(), t)
far_sel = (sel >= 64) & (sel < t - 2048)
print(f"D' skip_far 后 far 选择 token 数: {far_sel.sum().item()}（应大幅减少）")
assert far_sel.sum().item() < 128, "D' 应基本清空 far 候选"
print("PASS: TLIIndexer 对拍与 D' 行为均符合预期")
