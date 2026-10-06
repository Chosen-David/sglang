# TLIIndexer（sparse_attn 注册版）冒烟测试：用真实 trace 走 prepare_mask 全流程
import sys
import argparse
import torch
import glob

sys.path.insert(0, "/home/wangyuanshuo02/two-level-attention")

parser = argparse.ArgumentParser()
from sparse_attn.arguments import add_sparse_attn_args
add_sparse_attn_args(parser)
args = parser.parse_args([
    "--method", "tli", "--tia_level2_topk", "1024", "--tia_level2_cmp_ratio", "4",
])
print(f"method={args.method} subspace={args.tli_enable_subspace} kmeans={args.tli_enable_kmeans} "
      f"skip={args.tli_enable_layer_skip} far_blocks={args.tli_far_blocks}")

from sparse_attn.indexer import indexer_type_dict
TLI = indexer_type_dict["tli"]
TIA = indexer_type_dict["tia"]

# ---- 逐层对比 TLI vs TIA mask 质量（mass 覆盖）----
TRACE = "/tmp/trace/qwen3-8b/lb_hotpotqa_0"
layer_files = sorted(glob.glob(f"{TRACE}/layer*.pt"))
tli_cov, tia_cov, ious = [], [], []
for lf in layer_files[:6] + layer_files[17:19]:  # 含跳层和非跳层
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
    layer_id = int(lf.split("layer")[1].split(".")[0])
    # k 布局：patch 传 [1, S, Hkv, D]；q [1, 1, H, D]
    k_in = k.unsqueeze(0).to(torch.bfloat16)
    q_in = q[-1:].reshape(1, 1, H, D).to(torch.bfloat16)
    cu = torch.tensor([0, S], device="cuda", dtype=torch.int32)
    outs = {}
    for name, cls in [("tli", TLI), ("tia", TIA)]:
        idx = cls(args)
        idx.layer_idx = layer_id
        mask, blk = idx.prepare_mask(q_in, torch.tensor([t], device="cuda"), k_in, cu)
        # mask: [1,1,Hkv,S] bool（block_size=1 → token 级）；p[0] 是 [H,S] q-head 级
        p_kv = p[0]  # [Hkv, S]（einsum 后已是 kv-head 级）
        mass = p_kv[mask[0, 0]].sum().item()
        outs[name] = mask
        if name == "tli":
            tli_cov.append(mass / total)
        else:
            tia_cov.append(mass / total)
        idx.clear()
    a, b = outs["tia"][0, 0], outs["tli"][0, 0]
    ious.append((a & b).sum().item() / (a | b).sum().item())
    del k, q, s, p, d
    torch.cuda.empty_cache()

import statistics as st
print(f"层 {layer_id}: TLI mass cov mean={st.mean(tli_cov):.4f} min={min(tli_cov):.4f}")
print(f"TIA mass cov mean={st.mean(tia_cov):.4f} min={min(tia_cov):.4f}")
print(f"mask IoU(TLI,TIA) mean={st.mean(ious):.3f}")
assert st.mean(tli_cov) > 0.985, "TLI 覆盖率应≈TIA（trace 实测口径 0.997+）"
print("PASS")
