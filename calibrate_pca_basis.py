# M9：PCA 投影基离线校准（同 D' 哲学：trace 校准一次，运行期只读）
# 每 (layer, kv-head) 对 far 区 K 做 SVD 取 top-r 主成分 → basis [L, Hkv, D, r]
# 校准集 = hotpotqa_0（已验证跨任务迁移仅 −0.024，test_tli_proj_sweep.py）
# far 区口径与打分实验一致：[128, S-2048)
import json
import os
import sys

import torch

dev = "cuda:0"
SRC = "/tmp/trace/qwen3-8b/lb_hotpotqa_0"
R = int(os.environ.get("SGLANG_TLI_PROJ_R", "16"))
OUT = f"/home/wangyuanshuo02/sglang/tli_pca_basis_r{R}.pt"

layers = sorted(
    int(f[5:7]) for f in os.listdir(SRC) if f.startswith("layer") and f.endswith(".pt")
)
print(f"layers={len(layers)} R={R} src={SRC}")

basis = []
far_stats = []
for li in layers:
    d = torch.load(f"{SRC}/layer{li:02d}.pt", map_location=dev)
    k = d["k"].float()  # [S, Hkv, D]
    S, Hkv, D = k.shape
    far = k[128 : max(129, S - 2048)]  # [Tfar, Hkv, D]
    vb = []
    for h in range(Hkv):
        _, _, Vt = torch.linalg.svd(far[:, h], full_matrices=False)
        vb.append(Vt[:R].T)  # [D, R]
    basis.append(torch.stack(vb))  # [Hkv, D, R]
    far_stats.append({"layer": li, "S": S, "tfar": far.shape[0]})
    del d, k, far
    torch.cuda.empty_cache()

B = torch.stack(basis).contiguous()  # [L, Hkv, D, R] fp32
torch.save(B, OUT)
json.dump(far_stats, open(f"/home/wangyuanshuo02/sglang/tli_pca_basis_r{R}_meta.json", "w"), indent=1)
print(f"saved {OUT} shape={tuple(B.shape)} dtype={B.dtype}")
print(f"显存占用（常驻）: {B.numel() * 4 / 1e6:.2f} MB")
