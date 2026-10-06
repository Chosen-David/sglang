# sup_wsvd 基校准（sglang 主表重跑用）：从 hotpotqa_0 trace 离线校准注意力加权 PCA 基，
# 导出 SGLANG_TLI_PROJ_BASIS 兼容格式 .pt：fp32 [n_layers, Hkv, D=128, r]。
# 数学：基作用在 NoPE 尾维 32 子空间内（B_sub [Hkv,32,r]），
#   full-D 基 = E @ B_sub，其中 E 是 128→32 的子空间选择（尾维 16+16），
#   即 W[:, i] = basis_sub 作用回全维：W[h][:32维处] ... 直接构造 [128, r]：
#   W_full[h][:, j] = sum_k E_sel[k] * B_sub[h][k, j]（E_sel[k]=1 当 k ∈ D2I）
#   等价于把 B_sub 行 scatter 到全维 128 位置。
# r ∈ {4, 8}（E64f 冠军 d8 / 极限 d4）各导一份。
import json

import torch

TRACE = "/tmp/trace/qwen3-8b"
CAL = "lb_hotpotqa_0"
D2I = list(range(48, 64)) + list(range(112, 128))
SINK, SWA, NEAR_BAND = 128, 1024, 4096
NQ_CAL = 8
OUT = "/home/wangyuanshuo02/two-level-attention/exp/trace/results/sup_wsvd_basis_qwen3-8b.pt"


def calibrate(lf):
    d = torch.load(lf, map_location="cpu", weights_only=False)
    k, q, qpos, S = d["k"].float(), d["q"].float(), d["qpos"], d["S"]
    Hkv, H, D = k.shape[1], q.shape[1], k.shape[-1]
    G = H // Hkv
    t = int(qpos[-1])
    mid_hi = t + 1 - SWA
    far_lo, far_hi = SINK, mid_hi - NEAR_BAND
    if far_hi - far_lo < 8192:
        return None
    idx = torch.tensor(D2I)
    k32 = k[..., idx]
    C = torch.zeros(Hkv, 32, 32)
    for ri in range(NQ_CAL):
        q_head = q[-NQ_CAL + ri].reshape(Hkv, G, D).sum(1)
        s = torch.einsum("hd,shd->hs", q_head, k) * (D ** -0.5)
        s = s.masked_fill(torch.arange(S).view(1, -1) > int(qpos[-NQ_CAL + ri]), float("-inf"))
        p = torch.softmax(s, dim=-1)[:, far_lo:far_hi]
        kf = k32[far_lo:far_hi]
        C += torch.einsum("ht,thd,the->hde", p, kf, kf)
    bases = {}
    for r in (4, 8, 32):
        B = []
        for h in range(Hkv):
            eig, vec = torch.linalg.eigh(C[h])
            B.append(vec[:, torch.argsort(eig, descending=True)][:, :r])   # [32, r]
        B = torch.stack(B)                                                  # [Hkv,32,r]
        # scatter 回全维：W_full[h][D2I[k], j] = B[h][k, j]
        W = torch.zeros(Hkv, D, r)
        W[:, torch.tensor(D2I), :] = B
        bases[r] = W                                                        # [Hkv,128,r]
    return bases


def main():
    n_layers = json.load(open(f"{TRACE}/{CAL}/meta.json"))["n_layers"]
    layers = list(range(0, n_layers, max(1, n_layers // 12)))
    out4, out8, out32 = [], [], []
    ref = None
    for li in layers:
        b = calibrate(f"{TRACE}/{CAL}/layer{li:02d}.pt")
        if b is None:
            b = ref  # far 区不足的层复用前一层基
        ref = b
        out4.append(b[4]); out8.append(b[8]); out32.append(b[32])
    # 展开到全部 n_layers（最近邻复制）
    def expand(out):
        full = []
        for li in range(n_layers):
            j = min(range(len(layers)), key=lambda x: abs(layers[x] - li))
            full.append(out[j])
        return torch.stack(full)   # [n_layers, Hkv, 128, r]
    payload = {"r4": expand(out4), "r8": expand(out8), "r32": expand(out32), "layers_cal": layers, "cal": CAL}
    torch.save(payload, OUT)
    print(f"saved {OUT}: r4{list(payload['r4'].shape)} r8{list(payload['r8'].shape)} r32{list(payload['r32'].shape)}")


if __name__ == "__main__":
    main()
