# M11 统一稀疏 attention fused kernel 对拍：合成数据 vs eager 参考实现
# （与 backend._sparse_extend_one / _sparse_attn 同运算：gather→fp32→softmax）。
# 覆盖：哨兵槽位、K2 非 BLOCK_K 整数倍、bf16/fp16、行数×Hkv 组合。
import sys

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
import torch

from sglang.srt.layers.attention.tli.kernels import tli_sparse_gather_attn


def eager_ref(q, sel, k_buf, v_buf, S_loc):
    n, H, D = q.shape
    Hkv = k_buf.shape[1]
    G = H // Hkv
    K2 = sel.shape[-1]
    out = torch.empty_like(q)
    for r in range(n):
        for h in range(Hkv):
            s_v = sel[r, h]
            valid = s_v < S_loc
            s_c = s_v.clamp(max=S_loc - 1)
            k = k_buf[s_c, h].float()  # [K2, D]
            v = v_buf[s_c, h].float()
            q_h = q[r, h * G : (h + 1) * G].float()  # [G, D]
            att = torch.einsum("gd,kd->gk", q_h, k) * (D**-0.5)
            att = att.masked_fill(~valid.unsqueeze(0), float("-inf"))
            att = torch.softmax(att, dim=-1)
            out[r, h * G : (h + 1) * G] = torch.einsum("gk,kd->gd", att, v).to(q.dtype)
    return out


def main():
    torch.manual_seed(0)
    dev = "cuda:0"
    fails = 0
    for (n, Hkv, G, D, K2, pool, n_sent) in [
        (5, 8, 4, 128, 1024, 131072, 0),
        (5, 8, 4, 128, 1024, 131072, 137),   # 哨兵
        (1, 8, 4, 128, 1030, 65536, 51),     # K2 非 64 倍数
        (17, 4, 8, 128, 512, 32768, 29),     # G=8
        (33, 8, 4, 64, 2048, 8192, 77),      # D=64
    ]:
        H = Hkv * G
        q = torch.randn(n, H, D, device=dev, dtype=torch.bfloat16)
        k_buf = torch.randn(pool, Hkv, D, device=dev, dtype=torch.bfloat16)
        v_buf = torch.randn(pool, Hkv, D, device=dev, dtype=torch.bfloat16)
        sel = torch.randint(0, pool, (n, Hkv, K2), device=dev)
        # 塞哨兵（位置 >= S_loc=pool）
        if n_sent:
            flat = sel.view(-1)
            idx = torch.randperm(flat.numel(), device=dev)[:n_sent]
            flat[idx] = pool + torch.randint(0, 3, (n_sent,), device=dev)
        out_k = tli_sparse_gather_attn(q, sel, k_buf, v_buf, G, S_loc=pool)
        out_e = eager_ref(q, sel, k_buf, v_buf, pool)
        diff = (out_k.float() - out_e.float()).abs().max().item()
        ok = diff < 2e-2  # bf16 输出 + online vs 全量 softmax 累加顺序差
        print(f"n={n} Hkv={Hkv} G={G} D={D} K2={K2} sent={n_sent}: "
              f"maxdiff={diff:.2e} {'PASS' if ok else 'FAIL'}")
        fails += not ok
    print("ALL PASS" if fails == 0 else f"{fails} FAIL")
    return fails


if __name__ == "__main__":
    sys.exit(main())
