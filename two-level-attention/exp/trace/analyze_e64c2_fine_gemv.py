# E64c-2：细筛 token GEMV 修正版成本模型（合成 microbench，标注：合成数据，双口径铁律之 kernel 侧）
# E64c-1 遗留问题：细筛 GEMV 逐 head Python 循环 + einsum("d,thd->t") 路径导致 d4=1.86ms 反常高于 d32，
# 掩盖了降维的真实算力收益。本脚本用三条正确路径重测：
#   A) gather 后 batched bmm（[Hkv, T, d] × [Hkv, d, 1]）——常规路径
#   B) 先投影再 gather（把降维矩阵乘在 KV 上是离线摊销，在线 gather 的数据量就是 d 维）
#   C) 全量 K 先降维投影再 GEMV（无需 gather 的极端形态：S×d 与 q 的 GEMM）
# 输出各 d 的 ms + 与 d=32 的比例（降维收益曲线），cudaEvent 计时，H20 GPU1。
import json
import os

import torch

DEV = "cuda:" + os.environ.get("E64C_GPU", "0")
S = 65536
Hkv, D = 8, 128
BS = 64
D2I = list(range(48, 64)) + list(range(112, 128))
DIMS = [4, 8, 16, 32, 64, 128]
PAGES = 128
OUT = "/home/wangyuanshuo02/two-level-attention/exp/trace/results/e64c2_fine_gemv.json"


def timed(fn, warm=5, rep=20):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    st.record()
    for _ in range(rep):
        fn()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / rep


def bench(d_dim):
    g = torch.Generator(device=DEV).manual_seed(0)
    # 预降维后的 KV（模拟离线已存 d 维特征，这是生产形态：索引侧存的就是 d 维）
    k_d = torch.randn(S, Hkv, d_dim, device=DEV, generator=g, dtype=torch.bfloat16)
    q = torch.randn(Hkv, d_dim, device=DEV, generator=g, dtype=torch.bfloat16)
    nblk = S // BS
    res = {}
    # 页池 gather 索引（每 head 独立 top 页池）
    sc = torch.randn(Hkv, nblk, device=DEV, generator=g)
    ib = torch.topk(sc, PAGES, dim=-1).indices                 # [Hkv, PAGES]
    pool_idx = (ib.unsqueeze(-1) * BS + torch.arange(BS, device=DEV))  # [Hkv, PAGES, 64]
    flat = pool_idx.reshape(Hkv, -1)                            # [Hkv, PAGES*64]
    T = flat.shape[-1]
    # A) gather + batched bmm（页池 token 细筛，生产主路径）
    # k_d [S,Hkv,d] 展平 [Hkv*S,d] + per-head offset：pos=flat+h*S 一维索引 → [Hkv,T,d] 语义正确
    kt = k_d.permute(1, 0, 2).reshape(Hkv * S, -1).contiguous()
    pos = flat + (torch.arange(Hkv, device=DEV) * S).unsqueeze(-1)   # [Hkv, T]
    qf = q.float().unsqueeze(-1)                                # [Hkv, d, 1]
    def gemm_bmm():
        gathered = kt[pos]                                      # [Hkv, T, d]
        return torch.bmm(gathered.float(), qf).squeeze(-1)
    res["A_gather_bmm_ms"] = timed(gemm_bmm)
    # B) gather 单独计时（与打分分离，看 gather vs GEMV 占比）
    res["B_gather_only_ms"] = timed(lambda: kt[pos])
    # C) gather 后 einsum 批量（对照路径，一次 einsum 不循环）
    res["C_gather_einsum_ms"] = timed(lambda: torch.einsum("htd,hd->ht", kt[pos].float(), q.float()))
    # D) 全量 S×d GEMV（无 gather 极端形态：如果直接对全序列打分）
    res["D_full_gemv_ms"] = timed(lambda: torch.einsum("shd,hd->hs", k_d.float(), q.float()))
    return res


def main():
    out = {}
    for d in DIMS:
        out[f"d{d}"] = bench(d)
        print(f"[d={d}] " + " ".join(f"{k}={v:.3f}" for k, v in out[f"d{d}"].items()), flush=True)
    # 降维收益比例（相对 d=128）
    base = {k: out["d128"][k] for k in out["d128"]}
    for d in DIMS:
        out[f"d{d}"]["speedup_vs_d128_A"] = round(base["A_gather_bmm_ms"] / out[f"d{d}"]["A_gather_bmm_ms"], 2)
    json.dump(out, open(OUT, "w"), indent=1)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
