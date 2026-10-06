# E64c-1：选择阶段 kernel 成本模型（合成 microbench，标注：合成数据，双口径铁律之 kernel 侧）
# 比较四种 (far_method × near_method) 组合在选择阶段的 GPU 成本（S=64K，单层单请求口径）：
#   mavg（minmax 粗筛+细筛）: build O(S·d')/64 块 minmax + GEMV 块分 + 池内 token GEMV topk
#   cavg（cluster+avg）    : 贪心 build（离线摊销 decode，这里测在线成本上限）+ 簇分 + token 簇分 topk
#   aavg（avg+avg）        : avg 块分 GEMV + token GEMV topk
# 维度臂：d' ∈ {4, 8, 32}（对应 E65/E65b 降维结论的算力兑现）
# 输出：各组件 ms + 总 ms（cudaEvent 计时，H20，GPU 合成张量）
import json

import torch

import os
DEV = "cuda:" + os.environ.get("E64C_GPU", "1")   # GPU1 与 E4d 分时复用（E4d util 24% 有余量）
S = 65536
Hkv, D = 8, 128
BS = 64
D2I = list(range(48, 64)) + list(range(112, 128))
DIMS = [4, 8, 32]
PAGES = 128
TOK_BUD = 2048
OUT = "/home/wangyuanshuo02/two-level-attention/exp/trace/results/e64c_cost_model.json"


def timed(fn, warm=3, rep=10):
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
    k = torch.randn(S, Hkv, D, device=DEV, generator=g, dtype=torch.bfloat16)
    q = torch.randn(Hkv, D, device=DEV, generator=g, dtype=torch.bfloat16)
    idx = torch.tensor(D2I[:32] if d_dim == 32 else D2I[:d_dim], device=DEV)
    nblk = S // BS
    res = {}
    # 1) 块统计 build（minmax 或 avg，一次性 O(S·d)）
    kc = k[..., idx].reshape(nblk, BS, Hkv, -1).float()
    res["build_minmax_ms"] = timed(lambda: (kc.amin(1), kc.amax(1)))
    res["build_avg_ms"] = timed(lambda: kc.mean(1))
    # 2) 块打分 GEMV（minmax 拆正负 / avg）
    kmin, kmax = kc.amin(1), kc.amax(1)
    qf = q[..., idx].float()
    res["score_minmax_ms"] = timed(lambda: (
        torch.einsum("hd,nhd->hn", qf.clamp(min=0), kmax) +
        torch.einsum("hd,nhd->hn", qf.clamp(max=0), kmin)))
    kavg = kc.mean(1)
    res["score_avg_ms"] = timed(lambda: torch.einsum("hd,nhd->hn", qf, kavg))
    # 3) 页 topk
    sc = torch.einsum("hd,nhd->hn", qf, kavg)
    res["topk_page_ms"] = timed(lambda: torch.topk(sc, PAGES, dim=-1))
    # 4) 池内 token GEMV（PAGES×64 token × d 维，gather+GEMV 合成成本）
    ib = torch.topk(sc, PAGES, dim=-1).indices                       # [Hkv, PAGES]
    pool_idx = (ib.unsqueeze(-1) * BS + torch.arange(BS, device=DEV))  # [Hkv, PAGES, 64]
    flat = pool_idx.reshape(Hkv, -1)                                  # [Hkv, PAGES*64]

    def tok_gemm():
        # 逐 head gather（k[flat] 是跨维 gather，语义错；Hkv=8 循环成本可接受）
        return torch.stack([torch.einsum("d,thd->t", qf[h], k[flat[h]][..., idx].float()) for h in range(Hkv)])

    res["fine_gemm_ms"] = timed(tok_gemm)
    # 5) token topk
    ts = torch.stack([torch.einsum("d,thd->t", qf[h], k[flat[h]][..., idx].float()) for h in range(Hkv)])
    res["topk_token_ms"] = timed(lambda: torch.topk(ts, TOK_BUD, dim=-1))
    # 6) 贪心聚类在线成本上限（全 far 逐 token——实际有增量摊销，此处为上限）
    def greedy_full():
        k32 = k[..., torch.tensor(D2I, device=DEV)].float()   # [S,Hkv,32]
        k_n = k32.norm(dim=-1)
        sums = torch.zeros(Hkv, 4096, 32, device=DEV)
        for i in range(0, min(S, 8192)):   # 上限采样 8K token 外推
            xi = k32[i]
            cos = torch.einsum("hkd,hd->hk", sums, xi) / (sums.norm(dim=-1) * k_n[i].unsqueeze(-1) + 1e-9)
            _ = cos.argmax(-1)
        return _

    t_g = timed(greedy_full, warm=1, rep=2)
    res["greedy_online_8k_ms"] = t_g
    res["greedy_online_64k_extrap_ms"] = t_g * 8
    return res


def main():
    out = {}
    for d in DIMS:
        out[f"d{d}"] = bench(d)
        print(f"[d={d}] " + " ".join(f"{k}={v:.3f}" for k, v in out[f"d{d}"].items()), flush=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
