# TLI 维度压缩敏感性扫描（trace 离线，真实 Qwen3-8B 权重）：
#   - L2 细筛维数 2δ（SGLANG_TLI_DELTA，kq 存储 nd2+8 B/token-head）
#   - L1 粗筛维数 d'（SGLANG_TLI_COARSE_DIM，kmin/kmax 2*d'*4B/block-head）
# 动机：聚类路线（另一项目 opt5）显示 128→32 压缩后聚类质量不掉；
# TLI 的两级打分维数是否同样可压——L2 4bit 细筛压到 2δ=16/8 可进一步
# 减半 kq 存储（40→24/16 B）与 L2 打分 GEMV；L1 压到 d'=16 可减半
# kmin/kmax 与 L1 GEMV。
# 口径：行级总 mass coverage + 剩余 mass coverage（竞争区 = 去除 sink
# [0,128) + 滑窗 [t-127,t] 强制位——强制位质量与索引器无关，总口径被
# sink mass 稀释到 0.99+ 区分度低）；竞争区占比一并报告。
# 数据：/tmp/trace/qwen3-8b/lb_hotpotqa_0（L03/L05 far-heavy，其余典型）
import sys

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
import json
import os

import torch

from sglang.srt.layers.attention.tli.config import TLIProfile
from sglang.srt.layers.attention.tli.indexer import TLIIndexer

dev = "cuda:0"
TRACE = "/tmp/trace/qwen3-8b/lb_hotpotqa_0"
LAYERS = [3, 5, 10, 20, 33]
CONFIGS = [
    (16, 32),  # 基线（默认 delta=16, coarse_dim=32）
    (8, 32),   # 仅压 L2
    (4, 32),
    (16, 16),  # 仅压 L1
    (8, 16),   # 双压
]
OUT = "/home/wangyuanshuo02/sglang/tli_dim_sweep.json"

rows_out = []
print(f"{'cfg':>10} {'layer':>5} {'t':>6} {'cov总':>7} {'cov剩余':>8} {'竞争区占比':>9}")
for layer in LAYERS:
    d = torch.load(f"{TRACE}/layer{layer:02d}.pt", map_location=dev)
    k_real = d["k"].float()  # [S, Hkv, D]
    q_real = d["q"].float()  # [nq, H, D]
    qpos = d["qpos"].cuda()
    S, Hkv, D = k_real.shape
    H = q_real.shape[1]
    G = H // Hkv
    for delta, coarse in CONFIGS:
        os.environ["SGLANG_TLI_DELTA"] = str(delta)
        os.environ["SGLANG_TLI_COARSE_DIM"] = str(coarse)
        prof = TLIProfile()
        idxer = TLIIndexer(prof, head_dim=D).to(dev)
        idx = idxer.build_block_index(k_real)
        for t in [S - 1, S // 2]:
            in_range = torch.nonzero(qpos <= t).squeeze(1)
            q_t = q_real[int(in_range[-1]) : int(in_range[-1]) + 1]
            sel = idxer.select(idx, q_t, t)  # [Hkv, K2]（eager，全有效位）
            qg = q_t.reshape(1, Hkv, G, D)
            s = (
                torch.einsum("bhgd,chd->bhgc", qg, k_real[: t + 1]).sum(-2)
                * (D**-0.5)
            )
            p_ = torch.softmax(s, dim=-1)[0]  # [Hkv, t+1]
            # 选中位 one-hot（scatter 天然去重 far/near/滑窗交叠）
            selm = torch.zeros(Hkv, t + 1, device=dev)
            selm.scatter_(1, sel.clamp(max=t), 1.0)
            cov = (p_ * selm).sum().item() / p_.sum().item()
            sink_hi = prof.sink_blocks * prof.block_size
            sw_lo = max(0, t - prof.sliding_window + 1)
            contested = (
                (torch.arange(t + 1, device=dev) >= sink_hi)
                & (torch.arange(t + 1, device=dev) < sw_lo)
            )
            denom = p_[:, contested].sum().item()
            res_frac = denom / p_.sum().item()
            cov_res = (
                (p_ * selm * contested.view(1, -1)).sum().item() / denom
            )
            print(f"δ={delta:2d},d'={coarse:2d} L{layer:02d} t={t:6d} "
                  f"{cov:7.5f} {cov_res:8.5f} {res_frac:9.5f}")
            rows_out.append({
                "layer": layer, "t": t, "delta": delta, "coarse_dim": coarse,
                "cov_total": round(cov, 5), "cov_residual": round(cov_res, 5),
                "residual_frac": round(res_frac, 5),
                "kq_bytes_per_token_head": 2 * delta + 8,
            })
json.dump(rows_out, open(OUT, "w"), indent=1)
print(f"\nsaved -> {OUT}")

# 汇总：每配置跨层聚合
print("\n==== 汇总（5 层 × 2 个 t 共 10 行）====")
for delta, coarse in CONFIGS:
    rs = [r for r in rows_out if r["delta"] == delta and r["coarse_dim"] == coarse]
    tot = sum(r["cov_total"] for r in rs) / len(rs)
    res = sum(r["cov_residual"] for r in rs) / len(rs)
    res_mn = min(r["cov_residual"] for r in rs)
    print(f"δ={delta:2d} d'={coarse:2d}: cov总 mean={tot:.5f} | "
          f"cov剩余 mean={res:.5f} min={res_mn:.5f} | "
          f"kq {2*delta+8}B/token-head（基线 40B）")
