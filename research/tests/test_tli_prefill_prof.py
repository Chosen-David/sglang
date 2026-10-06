# M7 前置：prefill 开销归因（trace 级微基准，真实 Qwen3-8B 权重）
# forward_extend 单请求各阶段墙钟拆解：gather / build_block_index /
# select_batched（nq×S 打分+topk）/ _sparse_extend_one（稀疏前向）。
# S 梯度：5K / 10K / 20K / 40K（trace 复制扩展，K 权重真实）。
# 用法：CUDA_VISIBLE_DEVICES=1 PYTHONPATH=... python3 test_tli_prefill_prof.py
import sys

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
import time

import torch

from sglang.srt.layers.attention.tli.config import TLIProfile
from sglang.srt.layers.attention.tli.indexer import TLIIndexer

dev = "cuda:0"
d = torch.load("/tmp/trace/qwen3-8b/lb_hotpotqa_0/layer03.pt", map_location=dev)
k_real = d["k"].float()  # [S, Hkv, D]
q_real = d["q"].float()
qpos = d["qpos"].cuda()
S0, Hkv, D = k_real.shape
H = q_real.shape[1]
G = H // Hkv

prof = TLIProfile()
idxer = TLIIndexer(prof, head_dim=D).to(dev)


def bench(fn, n_warm=2, n_iter=5):
    for _ in range(n_warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iter):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n_iter


print(f"trace S0={S0}；prefill 各阶段（单请求，nq=S 整段一次性 extend）\n")
print(f"{'S':>6} {'gather':>8} {'build':>8} {'select':>8} {'sp_fwd':>8} {'total':>8} {'build占比':>9}")
for S in [5000, 10000, 20000, 40000]:
    # K 权重真实，token 重复扩展（trace 不足时 tile）
    if S <= S0:
        k = k_real[:S]
    else:
        reps = (S + S0 - 1) // S0
        k = k_real.repeat(reps, 1, 1)[:S]
    q = q_real[-256:].repeat(1, 1, 1)  # 仅计时口径，位置不影响成本
    nq = S  # 整段 prefill：nq = S
    q_b = q[0:1].repeat(nq, 1, 1)
    # phase 1: gather（模拟 k_buf[locs].float()）
    locs = torch.randperm(S + 512, device=dev)[:S]
    k_buf = torch.zeros(S + 512, Hkv, D, device=dev)
    k_buf[locs] = k
    t_g = bench(lambda: k_buf[locs].float())
    # phase 2: build
    k_all = k_buf[locs].float()
    t_b = bench(lambda: idxer.build_block_index(k_all))
    # phase 3: select_batched（nq 行，t_arr 覆盖全序列）
    index = idxer.build_block_index(k_all)
    t_arr = torch.arange(nq, device=dev)
    t_s = bench(lambda: idxer.select_batched(index, q_b, t_arr), n_warm=1, n_iter=3)
    # phase 4: 稀疏前向近似（gather K2=1024 + einsum）——与 _sparse_extend_one
    # 同量级：nq 行 × Hkv × K2 的 gather + 两个 einsum
    sel = idxer.select_batched(index, q_b[:64], t_arr[:64])
    sel_full = sel.repeat((nq + 63) // 64, 1, 1)[:nq]

    def sp_fwd():
        sel_c = sel_full.clamp(max=S - 1)
        pool_pos = locs[sel_c]  # [nq, Hkv, K2]
        flat = (
            pool_pos.unsqueeze(-1) * (Hkv * D)
            + torch.arange(Hkv, device=dev).view(1, Hkv, 1, 1) * D
            + torch.arange(D, device=dev).view(1, 1, 1, D)
        )
        ks = k_buf.reshape(-1)[flat.view(-1)].view(-1, Hkv, 1024, D)
        att = torch.einsum("nhd,nhkd->nhk", q_b.reshape(nq, Hkv, G, D).mean(2), ks)
        torch.softmax(att, dim=-1)

    t_f = bench(sp_fwd, n_warm=1, n_iter=3)
    tot = t_g + t_b + t_s + t_f
    print(f"{S:6d} {t_g*1e3:8.1f} {t_b*1e3:8.1f} {t_s*1e3:8.1f} {t_f*1e3:8.1f} "
          f"{tot*1e3:8.1f} {t_b/tot*100:8.1f}%")
    del k, k_buf, k_all, index, sel_full
    torch.cuda.empty_cache()
