# M7 prefill S 梯度基准（backend forward_extend 全链路，真实 trace 权重）：
#   gather + build + select_batched（M7 快路径）+ _sparse_extend_one（稀疏前向）
# 对比口径：优化前数字来自 git stash 前的同脚本（select_batched scatter 版）。
# trace 复制扩展到长 S（K 权重真实，token 重复——成本口径不失真）。
import sys

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
import time

import torch

from sglang.srt.layers.attention.tli.backend import TLISparseAttnBackend

dev = "cuda:0"
d = torch.load("/tmp/trace/qwen3-8b/lb_hotpotqa_0/layer03.pt", map_location=dev)
k_real = d["k"].float()  # [S0, Hkv, D]
S0, Hkv, D = k_real.shape
q_real = d["q"].float()
H = q_real.shape[1]


class FakePool:
    def __init__(self, k_pool, v_pool):
        self.k_pool, self.v_pool = k_pool, v_pool

    def get_kv_buffer(self, layer_id):
        return (self.k_pool, self.v_pool)

    def set_kv_buffer(self, layer, locs, k, v):
        pass


class FakeRTP:
    def __init__(self, req_to_token):
        self.req_to_token = req_to_token


class FakeMC:
    head_dim = D
    num_key_value_heads = Hkv
    num_hidden_layers = 1


class FakeRunner:
    def __init__(self, req_to_token, kvp):
        self.model_config = FakeMC()
        self.device = dev
        self.token_to_kv_pool = kvp
        self.req_to_token_pool = FakeRTP(req_to_token)


class FakeLayer:
    layer_id = 3
    scaling = D**-0.5


class FakeFB:
    token_to_kv_pool = None
    req_to_token_pool = None
    req_pool_indices = None
    out_cache_loc = None
    seq_lens = None
    extend_prefix_lens = None
    extend_seq_lens = None


def bench_extend(S):
    if S <= S0:
        k = k_real[:S]
    else:
        reps = (S + S0 - 1) // S0
        k = k_real.repeat(reps, 1, 1)[:S]
    torch.manual_seed(0)
    v = (torch.randn_like(k) * 0.05).float()
    POOL_N = S + 64
    perm = torch.randperm(POOL_N, device=dev)
    k_pool = torch.zeros(POOL_N, Hkv, D, device=dev)
    v_pool = torch.zeros(POOL_N, Hkv, D, device=dev)
    k_pool[perm[:S]] = k
    v_pool[perm[:S]] = v
    kvp = FakePool(k_pool, v_pool)
    req_to_token = torch.zeros(2, S + 64, dtype=torch.long, device=dev)
    req_to_token[1, :S] = perm[:S]
    be = TLISparseAttnBackend(runner=FakeRunner(req_to_token, kvp))
    fb = FakeFB()
    fb.req_pool_indices = torch.tensor([1], device=dev)
    nq = 256  # 稀疏 prefill 的 q 行 = 尾部 256（select_batched 行数口径）
    q_ext = q_real[-nq:]
    t_start = S - nq
    fb.extend_prefix_lens = torch.tensor([t_start], device=dev)
    fb.extend_seq_lens = torch.tensor([nq], device=dev)
    args = (
        q_ext.reshape(nq, H * D), k[S - nq : S].reshape(nq, Hkv * D),
        v[S - nq : S].reshape(nq, Hkv * D), FakeLayer(), fb,
    )
    for _ in range(2):
        be.forward_extend(*args, save_kv_cache=False)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(3):
        be.forward_extend(*args, save_kv_cache=False)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / 3
    del k_pool, v_pool, k, v, be
    torch.cuda.empty_cache()
    return dt


print(f"{'S':>7} {'ms/layer':>10}   （nq=256 尾 chunk，prefill tli 单层 forward_extend）")
for S in [5000, 10000, 20000, 40000, 80000, 130000]:
    dt = bench_extend(S)
    print(f"{S:7d} {dt*1e3:10.1f}")
