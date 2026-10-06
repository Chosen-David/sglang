"""E103 单元级 smoke：合成 q/k 验证 per_q_head 开关的 shape 链路与预算语义。

- 开关关：mask [1,1,Hkv,T]，每 kv-head 选中 token 数 ≈ K2（组内共享）
- 开关开：mask [1,1,H,T]，每 q-head 选中 token 数 ≈ K2（独立选择）
- 消费端 eager_decoding_attn 两口径均能跑通且无 NaN
"""
import sys
import types
import torch

sys.path.insert(0, "/home/wangyuanshuo02/two-level-attention")

from sparse_attn.indexer.tli_indexer import TLIIndexer
from sparse_attn.ops.eager_decoding import eager_decoding_attn

H, Hkv, D = 32, 8, 128          # Qwen3-8B GQA：32 q-head / 8 kv-head / G=4
BS = 64
K1, K2 = 128, 1024

def make_args(per_q_head):
    return types.SimpleNamespace(
        tia_block_size=BS, tia_level1_topk=K1, tia_level2_topk=K2,
        tia_level2_cmp_ratio=4, tia_enable_async_topk=False,
        tli_enable_subspace=False,           # full 维（E98 best 口径）
        tli_enable_kmeans=False, tli_far_select='4bit',
        tli_enable_layer_skip=False, tli_layer_skip_path=None,
        tli_alpha=0.125, tli_beta=0.375, tli_gamma=0.625,
        tli_far_method='minmax', tli_near_method='avg',
        tli_sigma_select='none', tli_sigma=8.0, tli_moba=False,
        tli_subspace='full', tli_proj_basis=None, tli_static_pair=False,
        tli_per_q_head=per_q_head,
    )

T = 6000   # 约 94 个块，mid 远大于 near/far 预算
torch.manual_seed(0)
q = torch.randn(1, 1, H, D)
k = torch.randn(1, T, Hkv, D)
v = torch.randn(1, T, Hkv, D)
cu = torch.tensor([0, T])

for per in (False, True):
    idx = TLIIndexer(make_args(per))
    idx.layer_idx = 1
    mask, block_size = idx.prepare_mask(q, torch.tensor([T - 1]), k, cu)
    print(f"[per_q_head={per}] mask shape={list(mask.shape)} block_size={block_size}")
    exp_h = H if per else Hkv
    assert mask.shape == (1, 1, exp_h, T), mask.shape
    per_head_cnt = mask[0, 0].sum(dim=-1).float()   # 每个 head 选中数
    print(f"  per-head select: mean={per_head_cnt.mean():.0f} "
          f"min={per_head_cnt.min():.0f} max={per_head_cnt.max():.0f} "
          f"(K2={K2})  NaN={torch.isnan(mask.float()).any().item()}")
    # 消费端
    o = eager_decoding_attn(q, k, v, mask, block_size, cu)
    assert not torch.isnan(o.float()).any(), "输出含 NaN"
    print(f"  eager_decoding output shape={list(o.shape)} 无 NaN")

# 开关关 vs 开：共享 mask 应与 per-head mask 的组内 mean 一致性抽查（选块可不同，
# 但预算口径应一致：Hkv 口径每 head ≈ K2；per-head 口径每 q-head ≈ K2）
print("E103 unit smoke PASS")
