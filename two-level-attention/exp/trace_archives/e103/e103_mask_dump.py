import sys, types, torch
sys.path.insert(0, "/home/wangyuanshuo02/two-level-attention")
from sparse_attn.indexer.tli_indexer import TLIIndexer
a = types.SimpleNamespace(
    tia_block_size=64, tia_level1_topk=128, tia_level2_topk=1024,
    tia_level2_cmp_ratio=4, tia_enable_async_topk=False,
    tli_enable_subspace=False, tli_enable_kmeans=False, tli_far_select='4bit',
    tli_enable_layer_skip=False, tli_layer_skip_path=None,
    tli_alpha=0.125, tli_beta=0.375, tli_gamma=0.625,
    tli_far_method='minmax', tli_near_method='avg',
    tli_sigma_select='none', tli_sigma=8.0, tli_moba=False,
    tli_subspace='full', tli_proj_basis=None, tli_static_pair=False)
T = 6000
torch.manual_seed(42)
q = torch.randn(1,1,32,128); k = torch.randn(1,T,8,128)
idx = TLIIndexer(a); idx.layer_idx = 3
mask, _ = idx.prepare_mask(q, torch.tensor([T-1]), k, torch.tensor([0,T]))
torch.save(mask.cpu(), sys.argv[1])
print("saved", sys.argv[1], mask.shape)
