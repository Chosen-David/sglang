import torch

def prepare_pad_mask(block_size: int, cu_seqlens: torch.Tensor):
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    pad_seqlens = torch.stack([(x + block_size - 1) // block_size * block_size for x in seqlens])
    pad_cu_seqlens = torch.cat((pad_seqlens.new_tensor([0]), pad_seqlens.cumsum(dim=0)))
    pad_mask = torch.cat([torch.cat((seqlens.new_ones(x), seqlens.new_zeros(y - x))) for x, y in zip(seqlens, pad_seqlens)]).to(torch.bool)
    return pad_mask, pad_cu_seqlens

def pad_tensor(x: torch.Tensor, pad_mask: torch.Tensor):
    pad_x = x.new_zeros((1, pad_mask.size(0)) + x.shape[2:])
    pad_x[:, pad_mask] = x
    return pad_x

def prepare_seqlens(cu_seqlens: torch.Tensor):
    return cu_seqlens[1:] - cu_seqlens[:-1]
