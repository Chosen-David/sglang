import torch
from typing import Tuple

def prepare_seqlens(cu_seqlens: torch.Tensor):
    return cu_seqlens[1:] - cu_seqlens[:-1]


def prepare_cu_seqlens(key_states: torch.Tensor):
    input_shape = key_states.shape[:2]
    return torch.tensor([0] + [input_shape[1]] * input_shape[0], dtype=torch.int32, device=key_states.device).cumsum(0)

def prepare_cu_seqlens_from_mask(attn_mask: torch.Tensor):
    seqlens = attn_mask.sum(dim=-1)
    cu_seqlens = torch.cat((seqlens.new_zeros((1,)), seqlens.cumsum(dim=0)), dim=0)
    return cu_seqlens

def unpad_tensor(x: torch.Tensor, attn_mask: torch.Tensor):
    return x[attn_mask].unsqueeze(0)
