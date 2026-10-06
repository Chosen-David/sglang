import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Any
from einops import rearrange, einsum, repeat

from .base import Indexer
from .utils import prepare_pad_mask, pad_tensor


class TwilightIndexer(Indexer):

    def __init__(self, args) -> None:
        super().__init__(args)
        self.group_size = None
    
    def min_max_per_token_quant(self, x: torch.Tensor, num_bits: int = 4) -> torch.Tensor:
        max_val = x.amax(dim=-1, keepdim=True)
        min_val = x.amin(dim=-1, keepdim=True)

        max_int = 2**num_bits - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-9) / max_int
        zeros = -min_val
        # print(zeros)
        return torch.clamp(torch.round((x + zeros) / scales), min_int, max_int) * scales - zeros

    def prepare_index(self, 
        k: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        assert k.shape[0] == 1

        pad_mask, pad_cu_seqlens_k = prepare_pad_mask(self.args.twi_block_size, cu_seqlens_k)
        pad_k = pad_tensor(k, pad_mask)

        # self.metrics.add_k_delta(k, self.args.fsa_block_size_coarse)

        k_coarse = rearrange(pad_k, 'b (t bs) h d -> b t bs h d', bs=self.args.twi_block_size)
        k_min = k_coarse.amin(dim=2)
        k_max = k_coarse.amax(dim=2)
        cu_seqlens_k_coarse = pad_cu_seqlens_k // self.args.twi_block_size

        k_qat = self.min_max_per_token_quant(k)
        cu_seqlens_k_fine = cu_seqlens_k

        return {
            "k_min": k_min,
            "k_max": k_max,
            "k_qat": k_qat,
            "cu_seqlens_k_coarse": cu_seqlens_k_coarse,
            "cu_seqlens_k_fine": cu_seqlens_k_fine,
        }
    
    def compute_score(self,
        q: torch.Tensor, 
        q_ids: torch.Tensor,
        index_dict: Dict[str, torch.Tensor],
        softmax_scale: float
    ):
        assert q.shape[0] == 1 and q.shape[1] == 1
        cu_seqlens_k_coarse = index_dict["cu_seqlens_k_coarse"]
        cu_seqlens_k_fine   = index_dict["cu_seqlens_k_fine"]
        
        q, k_min, k_max, k_qat = (x.squeeze(0) for x in (q, index_dict["k_min"], index_dict["k_max"], index_dict["k_qat"]))

        self.group_size = q.shape[1] // k_min.shape[1]

        coarse_shape = (q.shape[0], q.shape[1], k_min.shape[0]) # (qt, h, kt)
        fine_shape = (q.shape[0], q.shape[1], k_qat.shape[0])
        score_coarse = q.new_full(coarse_shape, float('-inf'))
        score_fine = q.new_full(fine_shape, float('-inf'))

        b_q = (q[0] * softmax_scale).to(torch.float32)

        b_k_min = repeat(k_min, 't h d -> t (h g) d', g=self.group_size).to(torch.float32)
        b_k_max = repeat(k_max, 't h d -> t (h g) d', g=self.group_size).to(torch.float32)
        b_score_corase = \
            einsum(b_q.clamp(max=0), b_k_min, 'h d, kt h d -> h kt') + \
            einsum(b_q.clamp(min=0), b_k_max, 'h d, kt h d -> h kt')
        score_coarse[0] = b_score_corase.to(q.dtype)

        b_k_qat = repeat(k_qat, 't h d -> t (h g) d', g=self.group_size).to(torch.float32)
        b_score_fine = einsum(b_q, b_k_qat, 'h d, kt h d -> h kt')
        score_fine[0] = b_score_fine.to(q.dtype)

        return {
            "score_coarse": score_coarse.unsqueeze(0),
            "score_fine": score_fine.unsqueeze(0),
        }
    
    def compute_mask(self,
        q_ids: torch.Tensor,
        score_dict: Dict[str, torch.Tensor],
    ):
        block_size = self.args.twi_block_size
        score_coarse = score_dict["score_coarse"]
        score_fine   = score_dict["score_fine"]

        score_coarse = rearrange(score_coarse, 'b qt (h g) kt -> b qt h g kt', g=self.group_size).mean(dim=-2)
        score_coarse[..., -1] = float('inf')
        values, indices = torch.topk(score_coarse, dim=-1, k=min(score_coarse.shape[-1], self.args.twi_level1_topk))
        topk_mask = torch.zeros_like(score_coarse, dtype=torch.bool)\
            .scatter_(-1, indices, torch.ones_like(values, dtype=torch.bool))
        topk_mask = repeat(topk_mask, 'b qt h kt -> b qt (h g) (kt bs)', g=self.group_size, bs=block_size)

        topk_mask = topk_mask[..., :score_fine.shape[-1]]
        score_fine = torch.where(topk_mask, score_fine, float('-inf'))
        p = F.softmax(score_fine, dim=-1).nan_to_num(nan=0)

        # merge head
        # p = rearrange(p, 'b qt (h g) kt -> b qt h g kt', g=self.group_size).mean(dim=-2)
        # values, indices = torch.sort(p, dim=-1, descending=True)
        # cumsum = torch.cumsum(values, dim=-1)
        # cumsum_mask = (cumsum <= self.args.twi_level2_topp)
        # cumsum_mask = F.pad(cumsum_mask, (1, -1), value=True)
        # topp_mask = torch.zeros_like(p, dtype=torch.bool).scatter(-1, indices, cumsum_mask)
        # return topp_mask

        values, indices = torch.sort(p, dim=-1, descending=True)
        cumsum = torch.cumsum(values, dim=-1)
        cumsum_mask = (cumsum <= self.args.twi_level2_topp)
        cumsum_mask = F.pad(cumsum_mask, (1, -1), value=True)
        topp_mask = torch.zeros_like(p, dtype=torch.bool).scatter(-1, indices, cumsum_mask)
        topp_mask = (rearrange(topp_mask, 'b qt (h g) kt -> b qt h g kt', g=self.group_size).sum(dim=-2) != 0)
        return topp_mask

    def get_block_size(self):
        return 1
