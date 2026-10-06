import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Any
from einops import rearrange, einsum, repeat

from .base import Indexer
from .utils import prepare_pad_mask, pad_tensor


class TIAIndexer(Indexer):

    def __init__(self, args) -> None:
        super().__init__(args)
        self.group_size = None
        self.sliding_window_size = 128
        self.cmp_ratio = args.tia_level2_cmp_ratio
        self.enable_async = args.tia_enable_async_topk
        self.prev_mask = None
    
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

        pad_mask, pad_cu_seqlens_k = prepare_pad_mask(self.args.tia_block_size, cu_seqlens_k)
        pad_k = pad_tensor(k, pad_mask)

        # self.metrics.add_k_delta(k, self.args.fsa_block_size_coarse)

        k_coarse = rearrange(pad_k, 'b (t bs) h d -> b t bs h d', bs=self.args.tia_block_size)
        k_min = k_coarse.amin(dim=2)
        k_max = k_coarse.amax(dim=2)
        cu_seqlens_k_coarse = pad_cu_seqlens_k // self.args.tia_block_size

        assert k.shape[-1] == 128
        delta = 64 // self.cmp_ratio
        indices = torch.tensor(list(range(64 - delta, 64)) + list(range(128 - delta, 128))).to(k.device)
        k_qat = torch.zeros_like(k)
        k_qat[..., indices] = self.min_max_per_token_quant(k[..., indices])
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
        score_coarse = score_dict["score_coarse"]
        score_fine   = score_dict["score_fine"]

        score_coarse = rearrange(score_coarse, 'b qt (h g) kt -> b qt h g kt', g=self.group_size).mean(dim=-2)
        score_coarse[..., -1] = float('inf')
        values, indices = torch.topk(score_coarse, dim=-1, k=min(score_coarse.shape[-1], self.args.tia_level1_topk))
        topk_mask = torch.zeros_like(score_coarse, dtype=torch.bool)\
            .scatter_(-1, indices, torch.ones_like(values, dtype=torch.bool))
        topk_mask = repeat(topk_mask, 'b qt h kt -> b qt (h g) (kt bs)', g=self.group_size, bs=self.args.tia_block_size)

        if self.enable_async:
            if self.prev_mask is not None:
                topk_mask, self.prev_mask = self.prev_mask, topk_mask
                if topk_mask.shape[-1] < score_fine.shape[-1]:
                    pad_len = score_fine.shape[-1] - topk_mask.shape[-1]
                    topk_mask = F.pad(topk_mask, (0, pad_len), value=True)
            else:
                self.prev_mask = topk_mask

        topk_mask = topk_mask[..., :score_fine.shape[-1]]
        score_fine = torch.where(topk_mask, score_fine, float('-inf'))
        p = F.softmax(score_fine, dim=-1).nan_to_num(nan=0)
        p = rearrange(p, 'b qt (h g) kt -> b qt h g kt', g=self.group_size).mean(dim=-2)
        p[..., -self.sliding_window_size:] = 1.0
        # k_idx = torch.arange(0, score_fine.shape[-1].shape[-1]).expand_as(p)
        # q_idx = q_ids[None, :, None, None].expand_as(p)
        # swa_mask = ((q_idx - self.sliding_window_size) < k_idx) & (q_idx >= k_idx)
        # p = torch.where(swa_mask, torch.ones_like(p), p)

        values, indices = torch.topk(p, k=min(p.shape[-1], self.args.tia_level2_topk), dim=-1)
        topk_mask = torch.zeros_like(p, dtype=torch.bool).scatter_(-1, indices, torch.ones_like(values, dtype=torch.bool))
        return topk_mask

    def get_block_size(self):
        return 1

    def clear(self):
        self.prev_mask = None
