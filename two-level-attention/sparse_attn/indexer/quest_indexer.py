import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Any
from einops import rearrange, einsum, repeat

from .base import Indexer
from .utils import prepare_pad_mask, pad_tensor

class QuestIndexer(Indexer):

    def __init__(self, args) -> None:
        super().__init__(args)
 
    def prepare_index(self, 
        k: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        pad_mask, pad_cu_seqlens = prepare_pad_mask(self.args.quest_block_size, cu_seqlens_k)
        k = pad_tensor(k, pad_mask)
        k = rearrange(k, 'b (t bs) h d -> b t bs h d', bs=self.args.quest_block_size)
        k_min = k.amin(dim=2)
        k_max = k.amax(dim=2)
        cu_seqlens_k = pad_cu_seqlens // self.args.quest_block_size
        return {
            "k_min": k_min, 
            "k_max": k_max,
            "cu_seqlens_k": cu_seqlens_k,
        }
    
    def compute_score(self,
        q: torch.Tensor,
        q_ids: torch.Tensor,
        index_dict: Dict[str, torch.Tensor],
        softmax_scale: float,
    ):
        cu_seqlens_k: torch.Tensor = index_dict["cu_seqlens_k"]
        q, k_min, k_max = (x.squeeze(0) for x in (q, index_dict["k_min"], index_dict["k_max"]))
        score_shape = (q.shape[0], k_min.shape[1], k_min.shape[0])
        score = q.new_full(score_shape, float('-inf'))
        G = q.shape[-2] // k_min.shape[-2]

        bos_k, eos_k = cu_seqlens_k[0], cu_seqlens_k[1]

        k_ids = torch.arange(0, eos_k - bos_k, dtype=torch.int32, device=q.device)
        b_mask = (q_ids // self.args.quest_block_size) > k_ids

        b_q = (q[0] * softmax_scale).to(torch.float32)
        b_k_min = repeat(k_min[bos_k:eos_k], 't h d -> t (h g) d', g=G).to(torch.float32)
        b_k_max = repeat(k_max[bos_k:eos_k], 't h d -> t (h g) d', g=G).to(torch.float32)
        b_score_min = einsum(b_q, b_k_min, 'h d, tk h d -> h tk d')
        b_score_max = einsum(b_q, b_k_max, 'h d, tk h d -> h tk d')
        b_score = torch.stack((b_score_min, b_score_max), dim=-1).amax(dim=-1).sum(dim=-1) # [h, tk]
        b_score = rearrange(b_score, '(h g) tk -> h g tk', g=G).mean(dim=-2)
        b_score = torch.where(b_mask[None, :], b_score, float('-inf'))
        score[0] = b_score.to(q.dtype)

        return {
            "score": score.unsqueeze(0)
        }
    
    def compute_mask(self,
        q_ids: torch.Tensor,
        score_dict: Dict[str, torch.Tensor],
    ):
        score = score_dict["score"]
        score[..., -1] = float('-inf')
        values, indices = torch.sort(score, dim=-1, descending=True)

        topk_mask = torch.zeros_like(values, dtype=torch.bool)
        topk_mask[..., :self.args.quest_topk - 1] = True

        mask = torch.zeros_like(score, dtype=torch.bool)
        mask.scatter_(-1, indices, topk_mask)

        position_ids = torch.arange(0, q_ids.shape[0]).type_as(q_ids)
        mask[:, position_ids, :, q_ids // self.args.quest_block_size] = True

        return mask

    def get_block_size(self):
        return self.args.quest_block_size
