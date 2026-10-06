import torch
from typing import Tuple, Dict, Any
from .utils import prepare_pad_mask, pad_tensor
from ..metrics import get_metrics

class Indexer:

    def __init__(self, args) -> None:
        self.args = args
        self.metrics = get_metrics()

    def prepare_index(self,
        k: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
    ):
        pass

    def compute_score(self,
        q: torch.Tensor, 
        q_ids: torch.Tensor,
        index_dict: Dict[str, torch.Tensor],
        cu_seqlens_idx: torch.Tensor,
        softmax_scale: float,
    ):
        pass

    def compute_mask(self,
        q_ids: torch.Tensor,
        score_dict: Dict[str, torch.Tensor],
    ):
        pass

    def get_block_size(self):
        pass

    def prepare_mask(self,
        q: torch.Tensor,
        q_ids: torch.Tensor,
        k: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        softmax_scale: float = None
    ) -> torch.Tensor:
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** -0.5
        index_dict = self.prepare_index(k, cu_seqlens_k)
        score_dict = self.compute_score(q, q_ids, index_dict, softmax_scale)
        mask = self.compute_mask(q_ids, score_dict)
        block_size = self.get_block_size()
        self.metrics.add_select_result(q_ids, mask, block_size)
        return mask, block_size

    def clear(self):
        pass
