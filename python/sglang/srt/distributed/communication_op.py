# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/communication_op.py

from typing import Any, Dict, Optional, Union

import torch
import torch.distributed

from .parallel_state import get_tp_group

from .sparse_tp_comm import (
      ResShard,
      all_gather_and_resolve_res_shard,
      all_gather_res_shard,
      is_sparse_tp_comm_enabled,
  )

def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_tp_group().all_reduce(input_)


def tensor_model_parallel_all_gather(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    return get_tp_group().all_gather(input_, dim)


def tensor_model_parallel_gather(
    input_: torch.Tensor, dst: int = 0, dim: int = -1
) -> Optional[torch.Tensor]:
    """Gather the input tensor across model parallel group."""
    return get_tp_group().gather(input_, dst, dim)


def broadcast_tensor_dict(
    tensor_dict: Optional[Dict[Any, Union[torch.Tensor, Any]]] = None, src: int = 0
):
    if not torch.distributed.is_initialized():
        return tensor_dict
    return get_tp_group().broadcast_tensor_dict(tensor_dict, src)


def tensor_model_parallel_all_gather_res_shard(
      local_shard: ResShard,
      tp_group: Optional[GroupCoordinator] = None,
  ):
      return all_gather_res_shard(local_shard, tp_group=tp_group or get_tp_group())


def tensor_model_parallel_sparse_resolve_res(
      local_shard: ResShard,
      S: int,
      F: int,
      tp_group: Optional[GroupCoordinator] = None,
  ):
      return all_gather_and_resolve_res_shard(
          local_shard, S=S, F=F, tp_group=tp_group or get_tp_group()
      )

def should_use_sparse_tp_comm() -> bool:
    return is_sparse_tp_comm_enabled()
