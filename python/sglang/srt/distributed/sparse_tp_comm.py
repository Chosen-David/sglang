from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import torch.distributed as dist

from sglang.srt.environ import envs

from .parallel_state import GroupCoordinator, get_tp_group


TensorLike = Union[np.ndarray, torch.Tensor]


@dataclass
class ResShard:
    k: int
    i: int
    C_k: TensorLike  # shape=[S_loc], dtype=int32
    Res_vals: TensorLike  # shape=[S_loc, F_loc], dtype=float16/float32/bfloat16
    Z_i: TensorLike  # shape=[F_loc], dtype=int32


def is_sparse_tp_comm_enabled() -> bool:
    return envs.SGLANG_ENABLE_SPARSE_TP_COMM.get()


def _to_device_tensor(
    x: TensorLike,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x.to(device=device, non_blocking=True)
        if dtype is not None and t.dtype != dtype:
            t = t.to(dtype=dtype)
        return t.contiguous()
    t = torch.as_tensor(x, device=device)
    if dtype is not None and t.dtype != dtype:
        t = t.to(dtype=dtype)
    return t.contiguous()


def _all_gather_fixed(local: torch.Tensor, group: GroupCoordinator) -> torch.Tensor:
    world_size = group.world_size
    out = torch.empty(
        (world_size,) + tuple(local.shape), dtype=local.dtype, device=local.device
    )
    dist.all_gather_into_tensor(out, local, group=group.device_group)
    return out


def all_gather_res_shard(
    local_shard: ResShard,
    tp_group: Optional[GroupCoordinator] = None,
) -> List[ResShard]:
    group = tp_group or get_tp_group()
    device = group.device
    world_size = group.world_size

    C_local = _to_device_tensor(local_shard.C_k, device=device, dtype=torch.int32)
    Z_local = _to_device_tensor(local_shard.Z_i, device=device, dtype=torch.int32)
    Res_local = _to_device_tensor(local_shard.Res_vals, device=device)

    if C_local.dim() != 1 or Z_local.dim() != 1 or Res_local.dim() != 2:
        raise ValueError("ResShard shapes must be C_k=[S_loc], Z_i=[F_loc], Res_vals=[S_loc,F_loc]")
    if Res_local.shape != (C_local.numel(), Z_local.numel()):
        raise ValueError(
            "Res_vals shape mismatch: expected [len(C_k), len(Z_i)], "
            f"got {tuple(Res_local.shape)}"
        )

    shape_local = torch.tensor(
        [C_local.numel(), Z_local.numel()], dtype=torch.int32, device=device
    )
    shape_all = _all_gather_fixed(shape_local, group)
    s_sizes = shape_all[:, 0].tolist()
    f_sizes = shape_all[:, 1].tolist()
    max_s = int(max(s_sizes))
    max_f = int(max(f_sizes))

    c_pack = torch.full((max_s,), -1, dtype=torch.int32, device=device)
    z_pack = torch.full((max_f,), -1, dtype=torch.int32, device=device)
    r_pack = torch.zeros((max_s, max_f), dtype=Res_local.dtype, device=device)
    c_pack[: C_local.numel()] = C_local
    z_pack[: Z_local.numel()] = Z_local
    r_pack[: C_local.numel(), : Z_local.numel()] = Res_local

    c_all = _all_gather_fixed(c_pack, group)
    z_all = _all_gather_fixed(z_pack, group)
    r_all = _all_gather_fixed(r_pack.flatten(), group).view(world_size, max_s, max_f)

    ki_local = torch.tensor([local_shard.k, local_shard.i], dtype=torch.int32, device=device)
    ki_all = _all_gather_fixed(ki_local, group)

    shards: List[ResShard] = []
    for r in range(world_size):
        s_loc = int(shape_all[r, 0].item())
        f_loc = int(shape_all[r, 1].item())
        shards.append(
            ResShard(
                k=int(ki_all[r, 0].item()),
                i=int(ki_all[r, 1].item()),
                C_k=c_all[r, :s_loc].clone(),
                Res_vals=r_all[r, :s_loc, :f_loc].clone(),
                Z_i=z_all[r, :f_loc].clone(),
            )
        )
    return shards


def resolve_full_res(
    shards: Sequence[ResShard],
    S: int,
    F: int,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if len(shards) == 0:
        raise ValueError("No shards provided")
    if device is None:
        first = shards[0].Res_vals
        if isinstance(first, torch.Tensor):
            device = first.device
        else:
            device = torch.device("cpu")

    if dtype is None:
        first = shards[0].Res_vals
        if isinstance(first, torch.Tensor):
            dtype = first.dtype
        else:
            dtype = torch.float32

    out = torch.zeros((S, F), dtype=dtype, device=device)
    for shard in shards:
        rows = _to_device_tensor(shard.C_k, device=device, dtype=torch.int64)
        cols = _to_device_tensor(shard.Z_i, device=device, dtype=torch.int64)
        vals = _to_device_tensor(shard.Res_vals, device=device, dtype=dtype)
        if vals.shape != (rows.numel(), cols.numel()):
            raise ValueError(
                "Shard value/index shape mismatch, got "
                f"vals={tuple(vals.shape)} rows={rows.numel()} cols={cols.numel()}"
            )
        out[rows[:, None], cols[None, :]] = vals
    return out


def all_gather_and_resolve_res_shard(
    local_shard: ResShard,
    S: int,
    F: int,
    tp_group: Optional[GroupCoordinator] = None,
) -> torch.Tensor:
    shards = all_gather_res_shard(local_shard, tp_group=tp_group)
    return resolve_full_res(shards, S=S, F=F)
