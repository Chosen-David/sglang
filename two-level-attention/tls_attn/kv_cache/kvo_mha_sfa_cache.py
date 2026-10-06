import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen3Config
from transformers.cache_utils import (
    _is_torch_greater_or_equal_than_2_7,
    Cache, CacheLayerMixin
)
from typing import Any

from ..ops import (
    KVO_MHAStoreKVInterface,
    KVO_MHAStoreKIndexInterface,
)

from transformers.cache_utils import DynamicCache


class MHALayer(CacheLayerMixin):
    """
    A static cache layer that stores the key and value states as static tensors of shape `[batch_size, num_heads, max_cache_len), head_dim]`.
    It lazily allocates its full backing tensors, and then mutates them in-place. Built for `torch.compile` support.

    Args:
        max_cache_len (`int`):
            Maximum number of tokens that can be stored, used for tensor preallocation.
    """

    is_compileable = True
    is_sliding = False

    def __init__(self, 
        max_batch_size: int,
        max_cache_len: int, 
        sfa_block_size: int,
        sfa_level1_topk: int,
        sfa_level2_topk: int,
        sfa_cmp_ratio: int,
        store_kv_interface: KVO_MHAStoreKVInterface,
        store_k_index_interface: KVO_MHAStoreKIndexInterface,
    ):
        super().__init__()
        self.max_batch_size = max_batch_size
        self.max_cache_len = max_cache_len
        self.sfa_block_size = sfa_block_size
        self.sfa_level1_topk = sfa_level1_topk
        self.sfa_level2_topk = sfa_level2_topk
        self.sfa_cmp_ratio = sfa_cmp_ratio
        self.store_kv_interface = store_kv_interface
        self.store_k_index_interface = store_k_index_interface
        self.lengths = None

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        """
        Lazy initialization of the keys and values tensors. This allows to get all properties (dtype, device,
        num_heads in case of TP etc...) at runtime directly, which is extremely practical as it avoids moving
        devices, dtypes etc later on for each `update` (which could break the static dynamo addresses as well).

        If this is unwanted, one can call `early_initialization(...)` on the Cache directly, which will call this
        function ahead-of-time (this is required for `torch.export` for example). Note that for `compile`, as we
        internally don't compile the prefill, this is guaranteed to have been called already when compiling.
        If compiling the prefill as well, e.g. calling `model.compile(...)` before `generate` with a static cache,
        it is still supported in general, but without guarantees depending on the compilation options (e.g. cuda graphs,
        i.e. `mode="reduce-overhead"` is known to fail). But it will in general work correctly, and prefill should
        not be compiled anyway for performances!
        """
        self.dtype, self.device = key_states.dtype, key_states.device
        _, _, self.num_heads = key_states.shape[:3]
        self.v_head_dim = value_states.shape[-1]
        self.k_head_dim = key_states.shape[-1]

        self.lengths = torch.zeros(self.max_batch_size, dtype=torch.int32, device=self.device)
        self.keys = torch.zeros(
            (self.max_batch_size, self.max_cache_len, self.num_heads, self.k_head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.values = torch.zeros(
            (self.max_batch_size, self.max_cache_len, self.num_heads, self.v_head_dim),
            dtype=self.dtype,
            device=self.device,
        )

        self.level1_keys_min = torch.zeros(
            (self.max_batch_size, self.max_cache_len // self.sfa_block_size, self.num_heads, self.k_head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.level1_keys_max = torch.zeros(
            (self.max_batch_size, self.max_cache_len // self.sfa_block_size, self.num_heads, self.k_head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.level2_keys = torch.zeros(
            (self.max_batch_size, self.max_cache_len, self.num_heads, self.k_head_dim // self.sfa_cmp_ratio),
            dtype=self.dtype,
            device=self.device,
        )

        self.local_keys = torch.zeros(
            (self.max_batch_size, self.sfa_block_size, self.num_heads, self.k_head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.prefetch_block_indices = None
        self.prefetch_keys = torch.zeros(
            (self.max_batch_size, self.sfa_block_size * self.sfa_level1_topk, self.num_heads, self.k_head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.prefetch_values = torch.zeros(
            (self.max_batch_size, self.sfa_block_size * self.sfa_level1_topk, self.num_heads, self.k_head_dim),
            dtype=self.dtype,
            device=self.device,
        )

        self.is_initialized = True

    def offload(self):
        """Offload this layer's data to CPU device."""
        if self.is_initialized and self.keys.device == self.device:
            self.keys = self.keys.transpose(1, 2).contiguous().to("cpu", non_blocking=True)
            self.values = self.values.transpose(1, 2).contiguous().to("cpu", non_blocking=True)

    def prefetch(self, block_indices: torch.Tensor, layer_idx: int = 0):
        """In case of layer offloading, this allows to move the data back to the layer's device ahead of time."""
        if self.is_initialized:
            if self.prefetch_block_indices is None:
                self.prefetch_block_indices = torch.full_like(block_indices, fill_value=-1)

            next_block_indices, src_indices, dst_offsets, load_blocks = \
                self.store_kv_interface.update_block_indices(
                    self.prefetch_block_indices, block_indices
                )

            src_indices = src_indices.view(-1)
            src_indices = src_indices[src_indices != -1]
            # src_indices: [batch, heads, topk]

            cumsum_load_blocks = \
                torch.cat((load_blocks.new_tensor([0]), load_blocks.view(-1)[:-1]), dim=-1)\
                .cumsum(dim=0)\
                .view(load_blocks.shape)\
                .to(torch.int32)

            # max_batch x heads x num_blocks
            src_indices_cpu = src_indices.cpu()

            keys = self.keys.view(-1, self.sfa_block_size, self.k_head_dim) 
            values = self.values.view(-1, self.sfa_block_size, self.v_head_dim)
            load_keys = keys[src_indices_cpu].to(self.device, non_blocking=True)
            load_values = values[src_indices_cpu].to(self.device, non_blocking=True)

            self.store_kv_interface.update_kv(
                load_keys,
                load_values,
                cumsum_load_blocks,
                load_blocks,
                dst_offsets,
                self.prefetch_keys,
                self.prefetch_values,
            )
            
            self.prefetch_block_indices = next_block_indices
            return (
                self.prefetch_block_indices,
                self.prefetch_keys,
                self.prefetch_values,
            )
        else:
            raise ValueError
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str],
    ):
        """
        Update the key and value caches in-place, and return the necessary keys and value states.

        Args:
            key_states (`torch.Tensor`): The new key states to cache.
            value_states (`torch.Tensor`): The new value states to cache.
            cache_kwargs (`dict[str, Any]`, *optional*): Additional arguments for the cache.

        Returns:
            tuple[`torch.Tensor`, `torch.Tensor`]: The key and value states.
        """
        # Lazy initialization
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        prefill_mode = cache_kwargs.get("prefill_mode")
        if prefill_mode:
            batch, seqlen_kv = key_states.shape[:2]
            attention_mask = cache_kwargs.get("attention_mask", None)
            if attention_mask is not None:
                offsets = (attention_mask == 0).sum(dim=-1).to(torch.int32)
            else:
                offsets = key_states.new_zeros([batch], dtype=torch.int32)
            self.lengths[:batch] = seqlen_kv - offsets
            self.store_kv_interface.prefill_forward(
                key_states,
                value_states,
                offsets,
                self.keys,
                self.values,
            )
            self.store_k_index_interface.prefill_forward(
                key_states,
                offsets,
                self.local_keys,
                self.level1_keys_min,
                self.level1_keys_max,
                self.level2_keys,
            )

        else:
            batch, seqlen_kv = key_states.shape[:2]
            self.lengths[:batch] += 1

            batch_ids = torch.arange(batch, dtype=torch.int64)
            indices = self.lengths[:batch].to("cpu") - 1
            key_states_cpu = key_states.to("cpu")
            value_states_cpu = value_states.to("cpu")
            self.keys[batch_ids, :, indices, :] = key_states_cpu.squeeze(1)
            self.values[batch_ids, :, indices, :] = value_states_cpu.squeeze(1)

            if self.prefetch_block_indices is not None:
                self.store_kv_interface.decode_forward(
                    key_states,
                    value_states,
                    self.lengths[:batch],
                    self.prefetch_block_indices,
                    self.prefetch_keys,
                    self.prefetch_values,
                )

            self.store_k_index_interface.decode_forward(
                key_states,
                self.lengths[:batch],
                self.local_keys,
                self.level1_keys_min,
                self.level1_keys_max,
                self.level2_keys,
            )

        return (
            self.prefetch_block_indices,
            self.prefetch_keys,
            self.prefetch_values, 
            self.level1_keys_min, 
            self.level1_keys_max, 
            self.level2_keys, 
            self.lengths,
        )

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        """Return the length and offset of the cache, used to generate the attention mask"""
        kv_offset = 0
        kv_length = self.max_cache_len
        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        return self.lengths.amax().item() if self.lengths is not None else 0

    def get_max_cache_shape(self) -> int:
        """Return the maximum cache shape of the cache"""
        return self.max_cache_len


class KVO_MHASfaCache(Cache):

    def __init__(
        self,
        config: Qwen3Config,
        max_batch_size: int,
        max_cache_len: int,
        sfa_block_size: int = 64,
        sfa_level1_topk: int = 128,
        sfa_level2_topk: int = 1024,
        sfa_cmp_ratio: int = 4,
        **kwargs,
    ):
        config = config.get_text_config(decoder=True)

        store_kv_interface = KVO_MHAStoreKVInterface(
            max_batch=max_batch_size,
            max_seqlen=max_cache_len,
            num_kv_heads=config.num_key_value_heads,
            dim_k=config.hidden_size // config.num_attention_heads,
            dim_v=config.hidden_size // config.num_attention_heads,
            topk=sfa_level1_topk,
            block_size=sfa_block_size,
        )
        store_k_index_interface = KVO_MHAStoreKIndexInterface(
            max_batch=max_batch_size,
            max_seqlen=max_cache_len,
            num_kv_heads=config.num_key_value_heads,
            dim_k=config.hidden_size // config.num_attention_heads,
            sfa_block_size=sfa_block_size,
            sfa_cmp_ratio=sfa_cmp_ratio,
        )

        layers = []
        for _ in range(config.num_hidden_layers):
            layer = MHALayer(
                max_batch_size=max_batch_size,
                max_cache_len=max_cache_len, 
                sfa_block_size=sfa_block_size,
                sfa_level1_topk=sfa_level1_topk,
                sfa_level2_topk=sfa_level2_topk,
                sfa_cmp_ratio=sfa_cmp_ratio,
                store_kv_interface=store_kv_interface,
                store_k_index_interface=store_k_index_interface,
            )
            layers.append(layer)

        super().__init__(layers=layers, offloading=True, offload_only_non_sliding=True)

        assert _is_torch_greater_or_equal_than_2_7

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ):
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`dict[str, Any]`, *optional*):
                Additional arguments for the cache subclass. These are specific to each subclass and allow new types of
                cache to be created.

        Return:
            A tuple containing the updated key and value states.
        """
        # In this case, the `layers` were not provided, and we must append as much as `layer_idx`
        if self.layer_class_to_replicate is not None:
            while len(self.layers) <= layer_idx:
                self.layers.append(self.layer_class_to_replicate())

        # Wait for the stream to finish if needed, and start prefetching the next layer
        torch.cuda.default_stream(key_states.device).wait_stream(self.prefetch_stream)

        prefetch_block_indices, prefetch_keys, prefetch_values, \
        level1_keys_min, level1_keys_max, level2_keys, lengths \
            = self.layers[layer_idx].update(key_states, value_states, cache_kwargs)

        self.offload(layer_idx)

        return prefetch_block_indices, prefetch_keys, prefetch_values, level1_keys_min, level1_keys_max, level2_keys, lengths

    def offload(self, layer_idx: int):
        self.layers[layer_idx].offload()            

    def prefetch(self, layer_idx: int, block_indices: torch.Tensor, sync: bool = False):
        # Prefetch
        with self.prefetch_stream:
            prefetch_block_indices, prefetch_keys, prefetch_values \
                = self.layers[layer_idx].prefetch(block_indices, layer_idx)
        if sync:
            torch.cuda.default_stream(block_indices.device).wait_stream(self.prefetch_stream)
        return prefetch_block_indices, prefetch_keys, prefetch_values
