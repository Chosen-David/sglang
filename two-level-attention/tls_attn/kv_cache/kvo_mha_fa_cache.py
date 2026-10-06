import torch
import torch.nn as nn
from transformers import Qwen3Config
from transformers.cache_utils import Cache, CacheLayerMixin
from typing import Any

from ..ops import (
    MHAStoreKVInterface,
    MHAStoreKIndexInterface,
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
        store_kv_interface: MHAStoreKVInterface,
    ):
        super().__init__()
        self.max_batch_size = max_batch_size
        self.max_cache_len = max_cache_len
        self.store_kv_interface = store_kv_interface
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

        self.is_initialized = True

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        else:
            batch, seqlen_kv = key_states.shape[:2]
            self.lengths[:batch] += 1
            self.store_kv_interface.decode_forward(
                key_states,
                value_states,
                self.lengths[:batch],
                self.keys,
                self.values,
            )

        return self.keys, self.values, self.lengths

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


class KVO_MHAFaCache(Cache):

    def __init__(
        self,
        config: Qwen3Config,
        max_batch_size: int,
        max_cache_len: int,
        **kwargs,
    ):
        config = config.get_text_config(decoder=True)

        store_kv_interface = MHAStoreKVInterface(
            max_batch=max_batch_size,
            max_seqlen=max_cache_len,
            num_kv_heads=config.num_key_value_heads,
            dim_k=config.hidden_size // config.num_attention_heads,
            dim_v=config.hidden_size // config.num_attention_heads,
        )

        layers = []
        for _ in range(config.num_hidden_layers):
            layer = MHALayer(
                max_batch_size=max_batch_size,
                max_cache_len=max_cache_len, 
                store_kv_interface=store_kv_interface,
            )
            layers.append(layer)

        super().__init__(layers=layers, offloading=True, offload_only_non_sliding=True)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

        if self.offloading:
            # Wait for the stream to finish if needed, and start prefetching the next layer
            torch.cuda.default_stream(key_states.device).wait_stream(self.prefetch_stream)
            self.prefetch(layer_idx + 1, self.only_non_sliding)

        keys, values, lengths \
            = self.layers[layer_idx].update(key_states, value_states, cache_kwargs)

        if self.offloading:
            self.offload(layer_idx, self.only_non_sliding)

        return keys, values, lengths
