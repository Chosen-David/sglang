import torch
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3.modeling_qwen3 import (
    apply_rotary_pos_emb,
    eager_attention_forward
)

from .attn_utils import flash_attention_forward

from typing import Optional, Callable, Tuple
from ..kv_cache import KVO_MHASfaCache
from ..ops import (
    KVO_MHAIndexerLevel1Interface,
    KVO_MHAIndexerLevel2Interface,
    KVO_SparseMHAInterface,
)


def kvo_qwen3_sfa_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: KVO_MHASfaCache | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape))
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))
    value_states = self.v_proj(hidden_states).view(hidden_shape)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=-2)

    if past_key_values.get_seq_length(self.layer_idx) == 0:
        cache_kwargs = {"attention_mask": attention_mask, "prefill_mode": True, "layer_idx": self.layer_idx}
        _ = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
        attn_output, attn_weights = flash_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # diff with Llama
            **kwargs,
        )
    else:
        sync_decode_step = False
        cache_kwargs = {"attention_mask": None,  "prefill_mode": False, "layer_idx": self.layer_idx}
        prefetch_block_indices, prefetch_keys, prefetch_values, level1_keys_min, level1_keys_max, level2_keys, lengths \
            = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
                
        sfa_args: Tuple[
            KVO_MHAIndexerLevel1Interface, 
            KVO_MHAIndexerLevel2Interface,
            KVO_SparseMHAInterface,
            int, int, int
        ] = self.sfa_args
        (
            level1_indexer_interface, # type
            level2_indexer_interface,
            decode_interface,
            sfa_block_size,
            sfa_half_dim,
            sfa_half_cmp_dim,
        ) = sfa_args

        level1_query_states = query_states
        level1_lengths = (lengths + sfa_block_size - 1) // sfa_block_size
        _, _, level1_topk_indices = level1_indexer_interface.forward(
            level1_query_states,
            level1_keys_min,
            level1_keys_max,
            level1_lengths,
        )

        if prefetch_block_indices is None: # or some other conditions
            sync_decode_step = True
            prefetch_block_indices, prefetch_keys, prefetch_values \
                = past_key_values.prefetch(self.layer_idx, level1_topk_indices, sync=True)

        level2_query_states = torch.cat(
            (
                query_states[..., sfa_half_dim-sfa_half_cmp_dim:sfa_half_dim],
                query_states[..., -sfa_half_cmp_dim:],
            ), dim=-1
        ).contiguous()
        level2_lengths = lengths
        _, _, _, level2_topk_indices = level2_indexer_interface.forward(
            level2_query_states,
            level2_keys,
            prefetch_block_indices,
            level2_lengths,
        )

        attn_output = decode_interface.forward(
            query_states,
            prefetch_keys,
            prefetch_values,
            level2_topk_indices,
        )
        attn_weights = None

        if not sync_decode_step:
            past_key_values.prefetch(self.layer_idx, level1_topk_indices)

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights
