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

from typing import Optional, Callable
from ..kv_cache import MHAFaCache
from ..ops import MHAInterface


def qwen3_fa_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: MHAFaCache | None = None,
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
        cache_kwargs = {"attention_mask": attention_mask, "prefill_mode": True}
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
        cache_kwargs = {"attention_mask": None,  "prefill_mode": False}
        key_states, value_states, lengths \
            = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        decode_interface: MHAInterface = self.sfa_args[0]
        attn_output = decode_interface.forward(
            query_states,
            key_states,
            value_states,
            lengths,
        )
        attn_weights = None

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights
