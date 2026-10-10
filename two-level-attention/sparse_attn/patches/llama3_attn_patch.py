import torch
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    eager_attention_forward
)

from typing import Optional, Callable
from ..indexer import Indexer
from ..ops.eager_decoding import eager_decoding_attn
from .utils import (
    prepare_seqlens,
    prepare_cu_seqlens, 
    prepare_cu_seqlens_from_mask, 
    unpad_tensor
)


def llama3_attn_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    # 【B2 修复（kimi3 清单 F2，2026-10-08）】与 qwen3_attn_patch 同构的
    # 分支判定守卫（use_cache=False / chunked prefill / batch>1 / 4D float
    # mask 一律显式拒绝，详见 qwen3_attn_patch.py 注释）。
    if (past_key_values is not None
            and past_key_values.get_seq_length(self.layer_idx) == input_shape[1]):
        indexer: Indexer = self.indexer
        indexer.clear()
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
    else:
        # 【B2 修复（kimi3 清单 F2，2026-10-08）】decode 分支入口硬守卫
        # （与 qwen3_attn_patch 同款）。
        if past_key_values is None:
            raise RuntimeError(
                "TLI 稀疏 patch 需要 KV cache（use_cache=True）；当前 "
                "past_key_values=None（use_cache=False 形态，训练/PPL/"
                "logprob 打分不支持稀疏索引路径），拒绝执行。"
            )
        if input_shape[0] != 1 or input_shape[1] != 1:
            raise RuntimeError(
                f"TLI 稀疏 decode 路径仅支持 batch=1 且 q_len=1 的逐 token "
                f"解码；当前 batch={input_shape[0]}、q_len={input_shape[1]}"
                f"（chunked prefill / batch>1 会静默只算首 token 或形状"
                f"崩溃），拒绝执行。"
            )
        if attention_mask is not None and (
            attention_mask.dim() != 2 or attention_mask.dtype != torch.bool
        ):
            raise RuntimeError(
                f"TLI 稀疏 decode 路径的 attention_mask 分支仅支持 2D bool "
                f"padding mask；当前 dim={attention_mask.dim()}、"
                f"dtype={attention_mask.dtype}（HF 实际传 4D float causal "
                f"mask，按 2D 假设处理不正确），拒绝执行。"
            )
        indexer: Indexer = self.indexer
        query_states, key_states, value_states = (x.transpose(1, 2) for x in (query_states, key_states, value_states))
        if attention_mask is not None:
            cu_seqlens_k = prepare_cu_seqlens_from_mask(attention_mask)
            query_states, key_states, value_states = (unpad_tensor(x, attention_mask) for x in (query_states, key_states, value_states))
        else:
            cu_seqlens_k = prepare_cu_seqlens(key_states)
            query_states, key_states, value_states = (x.view(1, -1, *x.shape[-2:]) for x in (query_states, key_states, value_states))
        query_position_ids = prepare_seqlens(cu_seqlens_k) - 1
        mask, block_size = indexer.prepare_mask(
            query_states, 
            query_position_ids, 
            key_states, 
            cu_seqlens_k
        )
        attn_output = eager_decoding_attn(
            query_states,
            key_states,
            value_states,
            mask,
            block_size,
            cu_seqlens_k,
            softmax_scale=self.scaling,
        )
        attn_weights = None

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights
