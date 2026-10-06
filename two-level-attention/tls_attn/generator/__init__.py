import torch
from typing import List, Optional, Tuple
from transformers import (
    PreTrainedModel,
    Qwen3ForCausalLM,
    Glm4MoeLiteForCausalLM,
)

from .base import BaseGenerator
from .mha_generator import MHAGenerator

SUPPORT_MODELS = [Qwen3ForCausalLM, Glm4MoeLiteForCausalLM]

def create_generator(
    model: PreTrainedModel,
    max_batch_size: int,
    max_seq_len: int,
    max_new_tokens: int
) -> MHAGenerator:
    """    
    Args:
        model: 预训练模型
        max_batch_size: 最大批处理大小
        max_seq_len: 最大序列长度
        max_new_tokens: 最大生成token数
        
    Returns:
        对应的静态KV生成器实例
    """
    # 检查模型是否支持
    if not any(isinstance(model, model_class) for model_class in SUPPORT_MODELS):
        raise ValueError(f"Only models of type {SUPPORT_MODELS} are supported.")
        
    # 检查是否为MHA模型
    if isinstance(model, Qwen3ForCausalLM):
        return MHAGenerator(model, max_batch_size, max_seq_len, max_new_tokens)
    elif isinstance(model, Glm4MoeLiteForCausalLM):
        raise NotImplementedError
    else:
        raise ValueError(f"Model type '{type(model)}' is not supported.")
