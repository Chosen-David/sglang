import torch
import torch.nn as nn
from types import MethodType
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
from transformers.models.llama.modeling_llama import LlamaAttention


from .qwen3_attn_patch import qwen3_attn_forward
from .llama3_attn_patch import llama3_attn_forward
from ..indexer import indexer_type_dict

def register_patch(model: nn.Module, args):
    IndexerType = indexer_type_dict.get(args.method, None)
    if args.method != 'none' and IndexerType is None:
        raise ValueError(f"not support {args.method}")
    if IndexerType is not None:
        for name, module in model.named_modules():
            if isinstance(module, Qwen3Attention):
                module.indexer = IndexerType(args)
                if hasattr(module, "layer_idx"):
                    module.indexer.layer_idx = module.layer_idx  # TLI D' 层掩码用
                module.forward = MethodType(qwen3_attn_forward, module)
                if hasattr(module, "layer_idx") and module.layer_idx == 0:
                    print(f"register qwen3 attn: [{name}]")
            elif isinstance(module, LlamaAttention):
                module.indexer = IndexerType(args)
                if hasattr(module, "layer_idx"):
                    module.indexer.layer_idx = module.layer_idx
                module.forward = MethodType(llama3_attn_forward, module)
                if hasattr(module, "layer_idx") and module.layer_idx == 0:
                    print(f"register llama3 attn: [{name}]")
