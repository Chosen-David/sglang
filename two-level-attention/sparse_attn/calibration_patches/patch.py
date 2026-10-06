import torch
import torch.nn as nn
from types import MethodType
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
from transformers.models.llama.modeling_llama import LlamaAttention


from .qwen3_attn_patch import qwen3_attn_forward
from .llama3_attn_patch import llama3_attn_forward
from .recorder import AttnWeightsRecorder

def register_calibration_patch(model: nn.Module, args):
    recorder = AttnWeightsRecorder()
    for name, module in model.named_modules():
        if isinstance(module, Qwen3Attention):
            module.forward = MethodType(qwen3_attn_forward(recorder), module)
            print(f"register qwen3 attn: [{name}]")
        elif isinstance(module, LlamaAttention):
            module.forward = MethodType(llama3_attn_forward(recorder), module)
            print(f"register llama3 attn: [{name}]")
    return recorder
