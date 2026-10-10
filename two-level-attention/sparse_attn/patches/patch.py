import torch
import torch.nn as nn
from types import MethodType
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
from transformers.models.llama.modeling_llama import LlamaAttention


from .qwen3_attn_patch import qwen3_attn_forward
from .llama3_attn_patch import llama3_attn_forward
from ..indexer import indexer_type_dict

def register_patch(model: nn.Module, args) -> int:
    """注册稀疏 attention patch，返回成功 patch 的模块数。

    【B3 修复（kimi3 清单 F3，2026-10-08）】method≠none 而零模块匹配时，
    原实现静默返回——GLM-4 等未支持模型跑 method≠none → 零匹配 → 实际
    dense 输出被打上稀疏方法标签（实验结论污染）。改为：
      ① 返回 patch 计数（调用方可显式校验）；
      ② method≠none 且 patched=0 时显式 raise 拒绝。
    GLM 说明：glm4_moe_lite_attn_patch.py 是无 indexer 调用的半成品
    （MLA 结构），注册它反而会复现「标稀疏跑 dense」，故不注册、
    显式拒绝（GLM 支持留待补完 indexer 集成后再放开）。
    """
    IndexerType = indexer_type_dict.get(args.method, None)
    if args.method != 'none' and IndexerType is None:
        raise ValueError(f"not support {args.method}")
    n_patched = 0
    if IndexerType is not None:
        for name, module in model.named_modules():
            if isinstance(module, Qwen3Attention):
                module.indexer = IndexerType(args)
                if hasattr(module, "layer_idx"):
                    module.indexer.layer_idx = module.layer_idx  # TLI D' 层掩码用
                module.forward = MethodType(qwen3_attn_forward, module)
                n_patched += 1
                if hasattr(module, "layer_idx") and module.layer_idx == 0:
                    print(f"register qwen3 attn: [{name}]")
            elif isinstance(module, LlamaAttention):
                module.indexer = IndexerType(args)
                if hasattr(module, "layer_idx"):
                    module.indexer.layer_idx = module.layer_idx
                module.forward = MethodType(llama3_attn_forward, module)
                n_patched += 1
                if hasattr(module, "layer_idx") and module.layer_idx == 0:
                    print(f"register llama3 attn: [{name}]")
    if args.method != 'none' and n_patched == 0:
        mod_types = sorted({type(m).__name__ for _, m in model.named_modules()})
        hint = ""
        if any('glm' in t.lower() for t in mod_types):
            hint = ("（检测到 GLM 系模块：glm4_moe_lite_attn_patch.py 为无 "
                    "indexer 集成的半成品，未注册支持）")
        raise RuntimeError(
            f"method={args.method!r} 但模型中没有任何可 patch 的 "
            f"Qwen3Attention/LlamaAttention 模块（patched=0）——继续运行会"
            f"静默产出 dense 结果却被标记为稀疏方法{hint}。请确认模型架构"
            f"或改 --method none。模块类型样本: {mod_types[:8]}"
        )
    return n_patched
