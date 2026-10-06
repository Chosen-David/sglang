import logging
import torch
from typing import List, Optional, Tuple, Dict, Any
from types import MethodType
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3ForCausalLM,
    Qwen3Attention,
)

from .base import BaseGenerator
from ..kv_cache import (
    MHAFaCache,
    MHASfaCache,
    KVO_MHAFaCache,
    KVO_MHASfaCache,
)
from ..patches import (
    qwen3_fa_forward,
    qwen3_sfa_forward,
    kvo_qwen3_fa_forward,
    kvo_qwen3_sfa_forward,
)
from ..ops import (
    MHAInterface,
    MHAIndexerLevel1Interface,
    MHAIndexerLevel2Interface,
    SparseMHAInterface,
    KVO_MHAIndexerLevel1Interface,
    KVO_MHAIndexerLevel2Interface,
    KVO_SparseMHAInterface,
)

logger = logging.getLogger(__name__)


class MHAGenerator(BaseGenerator):

    def __init__(
        self,
        model: Qwen3ForCausalLM,
        max_batch_size: int,
        max_seq_len: int,
        max_new_tokens: int,
        enable_sfa: bool = False,
        enable_offloading: bool = False,
        sfa_block_size: int = 64,
        sfa_level1_topk: int = 128, # [8k]
        sfa_level2_topk: int = 1024, # [1k]
        sfa_cmp_ratio: int = 4,
        sfa_sliding_blocks: int = 3,
        sfa_slidiing_window: int = 128,
    ):
        """
        初始化静态缓存生成器
        
        Args:
            model: 预训练模型
            max_batch_size: 最大批处理大小
            max_seq_len: 最大序列长度
            max_new_tokens: 最大生成token数
        """
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.max_new_tokens = max_new_tokens
        self.enable_sfa = enable_sfa
        self.enable_offloading = enable_offloading
        self.sfa_block_size = sfa_block_size
        self.sfa_level1_topk = sfa_level1_topk
        self.sfa_level2_topk = sfa_level2_topk
        self.sfa_cmp_ratio = sfa_cmp_ratio
        self.sfa_sliding_blocks = sfa_sliding_blocks
        self.sfa_sliding_window = sfa_slidiing_window
        
        # 获取模型配置
        self.config = model.config
        self.num_layers = self.config.num_hidden_layers
        self.num_heads = self.config.num_attention_heads
        self.num_kv_heads = self.config.num_key_value_heads
        self.head_dim = self.config.hidden_size // self.num_heads
        
        # 预分配静态KV缓存
        self.cache = self._prepare_cache()
        
        # 缓存使用状态
        self.lengths = torch.zeros(max_batch_size, dtype=torch.int32, device=model.device)

        if model.config._attn_implementation != "flash_attention_2":
            model.config._attn_implementation = "flash_attention_2"
            logger.warning("Setting the attention implementation as FlashAttention2")
        
        (
            level1_indexer_interface,
            level2_indexer_interface,
            decode_interface
        ) = self._preprae_interface()

        for name, module in self.model.named_modules():
            if isinstance(module, Qwen3Attention):
                if enable_sfa:
                    if enable_offloading:
                        module.forward = MethodType(kvo_qwen3_sfa_forward, module)
                    else:
                        module.forward = MethodType(qwen3_sfa_forward, module)
                    module.sfa_args = (
                        level1_indexer_interface,
                        level2_indexer_interface,
                        decode_interface,
                        self.sfa_block_size,
                        self.head_dim // 2,
                        (self.head_dim // 2) // self.sfa_cmp_ratio,
                    )
                    logger.info(f"register qwen3 SFA patch: [{name}]")
                else:
                    if enable_offloading:
                        module.forward = MethodType(qwen3_fa_forward, module)
                    else:
                        module.forward = MethodType(kvo_qwen3_fa_forward, module)
                    module.sfa_args = (decode_interface,)
                    logger.info(f"register qwen3 FA patch: [{name}]")
    
    def _sort_and_batch_prompts(
        self,
        prompts: List[str],
        tokenizer
    ) -> List[Dict[str, Any]]:
        """
        根据长度排序prompts并划分为批次
        
        Args:
            prompts: 输入prompt列表
            tokenizer: 分词器
            
        Returns:
            批次列表，每个批次包含排序后的prompts和对应的索引
        """
        # 计算每个prompt的token长度
        prompt_lengths = []
        for prompt in prompts:
            tokens = tokenizer.encode(prompt, add_special_tokens=False)
            prompt_lengths.append(len(tokens))
        
        # 按长度排序（从长到短，提高填充效率）
        sorted_indices = sorted(
            range(len(prompts)),
            key=lambda i: prompt_lengths[i],
            reverse=True
        )
        
        # 创建批次
        batches = []
        current_batch = []
        current_batch_indices = []
        current_batch_lengths = []
        
        for idx in sorted_indices:
            prompt_len = prompt_lengths[idx]
            
            # 检查是否超过最大序列长度
            if prompt_len > self.max_seq_len:
                logger.warning(f"Prompt length {prompt_len} exceeds max_seq_len {self.max_seq_len}, will be truncated")
            
            # 检查是否可以加入当前批次
            if len(current_batch) <= self.max_batch_size:  # 长度相近的放在一起
                current_batch.append(prompts[idx])
                current_batch_indices.append(idx)
                current_batch_lengths.append(prompt_len)
            else:
                # 开始新的批次
                if current_batch:
                    batches.append({
                        'prompts': current_batch,
                        'indices': current_batch_indices,
                        'lengths': current_batch_lengths
                    })
                
                current_batch = [prompts[idx]]
                current_batch_indices = [idx]
                current_batch_lengths = [prompt_len]
        
        # 添加最后一个批次
        if current_batch:
            batches.append({
                'prompts': current_batch,
                'indices': current_batch_indices,
                'lengths': current_batch_lengths
            })
        
        logger.info(f"Sorted {len(prompts)} prompts into {len(batches)} batches")
        for i, batch in enumerate(batches):
            logger.info(f"Batch {i}: {len(batch['prompts'])} prompts, lengths: {batch['lengths']}")
        
        return batches
        
    def _prefill(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预填充阶段：处理输入prompt并填充KV缓存
        
        Args:
            input_ids: 输入token IDs [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            
        Returns:
            logits: 最后一个位置的logits [batch_size, vocab_size]
            next_token_logits: 用于生成下一个token的logits
        """
        batch_size, seqlen = input_ids.shape
                
        # 准备模型输入
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": True,  # 使用自定义Cache
            "past_key_values": self.cache,
        }
        
        # 调用模型forward
        outputs = self.model(**model_inputs)
        
        # 获取最后一层的logits
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        last_logits = logits[:, -1, :]  # [batch_size, vocab_size]

        self.lengths[:batch_size] = seqlen
                
        return logits, last_logits
    
    def _decode_step(
        self,
        next_tokens: torch.Tensor,
        batch_indices: List[int]
    ) -> torch.Tensor:
        """
        解码单步：使用静态缓存生成下一个token
        
        Args:
            next_tokens: 下一个token IDs [batch_size]
            batch_indices: 当前处理的batch索引列表
            
        Returns:
            next_token_logits: 下一个token的logits [batch_size, vocab_size]
        """
        
        # 准备模型输入
        model_inputs = {
            "input_ids": next_tokens.unsqueeze(1),  # [batch_size, 1]
            "attention_mask": None,  # 解码阶段不需要attention_mask
            "use_cache": True,
            "past_key_values": self.cache,
        }
        
        # 调用模型forward
        outputs = self.model(**model_inputs)
        
        # 获取logits
        logits = outputs.logits  # [batch_size, 1, vocab_size]
        next_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]
        
        # 更新缓存位置
        for idx in batch_indices:
            self.lengths[idx] += 1
        
        return next_token_logits
    
    def _batch_generate(
        self,
        batch_prompts: List[str],
        tokenizer,
        max_new_tokens: int,
        temperature: float,
        do_sample: bool
    ) -> List[str]:
        """
        单个批次的生成
        
        Args:
            batch_prompts: 批次内的prompts
            batch_indices: 原始索引
            tokenizer: 分词器
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            do_sample: 是否采样
            
        Returns:
            生成的文本列表（按原始顺序）
        """
        batch_size = len(batch_prompts)
        
        # 重置缓存
        self._reset_cache()
        
        # 编码输入
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(self.model.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # 预填充阶段
        _, next_token_logits = self._prefill(input_ids, attention_mask)
        
        # 初始化生成结果
        generated_ids = input_ids.clone()
        current_batch_indices = list(range(batch_size))
        
        # 解码阶段
        for step in range(max_new_tokens):
            # 生成下一个token
            if do_sample:
                # 采样模式
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # 贪心解码
                next_tokens = torch.argmax(next_token_logits, dim=-1)
                       
            # 更新生成结果
            generated_ids = torch.cat([generated_ids, next_tokens.unsqueeze(1)], dim=1)
            
            # 检查是否所有序列都生成了结束token
            unfinished = next_tokens != tokenizer.eos_token_id
            if not unfinished.any():
                break
            
            # 更新batch_indices（只继续未完成的序列）
            current_batch_indices = [i for i, keep in enumerate(unfinished.tolist()) if keep]
            if not current_batch_indices:
                break
            
            # 解码下一步
            next_token_logits = self._decode_step(next_tokens, current_batch_indices)
        
        # 解码生成结果
        batch_results = []
        for i in range(batch_size):
            text = tokenizer.decode(generated_ids[i, :self.lengths[i]], skip_special_tokens=True)
            batch_results.append(text)
        
        return batch_results
    
    @torch.no_grad()
    def generate(
        self,
        prompts: List[str],
        tokenizer,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.7,
        do_sample: bool = False
    ) -> List[str]:
        """
        批量生成文本（支持排序和分批）
        
        Args:
            prompts: 输入prompt列表
            tokenizer: 分词器
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            do_sample: 是否采样
            
        Returns:
            生成的文本列表（保持原始顺序）
        """
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        
        # 1. 排序并分批
        batches = self._sort_and_batch_prompts(prompts, tokenizer)
        
        # 2. 初始化结果数组
        results = [None] * len(prompts)
        
        # 3. 按批次处理
        for batch_info in batches:
            batch_prompts = batch_info['prompts']
            batch_indices = batch_info['indices']
            
            logger.info(f"Processing batch with {len(batch_prompts)} prompts")
            
            # 处理当前批次
            batch_results = self._batch_generate(
                batch_prompts=batch_prompts,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample
            )
            
            # 将结果放回原始位置
            for idx, result in zip(batch_indices, batch_results):
                results[idx] = result
        
        # 4. 验证所有结果都已生成
        if None in results:
            raise RuntimeError("Some prompts were not processed")
        
        return results
    
    def _reset_cache(self):
        """重置缓存状态"""
        self.cache = self._prepare_cache()

    def _prepare_cache(self):
        if self.enable_sfa:
            if self.enable_offloading:
                cache = KVO_MHASfaCache(
                    self.config, 
                    self.max_batch_size, 
                    self.max_seq_len,
                    sfa_block_size=self.sfa_block_size,
                    sfa_cmp_ratio=self.sfa_cmp_ratio,
                )
            else:
                cache = MHASfaCache(
                    self.config, 
                    self.max_batch_size, 
                    self.max_seq_len,
                    sfa_block_size=self.sfa_block_size,
                    sfa_cmp_ratio=self.sfa_cmp_ratio,
                )
        else:
            if self.enable_offloading:
                cache = KVO_MHAFaCache(
                    self.config, 
                    self.max_batch_size, 
                    self.max_seq_len,
                )
            else:
                cache = MHAFaCache(
                    self.config, 
                    self.max_batch_size, 
                    self.max_seq_len,
                )
        return cache
    
    def _preprae_interface(self):
        if not self.enable_sfa:
            level1_indexer_interface = None
            level2_indexer_interface = None
            decode_interface = MHAInterface(
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.head_dim,
            )
        else:
            if not self.enable_offloading:
                level1_indexer_interface = MHAIndexerLevel1Interface(
                    self.num_heads,
                    self.num_kv_heads,
                    self.head_dim,
                    self.sfa_level1_topk,
                    self.sfa_sliding_blocks
                )
                level2_indexer_interface = MHAIndexerLevel2Interface(
                    self.num_heads,
                    self.num_kv_heads,
                    self.head_dim // self.sfa_cmp_ratio,
                    self.sfa_level1_topk,
                    self.sfa_level2_topk,
                    self.sfa_block_size,
                    self.sfa_sliding_blocks,
                )
                decode_interface = SparseMHAInterface(
                    self.num_heads,
                    self.num_kv_heads,
                    self.head_dim,
                    self.head_dim,
                    self.sfa_level2_topk,
                )
            else:
                level1_indexer_interface = KVO_MHAIndexerLevel1Interface(
                    self.num_heads,
                    self.num_kv_heads,
                    self.head_dim,
                    self.sfa_level1_topk,
                    self.sfa_sliding_blocks
                )
                level2_indexer_interface = KVO_MHAIndexerLevel2Interface(
                    self.num_heads,
                    self.num_kv_heads,
                    self.head_dim // self.sfa_cmp_ratio,
                    self.sfa_level1_topk,
                    self.sfa_level2_topk,
                    self.sfa_block_size,
                    self.sfa_sliding_window,
                )
                decode_interface = KVO_SparseMHAInterface(
                    self.num_heads,
                    self.num_kv_heads,
                    self.head_dim,
                    self.head_dim,
                    self.sfa_level2_topk,
                )
        return (
            level1_indexer_interface,
            level2_indexer_interface,
            decode_interface
        )