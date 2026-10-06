import os
import torch
from transformers.utils import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from tls_attn.generator.mha_generator import MHAGenerator  # 导入MHAGenerator
from tls_attn.generator.mha_generator import MHAGenerator  # 导入MHAGenerator

logger = logging.get_logger(__name__)

def main(
    max_batch_size=4,
    max_seqlen=2048,
    max_new_tokens=64,
):
    # 设置日志
    logging.set_verbosity_info()
    
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    model_name = "hf_models/downloads/Qwen/Qwen3-8B" # meta-llama/Meta-Llama-3.1-8B-Instruct
    
    logger.info("model_name: {}".format(model_name))
            
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
    # 添加MHAGenerator使用示例
    # 加载模型
    logger.info("开始加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    ).eval()
    
    # 创建MHAGenerator实例
    logger.info("创建MHAGenerator实例...")
    generator = MHAGenerator(
        model=model,
        max_batch_size=max_batch_size,
        max_seq_len=max_seqlen,
        max_new_tokens=max_new_tokens,
        enable_sfa=False,
        enable_offloading=True,
    )

    messages = [
        [
            {
                "role": "user", "content": (
                    "Please help me implement the python code that print \"Hello World!\"."
                )
            },
        ],
        [
            {
                "role": "user", "content": (
                    "Please help me implement the python code that sort an arrary."
                )
            },
        ],
    ]
    # 准备输入prompts（转换为字符串格式）
    prompts = []
    for message in messages:
        # 将对话格式转换为字符串
        formatted_prompt = tokenizer.apply_chat_template(
            message, 
            tokenize=False, 
            add_generation_prompt=True,
            enable_thinking=False
        )
        prompts.append(formatted_prompt)
    
    # 使用generator生成文本
    logger.info("开始生成文本...")
    print("\n=== 使用MHAGenerator生成文本 ===")
    results = generator.generate(
        prompts=prompts,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        do_sample=False
    )
    logger.info("文本生成完成")
    
    # 打印结果
    for i, (prompt, result) in enumerate(zip(prompts, results)):
        print(f"\n--- Prompt {i+1} ---")
        print(f"输入: {prompt}")
        print(f"输出: {result}")
        print("-" * 50)
    
    logger.info("程序运行结束")

if __name__ == '__main__':
    main()
