import os
import torch
import json
from datetime import datetime
from argparse import ArgumentParser
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from sparse_attn.arguments import add_sparse_attn_args
from sparse_attn.patches import register_patch
from sparse_attn.metrics import get_metrics
from sparse_attn.info import (
    get_method_name_with_info,
    resolve_treatment_snapshot,
    TREATMENT_MANIFEST_SIDECAR_SUFFIX,
)

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default="Qwen3-8B") # meta-llama/Meta-Llama-3.1-8B-Instruct
    parser.add_argument('--model', type=str, default="hf_models/downloads/Qwen/Qwen3-8B") # meta-llama/Meta-Llama-3.1-8B-Instruct
    parser.add_argument('--lm_eval_batch_size', type=int, default=1)
    parser.add_argument('--tasks', nargs='+',
        default=[
            "niah_single_1", "niah_single_2",
            "niah_multikey_1", "niah_multikey_2",
            "niah_multiquery", "niah_multivalue",
            # "ruler_qa_squad", "ruler_qa_hotpot",
        ],
        help='Tasks to evaluate on LM Eval.'
    )
    # "arc_easy", "arc_challenge", "boolq", "hellaswag", "lambada_openai", "piqa", "siqa", "winogrande",
    parser.add_argument('--max_length', type=int, default=1024 * 64)
    parser.add_argument('--output_dir', type=str, default="./exp/results_ruler", help='Directory to save evaluation results')

    add_sparse_attn_args(parser)
    
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.model_name)
    return args

@torch.no_grad()
def main():
    args = parse_args()
    print("model: {}".format(args.model))

    # 082（TL-E121-LLM-EVAL-SNAPSHOT-082）：tli 臂在首次求 method name
    # 之前一次冻结 treatment snapshot（081 同款契约）——本入口此前在
    # 无 snapshot 状态下先固化名称、后逐层 patch，配置文件在窗口内
    # 替换可形成「名 A / 运行时 B」乃至跨层混代。name / register_patch /
    # 结果落盘 manifest 全程消费同一 snapshot。非 tli 臂零改动。
    snapshot = (
        resolve_treatment_snapshot(args) if args.method == "tli" else None)
    method_name = get_method_name_with_info(args, snapshot)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 生成时间戳用于文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_filename = f"{method_name}_{timestamp}"
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    ).eval()

    prompts = [
        [
            {
                "role": "user", "content": (
                    "Please help me implement the python code that print \"Hello World!\"."
                )
            },
        ],
    ]
    formatted_prompts = [tokenizer.apply_chat_template(messages, tokenize=False) for messages in prompts]

    inputs = tokenizer(formatted_prompts, return_tensors="pt", padding=True).to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=16, do_sample=True)
    results = tokenizer.batch_decode(outputs)
    print(results)

    register_patch(model, args, snapshot)

    import lm_eval
    from lm_eval import utils as lm_eval_utils
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager

    hflm = HFLM(pretrained=model, tokenizer=tokenizer, max_batch_size=args.lm_eval_batch_size, max_length=args.max_length)

    task_manager = TaskManager(
        include_path="./benchmark/RULER/lm_eval_tasks", 
        include_defaults=False,
        metadata={"max_seq_lengths": [8192, 8192 * 2, 8192 * 4], "pretrained": args.model}
    )
    task_names = lm_eval_utils.pattern_match(args.tasks, task_manager.all_tasks)

    print("task_names: {}".format(task_names))
    result_dict = lm_eval.simple_evaluate(
        hflm, tasks=task_names, 
        batch_size=args.lm_eval_batch_size, 
        max_batch_size=args.lm_eval_batch_size,
        task_manager=task_manager,
        confirm_run_unsafe_code=True,
    )
    result_table = lm_eval_utils.make_table(result_dict)
    
    # 获取稀疏注意力指标
    metrics = {
        "mean_select_tokens": get_metrics().get_select_tokens(),
        "k_delta_max": get_metrics().get_k_delta_max(),
        "k_delta_mean": get_metrics().get_k_delta_mean()
    }
    
    # 打印结果
    print(args.model)
    print(result_table)
    print("mean select tokens:", metrics["mean_select_tokens"])
    
    # 保存结果到文件
    save_results(args, result_filename, result_dict, result_table, metrics,
                 snapshot)

def save_results(args, filename, result_dict, result_table, metrics,
                 snapshot=None):
    """保存评估结果到文件"""
        
    # 保存文本格式的结果表格
    txt_path = os.path.join(args.output_dir, f"{filename}.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Evaluation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Tasks: {', '.join(args.tasks)}\n")
        f.write(f"Batch Size: {args.lm_eval_batch_size}\n")
        f.write(f"Max Length: {args.max_length}\n")
        f.write("\n" + "="*80 + "\n\n")
        f.write("Evaluation Results:\n")
        f.write(str(result_table))
        f.write("\n\n" + "="*80 + "\n\n")
        f.write("Sparse Attention Metrics:\n")
        f.write(f"Mean Select Tokens: {metrics['mean_select_tokens']}\n")
    print(f"Text results saved to: {txt_path}")

    # 082：tli 臂结果旁挂 canonical treatment manifest sidecar（与
    # LongBench 写门/sidecar 同一 manifest 对象、同一后缀语义）——
    # 结果身份不靠短哈希单点承载，可独立复核。
    if snapshot is not None:
        sidecar_path = txt_path + TREATMENT_MANIFEST_SIDECAR_SUFFIX
        with open(sidecar_path, "w", encoding="utf-8") as f:
            f.write(snapshot.manifest_json)
        print(f"Treatment manifest saved to: {sidecar_path}")
    

if __name__ == '__main__':
    main()
