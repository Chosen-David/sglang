# Modified from: https://github.com/mit-han-lab/Quest/blob/main/evaluation/LongBench/pred.py

import os
import re
from datasets import load_dataset
import torch
import json
from transformers import (
    AutoTokenizer,
    AutoConfig,
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoModelForCausalLM,
)
from tqdm import tqdm
import numpy as np
import random
import argparse

from benchmark.LongBench.lbv2_choice import extract_choice_official
from sparse_attn.arguments import add_sparse_attn_args
from sparse_attn.patches import register_patch
from sparse_attn.metrics import get_metrics
from sparse_attn.info import get_method_name_with_info


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=[
            "llama2-7b-chat-4k",
            "longchat-v1.5-7b-32k",
            "xgen-7b-8k",
            "internlm-7b-8k",
            "chatglm2-6b",
            "chatglm2-6b-32k",
            "chatglm3-6b-32k",
            "vicuna-v1.5-7b-16k",
            "Mistral-7B-Instruct-v0.3",
            "Meta-Llama-3.1-8B-Instruct",
            'Qwen3-8B',
            'Qwen3-14B',
            'Qwen3-32B',
        ],
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="If model has been downloaded locally",
    )
    parser.add_argument("--e", type=int, default=0, help="Evaluate on LongBench-E")
    parser.add_argument("--task", type=str, help="task name", required=True)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--dataset-path", type=str)
    parser.add_argument("--config-path", type=str)
    parser.add_argument("--t", type=str, default="benchmark launch time")
    parser.add_argument(
        "--algo-config-path",
        type=str,
        help="Algorithm config path",
    )
    parser.add_argument('--pred_postfix', type=str, default="")
    # 稀疏 prefill（用户 2026-10-08 指令，MoBA/NSA/DSA 口径）：设
    # TLI_SPARSE_PREFILL=1 环境变量（须在模型加载/patch 挂载前；patch 层
    # 逐 forward 读 env，此处 parse 后立即设置即可）。默认不传 = 0 =
    # prefill 走原 dense 分支逐位不变（在跑链零扰动）。
    parser.add_argument(
        '--tli-sparse-prefill', action='store_true',
        help='prefill 也走 indexer 稀疏选择（chunk 共享选择，MoBA 口径）',
    )

    add_sparse_attn_args(parser)

    return parser.parse_args(args)


# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template

        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    elif "Llama-3.1" in model_name:
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|> {prompt} <|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
    elif "Qwen3" in model_name:
        prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    elif "GLM-4" in model_name:
        message = [{"role": "user", "content": prompt},]
        prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return prompt


def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response


def extract_choice_letter(text):
    """LongBench-v2 四选一解析（E117b 统一到 lbv2_choice.py 官方口径）：
    仅 'The correct answer is (X)' / 'The correct answer is X' 两条大小写
    敏感格式，无匹配返回 None。历史兜底（IGNORECASE 首个独立字母）会把
    冠词 a 判成选项 A，已删除——pred_choice 落盘值与正式评分口径一致。
    """
    return extract_choice_official(text)


def get_pred(
    model,
    tokenizer,
    data,
    max_length,
    max_gen,
    prompt_format,
    dataset,
    device,
    model_name,
    data_fp="",
):
    preds = []

    # HACK: sample 10 examples for debugging
    # data = np.random.choice(data, 10)

    for row_idx, json_obj in enumerate(tqdm(data)):
        torch.cuda.empty_cache()
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(
            prompt, truncation=False, return_tensors="pt"
        ).input_ids[0]
        if "chatglm3" in model_name:
            tokenized_prompt = tokenizer(
                prompt, truncation=False, return_tensors="pt", add_special_tokens=False
            ).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(
                tokenized_prompt[:half], skip_special_tokens=True
            ) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in [
            "trec",
            "triviaqa",
            "samsum",
            "lsht",
            "lcc",
            "repobench-p",
        ]:  # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)

        # split the prompt and question (simulate decoding in the question stage)
        if dataset in ["qasper", "hotpotqa"]:
            q_pos = prompt.rfind("Question:")
        elif dataset == "lbv2":
            # 官方模板无 "Question:" 锚点；question stage 取选项块起点
            q_pos = prompt.rfind("Choices:")
        elif dataset in ["multifieldqa_en", "gov_report"]:
            q_pos = prompt.rfind("Now,")
        elif dataset in ["triviaqa"]:
            q_pos = prompt.rfind("Answer the question")
        elif dataset in ["narrativeqa"]:
            q_pos = prompt.rfind("Do not provide")
        else:
            q_pos = -1

        # max simulation length is 100
        q_pos = max(len(prompt) - 100, q_pos)

        if q_pos != None:
            question = prompt[q_pos:]
            prompt = prompt[:q_pos]

        if "chatglm3" in model_name:
            # input = prompt.to(device)
            input = prompt.to("cuda")
        else:
            # input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to("cuda")
            q_input = tokenizer(question, truncation=False, return_tensors="pt").to(
                "cuda"
            )
            q_input.input_ids = q_input.input_ids[:, 1:]

        context_length = input.input_ids.shape[-1] + q_input.input_ids.shape[-1]

        if (
            dataset == "samsum"
        ):  # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            # B03 修复（GPT 审查 2026-10-08）：原版 model.generate(**input) 缺两件事——
            # ① q_input 切出的 question 没接回，只喂了 prefix；② output 赋值后
            # generated_content 从未赋值，下方共用 decode 必 UnboundLocalError。
            # 修法：拼接完整输入（与 else 分支同协议），generate 后取新增 token。
            full_ids = torch.cat([input.input_ids, q_input.input_ids], dim=-1)
            output = model.generate(
                input_ids=full_ids,
                attention_mask=torch.ones_like(full_ids),
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=full_ids.shape[-1] + 1,
                eos_token_id=[
                    tokenizer.eos_token_id,
                    tokenizer.encode("\n", add_special_tokens=False)[-1],
                ],
            )[0]
            generated_content = output[full_ids.shape[-1]:].tolist()
        else:
            with torch.no_grad():
                output = model(
                    input_ids=input.input_ids,
                    past_key_values=None,
                    use_cache=True,
                )
                past_key_values = output.past_key_values
                for input_id in q_input.input_ids[0]:
                    output = model(
                        input_ids=input_id.unsqueeze(0).unsqueeze(0),
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    past_key_values = output.past_key_values

                pred_token_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                generated_content = [pred_token_idx.item()]
                for _ in range(max_gen - 1):
                    outputs = model(
                        input_ids=pred_token_idx,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )

                    past_key_values = outputs.past_key_values
                    pred_token_idx = (
                        outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                    )
                    generated_content += [pred_token_idx.item()]
                    if pred_token_idx.item() == tokenizer.eos_token_id:
                        break

            # output = model.generate(
            #     **input,
            #     max_new_tokens=max_gen,
            #     num_beams=1,
            #     do_sample=False,
            #     temperature=1.0,
            # )[0]

        pred = tokenizer.decode(generated_content, skip_special_tokens=True)
        # pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)

        # print("Score: ", flush=True)
        # score_info.print_min_sum_single_query()

        # Record budget
        # avg_budget = budget_info.get_total_avg_budget()
        # avg_score = score_info.get_total_avg_score()
        # budget_info.reset()
        # score_info.reset()
        # avg_B0 = budget_info.get_total_avg_budget_B0()

        metrics = get_metrics()
        budget = metrics.get_select_tokens()
        metrics.clear()

        record = {
            "pred": pred,
            "answers": json_obj["answers"],
            "all_classes": json_obj["all_classes"],
            "length": json_obj["length"],
            "budget": budget,
            "score_sum": None,
            # "B0": avg_B0,
            # "B1": avg_budget,
        }
        if dataset == "lbv2":
            # LongBench-v2：额外落盘解析字母与样本 _id（打分口径 accuracy = pred_choice == answer）
            record["pred_choice"] = extract_choice_letter(pred)
            record["_id"] = json_obj["_id"]
        else:
            # E116a（GPT TL-LBV1-SAMPLE-GATE-024）：v1 记录补稳定 sample_id，
            # 绑定 任务名 + 数据指纹 + 源行 index——评分端据此做 fail-closed
            # 完整性门禁（无重复/与 manifest 集合相等），缺行混行不再静默通过。
            record["_id"] = f"{dataset}:{data_fp}:{row_idx}"
        preds.append(record)
    return preds


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(path, model_name, device, args):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = model.eval()

    register_patch(model, args)

    return model, tokenizer


if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()
    # 稀疏 prefill 门控：在模型加载/patch 挂载前设置环境变量
    # （qwen3_attn_patch 逐 forward 读取；默认不设 = dense 逐位不变）
    if args.tli_sparse_prefill:
        os.environ["TLI_SPARSE_PREFILL"] = "1"

    config_path = args.config_path
    model2maxlen = json.load(open(f"{config_path}/model2maxlen.json", "r"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define your model
    model_name = args.model
    if args.model_path is None:
        model2path_list = json.load(open(f"{config_path}/model2path.json", "r"))
        model2path = model2path_list[model_name]
    else:
        model2path = args.model_path
    model, tokenizer = load_model_and_tokenizer(
        model2path, model_name, device, args
    )
    max_length = model2maxlen[model_name]

    datasets = [args.task]
    dataset2prompt = json.load(open(f"{config_path}/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open(f"{config_path}/dataset2maxlen.json", "r"))

    if args.e:
        dir = f"{args.output_dir}/pred_e{args.pred_postfix}"
        if not os.path.exists(dir):
            os.makedirs(dir)
    else:
        dir = f"{args.output_dir}/pred{args.pred_postfix}"
        if not os.path.exists(dir):
            os.makedirs(dir)

    local_data_set = args.dataset_path
    has_local_data_set = os.path.exists(local_data_set)
    for dataset in datasets:
        if dataset == "lbv2":
            # LongBench-v2（全量三件套之一）：503 题四选一 MCQ，单个 data.json
            # （JSON 数组，非 jsonl）。--dataset-path 可直接给 data.json 文件路径，
            # 也可给目录（依次找 lbv2.json / data.json）。
            # 断点续跑/SKIP 与 v1 同口径：由链脚本按输出 jsonl 行数（全集 503）判定。
            v2_path = local_data_set
            if os.path.isdir(v2_path):
                for cand in ("lbv2.json", "data.json"):
                    if os.path.exists(os.path.join(v2_path, cand)):
                        v2_path = os.path.join(v2_path, cand)
                        break
            if not os.path.isfile(v2_path):
                raise FileNotFoundError(
                    f"lbv2 需要 --dataset-path 指向 LongBench-v2 data.json（或含它的目录），当前路径不存在: {v2_path}"
                )
            with open(v2_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            # 归一化到 v1 管线字段约定：answers 列表 + all_classes（打分走 eval.py 的 lbv2 scorer）
            data = []
            for obj in raw:
                obj = dict(obj)
                obj["answers"] = [obj["answer"]]
                obj["all_classes"] = ["A", "B", "C", "D"]
                data.append(obj)
            print(f"lbv2: loaded {len(data)} examples from {v2_path}", flush=True)
        elif args.e:
            if not has_local_data_set:
                data = load_dataset("THUDM/LongBench", f"{dataset}_e", split="test")
            else:
                data = load_dataset(
                    "json", data_files=f"{local_data_set}/{dataset}_e.jsonl", split="train"
                )
        else:
            if not has_local_data_set:
                data = load_dataset("THUDM/LongBench", dataset, split="test")
            else:
                data = load_dataset(
                    "json", data_files=f"{local_data_set}/{dataset}.jsonl", split="train"
                )

        # Create output.jsonl file name
        dataset_prefix = dataset.split("-")[0]

        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        # E116a：数据指纹（answers+length 逐行 canonical JSON 的 SHA256 前 12 位），
        # 进 _id 绑定冻结输入身份；lbv2 已有官方 _id 不需要
        if dataset == "lbv2":
            data_fp = ""
        else:
            import hashlib as _hashlib
            _fp_src = json.dumps(
                [[obj.get("answers"), obj.get("length")] for obj in data],
                ensure_ascii=False, sort_keys=True,
            )
            data_fp = _hashlib.sha256(_fp_src.encode("utf-8")).hexdigest()[:12]
        preds = get_pred(
            model,
            tokenizer,
            data,
            max_length,
            max_gen,
            prompt_format,
            dataset,
            device,
            model_name,
            data_fp=data_fp,
        )

        method_name = get_method_name_with_info(args)
        out_fn = f"{dataset_prefix}-{method_name}-{args.t}"
        # avoid too long file name
        out_fn = out_fn.replace(" ", "")
        out_fn = out_fn.replace("'", "")
        if len(out_fn) > 245:
            out_fn = out_fn[:245] + "..."
        out_path = f"{dir}/{out_fn}.jsonl"

        with open(out_path, "w", encoding="utf-8") as f:
            for pred in preds:
                json.dump(pred, f, ensure_ascii=False)
                f.write("\n")

        budgets = [pred["budget"] for pred in preds if pred["budget"] is not None]
        ave_budget = sum(budgets) / len(budgets) if len(budgets) > 0 else None
        print(f"ave_budget: {ave_budget}")
