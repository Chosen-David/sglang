# #66 RULER 评测适配器：把 KVCache-Factory 预生成的 RULER 数据
# （data/RULER/{4096,8192,16384,32768}/×11 任务×100 条，零外网；65536/131072
# 由 gen_ruler_long.py 循环体扩展生成）喂进本仓库 sparse_attn 管线
# （TIA/TLI/Quest/FullKV 同一 monkeypatch）。
# 口径对齐官方 RULER：prompt 原样（无 chat template）、max_new=64、
# min_length=context+1 防复读、string_match_all 打分（score_ruler.py）。
# 用法（run_ruler.sh 批量调度）：
#   python -u -m benchmark.RULER.pred_ruler \
#     --model Qwen3-8B --model_path $MODEL \
#     --task niah_single_1 --context_length 8192 --method tli --t $TS \
#     --data-root /home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/RULER \
#     --output-dir exp/results_ruler/Qwen3-8B \
#     --tia_level2_topk 1024 --tia_level2_cmp_ratio 4 --pred_postfix _1024 \
#     --max-num 500
# ---- E109 三件套 #3：YaRN 长上下文扩展（2026-10-09）----
# Qwen3-8B 原生 max_position_embeddings=40960，65536/131072 档须 YaRN。
# 配方按 Qwen3 官方 model card：rope_scaling={yarn, factor, original 40960,
# beta_fast 32, beta_slow 1}；128K 档 factor=4.0（官方原配方，effective
# 163840），64K 档取半档 factor=2.0（effective 81920，留 25% 余量；与官方
# 整档倍率风格一致，且 ramp 插值更平缓、近程精度损失更小）。
# 设计文档口径（逐层参数求解器 §10.4）：dense 与所有稀疏方法同一位置缩放/
# tokenizer/最大输出；先报告 dense 长度能力再看稀疏退化。
# YaRN 对 TLI 索引透明：indexer 消费的是 post-RoPE 的 K（qwen3_attn_patch
# 在 apply_rotary_pos_emb + cache.update 之后传入），位置缩放自动生效。
# logits 显存账（超 32K 档自动 logits_to_keep=1，等价性审计 0233 已 A/B
# 实测 bitwise 相等）：131072×151936 vocab bf16 全序列 logits ≈ 39.8GB 纯
# 浪费（只消费末 token），32K 及以下档路径逐位不变（零扰动纪律）。
# ---- 057（GPT 2026-10-10 0330 审计 TL-E119-YARN-IDENTITY-057）----
# 本脚本此前只在 stdout 打印 effective factor、不落任何原生配置证据，formal
# 汇总只能消费操作者事后 CLI 声明——128K 批次出现「manifest 声明 2.0 vs
# 生成链自动档 4.0」的身份冲突。现每次运行把 effective yarn 配置原子写入
# 预测产物旁挂 receipt（{pred 基名}-yarn_receipt.json，见 yarn_receipt.py），
# 生成路径/行格式/自动档语义零改动。
import argparse
import hashlib
import json
import os
import random
import sys

# 057：生产者原生 effective-config receipt（yarn_receipt.py 零重依赖，
# 生成/消费/测试三方共享同一口径）。REPO 入 sys.path 前置于一切仓库内
# 导入——保证以普通脚本方式（python benchmark/RULER/pred_ruler.py）被
# 调用时 sparse_attn 与包内导入同样可达。
REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import (  # noqa: E402
    AutoConfig, AutoModelForCausalLM, AutoTokenizer)

from benchmark.RULER.yarn_receipt import (  # noqa: E402
    _file_sha256, build_yarn_receipt, resolve_yarn_config,
    write_yarn_receipt,
)
from sparse_attn.arguments import add_sparse_attn_args  # noqa: E402
from sparse_attn.patches import register_patch  # noqa: E402
from sparse_attn.metrics import get_metrics  # noqa: E402
from sparse_attn.info import get_method_name_with_info  # noqa: E402

RULER_TASKS = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multiquery", "niah_multivalue", "cwe", "fwe", "vt",
]
MAX_GEN = 64  # 官方 RULER 口径（全部任务统一）
SEED = 42     # seed_everything 默认值（receipt 身份字段，057）
# Qwen3-8B 原生上下文上限（config.max_position_embeddings）
QWEN3_NATIVE_MPE = 40960
# YaRN factor 自动档位（官方 128K 配方 factor 4.0 的等比半档）
YARN_FACTOR_AUTO = {65536: 2.0, 131072: 4.0}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen3-8B")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--task", type=str, choices=RULER_TASKS, required=True)
    parser.add_argument("--context_length", type=int,
                        choices=[4096, 8192, 16384, 32768, 65536, 131072],
                        required=True)
    parser.add_argument("--data-root", type=str, required=True,
                        help="KVCache-Factory data/RULER 目录")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--t", type=str, default="ruler run")
    parser.add_argument("--pred_postfix", type=str, default="")
    parser.add_argument("--max-num", type=int, default=500,
                        help="每任务样本数（0=全量）")
    parser.add_argument("--yarn", action="store_true",
                        help="启用 YaRN 位置扩展（Qwen3 官方配方；"
                             "65536/131072 档必开，32768 及以下档保持关闭"
                             "以维持既有口径零扰动）")
    parser.add_argument("--yarn_factor", type=float, default=None,
                        help="YaRN factor（默认按档位自动：65536→2.0、"
                             "131072→4.0；显式传入可覆盖做消融）")
    add_sparse_attn_args(parser)
    return parser.parse_args()


def seed_everything(seed=SEED):   # 057：默认值单源化（receipt 身份字段）
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    args = parse_args()
    seed_everything()

    # ---- YaRN 校验与配置（E109 三件套 #3 + 057 effective 单点解析）----
    # 超过原生 40960 的档位必须显式 --yarn（防误用原生 RoPE 跑超限位置）；
    # 原生档位默认关闭（既有 4K-32K 路径零扰动），显式开也允许（消融用）。
    use_yarn = args.yarn
    if args.context_length > QWEN3_NATIVE_MPE and not use_yarn:
        raise SystemExit(
            f"[yarn] context_length={args.context_length} 超出 Qwen3-8B 原生 "
            f"{QWEN3_NATIVE_MPE}，必须加 --yarn（run_ruler_e109.sh 已自动）")
    # 057：effective 配置单点解析（修复前 print 与 config 构造两处内联各
    # 算一遍；resolve_yarn_config 与原内联语义逐位一致——显式 factor 覆盖，
    # 否则档位自动表回退），后续 receipt/config 全部消费同一份结果，
    # 「实际生效值」从此有生产者原生证据。
    factor, rope_scaling = resolve_yarn_config(
        use_yarn, args.yarn_factor, args.context_length,
        YARN_FACTOR_AUTO, QWEN3_NATIVE_MPE)
    if use_yarn:
        print(f"[yarn] enabled: factor={factor} "
              f"(effective context = {int(QWEN3_NATIVE_MPE * factor)})")

    data_file = os.path.join(args.data_root, str(args.context_length),
                             f"{args.task}.jsonl")
    rows = [json.loads(l) for l in open(data_file)]
    if args.max_num and len(rows) > args.max_num:
        rows = rows[:args.max_num]  # topk 口径（确定性）
    print(f"[ruler] task={args.task} L={args.context_length} "
          f"n={len(rows)} method={args.method}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True)
    # YaRN：在模型加载前把 rope_scaling 写进 config（transformers 从
    # config 构造 YarnRotaryEmbedding；同时写 rope_type/type 兼容新旧字段）
    config = AutoConfig.from_pretrained(args.model_path,
                                        trust_remote_code=True)
    if use_yarn:
        config.rope_scaling = rope_scaling
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, config=config, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto").eval()
    register_patch(model, args)

    # 超原生档位启用 logits_to_keep=1：prefill 只回传末 token logits
    # （131072×151936 bf16 全序列 logits ≈ 39.8GB 纯浪费；等价性由审计
    # 0233 A/B 实测 bitwise 相等保证；≤32K 档不启用，路径逐位不变）
    keep_logits = args.context_length > 32768

    eos_ids = [tokenizer.eos_token_id]
    nl = tokenizer.encode("\n", add_special_tokens=False)
    if nl:
        eos_ids.append(nl[-1])

    out_dir = os.path.join(args.output_dir, f"L{args.context_length}",
                           f"pred{args.pred_postfix}")
    os.makedirs(out_dir, exist_ok=True)
    method_name = get_method_name_with_info(args)
    out_path = os.path.join(
        out_dir, f"{args.task}-{method_name}-{args.t}.jsonl")
    fout = open(out_path, "w", encoding="utf-8")

    # ---- 057：生产者原生 effective-config receipt（旁挂原子写）----
    # 记录本进程实际生效配置（自动档解析/显式覆盖之后），供 formal 消费
    # 侧以生产者证据闭合 yarn 身份（修复「manifest 声明 2.0 vs 生成链
    # 自动档 4.0」的 128K 身份冲突）。三点设计：
    #   ① 旁挂 .json 不进任何 {task}-*.jsonl glob——下游 wc -l /
    #      best-file 仲裁 / SKIP 幂等零扰动，既有产物行格式不动；
    #   ② 临时文件 + fsync + os.replace 原子落盘——中断不留半写
    #      receipt（半写比缺失危险：消费侧把存在当证据，损坏即 fail）；
    #   ③ 写在生成循环之前：receipt 描述的是产出这些预测字节的运行
    #      配置；进程中途崩溃留下的 partial 文件由 formal 的
    #      min-samples / best-file 仲裁兜底，receipt 本身仍如实。
    model_cfg_path = os.path.join(args.model_path, "config.json")
    yarn_receipt = build_yarn_receipt(
        yarn_enabled=use_yarn,
        effective_factor=factor,
        yarn_factor_cli=args.yarn_factor,
        rope_scaling=rope_scaling,
        context_length=args.context_length,
        task=args.task,
        model_path=os.path.abspath(args.model_path),
        model_config_sha256=(
            _file_sha256(model_cfg_path)
            if os.path.isfile(model_cfg_path) else None),
        native_mpe=QWEN3_NATIVE_MPE,
        generation_params={
            "max_gen": MAX_GEN, "max_num": args.max_num, "seed": SEED,
            "method": args.method, "pred_postfix": args.pred_postfix,
            "t": args.t,
        },
        producer_script_path="benchmark/RULER/pred_ruler.py",
        producer_script_sha256=_file_sha256(os.path.abspath(__file__)))
    yarn_rcp_path = write_yarn_receipt(out_path, yarn_receipt)
    print(f"[yarn-receipt] effective config -> {yarn_rcp_path}")

    for row in tqdm(rows):
        torch.cuda.empty_cache()
        input_ids = tokenizer(row["input"], truncation=False,
                              return_tensors="pt").input_ids.to("cuda")
        context_length = input_ids.shape[-1]
        with torch.no_grad():
            output = model(input_ids=input_ids, past_key_values=None,
                           use_cache=True,
                           **({"logits_to_keep": 1} if keep_logits else {}))
            past = output.past_key_values
            pred_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            gen = [pred_idx.item()]
            for _ in range(MAX_GEN - 1):
                # min_length 语义：禁止在 context 结束前停（RULER 官方口径）
                if len(gen) < 2 and pred_idx.item() in eos_ids:
                    pass  # 首位不许停（近似 min_length=context+1）
                out = model(input_ids=pred_idx, past_key_values=past,
                            use_cache=True)
                past = out.past_key_values
                pred_idx = out.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                gen.append(pred_idx.item())
                if pred_idx.item() in eos_ids and len(gen) > 1:
                    break
        pred = tokenizer.decode(gen, skip_special_tokens=True)
        metrics = get_metrics()
        budget = metrics.get_select_tokens()
        metrics.clear()
        # E116c（GPT 0826 审计 TL-RULER-SAMPLE-GATE-026）：样本身份绑定源行
        # index（RULER 源 jsonl 自带 0..99 int，比 LB v1 的 data_fp 指纹更直接）
        # + answers canonical SHA256 前 16 位（与 LongBench manifest 口径一致），
        # 评分端据此做 fail-closed 集合闭包校验；其余字段与行为零改动。
        fout.write(json.dumps({
            "pred": pred, "answers": row["outputs"],
            "length": row["length"], "budget": budget,
            "_id": f"{args.task}:{row['index']}",
            "_answers_sha": hashlib.sha256(json.dumps(
                row["outputs"], ensure_ascii=False, sort_keys=True)
                .encode("utf-8")).hexdigest()[:16],
        }, ensure_ascii=False) + "\n")
        fout.flush()
    fout.close()
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
