# #66 RULER 评测适配器：把 KVCache-Factory 预生成的 RULER 数据
# （data/RULER/{4096,8192,16384,32768}/×11 任务×100 条，零外网；65536/131072
# 由 gen_ruler_long.py 循环体扩展生成）喂进本仓库 sparse_attn 管线
# （TIA/TLI/Quest/FullKV 同一 monkeypatch）。
# 口径对齐官方 RULER：prompt 原样（无 chat template）、max_new=64、
# min_length=context+1 防复读、string_match_all 打分（正式打分入口
# score_ruler_formal.py——legacy + #198 指针协议双通道；score_ruler.py
# 直接入口仅支持 legacy-direct 产物，pointer root 下 fail-closed，069）。
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
# ---- 059（GPT 2026-10-10 0428 审计 TL-E119-YARN-RECEIPT-BINDING）----
# v1 回执在生成循环之前落最终旁挂路径、且不含预测内容 SHA/行数/run ID/
# 完成标记——「A 进程的预测配 B 进程的回执」可达（同名重跑/中断重跑/
# 并发/事后改写）。升级 producer-yarn-config-v2：预测写不可变 generation，
# 循环结束关闭后算 SHA256+行数，构建 status=complete 完成回执，经
# 原子提交点落位；全生命周期持规范化输出路径 flock。
# ---- 066/crash-recovery（GPT 2026-10-10 1130 审计，#198）----
# 059 的两步 os.replace（预测先替换最终路径 → 回执后替换旁挂）在
# 「B 预测与 A 字节完全相同」的死亡中间态下不可检（三方 SHA 全等，
# formal 接受 A config 标 verified_same_generation=true）。升级为
# 不可变 generation + 单指针原子切换：预测与完成回执同置
# {out}.gen-{attempt_id}/ 目录，提交 = 单次 os.replace 切指针
# {out}.tli_gen（内容 = gen 目录名；指针切换 = 唯一提交信号，E116f
# 同语义）。崩溃三阶段：指针未切 → 旧完整代可见；指针已切 → 新完整
# 代可见；混合代不可达（预测与回执同目录，消费侧从指针同时读两文件）。
import argparse
import hashlib
import json
import os
import random
import sys
from datetime import datetime

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
    _file_sha256, acquire_output_lock, build_yarn_receipt,
    commit_yarn_generation, generation_pointer_path, release_output_lock,
    resolve_yarn_config, stage_yarn_generation, stage_yarn_receipt,
)
from sparse_attn.arguments import add_sparse_attn_args  # noqa: E402
from sparse_attn.patches import register_patch  # noqa: E402
from sparse_attn.metrics import get_metrics  # noqa: E402
from sparse_attn.info import (  # noqa: E402
    get_method_name_with_info, get_treatment_manifest_json,
    resolve_treatment_snapshot)

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

    # 081（TL-E121-OUTPUT-SNAPSHOT-081）：tli 臂在模型加载（文件被任何
    # attention 层消费）之前冻结一次解析的 treatment snapshot——D′ 掩码/
    # 投影基每个文件只 open 一次；后续 method_name/out_path（190 行）、
    # register_patch（175 行）、receipt 的 treatment_manifest_json
    #（297 行）全部消费同一冻结对象。修复前：每层各读一次 + 190 行与
    # 297 行各重读一遍 → 生成窗口内文件被替换时 runtime A / 文件名 B /
    # receipt C 三方混装可达。非 tli 臂零改动（snapshot=None）。
    if args.method == "tli":
        snapshot = resolve_treatment_snapshot(args)
    else:
        snapshot = None

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
    register_patch(model, args, snapshot)

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
    # 081：method_name/out_path 从冻结 snapshot 派生（tli 臂）——与
    # runtime 各层、receipt 的 treatment manifest 同一内容身份。
    method_name = get_method_name_with_info(args, snapshot)
    out_path = os.path.join(
        out_dir, f"{args.task}-{method_name}-{args.t}.jsonl")
    # 076（TL-E121-OUTPUT-ID）：method_name 尾段含 canonical treatment
    # hash _h<hash10>（sparse_attn/info.py——全部输出相关参数排序键
    # JSON→sha256）→ 逻辑路径与 {out}.tli_gen 指针均为 treatment 单射：
    # 异配置不可能命中同一路径（「不切指针」结构性成立），同路径重跑
    # = 同 treatment 的 066/crash-recovery 设计语义（新 gen + 单次原子
    # 切指针）。残余 10-hex 碰撞由回执 v2 同代绑定（062 三方 SHA 一致）
    # 与消费侧 effective_config_sha256 门禁纵深兜底，不在生成侧重复比对。

    # ---- 059 + 066/crash-recovery：不可变 generation + 单指针原子提交 ----
    # 修复前缺陷链：①（059 前）直接以 open(..., "w") 截断最终预测路径
    #   ——生成中途崩溃毁掉上一代完整 best-file；②（059 前）回执在生成
    #   循环之前落最终旁挂路径且无完成标记/预测绑定；③（066/crash-
    #   recovery）059 的两步 os.replace 在「B 预测与 A 字节完全相同」
    #   的死亡中间态下不可检——第一步后、第二步前死亡 → 盘上为「B
    #   物理写入的预测 + A 旧回执」，三方 SHA 全等，formal 接受 A
    #   config 标 verified_same_generation=true（混合代不可检）。
    # 设计（E116f generation/单指针提交同款语义 + 056 flock 口径）：
    #   ① 全生命周期持规范化输出路径锁（realpath(父目录)+basename 键，
    #      042 加固防锁键漂移）——覆盖「gen 目录写入 → SHA/行数计算 →
    #      完成回执 → 指针切换」全程，后到者阻塞等待，崩溃由内核
    #      自动释放；
    #   ② 预测与完成回执同置不可变 generation 目录
    #      {out}.gen-{attempt_id}/（目录名不以 .jsonl 结尾 → 不进
    #      {task}-*.jsonl glob——中断残留不污染 best-file 仲裁 / SKIP
    #      幂等的行数口径）；
    #   ③ 循环结束 fout.close() 后计算预测 SHA256 + 最终行数，构建
    #      status=complete 完成回执（producer-yarn-config-v2，含
    #      prediction_basename/SHA/行数/run_id）写进同一 gen 目录；
    #   ④ 单次提交点：commit_yarn_generation 单次 os.replace 原子切
    #      指针 {out}.tli_gen（内容 = gen 目录名；指针切换 = 唯一
    #      提交信号）。
    # 崩溃语义（066/crash-recovery 验收口径）：gen 未就绪/指针未切 →
    #   旧完整代可见（指针仍指旧 gen）；指针已切 → 新完整代可见；
    #   混合代不可达（预测与回执同目录，消费侧从指针同时读两文件）。
    #   注：059 时代 docstring 的「两步之间死亡 → SHA 必失配」前提在
    #   同字节场景不成立（066），已废止。
    attempt_id = (datetime.now().strftime("%Y%m%d%H%M%S") +
                  f"-{os.getpid()}-{random.randint(1000, 9999)}")
    lock_fd = acquire_output_lock(out_path)
    try:
        tmp_pred = stage_yarn_generation(out_path, attempt_id)
        fout = open(tmp_pred, "w", encoding="utf-8")
    except OSError as e:
        release_output_lock(lock_fd)
        raise SystemExit(f"[pred] 打开临时 generation 失败（{e}）")

    try:
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

        # ---- 059+066：完成回执（v2 同代绑定）→ 同 gen 目录 → 切指针 ----
        # 057 的配置指纹字段全部保留（effective factor/完整 rope_scaling/
        # 模型路径与 config hash/生成参数/生产脚本身份）；v2 新增同代绑定
        # 四件：status=complete + prediction_basename/SHA256/行数 + run_id
        # ——formal 消费侧据指针解析 generation 后同时读预测与回执，
        # 三方一致校验（062 纵深）+ 指针同源保证（066/crash-recovery）。
        pred_sha = _file_sha256(tmp_pred)
        pred_lines = sum(1 for _ in open(tmp_pred, "rb"))
        model_cfg_path = os.path.join(args.model_path, "config.json")
        # 079（TL-E121-OUTPUT-ID-079）：tli 臂把 resolved treatment
        # manifest 写入回执——与 method hash（method_name 尾段）、
        # LongBench sidecar 共用 sparse_attn/info.py 同一份 resolved
        # manifest（单一事实源），本入口不再各自维护治疗身份字段子集；
        # 非 tli 臂（none/quest/twia/tia）treatment 身份由可读名整体
        # 编码，不写。schema 校验 sha256(json) 自洽（yarn_receipt 079）。
        # 081：从冻结 snapshot 派生（与 out_path 的 _h 段、runtime 各层
        # 同一对象）——生成窗口内文件替换不再产生 receipt 混装；yarn_
        # receipt 的 081 闭包校验（tm hash[:10] == basename _h 段）由此
        # 结构性地成立。
        treatment_manifest_json = (
            get_treatment_manifest_json(args, snapshot)
            if args.method == "tli" else None)
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
            producer_script_sha256=_file_sha256(os.path.abspath(__file__)),
            run_id=attempt_id,
            prediction_basename=os.path.basename(out_path),
            prediction_sha256=pred_sha,
            prediction_lines=pred_lines,
            treatment_manifest_json=treatment_manifest_json)
        tmp_rcp = stage_yarn_receipt(out_path, yarn_receipt, attempt_id)
        commit_yarn_generation(tmp_pred, tmp_rcp, out_path)
        print(f"[yarn-receipt] effective config + 同代绑定 -> {tmp_rcp}"
              f"（generation 目录内）")
        print(f"[pred] 提交：指针 {generation_pointer_path(out_path)} -> "
              f"{os.path.basename(os.path.dirname(tmp_pred))}")
        # 「saved -> {out_path}」行保留（调度端 provenance 对账依赖该
        # 口径）；实体 = 指针所指 generation 内的预测文件
        print(f"saved -> {out_path} (via generation pointer)")
    finally:
        # 崩溃/异常路径：未指向的 gen 目录残留（目录名不以 .jsonl 结尾，
        # 不污染 {task}-*.jsonl glob），指针保持上一完整代；锁由本处
        # 显式释放或进程退出时内核自动释放（flock 不持久，无死锁）。
        release_output_lock(lock_fd)


if __name__ == "__main__":
    main()
