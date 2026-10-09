# 生产者原生 effective YaRN 配置 receipt（GPT 2026-10-10 0330 审计
# TL-E119-YARN-IDENTITY-057，任务链 #194）
#
# 背景（057 违反事实）：pred_ruler.py 的 --yarn_factor 默认 None，未显式
# 传入时按档位自动表（65536→2.0、131072→4.0）解析 effective factor；而
# formal 汇总入口（score_ruler_formal.py）此前把操作者事后 CLI 声明原样
# 写进 run_identity。E119 128K 批次派单只传 --yarn 未传显式 factor，生成
# 进程实际按 4.0 自动档路径解析，三份 128K manifest 却声明 2.0——声明与
# 代码链推导冲突，effective factor 未闭合。
#
# 修复协议（producer-yarn-config-v1）：
#   - 生成侧（pred_ruler.py）：解析 effective 配置（自动档/显式覆盖）之后，
#     把 effective factor、完整 rope_scaling、上下文档位、模型路径、
#     config.json SHA256、生成参数原子写入预测产物旁挂 receipt
#     （{pred 基名}-yarn_receipt.json）——不改既有 jsonl 行格式，下游
#     wc -l / best-file 仲裁 / SKIP 幂等零扰动（.json 后缀不进任何
#     {task}-*.jsonl glob）；
#   - 消费侧（score_ruler_formal.py）：优先消费 best-file 旁挂 receipt 的
#     effective 值写 run_identity；操作者 CLI 声明与生产者证据冲突 →
#     fail-closed（_fail 走 SystemExit，python -O 不失效）；legacy 产物
#     无 receipt → CLI 值如实标注 operator_declared，不冒充实际生效值；
#     既有 128K 三臂 manifest 字节不动，旁挂版本化 correction JSON
#     （.manifest.yarn_correction.json，不脑补 actual=4.0——佐证≠同代
#     哈希绑定，与 051 同纪律）。
#
# 本模块刻意零重依赖（不 import torch/transformers/sparse_attn）——
# 生成侧、消费侧与 CPU 红绿测试三方共享同一解析/校验口径，干净检出
# 恒可单测。
import hashlib
import json
import os
from datetime import datetime

RECEIPT_VERSION = "producer-yarn-config-v1"
AUDIT_REF = "TL-E119-YARN-IDENTITY-057"
RECEIPT_SUFFIX = "-yarn_receipt.json"


def resolve_yarn_config(use_yarn, yarn_factor, context_length, auto_map,
                        native_mpe):
    """effective YaRN 配置解析（纯函数，057 抽出供单测）。

    语义与修复前 pred_ruler.py 两处内联逻辑逐位一致：
        factor = yarn_factor or auto_map.get(context_length, 4.0)
    （显式 0/None 同样回退自动档——保持既有 `or` 语义零行为变更）。
    返回 (factor, rope_scaling)；use_yarn=False → (None, None)。"""
    if not use_yarn:
        return None, None
    factor = yarn_factor or auto_map.get(context_length, 4.0)
    rope_scaling = {
        "rope_type": "yarn", "type": "yarn",
        "factor": factor,
        "original_max_position_embeddings": native_mpe,
        "beta_fast": 32, "beta_slow": 1,
    }
    return factor, rope_scaling


def _file_sha256(path):
    """文件 SHA256（receipt 生产/消费共用同一实现口径）。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_yarn_receipt(*, yarn_enabled, effective_factor, yarn_factor_cli,
                       rope_scaling, context_length, task, model_path,
                       model_config_sha256, native_mpe, generation_params,
                       producer_script_path, producer_script_sha256):
    """构造 receipt dict（写入前内存成型，字段集稳定可校验）。

    effective_factor = 实际写进 model config 的值（自动档解析或显式覆盖
    之后）；yarn_factor_cli = 操作者原始 CLI 输入（None=未传，走自动档）。
    yarn_factor_source 三态：explicit（CLI 显式）/ auto（档位自动表）/
    off（未启用 YaRN，effective 为 None）。"""
    return {
        "receipt_version": RECEIPT_VERSION,
        "audit_ref": AUDIT_REF,
        "produced_at": datetime.now().isoformat(),
        "yarn_enabled": bool(yarn_enabled),
        "effective_yarn_factor": effective_factor,
        "yarn_factor_source": ("explicit" if yarn_factor_cli
                               else ("auto" if yarn_enabled else "off")),
        "rope_scaling": rope_scaling,
        "context_length": context_length,
        "native_max_position_embeddings": native_mpe,
        "task": task,
        "model_path": model_path,
        "model_config_sha256": model_config_sha256,
        "generation_params": generation_params,
        "producer_script": {"path": producer_script_path,
                            "sha256": producer_script_sha256},
    }


def write_yarn_receipt(pred_out_path, receipt):
    """原子写入旁挂 receipt（{pred 基名}-yarn_receipt.json）。

    临时文件 + fsync + os.replace：进程中断不留半写 receipt（半写比缺失
    更危险——消费侧把「存在」当证据，半写/损坏须 fail-closed 拒收）。
    返回 receipt 路径。"""
    path = pred_out_path[: -len(".jsonl")] + RECEIPT_SUFFIX
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=1, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def producer_receipt_path_for(pred_path):
    """best-file → 旁挂 receipt 约定路径（消费侧据此探查）。"""
    return pred_path[: -len(".jsonl")] + RECEIPT_SUFFIX


def validate_producer_receipt(receipt, pred_path):
    """消费侧 receipt 自洽校验（057：生产者证据本身不得自相矛盾）。

    返回错误消息字符串（非法时），合法返回 None。校验项：
      - receipt_version 必须是本协议（存在即证据，半写/旧格式拒收）；
      - receipt.task 与产物文件名前缀一致；
      - context_length 为正整数（与所在 L 档位的一致性由调用方比对）；
      - yarn_enabled=True → effective_yarn_factor 必须为正数且与
        rope_scaling.factor 逐位一致（057 核心不变量）；
      - yarn_enabled=False → effective 必须为 None。"""
    if not isinstance(receipt, dict):
        return f"{pred_path}: receipt 不是 JSON 对象"
    if receipt.get("receipt_version") != RECEIPT_VERSION:
        return (f"{pred_path}: receipt_version="
                f"{receipt.get('receipt_version')!r} != "
                f"{RECEIPT_VERSION!r}（存在即证据，非本协议拒收）")
    task = receipt.get("task")
    base = os.path.basename(pred_path)
    if not isinstance(task, str) or not base.startswith(task + "-"):
        return (f"{pred_path}: receipt.task={task!r} 与产物文件名前缀"
                f"不一致——fail closed")
    cl = receipt.get("context_length")
    if not isinstance(cl, int) or isinstance(cl, bool) or cl <= 0:
        return f"{pred_path}: receipt.context_length={cl!r} 非法"
    eff = receipt.get("effective_yarn_factor")
    enabled = receipt.get("yarn_enabled")
    if not isinstance(enabled, bool):
        return f"{pred_path}: receipt.yarn_enabled={enabled!r} 非 bool"
    if enabled:
        if not isinstance(eff, (int, float)) or isinstance(eff, bool) \
                or eff <= 0:
            return (f"{pred_path}: yarn_enabled=True 但 "
                    f"effective_yarn_factor={eff!r} 非正数——fail closed")
        rs = receipt.get("rope_scaling") or {}
        if rs.get("factor") != eff:
            return (f"{pred_path}: rope_scaling.factor={rs.get('factor')!r} "
                    f"与 effective_yarn_factor={eff!r} 不一致"
                    f"（receipt 自相矛盾——057 核心不变量）——fail closed")
    else:
        if eff is not None:
            return (f"{pred_path}: yarn_enabled=False 但 "
                    f"effective_yarn_factor={eff!r} 非 None——fail closed")
    return None
