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
# ===== #195（GPT 2026-10-10 0428 审计三项修复）=====
# TL-E119-YARN-RECEIPT-BINDING-059（P1）：v1 回执在生成循环之前发布、
#   不含预测内容 SHA/行数/run ID/完成标记——「A 进程的预测配 B 进程的
#   回执」可达（同名重跑/中断重跑/并发/事后改写）。协议升级
#   producer-yarn-config-v2：预测先写不可变临时 generation，循环结束
#   关闭后计算预测 SHA256 + 最终行数，构建 status=complete 完成回执
#   （含 prediction_basename/SHA/行数/run_id），经单次原子提交点
#   （os.replace 临时预测 → 最终路径，再 os.replace 完成回执 → 旁挂
#   路径——回执最后落盘 = 提交信号，与 E116f 同语义）；全生命周期持
#   规范化输出路径 flock（056 口径 + 042 锁键加固：realpath(父目录) +
#   词法 basename）。消费侧校验回执与 best-file 逐位同代，v1 回执降级
#   标注 producer_receipt_v1_partial（只证 factor 口径不证同代）。
# TL-E119-YARN-RECEIPT-CLOSURE-060（P2）：validate_producer_receipt 只校验
#   开关与 factor——beta_fast/seed/模型 config hash/生产脚本 hash 写入
#   零校验，改 999/假 hash 均通过。升级为严格 schema（必需键/类型/
#   64 位十六进制/rope_scaling 完整规范/source 枚举/generation_params
#   必需键），并计算 effective_config_sha256（配置指纹规范化摘要），
#   formal 对同档（context_length 分组）内应逐位一致的字段做跨格一致
#   性门禁；task/method/t/pred_postfix 为分组自由字段不进指纹。
# TL-E119-YARN-CORRECTION-DISCOVERY-061（P2）：三份 128K
#   .manifest.yarn_correction.json 此前零机器消费者，旧 manifest 仍公开
#   yarn_factor=2.0 无 provenance。新增 resolve_manifest_yarn_identity
#   解析入口：解析正式 manifest 时探查同路径旁挂纠偏，存在且
#   target_manifest_sha256 与当前字节一致 → 旧 yarn_factor 不得再当
#   effective，返回 effective_yarn_factor=null +
#   provenance=operator_declared_not_effective + 纠偏绑定三重哈希；
#   不一致 → fail-closed（不脑补 2.0 也不脑补 4.0）。
#
# 本模块刻意零重依赖（不 import torch/transformers/sparse_attn）——
# 生成侧、消费侧与 CPU 红绿测试三方共享同一解析/校验口径，干净检出
# 恒可单测。
import fcntl
import hashlib
import json
import os
from datetime import datetime

# 059：v2 = 当前协议（同代绑定）；v1 = 057 时代协议（消费侧降级标注，
# 只证 factor 口径不证回执与预测同代——历史产物不撤销、不冒充）
RECEIPT_VERSION = "producer-yarn-config-v2"
RECEIPT_V1_VERSION = "producer-yarn-config-v1"
RECEIPT_VERSIONS = (RECEIPT_V1_VERSION, RECEIPT_VERSION)
AUDIT_REF = "TL-E119-YARN-IDENTITY-057"
RECEIPT_SUFFIX = "-yarn_receipt.json"
# 061：正式 manifest 的旁挂纠偏文件命名（与 #194 已落盘的三份 128K
# sidecar 命名逐位一致——manifest 路径直接追加后缀）
CORRECTION_SUFFIX = ".yarn_correction.json"
# 060：yarn_factor_source 合法枚举（与 build_yarn_receipt 三态一致）
YARN_FACTOR_SOURCES = ("explicit", "auto", "off")


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


def _is_hex64(v):
    """64 位十六进制 SHA256 格式检查（060：hash 字段的显式格式门禁）。"""
    return (isinstance(v, str) and len(v) == 64 and
            all(c in "0123456789abcdef" for c in v))


def _is_pos_int(v):
    """正整数检查（排除 bool——bool 是 int 子类，True 会被误判为 1）。"""
    return isinstance(v, int) and not isinstance(v, bool) and v > 0


def build_yarn_receipt(*, yarn_enabled, effective_factor, yarn_factor_cli,
                       rope_scaling, context_length, task, model_path,
                       model_config_sha256, native_mpe, generation_params,
                       producer_script_path, producer_script_sha256,
                       receipt_version=RECEIPT_VERSION, run_id=None,
                       prediction_basename=None, prediction_sha256=None,
                       prediction_lines=None):
    """构造 receipt dict（写入前内存成型，字段集稳定可校验）。

    effective_factor = 实际写进 model config 的值（自动档解析或显式覆盖
    之后）；yarn_factor_cli = 操作者原始 CLI 输入（None=未传，走自动档）。
    yarn_factor_source 三态：explicit（CLI 显式）/ auto（档位自动表）/
    off（未启用 YaRN，effective 为 None）。

    059：receipt_version=producer-yarn-config-v2（默认）要求同代绑定四件
    齐备（run_id/prediction_basename/prediction_sha256/prediction_lines）
    ——生成侧必须在预测循环结束、文件关闭并算出 SHA/行数之后才能构建
    v2 回执；缺任一件属生产代码接线错误，当场 ValueError（fail loudly，
    不产出无绑定字段的 v2 回执）。v1（RECEIPT_V1_VERSION）为历史协议，
    仅供既有产物兼容消费，不写绑定字段。"""
    if receipt_version not in RECEIPT_VERSIONS:
        raise ValueError(f"未知 receipt_version={receipt_version!r}"
                         f"（合法值 {RECEIPT_VERSIONS}）")
    receipt = {
        "receipt_version": receipt_version,
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
    if receipt_version == RECEIPT_VERSION:
        missing = [k for k, v in (
            ("run_id", run_id), ("prediction_basename", prediction_basename),
            ("prediction_sha256", prediction_sha256),
            ("prediction_lines", prediction_lines)) if v is None]
        if missing:
            raise ValueError(
                f"producer-yarn-config-v2 回执缺同代绑定字段 {missing}——"
                f"v2 必须在生成循环结束、预测关闭并算出 SHA/行数后构建"
                f"（059），不得产出无绑定回执")
        receipt["status"] = "complete"
        receipt["run_id"] = run_id
        receipt["prediction_basename"] = prediction_basename
        receipt["prediction_sha256"] = prediction_sha256
        receipt["prediction_lines"] = prediction_lines
    return receipt


def write_yarn_receipt(pred_out_path, receipt):
    """原子写入旁挂 receipt（{pred 基名}-yarn_receipt.json）。

    临时文件 + fsync + os.replace：进程中断不留半写 receipt（半写比缺失
    更危险——消费侧把「存在」当证据，半写/损坏须 fail-closed 拒收）。
    返回 receipt 路径。

    注意（059）：生产路径 pred_ruler.py 已改走 stage_yarn_receipt +
    commit_yarn_generation 两段式（完成回执不落最终路径直到提交点）；
    本函数保留给测试/工具直写场景。"""
    path = pred_out_path[: -len(".jsonl")] + RECEIPT_SUFFIX
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=1, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


# ---- 059：generation staging + 单次原子提交 + 输出路径锁 ----

def producer_receipt_path_for(pred_path):
    """best-file → 旁挂 receipt 约定路径（消费侧据此探查）。"""
    return pred_path[: -len(".jsonl")] + RECEIPT_SUFFIX


def attempt_lock_path(pred_out_path):
    """059：输出路径跨进程锁键（056 口径 + 042 加固）。

    realpath(父目录) + 词法 basename：最终分量（预测文件名）会被提交点
    的 os.replace 改写，realpath(整个路径) 在「目标初始不存在/symlink」
    与「发布后普通文件」两种状态间漂移（042 的锁键漂移根因）——只解析
    父目录，键不随提交漂移；symlink 父目录别名归一到同一把锁。
    锁文件名 .attempt.lock 与临时产物名（.gen-/.tmp-）均不冲突。"""
    return os.path.join(
        os.path.realpath(os.path.dirname(os.path.abspath(pred_out_path))
                         or "."),
        os.path.basename(pred_out_path)) + ".attempt.lock"


def acquire_output_lock(pred_out_path):
    """059：以（规范化后的）输出路径为粒度的跨进程互斥
    （fcntl.flock，LOCK_EX 阻塞等待）。

    生命周期持锁（056 同口径）：调用点在创建临时 generation 之前，锁
    覆盖「临时 generation 写入 → SHA/行数计算 → 完成回执 → 单次提交」
    全程，到提交完成 / 进程退出为止。后到者阻塞等待；flock 属内核锁，
    持锁进程崩溃即自动释放，不留死锁。锁是咨询锁：仅约束同样走
    pred_ruler.py 生成路径的进程；即便两个进程交错（如旧版本进程），
    消费侧同代绑定校验也会把「A 预测配 B 回执」的混合代际 fail-closed
    拒收（锁保证良态一致，绑定校验兜底检测坏态）。"""
    path = attempt_lock_path(pred_out_path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


def release_output_lock(fd):
    """059：显式释放（正常路径提交完成后调用；异常/崩溃路径由进程退出
    或内核自动释放兜底——flock 不持久）。"""
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def stage_yarn_generation(pred_out_path, attempt_id):
    """059：预测的不可变临时 generation 路径（{out}.gen-{attempt_id}）。

    临时名不以 .jsonl 结尾 → 不进任何 {task}-*.jsonl glob——生成中断
    留下的 partial 临时文件不污染 best-file 仲裁 / SKIP 幂等的行数
    口径，最终路径保持上一代完整产物。"""
    return f"{pred_out_path}.gen-{attempt_id}"


def stage_yarn_receipt(pred_out_path, receipt, attempt_id):
    """059：完成回执写入临时旁挂（不落最终路径）。

    仅在生成循环结束、预测文件关闭并算出 SHA256/行数后调用（v2 回执
    的绑定字段由此闭合）。临时文件 + fsync；提交点由
    commit_yarn_generation 统一执行。返回临时 receipt 路径。"""
    path = producer_receipt_path_for(pred_out_path)
    tmp = f"{path}.tmp-{attempt_id}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=1, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    return tmp


def commit_yarn_generation(tmp_pred, tmp_receipt, pred_out_path):
    """059：单次原子提交点（generation → 最终路径 + 完成回执）。

    顺序：先 os.replace 临时预测 → 最终路径，再 os.replace 临时完成
    回执 → 旁挂路径——回执最后落盘 = 提交信号（与 E116f「receipt 最后
    落盘 = 唯一提交信号」同语义）。两步之间死亡 → 最终路径为新预测配
    旧回执（SHA 必失配）→ 消费侧同代绑定校验 fail-closed（坏态可检，
    不静默）。提交前死亡 → 最终路径保持上一代完整产物（旧预测配旧
    回执，同代自洽），临时文件残留无 glob 污染。"""
    os.replace(tmp_pred, pred_out_path)
    os.replace(tmp_receipt, producer_receipt_path_for(pred_out_path))


# ---- 060：严格 schema 校验 + 配置指纹 ----

def _validate_common_schema(receipt, pred_path):
    """060：v1/v2 共用的严格 schema 校验（返回错误消息或 None）。

    必需键集合、类型、64 位十六进制 hash、rope_scaling 完整规范
    （rope_type/type=yarn、factor>0、original_max_position_embeddings
    正 int 且等于 native_mpe、beta_fast=32、beta_slow=1——与
    resolve_yarn_config 产出的 Qwen3 官方配方逐位一致）、
    yarn_factor_source 枚举且与开关自洽、generation_params 必需键
    （seed 正 int、max_gen 正 int、max_num 非负 int、method/t/
    pred_postfix 非空 str）、producer_script 路径+SHA 格式。
    允许额外键（前向兼容），但必需键缺一/类型错 → 拒收。"""
    base = os.path.basename(pred_path)
    required = ("receipt_version", "audit_ref", "produced_at",
                "yarn_enabled", "effective_yarn_factor",
                "yarn_factor_source", "rope_scaling", "context_length",
                "native_max_position_embeddings", "task", "model_path",
                "model_config_sha256", "generation_params",
                "producer_script")
    missing = [k for k in required if k not in receipt]
    if missing:
        return (f"{pred_path}: receipt 缺必需键 {missing}——schema 不"
                f"完整，fail closed（060）")
    # 标量类型与格式
    if not isinstance(receipt["audit_ref"], str) or \
            not receipt["audit_ref"]:
        return f"{pred_path}: receipt.audit_ref 非非空 str——fail closed（060）"
    if not isinstance(receipt["produced_at"], str) or \
            not receipt["produced_at"]:
        return f"{pred_path}: receipt.produced_at 非非空 str——fail closed（060）"
    task = receipt["task"]
    if not isinstance(task, str) or not base.startswith(task + "-"):
        return (f"{pred_path}: receipt.task={task!r} 与产物文件名前缀"
                f"不一致——fail closed")
    if not _is_pos_int(receipt["context_length"]):
        return (f"{pred_path}: receipt.context_length="
                f"{receipt['context_length']!r} 非正整数——fail closed（060）")
    if not _is_pos_int(receipt["native_max_position_embeddings"]):
        return (f"{pred_path}: native_max_position_embeddings="
                f"{receipt['native_max_position_embeddings']!r} 非正整数"
                f"——fail closed（060）")
    if not isinstance(receipt["model_path"], str) or \
            not receipt["model_path"]:
        return (f"{pred_path}: receipt.model_path 非非空 str——"
                f"fail closed（060）")
    mcs = receipt["model_config_sha256"]
    if mcs is not None and not _is_hex64(mcs):
        return (f"{pred_path}: model_config_sha256={mcs!r} 非 None 且非 64"
                f"位十六进制——fail closed（060）")
    ps = receipt["producer_script"]
    if not isinstance(ps, dict) or not isinstance(ps.get("path"), str) or \
            not ps.get("path") or not _is_hex64(ps.get("sha256")):
        return (f"{pred_path}: producer_script={ps!r} 非法（须为 "
                f"{{path: 非空 str, sha256: 64 位十六进制}}）——"
                f"fail closed（060）")
    # yarn_factor_source 枚举 + 与开关自洽
    src = receipt["yarn_factor_source"]
    if src not in YARN_FACTOR_SOURCES:
        return (f"{pred_path}: yarn_factor_source={src!r} 不在合法枚举 "
                f"{YARN_FACTOR_SOURCES}——fail closed（060）")
    enabled = receipt["yarn_enabled"]
    if not isinstance(enabled, bool):
        return f"{pred_path}: receipt.yarn_enabled={enabled!r} 非 bool"
    if enabled and src == "off":
        return (f"{pred_path}: yarn_enabled=True 但 source='off'——自相"
                f"矛盾，fail closed（060）")
    if not enabled and src != "off":
        return (f"{pred_path}: yarn_enabled=False 但 source={src!r}——"
                f"自相矛盾，fail closed（060）")
    # generation_params 必需键（seed 正 int / max_gen 正 int / max_num
    # 非负 int / method、t、pred_postfix 非空 str）
    gp = receipt["generation_params"]
    if not isinstance(gp, dict):
        return f"{pred_path}: generation_params 非对象——fail closed（060）"
    gp_req = {
        "seed": _is_pos_int, "max_gen": _is_pos_int,
        "max_num": lambda v: isinstance(v, int) and not isinstance(v, bool)
                             and v >= 0,
        "method": lambda v: isinstance(v, str) and bool(v),
        "pred_postfix": lambda v: isinstance(v, str) and bool(v),
        "t": lambda v: isinstance(v, str) and bool(v),
    }
    for k, chk in gp_req.items():
        if k not in gp:
            return (f"{pred_path}: generation_params 缺必需键 {k!r}——"
                    f"fail closed（060）")
        if not chk(gp[k]):
            return (f"{pred_path}: generation_params.{k}={gp[k]!r} 类型"
                    f"非法——fail closed（060）")
    # 057 核心不变量 + 060 rope_scaling 完整规范
    eff = receipt["effective_yarn_factor"]
    if enabled:
        if not isinstance(eff, (int, float)) or isinstance(eff, bool) \
                or eff <= 0:
            return (f"{pred_path}: yarn_enabled=True 但 "
                    f"effective_yarn_factor={eff!r} 非正数——fail closed")
        rs = receipt["rope_scaling"]
        if not isinstance(rs, dict):
            return (f"{pred_path}: rope_scaling 非对象——fail closed（060）")
        if rs.get("rope_type") != "yarn" or rs.get("type") != "yarn":
            return (f"{pred_path}: rope_scaling.rope_type/type="
                    f"{rs.get('rope_type')!r}/{rs.get('type')!r} 非 yarn"
                    f"——fail closed（060）")
        if rs.get("factor") != eff:
            return (f"{pred_path}: rope_scaling.factor={rs.get('factor')!r} "
                    f"与 effective_yarn_factor={eff!r} 不一致"
                    f"（receipt 自相矛盾——057 核心不变量）——fail closed")
        mpe = rs.get("original_max_position_embeddings")
        if not _is_pos_int(mpe) or \
                mpe != receipt["native_max_position_embeddings"]:
            return (f"{pred_path}: rope_scaling.original_max_position_"
                    f"embeddings={mpe!r} 非正整数或与 native_mpe="
                    f"{receipt['native_max_position_embeddings']!r} 不一致"
                    f"——fail closed（060）")
        # beta_fast=32 / beta_slow=1：Qwen3 官方 YaRN 配方（与
        # resolve_yarn_config 逐位一致）——改 999 等假值在此拒收
        if rs.get("beta_fast") != 32 or rs.get("beta_slow") != 1:
            return (f"{pred_path}: rope_scaling.beta_fast/beta_slow="
                    f"{rs.get('beta_fast')!r}/{rs.get('beta_slow')!r} != "
                    f"32/1（Qwen3 官方配方）——fail closed（060）")
    else:
        if eff is not None:
            return (f"{pred_path}: yarn_enabled=False 但 "
                    f"effective_yarn_factor={eff!r} 非 None——fail closed")
        if receipt["rope_scaling"] is not None:
            return (f"{pred_path}: yarn_enabled=False 但 rope_scaling 非 "
                    f"None——fail closed（060）")
    return None


def validate_producer_receipt(receipt, pred_path):
    """消费侧 receipt 自洽校验（057 + 059/060 严格化）。

    返回错误消息字符串（非法时），合法返回 None。校验分两层：
      - 协议层（057，保留）：receipt_version 必须是已发布协议之一
        （存在即证据，半写/未知格式拒收）；
      - schema 层（060，新增）：必需键集合、类型、64 位十六进制 hash、
        rope_scaling 完整规范（beta_fast=32/beta_slow=1）、source 枚举、
        generation_params 必需键——此前 beta_fast/seed/模型 hash/脚本
        hash 写入零校验，改 999/假 hash 均通过；
      - 绑定格式层（059，v2）：status=complete、run_id/prediction_basename
        非空、prediction_sha256 64 位十六进制、prediction_lines 正整数、
        basename 与产物文件名一致。
    注意：prediction_sha256/prediction_lines 与预测文件实际字节的比对
    需要读盘，属消费侧调用方职责（score_ruler_formal.
    _load_producer_yarn_receipt 现算比对）——本函数保持纯内存校验。"""
    if not isinstance(receipt, dict):
        return f"{pred_path}: receipt 不是 JSON 对象"
    ver = receipt.get("receipt_version")
    if ver not in RECEIPT_VERSIONS:
        return (f"{pred_path}: receipt_version={ver!r} 不在已发布协议 "
                f"{RECEIPT_VERSIONS}（存在即证据，非本协议拒收）")
    err = _validate_common_schema(receipt, pred_path)
    if err:
        return err
    if ver == RECEIPT_VERSION:
        # 059：v2 同代绑定格式校验（与预测字节的一致性由调用方比对）
        if receipt.get("status") != "complete":
            return (f"{pred_path}: receipt.status={receipt.get('status')!r} "
                    f"!= 'complete'——中断/未完成代际的回执，"
                    f"fail closed（059）")
        if not isinstance(receipt.get("run_id"), str) or \
                not receipt.get("run_id"):
            return (f"{pred_path}: receipt.run_id 非非空 str——"
                    f"fail closed（059）")
        pb = receipt.get("prediction_basename")
        if not isinstance(pb, str) or not pb or \
                pb != os.path.basename(pred_path):
            return (f"{pred_path}: prediction_basename={pb!r} 与产物文件"
                    f"名 {os.path.basename(pred_path)!r} 不一致——"
                    f"fail closed（059）")
        if not _is_hex64(receipt.get("prediction_sha256")):
            return (f"{pred_path}: prediction_sha256 非合法 64 位十六进制"
                    f"——fail closed（059）")
        if not _is_pos_int(receipt.get("prediction_lines")):
            return (f"{pred_path}: prediction_lines="
                    f"{receipt.get('prediction_lines')!r} 非正整数——"
                    f"fail closed（059）")
    return None


def effective_config_sha256(receipt):
    """060：配置指纹规范化摘要（跨格一致性门禁的比较锚）。

    只收录同一 context_length 档内应逐位一致的字段：模型路径/模型
    config hash/完整 rope_scaling/原生 MPE/yarn 开关与 effective
    factor/source/seed/max_gen/生产脚本（路径+SHA）。task/method/t/
    pred_postfix 允许逐格变化，不进指纹（formal 的显式分组规则）。
    同配置必同 hash（canonical json + sort_keys，无环境漂移）。"""
    gp = receipt.get("generation_params") or {}
    ps = receipt.get("producer_script") or {}
    canon = {
        "yarn_enabled": receipt.get("yarn_enabled"),
        "effective_yarn_factor": receipt.get("effective_yarn_factor"),
        "yarn_factor_source": receipt.get("yarn_factor_source"),
        "rope_scaling": receipt.get("rope_scaling"),
        "context_length": receipt.get("context_length"),
        "native_max_position_embeddings":
            receipt.get("native_max_position_embeddings"),
        "model_path": receipt.get("model_path"),
        "model_config_sha256": receipt.get("model_config_sha256"),
        "seed": gp.get("seed"),
        "max_gen": gp.get("max_gen"),
        "producer_script_path": ps.get("path"),
        "producer_script_sha256": ps.get("sha256"),
    }
    return hashlib.sha256(json.dumps(
        canon, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


# ---- 061：manifest 旁挂纠偏的机器消费解析入口 ----

def manifest_correction_path_for(manifest_path):
    """正式 manifest → 旁挂纠偏约定路径（与 #194 已落盘的三份 128K
    sidecar 命名逐位一致：manifest 去掉尾部 .json 后接
    .yarn_correction.json，即 xxx.json.manifest.json 的纠偏是
    xxx.json.manifest.yarn_correction.json——插在 .manifest 之后而非
    整名之后，追加式命名会与既有三份文件失配）。"""
    if manifest_path.endswith(".json"):
        return manifest_path[:-len(".json")] + CORRECTION_SUFFIX
    return manifest_path + CORRECTION_SUFFIX


def resolve_manifest_yarn_identity(manifest_path):
    """061：正式 manifest 的 effective yarn 身份解析入口（唯一消费口径）。

    解析 manifest 时探查同路径旁挂纠偏：
      - 存在 → 先 fail-closed 校验 correction.target_manifest_sha256 与
        manifest 当前字节 SHA 一致（纠偏指向的必须是这份 manifest；
        manifest 合法重发布后旧纠偏失配 → loudly 拒绝，须重新落纠偏），
        然后该 manifest 公开的 yarn_factor 不得再当 effective：
        返回 effective_yarn_factor=None +
        yarn_factor_provenance="operator_declared_not_effective" +
        纠偏绑定（target_manifest_sha256 / correction 文件自身 sha256 /
        correction_version）。不脑补 actual=4.0（佐证≠同代哈希绑定）。
      - 不存在 → manifest 原值透传（provenance 沿用 manifest 自带的
        yarn_factor_provenance；缺失标 manifest_declared_no_provenance
        ——E116e 之前发布的 legacy manifest 即此形态）。

    返回 dict（yarn/effective_yarn_factor/yarn_factor_provenance/
    operator_declared_yarn_factor/correction/manifest_run_identity_
    yarn_factor）；manifest 缺失/解析失败 → SystemExit fail-closed
    （python -O 不失效）。"""
    try:
        with open(manifest_path, "rb") as f:
            raw = f.read()
    except OSError as e:
        raise SystemExit(f"[GATE-FAIL] {manifest_path}: 读取失败（{e}）——"
                         f"yarn 身份解析 fail closed（061）")
    import json as _json
    try:
        manifest = _json.loads(raw)
    except (ValueError, UnicodeDecodeError) as e:
        raise SystemExit(f"[GATE-FAIL] {manifest_path}: JSON 解析失败"
                         f"（{e}）——yarn 身份解析 fail closed（061）")
    if not isinstance(manifest, dict) or \
            not isinstance(manifest.get("run_identity"), dict):
        raise SystemExit(f"[GATE-FAIL] {manifest_path}: 缺 run_identity "
                         f"对象——yarn 身份解析 fail closed（061）")
    ri = manifest["run_identity"]
    declared = ri.get("yarn_factor")
    correction_path = manifest_correction_path_for(manifest_path)
    if not os.path.isfile(correction_path):
        return {
            "yarn": bool(ri.get("yarn")),
            "effective_yarn_factor": declared,
            "yarn_factor_provenance": ri.get(
                "yarn_factor_provenance", "manifest_declared_no_provenance"),
            "operator_declared_yarn_factor": declared,
            "correction": None,
            "manifest_run_identity_yarn_factor": declared,
        }
    try:
        with open(correction_path, "rb") as f:
            craw = f.read()
        correction = _json.loads(craw)
    except (OSError, ValueError, UnicodeDecodeError) as e:
        raise SystemExit(f"[GATE-FAIL] {correction_path}: 纠偏文件读取/解析"
                         f"失败（{e}）——存在即证据，fail closed（061）")
    if not isinstance(correction, dict):
        raise SystemExit(f"[GATE-FAIL] {correction_path}: 纠偏非 JSON 对象"
                         f"——fail closed（061）")
    actual_manifest_sha = hashlib.sha256(raw).hexdigest()
    target_sha = correction.get("target_manifest_sha256")
    if not _is_hex64(target_sha) or target_sha != actual_manifest_sha:
        raise SystemExit(
            f"[GATE-FAIL] {correction_path}: target_manifest_sha256="
            f"{target_sha!r} 与 manifest 当前字节 SHA="
            f"{actual_manifest_sha} 不一致——纠偏指向的不是这份 manifest"
            f"（manifest 已被重新发布？须按 051 纪律重落版本化纠偏），"
            f"fail closed（061）")
    cver = correction.get("correction_version")
    if not isinstance(cver, str) or not cver:
        raise SystemExit(f"[GATE-FAIL] {correction_path}: correction_version"
                         f" 非非空 str——fail closed（061）")
    return {
        # 旧 manifest 公开的 yarn_factor 从此不再被当作 effective
        "yarn": bool(ri.get("yarn")),
        "effective_yarn_factor": None,
        "yarn_factor_provenance": "operator_declared_not_effective",
        "operator_declared_yarn_factor": declared,
        "correction": {
            "correction_version": cver,
            "correction_path": os.path.abspath(correction_path),
            "correction_sha256": hashlib.sha256(craw).hexdigest(),
            "target_manifest_path": os.path.abspath(manifest_path),
            "target_manifest_sha256": target_sha,
            "operator_declared_not_effective": True,
        },
        "manifest_run_identity_yarn_factor": declared,
    }
