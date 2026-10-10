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
# ===== #196（GPT 2026-10-10 0635 二轮复审 062/063/064/065）=====
#   TL-E119-YARN-CORRECTION-SCHEMA-063（P2，本文件）：纠偏 sidecar
#       schema 太弱——只校验 dict + target SHA + correction_version 任意
#       非空 str，缺 correction 主体的两字段 sidecar 也被解释成有效纠偏。
#       修复：correction_version 限已知枚举 {yarn-identity-057-v1}
#       （未知版本 fail closed）+ original_run_identity.yarn_factor 与
#       manifest run_identity 声明逐位核对 + correction 主体语义字段
#       （not_effective/null/closed=false）严格校验（按三份既有 128K
#       sidecar 实际结构适配）。
#   TL-E119-YARN-CONFIG-PARTIAL-064②（P2，本文件）：
#       effective_config_sha256 漏收录 schema 必需键
#       generation_params.max_num（必需+类型校验但不进指纹）——
#       max_num=1 与 100 两份合法回执同指纹。修复：max_num 纳入 canon；
#       「必需但不进指纹」的全部自由字段在 docstring 逐一声明。
#       064①（model_config_sha256=None 的 config_identity=missing 降级
#       标注）在消费侧 score_ruler_formal.py 落地。
#
# ===== #198（GPT 2026-10-10 1130 审计 TL-E119-YARN-SAME-BYTES-
#   PROVENANCE-066/crash-recovery，P2）=====
#   059 的两步 os.replace 提交（预测先替换 → 回执后替换）在「B 预测与
#   A 字节完全相同」的死亡中间态下不可检：第一步完成后、第二步完成前
#   进程死亡 → 盘上为「B 物理写入的预测 + A 旧回执」，三方 SHA 全等
#   （staging == 回执声明 == 源当前字节），消费侧同代绑定校验全过，
#   formal 接受 A 的 run_id/seed/config 并标 verified_same_generation=
#   true——物理运行来源保证过强（GPT 复现：同字节异配置 run-B 崩溃后
#   仍接受 run-A/seed=42 且 verified=true；异字节则 SHA 失配 fail-closed
#   正确拒绝）。066 回应里「两步之间死亡 → SHA 必失配 → fail-closed」
#   的前提在同字节场景不成立——SHA 校验只能拒绝【内容失配】的坏态，
#   不能证明【运行同代】。
#   修复（主 AI 已定，GPT 建议 1：不可变 generation + 单指针原子切换）：
#   预测与完成回执最终写入不可变 generation 目录 {out}.gen-{attempt_id}/
#   （两文件同目录共存，059 已有 .gen- 命名前例；目录名不以 .jsonl 结尾
#   → 不污染 best-file glob），提交 = 单次 os.replace 原子切指针
#   （{out}.tli_gen，内容为 gen 目录名；E116f「指针切换 = 唯一提交信号」
#   同语义）。崩溃三阶段只剩：指针未切 → 旧完整代可见；指针已切 →
#   新完整代可见；混合代不可达（消费侧从指针解析 generation 后【同时】
#   读预测与回执——物理来源 = 指针所指 gen 目录，不再依赖内容等价推断）。
#   flock 语义保留（066 活进程负例 B3 不回归）：锁窗口内完成
#   「stage → gen 就绪 → 指针切换」。既有产物无指针文件 → 消费侧维持
#   直接读路径（generation_binding="legacy-direct"），有指针则标
#   "pointer-v1"（既有收口数据/manifest 字节零改动）。
#   TL-E119-B4-SKIP-AS-PASS-067（P2，测试侧落地）：binding 套件 main
#   的 n += 1 无条件计数把 B4 的 SKIP 计为 PASS——PASS/SKIP/FAIL 三分
#   显式计数，SKIP>0 时不打 ALL PASS（详见测试文件）。
#
# ===== #200（GPT 2026-10-10 1528 审计 TL-E119-PROBE-RECEIPT-
#   VALIDATION-070，P2）=====
#   完成探针 gen_completion_probe.py（068）对 pointer 候选只调
#   resolve_generation_pointer（本模块，明确不校验回执内容）后数预测
#   行数判 complete——「预测足量 + 回执存在但无效（半写 JSON/未知
#   schema/非 complete/basename/SHA/行数失配）」被判 complete →
#   run_ruler_e109.sh SKIP 该格，而正式汇总 score_ruler_formal.py 会
#   fail-closed 拒收：调度与交付的完成定义分裂，坏格永久 SKIP 且正式
#   评分永远拒绝。修复：抽出共享校验器 validate_committed_generation
#   （回执 bytes 快照 → JSON → validate_producer_receipt → v2 同代
#   字节绑定），probe 与 formal 共用——单一完成定义，防第三套回执
#   判断漂移。版本接受集合与 formal 现行口径逐位一致（v1+v2；v1 无
#   绑定字段 → 不做字节比对，与 formal 的降级消费同口径——指针协议
#   生产路径只产 v2，v1 属理论边界，按 formal 既有行为不在此拒收）。
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
# 063（TL-E119-YARN-CORRECTION-SCHEMA）：纠偏 sidecar 的已发布版本枚举。
# 修复前 resolve_manifest_yarn_identity 只要求 correction_version 是任意
# 非空 str——缺 correction 主体的两字段 sidecar / 未知版本 / 语义矛盾
# 字段都会被静默解释成有效纠偏（GPT 审计最小复现：correction_version=
# "unrecognized-garbage-version" 仍返回 not_effective/null）。未知/缺失
# 版本 → fail closed（未来协议须先在此登记才可被消费）。
CORRECTION_VERSION = "yarn-identity-057-v1"
CORRECTION_VERSIONS = (CORRECTION_VERSION,)
# 060：yarn_factor_source 合法枚举（与 build_yarn_receipt 三态一致）
YARN_FACTOR_SOURCES = ("explicit", "auto", "off")
# 066/crash-recovery（#198）：generation 指针协议——预测与完成回执同置
# 不可变 generation 目录，{pred}.tli_gen 指针文件（内容 = gen 目录名）
# 单次原子切换 = 唯一提交信号。指针不以 .jsonl 结尾 → 不污染
# {task}-*.jsonl best-file glob。
GENERATION_POINTER_SUFFIX = ".tli_gen"
# manifest 的 generation 绑定口径标注：pointer-v1 = 指针协议产物
# （消费侧从指针解析后同时读预测与回执）；legacy-direct = 既有产物
# （无指针文件，按最终路径直接读——059 v2 与更早协议，历史数据不撤销）
GENERATION_BINDING_POINTER = "pointer-v1"
GENERATION_BINDING_LEGACY = "legacy-direct"


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

    注意（059 + 066/crash-recovery）：生产路径 pred_ruler.py 已走
    stage_yarn_generation/stage_yarn_receipt（预测与回执同置不可变
    generation 目录）+ commit_yarn_generation（单次原子切指针）；
    本函数保留给测试/工具与 legacy-direct fixture 直写场景。"""
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
    覆盖「generation 目录写入 → SHA/行数计算 → 完成回执 → 指针切换」
    全程，到提交完成 / 进程退出为止。后到者阻塞等待；flock 属内核锁，
    持锁进程崩溃即自动释放，不留死锁。锁是咨询锁：仅约束同样走
    pred_ruler.py 生成路径的进程；066/crash-recovery 后消费侧从指针
    解析 generation 后同时读预测与回执（混合代不可达），锁额外保证
    formal 冻结窗口与生产者提交互斥（B3 活进程负例），legacy-direct
    产物仍由锁 + 同代绑定校验双层兜底。"""
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


def generation_dir_for(pred_out_path, attempt_id):
    """066/crash-recovery（#198）：不可变 generation 目录名。

    {out}.gen-{attempt_id}/——预测与完成回执同目录共存；目录名不以
    .jsonl 结尾 → 不进任何 {task}-*.jsonl glob（059 命名前例保留）。"""
    return f"{pred_out_path}.gen-{attempt_id}"


def generation_pointer_path(pred_out_path):
    """066/crash-recovery（#198）：generation 指针文件路径。

    {out}.tli_gen（内容 = gen 目录名）。指针不以 .jsonl 结尾 → 不污染
    best-file glob；消费侧以指针解析当前 generation（物理来源 = 指针
    所指目录，预测与回执同源同代）。"""
    return pred_out_path + GENERATION_POINTER_SUFFIX


def stage_yarn_generation(pred_out_path, attempt_id):
    """059+066/crash-recovery：预测写入不可变 generation 目录。

    返回 gen 目录内的预测文件路径（生成循环直接写这里）：
    {out}.gen-{attempt_id}/{basename}。修复前（059 两步提交）临时
    预测是 pred_dir 下的散文件、提交点第一步把它 os.replace 到最终
    路径——两步之间死亡且 B 与 A 字节完全相同时留下「B 物理写入的
    预测 + A 旧回执」（三方 SHA 全等，混合代不可检，066）。指针协议
    下预测不再落最终路径，与完成回执同置 gen 目录，提交 =
    commit_yarn_generation 单次原子切指针。目录名不以 .jsonl 结尾
    → 生成中断留下的 partial gen 目录不污染 best-file 仲裁 / SKIP
    幂等的行数口径。"""
    gdir = generation_dir_for(pred_out_path, attempt_id)
    os.makedirs(gdir, exist_ok=True)
    return os.path.join(gdir, os.path.basename(pred_out_path))


def stage_yarn_receipt(pred_out_path, receipt, attempt_id):
    """059+066/crash-recovery：完成回执写入同一不可变 generation 目录。

    仅在生成循环结束、预测文件关闭并算出 SHA256/行数后调用（v2 回执
    的绑定字段由此闭合）。回执与预测同目录共存 → 指针切换后消费侧
    从同一 gen 目录同时读到两文件（同代同源）。目录内临时文件 + fsync
    + os.replace 落位；指针切换由 commit_yarn_generation 统一执行。
    返回 gen 目录内的回执路径。"""
    gdir = generation_dir_for(pred_out_path, attempt_id)
    os.makedirs(gdir, exist_ok=True)
    path = os.path.join(
        gdir, os.path.basename(producer_receipt_path_for(pred_out_path)))
    tmp = f"{path}.tmp-{attempt_id}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=1, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def commit_yarn_generation(tmp_pred, tmp_receipt, pred_out_path):
    """066/crash-recovery（#198）：单次原子提交点 = 切指针。

    修复前（059）顺序为「os.replace 临时预测 → 最终路径，再
    os.replace 完成回执 → 旁挂路径」，docstring 声称「两步之间死亡
    → 新预测配旧回执（SHA 必失配）→ 消费侧 fail-closed」——该前提
    在 B 与 A 字节完全相同时【不成立】（066 审计复现：同字节异配置
    崩溃中间态被 verified_same_generation=true 接受），SHA 校验只能
    拒绝内容失配的坏态、不能证明运行同代。

    指针协议：预测与回执已在不可变 gen 目录内就绪（tmp_pred /
    tmp_receipt 须同目录），提交 = 写指针临时文件 + fsync + 单次
    os.replace 原子切换 {out}.tli_gen（E116f「指针切换 = 唯一提交
    信号」同语义）。崩溃三阶段：
      - gen 未就绪 / 指针未切 → 旧完整代可见（指针仍指旧 gen 目录）；
      - 指针已切 → 新完整代可见；
      - 混合代不可达（预测与回执同目录，消费侧从指针同时读两文件）。
    提交后的旧 gen 目录不清理（指针已不再引用；崩溃语义需要旧完整
    代可见，与 059「临时残留不污染 glob」同纪律——.gen 目录名不以
    .jsonl 结尾）。返回指针路径。"""
    gdir = os.path.dirname(tmp_pred)
    if os.path.dirname(tmp_receipt) != gdir:
        raise ValueError(
            f"commit_yarn_generation: 预测与回执不在同一 generation 目录"
            f"（{tmp_pred!r} vs {tmp_receipt!r}）——指针协议要求两文件同"
            f"目录共存，否则指针切换后无法保证同代同源（066/crash-"
            f"recovery）")
    if not os.path.isfile(tmp_pred) or not os.path.isfile(tmp_receipt):
        raise ValueError(
            f"commit_yarn_generation: generation {gdir} 缺预测或回执"
            f"（isfile(pred)={os.path.isfile(tmp_pred)}, "
            f"isfile(receipt)={os.path.isfile(tmp_receipt)}）——不得提交"
            f"半成品 generation（066/crash-recovery）")
    pointer_path = generation_pointer_path(pred_out_path)
    tmp_pointer = f"{pointer_path}.tmp-{os.getpid()}"
    with open(tmp_pointer, "w", encoding="utf-8") as f:
        f.write(os.path.basename(gdir) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_pointer, pointer_path)
    return pointer_path


def resolve_generation_pointer(pred_out_path):
    """066/crash-recovery（#198）消费侧：指针 → generation 解析入口。

    {out}.tli_gen 存在 → 读其内容为 gen 目录名，校验 gen 目录及其内
    预测/回执两文件齐备，返回 dict：
        {"binding": "pointer-v1",
         "pointer_path": ..., "gen_dir": <gen 目录绝对路径>,
         "pred_path": <gen 内预测>, "rcp_path": <gen 内回执>}
    指针不存在 → 返回 None（legacy-direct：调用方按既有最终路径直接
    读，既有产物零改动）。
    指针存在但内容为空/指向不存在的 gen 目录/缺预测或缺回执 →
    SystemExit fail-closed（存在即证据：不静默回退 legacy-direct——
    回退会把指针协议的死亡中间态静默解释成完整代；python -O 不失效）。
    本函数不校验回执内容（schema/同代绑定由消费侧
    score_ruler_formal._load_producer_yarn_receipt 承载）。"""
    pointer_path = generation_pointer_path(pred_out_path)
    if not os.path.isfile(pointer_path):
        return None
    try:
        with open(pointer_path, "r", encoding="utf-8") as f:
            gen_name = f.read().strip()
    except OSError as e:
        raise SystemExit(
            f"[GATE-FAIL] {pointer_path}: generation 指针读取失败（{e}）"
            f"——存在即证据，fail closed（066/crash-recovery）")
    if not gen_name or os.path.basename(gen_name) != gen_name or \
            gen_name in (".", "..") or "/" in gen_name:
        raise SystemExit(
            f"[GATE-FAIL] {pointer_path}: 指针内容 {gen_name!r} 非合法"
            f" generation 目录名——fail closed（066/crash-recovery）")
    gen_dir = os.path.join(os.path.dirname(os.path.abspath(pred_out_path)),
                           gen_name)
    pred_path = os.path.join(gen_dir, os.path.basename(pred_out_path))
    rcp_path = os.path.join(gen_dir, os.path.basename(
        producer_receipt_path_for(pred_out_path)))
    if not os.path.isdir(gen_dir) or not os.path.isfile(pred_path):
        raise SystemExit(
            f"[GATE-FAIL] {pointer_path}: 指向的 generation {gen_name} 目录"
            f"或其中预测缺失（{pred_path}）——指针存在即证据，不得静默"
            f"回退直接读路径，fail closed（066/crash-recovery）")
    if not os.path.isfile(rcp_path):
        raise SystemExit(
            f"[GATE-FAIL] {pointer_path}: 指向的 generation {gen_name} 缺"
            f"完成回执（{rcp_path}）——gen 未就绪的死亡中间态，指针未切换"
            f"才对（存在即证据），fail closed（066/crash-recovery）")
    return {
        "binding": GENERATION_BINDING_POINTER,
        "pointer_path": os.path.abspath(pointer_path),
        "gen_dir": os.path.abspath(gen_dir),
        "pred_path": os.path.abspath(pred_path),
        "rcp_path": os.path.abspath(rcp_path),
    }


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


def _count_lines(path):
    """行数计数（与 wc -l / score_ruler._nlines / 探针同语义：逐行迭代）。"""
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def validate_committed_generation(pred_path, rcp_path):
    """070（TL-E119-PROBE-RECEIPT-VALIDATION，#200）：共享提交代回执
    校验器——probe（完成探针）与 formal（正式汇总）单口径，防第三套
    回执判断漂移。

    参数：
      pred_path：同代绑定比对对象（回执 prediction_basename/SHA256/行数
        所指的预测文件）。probe 传 generation 内预测本体（指针解析的
        物理同源文件）；formal 传 staging 评分副本（062② 语义：同代
        绑定以评分对象比对——副本与源 basename 相同，schema 的
        basename/task 前缀校验两种传法逐位一致）。
      rcp_path：回执文件路径（单一 bytes 快照源）。

    校验链（与 formal _load_producer_yarn_receipt 修复前行为同口径）：
      ① 回执单一 bytes 快照（062①：解析与 receipt SHA 同源，防「读-
         算之间被推进」的混合态回执摘要）；
      ② JSON 解析失败 → SystemExit（存在即证据，半写 fail-closed）；
      ③ validate_producer_receipt（协议版本集 v1+v2 + 060 严格 schema
         + 059 v2 绑定格式层：status=complete / basename / SHA 格式 /
         行数格式）；
      ④ v2 同代字节绑定：实际预测 SHA256/行数 与回执声明逐位比对
         （v1 无绑定字段 → 跳过，与 formal 的降级消费同口径）。
    任一失败 raise SystemExit（[GATE-FAIL] 前缀，python -O 不失效）。

    返回 dict：{"receipt": 解析后回执, "rcp_sha256": bytes 快照 SHA,
    "binding_sha256": 实际预测 SHA（v2；v1 为 None）,
    "binding_lines": 实际预测行数（v2；v1 为 None）}。
    本函数保持零重依赖、纯只读（不写任何文件）。"""
    # ---- 062①：回执单一 bytes 快照（解析与 SHA 同源）----
    try:
        with open(rcp_path, "rb") as f:
            rcp_raw = f.read()
    except OSError as e:
        raise SystemExit(
            f"[GATE-FAIL] {rcp_path}: 生产者 yarn receipt 读取失败（{e}）"
            f"——存在即证据，半写/损坏 fail closed（057+070）")
    try:
        rcp = json.loads(rcp_raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise SystemExit(
            f"[GATE-FAIL] {rcp_path}: 生产者 yarn receipt 解析失败（{e}）"
            f"——存在即证据，半写/损坏 fail closed（057+070）")
    rcp_sha = hashlib.sha256(rcp_raw).hexdigest()
    err = validate_producer_receipt(rcp, pred_path)
    if err:
        raise SystemExit(f"[GATE-FAIL] {err}")
    binding_sha = None
    binding_lines = None
    if rcp["receipt_version"] == RECEIPT_VERSION:
        # ---- 070 核心新增：v2 同代字节绑定（回执声明 vs 实际预测）----
        # 修复前只有 formal 做此比对，probe 只数行数 → 「预测足量 + 回执
        # 声明错代」被调度判 complete/SKIP，正式评分却拒收（完成定义分裂）
        binding_sha = _file_sha256(pred_path)
        binding_lines = _count_lines(pred_path)
        if binding_sha != rcp["prediction_sha256"]:
            raise SystemExit(
                f"[GATE-FAIL] {rcp_path}: 回执声明 prediction_sha256="
                f"{rcp['prediction_sha256']} 与实际预测 "
                f"{os.path.basename(pred_path)} 字节 SHA256={binding_sha} "
                f"不一致——预测与回执不同代（同名重跑/中断重跑/并发/"
                f"事后改写，或生产者在复制与验证之间提交了新一代），"
                f"fail closed（059+062+070）")
        if binding_lines != rcp["prediction_lines"]:
            raise SystemExit(
                f"[GATE-FAIL] {rcp_path}: 回执声明 prediction_lines="
                f"{rcp['prediction_lines']} 与实际预测 "
                f"{os.path.basename(pred_path)} 行数 {binding_lines} "
                f"不一致——预测与回执不同代，fail closed（059+070）")
    return {"receipt": rcp, "rcp_sha256": rcp_sha,
            "binding_sha256": binding_sha, "binding_lines": binding_lines}


def effective_config_sha256(receipt):
    """060：配置指纹规范化摘要（跨格一致性门禁的比较锚）。

    只收录同一 context_length 档内应逐位一致的字段：模型路径/模型
    config hash/完整 rope_scaling/原生 MPE/yarn 开关与 effective
    factor/source/seed/max_gen/max_num/生产脚本（路径+SHA）。
    同配置必同 hash（canonical json + sort_keys，无环境漂移）。

    064②（TL-E119-YARN-CONFIG-PARTIAL）：schema 必需键与指纹收录集的
    差异逐一声明如下——「必需但不进指纹」的全部自由字段：
      - receipt.task / generation_params.method / t / pred_postfix：
        逐格变化的派单/命名自由字段（formal 的显式分组规则，S3 正例
        锁定该语义；跨格漂移由样本 ID 集合门禁承载）；
      - produced_at：生产时间戳（逐回执自然不同，非配置语义）；
      - receipt_version / audit_ref：协议层常量（版本混装已被 059 门禁
        拒绝、audit_ref 由 schema 校验约束非空，均无逐格区分意义）。
    064② 修复：generation_params.max_num 此前「schema 必需 + 类型校验
    但不进指纹」——GPT 复现 max_num=1 与 100 两份合法回执同指纹
    a09b4ef6...，属「必需但不比较」漏洞；现纳入 canon（负例
    max_num=1 vs 100 必须不同指纹）。"""
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
        # 064②：max_num 是 schema 必需键（060 校验非负 int），必须进
        # 指纹——否则「必需但不比较」形成配置闭包缺口
        "max_num": gp.get("max_num"),
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
        再按 063 严格版本化 schema 校验纠偏主体（correction_version ∈
        CORRECTION_VERSIONS、original_run_identity 与 manifest 实际声明
        一致、correction 语义字段 not_effective/null/closed=false 齐备
        自洽——未知版本/缺主体/身份张冠李戴/语义矛盾全部 fail closed），
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
    # ---- 063（TL-E119-YARN-CORRECTION-SCHEMA）：严格版本化 schema ----
    # 修复前任意非空 correction_version 都被接受，且缺 correction 主体、
    # original 身份与 manifest 不符、语义字段矛盾均零校验——损坏/错误
    # 生成/未来未知协议的 sidecar 会被静默解释成有效纠偏。以下逐项
    # fail closed（按三份既有 128K sidecar 实际结构适配字段名）：
    if cver not in CORRECTION_VERSIONS:
        raise SystemExit(
            f"[GATE-FAIL] {correction_path}: correction_version={cver!r} "
            f"不在已发布纠偏协议枚举 {CORRECTION_VERSIONS}——未知版本"
            f" fail closed（未来协议须先在 yarn_receipt.CORRECTION_VERSIONS"
            f" 登记才可被消费，063）")
    # ① 纠偏声明的历史身份必须与 manifest run_identity 实际声明一致
    #    （纠偏描述的是「这份 manifest」，张冠李戴的 sidecar 拒收）
    ori = correction.get("original_run_identity")
    if not isinstance(ori, dict) or "yarn_factor" not in ori:
        raise SystemExit(
            f"[GATE-FAIL] {correction_path}: 缺 original_run_identity."
            f"yarn_factor（纠偏必须声明其纠正的历史身份，与 manifest "
            f"run_identity 实际声明核对）——fail closed（063）")
    if ori["yarn_factor"] != declared:
        raise SystemExit(
            f"[GATE-FAIL] {correction_path}: original_run_identity."
            f"yarn_factor={ori['yarn_factor']!r} 与 manifest "
            f"run_identity.yarn_factor={declared!r} 不一致——纠偏声明的"
            f"历史与实际 manifest 不符（sidecar 指向的不是这份 manifest"
            f"的身份历史），fail closed（063）")
    if "yarn" in ori and ori["yarn"] != bool(ri.get("yarn")):
        raise SystemExit(
            f"[GATE-FAIL] {correction_path}: original_run_identity."
            f"yarn={ori['yarn']!r} 与 manifest run_identity.yarn="
            f"{bool(ri.get('yarn'))!r} 不一致——fail closed（063）")
    # ② 纠偏主体的语义字段必须明确表达 not-effective/null/未闭合
    #    （字段名按三份既有 128K sidecar 实际结构：yarn_factor_status /
    #    effective_yarn_factor / effective_factor_closed）
    body = correction.get("correction")
    if not isinstance(body, dict):
        raise SystemExit(
            f"[GATE-FAIL] {correction_path}: 缺 correction 主体（纠偏语义"
            f"字段 yarn_factor_status/effective_yarn_factor/"
            f"effective_factor_closed 必须齐备且语义自洽）——"
            f"fail closed（063）")
    if body.get("yarn_factor_status") != "operator_declared_not_effective":
        raise SystemExit(
            f"[GATE-FAIL] {correction_path}: correction.yarn_factor_status="
            f"{body.get('yarn_factor_status')!r} != "
            f"'operator_declared_not_effective'——{cver} 协议的纠偏语义"
            f"必须明确声明旧 factor 不再是 effective，fail closed（063）")
    if body.get("effective_yarn_factor") is not None:
        raise SystemExit(
            f"[GATE-FAIL] {correction_path}: correction."
            f"effective_yarn_factor={body.get('effective_yarn_factor')!r}"
            f" 非 null——{cver} 协议不脑补实际生效值（佐证≠同代哈希"
            f"绑定，与 051 同纪律），fail closed（063）")
    if body.get("effective_factor_closed") is not False:
        raise SystemExit(
            f"[GATE-FAIL] {correction_path}: correction."
            f"effective_factor_closed="
            f"{body.get('effective_factor_closed')!r} 非 False——"
            f"factor 身份未闭合正是纠偏存在的前提，声明已闭合则自相"
            f"矛盾，fail closed（063）")
    if "operator_declared_yarn_factor" in body and \
            body["operator_declared_yarn_factor"] != declared:
        raise SystemExit(
            f"[GATE-FAIL] {correction_path}: correction."
            f"operator_declared_yarn_factor="
            f"{body['operator_declared_yarn_factor']!r} 与 manifest 声明"
            f"{declared!r} 不一致——fail closed（063）")
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
