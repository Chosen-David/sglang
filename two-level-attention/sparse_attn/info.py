import dataclasses
import hashlib
import io
import json
import os
import re

import torch

# ---- 076（TL-E121-OUTPUT-ID，GPT 2026-10-10 2131 审计）：canonical
# treatment manifest——输出身份必须是 treatment 的单射。
# B09 只把 α/β/γ 加进可读名，仍遗漏多个改变算法行为的参数（GPT CPU
# 复现 6 配置仅 2 名/3 组碰撞：far/near method、far/near select、
# subspace 具体值、投影基、per_q_head、sim/σ/MoBA 等），不同
# treatment 同名 → LongBench 同路径 "w" 模式静默互覆、RULER 同逻辑
# 指针被后跑代切换。修复：全部输出相关参数排序键 JSON 序列化 →
# sha256 前 10 hex 作 treatment hash，追加 "_h<hash10>" 尾段；可读段
# 逐位不动（B09 口径），既有测试锚点只补 hash 段。
# 字段缺省用 getattr 兜底，与 TLIIndexer.__init__ 的运行时缺省逐字段
# 同源（sparse_attn/indexer/tli_indexer.py __init__ getattr 序列）——
# 缺省 namespace 与显式默认值 namespace 必须同 hash（同配置稳定）。
# 【081 + kimi3 0316 追加修复 2】tli_enable_kmeans/tli_enable_layer_skip
# 此前刻意不进 manifest（076 residual 声明「可读段 B/D 二值编码已单射」
# ——被 kimi3 2026-10-11 0316 hourly review 反例推翻）：极端长可读前缀
# 触发 truncate_output_name_keep_hash 截断时，B/D 段被可读中段的 "..."
# 占位吃掉 → 仅在这两个开关上不同的两 treatment 截断后文件名相同 +
# manifest 相同 + hash 相同（kimi3 实测 trunc same? True、hash same?
# True）→ 写门放行互覆。两字段入 manifest 后 hash 恒单射，截断只影响
# 可读性不影响身份。默认 True 与 argparse 缺省 / TLIIndexer.__init__
# 运行时缺省三方同源。
#
# ---- 079（TL-E121-OUTPUT-ID-079，GPT 2026-10-11 0332 复审）两个残余 ----
# ① tia_enable_async_topk 进 manifest：TLI 继承 TIA
#   （sparse_attn/indexer/tia_indexer.py:18 self.enable_async =
#   args.tia_enable_async_topk；tli_indexer.py compute_mask async 分支
#   缓存 prev_mask 并在下一步替换）——异步第 2 个 decode 步起的细筛
#   候选集与同步路径可不同，是输出相关参数而非性能提示。getattr 缺省
#   False 与 argparse store_true 缺省/TIAIndexer 运行时读取三方同源。
# ② tli_proj_basis/tli_layer_skip_path 由「词法路径」升级为「内容身份」：
#   只记路径时同路径不同内容 → 同 manifest/hash → LongBench sidecar
#   写门放行覆盖、RULER 复用同逻辑指针（GPT CPU 复现实锤）。现于
#   manifest 构造时解析为 {path, sha256, shape/n_skip}（值非 None 时），
#   文件缺失/读取失败/解析失败一律 [GATE-FAIL] 079 SystemExit fail
#   closed——解析发生在 get_method_name_with_info 内，即输出路径确定
#   之前；不再用 default=str 隐去身份。LongBench sidecar（pred.py 写
#   门）与 method hash 共用本函数产出的同一份 resolved manifest，
#   两个入口零字段子集各自维护。
#
# ---- 081（TL-E121-OUTPUT-SNAPSHOT-081，GPT 2026-10-11 复审 P1）----
# 079 只共用了解析「函数」，没有冻结一次解析后的「值」——同一文件在
# 一次运行内被多次打开：TLIIndexer 逐层各 open 一次（patch.py 对每个
# attention 层建实例），benchmark 入口在生成后重新调 get_method_name_
# with_info / get_treatment_manifest_json 再读一遍 → 生成窗口内文件被
# 替换时 runtime A / 文件名 B / sidecar C 三方混装可达。配套两个身份缺口：
#   ① 默认 D′ 掩码遗漏：arguments.py 缺省 tli_enable_layer_skip=True、
#     tli_layer_skip_path=None，运行时（tli_indexer）展开 tracked
#     DEFAULT_MASK 并读取；但 079 manifest 只在路径显式非 None 时解析
#     → 默认配置 manifest 恒 null 而可读名含 D（身份缺口）；
#   ② 反向不对称：显式给 tli_layer_skip_path 但 enable_layer_skip=False
#     时，manifest 强制解析文件求 hash，而运行时根本不读它——身份按
#     「argv 声明」而非「实际消费」解析。
# 修复：
#   A. resolve_treatment_snapshot(args) 一次解析 → 不可变 TreatmentSnapshot
#      （manifest / manifest_json / skip_ids / basis_tensor），每文件只
#      open 一次、SHA 与内容解析共用同一批 bytes（不再「open 算 SHA 再
#      torch.load 读第二次」）；
#   B. D′ 掩码按有效值语义解析：只在 effective tli_enable_layer_skip=True
#      时进入（缺省 True + path=None → 展开 DEFAULT_MASK 并解析其内容；
#      enable=False 时无论路径是否声明都不解析、manifest 记 null）——
#      同时修复 ①②；
#   C. benchmark 两个入口在模型加载（文件被任何 attention 层消费）之前
#      冻结 snapshot，文件名 / sidecar / receipt 全程消费同一对象；
#      TLIIndexer(args, snapshot=...) 注入已解析对象，不再按层重开文件。
# 默认掩码路径定义上移到本模块（单一事实源；tli_indexer.DEFAULT_MASK
# 现引用本常量）——manifest 侧与运行时侧展开的必须是同一个文件。
DEFAULT_LAYER_SKIP_MASK_PATH = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "exp", "trace",
    "results", "tli_layer_skip_mask.json"))
_TREATMENT_FIELD_DEFAULTS = {
    # 既有可读名字段（hash 同样纳入——可读段整体单射的最强保证）
    "tia_block_size": 64,
    "tia_level1_topk": 128,
    "tia_level2_topk": 1024,
    "tia_level2_cmp_ratio": 2,      # argparse 缺省（可读段直接属性访问，兜底不可达）
    "tli_enable_subspace": True,
    "tli_subspace": "full",
    "tli_alpha": 0.0,
    "tli_beta": 0.0,
    "tli_gamma": 1.0,               # γ 兜底 1.0（B09 既有约定保持；off=None 走 JSON null）
    "tli_sparse_prefill": False,
    # 081 + kimi3 0316 追加修复 2：B/D 开关入 manifest（见文件头注释——
    # 截断可吃掉可读段 B/D 位，身份必须由 manifest/hash 承载）
    "tli_enable_kmeans": True,
    "tli_enable_layer_skip": True,
    # 076 遗漏字段：改变算法行为但不在可读段
    "tli_far_method": "minmax",
    "tli_near_method": "avg",
    "tli_far_select": "4bit",
    "tli_near_select": "4bit",
    "tli_far_clusters": 256,
    "tli_far_niter": 10,
    "tli_far_blocks": 16,
    "tli_far_tokens": 512,
    "tli_sim": 0.9,
    "tli_sim_dims": "subspace",
    "tli_sigma_select": "none",
    "tli_moba": False,
    "tli_sigma": 8.0,
    "tli_per_q_head": False,
    # 079①：TIA 继承的异步开关改变第 2 步起的细筛候选集，必须进身份
    "tia_enable_async_topk": False,
    # 079②：文件型配置——值非 None 时经 _resolve_file_identity 解析为
    # 内容身份 {path, sha256, shape/n_skip}（fail closed），不再只记路径
    "tli_proj_basis": None,
    "tli_static_pair": False,
    "tli_layer_skip_path": None,
}

# 079②：文件型配置字段集合（值非 None → 内容身份解析，见上）
_FILE_IDENTITY_FIELDS = ("tli_proj_basis", "tli_layer_skip_path")

# treatment hash 尾段格式：_h + 10 hex。锚定尾正则拆分（可读段以
# g 段/_P 结尾、非 tli 名以数值/async/none 结尾，不与该模式误撞）。
_TREATMENT_HASH_HEX_LEN = 10
_TREATMENT_HASH_RE = re.compile(
    r"^(?P<readable>.+)_(?P<hash>h[0-9a-f]{%d})$" % _TREATMENT_HASH_HEX_LEN,
    re.DOTALL)


def _resolve_skip_content(path, is_default=False):
    """081：D′ 掩码文件的「单次 open」内容解析（fail closed）。

    同一批 bytes 同时产出内容身份与解析好的 skip 列表——不再「open 算
    SHA、再 open 解析 JSON」双开（079 的 _resolve_file_identity 即此
    形态，081 起文件只被读一次，快照冻结才有单一事实源）。is_default=
    True 时错误文案附注该文件为缺省 D′ 掩码展开（tli_layer_skip_path
    未声明时的 DEFAULT_MASK）。

    081 身份口径：is_default=True 时内容身份只记 {default, sha256,
    n_skip}，刻意不含 checkout 相关的 realpath——默认掩码是 repo
    tracked 常量（非 argv 声明），其 provenance 就是「仓库这份内容」，
    记 realpath 会让同一份内容在不同 checkout 下得到不同 hash（WIP
    版实测缺陷：同内容同 SHA 在 worktree/主仓两 checkout 漂移为
    h7c17d0b763 / h3bbe917587，锚点随 checkout 必红）。显式声明的
    path（argv provenance）保留 realpath——路径本身是声明的一部分，
    与 079② 口径一致。"""
    if not isinstance(path, str) or not path:
        raise SystemExit(
            f"[GATE-FAIL] 079: tli_layer_skip_path 路径非法（{path!r}）"
            f"——文件型配置必须绑定实际内容，拒绝运行，fail loudly")
    rp = os.path.realpath(path)
    hint = ("（081：此文件为缺省 D′ 掩码 DEFAULT_MASK 的展开——"
            "tli_layer_skip_path 未声明时运行时同样读它）" if is_default
            else "")
    try:
        with open(rp, "rb") as f:
            raw = f.read()
    except OSError as e:
        raise SystemExit(
            f"[GATE-FAIL] 079: tli_layer_skip_path 文件读取失败（{path} → "
            f"{type(e).__name__}: {e}）——文件型配置必须绑定实际内容，"
            f"拒绝运行，fail loudly{hint}")
    if is_default:
        # 081：默认掩码身份只记内容闭包（default 标记 + sha256 +
        # n_skip），不含 checkout 相关 realpath——见 docstring。
        ident = {"default": True, "sha256": hashlib.sha256(raw).hexdigest()}
    else:
        ident = {"path": rp, "sha256": hashlib.sha256(raw).hexdigest()}
    try:
        doc = json.loads(raw.decode("utf-8"))
        ident["n_skip"] = len(doc["skip"])
        skip_list = list(doc["skip"])
    except (UnicodeDecodeError, ValueError, KeyError, TypeError) as e:
        raise SystemExit(
            f"[GATE-FAIL] 079: tli_layer_skip_path 解析失败（{path} → "
            f"{type(e).__name__}: {e}）——须为含 \"skip\" 键的 JSON；"
            f"旧口径静默关 D' 属静默退化，生产入口拒绝运行{hint}")
    return ident, skip_list


def _resolve_basis_content(path):
    """081：投影基文件的「单次 open」内容解析（fail closed）。

    同一批 bytes 同时产出内容身份 {path, sha256, shape} 与加载好的
    CPU tensor（torch.load(io.BytesIO(raw)) 等价于 torch.load(路径)
    的既有加载口径——weights_only、map_location=cpu）——SHA 与 tensor
    同源，杜绝「算 SHA 的字节 ≠ 加载进显存的字节」的混装窗口。"""
    if not isinstance(path, str) or not path:
        raise SystemExit(
            f"[GATE-FAIL] 079: tli_proj_basis 路径非法（{path!r}）——"
            f"文件型配置必须绑定实际内容，拒绝运行，fail loudly")
    rp = os.path.realpath(path)
    try:
        with open(rp, "rb") as f:
            raw = f.read()
    except OSError as e:
        raise SystemExit(
            f"[GATE-FAIL] 079: tli_proj_basis 文件读取失败（{path} → "
            f"{type(e).__name__}: {e}）——文件型配置必须绑定实际内容，"
            f"拒绝运行，fail loudly")
    ident = {"path": rp, "sha256": hashlib.sha256(raw).hexdigest()}
    try:
        t = torch.load(io.BytesIO(raw), map_location="cpu",
                       weights_only=True)
        ident["shape"] = [int(x) for x in t.shape]
    except SystemExit:
        raise
    except Exception as e:   # 加载失败/非裸 tensor——运行时同样会失败
        raise SystemExit(
            f"[GATE-FAIL] 079: tli_proj_basis 加载/取 shape 失败"
            f"（{path} → {type(e).__name__}: {e}）——生产路径要求裸"
            f"tensor [n_layers,Hkv,128,r]，TLIIndexer 稍后同样会失败，"
            f"拒绝运行，fail loudly")
    return ident, t


def _resolve_file_identity(field, path):
    """079②：文件型配置的内容身份解析（fail closed；081 兼容包装）。

    返回 {"path": 规范路径, "sha256": 文件字节 sha256, 摘要}：
      - tli_proj_basis：附加 shape（torch.load weights_only 提取——
        与 TLIIndexer.__init__ 同一加载口径；加载失败意味着运行时
        同样会失败，提前 fail closed 不留「能算名不能跑」的半配置）；
      - tli_layer_skip_path：附加 n_skip（解析 JSON 并要求含 "skip"
        键——TLIIndexer.__init__ 对坏文件静默关 D' 的旧语义属「静默
        退化」家族，生产入口不再放行）。

    文件缺失/读取失败/解析失败 → [GATE-FAIL] 079 SystemExit。
    刻意不做 mtime/size 键的解析缓存：同 size 同 mtime_ns 的陈旧缓存
    会让「同路径、内容已变」拿到旧身份（079 正是要修的缺陷形态）；
    身份宁可每次重读（两文件均为 MB 级以内），不允许任何陈旧身份。
    081：内部已改为单次 open 助手（SHA 与内容解析共用同一批 bytes）；
    D′ 掩码的有效值语义（enable=False 不解析、path=None 展开
    DEFAULT_MASK）见 resolve_treatment_snapshot——本包装保持 079 的
    「声明即解析」口径仅供兼容，生产路径不再经此入口。"""
    if field == "tli_proj_basis":
        return _resolve_basis_content(path)[0]
    return _resolve_skip_content(path)[0]


# ================================================================ 081：冻结快照

@dataclasses.dataclass(frozen=True)
class TreatmentSnapshot:
    """081：一次解析后的冻结 treatment 快照（不可变结构）。

    生产入口（benchmark pred.py / pred_ruler.py）在模型加载前调用
    resolve_treatment_snapshot(args) 取得唯一实例，随后：
      - register_patch → 每个 attention 层的 TLIIndexer 注入同一对象
        （层实例仍逐层建，但共享同一 snapshot——不再按层重开文件）；
      - method_name（文件名 _h<hash10> 段）、LongBench sidecar 写门、
        RULER receipt 的 treatment_manifest_json 全部从本对象派生。
    生成窗口内文件被替换时，runtime / 文件名 / sidecar / receipt 仍
    全部等于冻结时刻的内容身份（081 主缺陷的修复点）。

    字段（frozen——属性不可再赋值；manifest 内容约定只读）：
      manifest      resolved manifest dict（与逐字段重新解析逐位一致，
                    含默认 D′ 掩码的 {path, sha256, n_skip} 身份）
      manifest_json canonical JSON 串（sort_keys；hash 输入即本串）
      skip_ids      D′ 生效时解析好的 skip 集合（frozenset）；
                    enable_layer_skip=False → None（不消费掩码）
      basis_tensor  投影基 CPU float tensor；未声明 → None
    """
    manifest: dict
    manifest_json: str
    skip_ids: object     # frozenset[int] | None
    basis_tensor: object  # torch.Tensor | None


def resolve_treatment_snapshot(args):
    """081：把全部文件型 treatment 配置一次解析并冻结为 TreatmentSnapshot。

    有效值语义（D′ 掩码，081 修复身份缺口 ①②）：
      - effective tli_enable_layer_skip=True（缺省 True）且
        tli_layer_skip_path=None → 展开 DEFAULT_LAYER_SKIP_MASK_PATH
        并解析其内容（默认配置的 manifest 从此含掩码内容身份，
        而非恒 null 而可读名含 D）；
      - tli_enable_layer_skip=False → 无论路径是否声明都不解析、
        manifest 记 null（身份按「实际消费」而非「argv 声明」——
        修复 079 的反向不对称）。
    投影基：声明即解析（与 TLIIndexer 运行时同口径），同一批 bytes
    同时求 SHA256 / shape / tensor。文件缺失/损坏一律 [GATE-FAIL]
    079 SystemExit fail closed（沿用 079 错误口径）——不允许静默关 D′
    或「能算名不能跑」的半配置。

    保证：get_method_name_with_info / get_treatment_manifest_json /
    gate_output_treatment_identity（经 manifest_json）三者共用同一
    snapshot 时，输出与逐字段重新解析逐位一致（manifest 构造只有本
    函数一条路径，无第二套字段子集逻辑）。"""
    manifest = {}
    skip_ids = None
    basis_tensor = None
    # ---- D′：有效值语义 ----
    enable_skip = bool(getattr(args, "tli_enable_layer_skip", True))
    if enable_skip:
        mask_path = getattr(args, "tli_layer_skip_path", None)
        is_default = mask_path is None
        if is_default:
            mask_path = DEFAULT_LAYER_SKIP_MASK_PATH
        ident, skip_list = _resolve_skip_content(mask_path,
                                                 is_default=is_default)
        manifest["tli_layer_skip_path"] = ident
        skip_ids = frozenset(skip_list)
    else:
        # 081②：enable=False 时路径声明不进身份（运行时不消费）
        manifest["tli_layer_skip_path"] = None
    # ---- 投影基：声明即解析（单次 open，身份与 tensor 同 bytes）----
    bp = getattr(args, "tli_proj_basis", None)
    if bp is not None:
        ident, basis_tensor = _resolve_basis_content(bp)
        manifest["tli_proj_basis"] = ident
        basis_tensor = basis_tensor.float()
    else:
        manifest["tli_proj_basis"] = None
    # ---- 其余标量字段（getattr 兜底缺省，与 TLIIndexer 运行时同源）----
    for f, d in _TREATMENT_FIELD_DEFAULTS.items():
        if f in _FILE_IDENTITY_FIELDS:
            continue
        manifest[f] = getattr(args, f, d)
    manifest_json = json.dumps(manifest, sort_keys=True,
                              ensure_ascii=False, default=str)
    return TreatmentSnapshot(manifest=manifest, manifest_json=manifest_json,
                             skip_ids=skip_ids, basis_tensor=basis_tensor)


def _treatment_manifest(args, snapshot=None):
    """076：canonical treatment manifest——全部输出相关参数的
    规范化字典（getattr 兜底缺省与 TLIIndexer 运行时同源）。

    079②：文件型配置字段（_FILE_IDENTITY_FIELDS）解析为内容身份
    （realpath + 字节 sha256 + shape/n_skip，fail closed）。
    081：D′ 掩码按有效值语义（见 resolve_treatment_snapshot）；
    snapshot 提供时直接消费冻结值（不重开文件），未提供时逐字段
    重新解析（非 benchmark 入口的既有行为，确定性不依赖缓存）。"""
    if snapshot is not None:
        return dict(snapshot.manifest)
    return resolve_treatment_snapshot(args).manifest


def get_treatment_manifest_json(args, snapshot=None):
    """076：canonical treatment manifest 的规范化 JSON 串。

    与 _treatment_hash 同源同字节（hash 输入 = 本串 utf-8 编码）——
    sidecar 记录与 hash 计算共用单一序列化口径，sidecar 内容可独立
    复核 hash（审计可再算）。排序键 + default=str 兜不可序列化值。
    079：文件型配置已解析为内容身份（realpath/sha256/shape/n_skip），
    同配置多次调用逐位稳定（确定性不依赖缓存）；LongBench 写门与
    RULER receipt 均消费本函数，不各自维护字段子集。
    081：snapshot 提供时返回冻结的 manifest_json（与逐字段重新解析
    逐位一致——单一构造路径保证）；生产入口必须传 snapshot，生成
    窗口内的文件替换不再影响本函数输出。"""
    if snapshot is not None:
        return snapshot.manifest_json
    return json.dumps(_treatment_manifest(args), sort_keys=True,
                      ensure_ascii=False, default=str)


def _treatment_hash(args, snapshot=None):
    """076：manifest 排序键 JSON 序列化 → sha256 前 10 hex。
    同一 args 对象/同配置多次调用稳定；default=str 兜不可序列化值。
    081：snapshot 提供时从冻结 manifest_json 求 hash（与文件名/
    sidecar/receipt 同源）。"""
    payload = get_treatment_manifest_json(args, snapshot)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[
        :_TREATMENT_HASH_HEX_LEN]


# 076 fail-closed 写入门的 sidecar 后缀：{out_path}.tli_manifest.json。
# 刻意不以 .jsonl 结尾（不进 LongBench/RULER 任何 *.jsonl glob）且与
# .tli_gen 指针后缀互不混淆。
TREATMENT_MANIFEST_SIDECAR_SUFFIX = ".tli_manifest.json"


def gate_output_treatment_identity(out_path, manifest_json):
    """076（TL-E121-OUTPUT-ID）fail-closed 写入门：目标路径已存在时，
    必须以 sidecar manifest 证明其与本次 treatment 同一（逐字节一致）
    才允许重写——「文件名同 hash」不再单独作为身份证明（10-hex 碰撞/
    手工改名/异配置同名一律字节级复核）。同配置重跑幂等通过
    （SKIP 语义：同 manifest → 允许重写，内容同配置等价）。

    三类拒绝（fail loudly 拒绝运行，不覆盖、不切指针）：
      ① 目标存在但缺 sidecar（provenance 未知：旧协议残留/手工放置）；
      ② sidecar 读取失败（I/O）；
      ③ sidecar manifest ≠ 本次 manifest（hash 碰撞或异配置同路径）。
    目标不存在 → 直接放行（返回 sidecar 路径）。

    调用方约定：先写 sidecar 再写数据——中断在数据写之前只留孤儿
    sidecar，下次运行目标不存在 → 门自然放行，无死锁态；目标已存在
    且 manifest 相同时重写 sidecar 为同字节幂等操作。无锁：LongBench
    生成侧单写者纪律，并发同路径写 declare residual（RULER 侧同路径
    互斥由 056/059 flock 承载，指针协议见 yarn_receipt.py）。
    """
    sidecar = out_path + TREATMENT_MANIFEST_SIDECAR_SUFFIX
    if os.path.lexists(out_path):
        if not os.path.isfile(sidecar):
            raise SystemExit(
                f"[GATE-FAIL] 076: 输出目标 {out_path} 已存在但缺 "
                f"treatment manifest 记录（{sidecar}）——provenance 未知"
                f"（旧协议残留/手工放置），拒绝覆盖（不覆盖、不切指针），"
                f"fail loudly")
        try:
            with open(sidecar, "r", encoding="utf-8") as f:
                existing = f.read()
        except OSError as e:
            raise SystemExit(
                f"[GATE-FAIL] 076: {sidecar} 读取失败（{e}）——无法证明"
                f"目标与本次 treatment 同一，拒绝覆盖，fail loudly")
        if existing != manifest_json:
            raise SystemExit(
                f"[GATE-FAIL] 076: 输出目标 {out_path} 已存在且 treatment "
                f"manifest 与本次不同（10-hex hash 碰撞或异配置同路径）"
                f"——拒绝覆盖。existing 前 200 字节={existing[:200]!r}，"
                f"current 前 200 字节={manifest_json[:200]!r}")
    return sidecar


def split_method_name_hash(method_name):
    """076 配套：把 method_name 拆成 (可读段, 身份 hash 段)。

    hash 段形如 "_h<10 hex>"（含前导下划线，可直接拼回）；非 tli 名
    （quest/twia/tia/none——无 hash 段）返回 ("原名", "")。供 LongBench
    pred.py 的超长名截断「先截可读中段、保 hash 尾段」使用。"""
    m = _TREATMENT_HASH_RE.match(method_name)
    if not m:
        return method_name, ""
    return m.group("readable"), "_" + m.group("hash")


def truncate_output_name_keep_hash(dataset_prefix, method_name, t,
                                    limit=230):
    """076 配套：LongBench 输出文件名超长截断——身份 hash 尾段优先保留。

    旧口径 out_fn[:245]+"..." 会从尾部截，method_name 尾部的
    "_h<hash10>" 被截后不同 treatment 可再次同名互覆（076 回归）。
    新口径：可读中段先截（"..." 占位），dataset 前缀 / hash 尾段 /
    "-t" 后缀全保；非 tli 名（无 hash 段）回退旧口径（其可读段已全
    参数编码）。

    【081 + kimi3 0316 追加修复 1】limit 缺省 245 → 230：上限与
    sidecar 后缀链绑定——out_fn 落盘时实际还要拼 ".jsonl"（6 字节），
    且 076 写门会为它创建 sidecar "{out_fn}.tli_manifest.json"（18
    字节），故 245 截断下 sidecar 实际文件名 = 245+6+18 = 269 字节
    > ext4 单文件名 255 上限，open(sidecar,"w") 抛 OSError 36 裸
    traceback，破坏 076 fail-closed 统一口径（kimi3 实测复现）。
    230 = 255 − 6 − 18 − 1 余量。极端长 prefix/t 下宁可略超限也
    不丢身份段（如实声明，不静默截 hash）；调用方传显式 limit 时
    须自行保证 sidecar 链长约束。
    """
    t = str(t)
    readable, hash_seg = split_method_name_hash(method_name)
    if not hash_seg:
        full = f"{dataset_prefix}-{method_name}-{t}"
        return full[:limit] + "..."
    room = limit - (len(dataset_prefix) + 1 + len("...") +
                    len(hash_seg) + 1 + len(t))
    return f"{dataset_prefix}-{readable[:max(room, 0)]}...{hash_seg}-{t}"


def get_method_name_with_info(args, snapshot=None):
    # 081：snapshot 提供时 tli 名的 _h<hash10> 段从冻结 manifest_json
    # 派生（与 sidecar/receipt 同源）；非 tli 分支不消费 snapshot。
    if args.method == "quest":
        return f"quest_{args.quest_block_size}_{args.quest_topk}"
    elif args.method == "twi":
        return f"twi_{args.twi_block_size}_{args.twi_level1_topk}_{args.twi_level2_topp}"
    elif args.method == "tia":
        async_info = "_async" if args.tia_enable_async_topk else ""
        return f"tia_{args.tia_block_size}_{args.tia_level1_topk}_{args.tia_level2_topk}_c{args.tia_level2_cmp_ratio}{async_info}"
    elif args.method == "tli":
        # 【F11 修复（kimi3 清单 2026-10-08）】ab 用「生效值」而非 flag 原值：
        # tli_subspace=full 时 TLIIndexer.__init__ 内部强制 enable_subspace=False
        # → 不写 "A"（原按 flag 原值写 "A"，默认 full 口径下文件名虚标子空间，
        # 与实际 treatment 不符——历史 ARM_CONTRACT 串 tli_64_128_1024_c4_A 的
        # "A" 即此虚标，既有文件名不追溯改动）。
        subspace = getattr(args, "tli_subspace", "full")
        eff_subspace = getattr(args, "tli_enable_subspace", True)
        if subspace == "full":
            eff_subspace = False          # 与 TLIIndexer.__init__ 同源
        elif subspace in ("rope", "nope"):
            eff_subspace = True
        ab = ("A" if eff_subspace else "") + ("B" if args.tli_enable_kmeans else "") + ("D" if args.tli_enable_layer_skip else "")
        # 【B09 修复（kimi3 清单 2026-10-08）】α/β/γ 是 treatment 的一部分，
        # 必须进输出文件名（a0.25_b0.125_g0.625 风格），否则不同分区配置
        # 共享同 method_name → 同路径 w 模式互相覆盖 / 同名混装不可区分。
        # γ=off（显式传 "off" → None）渲染 "off"；缺省属性按 1.0 兜底——与
        # TLIIndexer.__init__ 的运行时缺省 getattr(args,"tli_gamma",1.0) 同源，
        # 否则程序化构造的 args（无该属性）文件名标 goff 而实际跑 γ=1.0，
        # 文件名/行为失配（B09 族）。
        alpha = getattr(args, "tli_alpha", 0.0)
        beta = getattr(args, "tli_beta", 0.0)
        gamma = getattr(args, "tli_gamma", 1.0)
        g_str = "off" if gamma is None else f"{gamma:g}"
        # 【10-09 TL-PREFILL-PROVENANCE-001 修复】稀疏 prefill 开关必须进输出文件名，
        # 否则 P-off/P-on 同路径 w 模式互相覆盖（GPT 审计 B09 族新实例）。默认不传
        # flag 时后缀为空 → 在跑链文件名逐位不变。
        p = "_P" if getattr(args, "tli_sparse_prefill", False) else ""
        # E122 合并说明：γ=off 自由竞争经 g_str="off" 渲染进文件名
        # （a…_b…_goff），与 γ 数值臂天然区分，无需独立 _goff 后缀；
        # 默认缺省 γ 按 1.0 处理（getattr 兜底，只有显式 off 才是 None）。
        # 076（TL-E121-OUTPUT-ID）：可读段逐位不动（B09 口径），追加
        # canonical treatment hash 尾段 _h<hash10>——全部输出相关参数
        # （far/near method/select、subspace 具体值、投影基、per_q_head、
        # sim/σ/MoBA 选择配置等）排序键 JSON→sha256 的身份段。不同
        # treatment 必不同名（单射），同名互覆/指针错切不再可达。
        # 081：hash 从冻结 snapshot 派生（生产入口传入）——生成窗口内
        # 文件被替换时文件名与 runtime/sidecar 仍同源。
        return (f"tli_{args.tia_block_size}_{args.tia_level1_topk}_"
                f"{args.tia_level2_topk}_c{args.tia_level2_cmp_ratio}_"
                f"{ab}a{alpha:g}_b{beta:g}_g{g_str}{p}"
                f"_h{_treatment_hash(args, snapshot)}")
    elif args.method == "none":
        return "none"
    else:
        raise ValueError
