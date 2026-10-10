import hashlib
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
# tli_enable_kmeans/tli_enable_layer_skip 已由可读段 B/D 二值编码、
# 不重复进 manifest（取舍：ABD 三字符对这两个 flag 本身已单射）。
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
    "tli_proj_basis": None,         # 路径级身份（内容身份由 argv/receipt 承载）
    "tli_static_pair": False,
    "tli_layer_skip_path": None,
}

# treatment hash 尾段格式：_h + 10 hex。锚定尾正则拆分（可读段以
# g 段/_P 结尾、非 tli 名以数值/async/none 结尾，不与该模式误撞）。
_TREATMENT_HASH_HEX_LEN = 10
_TREATMENT_HASH_RE = re.compile(
    r"^(?P<readable>.+)_(?P<hash>h[0-9a-f]{%d})$" % _TREATMENT_HASH_HEX_LEN,
    re.DOTALL)


def _treatment_manifest(args):
    """076：canonical treatment manifest——全部输出相关参数的
    规范化字典（getattr 兜底缺省与 TLIIndexer 运行时同源）。"""
    return {f: getattr(args, f, d)
            for f, d in _TREATMENT_FIELD_DEFAULTS.items()}


def get_treatment_manifest_json(args):
    """076：canonical treatment manifest 的规范化 JSON 串。

    与 _treatment_hash 同源同字节（hash 输入 = 本串 utf-8 编码）——
    sidecar 记录与 hash 计算共用单一序列化口径，sidecar 内容可独立
    复核 hash（审计可再算）。排序键 + default=str 兜不可序列化值。"""
    return json.dumps(_treatment_manifest(args), sort_keys=True,
                      ensure_ascii=False, default=str)


def _treatment_hash(args):
    """076：manifest 排序键 JSON 序列化 → sha256 前 10 hex。
    同一 args 对象/同配置多次调用稳定；default=str 兜不可序列化值。"""
    payload = get_treatment_manifest_json(args)
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
                                    limit=245):
    """076 配套：LongBench 输出文件名超长截断——身份 hash 尾段优先保留。

    旧口径 out_fn[:245]+"..." 会从尾部截，method_name 尾部的
    "_h<hash10>" 被截后不同 treatment 可再次同名互覆（076 回归）。
    新口径：可读中段先截（"..." 占位），dataset 前缀 / hash 尾段 /
    "-t" 后缀全保；非 tli 名（无 hash 段）回退旧口径（其可读段已全
    参数编码）。limit 245 为既有软上限（ext4 255 减 ".jsonl" 余量）；
    极端长 prefix/t 下宁可略超限也不丢身份段（如实声明，不静默截 hash）。
    """
    t = str(t)
    readable, hash_seg = split_method_name_hash(method_name)
    if not hash_seg:
        full = f"{dataset_prefix}-{method_name}-{t}"
        return full[:limit] + "..."
    room = limit - (len(dataset_prefix) + 1 + len("...") +
                    len(hash_seg) + 1 + len(t))
    return f"{dataset_prefix}-{readable[:max(room, 0)]}...{hash_seg}-{t}"


def get_method_name_with_info(args):
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
        return (f"tli_{args.tia_block_size}_{args.tia_level1_topk}_"
                f"{args.tia_level2_topk}_c{args.tia_level2_cmp_ratio}_"
                f"{ab}a{alpha:g}_b{beta:g}_g{g_str}{p}"
                f"_h{_treatment_hash(args)}")
    elif args.method == "none":
        return "none"
    else:
        raise ValueError
