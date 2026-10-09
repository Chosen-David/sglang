# -*- coding: utf-8 -*-
# E117a-mavg —— 同方法（mavg = far/minmax + near/avg）参照回放（GPT 2026-10-09
# E117 MatchedFixedReference Addendum 最小回放契约，任务 #183）。
#
# 背景：原 E117a（e117a_wo_projection.json，NO-GO，gap 中位 8.045%）全程在
#   avg/avg 代理口径下评估——参照臂 G*=(.125,.125,.375) 是 aavg 海选领跑臂，
#   CHAMP_REF=(.25,.125,.625) 只是旁观记录（E_champ_ref_cal 是它在 avg/avg
#   分数下的校准误差）。GPT 审计：该判决只对 avg/avg 路径成立，不能外推
#   mavg(minmax/avg)；「平坦性第五证」表述过强已更正。
#
# 本脚本 = 薄 wrapper（原脚本 analyze_e117a_wo_projection.py **不许改**，
#   与其 import 复用 analyze_p0p_perlayer_potential.py 的纪律一致——不复制
#   700 行避免双份漂移），注入三组默认参数：
#   ① --far-method minmax --near-method avg  ：真正的 mavg 打分口径
#   ② --gstar 0.25,0.125,0.625               ：G* = mavg 已保存最好配置
#     （= CHAMP_REF = c_U 锚点；GPT addendum 第 2/4 条要求「统一三元组 c_U vs
#     逐层表 c_l 的同方法配对比较」+「已保存最好三元组纳入候选/作锚点，若与
#     c_U 相同则合并」——gap = (E(c_U)−E(c_l))/E(c_U) 正是该配对比较；
#     若保留 aavg 臂当 G*，则参照臂在 minmax 分数下无理由好，gap 虚高假 GO）
#   ③ --out results/e117a_mavg_ref.json       ：落袋路径
#   用户显式命令行参数可覆盖任何默认值（只在不出现时注入）。
#
# CHAMP_REF=(.25,.125,.625) 硬编码**保持不变**（主 AI 2026-10-09 已承诺
#   「CHAMP_REF 三元组不变——这次它才真正是 mavg 参照」，此时 G* 与
#   CHAMP_REF 重合，E_champ_ref_cal == E_gstar_cal，符合「相同则合并」）。
#
# 运行后自动读 v1（avg/avg）结果做新旧口径对比，注入本次输出 JSON 的
#   "comparison_with_avg_avg_v1" 字段。
#
# 053（GPT 2026-10-10 0125 审计 TL-E117A-OUT-BINDING-053）：显式输出与
#   后处理目标分叉——修复前 main() 用 out_default(argv)（只区分
#   --dry-run）定位注入目标，丢弃用户显式 --out PATH / --out=PATH，
#   comparison 元数据注入默认路径而用户输出无注入。修复：从
#   inject_defaults 返回后的 argv 解析出唯一 resolved output path
#   （注入后必含 --out，两种形式都解析），base.main() 写盘与
#   attach_comparison 注入用同一值（见 resolve_out_path）。
#
# 纯 CPU 零 GPU（与 GPU 主链零冲突）。
#
# 用法：
#   python3 analyze_e117a_mavg_ref.py --dry-run   # 合成 trace 干跑全链路自检
#   python3 analyze_e117a_mavg_ref.py             # 真实 trace + 真实 W_O（CPU，~2.5h）
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import analyze_e117a_wo_projection as base   # noqa: E402  原脚本只复用不改

# ---------------- 回放 harness 修复（mavg 口径专用，进程内 monkeypatch）--------
# 真跑首跑实锤的缺陷：eval_layer_sample 逐候选原地改 indexer.α/β/γ，下一 query
# 的 prepare_index/compute_score 携带**上一候选残留的 α/β/γ**——kept 域含单池
# 臂（α=0 或 β=0）时 _need_avg_score()=False → k_avg 不算 → score_coarse_avg
# 缺失 → 其后分区候选的 near=avg 静默退化 minmax（E109a bug 模式，正是原脚本
# L242 断言要抓的）。原 avg/avg 口径天然免疫（far="avg" 恒需分数源）；干跑未
# 触发纯属 kept 域末尾恰好是分区臂。
# 修复：far/near 含 "avg" 时无条件产出 avg 分数。语义无损——far=minmax 时
# score_coarse_avg 仅被 near 池消费（tli_indexer.py L863-866；far 池 L855 需
# far_method=="avg" 不触发），prepare_index/compute_score 无任何其他 α/β/γ
# 依赖（grep 核实）；单池候选多算一份分数不改变任何 mask 结果。
# 生产代码不改（生产固定配置 α,β>0 时 _need_avg_score 恒真，无此缺陷）。
from sparse_attn.indexer.tli_indexer import TLIIndexer   # noqa: E402

_orig_need_avg = TLIIndexer._need_avg_score


def _need_avg_score_mavg_replay(self):
    if "avg" in (self.far_method, self.near_method):
        return True
    return _orig_need_avg(self)


TLIIndexer._need_avg_score = _need_avg_score_mavg_replay

PATCH_SELFCHECK = {}   # patch 红绿自检结果（注入产出 JSON 供审计）


def patch_selfcheck(seed=20261010):
    """回放 harness 修复的红绿自检（S=4096 合成张量，纯 CPU，秒级）。

    红：原始 _need_avg_score 在单池残留参数（α=β=0）下返回 False——缺陷根因
        证据（残留会让下一 query 的 avg 分数源缺失）。
    绿 1：patch 后残留单池参数仍产出 k_avg → score_coarse_avg（分数源不缺失）。
    绿 2：残留参数构建的分数源上算分区候选 mask，与干净 prepare/score 的
        mask 逐位一致——证明 prepare_index/compute_score 确与 α/β/γ 无关
        （除分数源开关外），修复无副作用。
    """
    import torch
    p0p = base.p0p
    S = 4096
    torch.manual_seed(seed)
    k = torch.randn(1, S, 8, 128)
    q = torch.randn(1, 1, 32, 128)
    cu = torch.tensor([0, S], dtype=torch.long)
    q_ids = torch.tensor([S - 1], dtype=torch.long)
    scale = 128 ** -0.5
    part = (0.25, 0.125, 0.625)   # 分区候选（= G* = mavg 主配置）
    # 红：原始语义在单池残留参数下 = False
    idx_r = p0p.make_indexer("minmax", "avg", *part, layer_idx=1)
    idx_r.alpha, idx_r.beta, idx_r.gamma = 0.0, 0.0, 0.0
    red = not _orig_need_avg(idx_r)
    # 绿 1：patch 后残留单池参数 → 分数源仍在
    idict = idx_r.prepare_index(k, cu)
    sdict = idx_r.compute_score(q, q_ids, idict, scale)
    green_src = bool(idict.get("k_avg") is not None
                     and sdict.get("score_coarse_avg") is not None)
    # 绿 2：残留分数源上分区候选 mask vs 干净分数源 mask 逐位一致
    idx_r.alpha, idx_r.beta, idx_r.gamma = part
    m_residue = idx_r.compute_mask(q_ids, sdict)
    idx_c = p0p.make_indexer("minmax", "avg", *part, layer_idx=1)
    idict_c = idx_c.prepare_index(k, cu)
    sdict_c = idx_c.compute_score(q, q_ids, idict_c, scale)
    m_clean = idx_c.compute_mask(q_ids, sdict_c)
    bitwise_same = bool(torch.equal(m_residue, m_clean))
    ok = bool(red and green_src and bitwise_same)
    return {"red_orig_need_avg_false_on_single_pool_residue": red,
            "green_avg_score_present_under_residue": green_src,
            "residue_vs_clean_mask_bitwise_equal": bitwise_same,
            "pass": ok}


# mavg 同方法口径默认值（用户显式参数可覆盖）
METHOD_DEFAULTS = [
    "--far-method", "minmax",
    "--near-method", "avg",
    "--gstar", "0.25,0.125,0.625",
]
V1_PATH = os.path.join(HERE, "results", "e117a_wo_projection.json")


def out_default(argv):
    dry = "--dry-run" in argv
    return os.path.join(
        HERE, "results",
        "e117a_mavg_ref_dryrun.json" if dry else "e117a_mavg_ref.json")


def inject_defaults(argv):
    """注入默认参数：仅当用户未显式给出时（不覆盖用户值）。"""
    out = list(argv)
    # argparse 接受 --k=v 与 --k v 两种形式，均视为已显式给出
    present = set()
    for a in out:
        if a.startswith("--"):
            present.add(a.split("=", 1)[0])
    for i, a in enumerate(out):
        if a.startswith("--") and "=" in a:
            present.add(a.split("=", 1)[0])
    for k, v in zip(METHOD_DEFAULTS[::2], METHOD_DEFAULTS[1::2]):
        if k not in present:
            out += [k, v]
    if "--out" not in present:
        out += ["--out", out_default(out)]
    return out


def resolve_out_path(argv):
    """053：解析（注入默认后的）argv 的唯一 resolved 输出路径。

    --out PATH 与 --out=PATH 两种形式都解析；多次出现取最后一个（与
    argparse 后值覆盖语义一致）。inject_defaults 返回后 argv 必含
    --out——base.main() 的写盘目标与 attach_comparison 的注入目标必须
    取同一值。修复前 main() 用 out_default(argv) 重新推导（只区分
    --dry-run），用户显式 --out 被丢弃 → comparison 注入错文件
    （默认路径有旧文件时被误改）或静默丢失（默认路径不存在时
    attach_comparison 直接 return）。"""
    out = None
    for i, a in enumerate(argv):
        if a == "--out":
            if i + 1 >= len(argv):
                raise SystemExit("resolve_out_path: --out 位于 argv 末尾缺值")
            out = argv[i + 1]
        elif a.startswith("--out="):
            out = a[len("--out="):]
    if out is None:
        raise SystemExit("resolve_out_path: argv 缺 --out"
                         "（inject_defaults 注入后不可达）")
    return out


def attach_comparison(out_path):
    """读 v1（avg/avg）与本次 mavg 回放，生成新旧口径对比注入结果 JSON。"""
    if not os.path.isfile(V1_PATH):
        print(f"[warn] v1 结果缺失，跳过对比注入: {V1_PATH}")
        return
    if not os.path.isfile(out_path):
        return
    v1 = json.load(open(V1_PATH))
    v2 = json.load(open(out_path))
    def _med(x):
        return x["global"]["median_gap_rel_cal"]
    def _medc(x):
        return x["global"]["median_gap_rel_conf"]
    cmp_rec = {
        "note": ("v1=e117a_wo_projection.json（avg/avg 代理口径，G*=aavg 领跑臂"
                 " (.125,.125,.375)）；v2=本回放（mavg=minmax/avg 同方法口径，"
                 "G*=CHAMP_REF=(.25,.125,.625)=mavg 已保存最好配置）。两份 gap "
                 "分母参照臂不同（E117 判决门语义=逐层表 c_l 比「部署固定臂」"
                 "好多少——各自口径下取该方法的部署臂），对比须连同口径一起读"),
        "v1": {
            "exp": v1.get("exp"),
            "far_method": v1["config"]["far_method"],
            "near_method": v1["config"]["near_method"],
            "gstar": v1["config"]["gstar"],
            "median_gap_rel_cal": _med(v1),
            "median_gap_rel_conf": _medc(v1),
            "gap_rel_cal_by_layer": v1["global"]["gap_rel_cal_by_layer"],
            "verdict": v1["global"]["verdict"],
        },
        "v2": {
            "exp": v2.get("exp"),
            "far_method": v2["config"]["far_method"],
            "near_method": v2["config"]["near_method"],
            "gstar": v2["config"]["gstar"],
            "median_gap_rel_cal": _med(v2),
            "median_gap_rel_conf": _medc(v2),
            "gap_rel_cal_by_layer": v2["global"]["gap_rel_cal_by_layer"],
            "verdict": v2["global"]["verdict"],
        },
    }
    m1, m2 = _med(v1), _med(v2)
    if m1 is not None and m2 is not None:
        cmp_rec["delta_median_gap_cal"] = round(m2 - m1, 6)
        cmp_rec["same_direction"] = (m1 <= 0.10) == (m2 <= 0.10)
    v2["comparison_with_avg_avg_v1"] = cmp_rec
    if PATCH_SELFCHECK:
        v2["replay_harness_patch"] = {
            "what": ("回放 harness 修复（进程内 monkeypatch，不改生产代码）："
                     "eval_layer_sample 逐候选改 α/β/γ 后残留进下一 query 的"
                     " prepare_index/compute_score，kept 域含单池臂时 "
                     "_need_avg_score()=False → avg 分数源缺失 → 分区候选 "
                     "near=avg 静默退化 minmax。patch 使 far/near 含 avg 时"
                     "无条件产出分数源（far=minmax 下仅 near 池消费，语义无损）"),
            "patch": "TLIIndexer._need_avg_score → avg method 恒 True（本进程）",
            "scope": "仅本回放进程；生产固定配置 α,β>0 时原语义恒真，无此缺陷",
            "selfcheck": PATCH_SELFCHECK,
        }
    json.dump(v2, open(out_path, "w"), ensure_ascii=False, indent=1)
    print(f"\n=== 新旧口径对比（校准 gap_rel 中位）===")
    print(f"  v1 avg/avg  (G*=aavg .125,.125,.375): {m1}")
    print(f"  v2 minmax/avg (G*=mavg .25,.125,.625): {m2}"
          f"  → 判决：{v2['global']['verdict']}")
    if "delta_median_gap_cal" in cmp_rec:
        print(f"  delta(median gap cal) = {cmp_rec['delta_median_gap_cal']:+.6f}"
              f"  方向一致: {cmp_rec['same_direction']}")
    print(f"对比已注入 -> {out_path} (comparison_with_avg_avg_v1)")


def main():
    argv = inject_defaults(sys.argv[1:])
    sys.argv = [sys.argv[0]] + argv
    # 053：显式 --out / 默认注入走同一解析——base.main() 写盘目标与
    # attach_comparison 注入目标必须是同一个文件（修复前 attach 用
    # out_default(argv) 丢弃用户显式 --out，注入错文件或静默丢失）
    out_path = resolve_out_path(argv)
    # patch 红绿自检先行（失败即退出，不进入长跑）
    global PATCH_SELFCHECK
    PATCH_SELFCHECK = patch_selfcheck()
    print(f"[patch-selfcheck] {json.dumps(PATCH_SELFCHECK, ensure_ascii=False)}")
    if not PATCH_SELFCHECK["pass"]:
        print("[patch-selfcheck] FAIL —— 回放 harness 修复红绿自检未过，中止")
        sys.exit(2)
    rc = 0
    try:
        base.main()
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else 0
    # 无论判决如何都注入对比（自检失败 rc≠0 也保留现场供审计）
    attach_comparison(out_path)
    if rc:
        sys.exit(rc)


if __name__ == "__main__":
    main()
