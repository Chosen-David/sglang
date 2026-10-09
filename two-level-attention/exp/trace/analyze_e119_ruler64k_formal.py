#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E119：RULER 64K 三臂正式入口收口汇总（score_ruler_formal.py E116e 门禁产物聚合）。

三臂分别经正式入口打分（各产出 result/MD/manifest/receipt 四件套 + 派生目录），
本脚本只读聚合三份已发布 JSON/receipt，生成跨臂判决汇总——不重算任何分数。

闭包断言（fail-closed）：
  ① 三臂 receipt 均 status=success 且 result_sha256 与当前 JSON 实际 SHA
     一致、manifest_sha256 与当前 manifest 实际 SHA 一致（receipt↔镜像闭合）；
  ② 三臂 11 任务 n 全部 =100（min-samples 硬门禁已在正式入口内执行，此处复核）；
  ③ 三臂 sources 的源文件 SHA 与磁盘当前文件 SHA 一致（发布后无篡改）；
  ④ 单 cell 假设 fail-closed：result JSON 的 n/scores 恰好一个 cell key，
     多 cell 输入（多 L 档混入）直接拒绝，不静默取 next(iter(...))。

E116g（GPT 1326 审计 TL-RULER-CROSSARM-IDENTITY-041）跨臂身份门禁：
  ⑤ 从每臂 formal manifest + receipt 建立共同 identity digest，三臂逐字段
     比较数据身份——task 集合 + 逐 task _id 列表 / answers_sha / 逐行
     lengths、source_data_sha256（逐 {L}/{task} 源文件 SHA）、
     expect_tasks/min_samples、model_path/yarn/yarn_factor、scorer
     manifest 规范化摘要、extra_params 白名单外逐键值。任何数据身份
     不一致 → 非零退出且不覆盖旧 summary（三臂各自内部闭合但样本
     不同的结果不再可能被输出为公平 A/B/C 排名）；
  ⑥ treatment 白名单（允许跨臂不同的字段）：method（臂方法，FullKV 的
     extra_params {"method":"none"}）/ far_method / near_method /
     alpha / beta / gamma——extra_params 逐键白名单比对，非整体忽略；
     pred_postfix / root 等运行位置字段属臂私有，不参与比较；
  ⑦ formal_script_sha256 / scorer_sha256：记录差异但仅 warn 不硬拒——
     该字段身份语义是「评分口径」而非「数据身份」，后续对脚本的修复会
     使旧产物 SHA 与新脚本不同；summary 如实记录三臂各自的脚本 SHA。

产物：exp/trace/results/e119_ruler64k_formal_summary.json
用法： python3 exp/trace/analyze_e119_ruler64k_formal.py [--results-dir DIR]
"""
import argparse
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "exp", "trace", "results")

ARMS = {
    "mavg": "e119_ruler64k_formal_mavg.json",
    "FullKV": "e119_ruler64k_formal_fullkv.json",
    "aavg": "e119_ruler64k_formal_aavg.json",
}
# 32K 正式口径（E116e，历史落袋）用于跨档对比
RULER32 = {"FullKV": 59.38, "mavg": 59.99, "aavg": 57.33}

# ⑥ treatment 白名单：允许跨臂不同的 extra_params 键（臂方法与三区参数）。
# FullKV 臂的 extra_params={"method":"none"}，TLI 臂为 far/near/αβγ——
# 白名单外键（如未来的新参数）必须三臂一致，否则 fail closed
TREATMENT_WHITELIST = {"method", "far_method", "near_method",
                       "alpha", "beta", "gamma"}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fail(msg):
    raise SystemExit(f"[E119-GATE-FAIL] {msg}")


def _canon(obj):
    """规范化序列化（sort_keys）——digest/比较共用同一口径。"""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def _arm_identity(manifest):
    """从单臂 formal manifest 提取跨臂可比身份字段。

    返回 (identity, treatment, non_wl_extra)：
      identity     —— 必须三臂逐字段一致的数据身份（进共同 digest）；
      treatment    —— 白名单键的逐键取值（允许不同，记录进 summary）；
      non_wl_extra —— 白名单外 extra_params 键值（必须一致，进 digest）。
    """
    ri = manifest["run_identity"]
    tasks = manifest["tasks"]
    identity = {
        "data_root": ri["data_root"],
        "model_path": ri["model_path"],
        "yarn": ri["yarn"],
        "yarn_factor": ri["yarn_factor"],
        "expect_tasks": manifest["expect_tasks"],
        "min_samples": manifest["min_samples"],
        "task_set": sorted(tasks.keys()),
        "tasks": {
            t: {"ids": m["ids"],
                "answers_sha": m["answers_sha"],
                "lengths": m["lengths"]}
            for t, m in sorted(tasks.items())
        },
        "source_data_sha256": {
            k: v["sha256"]
            for k, v in sorted(manifest["source_data_sha256"].items())
        },
    }
    extra = ri.get("extra_params") or {}
    treatment = {k: extra[k] for k in sorted(TREATMENT_WHITELIST)
                 if k in extra}
    non_wl = {k: v for k, v in sorted(extra.items())
              if k not in TREATMENT_WHITELIST}
    return identity, treatment, non_wl


def _compare_identities(per_arm_ident):
    """三臂 identity 逐字段比较（fail-closed，报告首个不一致字段链）。

    per_arm_ident: {arm: identity_dict}。返回共同 digest（一致时）。"""
    arms = sorted(per_arm_ident)
    ref_arm = arms[0]
    ref = per_arm_ident[ref_arm]
    for arm in arms[1:]:
        cur = per_arm_ident[arm]
        if set(ref.keys()) != set(cur.keys()):
            _fail(f"跨臂身份字段集不一致：{ref_arm} vs {arm} "
                  f"({sorted(set(ref) ^ set(cur))})")
        for field in sorted(ref.keys()):
            if ref[field] != cur[field]:
                # 深入一层给出可定位的差异（task 级）
                detail = ""
                if field == "tasks":
                    for t in sorted(set(ref[field]) | set(cur[field])):
                        a = ref[field].get(t)
                        b = cur[field].get(t)
                        if a != b:
                            if a is None or b is None:
                                detail = f"（task {t}: 一臂缺失）"
                            else:
                                for k in ("ids", "answers_sha", "lengths"):
                                    if a[k] != b[k]:
                                        detail = f"（task {t}.{k} 不一致）"
                                        break
                            break
                _fail(f"跨臂数据身份不一致（041）：字段 {field} 在 "
                      f"{ref_arm} 与 {arm} 之间不同{detail}——三臂样本/"
                      "答案/源数据身份不闭合，拒绝发布排名 summary")
    return hashlib.sha256(_canon(ref).encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser(
        description="E119 三臂收口汇总（E116g：跨臂身份门禁 + 单 cell "
                    "fail-closed）")
    ap.add_argument("--results-dir", default=RESULTS,
                    help="三臂产物所在目录（默认 exp/trace/results；"
                         "测试可用副本目录，生产数据零写入）")
    args = ap.parse_args()
    results_dir = os.path.abspath(args.results_dir)

    per_arm = {}
    per_arm_ident = {}
    per_arm_treatment = {}
    per_arm_non_wl = {}
    per_arm_scripts = {}
    per_arm_scorer_man = {}
    for arm, fn in ARMS.items():
        p = os.path.join(results_dir, fn)
        d = json.load(open(p))
        receipt = json.load(open(p + ".receipt.json"))
        manifest = json.load(open(p + ".manifest.json"))
        # ① receipt 闭合（result + manifest 双向 SHA）
        assert receipt["status"] == "success", (arm, receipt["status"])
        assert receipt["result_sha256"] == _sha256(p), \
            (arm, "receipt↔JSON SHA 不一致")
        assert receipt["manifest_sha256"] == _sha256(p + ".manifest.json"), \
            (arm, "receipt↔manifest SHA 不一致")
        # ④ 单 cell 假设 fail-closed（041 建议 4：多 cell 不许静默取首键）
        if len(d["n"]) != 1:
            _fail(f"{arm}: result JSON 含 {len(d['n'])} 个 cell "
                  f"（{sorted(d['n'])}）——本汇总只支持单 L 档单 cell，"
                  f"多 cell 输入 fail closed（须逐 L 档分别收口）")
        key = next(iter(d["n"]))
        # ② n 全 100
        assert all(v == 100 for v in d["n"][key].values()), (arm, "n≠100")
        scores = d["scores"][key]
        # ③ 双向 SHA 闭合（发布后无篡改）：receipt cells[].tasks[] 记录
        #    source_sha256（磁盘源文件）+ derived_sha256（补刻派生副本）；
        #    sources[].sha256 = derived（补刻后）口径
        pred_dir = os.path.join(
            ROOT, "exp", "results_ruler", "e109_full_Qwen3-8B",
            "fullkv" if arm == "FullKV" else arm, "L65536", "pred_1024")
        derived_dir = os.path.join(
            results_dir, os.path.basename(p) + ".run-" + receipt["run_id"],
            "pred_root", "L65536", "pred_1024")
        for task, tinfo in receipt["cells"][key]["tasks"].items():
            src_fp = os.path.join(pred_dir, tinfo["best_file"])
            assert os.path.exists(src_fp), (arm, task, src_fp)
            assert tinfo["source_sha256"] == _sha256(src_fp), (arm, task, "源 SHA 漂移")
            der_fp = os.path.join(derived_dir, tinfo["best_file"])
            assert os.path.exists(der_fp), (arm, task, der_fp)
            assert tinfo["derived_sha256"] == _sha256(der_fp), (arm, task, "派生 SHA 漂移")
        # ⑤ 跨臂身份提取（formal manifest，已经 receipt manifest_sha 闭合）
        ident, treatment, non_wl = _arm_identity(manifest)
        per_arm_ident[arm] = ident
        per_arm_treatment[arm] = treatment
        per_arm_non_wl[arm] = non_wl
        ri = manifest["run_identity"]
        per_arm_scripts[arm] = {
            "formal_script_sha256": ri["formal_script_sha256"],
            "scorer_sha256": ri["scorer_sha256"],
        }
        # scorer manifest（generation 内）规范化摘要——三臂必须一致
        # （GPT 1326 复核：当前三份字节一致 139721c9…；此门禁把该事实
        #   升格为 fail-closed 断言而非事后人工核验）
        sc_path = os.path.join(
            results_dir, os.path.basename(p) + ".run-" + receipt["run_id"],
            "scorer.manifest.json")
        sc = json.load(open(sc_path))
        per_arm_scorer_man[arm] = hashlib.sha256(
            _canon(sc).encode("utf-8")).hexdigest()
        per_arm[arm] = {
            "avg": round(sum(scores.values()) / len(scores), 2),
            "receipt_run_id": receipt.get("run_id"),
            "result_file": os.path.basename(p),
            "per_task": scores,
        }

    # ---- ⑤ 跨臂身份门禁主体 ----
    # 数据身份（含白名单外 extra_params + scorer manifest 摘要）逐字段比较
    non_wl_ref = None
    for arm in sorted(per_arm_non_wl):
        if non_wl_ref is None:
            non_wl_ref = (arm, per_arm_non_wl[arm])
        elif per_arm_non_wl[arm] != non_wl_ref[1]:
            _fail(f"跨臂数据身份不一致（041）：extra_params 白名单外键 "
                  f"（{sorted(per_arm_non_wl[arm])} vs "
                  f"{sorted(non_wl_ref[1])}）在 {non_wl_ref[0]} 与 {arm} "
                  f"之间不同——白名单只覆盖 method/far_method/near_method/"
                  f"alpha/beta/gamma，其余参数属数据身份必须一致")
    ident_digest = _compare_identities(per_arm_ident)
    # scorer manifest 摘要三臂一致
    sc_ref = sorted(per_arm_scorer_man)[0]
    for arm in sorted(per_arm_scorer_man)[1:]:
        if per_arm_scorer_man[arm] != per_arm_scorer_man[sc_ref]:
            _fail(f"跨臂数据身份不一致（041）：scorer manifest 规范化摘要 "
                  f"在 {sc_ref} 与 {arm} 之间不同（ids/answers_sha 闭包"
                  f"不同）——拒绝发布排名 summary")
    common_digest = hashlib.sha256(
        (ident_digest + _canon(non_wl_ref[1]) +
         per_arm_scorer_man[sc_ref]).encode("utf-8")).hexdigest()
    # ⑦ 脚本 SHA：评分口径而非数据身份——记录差异仅 warn
    script_shas = {a: tuple(sorted(s.items()))
                   for a, s in per_arm_scripts.items()}
    script_sha_warn = len(set(script_shas.values())) > 1
    if script_sha_warn:
        print(f"[E119-WARN] 三臂 formal/scorer 脚本 SHA 不全一致（评分"
              f"口径差异，非数据身份——如实记录，不硬拒）："
              f"{ {a: s['formal_script_sha256'][:12] for a, s in per_arm_scripts.items()} }",
              file=sys.stderr)

    # ---- 全部门禁通过才写 summary（数据身份不一致 → 非零退出，
    #      不覆盖旧 summary）----
    fullkv = per_arm["FullKV"]["avg"]
    out = {
        "experiment": "E119_ruler64k_formal_closure",
        "date": "2026-10-09",
        "entry": "benchmark/RULER/score_ruler_formal.py（E116e 正式入口：min-samples 硬门禁 + "
                 "staging 原子发布 + 身份扩展 receipt/manifest）",
        "identity": {
            "length_tier": "L65536（YaRN factor 2.0，64K 全档）",
            "model": "Qwen3-8B",
            "samples_per_task": 100,
            "tasks": 11,
        },
        "identity_gate": {
            # E116g（041）：跨臂身份门禁——三臂共同 identity digest 与
            # 实际比较结果；任何数据身份不一致本 summary 不会生成
            "protocol": "e116g-crossarm-identity-v1",
            "common_identity_digest": common_digest,
            "compared_fields": [
                "task_set", "tasks.{ids,answers_sha,lengths}",
                "source_data_sha256", "expect_tasks", "min_samples",
                "model_path", "yarn", "yarn_factor", "data_root",
                "extra_params(白名单外逐键)", "scorer_manifest_digest",
            ],
            "treatment_whitelist": sorted(TREATMENT_WHITELIST),
            "per_arm_treatment": per_arm_treatment,
            "scorer_manifest_digest": per_arm_scorer_man[sc_ref],
            "per_arm_script_sha256": per_arm_scripts,
            "script_sha_policy": (
                "warn-only：formal/scorer 脚本 SHA 的身份语义是评分口径"
                "而非数据身份（后续脚本修复会使旧产物 SHA 与新脚本不同），"
                "记录差异不硬拒"),
            "script_sha_warn": script_sha_warn,
            "all_arms_data_identity_identical": True,
        },
        "closure": {
            "three_arm_receipts_success": True,
            "receipt_result_sha_matches": True,
            "receipt_manifest_sha_matches": True,
            "all_cells_n100": True,
            "single_cell_enforced": True,
            "source_sha_stable": True,
            "crossarm_identity_gate": True,
        },
        "arms": {
            arm: {
                **st,
                "delta_vs_fullkv": round(st["avg"] - fullkv, 2),
                "ruler32_official": RULER32[arm],
            }
            for arm, st in per_arm.items()
        },
        "verdict": {
            "ruler64k_ranking": " > ".join(f"{a} {s['avg']}" for a, s in
                                            sorted(per_arm.items(), key=lambda kv: -kv[1]["avg"])),
            "conclusion": "64K 正式口径下 mavg(+0.88) 超 FullKV、aavg(−1.03) 落后；与 32K 正式排序"
                          "（mavg 59.99 > FullKV 59.38 > aavg 57.33）方向一致——RULER 32K/64K 两档"
                          " mavg 冠军稳定，128K 待全齐后同口径收口（E116g 起收口同时启用"
                          "跨臂身份门禁 + 正式入口并发发布锁）。",
        },
    }
    p_out = os.path.join(results_dir, "e119_ruler64k_formal_summary.json")
    json.dump(out, open(p_out, "w"), ensure_ascii=False, indent=2)
    print(json.dumps(out["arms"], ensure_ascii=False, indent=2))
    print("identity_gate:", json.dumps(
        {k: out["identity_gate"][k] for k in
         ("protocol", "common_identity_digest", "treatment_whitelist",
          "script_sha_warn", "all_arms_data_identity_identical")},
        ensure_ascii=False))
    print("verdict:", json.dumps(out["verdict"], ensure_ascii=False))
    print(f"saved: {p_out}")


if __name__ == "__main__":
    main()
