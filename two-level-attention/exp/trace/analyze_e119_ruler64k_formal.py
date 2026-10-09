#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E119：RULER 64K 三臂正式入口收口汇总（score_ruler_formal.py E116e 门禁产物聚合）。

三臂分别经正式入口打分（各产出 result/MD/manifest/receipt 四件套 + 派生目录），
本脚本只读聚合三份已发布 JSON/receipt，生成跨臂判决汇总——不重算任何分数。

闭包门禁（fail-closed，全部为显式条件 + _fail，不使用 assert——python -O
不会删除任何检查；见 E116h 审计 045）：
  ① 三臂 receipt 均 status=success 且 result_sha256 与当前 JSON 实际 SHA
     一致、manifest_sha256 与当前 manifest 实际 SHA 一致（receipt↔镜像闭合）；
  ② 每 task n == len(ids) == len(lengths) == len(answers_sha) 且 ID 键集闭合
     （set(answers_sha) == set(ids)、ids 无重复）、n >= manifest min_samples
     （045/046：正式入口 min-samples 硬门禁的复核，不再硬编码 n==100）；
  ③ 三臂 sources 的源文件 SHA 与磁盘当前文件 SHA 一致、派生副本 SHA 一致
     （发布后无篡改）；
  ④ 单 cell 假设 fail-closed：result JSON 的 n/scores 恰好一个 cell key，
     多 cell 输入（多 L 档混入）直接拒绝，不静默取 next(iter(...))；
  ⑤ result 的 cell/task 集与 manifest/scorer task 集完全相等（046①：
     cell 集在 result n/scores、manifest cells、receipt cells 四处一致；
     task 集在 result n/scores、manifest tasks、receipt cells[key].tasks、
     scorer.manifest 四处一致，缺一不可）。

E116g（GPT 1326 审计 TL-RULER-CROSSARM-IDENTITY-041）跨臂身份门禁：
  ⑥ 从每臂 formal manifest + receipt 建立共同 identity digest，三臂逐字段
     比较数据身份——task 集合 + 逐 task _id 列表 / answers_sha / 逐行
     lengths、source_data_sha256（逐 {L}/{task} 源文件 SHA）、
     expect_tasks/min_samples、model_path/yarn/yarn_factor、scorer
     manifest 规范化摘要、extra_params 白名单外逐键值。任何数据身份
     不一致 → 非零退出且不覆盖旧 summary（三臂各自内部闭合但样本
     不同的结果不再可能被输出为公平 A/B/C 排名）；
  ⑦ treatment 白名单（允许跨臂不同的字段）：method（臂方法，FullKV 的
     extra_params {"method":"none"}）/ far_method / near_method /
     alpha / beta / gamma——extra_params 逐键白名单比对，非整体忽略；
     pred_postfix / root 等运行位置字段属臂私有，不参与比较。

E116h（GPT 1429 审计 045/046/047）消费者闭包修复：
  ⑧ 显式 arm 契约（046②）：mavg 必须 far_method=minmax / alpha=0.25 /
     beta=0.125 / gamma=0.625，aavg 必须 far_method=avg / α=β=γ=0，
     FullKV 必须 method=none——从 manifest treatment（白名单键）逐字段
     校验，treatment 与契约任一字段不匹配 → fail-closed（交换 treatment
     把数值冠到错误臂名称之下的路径不可达）；
  ⑨ receipt 协议绑定（046③）：publish_protocol 缺失（legacy）→ generation
     目录按 results_dir + basename + run_id 构造路径读取，summary 逐臂
     标 legacy_protocol=true（只声明「数据身份门禁已执行」，不声称
     E116f/E116g 发布锁协议作用于该批数据）；"e116f-generation-v2" →
     从 receipt.outputs.derived_dir 单指针解析同一 generation 并复核
     generation_files 四规范文件 SHA 与 receipt 声明逐位一致；其他
     未知协议值 → fail-closed；
  ⑩ 评分口径公平门禁（047）：三臂 formal/scorer 脚本 SHA 必须完全一致，
     否则 fail-closed（无等价迁移证明时不产出冠军结论——数据身份相同
     不足以证明 A/B/C 评分公平，评分实现差异可改变得分与排名）；
  ⑪ 结论动态生成（046④）：ranking/delta/conclusion 全部从结构化数值
     生成，不硬编码任何 64K 结果数字。

E116h（GPT 1531 审计 TL-E119-SUMMARY-ATOMICITY-049）summary 原子发布：
  ⑫ summary 写入不再直接 `open(p_out, "w")` 截断——同目录临时文件完整
     序列化 + flush/fsync + 落盘后重新解析校验必要字段，全部通过才
     os.replace 原子替换公开路径；发布锁（锁键与 042 同口径
     realpath(parent)+lexical basename）串行化并发汇总器。任何写入异常
     （进程被杀/磁盘写满/写中断）只遗留或清理临时文件，上一份
     last-known-good summary 字节不变；summary 同时保留三臂不可变输入
     引用（result/manifest/receipt 逐臂 SHA），使 last-writer-wins 的
     输入代际可审计。

产物：exp/trace/results/e119_ruler64k_formal_summary.json
用法： python3 exp/trace/analyze_e119_ruler64k_formal.py \
          [--results-dir DIR] [--pred-root DIR]
  --pred-root：生产预测目录根（默认 exp/results_ruler/e109_full_Qwen3-8B，
     其下 {mavg|aavg|fullkv}/L65536/pred_1024/ 为各臂源预测；测试/干净
     检出可用已入库 fixture 的 pred_root 替代——048① 的路径参数化）。
"""
import argparse
import fcntl
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "exp", "trace", "results")
PRED_ROOT = os.path.join(ROOT, "exp", "results_ruler", "e109_full_Qwen3-8B")

ARMS = {
    "mavg": "e119_ruler64k_formal_mavg.json",
    "FullKV": "e119_ruler64k_formal_fullkv.json",
    "aavg": "e119_ruler64k_formal_aavg.json",
}
# 32K 正式口径（E116e，历史落袋）用于跨档对比
RULER32 = {"FullKV": 59.38, "mavg": 59.99, "aavg": 57.33}

# ⑦ treatment 白名单：允许跨臂不同的 extra_params 键（臂方法与三区参数）。
# FullKV 臂的 extra_params={"method":"none"}，TLI 臂为 far/near/αβγ——
# 白名单外键（如未来的新参数）必须三臂一致，否则 fail closed
TREATMENT_WHITELIST = {"method", "far_method", "near_method",
                       "alpha", "beta", "gamma"}

# ⑧ 显式 arm 契约（046②）：每臂 treatment（白名单键的逐键取值）必须与
# 契约完全相等（多键/少键/值不匹配均拒绝）——把数值冠到错误臂名称
# 之下的唯一通道是篡改 treatment，此门禁使其 fail-closed
ARM_CONTRACT = {
    "mavg": {"far_method": "minmax", "near_method": "avg",
             "alpha": "0.25", "beta": "0.125", "gamma": "0.625"},
    "aavg": {"far_method": "avg", "near_method": "avg",
             "alpha": "0", "beta": "0", "gamma": "0"},
    "FullKV": {"method": "none"},
}

# ⑨ 已知发布协议（receipt.publish_protocol）：缺失 = E116e 及更早的
# 逐文件替换协议（legacy）；e116f-generation-v2 = generation 单指针协议
KNOWN_PROTOCOLS = {None, "e116f-generation-v2"}


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


def _bind_generation(arm, p, receipt, results_dir):
    """⑨（046③）：从 receipt 解析 generation 目录（单指针优先）。

    返回 (gen_dir, legacy_protocol)。legacy receipt（publish_protocol
    缺失）按 results_dir + basename + run_id 构造路径；e116f-generation-v2
    从 outputs.derived_dir 单指针解析并复核 generation_files 四规范文件
    SHA 与 receipt 声明逐位一致；未知协议 fail-closed。"""
    protocol = receipt.get("publish_protocol")
    if protocol not in KNOWN_PROTOCOLS:
        _fail(f"{arm}: receipt publish_protocol={protocol!r} 不是已知协议"
              f"（{sorted(str(x) for x in KNOWN_PROTOCOLS)}）——消费者不"
              f"认未知发布协议的产物，fail closed")
    if protocol is None:
        # legacy（E116e 及更早）：generation 目录按固定命名约定构造
        gen_dir = os.path.join(
            results_dir, os.path.basename(p) + ".run-" + receipt["run_id"])
        return gen_dir, True
    # e116f-generation-v2：receipt 单指针（outputs.derived_dir）
    gen_dir = receipt.get("outputs", {}).get("derived_dir")
    if not gen_dir or not os.path.isdir(gen_dir):
        _fail(f"{arm}: receipt outputs.derived_dir 缺失或不存在: "
              f"{gen_dir!r}——generation 单指针断裂，fail closed")
    gen_files = receipt.get("outputs", {}).get("generation_files")
    if not gen_files:
        _fail(f"{arm}: e116f-generation-v2 receipt 缺 generation_files "
              f"映射——无法从单指针解析同一 generation，fail closed")
    # 四规范文件存在 + SHA 与 receipt 声明逐位闭合
    rc_sha = {
        "manifest": receipt.get("manifest_sha256"),
        "json": receipt.get("result_sha256"),
    }
    for role, fname in gen_files.items():
        fp = os.path.join(gen_dir, fname)
        if not os.path.isfile(fp):
            _fail(f"{arm}: generation 缺规范文件 {role}: {fp}")
        if role in rc_sha and _sha256(fp) != rc_sha[role]:
            _fail(f"{arm}: generation 规范文件 {role} SHA 与 receipt 声明"
                  f"不一致（{fp}）——单指针闭合失败，fail closed")
    return gen_dir, False


def _publish_summary(p_out, out):
    """⑫（049）：summary 原子发布——同目录临时文件完整序列化 + fsync +
    落盘后重新解析校验必要字段，全部通过才 os.replace 原子替换公开
    路径；发布锁（锁键与 042 同口径 realpath(parent)+lexical basename）
    串行化并发汇总器。任何写入异常只遗留或清理临时文件，公开 summary
    （last-known-good）字节不变。"""
    lock_path = os.path.join(
        os.path.dirname(os.path.realpath(p_out)),
        os.path.basename(p_out) + ".lock")
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    p_tmp = None
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        # 042 同款教训：获锁后复核公开路径非 symlink——锁键 canonicalization
        # 与安装目标必须同一口径，否则首替换后锁键漂移
        if os.path.islink(p_out):
            _fail(f"summary 目标 {p_out} 是符号链接——fail closed"
                  f"（049：锁键与安装路径须同一 canonicalization）")
        p_tmp = f"{p_out}.tmp-{os.getpid()}"
        with open(p_tmp, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        # 落盘后重新解析 + 必要字段校验：未通过则不替换公开文件
        check = json.load(open(p_tmp, encoding="utf-8"))
        for must in ("identity", "identity_gate", "closure", "arms",
                     "verdict"):
            if must not in check:
                _fail(f"summary 落盘重解析缺必要字段 {must!r}——临时文件"
                      f"不替换公开 summary（049 fail closed）")
        os.replace(p_tmp, p_out)
        p_tmp = None
        # 父目录 fsync（best-effort）：replace 后的目录项持久化，
        # 平台不支持时忽略——原子性由 os.replace 本身保证
        try:
            dfd = os.open(os.path.dirname(p_out) or ".", os.O_RDONLY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        except OSError:
            pass
    finally:
        if p_tmp is not None and os.path.exists(p_tmp):
            try:
                os.remove(p_tmp)
            except OSError:
                pass
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)


def main():
    ap = argparse.ArgumentParser(
        description="E119 三臂收口汇总（E116g 跨臂身份门禁 + E116h 消费者"
                    "闭包：arm 契约/协议绑定/评分口径公平/动态结论）")
    ap.add_argument("--results-dir", default=RESULTS,
                    help="三臂产物所在目录（默认 exp/trace/results；"
                         "测试可用副本目录或已入库 fixture，生产数据零写入）")
    ap.add_argument("--pred-root", default=PRED_ROOT,
                    help="各臂源预测目录根（其下 {mavg|aavg|fullkv}/"
                         "L65536/pred_1024/；默认生产 exp/results_ruler/"
                         "e109_full_Qwen3-8B；干净检出测试用 fixture 的"
                         "pred_root——048①）")
    args = ap.parse_args()
    results_dir = os.path.abspath(args.results_dir)
    pred_root = os.path.abspath(args.pred_root)

    per_arm = {}
    per_arm_ident = {}
    per_arm_treatment = {}
    per_arm_non_wl = {}
    per_arm_scripts = {}
    per_arm_scorer_man = {}
    per_arm_protocol = {}   # ⑨：{arm: {"publish_protocol", "legacy_protocol"}}
    per_arm_inputs = {}     # ⑫（049）：逐臂不可变输入引用（SHA 可审计）
    samples_seen = set()    # ⑪：样本数元信息从数据推导，不硬编码
    task_count = None
    for arm, fn in ARMS.items():
        p = os.path.join(results_dir, fn)
        d = json.load(open(p))
        receipt = json.load(open(p + ".receipt.json"))
        manifest = json.load(open(p + ".manifest.json"))
        # ① receipt 闭合（result + manifest 双向 SHA）——显式条件（045）
        if receipt.get("status") != "success":
            _fail(f"{arm}: receipt status={receipt.get('status')!r}（预期 "
                  f"success）——收口只聚合成功发布，fail closed")
        if receipt["result_sha256"] != _sha256(p):
            _fail(f"{arm}: receipt↔JSON SHA 不一致（receipt 声明的 "
                  f"result_sha256 与当前 result JSON 实际 SHA 不同——"
                  f"发布后篡改），fail closed")
        if receipt["manifest_sha256"] != _sha256(p + ".manifest.json"):
            _fail(f"{arm}: receipt↔manifest SHA 不一致（发布后篡改），"
                  f"fail closed")
        # ④ 单 cell 假设 fail-closed（041 建议 4：多 cell 不许静默取首键）
        if len(d["n"]) != 1:
            _fail(f"{arm}: result JSON 含 {len(d['n'])} 个 cell "
                  f"（{sorted(d['n'])}）——本汇总只支持单 L 档单 cell，"
                  f"多 cell 输入 fail closed（须逐 L 档分别收口）")
        key = next(iter(d["n"]))
        # ④ 扩展（046①）：cell key 在 result n/scores、manifest cells、
        #    receipt cells 四处一致；incomplete_cells 必须为空
        if len(d["scores"]) != 1 or next(iter(d["scores"])) != key:
            _fail(f"{arm}: result scores 的 cell 集与 n 不一致"
                  f"（n={sorted(d['n'])}, scores={sorted(d['scores'])}），"
                  f"fail closed")
        if set(manifest["cells"]) != {key} or set(receipt["cells"]) != {key}:
            _fail(f"{arm}: cell 集在 manifest/receipt 与 result 之间不一致"
                  f"（result={key}, manifest={sorted(manifest['cells'])}, "
                  f"receipt={sorted(receipt['cells'])}），fail closed")
        if d.get("incomplete_cells"):
            _fail(f"{arm}: result incomplete_cells 非空: "
                  f"{d['incomplete_cells']}——收口拒绝不完整 cell，"
                  f"fail closed")
        # ⑤（046①）：task 集闭包——result n/scores、manifest tasks、
        #    receipt cells[key].tasks、scorer manifest 四处完全相等
        manifest_tasks = manifest["tasks"]
        receipt_tasks = receipt["cells"][key]["tasks"]
        cell_tasks_n = d["n"][key]
        cell_tasks_scores = d["scores"][key]
        if not (set(cell_tasks_n) == set(cell_tasks_scores)
                == set(manifest_tasks) == set(receipt_tasks)):
            _fail(f"{arm}: task 集闭包失败（result n={sorted(cell_tasks_n)},"
                  f" result scores={sorted(cell_tasks_scores)}, manifest="
                  f"{sorted(manifest_tasks)}, receipt cells="
                  f"{sorted(receipt_tasks)}）——result 与 manifest/scorer "
                  f"的 task 集必须完全相等，fail closed")
        # ② 基数闭包（045/046①）：n == len(ids) == len(lengths) ==
        #    len(answers_sha)、ID 键集闭合、n >= manifest min_samples
        min_samples = manifest["min_samples"]
        for t, n in cell_tasks_n.items():
            samples_seen.add(n)
            m = manifest_tasks[t]
            ids, lengths = m["ids"], m["lengths"]
            ans_sha = m["answers_sha"]
            if not (n == len(ids) == len(lengths) == len(ans_sha)):
                _fail(f"{arm}/{t}: 基数闭包失败：n={n} 但 len(ids)="
                      f"{len(ids)}, len(lengths)={len(lengths)}, "
                      f"len(answers_sha)={len(ans_sha)}——result n 必须"
                      f"与 manifest 的 ids/lengths/answers_sha 基数相等，"
                      f"fail closed")
            if set(ans_sha) != set(ids):
                _fail(f"{arm}/{t}: ID 键集不闭合：answers_sha 键集与 ids "
                      f"集不一致，fail closed")
            if len(ids) != len(set(ids)):
                _fail(f"{arm}/{t}: ids 存在重复，fail closed")
            if n < min_samples:
                _fail(f"{arm}/{t}: n={n} < manifest min_samples="
                      f"{min_samples}——正式入口 min-samples 硬门禁复核"
                      f"失败，fail closed")
        # task 数与 manifest expect_tasks 一致
        if len(manifest_tasks) != manifest["expect_tasks"]:
            _fail(f"{arm}: task 数 {len(manifest_tasks)} ≠ manifest "
                  f"expect_tasks={manifest['expect_tasks']}，fail closed")
        task_count = len(manifest_tasks)
        # ③ 双向 SHA 闭合（发布后无篡改）：receipt cells[].tasks[] 记录
        #    source_sha256（磁盘源文件）+ derived_sha256（补刻派生副本）；
        #    sources[].sha256 = derived（补刻后）口径
        Lname = key.split("/", 1)[0]
        pred_dir = os.path.join(
            pred_root, "fullkv" if arm == "FullKV" else arm,
            Lname, "pred" + receipt["inputs"]["pred_postfix"])
        gen_dir, legacy = _bind_generation(arm, p, receipt, results_dir)
        per_arm_protocol[arm] = {"publish_protocol":
                                 receipt.get("publish_protocol"),
                                 "legacy_protocol": legacy,
                                 "generation_dir_source":
                                 ("constructed" if legacy
                                  else "outputs.derived_dir")}
        derived_dir = os.path.join(
            gen_dir, "pred_root", Lname,
            "pred" + receipt["inputs"]["pred_postfix"])
        for task, tinfo in receipt["cells"][key]["tasks"].items():
            src_fp = os.path.join(pred_dir, tinfo["best_file"])
            if not os.path.exists(src_fp):
                _fail(f"{arm}/{task}: 源预测文件缺失: {src_fp}——"
                      f"（--pred-root 未指向该批数据的源目录？）fail closed")
            if tinfo["source_sha256"] != _sha256(src_fp):
                _fail(f"{arm}/{task}: 源 SHA 漂移（{src_fp} 磁盘实际 SHA 与"
                      f" receipt 记录不一致——发布后被篡改），fail closed")
            der_fp = os.path.join(derived_dir, tinfo["best_file"])
            if not os.path.exists(der_fp):
                _fail(f"{arm}/{task}: 派生副本缺失: {der_fp}——generation "
                      f"派生目录不完整，fail closed")
            if tinfo["derived_sha256"] != _sha256(der_fp):
                _fail(f"{arm}/{task}: 派生 SHA 漂移（{der_fp} 与 receipt "
                      f"记录不一致——发布后被篡改），fail closed")
        # ⑤ 跨臂身份提取（formal manifest，已经 receipt manifest_sha 闭合）
        ident, treatment, non_wl = _arm_identity(manifest)
        # ⑧ 显式 arm 契约（046②）：treatment 逐字段校验
        expected = ARM_CONTRACT[arm]
        if treatment != expected:
            _fail(f"{arm}: arm 契约不匹配（046②）：treatment="
                  f"{treatment!r} 但该臂契约要求 {expected!r}——treatment "
                  f"与臂名称的绑定 fail closed（交换 treatment 冠名不可达）")
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
        sc_path = os.path.join(gen_dir, "scorer.manifest.json")
        if not os.path.isfile(sc_path):
            _fail(f"{arm}: generation 缺 scorer.manifest.json: {sc_path}"
                  f"——fail closed")
        sc = json.load(open(sc_path))
        if set(sc) != set(manifest_tasks):
            _fail(f"{arm}: scorer manifest 的 task 集与 manifest tasks 不"
                  f"一致（{sorted(sc)} vs {sorted(manifest_tasks)}）"
                  f"——task 集闭包失败，fail closed")
        per_arm_scorer_man[arm] = hashlib.sha256(
            _canon(sc).encode("utf-8")).hexdigest()
        per_arm[arm] = {
            "avg": round(sum(cell_tasks_scores.values())
                         / len(cell_tasks_scores), 2),
            "receipt_run_id": receipt.get("run_id"),
            "result_file": os.path.basename(p),
            "per_task": cell_tasks_scores,
        }
        # ⑫（049）：逐臂不可变输入引用——last-writer-wins 时输入代际可审计
        per_arm_inputs[arm] = {
            "result_file": os.path.basename(p),
            "result_sha256": receipt["result_sha256"],
            "manifest_sha256": receipt["manifest_sha256"],
            "receipt_sha256": _sha256(p + ".receipt.json"),
            "receipt_run_id": receipt.get("run_id"),
        }

    # ---- ⑥ 跨臂身份门禁主体 ----
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
    # ---- ⑩ 评分口径公平门禁（047）：三臂 formal/scorer 脚本 SHA 必须
    #      完全一致——数据身份相同不足以证明评分公平，实现差异可改变
    #      得分与排名；不一致时须以同一 scorer 重评三臂后才可收口 ----
    script_shas = {a: tuple(sorted(s.items()))
                   for a, s in per_arm_scripts.items()}
    if len(set(script_shas.values())) > 1:
        _fail(f"跨臂评分口径不一致（047）：三臂 formal/scorer 脚本 SHA "
              f"不全一致（{ {a: s['scorer_sha256'][:12] for a, s in per_arm_scripts.items()} }）——"
              f"不同评分实现产生的数值不可列为公平排名；须提供等价迁移"
              f"证明或以同一 scorer 重评三臂，fail closed")

    # ---- 全部门禁通过才写 summary（任何不一致 → 非零退出，
    #      不覆盖旧 summary）----
    fullkv = per_arm["FullKV"]["avg"]
    for arm, st in per_arm.items():
        st["delta_vs_fullkv"] = round(st["avg"] - fullkv, 2)
        st["ruler32_official"] = RULER32[arm]
    # ⑪ 结论动态生成（046④）：ranking/delta/conclusion 全部从结构化
    #    数值生成——本脚本不再硬编码任何 64K 结果数字
    ranked = sorted(per_arm.items(), key=lambda kv: -kv[1]["avg"])
    order64 = [a for a, _ in ranked]
    order32 = [a for a, _ in sorted(RULER32.items(), key=lambda kv: -kv[1])]
    champ, last = ranked[0][0], ranked[-1][0]
    d_champ = ranked[0][1]["delta_vs_fullkv"]
    d_last = ranked[-1][1]["delta_vs_fullkv"]
    ranking_txt = " > ".join(f"{a} {s['avg']}" for a, s in ranked)
    r32_txt = " > ".join(f"{a} {v}" for a, v in
                         sorted(RULER32.items(), key=lambda kv: -kv[1]))
    same_dir = order64 == order32
    all_legacy = all(v["legacy_protocol"] for v in per_arm_protocol.values())
    protocol_txt = (
        "三臂 receipt 均为 legacy 发布协议（生成早于 E116f/E116g 发布锁），"
        "本 summary 只声明：数据身份门禁（task/ID/答案/源数据 SHA/评分"
        "口径闭包）已对这些文件执行并通过；不声称发布锁协议作用于该批"
        "数据的生成" if all_legacy else
        "各臂发布协议见 identity_gate.per_arm_receipt_protocol")
    out = {
        "experiment": "E119_ruler64k_formal_closure",
        "date": "2026-10-09",
        "entry": "benchmark/RULER/score_ruler_formal.py（E116e 正式入口：min-samples 硬门禁 + "
                 "staging 原子发布 + 身份扩展 receipt/manifest）",
        "identity": {
            "length_tier": "L65536（YaRN factor 2.0，64K 全档）",
            "model": "Qwen3-8B",
            # ⑪：元信息从数据推导（单值保持原 int 口径，多值如实列出）
            "samples_per_task": (sorted(samples_seen)[0]
                                 if len(samples_seen) == 1
                                 else sorted(samples_seen)),
            "tasks": task_count,
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
            "arm_contract": {a: dict(c) for a, c in ARM_CONTRACT.items()},
            "per_arm_treatment": per_arm_treatment,
            "scorer_manifest_digest": per_arm_scorer_man[sc_ref],
            "per_arm_script_sha256": per_arm_scripts,
            # ⑩（047）：评分口径公平门禁——三臂脚本 SHA 不一致即失败，
            # 不再是 warn-only；能到达这里 = 三臂完全一致
            "script_sha_policy": (
                "fail-closed：三臂 formal/scorer 脚本 SHA 必须完全一致才"
                "产出排名（评分实现差异可改变得分与排名；旧产物因脚本"
                "修复产生 SHA 差异时，须等价迁移证明或同一 scorer 重评）"),
            "script_sha_identical": True,
            # ⑨（046③）：逐臂发布协议绑定与 legacy 标注
            "per_arm_receipt_protocol": per_arm_protocol,
            "protocol_note": (
                "legacy_protocol=true 表示该臂产物生成于旧发布协议"
                "（早于 E116f generation 单指针与 E116g 并发发布锁）；"
                "本 summary 声明数据身份门禁已执行并通过，不声称发布"
                "锁协议作用于该批数据的生成"),
            "all_arms_data_identity_identical": True,
        },
        "closure": {
            "three_arm_receipts_success": True,
            "receipt_result_sha_matches": True,
            "receipt_manifest_sha_matches": True,
            "n_ids_lengths_answers_sha_closure": True,
            "task_set_closure": True,
            "min_samples_enforced": True,
            "arm_contract_enforced": True,
            "publish_protocol_bound": True,
            "scorer_script_sha_identical": True,
            "single_cell_enforced": True,
            "source_sha_stable": True,
            "derived_sha_stable": True,
            "crossarm_identity_gate": True,
        },
        "arms": {
            arm: {
                **st,
                "legacy_protocol": per_arm_protocol[arm]["legacy_protocol"],
            }
            for arm, st in per_arm.items()
        },
        "verdict": {
            "ruler64k_ranking": ranking_txt,
            "conclusion": (
                f"64K 正式口径下 {champ}（vs FullKV {d_champ:+.2f}）居首、"
                f"{last}（{d_last:+.2f}）居末，完整排序 {ranking_txt}；"
                f"与 32K 正式排序（{r32_txt}）方向"
                f"{'一致' if same_dir else '不一致'}——RULER 32K/64K 两档 "
                f"{champ} 冠军{'稳定' if same_dir else '不稳定'}，128K 待"
                f"全齐后同口径收口（{protocol_txt}）。"),
        },
        # ⑫（049）：三臂不可变输入引用（已过 receipt↔文件 SHA 闭合门禁）
        "inputs": per_arm_inputs,
    }
    p_out = os.path.join(results_dir, "e119_ruler64k_formal_summary.json")
    # ⑫（049）：原子发布——写中断/进程被杀/磁盘满只毁临时文件，
    # 上一份 last-known-good summary 字节不变
    _publish_summary(p_out, out)
    print(json.dumps(out["arms"], ensure_ascii=False, indent=2))
    print("identity_gate:", json.dumps(
        {k: out["identity_gate"][k] for k in
         ("protocol", "common_identity_digest", "treatment_whitelist",
          "script_sha_identical", "all_arms_data_identity_identical")},
        ensure_ascii=False))
    print("verdict:", json.dumps(out["verdict"], ensure_ascii=False))
    print(f"saved: {p_out}")


if __name__ == "__main__":
    main()
