#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E116g 红绿测试（GPT 1326 审计 TL-RULER-CROSSARM-IDENTITY-041）。

041：E119 汇总逐臂验 SHA/n 但不比较三臂样本 ID、答案、长度和源数据
身份——三份各自「内部闭合」但样本不同的结果仍会被输出为公平排名。

测试策略：把已提交的 64K 三臂真实产物（result/manifest/receipt +
generation 派生目录）复制到临时目录，在副本上做「单臂篡改 + 同步修补
receipt 闭合」模拟审计场景（每臂内部闭合但跨臂身份错配）：

  P1 正例    三臂真实产物原样副本 → 汇总通过，summary 生成且含
             identity_gate 段，数值 49.42/48.54/47.51 逐位不变；
  N1 _id     篡改单臂 manifest 一个 task 的 ids[0] → 拒绝（exit≠0）；
  N2 answers 篡改单臂 manifest 一个 answers_sha 值 → 拒绝；
  N3 length  篡改单臂 manifest 一个 lengths[k] → 拒绝；
  N4 源数据  篡改单臂 manifest source_data_sha256 一个 SHA → 拒绝；
  N5 模型    篡改单臂 manifest run_identity.model_path（+receipt 同步）
             → 拒绝；
  N6 多 cell 单臂 result JSON 塞入第二个 cell key（+receipt result
             SHA 同步）→ fail-closed 拒绝（next(iter) 静默取首键修复）；
  W1 脚本SHA 篡改单臂 manifest formal_script_sha256 → 仅 warn 不拒
             （评分口径非数据身份），summary 记录三臂各自 SHA。

所有负例还断言「不覆盖旧 summary」：失败运行前放置哨兵 summary 文件，
失败后内容逐位不变。生产目录全程零写入（--results-dir 指向临时副本；
③ 的源/派生磁盘 SHA 复核只读生产 pred 目录）。

用法：
  python3 exp/trace/test_e119_crossarm_identity.py
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
RESULTS = os.path.join(REPO, "exp", "trace", "results")
SCRIPT = os.path.join(REPO, "exp", "trace",
                      "analyze_e119_ruler64k_formal.py")
ARMS = {"mavg": "e119_ruler64k_formal_mavg.json",
        "FullKV": "e119_ruler64k_formal_fullkv.json",
        "aavg": "e119_ruler64k_formal_aavg.json"}
SUMMARY = "e119_ruler64k_formal_summary.json"
SENTINEL = {"sentinel": "old-summary-must-not-be-overwritten"}

PASS = 0


def _sha(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_fixture(base):
    """三臂真实产物完整副本（含 generation 派生目录）。"""
    for fn in ARMS.values():
        for p in [os.path.join(RESULTS, fn),
                  os.path.join(RESULTS, fn + ".manifest.json"),
                  os.path.join(RESULTS, fn + ".receipt.json")]:
            shutil.copyfile(p, os.path.join(base, os.path.basename(p)))
        for run_dir in glob.glob(os.path.join(RESULTS, fn + ".run-*")):
            shutil.copytree(run_dir, os.path.join(
                base, os.path.basename(run_dir)))


def _run(base):
    """对副本目录跑汇总脚本（--results-dir 指向副本）。"""
    return subprocess.run(
        [sys.executable, SCRIPT, "--results-dir", base],
        capture_output=True, text=True, cwd=REPO)


def _tamper_manifest(base, arm, mutate, also_receipt_identity=None):
    """单臂篡改 + 同步修补 receipt 闭合（模拟「内部闭合但身份错配」）。

    mutate(manifest) 就地修改 manifest；随后重写 receipt.manifest_sha256
    使 receipt↔manifest SHA 闭合——确保拒绝来自跨臂身份门禁而非
    receipt 闭合检查。also_receipt_identity(receipt) 可同步篡改
    receipt.inputs.run_identity。"""
    mp = os.path.join(base, ARMS[arm] + ".manifest.json")
    rp = os.path.join(base, ARMS[arm] + ".receipt.json")
    m = json.load(open(mp))
    mutate(m)
    json.dump(m, open(mp, "w"), indent=1, ensure_ascii=False)
    r = json.load(open(rp))
    r["manifest_sha256"] = _sha(mp)
    if also_receipt_identity is not None:
        also_receipt_identity(r)
    json.dump(r, open(rp, "w"), indent=1, ensure_ascii=False)


def _place_sentinel_summary(base):
    p = os.path.join(base, SUMMARY)
    json.dump(SENTINEL, open(p, "w"))
    return p


def _expect_reject(base, tag):
    """负例断言：exit≠0 + 错误信息含 041 门禁标记 + 旧 summary 未被覆盖。"""
    sp = _place_sentinel_summary(base)
    r = _run(base)
    assert r.returncode != 0, \
        f"{tag}: 篡改后仍 exit=0（跨臂身份门禁失效）：\n{r.stdout[-800:]}"
    assert ("E119-GATE-FAIL" in r.stderr) or ("跨臂" in r.stderr) or \
           ("cell" in r.stderr), (tag, r.stderr[-800:])
    cur = json.load(open(sp))
    assert cur == SENTINEL, f"{tag}: 失败运行覆盖了旧 summary"
    print(f"  {tag} PASS  exit={r.returncode}，旧 summary 未覆盖；"
          f"拒绝原因: {r.stderr.strip().splitlines()[-1][:110]}")
    return True


def test_P1_positive(base):
    """P1：三臂真实产物原样副本 → 通过 + identity_gate 段 + 数值不变。"""
    r = _run(base)
    assert r.returncode == 0, (r.returncode, r.stdout[-1500:],
                               r.stderr[-1000:])
    s = json.load(open(os.path.join(base, SUMMARY)))
    assert {a: v["avg"] for a, v in s["arms"].items()} == \
        {"mavg": 49.42, "FullKV": 48.54, "aavg": 47.51}, s["arms"]
    ig = s["identity_gate"]
    assert ig["protocol"] == "e116g-crossarm-identity-v1"
    assert ig["all_arms_data_identity_identical"] is True
    assert len(ig["common_identity_digest"]) == 64
    assert ig["treatment_whitelist"] == ["alpha", "beta", "far_method",
                                         "gamma", "method", "near_method"]
    assert ig["per_arm_treatment"]["FullKV"] == {"method": "none"}
    assert ig["per_arm_treatment"]["mavg"]["gamma"] == "0.625"
    assert s["closure"]["crossarm_identity_gate"] is True
    print("P1 PASS  三臂真实产物副本 → 汇总通过；identity_gate 段生成；"
          "avg 49.42/48.54/47.51 逐位不变；treatment 白名单逐臂记录")


def test_N1_tamper_id(base):
    """N1（041 建议 4）：单臂一个 _id 篡改 → 拒绝发布。"""
    _tamper_manifest(base, "aavg", lambda m: m["tasks"]
                     ["niah_single_1"]["ids"].__setitem__(0, "tampered:999"))
    _expect_reject(base, "N1(_id)")


def test_N2_tamper_answers_sha(base):
    """N2：单臂一个 answers_sha 篡改 → 拒绝发布。"""
    def mut(m):
        m["tasks"]["niah_single_1"]["answers_sha"]["niah_single_1:0"] = \
            "deadbeefdeadbeef"
    _tamper_manifest(base, "FullKV", mut)
    _expect_reject(base, "N2(answers_sha)")


def test_N3_tamper_length(base):
    """N3：单臂一个逐行 length 篡改 → 拒绝发布。"""
    def mut(m):
        m["tasks"]["vt"]["lengths"][0] = m["tasks"]["vt"]["lengths"][0] + 1
    _tamper_manifest(base, "mavg", mut)
    _expect_reject(base, "N3(lengths)")


def test_N4_tamper_source_data_sha(base):
    """N4：单臂 source_data_sha256 一个 SHA 篡改 → 拒绝发布。"""
    def mut(m):
        key = sorted(m["source_data_sha256"])[0]
        m["source_data_sha256"][key]["sha256"] = "0" * 64
    _tamper_manifest(base, "aavg", mut)
    _expect_reject(base, "N4(source_data_sha256)")


def test_N5_tamper_model_identity(base):
    """N5：单臂模型身份篡改（manifest + receipt 同步）→ 拒绝发布。"""
    def mut(m):
        m["run_identity"]["model_path"] = "/home/other/Qwen3-8B-tampered"
    def mut_rc(r):
        r["inputs"]["run_identity"]["model_path"] = \
            "/home/other/Qwen3-8B-tampered"
    _tamper_manifest(base, "mavg", mut, also_receipt_identity=mut_rc)
    _expect_reject(base, "N5(model_path)")


def test_N6_multi_cell(base):
    """N6（041 建议 4）：单臂 result JSON 多 cell → fail-closed 拒绝
    （修 next(iter(d["n"])) 静默取首键假设 bug）。"""
    jp = os.path.join(base, ARMS["FullKV"])
    rp = os.path.join(base, ARMS["FullKV"] + ".receipt.json")
    d = json.load(open(jp))
    key = next(iter(d["n"]))
    fake = key + "_SECOND_L"
    d["n"][fake] = dict(d["n"][key])
    d["scores"][fake] = dict(d["scores"][key])
    json.dump(d, open(jp, "w"), indent=1, ensure_ascii=False)
    r = json.load(open(rp))
    r["result_sha256"] = _sha(jp)
    json.dump(r, open(rp, "w"), indent=1, ensure_ascii=False)
    _expect_reject(base, "N6(multi-cell)")


def test_W1_script_sha_warn(base):
    """W1（041 建议 1 的脚本 SHA 语义）：单臂 formal_script_sha256 篡改
    → 仅 warn 不硬拒（评分口径非数据身份），summary 记录三臂各自 SHA。"""
    _tamper_manifest(base, "aavg", lambda m: m["run_identity"]
                     .__setitem__("formal_script_sha256", "f" * 64))
    r = _run(base)
    assert r.returncode == 0, (r.returncode, r.stdout[-1000:],
                               r.stderr[-800:])
    assert "E119-WARN" in r.stderr, r.stderr[-800:]
    s = json.load(open(os.path.join(base, SUMMARY)))
    ig = s["identity_gate"]
    assert ig["script_sha_warn"] is True
    shas = {a: v["formal_script_sha256"]
            for a, v in ig["per_arm_script_sha256"].items()}
    assert shas["aavg"] == "f" * 64 and shas["mavg"] != "f" * 64
    assert ig["all_arms_data_identity_identical"] is True
    print("W1 PASS  脚本 SHA 差异仅 warn（stderr E119-WARN）不硬拒；"
          "summary 记录三臂各自 formal/scorer SHA；数据身份门禁仍全过")


def main():
    global PASS
    root_tmp = tempfile.mkdtemp(prefix="e119_ident_")
    try:
        # 每个用例独立新鲜副本：篡改互不累积，拒绝原因可精确归因到
        # 该用例注入的单字段错配（共享副本会让上一轮篡改先触发门禁）
        cases = [
            ("P1", test_P1_positive),
            ("N1", test_N1_tamper_id),
            ("N2", test_N2_tamper_answers_sha),
            ("N3", test_N3_tamper_length),
            ("N4", test_N4_tamper_source_data_sha),
            ("N5", test_N5_tamper_model_identity),
            ("N6", test_N6_multi_cell),
            ("W1", test_W1_script_sha_warn),
        ]
        for tag, fn in cases:
            base = os.path.join(root_tmp, f"fx_{tag}")
            os.makedirs(base)
            _copy_fixture(base)
            fn(base)
            PASS += 1
    finally:
        shutil.rmtree(root_tmp, ignore_errors=True)
    print(f"\nE119 crossarm identity ALL PASS ({PASS}/8)")


if __name__ == "__main__":
    main()
