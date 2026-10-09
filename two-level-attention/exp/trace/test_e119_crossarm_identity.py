#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E119 跨臂身份门禁红绿测试（041 + E116h 045/046/047/048①）。

048① 可移植性：本测试不再依赖仓库外的生产 prediction JSONL——改用已入库
的全合成最小 fixture（exp/trace/testdata/e119_min/，由
gen_e119_min_fixture.py 生成：4 任务 × 3 行/任务、三臂 treatment 与
ARM_CONTRACT 逐字段一致、全部 SHA 闭合、legacy receipt），干净检出即可
运行。analyzer 通过 --results-dir/--pred-root 只读指向临时副本，fixture
与生产数据零写入。

档位参数（E116i）：第一个可选参数选择 64k（默认，入库 fixture + 64K
analyzer）或 128k（128K analyzer + 现场以 gen_e119_min_fixture.py
--tier 128k 重建的最小 fixture——128K 批次 ARM_CONTRACT 与 64K 不同，
TLI 臂 treatment 仅落盘 method）。全矩阵对两档各跑一遍。

用例矩阵：

  P1 正例      fixture 三臂原样 → 汇总通过，arms 数值 40.0/35.0/30.0，
               identity_gate 段 + arm_contract + 逐臂 legacy_protocol 标注，
               结论动态生成（mavg 冠军 +5.00 / aavg 居末 -5.00）；
  N1 _id       篡改单臂 manifest 一个 task 的 ids[0] → 拒绝（exit≠0）；
  N2 answers   篡改单臂 manifest 一个 answers_sha 值 → 拒绝；
  N3 length    篡改单臂 manifest 一个 lengths[k] → 拒绝；
  N4 源数据    篡改单臂 manifest source_data_sha256 一个 SHA → 拒绝；
  N5 模型      篡改单臂 manifest run_identity.model_path（+receipt 同步）
               → 拒绝；
  N6 多 cell   单臂 result JSON 塞入第二个 cell key（+receipt result
               SHA 同步）→ fail-closed 拒绝（next(iter) 静默取首键修复）；
  N7 treatment 交换 mavg/aavg 的 treatment（046②：把 aavg 配置冠到 mavg
               臂名下，receipt 同步闭合）→ arm 契约 fail-closed 拒绝；
  N8 脚本SHA   篡改单臂 formal_script_sha256（047：评分口径公平）→
               fail-closed 拒绝（不再是 warn-only）；
  P2 结论动态  篡改 aavg result 分数使其反超（+receipt SHA 同步）→
               通过，但 verdict.ruler{tier}_ranking/conclusion 必须跟随新
               排序（aavg 冠军 +10.00），不得残留硬编码 mavg 冠军文本
               （046④）；
  O1 python -O 两类篡改（receipt SHA 不一致 / 跨臂 answers_sha）在
               python -O 下重跑 → 仍非零退出且不覆盖旧 summary
               （045：门禁全部为显式条件 + _fail，assert 零依赖）。

E116h 049（GPT 1531 审计 TL-E119-SUMMARY-ATOMICITY-049）summary 原子发布：
  N9a 写中断   注入 json.dump 写出 '{"partial":' 前缀后抛 OSError →
               非零退出、旧 summary SHA 逐位不变且仍可解析、无 .tmp
               残留（修复前 open(p_out,"w") 直接截断毁掉 last-known-good）；
  N9b 并发发布 双汇总器（一慢速写、一正常）并发写同一 summary →
               两进程均成功，最终文件为某一完整代际（与单跑逐字节
               一致，last-writer-wins），无混写/截断/临时残留；
  O2 python -O N9a 同款写中断在 python -O 下重跑 → 仍非零退出且旧
               summary 逐位不变（原子发布不依赖 assert）。

E116i 046③ 两残留（GPT 1633 审计 TL-E119-CONSUMER-BINDING-046③：
v2 必需角色未强制 + 读值与核验哈希未绑定同一快照）：
  P3 v2 正例   三臂 receipt 升级为 e116f-generation-v2（单指针 + 四规范
               文件）→ 汇总通过且数值不变、逐臂 publish_protocol/
               derived_dir 标注正确、legacy_protocol=false；
  N10 缺角色   v2 receipt 的 generation_files 被替换为单角色映射
               {"scorer": ...}（审计原始反例，修复前 exit=0 可绕过）或
               仅缺一个角色（md）→ 「缺必需角色」fail-closed 拒收，
               子例 b 在 python -O 下重跑；
  N11 读交换   消费者取到旧 result bytes 后、读 receipt 前，另一发布者
               完整安装新代际（分数 80.0、receipt/manifest 同步闭合、
               四 aliases 原子替换）→ bytes 快照绑定必须拒收（修复前
               exit=0 且 summary 记旧均值 40 配新 result SHA——旧值配
               新哈希）；post_swap 控制组（完整切换先于消费）必须以
               新值+新哈希一致通过（拒收只针对交错，不拒绝合法更新）。

E116j 052（GPT 2026-10-10 0125 审计 TL-E119-GEN-BINDING-052：
generation md/receipt 内容未哈希绑定——v2 对 md/receipt 只做存在性
检查，注释却宣称「四规范文件 SHA 逐位闭合」）：
  P4 v3 正例   三臂 receipt 升级为 e116i-generation-v3（entry receipt
               绑 gen_md_sha256/gen_receipt_sha256）→ 汇总通过且数值
               不变，closure.gen_content_bound 四角色全 true；
  N12 v3 md    v3 篡改 generation 内 result.md 内容 → fail-closed
               拒收（052 内容绑定），旧 summary 逐位不变（修复前
               exit=0 假闭合）；子例 b python -O 重跑；
  N13 v3 rcpt  v3 篡改 generation 内 receipt.json 内容 → fail-closed
               拒收，旧 summary 逐位不变；子例 b python -O 重跑；
  P5 v2 部分   v2（当前生产协议）篡改 generation 内 result.md /
               receipt.json → 汇总通过（v2 存在性检查确实抓不到，
               属协议能力边界），但 summary 必须如实标注
               closure.gen_content_bound 的 md/receipt=false——不得
               冒充四角色全绑定（修复前注释过度宣称）。

所有负例还断言「不覆盖旧 summary」：失败运行前放置哨兵 summary 文件，
失败后内容逐位不变。

用法：
  python3 exp/trace/test_e119_crossarm_identity.py [64k|128k]
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

# 049 故障注入 wrapper：importlib 加载 analyzer 模块，按模式 monkeypatch
# json.dump（analyzer 唯一的 json.dump 调用点 = summary 原子发布）。
#   interrupt —— 写出 '{"partial":' 前缀后抛 OSError（049 审计原始反例）
#   slow      —— 持锁慢速写（time.sleep 后真写），驱动并发发布窗口
_WRAPPER = '''
import json, os, sys, importlib.util
MODE, SCRIPT = sys.argv[1], sys.argv[2]
args = sys.argv[3:]
spec = importlib.util.spec_from_file_location("e119_analyzer", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
real_dump = json.dump
if MODE == "interrupt":
    def boom(obj, fh, *a, **k):
        fh.write('{"partial":')
        fh.flush()
        raise OSError("injected-write-interruption")
    json.dump = boom
elif MODE == "slow":
    def slow(obj, fh, *a, **k):
        import time
        time.sleep(float(os.environ.get("E119_SLOW_DUMP", "3")))
        real_dump(obj, fh, *a, **k)
    json.dump = slow
sys.argv = [SCRIPT] + args
mod.main()
'''

# E116i 046③ 残留 B 交错注入 wrapper：importlib 加载 analyzer，按模式
# 在消费窗口内模拟「另一发布者完整安装新代际」。
#   read_swap  —— hook analyzer 的 _read_bytes：消费者取到旧 result
#                 bytes 后、后续读取前，完整替换 mavg 臂代际（四
#                 aliases 原子替换 + gen-next 内分数 80.0/SHA 同步闭合）
#   post_swap  —— 开跑前先完整切换（控制组：合法代际更新必须通过）
_SWAP_WRAPPER = '''
import hashlib, importlib.util, json, os, shutil, sys
MODE, SCRIPT, RESULT, MARK = sys.argv[1:5]
args = sys.argv[5:]
spec = importlib.util.spec_from_file_location("e119_analyzer", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()

def save(p, obj):
    with open(p, "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)

def do_swap():
    """另一发布者完整安装新代际：分数全部 80.0、manifest/receipt 同步
    闭合、四 aliases 原子替换（GPT 1633 审计 read_swap 场景）。"""
    rp = RESULT + ".receipt.json"
    r = json.load(open(rp))
    d = json.load(open(RESULT))
    key = next(iter(d["scores"]))
    for t in d["scores"][key]:
        d["scores"][key][t] = 80.0
    oldg = r["outputs"]["derived_dir"]
    newg = oldg + "-next"
    if os.path.isdir(newg):
        shutil.rmtree(newg)
    shutil.copytree(oldg, newg)
    r["run_id"] = r["run_id"] + "-next"
    r["outputs"]["derived_dir"] = newg
    r["avg"][key] = 80.0
    save(newg + "/result.json", d)
    r["result_sha256"] = sha(newg + "/result.json")
    m = json.load(open(newg + "/manifest.json"))
    m["run_id"] = r["run_id"]
    save(newg + "/manifest.json", m)
    r["manifest_sha256"] = sha(newg + "/manifest.json")
    save(newg + "/receipt.json", r)
    for src, dst in [(newg + "/result.json", RESULT),
                     (newg + "/result.md",
                      RESULT[:-len(".json")] + ".md"),
                     (newg + "/manifest.json", RESULT + ".manifest.json"),
                     (newg + "/receipt.json", rp)]:
        shutil.copyfile(src, dst + ".next")
        os.replace(dst + ".next", dst)
    with open(MARK, "w") as f:
        f.write("swapped")

if MODE == "post_swap":
    do_swap()
    sys.argv = [SCRIPT] + args
    mod.main()
elif MODE == "read_swap":
    sys.argv = [SCRIPT] + args
    real_read = mod._read_bytes
    fired = []
    def hooked(path):
        raw = real_read(path)
        if os.path.abspath(path) == os.path.abspath(RESULT) and not fired:
            fired.append(1)
            do_swap()  # 消费者已取旧 result bytes，此刻完整替换代际
        return raw
    mod._read_bytes = hooked
    mod.main()
'''

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
FIXTURE = os.path.join(REPO, "exp", "trace", "testdata", "e119_min")
GEN_SCRIPT = os.path.join(FIXTURE, "gen_e119_min_fixture.py")
# E116i：档位参数——64k（默认）用入库 fixture + 64K analyzer；128k 用
# 现场重建的 128K 档最小 fixture + 128K analyzer（其 ARM_CONTRACT 与
# 64K 不同：TLI 臂 treatment 仅落盘 method=tli_64_128_1024_c4_A）
TIER = sys.argv[1].lower() if len(sys.argv) > 1 else "64k"
if TIER not in ("64k", "128k"):
    raise SystemExit(f"用法: {sys.argv[0]} [64k|128k]（未知档位 {TIER!r}）")
SCRIPT = os.path.join(
    REPO, "exp", "trace", f"analyze_e119_ruler{TIER}_formal.py")
SUMMARY = f"e119_ruler{TIER}_formal_summary.json"
RESULT_PREFIX = f"e119_ruler{TIER}_formal"
LNAME = "L65536" if TIER == "64k" else "L131072"
# fixture 合成分数（与生产排序方向一致：mavg > FullKV > aavg）
FIXTURE_AVG = {"mavg": 40.0, "FullKV": 35.0, "aavg": 30.0}
SENTINEL = {"sentinel": "old-summary-must-not-be-overwritten"}

PASS = 0


def _sha(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fixture_usable():
    """fixture 完整性探测：receipt 三件套 + 源预测 jsonl 均在位才可用
    （防止 .gitignore 漏白名单或异常检出导致半套 fixture 误判可用）。
    E116i：64k 档才允许用入库 fixture；128k 档总是现场重建。"""
    if TIER != "64k":
        return False
    probes = [
        os.path.join(FIXTURE, "results",
                     f"{RESULT_PREFIX}_mavg.json.receipt.json"),
        os.path.join(FIXTURE, "pred_root", "mavg", LNAME, "pred_1024"),
        os.path.join(FIXTURE, "pred_root", "fullkv", LNAME, "pred_1024"),
        os.path.join(FIXTURE, "pred_root", "aavg", LNAME, "pred_1024"),
    ]
    for p in probes:
        if not (os.path.isfile(p) or os.path.isdir(p)):
            return False
    return len(glob.glob(os.path.join(
        probes[1], "*.jsonl"))) == 4


def _ensure_fixture(root_tmp):
    """048①：fixture 源目录——已入库且完整则直接用；缺失（异常检出/
    漏白名单）或 128k 档则用已入库生成器现场重建到临时目录，保证
    干净路径可跑。"""
    if _fixture_usable():
        return FIXTURE
    dest = os.path.join(root_tmp, f"fixture_regen_{TIER}")
    os.makedirs(dest, exist_ok=True)
    r = subprocess.run([sys.executable, GEN_SCRIPT, dest,
                        "--tier", TIER],
                       capture_output=True, text=True)
    assert r.returncode == 0, (r.returncode, r.stdout[-500:], r.stderr[-500:])
    return dest


def _copy_fixture(base, fixture):
    """三臂产物完整副本（results 三件套 + generation 派生目录 + 源预测）。"""
    res = os.path.join(base, "results")
    shutil.copytree(os.path.join(fixture, "results"), res)
    shutil.copytree(os.path.join(fixture, "pred_root"),
                    os.path.join(base, "pred_root"))


def _run(base, opt_o=False):
    """对副本目录跑汇总脚本（--results-dir/--pred-root 只读指向副本）。"""
    cmd = [sys.executable] + (["-O"] if opt_o else []) + [
        SCRIPT, "--results-dir", os.path.join(base, "results"),
        "--pred-root", os.path.join(base, "pred_root")]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)


def _manifest_path(base, arm):
    return os.path.join(base, "results",
                        f"{RESULT_PREFIX}_{arm.lower()}.json"
                        ".manifest.json")


def _receipt_path(base, arm):
    return _manifest_path(base, arm).replace(".manifest.json",
                                             ".receipt.json")


def _result_path(base, arm):
    return os.path.join(base, "results",
                        f"{RESULT_PREFIX}_{arm.lower()}.json")


def _tamper_manifest(base, arm, mutate, also_receipt_identity=None):
    """单臂篡改 + 同步修补 receipt 闭合（模拟「内部闭合但身份错配」）。

    mutate(manifest) 就地修改 manifest；随后重写 receipt.manifest_sha256
    使 receipt↔manifest SHA 闭合——确保拒绝来自跨臂身份/契约门禁而非
    receipt 闭合检查。also_receipt_identity(receipt) 可同步篡改
    receipt.inputs.run_identity。"""
    mp = _manifest_path(base, arm)
    rp = _receipt_path(base, arm)
    m = json.load(open(mp))
    mutate(m)
    json.dump(m, open(mp, "w"), indent=1, ensure_ascii=False)
    r = json.load(open(rp))
    r["manifest_sha256"] = _sha(mp)
    if also_receipt_identity is not None:
        also_receipt_identity(r)
    json.dump(r, open(rp, "w"), indent=1, ensure_ascii=False)


def _place_sentinel_summary(base):
    p = os.path.join(base, "results", SUMMARY)
    json.dump(SENTINEL, open(p, "w"))
    return p


def _expect_reject(base, tag, opt_o=False, needle=None):
    """负例断言：exit≠0 + 门禁标记 + 旧 summary 未被覆盖。"""
    sp = _place_sentinel_summary(base)
    r = _run(base, opt_o=opt_o)
    assert r.returncode != 0, \
        f"{tag}: 篡改后仍 exit=0（跨臂身份门禁失效）：\n{r.stdout[-800:]}"
    assert ("E119-GATE-FAIL" in r.stderr) or ("跨臂" in r.stderr) or \
           ("cell" in r.stderr), (tag, r.stderr[-800:])
    if needle is not None:
        assert needle in r.stderr, (tag, needle, r.stderr[-900:])
    cur = json.load(open(sp))
    assert cur == SENTINEL, f"{tag}: 失败运行覆盖了旧 summary"
    mode = "python -O " if opt_o else ""
    print(f"  {tag} PASS  {mode}exit={r.returncode}，旧 summary 未覆盖；"
          f"拒绝原因: {r.stderr.strip().splitlines()[-1][:110]}")
    return True


def test_P1_positive(base):
    """P1：fixture 三臂原样 → 通过 + identity_gate/arm 契约 + 数值不变。"""
    r = _run(base)
    assert r.returncode == 0, (r.returncode, r.stdout[-1500:],
                               r.stderr[-1000:])
    s = json.load(open(os.path.join(base, "results", SUMMARY)))
    assert {a: v["avg"] for a, v in s["arms"].items()} == FIXTURE_AVG, \
        s["arms"]
    assert {a: v["legacy_protocol"] for a, v in s["arms"].items()} == \
        {"mavg": True, "FullKV": True, "aavg": True}, s["arms"]
    ig = s["identity_gate"]
    assert ig["protocol"] == "e116g-crossarm-identity-v1"
    assert ig["all_arms_data_identity_identical"] is True
    assert len(ig["common_identity_digest"]) == 64
    assert ig["script_sha_identical"] is True
    # 046②：arm 契约逐字段入 summary，且 per_arm_treatment 与契约相等
    # （E116i：128K 批次 TLI 臂 treatment 仅落盘 method，契约随档位）
    if TIER == "64k":
        assert ig["arm_contract"]["mavg"] == {
            "far_method": "minmax", "near_method": "avg",
            "alpha": "0.25", "beta": "0.125", "gamma": "0.625"}
        assert ig["arm_contract"]["aavg"] == {
            "far_method": "avg", "near_method": "avg",
            "alpha": "0", "beta": "0", "gamma": "0"}
    else:
        assert ig["arm_contract"]["mavg"] == \
            {"method": "tli_64_128_1024_c4_A"}
        assert ig["arm_contract"]["aavg"] == \
            {"method": "tli_64_128_1024_c4_A"}
    assert ig["arm_contract"]["FullKV"] == {"method": "none"}
    for arm, treat in ig["per_arm_treatment"].items():
        assert treat == ig["arm_contract"][arm], (arm, treat)
    # 046③：legacy receipt 显式标注，不得声称发布锁协议已作用
    for arm, proto in ig["per_arm_receipt_protocol"].items():
        assert proto["legacy_protocol"] is True, (arm, proto)
        assert proto["publish_protocol"] is None, (arm, proto)
    assert "不声称" in ig["protocol_note"]
    # 046④：结论从结构化数值动态生成（fixture 口径 mavg +5.00 冠军）
    assert s["identity"]["samples_per_task"] == 3
    assert s["identity"]["tasks"] == 4
    assert s["verdict"][f"ruler{TIER}_ranking"] == \
        "mavg 40.0 > FullKV 35.0 > aavg 30.0"
    assert "mavg（vs FullKV +5.00）居首" in s["verdict"]["conclusion"]
    assert "aavg（-5.00）居末" in s["verdict"]["conclusion"]
    assert "方向一致" in s["verdict"]["conclusion"]
    # 052（E116j）：legacy 协议 generation 内容绑定如实标注——md/receipt
    # 在 legacy 下完全未检查，不得冒充四角色全绑定
    assert s["closure"]["gen_content_bound"] == \
        {"json": True, "manifest": True, "md": False, "receipt": False}, \
        s["closure"]
    print("P1 PASS  fixture 三臂 → 汇总通过；arms 40.0/35.0/30.0；"
          "arm 契约逐臂记录且与 treatment 相等；三臂 legacy_protocol "
          "如实标注；结论从结构化 ranking 动态生成")


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
    jp = _result_path(base, "FullKV")
    rp = _receipt_path(base, "FullKV")
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


def test_N7_treatment_swap(base):
    """N7（046②）：交换 treatment——把 aavg 配置（far=avg/α=β=γ=0）
    冠到 mavg 臂名下（manifest + receipt 同步闭合）→ arm 契约 fail-closed
    拒绝。修复前唯一通道是篡改 treatment，此门禁使其不可达。"""
    def mut(m):
        m["run_identity"]["extra_params"] = {
            "far_method": "avg", "near_method": "avg",
            "alpha": "0", "beta": "0", "gamma": "0"}
    def mut_rc(r):
        r["inputs"]["run_identity"]["extra_params"] = {
            "far_method": "avg", "near_method": "avg",
            "alpha": "0", "beta": "0", "gamma": "0"}
    _tamper_manifest(base, "mavg", mut, also_receipt_identity=mut_rc)
    _expect_reject(base, "N7(treatment-swap)", needle="046②")


def test_N8_script_sha_fail_closed(base):
    """N8（047）：单臂 formal_script_sha256 篡改（+receipt 同步闭合）→
    fail-closed 拒绝——评分口径公平从 warn-only 升格为硬门禁（数据身份
    相同不足以证明 A/B/C 评分公平）。"""
    # 注意：fixture 三臂 FORMAL_SHA 本身是 "f"*64，须篡成不同值才构成反例
    def mut(m):
        m["run_identity"]["formal_script_sha256"] = "a" * 64
    def mut_rc(r):
        r["inputs"]["run_identity"]["formal_script_sha256"] = "a" * 64
    _tamper_manifest(base, "aavg", mut, also_receipt_identity=mut_rc)
    _expect_reject(base, "N8(script-sha)", needle="047")


def test_P2_dynamic_conclusion(base):
    """P2（046④）：篡改 aavg 分数反超 mavg（+receipt result SHA 同步）→
    门禁全过（分数属臂私有、receipt 闭合），但 verdict 必须跟随新排序
    aavg 冠军 +10.00——验证结论从结构化数值动态生成，无硬编码残留。"""
    jp = _result_path(base, "aavg")
    rp = _receipt_path(base, "aavg")
    d = json.load(open(jp))
    key = next(iter(d["n"]))
    d["scores"][key] = {t: 45.0 for t in d["scores"][key]}
    json.dump(d, open(jp, "w"), indent=1, ensure_ascii=False)
    r = json.load(open(rp))
    r["result_sha256"] = _sha(jp)
    json.dump(r, open(rp, "w"), indent=1, ensure_ascii=False)
    rr = _run(base)
    assert rr.returncode == 0, (rr.returncode, rr.stdout[-1200:],
                                rr.stderr[-800:])
    s = json.load(open(os.path.join(base, "results", SUMMARY)))
    assert {a: v["avg"] for a, v in s["arms"].items()} == \
        {"aavg": 45.0, "mavg": 40.0, "FullKV": 35.0}, s["arms"]
    assert s["arms"]["aavg"]["delta_vs_fullkv"] == 10.0
    assert s["verdict"][f"ruler{TIER}_ranking"] == \
        "aavg 45.0 > mavg 40.0 > FullKV 35.0"
    assert "aavg（vs FullKV +10.00）居首" in s["verdict"]["conclusion"]
    assert "FullKV（+0.00）居末" in s["verdict"]["conclusion"]
    # 排序与 32K 方向相反 → 结论如实写「不一致」，不得仍称 mavg 冠军稳定
    assert "不一致" in s["verdict"]["conclusion"]
    assert "mavg 冠军" not in s["verdict"]["conclusion"]
    print("P2 PASS  分数改动 → verdict/conclusion 逐字跟随新排序"
          "（aavg 冠军 +10.00、32K 方向不一致如实标注），无硬编码残留")


def test_O1_python_opt(base):
    """O1（045）：python -O 双跑负例——两类篡改在优化模式下仍必须
    非零退出且不覆盖旧 summary（门禁全部为显式条件 + _fail，
    assert 零依赖）。"""
    # 子例 a：aavg result JSON 篡改但不修补 receipt → receipt SHA 不一致
    jp = _result_path(base, "aavg")
    d = json.load(open(jp))
    key = next(iter(d["n"]))
    d["scores"][key]["cwe"] = 13.97
    json.dump(d, open(jp, "w"), indent=1, ensure_ascii=False)
    _expect_reject(base, "O1a(python -O, receipt SHA)", opt_o=True)
    # 子例 b：跨臂 answers_sha 篡改 + receipt 闭合 → 跨臂身份门禁
    base_b = os.path.join(os.path.dirname(base), "fx_O1b")
    os.makedirs(base_b)
    _copy_fixture(base_b, _FIXTURE)
    def mut(m):
        m["tasks"]["cwe"]["answers_sha"]["cwe:0"] = "0123456789abcdef"
    _tamper_manifest(base_b, "aavg", mut)
    _expect_reject(base_b, "O1b(python -O, crossarm identity)", opt_o=True)


def _run_wrapper(base, mode, opt_o=False, extra_env=None, popen=False):
    """049 故障注入：经 wrapper 进程跑 analyzer（monkeypatch json.dump）。"""
    wpath = os.path.join(base, f"wrapper_{mode}.py")
    with open(wpath, "w", encoding="utf-8") as f:
        f.write(_WRAPPER)
    cmd = [sys.executable] + (["-O"] if opt_o else []) + [
        wpath, mode, SCRIPT,
        "--results-dir", os.path.join(base, "results"),
        "--pred-root", os.path.join(base, "pred_root")]
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    if popen:
        return subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True,
                                cwd=REPO, env=env)
    return subprocess.run(cmd, capture_output=True, text=True,
                           cwd=REPO, env=env)


def _assert_no_tmp(base):
    res = os.path.join(base, "results")
    left = [f for f in os.listdir(res) if ".tmp-" in f]
    assert not left, f"summary 临时文件残留: {left}"


def test_N9_summary_atomic(base):
    """N9（049）：summary 原子发布红绿——a 写中断毁不了旧 summary、
    b 并发双发布无混写/截断。"""
    sp = _place_sentinel_summary(base)
    old_sha = _sha(sp)
    # -- a：写中断（049 审计原始反例：写前缀后抛 OSError）--
    r = _run_wrapper(base, "interrupt")
    assert r.returncode != 0, ("N9a: 写中断后仍 exit=0", r.stdout[-500:])
    cur = json.load(open(sp))
    assert cur == SENTINEL, "N9a: 写中断毁掉了旧 summary"
    assert _sha(sp) == old_sha, "N9a: 旧 summary 字节被改动"
    _assert_no_tmp(base)
    print("  N9a(写中断) PASS  exit=%d，旧 summary SHA 逐位不变且可解析，"
          "无 .tmp 残留（049: last-known-good 不再被 open(w) 截断）"
          % r.returncode)
    # -- b：并发双发布（慢速持锁写 × 正常写）--
    # 先单跑一次取「完整代际」期望字节
    r_single = _run(base)
    assert r_single.returncode == 0, (r_single.returncode, r_single.stderr[-500:])
    expected = open(sp, "rb").read()
    procs = [
        _run_wrapper(base, "slow",
                     extra_env={"E119_SLOW_DUMP": "4"}, popen=True),
        _run_wrapper(base, "slow",
                     extra_env={"E119_SLOW_DUMP": "0.5"}, popen=True),
    ]
    rcs = [p.wait() for p in procs]
    assert rcs == [0, 0], ("N9b: 并发发布进程失败", rcs,
                          [p.stderr.read()[-300:] for p in procs])
    assert open(sp, "rb").read() == expected, \
        "N9b: 并发发布后 summary 不是任一完整代际（混写/截断）"
    _assert_no_tmp(base)
    print("  N9b(并发发布) PASS  双汇总器并发（4s/0.5s 慢写）均成功，"
          "最终文件与单跑逐字节一致，无混写/截断/临时残留")


def test_O2_python_opt_summary_atomic(base):
    """O2（049 建议 4）：N9a 同款写中断在 python -O 下重跑 → 仍非零
    退出且旧 summary 逐位不变（原子发布路径不依赖 assert）。"""
    sp = _place_sentinel_summary(base)
    old_sha = _sha(sp)
    r = _run_wrapper(base, "interrupt", opt_o=True)
    assert r.returncode != 0, ("O2: python -O 写中断后仍 exit=0",
                               r.stdout[-500:])
    assert json.load(open(sp)) == SENTINEL, "O2: python -O 写中断毁掉旧 summary"
    assert _sha(sp) == old_sha
    _assert_no_tmp(base)
    print("  O2(python -O 写中断) PASS  exit=%d，旧 summary 逐位不变，"
          "无 .tmp 残留" % r.returncode)


# ==== E116i 046③ 两残留（GPT 1633 审计）红绿用例 ====

def _to_v2(base):
    """三臂 legacy receipt 升级为 e116f-generation-v2：在既有 run 目录
    （即 generation）内补齐四规范文件（result.json/manifest.json/
    result.md/receipt.json），receipt 写入 publish_protocol/derived_dir/
    generation_files——与生产 v2 产物（128K 三臂）同构。"""
    for arm in ("mavg", "FullKV", "aavg"):
        p = _result_path(base, arm)
        rp = p + ".receipt.json"
        r = json.load(open(rp))
        gen = p + ".run-" + r["run_id"]
        shutil.copyfile(p, os.path.join(gen, "result.json"))
        shutil.copyfile(p + ".manifest.json",
                        os.path.join(gen, "manifest.json"))
        with open(os.path.join(gen, "result.md"), "w") as f:
            f.write("synthetic v2 generation\n")
        r["publish_protocol"] = "e116f-generation-v2"
        r["outputs"]["derived_dir"] = gen
        r["outputs"]["generation_files"] = {
            "json": "result.json", "md": "result.md",
            "manifest": "manifest.json", "receipt": "receipt.json"}
        json.dump(r, open(rp, "w"), indent=1, ensure_ascii=False)
        shutil.copyfile(rp, os.path.join(gen, "receipt.json"))


def _run_swap_wrapper(base, mode, opt_o=False):
    """046③ 残留 B 交错注入：经 wrapper 进程跑 analyzer（hook
    _read_bytes 或消费前完整切换），返回 (CompletedProcess, mark 路径)。"""
    wpath = os.path.join(base, f"swap_wrapper_{mode}.py")
    with open(wpath, "w", encoding="utf-8") as f:
        f.write(_SWAP_WRAPPER)
    mark = os.path.join(base, f"swap_{mode}.mark")
    cmd = [sys.executable] + (["-O"] if opt_o else []) + [
        wpath, mode, SCRIPT, _result_path(base, "mavg"), mark,
        "--results-dir", os.path.join(base, "results"),
        "--pred-root", os.path.join(base, "pred_root")]
    return subprocess.run(cmd, capture_output=True, text=True,
                          cwd=REPO), mark


def test_P3_v2_positive(base):
    """P3（046③）：三臂升级为 e116f-generation-v2 → v2 消费路径全过：
    数值与 P1 相同、逐臂 publish_protocol/derived_dir 单指针标注、
    legacy_protocol=false、publish_protocol_bound 仍闭合。"""
    _to_v2(base)
    r = _run(base)
    assert r.returncode == 0, (r.returncode, r.stdout[-1200:],
                               r.stderr[-800:])
    s = json.load(open(os.path.join(base, "results", SUMMARY)))
    assert {a: v["avg"] for a, v in s["arms"].items()} == FIXTURE_AVG, \
        s["arms"]
    ig = s["identity_gate"]
    for arm in ("mavg", "FullKV", "aavg"):
        proto = ig["per_arm_receipt_protocol"][arm]
        assert proto["publish_protocol"] == "e116f-generation-v2", \
            (arm, proto)
        assert proto["legacy_protocol"] is False, (arm, proto)
        assert proto["generation_dir_source"] == "outputs.derived_dir", \
            (arm, proto)
        assert s["arms"][arm]["legacy_protocol"] is False, s["arms"]
    assert s["closure"]["publish_protocol_bound"] is True
    # 052（E116j）：v2 只内容绑定 json/manifest，md/receipt 仅存在性
    # 检查——summary 如实标注，不得冒充四角色全绑定
    assert s["closure"]["gen_content_bound"] == \
        {"json": True, "manifest": True, "md": False, "receipt": False}, \
        s["closure"]
    print("P3 PASS  v2 协议三臂 → 单指针解析全过，数值 40.0/35.0/30.0 "
          "不变；publish_protocol/derived_dir 逐臂闭合，legacy 标注为 "
          "false")


def test_N10_v2_missing_roles(base):
    """N10（046③ 残留 A）：v2 receipt 的 generation_files 缺必需角色
    → fail-closed 拒收。a = 审计原始反例（单角色映射 {"scorer": ...}，
    修复前非空即过、exit=0 可绕过四规范文件检查）；b = 仅缺一个角色
    （md），python -O 下重跑。"""
    _to_v2(base)
    rp = _receipt_path(base, "mavg")
    r = json.load(open(rp))
    r["outputs"]["generation_files"] = {"scorer": "scorer.manifest.json"}
    json.dump(r, open(rp, "w"), indent=1, ensure_ascii=False)
    _expect_reject(base, "N10a(v2 单角色映射)", needle="缺必需角色")
    # 子例 b：四角色只缺 md，python -O 下重跑（门禁不依赖 assert）
    base_b = os.path.join(os.path.dirname(base), "fx_N10b")
    os.makedirs(base_b)
    _copy_fixture(base_b, _FIXTURE)
    _to_v2(base_b)
    rp = _receipt_path(base_b, "mavg")
    r = json.load(open(rp))
    del r["outputs"]["generation_files"]["md"]
    json.dump(r, open(rp, "w"), indent=1, ensure_ascii=False)
    _expect_reject(base_b, "N10b(v2 缺 md 角色)", opt_o=True,
                   needle="缺必需角色")


def test_N11_read_swap(base):
    """N11（046③ 残留 B）：消费者取到旧 result bytes 后、读 receipt 前，
    另一发布者完整安装新代际 → bytes 快照绑定必须拒收（修复前 exit=0
    且 summary 记旧均值 40 配新 result SHA——旧值配新哈希）；
    a=常解释器拒收、b=python -O 拒收、c=post_swap 控制组（完整切换
    先于消费）必须以新值+新哈希一致通过。"""
    _to_v2(base)
    _place_sentinel_summary(base)
    # -- a：读交换 → 必须拒收，且注入确实发生（mark 在位）--
    r, mark = _run_swap_wrapper(base, "read_swap")
    assert os.path.exists(mark), \
        "N11a: 交错注入未触发（wrapper hook 失效，负例不可信）"
    assert r.returncode != 0, ("N11a: 读交换后仍 exit=0（旧值配新哈希）",
                               r.stdout[-500:])
    assert "E119-GATE-FAIL" in r.stderr, r.stderr[-800:]
    cur = json.load(open(os.path.join(base, "results", SUMMARY)))
    assert cur == SENTINEL, "N11a: 拒收路径覆盖了旧 summary"
    print("  N11a(读交换拒收) PASS  exit=%d，注入已触发，旧 summary 未"
          "覆盖；拒绝原因: %s"
          % (r.returncode, r.stderr.strip().splitlines()[-1][:110]))
    # -- b：python -O 同款读交换（新鲜副本；门禁不依赖 assert）--
    base_b = os.path.join(os.path.dirname(base), "fx_N11b")
    os.makedirs(base_b)
    _copy_fixture(base_b, _FIXTURE)
    _to_v2(base_b)
    r, mark = _run_swap_wrapper(base_b, "read_swap", opt_o=True)
    assert os.path.exists(mark), "N11b: python -O 交错注入未触发"
    assert r.returncode != 0, ("N11b: python -O 读交换后仍 exit=0",
                               r.stdout[-500:])
    print("  N11b(python -O 读交换) PASS  exit=%d，快照绑定门禁不依赖 "
          "assert" % r.returncode)
    # -- c：post_swap 控制组——完整切换先于消费 → 新值+新哈希通过 --
    base_c = os.path.join(os.path.dirname(base), "fx_N11c")
    os.makedirs(base_c)
    _copy_fixture(base_c, _FIXTURE)
    _to_v2(base_c)
    r, mark = _run_swap_wrapper(base_c, "post_swap")
    assert os.path.exists(mark), "N11c: 控制组切换未执行"
    assert r.returncode == 0, ("N11c: 完整切换先于消费仍被拒（拒收过度）",
                               r.stderr[-800:])
    s = json.load(open(os.path.join(base_c, "results", SUMMARY)))
    assert s["arms"]["mavg"]["avg"] == 80.0, s["arms"]
    new_sha = _sha(_result_path(base_c, "mavg"))
    assert s["inputs"]["mavg"]["result_sha256"] == new_sha, s["inputs"]
    assert s["arms"]["mavg"]["receipt_run_id"].endswith("-next"), \
        s["arms"]["mavg"]
    print("  N11c(post-swap 控制) PASS  新代际 avg 80.0 与新 result SHA "
          "一致通过——拒收只针对交错，不拒绝合法代际更新")


# ==== E116j 052 红绿用例（generation md/receipt 内容哈希绑定） ====

def _to_v3(base):
    """三臂 receipt 升级为 e116i-generation-v3（E116j 052）：在 v2 结构上，
    entry receipt（公开 {out}.receipt.json）追加绑定 generation 的
    result.md/receipt.json 内容哈希（gen_md_sha256/gen_receipt_sha256）。
    gen receipt 先落盘（不含 entry-only 字段），entry receipt 再对其取
    哈希写公开 receipt——两文件互异、无自哈希，与生产 v3 产物同构。"""
    for arm in ("mavg", "FullKV", "aavg"):
        p = _result_path(base, arm)
        rp = p + ".receipt.json"
        r = json.load(open(rp))
        gen = p + ".run-" + r["run_id"]
        shutil.copyfile(p, os.path.join(gen, "result.json"))
        shutil.copyfile(p + ".manifest.json",
                        os.path.join(gen, "manifest.json"))
        with open(os.path.join(gen, "result.md"), "w") as f:
            f.write("synthetic v3 generation\n")
        r["publish_protocol"] = "e116i-generation-v3"
        r["outputs"]["derived_dir"] = gen
        r["outputs"]["generation_files"] = {
            "json": "result.json", "md": "result.md",
            "manifest": "manifest.json", "receipt": "receipt.json"}
        json.dump(r, open(os.path.join(gen, "receipt.json"), "w"),
                  indent=1, ensure_ascii=False)
        r["gen_md_sha256"] = _sha(os.path.join(gen, "result.md"))
        r["gen_receipt_sha256"] = _sha(os.path.join(gen, "receipt.json"))
        json.dump(r, open(rp, "w"), indent=1, ensure_ascii=False)


def test_P4_v3_positive(base):
    """P4（052 绿）：三臂升级为 e116i-generation-v3 → v3 消费路径全过：
    数值与 P1 相同，closure.gen_content_bound 四角色全 true（md/receipt
    由 entry receipt 内容绑定，不再是仅存在性检查）。"""
    _to_v3(base)
    r = _run(base)
    assert r.returncode == 0, (r.returncode, r.stdout[-1200:],
                               r.stderr[-800:])
    s = json.load(open(os.path.join(base, "results", SUMMARY)))
    assert {a: v["avg"] for a, v in s["arms"].items()} == FIXTURE_AVG, \
        s["arms"]
    ig = s["identity_gate"]
    for arm in ("mavg", "FullKV", "aavg"):
        proto = ig["per_arm_receipt_protocol"][arm]
        assert proto["publish_protocol"] == "e116i-generation-v3", \
            (arm, proto)
        assert proto["legacy_protocol"] is False, (arm, proto)
    assert s["closure"]["gen_content_bound"] == \
        {"json": True, "manifest": True, "md": True, "receipt": True}, \
        s["closure"]
    print("P4 PASS  v3 协议三臂 → entry receipt 内容绑定（gen_md/"
          "gen_receipt SHA256）全过，数值 40.0/35.0/30.0 不变；"
          "gen_content_bound 四角色全 true")


def test_N12_v3_tamper_gen_md(base):
    """N12（052 红）：v3 篡改 generation 内 result.md 内容 → fail-closed
    拒收，旧 summary 逐位不变。修复前 md 只有存在性检查——篡改后 exit=0
    照常汇总，注释宣称的「四规范文件 SHA 逐位闭合」是假闭合。
    a=常解释器拒收；b=python -O 拒收（052 门禁为显式条件 + _fail）。"""
    _to_v3(base)
    r = json.load(open(_receipt_path(base, "mavg")))
    gen = r["outputs"]["derived_dir"]
    with open(os.path.join(gen, "result.md"), "a") as f:
        f.write("tampered-after-bind\n")
    _expect_reject(base, "N12a(v3 篡改 gen result.md)",
                   needle="052 内容绑定")
    # 子例 b：python -O 重跑（门禁不依赖 assert）
    base_b = os.path.join(os.path.dirname(base), "fx_N12b")
    os.makedirs(base_b)
    _copy_fixture(base_b, _FIXTURE)
    _to_v3(base_b)
    r = json.load(open(_receipt_path(base_b, "mavg")))
    gen = r["outputs"]["derived_dir"]
    with open(os.path.join(gen, "result.md"), "a") as f:
        f.write("tampered-after-bind\n")
    _expect_reject(base_b, "N12b(python -O v3 篡改 gen result.md)",
                   opt_o=True, needle="052 内容绑定")


def test_N13_v3_tamper_gen_receipt(base):
    """N13（052 红）：v3 篡改 generation 内 receipt.json 内容 →
    fail-closed 拒收（gen_receipt_sha256 内容绑定），旧 summary 逐位
    不变。a=常解释器拒收；b=python -O 拒收。"""
    _to_v3(base)
    r = json.load(open(_receipt_path(base, "mavg")))
    gen = r["outputs"]["derived_dir"]
    with open(os.path.join(gen, "receipt.json"), "a") as f:
        f.write("\n")
    _expect_reject(base, "N13a(v3 篡改 gen receipt.json)",
                   needle="052 内容绑定")
    # 子例 b：python -O 重跑（门禁不依赖 assert）
    base_b = os.path.join(os.path.dirname(base), "fx_N13b")
    os.makedirs(base_b)
    _copy_fixture(base_b, _FIXTURE)
    _to_v3(base_b)
    r = json.load(open(_receipt_path(base_b, "mavg")))
    gen = r["outputs"]["derived_dir"]
    with open(os.path.join(gen, "receipt.json"), "a") as f:
        f.write("\n")
    _expect_reject(base_b, "N13b(python -O v3 篡改 gen receipt.json)",
                   opt_o=True, needle="052 内容绑定")


def test_P5_v2_tamper_md_partial_binding(base):
    """P5（052 如实标注）：v2（当前生产协议）篡改 generation 内
    result.md / receipt.json → 汇总通过（v2 对 md/receipt 确实只有
    存在性检查，篡改抓不到属协议能力边界、非本修复可在消费侧弥补），
    但 summary 必须如实标注 closure.gen_content_bound 的 md/receipt
    =false——不得冒充「四角色内容全绑定」（修复前注释过度宣称）。"""
    _to_v2(base)
    r = json.load(open(_receipt_path(base, "mavg")))
    gen = r["outputs"]["derived_dir"]
    with open(os.path.join(gen, "result.md"), "a") as f:
        f.write("v2-tamper-existence-only\n")
    with open(os.path.join(gen, "receipt.json"), "a") as f:
        f.write("\n")
    rr = _run(base)
    assert rr.returncode == 0, ("P5: v2 篡改 md/receipt 被拒（应通过并"
                               "如实标注部分绑定）", rr.stderr[-800:])
    s = json.load(open(os.path.join(base, "results", SUMMARY)))
    cb = s["closure"]["gen_content_bound"]
    assert cb == {"json": True, "manifest": True, "md": False,
                  "receipt": False}, cb
    # 篡改内容确实留在盘上（证明 v2 确实没有 md 内容绑定，而非被
    # 某条未预期路径改回/重生成）
    assert "v2-tamper-existence-only" in \
        open(os.path.join(gen, "result.md")).read()
    print("P5 PASS  v2 篡改 gen md/receipt → 汇总照常通过（存在性检查"
          "抓不到，协议能力边界），但 summary 如实标注 "
          "gen_content_bound.md/receipt=false，不冒充全绑定")


def main():
    global PASS, _FIXTURE
    root_tmp = tempfile.mkdtemp(prefix="e119_ident_")
    try:
        # 048①：fixture 源（已入库优先，异常检出时现场重建）
        _FIXTURE = _ensure_fixture(root_tmp)
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
            ("N7", test_N7_treatment_swap),
            ("N8", test_N8_script_sha_fail_closed),
            ("P2", test_P2_dynamic_conclusion),
            ("O1", test_O1_python_opt),
            ("N9", test_N9_summary_atomic),
            ("O2", test_O2_python_opt_summary_atomic),
            ("P3", test_P3_v2_positive),
            ("N10", test_N10_v2_missing_roles),
            ("N11", test_N11_read_swap),
            ("P4", test_P4_v3_positive),
            ("N12", test_N12_v3_tamper_gen_md),
            ("N13", test_N13_v3_tamper_gen_receipt),
            ("P5", test_P5_v2_tamper_md_partial_binding),
        ]
        for tag, fn in cases:
            base = os.path.join(root_tmp, f"fx_{tag}")
            os.makedirs(base)
            _copy_fixture(base, _FIXTURE)
            fn(base)
            PASS += 1
    finally:
        shutil.rmtree(root_tmp, ignore_errors=True)
    print(f"\nE119 crossarm identity ALL PASS ({PASS}/20) [tier={TIER}]")


if __name__ == "__main__":
    main()
