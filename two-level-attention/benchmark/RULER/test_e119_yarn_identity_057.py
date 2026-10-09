# TL-E119-YARN-IDENTITY-057 红绿测试（GPT 2026-10-10 0330 审计，任务链 #194）
#
# 违反事实：pred_ruler.py --yarn_factor 默认 None（65536/131072 档自动
# 2.0/4.0），E119 128K 派单只传 --yarn → 生成链按 4.0 自动档路径解析；而
# score_ruler_formal.py 把事后 CLI 声明原样写进 run_identity → 三份 128K
# manifest 声明 2.0，effective factor 未闭合。
#
# 修复协议（producer-yarn-config-v1）用例矩阵：
#   U1 自动档解析（纯函数）  --yarn@65536 → effective 2.0、--yarn@131072
#           → 4.0、显式 X 持久化 X、yarn off → (None, None)；
#           rope_scaling 六键完整；
#   U2 receipt 落盘往返      build → 原子写（旁挂 {pred 基名}
#           -yarn_receipt.json，不改 jsonl 行格式）→ 读回校验通过；
#           receipt 自相矛盾（rope_scaling.factor≠effective）→ 校验拒绝；
#   U3 生成侧接线（有 torch/sparse_attn 环境时）pred_ruler.YARN_FACTOR_AUTO
#           与 resolve/write 同表闭环；缺依赖环境 SKIP（formal 侧用例
#           不依赖 torch，保持干净检出可运行）；
#   F1 生产者证据正例        native fixture + 11 格 receipt（factor 2.0）
#           → formal 成功：run_identity.yarn_factor=2.0、
#           provenance=producer_receipt、producer_evidence 11/11 闭合、
#           CLI 声明一致（--yarn-factor 2.0）时也通过；
#   F2 factor 声明冲突        CLI --yarn-factor 4.0 vs receipt 2.0 →
#           非零退出不发布（python 与 -O 双跑）；
#   F3 yarn 开关声明冲突      receipt yarn_enabled=True vs 不传 --yarn →
#           非零退出不发布；
#   F4 legacy operator_declared   无 receipt 产物 + --yarn --yarn-factor
#           2.0 → 成功但 provenance=operator_declared、
#           producer_evidence.status=missing、cells 逐格 receipt=null
#           （CLI 值不再冒充实际生效值）；
#   F5 覆盖不全              10/11 格有 receipt → fail-closed
#           （run_identity 不得混合两种口径；python 与 -O 双跑）；
#   F6 receipt 损坏          ①截断 JSON ②自相矛盾 → 都 fail-closed
#           （存在即证据，半写/篡改比缺失更危险）；
#   F7 档位端到端            合成 L65536/L131072 root + auto 档 receipt
#           → formal 分别证实 2.0/4.0（producer_receipt）。
#
# 用法：
#   PYTHONPATH=$PWD python3 -m benchmark.RULER.test_e119_yarn_identity_057
#   （生产入口 fail-closed 的 -O 覆盖内嵌于 F2b/F5b：以 python -O 起子进程）
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
TESTDATA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "testdata", "e116e")

from benchmark.RULER.score_ruler import TASKS  # noqa: E402
from benchmark.RULER.yarn_receipt import (  # noqa: E402
    RECEIPT_SUFFIX, build_yarn_receipt, producer_receipt_path_for,
    resolve_yarn_config, validate_producer_receipt, write_yarn_receipt,
)

YARN_AUTO = {65536: 2.0, 131072: 4.0}   # 与 pred_ruler.py 逐位同表（U3 断言）
NATIVE_MPE = 40960
PASS = 0


def _sha(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _formal(root, out, data_root=None, min_samples=2, extra=(),
            postfix="_fx", optimized=False):
    argv = [sys.executable] + (["-O"] if optimized else []) + \
        ["-u", "-m", "benchmark.RULER.score_ruler_formal",
         "--root", root, "--pred-postfix", postfix,
         "--data-root", data_root or os.path.join(TESTDATA, "data_root"),
         "--out", out, "--min-samples", str(min_samples)]
    argv += list(extra)
    return subprocess.run(argv, capture_output=True, text=True, cwd=REPO,
                          env={**os.environ, "PYTHONPATH": REPO})


def _products(out):
    return [out, out[:-len(".json")] + ".md",
            out + ".manifest.json", out + ".receipt.json"]


def _copy_fixture(base, name, native=False):
    dst = os.path.join(base, name)
    shutil.copytree(
        os.path.join(TESTDATA, "pred_root_native" if native else "pred_root"),
        dst)
    return dst


def _write_receipts(pred_root, context_length, factor, enabled=True,
                     tasks=None):
    """给 root 下每个 pred jsonl 写旁挂生产者 receipt（模拟新生成侧落盘）。"""
    n = 0
    for f in sorted(glob.glob(os.path.join(
            pred_root, "L*", "pred_*", "*.jsonl"))):
        if os.path.basename(f).endswith("-merged.jsonl"):
            continue
        task = os.path.basename(f).split("-", 1)[0]
        if tasks is not None and task not in tasks:
            continue
        if enabled:
            eff, scaling = factor, {
                "rope_type": "yarn", "type": "yarn", "factor": factor,
                "original_max_position_embeddings": NATIVE_MPE,
                "beta_fast": 32, "beta_slow": 1,
            }
        else:
            eff, scaling = None, None
        rcp = build_yarn_receipt(
            yarn_enabled=enabled, effective_factor=eff, yarn_factor_cli=None,
            rope_scaling=scaling, context_length=context_length, task=task,
            model_path="/synthetic/Qwen3-8B", model_config_sha256=None,
            native_mpe=NATIVE_MPE,
            generation_params={"max_gen": 64, "max_num": 500, "seed": 42,
                               "method": "tli", "pred_postfix": "_fx",
                               "t": "01010000"},
            producer_script_path="benchmark/RULER/pred_ruler.py",
            producer_script_sha256="0" * 64)
        write_yarn_receipt(f, rcp)
        n += 1
    return n


def test_U1_auto_tier_resolution():
    """U1（057 规格①②纯函数层）：--yarn 自动档 65536→2.0 / 131072→4.0；
    显式 --yarn_factor X 持久化 X；yarn off → (None, None)。"""
    f64, rs64 = resolve_yarn_config(True, None, 65536, YARN_AUTO, NATIVE_MPE)
    f128, rs128 = resolve_yarn_config(True, None, 131072, YARN_AUTO,
                                      NATIVE_MPE)
    assert f64 == 2.0 and f128 == 4.0, (f64, f128)
    # rope_scaling 六键完整且 factor=effective
    for f, rs in ((f64, rs64), (f128, rs128)):
        assert set(rs) == {"rope_type", "type", "factor",
                           "original_max_position_embeddings",
                           "beta_fast", "beta_slow"}
        assert rs["rope_type"] == "yarn" and rs["type"] == "yarn" and \
            rs["factor"] == f and \
            rs["original_max_position_embeddings"] == NATIVE_MPE and \
            rs["beta_fast"] == 32 and rs["beta_slow"] == 1
    # 显式覆盖持久化（自动档被覆盖）
    fx, rsx = resolve_yarn_config(True, 3.5, 131072, YARN_AUTO, NATIVE_MPE)
    assert fx == 3.5 and rsx["factor"] == 3.5
    # off
    fo, rso = resolve_yarn_config(False, 3.5, 131072, YARN_AUTO, NATIVE_MPE)
    assert fo is None and rso is None
    print("U1 PASS  自动档 65536→2.0 / 131072→4.0；显式 3.5 持久化 3.5；"
          "rope_scaling 六键完整；off → (None, None)")


def test_U2_receipt_roundtrip(base):
    """U2：receipt 原子落盘（旁挂命名约定 + jsonl 行格式零改动）+ 读回
    校验通过 + 自相矛盾 receipt 被校验拒绝。"""
    pred = os.path.join(base, "u2", "niah_single_1-fxm-01010000.jsonl")
    os.makedirs(os.path.dirname(pred), exist_ok=True)
    with open(pred, "w", encoding="utf-8") as f:
        f.write('{"pred": "x", "answers": ["x"], "length": 1, '
                '"budget": 0}\n')
    before = _sha(pred)
    eff, scaling = resolve_yarn_config(True, None, 131072, YARN_AUTO,
                                       NATIVE_MPE)
    rcp = build_yarn_receipt(
        yarn_enabled=True, effective_factor=eff, yarn_factor_cli=None,
        rope_scaling=scaling, context_length=131072,
        task="niah_single_1", model_path="/synthetic/Qwen3-8B",
        model_config_sha256=None, native_mpe=NATIVE_MPE,
        generation_params={"max_gen": 64}, producer_script_path="p",
        producer_script_sha256="0" * 64)
    path = write_yarn_receipt(pred, rcp)
    # 命名约定：{pred 基名}-yarn_receipt.json；jsonl 字节零改动
    assert path == pred[:-len(".jsonl")] + RECEIPT_SUFFIX
    assert producer_receipt_path_for(pred) == path
    assert _sha(pred) == before, "写 receipt 改动了 jsonl 产物字节"
    back = json.load(open(path, encoding="utf-8"))
    assert validate_producer_receipt(back, pred) is None
    assert back["effective_yarn_factor"] == 4.0 and \
        back["yarn_factor_source"] == "auto" and \
        back["receipt_version"] == "producer-yarn-config-v1"
    # 无 tmp 残留（原子写）
    assert not glob.glob(path + ".tmp-*")
    # 自相矛盾：rope_scaling.factor 与 effective 不一致 → 校验拒绝
    bad = dict(back)
    bad["rope_scaling"] = dict(back["rope_scaling"], factor=2.0)
    assert validate_producer_receipt(bad, pred) is not None
    # yarn_enabled=False 但 effective 非 None → 拒绝
    bad2 = dict(back, yarn_enabled=False)
    assert validate_producer_receipt(bad2, pred) is not None
    print("U2 PASS  receipt 原子落盘 + 旁挂命名约定 + jsonl 零改动 + "
          "读回校验通过 + 自相矛盾/状态矛盾 receipt 拒绝")


def test_U3_producer_wiring():
    """U3：生成侧接线（有 torch/sparse_attn 的环境）——pred_ruler 的
    YARN_FACTOR_AUTO 与本套件同表，resolve/write 走 pred_ruler 引用的
    同一实现。缺依赖环境 SKIP（formal 侧用例不受影响）。"""
    try:
        import benchmark.RULER.pred_ruler as pr
    except Exception as e:   # torch/transformers/sparse_attn 缺失
        print(f"U3 SKIP  pred_ruler 不可导入（{type(e).__name__}: {e}）"
              f"——生成侧接线断言需 torch 环境，formal 侧用例不受影响")
        return
    assert pr.YARN_FACTOR_AUTO == YARN_AUTO, pr.YARN_FACTOR_AUTO
    f64, _ = pr.resolve_yarn_config(True, None, 65536, pr.YARN_FACTOR_AUTO,
                                    pr.QWEN3_NATIVE_MPE)
    f128, _ = pr.resolve_yarn_config(True, None, 131072, pr.YARN_FACTOR_AUTO,
                                     pr.QWEN3_NATIVE_MPE)
    assert f64 == 2.0 and f128 == 4.0
    assert pr.write_yarn_receipt is write_yarn_receipt and \
        pr.resolve_yarn_config is resolve_yarn_config
    print("U3 PASS  pred_ruler.YARN_FACTOR_AUTO={65536: 2.0, 131072: 4.0} "
          "与 resolve/write 同一实现闭环（131072 → 4.0）")


def test_F1_producer_evidence_positive(base):
    """F1（057 规格①formal 端）：native fixture + 11 格 receipt
    （auto 档 2.0）→ formal 成功，run_identity 消费生产者证据。"""
    root = _copy_fixture(base, "f1_root", native=True)
    n = _write_receipts(root, 32768, 2.0)
    assert n == 11, n
    out = os.path.join(base, "f1.json")
    r = _formal(root, out, extra=("--yarn",))
    assert r.returncode == 0 and "DONE" in r.stdout, \
        (r.returncode, r.stdout[-3000:], r.stderr[-2000:])
    mf = json.load(open(out + ".manifest.json"))
    ri = mf["run_identity"]
    assert ri["yarn"] is True and ri["yarn_factor"] == 2.0 and \
        ri["yarn_factor_provenance"] == "producer_receipt", ri
    pe = ri["producer_evidence"]
    assert pe["status"] == "present" and pe["cells_with_receipt"] == 11 \
        and pe["cells_total"] == 11 and pe["effective_yarn_factor"] == 2.0
    # 逐格 receipt 字节绑定（path+sha 与磁盘一致）
    for cell in mf["cells"].values():
        for t, ti in cell["tasks"].items():
            rc = ti["producer_yarn_receipt"]
            assert rc is not None and _sha(rc["path"]) == rc["sha256"], \
                (t, rc)
            assert rc["effective_yarn_factor"] == 2.0
    # formal receipt 的 inputs.run_identity 同值（下游消费者单指针可恢复）
    frc = json.load(open(out + ".receipt.json"))
    fri = frc["inputs"]["run_identity"]
    assert fri["yarn_factor"] == 2.0 and \
        fri["yarn_factor_provenance"] == "producer_receipt"
    # CLI 声明与生产者证据一致时也通过（operator declared 保留记录）
    out2 = os.path.join(base, "f1b.json")
    r2 = _formal(root, out2, extra=("--yarn", "--yarn-factor", "2.0"))
    assert r2.returncode == 0, (r2.returncode, r2.stdout[-2000:])
    ri2 = json.load(open(out2 + ".manifest.json"))["run_identity"]
    assert ri2["yarn_factor"] == 2.0 and \
        ri2["yarn_factor_provenance"] == "producer_receipt" and \
        ri2["yarn_factor_operator_declared"] == 2.0
    print("F1 PASS  11 格生产者 receipt → formal 证实 effective=2.0"
          "（producer_receipt，cells/path+sha 逐格冻结）；CLI 声明一致时"
          "通过且 operator_declared 字段保留")


def test_F2_factor_conflict_fail_closed(base):
    """F2（057 核心）：CLI --yarn-factor 4.0 vs receipt 2.0 → fail-closed
    不发布；python 与 -O 双跑（-O 子例 F2b）。"""
    root = _copy_fixture(base, "f2_root", native=True)
    _write_receipts(root, 32768, 2.0)
    out = os.path.join(base, "f2.json")
    for optimized in (False, True):
        r = _formal(root, out, extra=("--yarn", "--yarn-factor", "4.0"),
                    optimized=optimized)
        assert r.returncode != 0, \
            (optimized, r.returncode, r.stdout[-2000:])
        assert "factor 声明冲突" in (r.stdout + r.stderr) and \
            "fail closed" in (r.stdout + r.stderr)
        assert not any(os.path.exists(p) for p in _products(out)), \
            "冲突仍发布了产物"
        assert not glob.glob(out + ".staging-*")
        assert glob.glob(out + ".failure-*.json")
    print("F2 PASS  CLI 声明 4.0 vs 生产者证据 2.0 → 非零退出不发布"
          "（python 与 -O 双跑，_fail 非 assert）")


def test_F3_yarn_flag_conflict(base):
    """F3：receipt yarn_enabled=True vs 不传 --yarn → 声明冲突 fail-closed。"""
    root = _copy_fixture(base, "f3_root", native=True)
    _write_receipts(root, 32768, 2.0)
    out = os.path.join(base, "f3.json")
    r = _formal(root, out, extra=())
    assert r.returncode != 0 and "yarn 声明冲突" in (r.stdout + r.stderr), \
        (r.returncode, r.stdout[-2000:])
    assert not any(os.path.exists(p) for p in _products(out))
    print("F3 PASS  生产者证据 yarn_enabled=True vs CLI 未声明 --yarn → "
          "fail-closed（事后声明不得与实际值矛盾）")


def test_F4_legacy_operator_declared(base):
    """F4（057 规格④）：legacy 产物无 receipt → CLI 值降级
    operator_declared（不冒充实际生效值）。"""
    root = _copy_fixture(base, "f4_root")          # legacy fixture，无 receipt
    out = os.path.join(base, "f4.json")
    r = _formal(root, out, extra=("--yarn", "--yarn-factor", "2.0"))
    assert r.returncode == 0 and "operator_declared" in r.stdout, \
        (r.returncode, r.stdout[-3000:], r.stderr[-2000:])
    ri = json.load(open(out + ".manifest.json"))["run_identity"]
    assert ri["yarn_factor"] == 2.0 and \
        ri["yarn_factor_provenance"] == "operator_declared" and \
        ri["yarn_factor_operator_declared"] == 2.0
    pe = ri["producer_evidence"]
    assert pe["status"] == "missing" and pe["cells_with_receipt"] == 0 and \
        "未闭合" in pe["note"]
    mf = json.load(open(out + ".manifest.json"))
    for cell in mf["cells"].values():
        for ti in cell["tasks"].values():
            assert ti["producer_yarn_receipt"] is None
    print("F4 PASS  legacy 无 receipt → yarn_factor=2.0 标注 "
          "operator_declared + producer_evidence missing + 逐格 "
          "receipt=null（不再冒充实际生效值）")


def test_F5_partial_coverage_fail_closed(base):
    """F5：10/11 格有 receipt → 混合口径 fail-closed（python 与 -O 双跑）。"""
    root = _copy_fixture(base, "f5_root", native=True)
    n = _write_receipts(root, 32768, 2.0, tasks=set(TASKS) - {"vt"})
    assert n == 10, n
    out = os.path.join(base, "f5.json")
    for optimized in (False, True):
        r = _formal(root, out, extra=("--yarn",), optimized=optimized)
        assert r.returncode != 0 and "覆盖不全" in (r.stdout + r.stderr), \
            (optimized, r.returncode, r.stdout[-2000:])
        assert not any(os.path.exists(p) for p in _products(out))
        assert glob.glob(out + ".failure-*.json")
    print("F5 PASS  10/11 格证据 → run_identity 不得混合『生产者证实』与"
          "『操作者声明』→ fail-closed（python 与 -O 双跑）")


def test_F6_corrupt_receipt_fail_closed(base):
    """F6：receipt 存在即证据——①截断 JSON ②自相矛盾 → 都 fail-closed。"""
    # ① 截断（模拟半写/磁盘损坏）
    root = _copy_fixture(base, "f6a_root", native=True)
    _write_receipts(root, 32768, 2.0)
    tgt = glob.glob(os.path.join(root, "L32768", "pred_fx",
                                 "vt-*.jsonl"))[0]
    rcp_path = producer_receipt_path_for(tgt)
    open(rcp_path, "w", encoding="utf-8").write('{"partial":')
    out = os.path.join(base, "f6a.json")
    r = _formal(root, out, extra=("--yarn",))
    assert r.returncode != 0 and "解析失败" in (r.stdout + r.stderr), \
        (r.returncode, r.stdout[-2000:])
    assert not any(os.path.exists(p) for p in _products(out))
    # ② 自相矛盾（rope_scaling.factor 与 effective 不一致）
    root = _copy_fixture(base, "f6b_root", native=True)
    _write_receipts(root, 32768, 2.0)
    tgt = glob.glob(os.path.join(root, "L32768", "pred_fx",
                                 "vt-*.jsonl"))[0]
    rcp_path = producer_receipt_path_for(tgt)
    rcp = json.load(open(rcp_path, encoding="utf-8"))
    rcp["rope_scaling"] = dict(rcp["rope_scaling"], factor=4.0)
    json.dump(rcp, open(rcp_path, "w", encoding="utf-8"))
    out = os.path.join(base, "f6b.json")
    r = _formal(root, out, extra=("--yarn",))
    assert r.returncode != 0 and "自相矛盾" in (r.stdout + r.stderr), \
        (r.returncode, r.stdout[-2000:])
    assert not any(os.path.exists(p) for p in _products(out))
    print("F6 PASS  ①截断 JSON ②自相矛盾 receipt → 均非零退出不发布"
          "（存在即证据：半写/篡改比缺失更危险）")


def test_F7_auto_tier_end_to_end(base):
    """F7（057 规格①formal 端档位覆盖）：合成 L65536/L131072 root +
    auto 档 receipt → formal 分别证实 2.0 / 4.0（producer_receipt）。"""
    for L, factor in ((65536, 2.0), (131072, 4.0)):
        root = _copy_fixture(base, f"f7_{L}_root", native=True)
        src_L = os.path.join(root, "L32768")
        os.rename(src_L, os.path.join(root, f"L{L}"))
        # data_root 对应档位目录（源数据存在性门禁；内容仅做 SHA 绑定）
        data_root = os.path.join(base, f"f7_data_{L}")
        os.makedirs(os.path.join(data_root, str(L)), exist_ok=True)
        for f in glob.glob(os.path.join(TESTDATA, "data_root", "32768",
                                        "*.jsonl")):
            shutil.copyfile(f, os.path.join(data_root, str(L),
                                            os.path.basename(f)))
        _write_receipts(root, L, factor)
        out = os.path.join(base, f"f7_{L}.json")
        r = _formal(root, out, data_root=data_root, extra=("--yarn",))
        assert r.returncode == 0 and "DONE" in r.stdout, \
            (L, r.returncode, r.stdout[-3000:], r.stderr[-2000:])
        ri = json.load(open(out + ".manifest.json"))["run_identity"]
        assert ri["yarn_factor"] == factor and \
            ri["yarn_factor_provenance"] == "producer_receipt", (L, ri)
        pe = ri["producer_evidence"]
        assert pe["cells_with_receipt"] == 11 and \
            pe["effective_yarn_factor"] == factor
    print("F7 PASS  L65536 → producer 证实 2.0；L131072 → producer 证实"
          " 4.0（自动档两端点 formal 端到端闭合）")


def main():
    global PASS
    base = tempfile.mkdtemp(prefix="e119_yarn_057_")
    try:
        test_U1_auto_tier_resolution()
        test_U2_receipt_roundtrip(base)
        test_U3_producer_wiring()
        test_F1_producer_evidence_positive(base)
        test_F2_factor_conflict_fail_closed(base)
        test_F3_yarn_flag_conflict(base)
        test_F4_legacy_operator_declared(base)
        test_F5_partial_coverage_fail_closed(base)
        test_F6_corrupt_receipt_fail_closed(base)
        test_F7_auto_tier_end_to_end(base)
        PASS = 10
    finally:
        shutil.rmtree(base, ignore_errors=True)
    print(f"\nE119-YARN-IDENTITY-057 ALL PASS ({PASS}/10)")


if __name__ == "__main__":
    main()
