# E116a 红绿测试（GPT 0728 审计 TL-LBV1-SAMPLE-GATE-024 / SCORER-BACKEND-025 修复）
# 验证：
#   A. metrics scorer 后端固定——difflib 默认路径与 fuzzywuzzy 无 Levenshtein 路径逐位一致
#   B. eval.py 门禁 fail closed——缺行/重复行/answers 错配/行数不足 四负例全拒 + 正例通过
# 用法: python benchmark/LongBench/test_e116a_gate.py
import json, os, sys, tempfile, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

PASS = 0

# ---------- A. scorer 后端 ----------
def test_A():
    global PASS
    from benchmark.LongBench.metrics import _fuzz_ratio, SCORER_BACKEND_ID, code_sim_score
    import difflib
    pairs = [("def foo():\n    return 1", "def foo():\n    return 2"),
             ("hello world", "hello worlds!"),
             ("", "abc"), ("abc", "abc"), ("完全不同字符串", "another thing")]
    for s1, s2 in pairs:
        # 与 fuzzywuzzy 无 Levenshtein 的 stdlib 路径语义逐位一致（含 round）
        ref = int(round(100 * difflib.SequenceMatcher(None, s1, s2).ratio()))
        assert _fuzz_ratio(s1, s2) == ref, (s1, s2, _fuzz_ratio(s1, s2), ref)
    # code_sim_score 走首行无注释行（LongBench 口径）
    v = code_sim_score("`code`\n# comment\nplain line here", "plain line here")
    assert 0.0 <= v <= 1.0
    print(f"A PASS  scorer backend fixed ({SCORER_BACKEND_ID})，5 对逐位一致")
    PASS += 1

# ---------- B. eval 门禁 ----------
def _mk_pred_file(d, fn, rows):
    with open(os.path.join(d, fn), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def _run_eval(d, manifest=None, expect=0):
    argv = ["--output-path", d]
    if manifest: argv += ["--manifest", manifest]
    if expect: argv += ["--expect-count", str(expect)]
    # subprocess 跑模块 __main__（真实控制流）
    import subprocess
    r = subprocess.run(
        [sys.executable, "-m", "benchmark.LongBench.eval"] + argv,
        capture_output=True, text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        env={**os.environ, "PYTHONPATH": os.pathsep.join(
            [os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
             os.path.expanduser("~/.local/pylibs")])},
    )
    return r

def test_B():
    global PASS
    base = tempfile.mkdtemp(prefix="e116a_")
    try:
        import hashlib
        def sha(a): return hashlib.sha256(json.dumps(a, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        rows = [{"pred": f"answer {i}", "answers": [f"gt {i}"], "all_classes": None,
                 "length": 100, "_id": f"narrativeqa:abcd1234:{i}"} for i in range(5)]
        rows_dedup = {r["_id"]: r for r in rows}

        # B1 正例：完整 5 行 + manifest 集合相等 + answers hash 一致 → 通过，写 result.json
        d1 = os.path.join(base, "ok"); os.makedirs(d1)
        _mk_pred_file(d1, "narrativeqa-m-001.jsonl", rows)
        m = {"narrativeqa": {"ids": [r["_id"] for r in rows],
                      "answers_sha": {r["_id"]: sha(r["answers"]) for r in rows}}}
        mf = os.path.join(base, "m.json"); json.dump(m, open(mf, "w"))
        r = _run_eval(d1, manifest=mf)
        assert r.returncode == 0, r.stderr
        assert json.load(open(os.path.join(d1, "result.json")))["_meta"]["scorer_backend"]
        print("B1 PASS  正例（5 行 + manifest 闭包 + hash 一致）→ result.json 落盘含 _meta")
        # B1 附：门禁通过时 result.json 必须未被负例场景写入过（每场景独立目录）

        # B2 负例：缺一行（4/5）
        d2 = os.path.join(base, "miss"); os.makedirs(d2)
        _mk_pred_file(d2, "narrativeqa-m-002.jsonl", rows[:4])
        r = _run_eval(d2, manifest=mf)
        assert r.returncode != 0 and "GATE-FAIL" in (r.stderr + r.stdout), (r.returncode, r.stderr, r.stdout)
        assert not os.path.exists(os.path.join(d2, "result.json")), "缺行仍写了 result.json"
        print("B2 PASS  缺行（4/5）→ 非零退出 + 不写 result.json")

        # B3 负例：重复一行（6 行含重复 id）
        d3 = os.path.join(base, "dup"); os.makedirs(d3)
        _mk_pred_file(d3, "narrativeqa-m-003.jsonl", rows + [rows[0]])
        r = _run_eval(d3, manifest=mf)
        assert r.returncode != 0 and "重复 _id" in (r.stderr + r.stdout)
        assert not os.path.exists(os.path.join(d3, "result.json"))
        print("B3 PASS  重复行（6 行重复 id）→ 非零退出 + 不写 result.json")

        # B4 负例：answers 错配（同 id 换答案）
        d4 = os.path.join(base, "swap"); os.makedirs(d4)
        bad = [dict(r) for r in rows]; bad[2]["answers"] = ["wrong answer"]
        _mk_pred_file(d4, "narrativeqa-m-004.jsonl", bad)
        r = _run_eval(d4, manifest=mf)
        assert r.returncode != 0 and "answers hash" in (r.stderr + r.stdout)
        assert not os.path.exists(os.path.join(d4, "result.json"))
        print("B4 PASS  answers 错配（同 id 换答案）→ 非零退出 + 不写 result.json")

        # B5 负例：行数不足（expect-count 10 > 实际 5）
        d5 = os.path.join(base, "short"); os.makedirs(d5)
        _mk_pred_file(d5, "narrativeqa-m-005.jsonl", rows)
        r = _run_eval(d5, expect=10)
        assert r.returncode != 0 and "expect-count" in (r.stderr + r.stdout)
        assert not os.path.exists(os.path.join(d5, "result.json"))
        print("B5 PASS  行数不足（5 < expect 10）→ 非零退出 + 不写 result.json")

        # B6 负例：旧文件无 _id 却给了 manifest
        d6 = os.path.join(base, "noid"); os.makedirs(d6)
        _mk_pred_file(d6, "narrativeqa-m-006.jsonl",
                      [{"pred": p, "answers": a, "all_classes": None, "length": 100}
                       for p, a in [(r["pred"], r["answers"]) for r in rows]])
        r = _run_eval(d6, manifest=mf)
        assert r.returncode != 0 and "缺 _id" in (r.stderr + r.stdout)
        assert not os.path.exists(os.path.join(d6, "result.json"))
        print("B6 PASS  无 _id 旧文件 + manifest → 非零退出（拒绝静默跳过校验）")
        PASS += 6
    finally:
        shutil.rmtree(base, ignore_errors=True)

if __name__ == "__main__":
    test_A(); test_B()
    print(f"\nE116a ALL PASS ({PASS}/7)")
