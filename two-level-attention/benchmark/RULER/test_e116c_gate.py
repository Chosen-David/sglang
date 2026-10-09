# E116c 红绿测试（GPT 0826 审计 TL-RULER-SAMPLE-GATE-026 修复）
# 验证 score_ruler.py fail-closed 门禁 + --merge-best 显式合并：
#   C1 正例：完整文件 + manifest 集合闭包 + answers hash 一致 → 通过
#   C2 负例：同 task/method-key 两完整文件（无 --merge-best）→ 非零退出
#   C3 --merge-best：行数少的 bad-file 被跳过，评分用 best-file，
#      规范文件 {task}-{method}-merged.jsonl 落盘
#   C4 负例：重复 _id 行 → 非零退出
#   C5 负例：缺行 vs manifest → 非零退出
#   C6 负例：answers hash 错配 → 非零退出
#   C7 负例：缺任一任务 + --expect-tasks 11 → 非零退出（不输出正式结果）
# 用法: python benchmark/RULER/test_e116c_gate.py
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
PASS = 0

TASK = "cwe"
N = 5


def sha16(a):
    return hashlib.sha256(json.dumps(
        a, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]


def _mk_rows(hit=True, n=N):
    rows = []
    for i in range(n):
        ans = [f"gt{i}"]
        rows.append({
            "pred": f"the answer is gt{i}" if hit else f"wrong {i}",
            "answers": ans, "length": 100, "budget": 0,
            "_id": f"{TASK}:{i}", "_answers_sha": sha16(ans),
        })
    return rows


def _mk_root(base, files):
    """files: {filename: rows} → root/L32768/pred_t/<filename>"""
    root = os.path.join(base, f"root{len(os.listdir(base))}")
    pd = os.path.join(root, "L32768", "pred_t")
    os.makedirs(pd)
    for fn, rows in files.items():
        with open(os.path.join(pd, fn), "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return root, os.path.join(base, f"table{len(os.listdir(base))}.json")


def _run(root, out, merge=False, manifest=None, expect=0):
    argv = ["--root", root, "--pred-postfix", "_t", "--out", out,
            "--min-samples", str(N)]
    if merge:
        argv += ["--merge-best"]
    if manifest:
        argv += ["--manifest", manifest]
    if expect:
        argv += ["--expect-tasks", str(expect)]
    return subprocess.run(
        [sys.executable, "-m", "benchmark.RULER.score_ruler"] + argv,
        capture_output=True, text=True, cwd=REPO,
        env={**os.environ, "PYTHONPATH": REPO},
    )


def _no_output(out):
    return not os.path.exists(out) and not os.path.exists(
        out.replace(".json", ".md"))


def test_C():
    global PASS
    base = tempfile.mkdtemp(prefix="e116c_")
    try:
        rows = _mk_rows()
        manifest_rows = {"ids": [r["_id"] for r in rows],
                         "answers_sha": {r["_id"]: r["_answers_sha"]
                                         for r in rows}}
        mf = os.path.join(base, "m.json")
        json.dump({TASK: manifest_rows}, open(mf, "w"))

        # C1 正例：完整文件 + manifest 闭包 + hash 一致 → 通过
        root, out = _mk_root(base, {f"{TASK}-none-01010000.jsonl": rows})
        r = _run(root, out, manifest=mf)
        assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
        got = json.load(open(out))
        assert got["scores"]["L32768/none"][TASK] == 100.0, got
        assert got["sources"]["L32768/none"][TASK]["file"] == \
            f"{TASK}-none-01010000.jsonl"
        assert len(got["sources"]["L32768/none"][TASK]["sha256"]) == 64
        print("C1 PASS  正例（5 行 + manifest 闭包 + hash 一致）→ "
              "result 落盘含 sources（文件名+SHA256）")

        # C2 负例：同键两完整文件（无 --merge-best）→ 非零退出
        root, out = _mk_root(base, {
            f"{TASK}-none-01010000.jsonl": rows,
            f"{TASK}-none-01020000.jsonl": _mk_rows(hit=False)})
        r = _run(root, out)
        assert r.returncode != 0 and "GATE-FAIL" in (r.stdout + r.stderr), \
            (r.returncode, r.stdout, r.stderr)
        assert "01010000" in (r.stdout + r.stderr) and \
            "01020000" in (r.stdout + r.stderr), "冲突文件清单未列出"
        assert _no_output(out), "门禁失败仍写了结果"
        print("C2 PASS  同键两完整文件（无 --merge-best）→ 非零退出 + "
              "冲突清单 + 不写结果")

        # C3 --merge-best：bad-file（行数少）被跳过，评分用 best-file
        root, out = _mk_root(base, {
            f"{TASK}-none-01010000.jsonl": _mk_rows(hit=False, n=3),
            f"{TASK}-none-01020000.jsonl": rows})
        r = _run(root, out, merge=True)
        assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
        canon = os.path.join(root, "L32768", "pred_t",
                             f"{TASK}-none-merged.jsonl")
        assert os.path.exists(canon), "规范文件未落盘"
        canon_rows = [json.loads(l) for l in open(canon)]
        assert [x["_id"] for x in canon_rows] == [x["_id"] for x in rows]
        got = json.load(open(out))
        assert got["scores"]["L32768/none"][TASK] == 100.0, \
            f"best-file 未生效: {got}"
        assert got["sources"]["L32768/none"][TASK]["file"] == \
            f"{TASK}-none-merged.jsonl"
        assert "01020000" in r.stdout and "01010000" in r.stdout, \
            "选择打印缺候选"
        print("C3 PASS  --merge-best：3 行 bad-file 被跳过，评分用 5 行 "
              "best-file，规范文件落盘 + 选择打印")

        # C4 负例：重复 _id 行 → 非零退出
        root, out = _mk_root(
            base, {f"{TASK}-none-01030000.jsonl": rows + [rows[0]]})
        r = _run(root, out)
        assert r.returncode != 0 and "重复 _id" in (r.stdout + r.stderr), \
            (r.returncode, r.stdout, r.stderr)
        assert _no_output(out)
        print("C4 PASS  重复 _id 行 → 非零退出 + 不写结果")

        # C5 负例：缺行 vs manifest（4/5）→ 非零退出
        root, out = _mk_root(base, {f"{TASK}-none-01040000.jsonl": rows[:4]})
        r = _run(root, out, manifest=mf)
        assert r.returncode != 0 and "GATE-FAIL" in (r.stdout + r.stderr) \
            and "manifest" in (r.stdout + r.stderr), (r.returncode, r.stdout)
        assert _no_output(out)
        print("C5 PASS  缺行（4/5）vs manifest → 非零退出 + 不写结果")

        # C6 负例：answers hash 错配（同 id 换答案且行内 _answers_sha 同步
        # 重算——只留 manifest 错配这一条触发路径）→ 非零退出
        bad = [dict(x) for x in rows]
        bad[2] = dict(bad[2], answers=["tampered answer"],
                      _answers_sha=sha16(["tampered answer"]))
        root, out = _mk_root(base, {f"{TASK}-none-01050000.jsonl": bad})
        r = _run(root, out, manifest=mf)
        assert r.returncode != 0 and "answers hash" in (r.stdout + r.stderr), \
            (r.returncode, r.stdout)
        assert _no_output(out)
        print("C6 PASS  answers hash 错配 → 非零退出 + 不写结果")

        # C6b 负例：行内 _answers_sha 与 answers 不一致（无 manifest 也拦）→
        # 非零退出（记录篡改/混行）
        tam = [dict(x) for x in rows]
        tam[1] = dict(tam[1], _answers_sha="0" * 16)
        root, out = _mk_root(base, {f"{TASK}-none-01060000.jsonl": tam})
        r = _run(root, out)
        assert r.returncode != 0 and "_answers_sha" in (r.stdout + r.stderr)
        print("C6b PASS 行内 _answers_sha 与 answers 重算不一致 → 非零退出")

        # C7 负例：缺任一任务 + --expect-tasks 11 → 非零退出
        root, out = _mk_root(base, {f"{TASK}-none-01070000.jsonl": rows})
        r = _run(root, out, expect=11)
        assert r.returncode != 0 and "expect-tasks" in (r.stdout + r.stderr), \
            (r.returncode, r.stdout, r.stderr)
        assert _no_output(out), "缺任务仍写了正式结果"
        print("C7 PASS  缺 10/11 任务 + --expect-tasks 11 → 非零退出 + "
              "不输出正式结果")

        # C7b 正例：--expect-tasks 1（预期集合 TASKS[:1]=niah_single_1）→ 通过
        ns_rows = [dict(r, _id=f"niah_single_1:{i}") for i, r in
                   enumerate(rows)]
        root, out = _mk_root(
            base, {"niah_single_1-none-01080000.jsonl": ns_rows})
        r = _run(root, out, expect=1)
        assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
        print("C7b PASS --expect-tasks 1 任务齐 → 通过")

        # C8 兼容：老数据无 _id、单文件、无 manifest → 旧版行为（评分通过）
        old_rows = [{"pred": r["pred"], "answers": r["answers"],
                     "length": r["length"], "budget": r["budget"]}
                    for r in rows]
        root, out = _mk_root(base, {f"{TASK}-none-01090000.jsonl": old_rows})
        r = _run(root, out)
        assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
        assert json.load(open(out))["scores"]["L32768/none"][TASK] == 100.0
        print("C8 PASS  老数据（无 _id 单文件无 manifest）→ 兼容通过")

        # C9 负例：无 _id 旧文件 + manifest → 非零退出（拒绝静默跳过）
        root, out = _mk_root(base, {f"{TASK}-none-01100000.jsonl": old_rows})
        r = _run(root, out, manifest=mf)
        assert r.returncode != 0 and "缺 _id" in (r.stdout + r.stderr)
        assert _no_output(out)
        print("C9 PASS  无 _id 旧文件 + manifest → 非零退出（拒绝静默跳过）")
        PASS += 11
    finally:
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    test_C()
    print(f"\nE116c ALL PASS ({PASS}/11)")
