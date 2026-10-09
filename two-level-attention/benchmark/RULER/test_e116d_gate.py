# E116d 红绿测试（GPT 0935 审计三 P1 修复回归）
#   TL-RULER-GATE-INTEGRATION-027（空 root 真空通过）→ D1/D2
#   TL-RULER-MANIFEST-HASH-028（manifest answers_sha 缺失静默跳过）→ D3-D6
#   TL-RULER-MERGED-STALE-029（merged 规范文件参与候选竞争）→ D7
#   任务闭包 → D8；真实数据生产正式入口回归 → D9；E116c 套件不回归 → D10
# 负例全部 fail closed（非零退出 + 不写任何输出文件）。
# 用法: PYTHONPATH=$PWD python3 benchmark/RULER/test_e116d_gate.py
import glob
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
    import hashlib
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


def test_D():
    global PASS
    base = tempfile.mkdtemp(prefix="e116d_")
    try:
        rows = _mk_rows()
        ids = [r["_id"] for r in rows]
        shas = {r["_id"]: r["_answers_sha"] for r in rows}

        # ---- D1 负例：空 root（存在但空）+ --expect-tasks 11 → 非零退出，
        # 不写任何输出（缺陷 1 回归：旧版零方法键 → 循环体零次 → 照样 exit 0）
        empty_root = os.path.join(base, "empty_root")
        os.makedirs(empty_root)
        out = os.path.join(base, "d1.json")
        r = _run(empty_root, out, expect=11)
        assert r.returncode != 0 and "0 个方法键" in (r.stdout + r.stderr), \
            (r.returncode, r.stdout, r.stderr)
        assert _no_output(out), "空 root 门禁失败仍写了输出"
        # D1b：root 路径根本不存在（拼错/挂载缺失）→ 同样 fail closed
        out = os.path.join(base, "d1b.json")
        r = _run(os.path.join(base, "no_such_root"), out, expect=11)
        assert r.returncode != 0 and "0 个方法键" in (r.stdout + r.stderr)
        assert _no_output(out)
        print("D1 PASS  空 root（存在/不存在两种）+ --expect-tasks 11 → "
              "非零退出 + 不写任何输出")

        # ---- D2 负例：root 存在但只有无关 .txt（零方法键）→ 非零退出
        junk_root = os.path.join(base, "junk_root")
        pd = os.path.join(junk_root, "L32768", "pred_t")
        os.makedirs(pd)
        open(os.path.join(pd, "readme.txt"), "w").write("not a pred file")
        out = os.path.join(base, "d2.json")
        r = _run(junk_root, out, expect=11)
        assert r.returncode != 0 and "0 个方法键" in (r.stdout + r.stderr), \
            (r.returncode, r.stdout, r.stderr)
        assert _no_output(out)
        print("D2 PASS  root 只有无关 .txt（零方法键）→ 非零退出 + 不写输出")

        # ---- D3 负例：manifest answers_sha={} 空映射 → schema 前置校验
        # fail closed（旧版 `if i in exp_sha` 会静默跳过全部校验照样写分）
        root, out = _mk_root(base, {f"{TASK}-none-01010000.jsonl": rows})
        mf3 = os.path.join(base, "m3.json")
        json.dump({TASK: {"ids": ids, "answers_sha": {}}}, open(mf3, "w"))
        r = _run(root, out, manifest=mf3)
        assert r.returncode != 0 and "answers_sha" in (r.stdout + r.stderr), \
            (r.returncode, r.stdout, r.stderr)
        assert _no_output(out)
        print("D3 PASS  manifest answers_sha={} 空映射 → 非零退出 + 不写输出")

        # ---- D4 负例：answers_sha 漏一个 ID → fail closed
        shas_miss = dict(list(shas.items())[:-1])
        mf4 = os.path.join(base, "m4.json")
        json.dump({TASK: {"ids": ids, "answers_sha": shas_miss}},
                  open(mf4, "w"))
        r = _run(root, out, manifest=mf4)
        assert r.returncode != 0 and "answers_sha 键集合与 ids 不等" \
            in (r.stdout + r.stderr), (r.returncode, r.stdout)
        assert _no_output(out)
        print("D4 PASS  answers_sha 漏一个 ID → 非零退出 + 不写输出")

        # ---- D5 负例：answers_sha 多一个 ID → fail closed
        shas_extra = dict(shas)
        shas_extra[f"{TASK}:999"] = "0" * 16
        mf5 = os.path.join(base, "m5.json")
        json.dump({TASK: {"ids": ids, "answers_sha": shas_extra}},
                  open(mf5, "w"))
        r = _run(root, out, manifest=mf5)
        assert r.returncode != 0 and "answers_sha 键集合与 ids 不等" \
            in (r.stdout + r.stderr), (r.returncode, r.stdout)
        assert _no_output(out)
        print("D5 PASS  answers_sha 多一个 ID → 非零退出 + 不写输出")

        # ---- D6 负例：manifest ids 重复 → fail closed
        mf6 = os.path.join(base, "m6.json")
        json.dump({TASK: {"ids": ids + [ids[0]], "answers_sha": shas}},
                  open(mf6, "w"))
        r = _run(root, out, manifest=mf6)
        assert r.returncode != 0 and "ids 存在重复" in (r.stdout + r.stderr), \
            (r.returncode, r.stdout)
        assert _no_output(out)
        print("D6 PASS  manifest ids 重复 → 非零退出 + 不写输出")

        # ---- D7 缺陷 3 回归：merged 派生物不得压过同长度更新时间戳的
        # 新原始文件——canonical 必须 = 新文件内容
        # 第一轮：A(5 行, ts 01010000, 全对) + C(3 行坏) → best=A，
        # merged 落盘 = A 内容，score=100
        a_rows, c_rows = rows, _mk_rows(hit=False, n=3)
        root, out = _mk_root(base, {
            f"{TASK}-none-01010000.jsonl": a_rows,
            f"{TASK}-none-01000000.jsonl": c_rows})
        r = _run(root, out, merge=True)
        assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
        canon = os.path.join(root, "L32768", "pred_t",
                             f"{TASK}-none-merged.jsonl")
        assert os.path.exists(canon)
        canon_rows = [json.loads(l) for l in open(canon)]
        assert canon_rows == a_rows, "第一轮 canonical 应为 A 内容"
        assert json.load(open(out))["scores"]["L32768/none"][TASK] == 100.0
        # 第二轮：放入同长度（5 行）更新时间戳的新原始文件 B（全错）→
        # 旧版 merged 的 _ts_of='merged' 字典序恒胜 → stale；新版必须重新
        # 仲裁选 B，canonical 内容 = B，score=0
        b_rows = _mk_rows(hit=False)
        with open(os.path.join(root, "L32768", "pred_t",
                               f"{TASK}-none-01020000.jsonl"), "w") as f:
            for rr in b_rows:
                f.write(json.dumps(rr, ensure_ascii=False) + "\n")
        r = _run(root, out, merge=True)
        assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
        canon_rows = [json.loads(l) for l in open(canon)]
        assert canon_rows == b_rows, \
            f"canonical 未刷新为新原始文件 B（stale-merged 回归）"
        assert "01020000" in r.stdout, "重仲裁打印未提到新文件 B"
        got = json.load(open(out))
        assert got["scores"]["L32768/none"][TASK] == 0.0, \
            f"评分应来自新文件 B（全错）: {got}"
        assert got["sources"]["L32768/none"][TASK]["file"] == \
            f"{TASK}-none-merged.jsonl"
        # 幂等语义：同一原始候选集重跑 → canonical 不变
        sha_before = subprocess.run(
            ["sha256sum", canon], capture_output=True, text=True).stdout
        r = _run(root, out, merge=True)
        assert r.returncode == 0
        sha_after = subprocess.run(
            ["sha256sum", canon], capture_output=True, text=True).stdout
        assert sha_before == sha_after, "同候选集重跑 canonical 应幂等"
        print("D7 PASS  merged 派生物被排除出候选集：新同长度文件 B 胜出，"
              "canonical=B 内容（score 100→0）+ 同候选集幂等")

        # ---- D8 负例：--expect-tasks 11 但只有 10 个任务 → 非零退出
        ten = {}
        for t in ["niah_single_1", "niah_single_2", "niah_single_3",
                  "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
                  "niah_multiquery", "niah_multivalue", "cwe", "fwe"]:
            ten[f"{t}-none-01030000.jsonl"] = [
                dict(r, _id=f"{t}:{i}") for i, r in enumerate(rows)]
        root, out = _mk_root(base, ten)
        r = _run(root, out, expect=11)
        assert r.returncode != 0 and "expect-tasks" in (r.stdout + r.stderr) \
            and "vt" in (r.stdout + r.stderr), (r.returncode, r.stdout)
        assert _no_output(out)
        print("D8 PASS  10/11 任务 + --expect-tasks 11 → 非零退出 + 不写输出")

        # ---- D9 正例：三臂 32K 真实数据（/tmp 副本）走生产正式入口
        # score_ruler_formal.py → 成功 + FULLKV AVG 与 59.38 逐位一致
        # E116e 适配：新正式入口要求 --data-root（源数据 SHA256 身份绑定），
        # legacy _id 补刻改写到 staging 派生目录（原始文件全程只读），
        # 故断言改为「原始拷贝零改动 + 派生副本含 _id」。
        src = os.path.join(REPO, "exp/results_ruler/e109_full_Qwen3-8B",
                           "L32768")
        data_root = os.environ.get(
            "E116E_DATA_ROOT",
            "/home/wangyuanshuo02/sparse-bench/third_party/"
            "KVCache-Factory/data/RULER")
        if not (os.path.isdir(src) and os.path.isdir(
                os.path.join(data_root, "32768"))):
            print("D9 SKIP  本机无真实 32K 三臂数据/源数据目录"
                  "（外部数据集成测试，不计入程序门禁）")
        else:
            real_base = os.path.join(base, "real")
            os.makedirs(real_base)
            shutil.copytree(src, os.path.join(real_base, "L32768"))
            avgs = {}
            for tag, postfix in [("FULLKV", "_E109_FULLKV"),
                                 ("mavg", "_E109_mavg_a0.25_b0.125_g0.625"),
                                 ("aavg", "_E109_aavg_a0_b0_g0")]:
                out = os.path.join(base, f"formal_{tag}.json")
                r = subprocess.run(
                    [sys.executable, "-u", "-m",
                     "benchmark.RULER.score_ruler_formal",
                     "--root", real_base, "--pred-postfix", postfix,
                     "--data-root", data_root,
                     "--out", out],
                    capture_output=True, text=True, cwd=REPO,
                    env={**os.environ, "PYTHONPATH": REPO})
                assert r.returncode == 0 and "DONE" in r.stdout, \
                    (tag, r.returncode, r.stdout[-3000:], r.stderr[-2000:])
                d = json.load(open(out))
                mkey = [k for k in d["scores"]][0]
                scores = d["scores"][mkey]
                assert len(scores) == 11, (tag, scores.keys())
                avg = round(sum(scores.values()) / 11, 2)
                avgs[tag] = avg
                # 036：原始（拷贝）pred 文件零改动——legacy 无 _id 保持在原样
                orig = os.path.join(real_base, "L32768", "pred_E109_FULLKV",
                                    "niah_single_1-none-10090537.jsonl")
                orig_row = json.loads(open(orig).readline())
                assert orig_row.get("_id") is None, \
                    f"原始文件被原地补刻（036 回归）: {orig_row.keys()}"
            assert avgs["FULLKV"] == 59.38, \
                f"FULLKV AVG {avgs['FULLKV']} != 59.38（历史口径 e109_full_ruler32）"
            assert avgs["mavg"] == 59.99 and avgs["aavg"] == 57.33, avgs
            # legacy 补刻发生在 staging 派生目录（成功后改名 {out}.run-*）
            stamped_glob = os.path.join(
                base, "formal_FULLKV.json.run-*", "pred_root", "L32768",
                "pred_E109_FULLKV", "niah_single_1-none-10090537.jsonl")
            stamped_files = glob.glob(stamped_glob)
            assert stamped_files, "派生目录中未找到补刻副本"
            stamped = json.loads(open(stamped_files[0]).readline())
            assert "_id" in stamped and "_answers_sha" in stamped, stamped
            print(f"D9 PASS  三臂真实数据走 score_ruler_formal.py 全部成功："
                  f"AVG FULLKV={avgs['FULLKV']}（59.38 逐位一致）/ "
                  f"mavg={avgs['mavg']} / aavg={avgs['aavg']}；legacy _id "
                  f"补刻在派生目录生效（原始文件零改动）")
            PASS += 1

        # ---- D10：现有 E116c 套件全部用例不回归（11/11）
        r = subprocess.run(
            [sys.executable, "-m", "benchmark.RULER.test_e116c_gate"],
            capture_output=True, text=True, cwd=REPO,
            env={**os.environ, "PYTHONPATH": REPO})
        assert r.returncode == 0 and "E116c ALL PASS (11/11)" in r.stdout, \
            (r.returncode, r.stdout[-3000:], r.stderr[-2000:])
        print("D10 PASS test_e116c_gate.py 11/11 无回归")
        PASS += 9
    finally:
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    test_D()
    # D9 为外部真实数据集成测试（E116e 起缺数据时 SKIP 不计入门禁）
    print(f"\nE116d ALL PASS ({PASS}/10"
          f"{'，D9 SKIP' if PASS == 9 else ''})")
