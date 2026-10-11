# -*- coding: utf-8 -*-
"""E123 cavg 小试配对判决脚本 v2（083 补链口径：可复验证据绑定）。

回答用户问题（S-T018）：cavg 的 (0.25,0.5,0.125) 与 (0.126,0.126,γ off
自由竞争) 能否优于海选胜出的相关方法（mavg 冠军 .25/.125/.625）。

判决规则（不预设方向）：
  cavg 臂 5 任务配对平均差（vs mavg 新口径锚点）显著为正（bootstrap
  95% CI 不含 0）→ GO，冠军配置换人进 E116b 全量；否则维持 mavg。

v2（TL-E123-VERDICT-PROVENANCE-083）：
  - input_manifest：20 格逐文件 {basename, rows, sha256}——汇总分数与
    原始预测的绑定不再缺位；
  - bootstrap 协议显式化（stdlib random.Random、B、seed、任务顺序、
    percentile 取法），seed 不再被当成「唯一可复验协议」的全部；
  - 结论措辞按 084 降格：CI 含 0 = 「未证明优于 / 未检出差异」，
    不写「持平/等价」。

Caveat（如实记录）：
  - cavg_off 臂未归一化混池（far GQA group-sum 点积 vs near softmax
    概率，GPT 7c31909 量纲更正）：其结果按「未归一化混池实测」口径报告。
  - cavg_off 臂 musique/repobench 两个文件带 _h<hash10> 尾段（081 合入
    后跑的），前三任务不带（合入前）；081 只改身份记录不改选择行为，
    臂内行为同质，判决可用。
  - fullkv 为锚点参照，不参与判决。
"""
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys

REPO = "/home/wangyuanshuo02/sglang/two-level-attention"
PY = sys.executable
# 原始数据默认用入库副本（083 可复验链）；E123_RAW_DIR 可覆盖指回 /tmp 原跑目录
OUT_ROOT = os.environ.get(
    "E123_RAW_DIR",
    "/home/wangyuanshuo02/sglang/two-level-attention/exp/trace/results/e123_trial_raw")
# 注意：run_eval 会把 result.json 写回 OUT_ROOT/pred_{arm}/——对入库副本
# 重跑时会原地刷新 result.json（同输入同输出，git diff 应为零）。
ARMS = ["mavg", "cavg_g", "cavg_off", "fullkv"]
TASKS = ["qasper", "hotpotqa", "gov_report", "musique", "repobench"]
EXPECT = {"qasper": 200, "hotpotqa": 200, "gov_report": 200,
          "musique": 200, "repobench": 500}
B, SEED = 10000, 42
# 重建判决落盘位置独立于原始数据目录（不污染 raw 目录）
OUT_JSON = os.environ.get("E123_OUT_JSON", "/tmp/e123_trial_verdict_rebuild.json")


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_lines(path):
    if not os.path.exists(path):
        return -1
    with open(path) as f:
        return sum(1 for _ in f)


def find_pred(arm, task):
    """臂目录下以 {task}- 开头的唯一 jsonl（容忍 _h 尾段口径混装）。"""
    d = os.path.join(OUT_ROOT, f"pred_{arm}")
    hits = [f for f in os.listdir(d)
            if f.startswith(task + "-") and f.endswith(".jsonl")]
    if len(hits) != 1:
        return None
    return os.path.join(d, hits[0])


def run_eval(arm_dir, label):
    cmd = [PY, "-m", "benchmark.LongBench.eval",
           "--output-path", arm_dir,
           "--expect-count", "200"]  # 行数下限门禁；逐文件精确行数另行对账
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                       env={**os.environ,
                            "PYTHONPATH": REPO + ":/tmp/e117_extra_pkgs"})
    if r.returncode != 0:
        print(f"[EVAL-FAIL] {label}: rc={r.returncode}")
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        sys.exit(1)
    result = json.load(open(os.path.join(arm_dir, "result.json")))
    scores = {}
    for fn, v in result.items():
        if fn == "_meta" or not fn.endswith(".jsonl"):
            continue
        scores[fn.split("-")[0]] = v["score"]
    return scores


def paired_bootstrap(deltas, B=B, seed=SEED):
    """percentile CI：任务级有放回重采样，stdlib random.Random(seed)。"""
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(B):
        s = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(s) / n)
    means.sort()
    return means[int(0.025 * B)], means[int(0.975 * B) - 1]


def main():
    # 1) 完成度对账 + 输入 manifest（083：逐文件 sha256/行数绑定）
    manifest = []
    for arm in ARMS:
        for t in TASKS:
            p = find_pred(arm, t)
            n = count_lines(p) if p else -1
            if n != EXPECT[t]:
                sys.exit(f"[ABORT] {arm}/{t}: rows={n} != {EXPECT[t]}")
            manifest.append({
                "arm": arm, "task": t,
                "basename": os.path.basename(p),
                "rows": n, "expect": EXPECT[t],
                "sha256": sha256_of(p)})
    print(f"[READY] 四臂 5 任务行数精确对账通过（repobench=500 其余=200），"
          f"manifest {len(manifest)} 格")

    # 2) 官方打分（四臂）
    scores = {}
    for arm in ARMS:
        print(f"=== 打分 {arm} ===")
        scores[arm] = run_eval(os.path.join(OUT_ROOT, f"pred_{arm}"), arm)

    # 3) 配对差值（vs mavg 新口径锚点）+ bootstrap CI
    verdict = {"per_task": {}, "input_manifest": manifest, "meta": {
        "champion_ref": "mavg(.25,.125,.625) 新口径（R01+B10+B04 后）",
        "cavg_g": "(.25,.5,.125)",
        "cavg_off": "(.126,.126, gamma off 自由竞争，未归一化混池 caveat)",
        "fullkv": "锚点参照，不参与判决",
        "n_tasks": 5,
        "bootstrap": {
            "unit": "任务级配对差（arm−mavg）有放回重采样",
            "B": B, "seed": SEED,
            "rng": "stdlib random.Random（Mersenne Twister）",
            "ci": "percentile 95%：sorted means[int(0.025*B)], means[int(0.975*B)-1]",
            "task_order": TASKS},
        "scorer": "benchmark.LongBench.eval（官方 scorer，difflib 后端）",
        "python": platform.python_version(),
        "provenance_v2": "TL-E123-VERDICT-PROVENANCE-083 补链："
                         "原始预测入 exp/trace/results/e123_trial_raw/，"
                         "本 JSON 与其 sha256 逐格绑定"}}
    for arm in ["cavg_g", "cavg_off", "fullkv"]:
        deltas = []
        print(f"\n=== {arm} vs mavg 配对 ===")
        for t in TASKS:
            a, m = scores[arm][t], scores["mavg"][t]
            d = round(a - m, 4)
            deltas.append(d)
            verdict["per_task"][f"{arm}/{t}"] = {
                "arm": a, "mavg": m, "delta": d}
            print(f"  {t:12s} {arm}={a:.4f}  mavg={m:.4f}  delta={d:+.4f}")
        avg = sum(deltas) / len(deltas)
        lo, hi = paired_bootstrap(deltas)
        sig = "显著" if (lo > 0 or hi < 0) else "不显著"
        verdict[f"{arm}_avg_delta"] = round(avg, 4)
        verdict[f"{arm}_ci95"] = [round(lo, 4), round(hi, 4)]
        verdict[f"{arm}_significant"] = lo > 0 or hi < 0
        print(f"  平均差 {avg:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  → {sig}")

    # 4) 判决（不预设方向；084 措辞：CI 含 0 = 未证明优于，非「持平」）
    go_arms = [a for a in ("cavg_g", "cavg_off")
               if verdict[f"{a}_significant"]
               and verdict[f"{a}_avg_delta"] > 0]
    verdict["verdict"] = ("GO: " + "+".join(go_arms) + " 显著优于 mavg，冠军换人"
                          if go_arms else
                          "NO-GO: cavg 两臂均未证明优于 mavg 冠军"
                          "（cavg_off 显著更差；cavg_g 未达换冠军证据门槛），"
                          "维持 mavg")
    print("\n判决：" + verdict["verdict"])

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(verdict, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"落盘 {OUT_JSON}")


if __name__ == "__main__":
    main()
