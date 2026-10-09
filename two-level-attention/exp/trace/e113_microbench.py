#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E113 microbench：sim_greedy 贪心聚类 Python 循环 vs Triton kernel 规模扫描。

== 050 身份闭包修复（TL-E113-BENCH-PROVENANCE-050，GPT 审计 2026-10-09_1727）==
旧版四缺陷已修，重跑性能数据前旧 `results/e113_microbench.json` 保留不动
（实现身份未闭合的历史观测，不撤销也不作为正式性能证据）：
  1. Python reference 默认从**当前 checkout**（`HERE/../..`，即本脚本所在的
     two-level-attention 树）加载，不再硬编码仓库外绝对路径；跨版本 A/B 必须
     显式传 `--ref-root`/`--tri-root`，两侧 git SHA/dirty 与实现文件 SHA256
     全部落 manifest（隐藏绝对路径禁止）。
  2. manifest 记录完整实现身份：git SHA/dirty、tli_indexer.py 与
     greedy_triton.py 双侧 SHA256、生成器参数/seed、torch/triton/CUDA/
     driver/GPU 版本、计时口径、warmup/rep、逐次原始延迟、输出内容 SHA
     （自洽口径：`output_content_sha256` = 剔除该键后规范序列化字节的
     SHA256，验证方读入 JSON → 删该键 → 同参 json.dumps → 比对）。
     发布 = 临时文件 + fsync + 原子 replace，另落 `<out>.sha256` sidecar
     （最终文件字节 SHA）。
  3. correctness fail-closed：assignment / k_live / cnt / sq / 簇心 sums
     任一超预定容差 → 非零退出 + 失败日志（含输入 hash 与身份）落盘
     `<out>.failure.json`，**性能 JSON 不发布**。
  4. singleton 臂 `torch.randn` 显式传入固定 seed 的 generator。

GPU 空闲时跑（默认 GPU0，加 --wait 可轮询等待空闲）。

口径：
  - 数据：base+noise 混合簇结构（与单测同款，覆盖归并路径）+ 纯高斯单例
    极端臂（C/N≈1，最坏 K 扫描量）；
  - 计时：torch.cuda.synchronize() 前后 wall time，Triton 预热一次去 JIT；
  - 对照：E108 probe 实测口径（Python 循环 T≈16K ~5.8s/层，sim=0.9 27s）。

用法：CUDA_VISIBLE_DEVICES=0 python3 e113_microbench.py [--wait]
输出：exp/trace/results/e113_microbench.json（+ .sha256 sidecar）
CPU-only 红绿单测（身份门/发布协议）：test_e113_microbench_identity.py
"""
import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import types

sys.dont_write_bytecode = True   # 只读 import，不落字节码

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
# 050：两侧实现默认都从当前 checkout 推导（HERE/../.. = 本 two-level-attention 树），
# 禁止仓库外隐藏绝对路径；跨版本 A/B 须显式 --ref-root/--tri-root（身份全落盘）。
DEFAULT_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))


# ---------------------------------------------------------------- 身份与发布工具
def file_sha256(path):
    """文件 SHA256（流式读，避免大文件整体驻留）。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha256(x):
    """输入张量内容 SHA256（失败日志/逐 case 输入闭包用）。"""
    return hashlib.sha256(x.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def git_identity(root):
    """root 子树的 git 身份：HEAD SHA + 该子树 dirty 状态（050 manifest 字段）。

    root 非 git 仓库 / git 不可用时显式返回 None（不静默冒充），由调用方
    与审计口径共同裁决——跨版本 A/B 时两侧身份必须可见。"""
    def run(*cmd):
        try:
            p = subprocess.run(["git", "-C", root, *cmd],
                               capture_output=True, text=True, timeout=60)
            return p.stdout.strip() if p.returncode == 0 else None
        except Exception:
            return None
    sha = run("rev-parse", "HEAD")
    if sha is None:
        return {"git_sha": None, "git_dirty": None,
                "note": f"{root} 非 git 仓库或 git 不可用（身份未闭合，须人工核验）"}
    porc = run("status", "--porcelain", "--", ".")   # 只看 root 子树的 dirty
    return {"git_sha": sha, "git_dirty": bool(porc),
            "dirty_files": porc.splitlines()[:20] if porc else []}


def build_identity(ref_root, tri_root):
    """050：双侧实现身份闭包（写入结果 meta.identity）。

    - Python reference = ref_root 下 sparse_attn/indexer/tli_indexer.py 的
      `TLIIndexer._greedy_cluster_pass_python`；
    - Triton candidate = tri_root 下 sparse_attn/indexer/greedy_triton.py
      （生产 kernel 本体；不再经 e113_greedy_triton.py 转发，直接按文件加载，
      使被测文件路径/SHA 显式可控）。"""
    return {
        "ref_root": os.path.abspath(ref_root),
        "tri_root": os.path.abspath(tri_root),
        "same_root": os.path.abspath(ref_root) == os.path.abspath(tri_root),
        "ref": {
            "git": git_identity(ref_root),
            "impl_file": os.path.join(ref_root, "sparse_attn", "indexer", "tli_indexer.py"),
            "impl_sha256": file_sha256(os.path.join(ref_root, "sparse_attn", "indexer", "tli_indexer.py")),
        },
        "tri": {
            "git": git_identity(tri_root),
            "impl_file": os.path.join(tri_root, "sparse_attn", "indexer", "greedy_triton.py"),
            "impl_sha256": file_sha256(os.path.join(tri_root, "sparse_attn", "indexer", "greedy_triton.py")),
        },
        "microbench_script_sha256": file_sha256(os.path.abspath(__file__)),
        "note": "050：双侧实现文件 SHA + git 身份全记录；跨版本 A/B 须显式 root 且核验各自 commit/dirty",
    }


def collect_versions():
    """050：运行环境版本闭包（torch/triton/CUDA/driver/GPU）。"""
    v = {"torch": torch.__version__, "cuda": torch.version.cuda}
    try:
        import triton
        v["triton"] = triton.__version__
    except Exception:
        v["triton"] = None
    drv = None
    try:
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30)
        if p.returncode == 0 and p.stdout.strip():
            drv = p.stdout.strip().splitlines()[0]
    except Exception:
        pass
    v["driver"] = drv
    v["gpu"] = torch.cuda.get_device_name(0)
    return v


def canonical_bytes(results):
    """输出内容 SHA 的规范字节：剔除 output_content_sha256 键后按发布参数
    （indent=1, ensure_ascii=False，dict 保序）序列化。验证方：读入 JSON →
    删该键 → 同参序列化 → 比对 SHA（防篡改自洽口径，无自指问题）。"""
    r2 = {k: v for k, v in results.items() if k != "output_content_sha256"}
    return json.dumps(r2, indent=1, ensure_ascii=False).encode("utf-8")


def output_content_sha256(results):
    return hashlib.sha256(canonical_bytes(results)).hexdigest()


def verify_output_sha256(path):
    """发布文件的内容 SHA 校验（供复验脚本/审计调用）。"""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    claimed = d.pop("output_content_sha256", None)
    return claimed is not None and claimed == output_content_sha256(d)


def atomic_write_bytes(path, data):
    """050 原子发布：临时文件 + flush + fsync + os.replace。"""
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ---------------------------------------------------------------- 实现加载
def load_sparse_attn(name, root):
    """root/sparse_attn 作为独立命名空间包加载（同 test_e110 手法）。"""
    mod = types.ModuleType(name)
    mod.__path__ = [os.path.join(root, "sparse_attn")]
    sys.modules[name] = mod
    return importlib.import_module(f"{name}.indexer")


def load_triton_kernel(tri_root):
    """按显式路径加载生产 Triton kernel 本体（050：路径可控可记 SHA）。"""
    path = os.path.join(tri_root, "sparse_attn", "indexer", "greedy_triton.py")
    if not os.path.isfile(path):
        sys.exit(f"[FATAL] Triton 实现不存在：{path}")
    spec = importlib.util.spec_from_file_location("e113_bench_triton_kernel", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------- 数据生成
def gen_structured(T, H, dd, seed, n_base=None, noise=0.10):
    """混合簇结构（归并路径）。n_base=T//8 时 C/N≈0.125。固定 seed generator。"""
    g = torch.Generator(device="cuda").manual_seed(seed)
    n_base = n_base or max(T // 8, 1)
    base = torch.randn(n_base, H, dd, generator=g, device="cuda")
    idx = torch.randint(0, n_base, (T,), generator=g, device="cuda")
    return (base[idx] + noise * torch.randn(T, H, dd, generator=g, device="cuda")).contiguous()


def gen_singleton(T, H, dd, seed):
    """纯高斯单例极端臂（C/N≈1，最坏 K 扫描量）。

    050 修复 4：显式固定 generator/seed——旧版裸 torch.randn 使该臂输入
    不可重放；seed 记入 meta.generators。"""
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(T, H, dd, generator=g, device="cuda").contiguous()


# ---------------------------------------------------------------- 计时与正确性门
def bench(fn, warmup=1, rep=3):
    """同步口径计时：预热 warmup 次后 rep 次取中位。

    050：返回 (中位数, 逐次原始延迟列表)——manifest 要求逐次样本全记录。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(rep):
        t0 = time.time()
        fn()
        torch.cuda.synchronize()
        ts.append(time.time() - t0)
    ts.sort()
    return ts[len(ts) // 2], ts


def check_pair(r, t, tag, atol_cnt=1e-6, rtol_sq=1e-5, atol_sq=1e-5, atol_sums=1e-5):
    """050 fail-closed 正确性门：返回错误列表（空 = 通过）。

    检查项与容差对齐 exp/trace/test_e113_greedy_triton.py::cmp_state：
      assignment 逐位、k_live 相等、cnt/sq/簇心 sums 容差；
    任一失败由调用方 fail-closed（非零退出 + 不发布性能 JSON）。"""
    errs = []
    sums_r, cnt_r, sq_r, kl_r, a_r = r
    sums_t, cnt_t, sq_t, kl_t, a_t = t
    if not torch.equal(kl_r, kl_t):
        errs.append(f"{tag}: k_live 不一致 {kl_r.tolist()} vs {kl_t.tolist()}")
    mism = int((a_r != a_t).sum())
    if mism:
        errs.append(f"{tag}: assign mismatch {mism}/{a_r.numel()} 位")
    live = int(kl_r.max().item())
    if live <= 0:
        errs.append(f"{tag}: k_live max=0，无活簇（数据/门异常）")
        return errs
    if not torch.allclose(cnt_r[:, :live], cnt_t[:, :live], atol=atol_cnt):
        errs.append(f"{tag}: cnt 超容差（atol={atol_cnt}）")
    if not torch.allclose(sq_r[:, :live], sq_t[:, :live], rtol=rtol_sq, atol=atol_sq):
        errs.append(f"{tag}: sq 超容差（rtol={rtol_sq}, atol={atol_sq}）")
    if not torch.allclose(sums_r[:, :live], sums_t[:, :live], atol=atol_sums):
        errs.append(f"{tag}: 簇心 sums 超容差（atol={atol_sums}）")
    return errs


GATE_TOLERANCES = {"assign": "逐位相等", "k_live": "逐位相等",
                   "cnt": {"atol": 1e-6}, "sq": {"rtol": 1e-5, "atol": 1e-5},
                   "sums": {"atol": 1e-5}}


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait", action="store_true", help="轮询等 GPU 空闲再跑")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "e113_microbench.json"))
    ap.add_argument("--ref-root", default=DEFAULT_ROOT,
                    help=f"Python 参考实现所在 two-level-attention 根（默认=当前 checkout "
                         f"{DEFAULT_ROOT}；跨版本 A/B 须显式传入，两侧 git/文件身份全落 manifest）")
    ap.add_argument("--tri-root", default=DEFAULT_ROOT,
                    help="Triton candidate 实现所在 two-level-attention 根（同上）")
    ap.add_argument("--structured-seed", type=int, default=7,
                    help="structured_c8 臂 generator seed（记入 manifest）")
    ap.add_argument("--singleton-seed", type=int, default=7,
                    help="singleton 臂 generator seed（050 修复 4，记入 manifest）")
    args = ap.parse_args()

    ref_root = os.path.abspath(args.ref_root)
    tri_root = os.path.abspath(args.tri_root)
    if ref_root != DEFAULT_ROOT or tri_root != DEFAULT_ROOT:
        # 050：跨版本 A/B 允许，但必须显式且身份可见——调用者负责核验两侧 commit/dirty
        print(f"[警告] 跨版本 A/B：ref_root={ref_root} tri_root={tri_root} "
              f"（默认当前 checkout={DEFAULT_ROOT}）；两侧 git SHA/dirty 与文件 SHA256 "
              f"将全部记录进 manifest，请自行核验两侧 commit 状态", flush=True)

    torch.set_grad_enabled(False)
    assert torch.cuda.is_available(), "需 GPU"
    if args.wait:
        while True:
            free, total = torch.cuda.mem_get_info()
            if (total - free) / total < 0.5:   # 已用 <50% 视为空闲
                break
            print(f"GPU 占用高（free {free/1e9:.0f}GB/{total/1e9:.0f}GB），60s 后重试…", flush=True)
            time.sleep(60)

    # ---- 实现加载（050：默认双侧同一 checkout；显式 root 走 A/B）----
    IDX = load_sparse_attn("sparse_attn_e113bench", ref_root)
    greedy_ref = IDX.TLIIndexer._greedy_cluster_pass_python   # E113b：参考实现固定取 Python 路径
    TRI = load_triton_kernel(tri_root)
    greedy_build_triton = TRI.greedy_build_triton

    identity = build_identity(ref_root, tri_root)
    print("[identity] " + json.dumps({
        "same_root": identity["same_root"],
        "ref_git": identity["ref"]["git"], "ref_sha": identity["ref"]["impl_sha256"][:16],
        "tri_git": identity["tri"]["git"], "tri_sha": identity["tri"]["impl_sha256"][:16],
    }, ensure_ascii=False), flush=True)

    H, dd = 8, 32   # Qwen3-8B kv-head × tail32 口径
    results = {"meta": {
        "probe": "E113 microbench：sim_greedy Python 循环 vs Triton kernel",
        "gpu": torch.cuda.get_device_name(0),
        "timing": ("cuda synchronize wall；python 臂 warmup=0 rep=1（单次），"
                   "triton 臂 warmup=1 rep=3 取中位；Triton 预热含去 JIT"),
        "warmup_rep": {"python": {"warmup": 0, "rep": 1},
                       "triton": {"warmup": 1, "rep": 3}},
        "e108_probe_ref": "Python 循环 T≈16K 实测 ~5.8s/层（sim=0.9 时 27s）",
        "versions": collect_versions(),
        "identity": identity,
        "generators": {
            "structured_c8": {"kind": "base+noise 混合簇", "n_base_rule": "max(T//8,1)",
                              "noise": 0.10, "seed": args.structured_seed,
                              "device": "cuda"},
            "singleton": {"kind": "纯高斯单例", "seed": args.singleton_seed,
                          "device": "cuda",
                          "note": "050 修复 4：显式固定 generator/seed"},
        },
        "correctness_gate": {"mode": "fail-closed（050 修复 3）",
                             "checks": ["assign 逐位", "k_live 相等", "cnt/sq/簇心容差"],
                             "tolerances": GATE_TOLERANCES,
                             "on_fail": "非零退出 + <out>.failure.json（含输入 hash），性能 JSON 不发布"},
        "started": time.strftime("%F %T"),
    }, "cases": []}

    for T in (1024, 4096, 8192, 16384, 32768):
        for sim, tag, gen in ((0.9, "structured_c8", lambda: gen_structured(T, H, dd, args.structured_seed)),
                              (0.9, "singleton", lambda: gen_singleton(T, H, dd, args.singleton_seed))):
            x = gen()
            in_sha = tensor_sha256(x)   # 050：逐 case 输入内容闭包

            def run_ref():
                return greedy_ref(x, sim, x.new_zeros(H, T, dd), x.new_zeros(H, T),
                                  x.new_zeros(H, T),
                                  torch.zeros(H, dtype=torch.long, device="cuda"))

            def run_tri():
                return greedy_build_triton(x, sim, chunk=16384)
            # Python 参考（慢，只跑 1 次且兼做正确性基准）
            t_ref, ref_samples = bench(run_ref, warmup=0, rep=1)
            # Triton（预热去 JIT，中位×3；chunk=16384 为 E113 调优最优）
            t_tri, tri_samples = bench(run_tri, warmup=1, rep=3)
            # ---- 050 fail-closed 正确性门（任一失败即中断，不发布性能 JSON）----
            r = run_ref()
            t = run_tri()
            errs = check_pair(r, t, f"T={T}/{tag}")
            mism = int((r[4] != t[4]).sum())
            live = int(r[3].max().item())
            rec = {"T": T, "H": H, "dd": dd, "sim": sim, "data": tag,
                   "input_sha256": in_sha,
                   "clusters_max_head": live, "C_over_N": round(live / T, 4),
                   "assign_mismatch": mism,
                   "wall_s_python": round(t_ref, 3),
                   "wall_s_python_samples": [round(s, 4) for s in ref_samples],
                   "wall_s_triton": round(t_tri, 4),
                   "wall_s_triton_samples": [round(s, 5) for s in tri_samples],
                   "us_per_token_python": round(1e6 * t_ref / T, 2),
                   "us_per_token_triton": round(1e6 * t_tri / T, 2),
                   "speedup": round(t_ref / t_tri, 1)}
            if errs:
                # fail-closed：失败日志落盘（含输入 hash + 身份），非零退出，性能 JSON 不发布
                rec["gate_errors"] = errs
                fail_doc = {"status": "failed",
                            "gate": "fail-closed（TL-E113-BENCH-PROVENANCE-050 修复 3）",
                            "identity": identity,
                            "started": results["meta"]["started"],
                            "timestamp": time.strftime("%F %T"),
                            "failed_case": rec, "errors": errs}
                fail_path = args.out + ".failure.json"
                os.makedirs(os.path.dirname(fail_path), exist_ok=True)
                atomic_write_bytes(fail_path, json.dumps(
                    fail_doc, indent=1, ensure_ascii=False).encode("utf-8"))
                print("[GATE-FAIL] " + json.dumps(errs, ensure_ascii=False), flush=True)
                print(f"correctness 门失败 → 不发布性能 JSON；失败日志 -> {fail_path}", flush=True)
                sys.exit(1)
            print(json.dumps(rec, ensure_ascii=False), flush=True)
            results["cases"].append(rec)
            del x, r, t
            torch.cuda.empty_cache()

    results["finished"] = time.strftime("%F %T")
    # ---- 050 原子发布：临时文件 + fsync + replace；内容 SHA 自洽 + sidecar ----
    results["output_content_sha256"] = output_content_sha256(results)
    final_bytes = json.dumps(results, indent=1, ensure_ascii=False).encode("utf-8")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    atomic_write_bytes(args.out, final_bytes)
    atomic_write_bytes(args.out + ".sha256",
                       (hashlib.sha256(final_bytes).hexdigest() + "\n").encode("utf-8"))
    assert verify_output_sha256(args.out), "发布后内容 SHA 自校验失败（不应发生）"
    print(f"\nsaved -> {args.out}（content_sha={results['output_content_sha256'][:16]}… "
          f"+ .sha256 sidecar）", flush=True)


if __name__ == "__main__":
    main()
