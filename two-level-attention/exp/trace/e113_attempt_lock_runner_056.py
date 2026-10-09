#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E113 056 并发锁验收的子进程 runner（被 test_e113_attempt_lock_056.py Popen 调用）。

本脚本是一个**真实独立进程**：加载生产 e113_microbench，stub 掉 GPU/张量/
实现加载（手法与 test_e113_state_machine_054_055.py / e113_failclosed_inject.py
一致——只替换环境桩，**不复制生产逻辑、不绕过生产状态机**），然后调用生产
main() 走真实的「隔离 → 执行 → 发布 → 终检」路径（含 056 flock）。

模式：
  success       tri 桩返回与 ref 一致的终态 → 生产 main() 走完整成功发布
                （attempt 状态机 + flock + sidecar + 终检全程真实执行）；
  failure       tri 桩篡改 assignment 一位 → 首个 case 即 fail-closed：
                failure.json 落盘 + 终检 + SystemExit(1)；
  crash-between 生产发布路径中，success JSON 已原子替换、sidecar 尚未写时
                用 os._exit(9) 硬杀本进程（不经 finally、不显式释放 flock——
                专测内核自动释放 + 下一 attempt 按隔离协议恢复一致）；
  lock-hold     只 acquire 生产锁后 sleep(--hold) 再释放（不跑 main()），
                供父进程做非阻塞互斥探测（LOCK_NB 必须失败）。

并发对齐：--barrier-dir + --barrier-count N——本进程创建 ready 文件后轮询
直到 N 个 ready 文件就绪才进入 main()（消除 import torch 耗时抖动，保证
两进程在同一时刻进入「quarantine → 执行」窗口，使修复前的竞态在红测中
确定性暴露：无锁时双方 quarantine 都看到空目录 → 后到者无隔离记录）。

用法（由测试调用，也可手动）：
  python3 e113_attempt_lock_runner_056.py --out /tmp/x/r.json --mode success \
      --barrier-dir /tmp/x/bar --barrier-count 2 --bench-sleep 0.25
"""
import argparse
import importlib.util
import json
import os
import sys
import time
import types

import torch   # 本 runner 只用 CPU 张量做 gate 桩（GPU 全部 stub 掉，不占卡）

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "e113_microbench.py")


def fake_state(tamper):
    """最小贪心终态（CPU 张量，与 check_pair 解包序一致：
    (sums, cnt, sq, k_live, assign)）。tamper=True 篡改 assignment 一位。"""
    g = torch.Generator().manual_seed(7)
    sums = torch.randn(2, 4, 8, generator=g)
    cnt = torch.rand(2, 4, generator=g) * 4 + 1
    sq = (sums * sums).sum(-1)
    kl = torch.tensor([2, 3], dtype=torch.long)
    assign = torch.stack([torch.randint(0, 2, (16,), generator=g),
                          torch.randint(0, 3, (16,), generator=g)]).long()
    if tamper:
        assign = assign.clone()
        assign.view(-1)[0] = 99999   # 不可能的簇索引，必 mismatch
    return (sums.clone(), cnt.clone(), sq.clone(), kl.clone(), assign.clone())


def barrier_wait(barrier_dir, count, timeout=60.0):
    """N 进程就绪栅栏：各创建 <pid>.ready，轮询直到目录内 ready 文件数 >= count。"""
    os.makedirs(barrier_dir, exist_ok=True)
    open(os.path.join(barrier_dir, f"{os.getpid()}.ready"), "wb").close()
    if count <= 1:
        return
    t0 = time.time()
    while time.time() - t0 < timeout:
        n = len([f for f in os.listdir(barrier_dir) if f.endswith(".ready")])
        if n >= count:
            return
        time.sleep(0.01)
    print(f"[RUNNER {os.getpid()}] 栅栏超时（{timeout}s），继续执行", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", required=True,
                    choices=["success", "failure", "crash-between", "lock-hold"])
    ap.add_argument("--barrier-dir", default=None)
    ap.add_argument("--barrier-count", type=int, default=1)
    ap.add_argument("--bench-sleep", type=float, default=0.25,
                    help="桩 bench 每次调用 sleep 秒数——拉开 quarantine 与发布之间的"
                         "窗口（红测确定性 / 绿测覆盖持锁等待期）")
    ap.add_argument("--hold", type=float, default=2.0, help="lock-hold 模式持锁秒数")
    args = ap.parse_args()

    # ---- 加载生产模块（唯一模块名，避免 sys.modules 串台）----
    spec = importlib.util.spec_from_file_location(f"e113_prod_mod_{os.getpid()}", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    if args.mode == "lock-hold":
        # 只测生产锁原语的跨进程互斥，不跑 main()
        fd = m.acquire_attempt_output_lock(args.out)
        print(f"[RUNNER {os.getpid()}] lock-hold：已获锁 {m.attempt_lock_path(args.out)}，"
              f"sleep {args.hold}s", flush=True)
        time.sleep(args.hold)
        m.release_attempt_output_lock(fd)
        print(f"[RUNNER {os.getpid()}] lock-hold：已释放", flush=True)
        return 0

    # ---- 并发栅栏（进入 main() 前对齐双方）----
    if args.barrier_dir:
        barrier_wait(args.barrier_dir, args.barrier_count)

    fail_mode = args.mode == "failure"

    class _FakeTLI:
        @staticmethod
        def _greedy_cluster_pass_python(*a, **k):
            return fake_state(False)

    class _FakeIDX:
        TLIIndexer = _FakeTLI

    fake_tri = types.SimpleNamespace(
        greedy_build_triton=lambda *a, **k: fake_state(fail_mode))
    fake_x = torch.randn(4, 2, 8)

    def fake_bench(fn, warmup=1, rep=3):
        # 桩样本 + 固定 sleep：保证 quarantine（main 入口附近）与发布之间有
        # 可观的时间窗口，双方（无锁时）都在对方发布前完成隔离检查
        time.sleep(args.bench_sleep)
        return [4_500_000, 3_100_000, 3_800_000] if rep == 3 else [123_456_789]

    real_zeros = torch.zeros

    def cpu_zeros(*a, **k):
        if k.get("device") == "cuda":
            k = dict(k)
            k["device"] = "cpu"
        return real_zeros(*a, **k)

    # ---- crash-between：JSON 已替换、sidecar 未写时硬杀本进程 ----
    if args.mode == "crash-between":
        real_awb = m.atomic_write_bytes

        def awb_hooked(path, data):
            if path == args.out + ".sha256":
                print(f"[RUNNER {os.getpid()}] crash-between：success JSON 已发布、"
                      f"sidecar 未写 → os._exit(9) 模拟进程死亡（持锁硬杀）", flush=True)
                os._exit(9)   # 硬退出：不经 finally、不显式释放 flock
            real_awb(path, data)
        m.atomic_write_bytes = awb_hooked

    # ---- 环境桩（GPU 侧；生产状态机/发布协议/锁 全程真实执行）----
    torch.cuda.is_available = lambda: True
    torch.cuda.synchronize = lambda *a, **k: None
    torch.cuda.empty_cache = lambda: None
    torch.cuda.get_device_name = lambda *a, **k: "stub-cpu-gpu"
    torch.zeros = cpu_zeros
    m.load_sparse_attn = lambda *a, **k: _FakeIDX()
    m.load_triton_kernel = lambda *a, **k: fake_tri
    m.gen_structured = lambda *a, **k: fake_x.clone()
    m.gen_singleton = lambda *a, **k: fake_x.clone()
    m.bench = fake_bench

    sys.argv = ["e113_microbench.py", "--out", args.out, "--Ts", "1024"]
    print(f"[RUNNER {os.getpid()}] mode={args.mode} out={args.out} 进入生产 main()",
          flush=True)
    try:
        m.main()
        print(f"[RUNNER {os.getpid()}] main 正常返回（成功发布）", flush=True)
        return 0
    except SystemExit as e:
        print(f"[RUNNER {os.getpid()}] SystemExit code={e.code}", flush=True)
        return e.code if isinstance(e.code, int) else 1


if __name__ == "__main__":
    sys.exit(main())
