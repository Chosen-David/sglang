#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""059 生产者侧并发/中断验收 runner（被 test_e119_yarn_binding_059_060_061.py
以真实子进程 Popen 调用；与 exp/trace/e113_attempt_lock_runner_056.py 同款
手法——只替换环境桩，**不复制生产逻辑、不绕过生产提交协议**）。

stub 范围（生产路径全部真实执行）：
  - transformers：tokenizer（encode → CPU 张量 / decode → 确定性 stub 预测
    文本，内容随 --pred-tag 变化便于区分代际）/ model（返回零 logits，
    argmax 恒 0，走满 MAX_GEN 解码循环）/ config；
  - GPU：torch.Tensor.to("cuda") 就地返回 CPU 张量（无卡环境可跑）、
    torch.cuda.empty_cache no-op；
  - sparse_attn 接线：register_patch / get_metrics / get_method_name_with_info
    为轻桩（与生成协议无关的旁路）。

生产路径（全部真实）：resolve_yarn_config → acquire_output_lock（056 口径
flock）→ stage_yarn_generation（不可变临时 {out}.gen-{attempt_id}）→ 生成
循环逐行写 → fout.close → SHA256/行数计算 → build_yarn_receipt（v2
status=complete 同代绑定）→ stage_yarn_receipt → commit_yarn_generation
（单次原子提交：预测先 replace、回执后 replace）→ release_output_lock。

模式：
  success        stub 生成 --rows 行预测 → 完整成功提交（v2 回执）；
  crash-mid      生成循环写到第 --crash-after 行后 os._exit(9) 硬杀
                 （模拟生成中途进程死亡——最终路径必须保持上一代完整
                 产物，partial 只留在临时 generation，不污染
                 {task}-*.jsonl glob）；
  crash-between  提交点两步之间死亡：预测已 os.replace 到最终路径、
                 完成回执尚未 replace 时 os._exit(9)（留下「新预测配旧
                 回执」的失配代际，供消费侧同代绑定 fail-closed 验收）；
  lock-hold      只 acquire 生产锁后 sleep(--hold) 再释放（不跑 main()），
                 供父进程做非阻塞互斥探测（LOCK_NB 必须失败）。

并发对齐：--barrier-dir + --barrier-count N——本进程创建 ready 文件后
轮询直到 N 个 ready 就绪才进入 main()（消除 import/stub 耗时抖动，保证
两进程同时进入「取锁 → 生成」窗口）。

用法（由测试调用，也可手动）：
  python3 benchmark/RULER/e119_yarn_producer_runner_059.py \
      --out-dir /tmp/x/out --data-root /tmp/x/data --mode success \
      --pred-tag A --rows 3
"""
import argparse
import importlib
import json
import os
import sys
import time
import types

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


def barrier_wait(barrier_dir, count, timeout=60.0):
    """N 进程就绪栅栏：各创建 <pid>.ready，轮询直到目录内 ready 数 >= count。"""
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
    print(f"[RUNNER {os.getpid()}] 栅栏超时（{timeout}s），继续执行",
          flush=True)


def install_stubs(pred_tag):
    """安装环境桩并加载生产 pred_ruler 模块（生产路径不 stub）。

    transformers 在导入 pred_ruler 之前整体替换（干净 CPU 环境无
    transformers 依赖也可跑本 runner）；torch 只改 .to 的 cuda 分支与
    cuda.empty_cache（GPU 桩），张量语义本身不 stub。"""
    import torch

    state = {"row": 0}

    class _Tok:
        # eos_token_id=7：与 stub 预测 argmax=0 不同 → 解码循环走满
        # MAX_GEN（生成行数真实反映 --rows）
        eos_token_id = 7

        def __call__(self, text, **k):
            # 生产调用链：input_ids = tokenizer(row["input"],
            # truncation=False, return_tensors="pt").input_ids
            return types.SimpleNamespace(
                input_ids=torch.tensor([[1, 2, 3, 4]]))

        @staticmethod
        def encode(text, **k):
            # 生产调用链：nl = tokenizer.encode("\n",
            # add_special_tokens=False)（直接下标取末 token id）
            return [10]

        def decode(self, ids, **k):
            state["row"] += 1
            # 确定性 stub 预测：内容随 --pred-tag 变化（区分代际）
            return f"stub-pred-{pred_tag}-{state['row'] - 1}"

    class _Model:
        def __call__(self, input_ids=None, past_key_values=None,
                     use_cache=True, **k):
            n = int(input_ids.shape[-1])
            return types.SimpleNamespace(
                logits=torch.zeros(1, n, 8),   # argmax 恒 0
                past_key_values=object())

        # 生产调用链：from_pretrained(...).eval() —— stub 同链返回自身
        def eval(self):
            return self

    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=
                                            staticmethod(lambda *a, **k:
                                                         _Tok())),
        AutoConfig=types.SimpleNamespace(from_pretrained=
                                          staticmethod(lambda *a, **k:
                                                       types.SimpleNamespace())),
        AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=
                                                   staticmethod(
                                                       lambda *a, **k:
                                                       _Model())),
    )
    # 真 transformers 在位（本机/E109 派单机）：只覆写 pred_ruler 顶层
    # `from transformers import Auto*` 消费到的三个类（sparse_attn.patches
    # 的 `transformers.models.qwen3` 导入链保持真实可用）；缺 transformers
    # 的裸环境：整体 fake 模块 + sparse_attn.patches 一并 stub（该桩与
    # 生成/提交协议无关，register_patch 本就被覆写为 no-op）。
    try:
        import transformers  # noqa: F401
        transformers.AutoTokenizer = fake_transformers.AutoTokenizer
        transformers.AutoConfig = fake_transformers.AutoConfig
        transformers.AutoModelForCausalLM = \
            fake_transformers.AutoModelForCausalLM
    except Exception:
        sys.modules["transformers"] = fake_transformers
        sys.modules["sparse_attn.patches"] = types.SimpleNamespace(
            register_patch=lambda model, args: None)

    # GPU 桩：.to("cuda") 就地返回（无卡环境可跑）；其余 .to 语义不变
    _orig_to = torch.Tensor.to

    def _cpu_to(self, *a, **k):
        dev = a[0] if a else k.get("device")
        if isinstance(dev, str) and dev.startswith("cuda"):
            return self
        return _orig_to(self, *a, **k)

    torch.Tensor.to = _cpu_to
    torch.cuda.empty_cache = lambda: None

    # 加载生产模块（真实包导入——sys.path 指向仓库根）
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    pred_ruler = importlib.import_module("benchmark.RULER.pred_ruler")

    # sparse_attn 旁路桩（与生成/提交协议无关）
    pred_ruler.register_patch = lambda model, args: None
    pred_ruler.get_metrics = lambda: types.SimpleNamespace(
        get_select_tokens=lambda: 0, clear=lambda: None)
    pred_ruler.get_method_name_with_info = lambda args: "stubm"
    return pred_ruler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="pred 输出目录（--output-dir）")
    ap.add_argument("--data-root", required=True, help="RULER 源数据根目录")
    ap.add_argument("--task", default="vt")
    ap.add_argument("--context-length", type=int, default=32768)
    ap.add_argument("--t", default="09090909")
    ap.add_argument("--mode", required=True,
                    choices=["success", "crash-mid", "crash-between",
                             "lock-hold"])
    ap.add_argument("--rows", type=int, default=3,
                    help="stub 源数据行数（现场合成最小 RULER jsonl）")
    ap.add_argument("--crash-after", type=int, default=1,
                    help="crash-mid 模式：写完第 N 行后 os._exit(9)")
    ap.add_argument("--pred-tag", default="A", help="stub 预测内容标记（区分代际）")
    ap.add_argument("--hold", type=float, default=2.0, help="lock-hold 持锁秒数")
    ap.add_argument("--barrier-dir", default=None)
    ap.add_argument("--barrier-count", type=int, default=1)
    args = ap.parse_args()

    if args.mode == "lock-hold":
        pred_ruler = install_stubs(args.pred_tag)
        # attempt_lock_path 定义在 yarn_receipt（pred_ruler 只按需导入
        # 生产消费的子集，不重导出），此处直接取锁模块拿锁路径
        yarn_mod = importlib.import_module("benchmark.RULER.yarn_receipt")
        out_path = os.path.join(
            args.out_dir, f"L{args.context_length}", "pred_stub",
            f"{args.task}-stubm-{args.t}.jsonl")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fd = pred_ruler.acquire_output_lock(out_path)
        print(f"[RUNNER {os.getpid()}] lock-hold：已获锁 "
              f"{yarn_mod.attempt_lock_path(out_path)}，sleep {args.hold}s",
              flush=True)
        time.sleep(args.hold)
        pred_ruler.release_output_lock(fd)
        print(f"[RUNNER {os.getpid()}] lock-hold：已释放", flush=True)
        return 0

    # ---- 现场合成最小 RULER 源数据（input/outputs/length/index）----
    task_dir = os.path.join(args.data_root, str(args.context_length))
    os.makedirs(task_dir, exist_ok=True)
    data_file = os.path.join(task_dir, f"{args.task}.jsonl")
    with open(data_file, "w", encoding="utf-8") as f:
        for i in range(args.rows):
            f.write(json.dumps({
                "input": f"stub context {i}", "outputs": [f"gt{i}"],
                "length": args.context_length, "index": i,
            }, ensure_ascii=False) + "\n")

    pred_ruler = install_stubs(args.pred_tag)

    # crash-mid：生成循环写完第 crash-after 行后硬杀（tqdm 桩计数）
    if args.mode == "crash-mid":
        class _CrashTqdm:
            def __init__(self, it):
                self._it, self._n = iter(it), 0

            def __iter__(self):
                return self

            def __next__(self):
                if self._n >= args.crash_after:
                    print(f"[RUNNER {os.getpid()}] crash-mid：已写 "
                          f"{self._n} 行 → os._exit(9) 模拟生成中途死亡"
                          f"（持锁硬杀，内核自动释放）", flush=True)
                    os._exit(9)
                self._n += 1
                return next(self._it)
        pred_ruler.tqdm = _CrashTqdm

    # crash-between：提交点两步之间死亡（预测已 replace、回执未 replace）
    if args.mode == "crash-between":
        def _hooked_commit(tmp_pred, tmp_receipt, out_path):
            os.replace(tmp_pred, out_path)   # 第一步已执行（生产同款）
            print(f"[RUNNER {os.getpid()}] crash-between：预测已 replace 到"
                  f"最终路径、完成回执未 replace → os._exit(9)（持锁硬杀）",
                  flush=True)
            os._exit(9)
        pred_ruler.commit_yarn_generation = _hooked_commit

    if args.barrier_dir:
        barrier_wait(args.barrier_dir, args.barrier_count)

    sys.argv = [
        "pred_ruler.py",
        "--model", "Qwen3-8B", "--model_path", "/synthetic/Qwen3-8B",
        "--task", args.task, "--context_length", str(args.context_length),
        "--data-root", args.data_root, "--output-dir", args.out_dir,
        "--t", args.t, "--method", "tli", "--pred_postfix", "_stub",
        "--max-num", "0",
    ]
    print(f"[RUNNER {os.getpid()}] mode={args.mode} tag={args.pred_tag} "
          f"进入生产 main()", flush=True)
    try:
        pred_ruler.main()
        print(f"[RUNNER {os.getpid()}] main 正常返回（成功提交）", flush=True)
        return 0
    except SystemExit as e:
        print(f"[RUNNER {os.getpid()}] SystemExit code={e.code}", flush=True)
        return e.code if isinstance(e.code, int) else 1


if __name__ == "__main__":
    sys.exit(main())
