#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E124a（S-T021 / A1）动态 controller v0 最小入口脚本。

用法：
  # dry-run：只输出解析后的配置（controller_spec + 给定几何下三档与
  # 固定档的整数预算编译结果），不读 trace、不载模型、不落决策。
  python3 exp/trace/run_e124a_dryrun.py --dry-run
  python3 exp/trace/run_e124a_dryrun.py --dry-run --k1 128 --k2 1024 \
      --n-valid 32768 --n-protected 256

  # 实跑（CPU only）：读 /tmp/trace/qwen3-8b-v 的逐层 q/k，取末位
  # query、非保护 middle K，逐 (seq, layer) 调 decide()，落
  # per_seq_layer_decisions.jsonl + controller_spec.json。
  python3 exp/trace/run_e124a_dryrun.py --trace-dir /tmp/trace/qwen3-8b-v/hotpotqa-0 \
      --out /tmp/e124a_dryrun/per_seq_layer_decisions.jsonl

边界（任务书约束）：不注册任何 CLI 参数进 sparse_attn/arguments.py；
不改生产 tli_indexer.py；不实现 M5 D′ gate；不跑 GPU；--dynamic 类
参数一律不伪造——本脚本的 argparse 参数只是控制器输入几何，不是
生产 CLI 扩展。
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from sparse_attn.indexer import dyn_controller as dc  # noqa: E402


def _geo_budgets(args):
    """给定几何下三档 + 固定参数档的编译预览（dry-run 的主体）。
    ok=False 时记录 reason——dry-run 的职责是「解析后配置」而非判定，
    三档是否可用如实展示，不预选。"""
    out = {}
    for tier in ("P_F", "P_C", "P_N"):
        ok, info = dc.compile_tier(tier, k1=args.k1, k2=args.k2, bs=args.bs,
                                   n_valid=args.n_valid,
                                   n_protected=args.n_protected,
                                   n_prefix=args.n_prefix, n_swa=args.n_swa)
        out[tier] = info if ok else {"reason": info.get("reason"),
                                     "detail": info}
    ok, info = dc.compile_fixed(args.method, k1=args.k1, k2=args.k2,
                                bs=args.bs, n_valid=args.n_valid,
                                n_protected=args.n_protected,
                                n_prefix=args.n_prefix, n_swa=args.n_swa)
    out["fixed_tier"] = info if ok else {"reason": info.get("reason"),
                                         "detail": info}
    return out


def _resolved_config(args):
    return {
        "controller_spec": dc.controller_spec(),
        "run_config": {
            "method": dc.canonical_method(args.method),
            "k1": args.k1, "k2": args.k2, "bs": args.bs,
            "n_valid": args.n_valid, "n_protected": args.n_protected,
            "n_prefix": args.n_prefix, "n_swa": args.n_swa,
            "phase": "prefill", "state_epoch": 0,
        },
        "compiled_preview": _geo_budgets(args),
    }


def _run_trace(args):
    """实跑：逐层读 trace pt，取末位 q（当前 query）与非保护 middle K，
    decide() 后追加落盘。CPU only；bf16 → 模块内部升 float64。"""
    import torch
    meta = json.load(open(os.path.join(args.trace_dir, "meta.json")))
    layers = args.layers or meta.get("layers")
    _check(layers, f"{args.trace_dir}/meta.json 无 layers 且未显式给 --layers")
    S = meta.get("S")
    n_valid = args.n_valid if args.n_valid else int(S)
    spec_path = args.spec_out or os.path.join(
        os.path.dirname(os.path.abspath(args.out)), "controller_spec.json")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    dc.write_controller_spec(spec_path)
    log = dc.DecisionLog(args.out)
    n = 0
    for layer in layers:
        # 兼容 layer_1.pt 与 layer01.pt 两种命名（实存 trace 为零填充后者）
        cands = [os.path.join(args.trace_dir, f"layer_{layer}.pt"),
                 os.path.join(args.trace_dir, f"layer{int(layer):02d}.pt")]
        pt_path = next((p for p in cands if os.path.exists(p)), None)
        _check(pt_path, f"trace 文件不存在: {args.trace_dir}/layer_{layer}.pt")
        blob = torch.load(pt_path, map_location="cpu")
        q_all, k_all = blob["q"], blob["k"]     # [Tq, H, D], [S, Hkv, D]
        q = q_all[-1].to(torch.float32)          # 当前 query（末位）[H, D]
        mid = k_all[args.n_prefix: n_valid - args.n_swa]
        k_mid = mid.to(torch.float32)
        dec = dc.decide(q, k_mid, seq_id=args.seq_id, layer_idx=int(layer),
                        method=args.method, k1=args.k1, k2=args.k2,
                        bs=args.bs, n_valid=n_valid,
                        n_protected=args.n_protected,
                        n_prefix=args.n_prefix, n_swa=args.n_swa)
        log.append(dec)
        n += 1
        print(f"[e124a] seq={args.seq_id} layer={layer} "
              f"s={dec['features']['s'] if dec['features']['s'] is not None else 'NA'} "
              f"requested={dec['profile']['requested']} "
              f"applied={dec['profile']['applied']} "
              f"Tn/Tf={dec['budget']['Tn']}/{dec['budget']['Tf']}"
              if dec["budget"] else "applied=unsupported", flush=True)
    log.close()
    print(f"[e124a] DONE {n} decisions -> {args.out}\n"
          f"[e124a] spec -> {spec_path}")
    return 0


def _check(cond, msg):
    if not cond:
        raise SystemExit(f"[E124A-DRYRUN-FAIL] {msg}")


def main():
    ap = argparse.ArgumentParser(description="E124a dyn controller v0 入口")
    ap.add_argument("--dry-run", action="store_true",
                    help="只输出解析后配置（spec+预算预览），不读 trace/不载模型")
    ap.add_argument("--trace-dir",
                    default="/tmp/trace/qwen3-8b-v/lb_hotpotqa_0",
                    help="trace 目录（含 layerNN.pt + meta.json）")
    ap.add_argument("--layers", default=None,
                    help="逗号分隔层号（缺省读 meta.json 的 layers）")
    ap.add_argument("--method", default="mavg")
    ap.add_argument("--k1", type=int, default=128)
    ap.add_argument("--k2", type=int, default=1024)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--n-valid", type=int, default=32768,
                    help="U：因果可见 key 数（实跑缺省取 meta.S）")
    ap.add_argument("--n-protected", type=int, default=256)
    ap.add_argument("--n-prefix", type=int, default=128)
    ap.add_argument("--n-swa", type=int, default=128)
    ap.add_argument("--seq-id", type=int, default=0)
    ap.add_argument("--out", default="/tmp/e124a_dryrun/per_seq_layer_decisions.jsonl")
    ap.add_argument("--spec-out", default=None)
    args = ap.parse_args()

    if args.layers:
        args.layers = [int(x) for x in str(args.layers).split(",") if x.strip()]

    if args.dry_run:
        print(json.dumps(_resolved_config(args), ensure_ascii=False, indent=2))
        return 0
    return _run_trace(args)


if __name__ == "__main__":
    sys.exit(main())
