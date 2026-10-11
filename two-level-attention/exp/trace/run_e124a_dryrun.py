#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E124a（S-T021 / A1）动态 controller v0 最小入口脚本。

用法：
  # dry-run：只输出解析后的配置（controller_spec + 给定几何下三档与
  # 固定档的整数预算编译结果），不读 trace、不载模型、不落决策。
  # dry-run 预览需要显式 --n-valid（实跑缺省才回退 trace meta.S）。
  python3 exp/trace/run_e124a_dryrun.py --dry-run --k1 128 --k2 1024 \
      --n-valid 32768 --n-protected 256

  # 实跑（CPU only）：读 /tmp/trace/qwen3-8b-v 的逐层 q/k，取末位
  # query、非保护 middle K，逐 (seq, layer) 调 decide()，落
  # per_seq_layer_decisions.jsonl + controller_spec.json。
  python3 exp/trace/run_e124a_dryrun.py --trace-dir /tmp/trace/qwen3-8b-v/lb_hotpotqa_0 \
      --out /tmp/e124a_dryrun/per_seq_layer_decisions.jsonl

几何一致性契约（GPT A1 验收反馈 2026-10-11，fail-closed 不静默）：
  1. --n-valid 缺省为 None → 回退 trace meta.S（本入口默认值曾硬编码
     32768，使回退分支永不触发、S=16957 被静默覆盖——本修复的根源）。
  2. 显式给 --n-valid 时必须与 trace 实际 K 长度一致（= meta.S，且与
     每 layer blob["k"] 实际行数、blob["S"] 双向核对），不一致 →
     [E124A-ABORT] 非零退出。
  3. mid 切片 k_all[n_prefix : n_valid-n_swa] 之前显式断言
     n_valid <= len(k_all)——Python 切片越界会**静默钳制**（旧版缺陷：
     S=16957 时 [128:32640] 被钳成 [128:16957]，SWA 尾段 128 token
     未切掉混进特征 middle），钳制正是缺陷根源，必须显式拒绝。
  4. decide() 消费侧同源校验：k_mid 行数必须 == n_valid − n_protected
     （dyn_controller.decide 内部 fail-closed），特征与预算共享同一
     合法前缀。
  5. query 因果位置：q_all[-1] 的 qpos（blob["qpos"] 末位）+1 必须等于
     n_valid，不一致拒绝；qpos 与来源如实记入每条决策的 geometry 段。

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
    三档是否可用如实展示，不预选。
    --n-valid 未给时（缺省 None）预算公式无输入几何 → 各档如实记
    reason=n_valid_unspecified（dry-run 预览要求显式 --n-valid；实跑
    才回退 trace meta.S）。"""
    out = {}
    if args.n_valid is None:
        for tier in ("P_F", "P_C", "P_N", "fixed_tier"):
            out[tier] = {"reason": "n_valid_unspecified",
                         "detail": "dry-run 预览需要显式 --n-valid；"
                                   "实跑缺省回退 trace meta.S"}
        return out
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


def _abort(msg):
    """几何一致性 fail-closed（GPT A1 验收反馈）：非零退出 + 可检索
    错误标记。区别于 _check（一般性前置校验）——所有「特征/预算两套
    几何」「声明与实际张量不一致」「切片会静默钳制」类错误一律走
    本出口，绝不静默覆盖/截断。"""
    raise SystemExit(f"[E124A-ABORT][GEOM-MISMATCH] {msg}")


def _check(cond, msg):
    if not cond:
        raise SystemExit(f"[E124A-DRYRUN-FAIL] {msg}")


def _run_trace(args):
    """实跑：逐层读 trace pt，取末位 q（当前 query）与非保护 middle K，
    decide() 后追加落盘。CPU only；bf16 → 模块内部升 float64。

    几何一致性门禁（见模块 docstring）：n_valid 与 trace 实际 K 长度
    双向核对、切片前防钳制断言、qpos 因果核对、k_mid 行数与
    n_valid−n_protected 同源校验——任何不一致 [E124A-ABORT] 非零退出。"""
    import torch
    meta = json.load(open(os.path.join(args.trace_dir, "meta.json")))
    layers = args.layers or meta.get("layers")
    _check(layers, f"{args.trace_dir}/meta.json 无 layers 且未显式给 --layers")
    S = meta.get("S")
    _check(isinstance(S, int) and S > 0,
           f"{args.trace_dir}/meta.json 缺合法 S（得 {S!r}）")

    # ---- n_valid 解析：缺省回退 meta.S；显式给值必须与 meta.S 一致 ----
    if args.n_valid is None:
        n_valid = int(S)
        n_valid_source = "meta.S"
    else:
        n_valid = int(args.n_valid)
        n_valid_source = "explicit"
        if n_valid != int(S):
            _abort(f"--n-valid={n_valid} 与 trace meta.S={int(S)} 不一致："
                   f"显式 n_valid 必须等于 trace 实际 K 长度，"
                   f"绝不静默覆盖（旧版默认 32768 即此缺陷，S=16957 被覆盖）")

    # ---- 保护集两分量闭合（prefix 头部 + SWA 尾部，无重叠） ----
    if args.n_prefix + args.n_swa != args.n_protected:
        _abort(f"n_prefix({args.n_prefix})+n_swa({args.n_swa}) != "
               f"n_protected({args.n_protected})：保护集两分量不闭合，"
               f"mid 切片 [n_prefix, n_valid-n_swa) 的行数"
               f"(n_valid-n_prefix-n_swa) 将与预算几何 (n_valid-n_protected) "
               f"两套口径——拒绝")

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

        # ---- meta 声明 vs 实际张量双向核对（逐 layer）----
        k_rows = int(k_all.shape[0])
        if k_rows != int(S):
            _abort(f"{pt_path}: blob['k'] 实际行数 {k_rows} != meta.S {int(S)}"
                   f"（meta 声明与实际张量不一致，拒绝）")
        blob_S = blob.get("S")
        if blob_S is not None:
            bsv = int(blob_S.item()) if torch.is_tensor(blob_S) else int(blob_S)
            if bsv != int(S):
                _abort(f"{pt_path}: blob['S']={bsv} != meta.S={int(S)}"
                       f"（trace 内部自报长度与 meta 不一致，拒绝）")
        if n_valid != k_rows:
            _abort(f"{pt_path}: n_valid={n_valid} != 实际 K 长度 {k_rows}"
                   f"（来源 {n_valid_source}；特征与预算必须共享同一合法前缀）")

        # ---- 切片前防钳制断言：越界切片会被 Python 静默钳制——
        #      旧版缺陷根源（[128:32640] 在 S=16957 时被钳成 [128:16957]，
        #      SWA 尾段未切除），必须显式拒绝而非静默截断 ----
        if n_valid > k_rows:
            _abort(f"n_valid={n_valid} > len(k_all)={k_rows}：切片会静默钳制，"
                   f"显式拒绝")

        # ---- 当前 query 的因果位置（q_all[-1]）----
        q = q_all[-1].to(torch.float32)          # 当前 query（末位）[H, D]
        if "qpos" in blob and blob["qpos"] is not None and len(blob["qpos"]) > 0:
            qpos = int(blob["qpos"][-1])
            qpos_source = "blob"
        else:
            # blob 无 qpos：prefill 末位 query 假设位于 S-1（如实记来源）
            qpos = int(S) - 1
            qpos_source = "assumed_S-1"
        if qpos + 1 != n_valid:
            _abort(f"当前 query 因果位置 qpos={qpos}（来源 {qpos_source}）"
                   f"的因果可见 keys={qpos + 1} != n_valid={n_valid}"
                   f"（query 因果位置与 K 长度关系不一致，拒绝）")

        mid = k_all[args.n_prefix: n_valid - args.n_swa]
        k_mid = mid.to(torch.float32)
        # ---- 特征/预算同源最终门：切片行数必须恰为 n_valid-n_protected
        #      （decide() 内部同源校验兜底，这里先给出可定位的错误信息） ----
        if int(k_mid.shape[0]) != n_valid - args.n_protected:
            _abort(f"{pt_path}: mid 切片行数 {int(k_mid.shape[0])} != "
                   f"n_valid-n_protected={n_valid - args.n_protected}"
                   f"（特征与预算两套几何，拒绝）")

        dec = dc.decide(q, k_mid, seq_id=args.seq_id, layer_idx=int(layer),
                        method=args.method, k1=args.k1, k2=args.k2,
                        bs=args.bs, n_valid=n_valid,
                        n_protected=args.n_protected,
                        n_prefix=args.n_prefix, n_swa=args.n_swa)
        # ---- 决策记录如实记录本次实跑的几何来源与 query 因果位置 ----
        dec["geometry"] = {
            "S": int(S), "k_rows": k_rows,
            "n_valid": n_valid, "n_valid_source": n_valid_source,
            "n_prefix": args.n_prefix, "n_swa": args.n_swa,
            "n_protected": args.n_protected,
            "n_mid": int(k_mid.shape[0]),
            "qpos": qpos, "qpos_source": qpos_source,
            "causal_visible_keys": qpos + 1,
        }
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
    ap.add_argument("--n-valid", type=int, default=None,
                    help="U：因果可见 key 数。缺省 None → 实跑回退 trace "
                         "meta.S（旧版默认 32768 使回退永不触发，已修）；"
                         "显式给值必须与 trace 实际 K 长度一致，否则 "
                         "[E124A-ABORT] 非零退出。dry-run 预览需显式给值。")
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
