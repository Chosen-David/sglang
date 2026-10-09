# -*- coding: utf-8 -*-
"""E117 8 层逐层配置 e2e 小试 harness（任务链 #191）。

背景：E117a-mavg 回放（exp/trace/analyze_e117a_mavg_ref.py +
exp/trace/results/e117a_mavg_ref.json）在 8 个代表层上发现逐层最优
(α,β,γ) 的 W_O 投影误差比全局固定冠军 (0.25, 0.125, 0.625) 低
（gap 中位 34.79%），判决 GO 授权 8 层小试。本 wrapper 做 e2e 实测：
  - perlayer 臂：8 层逐层覆写 + 其余层保持冠军；
  - uniform  臂：全程冠军配置（对照组，不覆写）。
两臂 LongBench 13 任务配对对比（n/seed 由 pred.py 内部固定 seed=42 +
全量数据保证一致）。trace 级增益可能在 e2e 反转（E85f 教训），不预设方向。

设计决策（为什么选 runpy 包装而不是改 pred.py / 复制其 main）：
  pred.py 的全部流程在 `if __name__ == "__main__":` 块内（无 main() 函数），
  复制会带来 ~120 行漂移面。本 wrapper 通过三层进程内 hook 实现零侵入：
  1) sys.argv 默认注入（缺省才补，用户显式参数优先）；
  2) sparse_attn.patches.register_patch 属性替换——pred.py 顶部
     `from sparse_attn.patches import register_patch` 在 runpy 执行时才
     发生，会拿到包装后的版本（先调原函数挂 indexer，再按 module.layer_idx
     覆写 indexer.alpha/beta/gamma）；
  3) TLIIndexer._need_avg_score 进程内 monkeypatch（E117a 回放同款）。
  因此不需要修改 pred.py 本体，也不需要复制其 main 流程。

必读技术事实（主会话已核验）：
  * register_patch 为每个 Qwen3Attention/LlamaAttention module 独立构造
    IndexerType(args) 并注入 module.layer_idx → indexer.layer_idx；
  * tli_indexer.py 中 α/β/γ 全部在 compute 路径动态读取，构造后覆写安全；
  * _need_avg_score 原语义：near=avg 时须 α>0 且 β>0 才产出 avg 分数源。
    单池层（α=β=0）+ near=avg 会让 k_avg/score_coarse_avg 缺失。
    e2e 下逐层 indexer 配置固定、无回放式的跨候选残留，本 patch 对 mask
    结果中立（mavg 单池层消费点 L898 须 far_method=="avg" 才触发，
    far=minmax 不触发；分区层 α,β>0 原语义本就 True）——但回放判决是在
    「avg 分数源恒在」语义下做出，两臂必须同 patch 语义才公平（用户
    指令硬性要求），故两臂都挂，receipt 记录自检结果。

身份闭包（051 教训）：每次运行落盘 sidecar receipt（原子写），含 git SHA、
pred.py/wrapper SHA256、逐层配置 map、全局 method/αβγ、模型路径、patch
描述与自检结果、时间戳——小试结果身份绑定的唯一凭证。

用法（与 pred.py 相同参数 + 两个 wrapper 参数）：
  python -m benchmark.LongBench.pred_e117_perlayer \
      --arm perlayer --model Qwen3-8B --task hotpotqa \
      --model_path /mnt/.../Qwen3-8B --dataset-path ~/datasets/LongBench/data \
      --config-path benchmark/LongBench/config --output-dir /tmp/e117_trial
  # --arm uniform 跑对照组；--perlayer-config 指定自定义 JSON（默认嵌 8 层配置）
"""
import argparse
import datetime
import hashlib
import json
import os
import runpy
import subprocess
import sys

# ---------------- 路径自举：兼容 -m 与直接路径两种启动 ----------------
# 直接 `python benchmark/LongBench/pred_e117_perlayer.py` 启动时，
# pred.py 内部的 `from benchmark.LongBench...` 绝对导入需要仓库根在 sys.path。
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
PRED_PATH = os.path.join(_HERE, "pred.py")

# ---------------- E117a 逐层最优配置（来源 e117a_mavg_ref.json per_layer，已核验）--
# c_star 逐层值与 e117a_mavg_ref.json 逐位一致（2026-10-10 主会话核验）：
#   L4/24/28 单池 (0,0,0)；L1 是对照层不覆写（保持冠军）。
PERLAYER_DEFAULT = {
    4:  (0.0, 0.0, 0.0),
    8:  (0.375, 0.625, 0.125),
    12: (0.875, 0.625, 0.125),
    16: (0.625, 0.625, 0.125),
    20: (0.25, 0.375, 0.125),
    24: (0.0, 0.0, 0.0),
    28: (0.0, 0.0, 0.0),
    35: (0.0625, 0.125, 0.5),
}
# 全局冠军（= E117a 回放的 G* = CHAMP_REF，mavg 口径已保存最好配置）
CHAMPION = (0.25, 0.125, 0.625)

# 默认注入的 pred.py 参数（缺省才补，用户显式参数优先）。
# 注意三点身份对齐：
#   ① method 组合 = mavg（far=minmax + near=avg），E117a 回放打分口径；
#   ② αβγ = 冠军 (0.25, 0.125, 0.625)，两臂共同的基配置；
#   ③ --tia_level2_cmp_ratio 4 + kmeans/layer_skip 关 → 输出文件名 method 段
#      为 tli_64_128_1024_c4_A，与既有生产臂（E98BEST/E100/e98_e2e_grid 及
#      E119 receipt 的 method 编码）一致；parser 裸默认 cmp_ratio=2+kmeans/skip
#      全开（c2_ABD）是旧口径，不能默认带上。
WRAPPER_DEFAULTS = [
    "--method", "tli",
    "--tli_far_method", "minmax",
    "--tli_near_method", "avg",
    "--tli_alpha", str(CHAMPION[0]),
    "--tli_beta", str(CHAMPION[1]),
    "--tli_gamma", str(CHAMPION[2]),
    "--tia_level2_cmp_ratio", "4",
    "--tli_enable_kmeans", "false",
    "--tli_enable_layer_skip", "false",
]

PATCH_DESCRIPTION = (
    "E117a 回放同款进程内 monkeypatch：TLIIndexer._need_avg_score 在 "
    "far/near 含 'avg' 时无条件返回 True（原始语义 near=avg 还须 α>0 且 β>0）。"
    "e2e 固定配置下对 mask 结果中立（mavg 单池层消费点需 far_method=='avg' "
    "不触发；分区层原语义本就 True），携带目的是与回放判决保持同分数源语义、"
    "两臂公平（用户指令硬性要求）。不改生产文件，仅本进程。"
)


# ---------------- 逐层配置加载与校验 ----------------
def load_perlayer_config(path=None):
    """加载逐层 (α,β,γ) 配置。path=None 用内置 8 层默认。

    JSON 格式：{"层号": [α, β, γ], ...}（层号字符串或整数均可）；
    也接受 {"per_layer": {...}} 包一层的形式。返回 {int: (float,float,float)}。
    """
    if path is None:
        raw = {str(k): list(v) for k, v in PERLAYER_DEFAULT.items()}
    else:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and set(obj.keys()) == {"per_layer"}:
            obj = obj["per_layer"]
        if not isinstance(obj, dict) or not obj:
            raise ValueError(f"perlayer 配置必须是非空 dict: {path}")
        raw = obj
    cfg = {}
    for k, v in raw.items():
        li = int(k)
        if li < 0:
            raise ValueError(f"层号必须非负: {k}")
        if not isinstance(v, (list, tuple)) or len(v) != 3:
            raise ValueError(f"层 {li} 的配置必须是 [α, β, γ] 三元组: {v}")
        a, b, g = (float(x) for x in v)
        for name, x in (("alpha", a), ("beta", b), ("gamma", g)):
            if not (0.0 <= x <= 1.0):
                raise ValueError(f"层 {li} 的 {name}={x} 超出 [0,1]")
        cfg[li] = (a, b, g)
    return cfg


# ---------------- argv 默认注入 ----------------
def _flag_present(argv, flag):
    """flag 显式出现判定：兼容 `--flag value` 与 `--flag=value` 两种形式。"""
    return flag in argv or any(t.startswith(flag + "=") for t in argv)


def inject_defaults(argv, arm):
    """对 pred.py 的 argv 补缺省参数（用户显式参数优先，绝不覆盖）。

    --pred_postfix 两臂分目录（避免覆盖既有生产文件）：
    perlayer → _e117per，uniform → _e117uni。
    """
    out = list(argv)
    for i in range(0, len(WRAPPER_DEFAULTS), 2):
        flag = WRAPPER_DEFAULTS[i]
        if not _flag_present(out, flag):
            out.extend(WRAPPER_DEFAULTS[i:i + 2])
    postfix = "_e117per" if arm == "perlayer" else "_e117uni"
    if not _flag_present(out, "--pred_postfix"):
        out.extend(["--pred_postfix", postfix])
    return out


# ---------------- _need_avg_score monkeypatch（E117a 回放同款）----------------
_ORIG_NEED_AVG = None
_PATCH_APPLIED = False


def apply_avg_score_patch():
    """TLIIndexer._need_avg_score → far/near 含 'avg' 时无条件 True（本进程）。

    与 exp/trace/analyze_e117a_mavg_ref.py L58-67 逐语义一致（不 import 该
    脚本——它 import 链拖 exp/trace 全家，本 wrapper 保持独立可测）。
    幂等：重复调用不二次包装。
    """
    global _ORIG_NEED_AVG, _PATCH_APPLIED
    if _PATCH_APPLIED:
        return _ORIG_NEED_AVG
    from sparse_attn.indexer.tli_indexer import TLIIndexer
    _ORIG_NEED_AVG = TLIIndexer._need_avg_score

    def _need_avg_score_e117(self):
        if "avg" in (self.far_method, self.near_method):
            return True
        return _ORIG_NEED_AVG(self)

    TLIIndexer._need_avg_score = _need_avg_score_e117
    _PATCH_APPLIED = True
    return _ORIG_NEED_AVG


def avg_score_patch_selfcheck():
    """patch 红绿自检（真实 TLIIndexer，纯 CPU，秒级）。

    红  ：原始 _need_avg_score 在单池残留参数（α=β=0, near=avg, far=minmax）
          下返回 False —— 缺陷根因证据（回放 E109a bug 模式）。
    绿 1：patch 后同参数返回 True（avg 分数源恒在）。
    绿 2：冠军配置（α,β>0）原始语义本就 True（patch 对生产冠军臂无行为差）。
    环境异常（torch/依赖缺失）时不抛错，降级为「静态断言 patch 已挂」
    （静态断言在 main() 里强制执行，此处只记录失败原因）。
    """
    try:
        import torch  # noqa: F401  环境探针
        from sparse_attn.arguments import add_sparse_attn_args
        from sparse_attn.indexer.tli_indexer import TLIIndexer
        p = argparse.ArgumentParser()
        add_sparse_attn_args(p)
        # 与生产冠军臂一致的参数面（构造真实 indexer，覆盖 __init__ 全路径）
        ns = p.parse_args([
            "--method", "tli",
            "--tli_far_method", "minmax", "--tli_near_method", "avg",
            "--tli_alpha", str(CHAMPION[0]), "--tli_beta", str(CHAMPION[1]),
            "--tli_gamma", str(CHAMPION[2]),
            "--tia_level2_cmp_ratio", "4",
            "--tli_enable_kmeans", "false", "--tli_enable_layer_skip", "false",
        ])
        idx = TLIIndexer(ns)
        orig = _ORIG_NEED_AVG
        # 红：单池残留参数下原始语义 = False
        idx.alpha, idx.beta, idx.gamma = 0.0, 0.0, 0.0
        red = not orig(idx)
        # 绿 1：patch 后单池残留参数 = True
        green_residue = idx._need_avg_score()
        # 绿 2：冠军配置下原始语义本就 True
        idx.alpha, idx.beta, idx.gamma = CHAMPION
        green_champ_orig = orig(idx)
        ok = bool(red and green_residue and green_champ_orig)
        return {
            "mode": "real_indexer_cpu",
            "red_orig_need_avg_false_on_single_pool": bool(red),
            "green_patched_true_on_single_pool_residue": bool(green_residue),
            "green_orig_true_on_champion": bool(green_champ_orig),
            "pass": ok,
        }
    except Exception as e:  # 环境降级：torch/CUDA/依赖不可用时
        return {
            "mode": "static_only",
            "error": f"{type(e).__name__}: {e}",
            "pass": None,   # 静态断言（patch 已挂）由 main() 强制兜底
        }


# ---------------- register_patch 包装：逐层覆写 ----------------
def apply_perlayer_overrides(model, perlayer_map, num_hidden_layers=None):
    """对 model 中所有带 indexer 的 attention module 按 layer_idx 覆写 α/β/γ。

    返回 (applied: {layer_idx: (α,β,γ)}, n_indexer_modules)。num_hidden_layers
    非 None 时断言全部覆写层号 < 层数（防错层，用户硬性要求）。
    """
    n_layers = num_hidden_layers
    if n_layers is not None:
        for li in perlayer_map:
            if li >= n_layers:
                raise AssertionError(
                    f"覆写层号 {li} >= num_hidden_layers={n_layers}（防错层断言失败）"
                )
    applied = {}
    n_modules = 0
    for _name, module in model.named_modules():
        indexer = getattr(module, "indexer", None)
        if indexer is None:
            continue
        n_modules += 1
        li = getattr(module, "layer_idx", None)
        if li in perlayer_map:
            a, b, g = perlayer_map[li]
            indexer.alpha, indexer.beta, indexer.gamma = a, b, g
            applied[li] = (a, b, g)
            tag = " (single-pool)" if (a == 0.0 and b == 0.0) else ""
            print(f"[E117-perlayer] override layer_idx={li} "
                  f"alpha={a} beta={b} gamma={g}{tag}", flush=True)
    # 覆写完整性：map 中的每个层号必须真的被找到并覆写（防层号拼写错漏覆写）
    missing = set(perlayer_map) - set(applied)
    if missing:
        raise AssertionError(f"perlayer 配置中的层号未在模型中找到 indexer: {sorted(missing)}")
    return applied, n_modules


def _sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _git_head():
    """当前检出（可能是 worktree）的 HEAD SHA + dirty 标记。"""
    try:
        sha = subprocess.run(
            ["git", "-C", _REPO_ROOT, "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        st = subprocess.run(
            ["git", "-C", _REPO_ROOT, "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return {"sha": sha, "dirty": bool(st), "dirty_files": st.splitlines()[:20]}
    except Exception as e:
        return {"sha": None, "error": f"{type(e).__name__}: {e}"}


def write_receipt(path, payload):
    """原子写 receipt（临时文件 + os.replace；E116h 042-049 同口径纪律）。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def patch_register_patch(perlayer_map, arm, selfcheck):
    """替换 sparse_attn.patches.register_patch 为「先原函数，再逐层覆写」包装。

    pred.py 顶部 `from sparse_attn.patches import register_patch` 在 runpy
    执行时才解析，拿到的必是本包装版本。覆写/receipt 都发生在原
    register_patch 之后（此刻每个 attention module 已有自己的 indexer）。
    返回 (原函数, 新函数) 供测试还原。
    """
    import sparse_attn.patches as patches_mod
    orig = patches_mod.register_patch
    state = {"receipt_written": False}

    def wrapped_register_patch(model, args):
        orig(model, args)
        num_layers = getattr(getattr(model, "config", None), "num_hidden_layers", None)
        if arm == "perlayer":
            applied, n_modules = apply_perlayer_overrides(model, perlayer_map, num_layers)
        else:
            applied, n_modules = {}, 0
            for _n, m in model.named_modules():
                if getattr(m, "indexer", None) is not None:
                    n_modules += 1
        if num_layers is None:
            print("[E117-perlayer] 警告: model.config.num_hidden_layers 缺失，"
                  "防错层断言未执行", flush=True)
        else:
            assert all(li < num_layers for li in perlayer_map), \
                f"覆写层号越界（num_hidden_layers={num_layers}）"
        digest = (f"[E117-{arm}] digest: {len(applied)}/{num_layers} layers "
                  f"overridden, {n_modules} indexer modules")
        print(digest + (f" (uniform arm: no override, champion="
                        f"{CHAMPION})" if arm == "uniform" else ""), flush=True)
        # ---- 身份闭包 receipt（051 教训：唯一身份凭证，原子落盘）----
        if not state["receipt_written"]:
            state["receipt_written"] = True
            _write_run_receipt(model, args, arm, perlayer_map, applied,
                                n_modules, num_layers, selfcheck)
        else:
            print("[E117-perlayer] 警告: register_patch 被二次调用，receipt 已写",
                  flush=True)
        return None

    patches_mod.register_patch = wrapped_register_patch
    return orig, wrapped_register_patch


def _pred_out_fn_base(args):
    """复刻 pred.py 的输出文件名基（L445-450 四行 sanitize 逻辑）。"""
    from sparse_attn.info import get_method_name_with_info
    method_name = get_method_name_with_info(args)
    base = f"{args.task.split('-')[0]}-{method_name}-{args.t}"
    base = base.replace(" ", "").replace("'", "")
    if len(base) > 245:
        base = base[:245] + "..."
    return base


def _write_run_receipt(model, args, arm, perlayer_map, applied, n_modules,
                       num_layers, selfcheck):
    out_dir = os.path.join(args.output_dir, f"pred{args.pred_postfix}")
    base = _pred_out_fn_base(args)
    model_path = (getattr(getattr(model, "config", None), "_name_or_path", None)
                  or args.model_path)
    payload = {
        "experiment": "E117_perlayer_e2e_trial",
        "task_chain": "#191",
        "arm": arm,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git": _git_head(),
        "repo_root": _REPO_ROOT,
        "pred_py_sha256": _sha256_file(PRED_PATH),
        "wrapper_sha256": _sha256_file(os.path.abspath(__file__)),
        "task": args.task,
        "output_dir": out_dir,
        "pred_postfix": args.pred_postfix,
        "model_path": model_path,
        "num_hidden_layers": num_layers,
        "n_indexer_modules": n_modules,
        "global_config": {
            "method": args.method,
            "tli_far_method": args.tli_far_method,
            "tli_near_method": args.tli_near_method,
            "tli_alpha": args.tli_alpha,
            "tli_beta": args.tli_beta,
            "tli_gamma": args.tli_gamma,
            "tia_block_size": args.tia_block_size,
            "tia_level1_topk": args.tia_level1_topk,
            "tia_level2_topk": args.tia_level2_topk,
            "tia_level2_cmp_ratio": args.tia_level2_cmp_ratio,
            "tli_enable_kmeans": args.tli_enable_kmeans,
            "tli_enable_layer_skip": args.tli_enable_layer_skip,
            "tli_enable_subspace": args.tli_enable_subspace,
            "tli_subspace": args.tli_subspace,
            "champion_ref": list(CHAMPION),
        },
        "perlayer_config_requested": {str(k): list(v) for k, v in perlayer_map.items()},
        "perlayer_config_applied": {str(k): list(v) for k, v in applied.items()},
        "n_layers_overridden": len(applied),
        "avg_score_patch": {
            "what": PATCH_DESCRIPTION,
            "applied": _PATCH_APPLIED,
            "selfcheck": selfcheck,
        },
        "seed": 42,   # pred.py __main__ 硬编码 seed_everything(42)
    }
    rc_path = os.path.join(out_dir, f"{base}-e117_perlayer_receipt.json")
    write_receipt(rc_path, payload)
    print(f"[E117-{arm}] receipt -> {rc_path}", flush=True)


# ---------------- wrapper 自身参数 ----------------
def parse_wrapper_args(argv):
    """只认 wrapper 两个参数，其余原序透传给 pred.py（parse_known_args）。"""
    p = argparse.ArgumentParser(
        description="E117 逐层配置 e2e 小试 wrapper（其余参数透传 pred.py）",
        add_help=False,
    )
    p.add_argument("--arm", choices=["perlayer", "uniform"], default="perlayer",
                   help="perlayer=8 层逐层覆写臂；uniform=冠军均匀对照组")
    p.add_argument("--perlayer-config", default=None,
                   help="逐层配置 JSON 路径（默认嵌入 e117a_mavg_ref 8 层配置）")
    known, rest = p.parse_known_args(argv)
    return known, rest


def main():
    argv = sys.argv[1:]
    wargs, rest = parse_wrapper_args(argv)
    perlayer_map = load_perlayer_config(wargs.perlayer_config)

    # ① 挂 avg 分数源 patch（两臂同语义，回放公平性硬性要求）
    apply_avg_score_patch()
    selfcheck = avg_score_patch_selfcheck()
    if not _PATCH_APPLIED:   # 静态断言兜底（正常不可达：上面刚挂过）
        raise RuntimeError("avg_score patch 未挂载，拒绝继续")
    print(f"[E117-{wargs.arm}] avg_score_patch selfcheck: "
          f"{json.dumps(selfcheck, ensure_ascii=False)}", flush=True)

    # ② 替换 register_patch（先原函数挂 indexer，再逐层覆写 + 写 receipt）
    patch_register_patch(perlayer_map, wargs.arm, selfcheck)

    # ③ argv 默认注入后把控制权交给 pred.py（runpy __main__ 全流程）
    new_argv = inject_defaults(rest, wargs.arm)
    sys.argv = [sys.argv[0]] + new_argv
    print(f"[E117-{wargs.arm}] wrapper defaults injected: "
          f"method=tli mavg champ={CHAMPION} arm={wargs.arm} "
          f"perlayer_layers={sorted(perlayer_map)}", flush=True)
    runpy.run_path(PRED_PATH, run_name="__main__")


if __name__ == "__main__":
    main()
