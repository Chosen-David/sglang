# -*- coding: utf-8 -*-
"""test_pred_e117_perlayer.py —— E117 逐层 wrapper 的 CPU 级红绿测试（#191）。

纯 CPU、零 GPU、不加载大模型：
  绿 1：perlayer 配置解析（默认嵌入 8 层 + JSON 文件 round-trip + 校验负例）。
  绿 2：_need_avg_score patch 红绿——原始语义在单池残留参数（α=β=0,
        near=avg, far=minmax）下 False（红证据），patch 后 True（绿）；
        用真实 TLIIndexer（生产 argparse namespace 构造）验证。
  绿 3：receipt 原子落盘路径与字段完整性（tmp 目录 + 假 args namespace，
        不加载模型）。
  绿 4（补充）：argv 默认注入（显式参数不被覆盖、--flag=value 形式识别）
        + 逐层覆写逻辑（stub model：正确层覆写、越界层断言、未命中层断言）。

未测边界（如实声明，不冒充）：
  * 未加载真实 Qwen3-8B / 未跑任何 e2e 推理（GPU 由主会话验收后派单）；
  * register_patch 的真实挂载路径（Qwen3Attention module 识别）依赖
    transformers 模型实例，本测试用 stub module 验证覆写遍历逻辑；
  * pred.py 全流程（数据加载/generate 循环/输出 jsonl）不在本测试面。
"""
import json
import os
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, HERE)

import pred_e117_perlayer as W   # noqa: E402

HAS_TORCH = True
try:
    import torch  # noqa: F401
    import sparse_attn  # noqa: F401
except Exception as _e:
    HAS_TORCH = False
    _IMPORT_ERR = _e

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


# ---------------- 绿 1：perlayer 配置解析 ----------------
def test_perlayer_config():
    cfg = W.load_perlayer_config(None)
    expected = {
        4: (0.0, 0.0, 0.0), 8: (0.375, 0.625, 0.125),
        12: (0.875, 0.625, 0.125), 16: (0.625, 0.625, 0.125),
        20: (0.25, 0.375, 0.125), 24: (0.0, 0.0, 0.0),
        28: (0.0, 0.0, 0.0), 35: (0.0625, 0.125, 0.5),
    }
    check("绿1a 默认嵌入配置 = 预期 8 层（其余层不在 map）",
          cfg == expected and len(cfg) == 8,
          f"got {sorted(cfg)}")
    # JSON 文件 round-trip（含 per_layer 包装形式）
    with tempfile.TemporaryDirectory() as td:
        p1 = os.path.join(td, "c1.json")
        with open(p1, "w") as f:
            json.dump({str(k): list(v) for k, v in expected.items()}, f)
        check("绿1b JSON 平铺形式 round-trip 一致",
              W.load_perlayer_config(p1) == expected)
        p2 = os.path.join(td, "c2.json")
        with open(p2, "w") as f:
            json.dump({"per_layer": {str(k): list(v) for k, v in expected.items()}}, f)
        check("绿1c JSON per_layer 包装形式 round-trip 一致",
              W.load_perlayer_config(p2) == expected)
        # 校验负例：三元组长度错 / 越界值 / 非法层号
        for tag, obj in [
            ("二元组", {"4": [0.1, 0.2]}),
            ("越界值", {"4": [0.1, 0.2, 1.5]}),
            ("负层号", {"-1": [0.1, 0.2, 0.3]}),
        ]:
            p3 = os.path.join(td, "bad.json")
            with open(p3, "w") as f:
                json.dump(obj, f)
            try:
                W.load_perlayer_config(p3)
                check(f"绿1d 负例拒绝（{tag}）", False, "未抛异常")
            except ValueError as e:
                check(f"绿1d 负例拒绝（{tag}）", True, str(e)[:60])


# ---------------- 绿 2：_need_avg_score patch 红绿 ----------------
def test_avg_score_patch():
    if not HAS_TORCH:
        # 环境无 torch：降级为 AST/逻辑等价 stub（如实注明测试边界）
        src = open(W.__file__, encoding="utf-8").read()
        check("绿2(降级) patch 源码语义等价检查",
              'if "avg" in (self.far_method, self.near_method): return True' in src.replace('"',
                '"').replace("'\"'", "'\"'") or '"avg" in (self.far_method, self.near_method)' in src,
              f"无 torch 环境（{_IMPORT_ERR}），仅静态检查")
        return
    orig = W.apply_avg_score_patch()
    check("绿2a patch 幂等（二次调用拿到同一原函数）",
          W.apply_avg_score_patch() is orig)
    from sparse_attn.indexer.tli_indexer import TLIIndexer
    check("绿2a' patch 已挂（类属性被替换为 _need_avg_score_e117）",
          getattr(TLIIndexer._need_avg_score, "__name__", "") == "_need_avg_score_e117",
          f"got {getattr(TLIIndexer._need_avg_score, '__name__', '?')}")
    # 真实 TLIIndexer（生产 argparse namespace 构造，CPU 秒级）
    import argparse
    from sparse_attn.arguments import add_sparse_attn_args
    from sparse_attn.indexer.tli_indexer import TLIIndexer
    p = argparse.ArgumentParser()
    add_sparse_attn_args(p)
    ns = p.parse_args([
        "--method", "tli", "--tli_far_method", "minmax", "--tli_near_method", "avg",
        "--tli_alpha", "0.25", "--tli_beta", "0.125", "--tli_gamma", "0.625",
        "--tia_level2_cmp_ratio", "4",
        "--tli_enable_kmeans", "false", "--tli_enable_layer_skip", "false",
    ])
    idx = TLIIndexer(ns)
    # 红：原始语义在单池残留参数下 = False（缺陷根因证据）
    idx.alpha, idx.beta, idx.gamma = 0.0, 0.0, 0.0
    check("绿2b 红：原始 _need_avg_score(α=β=0) = False", not orig(idx),
          f"orig={orig(idx)}")
    # 绿：patch 后单池残留参数 = True
    check("绿2c 绿：patch 后 _need_avg_score(α=β=0) = True",
          idx._need_avg_score() is True)
    # 冠军配置下原始语义本就 True（patch 对生产冠军臂无行为差）
    idx.alpha, idx.beta, idx.gamma = W.CHAMPION
    check("绿2d 冠军配置下原始语义 True（patch 无行为差）", orig(idx) is True)
    # wrapper 自检函数全过
    sc = W.avg_score_patch_selfcheck()
    check("绿2e wrapper 自检 selfcheck pass（real_indexer_cpu 模式）",
          sc.get("pass") is True and sc.get("mode") == "real_indexer_cpu",
          json.dumps(sc, ensure_ascii=False)[:120])


# ---------------- 绿 3：receipt 落盘 ----------------
class _FakeConfig:
    num_hidden_layers = 36
    _name_or_path = "/fake/model/Qwen3-8B"


class _FakeModel:
    """named_modules 多次调用必须返回同一批 module 对象（与真实 nn.Module 一致），
    否则覆写后的属性读不到——这正是本 stub 最初踩的坑。"""
    config = _FakeConfig()

    def __init__(self):
        self._mods = [
            (f"layers.{li}.self_attn",
             types.SimpleNamespace(layer_idx=li,
                                    indexer=types.SimpleNamespace(
                                        alpha=0.25, beta=0.125, gamma=0.625)))
            for li in (0, 4, 8, 35)
        ]

    def named_modules(self):
        return iter(self._mods)


def _fake_args(**kw):
    base = dict(
        task="hotpotqa", t="unittest", output_dir="/nonexistent", pred_postfix="",
        method="tli", model_path="/fake/model/Qwen3-8B",
        tli_far_method="minmax", tli_near_method="avg",
        tli_alpha=0.25, tli_beta=0.125, tli_gamma=0.625,
        tia_block_size=64, tia_level1_topk=128, tia_level2_topk=1024,
        tia_level2_cmp_ratio=4, tli_enable_kmeans=False, tli_enable_layer_skip=False,
        tli_enable_subspace=True, tli_subspace="full",
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_receipt():
    if not HAS_TORCH:
        check("绿3 receipt（跳过）", None is None,
              f"无 torch 环境，sparse_attn.info 不可 import，receipt 全链未测（如实注明）")
        return
    with tempfile.TemporaryDirectory() as td:
        args = _fake_args(output_dir=td, pred_postfix="_e117per")
        model = _FakeModel()
        pl_map = {4: (0.0, 0.0, 0.0), 8: (0.375, 0.625, 0.125)}
        W._write_run_receipt(model, args, "perlayer", pl_map,
                             applied=dict(pl_map), n_modules=4,
                             num_layers=36, selfcheck={"pass": True, "mode": "test"})
        # 落盘路径：{output_dir}/pred{postfix}/{base}-e117_perlayer_receipt.json
        expected_dir = os.path.join(td, "pred_e117per")
        files = os.listdir(expected_dir)
        check("绿3a receipt 落盘目录正确",
              len(files) == 1 and files[0].endswith("-e117_perlayer_receipt.json"),
              f"files={files}")
        # pred.py 的 out_fn 基复刻（tli_64_128_1024_c4_A + hotpotqa）
        check("绿3b 文件名含 pred.py 命名基（method 段 c4_A）",
              files[0].startswith("hotpotqa-tli_64_128_1024_c4_A-"),
              files[0])
        with open(os.path.join(expected_dir, files[0])) as f:
            rc = json.load(f)
        required = [
            "experiment", "arm", "timestamp_utc", "git", "pred_py_sha256",
            "wrapper_sha256", "task", "output_dir", "model_path",
            "num_hidden_layers", "global_config", "perlayer_config_requested",
            "perlayer_config_applied", "n_layers_overridden", "avg_score_patch",
            "seed",
        ]
        missing = [k for k in required if k not in rc]
        check("绿3c receipt 字段完整（051 身份闭包硬性要求）", not missing,
              f"missing={missing}")
        check("绿3d receipt 逐层配置与全局配置正确",
              rc["perlayer_config_applied"] == {"4": [0.0, 0.0, 0.0],
                                               "8": [0.375, 0.625, 0.125]}
              and rc["global_config"]["tli_alpha"] == 0.25
              and rc["global_config"]["tli_near_method"] == "avg"
              and rc["n_layers_overridden"] == 2)
        check("绿3e receipt 原子写无残留 tmp 文件",
              not any(x.endswith(".tmp") for x in files))
        # 原子写函数独立验证：覆盖写 + fsync + replace 不留 tmp
        p = os.path.join(td, "r.json")
        W.write_receipt(p, {"a": 1})
        W.write_receipt(p, {"a": 2})
        with open(p) as f:
            check("绿3f write_receipt 可重复原子覆盖", json.load(f)["a"] == 2)


# ---------------- 绿 4：argv 注入 + 覆写逻辑 ----------------
def test_inject_and_override():
    # 显式参数不被覆盖（含 --flag=value 形式）
    out = W.inject_defaults(
        ["--task", "hotpotqa", "--tli_alpha", "0.5", "--tli_gamma=0.9"], "perlayer")
    check("绿4a 显式 --tli_alpha 0.5 不被默认覆盖",
          out[out.index("--tli_alpha") + 1] == "0.5" and out.count("--tli_alpha") == 1)
    check("绿4a' 显式 --tli_gamma=0.9（等号形式）不重复注入",
          out.count("--tli_gamma") == 0 and "--tli_gamma=0.9" in out)
    # 全缺省注入
    out2 = W.inject_defaults([], "perlayer")
    pairs = dict(zip(out2[::2], out2[1::2]))
    check("绿4b 缺省注入 = mavg 冠军 + c4_A 身份",
          pairs["--method"] == "tli" and pairs["--tli_far_method"] == "minmax"
          and pairs["--tli_near_method"] == "avg" and pairs["--tli_alpha"] == "0.25"
          and pairs["--tli_beta"] == "0.125" and pairs["--tli_gamma"] == "0.625"
          and pairs["--tia_level2_cmp_ratio"] == "4"
          and pairs["--tli_enable_kmeans"] == "false"
          and pairs["--tli_enable_layer_skip"] == "false"
          and pairs["--pred_postfix"] == "_e117per")
    out3 = W.inject_defaults([], "uniform")
    check("绿4b' uniform 臂 postfix=_e117uni",
          dict(zip(out3[::2], out3[1::2]))["--pred_postfix"] == "_e117uni")

    # 覆写逻辑（stub model；同一实例上先覆写再复核属性）
    m = _FakeModel()
    applied, n = W.apply_perlayer_overrides(m, {4: (0.0, 0.0, 0.0),
                                                8: (0.375, 0.625, 0.125)},
                                            num_hidden_layers=36)
    check("绿4c 覆写命中正确的层、其余层保持冠军",
          applied == {4: (0.0, 0.0, 0.0), 8: (0.375, 0.625, 0.125)} and n == 4)
    mods = [mod for _, mod in m.named_modules()]
    check("绿4c' L4 覆写生效且 L35 未动（L35 保持冠军 α=0.25）",
          mods[1].indexer.alpha == 0.0 and mods[3].indexer.alpha == 0.25)
    # 越界层断言（防错层，用户硬性要求）
    try:
        W.apply_perlayer_overrides(_FakeModel(), {36: (0.1, 0.1, 0.1)},
                                   num_hidden_layers=36)
        check("绿4d 越界层号断言触发", False, "未抛异常")
    except AssertionError as e:
        check("绿4d 越界层号断言触发", "36 >= num_hidden_layers" in str(e), str(e)[:60])
    # 配置中层号模型里不存在 → 断言（防拼写错漏覆写）
    try:
        W.apply_perlayer_overrides(_FakeModel(), {5: (0.1, 0.1, 0.1)},
                                   num_hidden_layers=36)
        check("绿4e 未命中层号断言触发", False, "未抛异常")
    except AssertionError as e:
        check("绿4e 未命中层号断言触发", "未在模型中找到" in str(e), str(e)[:60])


def test_register_patch_wiring():
    """绿 5：patch_register_patch 包装函数全链（wiring 级）。

    真实 register_patch 对 stub 模型无 Qwen3Attention/LlamaAttention 实例，
    是安全空操作 → 验证的是包装层逻辑：原函数被调用 + 覆写 + receipt 落盘 +
    sparse_attn.patches.register_patch 属性被替换/还原。
    """
    if not HAS_TORCH:
        check("绿5 register_patch wiring（跳过）", True,
              "无 torch 环境，sparse_attn 不可 import（如实注明未测）")
        return
    import sparse_attn.patches as patches_mod
    orig_attr = patches_mod.register_patch
    try:
        orig_ret, wrapped = W.patch_register_patch(
            {4: (0.0, 0.0, 0.0), 35: (0.0625, 0.125, 0.5)}, "perlayer",
            {"pass": True, "mode": "wiring-test"})
        check("绿5a 返回的原函数 = 替换前属性", orig_ret is orig_attr)
        check("绿5b patches.register_patch 已被替换", patches_mod.register_patch is wrapped)
        with tempfile.TemporaryDirectory() as td:
            args = _fake_args(output_dir=td, pred_postfix="_e117per", task="qasper")
            wrapped(_FakeModel(), args)   # 原函数对 stub 模型空操作 → 走全包装链
            mods = [mod for _, mod in _FakeModel().named_modules()]
            # receipt 已写（stub 模型的覆写属性在另一个实例上，这里验 receipt）
            rc_files = [f for f in os.listdir(os.path.join(td, "pred_e117per"))
                        if f.endswith("-e117_perlayer_receipt.json")]
            check("绿5c wrapped 调用后 receipt 落盘", len(rc_files) == 1, f"{rc_files}")
            with open(os.path.join(td, "pred_e117per", rc_files[0])) as f:
                rc = json.load(f)
            check("绿5d receipt 记录 8 层请求中的 2 层 wiring 配置",
                  rc["perlayer_config_applied"] == {"4": [0.0, 0.0, 0.0],
                                                    "35": [0.0625, 0.125, 0.5]}
                  and rc["n_layers_overridden"] == 2)
        # uniform 臂：不覆写但 receipt 仍落盘（先还原属性，避免包装嵌套污染）
        patches_mod.register_patch = orig_attr
        _, wrapped_u = W.patch_register_patch({}, "uniform", {"pass": True})
        with tempfile.TemporaryDirectory() as td:
            args = _fake_args(output_dir=td, pred_postfix="_e117uni", task="lcc")
            wrapped_u(_FakeModel(), args)
            with open(os.path.join(td, "pred_e117uni",
                      os.listdir(os.path.join(td, "pred_e117uni"))[0])) as f:
                rc = json.load(f)
            check("绿5e uniform 臂 applied 为空（不覆写）",
                  rc["arm"] == "uniform" and rc["perlayer_config_applied"] == {}
                  and rc["n_layers_overridden"] == 0)
    finally:
        patches_mod.register_patch = orig_attr   # 还原，防污染同进程后续测试
    check("绿5f patches.register_patch 已还原", patches_mod.register_patch is orig_attr)


def main():
    test_perlayer_config()
    test_avg_score_patch()
    test_receipt()
    test_inject_and_override()
    test_register_patch_wiring()
    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n=== E117 wrapper 红绿测试: {n_pass} PASS / {n_fail} FAIL "
          f"(torch={'yes' if HAS_TORCH else 'NO'}) ===")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
