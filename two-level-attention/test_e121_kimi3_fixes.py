#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E121 kimi3 清单（2026-10-08）A 组修复验收套件（CPU only，python/-O 双跑安全）。

覆盖修复与验收点（红绿证据：主树未修态实测红、worktree 修复态绿，标注在各
测试 docstring 的【红】【绿】行）：

  E121-A1 (R01)  pred.py BOS 感知切片：Qwen3 系 tokenizer 不加 BOS，
                 原无条件 [:, 1:] 会切掉 question 真实首 token。
                 —— 三态行为测试（FakeModel/FakeTokenizer 全进程内 mock，
                    不依赖 GPU/真实权重）：
                    态1 加 BOS：切 BOS 后 1 个 q-stage 调用（主树同绿，回归位）
                    态2 不加 BOS：不切，1 个 q-stage 调用【红：主树切成 0 个】
                    态3 声明 bos_token_id 但实际未加：不切【红：主树切成 0 个】
  E121-A3 (B04)  pred.py 首 token EOS 判停：原循环漏判首 token。
                    正例 首 token=EOS → 0 次额外生成调用【红：主树 +3 次】
                    负例 首 token≠EOS → 照常生成 max_gen-1 步（两侧同绿）
  E121-A4 (B09+F11) info.py method_name 进 α/β/γ + ab 用生效值：
                    红主树名 = "tli_64_128_1024_c4_A"（无 αβγ 且 A 虚标）
  E121-A2 (B10)  near 左界从 swa 起点推（kimi3 §8 审计原案例 S=4416/swa=192）：
                    HF 建簇侧 far_hi 2368→2176（near POOL 1856→2048=α·mid）
                    SG _taskmd_regions 同源 34 块；(0,0) 单池分支逐位不变
  E121-A5 (F5)   compute_mask e64 臂用真实 S（score_fine 宽）而非 pad kt*bs：
                    非对齐 S=6145 mavg 冠军 mid 577→768（=K2_mid 满额）
  E121-A2-SG     B10 SG 侧区域公式（_taskmd_regions 审计案例 + 单池不变位）

依赖注入：sparse_attn / benchmark.LongBench 以 __path__ 注入（不污染进程外
环境）；transformers/datasets 走真实安装（本套件只用到 pred.py 的
get_pred 函数与 FakeModel，不加载权重）。

用法：python3 test_e121_kimi3_fixes.py   （two-level-attention/ 下）
  红态复跑（对主树）：TLI_E121_ROOT=/home/wangyuanshuo02/sglang/two-level-attention \
                      TLI_E121_SG_ROOT=/home/wangyuanshuo02/sglang/python \
                      python3 test_e121_kimi3_fixes.py
"""
import os
import sys
import types

sys.dont_write_bytecode = True

import torch

torch.set_num_threads(1)

# 根路径：默认本文件所在树（worktree 检出即测 worktree 代码）；TLI_E121_ROOT
# 可指主树做红态复跑（只读，不写主树）。
ROOT = os.environ.get("TLI_E121_ROOT", os.path.dirname(os.path.abspath(__file__)))
SG_ROOT = os.environ.get("TLI_E121_SG_ROOT",
                         os.path.normpath(os.path.join(ROOT, "..", "python")))

# 状态显式化：全部用显式条件检查（非 assert——python -O 会剥 assert，
# 门禁退化为恒过；本套件 -O 双跑是验收硬指标）。
RESULTS = []   # (name, status, detail)


def report(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    RESULTS.append((name, status, detail))
    print(f"{status}  {name}" + (f"  -- {detail}" if detail else ""))


def _require(cond, msg):
    """-O 安全断言：显式 raise，不依赖 assert。"""
    if not cond:
        raise RuntimeError(msg)


def _register_pkg(name, path):
    """命名空间包注入（sparse_attn 无顶层 __init__.py 的既有模式）。"""
    mod = types.ModuleType(name)
    mod.__path__ = [path]
    sys.modules[name] = mod
    return mod


_register_pkg("sparse_attn", os.path.join(ROOT, "sparse_attn"))
_register_pkg("benchmark", os.path.join(ROOT, "benchmark"))
_register_pkg("benchmark.LongBench", os.path.join(ROOT, "benchmark", "LongBench"))

import importlib  # noqa: E402

IDX_PKG = importlib.import_module("sparse_attn.indexer")
INFO_MOD = importlib.import_module("sparse_attn.info")

# SG 侧（B10 同步验收）：python/sglang CPU 可导入（e112 全套先例）
sys.path.insert(0, SG_ROOT)
from sglang.srt.layers.attention.tli.indexer import TLIIndexer as TLIIndexerSG  # noqa: E402
from sglang.srt.layers.attention.tli.config import TLIProfile  # noqa: E402

# pred.py：真实模块导入（transformers/datasets 用本机安装；get_pred 只在
# FakeModel 上跑，不触 GPU/权重）。导入耗时 ~10-30s 属正常。
PRED_MOD = importlib.import_module("benchmark.LongBench.pred")


# ================================================================ 公共构造
def make_args(**kw):
    a = types.SimpleNamespace(
        tia_block_size=64,
        tia_level1_topk=128,
        tia_level2_topk=1024,
        tia_level2_cmp_ratio=4,
        tia_enable_async_topk=False,
    )
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def gen_kq(S, seed=31):
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(1, S, 2, 128, generator=g) * 0.5
    q = torch.randn(1, 1, 4, 128, generator=g) * 0.5
    return k, q


def run_mask(args, k, q, swa=None):
    idx = IDX_PKG.TLIIndexer(args)
    if swa is not None:
        idx.sliding_window_size = swa     # 审计案例 swa=192（默认 128）
    idx.layer_idx = 3
    cu = torch.tensor([0, k.shape[1]])
    q_ids = torch.tensor([k.shape[1] - 1])
    mask, _ = idx.prepare_mask(q, q_ids, k, cu, q.shape[-1] ** -0.5)
    return idx, mask


CCLUSTER_CFG = dict(
    tli_far_select="cluster", tli_near_select="cluster",
    tli_enable_kmeans=True, tli_enable_layer_skip=False,
    tli_far_method="minmax", tli_near_method="avg",
)


# ================================================================ Fake 模型/分词器
class _TokOut:
    """tokenizer(...) 返回体：input_ids [1,L] + .to(device) 链式（设备无关）。"""

    def __init__(self, ids):
        self.input_ids = torch.tensor([ids])

    def to(self, *a, **kw):
        return self


class FakeTokenizer:
    """字符→token id 确定性映射：id = 3 + ord(c)%40（∈[3,42]，不撞 BOS=1/EOS=2）。

    adds_bos=True 时 __call__ 附加 BOS（模拟 Llama 系）；bos_token_id=None
    模拟 Qwen3 系（不加 BOS）。
    """

    def __init__(self, bos_token_id=None, eos_token_id=2, adds_bos=False):
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.adds_bos = adds_bos

    def __call__(self, text, truncation=False, return_tensors="pt"):
        ids = [3 + (ord(c) % 40) for c in text]
        if self.adds_bos and self.bos_token_id is not None:
            ids = [self.bos_token_id] + ids
        return _TokOut(ids)

    def decode(self, ids, skip_special_tokens=True):
        out = []
        for i in ids:
            if skip_special_tokens and i == self.eos_token_id:
                continue
            out.append(chr(60 + i))
        return "".join(out)

    def encode(self, text, add_special_tokens=False):
        return [3 + (ord(c) % 40) for c in text]


class _ModelOut:
    def __init__(self, logits, past):
        self.logits = logits
        self.past_key_values = past


class FakeModel:
    """按 script 逐调用发射 argmax token；记录每次收到的 input_ids。"""

    VOCAB = 16

    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self._past = object()

    def __call__(self, input_ids=None, past_key_values=None, use_cache=True, **kw):
        self.calls.append(input_ids.clone()
                          if isinstance(input_ids, torch.Tensor) else input_ids)
        step = len(self.calls) - 1
        tok = self.script[min(step, len(self.script) - 1)]
        L = int(input_ids.shape[-1])
        logits = torch.zeros(1, L, self.VOCAB)
        logits[0, -1, tok] = 10.0
        return _ModelOut(logits, self._past)


def run_get_pred(tok, model, max_gen=4, max_length=64):
    """跑 pred.get_pred 一行数据（dataset='zzz' 不进任何特殊分支）。"""
    data = [{"context": "abcdefgh",
             "answers": ["ans"], "all_classes": None, "length": 8}]
    preds = PRED_MOD.get_pred(
        model, tok, data, max_length=max_length, max_gen=max_gen,
        prompt_format="{context}", dataset="zzz", device="cpu",
        model_name="zzz-model", data_fp="fp",
    )
    return preds


# ================================================================ E121-A1 (R01)
def a1_bos_aware_slice():
    """R01：BOS 感知切片三态。

    【红】态2/态3 在主树（无条件 [:, 1:]）：q-stage 0 次调用、首 token 取自
        prefill 输出 → 本测试两态均 FAIL。
    【绿】修复态：态1 切 BOS；态2/态3 不切，q-stage 恰 1 次且喂真实首 token。
    """
    name = "A1(R01) BOS 感知切片三态（加BOS切/不加BOS不切/声明BOS未加不切）"
    try:
        # ---- 态1：tokenizer 加 BOS（Llama 系）→ 切掉 BOS，喂 1 个真实 q token
        tok = FakeTokenizer(bos_token_id=1, eos_token_id=2, adds_bos=True)
        mdl = FakeModel(script=[5, 5, 5, 5, 5, 5])
        run_get_pred(tok, mdl)
        # calls[0]=prefill(prompt)，calls[1..]=q-stage（1 次），其后生成
        h_id = 3 + (ord("h") % 40)
        _require(len(mdl.calls) >= 2, f"态1 q-stage 调用缺失（共 {len(mdl.calls)}）")
        q_calls = mdl.calls[1:len(mdl.calls) - (4 - 1)]  # max_gen=4 → 生成至多 3 次
        _require(len(q_calls) == 1,
                 f"态1 加 BOS：q-stage 应恰 1 次调用，实际 {len(q_calls)}")
        _require(int(q_calls[0].reshape(-1)[0]) == h_id,
                 f"态1 切后应喂真实首 token {h_id}，实际 {int(q_calls[0].reshape(-1)[0])}")

        # ---- 态2：tokenizer 不加 BOS（Qwen3 系）→ 不切，1 次 q-stage 喂首 token
        # 【红绿判别位】主树无条件切片 → 0 次 q-stage → 本段必 FAIL
        tok = FakeTokenizer(bos_token_id=None, eos_token_id=2, adds_bos=False)
        mdl = FakeModel(script=[5, 5, 5, 5, 5, 5])
        run_get_pred(tok, mdl)
        gen_calls = 4 - 1     # 首 token 非 EOS（script=5≠2）→ 生成 max_gen-1 次
        q_calls = mdl.calls[1:len(mdl.calls) - gen_calls]
        _require(len(q_calls) == 1,
                 f"态2 不加 BOS：q-stage 应恰 1 次（真实首 token 必须进模型），"
                 f"实际 {len(q_calls)}——R01 缺陷复现（question 首 token 被切）")
        _require(int(q_calls[0].reshape(-1)[0]) == h_id,
                 f"态2 应喂真实首 token {h_id}，实际 {int(q_calls[0].reshape(-1)[0])}")

        # ---- 态3：声明 bos_token_id=1 但实际未加 BOS → 首 token 非 BOS 不切
        # 【红绿判别位】主树按「声明了 bos」切片 → 0 次 q-stage → FAIL
        tok = FakeTokenizer(bos_token_id=1, eos_token_id=2, adds_bos=False)
        mdl = FakeModel(script=[5, 5, 5, 5, 5, 5])
        run_get_pred(tok, mdl)
        q_calls = mdl.calls[1:len(mdl.calls) - gen_calls]
        _require(len(q_calls) == 1,
                 f"态3 声明 BOS 未加：q-stage 应恰 1 次（首 token 非 BOS 不切），"
                 f"实际 {len(q_calls)}")
        _require(int(q_calls[0].reshape(-1)[0]) == h_id,
                 "态3 应喂真实首 token（非 BOS 不切）")
        report(name, True, "三态全过：切/不切均以「首 token 是否真为 BOS」为准")
    except RuntimeError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ E121-A3 (B04)
def a3_first_token_eos():
    """B04：首 token 即 EOS 必须停（原循环在 push 后才判停，首 token 漏判）。

    【红】主树：首 token=EOS 后继续生成 max_gen-1=3 次（共 5 次模型调用），
        本测试 FAIL。
    【绿】修复态：首 token=EOS → 0 次生成调用（共 2 次），pred 为空。
    """
    name = "A3(B04) 首 token EOS 判停（正例停/负例照常生成）"
    try:
        # 正例：首 token 即 EOS（FakeModel 逐调用发射：call0=prefill、call1=
        # q-stage——首个生成 token 取自 q-stage 输出，故 script[1]=EOS）
        tok = FakeTokenizer(bos_token_id=None, eos_token_id=2, adds_bos=False)
        mdl = FakeModel(script=[5, 2, 9, 9, 9, 9])   # 首生成 token=EOS
        preds = run_get_pred(tok, mdl, max_gen=4)
        _require(len(mdl.calls) == 2,
                 f"首 token=EOS 应立即停（prefill+q 共 2 次调用），"
                 f"实际 {len(mdl.calls)} 次——B04 缺陷复现（首 token 不停）")
        _require(preds[0]["pred"] == "",
                 f"首 token=EOS 的 pred 应为空串（skip_special_tokens），"
                 f"实际 {preds[0]['pred']!r}")

        # 负例：首 token 非 EOS → 照常生成 max_gen-1 步（回归位，两侧同绿）
        mdl = FakeModel(script=[5, 5, 5, 5, 5, 5])
        preds = run_get_pred(tok, mdl, max_gen=4)
        _require(len(mdl.calls) == 2 + 3,
                 f"首 token≠EOS 应生成 max_gen-1=3 步（共 5 次调用），"
                 f"实际 {len(mdl.calls)}")
        _require(len(preds[0]["pred"]) == 4,
                 f"非 EOS 应产出 4 个可见 token，实际 {preds[0]['pred']!r}")
        report(name, True, "EOS 首停 2 次调用；非 EOS 5 次（max_gen-1 生成步）")
    except RuntimeError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ E121-A4 (B09+F11)
def a4_method_name_abg():
    """B09+F11：method_name 进 α/β/γ；A 用生效值（subspace=full 强制关 → 不写 A）。

    【红】主树：冠军配置名 = "tli_64_128_1024_c4_A"（无 αβγ；A 虚标——full 口径
        下 enable_subspace 实为 False），本测试 FAIL。
    """
    name = "A4(B09+F11) method_name 进 α/β/γ + ab 生效值 + 非 tli 方法不回归"
    try:
        def mk(**kw):
            base = dict(method="tli", tia_block_size=64, tia_level1_topk=128,
                        tia_level2_topk=1024, tia_level2_cmp_ratio=4,
                        tli_subspace="full", tli_enable_subspace=True,
                        tli_enable_kmeans=False, tli_enable_layer_skip=False,
                        tli_alpha=0.25, tli_beta=0.125, tli_gamma=0.625,
                        tli_sparse_prefill=False)
            base.update(kw)
            return types.SimpleNamespace(**base)

        f = INFO_MOD.get_method_name_with_info
        # B09：冠军配置 α/β/γ 进名；F11：full 口径不写 A（生效值）
        got = f(mk())
        _require(got == "tli_64_128_1024_c4_a0.25_b0.125_g0.625",
                 f"冠军配置名应含 α/β/γ 且 full 不写 A，实际 {got!r}")
        # F11 正向：rope 子空间（生效开）→ 写 A，即使 flag 原值 False
        # （名格式：ab 直拼 a 段——c4_Aa0.25，无分隔下划线）
        got = f(mk(tli_subspace="rope", tli_enable_subspace=False))
        _require(got == "tli_64_128_1024_c4_Aa0.25_b0.125_g0.625",
                 f"rope 生效子空间应写 A（不受 flag 原值 False 影响），实际 {got!r}")
        # B/D 与 γ None（"off"）
        got = f(mk(tli_subspace="rope", tli_enable_kmeans=True,
                   tli_enable_layer_skip=True, tli_alpha=0.0, tli_beta=0.0,
                   tli_gamma=None))
        _require(got == "tli_64_128_1024_c4_ABDa0_b0_goff",
                 f"kmeans+layer_skip 应写 BD；γ=None 应写 off，实际 {got!r}")
        # 稀疏 prefill 后缀 _P
        got = f(mk(tli_sparse_prefill=True))
        _require(got == "tli_64_128_1024_c4_a0.25_b0.125_g0.625_P",
                 f"稀疏 prefill 应加 _P 后缀，实际 {got!r}")
        # 小数格式化（0.0625 不丢位）
        got = f(mk(tli_alpha=0.0625, tli_beta=0.03125, tli_gamma=0.5))
        _require(got == "tli_64_128_1024_c4_a0.0625_b0.03125_g0.5",
                 f"α/β/γ 小数应完整保位，实际 {got!r}")
        # 非 tli 方法不回归（quest/twia/tia/none）
        _require(f(types.SimpleNamespace(
            method="quest", quest_block_size=64, quest_topk=128)) == "quest_64_128",
            "quest 分支回归")
        _require(f(types.SimpleNamespace(
            method="twi", twi_block_size=64, twi_level1_topk=128,
            twi_level2_topp=0.9)) == "twi_64_128_0.9", "twia 分支回归")
        _require(f(types.SimpleNamespace(
            method="tia", tia_block_size=64, tia_level1_topk=128,
            tia_level2_topk=1024, tia_level2_cmp_ratio=4,
            tia_enable_async_topk=False)) == "tia_64_128_1024_c4", "tia 分支回归")
        _require(f(types.SimpleNamespace(
            method="tia", tia_block_size=64, tia_level1_topk=128,
            tia_level2_topk=1024, tia_level2_cmp_ratio=4,
            tia_enable_async_topk=True)) == "tia_64_128_1024_c4_async",
            "tia async 分支回归")
        _require(f(types.SimpleNamespace(method="none")) == "none", "none 分支回归")
        report(name, True, "α/β/γ/P/ABD/off 全格式过；quest/twia/tia/none 零回归")
    except RuntimeError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ E121-A2 (B10) HF
def a2_b10_hf_audit_case():
    """B10：kimi3 §8 审计原案例 S=4416, sink=128, swa=192, bs=64, α=0.5。

    spec：mid = 4416-128-192 = 4096；near_L = α·mid = 2048；
    near 区间 = [S−swa−near_L, S−swa) = [2176, 4224)。
    【红】主树：near_blks = (4416-2048)//64 = 37 → 起点 2368 → near POOL
        = 4224-2368 = 1856（swa 被扣两次，缺陷实锤值与审计文档逐位一致）。
    【绿】修复态：起点 2176 → POOL 2048 = α·mid。
    """
    name = "A2(B10) HF 审计案例 S=4416/swa=192：near POOL=2048（红 1856）"
    try:
        S, bs, sink_tok, swa_tok = 4416, 64, 128, 192
        # 公式重放（与 _maybe_build_kmeans / compute_mask B10 修复后口径同源）
        mid_len = S - sink_tok - swa_tok                  # 4096
        near_len_dyn = max(bs, int(0.5 * mid_len))        # 2048
        near_base = S - swa_tok                           # 4224（B10）
        old_far_hi = ((S - near_len_dyn) // bs) * bs      # 2368（审计缺陷值）
        new_far_hi = ((near_base - near_len_dyn) // bs) * bs   # 2176
        _require(mid_len == 4096 and near_len_dyn == 2048,
                 f"公式重放失败 mid_len={mid_len} near_len_dyn={near_len_dyn}")
        _require(new_far_hi == 2176 and old_far_hi == 2368,
                 f"公式重放 far_hi 应 新2176/旧2368，实际 新{new_far_hi}/旧{old_far_hi}")
        k, q = gen_kq(S)
        args = make_args(**CCLUSTER_CFG, tli_alpha=0.5, tli_beta=0.25, tli_gamma=0.5)
        idx, mask = run_mask(args, k, q, swa=swa_tok)
        fh = int(idx._km_far_hi_cached)
        _require(fh == 2176,
                 f"建簇侧 far_hi 应 2176（B10 后；主树缺陷态 2368），实际 {fh}")
        pool_w = (S - swa_tok) - fh
        _require(pool_w == 2048,
                 f"near POOL 宽应 = α·mid = 2048（主树缺陷态 1856），实际 {pool_w}")
        # 消费侧完整性：sink/swa 强制齐、mid 配额闭包（K2_mid=704 饱食）
        mm = mask[0, 0]
        _require(bool(mm[..., :sink_tok].all()), "sink 强制区缺失")
        _require(bool(mm[..., S - swa_tok:].all()), "swa 强制区缺失")
        n_mid = int(mm[0, sink_tok:S - swa_tok].sum())
        _require(n_mid == 704,
                 f"mid 应 = K2_mid = 704（1024-128-192），实际 {n_mid}")
        _require(int(mm[0].sum()) == 1024, f"总选中应 K2=1024，实际 {int(mm[0].sum())}")
        report(name, True, f"far_hi={fh}(红2368)、POOL={pool_w}=α·mid、mid={n_mid} 饱食")
    except RuntimeError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


def a2_b10_hf_single_pool_invariant():
    """B10 不变量：(0,0) 单池分支 far_hi 逐位不变（S−swa 恰为正确单池语义）。

    【红/绿同值】本项在主树与修复态都应过——回归保护位（B10 若破坏单池
    语义此处红）。旧口径 (S−swa_tok)//bs 与新 near_base=S 分支算术同值。
    """
    name = "A2(B10) (0,0) 单池分支不变：far_hi = (S−swa)块对齐 = 4224"
    try:
        S, bs, swa_tok = 4416, 64, 192
        k, q = gen_kq(S)
        args = make_args(tli_far_select="cluster", tli_near_select="4bit",
                         tli_enable_kmeans=True, tli_enable_layer_skip=False,
                         tli_far_method="minmax", tli_near_method="avg",
                         tli_alpha=0.0, tli_beta=0.0)
        idx, mask = run_mask(args, k, q, swa=swa_tok)
        fh = int(idx._km_far_hi_cached)
        expect = ((S - swa_tok) // bs) * bs      # 4224
        _require(fh == expect,
                 f"(0,0) 单池 far_hi 应保持旧式 {expect}（B10 只动 e64 臂），实际 {fh}")
        _require(bool(mask[..., :128].all()) and bool(mask[..., S - swa_tok:].all()),
                 "sink/swa 强制区缺失")
        report(name, True, f"far_hi={fh}=S−swa 块对齐（单池语义不受 B10 扰动）")
    except RuntimeError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ E121-A2-SG (B10)
def a2_b10_sg_regions():
    """B10 SG 侧同源：_taskmd_regions 审计案例 + 单池不变位。

    【红】主树 SG：near_blks = (4416-2048)//64 = 37（与 HF 同构缺陷）→ 本项 FAIL。
    """
    name = "A2(B10) SG _taskmd_regions：审计案例 34 块/POOL 2048；单池 66 块不变"
    try:
        def prof(alpha, beta, gamma, swa=128):
            p = TLIProfile()
            p.taskmd = True
            p.alpha, p.beta, p.gamma = alpha, beta, gamma
            p.far_method, p.near_method = "minmax", "avg"
            p.block_size = 64
            p.coarse_dim = 128
            p.k1_blocks = 128
            p.token_budget = 1024
            p.sliding_window = swa
            p.sink_blocks = 2
            p.delta = 16
            p.near_len = 2048
            p.sliding_blocks = 3
            p.far_tokens = 256
            p.dense_threshold = 2048
            return p

        # 审计案例（e64 臂，swa=192）：near_blks 34 → near POOL = 4224-2176 = 2048
        idxsg = TLIIndexerSG(prof(0.5, 0.25, 0.5, swa=192), head_dim=128)
        r = idxsg._taskmd_regions(4416)
        _require(r["e64"] is True, "α/β>0 应判 e64")
        _require(r["near_blks"] == 34,
                 f"SG near_blks 应 34（B10 后；主树同构缺陷 37），实际 {r['near_blks']}")
        pool_w = (4416 - 192) - r["near_blks"] * 64
        _require(pool_w == 2048, f"SG near POOL 应 2048 = α·mid，实际 {pool_w}")
        _require(r["far_hi_blk"] == 34, f"far_hi_blk 应 34，实际 {r['far_hi_blk']}")
        _require(r["swa_lo_blk"] == 66, f"swa_lo_blk 应 66（69-3），实际 {r['swa_lo_blk']}")

        # 单池不变位（α=0）：near_len_dyn=swa、near_base=S → 66 块（新旧同值）
        r0 = TLIIndexerSG(prof(0.0, 0.0, 1.0, swa=192), head_dim=128)._taskmd_regions(4416)
        _require(r0["e64"] is False and r0["near_blks"] == 66,
                 f"(0,0) 单池 near_blks 应 66（不受 B10 扰动），实际 {r0['near_blks']}")
        report(name, True, f"e64 near_blks={r['near_blks']}(红37)/POOL={pool_w}；"
                           f"单池 {r0['near_blks']} 不变")
    except RuntimeError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ E121-A5 (F5)
def a5_f5_real_s():
    """F5：compute_mask e64 臂 mid/near_base 用真实 S（score_fine 宽）而非 pad。

    非对齐 S=6145（decode 每步多数不整除 bs 的典型形态），mavg 冠军
    (.125/.375/.625) 4bit 臂：
    【红】主树（pad 口径 kt*bs=6208）：near 池上界被 pad 抬过真实 swa 起点
        6017 → mid 选中 577（池内截断 + 越界段不可见）。
    【绿】修复态：near_base=6017 → 池宽 769 ≥ 配额 768 → mid 满额 768。
    对齐 S（4352）行为逐位不变（回归位，两侧同绿）。
    """
    name = "A5(F5) 非对齐 S=6145 e64 臂真实 S：mid=768 满额（红 577）"
    try:
        mavg_cfg = dict(tli_far_select="4bit", tli_near_select="4bit",
                        tli_enable_kmeans=False, tli_enable_layer_skip=False,
                        tli_far_method="minmax", tli_near_method="avg",
                        tli_alpha=0.125, tli_beta=0.375, tli_gamma=0.625)
        # 非对齐案例
        S = 6145
        k, q = gen_kq(S)
        _, mask = run_mask(make_args(**mavg_cfg), k, q)
        n_mid = int(mask[0, 0][0, 128:S - 128].sum())
        _require(n_mid == 768,
                 f"S=6145 mid 应 = K2_mid = 768 满额（主树 pad 口径缺陷态 577），"
                 f"实际 {n_mid}")
        _require(bool(mask[..., :128].all()) and bool(mask[..., S - 128:].all()),
                 "sink/swa 强制区缺失")
        # 对齐回归位（F5 对对齐 S 逐位不变；S=4352 mavg 冠军 mid=512 是
        # 已判定的 N5 合法饥饿——γ 悬崖：nt_near=min(1920,768)=768 但 near 池
        # 宽 = α·mid = 512 < 768 → 池内截断；主树/worktree 同值 512）
        S2 = 128 + 4096 + 128
        k2, q2 = gen_kq(S2)
        _, mask2 = run_mask(make_args(**mavg_cfg), k2, q2)
        n_mid2 = int(mask2[0, 0][0, 128:S2 - 128].sum())
        _require(n_mid2 == 512,
                 f"对齐 S=4352 mid 应 512（N5 合法饥饿，F5 不改对齐行为），"
                 f"实际 {n_mid2}")
        report(name, True, f"S=6145 mid={n_mid}（红577）；对齐 S=4352 mid={n_mid2} 不变")
    except RuntimeError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ main
if __name__ == "__main__":
    a1_bos_aware_slice()
    a3_first_token_eos()
    a4_method_name_abg()
    a2_b10_hf_audit_case()
    a2_b10_hf_single_pool_invariant()
    a2_b10_sg_regions()
    a5_f5_real_s()
    n_pass = sum(1 for _, s, _ in RESULTS if s == "PASS")
    n_fail = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    _require(n_pass + n_fail == len(RESULTS) == 7, "门禁不允许漏项")
    print(f"\n===== E121 kimi3 A 组验收：{n_pass}/{n_pass + n_fail} PASS"
          f"（-O 安全：全部显式检查非 assert）=====")
    sys.exit(1 if n_fail else 0)
