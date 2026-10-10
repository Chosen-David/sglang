#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E121 kimi3 清单（2026-10-08）A+B+C 组修复验收套件（CPU only，python/-O 双跑安全）。

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

  E121-B1 (F1)   skip 层 sink 强制：skip_far 落 legacy 尾部 topk 时 sink
                    p 恒 0 永不入选（S=4352 实测 0/128）【红：主树 sink=0】；
                    非 skip 层走分区路径不受扰（回归位两侧同绿）
  E121-B2 (F2)   qwen3/llama3 patch decode 分支入口硬守卫：
                    use_cache=False / chunked prefill(q_len>1) / batch>1 /
                    4D float mask 一律显式 RuntimeError【红：主树分别
                    AttributeError / 静默进 decode / 静默错算】；prefill
                    分支回归位两侧同绿
  E121-B3 (F3)   register_patch 计数 + 零匹配显式拒绝：GLM 系等未支持
                    架构跑 method≠none 不再静默 dense【红：主树静默返回】
  E121-B4 (F4)   σ 死参数 fail-closed：sigma≠none × (无分区|skip_far|moba)
                    组合显式 ValueError【红：主树静默退化】；合法 σ 臂
                    （分区开、非 moba）回归位两侧同绿
  E121-B5 (F7)   clear() 重置 _basis（投影基跨请求残留）【红：主树残留】
  E121-B6 (F10)  metrics.add_k_delta 极差改减法【红：主树 (max+min).abs()】
  E121-B7 (F12)  eager_decoding 全 False 行防线【红：主树 NaN 静默返回】
  E121-C1 (S6)   backend PCA basis 实际秩 vs proj_rank 校验【红：主树不校验】
  E121-C2 (S7)   SG indexer empty 行 pad 哨兵化（运行时不可达已证：w ≤
                    token_budget ≤ i_g 宽 → 只可能截断，静态契约测试）
  E121-C3 (S8)   config.py M10 过期注释（「哨兵转 0」为 B01 前口径）更正

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
        # 076（TL-E121-OUTPUT-ID）锚点口径：tli 名尾段追加 canonical
        # treatment hash _h<hash10>（全部输出相关参数排序键 JSON→sha256
        # 前 10 hex；缺省 getattr 与 argparse/TLIIndexer 运行时缺省同源
        # ——单射性/稳定性单测见 test_e119_fixes_076_077_078.py）。hash
        # 依赖 _TREATMENT_FIELD_DEFAULTS 参数集固定后逐位稳定；改字段集
        # 时本锚点必须重算同步更新（同名互覆防线 = 076 的存在理由）。
        # 079 口径：manifest 新增 tia_enable_async_topk（缺省 False）→
        # 全部 tli hash 重算，下列五个锚点已同步更新。
        # 081 + kimi3 0316 口径（B/D 开关入 manifest + 默认 D′ 掩码内容
        # 身份）：kimi3 0316 追加修复 2 把 tli_enable_kmeans/
        # tli_enable_layer_skip 从「可读段 B/D 承载」升格入
        # _TREATMENT_FIELD_DEFAULTS（截断吃掉可读段 B/D 位时身份仍由
        # hash 单射承载）→ 两字段进 manifest 对全部 tli 配置生效，
        # 五锚点整体重算；叠加 081 身份缺口 ① 修复（layer_skip=True
        # 且未声明 tli_layer_skip_path → manifest 解析 tracked
        # DEFAULT_MASK 内容身份），第三个锚点（唯一 layer_skip=True
        # 案例）双重漂移。
        # B09：冠军配置 α/β/γ 进名；F11：full 口径不写 A（生效值）
        got = f(mk())
        _require(got == "tli_64_128_1024_c4_a0.25_b0.125_g0.625_h562141be42",
                 f"冠军配置名应含 α/β/γ 且 full 不写 A，实际 {got!r}")
        # F11 正向：rope 子空间（生效开）→ 写 A，即使 flag 原值 False
        # （名格式：ab 直拼 a 段——c4_Aa0.25，无分隔下划线）
        got = f(mk(tli_subspace="rope", tli_enable_subspace=False))
        _require(got == "tli_64_128_1024_c4_Aa0.25_b0.125_g0.625"
                         "_h08a067c3c5",
                 f"rope 生效子空间应写 A（不受 flag 原值 False 影响），实际 {got!r}")
        # B/D 与 γ None（"off"）
        got = f(mk(tli_subspace="rope", tli_enable_kmeans=True,
                   tli_enable_layer_skip=True, tli_alpha=0.0, tli_beta=0.0,
                   tli_gamma=None))
        _require(got == "tli_64_128_1024_c4_ABDa0_b0_goff_h1ee5503c61",
                 f"kmeans+layer_skip 应写 BD；γ=None 应写 off，实际 {got!r}")
        # 稀疏 prefill 后缀 _P
        got = f(mk(tli_sparse_prefill=True))
        _require(got == "tli_64_128_1024_c4_a0.25_b0.125_g0.625_P"
                         "_hd891baff79",
                 f"稀疏 prefill 应加 _P 后缀，实际 {got!r}")
        # 小数格式化（0.0625 不丢位）
        got = f(mk(tli_alpha=0.0625, tli_beta=0.03125, tli_gamma=0.5))
        _require(got == "tli_64_128_1024_c4_a0.0625_b0.03125_g0.5"
                         "_h3396bc828c",
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


# ================================================================ E121-B1 (F1)
def b1_skip_layer_sink_force():
    """F1：skip 层（skip_far）落 legacy 尾部 topk 时 sink 正交强制。

    机制：e64 分区臂 L1 双池按设计排除 sink 块（sink 不进粗筛管线），
    skip_far 使 use_partition=False 落 legacy 尾部 → sink token p=0 永不
    入选（attention sink 永久丢失）。
    【红】主树：S=4352/mavg 冠军/skip 层 sink 选中 0/128（swa 由 p=1.0
        兜住 128/128）→ 本测试 FAIL。
    【绿】修复态：skip 层 sink 强制 128/128（不占 K2 预算，总选中 1152）；
        非 skip 层走分区路径不受扰（回归位）。
    """
    name = "B1(F1) skip 层 sink 正交强制（红 sink=0/128；绿 128/128）"
    try:
        mavg_cfg = dict(tli_far_select="4bit", tli_near_select="4bit",
                        tli_enable_kmeans=False, tli_enable_layer_skip=False,
                        tli_far_method="minmax", tli_near_method="avg",
                        tli_alpha=0.125, tli_beta=0.375, tli_gamma=0.625)
        S = 4352
        k, q = gen_kq(S)
        idx = IDX_PKG.TLIIndexer(make_args(**mavg_cfg))
        idx.layer_idx = 3
        idx.skip_far = True   # _skip_ids 为 None 时 _resolve_skip 不覆盖
        cu = torch.tensor([0, S])
        q_ids = torch.tensor([S - 1])
        mask, _ = idx.prepare_mask(q, q_ids, k, cu, q.shape[-1] ** -0.5)
        mm = mask[0, 0]
        n_sink = int(mm[..., :128].sum(dim=-1).min())
        _require(n_sink == 128,
                 f"skip 层 sink 应强制 128/128 token（主树缺陷态 0），"
                 f"实际最小 head 选中 {n_sink}/128——F1 缺陷复现（attention "
                 f"sink 永久丢失）")
        _require(bool(mm[..., S - 128:].all()), "skip 层 swa 强制区缺失")
        # sink 强制不占 K2 预算：总选中 = K2 + sink（1152）
        _require(int(mm[0].sum()) == 1024 + 128,
                 f"skip 层总选中应 = K2+sink = 1152（sink 正交不占预算），"
                 f"实际 {int(mm[0].sum())}")
        # 回归位：非 skip 层走分区路径，sink/swa 照常强制（两侧同绿）。
        # 总选中 768 = sink 128 + mid 512 + swa 128——S=4352 mavg 冠军的
        # N5 合法饥饿（near 池宽 α·mid=512 < 配额 768 → 池内截断，E98
        # 判例），与 A5 测试对齐 S=4352 mid=512 同口径。
        _, mask2 = run_mask(make_args(**mavg_cfg), k, q)
        _require(bool(mask2[0, 0][..., :128].all()) and
                 bool(mask2[0, 0][..., S - 128:].all()),
                 "非 skip 层 sink/swa 强制缺失（B1 扰动分区路径——回归）")
        _require(int(mask2[0, 0][0].sum()) == 768,
                 f"非 skip 层总选中应 = sink+mid+swa = 128+512+128 = 768"
                 f"（N5 池截断 mid=512 判例，B1 不扰分区路径），"
                 f"实际 {int(mask2[0, 0][0].sum())}")
        report(name, True, f"skip sink={n_sink}/128 强制、总选中 1152；非 skip 768 不变")
    except RuntimeError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ E121-B2 (F2)
def b2_patch_branch_guards():
    """F2：qwen3/llama3 patch decode 分支入口硬守卫（三走廊外形态 fail-closed）。

    【红】主树：① past=None → AttributeError；② chunked prefill（q_len=2
        带 past）→ 静默进 decode 分支（stub 侧证）；③ 4D float mask →
        静默错算；④ llama3 同构。四例均非预期 RuntimeError → FAIL。
    【绿】修复态：三形态 + llama3 全部显式 RuntimeError；prefill 分支
        （cache 序==q_len）走 dense stub 正常返回（回归位两侧同绿）。
    """
    name = "B2(F2) patch decode 入口守卫（use_cache/chunk/batch/4D mask）"
    try:
        import torch.nn as nn
        QW3_MOD = importlib.import_module("sparse_attn.patches.qwen3_attn_patch")
        LL3_MOD = importlib.import_module("sparse_attn.patches.llama3_attn_patch")

        class _DecodeEntered(Exception):
            pass

        class StubIndexer:
            def __init__(self):
                self.calls = []

            def clear(self):
                self.calls.append("clear")

            def observe_prefill_q(self, q):
                self.calls.append("observe")

            def prepare_mask(self, *a, **kw):
                self.calls.append("decode")
                raise _DecodeEntered("decode branch entered")

        class FakeCache:
            """KV cache 桩：get_seq_length 决定分支（≠q_len → decode）；
            update 返回预置 k/v（真实语义=返回全历史），两 knob 解耦——
            ⑤ 用 kv_len=1 使 2D bool mask 宽度与 q/k 序长一致可走通
            unpad（生产 unpad_tensor 是 x[mask] 布尔索引，要求宽度相等）。"""
            def __init__(self, seqlen, kv_len=None, hkv=2, hd=4):
                self._s = seqlen
                n = kv_len if kv_len is not None else seqlen
                torch.manual_seed(seqlen * 10 + n)
                # HF cache 约定：update 返回 [B, H, S, D]（head-major）
                self.k = torch.randn(1, hkv, n, hd)
                self.v = torch.randn(1, hkv, n, hd)

            def get_seq_length(self, layer_idx=0):
                return self._s

            def update(self, k, v, layer_idx, cache_kwargs=None):
                return self.k, self.v

        def mk_self(hu=4, hkv=2, hd=4, hidden=16):
            return types.SimpleNamespace(
                q_proj=nn.Linear(hidden, hu * hd), q_norm=nn.Identity(),
                k_proj=nn.Linear(hidden, hkv * hd), k_norm=nn.Identity(),
                v_proj=nn.Linear(hidden, hkv * hd),
                o_proj=nn.Linear(hidden, hidden), head_dim=hd,
                config=types.SimpleNamespace(_attn_implementation="eager"),
                scaling=0.5, layer_idx=0, sliding_window=128,
                training=False, attention_dropout=0.0,
                indexer=StubIndexer(),
            )

        def call(fwd, fake_self, q_len, past, attn_mask=None, seq_len=None):
            sl = seq_len or q_len
            hs = torch.randn(1, q_len, 16)
            cos = torch.ones(1, sl, 4)
            sin = torch.zeros(1, sl, 4)
            cache_pos = torch.arange(sl)
            return fwd(fake_self, hs, (cos, sin), attn_mask, past, cache_pos)

        def expect_runtime_error(fn, needle, tag):
            try:
                fn()
            except RuntimeError as e:
                _require(needle in str(e),
                         f"{tag}: RuntimeError 应含 {needle!r}，实际 {str(e)!r}")
            except _DecodeEntered:
                raise RuntimeError(
                    f"{tag}: 静默进了 decode 分支（F2 缺陷路径——守卫缺失）")
            except Exception as e:
                raise RuntimeError(
                    f"{tag}: 预期 RuntimeError（含 {needle!r}），实际 "
                    f"{type(e).__name__}: {e}")
            else:
                raise RuntimeError(f"{tag}: 未抛异常（F2 缺陷：静默通过）")

        # ① use_cache=False（past=None，训练/PPL 形态）
        expect_runtime_error(
            lambda: call(QW3_MOD.qwen3_attn_forward, mk_self(), 2, None),
            "use_cache", "①use_cache=False")
        # ② chunked prefill：past 非空（序 3）+ q_len=2 → 主树静默进 decode
        expect_runtime_error(
            lambda: call(QW3_MOD.qwen3_attn_forward, mk_self(), 2, FakeCache(3)),
            "q_len", "②chunked_prefill")
        # ③ 4D float mask（HF 实际形态）+ decode（序 5 ≠ q_len 1）
        expect_runtime_error(
            lambda: call(QW3_MOD.qwen3_attn_forward, mk_self(), 1, FakeCache(5),
                         attn_mask=torch.zeros(1, 1, 1, 5, dtype=torch.float32)),
            "2D bool", "③4D_float_mask")
        # ④ llama3 同构（use_cache=False）
        expect_runtime_error(
            lambda: call(LL3_MOD.llama3_attn_forward, mk_self(), 2, None),
            "use_cache", "④llama3_use_cache=False")
        # ⑤ 2D bool mask（合法形态）decode 不被守卫拦截：mask 宽度须与
        # q/k 序长一致（unpad_tensor 是 x[mask] 布尔索引）→ cache 桩
        # kv_len=1、mask [1,1] 全 True，走通 unpad + stub mask + eager_decoding
        fake = mk_self()
        # stub prepare_mask 返回 (mask, block_size)：mask 须 4 维 [1,1,Hkv,tk]
        # （eager_decoding squeeze(0) 后 mask[0] 才是 2 维 [Hkv, tk]）
        def _pm(*a, **kw):
            fake.indexer.calls.append("decode")
            tk = a[2].shape[1]
            return torch.ones(1, 1, 2, tk, dtype=torch.bool), 1
        fake.indexer.prepare_mask = _pm
        call(QW3_MOD.qwen3_attn_forward, fake, 1, FakeCache(5, kv_len=1),
             attn_mask=torch.ones(1, 1, dtype=torch.bool))
        _require("decode" in fake.indexer.calls,
                 "⑤2D bool mask 合法 decode 路径被守卫误拦（过拦——回归）")
        # ⑥ prefill 分支回归：cache 序 == q_len → dense stub 正常返回
        fake = mk_self()
        _orig_eager = QW3_MOD.eager_attention_forward
        QW3_MOD.eager_attention_forward = (
            lambda self, q, k, v, am, dropout=0.0, scaling=1.0, **kw:
            (torch.zeros(1, 2, 16), None))
        try:
            out, _ = call(QW3_MOD.qwen3_attn_forward, fake, 2, FakeCache(2))
            _require(tuple(out.shape) == (1, 2, 16),
                     f"⑥prefill 输出形状应 (1,2,16)，实际 {tuple(out.shape)}")
            _require(fake.indexer.calls == ["clear", "observe"],
                     f"⑥prefill 应 clear+observe（sparse-prefill env 关），"
                     f"实际 {fake.indexer.calls}")
        finally:
            QW3_MOD.eager_attention_forward = _orig_eager
        report(name, True, "四守卫全 raise；2D bool/prefill 回归位不误拦")
    except RuntimeError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ E121-B3 (F3)
def b3_register_patch_count():
    """F3：register_patch 计数返回 + method≠none 零匹配显式拒绝。

    【红】主树：① GLM 系模型零匹配 → 静默返回（dense 被标稀疏）；
        ② 返回值 None（无计数 API）→ 两例 FAIL。
    【绿】修复态：GLM 零匹配 RuntimeError（含 GLM 提示）；Qwen3 匹配
        返回计数 1；method=none/未知方法回归不变。
    """
    name = "B3(F3) register_patch 计数 + 零匹配显式拒绝（GLM 不再静默 dense）"
    try:
        import torch.nn as nn
        PATCH_MOD = importlib.import_module("sparse_attn.patches.patch")

        class Holder:
            """named_modules 鸭子类型容器（绕开 nn.Module 注册机制）。"""
            def __init__(self, mods):
                self._mods = list(mods)

            def named_modules(self):
                yield from self._mods

        class GlmFake(nn.Module):
            pass

        from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

        class FakeQ3(Qwen3Attention):
            pass

        # ① GLM 系零匹配 → 显式拒绝（红：主树静默返回 None）
        glm_model = Holder([("glm_attn", GlmFake())])
        args_tli = make_args()   # method 需显式给（make_args 默认无 method）
        args_tli.method = "tli"
        raised = False
        try:
            PATCH_MOD.register_patch(glm_model, args_tli)
        except RuntimeError as e:
            raised = True
            _require("patched=0" in str(e) and "GLM" in str(e),
                     f"①GLM 拒绝信息应含 patched=0 与 GLM 提示，实际 {str(e)!r}")
        _require(raised, "①GLM 零匹配应 RuntimeError（主树缺陷态：静默返回，"
                         "dense 结果被标稀疏方法——F3 复现）")
        # ② 计数 API：Qwen3 匹配 1 个 → 返回 1（红：主树返回 None）
        q3 = FakeQ3.__new__(FakeQ3)
        q3_model = Holder([("attn0", q3)])
        n = PATCH_MOD.register_patch(q3_model, args_tli)
        _require(n == 1, f"②Qwen3 匹配应返回计数 1，实际 {n!r}")
        _require(getattr(q3, "indexer", None) is not None,
                 "②Qwen3 模块应被挂 indexer")
        # ③ method=none → 不 patch、返回 0、不 raise（回归位）
        n = PATCH_MOD.register_patch(q3_model, make_args(method="none"))
        _require(n == 0, f"③method=none 应返回 0，实际 {n!r}")
        # ④ 未知 method → ValueError（回归位）
        raised = False
        try:
            PATCH_MOD.register_patch(q3_model, make_args(method="nope"))
        except ValueError:
            raised = True
        _require(raised, "④未知 method 应 ValueError")
        report(name, True, "GLM 零匹配拒 + 计数 API + none/未知回归全过")
    except RuntimeError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ E121-B4 (F4)
def b4_sigma_dead_param_raise():
    """F4：sigma_select≠none 的死参数组合 fail-closed。

    【红】主树：① 无分区（α=β=0 且 kmeans 关）；② skip_far 跳层；③ moba
        组合——三例 σ 全部静默退化（跑的不是声明的方法）→ FAIL。
    【绿】修复态：三例显式 ValueError；合法 σ 臂（e64 分区开、非 moba）
        照常出 mask（回归位两侧同绿）。
    """
    name = "B4(F4) σ 死参数 fail-closed（无分区/skip_far/moba 三组合）"
    try:
        S = 4352
        k, q = gen_kq(S)
        # 注：tli_moba 不入 base（③单独传 True，避免 make_args 重复 kwarg；
        # 其余臂缺省时 indexer 走 getattr(args, "tli_moba", False)=False）
        base = dict(tli_far_select="4bit", tli_near_select="4bit",
                    tli_enable_kmeans=False, tli_enable_layer_skip=False,
                    tli_far_method="minmax", tli_near_method="avg",
                    tli_sigma=8.0, tli_per_q_head=False,
                    tli_static_pair=False, tli_proj_basis=None)

        def expect_sigma_value_error(args, tag, skip_far=False):
            idx = IDX_PKG.TLIIndexer(args)
            idx.layer_idx = 3
            if skip_far:
                idx.skip_far = True
            cu = torch.tensor([0, S])
            q_ids = torch.tensor([S - 1])
            try:
                idx.prepare_mask(q, q_ids, k, cu, q.shape[-1] ** -0.5)
            except ValueError as e:
                _require("sigma" in str(e).lower(),
                         f"{tag}: ValueError 应提 sigma，实际 {str(e)!r}")
            except Exception as e:
                raise RuntimeError(
                    f"{tag}: 预期 ValueError（σ 死参数），实际 "
                    f"{type(e).__name__}: {e}")
            else:
                raise RuntimeError(
                    f"{tag}: 未抛异常——σ 静默退化（F4 缺陷复现：实验跑的"
                    f"不是声明的方法）")

        # ① 无分区：σ × (α=β=0, kmeans 关) → use_partition=False
        expect_sigma_value_error(
            make_args(**base, tli_alpha=0.0, tli_beta=0.0, tli_gamma=1.0,
                      tli_sigma_select="far"),
            "①σ×无分区")
        # ② skip_far：σ × e64 分区 × 跳层 → use_partition=False（D' 命中）
        expect_sigma_value_error(
            make_args(**base, tli_alpha=0.5, tli_beta=0.25, tli_gamma=0.5,
                      tli_sigma_select="far"),
            "②σ×skip_far", skip_far=True)
        # ③ moba × σ：moba 提前 return → σ 死参数
        expect_sigma_value_error(
            make_args(**base, tli_alpha=0.5, tli_beta=0.25, tli_gamma=0.5,
                      tli_sigma_select="far", tli_moba=True),
            "③σ×moba")
        # ④ 合法 σ 臂回归位：e64 分区开、非 moba、非 skip → 正常出 mask
        idx = IDX_PKG.TLIIndexer(make_args(
            **base, tli_alpha=0.5, tli_beta=0.25, tli_gamma=0.5,
            tli_sigma_select="far"))
        idx.layer_idx = 3
        cu = torch.tensor([0, S])
        q_ids = torch.tensor([S - 1])
        mask, _ = idx.prepare_mask(q, q_ids, k, cu, q.shape[-1] ** -0.5)
        _require(int(mask[0, 0][0].sum()) > 0, "④合法 σ 臂应产出非空 mask")
        report(name, True, "三死参数组合全 raise；合法 σ 臂回归出 mask")
    except RuntimeError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ E121-B5 (F7)
def b5_clear_resets_basis():
    """F7：clear() 重置 _basis（投影基惰性切片跨请求残留）。

    【红】主树：prepare_index 提取 _basis 后 clear() 不重置 → 残留非 None。
    【绿】修复态：clear() 后 _basis=None；下次 prepare_index 从 _basis_all
        重新派生（语义自愈）。
    """
    name = "B5(F7) clear() 重置 _basis（红残留 / 绿 None+重派生）"
    try:
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            bp = os.path.join(td, "basis.pt")
            # _basis_all = [n_layers, Hkv, head_dim=128, r]（prepare_index 经
            # idx_sub（tail-32）切片 → _basis [Hkv, 32, r]；head_dim 维必须
            # 是 128，否则 [:, idx_sub, :] 越界）
            torch.save(torch.randn(2, 2, 128, 4), bp)
            args = make_args(tli_subspace="tail", tli_enable_subspace=True,
                             tli_proj_basis=bp, tli_enable_kmeans=False,
                             tli_enable_layer_skip=False,
                             tli_far_select="4bit", tli_near_select="4bit",
                             tli_far_method="minmax", tli_near_method="avg",
                             tli_alpha=0.0, tli_beta=0.0, tli_gamma=1.0)
            k, _q = gen_kq(128)
            idx = IDX_PKG.TLIIndexer(args)
            idx.layer_idx = 0
            cu = torch.tensor([0, 128])
            idx.prepare_index(k, cu)
            _require(idx._basis is not None,
                     "prepare_index 后 _basis 应已惰性提取（非 None）")
            _require(tuple(idx._basis.shape) == (2, 32, 4),
                     f"_basis 形状应 [Hkv,32,r]=(2,32,4)，实际 {tuple(idx._basis.shape)}")
            idx.clear()
            _require(idx._basis is None,
                     f"clear() 后 _basis 应 None（主树缺陷态：跨请求残留 "
                     f"{type(idx._basis)}——F7 复现）")
            # 重派生自愈：clear 后再次 prepare_index 应重新提取
            idx.prepare_index(k, cu)
            _require(idx._basis is not None and tuple(idx._basis.shape) == (2, 32, 4),
                     "clear() 后再次 prepare_index 应从 _basis_all 重新派生 _basis")
        report(name, True, "clear 后 None；重 prepare_index 自愈重派生")
    except RuntimeError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ E121-B6 (F10)
def b6_k_delta_range():
    """F10：add_k_delta 极差 = k_max − k_min（原 (k_max+k_min).abs() 无几何意义）。

    【红】主树：块 max=1、min=−2 → delta=|1+(−2)|=1（错误）。
    【绿】修复态：极差 3。死代码防御修复（无生产者）。
    """
    name = "B6(F10) add_k_delta 极差减法（红 |max+min|=1 / 绿 max−min=3）"
    try:
        METRICS_MOD = importlib.import_module("sparse_attn.metrics")
        m = METRICS_MOD.Metrics()
        # 单块 64 token：max=1.0, min=−2.0 → 极差 3.0
        k = torch.full((1, 64, 1, 1), -2.0)
        k[0, 0, 0, 0] = 1.0
        m.add_k_delta(k, 64)
        got = float(m.get_k_delta_max())
        _require(abs(got - 3.0) < 1e-6,
                 f"极差应 3.0 = max−min（主树缺陷态 |max+min|=1.0），实际 {got}")
        report(name, True, f"k_delta={got}（极差口径）")
    except RuntimeError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ E121-B7 (F12)
def b7_all_false_row_guard():
    """F12：eager_decoding_attn 全 False 行防线（消费端 fail-closed）。

    【红】主树：全 False mask → softmax NaN 静默返回。
    【绿】修复态：显式 RuntimeError。
    """
    name = "B7(F12) eager_decoding 全 False 行防线（红 NaN 静默 / 绿 raise）"
    try:
        EAGER_MOD = importlib.import_module("sparse_attn.ops.eager_decoding")
        torch.manual_seed(7)
        # mask 契约：4 维 [1, x, Hkv, tk]（eager_decoding squeeze(0) 后
        # mask[0] 才是 2 维 [Hkv, tk]，与生产 prepare_mask 输出同构）
        q = torch.randn(1, 1, 2, 8)
        k = torch.randn(1, 4, 2, 8)
        v = torch.randn(1, 4, 2, 8)
        mask = torch.zeros(1, 1, 2, 4, dtype=torch.bool)   # 全 False
        raised = False
        try:
            EAGER_MOD.eager_decoding_attn(
                q, k, v, mask, 1, torch.tensor([0, 4]), softmax_scale=0.5)
        except RuntimeError as e:
            raised = True
            _require("全 False" in str(e),
                     f"RuntimeError 应含「全 False」，实际 {str(e)!r}")
        _require(raised,
                 "全 False 行应 RuntimeError（主树缺陷态：NaN 静默返回——"
                 "F12 复现）")
        # 回归位：合法 mask（含 True）正常返回有限值
        mask_ok = torch.zeros(1, 1, 2, 4, dtype=torch.bool)
        mask_ok[0, 0, 0, 2] = True
        mask_ok[0, 0, 1, 3] = True
        o = EAGER_MOD.eager_decoding_attn(
            q, k, v, mask_ok, 1, torch.tensor([0, 4]), softmax_scale=0.5)
        _require(bool(torch.isfinite(o).all()),
                 "合法 mask 输出应有限（守卫不应误拦正常路径）")
        report(name, True, "全 False raise；合法 mask 回归有限输出")
    except RuntimeError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ E121-C1 (S6)
def c1_pca_rank_validation():
    """S6：backend PCA basis 实际秩 vs profile.proj_rank 校验。

    pool 槽宽按 refine_nd()=proj_rank 预分配，indexer nd2 按 basis.shape[-1]
    ——r 不符时静默错位。【红】主树不校验（构造成功）；【绿】ValueError。
    r 匹配时正常构造（回归位两侧同绿）。
    """
    name = "C1(S6) PCA basis 秩 vs proj_rank 校验（红不校验 / 绿 ValueError）"
    try:
        import tempfile
        from sglang.srt.layers.attention.tli.backend import TLISparseAttnBackend

        def fake_runner():
            return types.SimpleNamespace(
                device="cpu",
                model_config=types.SimpleNamespace(
                    head_dim=128, num_key_value_heads=2, num_hidden_layers=2),
                token_to_kv_pool=object(),
                req_to_token_pool=types.SimpleNamespace(req_to_token=object()),
            )

        env_bak = os.environ.get("SGLANG_TLI_PROJ_BASIS")
        r_bak = os.environ.get("SGLANG_TLI_PROJ_R")
        try:
            with tempfile.TemporaryDirectory() as td:
                # r=8 ≠ proj_rank=16（默认）
                bp = os.path.join(td, "basis.pt")
                torch.save(torch.randn(2, 2, 128, 8), bp)
                os.environ["SGLANG_TLI_PROJ_BASIS"] = bp
                raised = False
                try:
                    TLISparseAttnBackend(runner=fake_runner())
                except ValueError as e:
                    raised = True
                    _require("proj_rank" in str(e),
                             f"ValueError 应提 proj_rank，实际 {str(e)!r}")
                except Exception as e:
                    raise RuntimeError(
                        f"秩失配应 ValueError，实际 {type(e).__name__}: {e}")
                _require(raised,
                         "r=8≠proj_rank=16 应 ValueError（主树缺陷态：不校验"
                         "静默错位——S6 复现）")
                # r 匹配回归位：正常构造（两侧同绿）
                bp2 = os.path.join(td, "basis16.pt")
                torch.save(torch.randn(2, 2, 128, 16), bp2)
                os.environ["SGLANG_TLI_PROJ_BASIS"] = bp2
                b = TLISparseAttnBackend(runner=fake_runner())
                _require(b.proj_basis is not None and
                         int(b.proj_basis.shape[-1]) == 16,
                         "r 匹配时应正常加载 basis")
        finally:
            if env_bak is None:
                os.environ.pop("SGLANG_TLI_PROJ_BASIS", None)
            else:
                os.environ["SGLANG_TLI_PROJ_BASIS"] = env_bak
            if r_bak is None:
                os.environ.pop("SGLANG_TLI_PROJ_R", None)
            else:
                os.environ["SGLANG_TLI_PROJ_R"] = r_bak
        report(name, True, "秩失配 ValueError；r 匹配正常构造")
    except RuntimeError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ E121-C2 (S7)
def c2_empty_row_pad_sentinel():
    """S7：SG indexer empty 行兜底 pad 哨兵化（静态契约测试）。

    运行时不可达已证（empty 行须 t_c ≥ w，而 w ≤ token_budget ≤
    min(budget,S) = i_g 宽 → 只可能走截断分支，pad 分支矛盾不可达），
    故以源码契约断言：pad 调用必须带 value=SENT（哨兵=S；原 0 填充是
    B01 前旧语义——token 0 会被下游当有效位重复计权）。
    【红】主树 pad 无 value → 0 填充；【绿】修复态 value=SENT。
    """
    name = "C2(S7) empty 行 pad 哨兵化（静态契约：红 0 填充 / 绿 value=SENT）"
    try:
        import re as _re
        src_path = os.path.join(
            SG_ROOT, "sglang", "srt", "layers", "attention", "tli", "indexer.py")
        with open(src_path, encoding="utf-8") as f:
            src = f.read()
        # 定位 empty 行兜底段（res_k[empty] = i_g 前的 pad 调用）
        _require("res_k[empty] = i_g" in src, "未找到 empty 行兜底 scatter（锚点漂移）")
        pat = _re.compile(
            r"pad\(\s*i_g,\s*\(0,\s*w\s*-\s*i_g\.shape\[-1\]\)\s*,\s*value=SENT\s*\)")
        _require(bool(pat.search(src)),
                 "empty 行 pad 应带 value=SENT（主树缺陷态：默认 0 填充——"
                 "S7 复现）；截断分支 i_g[..., :w] 应保留")
        _require("i_g = i_g[..., :w]" in src, "截断分支应保留（口径未漂移）")
        report(name, True, "pad value=SENT 契约在位（运行时不可达，纯防御）")
    except RuntimeError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ E121-C3 (S8)
def c3_config_m10_comment():
    """S8：config.py M10 注释更正（「输出哨兵转 0」为 B01 前过期口径）。

    B01 后 M10 prefill kernel 输出哨兵=S，下游 _sparse_extend_one 以
    valid = sel < S 屏蔽。【红】主树注释仍写「哨兵转 0」；【绿】更正。
    """
    name = "C3(S8) config.py M10 过期注释更正（红「哨兵转 0」/ 绿哨兵=S）"
    try:
        cfg_path = os.path.join(
            SG_ROOT, "sglang", "srt", "layers", "attention", "tli", "config.py")
        with open(cfg_path, encoding="utf-8") as f:
            src = f.read()
        _require("输出哨兵转 0" not in src,
                 "config.py 仍含过期描述「输出哨兵转 0」（B01 前口径——S8 复现）")
        _require("哨兵=S" in src and "valid = sel < S" in src,
                 "M10 注释应描述 B01 后哨兵语义（哨兵=S + valid = sel < S）")
        report(name, True, "M10 注释已对齐 B01 哨兵语义")
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
    b1_skip_layer_sink_force()
    b2_patch_branch_guards()
    b3_register_patch_count()
    b4_sigma_dead_param_raise()
    b5_clear_resets_basis()
    b6_k_delta_range()
    b7_all_false_row_guard()
    c1_pca_rank_validation()
    c2_empty_row_pad_sentinel()
    c3_config_m10_comment()
    n_pass = sum(1 for _, s, _ in RESULTS if s == "PASS")
    n_fail = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    _require(n_pass + n_fail == len(RESULTS) == 17, "门禁不允许漏项")
    print(f"\n===== E121 kimi3 A+B+C 组验收：{n_pass}/{n_pass + n_fail} PASS"
          f"（-O 安全：全部显式检查非 assert）=====")
    sys.exit(1 if n_fail else 0)
