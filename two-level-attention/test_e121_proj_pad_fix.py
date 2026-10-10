#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""075 投影路径 padded-S 修复门禁（CPU 主跑 + GPU 冒烟，python / python -O 双跑）。

背景（GPT 审计 TL-E121-PROJ-PAD-S-075，P1，2026-10-10_2131 复审 §075）：
  投影路径（--tli_proj_basis）k_qat 从 pad 后 pad_k 派生 → score_fine 宽度
  = Tpad（块对齐 padding 后长度）而非真实 S。compute_mask 的 F5「真实 S =
  score_fine 宽度」在投影路径实取 Tpad → mid/near/far 边界右移、SWA 强制
  区落 padded 尾部。消费端 eager_decoding 裁回真实长度后：
  真实序列尾部只被保护 swa-(Tpad-S) 个 token（S=6145/bs=64 实测 65/128）。

修复（主会话拍板，与 GPT 建议一致）：
  ① prepare_index 在 index_dict 显式携带 valid_length（= pad 前 k 真实
     token 数），compute_score 透传进 score_dict；
  ② compute_mask 内 padding 段 [S_real, Tpad) 在任何选择之前永久置 -inf；
  ③ SWA 强制区 [S_real-swa, S_real) + near/far 边界全按真实 S；
  ④ 无 valid_length 的外部合成 score_dict → 回退宽度口径（兼容既有 harness）。

GPT 纯公式复现基准（S=6145, α=0.125, bs=64, sink=128, swa=128）：
  expected (mid, near_len, far_hi, 真实 SWA 强制位数) = (5889, 736, 5248, 128)
  actual   （修复前）                        = (5952, 744, 5312, 65)

测试矩阵：
  T1  GPT S=6145 精确复现（投影 on / shared）：几何逐项断言 + padding 永久
      -inf + valid_length 透传链
  T2  投影 on/off × shared/per_q_head × S%64 ∈ {0,1,32,63}
      （S ∈ {6144, 6145, 6176, 6207}）全矩阵：
      a) padding 永不入选（含强制区在内的整张 mask [S:] 段全 False）
      b) 真实 SWA 全保（mask[..., S-swa:S] 全 True）
      c) near/far 边界按真实 S（far 区全覆盖 + near 区预算清零 → 边界可观察）
  T3  非投影路径与修复前逐位一致（E121_BASE_ROOT 指向基线 checkout 时启用的
      mask 哈希比对；α/β/γ 冠军配置、(0,0) 单池、per_q_head、σ 分支 × 对齐/
      非对齐 S）。同时记录基线投影路径的实测红值（应为 5312/65——审计复现）。
  T4  外部合成 score_dict（无 valid_length）回退宽度口径——既有 harness 兼容
  T5  GPU 冒烟（cuda 可用时 T1 几何在 GPU 重放）

红/绿用法：
  绿：python3 test_e121_proj_pad_fix.py            （修复态全 PASS）
  红：把本文件拷到基线 commit 的 two-level-attention/ 下运行 → T1/T2 FAIL
      （观测值 5312/65 即审计 actual 列）。
  逐位对照：E121_BASE_ROOT=/path/to/base/two-level-attention python3 test_...py
"""
import hashlib
import importlib
import os
import sys
import tempfile
import types

sys.dont_write_bytecode = True

import torch

torch.set_num_threads(1)

REPO = os.path.dirname(os.path.abspath(__file__))

RESULTS = []   # (name, status, detail)  status ∈ {"PASS", "FAIL"}


def report(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    RESULTS.append((name, status, detail))
    print(f"{status}  {name}" + (f"  -- {detail}" if detail else ""))


def load_sparse_attn(name, root):
    mod = types.ModuleType(name)
    mod.__path__ = [os.path.join(root, "sparse_attn")]
    sys.modules[name] = mod
    return importlib.import_module(f"{name}.indexer")


FIX = load_sparse_attn("sparse_attn_075", REPO)
BASE_ROOT = os.environ.get("E121_BASE_ROOT", "")
BASE = None
if BASE_ROOT and os.path.isdir(os.path.join(BASE_ROOT, "sparse_attn")):
    BASE = load_sparse_attn("sparse_attn_075_base", BASE_ROOT)
else:
    print(f"[075] E121_BASE_ROOT 未提供或不存在，跳过 T3 逐位对照（独立红跑见任务报告）")

BS = 64
SINK = 128
SWA = 128
Hkv, H = 2, 4
NLAYER = 4
RANK = 8


def make_basis_file():
    """合成投影基 [n_layers, Hkv, 128, r]（fp32，weights_only 可加载）。"""
    g = torch.Generator().manual_seed(20261010)
    b = torch.randn(NLAYER, Hkv, 128, RANK, generator=g).float()
    fd, path = tempfile.mkstemp(suffix=".pt", prefix="tli075_basis_")
    os.close(fd)
    torch.save(b, path)
    return path


BASIS_PATH = make_basis_file()


def make_args(**kw):
    a = types.SimpleNamespace(
        tia_block_size=BS,
        tia_level1_topk=128,
        tia_level2_topk=2048,
        tia_level2_cmp_ratio=4,
        tia_enable_async_topk=False,
        # TLI 参数（对齐生产默认 + 本套件消融旋钮）
        tli_enable_subspace=True,
        tli_subspace="tail",
        tli_enable_kmeans=True,
        tli_enable_layer_skip=False,
        tli_far_select="4bit",
        tli_near_select="4bit",
        tli_far_method="minmax",
        tli_near_method="avg",
        tli_alpha=0.0,
        tli_beta=0.0,
        tli_gamma=1.0,
        tli_per_q_head=False,
        tli_static_pair=False,
        tli_moba=False,
        tli_sigma_select="none",
        tli_sigma=8.0,
        tli_sim=0.9,
        tli_sim_dims="subspace",
        tli_proj_basis=None,
    )
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def geom_expect(S, alpha):
    """期望几何（独立手算公式，GPT 审计口径；非生产代码复用）。

    mid = S - sink - swa；near_len = max(bs, α·mid)；near_blks =
    max(sink_blocks, (S - swa - near_len)//bs)；far_hi = near_blks·bs。
    """
    mid = max(0, S - SINK - SWA)
    near_len = max(BS, int(alpha * mid))
    near_blks = max(2, (S - SWA - near_len) // BS)
    far_hi = near_blks * BS
    return mid, near_len, near_blks, far_hi, S - SWA


def gen_input(S, seed=1234):
    """固定 seed 生成 decode 输入（cpu；跨模块副本逐位一致）。"""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(1, 1, H, 128, generator=g)
    k = torch.randn(1, S, Hkv, 128, generator=g)
    return q, k


def chain(mod, args, S, q=None, k=None, device="cpu", keep_score_dict=False):
    """全链驱动 prepare_index → compute_score → compute_mask（生产入口语义）。"""
    if q is None or k is None:
        q, k = gen_input(S)
    q, k = q.to(device), k.to(device)
    idx = mod.TLIIndexer(args)
    idx.layer_idx = 1
    cu = torch.tensor([0, S], device=device)
    index_dict = idx.prepare_index(k, cu)
    sd = idx.compute_score(q, torch.tensor([S - 1], device=device), index_dict,
                           128 ** -0.5)
    mask = idx.compute_mask(torch.tensor([S - 1], device=device), sd)
    if keep_score_dict:
        return idx, sd, mask
    return idx, mask


def observe_geometry(mask, S):
    """从返回 mask 观测实际几何（全 head 聚合后的边界）。"""
    m = mask.squeeze(0).squeeze(0)          # [H 或 Hkv, T]
    allh = m.all(dim=0)                     # [T] 全 head 均 True
    anyh = m.any(dim=0)
    swa_lo = S - SWA
    # far 覆盖上界观测：[sink, swa_lo) 内最后一个全 True 的位置 +1
    far_hi_obs = 0
    for i in range(SINK, swa_lo):
        if allh[i]:
            far_hi_obs = i + 1
    # near 空区观测：[far_hi_obs, swa_lo) 是否存在任何被选 token
    near_gap_clean = not anyh[far_hi_obs:swa_lo].any().item()
    swa_real_forced = int(allh[swa_lo:S].sum().item())
    pad_clean = not anyh[S:].any().item() if m.shape[-1] > S else True
    sink_ok = bool(allh[:SINK].all().item())
    return dict(width=m.shape[-1], far_hi_obs=far_hi_obs,
                near_gap_clean=near_gap_clean, swa_real_forced=swa_real_forced,
                pad_clean=pad_clean, sink_ok=sink_ok)


def mask_hash(t):
    return hashlib.sha256(t.cpu().contiguous().numpy().tobytes()).hexdigest()[:16]


# ================================================================ T1：GPT 精确复现
def t1_gpt_exact():
    S = 6145
    mid_e, near_e, nblk_e, far_hi_e, swa_lo_e = geom_expect(S, 0.125)
    assert (mid_e, near_e, far_hi_e) == (5889, 736, 5248), \
        f"手算公式自身与 GPT 审计基准不符: {(mid_e, near_e, far_hi_e)}"
    K2 = (far_hi_e - SINK) + SINK + SWA          # = far 宽 + sink + swa → far 全覆盖
    args = make_args(tli_proj_basis=BASIS_PATH, tli_alpha=0.125, tli_beta=0.125,
                     tli_gamma=0.0, tia_level2_topk=K2)
    idx, sd, mask = chain(FIX, args, S, keep_score_dict=True)

    ok = sd.get("valid_length") == S
    report("T1a valid_length 透传（prepare_index→compute_score→score_dict）",
           ok, f"valid_length={sd.get('valid_length')} (期望 {S})")

    sf = sd["score_fine"]
    ok = sf.shape[-1] == 6208
    report("T1b 投影路径 score_fine 宽 = Tpad=6208（pad 后）", ok,
           f"width={sf.shape[-1]} (期望 6208)")

    ok = bool((sf[..., S:] == float("-inf")).all().item()) and sf.shape[-1] > S
    report("T1c padding 段永久 -inf（compute_mask 后 score_dict 内实测）", ok,
           f"padding 段 max={sf[..., S:].max().item()} (期望 -inf)")

    g = observe_geometry(mask, S)
    ok = (g["width"] == 6208 and g["far_hi_obs"] == far_hi_e
          and g["near_gap_clean"] and g["swa_real_forced"] == SWA
          and g["pad_clean"] and g["sink_ok"])
    report("T1d GPT S=6145 几何：mid=5889 near=736 far_hi=5248 SWA=128 padding=0",
           ok, f"观测 width={g['width']} far_hi={g['far_hi_obs']} "
                f"near_gap_clean={g['near_gap_clean']} swa_real_forced="
                f"{g['swa_real_forced']}/128 pad_clean={g['pad_clean']} "
                f"sink_ok={g['sink_ok']}（修复前 actual: far_hi=5312 SWA=65）")


# ================================================================ T2：小矩阵
def t2_matrix(device="cpu"):
    S_list = [6144, 6145, 6176, 6207]           # S%64 ∈ {0,1,32,63}
    for proj in (True, False):
        for pqh in (False, True):
            for S in S_list:
                mid_e, near_e, nblk_e, far_hi_e, swa_lo_e = geom_expect(S, 0.125)
                K2 = (far_hi_e - SINK) + SINK + SWA
                kw = dict(tli_alpha=0.125, tli_beta=0.125, tli_gamma=0.0,
                          tia_level2_topk=K2, tli_per_q_head=pqh)
                if proj:
                    kw["tli_proj_basis"] = BASIS_PATH
                args = make_args(**kw)
                tag = f"proj={'on' if proj else 'off'}/pqh={'y' if pqh else 'n'}/S={S}"
                _, mask = chain(FIX, args, S, device=device)
                g = observe_geometry(mask, S)
                Tpad = ((S + BS - 1) // BS) * BS
                width_expect = Tpad if proj else S
                sub = []
                if g["width"] != width_expect:
                    sub.append(f"width={g['width']}≠{width_expect}")
                if g["far_hi_obs"] != far_hi_e:
                    sub.append(f"far_hi={g['far_hi_obs']}≠{far_hi_e}")
                if not g["near_gap_clean"]:
                    sub.append("near_gap_dirty")
                if g["swa_real_forced"] != SWA:
                    sub.append(f"swa={g['swa_real_forced']}≠{SWA}")
                if not g["pad_clean"]:
                    sub.append("padding_selected")
                if not g["sink_ok"]:
                    sub.append("sink_lost")
                report(f"T2[{tag}] a)padding不入选 b)SWA全保 c)边界按真实S",
                       not sub, "; ".join(sub) if sub else "OK")


# ================================================================ T3：非投影逐位对照 + 基线红值
def t3_bitidentity():
    if BASE is None:
        report("T3 非投影路径与基线逐位一致", True,
               "SKIP（E121_BASE_ROOT 未提供；独立红跑见任务报告）")
        return
    cases = [
        ("冠军 mavg 配置 α.125/β.125/γ.625", dict(tli_alpha=0.125,
         tli_beta=0.125, tli_gamma=0.625), [6144, 6145]),
        ("(0,0) 单池默认", dict(tli_alpha=0.0, tli_beta=0.0), [6145]),
        ("per_q_head", dict(tli_alpha=0.125, tli_beta=0.125, tli_gamma=0.625,
         tli_per_q_head=True), [6145]),
        ("σ far 分支", dict(tli_alpha=0.125, tli_beta=0.125, tli_gamma=0.625,
         tli_sigma_select="far"), [6145]),
    ]
    for name, kw, S_list in cases:
        for S in S_list:
            q, k = gen_input(S)
            _, m_fix = chain(FIX, make_args(**kw), S, q=q, k=k)
            _, m_base = chain(BASE, make_args(**kw), S, q=q, k=k)
            h_fix, h_base = mask_hash(m_fix), mask_hash(m_base)
            report(f"T3 非投影逐位一致 [{name}] S={S}",
                   h_fix == h_base, f"fix={h_fix} base={h_base}")
    # 基线投影路径红值观测（审计 actual 列的实测复核，只记录不断言修复态）
    S = 6145
    _, near_e, _, far_hi_e, _ = geom_expect(S, 0.125)
    K2 = (far_hi_e - SINK) + SINK + SWA
    kw = dict(tli_proj_basis=BASIS_PATH, tli_alpha=0.125, tli_beta=0.125,
              tli_gamma=0.0, tia_level2_topk=K2)
    q, k = gen_input(S)
    _, m_base = chain(BASE, make_args(**kw), S, q=q, k=k)
    g = observe_geometry(m_base, S)
    report("T3 基线投影路径红值 = 审计 actual（far_hi=5312, SWA=65）",
           g["far_hi_obs"] == 5312 and g["swa_real_forced"] == 65,
           f"基线实测 far_hi={g['far_hi_obs']} swa_real_forced="
           f"{g['swa_real_forced']}（审计 actual: 5312/65）")


# ================================================================ T4：无 valid_length 回退
def t4_fallback():
    """外部合成 score_dict（无 valid_length）→ 宽度口径，既有 harness 兼容。"""
    S = 1024
    kt = S // BS
    args = make_args(tli_alpha=0.125, tli_beta=0.125, tli_gamma=0.625)
    idx = FIX.TLIIndexer(args)
    idx.layer_idx = 3
    idx.group_size = 2
    fine = torch.randn(1, 1, H, S)
    score_dict = {
        "score_coarse": torch.ones(1, 1, H, kt),
        "score_fine": fine,
        "score_coarse_avg": None,
        "score_moba": None,
    }
    try:
        mask = idx.compute_mask(torch.tensor([S - 1]), score_dict)
        ok = mask.shape[-1] == S and bool(mask[..., :128].all().item()) \
            and bool(mask[..., S - SWA:].all().item())
        report("T4 合成 score_dict 无 valid_length → 回退宽度口径（SWA/SINK 正常）",
               ok, f"width={mask.shape[-1]}")
    except Exception as e:
        report("T4 合成 score_dict 无 valid_length → 回退宽度口径", False,
               f"异常: {type(e).__name__}: {e}")


# ================================================================ T5：GPU 冒烟
def t5_gpu():
    if not torch.cuda.is_available():
        report("T5 GPU 冒烟", True, "SKIP（无 cuda）")
        return
    S = 6145
    _, _, _, far_hi_e, _ = geom_expect(S, 0.125)
    K2 = (far_hi_e - SINK) + SINK + SWA
    args = make_args(tli_proj_basis=BASIS_PATH, tli_alpha=0.125,
                     tli_beta=0.125, tli_gamma=0.0, tia_level2_topk=K2)
    _, mask = chain(FIX, args, S, device="cuda")
    g = observe_geometry(mask, S)
    ok = (g["far_hi_obs"] == far_hi_e and g["near_gap_clean"]
          and g["swa_real_forced"] == SWA and g["pad_clean"])
    report("T5 GPU 冒烟（投影 on, S=6145）", ok,
           f"far_hi={g['far_hi_obs']} swa={g['swa_real_forced']}/128 "
           f"pad_clean={g['pad_clean']}")


def main():
    t1_gpt_exact()
    t2_matrix()
    t3_bitidentity()
    t4_fallback()
    t5_gpu()
    n_fail = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    n_pass = sum(1 for _, s, _ in RESULTS if s == "PASS")
    print(f"\n==== 075 suite: {n_pass} PASS / {n_fail} FAIL ====")
    if os.path.exists(BASIS_PATH):
        os.unlink(BASIS_PATH)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
