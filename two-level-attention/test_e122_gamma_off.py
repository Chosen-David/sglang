#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E122 γ=off 自由竞争门禁（CPU only，合成分数，python / python -O 双跑）。

背景（用户 2026-10-10 指令，cavg 探索臂前置）：
  --tli_gamma 接受 'off'（大小写不敏感）→ None。γ=off 时 L2 分区 topk
  （tli_indexer.compute_mask 的 use_partition 分支）取消 near/far 配额
  分割：全部 mid 候选在统一预算 K2_mid 内单池 topk 竞争。
    * 默认路径（far/near 同用细筛分 p）：p[..., far_tok_lo:swa_lo_tok]
      单池 topk(K2_mid)，索引 + far_tok_lo；
    * cavg/ccluster 路径（far_tok_score 存在）：concat(far 簇分数, near
      细筛分) 单次 topk → 索引回映射（< w_far 走 _km_far_lo 偏移，
      否则 far_tok_hi 偏移）；
    * sink/swa 强制区、L1 β near 块预算、K2_mid = K2 − sink − swa 口径
      全部不变；nt_near/far_budget/k2_far/k2_near 配额逻辑整体旁路。

测试矩阵：
  T1 默认路径自由竞争 vs γ=0.625 配额（红：off 若仍走配额路径，
     near 计数 640 而非 832，断言必失败）
  T2 cavg 路径拼接 topk 索引映射（far 簇分 + near 细筛分，
     选出 token id 落在预期区段；与配额模式行为不同）
  T3 (0,0) cavg 几何：near 区空（swa_lo_tok == far_tok_hi 边界）
  T4 边界：短序列 far-empty（C-1 几何）+ mid≤1 块 near 区倒挂
     （far_tok_hi > swa_lo_tok，swa token 不占竞争预算）
  T5 CLI 三态解析（off / OFF / 0.625）+ info.py method_name 渲染
  T6 全链 prepare_mask 真管线集成（默认 4bit 路径 + cavg 路径）

用法：python3 test_e122_gamma_off.py   （two-level-attention/ 下）
"""
import importlib
import os
import sys
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


IDX = load_sparse_attn("sparse_attn_e122", REPO)
_args_mod = importlib.import_module("sparse_attn_e122.arguments")
_info_mod = importlib.import_module("sparse_attn_e122.info")


def make_args(**kw):
    a = types.SimpleNamespace(
        tia_block_size=64,
        tia_level1_topk=128,
        tia_level2_topk=2048,
        tia_level2_cmp_ratio=4,
        tia_enable_async_topk=False,
    )
    for k, v in kw.items():
        setattr(a, k, v)
    return a


# ---------------------------------------------------------------- 直接驱动 harness
def drive_compute_mask(args, fine_1h, kt, far_inject=None):
    """直接驱动 compute_mask（合成分数，绕过 prepare_index/compute_score）。

    fine_1h: [T] 单 head 细筛分（p = softmax(fine)，排序等价）。
    返回 (indexer, mask[1,1,H,T])。group_size=1、H=Hkv=2（两 head 同分对称）。
    """
    idx = IDX.TLIIndexer(args)
    idx.layer_idx = 3
    idx.group_size = 1
    if far_inject is not None:
        idx._km_centroids = far_inject["centroids"]      # [Hkv, K_c, d']
        idx._km_token_assign = far_inject["assign"]      # [Hkv, Tfar]
        idx._km_far_lo = far_inject["far_lo"]
        idx._km_dims = far_inject["dims"]
        idx._last_q = far_inject["last_q"]               # [H, D]（已缩放语义）
    T = fine_1h.shape[-1]
    fine = fine_1h.view(1, 1, 1, T).expand(1, 1, 2, T).clone()
    score_dict = {
        "score_coarse": torch.ones(1, 1, 2, kt),   # 均匀分 → L1 池全过
        "score_fine": fine,
        "score_coarse_avg": None,
        "score_moba": None,
    }
    mask = idx.compute_mask(torch.tensor([T - 1]), score_dict)
    return idx, mask


def make_far_inject(Tfar, n_c0, d=4, far_lo=128):
    """合成 far 簇机制：2 簇，c0 分 1.0、c1 分 0.0；前 n_c0 个 token 归 c0。

    _last_q = e_0（128 维）→ q·c0 = 1.0、q·c1 = 0.0（两 head 同构）。
    """
    centroids = torch.zeros(2, 2, d)
    centroids[:, 0, 0] = 1.0                       # h∈{0,1} 的簇 0 = e_0
    assign = torch.zeros(2, Tfar, dtype=torch.long)
    assign[:, n_c0:] = 1
    last_q = torch.zeros(2, 128)
    last_q[:, 0] = 1.0
    return {
        "centroids": centroids,
        "assign": assign,
        "far_lo": far_lo,
        "dims": torch.arange(d),
        "last_q": last_q,
    }


# ================================================================ T1 默认路径自由竞争
def t1_default_path_free_vs_quota():
    name = "T1 γ=off 默认路径：mid 单池竞争（near 全选 832 + 特殊 far 50 入选）vs γ=0.625 配额（near=640）"
    # 几何：T=4096, bs=64, sink=128, swa=128 → K2_mid = 2048-256 = 1792
    #   α=0.25 → near_len_dyn=960 → near_blks=49 → far_tok_hi=3136
    #   far 区 [128,3136) 宽 3008；near 区 [3136,3968) 宽 832
    try:
        T, kt, bs = 4096, 64, 64
        sink_tok, swa_lo = 128, 3968
        far_lo, far_hi, near_lo, near_hi = 128, 3136, 3136, 3968
        K2_mid = 2048 - 128 - 128
        fine = torch.zeros(T)
        # far 区：常规分 1.0+梯度；50 个特殊 token 分 4.0+梯度
        far_pos = torch.arange(far_lo, far_hi)
        fine[far_lo:far_hi] = 1.0 + 1e-6 * (far_pos - far_lo)
        special = far_lo + torch.linspace(0, far_hi - far_lo - 1, 50).long()
        fine[special] = 4.0 + 1e-6 * (special - far_lo)
        # near 区：分 5.0+梯度（全区最高 → 自由竞争下 near 全选）
        near_pos = torch.arange(near_lo, near_hi)
        fine[near_lo:near_hi] = 5.0 + 1e-6 * (near_pos - near_lo)

        cfg = dict(tli_enable_layer_skip=False, tli_far_method="minmax",
                   tli_near_method="avg", tli_alpha=0.25, tli_beta=0.125,
                   tli_enable_kmeans=False)
        _, m_off = drive_compute_mask(make_args(**cfg, tli_gamma=None), fine, kt)
        _, m_q = drive_compute_mask(make_args(**cfg, tli_gamma=0.625), fine, kt)
        for tag, m in (("off", m_off), ("quota", m_q)):
            if m.dtype != torch.bool or m.shape[-1] != T:
                raise AssertionError(f"{tag} mask 形状/dtype 异常 {m.dtype} {m.shape}")
            if not bool(m[0, 0, 0, :sink_tok].all()):
                raise AssertionError(f"{tag} sink 强制区缺失")
            if not bool(m[0, 0, 0, swa_lo:].all()):
                raise AssertionError(f"{tag} swa 强制区缺失")
        # 预算精确：两模式 mid 选中数都 == K2_mid
        for tag, m in (("off", m_off), ("quota", m_q)):
            n_mid = int(m[0, 0, 0, sink_tok:swa_lo].sum())
            if n_mid != K2_mid:
                raise AssertionError(f"{tag} mid 预算 {n_mid} != K2_mid={K2_mid}")
        h0_off = m_off[0, 0, 0]
        h0_q = m_q[0, 0, 0]
        # off：near 区全选（832）、特殊 far 全入选、far 计数 = 1792-832 = 960
        n_near_off = int(h0_off[near_lo:near_hi].sum())
        if n_near_off != 832:
            raise AssertionError(f"off near 应全选 832，实际 {n_near_off}"
                                 "（若仍走配额路径会是 640——红项）")
        if not bool(h0_off[special].all()):
            raise AssertionError("off 特殊 far token 未全部入选（高分应跨区竞争）")
        n_far_off = int(h0_off[far_lo:far_hi].sum())
        if n_far_off != K2_mid - 832:
            raise AssertionError(f"off far 计数 {n_far_off} != {K2_mid - 832}")
        # 配额 γ=0.625：nt_near = 16·64·0.625 = 640 → near=640、far=1152
        n_near_q = int(h0_q[near_lo:near_hi].sum())
        if n_near_q != 640:
            raise AssertionError(f"quota near 应 640，实际 {n_near_q}")
        n_far_q = int(h0_q[far_lo:far_hi].sum())
        if n_far_q != K2_mid - 640:
            raise AssertionError(f"quota far 应 {K2_mid - 640}，实际 {n_far_q}")
        if n_near_off == n_near_q:
            raise AssertionError("off 与 quota 行为应不同（红：off 走了配额路径）")
        report(name, True, f"off: near={n_near_off} far={n_far_off}（特殊 far 50 全选、"
                           f"mid={K2_mid}）；quota: near={n_near_q} far={n_far_q}")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ T2 cavg 路径索引映射
def t2_cavg_concat_mapping():
    name = "T2 γ=off cavg 路径：concat(far 簇分, near 细筛分) 单次 topk 索引回映射"
    # 同 T1 几何；far 簇机制：前 1504 个 far token 簇 0（分 1.0），其余簇 1（0.0）
    #   near p（softmax 概率 > 0）> 簇 1 分 0.0 → off 选中 = 1504 far(c0) + 288 near
    try:
        T, kt = 4096, 64
        sink_tok, swa_lo = 128, 3968
        far_lo, far_hi, near_lo, near_hi = 128, 3136, 3136, 3968
        K2_mid = 1792
        Tfar = far_hi - far_lo          # 3008（注入簇分数宽度 = far 区宽）
        inj = make_far_inject(Tfar, n_c0=1504)
        fine = torch.zeros(T)
        # near 分 5.0+梯度：off 下被选中的 near = p 最高的 288 个（梯度最大端）
        near_pos = torch.arange(near_lo, near_hi)
        fine[near_lo:near_hi] = 5.0 + 1e-6 * (near_pos - near_lo)
        fine[far_lo:far_hi] = 1.0 + 1e-6 * torch.arange(far_hi - far_lo)

        cfg = dict(tli_enable_layer_skip=False, tli_far_method="minmax",
                   tli_near_method="avg", tli_alpha=0.25, tli_beta=0.125,
                   tli_far_select="cluster", tli_near_select="4bit",
                   tli_enable_kmeans=True)
        idx_off, m_off = drive_compute_mask(
            make_args(**cfg, tli_gamma=None), fine, kt, far_inject=inj)
        # 簇分数实际值核验（1.0/0.0 两档）
        fs = idx_off._far_token_score(idx_off._last_q)
        if fs is None or fs.shape != (2, Tfar):
            raise AssertionError(f"far_tok_score 形状异常 {None if fs is None else fs.shape}")
        if float(fs[0, :1504].min()) < 0.99 or float(fs[0, 1504:].max()) > 0.01:
            raise AssertionError("far_tok_score 档位构造失败（应 1.0/0.0）")
        h0 = m_off[0, 0, 0]
        if not bool(h0[:sink_tok].all()) or not bool(h0[swa_lo:].all()):
            raise AssertionError("sink/swa 强制区缺失")
        # far 段映射：选中的 far token 全部为簇 0（相对偏移 < 1504）
        far_sel = torch.nonzero(h0[far_lo:far_hi]).squeeze(-1)
        if int(far_sel.numel()) != 1504:
            raise AssertionError(f"off far 选中 {int(far_sel.numel())} != 1504（簇 0 全体）")
        if int(far_sel.max()) >= 1504:
            raise AssertionError(f"索引回映射错误：选中簇 1 far token（max 相对偏移 {int(far_sel.max())}）")
        # near 段映射：288 个 = p 最高的 288 个（梯度最大端 3967 往下数 288 个）
        near_sel = torch.nonzero(h0[near_lo:near_hi]).squeeze(-1)
        exp_near = torch.arange(near_hi - 288, near_hi) - near_lo
        if int(near_sel.numel()) != 288:
            raise AssertionError(f"off near 选中 {int(near_sel.numel())} != 288")
        if not bool(torch.equal(near_sel.sort().values, exp_near)):
            raise AssertionError(f"near 段索引映射错误：{near_sel.tolist()[:5]}...")
        n_mid = int(h0[sink_tok:swa_lo].sum())
        if n_mid != K2_mid:
            raise AssertionError(f"off mid 总预算 {n_mid} != {K2_mid}")
        # 配额对比（红：off 若走配额路径 near=640 far=1152，与 288/1504 必不同）
        _, m_q = drive_compute_mask(
            make_args(**cfg, tli_gamma=0.625), fine, kt, far_inject=inj)
        hq = m_q[0, 0, 0]
        n_near_q = int(hq[near_lo:near_hi].sum())
        n_far_q = int(hq[far_lo:far_hi].sum())
        if (n_near_q, n_far_q) == (288, 1504):
            raise AssertionError("off 与 quota 行为应不同（红：off 走了配额路径）")
        report(name, True, f"off: far=1504（全簇0，映射无簇1泄漏） near=288（映射=梯度"
                           f" top 端逐位）；quota: far={n_far_q} near={n_near_q}")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ T3 (0,0) cavg near 区空
def t3_zero_zero_near_empty():
    name = "T3 γ=off (0,0) cavg 几何：near 区空（swa_lo_tok == far_tok_hi 边界）"
    # α=0/β=0 + far_select=cluster：near_len_dyn = swa_tok → near_blks=62
    #   → far_tok_hi = 3968 == swa_lo_tok → near 段空，mid 全由 far 簇分竞争
    try:
        T, kt = 4096, 64
        sink_tok, swa_lo = 128, 3968
        far_lo, far_hi = 128, 3968
        K2_mid = 1792
        Tfar = far_hi - far_lo          # 3840
        inj = make_far_inject(Tfar, n_c0=2000)
        fine = torch.zeros(T)
        fine[sink_tok:swa_lo] = 1.0 + 1e-6 * torch.arange(swa_lo - sink_tok)
        cfg = dict(tli_enable_layer_skip=False, tli_far_method="avg",
                   tli_near_method="avg", tli_alpha=0.0, tli_beta=0.0,
                   tli_far_select="cluster", tli_near_select="4bit",
                   tli_enable_kmeans=True)
        _, m = drive_compute_mask(make_args(**cfg, tli_gamma=None), fine, kt,
                                  far_inject=inj)
        h0 = m[0, 0, 0]
        if not bool(h0[:sink_tok].all()) or not bool(h0[swa_lo:].all()):
            raise AssertionError("sink/swa 强制区缺失")
        n_mid = int(h0[sink_tok:swa_lo].sum())
        if n_mid != K2_mid:
            raise AssertionError(f"mid 预算 {n_mid} != {K2_mid}（near 段空时应全由 far 段出）")
        far_sel = torch.nonzero(h0[far_lo:far_hi]).squeeze(-1)
        if int(far_sel.max()) >= 2000:
            raise AssertionError("选中了簇 1 token（near 段空时 far 段应独占预算且只选簇 0）")
        report(name, True, f"near 区空（far_tok_hi==swa_lo_tok=3968）：mid={n_mid} "
                           f"全 far 段簇 0，无簇 1 泄漏")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ T4 边界
def t4_edges():
    name = "T4 边界：短序列 far-empty（C-1）+ mid≤1 块 near 区倒挂（swa 不占竞争预算）"
    try:
        # 4a: T=192（N6 几何）α=1：far 空（far_tok_hi=128=far_tok_lo）、
        #     swa_lo_tok=64 < far_tok_lo → mid 竞争区空 → 全靠强制区 = 全 True
        S = 192
        fine = torch.zeros(S)
        cfg_a = dict(tli_enable_layer_skip=False, tli_far_method="minmax",
                     tli_near_method="avg", tli_alpha=1.0, tli_beta=0.25,
                     tli_gamma=None, tli_enable_kmeans=False,
                     tia_level2_topk=100)
        _, m_a = drive_compute_mask(make_args(**cfg_a), fine, S // 64)
        if m_a.dtype != torch.bool or m_a.shape[-1] != S:
            raise AssertionError(f"4a mask 异常 {m_a.dtype} {m_a.shape}")
        if not bool(m_a.all()):
            raise AssertionError("4a 短序列 far-empty：强制区应覆盖全序列（全 True）")
        # 4b: T=320（mid 恰 1 块 [128,192)）α=0.25：near_blks=4 → far_tok_hi=256
        #     > swa_lo_tok=192（near 区倒挂）→ far 段上界收到 swa_lo_tok=192，
        #     swa 强制区 token [192,256) 不进竞争不占预算
        S2 = 320
        Tfar = 128                       # 簇覆盖 [128,256)
        inj = make_far_inject(Tfar, n_c0=64)   # 前 64（[128,192)）簇 0 分 1.0
        fine2 = torch.zeros(S2)
        cfg_b = dict(tli_enable_layer_skip=False, tli_far_method="minmax",
                     tli_near_method="avg", tli_alpha=0.25, tli_beta=0.125,
                     tli_far_select="cluster", tli_near_select="4bit",
                     tli_enable_kmeans=True, tli_gamma=None)
        idx_b, m_b = drive_compute_mask(make_args(**cfg_b), fine2, S2 // 64,
                                        far_inject=inj)
        h0 = m_b[0, 0, 0]
        if not bool(h0.all()):
            raise AssertionError("4b mid 1 块全选 + sink/swa 强制应覆盖全序列")
        n_mid = int(h0[128:192].sum())
        if n_mid != 64:
            raise AssertionError(f"4b mid 区 [128,192) 应全选 64（far 段截到 swa_lo_tok），"
                                 f"实际 {n_mid}——swa 越界块可能占了竞争预算")
        # 配额模式同几何不崩（边界兼容，两路径都健康）
        _, m_bq = drive_compute_mask(make_args(**{**cfg_b, "tli_gamma": 0.625}),
                                     fine2, S2 // 64, far_inject=inj)
        if m_bq.shape != m_b.shape:
            raise AssertionError("4b quota mask 形状异常")
        report(name, True, f"4a T=192 全 True（far-empty 不崩）；4b T=320 mid=64 "
                           f"全选、swa 越界块不占预算；quota 同几何不崩")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ T5 CLI + info
def t5_cli_and_info():
    name = "T5 CLI 三态解析（off/OFF/0.625）+ info.py γ=off 渲染"
    try:
        pg = _args_mod._parse_gamma
        if pg("off") is not None:
            raise AssertionError("_parse_gamma('off') 应为 None")
        if pg("OFF") is not None:
            raise AssertionError("_parse_gamma('OFF') 应为 None（大小写不敏感）")
        if pg(" Off ") is not None:
            raise AssertionError("_parse_gamma(' Off ') 应为 None（strip）")
        if pg("0.625") != 0.625:
            raise AssertionError("_parse_gamma('0.625') 应为 0.625")
        # argparse 全链：三态 + 非法值 fail loudly
        import argparse
        parser = argparse.ArgumentParser(prog="tli")
        parser = _args_mod.add_sparse_attn_args(parser)
        a1 = parser.parse_args(["--method", "tli", "--tli_gamma", "off"])
        a2 = parser.parse_args(["--method", "tli", "--tli_gamma", "0.625"])
        if a1.tli_gamma is not None:
            raise AssertionError("argparse off 应解析为 None")
        if a2.tli_gamma != 0.625:
            raise AssertionError(f"argparse 数值应 0.625，实际 {a2.tli_gamma}")
        if a1.tli_gamma == a2.tli_gamma:
            raise AssertionError("off 与 0.625 三态应可区分")
        try:
            parser.parse_args(["--method", "tli", "--tli_gamma", "abc"])
            raise AssertionError("非法 γ 值应 SystemExit（fail loudly）")
        except SystemExit:
            pass
        # info.py（E121+E122 合并口径）：γ=None → g"off" 渲染进名；数值 γ 亦进名
        # （E121 B09：α/β/γ 是 treatment 必须可区分）；缺省属性按 1.0 兜底与
        # TLIIndexer 运行时缺省同源。ab 按 F11 生效值：缺 tli_subspace → "full" → 无 A。
        ns = types.SimpleNamespace(
            method="tli", tia_block_size=64, tia_level1_topk=128,
            tia_level2_topk=1024, tia_level2_cmp_ratio=4,
            tli_enable_subspace=True, tli_enable_kmeans=True,
            tli_enable_layer_skip=True, tli_gamma=None,
        )
        name_off = _info_mod.get_method_name_with_info(ns)
        ns.tli_gamma = 1.0
        name_f = _info_mod.get_method_name_with_info(ns)
        if "goff" not in name_off:
            raise AssertionError(f"γ=off 应渲染 goff 后缀：{name_off!r}")
        if name_off == name_f:
            raise AssertionError("γ=off 与 γ=1.0 文件名应不同（同名互覆 = "
                                 "TL-PREFILL-PROVENANCE-001 同型缺陷）")
        if name_f != "tli_64_128_1024_c4_BDa0_b0_g1":
            raise AssertionError(f"数值 γ 文件名应含生效 α/β/γ（E121 B09 口径）：{name_f!r}")
        del ns.tli_gamma                      # 缺省属性按 1.0 兜底（不加 goff）
        if _info_mod.get_method_name_with_info(ns) != name_f:
            raise AssertionError("缺省 tli_gamma 不应渲染 goff（须与显式 1.0 同名）")
        report(name, True, f"off→None / OFF→None / 0.625→float；非法值 SystemExit；"
                           f"info: {name_f} vs {name_off}")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ T6 全链真管线
def gen_kq(S, seed=31):
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(1, S, 2, 128, generator=g) * 0.5
    q = torch.randn(1, 1, 4, 128, generator=g) * 0.5
    return k, q


def run_mask(args, k, q):
    idx = IDX.TLIIndexer(args)
    idx.layer_idx = 3
    cu = torch.tensor([0, k.shape[1]])
    q_ids = torch.tensor([k.shape[1] - 1])
    mask, _ = idx.prepare_mask(q, q_ids, k, cu, q.shape[-1] ** -0.5)
    return idx, mask


def t6_full_pipeline():
    name = "T6 全链 prepare_mask：默认 4bit 路径（off vs 0.625 行为不同）+ cavg 路径不崩"
    # S=4224：kt=66，sink=128，swa_tok=128，swa_lo_tok=4096，mid=3968；K2=1024 → K2_mid=768
    #   【B10 修复后几何】α=0.25 → mid_len=3968、near_len_dyn=992、
    #   near_base=S−swa=4096 → near_blks=(4096−992)//64=48 → far_tok_hi=3072，
    #   near 区 [3072,4096) 宽 1024、far 区 [128,3072) 宽 2944
    #   （旧口径从 kt*bs 推 → near_blks=50/far_hi=3200，B10 已改 swa 起点基准）
    try:
        S = 4224
        K2_mid = 1024 - 128 - 128
        sink_tok, swa_lo = 128, 4096
        far_hi, near_lo = 3072, 3072   # B10 几何：near_blks=48（旧 3200 为 pre-B10 值）
        k, q = gen_kq(S)
        base = dict(tli_enable_layer_skip=False, tli_far_method="minmax",
                    tli_near_method="avg", tli_alpha=0.25, tli_beta=0.125,
                    tia_level2_topk=1024, tli_enable_kmeans=False)
        _, m_off = run_mask(make_args(**base, tli_gamma=None), k, q)
        _, m_q = run_mask(make_args(**base, tli_gamma=0.625), k, q)
        for tag, m in (("off", m_off), ("quota", m_q)):
            if m.dtype != torch.bool or m.shape[-1] != S:
                raise AssertionError(f"{tag} mask 形状异常 {m.dtype} {m.shape}")
            if not bool(m[..., :sink_tok].all()) or not bool(m[..., swa_lo:].all()):
                raise AssertionError(f"{tag} sink/swa 强制区缺失")
            n_mid = int(m[0, 0, 0, sink_tok:swa_lo].sum())
            if n_mid != K2_mid:
                raise AssertionError(f"{tag} mid 预算 {n_mid} != {K2_mid}")
        # 配额 γ=0.625：nt_near = 16·64·0.625 = 640 → near=640、far=128
        n_near_q = int(m_q[0, 0, 0, near_lo:swa_lo].sum())
        if n_near_q != 640:
            raise AssertionError(f"quota near 应 640，实际 {n_near_q}")
        # off：near 计数由竞争决定（seeded 数据确定性），应 ≠ 配额模式
        n_near_off = int(m_off[0, 0, 0, near_lo:swa_lo].sum())
        if n_near_off == n_near_q:
            raise AssertionError("off 与 quota 的 near 计数相同——off 疑走配额路径")
        if bool(torch.equal(m_off, m_q)):
            raise AssertionError("off 与 quota mask 应不同")
        # cavg 全链（真 kmeans + 簇分数 + 自由竞争）
        cfg_cavg = dict(tli_enable_layer_skip=False, tli_far_method="avg",
                        tli_near_method="avg", tli_alpha=0.25, tli_beta=0.125,
                        tli_far_select="cluster", tli_near_select="4bit",
                        tli_enable_kmeans=True, tli_far_clusters=8,
                        tli_far_niter=3, tia_level2_topk=1024, tli_gamma=None)
        idx_c, m_c = run_mask(make_args(**cfg_cavg), k, q)
        if idx_c._km_centroids is None:
            raise AssertionError("cavg 全链未建簇（Tfar < far_clusters*4？）")
        if m_c.dtype != torch.bool or m_c.shape[-1] != S:
            raise AssertionError(f"cavg mask 形状异常 {m_c.dtype} {m_c.shape}")
        if not bool(m_c[..., :sink_tok].all()) or not bool(m_c[..., swa_lo:].all()):
            raise AssertionError("cavg sink/swa 强制区缺失")
        n_mid_c = int(m_c[0, 0, 0, sink_tok:swa_lo].sum())
        if n_mid_c != K2_mid:
            raise AssertionError(f"cavg off mid 预算 {n_mid_c} != {K2_mid}")
        sel = torch.nonzero(m_c[0, 0, 0, sink_tok:swa_lo]).squeeze(-1) + sink_tok
        if int(sel.min()) < sink_tok or int(sel.max()) >= swa_lo:
            raise AssertionError("cavg off 选中 token 越出 mid 区（索引映射错误）")
        report(name, True, f"4bit: off near={n_near_off} vs quota near={n_near_q}"
                           f"（mask 不同、mid 均 {K2_mid}）；cavg 全链 mid={n_mid_c} "
                           f"全部落在 mid 区")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


if __name__ == "__main__":
    t1_default_path_free_vs_quota()
    t2_cavg_concat_mapping()
    t3_zero_zero_near_empty()
    t4_edges()
    t5_cli_and_info()
    t6_full_pipeline()
    n_pass = sum(1 for _, s, _ in RESULTS if s == "PASS")
    n_fail = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    assert n_pass + n_fail == len(RESULTS) == 6, "门禁不允许 SKIP/漏项"
    print(f"\n===== E122 γ=off 门禁：executed {n_pass}/{n_pass + n_fail} PASS"
          f"（python -O 复跑须同绿）=====")
    sys.exit(1 if n_fail else 0)
