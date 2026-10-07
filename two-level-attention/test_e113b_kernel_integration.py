#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E113b 单测：Triton 贪心 kernel 接入 tli_indexer e2e 路径后的开关对拍。

覆盖（GPU 小规模合成数据，显存 <300MB，与 E109 v2 扫描共存）：
  T1  调度器对拍：SGLANG_TLI_GREEDY_KERNEL=1（Triton）vs =0（Python 循环），
      T=8192 多 head 结构化数据 × 多 sim —— assign/k_live 逐位相等、
      sums/cnt/sq/簇心 allclose（kernel 与 Python 归约顺序差异的 fp 容差）
  T2  E110 铁律（kernel 路径）：far greedy 增量续跑（S 逐步 +64，far_hi 多次
      前移）== 冷启动全量重放，_km_token_assign/_km_greedy_klive 逐位一致
  T3  全链路 mask 对拍（cavg_sim 与 ccluster_sim）：prefill + 多步 decode
      （S 逐步增长走缓存/增量路径），env 开/关两份 indexer 的 mask 逐位相等
  T4  clear() 跨请求重置（kernel 路径）：clear 后重跑 == 冷启动新实例
  T5  env=0 显式回退：dispatcher 确实走 Python 路径（spy 验证），
      非 2 的幂 dd 防御性回退不炸（cmp_ratio=3 → dd=42 走 Python）

用法：python3 test_e113b_kernel_integration.py   （two-level-attention/ 下，
  需 CUDA；与 E109 扫描共存的小规模口径）
"""
import importlib
import os
import sys
import types

sys.dont_write_bytecode = True   # 主树只读 import（E109 扫描在用），不落字节码

import torch

REPO = os.path.dirname(os.path.abspath(__file__))

RESULTS = []


def report(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))


# ---------------------------------------------------------------- 包加载
def load_sparse_attn(name, root):
    mod = types.ModuleType(name)
    mod.__path__ = [os.path.join(root, "sparse_attn")]
    sys.modules[name] = mod
    return importlib.import_module(f"{name}.indexer")


IDX = load_sparse_attn("sparse_attn_e113b", REPO)
TLI = IDX.TLIIndexer


# ---------------------------------------------------------------- 公共工具
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


def gen_kq(S, Hkv=8, G=4, D=128, seed=0, scale=0.5, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    H = Hkv * G
    k = torch.randn(1, S, Hkv, D, generator=g, device=device) * scale
    q = torch.randn(1, 1, H, D, generator=g, device=device) * scale
    return k, q


def gen_structured(T, H, dd, seed, n_base=None, noise=0.09, device="cuda"):
    """混合簇结构数据（保证归并路径实际发生；纯高斯 32 维近正交全单例）。"""
    g = torch.Generator(device=device).manual_seed(seed)
    n_base = n_base or max(T // 8, 1)
    base = torch.randn(n_base, H, dd, generator=g, device=device)
    idx = torch.randint(0, n_base, (T,), generator=g, device=device)
    return (base[idx] + noise * torch.randn(T, H, dd, generator=g, device=device)).contiguous()


ENV_KEY = "SGLANG_TLI_GREEDY_KERNEL"


class env_switch:
    """临时切换 SGLANG_TLI_GREEDY_KERNEL（调度器每次调用现读 env，可进程内切换）。"""

    def __init__(self, val):
        self.val = val

    def __enter__(self):
        self.old = os.environ.get(ENV_KEY)
        os.environ[ENV_KEY] = self.val
        return self

    def __exit__(self, *a):
        if self.old is None:
            os.environ.pop(ENV_KEY, None)
        else:
            os.environ[ENV_KEY] = self.old


def run_mask(idx, k, q, S):
    """一步 prepare_mask（q_ids=末位置，与 e109a/e110 口径一致）。"""
    cu = torch.tensor([0, S])
    q_ids = torch.tensor([S - 1])
    mask, _ = idx.prepare_mask(q, q_ids, k[:, :S], cu, q.shape[-1] ** -0.5)
    return mask


# ================================================================ T1 调度器开关对拍
def t1_dispatcher_bitwise():
    name = "T1 调度器对拍（kernel=1 vs kernel=0，T=8192 多 head 多 sim）"
    try:
        T, H, dd = 8192, 8, 32
        x = gen_structured(T, H, dd, seed=20261007)
        for sim in (0.80, 0.90, 0.95):
            states = {}
            for val in ("0", "1"):
                sums = x.new_zeros(H, T, dd)
                cnt = x.new_zeros(H, T)
                sq = x.new_zeros(H, T)
                kl = torch.zeros(H, dtype=torch.long, device=x.device)
                with env_switch(val):
                    sums, cnt, sq, kl, assign = TLI._greedy_cluster_pass(
                        x, sim, sums, cnt, sq, kl)
                states[val] = (sums, cnt, sq, kl, assign)
            (s0, c0, q0, kl0, a0), (s1, c1, q1, kl1, a1) = states["0"], states["1"]
            assert torch.equal(kl0, kl1), f"sim={sim}: k_live 不一致"
            n = int((a0 != a1).sum())
            assert n == 0, f"sim={sim}: assign 逐位不一致 {n}/{a0.numel()} 处"
            live = int(kl0.max().item())
            assert 0 < live < T, f"sim={sim}: 簇数 {live} 异常（应发生归并）"
            assert torch.allclose(c0[:, :live], c1[:, :live], atol=1e-6)
            assert torch.allclose(q0[:, :live], q1[:, :live], rtol=1e-5, atol=1e-5)
            assert torch.allclose(s0[:, :live], s1[:, :live], atol=1e-5)
        report(name, True, f"3 sim 全过：assign 逐位一致，平均簇数 "
                           f"{live}/{T}（sim={sim}）")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ T2 E110 铁律（kernel 路径）
def t2_incremental_equals_replay():
    name = "T2 far greedy 增量续跑 == 全量重放（kernel 路径，S 逐步 +64）"
    try:
        args = make_args(
            tli_far_select="sim_greedy", tli_near_select="4bit",
            tli_enable_kmeans=True, tli_enable_layer_skip=False,
            tli_alpha=0.25, tli_beta=0.25, tli_gamma=0.5,
        )
        S1, DS = 6144, 512
        k, _ = gen_kq(S1 + DS, seed=7)
        with env_switch("1"):
            # 增量：S1 → S1+64 → ... → S1+DS（far_hi 多次前移走续跑路径）
            idx = TLI(args)
            idx.layer_idx = 3
            idx.prepare_index(k[:, :S1], torch.tensor([0, S1]))
            for S in range(S1 + 64, S1 + DS + 1, 64):
                idx.prepare_index(k[:, :S], torch.tensor([0, S]))
            # 全量重放（同数据同 far 终态）
            idx2 = TLI(args)
            idx2.layer_idx = 3
            idx2.prepare_index(k[:, :S1 + DS], torch.tensor([0, S1 + DS]))
        assert torch.equal(idx._km_token_assign, idx2._km_token_assign), \
            "增量 assign 与全量重放不一致（E110 铁律破坏）"
        assert torch.equal(idx._km_greedy_klive, idx2._km_greedy_klive), "k_live 不一致"
        live = int(idx._km_greedy_klive.max().item())
        assert torch.allclose(idx._km_centroids[:, :live],
                              idx2._km_centroids[:, :live], atol=1e-6), "簇心不一致"
        assert live < S1, "sim=0.9 应有归并发生"
        report(name, True, f"S {S1}→{S1+DS} 增量 8 次，assign 逐位一致，簇数 {live}/{S1}")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ T3 全链路 mask 对拍
def t3_pipeline_mask_bitwise():
    name = "T3 全链路 mask 对拍（cavg_sim / ccluster_sim，prefill+多步 decode）"
    try:
        S0, DS = 5120, 256
        k, q = gen_kq(S0 + DS, seed=11)
        cfgs = {
            "cavg_sim": dict(tli_far_select="sim_greedy", tli_near_select="4bit"),
            "ccluster_sim(nope)": dict(tli_far_select="sim_greedy",
                                       tli_near_select="sim_greedy",
                                       tli_sim_dims="nope"),
        }
        for tag, extra in cfgs.items():
            args = make_args(
                tli_enable_kmeans=True, tli_enable_layer_skip=False,
                tli_alpha=0.25, tli_beta=0.25, tli_gamma=0.5,
                tli_far_method="minmax", tli_near_method="avg",
                tli_sim=0.9, **extra,
            )
            # env 开/关各一份 indexer，同步走 prefill + 多步 decode，逐步比对 mask
            idxs, masks = {}, {}
            for val in ("0", "1"):
                with env_switch(val):
                    ix = TLI(args)
                    ix.layer_idx = 3
                    for S in [S0] + list(range(S0 + 64, S0 + DS + 1, 64)):
                        m = run_mask(ix, k, q, S)
                        masks.setdefault(val, []).append(m)
                idxs[val] = ix
            for si, (m0, m1) in enumerate(zip(masks["0"], masks["1"])):
                assert m0.shape == m1.shape and m0.dtype == m1.dtype
                n = int((m0 != m1).sum())
                assert n == 0, f"{tag} step{si}: mask 逐位不一致 {n}/{m0.numel()} 处"
            # 终态贪心状态也对拍（assign/簇心）
            a0, a1 = idxs["0"]._km_token_assign, idxs["1"]._km_token_assign
            assert torch.equal(a0, a1), f"{tag}: 终态 token_assign 不一致"
            live = int(idxs["1"]._km_greedy_klive.max().item())
            assert torch.allclose(idxs["0"]._km_centroids[:, :live],
                                  idxs["1"]._km_centroids[:, :live], atol=1e-5), \
                f"{tag}: 终态簇心不一致"
            # kernel 版应真的走过 Triton 路径（加载成功缓存非 False）
            import sparse_attn_e113b.indexer.tli_indexer as tmod  # noqa: F401
            assert tmod._GREEDY_TRITON_FN is not None and tmod._GREEDY_TRITON_FN is not False, \
                "Triton kernel 未加载成功（T3 的 kernel=1 实际走的是 Python）"
        report(name, True, f"2 配置 × 5 步 mask 逐位一致（prefill {S0} + decode {DS}）")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ T4 clear() 跨请求重置
def t4_clear_reset():
    name = "T4 clear() 跨请求重置（kernel 路径，重跑 == 冷启动）"
    try:
        args = make_args(
            tli_far_select="sim_greedy", tli_near_select="sim_greedy",
            tli_enable_kmeans=True, tli_enable_layer_skip=False,
            tli_alpha=0.25, tli_beta=0.25, tli_gamma=0.5, tli_sim=0.9,
        )
        S, DS = 4608, 128
        k, q = gen_kq(S + DS, seed=21)
        with env_switch("1"):
            # 请求 1：跑两个请求长度（第二个请求更短 → far 收缩触发全量重建）
            ix = TLI(args)
            ix.layer_idx = 3
            for Sreq in (S + DS, S):
                ix.clear()
                ix.prepare_index(k[:, :Sreq], torch.tensor([0, Sreq]))
                a_short = ix._km_token_assign
                # 冷启动新实例（同数据）
                ix2 = TLI(args)
                ix2.layer_idx = 3
                ix2.prepare_index(k[:, :Sreq], torch.tensor([0, Sreq]))
                assert torch.equal(a_short, ix2._km_token_assign), \
                    f"S={Sreq}: clear 后重跑 assign != 冷启动（greedy 增量状态残留）"
                assert torch.equal(ix._km_greedy_klive, ix2._km_greedy_klive)
        report(name, True, f"两请求长度 {S+DS}/{S}，clear 后均 == 冷启动")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ T5 回退路径显式验证
def t5_fallback_paths():
    name = "T5 env=0 走 Python + 非 2 幂 dd 防御性回退（不炸）"
    try:
        # env=0：spy 验证 dispatcher 确实调 Python 实现
        calls = {"py": 0, "tri": 0}
        py_orig = TLI._greedy_cluster_pass_python
        tri_orig = TLI._greedy_cluster_pass_triton

        def spy_py(*a, **kw):
            calls["py"] += 1
            return py_orig(*a, **kw)

        def spy_tri(*a, **kw):
            calls["tri"] += 1
            return tri_orig(*a, **kw)

        TLI._greedy_cluster_pass_python = staticmethod(spy_py)
        TLI._greedy_cluster_pass_triton = staticmethod(spy_tri)
        try:
            x = gen_structured(256, 4, 32, seed=5)
            sums = x.new_zeros(4, 256, 32)
            cnt = x.new_zeros(4, 256)
            sq = x.new_zeros(4, 256)
            kl = torch.zeros(4, dtype=torch.long, device=x.device)
            with env_switch("0"):
                TLI._greedy_cluster_pass(x, 0.9, sums, cnt, sq, kl)
            assert calls == {"py": 1, "tri": 0}, f"env=0 应走 Python, 实际 {calls}"
            calls.update(py=0, tri=0)
            with env_switch("1"):
                TLI._greedy_cluster_pass(x, 0.9, sums, cnt, sq, kl)
            assert calls == {"py": 0, "tri": 1}, f"env=1 应走 Triton, 实际 {calls}"
            # 非 2 的幂 dd（42）：防御性回退 Python，不抛异常
            calls.update(py=0, tri=0)
            x42 = gen_structured(200, 2, 42, seed=6)
            s42 = x42.new_zeros(2, 200, 42)
            c42 = x42.new_zeros(2, 200)
            q42 = x42.new_zeros(2, 200)
            kl42 = torch.zeros(2, dtype=torch.long, device=x42.device)
            with env_switch("1"):
                r = TLI._greedy_cluster_pass(x42, 0.9, s42, c42, q42, kl42)
            assert calls == {"py": 1, "tri": 0}, f"dd=42 应回退 Python, 实际 {calls}"
            assert r[4].shape == (2, 200)
        finally:
            TLI._greedy_cluster_pass_python = staticmethod(py_orig)
            TLI._greedy_cluster_pass_triton = staticmethod(tri_orig)
        report(name, True, "env 路由 + dd 非 2 幂回退全过")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("无可用 GPU——本单测必须跑 GPU（Triton kernel）")
        sys.exit(2)
    torch.manual_seed(0)
    t1_dispatcher_bitwise()
    t2_incremental_equals_replay()
    t3_pipeline_mask_bitwise()
    t4_clear_reset()
    t5_fallback_paths()
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print("\n" + "=" * 60)
    print(f"总计 {len(RESULTS)} 项，通过 {len(RESULTS) - n_fail}，失败 {n_fail}")
    sys.exit(1 if n_fail else 0)
