#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E113 单测：Triton 贪心 kernel 与 tli_indexer._greedy_cluster_pass 语义对拍。

参考实现从 two-level-attention 主树**只读 import**（E109 扫描正在用主树，
sys.dont_write_bytecode 防 __pycache__ 落盘，口径同 test_e110_ccluster.py）。

覆盖（全部小规模 GPU——与 E109 扫描共存，显存占用 <100MB）：
  T1  冷启动全量对拍：assign 逐位相等 + k_live 相等 + cnt/sq/簇心 allclose
  T2  多尺寸扫描：(T,H,dd,sim) 网格 × 多 seed，报告逐位一致率（fp 边界注记）
  T3  增量续跑 == 全量重放（E110 T3 的 Triton 版：分三段续跑 vs 一次全量）
  T4  sim=0.99 全新建簇路径（新建分支压力测试）+ 手工向量语义（E110 T1 口径）
  T5  多 head 独立性：同数据复制到两 head → 两 head assign 相同

用法：python3 test_e113_greedy_triton.py   （exp/trace/ 目录下）
"""
import importlib
import os
import sys
import types

sys.dont_write_bytecode = True   # 主树只读 import，不落字节码

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from e113_greedy_triton import greedy_pass_triton, greedy_build_triton  # noqa: E402

MAIN_TREE = "/home/wangyuanshuo02/sglang/two-level-attention"

RESULTS = []


def report(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))


# ---------------------------------------------------------------- 参考实现加载
def load_sparse_attn(name, root):
    """root/sparse_attn 作为独立命名空间包加载（同 test_e110 手法）。"""
    mod = types.ModuleType(name)
    mod.__path__ = [os.path.join(root, "sparse_attn")]
    sys.modules[name] = mod
    return importlib.import_module(f"{name}.indexer")


IDX = load_sparse_attn("sparse_attn_e113ref", MAIN_TREE)
greedy_ref = IDX.TLIIndexer._greedy_cluster_pass_python   # E113b：参考实现固定取 Python 路径（_greedy_cluster_pass 已是调度器）   # 参考实现（静态方法）


# ---------------------------------------------------------------- 工具
def gen_structured(T, H, dd, g, n_base=64, noise=0.08):
    """混合簇结构数据：真实 trace 有簇结构（贪心会归并），纯高斯 32 维
    近正交全单例（E108 实测簇形成依赖真实分布）——单测要覆盖归并路径，
    用 base+噪声构造：同 base 的 token 间 cos ≈ 1/(1+dd·noise²) > 0.9。"""
    base = torch.randn(n_base, H, dd, generator=g, device="cuda")
    idx = torch.randint(0, n_base, (T,), generator=g, device="cuda")
    x = base[idx] + noise * torch.randn(T, H, dd, generator=g, device="cuda")
    return x


def greedy_cold_ref(x, sim):
    T, H, dd = x.shape
    sums = x.new_zeros(H, T, dd)
    cnt = x.new_zeros(H, T)
    sq = x.new_zeros(H, T)
    k_live = torch.zeros(H, dtype=torch.long, device=x.device)
    return greedy_ref(x, sim, sums, cnt, sq, k_live)


def greedy_cold_tri(x, sim):
    T, H, dd = x.shape
    sums = x.new_zeros(H, T, dd)
    cnt = x.new_zeros(H, T)
    sq = x.new_zeros(H, T)
    k_live = torch.zeros(H, dtype=torch.long, device=x.device)
    return greedy_pass_triton(x, sim, sums, cnt, sq, k_live)


def cmp_state(r, t, tag):
    """参考 (sums,cnt,sq,klive,assign) vs Triton 版的一致性检查。"""
    sums_r, cnt_r, sq_r, kl_r, a_r = r
    sums_t, cnt_t, sq_t, kl_t, a_t = t
    assert torch.equal(kl_r, kl_t), f"{tag}: k_live 不一致 {kl_r.tolist()} vs {kl_t.tolist()}"
    n_mism = int((a_r != a_t).sum())
    assert n_mism == 0, f"{tag}: assign 不一致 {n_mism}/{a_r.numel()} 处"
    live = int(kl_r.max().item())
    assert live > 0
    assert torch.allclose(cnt_r[:, :live], cnt_t[:, :live], atol=1e-6), f"{tag}: cnt 不一致"
    assert torch.allclose(sq_r[:, :live], sq_t[:, :live], rtol=1e-5, atol=1e-5), f"{tag}: sq 不一致"
    assert torch.allclose(sums_r[:, :live], sums_t[:, :live], atol=1e-5), f"{tag}: 簇心不一致"
    return live


# ================================================================ T1 冷启动全量对拍
def t1_cold_start():
    name = "T1 冷启动全量对拍（assign 逐位 + 状态 allclose）"
    try:
        g = torch.Generator(device="cuda").manual_seed(20261007)
        for (T, H, dd, sim) in [(512, 4, 32, 0.9), (1024, 8, 32, 0.85), (768, 2, 64, 0.92)]:
            x = gen_structured(T, H, dd, g)
            r = greedy_cold_ref(x, sim)
            t = greedy_cold_tri(x, sim)
            live = cmp_state(r, t, f"T={T},H={H},dd={dd},sim={sim}")
            assert live < T, "随机数据 sim<=0.92 应有归并发生（否则测试无鉴别力）"
        report(name, True, "3 组形状全过（含簇合并路径）")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ T2 多 seed 逐位一致率
def t2_bitwise_rate():
    name = "T2 多 seed 逐位一致率（fp 边界量化）"
    try:
        g = torch.Generator(device="cuda").manual_seed(314159)
        tot, mism, live_tot = 0, 0, 0
        for seed_i in range(8):
            x = gen_structured(2048, 8, 32, g, n_base=128, noise=0.10)
            for sim in (0.80, 0.90, 0.95):
                r = greedy_cold_ref(x, sim)
                t = greedy_cold_tri(x, sim)
                tot += r[4].numel()
                mism += int((r[4] != t[4]).sum())
                assert torch.equal(r[3], t[3]), f"seed{seed_i} sim={sim}: k_live 不一致"
                live = int(r[3].max().item())
                live_tot += live
                assert torch.allclose(r[0][:, :live], t[0][:, :live], atol=1e-4), \
                    f"seed{seed_i} sim={sim}: 簇心漂移超容差"
        rate = mism / tot
        assert rate == 0.0, f"逐位不一致率 {rate:.2e}（{mism}/{tot}）超 0 —— fp 边界翻转"
        report(name, True, f"24 组 (8 seed × 3 sim) assign 逐位 100% 一致 "
                           f"（{tot} token 位，平均簇数 {live_tot/24:.0f}）")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ T3 增量续跑 == 全量重放
def t3_incremental_equals_replay():
    name = "T3 Triton 增量续跑 == 全量重放（分三段续跑 vs 一次全量）"
    try:
        g = torch.Generator(device="cuda").manual_seed(271828)
        T, H, dd, sim = 4096, 8, 32, 0.9
        x = gen_structured(T, H, dd, g, n_base=96, noise=0.09)
        # 增量：三段 [0,1024) [1024,3072) [3072,4096)，容量逐步扩容（走 F.pad 路径）
        sums = x.new_zeros(H, 1024, dd)
        cnt = x.new_zeros(H, 1024)
        sq = x.new_zeros(H, 1024)
        kl = torch.zeros(H, dtype=torch.int64, device="cuda")
        assign = torch.empty(H, T, dtype=torch.int64, device="cuda")
        cuts = [0, 1024, 3072, 4096]
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            sums, cnt, sq, kl, a = greedy_pass_triton(x[lo:hi], sim, sums, cnt, sq, kl)
            assign[:, lo:hi] = a
        # 全量一次
        r = greedy_cold_ref(x, sim)
        assert torch.equal(kl, r[3]), "增量 k_live 与全量重放不一致"
        n = int((assign != r[4]).sum())
        assert n == 0, f"增量 assign 与全量重放不一致 {n} 处"
        live = int(kl.max().item())
        assert torch.allclose(sums[:, :live], r[0][:, :live], atol=1e-5), "增量簇心不一致"
        # 分段构建驱动（greedy_build_triton）同口径
        b = greedy_build_triton(x, sim, chunk=1536)
        assert torch.equal(b[3], r[3]) and int((b[4] != r[4]).sum()) == 0, \
            "greedy_build_triton 分段构建与全量不一致"
        report(name, True, f"三段续跑 + chunk 构建 vs 全量重放：assign 逐位一致，簇数 {live}/{T}")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ T4 新建簇路径 + 手工语义
def t4_new_cluster_path():
    name = "T4 sim=0.99 全新建簇路径 + 手工向量语义（E110 T1 口径）"
    try:
        # 手工向量（E110 T1 同款）：running-mean 语义 + 阈值分支
        # dd pad 到 4（Triton 的 tl.arange 需 2 的幂；零 pad 不改变参考/被测双方语义）
        v = torch.tensor([
            [1.0, 0.0],
            [0.91, 0.4124],
            [0.8, 0.6],
            [0.0, 1.0],
        ], device="cuda").unsqueeze(1)   # [T=4, H=1, dd=2]
        v4 = torch.nn.functional.pad(v, (0, 2)).contiguous()   # [4,1,4]
        r = greedy_cold_ref(v4, 0.9)
        t = greedy_cold_tri(v4.contiguous(), 0.9)
        assert t[4].tolist() == [[0, 0, 0, 1]], \
            f"手工向量 assign 应 [0,0,0,1]（running-mean 归并语义）, 实际 {t[4].tolist()}"
        cmp_state(r, t, "handcrafted")
        # sim=0.99：每 token 一簇（新建分支全走）
        g = torch.Generator(device="cuda").manual_seed(57721)
        x = torch.randn(512, 4, 32, generator=g, device="cuda")
        t99 = greedy_cold_tri(x, 0.99)
        assert t99[3].tolist() == [512] * 4, f"sim=0.99 应全新建 (k_live=512), 实际 {t99[3].tolist()}"
        r99 = greedy_cold_ref(x, 0.99)
        assert torch.equal(t99[4], r99[4]), "sim=0.99 assign 与参考不一致（新建路径）"
        # 但近似重复向量仍归并
        w = torch.tensor([[1.0, 0.0], [1.0, 0.05]], device="cuda").unsqueeze(1)
        w = torch.nn.functional.pad(w, (0, 2)).contiguous()
        tw = greedy_cold_tri(w, 0.99)
        assert tw[4].tolist() == [[0, 0]], "cos≈0.9988 相邻向量 sim=0.99 下应归并"
        report(name, True, "running-mean/阈值/新建三分支语义全过")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ T5 多 head 独立性
def t5_head_independence():
    name = "T5 多 head 独立性（同数据复制 → 同 assign）"
    try:
        g = torch.Generator(device="cuda").manual_seed(99991)
        x1 = torch.randn(1024, 1, 32, generator=g, device="cuda")
        x = x1.expand(1024, 4, 32).contiguous()
        t = greedy_cold_tri(x, 0.9)
        for h in range(1, 4):
            assert torch.equal(t[4][0], t[4][h]), f"head{h} 与 head0 assign 不一致"
        assert t[3].tolist() == [t[3][0].item()] * 4, "各 head k_live 不一致"
        report(name, True, f"4 head 同数据同结果（k_live={t[3][0].item()}）")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("无可用 GPU——本单测必须跑 GPU（Triton kernel）")
        sys.exit(2)
    torch.manual_seed(0)
    t1_cold_start()
    t2_bitwise_rate()
    t3_incremental_equals_replay()
    t4_new_cluster_path()
    t5_head_independence()
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print("\n" + "=" * 60)
    print(f"总计 {len(RESULTS)} 项，通过 {len(RESULTS) - n_fail}，失败 {n_fail}")
    sys.exit(1 if n_fail else 0)
