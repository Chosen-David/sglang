#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E110 单测：ccluster (cluster,cluster) 与 cluster_sim_greedy 的 e2e 实现（CPU only）。

覆盖（全部 CPU，绝不碰 GPU——E109 正在占满双卡）：
  T1  贪心聚类正确性：手工构造向量，验证归并/新建/running-mean 簇心语义
  T2  与 e64a 参考实现语义对拍：assign/k_live 逐位相等、簇心 allclose
      （e64a.greedy_cluster_assign 是 mass 侧口径源，E108 probe 同语义）
  T3  far greedy 增量续跑 == 全量重放：decode 期 far_hi 前移后续跑指派，
      与冷启动全量重建逐位一致（贪心时序语义的精确延续）
  T4  同接口跑通：ccluster(kmeans)/cavg_sim/ccluster_sim 三配置
      prepare_index→compute_score→compute_mask 全链路，mask 结构合法
  T5  预算语义：召回数 == mid_token 预算（K2−sink−swa），far/near 分区
      各自预算约束（γ 活化口径：nt_near=nb_near·bs·γ，far=max(0, K2_mid−nt_near)，
      TASK.md L172 严格式——E109a-γ 起 cavg/ccluster 全组 γ 生效且无保底）
  T6  near 簇覆盖 + 细筛回退混合：块对齐尾巴 token 经细筛分回退仍可被召回
  T7  回归保护：默认 flag（far=4bit, near=4bit）与 cavg/mavg 配置下，
      新代码 vs 主树旧代码（E109 在跑的工作树，只读 import）mask 逐位相同，
      含多步 decode 模拟（S 逐步 +1 走缓存/增量路径）
  T8  clear() 跨请求重置：greedy 增量状态残留清零，重跑 == 冷启动
  T9  flag 互斥守卫：per_q_head × cluster/sim_greedy、near_select × sigma/moba

用法：python3 test_e110_ccluster.py   （在 worktree 的 two-level-attention/ 下）
"""
import importlib
import importlib.util
import os
import sys
import types

# 关键：禁止写 __pycache__ —— 主树（/home/wangyuanshuo02/sglang/two-level-attention）
# 正在被 E109 扫描使用（每臂新起 python 即时加载工作树代码），只读 import 不落字节码
sys.dont_write_bytecode = True

import torch

# CPU 单测：贪心逐步循环里都是小张量操作，多线程同步开销反而拖慢（实测 18 线程
# 比 1 线程慢一个量级），且不影响正确性判定——统一限 1 线程
torch.set_num_threads(1)

REPO = os.path.dirname(os.path.abspath(__file__))                     # worktree 的 two-level-attention
MAIN_TREE = "/home/wangyuanshuo02/sglang/two-level-attention"          # 主树（只读！）
E64A_PATH = os.path.join(MAIN_TREE, "exp", "trace", "analyze_e64a_ab_grid.py")

RESULTS = []   # [(name, ok, detail)]


def report(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))


# ---------------------------------------------------------------- 包加载（新旧两份独立命名空间）
def load_sparse_attn(name, root):
    """把 root/sparse_attn 作为独立命名包加载（该目录无 __init__.py，是 namespace pkg；
    手动造 __path__ 使相对导入 from ..metrics 正常解析）。"""
    mod = types.ModuleType(name)
    mod.__path__ = [os.path.join(root, "sparse_attn")]
    sys.modules[name] = mod
    indexer = importlib.import_module(f"{name}.indexer")
    return indexer


IDX_NEW = load_sparse_attn("sparse_attn_e110new", REPO)
IDX_OLD = load_sparse_attn("sparse_attn_e110old", MAIN_TREE)


def load_e64a():
    spec = importlib.util.spec_from_file_location("e64a_ref", E64A_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["e64a_ref"] = mod
    spec.loader.exec_module(mod)
    return mod


E64A = load_e64a()


# ---------------------------------------------------------------- 公共工具
def make_args(**kw):
    """argparse 等价 namespace：TLI 全部 tli_* flag 走 getattr 默认，只需给直接访问项。"""
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


def gen_kq(S, Hkv=2, H=4, D=128, seed=0, scale=0.5):
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(1, S, Hkv, D, generator=g) * scale
    q = torch.randn(1, 1, H, D, generator=g) * scale
    return k, q


def run_mask(idxpkg, args, k, q, indexer=None):
    """完整 prepare_mask 链路（q_ids=末位置，softmax_scale=D^-0.5，与 patch 口径一致）。"""
    idx = IDX_NEW.TLIIndexer(args) if indexer is None else indexer
    idx.layer_idx = 3
    cu = torch.tensor([0, k.shape[1]])
    q_ids = torch.tensor([k.shape[1] - 1])
    mask, _ = idx.prepare_mask(q, q_ids, k, cu, q.shape[-1] ** -0.5)
    return idx, mask


def greedy_cold(x, sim):
    """冷启动跑一趟贪心（测试便捷封装）。x: [T, H, dd] → (cent, assign, k_live, sums, cnt, sq)。"""
    T, H, dd = x.shape
    sums = x.new_zeros(H, T, dd)
    cnt = x.new_zeros(H, T)
    sq = x.new_zeros(H, T)
    k_live = torch.zeros(H, dtype=torch.long)
    sums, cnt, sq, k_live, assign = IDX_NEW.TLIIndexer._greedy_cluster_pass(
        x, sim, sums, cnt, sq, k_live)
    cent = sums / cnt.clamp(min=1).unsqueeze(-1)
    return cent, assign, k_live, sums, cnt, sq


# ================================================================ T1 贪心聚类正确性
def t1_greedy_basics():
    name = "T1 贪心聚类正确性（归并/新建/running-mean 簇心）"
    try:
        sim = 0.9
        # 向量组（单位向量，手算余弦）：
        #   v0=[1,0]；v1=[0.91,0.4124] cos(v0,v1)=0.91>=0.9 → 归并 c0（mean=[0.955,0.2062]）
        #   v2=[0.8,0.6]：cos(v2, v0)=0.8<0.9（若比 v0 不会并），但 cos(v2, running-mean)=0.9085>=0.9
        #   → 归并 c0 —— 证明比较对象是 running mean 而非首成员
        #   v3=[0,1]：cos(v3, mean(3))≈0.35<0.9 → 新建 c1
        v = torch.tensor([
            [1.0, 0.0],
            [0.91, 0.4124],
            [0.8, 0.6],
            [0.0, 1.0],
        ]).unsqueeze(1)   # [T=4, H=1, dd=2]
        cent, assign, k_live, _, cnt, _ = greedy_cold(v, sim)
        assert k_live.tolist() == [2], f"k_live 应为 2（c0 三成员 + c1 一成员）, 实际 {k_live.tolist()}"
        assert assign.tolist() == [[0, 0, 0, 1]], f"assign 应 [0,0,0,1], 实际 {assign.tolist()}"
        assert cnt[0, :2].tolist() == [3.0, 1.0], f"簇计数应 [3,1], 实际 {cnt[0, :2].tolist()}"
        expect_c0 = v[:3, 0, :].mean(0)   # c0 = 前三个向量的算术均值（v3 在 c1）
        assert torch.allclose(cent[0, 0], expect_c0, atol=1e-6), \
            f"c0 簇心应=三向量算术均值 {expect_c0}, 实际 {cent[0, 0]}"
        assert torch.allclose(cent[0, 1], v[3, 0, :], atol=1e-6), "c1 簇心应=v3 本身"
        # v2 若与 v0 单独比较不会归并（0.8<0.9）——running mean 语义已由上面 assign 隐式验证

        # sim=0.99：同一组向量全部新建簇（cos 均低于 0.99）
        cent2, assign2, k_live2, _, _, _ = greedy_cold(v, 0.99)
        assert k_live2.tolist() == [4] and assign2.tolist() == [[0, 1, 2, 3]], \
            f"sim=0.99 应每 token 一簇, 实际 k_live={k_live2.tolist()} assign={assign2.tolist()}"
        # 但 [1,0] 与 [1,0.05]（cos≈0.9988>=0.99）仍应归并
        v4 = torch.tensor([[1.0, 0.0], [1.0, 0.05]]).unsqueeze(1)
        _, assign4, k_live4, _, cnt4, _ = greedy_cold(v4, 0.99)
        assert k_live4.tolist() == [1] and assign4.tolist() == [[0, 0]], \
            "sim=0.99 下 cos≈0.9988 的相邻向量仍应归并"

        # 多 head 独立聚类：H=2 同数据复制 → 两 head 结果一致
        v5 = v.expand(4, 2, 2).contiguous()
        _, assign5, k_live5, _, _, _ = greedy_cold(v5, sim)
        assert assign5[0].tolist() == assign5[1].tolist() and k_live5.tolist() == [2, 2], \
            "两个 head 相同数据应得到相同聚类"
        report(name, True)
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ T2 与 e64a 参考实现语义对拍
def t2_vs_e64a_reference():
    name = "T2 与 e64a.greedy_cluster_assign 语义对拍（mass 口径源）"
    try:
        g = torch.Generator().manual_seed(20261007)
        for T, H, dd, sim in [(200, 3, 8, 0.9), (300, 2, 16, 0.85), (150, 4, 4, 0.95)]:
            x = torch.randn(T, H, dd, generator=g)
            cent_ref, assign_ref, klive_ref = E64A.greedy_cluster_assign(x, sim=sim, k_max=T)
            cent, assign, k_live, _, _, _ = greedy_cold(x, sim)
            assert torch.equal(assign, assign_ref), \
                f"T={T} sim={sim}: assign 与 e64a 不一致（贪心时序语义漂移）"
            assert torch.equal(k_live, klive_ref), f"T={T} sim={sim}: k_live 不一致"
            live = int(k_live.max().item())
            assert torch.allclose(cent[:, :live], cent_ref[:, :live], atol=1e-5), \
                f"T={T} sim={sim}: 簇心不一致"
        report(name, True, "3 组 (T,H,d,sim) 全对拍一致")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ T3 far greedy 增量 == 全量重放
def t3_incremental_equals_replay():
    name = "T3 far greedy 增量续跑 == 全量重放（decode far_hi 前移精确性）"
    try:
        args = make_args(
            tli_far_select="sim_greedy", tli_near_select="4bit",
            tli_enable_kmeans=True, tli_enable_layer_skip=False,
            tli_alpha=0.25, tli_beta=0.25, tli_gamma=0.5,
        )
        S1, DS = 4096, 512
        k, _ = gen_kq(S1 + DS, seed=7)
        cu1 = torch.tensor([0, S1])
        # 逐步增长：S1 → S1+64 → S1+128 ... → S1+DS（模拟 decode 追加，far_hi 多次前移）
        idx = IDX_NEW.TLIIndexer(args)
        idx.layer_idx = 3
        idx.prepare_index(k[:, :S1], cu1)
        for S in range(S1 + 64, S1 + DS + 1, 64):
            idx.prepare_index(k[:, :S], torch.tensor([0, S]))
        # 冷启动全量重放（同数据同 far 终态）
        idx2 = IDX_NEW.TLIIndexer(args)
        idx2.layer_idx = 3
        idx2.prepare_index(k[:, :S1 + DS], torch.tensor([0, S1 + DS]))
        assert torch.equal(idx._km_token_assign, idx2._km_token_assign), \
            "增量 assign 与全量重放不一致（贪心时序续跑精确性破坏）"
        assert torch.equal(idx._km_greedy_klive, idx2._km_greedy_klive), "k_live 不一致"
        live = int(idx._km_greedy_klive.max().item())
        assert torch.allclose(idx._km_centroids[:, :live], idx2._km_centroids[:, :live], atol=1e-6), \
            "增量簇心与全量重放不一致"
        assert int(idx._km_greedy_klive.max().item()) < S1, \
            "sim=0.9 随机数据簇数应显著小于 token 数（归并实际发生）"
        report(name, True, f"S {S1}→{S1+DS} 增量 8 次，assign 逐位一致，簇数 {live}/{S1}")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ T4 同接口全链路跑通
def t4_pipeline_runs():
    name = "T4 同接口全链路（ccluster_kmeans / cavg_sim / ccluster_sim）"
    try:
        S = 4096
        k, q = gen_kq(S, seed=11)
        cfgs = {
            "ccluster_kmeans": dict(tli_far_select="cluster", tli_near_select="cluster"),
            "cavg_sim(s=0.9)": dict(tli_far_select="sim_greedy", tli_near_select="4bit",
                                    tli_sim=0.9),
            "ccluster_sim(s=0.9,nope)": dict(tli_far_select="sim_greedy", tli_near_select="sim_greedy",
                                             tli_sim=0.9, tli_sim_dims="nope"),
            "ccluster_sim(s=0.9,sub)": dict(tli_far_select="sim_greedy", tli_near_select="sim_greedy",
                                            tli_sim=0.9, tli_sim_dims="subspace"),
        }
        for tag, extra in cfgs.items():
            args = make_args(
                tli_enable_kmeans=True, tli_enable_layer_skip=False,
                tli_alpha=0.25, tli_beta=0.25, tli_gamma=0.5,
                tli_far_method="minmax", tli_near_method="avg",
                **extra,
            )
            idx, mask = run_mask(None, args, k, q)
            assert mask.dtype == torch.bool and mask.shape[-1] == S, \
                f"{tag}: mask 形状/类型异常 {tuple(mask.shape)} {mask.dtype}"
            n_sel = int(mask.sum())
            assert 0 < n_sel <= mask.numel(), f"{tag}: mask 全空/全满异常"
            # sink/swa 强制区必选
            assert bool(mask[..., :128].all()), f"{tag}: sink 区未全选"
            assert bool(mask[..., S - 128:].all()), f"{tag}: swa 区未全选"
        report(name, True, f"4 配置全过（S={S}），mid 选中数 " +
               "/".join(str(int(run_mask(None, make_args(
                   tli_enable_kmeans=True, tli_enable_layer_skip=False,
                   tli_alpha=0.25, tli_beta=0.25, tli_gamma=0.5, **extra), k, q)[1]
                   .sum())) for extra in cfgs.values()))
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ T5 预算语义
def t5_budget_semantics():
    name = "T5 预算语义（mid_token 预算 + far/near 分区约束，γ 活化）"
    try:
        S = 8192
        bs, sink_tok, swa_tok, K2 = 64, 128, 128, 1024
        k, q = gen_kq(S, seed=13)
        args = make_args(
            tli_far_select="sim_greedy", tli_near_select="sim_greedy", tli_sim=0.9,
            tli_enable_kmeans=True, tli_enable_layer_skip=False,
            tli_alpha=0.25, tli_beta=0.25, tli_gamma=0.5,
        )
        _, mask = run_mask(None, args, k, q)
        # 手算预算（与 compute_mask 同式，TASK.md L172 严格口径无保底；
        # 【B10 修复 2026-10-10】near 左界从 swa 起点推 → far_tok_hi 平移 −swa_tok）：
        #   mid_len=7936, near_len_dyn=1984 → near_blks=(8192−128−1984)//64=95
        #   → far_tok_hi=6080, swa_lo_tok=8064
        #   K2_mid=768；nb_near=round(128*0.25)=32；nt_near=min(32*64*0.5,768)=768
        #   far_budget=max(0,768-768)=0；near=768-0=768（γ 高值让渡满额 → far=0）
        mid_len = S - sink_tok - swa_tok
        near_len_dyn = max(bs, int(0.25 * mid_len))
        near_blks = max(2, (S - swa_tok - near_len_dyn) // bs)   # B10：swa 起点推
        far_tok_hi, swa_lo_tok = near_blks * bs, S - swa_tok
        K2_mid = K2 - sink_tok - swa_tok
        nb_near = max(1, int(round(128 * 0.25)))
        nt_near = min(int(nb_near * bs * 0.5), K2_mid)
        far_budget = max(0, K2_mid - nt_near)
        exp_far = min(far_budget, far_tok_hi - sink_tok, K2_mid)
        exp_near = K2_mid - exp_far
        m = mask[0, 0]   # [Hkv, S]
        n_far = int(m[:, sink_tok:far_tok_hi].sum(dim=-1)[0])
        n_near = int(m[:, far_tok_hi:swa_lo_tok].sum(dim=-1)[0])
        n_mid = int(m[:, sink_tok:swa_lo_tok].sum(dim=-1)[0])
        assert n_far == exp_far, f"far 侧选中 {n_far} != 预算 {exp_far}"
        assert n_near == exp_near, f"near 侧选中 {n_near} != 预算 {exp_near}"
        assert n_mid == K2_mid, f"mid 总选中 {n_mid} != mid_token 预算 {K2_mid}"
        # 每个 kv-head 独立同预算
        assert bool((m[:, sink_tok:swa_lo_tok].sum(-1) == K2_mid).all()), "各 kv-head 预算不一致"

        # 对照：cavg（near=4bit）far 预算同样按 γ 活化（TASK.md 严格口径，
        # 与 ccluster 同式——原「far_tokens=512 γ 死参数」回归口径已随
        # E109a 污染重跑废弃）
        args_cavg = make_args(
            tli_far_select="cluster", tli_near_select="4bit",
            tli_enable_kmeans=True, tli_enable_layer_skip=False,
            tli_alpha=0.25, tli_beta=0.25, tli_gamma=0.5,
        )
        _, mask_c = run_mask(None, args_cavg, k, q)
        mc = mask_c[0, 0]
        n_far_c = int(mc[:, sink_tok:far_tok_hi].sum(dim=-1)[0])
        assert n_far_c == exp_far, \
            f"cavg far 侧 {n_far_c} != γ 活化预算 {exp_far}（TASK.md 严格口径被改动！）"
        assert int(mc[:, sink_tok:swa_lo_tok].sum(dim=-1)[0]) == K2_mid, "cavg mid 总预算破坏"
        report(name, True, f"ccluster_sim: far={n_far} near={n_near} mid={n_mid}=K2_mid; "
                           f"cavg 对照 far={n_far_c}（γ 活化口径一致）")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ T6 near 回退混合
def t6_near_fallback():
    name = "T6 near 簇覆盖 + 细筛回退（块对齐尾巴 token 仍可召回）"
    try:
        # S 不整除 bs：near 簇覆盖区右缘块对齐（near_hi），消费端 near 池右缘 token 对齐
        # （swa_lo_tok=S-128）→ 尾巴 [near_hi, swa_lo_tok) 走细筛分回退
        S = 4096 + 30
        k, q = gen_kq(S, seed=17)
        q_dir = torch.nn.functional.normalize(q[0, 0].float(), dim=-1)   # [H, D]
        # 尾巴区高亮 token：k[3980] = 大幅 q 方向 → 细筛原始分全场最高，必须被召回
        k = k.clone()
        k[0, 3980] = q_dir[0] * 30.0
        # 簇覆盖区高亮 token：k[3500] 同样构造（经簇代表分召回）
        k[0, 3500] = q_dir[1] * 30.0
        bs, sink_tok, swa_tok = 64, 128, 128
        near_hi = (S - swa_tok) // bs * bs     # 3968
        swa_lo_tok = S - swa_tok               # 3998
        assert 3980 >= near_hi and 3980 < swa_lo_tok, "测试前提：3980 落在回退尾巴区"
        assert 3500 < near_hi, "测试前提：3500 落在簇覆盖区"
        args = make_args(
            tli_far_select="sim_greedy", tli_near_select="sim_greedy", tli_sim=0.9,
            tli_enable_kmeans=True, tli_enable_layer_skip=False,
            tli_alpha=0.25, tli_beta=0.25, tli_gamma=0.5,
        )
        _, mask = run_mask(None, args, k, q)
        m = mask[0, 0]
        assert bool(m[:, 3980].all()), "回退尾巴区最高分 token 未被召回（细筛回退路径失效）"
        assert bool(m[:, 3500].any()), "簇覆盖区高亮 token 未被任何 kv-head 召回"
        # 预算仍守恒（mid 总数 = K2_mid，S 不整除时 p 宽 = S；
        # 【B10】公式重放同步 swa 起点推 + 实现侧 pad 口径 kt*bs）
        K2_mid = 1024 - sink_tok - swa_tok
        kt_pad = (S + bs - 1) // bs * bs          # 4160（实现 pad 口径）
        mid_len = kt_pad - sink_tok - swa_tok
        near_len_dyn = max(bs, int(0.25 * mid_len))
        near_blks = max(2, (kt_pad - swa_tok - near_len_dyn) // bs)   # B10
        far_tok_hi = near_blks * bs
        n_mid = int(m[:, sink_tok:swa_lo_tok].sum(dim=-1)[0])
        assert n_mid == K2_mid, f"非整除 S 下 mid 总选中 {n_mid} != {K2_mid}"
        report(name, True, f"回退 token 3980 与簇区 token 3500 均召回，mid={n_mid}=K2_mid "
                           f"(far_hi={far_tok_hi}, near_hi={near_hi}, swa_lo={swa_lo_tok})")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ T7 回归保护（旧 vs 新逐位对拍）
def t7_regression_vs_main_tree():
    name = "T7 回归保护（vs 主树：非 e64 臂逐位相同；e64 臂 B10 结构不变量）"
    # 【10-10 B10 合入（e121）】e64 分区臂（α>0 且 β>0）near 左界改从 swa 起点
    # 推 → mask 与主树旧代码**有意不同**，逐位对拍只保留非 e64 臂（默认 flag /
    # 老逻辑 α=0，B10 明确不触碰）；e64 臂改断言 B10 结构不变量（预算守恒 +
    # 强制区 + 两侧各自预算语义自洽），防主树侧意外漂移以外的真实回归。
    try:
        S0 = 6144
        bs, sink_tok, swa_tok, K2 = 64, 128, 128, 1024
        K2_mid = K2 - sink_tok - swa_tok
        k_full, q = gen_kq(S0 + 8, seed=23)
        cfgs = {
            "默认flag(4bit/4bit)": dict(tli_enable_layer_skip=False),
            "cavg(far=cluster)": dict(tli_far_select="cluster", tli_enable_kmeans=True,
                                      tli_enable_layer_skip=False,
                                      tli_alpha=0.125, tli_beta=0.25, tli_gamma=0.75),
            "mavg(far=4bit,αβγ)": dict(tli_enable_kmeans=False, tli_enable_layer_skip=False,
                                       tli_alpha=0.125, tli_beta=0.375, tli_gamma=0.625),
            "tli老逻辑(α=0)": dict(tli_enable_kmeans=True, tli_enable_layer_skip=False),
        }
        for tag, extra in cfgs.items():
            e64 = extra.get("tli_alpha", 0.0) > 0 and extra.get("tli_beta", 0.0) > 0
            args = make_args(**extra)
            idx_new = IDX_NEW.TLIIndexer(args)
            idx_new.layer_idx = 3
            idx_old = IDX_OLD.TLIIndexer(args)
            idx_old.layer_idx = 3
            # prefill + 8 步 decode 模拟（S 每步 +1，走聚类缓存/增量路径）
            for step in range(9):
                S = S0 + step
                k = k_full[:, :S]
                cu = torch.tensor([0, S])
                q_ids = torch.tensor([S - 1])
                mn, _ = idx_new.prepare_mask(q, q_ids, k, cu, q.shape[-1] ** -0.5)
                mo, _ = idx_old.prepare_mask(q, q_ids, k, cu, q.shape[-1] ** -0.5)
                assert mn.shape == mo.shape and mn.dtype == mo.dtype, \
                    f"{tag} step{step}: mask 形状漂移 {tuple(mn.shape)} vs {tuple(mo.shape)}"
                if not e64:
                    assert torch.equal(mn, mo), \
                        f"{tag} step{step}: mask 与主树旧代码不一致（回归破坏！）"
                else:
                    # B10 结构不变量：两侧 sink/swa 强制 + mid 预算不超上限；
                    # 旧侧 mid 允许 < K2_mid（N5 已记录的合法饥饿：旧口径 near
                    # 池窄 swa_tok；非对齐 S 还有 ≤63 token 的 pad 边角 = F5
                    # 已知偏差，B10 后对齐步满额、非对齐步仍可截断）。
                    # B10 单调性：near 池左界只左移（⊇ 旧池）、本组两臂
                    # far_budget=0 → mid_new ≥ mid_old（只增不减）。
                    for tag2, m in (("new", mn), ("old", mo)):
                        assert m.dtype == torch.bool
                        assert bool(m[..., :sink_tok].all()), \
                            f"{tag}/{tag2} step{step}: sink 强制区缺失"
                        assert bool(m[..., S - swa_tok:].all()), \
                            f"{tag}/{tag2} step{step}: swa 强制区缺失"
                        per_head_mid = m[0, 0, :, sink_tok:S - swa_tok].sum(-1)
                        assert bool((per_head_mid > 0).all()) and \
                            bool((per_head_mid <= K2_mid).all()), \
                            f"{tag}/{tag2} step{step}: mid 预算越界 {per_head_mid.tolist()}"
                    mid_new = mn[0, 0, :, sink_tok:S - swa_tok].sum(-1)
                    mid_old = mo[0, 0, :, sink_tok:S - swa_tok].sum(-1)
                    assert bool((mid_new >= mid_old).all()), \
                        f"{tag} step{step}: B10 应使 near 池只增不减，" \
                        f"mid new {mid_new.tolist()} < old {mid_old.tolist()}"
            # 二次请求（clear 后重跑，验证 clear 重置等价性）
            idx_new.clear()
            idx_old.clear()
            mn, _ = idx_new.prepare_mask(q, torch.tensor([S0 - 1]), k_full[:, :S0],
                                         torch.tensor([0, S0]), q.shape[-1] ** -0.5)
            mo, _ = idx_old.prepare_mask(q, torch.tensor([S0 - 1]), k_full[:, :S0],
                                         torch.tensor([0, S0]), q.shape[-1] ** -0.5)
            if not e64:
                assert torch.equal(mn, mo), f"{tag}: clear 后重跑 mask 不一致"
        report(name, True, "非 e64 2 配置逐位一致；e64 2 配置 B10 结构不变量"
                           "（预算/强制区）9 步 × 2 侧全过")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ T8 clear 跨请求重置
def t8_clear_greedy_state():
    name = "T8 clear() 跨请求重置（greedy 增量状态不残留）"
    try:
        S = 4096
        k1, q1 = gen_kq(S, seed=31)
        k2, q2 = gen_kq(S, seed=32)
        args = make_args(
            tli_far_select="sim_greedy", tli_near_select="sim_greedy", tli_sim=0.9,
            tli_enable_kmeans=True, tli_enable_layer_skip=False,
            tli_alpha=0.25, tli_beta=0.25, tli_gamma=0.5,
        )
        idx = IDX_NEW.TLIIndexer(args)
        idx.layer_idx = 3
        _, m1 = run_mask(None, args, k1, q1, indexer=idx)
        idx.clear()   # 模拟新请求 prefill 前的 clear（patch 口径）
        assert idx._km_greedy_sums is None and idx._km_near_centroids is None \
            and idx._km_near_key is None and idx._km_far_hi_cached == -1, \
            "clear 后 greedy/near 聚类状态未清零（跨请求簇心污染风险）"
        _, m2 = run_mask(None, args, k2, q2, indexer=idx)
        # 冷启动对照（独立实例跑 k2）
        _, m2_ref = run_mask(None, args, k2, q2)
        assert torch.equal(m2, m2_ref), "clear 后重跑 != 冷启动（状态残留影响结果）"
        # 新旧两请求 mask 确实不同（数据不同，排除恒等假阳性）
        assert not torch.equal(m1, m2), "不同数据 mask 完全相同（测试假阳性）"
        report(name, True)
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ T9 互斥守卫
def t9_guards():
    name = "T9 flag 互斥守卫（per_q_head / sigma / moba × 新路径）"
    try:
        def must_raise(**kw):
            try:
                IDX_NEW.TLIIndexer(make_args(tli_enable_layer_skip=False, **kw))
                return False
            except (AssertionError, ValueError):
                return True
        assert must_raise(tli_per_q_head=True, tli_far_select="cluster"), "per_q_head × far cluster 未拦截"
        assert must_raise(tli_per_q_head=True, tli_far_select="sim_greedy"), "per_q_head × far sim_greedy 未拦截"
        assert must_raise(tli_per_q_head=True, tli_near_select="cluster"), "per_q_head × near cluster 未拦截"
        assert must_raise(tli_per_q_head=True, tli_near_select="sim_greedy"), "per_q_head × near sim_greedy 未拦截"
        assert must_raise(tli_near_select="cluster", tli_sigma_select="far"), "near cluster × sigma 未拦截"
        assert must_raise(tli_near_select="sim_greedy", tli_moba=True), "near sim_greedy × moba 未拦截"
        assert must_raise(tli_sim_dims="bogus"), "非法 sim_dims 未拦截"
        assert must_raise(tli_near_select="bogus"), "非法 near_select 未拦截"
        # 合法组合不炸
        IDX_NEW.TLIIndexer(make_args(tli_enable_layer_skip=False, tli_far_select="sim_greedy",
                                     tli_near_select="sim_greedy", tli_sim=0.9, tli_sim_dims="nope"))
        report(name, True, "9 拦截 + 1 合法组合")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


if __name__ == "__main__":
    torch.manual_seed(0)
    t1_greedy_basics()
    t2_vs_e64a_reference()
    t3_incremental_equals_replay()
    t4_pipeline_runs()
    t5_budget_semantics()
    t6_near_fallback()
    t7_regression_vs_main_tree()
    t8_clear_greedy_state()
    t9_guards()
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print("\n" + "=" * 60)
    print(f"总计 {len(RESULTS)} 项，通过 {len(RESULTS) - n_fail}，失败 {n_fail}")
    sys.exit(1 if n_fail else 0)
