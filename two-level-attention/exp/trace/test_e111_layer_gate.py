# E111 单测：动态 per-request 层 gate（D' 终态版）的判定逻辑、flag 语义、
#   省算等价性、method 组合正交性、预算语义与 HEAD 回归（全 CPU，无模型）。
#
# 验证七件事：
#   ① dynamic 判定逻辑：合成信号验证 τ 阈值行为（信号低于 τ → 跳 far），
#      far 区边界口径与部署分区（_partition_blks）一致；
#   ② clear() 跨请求重置 + 新信号重决策；
#   ③ flag 语义等价：gate=static ≡ 旧 --tli_enable_layer_skip true（默认掩码
#      DEFAULT_MASK）；gate 未设 + enable_layer_skip false 解析为 none；
#   ④ gate=none 回归保护：与 HEAD（12ef0d416）基线逐位一致（子进程跑
#      /tmp/e111_head_pkg 的 HEAD 版 sparse_attn，同 seed 同输入对比 mask）；
#   ⑤ 与 method 组合正交：mavg+dynamic / aavg+static 组合跑通；
#   ⑥ 预算语义：跳 far 层时 swa 尾部强制区全 True、far 区全 False、
#      mask 总数恒 K2=1024（near/sink/swa 预算不受 far 污染）；
#   ⑦ 省算等价：skip 全程（量化跳过 + einsum 分段）的 mask 与
#      「完整计算 + 手工 far 块 -inf」逐位一致（省算不改变语义）。
#
# 用法：
#   python exp/trace/test_e111_layer_gate.py                       # 主模式（worktree 新代码）
#   python exp/trace/test_e111_layer_gate.py --pkg /tmp/e111_head_pkg \
#       --emit-baseline /tmp/e111_baseline.pt                      # HEAD 基线发射（子进程）
# 基线包准备（bash）：
#   cp -r <wt>/two-level-attention/sparse_attn /tmp/e111_head_pkg/
#   git -C <wt> show HEAD:two-level-attention/sparse_attn/indexer/tli_indexer.py \
#       > /tmp/e111_head_pkg/sparse_attn/indexer/tli_indexer.py   （arguments/patch 同理）
import argparse
import os
import subprocess
import sys

import torch

# --pkg 决定 import 哪个 sparse_attn（必须在 import sparse_attn 前处理）
_PKGS = [a for a in sys.argv if a.startswith("--pkg")]
PKG = _PKGS[0].split("=", 1)[1] if _PKGS and "=" in _PKGS[0] else (
    sys.argv[sys.argv.index("--pkg") + 1] if "--pkg" in sys.argv else None)
if PKG is not None:
    sys.path.insert(0, PKG)
else:
    sys.path.insert(0, "/home/wangyuanshuo02/wt-e111-layerskip/two-level-attention")

from sparse_attn.arguments import add_sparse_attn_args
from sparse_attn.indexer.tli_indexer import TLIIndexer

S, HKV, H, D = 8192, 8, 32, 128
BS = 64
SINK_TOK, SWA_TOK = 128, 128            # sink_blocks=2×64 / sliding_window=128
ALPHA, BETA, GAMMA = 0.125, 0.375, 0.625  # E98 best mavg 占位
# 部署口径独立计算（不复用被测 _partition_blks）：
MID_LEN = S - SINK_TOK - SWA_TOK
NEAR_LEN_DYN = max(BS, int(ALPHA * MID_LEN))
NEAR_BLKS = max(2, (S - NEAR_LEN_DYN) // BS)
FAR_LO_TOK, FAR_HI_TOK = SINK_TOK, NEAR_BLKS * BS   # far token 区 [128, 7168)
QROWS = 16

# HEAD 基线三配置（gate=none 口径 = enable_layer_skip false = 主表口径）
BASELINE_CFGS = {
    "mavg_full": dict(sub="full", a=ALPHA, b=BETA, g=GAMMA, fm="avg", nm="minmax"),
    "mavg_tail": dict(sub="tail", a=ALPHA, b=BETA, g=GAMMA, fm="avg", nm="minmax"),
    "mavg_pool": dict(sub="full", a=0.0, b=0.0, g=0.0, fm="avg", nm="minmax"),
}


def make_args(**kw):
    p = argparse.ArgumentParser()
    add_sparse_attn_args(p)
    sub = kw.pop("sub", "full")
    a = p.parse_args([
        "--method", "tli", "--tli_subspace", sub,
        "--tli_alpha", str(kw.pop("a", ALPHA)), "--tli_beta", str(kw.pop("b", BETA)),
        "--tli_gamma", str(kw.pop("g", GAMMA)),
        "--tli_far_method", kw.pop("fm", "avg"), "--tli_near_method", kw.pop("nm", "minmax"),
        "--tli_enable_layer_skip", "false",
    ])
    for k2, v in kw.items():
        setattr(a, k2, v)
    return a


def fixed_inputs(seed, S_=S):
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(1, S_, HKV, D, generator=g)
    q = torch.randn(1, 1, H, D, generator=g)
    return k, q


def synth_prefill(far_massive):
    """合成 prefill q/k（patch 口径布局：q [B,H,S,D] / k [B,Hkv,S,D]）：
    e0 方向控制 far/near 区分数。
    far_massive=False → far 区 k 正交（far mass ≈ 0）；
    far_massive=True  → far 区 k 对齐（far mass ≈ far 区 token 占比）。"""
    g = torch.Generator().manual_seed(7)
    k = torch.randn(1, HKV, S, D, generator=g) * 0.01
    q = torch.randn(1, H, S, D, generator=g) * 0.01
    q[..., 0] = 8.0                                   # 所有 q 行同向 e0
    if far_massive:
        k[0, :, FAR_LO_TOK:FAR_HI_TOK, 0] = 8.0       # far 对齐 → far mass 大
    else:
        k[0, :, FAR_HI_TOK:S - SWA_TOK, 0] = 8.0      # near 对齐 → far mass ≈ 0
    return q, k


def run_decode_step(idx, k, q_dec=None):
    """跑一步 decode 全链（prepare_index→compute_score→compute_mask）。"""
    if q_dec is None:
        q_dec = torch.randn(1, 1, H, D)
    cu = torch.tensor([0, k.shape[1]])
    mask, _ = idx.prepare_mask(q_dec, torch.tensor([k.shape[1] - 1]), k, cu, D ** -0.5)
    return mask


def ref_far_frac(q, k):
    """独立重算信号（E67 mass_profile 同构，边界用测试硬算部署口径；
    k 布局 [B,Hkv,S,D] 与 patch 一致）。"""
    nrow = min(QROWS, S)
    qg = q[0, :, -nrow:].reshape(HKV, H // HKV, nrow, -1).sum(1).float()
    s = torch.einsum("hrd,hsd->hrs", qg, k[0].float()) * (D ** -0.5)
    t = (S - nrow) + torch.arange(nrow)
    pos = torch.arange(S)
    s = s.masked_fill(pos[None, :] > t[:, None], float("-inf"))
    p = torch.softmax(s, dim=-1)
    tot = p.sum(dim=-1)
    far = p[..., FAR_LO_TOK:FAR_HI_TOK].sum(dim=-1)
    return float((far / tot.clamp(min=1e-12)).mean())


def emit_baseline(path):
    """HEAD 基线发射：用 --pkg 指定的 HEAD 包跑三配置 decode mask 存盘。"""
    out = {}
    for name, cfg in BASELINE_CFGS.items():
        torch.manual_seed(0)
        idx = TLIIndexer(make_args(sub=cfg["sub"], a=cfg["a"], b=cfg["b"],
                                   g=cfg["g"], fm=cfg["fm"], nm=cfg["nm"]))
        idx.layer_idx = 5
        k, q = fixed_inputs(11)
        out[name] = run_decode_step(idx, k, q).clone()
    torch.save(out, path)
    print(f"[baseline] saved -> {path} ({list(out)})")


def main():
    torch.manual_seed(0)
    fails = []

    # ---- ① dynamic 判定逻辑（合成信号 × τ 阈值行为）----
    idx = TLIIndexer(make_args(tli_layer_gate="dynamic"))
    idx.layer_idx = 9
    if idx.layer_gate != "dynamic":
        fails.append(f"① layer_gate 解析错: {idx.layer_gate}")
    # 场景 A：far mass ≈ 0 → τ0.1 跳
    qA, kA = synth_prefill(far_massive=False)
    idx.observe_prefill_gate(qA, kA)
    sigA = idx._dyn_far_sig
    refA = ref_far_frac(qA, kA)
    if sigA is None or abs(sigA - refA) > 1e-6:
        fails.append(f"① far 信号值与独立重算不一致: {sigA} vs {refA}")
    if not idx._dyn_skip_far:
        fails.append(f"① far_mass≈0({sigA}) 未触发跳过 (τ={idx.gate_tau})")
    # far 区 token 全部不选（⑥ 在此合并验证 far 区清零）
    kA_dec, _ = fixed_inputs(3)
    maskA = run_decode_step(idx, kA_dec)
    if maskA[..., FAR_LO_TOK:FAR_HI_TOK].any():
        fails.append("① skip 层 far 区仍有 token 被选（预算污染！）")
    if not maskA[..., -SWA_TOK:].all():
        fails.append("① skip 层 swa 强制区未全 True")
    totA = maskA.sum(-1).flatten().tolist()
    if any(abs(x - 1024) > 1e-6 for x in totA):
        fails.append(f"① skip 层 mask 预算 {set(totA)} != 1024")
    print(f"① far_mass≈{sigA:.4f} < τ0.1 → skip ✓  far 区全 False / swa 全 True / "
          f"预算 1024 ✓")

    # 场景 B：far mass ≈ 1（softmax 全压 far）→ τ0.1 不跳；τ1.5 跳（阈值方向性；
    #   合成场景下 far/near 分数对比极端，far_frac 逼近 0/1 两端，用 τ>1 的
    #   极端臂单测「信号 ≥ τ 不跳 / 信号 < τ 跳」两方向）
    qB, kB = synth_prefill(far_massive=True)
    idxB = TLIIndexer(make_args(tli_layer_gate="dynamic"))
    idxB.layer_idx = 9
    idxB.observe_prefill_gate(qB, kB)
    sigB = idxB._dyn_far_sig
    refB = ref_far_frac(qB, kB)
    if abs(sigB - refB) > 1e-6:
        fails.append(f"① 场景B 信号与独立重算不一致: {sigB} vs {refB}")
    if idxB._dyn_skip_far:
        fails.append(f"① far_mass≈{sigB} 高但 τ0.1 触发了跳过（阈值方向错）")
    idxB2 = TLIIndexer(make_args(tli_layer_gate="dynamic", tli_layer_gate_tau=1.5))
    idxB2.layer_idx = 9
    idxB2.observe_prefill_gate(qB, kB)
    if not idxB2._dyn_skip_far:
        fails.append(f"① far_mass≈{sigB} 但 τ1.5 未触发跳过（阈值边界错）")
    print(f"① far_mass≈{sigB:.4f}: τ0.1 不跳 ✓ / τ1.5 跳 ✓（信号独立重算对上 "
          f"{refA:.6f}/{refB:.6f}）")

    # ---- ② clear() 跨请求重置 + 重决策 ----
    idx.clear()
    if idx._dyn_skip_far or idx._dyn_far_sig is not None:
        fails.append("② clear() 未重置动态 gate 状态")
    idx.observe_prefill_gate(qB, kB)      # 新请求：far mass 大 → 不跳
    if idx._dyn_skip_far:
        fails.append("② clear 后新信号未重新决策（残留旧 skip）")
    print("② clear() 重置 + 新信号重决策 ✓")

    # ---- ③ flag 语义等价：static ≡ 旧 enable_layer_skip true ----
    for li, in_mask in ((7, True), (5, False)):     # 7 ∈ DEFAULT_MASK, 5 ∉
        old = TLIIndexer(make_args(tli_enable_layer_skip="true"))
        new = TLIIndexer(make_args(tli_enable_layer_skip="false", tli_layer_gate="static"))
        old.layer_idx = new.layer_idx = li
        if new.layer_gate != "static" or old.layer_gate != "static":
            fails.append(f"③ L{li} layer_gate 解析错: old={old.layer_gate} new={new.layer_gate}")
        k, q = fixed_inputs(23)
        m_old = run_decode_step(old, k, q)
        m_new = run_decode_step(new, k, q)
        if old.skip_far != in_mask or new.skip_far != in_mask:
            fails.append(f"③ L{li} skip_far 错: old={old.skip_far} new={new.skip_far} (期望 {in_mask})")
        if not torch.equal(m_old, m_new):
            fails.append(f"③ L{li} static 模式 mask 与旧 flag 口径不一致")
    # 未传 gate + enable_layer_skip false → 解析为 none（主表口径）
    idxN = TLIIndexer(make_args())
    if idxN.layer_gate != "none":
        fails.append(f"③ gate 未设 + skip false 应解析为 none，实为 {idxN.layer_gate}")
    print("③ static ≡ 旧 flag（L7 skip / L5 不 skip，mask 逐位一致）；"
          "gate 未设+false → none ✓")

    # ---- ④ gate=none 回归：与 HEAD 基线逐位一致（子进程）----
    bl_path = "/tmp/e111_baseline.pt"
    here = os.path.abspath(__file__)
    if "--pkg" in sys.argv:
        print("[skip] HEAD 对照在 emit 模式下跳过")
    else:
        if not os.path.isfile(bl_path):
            r = subprocess.run(
                [sys.executable, here, "--pkg", "/tmp/e111_head_pkg",
                 "--emit-baseline", bl_path], capture_output=True, text=True)
            if r.returncode != 0:
                fails.append(f"④ HEAD 基线子进程失败: {r.stderr[-400:]}")
        if os.path.isfile(bl_path):
            bl = torch.load(bl_path, weights_only=False)
            for name, cfg in BASELINE_CFGS.items():
                idxH = TLIIndexer(make_args(sub=cfg["sub"], a=cfg["a"], b=cfg["b"],
                                            g=cfg["g"], fm=cfg["fm"], nm=cfg["nm"],
                                            tli_layer_gate="none"))
                idxH.layer_idx = 5
                k, q = fixed_inputs(11)
                m_new = run_decode_step(idxH, k, q)
                if not torch.equal(m_new, bl[name]):
                    n_diff = (m_new != bl[name]).sum().item()
                    fails.append(f"④ {name} gate=none 与 HEAD 不一致（{n_diff} 位）")
            print("④ gate=none 三配置（mavg_full/mavg_tail/mavg_pool）与 HEAD "
                  "逐位一致 ✓")

    # ---- ⑤ method 组合正交：mavg+dynamic / aavg+static ----
    for tag, kw, gate_kw, li in (
        ("mavg+dynamic", dict(fm="avg", nm="minmax"), dict(tli_layer_gate="dynamic"), 9),
        ("aavg+static", dict(fm="avg", nm="avg"), dict(tli_layer_gate="static"), 7),
    ):
        idxC = TLIIndexer(make_args(**kw, **gate_kw))
        idxC.layer_idx = li
        k, q = fixed_inputs(31)
        if gate_kw.get("tli_layer_gate") == "dynamic":
            idxC.observe_prefill_gate(qA, kA)      # far mass 低 → skip
        m = run_decode_step(idxC, k, q)
        tot = m.sum(-1).flatten().tolist()
        if any(abs(x - 1024) > 1e-6 for x in tot):
            fails.append(f"⑤ {tag} mask 预算 {set(tot)} != 1024")
        if not m[..., -SWA_TOK:].all():
            fails.append(f"⑤ {tag} swa 强制区未全 True")
        sk = (tag.endswith("dynamic") and idxC.skip_far) or (tag.endswith("static") and idxC.skip_far)
        print(f"⑤ {tag} skip_far={idxC.skip_far} 预算 1024 swa 强制 ✓")

    # ---- ⑦ 省算等价：skip 全程（量化跳过 + einsum 分段）vs 完整计算+手工 -inf ----
    #   覆盖四种配置：tail/full × (分区 α.125β.375 / 单池 α0β0)——E109 的 α=0
    #   单池纯 method 臂同样须与 gate 正交
    POOL_NEAR_BLKS = max(2, (S - 2048) // BS)       # α=0 时 near_len=2048（独立口径）
    for sub, a, b, hi_exp in (("tail", ALPHA, BETA, NEAR_BLKS),
                              ("full", ALPHA, BETA, NEAR_BLKS),
                              ("full", 0.0, 0.0, POOL_NEAR_BLKS)):
        # A：skip 全程省算路径
        idxA = TLIIndexer(make_args(sub=sub, a=a, b=b, tli_layer_gate="dynamic"))
        idxA.layer_idx = 9
        idxA.observe_prefill_gate(qA, kA)           # far mass 低 → skip_far=True
        k, q = fixed_inputs(41)
        maskA = run_decode_step(idxA, k, q)
        # B：完整计算（gate=none 不省算），手工把 far 块 coarse -inf 再走 skip 版 compute_mask
        idxB = TLIIndexer(make_args(sub=sub, a=a, b=b, tli_layer_gate="none"))
        idxB.layer_idx = 9
        cu = torch.tensor([0, k.shape[1]])
        idxd = idxB.prepare_index(k, cu)
        sd = idxB.compute_score(q, torch.tensor([k.shape[1] - 1]), idxd, D ** -0.5)
        sd["score_coarse"][..., 2:hi_exp] = float("-inf")
        idxB.skip_far = True
        maskB = idxB.compute_mask(torch.tensor([k.shape[1] - 1]), sd)
        if not torch.equal(maskA, maskB):
            n_diff = (maskA != maskB).sum().item()
            fails.append(f"⑦ subspace={sub} α{a} 省算路径 mask 与完整计算+手工-inf "
                         f"不一致（{n_diff} 位）")
        # 口径一致性：_partition_blks == 测试独立计算的部署边界
        lo, hi = idxA._partition_blks(k.shape[1] // BS)
        if (lo, hi) != (2, hi_exp):
            fails.append(f"⑦ subspace={sub} α{a} _partition_blks ({lo},{hi}) "
                         f"!= 独立计算 (2,{hi_exp})")
        tag = "分区" if a > 0 else "单池"
        print(f"⑦ subspace={sub} {tag} 省算（量化跳过+einsum 分段）≡ 完整计算+far -inf "
              f"（逐位一致）；分区口径 ({lo},{hi}) ✓")

    print("\n" + ("ALL PASS" if not fails else "FAIL:\n" + "\n".join(fails)))
    return 0 if not fails else 1


if __name__ == "__main__":
    if "--emit-baseline" in sys.argv:
        emit_baseline(sys.argv[sys.argv.index("--emit-baseline") + 1])
        sys.exit(0)
    sys.exit(main())
