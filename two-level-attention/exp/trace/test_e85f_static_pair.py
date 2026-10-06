# E85f 干跑单测：per-layer 静态 pair 选取的 e2e 管线接入正确性（CPU，无模型）。
# 验证四件事：
#   ① observe_prefill_q 的 pair 选取正确（受控幅值 → top-16 完整对、长度 32、
#      配对完整性 j∈前半 ⇔ j+64∈后半、排序）；
#   ② clear() 跨请求重置（残留 pair 会被新请求的 q 统计覆盖）；
#   ③ L1 粗筛（k_min/k_max）与 L2 细筛（k_qat）用同一静态 pair 子空间；
#   ④ E64 分区口径下 mask 预算恒 K2=1024（与 B7/E72 干跑同口径）。
# 用法：python exp/trace/test_e85f_static_pair.py
import argparse
import sys

import torch

sys.path.insert(0, "/home/wangyuanshuo02/two-level-attention")
from sparse_attn.arguments import add_sparse_attn_args
from sparse_attn.indexer.tli_indexer import TLIIndexer

S, HKV, H, D = 16384, 8, 32, 128


def make_args(**kw):
    p = argparse.ArgumentParser()
    add_sparse_attn_args(p)
    a = p.parse_args(["--method", "tli", "--tli_subspace", "tail",
                      "--tli_alpha", "0.125", "--tli_beta", "0.375",
                      "--tli_gamma", "0.125",
                      "--tli_enable_layer_skip", "false"])
    for k2, v in kw.items():
        setattr(a, k2, v)
    return a


def fake_prefill_q(strong_pairs):
    """post-RoPE q [B,H,S,D]，指定 pair j 的两维幅值大 → 期望入选 top-16。"""
    q = torch.randn(1, H, 300, D) * 0.1
    for j in strong_pairs:
        q[..., j] += 10.0
        q[..., j + 64] += 10.0
    return q


def main():
    torch.manual_seed(0)
    fails = []

    # ---- ①② 静态 pair 选取 + 跨请求重置 ----
    idx = TLIIndexer(make_args(tli_static_pair=True))
    idx.layer_idx = 1
    strong = [3, 17, 40, 55]
    idx.observe_prefill_q(fake_prefill_q(strong))
    if idx._pair_idx is None:
        fails.append("① _pair_idx 未生成")
    else:
        pi = idx._pair_idx.tolist()
        assert len(pi) == 32 and pi == sorted(pi), f"① pair 长度/排序错: {pi}"
        first = set(j for j in pi if j < 64)
        if len(first) != 16 or any(j + 64 not in pi for j in first):
            fails.append(f"① 配对完整性破坏: {sorted(first)}")
        for j in strong:
            if j not in first:
                fails.append(f"① 受控强 pair {j} 未入选: {sorted(first)}")
        print(f"① 静态 pair（频率 j）: {sorted(first)}  含受控 {strong} ✓")

    # 跨请求重置：clear 后旧 pair 消失，新 q 统计生效
    idx.clear()
    if idx._pair_idx is not None:
        fails.append("② clear() 未重置 _pair_idx")
    strong2 = [0, 63]
    idx.observe_prefill_q(fake_prefill_q(strong2))
    first2 = set(j for j in idx._pair_idx.tolist() if j < 64)
    for j in strong2:
        if j not in first2:
            fails.append(f"② 重选后受控 pair {j} 未入选: {sorted(first2)}")
    print(f"② clear→重选 pair: {sorted(first2)} ✓")

    # ---- ③ L1/L2 同子空间 ----
    k = torch.randn(1, S, HKV, D)
    cu = torch.tensor([0, S])
    idxd = idx.prepare_index(k, cu)
    n_sub = idxd["k_min"].shape[-1]
    if n_sub != 32:
        fails.append(f"③ k_min 维度 {n_sub} != 32")
    # k_qat 非零维 == _pair_idx（量化后残差可能非零？零维乘 scale 减 zeros 仍 0）
    nz = (idxd["k_qat"][0, 100] != 0).any(dim=0).nonzero().flatten().tolist()
    if sorted(nz) != idx._pair_idx.tolist():
        fails.append(f"③ k_qat 非零维 {sorted(nz)[:8]}... != pair idx")
    print(f"③ L1 k_min d={n_sub}, L2 k_qat 非零维=pair idx ✓")

    # ---- ④ 预算恒 1024 ----
    q_dec = torch.randn(1, 1, H, D)
    sd = idx.compute_score(q_dec, torch.tensor([S - 1]), idxd, D ** -0.5)
    mask = idx.compute_mask(torch.tensor([S - 1]), sd)
    sums = mask.sum(-1).flatten().tolist()
    if any(abs(s - 1024) > 1e-6 for s in sums):
        fails.append(f"④ mask 预算 {set(sums)} != 1024")
    print(f"④ mask 预算 per kv-head = {set(int(round(s)) for s in sums)} ✓")

    # ---- ⑤ tail 臂回归：不开 static_pair 时 idx_sub 与旧硬编码逐位一致 ----
    idx_t = TLIIndexer(make_args(tli_static_pair=False))
    old = torch.tensor(list(range(32, 64)) + list(range(96, 128)))
    new = idx_t._subspace_indices(torch.device("cpu"))
    if not torch.equal(new, old):
        fails.append("⑤ tail 臂 idx_sub 与旧硬编码不一致（回归！）")
    print("⑤ tail 臂回归逐位一致 ✓")

    print("\n" + ("ALL PASS" if not fails else "FAIL:\n" + "\n".join(fails)))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
