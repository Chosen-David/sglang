# M10 对拍：select_batched 慢路径 kernel 化（M8 移植）vs 原版 eager 慢路径。
# 判据：候选充足行（t 大、far/near 池均有足够候选）位置集合逐行 head 级一致
# （两边均无垃圾位）；empty/短行（首 chunk）冒烟（形状正确、值域合法）。
# 双口径：合成随机 K + 真实 trace K（narrativeqa layer03，S=32K far-heavy）。
import os
import sys

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")

import torch

os.environ.setdefault("SGLANG_TLI_PREFILL_KERNEL", "1")

from sglang.srt.layers.attention.tli.config import TLIProfile
from sglang.srt.layers.attention.tli.indexer import TLIIndexer

dev = "cuda:0"
torch.manual_seed(0)

S, Hkv, D, G = 30720, 8, 128, 4
H = Hkv * G
NQ = 512  # 8 个 row_chunk（t 混合：前段 empty/短行 + 末段候选充足行）


def build_indexer():
    os.environ["SGLANG_TLI_PREFILL_KERNEL"] = "1"
    p1 = TLIProfile()
    os.environ["SGLANG_TLI_PREFILL_KERNEL"] = "0"
    p0 = TLIProfile()
    os.environ["SGLANG_TLI_PREFILL_KERNEL"] = "1"
    return TLIIndexer(p1, head_dim=D).to(dev), TLIIndexer(p0, head_dim=D).to(dev)


def run_case(name, K):
    idx_k, idx_e = build_indexer()
    index = idx_k.build_block_index(K)
    # 两 indexer 共享同一 index dict（build 确定性，直接复用）
    q = torch.randn(NQ, H, D, device=dev) * 0.3
    # t 混合：1/4 首段（t 小，empty/短 far），3/4 末段（t 大，候选充足）
    t_arr = torch.cat(
        [
            torch.arange(0, NQ // 4, device=dev),
            torch.arange(S - 3 * NQ // 4, S, device=dev),
        ]
    )
    sel_k = idx_k.select_batched(index, q, t_arr)
    sel_e = idx_e.select_batched(index, q, t_arr)
    torch.cuda.synchronize()
    assert sel_k.shape == sel_e.shape, f"{name}: 形状 {sel_k.shape} vs {sel_e.shape}"
    # 逐行逐 head 集合比对（multiset；判据分档）：
    # - 候选充足行：对称差 ≤2 容忍（KernelC GEMV 归约顺序 vs eager einsum 的
    #   1e-7 级尾差在 topk 第 k 名边界翻转 tie——已验证实例分数差 2.4e-07，
    #   M8 decode 对拍同款现象；统计口径 mismatch 行对占比须 <2%）
    # - 短行/empty：有效集（>0）一致 + 值域合法 [0,S)
    import collections

    n_pair = 0
    n_mismatch = 0
    for i in range(NQ):
        big_t = bool(t_arr[i] > 4096)  # 候选充足行（远端池 > far_tokens）
        for h in range(Hkv):
            a = collections.Counter(sel_k[i, h].tolist())
            b = collections.Counter(sel_e[i, h].tolist())
            n_pair += 1
            if big_t:
                sym_diff = sum((a - b).values()) + sum((b - a).values())
                if sym_diff > 2:
                    n_mismatch += 1
            else:
                sa = set(x for x in a if x > 0)
                sb = set(x for x in b if x > 0)
                if sa != sb or sel_k[i, h].min() < 0 or sel_k[i, h].max() >= S:
                    n_mismatch += 1
    rate = n_mismatch / n_pair
    status = "PASS" if n_mismatch == 0 else f"FAIL ({n_mismatch}/{n_pair})"
    print(f"[{name}] 对拍 {status}")
    return n_mismatch == 0


ok = True
# 合成 K（慢路径 128/128，候选充足行 far 池随机分数）
ok &= run_case("合成K 30K", torch.randn(S, Hkv, D, device=dev) * 0.3)
# 真实 K（narrativeqa layer03 far-heavy）
d = torch.load(
    "/tmp/trace/qwen3-8b/lb_narrativeqa_0/layer03.pt",
    map_location="cpu",
    weights_only=False,
)
Kr = d["k"].to(dev).float()
ok &= run_case("真实K narrativeqa L03", Kr)
print("M10 select_batched kernel 化对拍:", "ALL PASS" if ok else "FAILED")
sys.exit(0 if ok else 1)
