"""MoBA: training-free chunk-mean gate 稀疏 attention（E89 复现 e2e 臂）。

算法口径对齐 two-level-attention/sparse_attn/indexer/tli_indexer.py 的
moba_gate 路径（chunk 64 × top-16 块 = 1024 token 预算 + sink/swa 保送，
与 TLI@1024 同预算）。改造基底 = Quest backend（paged 寻址 / 共享
index pool / fused attention 消费端复用）。
"""

from sglang.srt.layers.attention.moba.config import MoBAProfile
from sglang.srt.layers.attention.moba.indexer import MoBAIndexer

__all__ = ["MoBAProfile", "MoBAIndexer", "MoBASparseAttnBackend"]


def __getattr__(name):
    if name == "MoBASparseAttnBackend":
        from sglang.srt.layers.attention.moba.backend import MoBASparseAttnBackend

        return MoBASparseAttnBackend
    raise AttributeError(name)
