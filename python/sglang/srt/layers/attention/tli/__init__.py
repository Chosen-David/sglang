"""TLI: Two-Level Indexer sparse attention (TIA 下一代, 创新点 A/B/D')。

实测支撑：two-level-attention/exp/trace/results/（2026-09-23, Qwen3-8B,
10 条真实 trace）：两级 pipeline mass coverage 0.9967–1.0000。
"""

from sglang.srt.layers.attention.tli.config import TLIProfile
from sglang.srt.layers.attention.tli.indexer import TLIIndexer

__all__ = ["TLIProfile", "TLIIndexer", "TLISparseAttnBackend"]


def __getattr__(name):
    if name == "TLISparseAttnBackend":
        from sglang.srt.layers.attention.tli.backend import TLISparseAttnBackend

        return TLISparseAttnBackend
    raise AttributeError(name)
