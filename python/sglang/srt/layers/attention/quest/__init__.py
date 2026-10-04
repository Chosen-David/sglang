"""Quest: page 级全维 min/max 上界稀疏 attention（审稿 C3 e2e 对照臂）。

算法口径对齐 two-level-attention quest_indexer.py（page 64 × topk 16 =
1024 token 预算，与 TLI@1024 同预算）。改造基底 = TLI backend（paged
寻址 / 共享 index pool / fused attention 消费端复用）。
"""

from sglang.srt.layers.attention.quest.config import QuestProfile
from sglang.srt.layers.attention.quest.indexer import QuestIndexer

__all__ = ["QuestProfile", "QuestIndexer", "QuestSparseAttnBackend"]


def __getattr__(name):
    if name == "QuestSparseAttnBackend":
        from sglang.srt.layers.attention.quest.backend import QuestSparseAttnBackend

        return QuestSparseAttnBackend
    raise AttributeError(name)
