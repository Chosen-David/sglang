"""Quest attention backend（审稿 C3 修复：Quest e2e 对照臂）。

算法口径对齐 two-level-attention/sparse_attn/indexer/quest_indexer.py
（Quest 论文口径，page 64 × topk 16 = 1024 token 预算，与 TLI@1024 同预算）：
  - 索引：每 page=64 token 建全维（D=128）k_min/k_max 上界，无量化
    （Quest 论文口径 = 无量化全维 min/max，保持一致）
  - decode 选择：page score = sum_d max(q_d·k_min_d, q_d·k_max_d)（上界），
    GQA group-mean 对齐（kv head 内 G 个 q head 的上界分取平均）
  - top-16 page 选中后整页 64 token 全展开参与 attention；当前 query 所在
    page 强制入选（quest_indexer.py 的 mask[:, pos, :, q_id//bs]=True 语义）
  - sink/swa：Quest 原版无 sink 保送——仅当前页强制入选提供 64 token 局部
    性。保持原版行为（baseline 诚实口径，已记录供论文 Limitations 引用）

存储口径（论文对照点，如实记录算术）：
  - Quest 全维 min/max：2 × D × 4B = 1KB / page / kv-head（fp32 口径）；
    page=64 摊销后 16B/token/kv-head，Hkv=8 时 128B/token（fp32）/
    64B/token（bf16 存储，本实现：bound 是 KV 值本身的 min/max，
    用 KV 同 dtype 存储对 bound 精确无损）
  - TLI 对照：kq 4bit 40B + 粗筛界 4B = 44B/token/kv-head ≈ 336B/token
  - 注意：Quest 的 page 级索引本身比 TLI 的 token 级 4bit 缓存更省，
    但其索引粒度只到 page（上界更松、选择质量更低）——存储与精度
    两个维度分开报告，不做单数字对比

配置来源：环境变量 SGLANG_QUEST_*（与 TLI prototype 同款覆盖方式）。
"""

from __future__ import annotations

import os


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes")


class QuestProfile:
    """Quest 配置（默认 = page 64 × topk 16，与 two-level quest_64_16 臂一致）。"""

    def __init__(self) -> None:
        self.page_size: int = _env_int("SGLANG_QUEST_PAGE_SIZE", 64)
        self.topk_pages: int = _env_int("SGLANG_QUEST_TOPK_PAGES", 16)
        # 短序列退 dense（与 TLI harness 同口径，S≤2048 走 dense）
        self.dense_threshold: int = _env_int("SGLANG_QUEST_DENSE_THRESHOLD", 2048)
        # 共享 index pool 初始行数（graph 禁用版无捕获约束，可动态扩）
        self.pool_rows: int = _env_int("SGLANG_QUEST_POOL_R", 32)
        # pool S 容量显式封顶（0 = 跟随 req_to_token 全宽）
        self.pool_s_cap: int = _env_int("SGLANG_QUEST_POOL_S_CAP", 0)
        # decode 稀疏前向复用 TLI 的 fused gather+attn kernel（同 kernel =
        # 公平对照：Quest 与 TLI 的 attention 消费端同款实现，差异只在索引）
        self.use_sparse_attn_kernel: bool = _env_bool(
            "SGLANG_QUEST_SPARSE_KERNEL", True
        )

    @property
    def token_budget(self) -> int:
        """选择预算 = topk_pages × page_size（默认 16×64 = 1024，同 TLI@1024）。"""
        return self.topk_pages * self.page_size


__all__ = ["QuestProfile"]
