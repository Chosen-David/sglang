"""MoBA profile 配置（E89 复现臂，e2e 延迟对比用）。

口径声明（必须随代码保留）：
  - 原版 MoBA（Moonshot AI, "Mixture of Block Attention"）的 gate 是
    **训练得到**的（MoBA 层与 LM 联合训练）；
  - 本复现是 **training-free chunk-mean gate**：gate 分数 = 全维（D=128）
    q · chunk 内 key 均值（GQA 组内 mean 到 kv-head 级），无训练参数。
  - 质量侧已验证：E89（two-level-attention 仓 transformers monkeypatch 臂，
    --tli_moba）LongBench 13 任务 49.19。

算法口径（对齐 two-level-attention/sparse_attn/indexer/tli_indexer.py 的
moba 路径：L60 moba_gate 开关 / L224 k_avg_full / L356-367 score_moba /
L414-446 _moba_mask）：
  - chunk = 64 token（tia_block_size 默认），gate 用零 pad 后的 chunk-mean
    （尾块均值 = 真实 token 和 / 64，E89 pad_tensor 零填充口径）
  - 选 top-(K2/BS)=16 块全展开（无两级 / 无量化 / 无 L2 细筛 / 无分区）
  - sink（前 128 token = 2 块）+ swa（尾 128 token）保送，与 PSI 同口径，
    不占 gate 预算；块排名不排除 sink/swa 块（E89 原语义：全部块参与
    topk，sink/swa 由 mask 直接置位）
  - K2 = tia_level2_topk = 1024 → 16 块（E89 运行脚本
    exp/trace/run_scripts/e89_moba_smoke.sh 的实参）

配置来源：环境变量 SGLANG_MOBA_*（与 tli/quest prototype 同款覆盖方式）。
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


class MoBAProfile:
    """MoBA training-free 复现配置（默认 = E89 运行臂实参）。"""

    def __init__(self) -> None:
        # chunk（块）大小 = E89 tia_block_size=64（Moonshot MoBA 论文同值）
        self.chunk_size: int = _env_int("SGLANG_MOBA_CHUNK", 64)
        # token 预算 K2 = E89 tia_level2_topk=1024 → top-(K2/BS)=16 块
        self.token_budget: int = _env_int("SGLANG_MOBA_TOKEN_BUDGET", 1024)
        # sink 保送 token 数 = E89 sink_blocks(2) × bs(64) = 128
        self.sink_tokens: int = _env_int("SGLANG_MOBA_SINK", 128)
        # swa 保送 token 数 = E89 sliding_window_size=128（tia_indexer.py L16）
        self.sliding_window: int = _env_int("SGLANG_MOBA_SWA", 128)
        # 短序列退 dense（与 tli/quest harness 同口径，S≤2048 走 dense）
        self.dense_threshold: int = _env_int("SGLANG_MOBA_DENSE_THRESHOLD", 2048)
        # 共享 index pool 初始行数（graph 禁用版无捕获约束，可动态扩）
        self.pool_rows: int = _env_int("SGLANG_MOBA_POOL_R", 32)
        # pool S 容量显式封顶（0 = 跟随 req_to_token 全宽）
        self.pool_s_cap: int = _env_int("SGLANG_MOBA_POOL_S_CAP", 0)
        # 稀疏前向复用 TLI 的 fused gather+attn kernel（同 kernel = 公平
        # 对照：MoBA / Quest / PSI 三臂 attention 消费端同款，差异只在索引）
        self.use_sparse_attn_kernel: bool = _env_bool(
            "SGLANG_MOBA_SPARSE_KERNEL", True
        )

    @property
    def select_blocks(self) -> int:
        """gate top-K 块数 = K2 / chunk_size（E89 nb_sel = max(1, K2//bs)）。"""
        return max(1, self.token_budget // self.chunk_size)

    @property
    def sel_width(self) -> int:
        """选择输出静态总宽 = top-K 块展开 + sink + swa（默认 1024+128+128）。

        这是选中 token 数的上界（sink/swa 块与 top-K 块重叠时实际更少，
        去重逻辑见 indexer.select_decode_batched）。
        """
        return (
            self.select_blocks * self.chunk_size
            + self.sink_tokens
            + self.sliding_window
        )


__all__ = ["MoBAProfile"]
