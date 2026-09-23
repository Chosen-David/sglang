"""TLI (Two-Level Indexer) profile parsing.

继承 TIA（本组第一代）语义并叠加 proposal 三个实测 Go 的创新点：
  A: position-stable subspace 粗筛（Qwen3 rotate_half 低频尾维, d'=32）
  B: 远端 kmeans 聚类代表（可选, 默认走 TIA 的块 min/max 上界）
  D': 层自适应级联跳过（离线校准的静态层掩码）

配置来源：server_args 上的 tli_* 字段（prototype 阶段由环境变量
SGLANG_TLI_* 覆盖，便于不重写 arg parser 先跑通）。
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


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


class TLIProfile:
    """两级索引器配置（默认值全部对齐 2026-09-23 trace 实测的 Go 配置）。"""

    def __init__(self) -> None:
        # ---- Level-1（创新点 A：子空间块上界粗筛）----
        self.block_size: int = _env_int("SGLANG_TLI_BLOCK_SIZE", 64)
        self.coarse_dim: int = _env_int("SGLANG_TLI_COARSE_DIM", 32)  # d'
        self.k1_blocks: int = _env_int("SGLANG_TLI_K1_BLOCKS", 128)
        # ---- Level-2（4bit 部分维 token 精筛，TIA 语义）----
        self.delta: int = _env_int("SGLANG_TLI_DELTA", 16)  # 每半取的维数
        self.token_budget: int = _env_int("SGLANG_TLI_TOKEN_BUDGET", 1024)  # K2
        self.sliding_window: int = _env_int("SGLANG_TLI_SLIDING_WINDOW", 128)
        self.sliding_blocks: int = _env_int("SGLANG_TLI_SLIDING_BLOCKS", 3)
        # ---- 创新点 B：远端 kmeans 聚类代表 ----
        self.far_kmeans: bool = _env_bool("SGLANG_TLI_FAR_KMEANS", False)
        self.far_clusters: int = _env_int("SGLANG_TLI_FAR_CLUSTERS", 256)
        # ---- 创新点 D'：层自适应跳过（离线校准掩码文件路径）----
        self.layer_skip_path: str | None = os.environ.get("SGLANG_TLI_LAYER_SKIP")
        # 短序列退 dense
        self.dense_threshold: int = _env_int("SGLANG_TLI_DENSE_THRESHOLD", 2048)

    def subspace_idx(self, head_dim: int) -> list[int]:
        """position-stable 子空间维度索引（Qwen3 rotate_half 两半的尾维）。

        head_dim=128, coarse_dim=32 → [48..63] + [112..127]（低频维）。
        """
        assert head_dim % 2 == 0, "rotate_half 布局要求偶数 head_dim"
        half = head_dim // 2
        d_half = self.coarse_dim // 2
        assert 0 < d_half <= half, f"coarse_dim={self.coarse_dim} 超出 head_dim={head_dim}"
        return list(range(half - d_half, half)) + list(range(head_dim - d_half, head_dim))

    def refine_idx(self, head_dim: int) -> list[int]:
        """Level-2 精筛维度（每半 delta 维，总 2*delta；TIA 语义）。"""
        half = head_dim // 2
        assert 0 < self.delta <= half
        return list(range(half - self.delta, half)) + list(
            range(head_dim - self.delta, head_dim)
        )

    def load_layer_skip(self, n_layers: int) -> list[bool] | None:
        """D' 静态层掩码：JSON 文件 {"skip": [layer_idx,...]}。"""
        if not self.layer_skip_path:
            return None
        import json

        with open(self.layer_skip_path) as f:
            skip_ids = set(json.load(f)["skip"])
        return [i in skip_ids for i in range(n_layers)]


__all__ = ["TLIProfile"]
