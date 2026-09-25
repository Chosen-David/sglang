"""TLI (Two-Level Indexer) profile parsing.

继承 TIA（本组第一代）语义并叠加三个实测支撑的创新点：
  A: position-stable subspace 粗筛（Qwen3 rotate_half 低频尾维, d'=32）
  B': far/near 分区 L2 预算（E4c 修正：聚类代表降级为消融，far 区走 4bit 精筛）
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
        # ---- 创新点 B'：far/near 分区 L2 预算（E4c 修正后的语义）----
        # E4c 严格预算实测：聚类代表（km_blk 0.09–0.39 / km_tok 0.45–0.79）
        # 无一致优势，far 区保留 4bit token 级精筛（≈ oracle）；B' 的贡献是
        # far 池独立预算防挤出（far-heavy 层 TLI 反超 TIA）
        self.far_tokens: int = _env_int("SGLANG_TLI_FAR_TOKENS", 256)  # K2_far（128 即饱和）
        self.far_select: str = os.environ.get("SGLANG_TLI_FAR_SELECT", "4bit")  # 4bit | cluster（消融）
        self.near_len: int = _env_int("SGLANG_TLI_NEAR_LEN", 2048)
        self.sink_blocks: int = _env_int("SGLANG_TLI_SINK_BLOCKS", 2)
        # ---- 聚类代表（已降级为消融，仅 far_select=cluster 时构建）----
        self.far_kmeans: bool = _env_bool("SGLANG_TLI_FAR_KMEANS", False)
        self.far_clusters: int = _env_int("SGLANG_TLI_FAR_CLUSTERS", 256)
        # ---- 创新点 D'：层自适应跳过（离线校准掩码文件路径）----
        self.layer_skip_path: str | None = os.environ.get("SGLANG_TLI_LAYER_SKIP")
        # ---- kernel 化（E8-2：fused L1 单 launch，跳层 3.6×/非跳层 1.6×；
        # 并列截断多选块由 L2 精筛淘汰，trace 对拍 cov 一致）----
        self.use_l1_kernel: bool = _env_bool("SGLANG_TLI_L1_KERNEL", False)
        # ---- L2 级联 fused（M3-b 接入：单 launch/head 分区精筛，原型 1.63×）----
        self.use_l2_kernel: bool = _env_bool("SGLANG_TLI_L2_KERNEL", False)
        # ---- M9：L2 精筛 PCA 投影降维（离线校准基，替代 refine_idx 维度选择）----
        # 实验依据（test_tli_proj_sweep/explore，2026-09-25）：PCA16 0.532 ≈
        # 选择32 0.557（打分维砍半）、4bit 投影仅 −0.022、跨任务基迁移 −0.024、
        # 2048 token 小校准集=全量同值；随机投影崩溃（JL 不保 GQA 点积排序）
        # → 必须用 K 协方差（per-layer per-head SVD top-r）。
        # basis 文件 = .pt，fp32 [n_layers, Hkv, D, r]（离线脚本 calibrate_pca_basis.py）
        self.proj_basis_path: str | None = os.environ.get("SGLANG_TLI_PROJ_BASIS")
        self.proj_rank: int = _env_int("SGLANG_TLI_PROJ_R", 16)
        # ---- M4 批量化 decode：n≥2 走共享 pool + 批量 eager select ----
        # （关闭可回退 per-request 路径做 A/B 对拍；n==1 恒走 per-request
        #   以保留 L1/L2 fused kernel 的 bs=1 延迟优势）
        self.use_batch_select: bool = _env_bool("SGLANG_TLI_BATCH_SELECT", True)
        # 共享 index pool 初始行数（请求行数不足时自动扩）
        self.pool_rows: int = _env_int("SGLANG_TLI_POOL_R", 32)
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

    def refine_nd(self) -> int:
        """L2 精筛表示维度：投影基存在时 = r（PCA），否则 = 2*delta（选择）。"""
        return self.proj_rank if self.proj_basis_path else 2 * self.delta

    def load_layer_skip(self, n_layers: int) -> list[bool] | None:
        """D' 静态层掩码：JSON 文件 {"skip": [layer_idx,...]}。"""
        if not self.layer_skip_path:
            return None
        import json

        with open(self.layer_skip_path) as f:
            skip_ids = set(json.load(f)["skip"])
        return [i in skip_ids for i in range(n_layers)]


__all__ = ["TLIProfile"]
