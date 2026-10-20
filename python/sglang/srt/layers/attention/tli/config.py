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
        # ---- M8：批量 L2 fused gather+dequant+GEMV（select_decode_batched 的
        # P5 瓶颈，87.6%@bs32/131K：eager 物化 kq_c fp32 2.1GB×2 + 逐元素 flat
        # gather ~240GB/s → kernel 寄存器内反量化+每 token 256B 连续段 gather，
        # 20.5→0.48ms（43×，有效带宽 1.55TB/s）；CHUNK>1024 会寄存器溢出反而
        # 变慢（CHUNK=1024 实测 3.2ms），对拍 s2 max diff 2.4e-07）----
        self.use_l2_batched_kernel: bool = _env_bool("SGLANG_TLI_L2B_KERNEL", True)
        # ---- M8-KernelD：批量 L1 fused gather+GEMV（P1 行 gather 262μs +
        # P2 einsum permute 拷贝 ~346μs → 单 kernel 直读 pool）----
        self.use_l1_batched_kernel: bool = _env_bool("SGLANG_TLI_L1B_KERNEL", True)
        # ---- M8-TC：L1 批量打分 Tensor Core 化（tl.dot tf32 MMA 替代广播
        # mul+sum；DP≥16 才生效否则静默回退广播版。A/B 后定默认值）----
        self.use_l1_tc_kernel: bool = _env_bool("SGLANG_TLI_L1TC_KERNEL", False)
        # ---- M8-KernelC：双池直写（far/near -inf 烘进 KernelA 写出口径，
        # 消除 P6 masked_fill 链与 P7 的 far_sc/near_sc 物化；关闭可回退
        #   s2 单输出路径做 A/B 对拍）----
        self.use_l2_dual_kernel: bool = _env_bool("SGLANG_TLI_L2D_KERNEL", True)
        # ---- M8-topk：near 池压缩直写（near topk 输入 98.6% 为 -inf——
        # near 有限项仅 ~920/65728；静态上界 WNCAP=sink+(near_len-sw)=2048，
        # topk 宽度 30×↓；slot 确定性保 CUDA graph 逐位一致）----
        self.use_near_compact: bool = _env_bool("SGLANG_TLI_NEAR_COMPACT", True)
        # ---- M8：候选压实块展开 kernel（P4：topk-min 全排序 0.88ms →
        # cumsum+块展开 ~0.3-0.5ms；哨兵可在中段，下游 valid 掩掉，有效集一致）----
        self.use_compact_kernel: bool = _env_bool("SGLANG_TLI_COMPACT_KERNEL", True)
        # ---- M10：prefill select_batched 慢路径 kernel 化（M8 decode 侧全套
        # 移植：tli_compact 候选压实 + 双池直写 + 静态宽度配额 topk）。30K e2e
        # 归因：慢路径（S>nblk 阈值后快路径失效）占 prefill ~100%，1042ms/
        # 调用@末chunk。【C3 修复（kimi3 清单 S8，2026-10-08）】输出哨兵=S
        # （B01 后语义：下游 _sparse_extend_one 以 valid = sel < S 逐槽
        # -inf 屏蔽；原注释「哨兵转 0」是 B01 前过期口径——转 0 会使
        # token 0 被重复计权。对拍口径=有效集一致）----
        self.use_prefill_kernel: bool = _env_bool("SGLANG_TLI_PREFILL_KERNEL", True)
        # M11-decode：_sparse_attn_batched 走 fused kernel（哨兵=per-lane
        # valid 掩码，与 eager 语义一致）。2026-09-27 默认开：graph replay
        # 与 eager 逐字一致（333 chars）+ E5b PK 分诊 45/48 逐字 + decode
        # e2e AB 1.19-1.53×@bs8-32；旧行为可 SGLANG_TLI_SPARSE_KERNEL=0 回退
        self.use_sparse_attn_kernel: bool = _env_bool("SGLANG_TLI_SPARSE_KERNEL", True)
        # #64：select_batched 三处 torch.topk（占 select CUDA 62%）换
        # DeepSeek 官方 DeepSelect topk kernel（microbench 4.4-10.2×）
        self.use_ds_topk: bool = _env_bool("SGLANG_TLI_DS_TOPK", False)
        # 共享 index pool 初始行数（请求行数不足时自动扩）
        self.pool_rows: int = _env_int("SGLANG_TLI_POOL_R", 32)
        # 短序列退 dense
        self.dense_threshold: int = _env_int("SGLANG_TLI_DENSE_THRESHOLD", 2048)
        # ---- P2（#129）：forward_extend 索引构建侧流（SGLANG_TLI_SIDE_STREAM，
        # 默认开）。增量/全量 build 提交到 persistent side stream，主流继续
        # dense 计算，select 前 event wait（Ov-3(a) 的单 forward 内子集；
        # 跨 forward 的双流乒乓见 overlap_kernel_design.md，未实现）。关 =
        # 原同步执行（对拍口径）。decode / M5 CUDA graph 路径零接触。----
        self.use_side_stream: bool = _env_bool("SGLANG_TLI_SIDE_STREAM", True)
        # ---- #58 消融开关：L2 打分的 q 侧 GQA 聚合方式（sum | max）。
        # max = 逐 q-head 打分取组内 max。注：G-sum 符号冲突曾被怀疑为
        # 30B 崩坏根因，终局诊断（#58）证实真根因是 select_batched 早期
        # 行因果越界，与本聚合无关；留作打分质量消融口径。----
        self.q_agg: str = os.environ.get("SGLANG_TLI_Q_AGG", "sum")
        # ---- E112（#149）：α/β/γ 分区参数化 + far/near method 组合 ----
        # 口径对齐 TASK.md（/home/wangyuanshuo02/sglang/TASK.md L155-239）与
        # two-level-attention 权威实现（sparse_attn/indexer/tli_indexer.py，
        # two-level-indexer 分支）。五个环境变量**任一显式出现**即进入
        # taskmd 模式（α/β/γ 分区 + method 分池打分 + sink/swa 正交语义，
        # M8/M10 kernel 路径旁路）；全部缺省时保持旧 B' 语义逐位不变
        # （回归保护：已落袋的 e2e 主表数据零扰动）。
        # 语义（与 TASK.md 一致）：
        #   mid_L = T - sink_tok - swa_tok；near_L = max(bs, α·mid_L)
        #   nb_near = max(1, round(k1·β))；nb_far = k1 - nb_near
        #   K2_mid = K2 - sink_tok - swa_tok（sink/swa 正交，不占预算）
        #   nt_near = min(int(nb_near·bs·γ), K2_mid)；far_budget = K2_mid - nt_near
        #   e64_partition = α>0 且 β>0（(α,β)=(0,0) 为单池退化点，
        #   far_method 独占全管线；γ 严格语义无保底）
        _taskmd_envs = (
            "SGLANG_TLI_ALPHA", "SGLANG_TLI_BETA", "SGLANG_TLI_GAMMA",
            "SGLANG_TLI_FAR_METHOD", "SGLANG_TLI_NEAR_METHOD",
            # E155d 几何 env（nb/nt）同入 taskmd 检测——几何选择走
            # taskmd 专用路径；NEAR_LEN 不入（旧 B' 字段，单独显式
            # 设置是旧路径合法用法，不触发 taskmd）
            "SGLANG_TLI_NEAR_PAGES", "SGLANG_TLI_NEAR_TOKENS",
        )
        self.taskmd: bool = any(k in os.environ for k in _taskmd_envs)
        # γ 三态（对齐 HF tli_indexer L357-364）："off" = 自由竞争
        # （L2 取消 near/far 配额分割，单池竞争；L1 双池照旧）；
        # 未传缺省 1.0；数值 = 配额折扣。
        _g_raw = os.environ.get("SGLANG_TLI_GAMMA")
        if _g_raw is not None and _g_raw.strip().lower() == "off":
            self.gamma: float | None = None
        else:
            self.gamma = _env_float("SGLANG_TLI_GAMMA", 1.0)
        self.alpha: float = _env_float("SGLANG_TLI_ALPHA", 0.0)
        self.beta: float = _env_float("SGLANG_TLI_BETA", 0.0)
        # method 默认值 = mavg 冠军口径（far=minmax 粗筛 + near=avg 粗筛）
        self.far_method: str = os.environ.get("SGLANG_TLI_FAR_METHOD", "minmax")
        self.near_method: str = os.environ.get("SGLANG_TLI_NEAR_METHOD", "avg")
        assert self.far_method in ("avg", "minmax"), (
            f"SGLANG_TLI_FAR_METHOD 须为 avg|minmax，当前 {self.far_method!r}"
        )
        assert self.near_method in ("avg", "minmax"), (
            f"SGLANG_TLI_NEAR_METHOD 须为 avg|minmax，当前 {self.near_method!r}"
        )
        # ---- E155d 固定配额几何（2026-10-20 sglang 移植；口径对齐
        # two-level 权威实现 sparse_attn/indexer/tli_indexer.py L367-444）。
        # 用户 2026-10-19 拍板：全面转向 nt/nb 绝对配额。语义映射（geo
        # 显式优先，legacy 回退见括号）：
        #   nb = NEAR_PAGES（legacy = max(1, round(k1·β))）
        #   nt = NEAR_TOKENS，"off" = 自由竞争哨兵 ≡ γ-off（legacy =
        #        min(nb·bs·γ, K2_mid)）；nt=off 与数值 nt=K2_mid 语义
        #        严格分开（后者 = near 独吞 far=0 退化臂）
        #   nl = NEAR_LEN（legacy = max(bs, α·mid_len)）
        #   far 侧均为减法：nb_far = K1−nb；far_budget = K2_mid−nt。
        # fail-closed：nb/nt 任一显式 + α/β/γ 任一显式 → 构造期拒绝
        # （两口径混配即歧义配置，含 α0β0 单池意图 + 几何的事故面；
        # 单池派法 = 只传 α0β0，不传任何几何参数）。
        # 【与 HF 的一处已注记差异】nl-only（只设 NEAR_LEN）在 sglang
        # 不拒绝——NEAR_LEN 是旧 B' 既有 env（default 2048），单独显式
        # 设置是旧路径合法用法；geo 检测只看 nb/nt 两个 env。
        _nt_raw = os.environ.get("SGLANG_TLI_NEAR_TOKENS")
        _nt_off = _nt_raw is not None and _nt_raw.strip().lower() == "off"
        self.nt_near_pages: int | None = (
            int(os.environ["SGLANG_TLI_NEAR_PAGES"])
            if "SGLANG_TLI_NEAR_PAGES" in os.environ else None
        )
        self.nt_near_tokens: int | None = (
            None if (_nt_raw is None or _nt_off) else int(_nt_raw)
        )
        self.nt_near_len: int | None = None
        _geo = (self.nt_near_pages is not None or _nt_raw is not None)
        if _geo:
            if any(k in os.environ for k in
                   ("SGLANG_TLI_ALPHA", "SGLANG_TLI_BETA", "SGLANG_TLI_GAMMA")):
                raise ValueError(
                    "SGLANG_TLI_NEAR_PAGES/TOKENS（固定配额几何）与 "
                    "SGLANG_TLI_ALPHA/BETA/GAMMA（legacy 比例口径）不可同传"
                    "——两口径混配即歧义配置（E155d 参数退役，fail-closed；"
                    "单池派法 = 只传 ALPHA=0/BETA=0，不传任何几何参数）"
                )
            # 几何齐备性：nb 必传；nt 传数值或 off；nl 必传（几何下
            # α/β/γ 禁传 → near_L 无比例回退）。
            if self.nt_near_pages is None:
                raise ValueError(
                    "固定配额几何须显式传 SGLANG_TLI_NEAR_PAGES（配 "
                    "SGLANG_TLI_NEAR_TOKENS 数值或 off；SGLANG_TLI_NEAR_LEN "
                    "亦须显式——几何下无比例回退）")
            if self.nt_near_tokens is None and not _nt_off:
                raise ValueError(
                    "SGLANG_TLI_NEAR_TOKENS 须传非负整数或 'off'（自由竞争"
                    "哨兵）——只传 NEAR_PAGES 而缺 nt 属歧义配置")
            if "SGLANG_TLI_NEAR_LEN" not in os.environ:
                raise ValueError(
                    "固定配额几何须显式传 SGLANG_TLI_NEAR_LEN（E155d：几何"
                    "派单下 α/β/γ 禁传，near 区宽无比例规则回退）")
            self.nt_near_len = int(os.environ["SGLANG_TLI_NEAR_LEN"])
            if _nt_off:
                # nt=off 归一：γ → 自由竞争（消费端与 γ-off 同一分支）；
                # nb/nl 显式覆盖照常生效。
                self.gamma = None
            # 激活分区：几何显式时 nb/nt/nl 三处消费点均显式优先；
            # α/β 置 1 使 e64_partition（α>0 且 β>0）成立。
            self.alpha = 1.0
            self.beta = 1.0
            # 构造期几何约束（用户 2026-10-19 三维网格）：nb·bs ≤ nl；
            # nt ≤ nb·bs；far = K2−sink−swa−nt ≥ 0。
            _bs_c, _k2_c = self.block_size, self.token_budget
            _nb_c, _nl_c = int(self.nt_near_pages), int(self.nt_near_len)
            if _nb_c < 0:
                raise ValueError(f"SGLANG_TLI_NEAR_PAGES 须非负，收到 {_nb_c}")
            if _nb_c * _bs_c > _nl_c:
                raise ValueError(
                    f"网格约束破坏：nb·bs ≤ nl（nb={_nb_c}·bs={_bs_c}="
                    f"{_nb_c * _bs_c} > nl={_nl_c}）——near 页配额超出 "
                    "near 区宽度能供给的页数")
            if not _nt_off:
                _nt_c = int(self.nt_near_tokens)
                if _nt_c < 0:
                    raise ValueError(
                        f"SGLANG_TLI_NEAR_TOKENS 须非负，收到 {_nt_c}")
                if _nt_c > _nb_c * _bs_c:
                    raise ValueError(
                        f"网格约束破坏：nt ≤ nb·bs（nt={_nt_c} > nb={_nb_c}·"
                        f"bs={_bs_c}={_nb_c * _bs_c}）——near token 配额超出 "
                        "near 页几何容量")
                _far_c = (_k2_c - self.sink_blocks * _bs_c
                          - self.sliding_window - _nt_c)
                if _far_c < 0:
                    raise ValueError(
                        f"网格约束破坏：far = K2−sink−swa−nt = {_far_c} < 0"
                        f"（K2={_k2_c}, sink={self.sink_blocks * _bs_c}, "
                        f"swa={self.sliding_window}, nt={_nt_c}）")
        # ---- #60 D' 升级：prefill 动态测层 → decode 动态跳 far。
        # 离线验证（e60_prefill_dynamic_gate.json，32B 7 任务）：prefill
        # 末段行 per-layer far mass 与 decode far mass corr 0.86-0.99，
        # 无静态掩码跨任务泛化假设（E5b 已证静态 gate No-Go）。
        # prefill 末 chunk 统计 per-layer far mass 存 indexer，
        # decode 侧 select 按阈值置 skip_far。阈值口径 = per-layer
        # far mass（行×Hkv 平均；8B 实测安全任务 ~0.001-0.008、
        # 多跳 ~0.015-0.026）。----
        self.dyn_far_gate: bool = _env_bool("SGLANG_TLI_DYN_GATE", False)
        self.dyn_far_thresh: float = _env_float("SGLANG_TLI_DYN_GATE_THRESH", 0.01)

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
