"""TLI (Two-Level Indexer)：TIA 下一代，组合三个实测 Go 的创新点。

创新点（全部有 2026-09-23 真实 trace 实测支撑，见 exp/trace/results/）：
  A  position-stable subspace L1：块 min/max 只在低频尾维 d'=32 上算
     （E3：d'=32 mass recall 0.729 ≈ 全维 0.732；random/highfreq 崩溃）
  B  far/near 分区 L2 预算：远端与近端独立 top-K2 池，防远端被近端高分挤出；
     远端用 TIA 4bit token 级精筛（E4c：≈oracle，L03 0.999）。原「远端 kmeans
     聚类代表」方案为 E4c negative result（E4b 有整簇超选 bug 虚高 4-10×，
     严格预算下 km_blk 0.09-0.39 / km_tok 0.45-0.79 均无一致优势），
     cluster 模式保留作消融（--tli_far_select cluster）
  D' 层自适应跳过：离线校准的静态层掩码，far 质量低的层把远端块从 L1 剔除
     （E6：precision 0.92-1.00，far 质量损失 <0.3%；在线信号版已否定）

继承 TIAIndexer，复用其 L2（4bit 部分维 token 精筛 + 滑窗强制）语义。
"""
import json
import os

import torch
import torch.nn.functional as F
from einops import rearrange, repeat, einsum

from .tia_indexer import TIAIndexer

DEFAULT_MASK = os.path.join(
    os.path.dirname(__file__), "..", "..", "exp", "trace", "results", "tli_layer_skip_mask.json"
)


class TLIIndexer(TIAIndexer):
    """A+B+D' 全开为 method='tli'；参数可单独关闭做消融。"""

    def __init__(self, args) -> None:
        super().__init__(args)
        self.layer_idx = None            # 由 register_patch 注入
        # ---- A：子空间维度选择（2026-09-29 用户指令：参数化 rope/nope/full，默认 full）----
        #   full = 全 128 维（默认）；rope = 前 64 旋转维；nope = 后 64 非旋转维；
        #   tail = 旧口径低频尾维 32（E71/B7/C0 主表口径，向后兼容保留）
        self.subspace = getattr(args, "tli_subspace", "full")
        assert self.subspace in ("full", "rope", "nope", "tail", "random", "highfreq"), \
            f"tli_subspace 非法取值: {self.subspace}"
        self.enable_subspace = getattr(args, "tli_enable_subspace", True)
        if self.subspace == "full":
            self.enable_subspace = False      # 全维（用户默认）
        elif self.subspace in ("rope", "nope"):
            self.enable_subspace = True       # 显式子空间段
        # ---- B：远端 kmeans（far_select='cluster' 时才构建，默认 4bit 分区）----
        self.enable_kmeans = getattr(args, "tli_enable_kmeans", True)
        self.far_select = getattr(args, "tli_far_select", "4bit")
        self.far_clusters = getattr(args, "tli_far_clusters", 256)
        self.far_niter = getattr(args, "tli_far_niter", 10)
        self.far_blocks = getattr(args, "tli_far_blocks", 16)   # 远端块预算 K1_far
        self.far_tokens = getattr(args, "tli_far_tokens", 512)  # 远端 token 预算 K2_far
        self.near_len = 2048                                      # 近端窗（D'/B 共用）
        # ---- E64j method 组合参数化（用户命名: mminmax/cavg/aavg 等）----
        self.far_method = getattr(args, "tli_far_method", "minmax")
        self.near_method = getattr(args, "tli_near_method", "avg")
        # ---- E110：ccluster / sim_greedy（用户 2026-10-07 定义，TASK.md「关于cluster的方法」节）----
        #   far_select/near_select 独立选择 L2 侧「簇代表打分」选择方式：
        #     ccluster     = (far=cluster,    near=cluster)
        #     cavg_sim     = (far=sim_greedy, near=4bit)  —— 与 cavg 只差聚类方式
        #     ccluster_sim = (far=sim_greedy, near=sim_greedy)
        #   默认双双 '4bit'（现状，回归保护：不开新 flag 行为逐位不变）。
        #   注意：cluster/sim_greedy 臂须 --tli_enable_kmeans true（沿用 cavg 惯例）。
        self.near_select = getattr(args, "tli_near_select", "4bit")
        assert self.near_select in ("4bit", "cluster", "sim_greedy"), \
            f"tli_near_select 非法取值: {self.near_select}"
        self.sim = getattr(args, "tli_sim", 0.9)
        self.sim_dims = getattr(args, "tli_sim_dims", "subspace")
        assert self.sim_dims in ("subspace", "nope", "tail", "full"), \
            f"tli_sim_dims 非法取值: {self.sim_dims}"
        # ---- E87：top-σ 选择（far/near/mid 侧不做两级，细筛分 ≥ sink max − σ 即选中）----
        self.sigma_select = getattr(args, "tli_sigma_select", "none")
        self.moba_gate = getattr(args, "tli_moba", False)
        self.sigma = getattr(args, "tli_sigma", 8.0)
        self.sink_blocks = 2                                      # sink = 前 2 块（128 tok）
        if self.near_select != "4bit":
            # E110：near 侧簇分路径与 σ/MoBA 均绕过/改写两级选择语义，显式互斥
            assert self.sigma_select == "none", \
                "--tli_near_select 与 --tli_sigma_select 互斥（σ 路径绕过两级选择）"
            assert not self.moba_gate, "--tli_near_select 与 --tli_moba 互斥"
        # ---- E103：kv-head 共享消融（审稿 MAJOR）----
        #   论文 §2.2 口径：L1/L2 分数先在 GQA 组内 mean 聚合到 kv-head 级再 topk
        #   （32 q-head → 8 kv-head，索引量省 4 倍）。本开关打开时旁路全部聚合点，
        #   per-q-head 分数直接进 topk（topk 沿最后一维，H 维天然独立），
        #   mask 从 [1,1,Hkv,T] 变 [1,1,H,T]（消费端 eager_decoding 按组拆回）。
        #   互斥守卫：cluster/moba/σ 路径的簇分数与 gate 均为 kv-head 级语义，
        #   与 per-q-head 组合无定义，显式禁止（E103 臂不经过这些路径）。
        self.per_q_head = getattr(args, "tli_per_q_head", False)
        if self.per_q_head:
            assert not self.moba_gate, "--tli_per_q_head 与 --tli_moba 互斥（gate 是 kv-head 级语义）"
            assert self.far_select == "4bit", \
                "--tli_per_q_head 与 --tli_far_select cluster/sim_greedy 互斥（簇分数是 kv-head 级语义）"
            assert self.near_select == "4bit", \
                "--tli_per_q_head 与 --tli_near_select cluster/sim_greedy 互斥（簇分数是 kv-head 级语义）"
            assert self.sigma_select == "none", \
                "--tli_per_q_head 与 --tli_sigma_select 互斥（E103 臂不经过 σ 路径）"
        # ---- D'：静态层掩码 ----
        self.enable_layer_skip = getattr(args, "tli_enable_layer_skip", True)
        self.skip_far = False
        mask_path = getattr(args, "tli_layer_skip_path", None) or DEFAULT_MASK
        self._skip_ids: set[int] | None = None
        if self.enable_layer_skip:
            try:
                with open(mask_path) as f:
                    self._skip_ids = set(json.load(f)["skip"])
            except (OSError, KeyError, ValueError):
                print(f"[TLI] 警告: 层掩码 {mask_path} 不可读, D' 关闭")
                self._skip_ids = None
        # B 的聚类缓存（prefill 后一次；decode 期 far 区不变）
        self._km_centroids = None    # [Hkv, K_c, d']
        self._km_token_assign = None  # [Hkv, Tfar]
        self._km_far_lo = None       # far 区起点（token idx）
        self._km_far_blk_lo = None   # far 区块起点
        self._km_blk_ids = None      # [Hkv, Tfar] 每个 far token 的块 id
        self._km_dims = None         # E110：far 侧建簇时实际用的维度索引（打分端对齐用）
        # E110：near 侧聚类缓存（ccluster；near 区左右缘 decode 期都会动 → 按 key 全量重建）
        self._km_near_centroids = None   # [Hkv, K_c, d']
        self._km_near_assign = None      # [Hkv, Tn]（相对 _km_near_lo 的 token 偏移）
        self._km_near_lo = -1            # near 簇覆盖区起点（token idx，块对齐）
        self._km_near_hi = -1            # near 簇覆盖区终点（token idx，块对齐）
        self._km_near_key = None         # (far_hi, near_hi) 重建 key
        self._km_near_dims = None        # near 侧建簇维度索引
        # E110：far 侧 sim_greedy 增量状态（贪心是时序过程，可对新增段精确续跑）
        self._km_greedy_sums = None      # [Hkv, Kcap, d']  簇成员和（簇心 = sums/cnt）
        self._km_greedy_cnt = None       # [Hkv, Kcap]
        self._km_greedy_sq = None        # [Hkv, Kcap]  ||sums||^2 增量维护（免每步全量 norm）
        self._km_greedy_klive = None     # [Hkv] 活簇数
        # ---- E64 框架参数化：α/β/γ 分区 + sup_wsvd 投影基（B 配置帕累托点）----
        self.alpha = getattr(args, "tli_alpha", 0.0)
        self.beta = getattr(args, "tli_beta", 0.0)
        self.gamma = getattr(args, "tli_gamma", 1.0)
        self._basis_all = None
        bp_path = getattr(args, "tli_proj_basis", None)
        if bp_path:
            if self.subspace != "tail":
                raise ValueError(
                    f"tli_proj_basis 定义在 tail-32 子空间上，与 tli_subspace={self.subspace} "
                    "不兼容（投影基的输入是 32 维特征）；请用 --tli_subspace tail"
                )
            self._basis_all = torch.load(bp_path, map_location="cpu", weights_only=True).float()
            self.enable_subspace = True  # 投影基定义在子空间内，强制开启
            print(f"[TLI] 投影基已加载 {bp_path} shape={list(self._basis_all.shape)}")
        self._basis = None   # 当前层子空间基 [Hkv, 32, r]（惰性提取于 prepare_index）
        # ---- E85f：per-layer 静态 pair 选取（e2e 判决）----
        #   E85e 重放：prefill 尾 256 q 平均 |q| 的 pair 幅值 |q̄_j|+|q̄_{j+64}|
        #   top-16 完整对，层固定、可索引化，recall 0.849 vs tail32 0.811。
        #   E85f 判决问题：该增益在 e2e（decode q 分布漂移下）是否保住
        #   （E66 投影基 trace 成立 e2e 崩的前车之鉴）。pair 未定前回退 tail32。
        self.static_pair = getattr(args, "tli_static_pair", False)
        self._pair_idx = None
        if self.static_pair:
            if self.subspace != "tail":
                raise ValueError(
                    f"--tli_static_pair 需 --tli_subspace tail（当前 {self.subspace}）；"
                    "静态 pair 是 tail32 的同宽度替换，非独立子空间段"
                )
            self.enable_subspace = True

    # ---------------- E85f：prefill q 统计 → per-layer 静态 pair ---------------- #

    def observe_prefill_q(self, q):
        """prefill 分支旁路采集（qwen3_attn_patch 调用）：post-RoPE q [B,H,S,D]。
        取尾 256 行全体 head 展平的平均 |q|，按 pair 幅值 |q̄_j|+|q̄_{j+64}|
        取 top-16 完整旋转对（与 E85e 重放口径一致）。只算一次（首个完整 prefill）。
        """
        if self._pair_idx is not None or not self.static_pair:
            return
        qm = q.detach().float()[:, :, -256:].reshape(-1, q.shape[-1]).abs().mean(0)  # [128]
        pair_mag = qm[:64] + qm[64:]                     # [64]（频率 j）
        top_pairs = torch.topk(pair_mag, 16).indices     # 频率 j
        self._pair_idx = torch.cat([top_pairs, top_pairs + 64]).sort().values.to(q.device)
        if os.environ.get("TLI_DEBUG") and self.layer_idx is not None:
            print(f"[E85f] L{self.layer_idx} static pairs (freq j): "
                  f"{sorted(top_pairs.tolist())}", flush=True)

    # ---------------- D' ---------------- #

    def _resolve_skip(self):
        if self._skip_ids is not None and self.layer_idx is not None:
            self.skip_far = self.layer_idx in self._skip_ids

    # ---------------- A：子空间 L1 ---------------- #

    def _subspace_indices(self, device):
        # 子空间选择（用户指令参数化）：rope=前 64 旋转维 / nope=后 64 非旋转维 /
        # tail=低频尾维（旧口径，cmp_ratio 控制宽度）/ full 不进此函数
        # E85f：静态 pair 已定（prefill q 统计）→ 用 per-layer 完整旋转对替换 tail32
        if self.static_pair and self._pair_idx is not None:
            return self._pair_idx.to(device)
        if self.subspace == "rope":
            return torch.arange(0, 64, device=device)
        if self.subspace == "nope":
            return torch.arange(64, 128, device=device)
        # E90 消融臂（对齐 E3 mass 口径）：random=固定 seed 随机 32 维 /
        # highfreq=头 16+头 16（RoPE 旋转对前元素，最高频对）
        if self.subspace == "random":
            g = torch.Generator().manual_seed(20261002)
            return torch.randperm(128, generator=g)[:32].sort().values.to(device)
        if self.subspace == "highfreq":
            delta = 64 // self.cmp_ratio
            return torch.tensor(
                list(range(0, delta)) + list(range(64, 64 + delta)), device=device)
        delta = 64 // self.cmp_ratio
        return torch.tensor(
            list(range(64 - delta, 64)) + list(range(128 - delta, 128)), device=device
        )

    def prepare_index(self, k, cu_seqlens_k):
        from .utils import prepare_pad_mask, pad_tensor
        assert k.shape[0] == 1
        self._resolve_skip()
        idx_sub = self._subspace_indices(k.device) if self.enable_subspace else None

        pad_mask, pad_cu_seqlens_k = prepare_pad_mask(self.args.tia_block_size, cu_seqlens_k)
        pad_k = pad_tensor(k, pad_mask)
        # ---- 投影基惰性提取（每层一次）：full-D [128,r] → 子空间 [32,r] ----
        if self._basis_all is not None and self._basis is None and self.layer_idx is not None:
            li = min(self.layer_idx, self._basis_all.shape[0] - 1)
            # 校准时 W[:, D2I, :] = B_sub，故 [:, idx_sub, :] 精确还原子空间基
            idx_cpu = idx_sub.cpu() if idx_sub is not None else slice(None)
            self._basis = self._basis_all[li][:, idx_cpu, :].to(k.device)  # [Hkv, 32, r]
        k_for_coarse = pad_k[..., idx_sub] if idx_sub is not None else pad_k
        if self._basis is not None:
            # 投影路径（E64f 协议）：粗筛+细筛都在 d 维投影特征上
            kp = torch.einsum("bshd,hde->bshe", k_for_coarse.float(), self._basis)
            k_coarse = rearrange(kp, "b (t bs) h d -> b t bs h d", bs=self.args.tia_block_size)
            k_min = k_coarse.amin(dim=2)
            k_max = k_coarse.amax(dim=2)
            cu_seqlens_k_coarse = pad_cu_seqlens_k // self.args.tia_block_size
            # L2 4bit 精筛：投影特征逐 token 量化（任意 last-dim 通用）
            k_qat = self.min_max_per_token_quant(kp)
            index_dict = {
                "k_min": k_min,
                "k_max": k_max,
                "k_qat": k_qat,
                "cu_seqlens_k_coarse": cu_seqlens_k_coarse,
                "cu_seqlens_k_fine": cu_seqlens_k,
            }
            if self.alpha > 0:
                # near 粗筛 avg 分数（E64f：near=avg / far=minmax）
                index_dict["k_avg"] = k_coarse.mean(dim=2)
            # ---- B：远端/近端聚类（cluster/sim_greedy 消融模式构建；默认 4bit 分区无需聚类）----
            if (self.enable_kmeans and not self.skip_far
                    and (self.far_select != "4bit" or self.near_select != "4bit")):
                self._maybe_build_kmeans(k, index_dict)
            return index_dict
        k_coarse = rearrange(
            k_for_coarse, "b (t bs) h d -> b t bs h d", bs=self.args.tia_block_size
        )
        k_min = k_coarse.amin(dim=2)
        k_max = k_coarse.amax(dim=2)
        if self.alpha > 0:
            k_avg = k_coarse.mean(dim=2)
        else:
            k_avg = None
        # E89 MoBA 臂：全维 chunk-mean（gate 分数用完整 128 维 pad_k，严格 MoBA 口径）
        k_avg_full = None
        if self.moba_gate:
            kc_full = rearrange(
                pad_k, "b (t bs) h d -> b t bs h d", bs=self.args.tia_block_size
            )
            k_avg_full = kc_full.mean(dim=2)
        cu_seqlens_k_coarse = pad_cu_seqlens_k // self.args.tia_block_size

        assert k.shape[-1] == 128
        # E85f 口径修正：L2 细筛 k_qat 与 L1 粗筛用同一子空间（idx_sub）——
        # E85e 重放口径是粗筛+细筛同特征；tail 臂 idx_sub 与旧硬编码 tail32
        # 逐位相同（cmp_ratio=2），仅 rope/nope 探索臂与静态 pair 臂行为改变
        delta = 64 // self.cmp_ratio
        indices = idx_sub if idx_sub is not None else torch.tensor(
            list(range(64 - delta, 64)) + list(range(128 - delta, 128))
        ).to(k.device)
        k_qat = torch.zeros_like(k)
        k_qat[..., indices] = self.min_max_per_token_quant(k[..., indices])
        index_dict = {
            "k_min": k_min,
            "k_max": k_max,
            "k_avg": k_avg,
            "k_avg_full": k_avg_full,
            "k_qat": k_qat,
            "cu_seqlens_k_coarse": cu_seqlens_k_coarse,
            "cu_seqlens_k_fine": cu_seqlens_k,
        }
        # ---- B：远端/近端聚类（cluster/sim_greedy 消融模式构建；默认 4bit 分区无需聚类）----
        if (self.enable_kmeans and not self.skip_far
                and (self.far_select != "4bit" or self.near_select != "4bit")):
            self._maybe_build_kmeans(k, index_dict)
        return index_dict

    def _maybe_build_kmeans(self, k, index_dict):
        S = k.shape[1]
        # far_lo 对齐 sink 边界（sink_blocks×block_size），与 compute_mask 的远端块区间一致
        far_lo = self.sink_blocks * self.args.tia_block_size
        # 修复（2026-09-29 E72 预检）：far_hi 判据块对齐——decode 时 S 每步 +1，
        # 若用 token 级 far_hi = S - near_len 则每步每层重建 kmeans（实测 61s/step）。
        # far 区真正参与选择的是 [far_lo, near_blks) 块区间（compute_mask 口径），
        # 新 token 落在 near 区不改变 far 块集合 → 块对齐后缓存跨 step 命中。
        # E7 已证：far 区内容不变时复用聚类无损（增量 assign recall 衰减 ≤0.03）。
        # near_len_dyn 与 compute_mask 同源（α>0 时 = α·mid_len），保证簇 token
        # 覆盖范围与消费端 far_tok_hi = near_blks·bs 一致。
        bs = self.args.tia_block_size
        swa_tok = self.sliding_window_size
        if self.alpha > 0 and self.beta > 0:
            sink_tok = self.sink_blocks * bs
            mid_len = max(0, S - sink_tok - swa_tok)
            near_len_dyn = max(bs, int(self.alpha * mid_len))
        else:
            near_len_dyn = self.near_len
        far_hi_blk = max(self.sink_blocks + 1, (S - near_len_dyn) // bs)
        far_hi = far_hi_blk * bs
        # ---- E110：far/near 两侧各自独立构建 ----
        #   far=cluster（cavg 原路径，原样保留）/ far=sim_greedy（增量贪心，见下）
        #   near=cluster/sim_greedy（ccluster，复用 far 侧逻辑，见 _update_near_cluster）
        if self.far_select == "sim_greedy":
            self._update_far_greedy(k, far_lo, far_hi)
        elif self.far_select == "cluster":
            # 已缓存且 far 块区间未增长（decode 新 token 全在近端）→ 复用
            if not (self._km_centroids is not None and self._km_far_hi_cached == far_hi):
                self._build_far_kmeans(k, far_lo, far_hi)
        if self.near_select in ("cluster", "sim_greedy"):
            self._update_near_cluster(k, far_hi)

    def _build_far_kmeans(self, k, far_lo, far_hi):
        """far 侧 kmeans 建簇（cavg 原逻辑抽出，行为逐位不变；E110 加 _km_dims 记录）。"""
        self._km_far_hi_cached = far_hi
        k_f = k[0, far_lo:far_hi].float()  # [Tfar, Hkv, D]
        idx_sub = self._subspace_indices(k.device)
        k_sub = k_f[..., idx_sub]           # [Tfar, Hkv, d']
        Hkv = k_sub.shape[1]
        Tfar = k_sub.shape[0]
        if Tfar < self.far_clusters * 4:
            self._km_centroids = None
            return
        centroids, assign = [], []
        for h in range(Hkv):
            c, a = self._gpu_kmeans(k_sub[:, h, :], self.far_clusters, self.far_niter)
            centroids.append(c)
            assign.append(a)
        self._km_centroids = torch.stack(centroids)      # [Hkv, K_c, d']
        self._km_token_assign = torch.stack(assign)      # [Hkv, Tfar]
        self._km_far_lo = far_lo
        self._km_far_blk_lo = far_lo // self.args.tia_block_size
        self._km_dims = idx_sub                            # E110：打分端 q 取同维（对齐建簇口径）
        # 每个 far token 的块 id（用于簇分数 → 块分数的 scatter-max）
        tok_blk = torch.arange(Tfar, device=k.device) // self.args.tia_block_size
        self._km_blk_ids = tok_blk.unsqueeze(0).expand(Hkv, Tfar)

    def _update_far_greedy(self, k, far_lo, far_hi):
        """E110：far 侧 sim_greedy（增量贪心聚类，E108 probe greedy_cluster_assign
        的 e2e 移植，语义逐位一致）。

        贪心是时序过程：token i 与「i 时刻的簇心」（running mean）比较。decode 期
        far_hi 块对齐前移时只对新增段 [old_far_hi, far_hi) 续跑增量指派——新 token
        与当前簇心比较恰为严格时序语义，故续跑 ≡ 全量重放（逐位一致）。
        这使 decode 期贪心代价 = 新增段长度（~1 块），而非全 far 区重放。
        """
        if self._km_greedy_sums is not None and self._km_far_hi_cached == far_hi:
            return  # 缓存命中（decode 每步 far_hi 不动时零成本）
        idx = self._sim_dims_indices(k.device)
        if (self._km_greedy_sums is None or far_hi <= self._km_far_hi_cached
                or far_hi - far_lo < 128):
            # 冷启动 / far 收缩（clear 后跨请求）/ far 区过小 → 全量重建
            self._km_greedy_sums = self._km_greedy_cnt = self._km_greedy_sq = None
            self._km_greedy_klive = None
            self._km_token_assign = None
            if far_hi - far_lo < 128:
                self._km_centroids = None
                self._km_far_hi_cached = far_hi
                return
            x = k[0, far_lo:far_hi].float()[..., idx]      # [Tfar, Hkv, dd]
            H, Tfar, dd = x.shape[1], x.shape[0], x.shape[2]
            sums = x.new_zeros(H, Tfar, dd)                # Kcap = Tfar（最坏每 token 一簇）
            cnt = x.new_zeros(H, Tfar)
            sq = x.new_zeros(H, Tfar)
            k_live = torch.zeros(H, dtype=torch.long, device=x.device)
            sums, cnt, sq, k_live, assign = self._greedy_cluster_pass(
                x, self.sim, sums, cnt, sq, k_live)
            self._km_token_assign = assign
        else:
            # 增量续跑：只处理新增段（严格时序语义的精确延续）
            x = k[0, self._km_far_hi_cached:far_hi].float()[..., idx]
            sums, cnt, sq, k_live, assign_new = self._greedy_cluster_pass(
                x, self.sim, self._km_greedy_sums, self._km_greedy_cnt,
                self._km_greedy_sq, self._km_greedy_klive)
            self._km_token_assign = torch.cat(
                [self._km_token_assign, assign_new], dim=1)
        self._km_greedy_sums, self._km_greedy_cnt = sums, cnt
        self._km_greedy_sq, self._km_greedy_klive = sq, k_live
        self._km_far_hi_cached = far_hi
        self._km_far_lo = far_lo
        self._km_dims = idx
        # 簇代表 = 成员算术均值（零簇行无害：assign 永不指向空槽）
        self._km_centroids = sums / cnt.clamp(min=1).unsqueeze(-1)

    def _update_near_cluster(self, k, far_hi):
        """E110：near 侧聚类（ccluster，复用 far 侧逻辑：kmeans 或 sim_greedy）。

        near 区 = [far_hi, near_hi)，near_hi 取块对齐 swa 起点（(S-swa)//bs·bs）。
        与 far 侧不同：near 区 decode 期右缘每 token 前移、左缘随 far_hi 前移，
        均值簇心无法精确撤销成员 → near 侧不做增量，按 key=(far_hi, near_hi)
        变化全量重建（near 区 ~α·mid 较小，重建代价可控；重建间隙的右缘新 token
        由消费端用细筛原始分回退覆盖——细筛分质量高于簇代表分，语义无损偏保守）。
        """
        S = k.shape[1]
        bs = self.args.tia_block_size
        near_hi = max(far_hi, (S - self.sliding_window_size) // bs * bs)
        if self._km_near_key == (far_hi, near_hi) and self._km_near_centroids is not None:
            return
        self._km_near_key = (far_hi, near_hi)
        self._km_near_centroids = None
        self._km_near_assign = None
        Tn = near_hi - far_hi
        if Tn < 256:   # near 区过小（< 4 块）不值得聚类，消费端回退细筛分
            return
        idx = (self._sim_dims_indices if self.near_select == "sim_greedy"
               else self._subspace_indices)(k.device)
        x = k[0, far_hi:near_hi].float()[..., idx]   # [Tn, Hkv, dd]
        Hkv = x.shape[1]
        if self.near_select == "sim_greedy":
            T, dd = x.shape[0], x.shape[2]
            sums = x.new_zeros(Hkv, T, dd)
            cnt = x.new_zeros(Hkv, T)
            sq = x.new_zeros(Hkv, T)
            k_live = torch.zeros(Hkv, dtype=torch.long, device=x.device)
            sums, cnt, sq, k_live, assign = self._greedy_cluster_pass(
                x, self.sim, sums, cnt, sq, k_live)
            centroids = sums / cnt.clamp(min=1).unsqueeze(-1)
        else:
            centroids, assign = [], []
            for h in range(Hkv):
                c, a = self._gpu_kmeans(x[:, h, :], self.far_clusters, self.far_niter)
                centroids.append(c)
                assign.append(a)
            centroids = torch.stack(centroids)   # [Hkv, K_c, dd]
            assign = torch.stack(assign)         # [Hkv, Tn]
        self._km_near_centroids = centroids
        self._km_near_assign = assign            # 相对 near_lo 的 token 偏移
        self._km_near_lo = far_hi
        self._km_near_hi = near_hi
        self._km_near_dims = idx

    @staticmethod
    def _greedy_cluster_pass(x, sim, sums, cnt, sq, k_live):
        """增量贪心聚类一趟（语义 = e64a / E108 probe 的 greedy_cluster_assign：
        token 按序到达，与现有簇心（成员 running mean）余弦相似度最大的活簇
        cos >= sim 则归并，否则新建簇；簇心由 sums/cnt 增量维护（算术均值）。

        x: [T, H, dd]（按序）。sums [H,K,dd] / cnt [H,K] / sq [H,K]（||sums||^2
        增量维护，免每步全量 norm）/ k_live [H]：既有簇状态（全零 = 冷启动；
        非零 = 续跑，与全量重放逐位一致——贪心决策只依赖先验状态）。
        返回 (sums, cnt, sq, k_live, assign[H,T])；容量不足时返回扩容新张量。
        实现注（E108 probe 实测坑原样规避）：归并步的 dot_a 从 dot 直接 gather
        （恒有限）——若用 m·norm·x_n 推导，无活簇首 token 时 m=-inf 经 0 权重
        乘出 NaN 污染 sq；非归并头 index_add 加 0 权重无害（目标槽均为活簇或
        全新零槽）。
        """
        T, H, dd = x.shape
        K = sums.shape[1]
        need = int(k_live.max().item()) + T
        if need > K:   # 扩容（新簇上界 = T）
            pad = need - K
            sums = F.pad(sums, (0, 0, 0, pad))
            cnt = F.pad(cnt, (0, pad))
            sq = F.pad(sq, (0, pad))
            K = sums.shape[1]
        dev = x.device
        EPS = 1e-9
        assign = torch.zeros(H, T, dtype=torch.long, device=dev)
        x_n = x.norm(dim=-1)                                  # [T,H]
        ar_h = torch.arange(H, device=dev)
        live_row = torch.arange(K, device=dev).unsqueeze(0)   # [1,K]
        for i in range(T):
            xi = x[i]                                          # [H,dd]
            xi_n = x_n[i]                                      # [H]
            norm = sq.clamp(min=0).sqrt()                      # [H,K]
            dot = torch.bmm(sums, xi.unsqueeze(-1)).squeeze(-1)  # [H,K]
            cos = dot / (norm * xi_n.unsqueeze(-1) + EPS)
            cos = cos.masked_fill(live_row >= k_live.unsqueeze(-1), float("-inf"))
            a = cos.argmax(-1)                                 # [H]
            m = cos.gather(1, a.unsqueeze(-1)).squeeze(-1)
            upd = m >= sim                                     # [H]
            w = upd.float()
            dot_a = dot.gather(1, a.unsqueeze(-1)).squeeze(-1)  # sums_a·xi（恒有限）
            flat_upd = ar_h * K + a
            sums.view(-1, dd).index_add_(0, flat_upd, xi * w.unsqueeze(-1))
            cnt.view(-1).index_add_(0, flat_upd, w)
            sq.view(-1).index_add_(0, flat_upd, (2 * dot_a + xi_n * xi_n) * w)
            nw = (~upd).float()                                # 新建簇头
            flat_new = ar_h * K + k_live
            sums.view(-1, dd).index_add_(0, flat_new, xi * nw.unsqueeze(-1))
            cnt.view(-1).index_add_(0, flat_new, nw)
            sq.view(-1).index_add_(0, flat_new, xi_n * xi_n * nw)
            assign[:, i] = torch.where(upd, a, k_live)
            k_live = k_live + (~upd).long()
        return sums, cnt, sq, k_live, assign

    def _sim_dims_indices(self, device):
        """E110：sim_greedy 聚类维度（用户设定：nope 维或压缩维）。
        subspace = 与 cluster/kmeans 路径同维（_subspace_indices 全分支；
        注意 'full' 落到 tail 分支——与 far kmeans 构建口径一致）；nope/tail/full
        为显式覆盖（tail 宽度由 cmp_ratio 控制）。"""
        if self.sim_dims == "subspace":
            return self._subspace_indices(device)
        if self.sim_dims == "nope":
            return torch.arange(64, 128, device=device)
        if self.sim_dims == "full":
            return torch.arange(128, device=device)
        delta = 64 // self.cmp_ratio
        return torch.tensor(
            list(range(64 - delta, 64)) + list(range(128 - delta, 128)), device=device
        )

    _km_far_hi_cached = -1

    @staticmethod
    def _gpu_kmeans(x, K, niter, seed=0):
        g = torch.Generator().manual_seed(seed)
        N, d = x.shape
        c = x[torch.randperm(N, generator=g)[:K]].clone()
        for _ in range(niter):
            a = (x @ c.T).argmax(dim=1)
            cnt = torch.bincount(a, minlength=K).float()
            sums = torch.zeros(K, d, device=x.device).index_add_(0, a, x)
            ne = cnt > 0
            c[ne] = sums[ne] / cnt[ne, None]
        return c, a

    # ---------------- L1 打分（A 的子空间 + B 的远端簇分数 + D' 剔除）---------------- #

    def compute_score(self, q, q_ids, index_dict, softmax_scale):
        # 缓存 squeeze 后的 q 供远端簇分数使用
        if q.shape[0] == 1 and q.shape[1] == 1:
            self._last_q = q.squeeze(0).squeeze(0).to(torch.float32)  # [H, D]
        if not self.enable_subspace:
            return super().compute_score(q, q_ids, index_dict, softmax_scale)
        # A：L1 上界只在 d' 子空间维上算（q 同步取子空间，与 k_min/k_max 对齐）
        assert q.shape[0] == 1 and q.shape[1] == 1
        from einops import rearrange as _rearrange
        k_min, k_max = (x.squeeze(0) for x in (index_dict["k_min"], index_dict["k_max"]))
        k_qat = index_dict["k_qat"]
        k_avg = index_dict.get("k_avg")
        if k_avg is not None:
            k_avg = k_avg.squeeze(0)  # [T, Hkv, d]（与 k_min/k_max 同形）
        q_sq = q.squeeze(0)  # [1, H, D]
        H = q_sq.shape[1]
        self.group_size = H // k_min.shape[1]
        coarse_shape = (1, H, k_min.shape[0])
        score_coarse = q_sq.new_full(coarse_shape, float("-inf"))
        idx_sub = self._subspace_indices(q_sq.device)
        if self._basis is not None:
            # 投影路径（E64f）：q 子空间投影到 d 维（GQA 组共享 kv 基）
            q_sub = (q_sq[0] * softmax_scale).to(torch.float32)[..., idx_sub]  # [H,32]
            Hkv = self._basis.shape[0]
            G = H // Hkv
            b_q = torch.einsum(
                "hgd,hde->hge", q_sub.reshape(Hkv, G, -1), self._basis
            ).reshape(H, -1)  # [H, r]
        else:
            b_q = (q_sq[0] * softmax_scale).to(torch.float32)[..., idx_sub]  # [H, d']
        b_k_min = repeat(k_min, "t h d -> t (h g) d", g=self.group_size).to(torch.float32)
        b_k_max = repeat(k_max, "t h d -> t (h g) d", g=self.group_size).to(torch.float32)
        b_score = (
            einsum(b_q.clamp(max=0), b_k_min, "h d, kt h d -> h kt")
            + einsum(b_q.clamp(min=0), b_k_max, "h d, kt h d -> h kt")
        )
        score_coarse[0] = b_score.to(q_sq.dtype)
        # E89 MoBA 臂：全维 chunk-mean gate 分（gate 用完整 128 维，严格 MoBA 口径）
        score_moba = None
        if self.moba_gate:
            k_avg_full = index_dict.get("k_avg_full")
            b_k_avg_full = repeat(
                k_avg_full.squeeze(0), "t h d -> t (h g) d", g=self.group_size
            ).to(torch.float32)
            b_q_full_gate = (q_sq[0] * softmax_scale).to(torch.float32)
            score_moba = q_sq.new_full(coarse_shape, float("-inf"))
            score_moba[0] = einsum(
                b_q_full_gate, b_k_avg_full, "h d, kt h d -> h kt"
            ).to(q_sq.dtype)
        # E64f：near 粗筛 avg 分数（块均值精确分，不 clamp）
        score_coarse_avg = None
        if k_avg is not None:
            b_k_avg = repeat(k_avg, "t h d -> t (h g) d", g=self.group_size).to(torch.float32)
            score_coarse_avg = q_sq.new_full(coarse_shape, float("-inf"))
            score_coarse_avg[0] = einsum(b_q, b_k_avg, "h d, kt h d -> h kt").to(q_sq.dtype)
        # L2 精筛分数：非投影=全维 k_qat；投影=投影特征 k_qat（E64f 协议）
        # 投影路径必须 fp32：8 维点积幅度 ~O(0.5)，bf16 绝对步长会折叠 top-K 边界
        # 的小 gap 分数成同值 → topk 退化为位置偏置（e2e mass 0.99→0.81 实测）
        fine_shape = (1, H, k_qat.shape[1])
        if self._basis is not None:
            score_fine = q_sq.new_full(fine_shape, float("-inf"), dtype=torch.float32)
        else:
            score_fine = q_sq.new_full(fine_shape, float("-inf"))
        if self._basis is not None:
            b_q_full = b_q
        else:
            b_q_full = (q_sq[0] * softmax_scale).to(torch.float32)
        b_k_qat = repeat(k_qat.squeeze(0), "t h d -> t (h g) d", g=self.group_size).to(torch.float32)
        if self._basis is not None:
            score_fine[0] = einsum(b_q_full, b_k_qat, "h d, kt h d -> h kt")
        else:
            score_fine[0] = einsum(b_q_full, b_k_qat, "h d, kt h d -> h kt").to(q_sq.dtype)
        return {
            "score_coarse": score_coarse.unsqueeze(0),
            "score_coarse_avg": None if score_coarse_avg is None else score_coarse_avg.unsqueeze(0),
            "score_moba": None if score_moba is None else score_moba.unsqueeze(0),
            "score_fine": score_fine.unsqueeze(0),
        }

    def _far_token_score(self, q):
        """B：远端 token 的簇分数（E4/E7 语义：按中心分数 token 级展开）。

        q: [H(全部 head), D]。返回 [Hkv, Tfar]（相对 far_lo 的 token 偏移）或 None。
        E110：q 取维改用建簇时记录的 _km_dims（sim_greedy 的 sim_dims 可异于
        _subspace_indices——打分/建簇维度必须同源；kmeans 路径 _km_dims 即
        _subspace_indices 的返回值，行为不变）。
        """
        if self._km_centroids is None:
            return None
        idx_sub = self._km_dims if self._km_dims is not None \
            else self._subspace_indices(q.device)
        q_sub = q[:, idx_sub].float()                       # [H, d']
        H = q.shape[0]
        Hkv = self._km_centroids.shape[0]
        G = H // Hkv
        q_g = q_sub.reshape(Hkv, G, -1).sum(1)              # [Hkv, d']（group 求和，与 L1 量纲一致）
        cscore = torch.einsum("hd,hkd->hk", q_g, self._km_centroids)  # [Hkv, K_c]
        tok_score = cscore.gather(1, self._km_token_assign)  # [Hkv, Tfar]
        return tok_score

    def _near_token_score(self, q):
        """E110：near 区 token 的簇分数（与 _far_token_score 同构）。

        q: [H, D]。返回 [Hkv, Tn]（相对 _km_near_lo 的 token 偏移）或 None。
        与 far 侧的唯一差别：group 聚合用 mean——far 侧单独 topk 用 sum 无所谓
        量纲，near 侧簇分数要与「未覆盖段的细筛回退分 sf_g（group-mean 原始
        点积分）」在同一 topk 内竞争，sum 会放大 G 倍导致簇段霸榜，必须 mean。
        """
        if self._km_near_centroids is None:
            return None
        q_sub = q[:, self._km_near_dims].float()            # [H, dd]
        H = q.shape[0]
        Hkv = self._km_near_centroids.shape[0]
        G = H // Hkv
        q_g = q_sub.reshape(Hkv, G, -1).mean(1)             # [Hkv, dd]（group 均值，对齐 sf_g 量纲）
        cscore = torch.einsum("hd,hkd->hk", q_g, self._km_near_centroids)  # [Hkv, K_c]
        return cscore.gather(1, self._km_near_assign)       # [Hkv, Tn]

    def _moba_mask(self, score_dict, kt, bs):
        """E89 MoBA 复现臂：chunk gate top-K 块全展开（training-free，统一 harness）。

        gate = 全维 q·chunk-mean 分数；选 top-(K2/BS) 块 + 当前块强制，
        sink/swa 保送与 PSI 同口径。返回与 score_fine 同宽的 bool mask。
        """
        sm = score_dict.get("score_moba")
        assert sm is not None, "moba 臂需 prepare_index/compute_score 走 moba_gate 路径"
        sm = rearrange(sm, "b qt (h g) kt -> b qt h g kt", g=self.group_size).mean(dim=-2)
        sf = score_dict["score_fine"]
        K2 = min(sf.shape[-1], self.args.tia_level2_topk)
        nb_sel = max(1, K2 // bs)
        sink_tok = self.sink_blocks * bs
        swa_tok = self.sliding_window_size
        # 未来块屏蔽（因果）：gate 只对已见块打分，此处全为历史块，无需额外 mask
        i_blk = torch.topk(sm, min(nb_sel, sm.shape[-1]), dim=-1).indices  # [1,1,Hkv,nb]
        m = torch.zeros(
            sm.shape[:-1] + (sf.shape[-1],), dtype=torch.bool, device=sm.device
        )
        # 块展开：每块 bs 个 token（尾块按序列长度截断）
        tok_hi = min(sf.shape[-1], kt * bs)
        tok = (i_blk.unsqueeze(-1) * bs
               + torch.arange(bs, device=sm.device).view(1, 1, 1, bs)
               ).clamp(max=tok_hi - 1).reshape(i_blk.shape + (bs,))  # [1,1,Hkv,nb*bs]
        m.scatter_(-1, tok.reshape(*m.shape[:-1], -1), True)
        # 正交强制区：sink 头部 + swa 尾部（与 PSI/sigma 臂同口径，不占 gate 预算）
        m[..., :sink_tok] = True
        m[..., max(0, tok_hi - swa_tok):] = True
        import os as _os
        if _os.environ.get("TLI_DEBUG") and self.layer_idx == 1:
            n_sel = int(m[..., :tok_hi].sum(dim=-1).float().mean().item())
            print(f"[MObadb] nb_sel={nb_sel} sel_total={n_sel}", flush=True)
        return m

    def compute_mask(self, q_ids, score_dict):
        score_coarse = score_dict["score_coarse"]
        score_fine = score_dict["score_fine"]
        # ---- E103：kv-head 共享消融----
        # 共享口径（默认）：组内 mean 聚合到 kv-head 级 [1,1,Hkv,kt]；
        # per_q_head：旁路聚合，保留 per-q-head 分数 [1,1,H,kt] 直接进 topk
        # （topk 沿最后一维，H 维天然独立——32 个 q-head 各选各的块/token）。
        if not self.per_q_head:
            score_coarse = rearrange(
                score_coarse, "b qt (h g) kt -> b qt h g kt", g=self.group_size
            ).mean(dim=-2)  # [1,1,Hkv,kt]

        kt = score_coarse.shape[-1]
        bs = self.args.tia_block_size
        t = kt * bs - 1  # 当前序列长度（pad 后）近似
        # ---- E64 α/β/γ 分区：near 区动态长度（α=0 走老逻辑 near_len=2048）----
        # 区域权威定义（2026-09-29 用户澄清）：sink/swa 为正交强制区（不进创新管线
        # 不占配额）；mid = 除 sink/swa；near = mid 靠 q 的 α 比例；far = mid 其余。
        # 修复：near 池上界切到 swa_lo_blk（旧口径 near 池含 swa 8 块，抢 β 配额 25%）
        e64_partition = self.alpha > 0 and self.beta > 0
        nb_near = 0
        swa_tok = self.sliding_window_size   # e9acd1e 回归修复：α=0 路径 swa_tok 未定义
        if e64_partition:
            sink_tok = self.sink_blocks * bs
            mid_len = max(0, kt * bs - sink_tok - swa_tok)
            near_len_dyn = max(bs, int(self.alpha * mid_len))  # near 只算 mid 部分（不含 swa）
        else:
            near_len_dyn = self.near_len
        near_blks = max(self.sink_blocks, (kt * bs - near_len_dyn) // bs)
        far_lo_blk, far_hi_blk = self.sink_blocks, near_blks  # 远端块区间（含 far_hi 前一块）
        # near 池上界 = swa 起点（swa 块完全排除出双池，纯靠 mask 强制）
        swa_lo_blk = max(near_blks, kt - max(1, swa_tok // bs))

        # ---- D'：跳层 → 远端块全部 -inf（保留 sink + 近端）----
        if self.skip_far:
            far_range = torch.arange(far_lo_blk, far_hi_blk, device=score_coarse.device)
            score_coarse[..., far_range] = float("-inf")

        # ---- B：L2 分区 topk（far/near 独立预算，防远端被近端高分挤出）----
        # E4c 修正（E4b 有整簇超选 bug）：簇分数 token 级 topk 严格预算下平均捕获
        # 0.45-0.79，相对 minmax 块上界无一致优势；far 区保留 TIA 4bit token 级
        # 精筛（质量 ≈ oracle，L03 0.999），分区仅保证 far 预算不被挤出。
        # cluster 模式（簇分数选 token）留作消融 + negative result。
        # 门控修正（2026-09-29）：E64 α/β/γ 分区路径也须走 L2 分区预算，否则 γ 是死参数
        # （B5/B6 输出 200/200 逐字一致的根因——use_partition 仅挂在 enable_kmeans 下）
        use_partition = (self.enable_kmeans or e64_partition) and not self.skip_far
        far_tok_score = None
        if use_partition and self.far_select in ("cluster", "sim_greedy"):
            q_last = getattr(self, "_last_q", None)
            if q_last is not None and self._km_centroids is not None:
                far_tok_score = self._far_token_score(q_last)  # [Hkv, Tfar]
        # E110：near 侧簇分数（ccluster / ccluster_sim 臂）
        near_tok_score = None
        near_cluster_on = self.near_select in ("cluster", "sim_greedy")
        if use_partition and near_cluster_on:
            q_last = getattr(self, "_last_q", None)
            if q_last is not None:
                near_tok_score = self._near_token_score(q_last)  # [Hkv, Tn]

        score_coarse[..., -1] = float("inf")  # TIA 语义：当前块强制
        k1 = self.args.tia_level1_topk
        if e64_partition:
            # ---- E64 L1 双池：far 池=minmax 上界分（α 区外），near 池=avg 分（近区）----
            score_coarse_avg = score_dict.get("score_coarse_avg")
            if score_coarse_avg is not None and not self.per_q_head:
                # E103：per_q_head 时同样旁路聚合（near 池 avg 分 per-q-head 独立）
                score_coarse_avg = rearrange(
                    score_coarse_avg, "b qt (h g) kt -> b qt h g kt", g=self.group_size
                ).mean(dim=-2)
            nb_near = max(1, int(round(k1 * self.beta)))
            nb_far = max(1, k1 - nb_near)
            if self.skip_far:
                # D' 兑现省算：far 块全 -inf 时收缩 far 池
                n_valid_far = int(
                    (score_coarse[..., far_lo_blk:far_hi_blk] > float("-inf")).sum(dim=-1).max().item()
                )
                nb_far = min(nb_far, max(n_valid_far, 0))
            # far 池分数源：minmax（默认=块上界）或 avg（块均值，aavg 组合）
            if self.far_method == "avg" and score_coarse_avg is not None:
                sc_far = score_coarse_avg[..., far_lo_blk:far_hi_blk]
            else:
                sc_far = score_coarse[..., far_lo_blk:far_hi_blk]
            i_f = torch.topk(sc_far, min(nb_far, sc_far.shape[-1]), dim=-1).indices + far_lo_blk
            # near 池上界 = swa_lo_blk：swa 块（含当前块）完全排除出双池，
            # 纯靠 mask 强制（p[-swa:]=1）——正交区不占创新管线配额
            # near 池分数源：avg（默认=块均值）或 minmax（块上界，mminmax 组合）
            if self.near_method == "minmax" or score_coarse_avg is None:
                sc_near = score_coarse[..., near_blks:swa_lo_blk]
            else:
                sc_near = score_coarse_avg[..., near_blks:swa_lo_blk]
            sc_near = sc_near.clone()
            if sc_near.shape[-1] > 0:
                i_n = torch.topk(sc_near, min(nb_near, sc_near.shape[-1]), dim=-1).indices + near_blks
            else:
                i_n = i_f[..., :0]  # near 池空（mid 不足一块）→ 不选
            # 严格口径（2026-09-29 用户最终澄清）：sink 块完全不进 L1 池——
            # 正交强制区不进粗筛管线、不占 K1 配额。sink token 由 L2 最终
            # mask 直接置位（必选），mass 覆盖不受影响（旧 cat 版 sink 走管线
            # 抢配额；此前「漏 sink 即崩」由 mask 强制兜住）
            indices = torch.cat([i_f, i_n], dim=-1)
            topk_mask = torch.zeros_like(score_coarse, dtype=torch.bool).scatter_(
                -1, indices, torch.ones_like(indices, dtype=torch.bool)
            )
            import os as _os
            if _os.environ.get("TLI_DEBUG") and self.layer_idx == 1:
                i_n_min = i_n.min().item() if i_n.numel() else -1
                i_n_max = i_n.max().item() if i_n.numel() else -1
                print(f"[L1dbg] kt={kt} near_blks={near_blks} swa_lo_blk={swa_lo_blk} "
                      f"nb_far={nb_far} nb_near={nb_near} "
                      f"i_f_blk_min={i_f.min().item()} i_f_blk_max={i_f.max().item()} "
                      f"i_n_blk_min={i_n_min} i_n_blk_max={i_n_max}", flush=True)
        else:
            # D' 真正兑现省算：跳层时 far 块已 -inf，topk 只取有效块数
            # （否则 topk 会用 -inf far 块填满 K1，far token 照样进 L2，白算）
            if self.skip_far:
                n_valid = int((score_coarse > float("-inf")).sum(dim=-1).max().item())
                k1 = min(k1, max(n_valid, 1))
            values, indices = torch.topk(
                score_coarse, min(score_coarse.shape[-1], k1), dim=-1
            )
            topk_mask = torch.zeros_like(score_coarse, dtype=torch.bool).scatter_(
                -1, indices, torch.ones_like(values, dtype=torch.bool)
            )
        if self.per_q_head:
            # E103：mask 已是 per-q-head [1,1,H,kt]，只做块→token 展开（不按 G 扩）
            topk_mask = repeat(
                topk_mask, "b qt h kt -> b qt h (kt bs)", bs=bs
            )
        else:
            topk_mask = repeat(
                topk_mask, "b qt h kt -> b qt (h g) (kt bs)", g=self.group_size, bs=bs
            )

        if self.enable_async:
            if self.prev_mask is not None:
                topk_mask, self.prev_mask = self.prev_mask, topk_mask
                if topk_mask.shape[-1] < score_fine.shape[-1]:
                    pad_len = score_fine.shape[-1] - topk_mask.shape[-1]
                    topk_mask = F.pad(topk_mask, (0, pad_len), value=True)
            else:
                self.prev_mask = topk_mask

        topk_mask = topk_mask[..., : score_fine.shape[-1]]
        score_fine = torch.where(topk_mask, score_fine, float("-inf"))
        p = F.softmax(score_fine, dim=-1).nan_to_num(nan=0)
        if not self.per_q_head:
            # E103：共享口径组内 mean；per_q_head 保留 [1,1,H,T] 直接 topk
            p = rearrange(p, "b qt (h g) kt -> b qt h g kt", g=self.group_size).mean(dim=-2)
        p[..., -self.sliding_window_size:] = 1.0
        # ---- E89：MoBA 复现臂（统一 harness 公平口径）----
        # training-free chunk gate：全维 chunk-mean 分数 top-(K2/BS) 块全展开。
        # 无两级、无量化、无分区；sink/swa 保送与 PSI 同口径（预算 K2=1024
        # → 16 块 + sink 2 块 + swa 16 块，选中 token 数与 PSI 严格对齐）。
        if self.moba_gate:
            return self._moba_mask(score_dict, kt, bs)
        # ---- E87：top-σ 选择（用户 2026-10-02 定义）----
        # sigma 侧不做两级：该区全部 token 的细筛原始分（GQA group-mean 口径与 p
        # 一致）≥ sink token 分数 max − σ 即选中；预算可变（σ 为质量-预算旋钮）。
        # far=far 侧 top-σ + near 侧两级 / near=反向混合 / mid=整 mid 不分区 top-σ
        if use_partition and self.sigma_select != "none":
            K2_sig = min(p.shape[-1], self.args.tia_level2_topk)
            far_tok_lo = far_lo_blk * bs
            far_tok_hi = min(near_blks * bs, p.shape[-1])
            sink_tok = far_tok_lo
            swa_lo_tok = max(0, p.shape[-1] - self.sliding_window_size)
            sf_raw = score_dict["score_fine"][..., : p.shape[-1]].to(torch.float32)
            sf_g = rearrange(sf_raw, "b qt (h g) kt -> b qt h g kt",
                             g=self.group_size).mean(dim=-2)
            sink_max = sf_g[..., :sink_tok].max(dim=-1, keepdim=True).values
            thr = sink_max - self.sigma
            m = torch.zeros_like(p, dtype=torch.bool)
            if self.sigma_select in ("far", "mid"):
                m[..., far_tok_lo:far_tok_hi] = (
                    sf_g[..., far_tok_lo:far_tok_hi] >= thr)
            else:
                far_p = p[..., far_tok_lo:far_tok_hi]
                k2_far = min(max(64, K2_sig), far_tok_hi - far_tok_lo)
                i_f = torch.topk(far_p, k2_far, dim=-1).indices + far_tok_lo
                m.scatter_(-1, i_f, True)
            if self.sigma_select in ("near", "mid"):
                if swa_lo_tok > far_tok_hi:
                    m[..., far_tok_hi:swa_lo_tok] = (
                        sf_g[..., far_tok_hi:swa_lo_tok] >= thr)
            else:
                near_p = p[..., far_tok_hi:swa_lo_tok] if swa_lo_tok > far_tok_hi \
                    else p[..., :0]
                k2_near = min(K2_sig, near_p.shape[-1])
                i_n = torch.topk(near_p, k2_near, dim=-1).indices + far_tok_hi
                m.scatter_(-1, i_n, True)
            # 正交强制区直接置位（必选不占预算）：sink 头部 + swa 尾部
            m[..., :sink_tok] = True
            m[..., swa_lo_tok:] = True
            import os as _os
            if _os.environ.get("TLI_DEBUG") and self.layer_idx == 1:
                n_sig = int(m[..., far_tok_lo:swa_lo_tok].sum(dim=-1).float().mean().item())
                print(f"[SIGdbg] mode={self.sigma_select} sigma={self.sigma} "
                      f"mid_selected={n_sig} sink_max={float(sink_max.mean()):.3f}",
                      flush=True)
            return m
        # ---- B：L2 分区 topk（C = C_near + C_far 独立预算）----
        if use_partition:
            K2 = min(p.shape[-1], self.args.tia_level2_topk)
            far_tok_lo = far_lo_blk * bs
            # far 细筛池上界 = far/near 块边界（near_len_dyn 修复后只含 α·mid，
            # 不能再用 kt*bs-near_len_dyn——那会把 near 区吞进 far 池）
            far_tok_hi = min(near_blks * bs, p.shape[-1])
            # 严格口径（2026-09-29 用户最终澄清）：sink/swa 保护 token 必选且
            # 自动从预算排除——mid 预算 K2_mid = K2 − sink_tok − swa_tok；
            # 固定区 token 不进任何 topk 池，最终 mask 直接置位
            # （如 K2=2048、sink=128、swa=128 → mid_token = 1792 全给创新管线）
            sink_tok = far_tok_lo
            swa_lo_tok = max(0, p.shape[-1] - self.sliding_window_size)
            K2_mid = max(0, K2 - sink_tok - (p.shape[-1] - swa_lo_tok))
            nt_near = 0
            if far_tok_hi > far_tok_lo:
                if far_tok_score is not None:
                    # 消融：远端按簇分数 token 级 topk
                    Tfar = far_tok_score.shape[-1]
                    if near_cluster_on and e64_partition and near_tok_score is not None:
                        # E110 ccluster：两侧都走簇分 → γ 活化（TASK.md 预算语义：
                        # γ 切 near_token/far_token）。near=4bit 的 cavg 保持现状
                        # far_tokens 语义（γ 死参数）不变——E105/E109 在跑口径，
                        # 回归保护，只对新 ccluster 臂生效
                        nt_near = min(int(nb_near * bs * self.gamma), K2_mid)
                        far_budget = max(64, K2_mid - nt_near)
                    else:
                        far_budget = self.far_tokens
                    k2_far = min(far_budget, Tfar, K2_mid)
                    i_f = (torch.topk(far_tok_score, k2_far, dim=-1).indices
                           + self._km_far_lo).unsqueeze(0).unsqueeze(0)
                else:
                    # E64 γ：near 细筛折扣 nt_near = nb_near·bs·γ，far 拿剩余预算
                    if e64_partition:
                        nt_near = min(int(nb_near * bs * self.gamma), K2_mid)
                        far_budget = max(64, K2_mid - nt_near)
                    else:
                        far_budget = self.far_tokens
                    # 默认：远端与近端同用 4bit 精筛分数，仅在独立池内 topk
                    k2_far = min(far_budget, far_tok_hi - far_tok_lo, K2_mid)
                    far_p = p[..., far_tok_lo:far_tok_hi]
                    i_f = torch.topk(far_p, k2_far, dim=-1).indices + far_tok_lo
                    import os as _os
                    if _os.environ.get("TLI_DEBUG") and self.layer_idx == 1:
                        nfin = int((far_p > float("-inf")).sum(dim=-1).max().item())
                        print(f"[L2dbg] K2={K2} K2_mid={K2_mid} sink_tok={sink_tok} "
                              f"swa_tok={p.shape[-1] - swa_lo_tok} nt_near={nt_near} "
                              f"far_budget={far_budget} far_lo={far_tok_lo} "
                              f"far_hi={far_tok_hi} k2_far={k2_far} "
                              f"far_p_finite={nfin} pshape={list(p.shape)}", flush=True)
                # near 池只取 mid-near 区 [far_tok_hi, swa_lo_tok)——
                # sink/swa 不进 topk 竞争（旧版 near 池=全序列减 far 区，
                # swa p=1 满分 + sink 残留都会进池占预算）
                k2_near = max(0, K2_mid - k2_far)
                if swa_lo_tok > far_tok_hi:
                    near_p = p[..., far_tok_hi:swa_lo_tok]
                    if near_tok_score is not None:
                        # E110 ccluster：near 池换「簇分数（簇覆盖段）+ 细筛原始分
                        # （未覆盖段）」。两源均为 group-mean 原始点积分（量纲一致，
                        # _near_token_score 用 mean 聚合即为此）；且都不过 L1 块池
                        # 门控——簇代表分本身就是粗筛（TASK.md cluster 语义），
                        # 与 far 簇分路径口径一致。未覆盖段 = 块对齐右缘尾巴 +
                        # 建簇/消费边界漂移，用细筛分回退（细筛分质量 ≥ 簇代表分，
                        # 语义无损偏保守）
                        sf_raw = score_dict["score_fine"][..., : p.shape[-1]].to(torch.float32)
                        sf_g = rearrange(
                            sf_raw, "b qt (h g) kt -> b qt h g kt", g=self.group_size
                        ).mean(dim=-2)
                        near_sc = sf_g[..., far_tok_hi:swa_lo_tok].clone()
                        cs_lo = max(self._km_near_lo, far_tok_hi)
                        cs_hi = min(self._km_near_hi, swa_lo_tok)
                        if cs_hi > cs_lo:
                            seg = near_tok_score[..., cs_lo - self._km_near_lo:
                                                  cs_hi - self._km_near_lo]
                            near_sc[..., cs_lo - far_tok_hi:cs_hi - far_tok_hi] = \
                                seg.unsqueeze(0).unsqueeze(0)
                        near_p = near_sc
                    i_n = torch.topk(near_p, min(k2_near, near_p.shape[-1]), dim=-1).indices + far_tok_hi
                else:
                    i_n = i_f[..., :0]
                topk_mask = torch.zeros_like(p, dtype=torch.bool)
                topk_mask.scatter_(-1, i_f, True)
                topk_mask.scatter_(-1, i_n, True)
                # 正交强制区直接置位（必选不占预算）：sink 头部 + swa 尾部
                topk_mask[..., :sink_tok] = True
                topk_mask[..., swa_lo_tok:] = True
                return topk_mask
        values, indices = torch.topk(p, min(p.shape[-1], self.args.tia_level2_topk), dim=-1)
        topk_mask = (
            torch.zeros_like(p, dtype=torch.bool)
            .scatter_(-1, indices, torch.ones_like(values, dtype=torch.bool))
        )
        return topk_mask

    def clear(self):
        super().clear()
        self._km_centroids = None
        self._km_token_assign = None
        self._km_blk_ids = None
        self._km_far_hi_cached = -1
        self._km_dims = None
        # E110：near 侧簇缓存 + far 侧 greedy 增量状态跨请求必须重置
        # （clear() 在每次 prefill 前调用；far greedy 增量状态残留会把上一请求
        # 的簇心带入新请求的增量指派——贪心时序语义跨请求不成立）
        self._km_near_centroids = None
        self._km_near_assign = None
        self._km_near_lo = self._km_near_hi = -1
        self._km_near_key = None
        self._km_near_dims = None
        self._km_greedy_sums = None
        self._km_greedy_cnt = None
        self._km_greedy_sq = None
        self._km_greedy_klive = None
        # E85f：跨请求必须重置——clear() 在每次 prefill 前调用，随后
        # observe_prefill_q 用本请求的 q 统计重选 pair（否则残留上一请求）
        self._pair_idx = None

    def get_block_size(self):
        return 1
