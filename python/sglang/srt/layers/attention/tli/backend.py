"""TLI sparse attention backend（M2：paged 寻址 + 增量索引 + 稀疏 prefill）。

--attention-backend tli 启动。选择逻辑 = tli/indexer.py（与
two-level-attention/exp/trace 实测口径一致，mass coverage 0.9967–1.0000）。

M2 关键设计（correctness-first，kernel 化路线见 TWO_LEVEL_INDEXER_DESIGN.md §5）：
  - paged 寻址：所有 KV 读取经 req_to_token[req, :S] 间接寻址（不再假设
    page_size=1 的 req 起始连续布局）；选择返回的逻辑位置先映射到 pool 槽位
  - decode 增量索引：首步全量 build，之后每步只取新 token 调
    update_block_index（O(n)）——修复 E5b 实测的每步全量重建 O(S) 问题
    （gov_report 186min 根因）
  - extend（prefill）：S > dense_threshold 时走两级稀疏（build 全量索引 +
    select_batched 批量选择 + torch 稀疏前向）；短序列 dense
  - M4 批量化 decode：per-layer 共享 index pool（kq/kmin/kmax 预分配 +
    几何扩容，行随请求生命周期回收）+ n≥2 时两级选择批量 eager 化
    （select_decode_batched，launch 数与 bs 无关）+ 批量稀疏前向；
    n==1 保留 per-request L1/L2 fused kernel 路径（bs=1 延迟优势）
  - M5 CUDA graph decode（3 方法契约）：
      init_cuda_graph_state   捕获前一次性预分配——全部层建池并预扩到
                              req_to_token 全宽（pool 张量捕获后不可替换，
                              否则已捕获图引用已释放内存）；行 0 保留为
                              哨兵行（pad/dummy 批专用）
      init_forward_metadata_out_graph  replay 前 host 侧维护：行生命周期
                              回收 + 稳态不变式（每活跃行 pool 内容 =
                              [0, seq_len-1)）+ 填 _graph_rows_l（batch
                              行号 → pool 行号，pad 行 → 哨兵行 0）
      init_forward_metadata_in_graph   no-op（图内全部工作在 forward_decode
                              的 graph 分支，读静态 buffer）
    图内路径 = 统一稀疏 + 统一增量（静态形状、零 host 同步）：
      - 所有行（含 pad/短序列）统一走两级稀疏——S ≤ token_budget 时
        far 池空、near+forced 配额 ≥ S，数学上等价 dense；
        1024 < S ≤ dense_threshold 的行由 dense 改稀疏（top-1024 mass
        ≈1.0，H1 实测），与 eager 路径的质量差异属已知偏差
      - 稳态不变式：S_old = seq_len-1 恒成立（out_graph 维护），图内
        update_pool_rows_decode 统一追加当前 token（pad 行 seq_len=1 →
        写哨兵行 0，KV 写 slot 0——框架既有 sacrificial 语义）
"""

from __future__ import annotations

import os
import time

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.tli.config import TLIProfile
from sglang.srt.layers.attention.tli.indexer import TLIIndexer
from sglang.srt.layers.attention.tli.kernels import tli_sparse_gather_attn_dot

# 增量维护的单步上限：decode n=1 常态；超过（如 chunked/speculative 或
# req_pool_idx 被新请求复用导致 S 倒退/跳变）则全量重建
_MAX_INCREMENTAL_NEW = 512

# M3-b 开销归因：TLI_PROFILE_TIMING=1 时累计 decode 各阶段墙钟，
# 每 64 步打一行（torch.cuda.synchronize 后计时，含同步代价，仅供归因）
_TIMING = os.environ.get("TLI_PROFILE_TIMING", "0") == "1"


class _PhaseTimer:
    def __init__(self):
        self.acc = {}  # phase -> 秒（累计）
        self.steps = 0

    def add(self, phase, dt):
        self.acc[phase] = self.acc.get(phase, 0.0) + dt

    def tick(self):
        if _TIMING:
            torch.cuda.synchronize()
        return time.time()

    def maybe_report(self):
        if not _TIMING:
            return
        self.steps += 1
        if self.steps % 64 == 0:
            total = sum(self.acc.values())
            parts = " ".join(
                f"{k}={v * 1000:.1f}ms" for k, v in sorted(self.acc.items())
            )
            print(
                f"[TLI timing] steps={self.steps} total={total * 1000:.1f}ms "
                f"({total / self.steps * 1000:.2f}ms/step) {parts}"
            )
            for k in self.acc:
                self.acc[k] = 0.0


class TLISparseAttnBackend(AttentionBackend):
    def __init__(self, runner=None) -> None:
        super().__init__()
        self.runner = runner
        self.profile = TLIProfile()
        self.indexers: dict[int, TLIIndexer] = {}
        # M4：per-layer 共享 index pool（kq/kmin/kmax 预分配 [R, cap, ...]，
        # 几何扩容），替代 per-request block_indices dict——跨请求批量
        # select/gather 的寻址前提。有效长度由 pool["S"][row] 跟踪，
        # [S:cap) 是垃圾，消费方必须按 S/nblk 切片或掩码访问。
        self.index_pools: dict[int, dict] = {}
        self.layer_skip: list[bool] | None = None
        self.token_to_kv_pool = None  # _init_from_runner 填充（新版挂在 model_runner 上）
        self.req_to_token = None
        # ---- M5 CUDA graph 状态 ----
        self._num_layers = None
        self._use_graph_path = False  # out_graph 置 True / init_forward_metadata 置 False
        self._graph_locked = False  # 捕获后禁止 pool 扩容（张量替换 = 悬垂引用）
        self._pool_s_cap_floor = 0  # 建池 S 容量下限（graph 预分配时抬高）
        self._pool_r_cap_floor = 0
        self._graph_rows_l: dict[int, torch.Tensor] = {}  # layer → [max_bs] pool 行号
        self._graph_arange: torch.Tensor | None = None
        # P2（#129）：forward_extend 索引构建的 persistent 侧流（lazy 创建；
        # 仅 prefill 路径使用，decode/M5 CUDA graph 零接触）。SGLANG_TLI_
        # SIDE_STREAM=0 关闭回退同步执行。
        self._side_stream: torch.cuda.Stream | None = None
        self._side_stream_tried = False
        self.timer = _PhaseTimer()
        if runner is not None:
            self._init_from_runner(runner)

    def _init_from_runner(self, runner) -> None:
        model_config = runner.model_config
        self.head_dim = model_config.head_dim if model_config.head_dim else 128
        # TP 兼容（#58）：取 per-rank kv head 数（TP2 下 Qwen3-8B Hkv 8→4）。
        # 原版直接取全局 num_key_value_heads=8，view(bs,8,128) 与每卡实际
        # [bs,4,128] 尺寸不符 → graph capture 即崩。
        try:
            from sglang.srt.distributed import get_parallel

            self.num_kv_heads = model_config.get_num_kv_heads(
                get_parallel().attn_tp_size, get_parallel().attn_dcp_size
            )
        except Exception:
            self.num_kv_heads = getattr(model_config, "num_key_value_heads", None) or 1
        n_layers = model_config.num_hidden_layers
        self._num_layers = n_layers
        if self.profile.layer_skip_path:
            self.layer_skip = self.profile.load_layer_skip(n_layers)
        # M9：PCA 投影基（离线校准 .pt，[n_layers, Hkv, D, r] fp32；
        # 同 D' 哲学——校准一次，运行期只读）
        self.proj_basis = None
        if self.profile.proj_basis_path:
            self.proj_basis = torch.load(
                self.profile.proj_basis_path,
                map_location=runner.device if self.runner else "cuda",
            )
            assert self.proj_basis.shape[:1] == (n_layers,), (
                f"PCA basis 层数 {self.proj_basis.shape[0]} != 模型 {n_layers}"
            )
        self.dense_threshold = self.profile.dense_threshold
        # 新版 sglang：pool 挂在 model_runner 上（ForwardBatch 不再携带）
        self.token_to_kv_pool = runner.token_to_kv_pool
        self.req_to_token = runner.req_to_token_pool.req_to_token

    def _get_indexer(self, layer_id: int) -> TLIIndexer:
        if layer_id not in self.indexers:
            basis = None
            if self.proj_basis is not None:
                li = layer_id - (self._layer_offset or 0)
                basis = self.proj_basis[li if 0 <= li < self.proj_basis.shape[0] else 0]
            idx = TLIIndexer(self.profile, head_dim=self.head_dim, basis=basis).to(
                self.runner.device if self.runner else "cuda"
            )
            if self.layer_skip is not None:
                idx.skip_far = self.layer_skip[
                    layer_id - (self._layer_offset or 0) if layer_id >= (self._layer_offset or 0) else 0
                ]
            self.indexers[layer_id] = idx
        return self.indexers[layer_id]

    _layer_offset = 0

    # ---------------- 共享 index pool（M4） ---------------- #

    def _get_pool(self, layer_id: int) -> dict:
        if layer_id not in self.index_pools:
            p = self.profile
            Hkv = self.num_kv_heads
            dev = self.runner.device if self.runner else "cuda"
            # M5：容量下限由 init_cuda_graph_state 抬高（graph 预分配）；
            # 行 0 保留为哨兵行（CUDA graph pad/dummy 批的写入目标，
            # 内容恒为垃圾，永不分配给真实请求）
            s_cap = max(4096, p.dense_threshold + 1, self._pool_s_cap_floor)
            nblk_cap = (s_cap + p.block_size - 1) // p.block_size
            r_cap = max(p.pool_rows, self._pool_r_cap_floor)
            self.index_pools[layer_id] = {
                # M6：kq 真 4bit 存储（128→40 B/token-head，S=131K 前提）：
                # uint8 格点 [R,S,Hkv,nd2] + fp32 scale/mn [R,S,Hkv]
                # （scale 用 fp32：重建 grid*sc+mn 与 fp32 存储版逐位一致）
                # M9：PCA 投影时 nd2 = r（40→r+8 B/token-head）
                "kq_q": torch.zeros(r_cap, s_cap, Hkv, p.refine_nd(), dtype=torch.uint8, device=dev),
                "kq_sc": torch.zeros(r_cap, s_cap, Hkv, device=dev),
                "kq_mn": torch.zeros(r_cap, s_cap, Hkv, device=dev),
                "kmin": torch.zeros(r_cap, nblk_cap, Hkv, p.coarse_dim, device=dev),
                "kmax": torch.zeros(r_cap, nblk_cap, Hkv, p.coarse_dim, device=dev),
                "S_cap": s_cap,
                "R_cap": r_cap,
                "free": list(range(1, r_cap)),  # 行 0 = 哨兵，不进 free
                "row_of": {},  # req_pool_idx -> row
                "S": [-1] * r_cap,  # row -> 有效长度（-1 = 空）
            }
        return self.index_pools[layer_id]

    def _alloc_row(self, pool_l: dict, req: int) -> int:
        row = pool_l["row_of"].get(req)
        if row is None:
            if not pool_l["free"]:
                self._grow_pool_r(pool_l)
            row = pool_l["free"].pop()
            pool_l["row_of"][req] = row
            pool_l["S"][row] = -1
        return row

    def _ensure_pool_s(self, pool_l: dict, need: int) -> None:
        """S 维几何扩容（need ≤ S_cap 时 no-op）。扩容后旧行内容前缀保留，
        [old:cap) 新容量零初始化。

        M5：CUDA graph 捕获后（_graph_locked）禁止扩容——pool 张量替换
        会使已捕获图引用已释放内存（静默数据损坏）。容量在
        init_cuda_graph_state 一次性预扩到 req_to_token 全宽。
        """
        if need <= pool_l["S_cap"]:
            return
        if self._graph_locked:
            raise RuntimeError(
                f"tli: pool S 容量 ({pool_l['S_cap']}) 在 CUDA graph 捕获后"
                f"不可扩容（need={need}）——请增大 --context-length 预算或"
                "设置 SGLANG_TLI_POOL_S_CAP"
            )
        old = pool_l["S_cap"]
        new_cap = max(need + 2048, old * 2)
        bs = self.profile.block_size
        old_nblk = (old + bs - 1) // bs
        new_nblk = (new_cap + bs - 1) // bs
        for key, width in (
            ("kq_q", old), ("kq_sc", old), ("kq_mn", old),
            ("kmin", old_nblk), ("kmax", old_nblk),
        ):
            t = pool_l[key]
            s_dim = new_cap if key.startswith("kq") else new_nblk
            new = t.new_zeros((t.shape[0], s_dim, *t.shape[2:]))
            new[:, :width] = t
            pool_l[key] = new
        pool_l["S_cap"] = new_cap

    def _grow_pool_r(self, pool_l: dict, add: int = 16) -> None:
        if self._graph_locked:
            # 行维同理：捕获后不可替换张量；行数应在 init_cuda_graph_state
            # 按 max_bs 预扩（含哨兵行 0）
            raise RuntimeError(
                f"tli: pool 行数 ({pool_l['R_cap']}) 在 CUDA graph 捕获后"
                "不可扩容——请增大 --cuda-graph-max-bs"
            )
        r0 = pool_l["R_cap"]
        r1 = r0 + add
        for key in ("kq_q", "kq_sc", "kq_mn", "kmin", "kmax"):
            t = pool_l[key]
            new = t.new_zeros((r1, *t.shape[1:]))
            new[:r0] = t
            pool_l[key] = new
        pool_l["free"].extend(range(r0, r1))
        pool_l["S"].extend([-1] * add)
        pool_l["R_cap"] = r1

    def _get_side_stream(self) -> torch.cuda.Stream | None:
        """P2（#129）：forward_extend 索引构建侧流（lazy 单例）。

        关闭（SGLANG_TLI_SIDE_STREAM=0）或 CUDA 不可用时返回 None =
        原同步行为。**只用于 forward_extend**——decode 与 M5 CUDA graph
        路径零接触（graph 只包 decode；侧流操作永不进入 capture 区域）。
        """
        if self._side_stream_tried:
            return self._side_stream
        self._side_stream_tried = True
        if not self.profile.use_side_stream or not torch.cuda.is_available():
            return None
        dev = self.runner.device if self.runner is not None else None
        if dev is not None and dev != torch.cuda.current_device():
            with torch.cuda.device(dev):
                self._side_stream = torch.cuda.Stream(device=dev)
        else:
            self._side_stream = torch.cuda.Stream()
        return self._side_stream

    def _row_views(self, pool_l: dict, row: int, S: int) -> dict:
        """构造 per-request select()/update_block_index() 兼容的 view dict。

        张量是 pool 行的 view（[cap,...] 带容量 padding）——调用方须保证
        pool 容量足够（update 内的 _ensure 不触发，否则会静默脱离 pool）。
        """
        nblk = (S + self.profile.block_size - 1) // self.profile.block_size
        return {
            "kmin": pool_l["kmin"][row],
            "kmax": pool_l["kmax"][row],
            "kq_q": pool_l["kq_q"][row],
            "kq_sc": pool_l["kq_sc"][row],
            "kq_mn": pool_l["kq_mn"][row],
            "nblk": nblk,
            "S": S,
        }

    # ---------------- AttentionBackend 必须实现 ---------------- #

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        """M5：CUDA graph capture 前的一次性预分配（runner 在捕获前调用）。

        - 全部层立即建池，S 维预扩到 req_to_token 全宽（捕获后 pool 张量
          不可替换，见 _ensure_pool_s 注释）；R 维 ≥ max_bs+1（含哨兵行 0）
        - _graph_rows_l[layer]：batch 行号 → pool 行号映射（out_graph 每步
          填充，图内 forward_decode 读取）；_graph_arange：批内行号
        - 显存量级（fp32 kq ≈ 1KB/token/行 + kmin/kmax 32B/token/行）：
          16K context × 33 行 × 36 层 ≈ 20GB；131K 需 ~160GB —— 长上下文
          主表须先落 M6（kq 4bit）。SGLANG_TLI_POOL_S_CAP 可显式封顶
          （超出该长度的请求会在稳态维护处报错，而非静默错）。
        """
        if self._graph_rows_l:
            return
        if self.req_to_token is None or self.token_to_kv_pool is None:
            raise RuntimeError("tli CUDA graph 需要 runner 的 req_to_token/KV pool")
        if self._num_layers is None:
            raise RuntimeError("tli CUDA graph 需要 runner.model_config")
        dev = self.req_to_token.device
        s_cap_env = int(os.environ.get("SGLANG_TLI_POOL_S_CAP", "0"))
        s_cap = s_cap_env if s_cap_env > 0 else int(self.req_to_token.shape[1])
        r_cap = max(max_bs + 1, self.profile.pool_rows)
        self._pool_s_cap_floor = s_cap
        self._pool_r_cap_floor = r_cap
        for layer_id in range(self._num_layers):
            pool_l = self._get_pool(layer_id)  # 按 floor 建池（行 0 已保留）
            if pool_l["S_cap"] < s_cap:
                self._ensure_pool_s(pool_l, s_cap)  # 捕获前最后扩容机会
            if pool_l["R_cap"] < r_cap:
                self._grow_pool_r(pool_l, r_cap - pool_l["R_cap"])
            if 0 in pool_l["free"]:  # 防御：引擎启动期行 0 必未分配
                pool_l["free"].remove(0)
            self._graph_rows_l[layer_id] = torch.zeros(
                max_bs, dtype=torch.long, device=dev
            )
        self._graph_arange = torch.arange(max_bs, device=dev)
        # 之后禁止任何扩容（图已持有当前 pool 张量的引用）
        self._graph_locked = True

    def get_cuda_graph_seq_len_fill_value(self):
        # pad 行 seq_len=1：图内统一稀疏路径下 S≤1024+sw 数学等价 dense，
        # 且 forced 窗含位置 0（softmax 恒有有效 lane，无 NaN）
        return 1

    def veto_cuda_graph(self, forward_batch) -> bool:
        """M5：批内含真实短序列行（token_budget < S ≤ dense_threshold）
        → veto CUDA graph，整批回退 eager。

        图内路径统一稀疏：S ≤ token_budget 的行 far 池空 + near/forced
        配额 ≥ S → 选择集 = 全体位置，数学等价 dense（m5 单测 S=600
        mass coverage = 1.00000）；超过预算的短行会以 4bit 近端排序淘汰
        位置（S=1500 实测 mass 0.835）——此类行必须走 eager 的 dense
        分支。pad 行（seq_len ≤ 1）不触发 veto（哨兵行语义安全）。
        读 seq_lens_cpu（cpu 镜像，无 GPU 同步）。
        """
        slc = getattr(forward_batch, "seq_lens_cpu", None)
        if slc is not None:
            lens = slc.tolist()
        else:
            lens = forward_batch.seq_lens.tolist()  # 兜底一次同步
        lo = self.profile.token_budget
        return any(lo < int(L) <= self.dense_threshold for L in lens)

    def init_forward_metadata(self, forward_batch):
        """eager 入口：标记非图路径 + 行生命周期回收。"""
        self._use_graph_path = False
        if not self.index_pools:
            return
        try:
            active = {int(x) for x in forward_batch.req_pool_indices.tolist()}
        except Exception:
            return
        for pool_l in self.index_pools.values():
            for req in list(pool_l["row_of"]):
                if req not in active:
                    row = pool_l["row_of"].pop(req)
                    pool_l["S"][row] = -1
                    pool_l["free"].append(row)

    def init_forward_metadata_out_graph(self, forward_batch, in_capture: bool = False):
        """M5：capture / replay 前的 host 侧维护（允许同步，图外执行）。

        replay（in_capture=False）职责：
          1) pool 行生命周期回收（请求退出）
          2) 稳态不变式：每层每活跃请求 pool 行内容 = [0, seq_len-1)
             ——图内统一增量追加当前 token 后即 [0, seq_len)，与 eager
             路径每步结束时的语义一致（混跑可互换）
          3) 填 _graph_rows_l[layer][batch 行号] = pool 行号
             （pad 行 → 哨兵行 0；捕获后读取路径形状静态）
        capture（in_capture=True）：dummy 批（req=0 / seq_len=1）只做行
        映射（全部 → 哨兵行 0），不做稳态维护（kv pool 尚无真实数据，
        且 dummy req 会污染 row_of）。
        """
        self._use_graph_path = True
        bs = int(getattr(forward_batch, "batch_size", 0) or 0)
        if not self._graph_rows_l or bs <= 0:
            return
        if in_capture:
            for t in self._graph_rows_l.values():
                t[:bs].zero_()
            return
        if getattr(forward_batch, "spec_info", None) is not None:
            raise NotImplementedError("tli CUDA graph 暂不支持 speculative decoding")
        num_padding = int(getattr(forward_batch, "num_padding", 0) or 0)
        # seq_lens_cpu 是 replay 静态镜像（fill_from 已按 pad 策略填充），
        # 读取无 GPU 同步；缺失时兜底一次 .tolist() 同步
        seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
        if seq_lens_cpu is not None:
            lens_l = seq_lens_cpu[:bs].tolist()
        else:
            lens_l = forward_batch.seq_lens[:bs].tolist()
        reqs_l = forward_batch.req_pool_indices[:bs].tolist()
        active: list[tuple[int, int, int]] = []  # (batch 行号, req, seq_len)
        seen = set()
        for i in range(max(bs - num_padding, 0)):
            req = int(reqs_l[i])
            L = int(lens_l[i])
            if L <= 1 or req in seen:  # L=1 即 pad/异常行；重复 req 不可能
                continue
            seen.add(req)
            active.append((i, req, L))
        act_set = {req for _, req, _ in active}
        for layer_id, pool_l in self.index_pools.items():
            for req in list(pool_l["row_of"]):
                if req not in act_set:
                    row = pool_l["row_of"].pop(req)
                    pool_l["S"][row] = -1
                    pool_l["free"].append(row)
            k_buf = self.token_to_kv_pool.get_kv_buffer(layer_id)[0]
            indexer = self._get_indexer(layer_id)
            rows_arr = [0] * bs  # pad 行 → 哨兵行 0
            for i, req, L in active:
                row = self._alloc_row(pool_l, req)
                rows_arr[i] = row
                if pool_l["S"][row] != L - 1:
                    # 新请求首步 / S 跳变（chunked/spec/混跑遗留）：全量
                    # 重建到 L-1（当前 token 由图内增量追加）
                    self._ensure_pool_s(pool_l, L)
                    k_all = k_buf[self.req_to_token[req, : L - 1]].float()
                    idx_new = indexer.build_block_index(k_all)
                    pool_l["kq_q"][row, : L - 1] = idx_new["kq_q"]
                    pool_l["kq_sc"][row, : L - 1] = idx_new["kq_sc"]
                    pool_l["kq_mn"][row, : L - 1] = idx_new["kq_mn"]
                    pool_l["kmin"][row, : idx_new["nblk"]] = idx_new["kmin"]
                    pool_l["kmax"][row, : idx_new["nblk"]] = idx_new["kmax"]
                # 预记图内追加当前 token 后的有效长度（== eager 路径每步
                # 结束时的 bookkeeping 语义，混跑无缝切换）
                pool_l["S"][row] = L
            self._graph_rows_l[layer_id][:bs] = torch.tensor(
                rows_arr, device=self._graph_rows_l[layer_id].device
            )

    def init_forward_metadata_in_graph(self, forward_batch):
        """图内元数据钩子：no-op——全部图内工作在 forward_decode 的 graph
        分支（读静态 buffer + _graph_rows_l，形状静态、零 host 同步）。"""
        pass

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: torch.nn.Module,
        forward_batch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        # 新版 q/k/v 是 2D [T, H*D] / [T, Hkv*D]（RoPE 后），统一 view 成 3D
        bs = q.shape[0]
        H = q.shape[1] // self.head_dim
        Hkv = self.num_kv_heads
        G = H // Hkv
        q = q.view(bs, H, self.head_dim)
        k = k.view(bs, Hkv, self.head_dim)
        v = v.view(bs, Hkv, self.head_dim)
        if save_kv_cache:
            self._save_kv_cache(k, v, layer, forward_batch)
        if self._use_graph_path:
            # M5 图内路径（capture 时录制 / replay 时整图重放）：
            # 统一稀疏 + 统一增量，形状静态、零 host 同步
            return self._forward_decode_graph(
                q, k, layer, forward_batch, bs, H, Hkv, G
            )
        kv_pool = self.token_to_kv_pool
        req_to_token = self.req_to_token
        out = torch.empty_like(q)
        layer_id = layer.layer_id
        indexer = self._get_indexer(layer_id)
        k_buf = kv_pool.get_kv_buffer(layer_id)[0]
        pool_l = None
        # M4：稀疏请求的索引写入 per-layer 共享 pool 行（增量 O(n)），
        # n≥2 时两级选择批量 eager 化（launch 数与 bs 无关）；dense 逐请求
        # phase-3：req/seq_len 一次 .tolist()（2 次同步替代逐行 int() 的
        # 2×bs 次）；单 token 增量行收集后批量维护（update_pool_rows_decode）
        reqs_l = forward_batch.req_pool_indices.tolist()
        lens_l = forward_batch.seq_lens.tolist()
        sparse_rows: list[tuple[int, int, int]] = []  # (batch_row, pool_row, seq_len)
        inc_rows: list[int] = []  # 批量增量维护的 (pool 行, 旧 S, req)
        inc_S_old: list[int] = []
        inc_reqs: list[int] = []
        for i in range(bs):
            req = reqs_l[i]
            seq_len = lens_l[i]  # 含当前 token
            if seq_len <= self.dense_threshold:
                t0 = self.timer.tick()
                out[i] = self._dense_attn(q[i], req_to_token[req, :seq_len], kv_pool, layer_id)
                self.timer.add("dense", time.time() - t0)
                continue
            pool_l = self._get_pool(layer_id)
            row = self._alloc_row(pool_l, req)
            S_st = pool_l["S"][row]
            if S_st < 0 or S_st >= seq_len or seq_len - S_st > _MAX_INCREMENTAL_NEW:
                # 新请求首步 / req 槽位复用 / 大跳变：全量 build 写入 pool 行
                t0 = self.timer.tick()
                self._ensure_pool_s(pool_l, seq_len)
                k_all = k_buf[req_to_token[req, :seq_len]].float()  # [S, Hkv, D]
                idx_new = indexer.build_block_index(k_all)
                nblk = idx_new["nblk"]
                pool_l["kq_q"][row, :seq_len] = idx_new["kq_q"]
                pool_l["kq_sc"][row, :seq_len] = idx_new["kq_sc"]
                pool_l["kq_mn"][row, :seq_len] = idx_new["kq_mn"]
                pool_l["kmin"][row, :nblk] = idx_new["kmin"]
                pool_l["kmax"][row, :nblk] = idx_new["kmax"]
                pool_l["S"][row] = seq_len
                self.timer.add("build", time.time() - t0)
            elif seq_len > S_st:
                # 增量：只取新 token（O(n)，E5b gov_report 186min 根因修复）
                if seq_len - S_st == 1:
                    # decode 常态（每行恰 1 新 token）→ 收集后批量维护
                    inc_rows.append(row)
                    inc_S_old.append(S_st)
                    inc_reqs.append(req)
                else:
                    t0 = self.timer.tick()
                    self._ensure_pool_s(pool_l, seq_len)
                    k_new = k_buf[req_to_token[req, S_st:seq_len]].float()
                    indexer.update_block_index(self._row_views(pool_l, row, S_st), k_new)
                    self.timer.add("increment", time.time() - t0)
                pool_l["S"][row] = seq_len
            sparse_rows.append((i, row, seq_len))
        if inc_rows:
            # 批量增量维护（~10 launch 总量）：新 token K 从 pool 槽位一次
            # gather 出 [n, Hkv, D]，flat 索引 scatter 写 kq/kmin/kmax
            t0 = self.timer.tick()
            self._ensure_pool_s(pool_l, max(lens_l))
            reqs_t = torch.tensor(inc_reqs, device=q.device)
            S_old_t = torch.tensor(inc_S_old, device=q.device)
            slots = req_to_token[reqs_t, S_old_t]  # [n]（req_to_token 已含新 token 槽位）
            k_new_b = k_buf[slots].float()  # [n, Hkv, D]
            indexer.update_pool_rows_decode(
                pool_l, torch.tensor(inc_rows, device=q.device), inc_S_old, k_new_b
            )
            self.timer.add("increment", time.time() - t0)
        if sparse_rows:
            t0 = self.timer.tick()
            if len(sparse_rows) >= 2 and self.profile.use_batch_select:
                rows_t = torch.tensor([r for _, r, _ in sparse_rows], device=q.device)
                S_list = [S for _, _, S in sparse_rows]
                idx_t = torch.tensor(
                    [i for i, _, _ in sparse_rows], device=q.device
                )
                sel = indexer.select_decode_batched(
                    pool_l, rows_t, S_list, q[idx_t].float(),
                )  # [n, Hkv, K2']（池不足槽位为哨兵 S_cap）
            else:
                # n==1 / A/B 对拍：per-request select（含 L1/L2 fused kernel 路径）
                sels = []
                for i, row, seq_len in sparse_rows:
                    sels.append(
                        indexer.select(
                            self._row_views(pool_l, row, seq_len),
                            q[i : i + 1].float(),
                            seq_len - 1,
                            use_l1_kernel=self.profile.use_l1_kernel,
                            use_l2_kernel=self.profile.use_l2_kernel,
                        )
                    )
                sel = torch.stack(sels)  # [n, Hkv, K2]
            self.timer.add("select", time.time() - t0)
            t0 = self.timer.tick()
            out_sparse = self._sparse_attn_batched(
                q,
                [i for i, _, _ in sparse_rows],
                sel,
                [S for _, _, S in sparse_rows],
                forward_batch, req_to_token, kv_pool, layer_id, Hkv, G,
            )
            rows = [i for i, _, _ in sparse_rows]
            out[rows] = out_sparse.to(out.dtype)  # 张量索引赋值要求 dtype 一致
            self.timer.add("sparse_attn", time.time() - t0)
        self.timer.maybe_report()
        # 返回约定：[T, H*D]（与 triton backend 的 reshape(-1, H*D) 一致）
        return out.reshape(bs, H * self.head_dim)

    def _forward_decode_graph(
        self, q, k, layer, forward_batch, bs, H, Hkv, G
    ):
        """M5 图内 decode 路径（录制时确定全部形状，replay 零 host 同步）。

        统一处理所有行（真实 / pad / 短序列）：
          1. 统一增量：update_pool_rows_decode(S_old = seq_lens-1, k)，
             当前 token 直接取自输入 k（eager 路径从 KV pool 回读同一值）
          2. 统一稀疏：select_decode_batched（S_list 传 seq_lens 张量 +
             静态宽度）+ _sparse_attn_batched（tensor 化后全程可录制）
        稳态不变式由 init_forward_metadata_out_graph 维护（pool 行内容 =
        [0, seq_len-1)）；pad 行映射哨兵行 0——增量写哨兵行、KV 写 slot 0
        （框架既有 sacrificial 语义），选择输出为哨兵/垃圾位置但被
        valid 掩码屏蔽（forced 窗恒含位置 0，softmax 无 NaN）。
        短序列行（S ≤ dense_threshold）由 dense 改稀疏：S ≤ 1024 时
        far 池空、near+forced 配额 ≥ S 数学等价 dense；更长则 top-1024
        mass ≈ 1.0（H1 实测），与 eager 的质量差异属已知偏差。
        """
        layer_id = layer.layer_id
        indexer = self._get_indexer(layer_id)
        pool_l = self._get_pool(layer_id)
        rows_t = self._graph_rows_l[layer_id][:bs]
        seq_lens_t = forward_batch.seq_lens.to(torch.long)  # 静态 buffer [bs]
        # 统一增量追加当前 token（pad 行 seq_len=1 → S_old=0 写哨兵行 0）
        indexer.update_pool_rows_decode(pool_l, rows_t, seq_lens_t - 1, k.float())
        # 统一两级稀疏选择（静态宽度 W_far/W_near/W_forced）
        sel = indexer.select_decode_batched(
            pool_l, rows_t, seq_lens_t, q.float(),
        )  # [bs, Hkv, W_far+W_near+W_forced]（哨兵 = S_cap）
        # 统一稀疏前向（全批一次 gather + 两个批量 einsum）
        out = self._sparse_attn_batched(
            q, self._graph_arange[:bs], sel, seq_lens_t,
            forward_batch, self.req_to_token, self.token_to_kv_pool,
            layer_id, Hkv, G,
        )
        # 返回约定：[T, H*D]
        return out.to(q.dtype).reshape(bs, H * self.head_dim)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: torch.nn.Module,
        forward_batch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        # 新版 q/k/v 是 2D [T, H*D] / [T, Hkv*D]（RoPE 后），统一 view 成 3D
        T = q.shape[0]
        H = q.shape[1] // self.head_dim
        Hkv = self.num_kv_heads
        G = H // Hkv
        q = q.view(T, H, self.head_dim)
        k = k.view(T, Hkv, self.head_dim)
        v = v.view(T, Hkv, self.head_dim)
        if save_kv_cache:
            self._save_kv_cache(k, v, layer, forward_batch)
        pool = self.token_to_kv_pool
        req_to_token = self.req_to_token
        extend_seq_lens = forward_batch.extend_seq_lens
        extend_prefix_lens = forward_batch.extend_prefix_lens
        out = torch.empty(T, H * self.head_dim, dtype=q.dtype, device=q.device)
        layer_id = layer.layer_id
        # 新版 ForwardBatch 无 extend_seq_lens_cumulative：自行 cumsum。
        # F5（#129）：优先读 CPU 镜像（forward_batch_info.py L526-527，
        # chunked prefill 恒填充），避免 device cumsum→.tolist() 的每层
        # 一次 GPU 队列排空（36 层 × 60 forward = e2e 隐藏大头之一）；
        # 镜像缺失（非 chunked 路径 / mock 测试）兜底原 device 路径。
        _esl_cpu = getattr(forward_batch, "extend_seq_lens_cpu", None)
        _epl_cpu = getattr(forward_batch, "extend_prefix_lens_cpu", None)
        if _esl_cpu is not None and _epl_cpu is not None:
            lens_l = [int(x) for x in _esl_cpu]
            prefix_l = [int(x) for x in _epl_cpu]
        else:
            lens_l = torch.cumsum(extend_seq_lens, dim=0).tolist()
            prefix_l = (
                extend_prefix_lens.tolist()
                if extend_prefix_lens is not None
                else [0] * len(lens_l)
            )
        ends = [0] * len(lens_l)
        acc = 0
        for i, x in enumerate(lens_l):
            acc += x
            ends[i] = acc
        starts = [0] + ends[:-1]  # starts[b]..ends[b] = 第 b 个请求的 token 范围
        # F5：req 行号一次 .tolist()（或 CPU 镜像），替代循环内逐请求
        # int(device_tensor[b]) 的 b 次 GPU 同步
        try:
            reqs_l = list(forward_batch.req_pool_indices.tolist())
        except (TypeError, AttributeError):
            reqs_l = [int(x) for x in forward_batch.req_pool_indices]
        # ---- P2（#129）：侧流索引构建三阶段编排 ----
        # TODO(#129 P3，跨请求流水，只设计不实现)：本 for 循环即
        # overlap_kernel_design.md 的 Ov-3(a)/F4 站点——build(i+1) ∥
        # select(i) 双流乒乓 + 跨请求批量化（ragged build + batched
        # select）。当前实现只做「单次 forward_extend 内」的侧流提交
        # （阶段 A 提交 → 阶段 B dense → 阶段 C select+attn），跨
        # forward 调用的双 buffer 轮转见设计文档 §4 Ov-3(a)。
        #
        # 事件链（防竞态的完整依赖图）：
        #   主流: save_kv_cache 写 KV pool ──record(ev_kv)──┐
        #   侧流: wait(ev_kv) → update/build（读 k_buf、写 index pool 行）
        #         ──record(ev_done_b)──┐
        #   主流: wait(ev_done_b) → select_batched（读 index pool 行）→
        #         _sparse_extend_one（读 k_buf）
        # 约束：
        #   - 全部 _alloc_row/_ensure_pool_s（可能 realloc/替换 pool 张量）
        #     必须在首个侧流提交**之前**完成（阶段 A 前置）——主流 copy
        #     旧张量与侧流写旧张量并发 = 撕裂写；
        #   - 本函数返回前，每个 ev_done 都已被主流 wait → 侧流在飞工作
        #     清零，后续 forward 的 pool realloc 安全（无悬垂侧流引用）；
        #   - 侧流上下文内分配的临时张量由 allocator 按流归属管理，
        #     pool 主张量长生命周期且返回前已同步，无需 record_stream。
        side = self._get_side_stream()
        ev_kv = None
        if side is not None:
            ev_kv = torch.cuda.Event()
            ev_kv.record()  # 主流：本 forward 全部 KV 写入完成点
        # 阶段 A0：dense/sparse 分诊 + 容量预扩 + 行分配（全部 host 侧，
        # 必须先于任何侧流提交，见上事件链约束）
        dense_jobs = []  # (b, S)
        sparse_jobs = []  # (b, req, nq, prefix, S, row)
        pool_l = None
        indexer = None
        k_buf = None
        for b in range(len(starts)):
            req = reqs_l[b]
            nq = ends[b] - starts[b]
            # F5：prefix 优先用 CPU 镜像（与上方 lens_l 同源），避免
            # device tensor 的 int() 同步；镜像缺失兜底原 device 路径
            prefix = (
                prefix_l[b] if _epl_cpu is not None
                else int(extend_prefix_lens[b]) if extend_prefix_lens is not None
                else 0
            )
            S = prefix + nq  # 已写池总长（前缀 + 当前 chunk）
            if S <= self.dense_threshold:
                dense_jobs.append((b, S))
                continue
            if pool_l is None:
                indexer = self._get_indexer(layer_id)
                k_buf = pool.get_kv_buffer(layer_id)[0]
                # M4：写入共享 index pool（decode 增量起点）
                pool_l = self._get_pool(layer_id)
            self._ensure_pool_s(pool_l, S)
            row = self._alloc_row(pool_l, req)
            sparse_jobs.append((b, req, nq, prefix, S, row))
        # 阶段 A1：索引 build/update 提交（侧流 or 原地主流）
        evs_done = [None] * len(sparse_jobs)
        for ji, (b, req, nq, prefix, S, row) in enumerate(sparse_jobs):
            locs = req_to_token[req, :S]
            S_st = pool_l["S"][row]
            incremental = (
                S_st == prefix
                and prefix > 0
                and not self.profile.far_kmeans
                and self.profile.far_select != "cluster"
            )
            t0 = self.timer.tick()
            if side is not None:
                side.wait_event(ev_kv)  # 侧流等主流 KV 写入完成
                with torch.cuda.stream(side):
                    self._extend_update_index(
                        indexer, pool_l, row, k_buf, locs,
                        prefix, S, S_st, incremental,
                    )
                evs_done[ji] = torch.cuda.Event()
                evs_done[ji].record(side)  # 侧流本请求索引就绪点
            else:
                self._extend_update_index(
                    indexer, pool_l, row, k_buf, locs,
                    prefix, S, S_st, incremental,
                )
            self.timer.add("increment" if incremental else "build", time.time() - t0)
        # 阶段 B：dense 请求主流计算（与侧流 build 并发——读 k_buf 无
        # pool 交互，天然无竞态）
        for b, S in dense_jobs:
            locs = req_to_token[reqs_l[b], :S]
            q_b = q[starts[b] : ends[b]].float()
            out[starts[b] : ends[b]] = self._dense_extend_one(
                q_b, locs, pool, layer_id, Hkv, G
            ).to(q.dtype)
        # 阶段 C：sparse 请求 select + attention（主流；select 前等该请求
        # 的侧流 ev_done——build(ji+1) 侧流工作与 select(ji) 主流计算重叠）
        for ji, (b, req, nq, prefix, S, row) in enumerate(sparse_jobs):
            if evs_done[ji] is not None:
                torch.cuda.current_stream().wait_event(evs_done[ji])
            locs = req_to_token[req, :S]
            q_b = q[starts[b] : ends[b]].float()
            # select 统一读 pool 行 view（增量/全量两分支同源——
            # test_incremental_prefill.py [5] 已证 view 与 build dict
            # 输出 torch.equal；容量 padding 由 select_batched 按
            # nblk/S 切片规避）
            index = self._row_views(pool_l, row, S)
            t_arr = torch.arange(prefix, S, device=q.device)
            sel = indexer.select_batched(
                index, q_b, t_arr, t_min_hint=prefix
            )  # [nq, Hkv, K2] 逻辑位置（t_min_hint=prefix 消 empty/early 同步）
            out[starts[b] : ends[b]] = self._sparse_extend_one(
                q_b, sel, locs, pool, layer_id, Hkv, G,
                q_raw=q[starts[b] : ends[b]],
            ).to(q.dtype)
        # 返回约定：[T, H*D]（helper 已按此形状返回）
        return out

    def _extend_update_index(
        self, indexer, pool_l, row, k_buf, locs, prefix, S, S_st, incremental
    ) -> None:
        """P2（#129）：forward_extend 的索引维护段（在调用方给定的当前
        stream 上执行——主流或侧流皆可，内部零 host 同步）。
        增量分支 = F1；全量分支 = 首 chunk / 行复用 / S 跳变 / kmeans
        消融臂。写 index pool 行，host 簿记 pool_l["S"]。
        """
        if incremental:
            # F1（#127，Quest 对齐）：chunked prefill 增量分支——S_st ==
            # prefix 说明 pool 行内容恰为 [0, prefix)，只取本 chunk 新
            # token 增量更新（O(nq)，替代每 chunk O(S) 全量重建）。
            # kq 4bit 逐 token 独立 append；kmin/kmax 块界结合律合并
            # （复用 decode 侧 update_block_index，含尾块精确界口径），
            # 与全量重建逐位一致（test_incremental_prefill.py 对拍）。
            # F2：只物化新 chunk 的 fp32（不再 k_buf[locs].float() 全宽，
            # 64K 请求 8K chunk 少付 ~7/8 的 gather+cast 带宽）。
            k_new = k_buf[locs[prefix:S]].float()  # [nq, Hkv, D]
            indexer.update_block_index(
                self._row_views(pool_l, row, prefix), k_new
            )
            pool_l["S"][row] = S
        else:
            # 首 chunk / 行复用 / S 跳变（branch miss）/ kmeans 消融臂
            # （far_centroids/far_assign 是 prefill 全局聚类，无法增量）：
            # 全量重建
            k_all = k_buf[locs].float()  # [S, Hkv, D]
            index = indexer.build_block_index(k_all)
            pool_l["kq_q"][row, :S] = index["kq_q"]
            pool_l["kq_sc"][row, :S] = index["kq_sc"]
            pool_l["kq_mn"][row, :S] = index["kq_mn"]
            pool_l["kmin"][row, : index["nblk"]] = index["kmin"]
            pool_l["kmax"][row, : index["nblk"]] = index["kmax"]
            pool_l["S"][row] = S

    # ---------------- 内部工具 ---------------- #

    def _save_kv_cache(self, k, v, layer, forward_batch):
        self.token_to_kv_pool.set_kv_buffer(
            layer, forward_batch.out_cache_loc, k, v
        )

    def _dense_attn(self, q_i, locs, pool, layer_id):
        """decode 短序列 dense（GQA）。locs: [S] pool 槽位。"""
        k_all, v_all = pool.get_kv_buffer(layer_id)
        k_all = k_all[locs].float()  # [S, Hkv, D]
        v_all = v_all[locs].float()
        q_i = q_i.float()
        H = q_i.shape[0]
        Hkv = k_all.shape[1]
        G = H // Hkv
        q_g = q_i.reshape(Hkv, G, -1)
        k_e = k_all.transpose(0, 1)  # [Hkv, S, D]
        v_e = v_all.transpose(0, 1)
        att = torch.einsum("hgd,hsd->hgs", q_g, k_e) * (self.head_dim**-0.5)
        att = torch.softmax(att, dim=-1)
        o = torch.einsum("hgs,hsd->hgd", att, v_e)
        return o.reshape(H, -1)

    def _sparse_attn(self, q_i, sel, locs, pool, layer_id, Hkv, G):
        """decode 稀疏：每 kv head 在 K2 候选上做 GQA attention（批量向量化）。

        sel: [Hkv, K2] 逻辑位置；locs[sel] → pool 槽位。
        M3-b：Hkv Python 循环（每 head 4-5 个小 kernel × 8 head ≈ 40 launch）
        → 展平成 [P, Hkv*D] 后一次 flat gather + 两个批量 einsum（~6 launch）。
        数值与循环版逐位一致（同 fp32 累加顺序）。
        """
        H = q_i.shape[0]
        # 注意顺序：get_kv_buffer 返回 (k, v)——原实现写反（v_buf, k_buf），
        # 因 smoke 的 decode 全走 dense 路径（S<2048）而潜伏，稀疏 decode
        # 首次触发（M3 吞吐基线）才暴露
        k_buf, v_buf = pool.get_kv_buffer(layer_id)
        # 哨兵 pad 处理（L2 fused kernel 路径返回 [Hkv, K2+pad]，pad=S）：
        # 位置合法值域 [0, locs.shape[0])，== S 即 pad，softmax 前屏蔽
        S_loc = locs.shape[0]
        valid = sel < S_loc  # [Hkv, K2p]（eager 路径全 True，零开销约定）
        sel_c = sel.clamp(max=S_loc - 1)
        pool_pos = locs[sel_c]  # [Hkv, K2p] pool 槽位
        K2 = pool_pos.shape[-1]
        D = self.head_dim
        # flat 索引：槽位 s、kv head h、维 d → s*(Hkv*D) + h*D + d
        d_off = torch.arange(D, device=pool_pos.device)
        h_off = torch.arange(Hkv, device=pool_pos.device).view(Hkv, 1, 1) * D
        flat = pool_pos.unsqueeze(-1) * (Hkv * D) + h_off + d_off.view(1, 1, D)
        # 注意：必须 reshape(-1) 成 1D 再索引——reshape(-1, Hkv*D)[idx] 取的是
        # 整行（[N, 1024]），第一次实现就栽在这（gathered 1024× 大小）
        k_sel = k_buf.reshape(-1)[flat.view(-1)].view(Hkv, K2, D).float()
        v_sel = v_buf.reshape(-1)[flat.view(-1)].view(Hkv, K2, D).float()
        q_g = q_i.view(Hkv, G, D)  # [Hkv, G, D]
        att = torch.einsum("hgd,hkd->hgk", q_g, k_sel) * (D**-0.5)
        att = att.masked_fill(~valid.unsqueeze(1), float("-inf"))
        att = torch.softmax(att, dim=-1)
        o = torch.einsum("hgk,hkd->hgd", att, v_sel)  # [Hkv, G, D]
        return o.reshape(H, D)

    def _sparse_attn_batched(
        self, q, row_idx, sel, seq_lens, forward_batch, req_to_token, pool, layer_id, Hkv, G
    ):
        """M4：批量稀疏前向（所有稀疏请求一次 gather + 两个批量 einsum）。

        row_idx: 批内行号（列表或 device tensor）；sel: [n, Hkv, K2]（批量
        或 per-request stack，哨兵可为 S_i 或 S_cap——只需 ≥ seq_len_i）；
        seq_lens: list[int] 或 device tensor（M5：图内路径必须传 tensor，
        torch.tensor(list) 的 H2D 不能出现在 capture 区域）。
        哨兵 pad：valid = sel < seq_len_i（逐行界），softmax 前屏蔽。
        clamp 上界用逐行 torch.minimum（原 Python 标量 max；数学等价——
        valid lane 的位置恒 ≤ t_i = seq_len_i-1，clamp 不改变任何 valid
        lane，仅保证 invalid/garbage lane 的 gather 索引在界内）。
        数值与逐请求版一致（同 fp32 累加顺序）。
        """
        n = sel.shape[0]
        H, D = q.shape[1], self.head_dim
        if torch.is_tensor(row_idx):
            rows = row_idx.to(torch.long)
        else:
            rows = torch.tensor(row_idx, device=q.device)
        if torch.is_tensor(seq_lens):
            seq_lens_t = seq_lens.to(torch.long)
        else:
            seq_lens_t = torch.tensor(seq_lens, device=q.device, dtype=torch.long)
        seq_v = seq_lens_t.view(-1, 1, 1)
        valid = sel < seq_v  # [n, Hkv, K2]
        sel_c = torch.minimum(sel, seq_v - 1)
        # req_to_token 是 2D [max_req, max_S]：一次 gather 出全部请求槽位
        reqs = forward_batch.req_pool_indices[rows]  # [n]
        locs_full = req_to_token[reqs]  # [n, max_S]（每行 = 该请求逻辑位置 → 槽位）
        pool_pos = torch.gather(
            locs_full, 1, sel_c.view(n, -1)
        ).view(n, Hkv, sel.shape[-1])  # [n, Hkv, K2] pool 槽位
        K2 = pool_pos.shape[-1]
        k_buf, v_buf = pool.get_kv_buffer(layer_id)
        # M11-decode：fused kernel 路径（SGLANG_TLI_SPARSE_KERNEL=1）——
        # eager 的 [n,Hkv,K2,D] fp32 双物化（+flat 索引大张量）全部省掉，
        # per-lane valid 掩码与 eager 语义一致（哨兵=softmax 前屏蔽）。
        # 全部为 tensor 运算（无 host 同步），CUDA graph 可录制。
        q_raw = q[rows]
        if (
            self.profile.use_sparse_attn_kernel
            and q_raw.dtype in (torch.bfloat16, torch.float16)
            and q_raw.is_contiguous()
            and (G & (G - 1)) == 0
            and (D & (D - 1)) == 0
        ):
            return tli_sparse_gather_attn_dot(
                q_raw, pool_pos, k_buf, v_buf, G,
                S_loc=k_buf.shape[0], valid=valid,
            )
        # flat 索引：槽位 s、kv head h、维 d → s*(Hkv*D) + h*D + d
        d_off = torch.arange(D, device=q.device)
        h_off = torch.arange(Hkv, device=q.device).view(1, Hkv, 1, 1) * D
        flat = (
            pool_pos.unsqueeze(-1) * (Hkv * D) + h_off + d_off.view(1, 1, 1, D)
        )  # [n, Hkv, K2, D]
        flat_v = flat.view(n, -1)
        k_sel = k_buf.reshape(-1)[flat_v].view(n, Hkv, K2, D).float()
        v_sel = v_buf.reshape(-1)[flat_v].view(n, Hkv, K2, D).float()
        q_g = q[rows].float().view(n, Hkv, G, D)  # [n, Hkv, G, D]
        att = torch.einsum("nhgd,nhkd->nhgk", q_g, k_sel) * (D**-0.5)
        att = att.masked_fill(~valid.unsqueeze(2), float("-inf"))
        att = torch.softmax(att, dim=-1)
        o = torch.einsum("nhgk,nhkd->nhgd", att, v_sel)  # [n, Hkv, G, D]
        return o.view(n, H, D)

    def _dense_extend_one(self, q_b, locs, pool, layer_id, Hkv, G):
        """单个请求的 dense causal attention（短序列 prefill / chunk）。"""
        k_all, v_all = pool.get_kv_buffer(layer_id)
        k_e = k_all[locs].float().transpose(0, 1)  # [Hkv, S, D]
        v_e = v_all[locs].float().transpose(0, 1)
        nq = q_b.shape[0]
        S = locs.shape[0]
        q_g = q_b.reshape(nq, Hkv, G, self.head_dim)
        att = torch.einsum("ahgd,hsd->ahgs", q_g, k_e) * (self.head_dim**-0.5)
        # q 行 r 的全局位置 = S - nq + r；因果 mask 相对块尾
        qpos = torch.arange(S - nq, S, device=q_b.device).view(nq, 1)
        causal = torch.arange(S, device=q_b.device).view(1, S) <= qpos
        att = att.masked_fill(~causal.view(nq, 1, 1, S), float("-inf"))
        att = torch.softmax(att, dim=-1)
        o = torch.einsum("ahgs,hsd->ahgd", att, v_e)
        return o.reshape(nq, -1)

    def _sparse_extend_one(self, q_b, sel, locs, pool, layer_id, Hkv, G, q_raw=None):
        """单个请求的稀疏 prefill（select_batched 选择 + gather 前向）。

        sel: [nq, Hkv, K2] 逻辑位置（far+near 拼接，scatter 口径无重复）。
        M11：SGLANG_TLI_PREFILL_KERNEL=1 时走 Triton fused gather+online
        softmax（省 [n,K2,D] fp32 物化的 3× 带宽，kernel 级 11-16×），
        q_raw 为 bf16 原 view（kernel 输入；eager 路径用 fp32 的 q_b）。
        """
        k_buf, v_buf = pool.get_kv_buffer(layer_id)
        nq, H = q_b.shape[0], q_b.shape[1]
        # M11 fused 路径：sel 逻辑位置 → pool 槽位（与 eager 相同的一次小 gather）
        # 注意：q_raw 是父张量 q[starts[b]:ends[b]] 的切片 view，行 stride 可能
        # 不等于 H*D（实测 stride=6144 ≠ 4096）；kernel 寻址假设行连续，
        # 必须先连续化（一次 bf16 拷贝仍远快于 eager 的 fp32 双物化），
        # 否则 row≥1 全部读错位置（e2e 乱文根因，2026-09-27 修复）。
        if (
            q_raw is not None
            and self.profile.use_prefill_kernel
            and q_raw.dtype in (torch.bfloat16, torch.float16)
            and (G & (G - 1)) == 0
            and (self.head_dim & (self.head_dim - 1)) == 0
        ):
            pool_sel = locs[sel]  # [nq, Hkv, K2] pool 槽位
            q_c = q_raw if q_raw.is_contiguous() else q_raw.contiguous()
            out_k = tli_sparse_gather_attn_dot(
                q_c, pool_sel, k_buf, v_buf, G, S_loc=k_buf.shape[0]
            )
            return out_k.view(nq, H * self.head_dim)
        K2 = sel.shape[-1]
        pool_sel = locs[sel]  # [nq, Hkv, K2] pool 槽位
        out = torch.empty(nq, H, self.head_dim, device=q_b.device, dtype=q_b.dtype)
        row_chunk = max(1, min(nq, 512))  # [512, K2=1024, D=128] fp32 ≈ 268MB/head
        for h in range(Hkv):
            q_h = q_b[:, h * G : (h + 1) * G]  # [nq, G, D]
            for r0 in range(0, nq, row_chunk):
                r1 = min(r0 + row_chunk, nq)
                pp = pool_sel[r0:r1, h]  # [n, K2]
                k_sel = k_buf[pp, h].float()  # [n, K2, D]
                v_sel = v_buf[pp, h].float()
                att = torch.einsum("ngd,nkd->ngk", q_h[r0:r1], k_sel) * (
                    self.head_dim**-0.5
                )
                att = torch.softmax(att, dim=-1)
                out[r0:r1, h * G : (h + 1) * G] = torch.einsum(
                    "ngk,nkd->ngd", att, v_sel
                ).to(q_b.dtype)
        # forward_extend 的 out 是 2D [T, H*D]，此处须展平返回
        # （smoke 测试长 prompt 未过 dense_threshold，稀疏 prefill 路径
        #   首次被 e2e 触发时暴露的形状 bug）
        return out.view(nq, H * self.head_dim)


__all__ = ["TLISparseAttnBackend"]
