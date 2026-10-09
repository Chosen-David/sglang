# 共享表示路径矩阵验收表（HF 参考入口 × SG 生产入口 × 投影 off/on）

- **任务来源**：GPT addendum `agent_doc/advice/2026-10-09_SharedRepresentation_PathMatrix_Addendum_by_gpt.md`（共享表示验收补充）+ 主 AI 回应（验收表任务，发布共享表示图示/投影收益结论前的门禁）
- **源码基线**：`/home/wangyuanshuo02/sglang`，分支 `two-level-indexer`。源文件审计与行号基于 `bb44ba8fe649e7eef996a72d5b334ffc7567a0fb`；提交时 rebase 至 `d47f7bb453269fa351d68903cf8446e48f8abd37`（仅追加一篇 advice 文档，**引用的全部源文件零改动，行号不变**）。GPT 审计原文引用 SHA `29cf65f88e7c50c23ccba31008dd99a590c494d3`，此后代码有演进——GPT 所引行号（HF L234/L292-296、SG build L327/L340）经实测**与当前代码一致**；另 SG `_k_refine/_q_refine` 位于 `indexer.py:270-289`（与其 addendum 所引 270–289 相同）
- **方法**：静态源码审计 + 最小 CPU 断言核验（torch 单线程、固定小输入，零 GPU 占用；见 §5）
- **结论先行**：**HF 入口（tail 口径与投影口径）满足「共享表示分区前作用于两区、L1/L2 同源」契约；SG 入口投影 off（默认参数下）坐标集合一致但为条件共享（两个独立旋钮）；SG 入口投影 on 为 L1/L2 异构来源，显式标注 `unsupported`，不得声称「共享两级」**——`SGLANG_TLI_PROJ_BASIS` 只作用于 L2。

---

## 1. 共享表示契约（验收判据）

同表示契约成立需同时满足：

1. **同源派生**：L1（粗筛）与 L2（精筛）消费的特征由**同一个**坐标选择/投影基**一次派生**（同 kp 或同 idx），而非两套独立来源；
2. **分区前作用于两区**：near/far 分区只切预算与区间，不引入第二套表示——L1-near 与 L1-far 消费同一分数源张量，L2-near 与 L2-far 同理；
3. **Q/K 同基配对**：q 侧与 k 侧使用同一坐标选择或同一（kv-head 级共享）投影基；
4. **GQA 映射一致**：两级分数聚合到 kv-head 级的规则在同格内一致。

## 2. 2×2 矩阵总表

| 入口 × 投影 | L1 坐标来源 | L2 坐标来源 | near/far 是否同表示 | 共享两级判决 |
|---|---|---|---|---|
| **HF × 投影 off**（`--tli_subspace tail`，主表/E98/E109 口径） | `idx_sub`（tail32 = [48..63]+[112..127]，`cmp_ratio=4`） | **同 `idx_sub`** 逐 token 4bit 量化 | 是（同一 `score_coarse`/`score_fine` 切区） | **共享两级 ✓** |
| **HF × 投影 off**（`--tli_subspace full`，**代码默认**） | **全 128 维**（`enable_subspace=False`，`idx_sub=None`） | **尾部 32 维回退坐标**（`idx_sub=None` 时硬编码 tail） | 是（near/far 内部一致） | **✗ 异构：L1 全维 / L2 tail32**。full 名称 ≠ 两级全维一致 |
| **HF × 投影 on**（`--tli_proj_basis <pt>`，强制 `subspace=tail`） | 同一 `kp = K[...,idx_sub] @ basis` 的块 min/max/avg（r 维投影空间） | **同 `kp`** 逐 token 4bit 量化 | 是（同一投影 `score_coarse`/`score_fine` 切区） | **共享两级 ✓**（分区前作用于两区） |
| **SG × 投影 off**（默认 env，无 `SGLANG_TLI_PROJ_BASIS`） | `idx1`（`subspace_idx`，coarse_dim=32 → tail32） | `idx2`（`refine_idx`，delta=16 → 2δ=32；**独立旋钮**） | 是（同一 `kmin/kmax` 与 `kq` 切区） | **条件共享 ✓（默认 idx1 == idx2 数值一致）；旋钮分叉（coarse_dim ≠ 2·delta）即破** |
| **SG × 投影 on**（`SGLANG_TLI_PROJ_BASIS=<pt>`） | **`idx1` 原始 tail32 坐标**（无投影） | **`_k_refine = K @ basis`（PCA r=16 投影）** 4bit 量化 | 是（near/far 内部一致） | **✗ `unsupported`：L1/L2 异构来源**。投影基仅经 `_k_refine/_q_refine` 进入 L2，L1 不切换 |

> 注：矩阵为 2×2 主干，但「HF × 投影 off」必须按 `tli_subspace` 拆两个子格（GPT 补充第 2 条：不能由一个分支推广全部模式）；「SG × 投影 off」在默认参数下成立、在旋钮分叉参数下退化为异构，逐条件列明。

---

## 3. 每格证据明细（file:line 均相对 HEAD `bb44ba8fe`）

### 3.1 HF × 投影 off — 子格 A：`tli_subspace=tail`（生产主表口径）——**共享两级 ✓**

| 项 | 记录 |
|---|---|
| L1-near / L1-far 坐标 | 同一 `k_min/k_max`（块 min/max，`k_for_coarse = pad_k[..., idx_sub]`），`two-level-attention/sparse_attn/indexer/tli_indexer.py:244` + `:269-273`；near/far 双池只切区间与预算（`:843-876`），不换表示 |
| L2-near / L2-far 坐标 | 同一 `k_qat`：`indices = idx_sub`（`:292-294`），`k_qat[..., indices] = quant(k[..., indices])`（`:295-296`），全 D 维零填充张量、仅 idx_sub 维非零；near/far 双池切同一 `score_fine`（`:990-1041`） |
| 输入/输出维度 | 输入 K `[1, S, Hkv, 128]`；L1 特征 d'=32（`cmp_ratio=4` → `delta=16`，`:225-228`）；L2 特征名义 128 维、有效 32 维；输出 bool mask `[1,1,Hkv 或 H, T]` |
| 量化位置 | `prepare_index` 一次量化**并立即反量化为 fp32 存储**（`sparse_attn/indexer/tia_indexer.py:21-30` `min_max_per_token_quant`：per-token min-max 4bit，round 后 `*scales - zeros` 还原），分区之前、near/far 共享 |
| Q/K 配对 | q 同步取 `idx_sub`（`tli_indexer.py:638` `idx_sub = ... if self.enable_subspace else None`，`:648-649` `b_q = q_full[..., idx_sub]`）；L1 正 q 配 `k_max`/负 q 配 `k_min`（`:652-655`）；L2 `b_q_full · b_k_qat`（`:691`） |
| KV head 映射（GQA） | k 侧 repeat `Hkv→H` 打分（`:650-651`），分数 group-mean 回 kv-head 级进 topk（`:779-782`）；L2 softmax 后 group-mean（`:935-938`）；`per_q_head` 消融默认关（`:110`） |

### 3.2 HF × 投影 off — 子格 B：`tli_subspace=full`（**代码默认**）——**✗ L1/L2 异构**

| 项 | 记录 |
|---|---|
| L1-near / L1-far 坐标 | **全 128 维**：`full` 强制 `enable_subspace=False`（`tli_indexer.py:60-65`），`idx_sub=None`（`:234`），`k_for_coarse = pad_k`（`:244`），`k_min/k_max` 为全维块界（`:269-273`） |
| L2-near / L2-far 坐标 | `idx_sub=None` 时 `:292-296` **回退硬编码尾部坐标**（`delta = 64//cmp_ratio` → tail32）：`indices = torch.tensor(64-delta..64) + (128-delta..128)`，仅这 32 维被量化 |
| 判决 | L1 消费全 128 维、L2 消费 tail32 量化坐标——**同一格内两级表示不同源**。CPU 断言 §5-A3 实测：full 模式 `k_min` 尾维=128、`k_qat` 非零维数=32。论文/文档不得把 `full` 名称或该默认值写成「两级全维一致」；主表口径（tail）不受影响 |

### 3.3 HF × 投影 on（`--tli_proj_basis`）——**共享两级 ✓**

| 项 | 记录 |
|---|---|
| L1-near / L1-far 坐标 | 同一 `kp`：`k_for_coarse = pad_k[..., idx_sub]`（32 维）后 `kp = einsum(k_for_coarse, basis)`（`tli_indexer.py:244-247`），`k_min/k_max = kp 块界`（`:248-250`），`k_avg = kp 块均值`（`:263`）——**near avg 分数源也在投影空间** |
| L2-near / L2-far 坐标 | **同 `kp`**：`k_qat = min_max_per_token_quant(kp)`（`:253`），L2 分数 `einsum(b_q, b_k_qat)`（`:689`） |
| 输入/输出维度 | basis 离线文件 `[n_layers, Hkv, 128, r]`，按层惰性切到子空间 `[Hkv, 32, r]`（`:239-243`）；L1/L2 特征均为 r 维（本仓校准口径 r=16）；CPU 断言 §5-A4 实测 `k_min/k_qat/k_avg` 尾维全 = 16 |
| 量化位置 | 投影特征逐 token 4bit 量化（`:253`，量化于投影之后、分区之前）；打分强制 fp32（`:675-682` 注释：8 维点积幅度小，bf16 步长会折叠 top-K 边界 gap） |
| Q/K 配对 | q 先取 `idx_sub` 再与**同层同 kv-head 基**投影：`q_sub = q[..., idx_sub]` → `b_q = einsum(q_sub.reshape(Hkv,G,-1), basis)`（`:641-646`，GQA 组共享 kv 基）；L1 与 L2 都用这个 `b_q`（L2 处 `b_q_full = b_q`，`:684`）——**Q/K 两级同基配对** |
| KV head 映射 | 与 3.1 同（kv-head 级 group-mean 聚合） |
| 附注 | 投影基要求 `subspace=tail` 且强制 `enable_subspace=True`（`:156-164`）——投影 on 的可达组合只有本格，不存在「full+投影」 |

### 3.4 SG × 投影 off（默认，无 `SGLANG_TLI_PROJ_BASIS`）——**条件共享 ✓（默认参数）**

| 项 | 记录 |
|---|---|
| L1-near / L1-far 坐标 | `idx1 = subspace_idx`（`python/sglang/srt/layers/attention/tli/config.py:169-178`；`coarse_dim=32` → [48..63]+[112..127]）；构建/增量/打分全部 `k[..., idx1]`：`indexer.py:327`（build）、`:403`（update）、`:709`（`_select_taskmd` L1 分数 `qs = q[..., idx1]`）、`:527/:536`（非 taskmd 与 L1 kernel 路径）、`:1416/:2207`（批量 taskmd）；near/far 双池切同一 `sc1`/`sc_avg`（`:726-753`） |
| L2-near / L2-far 坐标 | `idx2 = refine_idx`（`config.py:180-186`；`delta=16` → 2δ=32）：`_k_refine = k[..., idx2]`（`indexer.py:270-274`），`kq = quant4_pack(_k_refine(k))`（`:340` build、`:396` update、`:1722` pool decode）；near/far 双池切同一 `fine`/`kq`（`:774-799` + `:814-860`）；打分 `_q_refine`（`:276-289`），basis=None 时即 `q[..., idx2]` |
| 共享判决 | **默认 `idx1 == idx2`（数值同一 tail32 集合，CPU 断言 §5-A1）→ 两级同坐标共享**；但二者是**两个独立旋钮**（`SGLANG_TLI_COARSE_DIM` vs `SGLANG_TLI_DELTA`），分叉参数下（如 coarse_dim=64 且 delta=16）L1 与 L2 坐标集合不同——判**条件共享**，声明共享时必须钉住两参数 |
| 输入/输出维度 | 输入 K `[S, Hkv, 128]` fp32（RoPE 后）；L1 特征 32 维；L2 特征 `2*delta`=32 维；输出 `[Hkv, K2]` token 位置（哨兵 S），批量版同 |
| 量化位置 | `quant4_pack`（`indexer.py:120-133`）**真 4bit 存储**：uint8 格点 [0,15] + fp32 双 scale（128→40 B/token-head），打分时 `kq_unpack`（`:136-138`）反量化，与 HF 的立即反量化 fp32 存储在数值上**逐位一致**（格点值 fp32 精确，`grid*sc+mn` 同操作数同序）；量化在 build/增量更新点、分区之前、near/far 共享 |
| Q/K 配对 | L1 正 q 配 `kmax`/负 q 配 `kmin`（`:710-715`）；avg 分数源 `kavg_sum`（idx1 维块和，`:344`，消费端 `/bs` = padded-mean 口径 `:721`）；L2 `group-sum(q2)·kq`（`:776` + `:794`）或 `q_agg=max` 消融（`:787-791`） |
| KV head 映射 | L1 group-`sum(q)`（`:715`，≡ HF group-mean 的 G 倍，topk 等价）；L2 group-sum（**与 HF 的 softmax 后 group-mean 存在固有聚合口径差异**，`_select_taskmd` docstring `:686-689` 显式声明为非对齐项） |

### 3.5 SG × 投影 on（`SGLANG_TLI_PROJ_BASIS=<pt>`）——**✗ `unsupported`：L1/L2 异构来源**

| 项 | 记录 |
|---|---|
| L1-near / L1-far 坐标 | **不变，仍 `idx1` 原始 tail32 坐标**——投影基的注入点只有 `_k_refine/_q_refine`（`indexer.py:270-289`），L1 全部打分路径（`:709/:715`、`:527/:536`、`:1416/:1885/:2207`）与 `kavg_sum`（`:344`，avg 分数源仍 idx1 维）均不经过 basis。CPU 断言 §5-A2 实测：basis 给定后 `idx1` 不变、`_k_refine` 输出 r 维 |
| L2-near / L2-far 坐标 | `basis = K @ basis`（PCA r=16 投影，`config.py:70` `proj_rank` 默认 16）→ `quant4_pack` 4bit（`:340`）；打分 `_q_refine` 投影 q（`:774`，basis 路径 r 维直接归约 `:795-799`，无零填充对拍口径——`:303` 断言 `assert self.basis is None` 于 `_pad_refine_full`） |
| 判决 | **`unsupported`（不得标共享两级）**：L1 = 原始坐标选择（tail32），L2 = PCA 投影（r 维），两级不同源。`SGLANG_TLI_PROJ_BASIS` 描述为 L2 专用（`indexer.py:169-171` docstring「L2 精筛表示从维度选择切换为 PCA 投影」）；启用该开关**不代表 L1 也切换到该基**。此外 avg 分数源仍固定 idx1 维 → 投影 on + near_method=avg 时 L1-near 与 L2-near 表示进一步异构。**历史投影代理结果（M9/e64f 等）不能证明「两区两级共享投影」的端到端质量** |
| basis 注入链 | `config.py:69`（env）→ `backend.py:141-179`（加载 + TP 切片 + Hkv 对齐断言）→ `backend.py:185-199`（按层注入 `TLIIndexer(basis=...)`）；与 far_kmeans 互斥（`indexer.py:178`） |
| 量化位置 | 同 3.4（quant4_pack 于投影特征之上，L2 内部 near/far 共享） |
| Q/K 配对 | L2 内部 q/k 同基（`_q_refine` 与 `_k_refine` 同一 basis，且「先投影后 GQA-sum == 先 sum 后投影」线性性注释 `:277-279`）；但 **L1 的 q/k 配对在 idx1 原始坐标，与 L2 不同基** |

### 3.6 两入口公共事实（所有格共享）

| 项 | 记录 |
|---|---|
| near/far 分区不引入第二表示 | 两入口中 near/far 均只切区间与预算：HF `tli_indexer.py:843-876`（L1 双池切 `score_coarse`/`score_coarse_avg`）+ `:990-1041`（L2 双池切同一 `p`）；SG `indexer.py:726-753`（L1）+ `:814-860`（L2）。**四格 L1-near/L1-far/L2-near/L2-far 在每格内部 near/far 对称**，异构只出现在「L1 vs L2」或「入口 vs 入口」之间 |
| 保护集合 | sink 头部（HF `sink_blocks=2`；SG `sink_blocks=2`）+ swa 尾部为正交强制区，不进双池不占预算；D' gate 只作用于 far 区（HF `:808-811`；SG select `:548-563` / 动态 gate `:497-501`） |
| 最终注意力消费原始全维 KV | HF：`sparse_attn/ops/eager_decoding.py:39-49`——全维 fp32 K/V，mask 按 kv-head 广播到组内 G 个 q-head；SG：`python/sglang/srt/layers/attention/tli/backend.py:921-955` `_sparse_attn`——从 pool gather 全 D=128 的 `k_buf/v_buf`，per q-head softmax。**索引/量化特征仅用于选择，不进入最终 attention 数值** |

---

## 4. unsupported / 历史异构变体汇总

| 组合 | 标注 | 原因 |
|---|---|---|
| SG × 投影 on | **`unsupported`（L1 异构）** | L1 固定 `idx1` 原始坐标，仅 L2 切 PCA 投影；同格内两级不同源（§3.5）。论文不得以该开关声称「共享两级投影」 |
| HF × `tli_subspace=full`（投影 off 默认） | **历史异构变体** | L1 全 128 维 vs L2 tail32 回退（§3.2）；full 名称 ≠ 两级全维一致。生产主表口径为 tail，full 是参数化默认值而非验收口径 |
| SG × 投影 off × 旋钮分叉（`coarse_dim ≠ 2·delta`） | **条件共享失效** | `idx1`/`idx2` 是独立旋钮，默认值巧合相同（§3.4）；声明共享必须钉住 `SGLANG_TLI_COARSE_DIM=32` 且 `SGLANG_TLI_DELTA=16` |
| HF 投影 on + `tli_subspace≠tail` | **不可达（显式拒绝）** | `tli_indexer.py:156-164` 直接 `ValueError`；投影基定义在 tail-32 子空间上 |
| SG 投影 on + `far_kmeans` | **不可达（显式互斥）** | `indexer.py:178` `assert basis is None or not p.far_kmeans` |

## 5. 最小 CPU 断言核验（2026-10-10，零 GPU）

环境：HEAD `bb44ba8fe` 工作树；`torch.set_num_threads(1)`；CPU-only；固定小输入（S=128, Hkv=8, H=32, D=128）。协议：**源码阅读不给通过，以下每条断言实际执行通过**；投影 on/off 不要求互相一致，只要求各组内部与声明坐标一致。

| # | 断言 | 结果 |
|---|---|---|
| A1 | SG 默认 `subspace_idx(128) == refine_idx(128)`，均 32 维 [48..63]+[112..127]；`refine_nd()=32` | **PASS** |
| A2 | SG basis 给定时 `_k_refine(k) == k @ basis`（r=16 维）且 `idx1` 保持 tail32 不变（L1 不随投影切换） | **PASS** |
| A3 | HF `full` 模式：`enable_subspace=False`，`prepare_index` 产 `k_min` 尾维 **128**、`k_qat` 非零维数 **32**（L2 回退 tail）；`tail` 模式：`k_min` 尾维 32、L1 `idx_sub` 与 L2 回退坐标逐元素相等 | **PASS** |
| A4 | HF 投影 on（tail + 随机基 [1,8,128,16]）：`k_min/k_qat/k_avg` 尾维全部 = 16——L1 与 L2 消费同一投影空间特征；`compute_score` 正常产出 `score_coarse/score_coarse_avg/score_fine` | **PASS** |
| A5 | HF 量化语义确认：`min_max_per_token_quant` 为 per-token min-max 4bit、量化即还原 fp32；SG `quant4_pack/kq_unpack` 同语义（uint8 格点 + 双 scale，重建逐位一致，`indexer.py:120-138` 注释） | **PASS** |

未覆盖（如实标注为**待验**，不以源码阅读替代）：
- 端到端小输入逐格「声明坐标 vs 实际消费特征」哈希比对（需跑完整 select→mask 链路的 instrumented 版本，属 E120 冻结协议范围）；
- M8/M10 fused kernel 路径（`SGLANG_TLI_L1B/L2B/L2D_KERNEL` 等）的 L1/L2 表示口径——kernel 侧逐位对齐另行处理（`indexer.py:296-299` 注释），本表只覆盖 eager/taskmd 路径；
- HF `near_select=cluster/sim_greedy` 消融臂（簇分数路径的 `_km_dims` 同源保证见 `tli_indexer.py:369/:417/:466`，非主表口径，未做输入级核验）。

## 6. 论文表述边界（本验收表的直接产出）

1. **可以声称**：「共享表示分区前作用于两区（near/far），两级粗筛与细筛同源」——成立范围 = **HF 参考入口**（tail 口径与投影口径），证据 `tli_indexer.py:234-268`（同 kp 派生 L1 min/max/avg 与 L2 量化特征）。
2. **必须限定**：HF `full` 模式不满足两级全维一致（L1 全维 / L2 tail32），论文不得用 `full` 默认值反推「两级全维一致」。
3. **不得声称**：SG 生产入口投影 on 满足共享两级——`SGLANG_TLI_PROJ_BASIS` 仅作用于 L2（`_k_refine/_q_refine`），L1 恒为 `idx1` 原始坐标。若 SG 侧需要真正的共享两级投影，须实现「L1 也切换到该基」的改造（留给实现者任务决策，见 addendum 主 AI 回应）。
4. **可以声称**：SG 投影 off 在默认参数（coarse_dim=32, delta=16）下与 HF tail 口径坐标集合一致（idx1 == idx2 == tail32），量化语义等价（per-token min-max 4bit，存储形式不同：HF fp32 还原存储 vs SG 真 4bit 打包，数值逐位一致）。
5. 固有口径差异（非「共享」缺陷，须如实并报）：GQA 聚合 L1 sum-vs-mean（topk 等价）、L2 sum-vs-softmax 后 mean（topk 非严格等价，`indexer.py:686-689` 显式声明）。
