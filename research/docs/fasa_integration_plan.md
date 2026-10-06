# FASA（ICLR 2026）接入准备：源码逆向 + Qwen3 适配 + mass 对照实验方案

任务编号：#138（审稿意见 S3 节要求 FASA baseline 正面对照）。
日期：2026-10-06。本文件为纯 CPU 准备工作产物：clone 仓库、通读源码、逆向校准算法、设计适配与实验方案。**未运行任何 GPU 推理。**

- 仓库：`git@github.com:wangyifei0047/FASA-ICLR2026.git`（clone 到 `/tmp/two_level/related/FASA`，HEAD `96c00c4`）
- 审稿意见源：`research/docs/PSI_子空间部分审稿意见_20261006 (1).md` §S3
- 我们的重放基础设施：`/home/wangyuanshuo02/two-level-attention/exp/trace/`（口径源 `analyze_e64a_ab_grid.py`，探针模板 `probe_e108_sim_greedy.py`），dump 在 `/tmp/trace/qwen3-8b`（16 个 dump：8 LongBench 任务 × 2 + needle32k + natural32k，每 dump 含 `layer{NN}.pt`（`k` [S,Hkv=8,128]、`q` [T,H=32,128]、`qpos`、`S`）与 `meta.json`）

---

## 1. FASA 源码逆向：校准规则的精确算法描述

FASA 两阶段结构：Phase 1 离线校准出每层每 head 的「主导频率块」集合，Phase 2 decode 时用该低维子空间打分选 top-budget token、再在选中 token 上做全维 attention（「低维筛选 + 全维计算」）。

### 1.1 Phase 1：校准（`identify_important_fre.py`）

monkey-patch attention forward，只在 **decode 步**（`query_states.shape[2] == 1`，L101/L336）触发统计，且 K 先经 `repeat_kv`（L105/L332）——即统计单位是 **q-head**（Qwen3-8B 对应 32 个，不是 kv-head）。核心函数 `partial_frequency_against_full_frequency`（L28-62）：

1. 全量分数：`attn_weights = q @ k^T`（post-RoPE 全 128 维点积，无 softmax、无 scale，L31）。
2. 频率块枚举：`for frequency_group_idx in range(0, head_dim, 2)`（L34）——**每个候选单元是物理上相邻的 2 个坐标 (2j, 2j+1)**，取 `q[..., fg:fg+2] @ k[..., fg:fg+2]^T` 为该块的偏分数（L35-37）。
3. 一致性统计量：对 `sparsity ∈ [128,256,512,1024,2048,4096]`（L427），取偏分数 top-k 与全量分数 top-k 的**索引交集占比** `|topk(partial) ∩ topk(full)| / k`（L40-55，逐 head）。
4. 数据组织：`res_dict[sparsity][freq_group][layer].append(per-head list)`（L62），**每个 decode step append 一条**；跑完校准样本后整体 dill.dump 到 `dimension_rankings/<model>/<dataset>_<evaluate_num>.pkl`（L441/L446）。
5. 校准数据口径：LongBench 任务（默认 `qasper`），`max_gen ≥ 512` 的任务用 1 个样本、否则用 `--evaluate_num`（默认 8）个样本、逐样本完整 generate（L442-445 + `eval_longbench`）。**审稿人「附录 B.1 单 Qasper 样本校准」对应的是 max_gen≥512 分支；qasper 本身 max_gen=128 走的是 8 样本分支（`bash.sh` L16-20 亦为 8）**。校准成本 = ≤8 个短生成长样本的普通前向，确实不可视为巨大。
6. **仓库缺口（重要）**：消费端 `monkey_patch/frequency/utils.py` L67 读取的是 `<dataset>_agreement.pkl`，注释期望形状 `[budget_num, layer_num, frequency_group_num, head_num]` 张量；但仓库**没有提交任何把 `res_dict`（嵌套 list、含逐 decode step 采样）聚合成该张量的脚本**（`bash.sh` L13-14 声称 Step 1 直接产出 agreement.pkl，与实际 dump 文件名 `{dataset}_{evaluate_num}.pkl` 不符）。从 `identify_important_fre_distribution.py` L463 的用法（`fc_map[1][layer_idx]` → `[freq_group, head]`）可反推聚合语义 = **对 decode-step 维度取均值**后按 `[budget, layer, freq_group, head]` 重排。我们自实现时按此语义写聚合器，并在文中注明「聚合脚本未随仓库发布，语义为反推」。

### 1.2 Phase 1.5：频率块选择规则（`monkey_patch/frequency/utils.py`）

`constructe_adaptive_selection`（L64-98）+ `adaptive_selection_topk`（L19-35）：

1. 载入 agreement 张量，按运行时 budget 索引切片：`budget_dict = {128:0, 256:1, 512:2, 1024:3, 2048:4, 4096:5}`（L66，与 Phase 1 的 sparsity_list 顺序一一对应）；budget 不在表内或 <1 时**回退用 k=1024 切片**（L69-70）。
2. 对每层：矩阵 `[freq_group_num, head_num]`（行为频率块、列为 head）。**逐 head（逐列）**：
   - `threshold = np.quantile(col_data, threshold_ratio)`（默认 0.8，L25）；
   - `indices = where(col >= threshold)`；不足 `min_k` 则取值 top-`min_k`，超过 `max_k` 则在入围者中取值 top-`max_k`（L27-30）；
   - 物理坐标展开：`head_selection = sorted(indices*2 ∪ indices*2+1)`（L32）——**候选单元 j 展开为物理坐标 (2j, 2j+1)，即再次确认单元定义是相邻坐标对**。
3. README/bash.sh 默认 `min_k=16, max_k=16`（README L58、bash.sh L33-34）：在此配置下 quantile 路径退化为**「每层每 head 取 agreement top-16 的频率块」**（threshold_ratio 仅在 min_k<max_k 时起作用）→ 16 块 × 2 维 = **32 个标量维**，与审稿 S3「统一 32 个标量维」天然对齐。
4. 输出 `records[layer]["head_selection"]`（padding 后的 [num_heads, max_len] 索引表）+ `selection_masks`，供 forward 用 `torch.gather` 按维度 gather（`core_module_with_padding` L186-232）。

### 1.3 Phase 2：运行时选择（`monkey_patch/frequency/llama_df.py` + `utils.py`）

- **prefill（q_len≠1）**：不淘汰。仅把 K 按 head 的选中/未选中维平面切开后存入自定义三槽 cache（选中维 K / 未选中维 K / V，`cache_design.py` L27-65），attention 仍是全量全维（llama_df.py L46-51 后 key_states 未被替换）。
- **decode（q_len==1）**：`core_module_with_padding` 按维 gather 出 q/K 的选中维平面 → 偏分数 `q_sel @ k_sel^T` → **逐 q-head 每步 topk 取 `budget` 个 token**（llama_df.py L67-80）→ gather 选中 token 的全维 K（选中维+未选中维平面 scatter 回原坐标，L88-90）与 V（L84）→ flash attention **全维全精度**计算（L125-135）。即每步动态、query-aware、逐 head 选举，缓存不淘汰只筛 attention 范围。
- **层范围**：`layer_control=-1` 时 `control_layer_list = range(1, num_hidden_layers)`（L62）——**第 0 层不筛选走全量**；Qwen 路径同样 `self.layer_idx > 0`（qwen_df.py L78）。
- **Qwen 路径差异**：`qwen_df.py` 用的是 `core_module_with_padding_old`（utils.py L101-138）——没有维平面拆分 cache，直接「选中维打分 → topk → gather 全维 K/V」，等价且更简单。
- **无强制集**：FASA 没有 sink/sliding-window 强制保留，纯 top-budget（与 PSI 的 sink128+swa1024 结构不同）。
- **Partial-RoPE**：论文 §6 报告了实验，但**发布代码不含任何 partial-RoPE 模型支持**（支持列表：LLaMA/Qwen2.5/Mistral/DeepSeek-R1-Distill，README L175）。复现其 partial-RoPE 结论须自实现适配，不能称官方原样。

### 1.4 关键发现：发布代码的「频率块」在 split-half 模型上不是完整 RoPE 对

Phase 1/1.5 的单元是**相邻物理坐标 (2j, 2j+1)**（identify L34-35；utils L32）。而 HF Llama/Qwen 的 `apply_rotary_pos_emb` 走 `rotate_half`（split-half 布局），真正的旋转对是 **(j, j+d/2)**。因此在 Llama/Qwen 上，FASA 发布代码的 2 维块 = 分属两个不同旋转对、频率相邻的两个坐标（各取一半），**不是论文口径的「完整频率对」**。其 agreement 统计是纯经验的（对任何切块都成立），方法学上无损，但「完整对」叙事与发布代码在 split-half 模型上不一致。

对我们的意义：审稿 S1/S3 要求按**完整频率对**为单位做对照（Qwen3 全 128 维 RoPE rotate_half，完整低频对 = (j, j+64) 第二元素即 coords [48:64]+[112:128]，正是我们的 tail32）。因此我们的 FASA 适配**以完整对为候选单元**（j ∈ [0,64)，对 (j, j+64)）——与审稿口径一致、与论文叙事一致，**与发布代码的相邻切块不一致**，此偏差必须在实验报告中如实标注（「FASA 规则的 pair 级重实现，非官方代码原样」）。

### 1.5 一句话总结

**FASA 校准算法 = 在校准样本（Qasper，1-8 个样本）的每个 decode 步、每层、每 q-head 上，测每个候选频率块的 2 维偏分数 top-k 与全维分数 top-k 的交集率（k 对齐运行时 budget），对 decode 步取均值后，每层每 head 用 quantile(0.8)+min/max_k（默认 16/16 即 top-16）选出 16 个块（32 标量维）；运行时逐 head 逐 decode 步用该 32 维打分选 top-budget token 后做全维 attention，prefill 不筛、第 0 层不筛。**

---

## 2. Qwen3-8B 适配设计

| FASA 侧 | Qwen3-8B 适配 | 说明 |
|---|---|---|
| 候选单元：相邻坐标 (2j,2j+1) | **完整旋转对 (j, j+64)，j∈[0,64)** | 按审稿 S1/S3 口径；发布代码偏差见 §1.4，报告标注 |
| 每层每 q-head（32 头，repeat_kv 后统计） | **每层每 kv-head（8 头）+ GQA 组内 sum 聚合 q** | 我们的 L2/head 规则固定为 kv-head 级（qsub=GQA sum，e64a 口径）；校准统计在同一 head 规则下测，避免引入第二个自由度。FASA 原版 32 q-head 粒度的适配差异如实记录 |
| 校准数据：Qasper（1-8 样本完整 generate） | **held-out trace dump：`lb_qasper_0` + `lb_qasper_1`**（测试集 14 dump 全部排除 qasper） | 对齐 FASA 的 qasper 校准口径；trace 已有，零 GPU 成本 |
| agreement 的 k | **k = K_mid = 1024**（对齐我们的细筛预算） | FASA 原版按 budget 索引 {128..4096} 切片，我们取 1024 桶 |
| min_k=max_k=16, threshold 0.8 | 同配置 → 每层每 kv-head **top-16 完整对**（32 维） | 与固定 tail32（也是 16 对）等维 |
| 无 partial-RoPE 支持 | Qwen3 无 partial-RoPE（128 维全 RoPE），无需处理 | 论文 §6 partial-RoPE 属其自有实验，不在本轮范围 |
| 层范围 1..N-1 | 重放覆盖全部采样层（与既有锚点同口径） | 差异记录：FASA 跳过 layer0 |
| 逐 q-head topk budget、无强制集 | **沿用我们骨架**：sink128+swa1024 强制 + L1 页粗筛 + L2 token 细筛 | 「FASA 规则选对」只是 L2 表示组件消融，**不能冒称完整 FASA 复现**（审稿原话） |

打分语义统一：所有臂的 L2 token 分 = 在所选 32 个坐标上的裸点积 `qsub · ksub`（与 tail32 臂完全同式，post-RoPE K 直接切片），不做 softmax/scale——与 FASA 偏分数语义一致、与我们既有重放一致。

---

## 3. mass 重放组件对照实验方案（S3 最小回放矩阵）

**固定骨架（全部臂严格相同）**：trace dump 重放（`analyze_e64a_ab_grid.py` 口径）——BS=64、SINK=128、SWA=1024、TAIL_N=4（末 4 个 decode query）、mid 单池；**L1 = minmax 块上界页粗筛（全 128 维、BP=64 页，臂间不变——生产主配置 L1 full128 的忠实重放且对 L2 表示中立）**；L2 = 池内 token 32 维裸点积 top-K_mid；强制集 sink+swa；head 规则 = kv-head 级 GQA sum；真值 = 全维 softmax 行级 mass coverage（`cov_mass`）。K_mid 主口径 1024（审稿 K=1024），敏感性副口径 2048。

**唯一变量 = L2 的 32 个标量维选取**（全部 16 完整对=32 维，除错配臂）：

| 臂 | 32 维选取 | 粒度 |
|---|---|---|
| `low_tail32`（incumbent） | 固定低频 16 对：j∈[48,64) → [48:64]+[112:128] | 全局固定 |
| `mid_pairs32` | 中频 16 对：j∈[24,40) | 全局固定 |
| `high_pairs32` | 高频 16 对：j∈[0,16) | 全局固定 |
| `random_pairs32` | 随机 16 个完整对 | **每 (层, kv-head) 独立随机**（seed 固定），与 FASA 臂同粒度；另跑 `random_global32`（全局一组）作副臂 |
| `mismatch_coords32` | 32 个坐标、**不含任何完整对**（如 first-half 偶数下标 16 个 + second-half 对应奇数偏移 16 个，保证 (j,j+64) 无一完整） | 全局固定；S1 的「固定物理切片负对照」精神 |
| `fasa_pairs32` | FASA 规则：held-out qasper dump 校准的每 (层, kv-head) top-16 agreement 对 | 每 (层, kv-head) |

**校准器（fasa_pairs32 专用，先跑）**：在 `lb_qasper_0/1` 两个 dump 上，逐层逐 kv-head：64 个完整对的偏分数（2 维裸点积）top-K_mid 与全维分数（128 维、带 scale 的 s_full）top-K_mid 的交集率，对 TAIL_N=4 个 decode query 取均值 → agreement[层, kv-head, 64 对] → 每 (层, kv-head) 取 top-16 对。产物落袋 `fasa_pair_tables.json`（含每对坐标表，供复查与后续 e2e 复用）。

**锚点（同脚本同 dump 重算，不引旧数字）**：`full_fine128`（全维 token 级打分 top-K_mid，无 L1，选择质量上界）、`tail32_fine`（tail32 token 级打分无 L1，e108 的 full_fine 口径）、`minmax_mono`（= low_tail32 双层臂，生产样管线）。

**统计与判决（预设，不事后挑赢家）**：
- 测试集 = 14 个 dump（16 dump 去掉 2 个 qasper 校准 dump）；每 dump 采样 12 层（既有惯例）；报告逐样本 + 全体均值 + 与 `low_tail32` 的逐 dump 配对 Δ。
- 判决门（对齐审稿 S3 判定段）：`fasa_pairs32` 配对均值 Δ ≥ +0.005（0.5pt mass）且 ≥10/14 dump 非负 → 「校准对更可靠」，论文把固定低频重定位为「特定条件下的免校准点」；Δ ∈ (−0.005, +0.005) → 等维下校准与固定低频打平，如实写「免校准不损失」；Δ ≤ −0.005 → 固定低频更强，收窄 FASA 侧表述。random/mismatch/high/mid 四臂给出结构信号（预期 mismatch/high 显著崩、random 居中、mid 中间——复刻 E85b/E90 型梯度）。
- 附带诊断（同一循环免费产出）：FASA 校准对的频率分布直方图（各层各 head 选中的 j 落在哪个频段）——直接回答「校准是否收敛到低频」这一审稿隐含问题。
- 诚实边界写入结果 JSON：组件消融≠完整 FASA 复现（无逐 q-head 选举、无 FASA 自身 budget 语义、L1/head 规则/强制集均为我们的骨架）；mass ≠ e2e（铁律），e2e 对照是下一阶段独立判据。

---

## 4. 后续 e2e 对照（Phase 2，另行立项）

1. `fasa_pairs32` 落袋后，在 sglang TLI backend 加 `--tli_fasa_pairs <json>`：L2 细筛子空间从固定 tail32 换成逐 (层, kv-head) 对表（indexer 侧 `_k_refine/_q_refine` 的维度 gather 已支持任意索引集合，改动集中在索引表加载）。
2. 校准 profiling 在 sglang 内重做（E85f `observe_prefill_q` 基础设施可复用：prefill 期采集 q/K 统计算 agreement），或直接导入 trace 校准的表（先导后训两档都跑）。
3. e2e 口径：13 任务全量 vs FullKV/tail32 主臂，双口径铁律（mass + e2e 都报）。校准成本单列（审稿 S5：等维≠等成本，须报 profiling 前向耗时）。

## 5. 工作量估计

- **Phase 1（mass 组件对照，本方案主体）**：1 个脚本 `probe_e110_fasa_pairs.py`（~400 行，模板 probe_e108 + e64a 组装）+ 校准器内嵌。实施步骤：① 校准器 + 对表落袋并 smoke（1 dump）；② 六臂 + 三锚点重放 smoke（1 dump）；③ 全量 14 dump 跑（GPU 轻负载，估 <1h，可与 E105 错峰）；④ JSON 落袋 + 分析报告。**约 4 步、0.5-1 天**。
- **Phase 2（e2e）**：sglang 集成 + 校准管线 + 13 任务全量，约 2-3 天，等 Phase 1 判决后启动（若 fasa 臂打平/更差，e2e 仅需单臂验证而非网格）。
- 风险点：① 重放层采样 12 层 vs FASA 全层——对表按全层生成、重放仍采 12 层（对表查全层无成本）；② K_mid 语义与审稿 K=1024 的解释差异须在结果报告脚注写死；③ FASA 发布代码相邻块 vs 我们 pair 级的偏差标注（§1.4）。

## 6. 证据索引（源码行号）

| 事实 | 位置 |
|---|---|
| decode 步触发 + repeat_kv（q-head 粒度） | identify_important_fre.py L101-107, L332-337 |
| 频率块 = 相邻 (2j,2j+1)、偏分数 2 维点积 | identify_important_fre.py L34-37 |
| agreement = top-k 交集率、k∈{128..4096} | identify_important_fre.py L40-62, L427 |
| 校准样本数分支（1 vs evaluate_num） | identify_important_fre.py L442-445 |
| agreement.pkl 消费 + budget 切片/回退 | utils.py L64-73 |
| quantile + min_k/max_k + indices*2 展开 | utils.py L19-35 |
| prefill 不筛 / decode 逐 head topk / 全维重算 / scatter 重构 | llama_df.py L46-90, L125-135 |
| layer0 跳过 | llama_df.py L62；qwen_df.py L78 |
| Qwen 路径 old 版（无维平面 cache） | qwen_df.py L79；utils.py L101-138 |
| 三槽 cache（选中维/未选中维/V） | cache_design.py L27-65 |
| 仓库无 agreement 聚合脚本（bash.sh 声称 vs 实际 dump 名） | bash.sh L13-14 vs identify L441/L446；distribution 脚本 L463 反推语义 |
| min_k=max_k=16（32 维）默认 | README L58；bash.sh L33-34 |
| partial-RoPE 不在发布代码支持列表 | README L175 |
