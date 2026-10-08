# SGLang / Two-Level 代码与测试审查_by_gpt

审查日期：2026-10-08。只读审查，未修改或推送 SGLang 源码。

审查分支 **two-level-indexer**，固定 commit **ee256310b2e8898d9ed3607da9c325b872fc10ff**。远端再次核对仍为此版本；main 不含本次 Two-Level 开发内容，因此没有用 main 代替用户实际实现。

## 优先处理的六项问题

| ID | 范围 | 问题与可复核影响 | 修复重点 |
|---|---|---|---|
| C1 / P1 | SGLang 默认 eager TLI | near 候选不足时 -inf top-k 下标仍进入注意力；构造条件下至少 576 个非法候选 | 按选中分数过滤并写 sentinel，覆盖单请求回退 |
| C2 / P1 | TASK.md CUDA graph TLI | 短序列 sink 与滑窗重复；S=160 的反例输出 0.75，正确值 0.6 | 短行全量唯一位置+sentinel，或暂回退 eager |
| C3 / P1 | Two-Level near 聚类 | 簇分未乘 softmax_scale，与已缩放尾段分直接 top-k，排名翻转 | 两源使用同一个实际 scale；两种聚类均回归 |
| T1 / P1 | E98 离线 mass 扫描 | 实际 grid 点 far 池只有512 token却取2048，至少1536个池外位置计入mass | 约束有效池容量、屏蔽无效top-k、重算受影响网格 |
| T2 / P1 | E104 评分 | 不同完成任务集合的AVG相减，可虚构+50优势 | 完整集AVG或同一交集的显式partial配对比较 |
| T3 / P2 | KVCache baseline复用 | 改预算/协议后仅凭行数复用旧预测，实际新配置没执行 | 配置/代码/模型/样本指纹、独立完成清单 |

P1 表示优先修复正确性或实验结论可信度；不代表已观察到生产事故。五项数值问题已核对源码并运行 CPU 算术/控制流反例；T3 为确定的静态控制流缺陷。当前环境无 Torch，未运行原 PyTorch/CUDA 全链、模型精度或性能基准。

## 源码定位（不可变链接）

- [C1: python/sglang/srt/layers/attention/tli/indexer.py:654–662](https://github.com/Chosen-David/sglang/blob/ee256310b2e8898d9ed3607da9c325b872fc10ff/python/sglang/srt/layers/attention/tli/indexer.py#L654-L662)
- [C2 选择: python/sglang/srt/layers/attention/tli/indexer.py:2468–2495](https://github.com/Chosen-David/sglang/blob/ee256310b2e8898d9ed3607da9c325b872fc10ff/python/sglang/srt/layers/attention/tli/indexer.py#L2468-L2495)
- [C2 引擎路由: python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py:710–718](https://github.com/Chosen-David/sglang/blob/ee256310b2e8898d9ed3607da9c325b872fc10ff/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py#L710-L718)
- [C3 near分数: two-level-attention/sparse_attn/indexer/tli_indexer.py:712–728](https://github.com/Chosen-David/sglang/blob/ee256310b2e8898d9ed3607da9c325b872fc10ff/two-level-attention/sparse_attn/indexer/tli_indexer.py#L712-L728)
- [C3 混合评分: two-level-attention/sparse_attn/indexer/tli_indexer.py:1048–1064](https://github.com/Chosen-David/sglang/blob/ee256310b2e8898d9ed3607da9c325b872fc10ff/two-level-attention/sparse_attn/indexer/tli_indexer.py#L1048-L1064)
- [T1: two-level-attention/exp/trace/analyze_e98_abg_full_grid.py:98–129](https://github.com/Chosen-David/sglang/blob/ee256310b2e8898d9ed3607da9c325b872fc10ff/two-level-attention/exp/trace/analyze_e98_abg_full_grid.py#L98-L129)
- [T2: two-level-attention/exp/trace/analyze_e104_ruler_32k.py:48–69](https://github.com/Chosen-David/sglang/blob/ee256310b2e8898d9ed3607da9c325b872fc10ff/two-level-attention/exp/trace/analyze_e104_ruler_32k.py#L48-L69)
- [T3: two-level-attention/exp/trace/pred_kvcf.py:169–181](https://github.com/Chosen-David/sglang/blob/ee256310b2e8898d9ed3607da9c325b872fc10ff/two-level-attention/exp/trace/pred_kvcf.py#L169-L181)

## 测试为什么可能漏掉

- 修复测试主要直接调用 fused L2，未覆盖 C1 的默认 eager 分支。
- 集合/mass coverage 验收会丢掉重复次数，无法排除 C2 的注意力重复计权。
- 主要聚类全链测试的预算实际令 far_budget=0，不能证明 far 聚类选择有效。
- near 尾段测试使用30倍亮点，无法验证 C3 的接近 top-k 边界排序。
- E110 的旧树参考是硬编码绝对目录，没有冻结基线SHA，可能实际比较同一份代码。

建议先补失败回归并修正 C1/C2/C3，再重算受影响离线结果；不需要立刻重跑全部数据集。当前版本已修的 S-T002 三处问题没有重复计为未修复。

## 主控复核补充

已定位实际新路径 `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py`，710–718行调用 backend veto；C2 的静态引擎接入链已核实，补足下方首轮报告记录的缺口。实际部署是否启用该路径、GPU输出仍需目标环境对拍。

未找到此版本被跟踪的 E109 执行脚本与 score_v2.py，无法审查当前外部运行目录中的调度/AVG5实现。E98/E104问题不自动推断为E109已受污染。审查不构成全仓无bug保证。


---

## 核心独立审查原始记录

# SGLang TLI core read-only audit

Source snapshot: `Chosen-David/sglang`, branch `two-level-indexer`, SHA `ee256310b2e8898d9ed3607da9c325b872fc10ff`, exported under `/workspace/scratch/c12f3d9f92bd/sglang-source`. Root `TASK.md`, `agent_doc/task/TASK.md`, and S-T002 were read. No source file was edited and no server, GPU job, dependency install, or download was run.

Two strong remaining findings follow. These are **static source findings with executed pure-Python mathematical witnesses**, not executed Torch/GPU tests. Both system Python and the existing primary runtime Python lack Torch. Witness values are saved alongside this report in `core_witnesses.json`.

## C1 — P1: default eager decode still returns invalid top-k slots as real tokens

**Root:** `python/sglang/srt/layers/attention/tli/indexer.py:654–662`, especially raw `i_n = torch.topk(...).indices` at 660 and concatenation at 661. Absolute source: `/workspace/scratch/c12f3d9f92bd/sglang-source/python/sglang/srt/layers/attention/tli/indexer.py`.

**Reachability:** `backend.py:614–635` takes per-request `select()` when there is one sparse request, or `SGLANG_TLI_BATCH_SELECT=0`. `config.py:62` defaults `use_l2_kernel=False`; thus this is the standard eager per-request route, not a dead fallback. `q_agg=max` also bypasses the fused route at `indexer.py:590`. `backend.py:983` accepts any returned index below that request's sequence length, and 1019–1022 recomputes actual attention logits for these slots. The consumer has no access to the selector's `-inf` scores.

**Trigger:** the near/far L2 pool has fewer valid L1 candidates (score > -inf) than its reserved quota. With default parameters and `S=16384`, construct a one-KV-head query/key case where L1's 128 highest-scoring blocks are far blocks 2 through 129. This is realizable by making their coarse-subspace dot products higher than every other block. `select()` additionally forces the final three blocks (`indexer.py:513–516, 567–581`). Thus there are 8192 far candidates and only 192 valid near candidates (including forced +inf scores). The near quota remains `1024−256=768`. At least **576** slots returned by the near top-k necessarily have score `-inf`; nevertheless all their raw indices are in `[0,S)` and pass consumer validity. The precise tie ordering is irrelevant to this counting proof.

**Impact:** at least 576 tokens excluded by L1 or the near partition enter actual attention. Some tie outcomes can duplicate far selections. Output can depend on batch size and kernel/fallback configuration. This is the same *class* of sentinel defect as S-T002 bug1, but a **different still-unfixed reachable eager location**: the repaired `kernels.tli_l2_partition_topk` is not being reported as broken.

**Suggested fix:** gather top-k values for each branch, convert `-inf` selections to sentinel `S`, including the final unpartitioned return where candidates may be insufficient. Add a regression covering the above near-starvation construction across default eager, fused, and batched selection, checking that every valid output belonged to that head's permitted valid pool (score > -inf). The existing `test_b1_b2_fix.py:35–66` bug1 case directly tests the fused kernel, so it does not cover this eager branch.

## C2 — P1: TASK.md graph decode double-counts protected tokens on short real requests

**Root:** `indexer.py:2468–2495` concatenates sink positions and sliding-window positions without deduplication or a short-row identity replacement. `backend.py:397–414` exempts `S <= token_budget` from graph veto on the assumption that selection is dense-equivalent. Absolute backend source: `/workspace/scratch/c12f3d9f92bd/sglang-source/python/sglang/srt/layers/attention/tli/backend.py`.

**Reachability:** enable TASK.md partition mode, e.g. `ALPHA=.125, BETA=.125, GAMMA=.375` (normal positive partition configuration), with graph decode enabled. `backend.py:661–685` explicitly routes all rows, including short real requests, through batched selection. `indexer.py:1817–1818` forwards them to `_select_decode_taskmd`. A real request with `S=160` is not vetoed (`1024 < 160 <= 2048` is false). Its metadata is treated as a real request (`backend.py:465–470`, only `L<=1` is ignored).

**Witness:** defaults give sink `[0,128)` and sliding window `[32,160)`. Far/near L1 pools are empty at this length: the source `_taskmd_regions(160)` function was extracted using AST and executed without importing the package; it gives `near_blks=far_hi_blk=swa_lo_blk=2`. The batched formulas give the same boundaries. The output therefore contains 256 valid protected lanes covering only 160 distinct tokens, with positions 32 through 127 duplicated. Every duplicate is below `S`, so `backend.py:983` accepts it. Choose Q=0 (all true attention logits zero), V=1 for those 96 overlap tokens and V=0 elsewhere. Dense/eager output is **96/160 = 0.6**; the graph sparse output is **192/256 = 0.75**. This witness is independent of top-k ties and GPU arithmetic.

**Impact:** short prompts get an incorrect attention distribution with graph decode, even though they are under both the token budget and dense threshold. The behavior changes when graph execution is disabled. This is separate from the already-fixed prefill early-row issue: `_select_batched_taskmd` has an identity-plus-sentinel repair at 1666–1677, while `_select_decode_taskmd` does not.

**Suggested fix:** for real rows with `S <= token_budget`, replace the selected lanes with one occurrence of each position `[0,S)`, padding the fixed output width with a sentinel, or veto TASK.md short rows to eager dense until equivalence is implemented. Merely masking sink/window overlap is insufficient to guarantee all short rows retain every causal token under independent quotas. Add TASK.md graph/eager output comparisons at lengths 129, 160, 255, 256, 600, and 1024, including nonconstant V; a set/mass-coverage check alone misses duplicates. Existing `test_tli_m5.py:317–359` checks old-profile lengths 600/1500 and does not establish TASK.md protected-region uniqueness.

## Scope and limits

Reviewed configuration, scalar/batched TASK.md selection, default eager selection, sentinel consumers, graph routing and relevant previous fix tests. No claim is made about measured model accuracy or throughput. The exported source subset lacks `python/sglang/srt/model_executor/cuda_graph_runner.py`; external engine hook registration was not independently traced beyond the backend's graph entry points and dedicated graph tests. C2 is therefore conditional on the backend's supported graph path being enabled and reached. No findings are asserted against the three already-fixed S-T002 locations themselves.

---

## 聚类与五种组合覆盖

# 两种聚类原语及五种组合：只读补充审查

快照：`Chosen-David/sglang`，`two-level-indexer`，`ee256310b2e8898d9ed3607da9c325b872fc10ff`。源根目录 `/workspace/scratch/c12f3d9f92bd/sglang-source`；未导出的生产 Triton 原语通过 `/workspace/scratch/c12f3d9f92bd/sglang-audit` 中固定 SHA 的 `git show` 只读查看。本报告不重复 C1/C2。未安装依赖、未运行 GPU、未改源。Torch 在现有两个 Python 运行时均不可用。

## C3 — P1：near 聚类分与细筛尾段分缺少统一 softmax_scale，选取排名会翻转

**位置：** `two-level-attention/sparse_attn/indexer/tli_indexer.py:722–728` 与 `1048–1061`；分数来源见 `605–606`、`675–683`。绝对源路径：`/workspace/scratch/c12f3d9f92bd/sglang-source/two-level-attention/sparse_attn/indexer/tli_indexer.py`。

`compute_score()` 把未缩放的 q 保存为 `_last_q`，但 `score_fine` 使用 `q * softmax_scale` 算点积。`_near_token_score(_last_q)` 使用未缩放 q 与簇心算点积，只修正了 GQA 的 mean 聚合，没有应用 softmax_scale。随后 near 侧直接把这一簇分覆盖到 `score_fine` 的 group-mean 切片，再在同一个池内 top-k。注释称两者量纲一致，实际不一致。

**可达条件：** `near_select=cluster` 或 `sim_greedy`，近端簇已构建，且同一个 near 池包含簇覆盖段与未覆盖的细筛回退段。`_update_near_cluster()` 在 431–440 把覆盖右边界向下对齐到块，而消费端窗口起点为精确 token 位置；例如 `S=4126, bs=64, swa=128` 时，覆盖到 3968，`[3968,3998)` 是回退尾段。正常解码每个非对齐 step 都可能进入这种混合情况；影响 ccluster_kmeans 和 ccluster_sim_greedy，far-only cavg 不受这个 near 混合缺陷影响。

**已执行的纯 Python 算术反例：** D=128 时 scale=0.08838834764831845。设簇覆盖 token 的原始代表点积为 1、尾段 token 的原始细筛点积为 2（信号可放在二者共享维度，排除子空间差异）。代码实际比较 **1 vs 0.1767766953**，偏选簇段；统一缩放后应比较 **0.08838834765 vs 0.1767766953**，应选尾段。给 near 配额 384，再加 383 个必胜候选，这个逆序直接改变最后一个入选 token。结果见 `cluster_scale_witness.json`。这是源码量纲核验和执行的算术见证，不是 Torch 全链或 GPU 对拍。

**影响：** D=128 时，簇覆盖区域的正分相对于回退尾段被额外放大约 11.31 倍；负分也发生不等比例变化。因此不能把它解释成对整个池统一乘常数的无害操作。块边界前后的选取存在额外位置偏差，可能污染聚类组合与非聚类组合的效果比较，但未实测模型精度下降幅度。

**建议修复：** 在同一评分协议中给 near 簇分补实际调用的 `softmax_scale`，或以可靠方式把回退分还原到一致量纲。不要只硬编码 `sqrt(128)`；调用接口已有 scale。回归用非对齐 S、明确位于 top-k 边界的簇段/尾段两个分数，以及 G=1/G>1；同时覆盖 kmeans 和 sim_greedy。现有 E110 T6 虽覆盖尾段可召回，但构造了 30 倍亮点，且配额很大，不能检验边界排序是否同量纲。

## 已检查的原语与测试范围

| 对象 | 已核查内容 | 现有直接测试 | 本轮结论/限制 |
|---|---|---|---|
| kmeans 原语 | `_gpu_kmeans` 579–590；far 构建 349–372；near 构建 421–466；簇分展开 691–728 | E110 T4 有 kmeans 双侧全链路烟测，T7 有 cavg 新旧对拍 | 没有本轮执行的独立数值 oracle；看到的是 dot-argmax 指派与算术均值更新，未把其命名/目标函数差异提升为已证业务 bug |
| sim_greedy Python | 504–558：按序余弦阈值、running sum/count、平方范数更新、容量扩容；374–419 增量 far；1085 以后 clear | E110 T1 手算归并/新簇/mean；T2 对照 e64a；T3 增量=冷启动；T8 clear；T9 互斥 | 对这些路径未发现另一个达到本报告证据标准的新 bug；未运行 Torch |
| sim_greedy Triton | 生产 `sparse_attn/indexer/greedy_triton.py`：逐 head 顺序 token、tile argmax、写回状态、容量与分块 wrapper；tli_indexer 469–501 调度 | `exp/trace/test_e113_greedy_triton.py` T1–T5；`test_e113b_kernel_integration.py` T1–T5 开关/增量/重置/回退 | 静态比较了状态与调度；未做 GPU 编译、数值边界或性能验证 |

## 五种组合的覆盖核对

下表是**测试源码中存在的断言覆盖**，不是声称本轮已运行通过。检索了固定 SHA 的全部 two-level `test*` 文件名以及导出的相关测试内容。

| 用户组合 | 可定位覆盖 | 具体盲区 |
|---|---|---|
| aavg | `test_e112_sglang_port.py:75–85`：分区与 (0,0) 单池；G=1 集合等价、G=4 差异报告 | 使用集合对拍不能发现重复槽位，不能代替最终 attention 输出验证；未覆盖全 αβγ 网格 |
| mminmax | 同一文件：分区、far 激活配置、(0,0) 单池，G=1/G=4 | 同上；少量参数点不是全面组合证明 |
| mavg | E112 冠军参数与 far 激活参数；E110 T7 默认分数管线新旧对拍 | 旧实现相等只证明回归一致，不能排除双方共享错误 |
| cavg | E110 T5 对照及 T7 新旧 mask 对拍；T4 有 cavg_sim | T5 β=.25、γ=.5，T7 β=.25、γ=.75 都使 K2_mid=768 全给 near，far 最终配额为零；没有用这些用例证明 far 簇分真正决定选取 |
| ccluster_sim_greedy（另含 kmeans 对照） | E110 T1–T9；E113b T1–T5；原语状态与增量覆盖较丰富 | E110 T4/T5 与 E113b T3 均 β=.25、γ=.5，near=768、far=0；far 原语虽被构建并直接验证状态，全链 mask 对拍并未验证活跃 far 配额。T6 未检查尺度一致性，C3 可漏过 |

E110 的覆盖依赖还存在可复现性限制：`test_e110_ccluster.py:40–42,70–84` 硬编码读取 `/home/wangyuanshuo02/sglang/two-level-attention` 为“旧树”和 e64a 参考；它不是固定 SHA 的自足回归基线。若在该主树直接运行而非旧 worktree，“新旧”可能实际加载同一份文件，不能据此保证跨版本回归。这里将其归为测试可信度/可迁移性缺口，不冒充另一个已复现算法错误。

建议最小补强：为 cavg/ccluster 的两种聚类方式各加一个 `far_budget>0` 且 `near_budget>0` 的精确排名 oracle，并加入非对齐长度下的 near 混合分数排序测试；保留已有增量、clear、fallback 测试即可。无需把本轮只读审查扩大成全数据集重跑。

---

## 测试脚本独立审查

# 测试与实验脚本只读复核

固定源码：Chosen-David/sglang `two-level-indexer`，`ee256310b2e8898d9ed3607da9c325b872fc10ff`。阅读了 TASK.md、agent_doc/task/TASK.md、S-T001 和 S-T002；未修改源码、未加载模型、未启动 GPU/服务、未安装依赖。以下是 3 个可由源码保证触发的独立问题。CPU 验证只使用 Python 标准库，设置 `PYTHONDONTWRITEBYTECODE=1`；环境没有 torch，未声称完成 GPU/PyTorch 实测。

## 1. P1：E98 离线 mass 扫描把 coarse pool 外的 -inf 槽位当成有效 token

- 位置：`two-level-attention/exp/trace/analyze_e98_abg_full_grid.py:98-101`；可达扫描及约束 `:104-129`。
- `select_sub()` 将池外 token 分数设为 -inf，然后直接对整个 mid 长度做 `topk(n_tokens)`，将所有返回下标 scatter 为 True。未检查选中 score 是否有限，也未验证 far coarse pool 容量足以满足细筛预算。
- **真实扫描中的最小参数反例**：mid_len=8192、a=.5、b=.875、g=0。源码计算 near_L=4096、far_L=4096、nb_near=56、nt_near=0、nt_far=2048。现有约束都通过，但 far 只获 8 页 × 64=512 个候选，topk 却要求 2048，故至少 1536 个池外 token 被无条件选为 True。任意 topk tie-breaking 都无法避免此错误；具体哪些池外位置取决于实现，不能声称必然都落在 near。
- 影响：该臂不再是声明的两级筛选，cov_mass 对未经粗筛入围的位置也计入概率质量。mass 比较、排名和可视化受到污染；这是离线 trace 指标问题，不能据此推断当前 E109 e2e 精度被污染。
- 与已修 bug 的关系：S-T002 bug1 修复的是 sglang fused L2 kernel。本条是另一份仍有问题的离线实验实现，不是宣称原修复未完成。
- 建议：扫描约束补 `far_pages*BS >= nt_far`（边界页后还需实际有效候选数校验）；topk 后只 scatter 有效 score 的下标。加入此高 beta、低 gamma 反例以及分区边界页反例，并重算受影响 mass 网格。
- CPU 验证：通过 AST 提取并执行该文件原始预算赋值，得到上述参数；用标准库构建 512 个 finite + 7680 个 -inf 的分数并取最大 2048，确认 1536 个非法槽位。没有运行 torch/GPU。

## 2. P1：E104 允许部分收割，却用不同任务集合的 AVG 直接判定方法差值

- 位置：`two-level-attention/exp/trace/analyze_e104_ruler_32k.py:48-57`、`:62-69`。
- 无完整 100 条预测的任务被记 None；AVG 随即对每个 arm 各自非空任务取平均。之后只要两边 AVG 非 None，就直接生成 `delta_main_minus_ref['AVG']`、`delta_ref_minus_fullkv['AVG']`、`delta_ref_minus_quest['AVG']`。任务交集、完整性和覆盖数未纳入判断。
- **反例**：两个方法真实结果完全相同：easy=100、hard=0。A 仅 easy 完成，B 两项均完成，输出 A.AVG=100、B.AVG=50、delta.AVG=+50，而共同任务差值和真实完整差值都是 0。11 个任务版本只需把其他 9 项设 None 即同样触发。
- 影响：脚本明确允许部分收割，在常见的多臂完成进度不一致时，会产出没有可比意义的总体优势/劣势；输出 note 仍描述“11 任务×n=100”。完整收官且所有任务都齐时，本条不触发。
- 建议：正式 AVG 仅在完整任务集合齐备时输出；部分结果另列 `partial_avg` 和任务列表。若需要中途比较，所有 arm 使用同一个显式交集，结果标为 partial paired comparison。
- CPU 验证：从该文件 AST 提取实际 `row['AVG']` 赋值在上述两个 row 上运行，输出 100、50 和伪 delta=50。无需 GPU。

## 3. P2：KVCache baseline 断点复用不包含实验参数，改变配置会静默复用旧输出

- 位置：`two-level-attention/exp/trace/pred_kvcf.py:169-181`；参数入口 `:134-145`，方法应用 `:162-164`，实际协议传入 `:183-185`。
- 输出目录/文件匹配只包含 method、可选 output_suffix、task。复用条件仅是任一旧文件行数 >= args.n，未验证 max_capacity、window_size、kernel_size、pooling、sink_guard、split_question、attn_impl 或模型/数据版本。
- **确定可达反例**：先运行 `--method snapkv --task hotpotqa --max-capacity 1024 --n 200`，再以同一默认 suffix 运行 `--max-capacity 2048`。只要第一个输出有 200 行，第二次在 get_pred 前打印 SKIP 并 return，2048 实验实际没有执行。同理默认 full-prefill 的输出会使新增 `--split-question` 被 SKIP。
- 影响：参数扫描/修复后重测可被误判为已完成，后续 scorer 继续吃旧配置预测，产生“参数不影响精度”或旧结果被新标签解释的风险。不是声称已发生于 S-T001；当前脚本的复用条件本身即可证明。
- 建议：保存可校验 manifest/config hash，将所有影响预测的配置、代码版本、数据身份写入输出；仅同 fingerprint、同样本 ID 的完整输出可复用。手工 output_suffix 是可选逃生口，不是当前默认行为的校验。
- 验证：控制流/路径键静态核验；未运行会加载模型的 main。

## 次级观察与覆盖边界

- `benchmark/LongBench/pred.py:178-182` 对所有 tokenizer 无条件删除 question 后缀的第一个 token。对不自动插入 BOS 的 tokenizer，这会删除真实 prompt 内容；使用 no-BOS character tokenizer 的 CPU 反例将 `Question: What?` 变成 `uestion: What?`。建议 tokenize 完整 prompt 一次后在 token ID 层分段。此导出没有真实 tokenizer 文件，故这里未声称已验证当前 Qwen3 tokenizer 配置，列为待配置核实项。`pred_kvcf.py:89` 的 split-question 分支有同样处理。
- `benchmark/LongBench/pred.py:224-239`、`pred_kvcf.py:101-109` 只检查第二个及以后的生成 token 的 EOS，第一个 token 已为 EOS 时仍继续 decode；源码确定可触发，但尚无真实输出样本证明影响规模。
- `run_scripts/e89_moba_smoke.sh` 无 set -e/显式 return-code 检查，尾部 echo 会掩盖 python 失败；属于较旧脚本，优先级低于上面三个指标污染问题。
- 没有发现当前导出中的 E109 执行脚本或 score_v2.py；父代理也已核实 git tree 无这些脚本。不能审查 S-T001 当前 12 卡调度、resume 或当前 AVG5 聚合实现，不以旧 E98/E104 推断它们同样有问题。
- 研究 microbench 所读文件通常已显式同步 CUDA；没有将“没有 GPU 验证”冒充计时错误。没有重复报告 S-T002 已修复的 q_agg/max 或 PCA TP bug。

---

## 可重放 CPU 见证脚本

下列脚本不导入Torch、不执行GPU；T2执行冻结源码中的AVG赋值AST，其他项核验源码公式的数学反例。需要源码目录sglang-source与报告目录同级。

```python
"""CPU mathematical/control-flow witnesses, not execution of Torch/CUDA kernels."""
import ast
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "sglang-source"

def main():
    results = {}
    # C1: reproduce the quota and candidate count, independent of -inf tie order.
    scores = [1.0] * 192 + [float("-inf")] * (16384 - 192)
    selected = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:768]
    invalid = sum(scores[i] == float("-inf") for i in selected)
    assert invalid == 576 and all(0 <= i < 16384 for i in selected)
    results["C1"] = {"quota": 768, "finite_candidates": 192, "invalid_admitted": invalid}

    # C2: protected regions are concatenated; softmax for Q=0 is uniform per lane.
    S = 160
    sink, window = list(range(128)), list(range(S - 128, S))
    overlap = set(sink) & set(window)
    lanes = sink + window
    dense = len(overlap) / S
    sparse = sum(i in overlap for i in lanes) / len(lanes)
    assert len(overlap) == 96 and dense == .6 and sparse == .75
    results["C2"] = {"S": S, "lanes": len(lanes), "unique": len(set(lanes)),
                     "dense": dense, "duplicated_sparse": sparse}

    scale = 1 / math.sqrt(128)
    actual_cluster, fallback = 1.0, 2.0 * scale
    correct_cluster = 1.0 * scale
    assert actual_cluster > fallback > correct_cluster
    results["C3"] = {"scale": scale, "actual_cluster": actual_cluster,
                     "fallback": fallback, "correct_cluster": correct_cluster}

    # T1: one actual E98 grid point passes its constraints yet exceeds far pool.
    mid, a, b, g, BP, BS, budget = 8192, .5, .875, 0, 64, 64, 2048
    near = int(a * mid)
    pages = int(round(BP * b))
    nt_near = min(int(pages * BS * g), budget)
    nt_far = max(64, budget - nt_near)
    assert near >= pages * BS and nt_far <= mid - near
    capacity = (BP - pages) * BS
    assert nt_far - capacity == 1536
    results["T1"] = {"far_pool": capacity, "far_topk": nt_far, "invalid_min": nt_far-capacity}

    # T2: execute the original AVG assignment from the source AST.
    path = ROOT / "two-level-attention/exp/trace/analyze_e104_ruler_32k.py"
    tree = ast.parse(path.read_text())
    stmt = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
                        and t.value.id == "row" and isinstance(t.slice, ast.Constant)
                        and t.slice.value == "AVG" for t in n.targets))
    code = compile(ast.fix_missing_locations(ast.Module(body=[stmt], type_ignores=[])), str(path), "exec")
    rows = [{"easy": 100, "hard": None}, {"easy": 100, "hard": 0}]
    for row in rows:
        exec(code, {"row": row, "vals": [v for v in row.values() if v is not None]})
    delta = rows[0]["AVG"] - rows[1]["AVG"]
    assert delta == 50
    results["T2"] = {"rows": rows, "invalid_unpaired_delta": delta, "paired_delta": 0}
    print(json.dumps(results, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
```

### 聚类尺度反例结果

```json
{
  "D": 128,
  "softmax_scale": 0.08838834764831845,
  "covered_raw_dot": 1.0,
  "tail_raw_dot": 2.0,
  "actual_covered_score": 1.0,
  "actual_tail_score": 0.1767766952966369,
  "aligned_covered_score": 0.08838834764831845,
  "actual_last_selected": "covered",
  "aligned_last_selected": "tail",
  "E110_T4_T5_E113_T3_far_budget": 0
}

```
