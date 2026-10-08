# SGLang TASK.md 五种 method 组合：Kernel 性能设计与优化建议

作者标识：by_gpt  
日期：2026-10-08  
修订：v2，按更新后的代码优化技能重审；新增逐路线判别实验、后端切换条件与测量合同。  
核查版本：`Chosen-David/sglang`，`two-level-indexer` 分支，`9049ba8759896e5efc13f287c442fb72d5e1b563`。

> 本文给出面向高性能实现的工程设计，不声称已达到硬件极限。已完成源码核查、两个代码 Agent 的独立分析及主执行者复核；没有实施本文改动，没有运行新的 GPU 性能测试，没有修改或推送 SGLang。
>
> 范围按用户最新要求收窄：仅依据 SGLang `TASK.md` 的 **mavg、aavg、mminmax、cavg、ccluster** 五种基础组合；cluster 分别设计 **kmeans 与 sim_greedy** 两种实现。下文展开为七条具体路线，不增加新的组合或研究方向。相关缓存、并行、数值合同和验收均服务于这五种组合。

## 1. 主要判断与工程顺序

本次修订的实质变化：把“推荐某个 kernel”补成“在什么证据下实现、与谁比较、什么结果会否定它”。§3.2 给七条路线各自的首实验；§9.1–9.3 给摊销与 dispatch 判据；§11.5 固定状态型 kernel 的计时和候选证据；§13 记录本次借鉴与拒用的 kernel Agent 方法。这里的极致性能是受正确性、质量和维护预算约束的优化目标，当前尚无达到极限的实测结论。

当前最大的机会是：**让实际运行的 method 组合真正执行候选式细筛，补齐已有融合 kernel 对组合语义的支持，消除未使用的计算与主机同步。** 然后才是近远端双流、更复杂的 persistent kernel 或硬件专用指令。

1. 明确参考语义：HF 实验实现、SGLang taskmd 实现、用户目标流程是三个需要对齐的对象。
2. 先做小范围等价优化：K 零填充移出循环、按 method 删除死计算、使用长度提示消除同步。
3. 普通 avg/minmax 组合：候选压实、候选式 L2、反量化与打分融合、避免全序列分数物化。
4. cluster 组合：区分“当前簇代表分直接选 token”与“簇粗筛后逐 token 细筛”的目标路线。
5. sim_greedy：已有 Triton 作为基线；优先状态预分配和独立链批处理，再评估 CUDA persistent。
6. 以端到端 TTFT、TPOT、吞吐、p95 和精度判定，不以 launch 数、SM 数或估算 FLOPs 直接宣布收益。

## 2. 必须冻结的语义合同

### 2.1 三种模式分开验收

| 模式 | 目的 | 可以声称什么 |
|---|---|---|
| C：当前代码兼容 | 不改变当前参考实现的选择与结果 | 通过对拍后可称实现加速 |
| R：目标流程修正 | 满足用户“子空间→near/far→各端 L1→各端 L2”要求，修复偏离 | 算法/语义修正，必须重测精度与基线 |
| A：近似探索 | 低精度排序、冻结簇心、滑窗近似维护等 | 独立算法消融，不称无损 kernel |

不要把 C、R、A 合并成一次提交后只测速度。即使更改的结果更合理，也需要记录旧行为和新行为。

### 2.2 参数、区域与预算

记真实因果长度为 `S`，块长 `B`，query 数为 `Nq`，KV head 数为 `Hkv`，GQA 比为 `G`；三种特征维分别记录 `D1`（L1）、`D2`（L2）和 `Dc`（cluster）。禁止只写“full=128”而隐去 L2 或 cluster 实际使用的维度。

目标合同中，Sink 与 SWA 是保护集合；短序列重叠时按集合并集计数。`M = S - |Sink ∪ SWA|`。near/far 只分割 mid 集合。建议所有路径消费同一个边界描述符，而不是分别用真实 `S`、padding 后 `padS` 和静态 near 长度重算。

当前普通双池代码的 γ 口径大致是：

```text
Kmid = max(0, K2 - fixed_count)
K1_near = round_rule(K1 * beta)
K1_far  = K1 - K1_near
K2_near = min(integer_rule(K1_near * B * gamma), Kmid)
K2_far  = max(0, Kmid - K2_near)
```

其中 `round_rule`、最少一个 near 块等实现细节必须从选定参考路径固化，不能把 γ 擅自解释成 `K2_near=γ*Kmid`。边界块、候选不足、零预算是否让渡都写入合同；没有授权的预算让渡不得隐藏在 kernel 中。

现状需独立核验：HF near 边界使用 `padS`，聚类构建用真实 `S`；near 长度的定义与扣除 SWA 的位置可能不一致。prefill 缺位以 token 0 补齐而最终算子缺少有效掩码，也可能改变 softmax。优化前先构造短序列、块边界和不足预算例子核对，不能用加速掩盖这些问题。

### 2.3 GQA、排序与数值

HF 普通 L2 在 L1 候选域上计算每个 query head 的 softmax，随后组内平均：

\[
p_t=\frac1G\sum_{g=1}^G\frac{\exp(z_{g,t})}{\sum_{u\in C_g}\exp(z_{g,u})}.
\]

mask 外项按零概率处理；实际 `C_g` 是否共享由配置决定。一般不能替换成 `sum_g(z[g,t])`，也不能让 near/far 各自归一化。SGLang taskmd 当前采用 group-sum logits，属于另一参考语义。

精确融合必须保留：输入缩放、特征维、量化解码、必要的 dtype 舍入、归约次序、NaN 处理、有效候选、tie 行为。FP32 并不自动意味着逐位一致。原路径零填充到全 D 是为复现归约树，不应仅因“非零只有 D2 维”就删掉并宣称精确等价。

`torch.topk` 的并列选择不应假定为稳定最小 token ID；如需跨 kernel 可复现，显式引入 `(score, token_id)` 次序并将其作为 R 模式合同变更，给旧新基线同步应用。Greedy 的 `argmax` 则必须保留当前最小簇 ID 的平局规则。

## 3. 全 method 覆盖矩阵

表中 K 表示当前命名为 kmeans 的聚类原语，S 表示 sim_greedy。K 的源码实际上执行 `argmax(x @ centroid.T)` 后求簇均值，不是带距离范数项的标准欧氏 Lloyd。

| ID | 组合（far, near） | 当前可核验路线 | 本文对应设计 |
|---|---|---|---|
| M1 | mavg=(minmax,avg) | HF 与 SGLang taskmd 都有，L2/GQA 合同不同 | §6.1 |
| M2 | aavg=(avg,avg) | 同上 | §6.2 |
| M3 | mminmax=(minmax,minmax) | 同上 | §6.3 |
| M4 | cavg-K=(cluster-K,avg) | HF far 簇代表分选 token；near 普通两级 | §7.1 |
| M5 | cavg-S=(cluster-S,avg) | HF 名称含 cavg_sim/cavgsim，已有 Triton greedy | §7.2 |
| M6 | ccluster-KK=(cluster-K,cluster-K) | HF 双侧簇代表分，near 有尾部回退 | §7.3 |
| M7 | ccluster-SS=(cluster-S,cluster-S) | HF ccluster_sim；far 增量、near 重建 | §7.4 |

### 3.1 七条实现路线的首选 kernel 组合

这些是首轮实现候选，不是未经测量的最快配置。A/B 对手应是同 method 的当前可达实现。

| 路线 | Prefill 构建与选择 | Decode 构建与选择 | 首选后端 |
|---|---|---|---|
| mavg | block 统计共享加载；分区 L1；候选式 L2 | append 更新必要统计；区域 GEMV；compact+refine | Triton 统计/融合；多 query 与 cuBLAS 对照 |
| aavg | 仅 block sum/count；batched avg L1；候选 L2 | 仅尾块 avg 更新；单原语双池选择 | Triton；大批规则 L1 对照 cuBLAS |
| mminmax | 仅 min/max；正负 Q 分解 L1；候选 L2 | 增量 min/max；融合 GEMV 与候选分数 | Triton；不额外维护 avg |
| cavg-kmeans | far grouped assignment/update；near avg；按 C/R 选择 | 按边界重建 far；near append；分区查询 | cuBLAS/Triton assignment + CUDA/Triton reduction |
| cavg-sim | far greedy 冷启动；near avg | far greedy append；near append | 现有 Triton 基线；CUDA 单 CTA/链候选 |
| ccluster-kmeans | 两 region 的 grouped assignment/update | 各自边界触发重建；双侧簇查询 | 同一个 K 原语库，region 作为任务轴 |
| ccluster-sim | 两独立 greedy 链，分别构建 | far append、near 按窗口重放 | 同一个 S 原语库；独立链批处理 |

SGLang taskmd 当前只接受 `avg|minmax` 的 far_method/near_method。不能因 HF E113b 写了“生产路径”就认为 M4–M7 已接入 SGLang serving backend。

还必须按阶段区分基线：HF 在全序列 fine score 后做 L1 掩码；SGLang taskmd prefill 也有全宽反量化/fine score；但 **SGLang taskmd decode 已先压实候选，再 eager gather/dequant/dot**。decode 的首要候选是消除这些中间量与调用开销，不是再次声称“从 S 降到 Tc”；其 head union 冗余与 compaction 另作消融。HF `prepare_index` 还存在全序列统计/量化重建成本，端到端分析必须计入，不能只测 selector。

### 3.2 每条路线的第一项判别实验

以下为待执行协议。所有时间、计数器和收益目前未知；不是已观察到的硬件瓶颈。C/R/A 依 §2.1 分开，按固定同一路由测试，不能拿旧 Python 路径代替已有 Triton 对手。

| 路线 | 源码证据与首假设 | 最小对照与观测量 | 会否定该方向的结果 |
|---|---|---|---|
| M1 mavg | taskmd prefill 全宽 L2；decode 已候选化但 eager 物化索引/反量化张量 | prefill 固定 C 合同比较 full-score 与候选式路线；decode 首轮保持原 tok/top-k，仅融合 gather/dequant/dot。分别记录 `Tc/S`、head union 膨胀、中间量、分段及 selector 时间 | 候选集合漂移；prefill compact+不规则加载抵消收益；decode 融合归约不满足合同或资源压力增加 |
| M2 aavg | 当前 taskmd 仍存在 min/max 计算；是否为死计算取决于其他消费者 | 先做消费者读集审计；仅关掉确实不被读取的 min/max 分配/维护，对照 build、append、显存与输出 | 某个分流/回退仍读取 min/max；或 shared build 的分支开销抵消收益 |
| M3 mminmax | 当前 taskmd 已共享全域 sc1，再分两池 mask/top-k；剩余机会在分数物化与选择 | 以当前共享 sc1+双池 mask/top-k 为基线，对照融合区域过滤/双池输出；记录读写字节、寄存器/spill 和整段时间。双域独立打分只可作为额外候选 | 共享统计本已高复用、融合导致资源压力或整体变慢；不据 launch 减少认定成功 |
| M4 cavg-K | dot-assignment 物化相似度；簇更新具有阶段依赖 | 固定 seed/中心/迭代数，只替换 assignment 为 tiled argmax，再保持原更新；比较 assignment、临时量、完整 build 与返回 a | near-tie 分配漂移超合同；或小矩阵下库 GEMM+argmax 更快。CSR 查询另做独立实验 |
| M5 cavg-S | 现有 Triton wrapper 的 live count 回传及状态扩容可产生等待 | 保持同一 greedy kernel，比较当前 wrapper 与预分配/设备计数版本；测 host gap、append 延迟、overflow 恢复及内存 | live 上界导致显存超限；或消掉同步却增加扫描/容量浪费；禁止同时改阈值或簇决策 |
| M6 ccluster-KK | 两区域数学独立，但 shape 和簇数不均 | 固定两侧算法，先比较串行两次构建与 region/grouped 调度；分别看 build 最大尾项、ragged padding 和整个 selector | 较小区域被最大 shape padding 放大、调度成本增加；或窗口版本失配 |
| M7 ccluster-SS | far append 与 near 重放为独立链；链内 token 串行 | 固定完全相同状态，比较顺序、统一 grid、双流三种调度；测 near 重建步和普通步的完整 TPOT 分布 | persistent greedy 挤占 attention，p95 上升；或误用旧 near 状态。不得用平均构建时间遮蔽重建尖峰 |

M1 的候选化只有在候选域与归一化合同已固定时才是 C；若同时把当前代表分路径改为真正 L2，那是 R。M4–M7 首轮应在 HF 已可达路径实测；若将来接入 SGLang，需要另验 backend 接入与 serving，不把 HF 结果直接记为 serving 已通过。

## 4. 共用 kernel 库：复用原语而不是每种组合复制实现

### 4.1 接口与状态

以下是建议接口，不是已存在的 API。统一逻辑形状为 `Q:[Nq,Hkv,G,D]`，query 到请求用 `request_row:[Nq]` 映射，输出 `ids:[Nq,Hkv,Kcap]` 加同形 valid mask（或连续有效项的 valid_count）；地址空间允许时 ID 用 int32，超界则显式采用 int64。

```text
RegionDesc(request_id, layer_id, generation, true_S, causal_end,
           sink_range, far_range, near_range, swa_range,
           K1_far, K1_near, K2_far, K2_near, method_pair, semantic_mode)

build_or_append(K_new, physical_page_map, RegionDesc, feature_spec, Cache)
  -> block_stats, quantized_K2, cluster_state, ready_event

score_blocks(Q1, block_stats, RegionDesc, query_tile)
  -> block_scores or tile_local_block_topk
select_blocks(...) -> per_head_block_ids, valid_counts
compact_blocks(...) -> logical_token_ids, valid_counts
score_candidates(Q2, quantized_K2, token_ids, page_map, scale_contract)
  -> logits or tiled_logits_and_local_lse
normalize_gqa(...) -> candidate_scores
select_tokens(...) -> selected_ids, selected_valid
attention_selected(Q, full_K, full_V, selected_ids, selected_valid)
  -> output, optional_lse
```

Cache 必须携带 request generation、layer、子空间/投影版本、量化格式、构建边界及 method 需求位。取消请求、复用池行、改变区域参数或投影后，旧 cache 不能误读。参数改变只使受影响统计或 cluster 失效，不能无理由全量重建全部状态。

输出逻辑 token ID；读取 K/V 时通过 paged KV 的映射取得物理地址。可以在已选集合内部按物理页重新排列以改善局部性，但 attention 累加顺序会变，必须对拍。目标接口中缺位不得当 token 0，始终带 valid mask。当前 prefill 补 token 0 的源行为若要修正，必须放入 R 模式并单列旧行为基线；C 模式不得同时承诺兼容这一行为和改变它。

### 4.2 建索引与布局

建议以 `[request, kv_head, block, feature]` 或等价分块布局保存统计，feature 连续；实际 stride 根据 decode 与 prefill 的访问分别测量，不强制一次全量转置。

- Prefill：一个 CTA/program 处理一个 block×head 的 K tile，按 method 需求位只计算 sum/count、min/max 或投影特征；可共享一次 K 加载。
- Decode：只更新追加 token 所在块，尾块 sealed 后转只读；同一请求同一层单写入者。min/max 追加可增量，任意删除不能靠简单减法维护。
- 当前均值分母可能含 block padding；C 模式保留，R 模式才改真实 count 分母。
- 当前 SGLang 的量化值是 0–15 格点，但每个格点按 uint8 保存，并非两个值压入一个 byte；另有每 token/head 的 FP32 scale 和 min。因此该缓存净载荷约为 `D2+8` bytes/token/head（不含 padding/元数据），不能按 `D2/2+8` 算带宽。nibble 真打包是独立布局候选：可保持格点值但增加位解码、对齐与尾维处理成本，需单独 A/B。HF 的 QAT 张量表示还需按实际 dtype 另算，不能套用 SGLang 的字节账本。
- 4bit 数值量化 scale/min 的粒度按现有格式保留；块统计、投影、量化能否共载取决于所需特征是否一致。若引入二次全 K 转置，其成本纳入 build 和峰值显存。
- 默认保留原 full K/V，用低精度 K 仅做选择。最终 attention 的表示精度不能跟着 indexer 隐式下降。

### 4.3 L1 与 L2 的调度

Prefill 多 query：尝试 query-tile×block-tile 的 GEMM 型 L1，复用 block 统计。候选式 L2 在不同 query 候选高度重合时可用 union tile 加 per-query mask；union 膨胀严重时回退逐 query/head candidate list。按实测 overlap 选择，不能因为 GEMM 吞吐高就计算所有无用项。建议 grid 为 `(ceil(Nq/Bq), Hkv, block_or_candidate_tile)`；ragged request 必须由元数据映射到本请求长度和页表。

Decode 少 query：用 request×head×candidate-tile 的 GEMV 型 kernel，融合加载、页寻址、量化格点解码、打分及局部归约，避免生成 `[batch,Tc,Hkv,D2]` 的展开 int64 gather 索引和完整 FP32 K 临时量。

Top-k 先保持独立精确阶段，候选规模小时再尝试融合。每 tile 保留 local top-k 再全局归并在同一总序下是精确的；但 K2≈1024 时 local top-k 也很大，寄存器和归并成本可能比独立选择更差。不要盲目把 L1、L2、top-k、attention 全塞进一个 CTA。

## 5. GQA 与 attention 的可并行实现

### 5.1 HF L2：两遍或三遍算法

第一遍，各区域/候选 tile 独立计算每个 query head 的 logits，得到局部 `(max,sumexp)`。归并所有合法候选的统计，得到每个 head 的全域 LSE。下一遍用全域 LSE 计算 `mean_g(exp(z-LSE_g))`，再做 far/near 各自 top-k。

两种内存方案公平比较：

| 方案 | 工作量 | 适合情况 |
|---|---|---|
| 存储候选 logits 后归一化 | 额外写读 G×Tc 分数，避免第二次 K 解码/点积 | G 或 D2 较大、candidate 缓冲可控 |
| 只存局部 LSE，重算 candidate logits | 再读 K、再做点积，少存大矩阵 | D2 小、分数矩阵成本更高 |

必须在全域 LSE 就绪后计算 GQA 混合分数；提前按 raw logits 对混合 GQA 做 local top-k 一般不精确。G=1 可利用 softmax 的单调性省去仅为排序服务的归一化，但需排除后续确实消费概率或特殊覆盖分数的路径。

### 5.2 最终 attention：固定区与选择区

固定 Sink/SWA 可在中间区域检索时计算 partial attention。mid 选择完成后算另一份 partial，使用 log-sum-exp 组合：

\[
L=\operatorname{logaddexp}(L_f,L_m),\qquad
O=e^{L_f-L}O_f+e^{L_m-L}O_m.
\]

固定区与 mid 必须不重叠、因果掩码一致；空集合 LSE=-∞ 且权重为零，不能产生 NaN。这里的 LSE 是最终 full K/V attention 的归一化，和 §5.1 用于 indexer 选 token 的归一化不是同一个量。

收益条件：检索有足够延迟且两条 kernel 留有并发资源。固定区很小、额外 launch/merge 已超过可隐藏时间时采用一个最终 attention kernel。split-K attention 也按 batch×head 并行度不足时启用，不设为无条件默认。

## 6. avg/minmax 三种组合的专门设计

### 6.1 M1：mavg

**Prefill**：far 所需块维护 min/max；near 所需块维护 sum/count；读取 Q 时分解正负部分。

\[
s^{mm}_{b,g}=\sum_d q^+_{g,d}K^{max}_{b,d}+q^-_{g,d}K^{min}_{b,d},\quad
s^{avg}_{b,g}=\sum_d q_{g,d}\bar K_{b,d}.
\]

边界随 query 变化时，同一物理块可能在不同 query 中承担不同角色：可维护共享统计超集，或按 query-tile 所需范围构建，不能只按最后一行 near/far 切一次后复用给所有历史 query。融合打分 kernel 根据 RegionDesc 在 far 读 min/max、near 读 avg；不必两种分数都算满全域。

**Decode**：旧 near 块逐渐进入 far，如果此前只存 avg，进入 far 时还需 min/max。两种策略比较：全块一次保存三类统计；或转区时补算 min/max。前者缓存多、转区稳定；后者读写少但可能造成周期性延迟尖峰。默认先用共享统计超集，按方法永久不使用的字段才删除。

**L2**：普通 token 量化打分，而不是 avg 打分；沿 §5 保留各后端合同。near/far 的 candidate list 分开，头间 union 仅用于复用加载。

**并行与融合**：优先统一 grid 内的 region 任务，次选两 stream；cache append 与同请求读之间必须有依赖。最小消融为 region-aware L1、compact、candidate L2 各自单独开关。

### 6.2 M2：aavg

只需 avg 统计和普通 L2 量化 K；完整删除 min/max 分配、更新与分数构建，前提是没有其他共享消费者。near/far 转区无需重建 avg，同一块统计共享，区域只影响 top-k 池和配额。

Prefill 用 grouped GEMM 或 Triton query×block tile 批量计算均值打分。Decode 用 Q×block-mean GEMV，融合区域过滤与局部选择；小 Nq 时不为使用 Tensor Core 人工复制 query。

HF GQA 的 L1 聚合位置要复刻源码，不能从线性打分直接推导“所有阶段可预先 sum Q”。若合同允许线性聚合，可在一次小 kernel 中计算每 KV head 的 Q 汇总供 L1 使用；L2 仍按 §5。

预计是最规则、缓存最轻的一类，但最终是否最快还取决于 Tc、γ 和选择质量。不得以结构简单替代质量约束。

### 6.3 M3：mminmax

只保存 min/max 和普通量化 K，删除不使用的 avg。两域共用同一统计、同一打分原语，分别执行预算选择。decode 增量 min/max 已在现有代码存在，新增工作是方法专门化和融合，不是再次“引入增量”。

Prefill 可把正 Q 与负 Q 对应的 max/min 组织成两个乘积的融合 epilogue，也可逻辑拼成 2D1 维输入；比较拼接/布局开销，不物化重复 head 的 K。Decode 在寄存器中按 q 符号选择 min/max 元素。

L2 仍是量化 token 分数，**不是所谓“4bit minmax 界细筛”**。L1 上界只对其特征表示成立，不代表对被舍弃维度或最终 attention 概率的严格界。

## 7. cluster 组合的专门设计

### 7.1 M4：cavg-K

**当前兼容 C 路线**：far 建簇→Q 对簇代表打分→按 assignment 映射成 token 分→far token top-k；near 使用 avg 粗筛和普通细筛。HF near 的 GQA softmax 分母仍可能消费 far 的 L1 候选；因此 far 最终按簇分选 token，并不代表可删除所有 far 普通分数计算。C 模式必须先检查归一化依赖，R 模式才重新定义候选域。far 当前不是“选中簇以后再算各 token 的 L2”。

K 构建优化：将独立 head/request/region 合成 grouped 任务；assignment kernel 以 token tile×centroid tile 计算 `x@c.T`，只归约 argmax，避免写 T×C 完整相似度矩阵。每轮 assignment 完成后再更新簇统计，不能跨 Lloyd-like 迭代越过依赖。

簇更新有两种实现候选：block 内聚合后少量 atomicAdd；或 assignment 排序/分段归约以降低热点写冲突。前者快但 FP 累加顺序不确定；后者有排序代价且也未必复现旧 index_add 顺序。二者都需要数值和 assignment 序列验收，不能称天然逐位等价。保持 seed、初始化抽样顺序、空簇保留旧中心和迭代次数；当前返回 assignment 由最后一次更新前的中心产生，不额外补一次最终中心分配；不要用 Faiss 欧氏 kmeans 直接替换 dot-assignment。

Decode far 集合增长时，C 路线保留现有触发/重建规则；新 token 只分配到旧中心或 warm-start 少迭代属于 A 路线。

查询阶段可用 cluster CSR 倒排表避免全长 far score scatter。先对簇分数排序，结合簇成员计数确定达到 token 预算的前缀，再展开所需成员。**但部分簇的同分 token 截断顺序必须匹配参考 top-k**；不满足则保留 token top-k，或将稳定 token-ID 次序列为 R 模式。构建 CSR 的成本按实际复用 query 数摊销。

**目标 R 路线**：far 簇 L1→按成员数确定召回候选→候选 token 真正 L2→token top-k。新增 L2 会改变结果和成本。候选预算应该明确是簇数还是成员 token 数，不能把每簇当固定 B token。簇大小偏斜时，prefix-scan 计数、两遍写入构建 ragged candidate list。

### 7.2 M5：cavg-S

near 沿 M2 的 avg 设计。far 保留现有 greedy 顺序语义和精确增量；边界前移只处理新进入 far 的 token。基线是已经接入的 `greedy_triton.py`，不是早期 Python 多 launch 循环。

建议分三档实验：

1. **S0：现有 Triton 加状态管理优化**。预分配有界 Kcap；设备维护 live count；移除每次 `k_live.max().item()`；容量不足设置 overflow 并走显式恢复，不截断簇。失败输出禁止消费：若已有前缀原地更新，记录精确 committed-prefix 后仅重放未提交后缀；或从调用前快照恢复后全量重放。不能拿部分更新状态从第一个 token 直接重试。对 chunk、BT、warps 做小规模调参；减少重复 clone、F.pad 和 assignment cat。
2. **S1：单 CTA/独立链 CUDA persistent**。多个 warp 分担簇行，warp 内完成 Dc 点积，再做 CTA argmax，单 owner 更新获胜簇后 block barrier，再处理下一 token。每条链只有该 CTA 写，无跨 CTA 状态竞争。token 串行，簇扫描并行。与 Triton 比较寄存器、spill 和每 token 延迟；CUDA 语言本身不保证更快。
3. **S2：多 CTA/链协作**。仅当少量独立链确实无法占满资源且簇扫描成为瓶颈时试验；需要正确的 cooperative launch/驻留约束或显式分段 kernel。普通超额 grid 中用全局自旋栅栏可能死锁，禁止作为默认实现。每 token 全局同步可能吃掉簇扫描并行收益。

冷启动和增量续跑复用同一状态布局与决策 kernel；batch×head×region 提供独立链维度。在线不同层的 K 依赖前层输出，不能像离线 dump replay 那样任意并行层。

### 7.3 M6：ccluster-KK

双侧分别维护 K 聚类状态，用 region×head×request grouped assignment 复用一次调度。far/near 的中心、成员计数和 assignment 分开；边界变动不允许把整个 near 状态误当 far 初始状态。

C 路线查询用两份 centroid-score→token-top-k，near 的簇覆盖之外尾部仍使用当前 fine fallback。原始代码 near cluster 分来自未缩放 Q，fine fallback 已含 softmax_scale，存在混合量纲风险；C 模式先记录并复现，R 模式统一分数表示后重新评测。

near 滑窗每次重建是重点成本。可异步构建到非当前消费者使用的 buffer，但只在目标窗口的 K 已经就绪时启动；发布时绑定 region/version/event。不能靠读取旧 near 簇实现“零等待”。

R 路线两侧都增加簇候选 token L2，复用 §5 的全域归一化与两池选择；预计比当前代表分路径增加计算，却可能改善质量，应按质量–延迟曲线评价。

### 7.4 M7：ccluster-SS

far 采用 S0/S1 增量链；near 采用相同 greedy kernel 对新窗口精确重放，两条独立链可并行，或放入统一 grid。单请求只有 Hkv 个 head 时增加 region 可扩大可用任务，但不保证线性加速。

**near 删除旧 token 后减 sums 不等价于重建**：旧 token 影响过后续 token 的分配，因此保留旧 assignment 再减和会改变算法。精确优化包括减少不必要重建、重用输入变换、只在实际边界变化时调度、减少扩容和同步；近似滑动簇、冻结中心、周期刷新归为 A 模式。

双缓冲成本按 far/near 最坏 live clusters 计算。greedy 最坏每 token 一个簇，单条链的簇扫描工作为 O(T²·Dc)，persistent kernel 只减少调度和改善并行，不能改变此最坏复杂度。长序列若出现大量单例簇，应记录算法成本并选择已授权的替代配置，不悄悄限簇数。

## 8. 并行、融合、缓存：哪些值得做

| 机会 | 依赖条件 | 推荐处理 |
|---|---|---|
| near/far L1 或建簇 | 同层 K 已写完，两份状态独立 | 先统一 grid 分发，再比较双流 |
| build(request j+1) 与 select(request j) | 各自缓存互不覆盖、事件齐全 | 已有 side stream，优化残余等待而非重新搭建 |
| 固定区 attention 与 indexer | K/V ready，最终做 LSE merge | §5.2；只在端到端有收益时启用 |
| 同请求 L1→候选 L2 | 候选数据依赖 | 保留同步；流水化 query tiles 可行，但每 tile 仍需正确候选 |
| near/far L2 logits | GQA 全域归一化 | 并行算局部统计，归并后再选 |
| 跨层索引构建 | 后层 K 尚未产生 | 在线不可随意并行；离线 replay 可以 |
| CPU/GPU 协作 | 数据是否原本在 CPU、传输是否在关键路径 | CPU 做元数据/离线统计；不把 GPU-resident 每 token 贪心搬回 CPU |

双流只在有资源余量时隐藏时间。两个满带宽 kernel 并发可能互相拖慢；一条很长的 persistent greedy kernel 也可能挤压 attention。保留顺序基线及 region-aware 单 kernel 候选，公平比较时间线。

## 9. 硬件调优与成本模型

以部署 GPU 的实际型号、SM 数、共享内存、寄存器、L2、可用精度、CUDA/Triton 版本为输入。A100、A6000、H20/Hopper 不共用“已最优”配置；TMA/WGMMA 等只在支持的硬件单列后端，不把它们作为通用前提。

建议首轮有限搜索空间：query tile `{1,16,32,64}`，candidate tile `{64,128,256}`，greedy centroid tile `{128,256,512,1024}`，warps `{4,8}`。先剔除不适用形状、编译溢出和显存超限，再测少数候选；保留 JIT/autotune 时间，不能无限搜索。

近似的工作量账本：

```text
普通 L1: O(Nq * Hq * (S/B) * D1)
全域普通 L2: O(Nq * Hq * S * D2)
候选普通 L2: O(Nq * Hq * Tc * D2) + compact/selection 开销
K 聚类: O(iter * Tregion * C * Dc) + assignment/统计更新
Greedy: O(Dc * sum_i C_live(i))，最坏 O(Tregion^2 * Dc)
最终 sparse attention: O(Nq * Hq * Kselected * Dfull)
```

`Tc` 必须用实际每 head 候选量，不能用 head union 后更大的候选量冒充。candidate 不连续导致有效带宽下降，计算比 S/Tc 不等于最终加速比。

单 kernel 下界仅作诊断：`max(FLOPs/effective_compute, bytes/effective_bandwidth)`；分层 HBM/L2 流量分开计，串行依赖、launch、同步、KV 映射都在额外成本中。多阶段没有重叠时要累加各阶段，不能直接用全管线 FLOPs/峰值算力给出端到端目标。

若目标阶段占总时延 f，阶段加速 r 倍，端到端理论上限为 `1 / ((1-f)+f/r)`。例如 f=0.2、r=2 只对应约 1.11 倍理想端到端加速。这是数学示例，不是本仓库测量。

### 9.1 后端切换按形状和测量决定

| 算子 | 初始实现 | 升级触发证据 | 保留的回退 |
|---|---|---|---|
| decode L1/L2 | 支持选定合同的 Triton GEMV、直接页寻址、mask 候选 | 同合同 Triton 验收后，若定位到指令/布局或持续 spill，且 CUDA/CuTe 能具体解决，再比较小原型 | 对应 HF/SGLang taskmd 原参考路径；旧 B′ Triton 不是 taskmd 等价回退 |
| prefill L1 / K assignment | 规则 tile 的 Triton 与成熟 GEMM 比较 | Nq/T/C 足以摊销 launch 和布局；编译报告确认预期指令实际生成 | 小 shape 的 GEMV/原调用；不可把补零和转置时间漏掉 |
| cluster 更新 | 分块聚合+少量 atomic 与分段归约 | 实际热点冲突或不确定性不可接受，再付排序成本 | 原 index_add，保留与原数值合同兼容路线 |
| greedy | S0 后再比较 S1 | 状态管理已非主要耗时，单链簇扫描或编译资源是瓶颈 | 已有 Triton；S2 必须有少链不足并行度的证据 |
| top-k | 独立精确选择 | 对应 K、Tc 的计时占比足够高，且融合不会爆寄存器 | 原精确 top-k；不以 unsupported/unstable API 作为部署前提 |

首轮只保留当前实现和一个最有依据的替代实现。dispatch key 至少含 GPU 架构、编译器版本、dtype、D1/D2/Dc、Nq、候选量区间、连续/分页布局、GQA 与语义模式；切换阈值从开发形状测得，再用未见形状确认。不存在跨设备永久通用的 tile 或阈值。小尾块单独优化必须证明真实流量占比值得维护。

### 9.2 三个必须成立的收益不等式

令各 T 为相同执行条件下测得的时间；这些公式用于设计实验，不是此处已有测量。

**候选化 L2**：只有

`T_compact + T_page_gather_score + T_GQA + T_select < T_current_equivalent_segment`

才获得该段收益。右侧保留同一语义所需的 mask、归一化和选择，不只比较一个 dense dot。记录候选量和物理页分散度；若 fused gather 低带宽，`Tc << S` 仍可能不够。head union 与 query union 分别统计膨胀比。

**CSR / 布局预处理**：若增量构建成本为 ΔTbuild、每次有效查询节省为 ΔTquery>0，则在失效之前至少需要 `R > ΔTbuild/ΔTquery` 次复用才有时间收益，另外仍需满足显存约束。near 每步重建可能无法摊销；far 生命周期通常更长，但须从实际边界变化频率确认。没有正的 ΔTquery 则拒绝，无需继续计算复用阈值。

**双流 / partial attention**：理想重叠最多把 `Ta+Tb` 降为 `max(Ta,Tb)`，实际还加 event、merge 和资源竞争。令 `T_overlap` 包含从共同起点至两支完成的 event/等待/竞争，`T_merge` 只含后续合并，二者不重计。用 trace 比较完整关键路径；若 `T_overlap+T_merge >= T_serial` 则回退，不能把两条并发 kernel 各自变慢的时间简单相加或各取最小值宣称收益。

### 9.3 状态与带宽账本

对每个 shape 记录实际 buffer bytes、分配次数、读写 bytes 和寿命。低精度值、scale/min、ID/valid mask、页表、centroid/count/sum、候选 logits 与双缓冲均单列。特别检查 `G×Tc` logits 和 `T×C` assignment 矩阵是否物化，以及取消请求后内存何时可复用。

greedy 当前 FP32 sums/count/sq、int64 live/assignment 的每条独立链基本存储为：`sums=4*Ccap*Dc`、`count+sq=8*Ccap`、`live=8`、`assignment=8*Tstored` bytes。再加物化 centroid、输入、scratch、容量 padding 和双缓冲，最后对 request/head/region 的链求和。不可把整个 state 当一种元素，也不因消除 `.item()` 就默认分配最坏 T 个簇。若采用有界容量，容量/overflow 只决定调度与恢复，不能改变数学簇数。容量耗尽的实测必须包含恢复成本；报告普通步、扩容步、near 重建步和冷启动分布。

## 10. 已有设计文档需要纠正或降级的结论

| 原描述类型 | 本次核查 | 建议替换 |
|---|---|---|
| mavg 的 L2 是 avg、mminmax 的 L2 是 minmax | 与普通区域 token 量化打分源码不符 | 分清 L1 method 和 L2 token score |
| kmeans 是欧氏 Lloyd 距离 | 当前是 dot-assignment | 写明实际目标函数；换欧氏距离属于算法变更 |
| 唯一 launch-bound 段是 Python greedy | greedy 已 Triton 化，taskmd 仍有 eager 小算子和同步 | 重新 profile 当前实际路径 |
| CUDA persistent 必然终态最优 | 单 CTA 仍受链并行度约束，多 CTA 每 token 同步有成本 | Triton/CUDA 在同条件下实测选型 |
| 8 个 program 意味着固定 10% 性能或全体 SM 利用率 | program 数不是完整性能测量 | Nsight 查看 active cycles、占用、带宽、stall |
| 状态小于 L2 容量就完全常驻 | attention/KV 与其他请求可能争抢 L2 | 按实际并发测 L2 命中和流量 |
| near/far 各自归一化后即可合并 | HF GQA L2 共享分母，最终 attention 也需 LSE | 分别处理两种归一化阶段 |
| near 滑窗删除旧 token 就能精确增量 | greedy 历史分配受旧 token 影响 | 精确重放或单列近似算法 |

已有 SEG-GREEDY 仿真记录所报告的跳过率仅约 0.29%–2.9%，应保留负结果，不优先再次投入同一方案。这些是仓库历史记录，本次没有复跑；零 mismatch 仅说明已测样本，不能作为一般精确性证明。

E113b 记录 2026-10-07、Qwen3-8B、ccluster_sim、8 样本、共享 GPU 条件下 kernel on/off 为 168s/1290s。它支持已有集成尝试，不等于当前所有 method、独占 GPU 或 SGLang serving 的公平 7.7× 结论。

## 11. 实施任务链与预先验收

### 11.1 任务顺序

| 任务 | 具体产物 | 前置条件 | 停止/回退条件 |
|---|---|---|---|
| P0 合同冻结 | 每 method 的 route、flags、scales、regions、预算和 reference hash | 冻结 TASK 定义与当前代码差异 | TASK 与代码差异先登记，不进入无损加速宣称 |
| P1 小补丁 | hoist K 扩展；删除永久不用统计；静态长度 guard | 对拍 fixtures | 选中集合或输出异常则回退 |
| P2 普通三组合 | compact→candidate L2→GQA→top-k | P0/P1；有效掩码明确 | Tc 很大或 gather 代价抵消收益则保留 full-score dispatch |
| P3 Cluster 查询 | centroid-score/CSR 或兼容 token-score 路线 | tie、fallback、缩放合同 | 结果漂移或 CSR 建造不摊销则保留旧路径 |
| P4 Cluster 构建 | K grouped assignment；S0/S1 对照 | 聚类序列和状态对拍 | 额外同步/资源占用导致端到端退化 |
| P5 并行组合 | side-stream、固定 attention 重叠、split-K | 有清晰 Nsight critical path | 单 kernel 更快但 TTFT/TPOT 变差则拒绝 |
| P6 算法修正/探索 | 双侧真正 L1→L2、scale 修正 | 单独 R/A 实验协议 | 精度预算不达标，不能混入 C 结果 |

以上是实现建议，不是本次已经执行的任务或自动部署授权。

### 11.2 正确性矩阵

复用 `test_e112_sglang_port.py`、`test_e110_ccluster.py`、`test_e113b_kernel_integration.py` 和现有 microbench 入口；先检查它们覆盖的是哪种参考语义，再增补参数，而非为每个组合复制脚本。

必须覆盖：

- batch=1/8/32；G=1 及模型真实 G>1；prefill、chunked prefill、decode。
- `B-1/B/B+1`、短序列、非整块长度、不同 prefix、混合长度 batch。
- α/β 的单池退化点；γ 导致 far=0；配额不足、空候选、Sink/SWA 重叠。
- 每种组合及 cluster 原语、cluster 尾部未覆盖、量化 ties/near-ties。
- greedy 真正会归并的结构化样本、全单例、重复/零向量、阈值等值、冷启动/分段续跑/near 重建。
- paged KV 非连续物理映射、池扩容、请求取消/复用、CUDA Graph capture/replay、双流事件、区域参数更新。

C 模式先比较有效集合和簇 assignment，再比较最终 attention 输出与端到端预测。暂定输出阈值不能替代参考自身精度基线；FP32 累加与低精度输出分别锁定容差。若不再逐位一致，记录误差和 near-tie 变化，不以 `allclose` 单项通过宣称所有选择等价。

### 11.3 性能与噪声控制

1. 固定 commit、模型、输入 ID/hash、所有环境 flags、layer mask、GPU/驱动/依赖版本、精度模式及图捕获方式。
2. GPU 使用须有当前授权；没有空闲目标 GPU 就保存未执行状态，不在共享实验上抢跑。
3. JIT、autotune、warmup 单列；稳态测试两臂均先 warmup。构建/首次请求另测，不把它们永久排除。
4. microbench 每配置至少 20 次计时，长端到端至少 10 对 A/B，按 AB/BA 交替；用每对差值和置信区间判断，不能挑最快一次。
5. 异步路径使用覆盖所有相关 stream 的事件依赖后计时，同时测 host wall-clock。逐 op 全设备 synchronize 会破坏原有 overlap，只用于局部诊断。
6. 记录其他 GPU 进程、时钟/温度/功耗及 trace；发现干扰保留原始记录并按事先规则标 invalid，补跑匹配的 A/B 对，不只删慢样本。
7. Nsight Systems 看主机等待、launch 间隙、两流重叠及 PP bubble；Nsight Compute 看真实瓶颈 kernel 的寄存器、spill、occupancy、HBM/L2 流量。profile 单独运行，主性能表用低扰动测量。
8. 预注册维护收益门：目标端到端中位延迟改善至少 5%，配对 95% 区间不跨零，p95 不退化超过 3%，显存不突破已有预算；仅局部 kernel 变快则记局部结果。阈值是本设计建议，应在看结果前固定。

质量验证先用已固定开发集发现问题，再用未见任务确认；对 R/A 路线跑真实任务精度和与 Quest/ClusterKV/MoBA 等既定基线的公平对照。token 重合率、mass、attention 输出误差是诊断指标，不能代替任务精度。


### 11.4 最终结果表模板

| method/模式 | 输入与实现 hash | GPU/shape | build | L1 | compact/L2/top-k | attention | TTFT/TPOT p50/p95 | peak memory | 质量 | 判决 |
|---|---|---|---|---|---|---|---|---|---|---|
| 参考 | 待实测 | 待实测 | 未执行 | 未执行 | 未执行 | 未执行 | 未执行 | 未执行 | 未执行 | 不判通过 |
| 候选 | 待实测 | 同参考 | 未执行 | 未执行 | 未执行 | 未执行 | 未执行 | 未执行 | 未执行 | 不判通过 |

成本还需包含 greedy/cluster 周期性重建的每步分布，不能只有摊销平均。R/A 结果与 C 模式分表；tokens、GPU 时、能耗或费用没有采集则写未知。

### 11.5 修订版的候选与测量合同

每次候选保留 `candidate_id/parent_id`、源码与依赖 hash、入口 flags、语义模式、shape/stride/dtype、初始状态 hash、reference/test hash、预定判别指标、原始正确性/计时日志、profile 对应的具体 kernel invocation，以及采用/拒绝原因。测试之后改过代码即形成新候选；不能把 A 版本的计时与 B 版本的正确性拼成通过。

特别为两个聚类原语固定状态起点：kmeans 每次用相同初始化及完整轮数；greedy append 每次从相同 prefix 的状态开始。warmup 若修改 state，要在正式计时前恢复。明确 reset/clone 是测量准备还是实际服务成本：kernel-only 可另列准备，但完整 append/build 指标必须计入生产中确实发生的 clone/扩容/重建。反复对增长状态计时，会使 A/B 工作量不同。

shape、dtype 和 layout 可用于合法 dispatch；输入值、指针身份或测试顺序不能用于绕开计算。重复同一数据的热缓存与新数据/真实 trace 各自构成不同实验，整组比较中不切换口径。参考与评测脚本不由候选改写；不复制外部 Agent 的统一 1e-2 容差到聚类分配或 top-k。

profiling 只用于回答当前疑问：先以系统时间线找关键路径，再对指定 kernel 收集计数器，最后按需查 IR/PTX/SASS。多 kernel 路径不得取第一个非框架 kernel 的指标代表全部；高 SOL、高 occupancy 或更少 launch 均不替代正常计时。并行代码探索可以独立进行，共享 GPU 的排名测量必须串行或在已有授权的隔离资源中进行。

最终赢家需要以原始快照恢复，再跑完整验收；“最新一次修改”不是默认赢家。性能无收益但有诊断价值的版本保留记录，不替换当前部署。没有目标 GPU 时，本文件中的表项保持未执行，CPU 数学对拍和模型审读不能填入 GPU 验收栏。

### 11.6 本次实际执行的数学检查

2026-10-08 使用 Python 标准库、CPU 双精度计算执行下列三个小例。它们检查设计中的数学关系，不执行仓库 CUDA/Triton，也不是吞吐、精度数据集或 GPU 实验。

| 检查 | 输入与操作 | 观察结果 | 设计结论 |
|---|---|---|---|
| GQA 排序反例 | 两 head 的三项 logits 为 `[10,0,9]`、`[0,10,9]`；各自 softmax 后平均，与 sum-logits 比较 | 平均概率约 `[0.365534,0.365534,0.268932]`，前两项胜；sum-logits 为 `[10,10,18]`，第三项胜 | §5 的跨 head softmax 混合不能替换成 sum-logits 排序 |
| partial attention 合并 | 第一段 logits=`[1,-1]`、V=`[2,4]`；第二段 logits=`[2]`、V=`[7]`；分别计算 LSE/输出再合并 | 合并 `5.597160617409884`，直接整体计算 `5.597160617409886`，绝对差约 `1.78e-15` | 非重叠同合同子集满足 §5.2 数学恒等式；GPU 舍入/空集/多维仍待验 |
| greedy 滑窗删除反例 | 2D 单位向量角度 `[0°,40°,70°]`，余弦阈值 `cos45°`；逐 token 比较簇向量和的归一化方向 | 原 assignment `[0,0,1]`；删首项并保留后续分配得到 `[0,1]`，剩余两项重放为 `[0,0]` | §7.4 减掉旧 sums/count 不能替代精确 near 重建 |

独立修订审查还纠正了三处设计表述：mminmax 当前已共享全域 sc1；taskmd eager 回退不能写成旧 Triton；greedy 显存需计全部状态和链数。以上修正已落实到 §3.2、§9.1、§9.3，未把审查意见当作性能证据。

另以更新后的技能启动独立上下文代码审计，仅提供原始 SGLang 代码与任务范围，不提供本文或既有审查结论。它独立识别了 taskmd 分流、两套 GQA 差异、最大内积聚类与 greedy 时序，并细化出 decode 已候选化的阶段边界；主执行者回读对应源码后补入 §3.1/§3.2。这是一次真实模型的静态代码任务，不是旧新技能公平 A/B，也不证明 GPU 或模型能力已提速。

## 12. 来源与核验范围

以下仓库链接固定到本次 SHA；行号用于快速定位，以该快照正文为准。

1. [TASK.md：method、保护区域、预算及论文要求](https://github.com/Chosen-David/sglang/blob/9049ba8759896e5efc13f287c442fb72d5e1b563/TASK.md)。只读，不修改用户维护文件。
2. [HF tli_indexer.py](https://github.com/Chosen-David/sglang/blob/9049ba8759896e5efc13f287c442fb72d5e1b563/two-level-attention/sparse_attn/indexer/tli_indexer.py)：配置 68–118；聚类 312–594；score 603–689；GQA 919–924；cluster token 选择 992–1061。
3. [greedy_triton.py](https://github.com/Chosen-David/sglang/blob/9049ba8759896e5efc13f287c442fb72d5e1b563/two-level-attention/sparse_attn/indexer/greedy_triton.py)：顺序 token 循环与 wrapper 容量/同步。
4. [SGLang config.py](https://github.com/Chosen-David/sglang/blob/9049ba8759896e5efc13f287c442fb72d5e1b563/python/sglang/srt/layers/attention/tli/config.py)：默认 kernel 开关及 139–158 taskmd 分流。
5. [SGLang indexer.py](https://github.com/Chosen-David/sglang/blob/9049ba8759896e5efc13f287c442fb72d5e1b563/python/sglang/srt/layers/attention/tli/indexer.py)：prefill 1304–1600；decode 2051–2340；full D 对拍与 candidate 物化。
6. [backend.py](https://github.com/Chosen-David/sglang/blob/9049ba8759896e5efc13f287c442fb72d5e1b563/python/sglang/srt/layers/attention/tli/backend.py)：pool/avg 状态与 738–824 side stream/event 依赖。
7. [kernels.py](https://github.com/Chosen-David/sglang/blob/9049ba8759896e5efc13f287c442fb72d5e1b563/python/sglang/srt/layers/attention/tli/kernels.py)：既有 batched L1/L2、dual、compact 和 attention 原语。
8. [E113 实证篇](https://github.com/Chosen-David/sglang/blob/9049ba8759896e5efc13f287c442fb72d5e1b563/research/docs/e113_method_kernel_design.md)、[sim_greedy 设计篇](https://github.com/Chosen-David/sglang/blob/9049ba8759896e5efc13f287c442fb72d5e1b563/research/docs/sim_greedy_kernel_design.md)、[overlap 设计](https://github.com/Chosen-David/sglang/blob/9049ba8759896e5efc13f287c442fb72d5e1b563/research/docs/overlap_kernel_design.md)：已有提案和历史实验；文档结论与源码冲突处见 §10。
9. [E113b 历史 smoke](https://github.com/Chosen-David/sglang/blob/9049ba8759896e5efc13f287c442fb72d5e1b563/two-level-attention/exp/trace/results/e113b_e2e_smoke.json)：不是本次新测结果。
10. [Triton Fused Softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)，访问 2026-10-08：融合减少中间显存读写，受片上容量/寄存器约束。
11. [Triton Matrix Multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)，访问 2026-10-08：tile 与 grouped 调度是可复用工程方法，不证明本项目同样收益。
12. [CUDA Cooperative Groups](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html)，访问 2026-10-08：grid 同步需要满足对应 launch 与硬件约束。
13. [CCCL Top-K requirements](https://nvidia.github.io/cccl/unstable/cub/device_topk_requirements.html)，访问 2026-10-08：集合确定性、tie 和输出次序是不同合同；该页面为 unstable 文档，不能推定目标环境已有相关 API。

后续实现时还需固定实际安装的 CUDA、Triton、CCCL 版本。本文未安装新 runtime，未执行 GPU 实验，也未把示例预算或理论下界当作测量结果。

14. [PyTorch torch.topk](https://docs.pytorch.org/docs/stable/generated/torch.topk.html)，访问 2026-10-08：并列元素索引不保证稳定，目标安装版本仍需核验。

## 13. Kernel Agent 调研如何改变本设计

核查日期 2026-10-08。公开材料没有给出这五种组合在同一硬件/模型/预算上的公平 Agent 排名，因此不指定“通用最强”。本次吸收可维护的工程机制，未安装整套 runtime，未获得或训练 CUDA-Agent RL 权重。

| 一手实现与固定 commit | 实际核查范围 | 本设计采用 | 不照搬 |
|---|---|---|---|
| [Meta KernelAgent](https://github.com/meta-pytorch/KernelAgent/tree/e0647170da36ef9b059ac0bd3d60103aa4ed378b) | README、orchestrator 的 verify/benchmark/best 分支、timing.py | §3.2 的证据→假设→实验；§9.1 按瓶颈换后端 | SOL 等同最优；第一个 kernel 的计数器代表管线；默认转换 dtype |
| [CUDA-Agent](https://github.com/BytedTsinghua-SIA/CUDA-Agent/tree/473025c8af7878e928525138b5cb327d6bcdf9dd) | agent_workdir/SKILL.md、utils/verification.py | 独立参考、相同模型状态、多输入验证 | 固定 SM90、sudo、1e-2 容差和宣称加载 Skill 即获得训练收益 |
| [AKO4ALL](https://github.com/TongmingLAIC/AKO4ALL/tree/bbd0e1cf1ce2fb19d4322a932680f4c3d175d80e) | SKILL.md、bench-wrapper.sh | §11.5 原始候选/失败轨迹、固定计时条件、重验赢家 | 每轮强制提交、新建分支和无限迭代；最小耗时不作尾延迟 |
| [Atrex Kernel Agent](https://github.com/alibaba/atrex-kernel-agent/tree/3d27c1eb1d75f390df29e63aa93ddbeecd928e3f) | episode-loop/SKILL.md、optimization_policy.py；gen-plan 部分 | 逐级取证；实现/profile 身份绑定；独立数值发现转复验输入 | 外部 SSH/runtime、强制 DSL 转换、reset-hard 和项目外评测政策 |

另检索到 [KernelPro](https://arxiv.org/abs/2606.26453)、[Houmao](https://arxiv.org/abs/2608.14560)、[Atrex-Bench](https://arxiv.org/abs/2607.14541)、[KREX](https://arxiv.org/abs/2609.30057)。本次只检索其摘要/元数据，KernelPro 全文 HTML 获取失败，**不计全文精读，不用其榜单数字证明本设计有效**。十篇全文与完整发布门禁的持续优化大批次没有因此完成。

本次升级的是所使用代码优化技能的工作流程和这份工程设计；SGLang 的五种组合仍须按以上协议实现与实测。文中的理论条件、调度建议和实验预算均不能标记为已部署或已经提速。
