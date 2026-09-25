# TLI 论文骨架（PAPER_DRAFT v1，2026-09-26）

> 目标 venue：MLSys / ASPLOS / SC（算法边界 + 系统集成 + 负结果护城河 + 测量学方法论）。
> 本文档 = 论文骨架 + 各节素材映射（素材来源：TWO_LEVEL_PAPER_REPORT.md §1-§10、§8b-1~16、
> figures/、各 JSON 结果）。写作顺序建议：§4 Design → §5 Implementation → §6 Evaluation →
> §3 → §2 → §1 Intro → §7 Discussion → Abstract。
> 标注约定：【素材：报告§x】= 已有实测；【缺：xxx】= 待补实验；【图：figN】= 现成图资产。

---

## 标题候选

1. **TLI: A Training-Free Two-Level Indexer for Long-Context Inference**
   ——副标题带 negative results 卖点：*What Works, What Doesn't, and Why*
2. Asymmetric Compression and Budgeted Selection for Training-Free Sparse Attention Indexing
3. Two-Level Attention Indexing at Production Scale: Design Boundaries from 30+ Measured Hypotheses

推荐 1（训练免费 + 两级索引定位清晰；「30+ 实测假设」可放 Intro 末）。

## Abstract（~200 词，最后写）

骨架：长上下文 decode 的 HBM 瓶颈（triton 注意力流量 ∝S）→ training-free 两级索引器
TLI 三创新点（A 非对称压缩 / B' 分区预算 / D' 层跳过）→ 三层结果：
①质量：LongBench 13 子集 49.92 vs FullKV 50.36 / TIA 50.06（守恒口径）+ NIAH 极端口径 0.625；
②kernel：批量 select 12.9×（1.80ms@bs32/131K）、prefill 慢路径 7.0×、e2e decode 原型→
生产级 12.5×（323 tok/s）；③方法论：30+ 假设实测、每步对拍零漂移、两例主动复测撤回
自身 headline（测量学贡献）。
【素材：报告§1、§8b-12/13/15/16】

## 1. Introduction（1.5 页）

叙事链：
1. **问题**：长上下文 LLM decode 中 KV cache 读流量 ∝ S，HBM 带宽成为第一瓶颈
   （bs=32×131K 时 triton ~620GB/步纯 HBM）。【素材：报告§8b-14 翻转点分析】
2. **现有路径及其代价**：(a) 训练型 indexer（DSA）需 2.1B token warm-up，部署成本高；
   (b) training-free（Quest/TIA）在 token 级精筛开销或质量上受限。
   【素材：报告§8b-6 三方对比表——Quest 索引最快但 1KB/token 存储 + LB −2.2 分；
   DSA 0.5ms 需训练；TLI 每 token MAC 32×↓ 免训】
3. **核心洞察（H1）**：注意力质量的真实形态是 sink/near/far 三层分解（0.37-0.71 /
   0.16-0.29 / 0.02-0.29），dense top-1024 的 mass 覆盖恒为 1.0000——「覆盖率」
   口径本身是陷阱，竞争区仅占总 mass 0.515。【素材：报告§3 修正②、fig1】
4. **本文贡献**（四条，对应 §4/§5/§6/§7）：
   - 非对称压缩定律 + 两级预算选择设计（A/B'/D'）
   - 生产级系统实现（sglang 全链路：4bit 索引、批量 kernel 组合、CUDA graph）
   - 双口径诚实评估（kernel microbench + e2e；LongBench 均值 + NIAH 极端）
   - 12 项 negative results → design decisions 映射（护城河）

## 2. Background & Related Work（1 页）

- **稀疏注意力三代**：static/window（StreamingLLM sink 现象）→ KV 压缩/eviction
  （H2O/SnapKV，训练时超参敏感）→ 索引检索式（Quest/TIA/MoBA/DSA）。
- **DSA**【素材：INDEXER_RESEARCH.md】：I=Σ w·ReLU(q·k)，64head FP8，top-2048，
  需训练。定位=外部主 baseline。
- **Quest**：page min/max 上界 + 按页选择，token 粒度粗。
- **TIA（同门第一代）**：块 min/max + 4bit 部分维——TLI 是其直接后继，
  创新点必须以增量叙事（§3 关键叙事修正）。
- **HISA (COLM 2026)**：block-coarse+token-refine 已有——「two-level」本身非
  novelty；TLI 增量=子空间选择依据/分区预算/层自适应。
- **MoBA**：gating 需从头训练；chunk512×topk2 最省但语义最少。
- 差异化声明表（每行一个维度：训练需求/索引存储/token 粒度/滑窗/分区/层自适应）。

## 3. Motivation: The Three-Layer Structure of Attention Mass（1 页）

### 3.1 覆盖率口径陷阱【正文 v1】

对 10 条真实 trace（needle + natural + 8 条 LongBench 样本，Qwen3-8B）做
注意力质量的**位置分解**：每条 trace 的 token 级 attention mass 按位置
分为三段——**sink**（前 128 token，0.37–0.71）、**near**（近端 2048
token，0.16–0.29）、**far**（其余远端区，0.02–0.29）（fig1）。这不是
「近端集中」的双峰形态，而是 sink 主导 + 三段并存的结构。

由此得到本文第一个方法论发现——**覆盖率口径陷阱**：在「dense top-K2 的
mass 覆盖」口径下，K2=1024 的覆盖率对任何保留 sink+滑窗的方法都恒为
1.0000（sink+near 段 768 token 已占 mass ~0.485，剩余 256 个名额无论怎么选，
总覆盖都在 0.99 以上）。该口径下 TIA@1024 ≈ FullKV 是**口径饱和**而非选择
能力证明。把 sink+滑窗从 oracle 与被测方法中同时掩去、只看**竞争区**
（剩余 mass coverage）——竞争区 mass 仅占总 mass 0.515——选择方法的真实
差异才显形。本文所有微观质量评估（E3/E4c/分区消融）一律采用剩余 mass /
far recall 口径，这是 §7 测量学方法论的第一条实践。

### 3.2 far 名额的噪声驱动（B' 的动机）【正文 v1】

三层分解的直接工程问题是**名额分配**。在全局 top-K2 语义（TIA：所有
候选 token 全局竞争 1024 个名额）下实测（5 trace × 5 层 × 2 个解码位置）：
far 区平均占据 **569/1024** 个名额——但 far 区 mass 只有 0.02–0.29。名额
与质量的错配来自两个与质量无关的机制：①**基数效应**——far token 数
（S−2048）≫ 近端带 token 数（2048），候选池大小的悬殊使 far 在随机
扰动下天然多占名额；②**4bit 分数噪声**——量化后的细筛分数在 token 间
产生大量与真实注意力无关的排序扰动。二者叠加的后果是：当近端 token 的
4bit 分数被噪声压低时，真正承载 0.16–0.29 mass 的近端带反而被挤出
终选集。这把名额分配从「分数噪声驱动」改为「预算驱动」——即 §4.2 的
B' 分区预算设计。该动机与直觉叙事「防止 far 被近端挤出」方向相反：
**实测被挤出的方向是近端**（分区消融中 far-heavy 层的全部增益来自近端
名额恢复，far mass 覆盖同值 0.999，见 §4.2 表）。

### 3.3 GQA far mass 的跨 head 不均（评估口径）【正文 v1】

GQA 下 8 个 kv-head 各自服务的 query head 组的 far mass 极不均匀
（L03：[0.003, 0.306, 0.001, 0.804, 0, 0.001, 0.004, 0.117]——head3
独占 0.804）。任何对 head 取平均的评估口径都会系统性掩盖 far 选择质量
（E4b 的第二处口径 bug 即源于此）。本文微观评估一律 per-head 分布加权，
该口径同时解释了为什么 far 检索是「少数 head 的窄而深的问题」——这是
D' 层跳过（far-empty 层整层跳过远端选择）可行性的统计基础。

## 4. Design: TLI（3 页，核心章）

### 4.1 A：Position-Stable Subspace Coarse Filtering（L1）【正文 v1】

**符号与两级定义（全文统一）**：上下文长度 S、当前解码位置 t、块大小 B=64、
块数 nblk=⌈S/B⌉；L1 粗筛=从因果块中选 K1=128 块（全局竞争，滑窗块强制
加入不占名额）；L2 细筛=在选中块并集的 token 上选 K2=1024 个（far 池
K2_far=256 + 近端 K2_near，见 §4.2）。L1 表示 = 块内 min/max 上界在子空间
d'=32 维上（`kmin/kmax ∈ R^{nblk×Hkv×d'}`）；L2 表示 = 4bit 量化部分维
（选择口径 2δ=32 维，或投影口径 r=16，见下）。

**A 的设计**：Qwen3 的 rotate_half RoPE 使头维前半 [0,63] 与后半 [64,127]
各自的两半之间发生旋转，旋转速率沿维单调——尾维 [48..63]+[112..127] 是
旋转最慢的 32 维，构成近似 position-stable 子空间（Qwen3 无原生 noPE 维；
DSA 论文的 noPE 叙事在此语境下应改写为此抽象）。L1 上界打分
score(q, blk) = qg·kmax + qn·kmin（qg/qn = q 的正/负部，块内任意 K 的
点积上界）只在这 32 维上计算——相对全维 128 维是 4× 读/算削减。

**实测依据（E3/E3b，10 条真实 trace）**：低频尾维 mass recall 0.729 ≈ 全维
0.732（差距 0.4%）；随机 32 维 0.32、高频头维 0.137 崩溃——选择性证据排除
「任意降维可用」。32B 复验（7 任务×2 样本×64 层）：lowfreq32 entry recall
0.66-0.87 ≈ full128（差距 ≤0.08），random 退化幅度随规模弱化（0.19-0.60）
——「机制存在、强度递减」的诚实泛化结论。

**非对称压缩定律（本文核心经验发现）**：两级对降维的容忍度截然不同——
- L1 可粗：d'=32→16 逐位不掉（选中块集不同，但 L2 精筛吸收差异）；
- L2 选择口径不可压：δ=16→8 far recall 0.557→0.378（−32%），且 fp32 与
  4bit 仅差 2-5%（0.557/0.531）——**病因是维数不足而非量化精度**；
- L2 投影口径可压：per-kv-head SVD top-r 投影（K 协方差主方向）下
  PCA16 0.532 ≈ 选择 32 维 0.557（同质量打分维度砍半）、PCA32 0.682
  （同 FLOP +0.125）——**L2 的信息瓶颈在表示方式而非维数**。

**机制分离（投影深入实验）**：随机投影崩溃（r=16 recall 0.147 vs PCA 0.532）
——JL 变换保范数但不保 GQA（多 query head 共享 kv-head 求和）后的点积
排序，降维必须利用 K 协方差结构；PCA 与尾维子空间主角度 73°（近正交）
——旋转稳定性与协方差方差是两个独立机制；同预算混合打分（pca16+tail16）
0.555 < 纯 PCA32 0.660——双计交叠反降，纯 PCA 是最优表示。校准成本可
忽略：2048 token 小校准集与全量同值；跨任务基迁移仅 −0.024（training-free
成立）；跨层共享基 No-Go（0.234）→ per-layer 基存储 2.4MB 可忽略。

**Design decision（由 negative result 推出）**：L1 用尾维选择（免校准、
上界语义天然只依赖子空间），L2 用 PCA 投影（校准成本可忽略、质量 +27%
同成本口径）——两级表示异质化是非对称压缩定律的直接工程化。

### 4.2 B'：Far/Near Partitioned L2 Budget【正文 v1】

**设计**：因果区分三段——sink（前 B·2=128 token，强制保留）、近端带
[t+1−2048, t]（滑窗 + 4bit 精筛）、far 池 [sink, t+1−2048)。L2 终选在
far 池独立取 top-K2_far=256、近端带独立取 top-K2_near=640（K2_far+
K2_near+滑窗强制 128 = K2=1024，预算恒定）。滑窗强制位不占竞争配额。

**动机（名额分配的质量语义）**：全局 top-K2（TIA 语义）下，far 平均占
569/1024 名额——这不是质量驱动而是 4bit 分数噪声 + far 区基数效应
（far token 数 ≫ 近端）驱动；当近端 token 分数被噪声压低时，真正承载
0.16-0.29 mass 的近端带反而被挤出。B' 把名额分配从「分数噪声驱动」
改为「预算驱动」。

**消融定机制（三版本对照，5 trace×5 层×2 t）**：

| 模式 | far rec | far 名额 | far mass cov | L1 存活 | cov剩 |
|---|---|---|---|---|---|
| global（TIA） | 0.664 | 569.4 | 0.8456 | 0.929 | 0.9207 |
| late（B'） | 0.511 | 256.0 | 0.8414 | 0.929 | **0.9284** |
| hier（L1 分区保底） | 0.420 | 256.0 | 0.8105 | 0.677 | 0.9197 |

三条结论：①**「far 在 L1 被剪则 L2 救不回」实测不发生**——near 带+sink 仅
34 块 vs K1=128，far 池保底 ≥94 块，oracle far token 的 L1 存活率
0.93-1.00；②far rec 的 global>late 是名额口径混淆（far mass cov 两模式
几乎同值 0.846/0.841）；③**B' 真实增益机制 = 近端名额保障**——far-heavy
L05 cov剩 0.600→0.726 的全部增益来自近端带名额恢复（far mass cov 同值
0.999）。hier（L1 强制分区）No-Go：far 保底反而把池从 94 块砍到 16 块
（存活 0.677）——**分区只需在 L2 终选级**。

**聚类代表 = negative result（E4c 严格预算口径）**：kmeans 块代表 0.09-0.39
（最差）、kmeans token 代表 0.45-0.79（无一致优势）、minmax 块 0.52-0.74、
TIA 4bit token 级精筛 ≈ oracle（L03 0.999）。早期「kmeans 4-10× 占优」
（E4b）含整簇超选 bug（budget=512 实取 2769 token = 5.4×）——**严格预算
+ per-head 加权是远端候选质量评估的必要口径**（GQA far mass 跨 head 极不均，
L03 head3 独占 0.804，head 平均口径会系统性掩盖）。

**预算敏感性**：LongBench 主表口径 far_tokens 128-256 饱和（过大反抢近端
配额）；NIAH 极端口径下 [128,512] 双 seed 合并 n=40 完全持平 0.625——
限制因子是 far 池整体召回 ~0.66（与 E4c far rec@256 量级吻合）而非预算
切分比例。两口径合并的设计含义：**K2_far=256 是跨负载的稳健默认值**。

### 4.3 D'：Offline-Calibrated Layer Skipping【正文 v1】

**设计**：逐层 far-mass 轮廓呈双峰（far-heavy 层 vs far-empty 层）——
far-empty 层的远端检索不承载质量（far mass <0.3%），可整层跳过远端选择，
只保留 sink+滑窗。跳过掩码离线校准（trace 平均轮廓阈值化）。

**跨规模证据**：8B 跳 13/36 层（precision 0.92-1.00，far 质量损失 <0.3%，
索引 FLOP 4.88×）；32B 双峰更极端——跳 41/64 层（64%），precision 0.993。
**规模越大层冗余越高**（64 层中 2/3 的 far 检索可免），D' 的价值随模型
规模单调放大。

**gate 设计的失败史（三段，全部实测，论文如实写入）**：
1. 在线信号 gate No-Go：L1 分数、熵等在线信号与 far 质量相关 ≈0，
   churn 22%——层轮廓是任务级属性非请求级；
2. far 总量 gate 单位混淆：初版「安全任务 far 总量 0.02-0.29 vs 多跳
   0.55-0.95 gap 清晰」实为 36 层总和 vs per-layer 的单位错误——同口径
   复验 8B/32B 均重叠，判据不成立；
3. per-task 重校准长度失配：用任务最长样本 trace 重校准反而更差
   （musique 21.07 vs 27.45）——16-22K 校准 trace 与 ~8K 真实样本的层
   轮廓错位 + pm 平均口径低估 GQA head 级 far。

**最终形态**：离线 per-task 层轮廓校准（校准 trace 必须与推理分布同长度
同分布）+ 跨任务迁移失败如实报告（E5 跨任务 corr 0.05-0.89）。D' 价值
=「离线校准换 4.88× 索引省算」，其代价（校准管道 + 任务适配）在 Discussion
诚实讨论。【缺：held-out gate 验证 precision ≥0.98 硬阈值——审稿必问】

### 4.4 级联 kernel 化边界（跨级设计原则）【正文 v1】

两条从实测提炼的级联通用原则（对任何 coarse-to-fine 检索系统适用）：

1. **L1 能容忍多选，L2 不能**：L1 并列截断多选的块由 L2 精筛自然淘汰
   （trace 对拍 cov 一致）；L2 顶端没有下游兜底——纯 kernel L2 的阈值
   二分在 4bit 分数大量并列时每 head 过选 130-256，且 120 轮二分串行
   反慢 4.5×。**最终形态 = 打分 kernel 化 + 精确 topk 留给库**（torch
   radix topk）——「级联 kernel 化到哪一级为止」由下游是否有兜底决定。

2. **池边界即因果边界**：far 池上界 far_hi ≤ t+1−near_len、near 池下界
   sw_lo ≤ t——两池上界严格 < t 使 L1 误选的未来块（位置 > t 的 -inf 块）
   天然落在两池之外，**替代显式 causal mask**（省 [n,Hkv,S] masked_fill
   链，eager 版实测该链占慢路径 44%）。该不变式使双池直写 kernel 的
   正确性论证退化为池界不等式，是 M8/M10 kernel 化可行性的前提。

## 5. Implementation: Production-Scale Integration（2 页）

### 5.1 索引存储与维护【正文 v1】

**4bit 三张量格式**：每 token-head 的 L2 精筛表示存为 `kq_q`（uint8
[S,Hkv,nd2] 格点 0-15）+ `kq_sc`/`kq_mn`（fp32 每 token-head 双 scale），
共 40B/token-head（选择口径 nd2=32；投影口径 r=16 时 24B）vs fp32 128B
（3.2-5.3×）。**逐位一致论证**：格点值 0-15 在 fp32 精确可表，重建
`grid.float()·sc+mn` 与量化 `round(...)·sc+mn` 走同一 IEEE 运算序列 →
反量化零数值漂移（测试断言 torch.equal）——这使得 4bit 存储成为纯
存储优化而非近似。

**PCA 投影集成（M9）**：`SGLANG_TLI_PROJ_BASIS` 离线校准基（.pt
[n_layers,Hkv,D,r] 常驻 2.36MB，2048 token 校准集与全量同值）；k/q 侧
统一出口 `_k_refine/_q_refine` 六调用点同代码路径；同成本口径 far
recall +27%（0.4423 vs 0.3480）。一致性口径修正：投影是 GEMV，
cuBLAS 归约顺序使增量 vs 全量有 1e-7 尾差——逐位口径放宽为格点级
（格点一致率 1.0、select 集合一致）。

**增量维护（O(n) 精确索引）**：每 decode 步仅对新 token 量化+写入，
增量索引与全量重建**逐位一致**（测试断言）；预分配几何扩容避免
cat 版每步 4.8GB memcpy（S=131K 实测）。paged 寻址经 req_to_token
间接层，与 sglang KV pool 槽位管理解耦。

### 5.2 批量 kernel 组合（M8/M10）【正文 v1】

高并发批量化先决条件（M4）：per-layer 共享 index pool（kq/kmin/kmax
预分配 [R,cap]，R=请求数行池 + 堆行回收），select launch 数与 bs 无关。

**四 kernel 组合**（select_decode_batched，bs=32/131K 全函数
23.25→1.80ms，12.9×）：

| Kernel | 替代的 eager 阶段 | 机制 | 单项收益 |
|---|---|---|---|
| A | P5 flat gather+反量化物化+einsum（20.2ms/87.6%） | 每 token 256B 连续段 uint8 gather + 寄存器内反量化 + GEMV + 转置写 | 20.5→0.48ms（43×，1.55TB/s） |
| B | topk-min 全排序压实（0.88ms） | cumsum + 块展开；哨兵可在中段由下游 valid 掩掉 | ~0.3-0.5ms |
| C | masked_fill 链 + far/near 分数物化 | -inf 烘进打分 kernel 写出口径（双池直写） | 消 P6/P7 链 |
| D | 行 gather 268MB + permute 连续化拷贝 | rows 行间接直读 pool + GEMV（归约分组对齐 eager） | 消 2×268MB 拷贝 |

**工程经验（写入论文的定量教训）**：寄存器压力断崖——CHUNK×Hkv×nd2
fp32 元素/program ≤65K（8 warps），超限是 local memory 溢出的非线性
劣化（dual 输出 tile 512→128：1.06→0.43ms；CHUNK=1024 反而 3.2ms）；
Triton tile 大了不是慢是断崖。

**prefill 慢路径移植（M10）**：快路径失效边界——池==因果区的快路径
只在 nblk ≤ K1×Hkv 时成立，S=30K 时 nblk=480 ≫ 128 → 慢路径必然触发。
移植 M8 全套 + row_chunk 64→512（kernel 不物化 kq_c fp32 → 显存约束
与 chunk 宽度解耦）：末 chunk 983.8→140.0ms（7.0×），30K e2e prefill
双档 2.11×（787→373s / 1565→741s，两档加速比一致=线性项消除证据）。

**对拍口径（每 kernel 配套）**：C vs B torch.equal（topk 输入一致）；
full vs C 逐元素差 0；full vs eager jaccard 1.0（仅并列排序/哨兵 lane
不同）；GEMV 归约序尾差 2.4e-07 → topk 第 k 名 tie 翻转容忍对称差 ≤2。

### 5.3 CUDA Graph 兼容【正文 v1】

**三方法契约**：`init_cuda_graph_state`（捕获前预分配全层 pool：S_cap
封顶 `SGLANG_TLI_POOL_S_CAP`（131K 主表硬依赖 4bit），R_cap=max_bs+1，
行 0=哨兵；`_graph_locked` 禁扩容）；`init_forward_metadata_out_graph`
（host 维护行回收 + 稳态不变式「每活跃行内容=[0,seq_len)」+ pad→哨兵行
0，读 seq_lens_cpu 避同步）；`in_graph` no-op。

**图内路径**：统一增量 update_pool_rows_decode + 统一稀疏
select_decode_batched（静态宽度 W_far+W_near+W_forced）+ tensor 化
_sparse_attn——零 host 同步，replay vs eager **逐位一致 0.00e+00**。

**短行质量边界 → veto 钩子**：S ≤ K2（1024）时数学等价 dense；
1024 < S ≤ dense_threshold 有真实损失（S=1500 cov=0.835，4bit 近端
排名噪声）→ decode_cuda_graph_runner 的 duck-typed veto 钩子整批回退
eager。混跑（eager↔graph 交替）对拍 4.28e-08。

**显存账（H20 141GB）**：KV=144KB/token；kq pool fp32 ≈1KB/token/行
→ S_cap 全宽预分配会爆（32K×33 行×36 层 ≈40GB）→ 4bit + 封顶是
CUDA graph 与 131K 主表的共同前提。

## 6. Evaluation（3.5 页）

### 6.1 Setup【正文 v1】

**硬件**：2×NVIDIA H20-3e（每卡 141GB HBM，cc9.0，78 SM）。H20 的 Tensor Core
吞吐仅为 H100 的 ~15%——这一「缺陷」恰好构成**算力无关性试验台**：TLI 的收益
主张全部建立在 HBM 流量与索引效率上而非算力，若在 TC 受限的 H20 上质量与
流量收益成立，则其收益不依赖算力代际。**模型**：Qwen3-8B（36 层，GQA 8
kv-head × 128 维，rotate_half RoPE）为主表，Qwen3-32B（64 层）做跨规模复验。
**系统**：sglang fork（分支 two-level-indexer），TLI 以 attention backend 形式
全链路集成（§5）。**负载**：LongBench v1 英文 13 子集（200 样本/任务，
repobench 500；K2=1024、cmp_ratio=4、far_tokens=512、temperature=0）+ NIAH
（RULER 口径 S=32K，见 §6.2）。**baseline**：FullKV（triton dense）、
TIA@1024（同门第一代，内部消融基准）、TWI、Quest@1024、DSA（kernel 级
对比，模型不通用）。

**对拍文化声明（贯穿全部表格的数字可信度基础）**：每步 kernel 化均配
torch.equal / 逐元素 / jaccard 对拍（§5.2）；4bit 索引与增量维护逐位一致
（§5.1）；CUDA graph replay 与 eager 逐位一致（§5.3）；e2e 输出与 eager
路径逐字一致。§7 记录两例该体系主动撤回自身 headline 的案例。

### 6.2 Quality: Dual-Caliber Evaluation【正文 v1】

**主表（均值负载口径）**：LongBench 13 子集，FullKV/Quest/TIA/TWI/TLI 全部
同机同权重同打分器：

| task | FullKV | Quest@1024 | TIA@1024 | TWI | TLI(A+B'+D'gated) |
|---|---|---|---|---|---|
| hotpotqa | 53.48 | 45.74 | 53.89 | 48.98 | 53.96 |
| 2wikimqa | 38.29 | 38.46 | 38.27 | 36.16 | 39.07 |
| musique | 32.14 | 27.25 | 32.28 | 23.60 | 31.35 |
| passage_retrieval_en | 100.00 | 98.50 | 99.50 | 98.50 | 100.00 |
| qasper | 44.17 | 40.13 | 44.03 | 38.96 | 44.03 |
| multifieldqa_en | 53.40 | 51.01 | 53.19 | 48.11 | 52.98 |
| gov_report | 33.17 | 32.13 | 33.43 | 33.09 | 32.41 |
| qmsum | 23.53 | 22.21 | 23.90 | 23.98 | 22.77 |
| multi_news | 24.93 | 24.95 | 24.73 | 24.99 | 24.66 |
| narrativeqa | 25.61 | 20.64 | 22.18 | 26.19 | 23.14 |
| triviaqa | 90.71 | 87.55 | 89.82 | 89.59 | 90.22 |
| lcc | 68.81 | 68.34 | 69.14 | 66.22 | 68.74 |
| repobench-p | 66.50 | 63.41 | 66.40 | 63.87 | 65.59 |
| **AVG** | **50.36** | **47.72** | **50.06** | **47.86** | **49.92** |

TLI 距 TIA −0.14、距 FullKV −0.44——以 1024/9900 token（K2/S@9.9K）的
稀疏预算守住 dense 质量；9/13 子集不低于 TIA，3 子集反超
（hotpotqa 53.96 / 2wikimqa 39.07 / passage_retrieval 100.0）。

**负向任务逐项诊断（gate 机制的诚实拆解）**：TLI 相对 TIA 的三个掉分子任务
全部来自 D' 层跳过的 gate 错位，且消融 3/3 定位（关 D' 后 musique 31.35 /
qasper 44.03 / multifieldqa_en 52.98——qasper 与 TIA 精确同值）。诊断表把
「掉分」分解为「gate 判据误触发 + 校准 trace 与推理分布长度失配」，而非
两级索引本身的质量损失；这同时是 §4.3 gate 失败史的 e2e 证据。

**极端压力口径（NIAH，RULER 风格）**：S=32K、needle 深度 5–95% 十档、
双 seed × far 预算三点扫描（n=40/配置）：

| 后端 | seed1 / seed2 / 合并 | far 区命中（合并） |
|---|---|---|
| dense (triton) | 1.000 / 1.000 / 1.000 | — |
| TLI far=128 | 0.550 / 0.700 / **0.625** | 0.66 |
| TLI far=256（默认） | 0.650 / 0.600 / **0.625** | 0.66 |
| TLI far=512 | 0.550 / 0.700 / **0.625** | 0.66 |

三个结论：①far 区命中 0.66 与 §4.2 微观口径的 far recall@256（0.5–0.56）
量级吻合——微观-宏观口径自洽；②far 预算 [128,512] 完全不敏感——限制因子
是 far 池整体召回而非预算切分（K2_far=256 跨负载稳健的又一证据）；
③失败模式为「抓到 haystack 干扰数字」而非拒答——损失本质是 0.8% far 预算
对极端检索任务的物理上限，与 Quest/HISA 报告的 NIAH 损失同性质。seed1 的
「钟形」（0.550/0.650/0.550）被 seed2 反转、合并后持平——单 seed 差分结论
必须复测（§7 方法论案例二）。

**逐层质量**：36/36 层 mass 覆盖 diff<0.001，far-heavy 层（L03/L05）反超
TIA（0.9995 vs 0.9990 / 0.9997 vs 0.9929）——B' 近端名额保障的直接逐层
证据（fig6）。Qwen3-32B 复验：子空间选择 entry recall 差距 ≤0.08、
D' 跳 41/64 层 precision 0.993——机制跨规模存在、强度递减，如实报告。

### 6.3 Kernel Microbenchmark: Same-Machine Three-Way【正文 v1】

官方 kernel 原样接入统一 harness（Quest 摘官方 raft decode_select_k.cuh
编译 + page GEMV；DSA 官方 tilelang fp8_index 原样 import；计时 cudaEvent，
单 token per-layer-call 全链路，S=10K/40K/131K 三档）：

| indexer（ms/层） | S=10K | S=40K | S=131K | 索引存储/token | training-free | LongBench-13 |
|---|---|---|---|---|---|---|
| Quest 官方 | 0.066 | 0.077 | 0.107 | ~1KB | ✓ | 47.72 |
| DSA 官方 | 0.476 | 0.490 | 0.503 | ~132B | ✗（2.1B token warm-up） | — |
| TLI eager | 0.736 | 0.780 | 0.853 | ~336B | ✓ | 49.92 |
| TLI fused L1 | 0.604 | 0.658 | 0.787 | ~336B | ✓ | 49.92 |

四条诚实结论：①**Quest 索引最快**（HBM 带宽型 µs 级 GEMV+radix），但其
速度与 fp16 全 head min/max 的 1KB/token 存储（3× TLI）和 LongBench −2.2 分
绑定；②**DSA 0.5ms 与 S 无关**（TC 型生产级 kernel），代价是训练需求 +
每 token 8192 MAC 全量 GEMV；③**TLI 单 token 延迟当前不占优**（0.6–0.83ms，
eager topk/gather launch 主导，三家在 131K 都远离 HBM bound——排名反映
实现成熟度而非架构上限）；④TLI 的结构性优势在算法侧：每 token 索引 MAC
≈258 = DSA 的 1/32、存储 3×↓ vs Quest、+2.2 分 vs Quest——**算力/存储余量
由 M8 kernel 化兑现**：批量化后 select_decode_batched 全函数
23.25→1.80ms@bs=32/131K（12.9×），launch 数与 bs 无关（口径差异——单 token
vs batch——在表注中如实标注）。补充批量口径对比：Quest/DSA 官方无批量
decode_select 实现，此处只列单 token 口径 + TLI 批量数，避免跨口径直接
比较。

### 6.4 End-to-End Throughput【正文 v1】

**decode 优化轨迹（bs=32, S=9.9K, CUDA graph 口径）**：M3 eager 原型
1236.8 → M5 +graph 188.6 → M8 +kernel 99.0 ms/step——**累计 12.5×**
（323 tok/s@bs32），同一指标体系内逐级归因（批量化 3.8× / graph 1.7× /
kernel 1.9×）。vs triton dense 31.7 ms/step 仍慢 3.1×——9.9K 档 attention
流量尚未主导，诚实边界如实报告（fig9a）。

**S 增长收窄链（bs=16）**：9.9K 慢 3.48×（64.7/18.7）→ 30K 慢 1.55×
（67.1/43.4，N=256 双侧复测）——S 增长下差距单调收窄；tli 扩展比 1.49× ≈
triton 1.45×（30K/9.9K），两互不依赖口径交叉验证。两点线性外推（triton
attention 流量 ∝S 的斜率 1.20ms/K vs tli ≈K2 恒定）给出**翻转点 S≈44K**
（fig9c 虚线）：模型粗但方向明确——attention 成为 step 主导项后 triton 线性
劣化、tli 恒定。**H100 主表口径（bs≥16 × S=128K）远在翻转点之后**，是
收益验证位【缺：机器申请中，论文 headline 表】。

**prefill（30K，批分双档）**：M8 787/1565s → M10 kernel 化 373/741s，
**两档加速比完全一致（2.11×）**——慢路径成本 ∝bs×S 的线性项被消除的
直接证据（fig9b）。30K 档 tli prefill vs triton（58/115s）剩余 ~6.4×，
其中结构性 topk radix 机器占 select 侧 67%——算法级近似 topk 是后续工作，
如实划界。decode 与 prefill 数字均为「单次测量 + 差分法信噪比复核」口径
（N=256 复测、§7 方法论案例一）。

### 6.5 Overhead Analysis【正文 v1】

**索引效率（index-side efficiency，不冒充主算子胜利）**：TLI 索引 FLOP
2.57× 省算（vs dense 全维打分），D' 跳 13/36 层后 4.88×；每 token 索引
MAC ≈258 vs DSA 8192（1/32）。**存储账**：L2 索引 40B/token-head
（选择口径）/ 24B（PCA 投影口径，M9）vs Quest ~1KB/token（3–5×↓）、
DSA ~132B/token+训练；141GB 显存的 H20 上 131K×bs32 主表因此可行。
**CPU 捞取 overlap 三面墙（No-Go）**：far token 捞取 CPU 化提议被定量
否定——PCIe 带宽差 57×、选择集依赖 churn 22%/step、select 占比仅 12%
（捞不动也不值得捞）——入 negative results。

### 6.6 图表映射

fig9（a/b/c 三联）= §6.4 全部数字的图形化；fig5 = §6.5 FLOP/延迟；
fig6 = §6.2 逐层；fig1–4 归 §3/§4。所有数字可由 JSON（tli_m8_e2e_results /
tli_m8_e2e_long_results / tli_m10_bench / tli_niah_results /
kernel_comparison_indexers）复现，无手抄。

## 7. Discussion: Measurement Methodology（0.75 页，差异化卖点）

- 两例主动撤回：decode 差分法信噪比 <1（N=64 假打平 → N=256 复测修正）；
  NIAH 单 seed 钟形被 seed2 反转（合并 n=40 持平）。
- 原则：小样本差分方差系统性低估；复测不是推翻而是校准。
- phase 插桩会骗人 → kernel 级 profiler 复核（两次踩坑）。
- 【素材：报告§8b-14 测量学修正段、§8b-16 诚实口径修正段】

## 8. Conclusion

三层贡献重述 + 诚实边界（10K/30K 档 decode 未胜 dense，线性外推翻转点
S≈44K（fig9c），收益位在 S≥128K HBM 流量 + 大 batch）+ 未来工作（H100 主表、
held-out gate、RULER 全量）。

---

## 附录素材映射速查

| 论文节 | 报告章节 | 图/JSON |
|---|---|---|
| §1 Intro | §1/§8b-6/§8b-14 | fig1 |
| §3 Motivation | §3/§8b-7 | fig1、E4c |
| §4.1 A | §8b-7/E3/E3b/32B§8 | fig2 |
| §4.2 B' | §8b-7/E4c/§8b-16 | fig3、tli_niah_results.json |
| §4.3 D' | §8/E6/gate 失败史 | fig4 |
| §5 Impl | §8b-2~5/8b-8/8b-12/8b-13/8b-15 | — |
| §6.2 质量 | §4 主表/§8b-16 | fig6、tli_niah_results.json |
| §6.3 kernel | §8b-6 | kernel_comparison_indexers.json |
| §6.4 e2e | §8b-13/8b-14/8b-15 | fig9、tli_m8_e2e_results.json、tli_m10_bench.json、tli_m8_e2e_long_results.json |
| §7 测量学 | §8b-14 修正段/§8b-16 修正段 | — |

## 写作待办（并入任务链）

1. 【缺】H100 主表（S=131K×bs16/32）——§6.4 headline
2. 【缺】held-out gate 验证——§4.3 审稿防御
3. 【缺】RULER 全量（若 H100 短缺，NIAH 双口径可先行撑住质量叙事）
4. ~~图表升级 fig9~~ ✅（make_fig9.py：M3→M8 轨迹 + M10 prefill 双档 + S 收窄链含 44K 翻转点外推）
5. 多 seed 置信区间（主表 200 样本已有；e2e 曲线单次——按测量学 §7 原则标注）
