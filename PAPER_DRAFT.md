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

- fig1（H1 三层分解）+ 口径陷阱：mass coverage 恒 1.0 vs 剩余 mass coverage 0.515。
- far 挤出风险的形式化：全局 top-K2 下 far 平均占 569/1024 名额是分数噪声驱动
  （4bit 量化 + 基数效应），而非质量驱动→ B' 的动机。【素材：报告§8b-7 分区消融】
- GQA far mass 跨 head 极不均（L03 head3 独占 0.804）→ per-head 加权评估口径。

## 4. Design: TLI（3 页，核心章）

### 4.1 A：Position-Stable Subspace Coarse Filtering（L1）

- L1 块 min/max 上界只在低频尾维 d'=32（rotate_half 布局 [48..63]+[112..127]）。
- **非对称压缩定律**：L1 可粗（d'=16 免费）——L2 必精——但 L2 瓶颈在「表示方式」
  非「维数」（选择口径 δ=8 崩溃 0.378；投影口径 PCA16 0.532 ≈ 选择 32 维 0.557）。
  【素材：报告§8b-7 三实验、fig2】
- 机制证据：random/highfreq 崩溃（0.32/0.137）排除「任意降维」；随机投影崩溃
  （JL 保范数不保 GQA 点积排序）→ K 协方差结构是本质。
- 32B 泛化：lowfreq 仍最优但差距收窄 ≤0.08——「机制存在、强度递减」诚实结论。

### 4.2 B'：Far/Near Partitioned L2 Budget

- far 池 [sink, t+1-near_len) 独立 top-K2_far + 近端带独立 top-K2_near；滑窗
  强制位不占配额。
- **消融定机制**（三版本）：oracle far L1 存活 0.93-1.00（far 在 L1 被剪不发生）；
  hierarchical（L1 分区）No-Go；B' 增益=近端名额保障（L05 cov剩 0.600→0.726）。
  → 分区只需在 L2 终选级。【素材：报告§8b-7】
- 聚类代表 = negative result（E4c 严格预算：km_blk 0.09-0.39 最差）。
- far_tokens 敏感性：128-256 饱和；NIAH 口径下 [128,512] 不敏感（限制因子是
  整体召回 ~0.66）。【素材：报告§8b-16 双 seed】

### 4.3 D'：Offline-Calibrated Layer Skipping

- 层 far-mass 轮廓双峰 → far 质量低的层可整层跳过远端检索。
- 8B 跳 13/36、32B 跳 41/64（precision 0.993）——跨规模更强。
- **gate 失败史**（论文素材，三段）：在线信号 corr≈0 → far 总量判据单位混淆
  （32B 复验发现重叠）→ per-task 重校准长度失配。最终=离线 per-task 层轮廓
  校准 + held-out 验证（待办）。
- 【缺：held-out gate 验证（precision ≥0.98 硬阈值）——审稿人必问，H100 轮补】

### 4.4 级联 kernel 化边界（跨级设计原则）

- L1 容忍多选（L2 吸收）、L2 不能（无下游兜底）——纯 kernel L2 No-Go
  （4bit 并列过选 130-256/head + 120 轮二分串行反慢 4.5×）。
- 池边界即因果边界：far_hi≤t+1-near_len / sw_lo≤t → 垃圾块天然落池外，
  替代显式 causal mask。【素材：报告§M3-b、M8-KernelC】

## 5. Implementation: Production-Scale Integration（2 页）

### 5.1 索引存储与维护

- kq 真 4bit 三张量（uint8 格点 + fp32 双 scale），40B/token-head（3.2×）；
  重建≡量化逐位一致论证（格点 fp32 精确可表）。
- PCA 投影口径（M9）：r=16 → 24B/token-head，同成本 far recall +27%。
- O(n) 精确增量索引（增量==全量重建逐位一致）；预分配几何扩容。

### 5.2 批量 kernel 组合（M8/M10）

- 四 kernel：A（gather+寄存器反量化+GEMV，1.55TB/s）/ B（候选压实）/ C（双池
  直写，-inf 烘进写出口径）/ D（L1 行间接直读）。23.25→1.80ms（12.9×）。
- row_chunk 显存解耦：不物化 kq_c → 64→512 摊销。
- 寄存器压力经验值：CHUNK×Hkv×ND2 ≤ 65K 元素/program。
- prefill 慢路径移植（M10）：快路径失效边界（nblk>K1×Hkv）+ 双档 2.11×。

### 5.3 CUDA Graph 兼容

- 三方法契约（init 预分配/init_forward_metadata host 维护/in_graph no-op）；
  静态宽度 W_far+W_near+W_forced；replay vs eager 逐位 0.00e+00；
  短行 veto 钩子（S≤1024 数学等价 dense，1024<S≤2048 有真实损失）。

## 6. Evaluation（3.5 页）

### 6.1 Setup

- Qwen3-8B/32B、LongBench v1 + NIAH、2×H20-3e（141GB；TC 仅 H100 15%——
  算力无关性试验台叙事）、sglang 分支。
- 对拍文化声明：每步 kernel 化配 torch.equal/逐元素/jaccard 对拍 + e2e 逐字一致。

### 6.2 质量（两端口径）

- 主表：LongBench 13 子集（TLI 49.92 / TIA 50.06 / FullKV 50.36 / Quest 47.72）。
  【素材：报告§4 表】
- 负向任务逐项诊断表（musique/qasper=D' gate 错位，关后精确恢复）。
- NIAH（极端压力）：0.625 vs dense 1.0，far 区命中 0.66 与 E4c 吻合；
  far 预算不敏感。【素材：报告§8b-16】
- 逐层质量：36/36 层 diff<0.001、far-heavy 层反超（fig6）。

### 6.3 Kernel Microbenchmark（同机三方对比）

- Quest 官方（page GEMV + raft decode_select_k）0.066-0.107ms；
- DSA 官方（tilelang fp8_index）0.476-0.503ms；
- TLI eager 0.736-0.853 → fusedL1 0.604-0.787（单 token 口径）
  + 批量 select 12.9×=1.80ms@bs32/131K（batch 口径，口径差异如实标注）。
- 存储/训练/质量三列同表。【素材：报告§8b-6、fig5】

### 6.4 End-to-End Throughput

- decode 阶梯：M3 原型 1236.8 → M5 graph 188.6 → M8 kernel+graph 99.0 ms/step
  （bs=32，12.5×；vs triton 31.7 仍慢 3.1×——诚实边界）。
- 9.9K→30K 差距收窄链：bs16 慢 3.48×→1.55×（S 增长单调收窄，fig9c）；
  两点线性外推翻转点 S≈44K——H100 主表（S=128K）远在翻转点后；30K 扩展比
  tli 1.49× vs triton 1.45×。
- prefill：M7 快路径 2.08×@10K + M10 慢路径 kernel 化 2.11×@30K
  （累计短板 13.6×→6.4×，剩余结构性 topk 67%）。
- 【缺：H100 + S=131K + bs≥16 主表（机器申请中）——论文 headline 表】

### 6.5 Overhead Analysis

- 索引 FLOP 2.57×（跳层 4.88×）——index-side efficiency，不冒充主算子胜利。
- 存储账：40B（24B PCA）/token-head vs Quest ~1KB。
- CPU 捞取 overlap 三面墙 No-Go（PCIe 57×/依赖 churn 22%/占比 12%）。

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
