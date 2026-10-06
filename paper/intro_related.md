# 1. Introduction / Abstract / 5. Related Work / 6. Limitations（论文正文草稿，2026-09-28 晨）

> 事实锚点：PAPER_OUTLINE.md §0/§1/§5/§6 + §8b-31 竞争扫描 + 各章草稿。

## Abstract（草稿）

长上下文推理中，KV 缓存访问同时制约 decode 与 prefill。现有
training-free 稀疏注意力方案面临精度-速度的困难权衡：基于下界近
似的方法（Quest）在长上下文检索任务上质量崩塌；基于训练的索引
（DSA）引入额外参数与训练成本。我们提出 TLI，一个两级 training-free
稀疏注意力索引：块级 minmax **上界**粗筛保证无漏选，far/near 分区
token 级精筛保护远端关键 token，离线层跳过与动态 gate 进一步削减
无效层计算。在统一 harness 下，TLI 选择链相对 dense 打分实现
1.46×@32K → 5.09×@128K kernel 级加速；e2e 收益区（64K 单请求）
1.285× 领先 dense。质量上，RULER 官方口径（3 长度 × 11 任务）
TLI 对 Quest 领先 +2.91 → +8.49 超线性扩大；相对全预算 topk 上界
索引（TIA）的代价集中于两个可精确定位的任务族（机制清晰）。全部
结果含负结果在同机同口径下诚实报告。

## 1. Introduction

**瓶颈**：长上下文 LLM 推理的注意力计算受 KV 缓存带宽制约——
131K 上下文 × 8B 模型的 KV 读取远超计算量，decode 每步全量访问
KV 成为吞吐上限；prefill 同样受全注意力带宽制约。

**现有方案的困境**：(i) training-free 方案中 Quest 的 page-min
**下界**分数近似会漏选真实高分页——干扰项随长度超线性增长时
质量崩塌（我们实测 RULER 16K multikey_3 仅 21 分 vs FullKV 100）；
(ii) 训练类索引（DSA/MoBA）引入可学习参数与 warm-up 成本，
且 indexer 质量依赖训练分布；(iii) HISA 等 block-coarse+token-refine
两级结构已被验证有效，但**表示（上界 vs 下界）与预算组织（分区
vs 全池）的质量-速度影响未被系统刻画**。

**我们的方案**：TLI 两级索引——L1 块级 minmax 上界粗筛（低频
position-stable 子空间 d'=32 + 4bit 量化，无漏选保证）；L2 far/near
分区 token 级精筛（far 独立配额防挤出）；D' 离线层跳过 + prefill
动态测层 gate。kernel 化贯穿（fused L1 / dual 级联 / 统一稀疏
attention fused kernel / DS topk）。

**贡献**（与实验编号一一对应）：
1. **上界粗筛的实证依据**：位置稳定低频子空间质量不打折
   （E3：0.729 vs 全维 0.732），上界保证与 Quest 下界近似的
   长上下文质量分化（RULER +8.49@16K）；
2. **far/near 分区预算（B'）**：GQA far 集中特性下的防挤出设计
   （E4c 严格预算口径，聚类代表降级为消融——负结果）；
3. **动态测层 gate（D' 演进）**：静态掩码跨任务不泛化的教训
   （−4.8~−5.9）→ prefill 动态 gate 代价 −0.17（#60）；
4. **系统级 kernel 化与双口径评测方法论**：kernel microbench
   与 e2e 的口径差（host-GPU 流水线效应）系统案例（#65），
   负结果全程入册（TC 化/TMA/聚类/near-DS/gate 判据五项 No-Go）。

## 5. Related Work

- **稀疏注意力索引**：Quest（page 采样 + 下界近似）、SnapKV/
  PyramidKV（observation-based 压缩）、H2O（累积重要性）。TLI 的
  差异 = 上界（vs 下界）+ 分区预算（vs 全池）+ 两级级联（vs 单级）。
- **训练类 indexer**：DSA（DeepSeek V3.2，I=Σw·ReLU(q·k)，64×128
  FP8 top-2048，warm-up 1000 步）、MoBA（gated block 稀疏）。
  TLI training-free/weight-free，与其在 kernel 级统一 harness 对比。
- **两级结构**：HISA（COLM 2026，block-coarse + token-refine）
  与 TLI 同构——「two-level」本身非 novelty；TLI 的增量 = 表示
  选择（上界）+ 预算组织（分区）+ 质量代价定位的系统刻画。
- **推理系统**：FlashAttention 系（计算侧）；PagedAttention/
  vLLM（显存组织）。TLI 在 sglang 生产形态集成（paged 寻址 +
  增量索引 + CUDA graph）。

## 6. Limitations & Future Work

**精度侧**（诚实定位）：TLI 相对全预算 topk 上界索引的代价集中
于 multikey_3（16K −37：块级粗筛分辨率被超多干扰 key 稀释）与
cwe（16K −24：分区挤占逐词召回预算）两任务族；RULER 单跳口径
放大该代价，LongBench 多跳口径下 TLI 与 TIA 差 <0.3——两口径
互补并报，用户可按任务形态选择。

**速度侧**：多请求批量形态（TP2 bs16 64K）仍差 dense 8%——
launch-bound（逐请求 Python 循环放大固定开销，GPU 名义满载但
带宽 util 仅 10-11%）；跨请求批量化 forward_extend 是明确修复
方向（decode 侧同类方案已验证 3.8×），列为 future work。

**其他**：动态 gate 的 far_stat 为 per-layer 单值，混合 batch
跨请求污染（生产语义应 per-row）；H100 吞吐主表（TC 形态下
#65 优化链的预期收益不同）留机器可用后补。
