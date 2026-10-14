# 稀疏注意力 / 长上下文 / KV cache 稀疏化前沿论文调研报告（2025-2026）

> 调研对象：TLI（两级注意力索引，块级粗筛 + token 级细筛 + (α,β,γ) 预算配比）
> 与 AsyncTLS（两级稀疏 + channel/INT4 细筛 + 异步 KV 卸载，arXiv:2604.07815，美团+人大）
> 调研日期：2026-10-14
> 执行 agent：科研文献调研 agent

## 0. 调研方法与来源分级（重要，请先读）

**环境限制：本轮 WebSearch / WebFetch / arXiv API / SemanticScholar / GitHub 全部不可达**
（curl 实测 google/arxiv/github 均 connection timeout，代理 `no_proxy=*`；WebSearch 六次查询均返回空）。
因此本报告为「本地已核实资料 + 训练知识库」双源拼接，**无法逐条联网校验 2025 下半年至 2026 的最新版本与数字**。

来源分级（每条目标注）：
- **[本地·已核实]** 项目内调研文档逐条核对过的 arXiv ID / GitHub 仓库 / 实验数字（可信度高，可作为下一步复现依据）。
- **[知识库·中置信]** 训练知识中确证存在的工作（方法、机构可靠），但**具体数字、最新版本、开源状态需联网复核**。
- **[知识库·低置信]** 记忆印象、可能误记，**严禁直接写入论文，需先联网确证**。

建议：恢复网络后，用本报告第三部分给出的「待核实清单」逐条补齐 arXiv ID 与主表数字。

---

## 1. 直接竞品（两级 / 层次 / 粗筛+细筛 结构，与 TLI 正面交锋）

### 1.1 HISA — 层次稀疏注意力索引（最接近 TLI 的「two-level」结构）
- **出处**：[本地·已核实] MuLabPKU / TransArch，**COLM 2026**，arXiv:2603.28458（v4，2026-09-10）。项目页 github.com/MuLabPKU/TransArch/tree/main/HISA_COLM_2026；TileLang kernels: github.com/tile-ai/tilelang/tree/main/examples/dsa_hisa；作者 sglang 分支 github.com/xuyufei-a/sglang_hisa/tree/hisa_pr（sglang PR #24672，核查时已关闭未合并）。
- **一句话方法**：作为 DSA indexer 的 drop-in 无参数替代，用层次（coarse block + token refine）结构复现 DSA 的稀疏 pattern（IoU>99%），消除 prefill 期 O(L²) indexer 构建瓶颈。
- **与 TLI / AsyncTLS 相似度**：**最高**。block-coarse + token-refine 两级结构与 TLI 的「L1 块级粗筛 + L2 近/远分区 token 细筛」**同构**；也做 prefill+decode 双阶段。
- **关键差异**：HISA 面向 DSA（有参数 indexer）做替代、pattern 对齐是「学习式校准」；TLI 是纯 training-free + 上界保证（minmax 上界无漏选）+ 分区预算。AsyncTLS 对比点：HISA 无异步卸载引擎，速度优化集中在 indexer 本身。
- **开源**：有（TransArch 项目页 + TileLang + sglang 分支，但 PR 未合并，属「公开可读 + 有上游适配尝试」）。
- **主表可比性**：**[本地·已核实]** v4 Table1 在 DeepSeek-V3.2、8×H200/TP8、128K 输入 +512 输出报告 TTFT 1.70×、TPOT 1.55×、完整请求 1.66×；**并发 1/2/4 时 TPOT 1.05×/1.38×/1.55×（并发敏感）**。未核到 RULER/LongBench 精度数字。模型非 Qwen/Llama 系，需改造后才有可比主表。

### 1.2 LongCat LSA / HI — 层次索引（美团）
- **出处**：[本地·已核实] Meituan，arXiv:2608.01662（v2）。权重 huggingface.co/meituan-longcat/LongCat-Flash-Lite-Sparse；sglang 适配 PR #32918（核查时 Open）。
- **一句话方法**：块级稀疏注意力（LSA）+ 层次 indexer（HI）——HI 将 KV 组织为多层索引，prefill 时按层收紧候选。
- **与 TLI / AsyncTLS 相似度**：高。HI 是「层次索引」思想最直接的近亲；但用途与 TLI 相反——**HI 仅用于 ≥256K 的 prefill，decode 明确关闭**。
- **关键差异**：**[本地·已核实]** HI 在 32K–128K 反而比 flat indexer 慢（speedup 0.79–0.82×），到 256K 才转正——说明「层次索引的收益临界长度」是关键变量；TLI 两级在 32K 即有 kernel 级 1.46×。这是 TLI 论文「什么时候启用、什么时候回退」的直接对照素材。
- **开源**：部分（Sparse 权重公开 + 基础适配 PR；论文全部 HI/HFA/KVP 优化系统未完整发布）。
- **主表可比性**：LSA 有模型 TTFT/TPOT 实测；decode 关闭 HI。模型为 LongCat-Flash 系（非 Qwen/Llama），不可直接并表。

### 1.3 AsyncTLS — 两级稀疏 + 异步 KV 卸载（本项目已立为对手）
- **出处**：[本地·用户提供] arXiv:2604.07815，美团 + 人大。无开源仓库。
- **一句话方法**：Quest 块级粗筛做 GEMM 化选择 + DoubleSparsity 式 channel+INT4 细筛 + 异步 KV 卸载引擎（卸载与注意力计算重叠）。
- **与 TLI 相似度**：高——「两级稀疏（粗筛+细筛）」结构一致，且是 Quest 家族内做 GEMM 化的代表。
- **关键差异**：AsyncTLS 的粗筛是「Quest 块界」、细筛是「channel+INT4」；TLI 是「子空间（d'=32）+ 4bit 格点 + 上界保证」+ far/near 分区预算。AsyncTLS 的独特增量在**异步卸载引擎**（TLI 未做卸载；若 TLI 论文要覆盖系统侧，AsyncTLS 是最应引述的卸载侧对手）。
- **开源**：无。按项目铁律「一律用官方源码，绝不语义复现」→ 登记等待官方开源走 vendor；否则只引述不作为主表对比。
- **主表可比性**：无公开数字，无主表可比。

### 1.4 Double Sparsity（DoubleSparse）— channel 粗筛 + INT4 细筛
- **出处**：[本地·已核实] arXiv:2408.07092（2024，v2），作者仓库 github.com/andy-yang-1/DoubleSparse。
- **一句话方法**：先按 channel 自适应选少量完整计算 channel（块级粗筛），再对 KV 做 INT4 量化 + token 级 top-k 细筛。
- **与 TLI / AsyncTLS 相似度**：**结构与 TLI 几乎同构**（粗筛+细筛），且细筛同为 4bit 量化 top-k；AsyncTLS 的细筛即沿用此路线。是「子空间/低维粗筛 + 量化细筛」范式的源头代表。
- **关键差异**：粗筛维度是 channel 采样（近似、无保证），TLI 粗筛是 minmax 上界（无漏选保证）；TLI 在粗筛侧多了 far/near 分区预算。
- **开源**：有（官方仓库 + Triton/CUDA 实现）。
- **主表可比性**：算法面向 decode（generation throughput、MMLU/GSM8K 质量保持）；**未核到 LongBench/RULER 主表数字**。模型 Llama-2/3 系。

### 1.5 FASA — 频率感知两级稀疏（本项目已实测）
- **出处**：[本地·已核实] ICLR 2026，arXiv:2602.03152（v3），作者仓库 github.com/wangyifei0047/FASA-ICLR2026，**Apache-2.0**。
- **一句话方法**：按 query 的 token 频率（freq-group）先离线校准、在线粗筛候选 token，再细筛 + 4bit（FASA-C 变体）。
- **与 TLI 相似度**：高——两级（粗筛候选集 → 细筛 top-k）；粗筛维度是「频率/频域」，与 TLI「位置稳定低频子空间」同属频域派。
- **关键差异**：**[本地·已核实，本项目 2026-10-13 实测]** 预算 1024 下 freq 细筛选择力弱：RULER 32K 6 任务×100 打分 AVG 24.00（s1=0），LongBench 16 任务 AVG16=33.31（vs fullkv 49.39，Δ=-16.08），检索/知识类全线掉分。FASA 论文只声称 LongBench、未声称 RULER。粗筛选择力弱是方法局限而非适配 bug（budget 单调性、校准样本数不敏感均已确证）。
- **开源**：有（Apache-2.0，monkey_patch 全量 forward + LongBench/PPL 脚本）。
- **主表可比性**：LongBench 有（Qwen 系），RULER 无公开数字——**主表上 FASA 只有 LongBench 可比，RULER 侧本项目实测可用**。

### 1.6 ClusterKV — 聚类粗筛 + 精确细筛（TLI ccluster 组合的直接对应）
- **出处**：[本地·已核实] DAC 2025，arXiv:2412.03213（v2），官方仓库 github.com/sjtu-zhao-lab/ClusterKV。
- **一句话方法**：prefill 期 kmeans 聚类构建簇，decode 用簇代表打分选簇、取回簇内原始 KV 做精确 attention。
- **与 TLI 相似度**：高——聚类粗筛（≈TLI 的 ccluster/cavg 组合）+ 原始 KV 精确细筛；TLI 的 sim_greedy 聚类用法与 ClusterKV 惯例一致。
- **关键差异**：ClusterKV 主要加速 decode（prefill 聚类成本 6–8%，占总 inference <2%）；TLI 是 prefill+decode 都做两级稀疏（indexer 族路线）。聚类代表质量受簇质量影响，TLI 的 minmax 上界粗筛无此问题（本项目已作消融：聚类代表降级为负结果）。
- **开源**：有（官方仓库 efficiency/bench_textgen.py 分别计 prefill 与 decode）。
- **主表可比性**：Llama-3.1-8B，1K budget，输入 8K/16K/32K，输出 256/512，Fig.12/13 含 prefill 段 + 完整 inference latency；**含 RULER 质量表**。

### 1.7 SparQ Attention — channel 两级 + 残差近似
- **出处**：[本地·已核实] ICML 2024，arXiv:2312.04985（v4），graphcore-research/llm-inference-research（tag 2024-05-sparq + benchmarks 分支）。
- **一句话方法**：per-query 动态选 top-α 通道做精确打分 + 残差近似补其余通道，再按通道分数两级选 token。
- **与 TLI 相似度**：中高——两级（通道粗 → token 细），通道维度与 TLI 子空间粗筛思路同源；SparQ 是「低维代理评分」范式早期代表。
- **关键差异**：SparQ 动态 per-query 通道选择（近似、有误差累积），TLI 静态低频位置稳定子空间（离线验证质量 0.729 vs 全维 0.732）；TLI 用 minmax 上界做第一级保证。
- **开源**：有（论文 tag + PyTorch/IPU 实现；后续作者另有 sparq-llama.cpp / sparq-gpt-fast）。
- **主表可比性**：Llama-2-7B，60–70% 稀疏保持质量；isolated attention 微基准为主，**无 LongBench/RULER 完整主表**。

---

## 2. 对比 baseline（单级块/token top-k、训练类 indexer、KV 压缩）

### 2.1 Quest — 块级 page min/max 下界 top-k
- **出处**：[本地·已核实] ICML 2024，arXiv:2406.10774（v2），github.com/mit-han-lab/Quest（含 bench_textgen.py / bench_prefill.cu / bench_page.cu）。
- **一句话方法**：KV 分 page，维护每 page min/max，decode 用 query 对 page 下界打分取 top-k。
- **与 TLI 差异**：单级 page 选择 vs TLI 两级；**下界近似会漏选真实高分页**——本项目实测 RULER 16K multikey_3 仅 21 分 vs FullKV 100（干扰项超线性增长时质量崩塌）。sglang 的 Quest 臂是 TLI backend 的减法 fork（共享 fused kernel），两臂差距完全归因于「索引构建+选择」。
- **开源**：有（官方仓库 + NVIDIA 微基准）。
- **主表可比性**：**有**。原论文 RULER（Llama-3.1-8B 系）；本项目在同一 harness 有 Quest 对照臂全套数字，可直接并表。

### 2.2 MoBA — 块级 gating MoE 稀疏（主打 prefill）
- **出处**：[本地·已核实] arXiv:2502.13189，Moonshot/Kimi。README 明说需继续训练，非 drop-in。
- **一句话方法**：block 级无参数 gating 的 MoE 式稀疏，chunk mean 每次 forward 现算；**论文明说「MoBA is used for prefill only, full attention during generation」**（1M prefill 6.5×）。
- **与 TLI 差异**：prefill-only + 现算均值粗筛 vs TLI prefill+decode 两级 + 增量维护索引；训练适配成本存在。
- **开源**：有（官方 repo）。
- **主表可比性**：1M 上下文 prefill 加速为主；Qwen/Kimi 系，有 RULER 相关数字（待联网复核）。

### 2.3 DSA / DeepSeek-V3.2 — Lightning Indexer（训练类 indexer 标杆）
- **出处**：[本地·已核实] DeepSeek-AI 2025.09 技术报告；官方仓库 deepseek-ai/DeepSeek-V3.2-Exp + DeepGEMM + FlashMLA（注意 FlashMLA main 已移除 V3.2/Hopper 支持，复现需固定历史 commit）。
- **一句话方法**：MLA latent 空间上 FP8 单 GEMM 计算 indexer 分数 I=Σ w·ReLU(q·k)，top-2048 选 token，再细筛 attention；warm-up 1000 步训练。
- **与 TLI 差异**：有参数、需训练、MLA/MQA 架构耦合（Qwen 不能直接套用）；TLI training-free/weight-free。两者在 kernel 级统一 harness 对比是论文口径。
- **开源**：有（官方 + DeepGEMM CUDA kernel）。
- **主表可比性**：16 个 benchmark 对比 V3.1-Terminus（MMLU-Pro/GPQA/HLE…），H800 集群 per-million-token 成本曲线；**无 RULER/LongBench 常规主表**，且模型为 671B，与本项目 Qwen/Llama 主表不可并表。

### 2.4 NSA — 原生稀疏注意力（训练类）
- **出处**：[本地·已核实] ACL 2025，arXiv:2502.11089，社区实现 github.com/fla-org/native-sparse-attention（原论文未核到官方训练权重链接）。
- **一句话方法**：压缩 + 选择 + 滑窗三分支，端到端训练，论证 KV 须跨 query head 共享。
- **与 TLI 差异**：训练类 vs training-free；TLI 无需训练即可 drop-in。
- **开源**：社区实现（非作者官方）。
- **主表可比性**：**注意**——ACL 最终版 9.0×/6.0× 是 training forward/backward kernel 实测，11.6× decode 是「Expected Speedup」推算，**不可当 TTFT/TPOT 实测引用**。

### 2.5 MInference 1.0 — prefill 专用动态稀疏
- **出处**：[本地·已核实] NeurIPS 2024，arXiv:2407.02490，github.com/microsoft/MInference。
- **一句话方法**：动态识别 vertical / block-structural / A2 三种稀疏 pattern，只在 prefill 加速。
- **与 TLI 差异**：prefill-only + 动态 pattern 识别 vs TLI 静态两级 + prefill/decode 双阶段；SnapKV 结合的 Table5 是质量兼容性、非 MInference 自身 decode 加速证据。
- **开源**：有。
- **主表可比性**：prefill 延迟（1M context），benchmark_e2e 实际计一次 model forward、非多 token 生成请求——**引用时须说明计时边界**。

### 2.6 SeerAttention — 学习式 top-k KV 稀疏化（Apple）
- **出处**：[知识库·中置信] Apple（Mike Goat 等），arXiv:2501.02161（2025），github.com/mikegoat/seerAttention。
- **一句话方法**：用 gate 网络学习每头每 query 的 top-k 稀疏 mask（而非固定 pattern），并在 kernel 层实现足量 top-k 检索。
- **与 TLI 相似度/差异**：同为 token 级 top-k 稀疏化；但 SeerAttention 需要一次校准/微调学习 mask（「训练一次、推理通用」），TLI 完全 training-free。可作「学习式 vs 启发式」对照。
- **开源**：有。
- **主表可比性**：Llama-3.1-8B 系，RULER 上保留 top-5% KV 维持精度的数字（**数字待联网核实**）。

### 2.7 ShadowKV — SVD 低秩 KV 存储 + 精确 top-k + 卸载
- **出处**：[本地·已核实] ICML 2025，arXiv:2410.21465（v2），ByteDance-Seed，github.com/ByteDance-Seed/ShadowKV。
- **一句话方法**：低秩 SVD 压缩 K/V 存入片外（CPU/磁盘），decode 时按低秩近似检索、取回精确 top-k token。
- **与 TLI / AsyncTLS 相似度**：高——**子空间（低秩）存储 + 精确检索**，与 TLI 子空间粗筛同源；**片外卸载**与 AsyncTLS 异步卸载引擎同方向（但 ShadowKV 无「异步/重叠」流水线）。
- **关键差异**：ShadowKV 是 decode 为主 + 低秩近似存储（精度有损），TLI 保留原始 KV + 4bit 索引。
- **开源**：有（test/e2e.py、models/base.py；benchmark 计时从 batch_prefill/初次采样/H2D/warmup 后开始）。
- **主表可比性**：Llama-3.1-8B/70B 长上下文，RULER 有数字（**待联网复核**）；预处理 SVD 成本有分解。

---

## 3. KV 卸载 / 分层存储系统（AsyncTLS 卸载侧对比；多为系统方向，非精度主表）

| 工作 | 出处 | 一句话方法 | 与 AsyncTLS 关系 | 开源 |
|---|---|---|---|---|
| **TurboAttention** | [知识库·中置信] Microsoft，arXiv:2412.08585 | INT4 KV cache + kernel-aware 稀疏 + 卸载与注意力计算重叠（compute/offload overlap） | 卸载重叠思想与 AsyncTLS 异步引擎直接同方向；INT4 细筛与 TLI 4bit 索引同源 | 待联网核实 |
| **LeanKV** | [知识库·中置信] NSDI 2025 | 异构 KV offload（register cache + DRAM 分层），精确 top-k 检索 | KV 分层存储 + top-k，AsyncTLS 卸载引擎的系统级对照 | github.com/LeanKV（待核实） |
| **Mooncake** | [知识库·中置信] Moonshot，arXiv:2407.00079，MLSys 2025 | KV-centric MoE serving：KV cache 共享 + 预取 + 传输异步化 | 分布式 KV 分层/共享，disaggregated KV 家族代表 | 部分开源 |
| **Preble** | [知识库·中置信] Microsoft，EuroSys 2025 | 反序异步流水线（decode 早层与晚层重叠），含 KV 复用/卸载 | 异步流水线系统，卸载侧旁证 | github.com/microsoft/preble（待核实） |
| **DistAttention / MemServe / MiServer / vStore** | [知识库·低置信] 2024-2025 系统方向 | 分布式注意力 / 记忆服务化 / KV 分离存储 | disaggregated KV 泛家族，仅引述用 | 各待核实 |

> 上述系统方向论文**均无 LongBench/RULER 精度主表**，适合在 Related Work 的系统侧段落引述，不作为精度 baseline。

---

## 4. KV 驱逐 / 压缩族（TLI Related Work 必引，非直接竞品）

- **[本地·已核实]** H2O（累积重要性驱逐）、SnapKV（观察窗口压缩）、PyramidKV、StreamingLLM（token 重要性）、KIVI/KVQuant/GEAR（量化压缩）、MiniCache（ICML 2025，层间冗余压缩，[知识库·中置信]）——与 TLI 的差异：这些是「驱逐/压缩后只用压缩集」，TLI 是「全量保留 + 两级检索」，保精度取向不同。
- **[知识库·中置信]** SwiftKV（Apple，训练式 4× KV 压缩）、CacheBlend（FAST 2025，预计算 attention 融合复用）、SVD-LLM（ACL 2025，SVD 压缩）——与 TLI 子空间/压缩侧可引述。
- **与 TLI 层跳过（D'/动态 gate）相关的**：MiniCache（层间冗余）值得引述为「层选择/跳过」证据。

---

## 5. 评测基准（主表可比性基础设施）

| 基准 | 出处 | 说明 | 与 TLI 主表关系 |
|---|---|---|---|
| **RULER** | [本地·已核实] NVIDIA 官方预生成（本项目用 KVCache-Factory 镜像，3 长度 × 11 任务） | 多任务合成长上下文（检索/多跳/变量替换等），string_match_all ×100 口径 | **本项目主表**；Quest/ClusterKV/FASA 等竞品可并表 |
| **LongBench v1** | [本地·已核实] 本项目用全量 | 16 任务真实文档 | 主表补充；FASA（AVG16=33.31 本地实测）可比 |
| **LongBench v2** | [知识库·中置信] THU，arXiv:2504.10433 | 多篇 32K 单文档、真实长文档、推理型任务，难度更高 | 项目已有 LBv2 审查（archive 2026-10-08），可用作高阶补充 |
| **ZeroBench** | [知识库·中置信] arXiv:2502.19654 | 长上下文完整检索的极严格测试（小模型全崩） | 仅引述，不并表 |
| **M-RCR** | [知识库·低置信] RULER 多跳扩展 | 更困难的多跳 needle | 待核实 |

---

## 6. 值得引述但非竞品的间接相关

- **[本地·已核实]** InfLLM（NeurIPS 2024，块代表 + 全局 token，decode 块选择）——块级选择的祖先，Related Work 必引。
- **[本地·已核实]** HiP Attention（arXiv:2406.09827，DeepAuto-AI/hip-attention，FSL-1.1-MIT）——双阶段（coarse min/max + fine select）的早期代表；**计时层级须注意**（Appendix D：RTX4090，prefill batch=1、decode batch=32，两阶段不可直接比较）。
- **[本地·已核实]** YOCO（ICML 2024，自解码器 KV 截断）——自回归 KV 压缩派，结构不同。
- **[知识库·中置信]** DeepSeek Speculative Attention（2025，低秩 KV 近似做 draft + 精确 attention 校验）——「近似粗 → 精确细」的投机式思想，与两级索引精神相通，可引述为并行直觉。
- **[知识库·低置信]** MInference 2.0 —— 印象存在，细节不清，**待核实后再决定是否引用**。
- **[本地·已核实·行业趋势]** sglang model_config 白名单已注册 GLM-5.x DSA、Longcat-Flash、Dots3、HYV4、DeepSeek-V4（FP4 indexer）——说明 **indexer 已是 2025-2026 行业主流**，TLI 作为 training-free 两级 indexer 的定位论证时可引用这一趋势；但这些实现无公开论文细节，不能直接并表。

---

## 7. 结论：竞品 / baseline / 引述 分档

### 直接竞品（主表必须交锋，优先复现/对比）
1. **HISA**（COLM 2026）——两级结构最接近；sglang 集成路径现成；**但无公开 RULER 精度数字**，需改造模型后才有主表。
2. **LongCat LSA/HI**（arXiv 2608.01662）——层次索引，重点看其「32K–128K 反而更慢」的收益临界，反衬 TLI 的 32K 即生效。
3. **AsyncTLS**（arXiv 2604.07815）——两级 + 异步卸载；**无开源，按铁律等官方 vendor**，论文中引述为卸载侧对手。
4. **Double Sparsity**——channel 粗 + INT4 细，TLI 结构同构的源头代表；decode 侧有吞吐数字。
5. **FASA**（ICLR 2026）——频域粗筛；LongBench 主表可比（本地实测 AVG16=33.31），RULER 无官方数字。
6. **ClusterKV**（DAC 2025）——聚类粗筛 + 精确细筛；有 RULER + prefill 成本分解，主表格式与 TLI 最贴近。

### 对比 baseline（已在统一 harness / 可直接并表）
7. **Quest**——本项目已有同 harness 对照臂全套数字，RULER 主表并表。
8. **MoBA**——prefill-only 主打，Qwen/Kimi 系，RULER 相关数字待复核。
9. **DSA / DeepSeek-V3.2**——训练类 indexer 标杆，kernel 级对比；671B 不可并表，作方法论锚点。
10. **NSA / MInference**——prefill/训练类代表，引用时须注明计时口径（NSA 11.6× 是 Expected、非实测）。
11. **SeerAttention / ShadowKV / TurboAttention**——学习式 top-k / 低秩存储 / INT4+卸载，各自对应 TLI 的细筛 / 子空间 / AsyncTLS 卸载侧。

### 值得引述（Related Work / 系统侧）
12. H2O/SnapKV/PyramidKV/StreamingLLM（驱逐派）、KIVI/KVQuant/GEAR/MiniCache（量化压缩派）、InfLLM/HiP/YOCO（块级/双阶段祖先）、LeanKV/Mooncake/Preble（KV 卸载系统）、LongBench v2/ZeroBench（评测基准）、DeepSeek Speculative Attention（投机式两级思想）。

---

## 8. 待联网核实清单（网络恢复后第一优先级）

1. HISA v4（arXiv 2603.28458）当前版本与 sglang PR #24672 状态；是否有 RULER/LongBench 数字。
2. LongCat（arXiv 2608.01662）HI 的 256K+ 精度数字。
3. AsyncTLS（arXiv 2604.07815）是否已开源。
4. FASA 官方是否补充 RULER 数字（对比本项目 24.00 实测）。
5. SeerAttention / TurboAttention / LeanKV / Mooncake / Preble / ShadowKV / MInference 2.0 的最新 arXiv 版本与主表数字。
6. LongBench v2（arXiv 2504.10433）与 ZeroBench（arXiv 2502.19654）的现状。
7. GLM-5.x DSA / Dots3 / HYV4 / DeepSeek-V4 FP4 indexer 是否有公开技术报告可引。
8. 2025 下半年–2026 新增的「粗筛+细筛」两级工作（本报告无法联网发现的新论文）。

---

*报告完。所有 [知识库·中/低置信] 条目在并入 TLI 论文引用前，必须按项目铁律先联网核实原文。*
