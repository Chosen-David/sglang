# 稀疏 Attention 论文的 Prefill 与 Decode 测速范围调研

核查日期：2026-10-06

## 结论

不是所有相关论文都只测 decode。应把三个问题分开：方法优化哪个阶段、论文实际测了哪个阶段、计时覆盖多大范围。

- FASA、Quest、SparQ、Double Sparsity、ShadowKV、ClusterKV 这条后训练 KV 检索路线，主要优化 decode。
- 其中 ClusterKV 的 Fig.12/13 明确画出 prefill 段，Fig.13(b) 还测了 Quest 的 prefill；Quest 原论文也有阶段占比数据，公开代码另有 prefill/metadata 微基准。ShadowKV 给出 prefill 预处理分解。因此“主要加速 decode”不等于“没有 prefill 相关数据”。
- HISA 最新 v4、LongCat LSA 和 DSA 已有两阶段的系统级效率证据，不过指标与启用模块不同。
- NSA 的训练 forward 实测、decode 理论预期，不能合并成“双阶段端到端实测”。
- 对 PSI 最有参考价值的实验范式是：Quest 的 decode 分层计时、ClusterKV 的预处理与总请求核算、HISA 的 TTFT/TPOT/端到端服务实验，以及 LongCat HI 的启用阈值实验。

这是一轮原论文与公开代码核查，没有运行这些仓库。下文“未报告”表示未在所核论文定位到对应实测，不表示算法绝对不能支持。仓库中出现 prefill 分支、质量表覆盖 sparse prefill，都不能独立证明 prefill 加速。

## 1 十篇核心工作的实验范围

| 工作及核查版本 | Prefill 加速实测 | Decode 加速实测 | 完整请求与预处理核算 | 论文证据 |
|---|---|---|---|---|
| FASA，ICLR 2026，arXiv v3 | 未报告；实验设置明确不做 prefill KV 优化 | 有 latency 图，但 Fig.7 未明确 attention-only 还是全模型 decode | Fig.3 是阶段耗时占比；不是加速对比。离线 FC 校准不计在线成本 | [§5.3 Fig.7；Appendix B.1/B.4](https://arxiv.org/html/2602.03152v3) |
| Quest，ICML 2024，v2 | 原论文无独立 prefill 加速主图；§3.1 有阶段占比。ClusterKV Fig.13(b) 测了它的 prefill | 有：attention 流水线及全模型平均单 token decode 延迟 | 原论文 Fig.10 不含 prefill，并排除 sampling；公开代码有模型 prefill 与 kernel 基准 | [§3.1/§4.3](https://arxiv.org/html/2406.10774v2)；[ClusterKV Fig.13(b)](https://arxiv.org/pdf/2412.03213v2#page=6) |
| SparQ Attention，ICML 2024，v4 | 未报告 | 有：isolated attention 微基准 | 主实测不是模型 TPOT 或整请求；K/V 初始化在微基准计时外 | [§6 Fig.8/Table4；Appendix F](https://arxiv.org/html/2312.04985v4#S6) |
| Double Sparsity，arXiv 2024，v2 | 无独立 prefill/TTFT 提速实验；算法针对 decode | 有：attention speedup、全模型 generation throughput | 公开 e2e 程序计时包住 prefill+decode；论文 Fig.6 对边界说明不足，不直接等同当前脚本 | [§6.2 Fig.5–6](https://arxiv.org/html/2408.07092v2#S6.SS2) |
| ShadowKV，ICML 2025，v2 | 无自身 prefill attention 加速结论；有预处理成本实测 | 有：decode throughput、decode 分解 | 主 e2e 脚本先 prefill 后启动 decode 计时；论文另报 prefill 的 SVD 等成本 | [§5.2；§7.6 Table12–13](https://arxiv.org/html/2410.21465v2#S7.SS6) |
| ClusterKV，DAC 2025，v2 | 不宣称加速 prefill；Fig.12/13 显式画 prefill 段，并报告 clustering 成本 | 有：decode throughput | 有含 prefill 的完整 inference latency，区分输入长度与输出长度 | [§V-C Fig.12–13](https://arxiv.org/html/2412.03213v2#S5.SS3) |
| NSA，ACL 2025 最终版 | 实测 training forward kernel，不能直接当模型 prefill/TTFT | Table4 为 Expected Speedup，非实测 TPOT | 无由上述结果支持的两阶段请求端到端结论 | [§4.1 Fig.5；§4.2 Table4，印刷页23084–23085](https://aclanthology.org/2025.acl-long.1126.pdf) |
| DSA / DeepSeek-V3.2，2025 技术报告 | 有：实际服务 benchmark 折算的 prefill token 成本 | 有：实际服务 benchmark 折算的 decode token 成本 | 属于 serving 级证据，图中指标不是直接 TTFT/TPOT，也不是单请求总时长 | [§2.3 Fig.3](https://arxiv.org/html/2512.02556v1#S2.SS3) |
| LongCat LSA / HI，arXiv 2026，v2 | LSA 有模型 TTFT；HI 有 prefill-shaped indexer 实测 | LSA 有模型 TPOT；**该实验 decode 关闭 HI** | 双阶段结论属于 LSA 整体，不能归因于 HI 单模块 | [§4.1.2 Table4；§4.3 Fig.5](https://arxiv.org/html/2608.01662v2#S4) |
| HISA，2026-09-10 v4，arXiv 标注 COLM 2026 | 有：SGLang TTFT | 有：SGLang TPOT，并给并发敏感性 | 有：完整请求 latency 和 throughput；包括下游 attention、KV 管理、调度等 | [§5.2 Table1；Appendix B.1/C.1](https://arxiv.org/html/2603.28458v4#S5.SS2) |

注意：“全模型 decode 端到端”与“整个请求端到端”是两个不同口径。前者可以包含所有 Transformer 层、MLP 和通信，却仍不包含 prefill。

## 2 公开代码是否足以复现对应结论

| 工作 | 已核到的公开实现或入口 | 复现时的边界 |
|---|---|---|
| FASA | [作者仓库](https://github.com/wangyifei0047/FASA-ICLR2026)，Apache-2.0；[llama_df.py](https://github.com/wangyifei0047/FASA-ICLR2026/blob/main/monkey_patch/frequency/llama_df.py)、LongBench/PPL/math 脚本 | 公开 forward 在 q_len=1 分支做选 token，prefill 保留 full attention。本轮未定位到与 Fig.7 对应的完整 FASA-C Triton 测速入口；不能把可运行质量代码视为该速度已可直接复现 |
| Quest | [模型计时 bench_textgen.py](https://github.com/mit-han-lab/Quest/blob/main/scripts/bench_textgen.py#L67-L96)；[prefill NVBench](https://github.com/mit-han-lab/Quest/blob/main/kernels/src/bench/bench_prefill.cu)；[KV/metadata append NVBench](https://github.com/mit-han-lab/Quest/blob/main/kernels/src/bench/bench_page.cu) | 模型脚本分别输出 avg_prefill_latency / avg_decode_latency；kernel 基准还覆盖 prefill 和 metadata 写入。存在入口不等于原论文已经公布对应独立数值 |
| SparQ | [论文 tag 2024-05-sparq](https://github.com/graphcore-research/llm-inference-research/tree/2024-05-sparq)、[benchmarks 分支](https://github.com/graphcore-research/llm-inference-research/tree/benchmarks)；scripts/run_sweep_pytorch.py / run_sweep_ipu.py | 论文的 isolated attention 与作者后续 sparq-llama.cpp / sparq-gpt-fast 全模型实现应分开记录 |
| Double Sparsity | [作者仓库](https://github.com/andy-yang-1/DoubleSparse)；scripts/run_attn.sh；[generate 内部](https://github.com/andy-yang-1/DoubleSparse/blob/main/models/generate.py#L137-L205)与[外层计时](https://github.com/andy-yang-1/DoubleSparse/blob/main/models/generate.py#L350-L387) | generate 计时包括 setup_caches、prefill、decode_n_tokens；prefill 调普通 model，decode 才走 sparse_forward；离线 channel 校准在外 |
| ShadowKV | [作者仓库](https://github.com/ByteDance-Seed/ShadowKV)，[test/e2e.py](https://github.com/ByteDance-Seed/ShadowKV/blob/main/test/e2e.py)、[models/base.py](https://github.com/ByteDance-Seed/ShadowKV/blob/main/models/base.py#L273-L314) | benchmark=True 的计时开始于 batch_prefill、初次采样、H2D 和 warmup 之后；主 throughput 是 decode 口径 |
| ClusterKV | [官方仓库](https://github.com/sjtu-zhao-lab/ClusterKV)，[efficiency/bench_textgen.py](https://github.com/sjtu-zhao-lab/ClusterKV/blob/public/efficiency/bench_textgen.py) | 分别计 prefill forward 与逐步 decode，适合核算索引成本是否被输出长度摊销 |
| NSA | [fla-org Triton 实现](https://github.com/fla-org/native-sparse-attention) | 这是社区实现。本轮未在原论文核到作者官方实现或训练权重链接；社区实测不能倒写成原论文实测 |
| DSA | [官方 V3.2-Exp 仓库](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp)、[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)、[FlashMLA](https://github.com/deepseek-ai/FlashMLA) | 包括 inference demo、研究参考实现及优化 kernel。FlashMLA 当前 main 已移除 V3.2/Hopper 支持；复现应固定 README 指定历史 commit，而非盲用 main |
| LongCat LSA | [官方 Sparse 权重](https://huggingface.co/meituan-longcat/LongCat-Flash-Lite-Sparse)；[SGLang 基础适配 PR #32918](https://github.com/sgl-project/sglang/pull/32918) | 权重公开、基础适配代码可见，不等于论文全部 HI/HFA/KVP 优化系统都已完整发布；PR 核查时仍 Open |
| HISA | [作者项目页](https://github.com/MuLabPKU/TransArch/tree/main/HISA_COLM_2026)、[TileLang kernels](https://github.com/tile-ai/tilelang/tree/main/examples/dsa_hisa)、[作者 SGLang 分支](https://github.com/xuyufei-a/sglang_hisa/tree/hisa_pr) | [PR #24672](https://github.com/sgl-project/sglang/pull/24672)含 prefill、paged decode、pooled-key cache 维护；核查时已关闭且未合并。pure decode 路径与 speculative/CP/offload 支持不能混说 |

仓库状态为核查日快照。代码“公开可读”、带开源许可证、提供优化 kernel、已进入上游服务框架，是不同层级；本报告未把它们混为一谈。

## 3 最值得注意的证据

### Quest 和 ClusterKV 都有 Prefill 相关材料

需要明确回答“有”。Quest 原论文 §3.1（PDF 第3页）给出 16K 输入、512 输出时 decode 占总时间超过 86% 的阶段占比；Fig.10 的主加速图仍只计 decode。Algorithm1 说明 Min/Max metadata 随 KV 插入维护，但未定位到原论文单列首次 metadata 构建成本或 TTFT 加速表。该 v2 PDF 共11页，后部是参考文献，没有另附 appendix。[Quest 原文](https://arxiv.org/pdf/2406.10774v2)

更直接的图在 **ClusterKV PDF 第6页**：Fig.12 的整体耗时柱带有斜线 Prefill 段；Fig.13(b) 并排比较 Quest 与 ClusterKV，也分别画了 Prefill。该图使用 Llama-3.1-8B、1K budget，输入为 8K/16K/32K，输出为 256/512。图没有给各 prefill 段精确数值标签，不应凭图伪造精确毫秒数据。[原图 Fig.12/13](https://arxiv.org/pdf/2412.03213v2#page=6)

Quest 的 prefill NVBench 调用 paged FlashInfer prefill；append-prefill NVBench 计 KV 与 metadata 写入，buffer 分配在计时外。后者不是剥离 KV 写入后的纯 metadata 额外开销。模型脚本的外层使用主机计时，复用前仍应核查同步、warm-up 与输入构造是否纳入；不能仅看到 avg_prefill_latency 字段就认定计时协议已适合 PSI。

### HISA 的新版本已经包含两阶段服务实测

旧版仅有 indexer microbenchmark 的印象已经过时。v4 的 Table1 在 DeepSeek-V3.2、8×H200/TP8、128K 输入+512 输出上报告：TTFT 1.70×、TPOT 1.55×、完整请求延迟 1.66×。Appendix B.1 说明使用 10 个随机请求并在 JIT warm-up 后测量。

这些数字有明显条件：Appendix C.1 中，并发 1/2/4 的 TPOT 分别为 1.05×/1.38×/1.55×。因此不能把高并发的 1.55×当成单请求 decode 的普遍收益。另一个 3.75×来自 A100 上 query_len=1024、64K context、固定 8K candidate budget 的 indexer 微基准，不是服务提速。[原文](https://arxiv.org/html/2603.28458v4)

### LongCat 要区分 LSA 整体与 HI 单模块

LSA 的 TTFT 和 TPOT 均有实测，但作者明确在 decode 关闭 HI；HI 仅用于至少 256K 的 prefill。Table4 的 HI 总耗时在 32K–128K 反而比 flat indexer 慢，speedup 仅 0.79–0.82×，到 256K 才转为正收益。这是 PSI 应报告“什么时候启用、什么时候回退”的直接理由。[原文 §4](https://arxiv.org/html/2608.01662v2#S4)

### ClusterKV 示范了 decode 优化如何诚实核算全请求

其 Fig.12 扫描输入 8K–32K 与输出 256–1024，报告整体 inference latency。聚类开销占 prefill 的 6–8%，占总 inference 不到 2%。后一个数字依赖该输出长度范围，不能外推到只生成几个 token 的请求。对 PSI 而言，关键不是必须加速 prefill，而是别把必须支付的预处理成本从总收益中消失。[原文 §V-C](https://arxiv.org/html/2412.03213v2#S5.SS3)

### FASA 与 NSA 的速度数字需要保留限定

FASA Appendix B.1 明确隔离 decode 优化；Fig.3 的阶段比例不构成两阶段加速结果。Fig.7 的计时范围没有足够说明，官方仓库的[相关问题 #1](https://github.com/wangyifei0047/FASA-ICLR2026/issues/1)也未给出作者解答。宜写“decode-oriented latency result，具体计时范围待明确”，不替作者补成 TPOT。

NSA ACL 最终版的 9.0×/6.0×来自 training forward/backward kernel；11.6× decode 来自 KV 访问量推导的 Expected Speedup。它能支持设计动机，不能充当 TTFT/TPOT 实测。[NSA 最终版](https://aclanthology.org/2025.acl-long.1126.pdf)

## 4 两个对照案例

### HiP Attention 同时测两阶段 但要注意计时层级与许可证

HiP 的 [arXiv v3 §5.2/Table3](https://arxiv.org/html/2406.09827v3)同时报告 prefill/decode attention latency，§5.4/Fig.7/Table5 另报全模型 decode 分解与加速；不能把前者的大倍数移植为后者或完整请求倍数。它还区分 mask refresh 和 cache reuse。Appendix D 规定 RTX4090，prefill batch=1、decode batch=32，因此两阶段数字不是同一 batch 的直接比较。

[作者仓库](https://github.com/DeepAuto-AI/hip-attention)提供 Triton 实现与[复现说明](https://github.com/DeepAuto-AI/hip-attention/blob/deepauto/dev/docs/REPRODUCE.md)。当前仓库声明 FSL-1.1-MIT，并限制免费商用；应称公开源码的补充案例，不能笼统列作无约束 MIT 开源。

### MInference 1.0 主要测 Prefill

这是相反方向的典型：NeurIPS 2024 的 [MInference 1.0](https://arxiv.org/html/2407.02490)专门加速 prefill。Fig.1b 报模型 prefill 延迟，Appendix D.2/Fig.10 给 attention kernel 及动态索引构建分解。与 SnapKV 结合的 Table5 是质量兼容性，不是 MInference 自身加速 decode 的证明。

[官方代码](https://github.com/microsoft/MInference/blob/main/experiments/benchmarks/benchmark_e2e.py)虽然文件名叫 benchmark_e2e，主要路径实际计一次整段 model forward；它不是“生成很多 token 的整个请求”。该脚本再次说明：必须读计时边界，不能只读 e2e 文件名。

## 5 对 PSI 最小实验集的建议

下面是实验设计建议，不是已完成的实验或已验证的收益。

### 可以把主贡献聚焦 Decode

如果 PSI 的近远分区和子空间索引目前只在 decode 启用，可以明确写成“面向长上下文 decode 的索引/检索优化”。不用为了“两个阶段都快”仓促扩大主张，但应补齐以下成本与结果。

| 层级 | 最少应报告什么 | 防止什么误读 |
|---|---|---|
| 索引 primitive | 投影/量化、coarse score、candidate selection、fine score、Top-K 各自及总时间 | 把单个最快 kernel 当完整索引器 |
| 完整 attention | 完整索引流程+KV gather/必要传输+最终 attention+KV/metadata 更新 | 只省打分，却把搬运与选择开销漏掉 |
| Prefill/TTFT | 与相同 dense prefill 基线比较；包括在线 basis/索引构建、投影缓存初始化 | 声称 decode-only 后隐藏首 token 代价 |
| 全模型 decode | TPOT 或明确的平均 step latency；写明 sampling、通信、同步是否计入 | 混淆 kernel speedup 和用户可感知收益 |
| 全请求 | 总 latency=首次处理与在线初始化+实际输出过程；固定输出长度并另测自然结束 | 只在长输出下摊薄成本，掩盖短输出负收益 |
| 服务负载 | 并发/批大小、输入/输出长度、P50/P95 TTFT/TPOT、output 与 total-token throughput 分开 | 把高并发吞吐当单请求时延，或把输入 token 计入的吞吐当生成速度 |

建议输入长度覆盖短、中、长与索引启用临界点；输出长度至少覆盖很短、常规、长生成。保持模型、GPU、dtype、batch/并发、TP、最终 KV budget 与质量目标一致；若比较不同预算，同时画质量–延迟曲线，不只比一个更稀疏的点。

### 应画一个收益转正图

对于固定输入长度 L，可用下式组织测量：

ΔT_total(L,G) = ΔT_prefill_and_setup(L) + Σ[g=1..G] ΔT_decode(L+g)

其中 ΔT 是 PSI 减 baseline，负值才是净省时。若近似每步节省 s 毫秒、一次性多付 c 毫秒，则最少需要约 c/s 个输出 token 才能回本；正式图应采用真实逐步耗时，因为上下文、batch 和缓存状态会变化。

离线校准与每请求构建分开列：前者报告一次性耗时与适用模型/数据，后者必须进入 TTFT 与全请求。如果 near/far 或子空间有额外完整 KV 副本、投影 cache、双布局和 CPU offload，也要同步报告峰值内存，避免以不一致内存条件解释性能。

### 推荐的论文表述边界

- 只有 kernel：写“indexer/attention kernel acceleration”。
- 测了整个模型的 decode：写“end-to-end decoding acceleration”，同时声明是否排除 prefill。
- 测了含初始化的完整请求：才把该数值写成“request end-to-end latency reduction”。
- 两阶段都实测收益：分别列 TTFT 与 TPOT，说明启用策略及负收益区间。
- 质量与速度由不同栈或不同版本测得：写清楚并补交叉验证，避免默认等价。

## 6 阅读与复现优先级

1. **ClusterKV**：最接近“prefill 建索引、decode 得益”的成本核算，且与 DAC 系统论文的表达方式贴近。
2. **Quest**：清楚区分流水线与全模型 decode；可以对齐指标定义。
3. **HISA v4**：与层次索引最直接相关，TTFT/TPOT/服务结果及并发效应都值得对照。
4. **LongCat LSA/HI**：重点看 HI crossover 与 decode 关闭策略，而非只摘最大速度。
5. **SparQ、Double Sparsity、FASA**：适合解释低维代理评分及选择机制，但要分别核对基准范围。
6. **ShadowKV**：如果 PSI 涉及低秩、额外 cache 或 offload，重点对照预处理和数据搬运分解。

最终判断：**decode-only 的研究定位完全可以成立；“只报 decode 最快的 kernel，不核算在线构建、TTFT 和请求总收益”则是更容易被质疑的实验缺口。**
