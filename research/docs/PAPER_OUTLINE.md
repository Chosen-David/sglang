# TLI 论文骨架大纲（2026-09-28 晨定稿，供正文写作直接使用）

> 事实底稿 = `/home/wangyuanshuo02/sglang/TWO_LEVEL_PAPER_REPORT.md`（§1-§8b-32 全部实测已收齐）；
> 图表 = `two-level-attention/exp/figures/`（fig1-10 PDF+PNG）；PPT = `~/sglang/TLI_progress_v7.pptx`（终局数据 5 页）。
> 本大纲按「速度卖点」定位组织（用户指示二选一，TLI = 速度：同精度梯队下 kernel 加速 + 收益区吞吐）。

## 0. 摘要一句话
两级 training-free 稀疏注意力 TLI：块级 minmax 上界粗筛 + far/near 分区 token 级精筛 + 离线层跳过，
同精度梯队下 kernel 级 1.3-5.1×（32K→128K）、单请求 e2e 1.285×@64K，RULER 官方口径对 Quest 领先 +2.91→+8.49 超线性。

## 1. Introduction
- 长上下文 KV 访问是 decode/prefill 双侧瓶颈；DSA 需训练，Quest page-min 下界长上下文崩塌（§8b-32：16K multikey_3 21 分）
- 贡献四条（对应报告 §2）：A 子空间粗筛（E3：lowfreq d'=32 recall 0.729≈全维）/ B' far/near 分区 L2 预算（E4c 修正后）/ D' 离线层跳过（E6+dyn gate）/ 系统级 kernel 化（M2-M11，E1 1.46×@32K→5.09×@128K）
- 图锚：fig7_architecture

## 2. Method
- 2.1 两级选择算法（minmax 上界 + 4bit 量化低频尾维子空间；报告 §5.0 设计）
- 2.2 far/near 分区预算（B'：far_tokens 128-256 饱和，E63 配比消融；near floor 保护）
- 2.3 动态层 gate（#60：prefill softmax far 统计 → 双峰阈值 → 幂等 skip_far；AVG 代价 −0.17）
- 2.4 系统集成（paged 寻址 / O(n) 精确增量索引 / 共享 index pool / CUDA graph）
- 图锚：fig1（H1 三层分解）、fig7

## 3. Kernels（速度卖点主体）
- 3.1 fused L1 块打分（3.6×/1.6×，E8-2）
- 3.2 L2 级联 dual kernel（单 launch 分区 topk + bf16+pad 直出，1.63×；M8 CHUNK 调优 22%）
- 3.3 M11 统一稀疏 attention fused gather+online softmax（kernel 级 11-16×）
- 3.4 DS topk（L1+far；near No-Go：4bit tie jaccard 0.72——负结果入正文）
- 3.5 同机三方对比（§8b-6：TLI vs Quest 官方 kernel vs DSA FlashMLA-indexer，统一 harness）
- 图锚：fig10（M8 kernels）、make_fig8/9

## 4. Evaluation
- 4.1 质量：RULER 四方法×3 长度终表（§8b-32 定稿表）+ LongBench 13 子集主表（TLI 49.92 vs TIA 50.06 vs FullKV 50.36，E5b）——**两口径互补并报**（单跳放大 TLI 代价 / 多跳放大小收益）
- 4.2 kernel 级 microbench：1.3-5.1× 梯度 + 三方对比 + negative（TMA/TC No-Go）
- 4.3 e2e：30B 单请求 64K 1.285×（32K 0.77× 如实）；8B 稳态 4.00×/1.94×；TP2 bs16 0.92×（launch-bound 诚实归因 = future work）
- 4.4 消融：A（E3）/ B'（E4c 严格预算+聚类降级消融）/ D'（E6+E60+gate 三任务消融）/ 预算敏感性 / 32B 泛化（§8）
- 4.5 profiling 方法论（§#65：microbench vs e2e 口径差 = host-GPU 流水线；10 万次同步排空案例）
- 图锚：fig3（E4c）、fig4（E6）、fig9a（e2e）

## 5. Related Work
- HISA（COLM 2026，block-coarse+token-refine 同构）/ OpenDSA / Quest / MoBA / DSA（§8b-31 竞争扫描已备）
- 定位差异：training-free + 上界（vs Quest 下界）+ 分区预算（vs 全预算 topk）

## 6. Limitations & Future Work（诚实口径资产）
- multikey_3+cwe 两任务族精度代价（K1 配额稀释+预算粒度，机制清晰可定位）
- 跨请求批量化 forward_extend（TP2 bs16 差 8% 根因）/ H100 主表 / gate stat per-row

## 写作顺序建议
Method → Kernels → Evaluation（数据全齐直接填表）→ Intro/Related → Abstract。
每次写作前从 json 复核数字（fwe/vt 列手抄出错教训，§8b-32）。
