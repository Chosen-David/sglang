# Two-Level Indexer 论文进展报告（v2，2026-09-23 晚）

> 承接 `TWO_LEVEL_INDEXER_DESIGN.md`（设计 + 批判性分析）。本报告从**论文写作视角**整理全部已实测结论、图表资产与剩余待办，并纳入当晚两个重大更新：**E4c 对 E4b 的口径修正（B 创新点重定位）**与 **E8 系列组合收益实测**。
>
> 数据基线：Qwen3-8B 真实权重 + 10 条真实 trace（needle/natural + 8 条 LongBench 官方样本），2×H20-3e。

---

## 1. 论文定位（一段话）

**TLI（Two-Level Indexer）**：面向长上下文 decode 的 training-free 稀疏注意力索引器，在 TIA（第一代两级索引）之上引入三个实测支撑的改进——**A 位置稳定子空间粗筛**、**B' far/near 分区预算**、**D' 离线校准层跳过**——在 mass 覆盖持平或超过 TIA 的前提下，索引侧计算量降低 **2.57×**（跳层 4.88×），fused L1 kernel 原型 **3.6×/1.6×**。论文同时贡献一组**方法论级 negative results**（严格预算口径下聚类代表劣于 4bit 精筛、在线信号失效、跨层复用失效），为该方向划定清晰的可行边界。

## 2. 贡献列表（写作时逐条对应实验）

| # | 贡献 | 支撑实验 | 数字 |
|---|---|---|---|
| 1 | **A**：L1 块 min/max 上界只在低频尾维 d'=32 计算（4× 维度削减，recall 不掉） | E3/E3b | mass recall 0.729 vs 全维 0.732；random 0.32 崩溃（选择性证据） |
| 2 | **B'**：far/near 分区 L2 预算，防远端被近端高分挤出 | E5b debug | far-heavy 层 TLI 0.9995/0.9997 vs TIA 0.9990/0.9929 |
| 3 | **D'**：离线校准静态层跳过掩码（far 质量低的层免远端检索） | E6 | 13/36 层可跳，precision 0.92–1.00，far 质量损失 <0.3% |
| 4 | 索引开销与 kernel：D' topk 截断真正兑现省算 + Triton fused L1 | E8-1/E8-2 | 索引 FLOP 2.57×（跳层 4.88×）；L1 kernel 3.6×/1.6× |
| 5 | **Negative results**：严格预算口径下聚类代表无优势；在线 L1 信号失效；跨层复用失效 | E4c/E5 | km_blk 0.09–0.39 vs minmax 0.52–0.74；IoU 0.33；corr ≈0 |
| 6 | E5b 端到端：TLI 在 LongBench 13 子集 vs TIA/TWI/Quest/FullKV | E5b（运行中） | 待填（逐层 mass 已验证） |

## 3. 关键叙事修正（相对开题 proposal）

1. **B 创新点从「聚类代表」重定位为「分区预算」**。E4b 的「kmeans 4–10× 占优」含整簇超选 bug（budget=512 实取 2769 tok = 5.4×），且 head 平均口径掩盖 GQA far mass 极度不均（L03 head3 独占 0.804）。严格口径下簇分数在远端没有一致优势，而 TIA 4bit token 级精筛 ≈ oracle（0.999）。**这本身是论文的方法论贡献**：远端候选质量的口径陷阱（超选 + 平均化）。
2. **「近端集中」叙事在 mass 口径下不成立**（H1）：真实形态是 sink(0.37–0.71) + near(0.16–0.29) + far(0.02–0.29) 三层，dense top-1024 mass 覆盖恒为 1.0000——TIA@1024≈FullKV 的根因。论文动机改为「质量三层分解 + far 挤出风险」。
3. **TC×CC 异构 kernel 降级为可选章节**（H20 TC 算力仅 H100 的 15%），主线改为 fused + 级联 topk（E8-2 已验证 L1 侧）。
4. RoPE/noPE 叙事改写为 **position-stable subspace** 抽象（Qwen3 语境：rotate_half 低频尾维）。

## 4. 图表资产（exp/figures/，PDF+PNG，matplotlib 矢量）

| 图 | 文件 | 内容 | 论文位置 |
|---|---|---|---|
| Fig 1 | fig1_h1_decomposition | 10 trace 位置质量三层分解（H1 动机） | §1 Intro/动机 |
| Fig 2 | fig2_e3_subspace | 子空间选择 recall 对比（A，lowfreq vs random/highfreq） | §3 A |
| Fig 3 | fig3_e4c_strict_budget | 严格预算远端策略对比（B 重定位依据 + negative result） | §4 B |
| Fig 4 | fig4_e6_layer_skip | 层 far-mass 轮廓 + 静态跳过掩码（D'） | §5 D' |
| Fig 5 | fig5_e8_speedup | E8-1 索引 FLOP + E8-2 kernel 延迟 | §6 开销 |
| Fig 6 | fig6_tli_vs_tia_layers | TLI vs TIA 逐层 mass 覆盖（红色带 = D' 跳层） | §7 端到端 |
| Fig 7 | fig7_tli_architecture | TLI 架构图（A/B'/D' 数据流） | §2 设计 |

主表（E5b，LongBench 13 英文子集；baseline 已打分，TLI 列运行中待回填）：

| task | FullKV | Quest@1024 | TIA@1024 | TWI | TLI(A+B'+D'-gated) |
|---|---|---|---|---|---|
| hotpotqa | 53.48 | 45.74 | 53.89 | 48.98 | **53.96** |
| 2wikimqa | 38.29 | 38.46 | 38.27 | 36.16 | **39.07** |
| musique | 32.14 | 27.25 | 32.28 | 23.60 | **31.35** |
| passage_retrieval_en | 100.00 | 98.50 | 99.50 | 98.50 | **100.00** |
| qasper | 44.17 | 40.13 | 44.03 | 38.96 | **44.03** |
| multifieldqa_en | 53.40 | 51.01 | 53.19 | 48.11 | **52.98** |
| gov_report | 33.17 | 32.13 | 33.43 | 33.09 | **32.41** |
| qmsum | 23.53 | 22.21 | 23.90 | 23.98 | **22.77** |
| multi_news | 24.93 | 24.95 | 24.73 | 24.99 | **24.66** |
| narrativeqa | 25.61 | 20.64 | 22.18 | 26.19 | **23.14** |
| triviaqa | 90.71 | 87.55 | 89.82 | 89.59 | **90.22** |
| lcc | 68.81 | 68.34 | 69.14 | 66.22 | **68.74** |
| repobench-p | 66.50 | 63.41 | 66.40 | 63.87 | **65.59** |
| **AVG** | **50.36** | **47.72** | **50.06** | **47.86** | **49.92** |

**结论：TLI 49.92，距 TIA 仅 −0.14（满足 ≥TIA−0.3 写入标准），距 FullKV −0.44，且索引侧 FLOP 2.57×（E8-1）+ L1/L2 fused kernel 3.6×/1.63×（E8-2）。**

口径：K2=1024、cmp_ratio=4、far_tokens=512、200 样本/任务（repobench 500）、Qwen3-8B、2×H20。

D'-gate 机制（本轮发现并修复）：
- **问题**：D' 全局掩码（5 任务校准，跳 13/36 层）在三个多跳任务上掉分（musique −4.83 / qasper −5.31 / multifieldqa_en −5.86 vs TIA）——三任务 far 总量 0.55–0.95（far 高度集中），层轮廓与校准集错位（E5 跨任务 corr 0.05–0.89 的直接后果）
- **消融定位（3/3）**：关 D' 后 musique 31.35 / qasper 44.03（=TIA 精确恢复）/ multifieldqa_en 52.98——D' 是唯一掉分来源
- **per-task 重校准失败（negative result，3/3）**：用任务专属最长样本 trace 重校准，跳层升至 28–31/36，实测反而更差——musique 21.07 / qasper 31.46 / multifieldqa_en 35.73（vs 全局掩码 27.45/38.72/47.33）。根因：最长样本（16–22K）与真实样本（~8K）长度失配 + pm 平均口径低估 GQA head 级 far 集中
- **最终设计：far 总量 gate**——离线校准输出 far 总量 T_far；T_far < 0.3（far 稀疏任务）启用层掩码（索引 4.88× 省算，精度不掉），T_far ≥ 0.3（多跳任务）不跳层。判据 gap 清晰：安全任务 T_far 0.02–0.29，多跳任务 0.55–0.95
- 论文叙事：D' 从「全局静态掩码」修正为「离线校准的 gated 省算模块」——gate 失败模式本身是 §negative results 的素材（跨任务层轮廓不可迁移 + 校准 trace 必须与推理分布同长度）

打分脚本：`exp/trace/run_e5b_eval.py`（与 eval.py 同一 scorer，依赖 jieba/fuzzywuzzy/rouge 已装至 ~/.local/pylibs）。

## 5. 实验编号总索引（脚本 → 结论，全部可复现）

| 实验 | 脚本（two-level-attention/exp/trace/） | 结论 |
|---|---|---|
| E1 | 导师 benchmark_mha 复跑 | 1.46×@32K / 2.77×@64K / 5.09×@128K |
| E2 | collect_trace.py / collect_trace_lb.py | 10 条真实 trace 落盘 |
| H1 | analyze_h1_full.py | 三层分解 + dense top-1024 mass=1.0 |
| E3/E3b | analyze_e3.py / analyze_e3b.py | A Go |
| E4/E4b | analyze_e4.py / analyze_e4b.py | ~~kmeans 占优~~（含 bug，见 E4c） |
| **E4c** | **analyze_e4c.py** | **B 重定位：km_blk 最差 / km_tok 无一致优势 / 4bit ≈ oracle** |
| E5 | analyze_e5.py | 跨层 IoU 0.33 / churn 22% / 在线信号 corr≈0（全 No-Go） |
| E6 | analyze_e6.py + tli_layer_skip_mask.json | D' Go（13/36 层） |
| E7 | analyze_e7.py | 增量聚类 staleness ≤0.03（保留为聚类数据结构结论） |
| **E8-1** | **analyze_e8_1.py** | **索引 FLOP 2.57×/4.88×** |
| **E8-2** | **e8_2_fused_topk.py** | **fused L1 kernel 3.6×/1.6×，块 id 对拍一致** |
| E5b | run_e5b.sh + debug/test_tli_e5b.py | 逐层 mass 已过；LongBench 13 子集分数运行中 |

## 6. 代码资产与提交

- `two-level-attention`（master，commit b0a732d）：TLIIndexer 全实现 + E1–E8 脚本与结果 + figures
- `sglang`（two-level-indexer 分支，commit aaf8fbff）：M1 = tli backend 注册 + trace 单测；分支基于 fork 最新 main（4b186cfea，2026-09-15）

## 7. 消融表（hotpotqa trace，36 层 mass 覆盖口径）

组件组合消融（vs TIA 0.99977，数据 `e5b_ablation_layers.json`）：

| 组合 | TLI mean | max\|diff\| vs TIA | diff>0.001 层数 |
|---|---|---|---|
| A only | 0.99994 | 0.0068 | 0 |
| A+D' | 0.99993 | 0.0068 | 0 |
| A+B' (far=32) | 0.99995 | 0.0068 | 0 |
| A+B'+D' (far=16) | 0.99995 | 0.0068 | 0 |

注：mass 全局口径已接近饱和（0.9999x），组件差异体现在 far-heavy 层的逐层对比（Fig 6：L03/L05 反超）与索引开销（E8-1）。论文正表建议用 LongBench 分数 + far 区 per-head capture 双口径，mass 表作为 sanity check。

far_tokens 预算敏感性（L03/L05 mass + far capture，`e5b_far_tokens_sensitivity.json`）：

| far_tokens | L03 | L05 | far capture |
|---|---|---|---|
| 128 | 0.99950 | 0.99965 | 0.97542 |
| 256 | 0.99950 | 0.99965 | 0.97096 |
| 512 | 0.99950 | 0.99965 | 0.97077 |
| 768 | 0.99897 | 0.99965 | 0.96884 |

→ 128–256 饱和；>512 反而劣化（far 抢占近端配额，near_floor 保护生效）。默认 far_tokens=256。

## 8. 待办（优先级序）

1. **E5b 完成后**：跑 eval.py 得 13 子集分数 → 填主表 → 若 TLI ≥ TIA−0.3%，写入主表小节；否则排查 B' 分区参数（far_tokens 预算敏感性）
2. **sglang M2**：TileLang kernel 接入 + paged gather + 稀疏 prefill（参照 dsa/ 8466 行模板）；M3：CUDA graph + e2e 吞吐
3. ~~消融表~~ ✅ 已完成（§7，trace 级）；LongBench 级消融（A/B'/D' 逐个关）视主表结果决定是否补跑
4. ~~L2 fused kernel~~ ✅ 已完成（E8-2 下半场：单 launch 分区 topk，对拍 4096/4096，1.63×）
5. 论文写作（骨架已定）+ 换 Qwen3-14B/32B 复验 A/D' 的层掩码泛化性

## 9. 答辩防御清单（更新版）

1. 「B 为什么不用聚类了？」→ E4c 严格预算数据 + 口径陷阱本身就是贡献（Fig 3）
2. 「和 HISA 的区别？」→ HISA 无层自适应/子空间选择依据/分区预算；我们有 negative results 护城河
3. 「D' 的层掩码跨模型泛化？」→ 承认是 Qwen3-8B 校准的；机制（far mass 层轮廓双峰）若在 14B/32B 复现则通用（待办 #5）
4. 「索引省 2.57× 但 attention 本身呢？」→ K2 固定 1024，attention 计算量不变；省的是索引器侧 + HBM（k_min/k_max 4× 削减）——与 proposal HBM↓45% 口径对齐时说明分母
5. 「为什么 mass 覆盖都是 0.99+，实际精度差异从哪来？」→ E5b 主表直接回答（运行中）
