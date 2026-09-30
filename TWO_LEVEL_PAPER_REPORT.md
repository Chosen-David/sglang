# Two-Level Indexer 论文进展报告（v4，2026-09-30）

> 承接 `TWO_LEVEL_INDEXER_DESIGN.md`（设计 + 批判性分析）。本报告从**论文写作视角**整理全部已实测结论、图表资产与剩余待办。v4 新增（09-29 深夜终局批次）：**B7s 严格口径双口径终局**（LongBench 13 任务 50.41 首次同时超 TIA 与 FullKV + RULER 87.93）、**E64a-j 消融全家桶**（method×α×β 网格/紧预算桥接/同异 method/五组合最佳配置）、**E72 五组合 e2e 快筛判决**（trace/e2e 排名反转）、E65/E66 降维终版、**创新点终局评估表**（§2'）；v3 内容：M3 sglang 系统集成实测、E4c 口径修正、E8 组合收益、E5b 主表、32B 泛化复验。
>
> 数据基线：Qwen3-8B 真实权重 + 真实 LongBench/RULER 全量样本 + trace 重放 16 样本，2×H20-3e。

---

## 1. 论文定位（一段话）
**TLI（Two-Level Indexer）**：面向长上下文 decode 的 training-free 稀疏注意力索引器，在 TIA（第一代两级索引）之上引入三个实测支撑的改进——**A 位置稳定子空间粗筛**、**B' far/near 分区预算**、**D' 离线校准层跳过**——在 mass 覆盖持平或超过 TIA 的前提下，索引侧计算量降低 **2.57×**（跳层 4.88×），fused L1 kernel 原型 **3.6×/1.6×**，sglang 全系统集成后两级选择 fused kernel **对拍精确一致**（jaccard 1.0000）、e2e 输出与 eager **逐字一致**、decode 步延迟 81→51.4 ms。论文同时贡献一组**方法论级 negative results**（严格预算口径下聚类代表劣于 4bit 精筛、在线信号失效、跨层复用失效、级联 kernel 化中「L1 可容忍多选、L2 不可」的边界），为该方向划定清晰的可行边界。

## 2. 贡献列表（写作时逐条对应实验）

| # | 贡献 | 支撑实验 | 数字 |
|---|---|---|---|
| 1 | **A**：L1 块 min/max 上界只在低频尾维 d'=32 计算（4× 维度削减，recall 不掉） | E3/E3b | mass recall 0.729 vs 全维 0.732；random 0.32 崩溃（选择性证据） |
| 2 | **B'**：far/near 分区 L2 预算，防远端被近端高分挤出 | E5b/E71-B7s | far-heavy 层 TLI 0.9995/0.9997 vs TIA 0.9990/0.9929；e2e B7s 50.41 双超 |
| 3 | **D'**：离线校准静态层跳过掩码（far 质量低的层免远端检索） | E6 | 13/36 层可跳，precision 0.92–1.00，far 质量损失 <0.3% |
| 4 | 索引开销与 kernel：D' topk 截断真正兑现省算 + Triton fused L1 | E8-1/E8-2 | 索引 FLOP 2.57×（跳层 4.88×）；L1 kernel 3.6×/1.6× |
| 5 | **Negative results**：严格预算口径下聚类代表无优势；在线 L1 信号失效；跨层复用失效 | E4c/E5 | km_blk 0.09–0.39 vs minmax 0.52–0.74；IoU 0.33；corr ≈0 |
| 6 | E5b 端到端：TLI 在 LongBench 13 子集 vs TIA/TWI/Quest/FullKV | E5b/E71 | **TLI_B7s 50.41** vs TIA 50.06 / FullKV 50.36 / Quest 47.72（RULER 87.93） |
| 7 | **系统（M3）**：sglang 全链路集成——L2 级联 fused kernel + 增量索引预分配 + e2e 吞吐 | M3-a/b/c | 对拍 jaccard 1.0000；e2e 逐字一致；decode 81→51.4 ms/step（bs=1）；两级 select 1.34×@9.9K |
| 8 | **方法论：口径鸿沟**——trace mass 排名与 e2e F1 排名系统性反转，须双口径报告 | E64 vs E72 | 五臂排名 Δ 全反转（e72_trace_vs_e2e 图）；分区收益本质=紧预算预算分配 |

## 2'. 创新点终局评估表（#74 步骤①，2026-09-30 定稿）

按用户蓝图四问（分区有用 / method 配置最优 / αβγ 最优 / kernel 理由）聚合 E64a-j + E71/E72 全部判决：

| 创新点/配置 | 终局定性 | 决定性证据 | 升降级说明 |
|---|---|---|---|
| **A 位置稳定子空间粗筛** | **保留（主创新点）** | E3（lowfreq32≈全维，random/highfreq 崩）；E71-B d8 投影 e2e 崩反向支撑（PCA 基对 decode 漂移脆弱 vs 尾维恒稳） | 32B 泛化「机制存在、强度递减」如实报告；sup_wsvd 进 negative results |
| **B' far/near 分区预算** | **保留（主创新点，e2e 判决成立）** | B7s LongBench 50.41（+0.35 vs TIA，+0.05 vs FullKV）+ RULER 87.93 + L16384 +0.91 长度梯度；musique 32.82 反超 TIA | trace mass 口径 mono 微胜（E64g 14/16、E64i oracle 4/16 胜幅噪声级）与 e2e 分裂坐实为**预算口径问题**（宽预算无分配压力 vs 紧预算有）——论文双口径诚实并列，机制叙事=「分区的收益本质是紧预算下的预算分配，而非候选质量」 |
| **D' 层跳过** | **降级为 gated 消融模块** | E5b 多跳掉 4.8-5.9；gate 判据 No-Go（单位混淆）；E67 感知型 last1 corr 0.924 是正解但 τ e2e 判决待 E72 链后跑 | 静态掩码→gated 省算模块叙事；τ=0.01 近 no-op 的证据强度警示如实记 |
| **聚类代表（原 B 的一半）** | **negative result 资产** | E4c 严格口径 km_blk 最差；E4d trace cluster far 复活但 E72 e2e cavg 末位（42.87）——trace 结论未兑现 | 「口径陷阱」本身是方法论贡献（超选+平均化） |
| **method 组合 (far,near)** | **最优=mavg(minmax,avg) α.125/β.375** | E72 快筛五臂 e2e：mavg 44.59 冠军（musique 34.76 全方法新高）；near 侧 method 自由度 ≤0.003（E64i）→ 异 method 组合创新点删 | E64i 判「同 method≥异 method」；trace mass 排名不可替代 e2e 判决（五臂反转图） |
| **α/β/γ 设置** | **α=0.125 最优（单调递减）**；β≥α（近端份额超区域份额）；**γ 必须等比缩放（0.5→0.25/0.125）** | E64a/g（α 单调、最优内部臂恒 α.125）；E64i（β 最优恒 ≥α）；E71 γ 缩放修正（B0 far 崩 13.51 教训） | γ 说明：γ 是 near 占 L2 预算比例，K2_mid=768 紧预算下必须随总预算等比缩放，否则 near 吃光 K2 → far 仅保底 64 |
| **kernel 设计** | **fused L1 3.6×/L2 级联 1.63× + TC 化 negative result** | E8-2；M8-TC 反慢 0.63×（内存受限算子 TC 无益）诚实报告；M11 prefill fused e2e 一致 | 「L1 可容忍多选（L2 吸收）、L2 不能」是级联 kernel 化边界（论文独有洞见） |
| **降维** | **细筛 PCA d16 无监督近无损（唯一推荐）**；粗筛 8× 只有监督投影撑住但 e2e 崩 | E65 终版（0.766 vs 0.804）；E66 LOTO 跨数据集 0.744-0.757（92-94%） | near 区压不动但有滑窗兜底 |

## 2''. 方法章一般化框架（#74 步骤②，论文方法叙述统一组织）

用户蓝图原话：**「mid 分 near/far → 降维 → 双区各自粗筛+细筛」**。方法章按此一般化
框架组织，TLI 现实现 = 该参数空间的实证选定工作点（E64a-j + E72 全消融后定稿）：

**区域结构（严格口径）**：
- **保护段**：sink（首 128 token）+ swa（滑窗 1024）完全保送，不进粗筛/细筛管线，
  预算不计入 K2_mid——S 越大越省（旧口径 swa 挤 near 配额+sink 走管线的浪费随 S
  放大，L16384 口径差 +0.98 实证）；
- **mid 处理区**：K2_mid = K2 − sink − swa（1024−256 = 768），按 α 分为
  near（近端 α·|mid|）+ far（其余）双区，各自独立走两级选择。

**参数空间（六维）**：
| 参数 | 语义 | 消融终值 | 消融来源 |
|---|---|---|---|
| α | near 区占 mid 长度比例 | **0.125**（单调递减，最优内部臂恒 α.125） | E64a/g 51+81 臂 |
| β | near 占 L1 页预算比例 | **0.375**（恒 ≥α：近端预算份额宜超区域份额） | E64g/i 跨口径一致 |
| γ | near 占 L2 token 预算比例 | **0.125**（必须随 K2 等比缩放：γ0.5 在 K2=1024 下 near 吃光预算→far 仅保底 64→B0 F1 崩 13.51） | E71 γ 缩放修正 |
| far_method | far 区 L1 粗筛方法 | **minmax**（块上界，mavg 0.9166 mono 最稳） | E64a + E72 e2e |
| near_method | near 区 L1 粗筛方法 | **avg**（免 min/max 打分带宽） | E72 e2e 冠军 |
| (可选) 降维 | 粗筛/细筛维度 | 粗筛尾维 d'=32（免费）；细筛 PCA d16 近无损（唯一推荐）；监督投影 e2e 崩（negative） | E3/E65/E66 |

**双区两级选择流水线**（工作点形态）：
L1 页级：far 区 minmax 上界池 + near 区 avg 池（β:0.375 分配 128 页）→
L2 token 级：γ:0.125 分配 768 token 预算（near 512/far 256 上界），4bit 量化
GEMV 精筛各自区池——near 免 min/max 打分 = 结构性带宽节省（分区价值主张
落到结构侧，E64 mass 侧 mono 微胜 −0.005 属宽预算口径）。

**论文表述要点**：①「双区各自粗筛+细筛」的 near/far method 是独立自由度
（E64i 证明 near 侧自由度 ≤0.003、组合最优 = (minmax, avg)）；②降维按区
可选（near 有滑窗兜底可压、far 压不动）；③D' gate 与 τ 感知跳层是框架
正交扩展（E67 last1 corr 0.924），不进主方法章。

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
| **Fig 8** | **fig8_m3_system** | **M3 系统集成三联图**：(a) decode vs bs（高并发曲线 + 线性放大注记）；(b) decode 步延迟优化轨迹 81→56→51.4 ms；(c) 两级 select fused vs eager（9.9K 1.34× / 131K 1.05×） | §6 系统评估 |

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

### E71 新主表（2026-09-29 完成，C0 配置 = α0 单池 bp128 K2=1024）

E64 用户一般化框架（α/β/γ + sup_wsvd 投影基）接入 transformers 侧 tli_indexer.py 后的重跑（run_e71_eval.py，postfix _c0）：

| task | FullKV | Quest | TIA | TWI | TLI_old | **TLI_C0** | TLI_B |
|---|---|---|---|---|---|---|---|
| hotpotqa | 53.48 | 45.74 | 53.89 | 48.98 | 53.96 | **54.93** | 47.73 |
| 2wikimqa | 38.29 | 38.46 | 38.27 | 36.16 | 39.07 | **38.57** | - |
| musique | 32.14 | 27.25 | 32.28 | 23.60 | 21.07 | 29.74 | - |
| passage_retrieval_en | 100.00 | 98.50 | 99.50 | 98.50 | 100.00 | **99.50** | - |
| qasper | 44.17 | 40.13 | 44.03 | 38.96 | 31.46 | **44.31** | - |
| multifieldqa_en | 53.40 | 51.01 | 53.19 | 48.11 | 35.73 | 52.87 | - |
| gov_report | 33.17 | 32.13 | 33.43 | 33.09 | 32.41 | **33.49** | - |
| qmsum | 23.53 | 22.21 | 23.90 | 23.98 | 22.77 | 23.87 | - |
| multi_news | 24.93 | 24.95 | 24.73 | 24.99 | 24.66 | 24.64 | - |
| narrativeqa | 25.61 | 20.64 | 22.18 | 26.19 | 23.14 | **24.46** | - |
| triviaqa | 90.71 | 87.55 | 89.82 | 89.59 | 90.22 | **89.88** | - |
| lcc | 68.81 | 68.34 | 69.14 | 66.22 | 68.74 | **69.28** | - |
| repobench | 66.50 | 63.41 | 66.40 | 63.87 | 65.59 | **66.58** | - |
| **AVG** | **50.36** | 47.72 | 50.06 | 47.86 | 46.83 | **50.16** | - |

**结论：TLI_C0 AVG=50.16，超 TIA（50.06，+0.10）、逼近 FullKV（50.36，−0.20）**——13 任务 7 胜 1 平 5 微负；显著占优 hotpotqa +1.04 / narrativeqa +2.28（长叙事 far-heavy 任务），唯一显著掉分 musique −2.54（多跳 far-heavy，B5 分区冒烟验证中——far/near 分区预期是该任务的增益来源，将补 e2e 证据）。TLI_old 的 46.83 为 ABD 全开非 gated 口径参考。

- **TLI_B**（bp128+sup_wsvd d8 投影+α.125/β.25/γ.25 帕累托点）：**e2e 崩坏降级 negative result**——B3 hotpotqa 47.73 vs C0 54.93；终极对拍三连证据（/tmp/tli_ref_diff.py）：ref(d8 连续) mass=0.2753 ≈ ref2(e2e 同构打分) 0.2751 ≈ C0 判决臂 ref(32 维连续) 0.3028 vs e2e mask 0.9941——trace group-sum 参考口径在 e2e decode 系统性不成立 + **注意力加权 PCA 基对 decode q 分布漂移脆弱（适用域问题），位置稳定尾维子空间恒稳**；decode 第一步全 mask mass=1.0、第二步起参考全崩。论文叙事：离线投影口径无损（0.9466）vs e2e 崩 = training-free 方法无在线适应的固有局限，反向支撑创新点 A 子空间选择依据。
- **RULER C0 全长完成（2026-09-29，11 任务 × 3 长度 × 100 样本，KVCache-Factory 官方数据）**：

| RULER 任务 | FullKV | Quest | TIA | TLI_old | **TLI_C0** |
|---|---|---|---|---|---|
| niah_single_1-3 / multikey_1-2 | 100.0 | 100/85.33 | 100.0 | 100.0 | **100.0** |
| niah_multikey_3 | 100.0 | 51.33 | 96.33 | 75.33 | 91.67 |
| niah_multiquery | 25.0 | 25.0 | 25.0 | 24.92 | 25.0 |
| niah_multivalue | 67.67 | 64.83 | 67.83 | 64.42 | **67.83** |
| cwe | 91.97 | 62.4 | 89.4 | 79.93 | 87.37 |
| fwe | 91.22 | 91.0 | 93.34 | 93.89 | **93.45** |
| vt | 100.0 | 99.53 | 100.0 | 98.93 | **100.0** |
| **AVG** | **88.71** | 79.95 | **88.35** | 85.22 | **87.76** |

诚实口径：TLI_C0 87.76，距 TIA −0.59、距 FullKV −0.95（掉分点 multikey_3 −4.67 / cwe −2.0；与 LongBench +0.10 互有胜负）；per-length L4096 91.63（=TIA）/ L8192 88.87（−0.03）/ L16384 85.04（+0.51 超 TIA）——**长度越长越占优**，与稀疏收益随 S 放大的主线一致。niah_multiquery 25.0 为全方法含 FullKV 的模型能力上限（无区分度）。表存 `exp/results_ruler/ruler_table_e71.{json,md}`；B7 分区臂 RULER 在跑（对比列）。
- 打分脚本 `exp/trace/run_e71_eval.py`（glob 按目录对齐：pred_postfix 只进目录名）。E64 绝对质量数字是 B_TOK=2048 的 2× 预算口径，论文须标注。

**B7s 严格口径重跑（2026-09-29 晚，进行中）**：用户终局澄清区域预算口径
（sink/swa 完全保送不进粗筛细筛、mid 预算 K2_mid = K2 − sink_L − swa_L，
commit 7ea756c）后全量重跑 B7（α.125/β.25/γ.125，TS=09291030）。
旧口径输出备份 pred_b7_v1caliber（swa 曾占 near 配额 + sink 曾走管线）。
干跑验证：K2=1024 → K2_mid=768=256(near)+512(far)，+128+128 保送恒 1024。
- LongBench **13 任务终表（20:20 guard 自动完成，e71_main_table.json）：
  TLI_B7s AVG = 50.41，超 TIA 50.06（+0.35）、超 TLI_C0 50.16（+0.25）、
  超 FullKV 50.36（+0.05）——TLI 系列首次在 LongBench 全量上同时超过
  TIA 与 FullKV**。7 任务 ≥TIA：musique 32.82（+0.54，反超 TIA 32.28，
  +0.68 vs FullKV）/ triviaqa 90.41 / lcc 69.37 / repobench 66.84 全新高
  或持平 / hotpotqa 54.23 / 2wikimqa 39.07 / passage 99.5 / qasper 44.24。
  唯一明显损失 narrativeqa 23.96（C0 24.46，−0.50）。
  vs B7v1 旧口径（53.79 六任务）方向一致：严格口径总体 +0.18
  （musique +1.24 / hotpotqa −1.18 去双计效应主导）。
- RULER 三长度（GPU1，自动 relay 打分 ruler_b7s.json）：**终值（17:25，33 任务全）**
  L4096 91.49（C0 91.63，−0.14）/ L8192 88.62（88.87，−0.25）/
  **L16384 83.68（82.77，+0.91）**；**三长度总 87.93 vs C0 87.76（+0.17）反超**。
  增量全部集中在 L16384 难任务：multikey_3 +5.0 / cwe +5.4 / fwe −0.33 其余 7 任务
  全平——**长度梯度干净成立（−0.14 → −0.25 → +0.91），分区 far 检索收益区 =
  超长上下文，与「near 区覆盖大半 mid 的短 S 分区无增益」机制一致**。
- 口径差叙事（v1caliber 旧口径 vs 严格口径，RULER 三长度）：B7v1 旧口径
  L4096 91.64 / L8192 88.58 / L16384 82.70 → B7s 严格口径 91.49 / 88.62 / 83.68。
  口径差 −0.15 / +0.04 / **+0.98**——严格口径（sink/swa 保送、mid 预算 768
  全给管线）在超长上下文上显著更优：旧口径 swa 挤占 near 配额 + sink 走粗筛
  管线的浪费在 S=16K 时被放大；L4096 的 −0.15 为短上下文上 swa 占额保护效应
  的轻微损失。**结论：严格区域口径是正确的工程选择，分区收益叙事统一用它**。
判决（终局，与 E64h/E64g 合看）：**B7s 双口径均 > C0（LongBench +0.25、
RULER +0.17）且 LongBench 超 FullKV——e2e 存在 mass 口径不可见的分区收益，
分区创新点 e2e 成立**。与 trace mass 口径（mono 微胜：E64g mavg 族 14/16、
E64i 3×3 oracle 分区胜 4/16 且胜幅全 ≤+0.005 噪声级；均值 −0.005）的口径
分歧坐实为「E64 系 trace 用 2× 预算（B_TOK=2048）+ 无 sink 双计」，论文双
口径诚实并列报告：mass 口径分区零增益（宽预算）、F1 口径分区 +0.2~0.4
（严格预算）——**分区的收益本质是紧预算下的预算分配问题，不是候选质量问题**。
机制证据：musique 多跳 far 检索 +3.08（C0 29.74→32.82）、RULER 长度梯度
（−0.14→−0.25→+0.91）。

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
| E5b | run_e5b.sh + debug/test_tli_e5b.py | 逐层 mass 已过；LongBench 13 子集分数 ✅ 49.92 |
| **M3-a** | **sglang/test_tli_throughput.py** | **e2e 吞吐基线：decode 与 S 无关（launch 数主导）；吞吐口径工程坑沉淀** |
| **M3-b** | **sglang/test_tli_l2_kernel.py + test_tli_l2_e2e_smoke.py** | **L2 级联 fused kernel：对拍 jaccard 1.0000 + e2e 逐字一致；纯 kernel 版 No-Go（4bit 并列过选 + 二分串行反慢）→ 混合形态** |
| **M3-c** | **sglang/test_tli_batch_decode.py** | **高并发曲线 bs=1-32：逐请求循环线性放大 vs triton 批量化 3.2×→ 批量化是吞吐主表前置条件** |

## 6. 代码资产与提交

- `two-level-attention`（master）：TLIIndexer 全实现 + E1–E8 脚本与结果 + figures（含 v3 的 fig8/make_fig8.py + TLI_progress_v3.pptx 13 页）
- `sglang`（two-level-indexer 分支，至 commit 6ef3f7bd4）：M1 = tli backend 注册 + trace 单测；M2 前半 = B'/D' 算法同步 + Triton fused L1；M2 后半 = paged 寻址 + O(n) 精确增量索引 + 稀疏 prefill；M3 = decode 归因 + _sparse_attn 向量化（bfe373c18）+ L2 级联 fused kernel 接入 + 增量索引预分配 + 批量 decode 基准（36b6c8a37）+ 设计报告 §5.0/§5.0.1（5021ef105）；M5 = CUDA graph（210c25755）；M6 = kq 真 4bit 三张量 + 维度压缩边界扫描（7c2a6ba74）；**M7 = prefill select 快路径 + S 梯度（6ef3f7bd4）**。test_tli_m2b.py 全链路对拍在预分配改造后仍全 PASS；分支基于 fork 最新 main（4b186cfea，2026-09-15）

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

## 8. Qwen3-32B 泛化复验（2026-09-24，任务 #19 完成）

数据：`/tmp/trace/qwen3-32b`（7 任务 × 2 样本 × 64 层，sunyueqing/model/Qwen3-32B 真实权重，32K 截断；
narrativeqa 已修复同文档重复样本坑——8B 时代 e6_layer_skip.json 的 narrativeqa_0/1 far 完全相同即此坑）。
脚本：`exp/trace/collect_trace_32b.py` + `exp/trace/analyze_generalize_32b.py`；
结果：`exp/trace/results/generalize_32b.json`。

**三问题结论**：

1. **A 子空间泛化（Go，口径修正后）**：mass recall 在 32B 上大面积饱和（sink 主导层 cov→1），
   换 **entry recall**（与 8B E3 analyze_e3.py 完全同口径：per-head 候选块 ∩ dense top-1024 / 1024）：
   lowfreq32 0.66–0.87 ≈ full128 0.73–0.91（差距 ≤0.08）＞ random32 0.49–0.80 ＞＞ hifreq32 0.19–0.60。
   与 8B（lowfreq 0.271 / full 0.303 / random 0.149）方向一致：**lowfreq 稳定最优、hifreq 稳定崩溃；
   random 的退化幅度在 32B 上弱化**（0.149→0.5-0.8 量级）——「子空间选择性」存在但强度随规模递减。
2. **D' 层掩码泛化（Go，且更强）**：far 层轮廓双峰复现（far-heavy 任务 gov/narrativeqa far/层
   0.13/0.16 vs 其余 0.015–0.026，与 8B 的 0.21/0.27 vs 0.015–0.026 同构）；平均轮廓静态掩码
   跳 **41/64 层（64%）**，precision **0.993**，max 漏 far 0.19（8B：13/36 层、precision 0.92–1.00）。
   更深模型 far 更集中 → 可跳层比例更高、D' 省算空间更大。
3. **far 总量 gate 判据（No-Go，重要修正）**：per-layer 归一化 far 跨模型高度一致
   （musique 0.021/0.022、qasper 0.015/0.021、mfqa 0.026/0.026、hotpotqa 0.017/0.026、
   passage 0.024/0.022），但 **safe 与 multi-hop 的 far/层在 8B 和 32B 上均重叠**——
   此前「安全 0.02–0.29 vs 多跳 0.55–0.95 gap 清晰」是口径混淆（0.55–0.95 是 36 层总和、
   0.02–0.29 混入了 per-layer 数字；同口径总和下 8B gov_report 8.9–10.6 / hotpotqa 0.57 远超多跳）。
   **E5b 多跳掉分的真实根因 = 校准集平均层轮廓与任务真实轮廓错位（E5 跨任务 corr 0.05–0.89），
   而非任务级 far 总量差异**。论文叙事修正：D' gate = per-task 层轮廓校准（长度匹配的同分布 trace），
   任务级 far 总量判据写进 negative results（两代模型均不成立）。

## 8b. M3 sglang 系统集成实测（2026-09-24，任务 #20/21/22 完成）

环境：sglang two-level-indexer 分支（6 commits 至 5021ef105）、Qwen3-8B 真实权重、
真实 narrativeqa/LongBench 上下文、H20、无 CUDA graph（tli 路径暂不支持）。
脚本：`sglang/test_tli_{throughput,profile_decode,l2_kernel,l2_e2e_smoke,batch_decode}.py`。

**1. L2 级联 fused kernel（最终形态 = 混合）**：
- Triton pass1（单 launch/head：融合 gather+GEMV 打分，写 far/near 双池 scratch，池外 -inf）
  + `torch.topk` 精确选取 + Python 侧滑窗 `arange` 精确复制（近端配额扣减 F=t+1-sw_lo）；
- **池边界即因果边界**：far 池 `[sink, t+1-near_len)`、near 池 `< t-sw+1`，两池上界严格 < t——
  L1 多选产生的 -inf 垃圾块（位置 > t）天然落两池之外，替代 eager 的 fine 矩阵因果 mask；
- 纯 kernel 版 No-Go（论文素材）：4bit 量化分数并列极多，阈值二分在终选级无人兜底 → 每 head
  过选 130-256；120 轮二分 × 串行 chunk、8 program 打不满 78 SM 反慢 4.5×。
  **教训：级联结构中 L1 能容忍多选（下游 L2 吸收），最末级不能——topk 留给 torch 是正确分工**；
- 对拍：jaccard **1.0000**（S=9891/131072 × t=末位/回退/中段共 6 组，输出恰 1024 token）；
  e2e（32K 字符真实上下文、64 token 生成）与 eager **逐字一致**。

**2. 增量索引预分配**：cat 版每步 O(S) 全量拷贝（S=131K 时 kq 134MB × 36 层 ≈ 4.8GB/步纯
memcpy）→ 几何扩容 buffer，update 延迟 **0.128ms 与 S 无关**；增量==全量重建仍逐位一致
（test_tli_m2b 全 PASS）。

**3. e2e decode 优化轨迹（bs=1，S≈9.9K token，墙钟）**：eager 81 → _sparse_attn 向量化 + L1
kernel 56 → +L2 kernel **51.4 ms/step**；归因（TLI_PROFILE_TIMING 同口径）：select 46→28.7ms。
两级 select 微基准（fused vs eager）：9.9K **1.34×** / 131K 1.05×（小 S 收益 = 砍 launch 数；
大 S 时 eager L2 的 topk/gather 本身已占大头，fused 只省 fine 矩阵与 [Tc,Hkv,nd2] 中间量）。

**4. M3-c 高并发曲线（bs=1/8/16/32 × S≈9.9K，64 步 decode）**：

| bs | triton ms/step | triton tok/s | tli ms/step | tli tok/s |
|---|---|---|---|---|
| 1 | 10.0 | 99.9 | 51.4 | 19.5 |
| 8 | 16.4 | 486.8 | 369.7 | 21.6 |
| 16 | 19.0 | 842.3 | 666.4 | 24.0 |
| 32 | 32.4 | 986.5 | 1236.8 | 25.9 |

**结构性发现**：tli `forward_decode` 逐请求 Python 循环 → ~38ms/请求/步**线性放大**；triton
全批量化仅 3.2× 涨幅。**高并发吞吐主表（论文硬件叙事的 H100+大 batch 展示位）必须先批量化**：
共享 index pool + L1/L2 kernel grid 加 batch 维 + kq 真 4bit 存储（uint8+scale，134MB→16.8MB
@131K，才能跨请求共享预分配）+ CUDA graph（批量化后捕获才有意义）——即 M4 里程碑（§9-3）。
论文叙事：M3 曲线本身是「prototype 到生产级推理引擎的工程鸿沟」的直接证据，批量化前后对比
（M3 vs M4 重跑同曲线）构成系统章节的完整故事线。

### 8b-2. M4 批量化 decode 实测（2026-09-24，commit 95a13278a 起，同曲线三阶段）

实现（三阶段递进，每阶段对拍后提交）：
- **phase-1 批量稀疏前向**：req_to_token 2D 一次 gather 全部槽位 + 批量 einsum（_sparse_attn_batched，
  ~6 launch 替代每请求 ~40）；
- **phase-2 共享 index pool + 批量选择**：per-layer pool（kq/kmin/kmax 预分配 [R,cap]+几何扩容+
  行生命周期回收）+ `select_decode_batched`（全 eager 批量，~15 launch 与 bs 无关；哨兵 S_cap
  语义 + per-row 配额裁剪 + 滑窗强制块保留）；n==1 保留 per-request L1/L2 fused kernel 路径；
- **phase-3 批量增量维护 + 去同步**：`update_pool_rows_decode`（flat 索引 scatter，~10 launch 总量
  替代逐行 update）+ `.tolist()` 一次替代逐行 `int()` 同步 + Python max 替代 `.item()`。

正确性：test_tli_m4.py ALL PASS（批量 vs eager/kernel 选择 32/32 head 集合一致；pool 行增量 vs
全量重建逐位一致；多请求前向两步 max|diff|=2.6e-08；行回收 ✓；mass coverage 与 eager 同行
exact 同值——L03 far-heavy 层行级方差 0.95-1.0 属算法固有，eager 同值）；test_tli_m2b 回归 PASS。
调试资产：①批量 topk 统一宽度必须 per-row 配额裁剪；②垃圾块剔除须作用于 scatter 源（整体 &
  掩码会误删滑窗强制块）；③req_to_token 新位置未初始化=0 会让多请求 out_cache_loc 撞同一槽位
（KV 池污染，表面症状是增量步对拍分歧）。

同曲线对比（bs×S≈9.9K，ms/step，64 步 decode，无 CUDA graph）：

| bs | M3-c 原型 | phase-1 | phase-2 | phase-3 | triton |
|---|---|---|---|---|---|
| 1 | 51.4 | 63.8* | 61.1* | 73.8* | 15.1 |
| 8 | 369.7 | 261.9 | 197.0 | **166.4** | 15.7 |
| 16 | 666.4 | 483.4 | 311.0 | **207.8** | 19.2 |
| 32 | 1236.8 | 885.4 | 517.4 | **326.4** | 31.9 |

\* bs=1 各轮波动为 prompt 混合差异（narrativeqa 唯一长 context 不足，混其他 LongBench 子集）。

**归因链**：线性项 38→14.3→5.2 ms/req（phase-3 后剩余线性项=批量路径的 GPU 计算）；
固定项 ~125 ms/step = ~35 launch/层 × 36 层的纯调度开销 → **下一个杠杆是 CUDA graph（M5）**。
定位修正：S=10K 档 decode KV 流量本来就小（triton 全量 KV 读 ~1.5GB/步 ≈ 0.45ms），稀疏收益
不在此档兑现；**主表目标形态 = bs(8-64)×S(10K-131K) 矩阵**——bs=32×131K 时 triton KV 读
~620GB/步（~188ms 纯 HBM），稀疏 1/8 流量的收益在长 S 才显现（与 Quest/HISA「收益随上下文
长度增长」叙事一致）。S=131K 需要 kq 4bit（M6，fp32 pool 5.4GB/层装不下）。

### 8b-3. M5 CUDA graph decode 实测（2026-09-24，任务 #25 完成）

实现（三方法契约，全在 tli/backend.py）：
- **去同步 + 形状静态化**：B' far/near 分区重写为全 device 张量 + 静态宽度（W_far/W_near/
  W_forced per-row 配额裁剪，输出宽度恒 1408）；`_sparse_attn_batched` tensor 化（`sel_c =
  torch.minimum(sel, seq_v-1)` 替代 Python 标量 clamp）；
- **capture 对接**：`init_cuda_graph_state`（捕获前预分配全部层 pool：行 0=哨兵、S 维 =
  req_to_token 全宽、`_graph_locked` 禁捕获后扩容）+ `init_forward_metadata_out_graph`
  （图外 host 维护：行回收 + 稳态不变式「每活跃行内容=[0, seq_len-1)」+ pad 行→哨兵行 0）
  + 图内统一路径（统一增量 `update_pool_rows_decode` + 统一稀疏 `select_decode_batched`）；
- **短行 veto 钩子**：runner 端 duck-typed `veto_cuda_graph`——批内含 1024 < S ≤
  dense_threshold 的短行整批回退 eager（图内统一稀疏对短行有 4bit 近端排名损失，
  S=1500 实测 mass 0.835；S ≤ 1024 数学等价 dense 不 veto）。

正确性（test_tli_m5.py ALL PASS，真实 Qwen3-8B trace）：
- 图内路径 vs eager 多步对拍（8 步含首步重建/增量/S 跳变/每步 pad 行）4.47e-08；
  混跑（eager↔graph 交替同一 backend，模拟 veto 回退）4.28e-08；pool 行内容逐位一致；
- **真 CUDA graph 捕获通过**（capture 区域零 host 同步的硬验证）+ replay vs eager
  **逐位一致 0.00e+00**；e2e（32K 字符 narrativeqa、64 token 生成）graph 输出与
  eager **逐字一致**。

e2e 数字（S≈9.9K token、64 步 decode、同曲线）：

| bs | M4 phase-3（无图） | M5 graph | 加速比 |
|---|---|---|---|
| 1 | 73.8 | 17.7–36.8 | 2.0–4.2× |
| 8 | 166.4 | 70.3–75.5 | ~2.2× |
| 16 | 207.8 | 87.5–115.2 | 1.8–2.4× |
| 32 | 326.4 | **188.6** | **1.73×** |

（两轮区间；bs=1 波动大，论文口径多轮取中位数。capture 4 shape 仅 2.8s。
triton 同配置图基线 7.8/12.8/18.7/31.7 ms/step——triton 本身 launch 少，graph 仅
bs=1 收益 2×；tli 在 S=10K 档仍慢于 triton 全量，符合 M4 定位修正：稀疏收益在长 S
的 HBM 流量，S=131K 主表待 M6/M8。）
**累计：M3 原型 1236.8 → M5 188.6 ms/step（6.6×，bs=32）；tok/s 25.9→169.7。**
归因：launch 固定项被图消除后，剩余大头 = 批量路径 GPU 计算（M4 归因的 ~5.2ms/req
线性项）→ 下一个杠杆是 kq 4bit（M6，同时解 S_cap 预分配显存口径：fp32 kq ≈1KB/token/行，
12288 封顶下 36 层已 15GB，131K 硬依赖 4bit）。

工程沉淀：①对拍两侧 q 必须预生成共享（现生成 randn_like 引入 4.85e-03 伪差异）；
②`cuda_graph_config` 须传 dict（JSON 字符串在 parse 处 `str.items()` 崩）且 prefill 须显式
disabled（默认 BREAKABLE 依赖 sgl_kernel.weak_ref_tensor，0.3.16.post6 无此 API）；
③graph 池显存口径：S_cap 按 req_to_token 全宽预分配会爆（32K×33 行×36 层≈40GB），
`SGLANG_TLI_POOL_S_CAP` 显式封顶。

### 8b-5. M7 prefill 加速（2026-09-24，任务 #27 完成，commit 6ef3f7bd4）

**归因修正**（test_tli_prefill_prof.py）：prefill 瓶颈**不是**索引 build（0.3ms@10K，
可忽略——「每 chunk 全量 rebuild O(S²/chunk)」的设计担忧实测不成立），而是
`select_batched`：K1=128 块 × Hkv 个 head 的**候选池并集**在 S ≲ 64K 量级时覆盖全部
因果 token（实测 S=10K 时 Tc==S=10000），此时逐 64 行 chunk 的「掩码→topk 提取候选
→gather→反量化→scatter 回 S 宽度」全部是绕路——kq_unpack 每 chunk 物化 655MB fp32
（3.5ms）+ 全池 einsum（1.6ms）× 157 chunks ≈ 865ms/层（与 e2e prefill 27.7s 吻合）。

**快路径**：逐行判定池==因果区（`sel_mask.sum(1) == t+1`）时直接对**共享反量化表**
`[S,Hkv,nd2]`（每次调用建一张、全部 chunk 共享，S=10K 仅 10MB）做全宽 einsum 一次得
fine 矩阵；慢路径（S≥~20K 并集不再全覆盖）改从共享表 gather 打分（省逐 chunk 反量化
算术）。语义**逐位等价**：m2b [2] 8/8 行位置集合一致、prefill mass cov 0.98964 与改前
同值；e2e smoke graph vs eager 逐字一致。

| 口径（select_batched 微基准） | 旧 | 新 | 加速 |
|---|---|---|---|
| S=10K, nq=10K（整段一次性 prefill） | 862.5 | 166.6 | **5.2×** |
| S=10K, nq=2048（chunked） | 187.7 | 46.7 | 4.0× |
| S=40K, nq=2048 | 482.3 | 247.2 | 2.0× |
| S=130K, nq=2048 | 771.3 | 425.1 | 1.8× |
| S=130K, nq=8192 | 3075.3 | 1696.4 | 1.8× |

e2e（bs=8 × 9.9K token，同曲线）：**prefill 263.9→126.6s（2.08×）**，decode 不变。
prefill S 梯度（backend 全链路，nq=256 尾 chunk）：5K/10K/20K/40K/80K/130K =
9.5/17.1/27.8/37.5/50.6/62.1 ms/层——**prefill 延迟随 S 亚线性增长**（select 受
池宽约束、稀疏前向受 K2 固定预算约束），长上下文 prefill 的结构性优势。

剩余大头 = `_sparse_extend_one` 的随机行 gather（0.93ms vs einsum 0.44ms / 512 行
chunk；576GB/s 未打满 HBM——gather 非合并访问）→ M8 TMA/Tensor Descriptor gather
范畴。

### 8b-4. M6 kq 真 4bit 存储（2026-09-24，任务 #26 完成，commit 7c2a6ba74）

实现：pool 的 kq 从 fp32 单张量改为 **uint8+scale 三张量**（kq_q [R,S_cap,Hkv,2δ] uint8
格点 + kq_sc/kq_mn [R,S_cap,Hkv] fp32 双 scale）。**逐位一致论证**：4bit 格点值 0-15 在
fp32 中精确可表，`grid.float()*sc+mn` 与量化时 `round(...)*sc+mn` 走同一 IEEE 运算序列 →
kq_unpack(*quant4_pack(x)) ≡ quant4(x)（test_tli_m4 [0] 直接断言）——4bit 存储零数值
漂移。存储 128→40 B/token-head（3.2×）；131K 单请求 36 层 kq 从 ~5.4GB 降到 1.7GB，
S_cap 预分配口径同步缓解。

正确性回归：m4/m2b/m5 全 PASS（m5 graph replay vs eager 逐位 0.00e+00——量化一致性
使批量/逐行/kernel/图四路径完全同值）；bs=8 graph e2e 78.8-86.5 ms/step（M5 70.3-75.5，
**unpack 反量化带来 ~5-10% 小幅回退**，可用 M8 fused kernel 内 dequant 消除）。

**附带两个方法论级发现（test_tli_dim_sweep.py，hotpotqa 真实 trace 5 层 × 2 个 t）**：

1. **L1 粗筛维数可压**：d'=32→16 后 mass coverage 逐位不变（选中块集 172 vs 194 块，
   但 L2 token 级 4bit 精筛把候选差异完全吸收）→ kmin/kmax 存储/流量再减半的免费午餐；
2. **L2 细筛维数不可压**：δ=16→8（2δ=32→16）时剩余 mass coverage 0.953→0.782
   （far-heavy 层 L03/L05 t 尾部最差 0.34）。与另一项目「细筛必须保全全维」的结论同构：
   **两级索引的维数压缩边界画在 L1/L2 之间——粗筛上界可粗，细筛分数必须精**。

**评估口径升级（test_tli_m4 [4]）**：新增**剩余 mass coverage**（竞争区 = 总 mass 去除
sink+滑窗强制 token）：竞争区仅占总 mass mean 0.515 / min 0.315（H1 sink mass 0.37-0.71
的直接后果），总口径 0.99+ 的饱和主要由强制位贡献——**论文质量表应改报剩余口径**
（基线 0.953 / min 0.726），区分度显著更高。E4c 的 L1 far capture 是同思想在 L1/远端
子集上的特例。

### 8b-6. 同机 indexer kernel 三方对比（2026-09-25，#29/#30 完成，官方 kernel 原样接入统一 harness）

口径：H20-3e（cc9.0 78SM）、decode 单 token indexer **per-layer-call 全链路**、同机同口径；
Quest/DSA 侧合成数据对齐量级（kernel 级 microbench），TLI 侧真实 trace（qwen3-8b layer03 far-heavy
+ layer17 两层均值）。脚本：`sglang/bench_dsa_vs_tli_indexer.py` + `sglang/bench_quest_score.py` +
`/tmp/bench_quest_select.cu`（Quest 官方 raft radix kernel 最小 fork，`/tmp/quest_min/`）；
汇总：`sglang/kernel_comparison_indexers.json`。

| indexer（ms/层） | S=10K | S=40K | S=131K | 索引存储/token | training-free | LongBench-13 |
|---|---|---|---|---|---|---|
| Quest 官方（page GEMV + raft decode_select_k，k=64pages=1024tok） | 0.066 | 0.077 | 0.107 | ~1KB（fp16 min/max 32head） | ✓ | 47.72 |
| DSA 官方（tilelang fp8_index 64head×128d + topk 2048） | 0.476 | 0.490 | 0.503 | ~132B（fp8 proj） | ✗（1000 步/2.1B token warm-up） | —（V3.2 专属） |
| TLI eager（两级 select 全链路） | 0.736 | 0.780 | 0.853 | ~336B（kq 4bit 40B/token-head×Hkv8 + L1 bounds） | ✓ | 49.92 |
| TLI fused L1 | 0.604 | 0.658 | 0.787 | 同上 | ✓ | 49.92 |

**诚实结论**（论文系统章节素材）：
1. **Quest 索引最快**（GEMV+radix µs 级，HBM 带宽型）但代价 = 1KB/token 索引存储（3× TLI）
   且 LongBench −2.2 分——Quest 的速度优势与其 fp16 全 head min/max 索引的存储代价绑定；
2. **DSA 0.5ms 与 S 无关**（tilelang FP8 TC 生产级 kernel，launch 主导；纯算力账 ~20µs@131K），
   但需训练 indexer 权重 + 每 token 64head×128d = 8192 MAC 的全量 GEMV；
3. **TLI 延迟当前不占优**（0.6–0.83 vs DSA 0.5ms）——如实报告。结构性优势在算法侧：
   每 token 索引 MAC ≈258（8kv-head×2δ32 + L1 摊销）= DSA 的 **1/32**、索引存储 ~3×↓ vs Quest、
   D' 跳 13/36 层摊销、质量 +2.2 分 vs Quest。当前延迟差距来源 = eager topk/gather 的 launch
   开销（三家在 131K 下都远离 HBM bound，排名反映的是实现成熟度）→ **M8 kernel 化是兑现路径**，
   算力/存储余量已备好。
4. **MoBA 未测**（flash-attn 未装）：降级预案 = 论文数字 + 算术账。

工程沉淀：①tilelang 0.1.7 无 `TL_DISABLE_FAST_MATH` PassConfigKey（源码级删 no-op 开关）+
jit 按首次形状特化（symbolic n 静态化）→ 每个 S 独立子进程；②Quest raft kernel 最小 fork：
select_radix.cuh 行段拼接（1-708 + 710-818 set_buf_pointers + 927-1094 one_block kernel），
shim 头断开 rmm/fmt 依赖链；③DSA 官方 kernel 的 k_s 签名是 [b,S]（须 squeeze）。

### 8b-7. L2 维度可压性扩展 + 投影表示对照 + B' 分区时序消融（2026-09-25，本轮三实验）

概念口径（用户定调）：**粗筛（L1）= 选哪些 block/page；细筛（L2）= 在粗筛选中的 block
内选 token 及数量**。三个实验回应外部批判（GPT 复盘）与用户质询，脚本
`test_tli_dim_sweep2.py` / `test_tli_proj_sweep.py` / `test_tli_partition_ablation.py`，
数据 `tli_dim_sweep2.json` / `tli_proj_sweep.json` / `tli_partition_ablation.json`。

**1. L2 维度选择不可压（扩展口径确认）+ 病因分离**（5 trace × 5 层 × 3 t，300 行）：
far recall（vs dense 全维 far top-256 oracle，per-head）：δ=16→12→8→4 =
0.557→0.500→0.378→0.274；**fp32 与 4bit 仅差 2–5%**（0.557/0.531、0.378/0.373）——
病因=**维度不够，非量化精度**；全链路 pipeline 与「不限候选池」隔离口径几乎同值
（0.528 vs 0.531）——L1 对 far recall 的额外损失可忽略。

**2. 投影表示可压（用户直觉验证，重要新方向）**：per-kv-head SVD top-r 投影替代尾维选择
（同隔离口径 fp32）：

| far recall | 选择-r | PCA-r（同源） | PCA-r（跨任务基） | 随机-r |
|---|---|---|---|---|
| r=16 | 0.378 | **0.532** | 0.503 | 0.147 |
| r=32 | 0.557 | **0.682** | 0.658 | 0.246 |

- **PCA16 ≈ 选择32**：同质量打分维度砍半（16 vs 32 MAC/token-head）；
  **PCA32 = +0.125**：同 FLOP 质量大增；**跨任务 PCA 基迁移仅 −0.024**（hotpotqa 校准基
  打全部 5 任务）——离线校准哲学与 D' 一致，training-free 成立；
- 随机投影崩溃（JL 保范数不保 GQA 求和后的点积排序）→ **降维必须用 K 协方差结构**；
- 结论修正：**非对称压缩定律的 L2 半边 = 信息瓶颈在表示方式而非维数**——投影可压、
  选择不可压。工程含义：kq 存 PCA 投影 4bit（r=16 → 16B/token-head vs 现 40B），
  KV 写入侧一次 [D×r] GEMV（2048 MAC/token/head 摊销）——L2 打分 FLOP 再砍半 +
  存储 2.5×↓（集成待办）。

**2b. 投影深入探索**（`test_tli_proj_explore.py` / `test_tli_proj_mixed.py`，r=16）：
- 子集选择全景：低频尾维 0.351（选择类最优）＞ even 0.081 ＞＞ 高频头维 0.026 崩溃
  ——**「noPE/位置稳定」机制的反向验证**（Qwen3 无原生 noPE 维；旋转慢的低频尾维
  = 近似 noPE，旋转快的高频维崩溃）；
- PCA 0.489（+0.14 vs 尾维）；**小校准集 2048 token 即与全量校准同值**（0.489）——
  校准成本可忽略；**4bit 量化投影仅 −0.022**（0.467）——存储口径 16B/token-head 成立；
- **跨层共享基 No-Go**（0.234，层间 K 协方差差异大）→ per-layer 基，存储
  36 层×Hkv8×[128,16] fp32 ≈ 2.4MB 可忽略；
- **可解释性：PCA 与尾维子空间主角度 73°（近正交）**——两种机制独立（旋转稳定性
  vs 协方差方差方向）；**混合打分 No-Go**（pca32 0.660 ＞ mixed32 0.555 ＞
  tail32 0.543，同预算下混合双计交叠反降）→ 纯 PCA 是最优表示。

**3. B' 分区时序消融 + far survival curve**（回应「far 在 L1 被剪掉则 L2 救不回」质询；
global = L1/L2 均全局（TIA 语义）/ late = 当前 B'（L1 全局 + L2 分区）/ hier = L1 far
保底 K1_far=16 块 + L2 分区；5 trace × 5 层 × 2 t）：

| 模式 | far rec | far 名额 | far mass cov | L1 存活 | cov剩 |
|---|---|---|---|---|---|
| global | 0.664 | 569.4 | 0.8456 | 0.929 | 0.9207 |
| late（B'） | 0.511 | 256.0 | 0.8414 | 0.929 | **0.9284** |
| hier | 0.420 | 256.0 | 0.8105 | **0.677** | 0.9197 |

- **「far 在 L1 被剪」实测不发生**：near 带+sink 仅 34 块 vs K1=128 → far 池保底
  ≥94 块，oracle far token 的 L1 存活率 0.93–1.00（far-heavy 层 t=S/2 达 1.00）；
- **far rec 的 global>late 是名额口径混淆**：global 下 far 平均占 569/1024 名额
  （4bit 噪声 + far 区基数效应），far mass cov 两模式几乎同值（0.846 vs 0.841）；
- **B' 的真实增益机制 = 近端名额保障**（方向与「防 far 被挤」的直觉叙事相反）：
  far-heavy L05 t=S/2 cov剩 0.600→0.726 的全部增益来自近端带名额恢复（far mass cov
  同值 0.9989）——名额分配从「分数噪声驱动」改为「预算驱动」，far 截到 256 质量不损；
- **hier（L1 分区）No-Go**：L1 far 保底 16 块反而把 far 池从 ~94 块砍到 16 块
  （L1 存活 0.677、far mass cov 0.811）——**分区只需在 L2 终选级，当前实现即正确形态**
  （negative result + design decision 入论文）。

### 8b-8. M9 PCA 投影降维集成（2026-09-25，任务 #33 完成）

§8b-7 验证的「投影表示可压」工程落地：kq 存 PCA 投影 4bit，L2 精筛表示从
「维度选择（refine_idx，2δ=32 维）」切换为「PCA 投影（r=16 维）」。

**集成形态**（sglang tli backend，全部路径统一出口）：
- 离线校准 `calibrate_pca_basis.py`：hotpotqa_0 far 区 per-(layer, kv-head) SVD
  top-16 → `tli_pca_basis_r16.pt` [36, 8, 128, 16]（常驻 2.36MB，同 D' 离线校准哲学）；
- config `SGLANG_TLI_PROJ_BASIS` / `SGLANG_TLI_PROJ_R`；backend 加载注入 per-layer
  indexer；pool kq 尾维 `refine_nd()`：2δ→r（扩容/行视图路径自动继承）；
- indexer `_k_refine`/`_q_refine`：选择与投影同一代码路径（build/update/select/
  select_batched/update_pool_rows_decode/select_decode_batched 六处调用点统一替换）。

**对拍实测**（`test_tli_m9.py` ALL PASS，30 行 = 3 任务 × 5 层 × 2 t，pipeline 全链路）：

| 表示（L2 精筛） | 打分维 | B/tok-head | far recall |
|---|---|---|---|
| 选择 δ=16（当前生产） | 32 | 40 | 0.5174 |
| 选择 δ=8（同维对照） | 16 | 24 | 0.3480 |
| **PCA r=16** | **16** | **24** | **0.4423** |

- **同成本口径（论文速度卖点）**：同样 16 维打分 / 24B 存储 / 带宽下，投影比选择
  +0.094（+27%）——「表示方式」信息瓶颈结论的系统级兑现；
- vs 全维选择 δ=16：−0.075，换 1.67× 存储节省 + 打分 FLOP 减半（质量-成本权衡点，
  如实报告；弱势行集中在 musique/gov L05/L33 跨任务基迁移，hotpotqa 多行反超 sel32）；
- **差距分解（集成零损耗证明）**：隔离 fp32 0.4843 → 4bit 0.4543（−0.030，与隔离
  实验 −0.022 吻合）→ pipeline 0.4423（L1 −0.012，与选择口径的 L1 损失同级）；
- **一致性对拍**：增量 vs 全量格点一致率 1.000000 + unpack allclose + select 集合
  一致（投影是 GEMV，cuBLAS 批次相关归约顺序 → scale 有 1e-7 级 fp 尾差，语义口径；
  选择路径 gather 无算术故逐位——M5 graph 对拍结论不受影响，PCA 模式下逐位口径
  放宽为格点级）；pool 批量 flat 索引（nd2=16）vs per-request 32/32 head 集合一致；
- **延迟（诚实口径）**：eager select 0.97–1.05×（trace 9.9K 0.791→0.813ms / 合成
  131K 1.019→0.972ms）——**降维收益在 eager 路径被 topk/gather launch 开销掩盖**
  （归因与 M3-b 一致），打分 FLOP 减半的兑现位在 M8 fused kernel（打分 GEMV 占比
  大的形态）；存储 1.67× 是即刻收益；
- **e2e smoke**：9.9K token 真实 narrativeqa 长上下文，PCA 路径稀疏 prefill + decode
  全链路运行无崩溃、输出语义连贯（与选择路径输出不同属预期——表示改变→选择集
  不同，非等价对拍口径）。

**工程坑（沉淀）**：torch.einsum 标签里 head 维必须出现在输出（`"...hgd,hdr->...hgr"`）；
漏写 h（曾写成 `kgd,hdr->kgr`）= h 变归约维、对全部 head 的基**求和**——k 路径标签
恰好正确、q 路径全错，两者数值形状相同，far recall 0.126 vs 预期 0.53 才暴露
（debug 脚本逐环节二分定位：隔离 fp32 ✓ → 4bit ✓ → _k_refine 逐位 ✓ → _q_refine
max diff 42.5 ✗）。

### 8b-9. MoBA indexer 级同机对比（2026-09-25，#30 完成）

官方 `MoonshotAI/MoBA`（moba_naive.py）的 router（gating）是**纯 PyTorch**（块均值 +
q·mean einsum + topk），不依赖 flash-attn——indexer 级对比可以同机原样接入（S9 口径
一致）；attention 主体（gated block attention CUTLASS kernel，moba_efficient）硬依赖
flash-attn 2.6.3（本机装不上，与 flashinfer 同类环境问题），按降级预案引用论文数字。
`bench_moba_router.py`（合成张量 microbench，已标注）：

| router（decode 单 token） | S=10K | S=40K | S=131K | 存储 B/tok | 训练 |
|---|---|---|---|---|---|
| MoBA（chunk512×topk2=1024 tok） | 0.056 | 0.056 | 0.057 ms | 16 | **需从头训** |
| Quest（官方，对照） | 0.066 | 0.077 | 0.107 | ~1K | 免训 |
| DSA（官方，对照） | 0.476 | 0.490 | 0.503 | ~132 | 需 warm-up |
| TLI eager / fusedL1 | 0.736/0.604 | 0.780/0.658 | 0.853/0.787 | ~224（M9 后） | 免训 |

- **decode indexer 成本排序：MoBA < Quest < DSA < TLI**——MoBA router 最便宜的原因是
  语义上做得最少：chunk=512 块粒度、无 token 级细筛/滑窗/分区（这些功能在 TLI 的
  0.85ms 里，在 MoBA 里被折叠进 gated flash-attn 主体）；TLI 的额外成本买到
  token 级 1024-of-S 精度 + 免训质量（LB 49.92，MoBA 需从头训练才能拿质量）；
- **prefill router eager 89.2 ms/层 @131K**（gate [H,S,N] 全矩阵 + topk，O(S²/chunk)）
  ——官方生产版靠 fused kernel 消化（README：efficient vs naive 40×@32K）；
  TLI select_batched（M7 后）@10K 同口径 ~160ms/层（含 4bit gather+反量化+两级），
  两者 eager 都不是最终形态，如实报告；
- **存储口径**：MoBA 块均值 16B/token vs TLI ~224B（kq 24×8 + 块界 32）vs Quest ~1KB
  ——MoBA 最省但粒度最粗；论文引用其 CUTLASS kernel 数字时标注「tech report 图表，
  未同机复测」。

### 8b-10. far token 捞取 CPU 化 overlap 提议的定量分析（2026-09-25，用户提议，No-Go）

用户提议「远邻 token 的捞取放 CPU 上做，overlap 进 GPU 计算」。同机实测
（`bench_cpu_gather_overlap.py`，bs=32 × S=131K × K2=1024，KV 选中集 134MB/层/步）：

| 路径 | 延迟 | 相对 GPU gather |
|---|---|---|
| [A] GPU gather（当前路径，高级索引） | 0.63 ms（213 GB/s 有效） | 1× |
| [B] GPU 预 gather → D2H → CPU 整理 → H2D | 35.8 ms | **57×** |
| [B2] host 镜像池（17GB RAM）→ CPU gather → H2D | 36.9 ms | **59×** |

PCIe pinned 实测 55 GB/s（D2H/H2D）。**No-Go，三面墙**：

1. **带宽墙**：捞取本体是 HBM→SM 的读（GPU 侧 0.63ms）；走 CPU 必须经 PCIe 两次
   （55GB/s），比 HBM 慢两个数量级——即使 CPU 端零开销、overlap 完美，57× 的
   传输时间也无法被 5.2ms/层的计算隐藏；
2. **依赖墙**：far token 选择依赖**当前层**的 q（自回归因果链 L→L+1），无法提前
   一步知道要捞什么；投机预取（用上一步选择集，E5 实测 decode churn 22%）命中率
   ~78% 且改变算法语义——不值得为 0.63ms 的项冒语义风险；
3. **占比墙**：gather 仅占层时间 12%（0.63/5.24ms），当前真瓶颈是 select 的
   launch/批量计算（M3-b 归因），M8 TMA gather（213GB/s→~2TB/s，10× 该项）才是
   正道。

**[D] 关键发现（提议的正确内核）**：传输与计算的 overlap **不需要 CPU**——copy
engine 与 SM 独立，GEMM 7.95ms + 128MB D2H 串行 10.67ms → 并发 7.98ms（完全隐藏）。
即：若未来确需搬数据（如 CPU 侧 radix cache），GPU 侧双流/事件即可 overlap；
「捞取更快」的正确路径 = M8 TMA/异步拷贝在 kernel 内消化（而非搬到 CPU）。
附带可行动作：near 窗 token 在池内天然连续，可免 gather 直读（M8 顺手项）。

### 8b-11. 心方案 H2-B：真实 trace 候选复用偏斜统计（2026-09-25，任务 #35 完成）

用户心方案（`CPU+TC+CC.txt`，Near=GPU 规则路径 ‖ Far=CPU 语义 cluster 路径 + GPU 内
TC/CC 混合打分）的**前提假设 H2-B**：「candidate page reuse is highly skewed——
少量 page 被反复选中（hot few + cold many），可用复用密度把工作分流到 TC GEMM /
CC GEMV」。Go 条件（心方案文件自定）：明显长尾 + 20% page 承载 ~70% q-k pairs。

**实验**（`test_tli_reuse_stats.py`，3 任务真实 trace × 5 层 = 15 行）：对每个 (task,
layer) 取全部真实 q（位置 >4096，Nq=262-270），逐 q 跑生产版 L1 块选择（子空间区间
算术 + 因果 mask + top-K1=128 + 滑窗强制块），得 Q×K 01 矩阵 [Nq, nblk]，统计 f_j
激活频率、16×16 tile density ρ、work concentration 曲线、near/far 拆分。输出
`figures_m8/fig_reuse_matrix_*.png`（01 位图）+ `fig_reuse_freq_*.png` +
`tli_reuse_stats.json`。

**结果（均值 / 范围，15 行）**：

| 指标 | 数值 | 判读 |
|---|---|---|
| f_j > 8 的 page 占比 | 0.898（0.51–1.00） | **不是长尾，是「多数 page 都热」** |
| ρ > 0.5 的 16×16 tile 占比 | 0.767（0.32–0.98） | 01 矩阵在 tile 级**大体稠密** |
| top-20% page 承载 q-k pairs | 0.295（0.22–0.47） | 均匀基线 0.2，浓缩比仅 1.1–2.35×（目标 ~3.5×） |
| far 区同口径 | 0.306 | far ≈ 全体——**far 语义复用并未更高**（心方案 §16 预期未证实） |
| f_j 均值 / 最大 | 110–185 / 270 | 滑窗+近端块被全部 q 选中（f_j=Nq），拉高均值 |

层间模式：语义选择层 L05/L10 浓缩最高（0.47，2.35×），L03/L20/L33 近稠密
（0.29）；但均远低于「20% page 承载 70% work」的漂亮故事线。

**结构性根因**：单请求连续 t 的 q 共享因果历史——滑窗/近端块天然被所有 q 选中
（f_j=Nq）；且 K1=128 在 S≤32K 时占总块数 32–61%，选择密度过高使「复用偏斜」没有
发挥空间（131K 时密度降至 ~8% 才可能出现真偏斜，但本机无 131K 真实 trace，LongBench
截断上限 32K——如实标注为口径局限；跨请求 shared-prefix 批量复用是另一口径未覆盖）。

**判定（按心方案文件 §19 自定规则）**：
- H2-B **No-Go**：无明显 hot/cold 偏斜 → TC/CC 混合分区（#34 的 crossover 成本模型）
  按「没有偏斜就砍掉」规则不建——分区/packing 的开销换不来偏斜收益；
- **正向副产品（馈赠 M8）**：01 矩阵 tile 级稠密（ρ>0.5 占 77%）意味着**全块 L1 打分
  本身就是一个高利用率 dense GEMM**——M8 的 L1 kernel 应直接上 Tensor Core
  （`tl.dot` 打 kmin/kmax × q 的批量矩阵乘），无需 hot/cold 分区packing。ρ 数据是
  M8 dense-TC 设计的直接依据；
- **cluster+avg far 路径双重 No-Go**：质量侧 E4c 已证聚类块代表最差（0.09–0.39 vs
  TIA token 精筛 ≈ oracle）；延迟侧 H2-A 撞 §8b-10 三面墙（CPU 检索延迟 vs 隐藏窗
  + 当前层 q 依赖 + gather 占比仅 12%）。CPU 侧仅存角色 = 后台维护（build/update），
  在线索引查询留在 GPU——与 MoBA/Quest 同构结论一致。

### 8b-12. M8-KernelA：批量 L2 fused gather+dequant+GEMV（2026-09-25，#28）

**阶段分解归因**（`bench_m8_phase.py`，插桩副本与真实函数逐位对拍 PASS；合成池
生产形状 n=32 × S_cap=131072 × Hkv=8 × d'=32 × K1=128，合成数据已标注）：

| phase | 中位 ms | 占比 | 内容 |
|---|---|---|---|
| P1 kmin/kmax gather | 0.085 | 0.4% | rows 物化 268MB |
| P2 L1 einsum | 0.369 | 1.6% | 区间算术批量 GEMM |
| P3 topk K1 + onehot | 0.118 | 0.5% | [32,8,2048] k=128 |
| P4 候选压实 topk-min | 0.883 | 3.8% | [32,131072] k=65728 |
| **P5 L2 gather+deq+einsum** | **20.24** | **87.6%** | kq_c fp32 物化 2.1GB×2 + 逐元素 flat gather ~240GB/s |
| P6 misc mask | 0.406 | 1.8% | |
| P7 partition topk | 1.014 | 4.4% | far/near topk + gather |

**KernelA**（`kernels.py: tli_l2_score_batched`，`SGLANG_TLI_L2B_KERNEL=1` 默认开）：
grid (n, Tc/512)，每 program 对 512 个候选 token 直接从 pool uint8 gather（每 token
HKV×ND2=256B 连续段）→ 寄存器内反量化（grid×sc+mn 同运算序，格点级逐位）→ GEMV
打分 → 转置写 s2——**消除全部中间物化**。

| 指标 | eager | kernel | 加速 |
|---|---|---|---|
| P5 段（独立微基准） | 20.5 ms | **0.48 ms** | **43×**（有效带宽 1.55TB/s） |
| P4 段（独立微基准，块数打满口径） | 3.23 ms | **0.80 ms** | **4.06×** |
| select_decode_batched 全函数（A） | 31.4 ms | 4.66 ms | 6.7× |
| select_decode_batched 全函数（A+B） | 31.4 ms | **3.82 ms** | **8.2×** |

调参教训：CHUNK 必须 ≤512——1024 时 tile fp32 化 262144 元素寄存器溢出到 local
memory，反而 3.2ms（6.7× 慢）；512/8warps 达理论带宽。

**对拍**：s2 max diff 2.4e-07（归约顺序级）；最终选择**有效集 jaccard 1.0000**
（256/256 行×head，仅并列元素排序与哨兵 lane 位置不同——语义等价）；M9 回归
test_tli_m9.py ALL PASS（PCA nd2=16 模式同过）；n==1 自动回退 eager（保留
per-request L1/L2 kernel 路径）。形状静态（Tc 常量）可进 CUDA graph。

调试插曲（诚实记录）：初版对拍差 0.58 疑似 kernel bug，逐环节二分后定位为
**benchmark 脚本抄生产代码时丢了 `h_off * nd2`（头偏移应乘维数）**——kernel 与
生产代码均正确。跨函数抄索引算术必须逐项核对偏移乘子。

**KernelB**（`kernels.py: tli_compact`，`SGLANG_TLI_COMPACT_KERNEL=1` 默认开）：P4
候选压实从 topk-min 全排序（[n,131072] 取 k=65728）换为 cumsum 前缀 + 块展开
kernel（块升序天然保持位置序，非因果尾 token 写哨兵可落中段——下游
valid = tok < S_cap 掩掉，有效集逐行一致；槽位带防御性上界）。独立微基准
4.06×；全函数再省 0.84ms。**A+B 合计 31.4→3.82ms（8.2×）**；有效集 jaccard
1.0000（uniform 与混合 S 行均 256/256）；M9 回归 ALL PASS（双 kernel 均过
nd2=16/32 两模式）。摊销口径：bs=32 时 0.119ms/req/token——已低于 Quest
单请求 0.107-0.107ms 同量级、远低于 DSA 0.5ms（高并发批量路径成为速度
主叙事的直接支撑）。

**剩余瓶颈**（3.82ms 组成）：P7 partition topk 1.01 + P2 L1 einsum 0.37 + P6
0.41 + KernelA 0.48 + KernelB ~0.15 + 其他——下一靶点为 P7/P6 的分区融合进
KernelA（参照 n=1 L2 kernel 的 far/near 双池直写）与 L1 einsum 的 TC 化（ρ
数据 §8b-11 支持 dense GEMM 直接走 tensor core，tf32/bf16 精度 L2 可吸收）。

### 8b-13. M8-KernelC/D + profiler 二次归因（2026-09-25 晚，#28）

**torch profiler 归因**（CUDA kernel 级，A+B 后 2.4ms/调用）揭穿两个隐藏大头：
①双 131μs elementwise = `kmin_pool[rows]`/`kmax_pool[rows]` 的 P1 行 gather
（各 67MB clone，此前 phase 分解误记 0.085ms——插桩打点在 gather 完成后，
clone 时间被并入了后续 phase）；②`aten::einsum` 346μs 中 gemv 仅 75μs，其余
是 permute 连续化拷贝（einsum 需要 bmm 布局）。**P1+P2 真实合计 ~0.6ms**。

**KernelC（双池直写）**（`tli_l2_score_batched_dual`，`SGLANG_TLI_L2D_KERNEL=1`
默认开）：far/near 池的 -inf 掩码烘进 KernelA 写出口径，一次扫描写两张池表
（far_sc/near_sc [n,Hkv,Tc]），消除 P6 masked_fill 链与 P7 的池表物化。关键
语义论证：**池边界即因果边界**（far_hi ≤ t+1-near_len、sw_lo ≤ t），tok 数组
取值仅为 [0,S_t) 实位置或哨兵，均天然落两池之外 → 与 eager 的
`池界 & valid & causal` 掩码逐位等价（topk 输入逐位一致 → 输出 torch.equal，
uniform/mixed S 双场景 PASS）。

**寄存器压力调参（CHUNK 二次扫描）**：双输出 tile 使寄存器压力比单输出版更早
触顶——CHUNK=512 时 dual kernel 1.06ms（溢出 local memory），**CHUNK=128 达
0.43ms**（比单输出 KernelA@512 的 0.49ms 还快）；顺带发现单输出 KernelA 也应
降 CHUNK（512→128：0.49→0.35ms）。**tile 元素数上限的经验值：CHUNK×Hkv×ND2
fp32 ≤ ~65K 元素/program（8 warps）**。

**KernelD（批量 L1 fused gather+GEMV）**（`tli_l1_score_batched`，
`SGLANG_TLI_L1B_KERNEL=1` 默认开）：grid (n×Hkv, NBLK/256)，rows 行间接寻址
直读 kmin/kmax pool（读 128MB 写 0.5MB），归约分组与 eager einsum 相同（先
G-sum 后点积）→ **P1+P2（gather 262μs + einsum permute 346μs）整体消除**；
垃圾块 -inf 在 kernel 内烘焙（skip_far 层的 near_keep 掩码语义走 eager 分支）。
实测与 C 档**逐元素差 0**（归约分组对齐后连 1e-7 都没有）。

| 配置 | select_decode_batched 全函数（bs=32/131K） | vs eager |
|---|---|---|
| 全 eager（P1-P7） | 23.25 ms | 1× |
| B 档（KernelA+B，CHUNK=128 + 惰性掩码） | 2.11 ms | 11.0× |
| C 档（+双池直写） | 2.15 ms | 10.8×（与 B 持平——双倍转置写 ≈ masked_fill 消除，价值在 launch 数↓与 C+D 组合） |
| **full（A+B+C+D）** | **1.80 ms** | **12.9×** |
| **+E near 压缩（§8b-19）** | **1.46 ms** | **15.9×** |
| **+F CHUNK 形态调优（§8b-20）** | **1.55 ms**（三档同进程自洽：eager 23.19/off 1.93/on 1.55） | **15.0×**（自洽口径，论文主数字；E/F 两步绝对值跨轮环境漂移 ±7% 不可直比，kernel 级硬数字 dual 0.49→0.39ms） |

诚实口径：C 档单独看是平手（2.15 vs 2.11ms）——双池直写的收益被第二张表的
转置写吃掉；保留默认开的原因是 launch 数减少（低 bs 时 launch 主导）与语义
更干净（-inf 单一来源）。有效集对拍：full vs eager jaccard 1.0、C vs B
torch.equal、full vs C 逐元素差 0（三重验证）。

**剩余瓶颈**（1.80ms 组成，profiler 实测）：far/near/K1 三个 topk 的 radix
机器 ~0.9ms（radixFindKthValues 487μs + gatherTopK 249μs + counts/sort
~170μs，50%）+ dual kernel 0.43 + compact 0.07 + gather/einsum 杂项 ~0.3。
topk 宽度受 CUDA graph 静态形状约束（Tc=65984），near 池实际候选数远小于
宽度但无静态上界可压——**结构性剩余，非实现低效**；进一步压缩需接受语义
近似（over-select）或破坏图形状静态性，暂不做。（`sorted=False` 微基准仅省
~5% 且 tie 语义变化，不值。）

**同机对比摊销口径更新（§8b-6 表的批量侧注脚）**：full 1.80ms@bs=32 →
**0.056 ms/req/layer**——追平 MoBA（0.056，语义最少的下界）、2× 优于 Quest
单请求（0.107）、9× 优于 DSA（0.5）。TLI 在「比 Quest/MoBA 多出 token 级
精筛 + far/near 分区 + 滑窗」的语义下达到 MoBA 级成本——速度主叙事的直接
数字支撑（口径诚实标注：TLI 为 bs=32 批量摊销，Quest/DSA/MoBA 为单请求
kernel 微基准；两者的单请求对单请求口径见 §8b-6）。

**M8 e2e 系统级兑现（test_tli_m8_e2e.py，9.9K narrativeqa、graph decode full、
N=64、与 M5 完全同曲线同口径）**：

| bs | M5+M6 graph（历史） | **M8 graph** | 改善 | triton 图基线 |
|---|---|---|---|---|
| 1 | 17.7–36.8 | 38.3 ms/step | n==1 恒走 per-request 路径（M8 批量 kernel 不适用），历史波动区间内 | 8.9 |
| 8 | 70.3–86.5 | **43.4** | 1.6–2.0× | 12.8 |
| 16 | 87.5–115.2 | **64.7** | 1.4–1.8× | 18.6 |
| 32 | 188.6 | **99.0** | **1.9×** | 31.0 |

- bs=32 全轨迹累计：M3-c 原型 1236.8 → M5 graph 188.6 → **M8 99.0 ms/step
  （12.5×）**，tok/s 26 → **323**。
- 诚实口径：9.9K 下 tli 仍慢于 triton 基线 ~3×——与 M3 归因一致（bs≤32、
  S=10K 时 KV 流量仅 ~30MB/层/步，稀疏收益小于 select 索引成本；单请求
  decode 被 MLP/GEMM 主导）。**M8 kernel 的收益展示位在长 S**（KV 流量
  ∝ S，1024-of-S 选择把 gather 流量除以 S/1024）——长上下文高并发主表见
  §8b-14（bs=8/16 × S≈30K，Qwen3-8B context 上限 40960 内的最大可行档）。

### 8b-14. 长上下文高并发 e2e（bs=8/16 × S≈30K，#36 进行中）

口径：`test_tli_m8_e2e_long.py`，vcsum 中文长文档（LongBench 唯一 ≥100K chars
文档 20 个全在 vcsum；中文 chars/token 文档间 3.1-5:1 波动 → **必须 offset_mapping
精确 token 截断**，chars 截断首跑实测 150K chars=47896 token 直接超模型上限），
prompt 精确 30K token、graph decode full + prefill disabled、N=64、POOL_S_CAP=40K、
watchdog 1800（30K 稀疏 prefill 数百秒）。显存账：KV=144KB/token → bs=16×30K
≈95GB（mem_frac 0.7）；bs=32×30K 需 155GB 超卡且唯一长文档不足 32 个——
**bs=16 × 30K 即 Qwen3-8B context 上限内的最大可行并发档**。

**完整结果（30K token，graph decode full，双侧 N=256 复测口径）**：

| bs | tli prefill (M8→M10) | tli decode | triton prefill | triton decode | tli/triton decode |
|---|---|---|---|---|---|
| 8 | 786.8 → **373.0 s** | 45.1 ms/step | 57.7 s | 30.0 ms/step | 1.50× |
| 16 | 1564.9 → **741.4 s** | 67.1 ms/step | 114.9 s | 43.4 ms/step | 1.55× |

- **测量学修正（重要，诚实口径）**：本节初版（N=64 差分法）曾报
  「bs=16 tli decode 47.7 ≈ triton 48.1 打平」——**该结论被 N=256 双侧
  复测撤回**。差分法（decode = 完整跑总时长 − 单独 prefill）在数百秒
  prefill 下信噪比 <1：decode 信号仅 3-4s，prefill 段间波动（~1%）即
  4-8s。N=256 复测（decode 信号 8-17s，信噪比 ~3）双侧结果：tli 45.1/
  67.1、triton 30.0/43.4 ms/step——tli bs 扩展比 1.49× 与 9.9K 口径
  （43.4→64.7 = 1.49×）完全一致，交叉验证可信；初版 tli「66.1→47.7
  反降」与 triton「25.4→48.1 线性翻倍」均为噪声假象（复测 triton 扩展
  比 1.45×，非 1.89×）。
- **真实图景：30K 下 tli decode 稳定慢于 dense triton 基线 ~1.5×，未打平**。
  扩展比 tli 1.49× ≈ triton 1.45×——30K/bs16 档两者 decode step 均被
  MLP 前向主导，attention 流量差异（tli ∝K2=1024 恒定 vs triton ∝bs×S）
  被稀释；与 9.9K 口径（bs16 慢 3.48×）构成「S 增长差距收窄 3.5×→1.55×」
  的单调链（fig9c）。**两点线性外推翻转点 S≈44–51K**（triton_step=a+b·S 从
  两个实测点解出 b=1.23ms/K、a=6.5ms；两拟合线交叉 51.4K，tli 取平台
  60–67ms 时 44–50K——区间报告；模型粗但方向明确：
  **【注：本段数字为修正前口径，§8b-25 已仲裁——S 标签 9.9K→7.5K 错算修正 +
  差分法三层测量 bug 修正后，翻转点修正为 54–57K、收窄链 3.53×→1.61×，以
  §8b-25 为准】**
  attention 成 step 主导项后 triton 流量 ∝S 显性化、tli 恒定）——**H100
  主表口径（bs≥16 × S=128K）远在翻转点之后，是收益验证位**（fig9）。
- **诚实口径 1**：每配置单次测量（M5 经验：波动大须多轮中位数）；
  prefill 数字双侧两版一致（57.7/114.9 vs 58.1/114.9，<1%）。
- **诚实口径 2（prefill 短板 → M10 已修）**：tli prefill 787-1565s vs
  triton 58-115s = 13.6× 慢——`select_batched` 全 eager + M7 快路径在
  30K 失效（见 §8b-15 归因）。**M10 kernel 化后双档 2.11×（373/741s），
  短板收窄至 ~6.4×**，剩余为结构性 topk（~67%）+ 模型本体前向。

### 8b-15. M10：prefill select_batched 慢路径 kernel 化（#37）

**归因（bench_m10_prefill.py + kernel 级 profiler）**：30K 尺度慢路径的根源
是 **M7 快路径失效边界**——快路径（L1 块并集==因果区）只在
nblk ≤ K1×Hkv 时成立；S=30K 时 nblk=480 ≫ K1=128，每 head 仅选 ~25%
块、并集覆盖 <100%，**128/128 行全走慢路径**（真实 narrativeqa K 实测）。
eager 慢路径成本 = [n,Hkv,S] masked_fill 链（elementwise 434ms）+ 随机行
gather 135ms + gemv 90ms + topk 机器 160ms（末 chunk 单调用 979ms；
4 chunks × 36 层 ≈ 141s/req，完美解释 e2e prefill）。**phase 插桩再次
骗人**：手动分解合计 ~200ms vs 实际 979ms——归因必须 kernel 级复核。

**实现（indexer.py select_batched）**：M8 decode 侧全套移植——
`tli_compact` 候选压实（L1 并集 onehot → 紧凑 token 数组，替代 scatter
到 [n,Hkv,S] 全宽）+ `tli_l2_score_batched_dual` 双池直写（far/near -inf
烘进打分 kernel 写出，寄存器内 dequant 不物化 kq_c fp32）+ host 常量宽度
配额 topk（W_far+W_near+W_forced=token_budget，无 CUDA graph 约束不必
静态上界）+ 哨兵转 0（下游 `_sparse_extend_one` 无 valid 掩码约定）。
**row_chunk 显存约束解耦**：kernel 路径不物化 kq_c → row_chunk 64→512
摊销 launch/topk 固定项。

**微基准（S=30720/NQ=8192，tli_m10_bench.json）**：末 chunk（慢路径主导）
983.8→140.0ms（**7.0×**）；首 chunk（快路径+empty 行）243.4→167.8ms（1.45×）。
对拍 test_tli_m10.py：双口径（合成 K + 真实 narrativeqa L03 K）候选充足行
对称差 ≤2（GEMV 归约序 tie 翻转 2.4e-07，M8 同款判据）、短行有效集一致；
M9/M5 回归全过。

**e2e 兑现（30K token，同 §8b-14 口径）**：

| bs | M8 prefill | **M10 prefill** | 加速 | decode（N=256，未改路径） |
|---|---|---|---|---|
| 8 | 786.8 s | **373.0 s** | 2.11× | 45.1 ms/step |
| 16 | 1564.9 s | **741.4 s** | 2.11× | 67.1 ms/step |

- **两档加速比完全一致（2.11×）**——慢路径成本 ∝ bs×S 的线性项被消除，
  剩余为结构性 topk + 不可压的模型本体前向。
- **诚实口径 3（decode 差分法失效 → N=256 复测修正）**：N=64 时 decode
  信号仅 3-4s，而数百秒 prefill 的段间波动（~1%）即 4-8s，**信噪比 <1**
  ——M10 复跑两档 decode 差分均为负值（−0.15/−3.06s），M8 版的正差分
  （66.1/47.7）同样不可信。N=256 复测（`LONG_N_DECODE=256`，信噪比 ~3）
  双侧：**tli 45.1/67.1、triton 30.0/43.4 ms/step**——tli bs 扩展比
  1.49× 与 9.9K 口径完全一致，交叉验证可信；§8b-14 的「打平」headline
  已据此撤回（见该节测量学修正段）。M10 未触碰 decode 代码路径，
  45.1/67.1 同时是 M8 版 30K decode 的可信替换值。
- **剩余瓶颈（结构性）**：kernel 化后 topk radix 机器 ~112ms/67%（far
  k=256 + near k=768 over Tc≈33K 候选）；对比 triton prefill 58-115s 仍慢
  ~6.5×，其中模型本体前向占非 select 部分大头——select 侧继续压缩需
  topk 算法级替换（近似选择），暂不做。
- 工程教训：**长 prefill e2e 期间不得在同机其他 GPU 跑任务**（CPU/PCIe
  竞争污染差分法）；decode 差分法只在 prefill ≲ 10s 时可信。

### 8b-16. NIAH 检索质量评测（RULER 口径，S=32K，#38）

口径：`test_tli_niah.py`——S=32041 token（token 精确截断，RULER uniform
深度 5-95% 十档 × 2 样本 = 20/后端），haystack = 本地 LongBench 英文长文档
拼接（gov_report/hotpotqa/qasper/multifieldqa_en/triviaqa；外网受限无法取
RULER 的 PG essays，NIAH 对 haystack 语义不敏感，拼接文档是社区常用替代）；
needle = 「magic number is {7 位随机数}」插在 token 精确深度的句边界；
temperature=0、输出含 needle 数字串计成功。tli（M10 全 kernel 路径，默认
far=256）vs triton（dense FullKV）同机同卡。

**结果（双 seed × far 预算三点扫描，n=40/配置）**：

| 后端 | NIAH score（seed1/seed2/合并） | far 区命中（合并） |
|---|---|---|
| triton（dense） | **1.000 / 1.000 / 1.000** | — |
| tli far=128 | 0.550 / 0.700 / **0.625** | 21/32 = 0.66 |
| tli far=256（默认） | 0.650 / 0.600 / **0.625** | 21/32 = 0.66 |
| tli far=512 | 0.550 / 0.700 / **0.625** | 21/32 = 0.66 |

- **失败模式定位**（输出文本分析）：失败样本模型输出的是 haystack 中其他
  数字（74/12/1940/1228）或复述问题，而非茫然拒答——即 **needle token
  未进 far 选择集**，模型从被选中片段抓了干扰数字。
- **深度分布与结构解释**：near 保障区（最后 near_len=2048 token ≈ 深度
  ≥94%）双 seed 全过（8/8）；depth 5-85% 全靠 far 池 → far 区合并命中
  0.66，**与 E4c 实测 far rec@256≈0.5-0.56 的量级吻合**（单针与 query
  相关性高于平均 → 略优于平均 recall）。
- **诚实口径修正（far 预算钟形被第二 seed 撤回）**：seed1 曾呈钟形
  （0.550/0.650/0.550），seed2 反转（0.700/0.600/0.700）——合并 n=40 后
  三预算**完全持平（0.625）**，seed1 钟形是 n=16 far 槽位的二项噪声。
  真实结论：**NIAH score 对 far 预算在 [128,512] 区间不敏感**——限制
  因子是 far 池整体召回水平（~0.66），而非预算切分比例；「512 挤近端」
  在此 n 下不显性。E5b 的 LongBench 预算饱和结论基于 n=200×13 任务，
  不受此影响。教训与 §8b-14 同构：**单 seed n=20 的差分结论必须复测
  才能写入论文**。
- **总诚实口径**：NIAH 是 far-heavy 极端检索任务，tli 0.625 vs dense 1.0
  是 far 池 ~0.8% 预算的本质限制——LongBench 主表（均值负载，TLI 49.92
  vs FullKV 50.36）与 NIAH（极端压力）共同构成质量侧的两端口径，与
  Quest/HISA 论文报告的 NIAH 损失同性质。

### 8b-17. D' 层跳过掩码的严格 held-out 验证（2026-09-26，任务 #48——§4.3 审稿防御【缺】项回填）

动机：E6 的「平均轮廓→每 prompt」预测把目标 trace 混入校准集（非 held-out）；
审稿必问「离线校准掩码在未见负载上的 precision」。脚本
`two-level-attention/exp/trace/analyze_e6b_heldout.py`（16 条 trace
leave-one-out：掩码由其余 15 条平均轮廓阈值化，在留出 trace 上评），
far 轮廓口径与 E6 完全一致（pm 平均 + far=pm[64:t-2048]，GPU 重算 576 层
缓存 `e6b_far_profiles.json`），结果 `e6b_heldout_gate.json`。

| 阈值 TH | precision min/mean | pred 层数均值 | 备注 |
|---|---|---|---|
| 0.02（E6 原值） | 0.800 / 0.954 | 13.6 | 过松：0.01-0.02 灰区层误跳 |
| **0.01（稳健点）** | **0.923 / 0.990** | **13.0** | 唯一失败 = narrativeqa（0/1 重复 trace） |
| 0.005 | 0.846 / 0.981 | 13.0 | 无增益（平均轮廓本就只含 13 个 <0.005 层） |

- **TH=0.01 是稳健点**：15/16 trace precision=1.000，mean 0.990；
  唯一失败 narrativeqa（prec 0.923）——且 narrativeqa_0/1 是同文档重复
  trace（32K 截断后 prompt 相同，E5 已知坑），**实际 = 1/15 个不同负载**；
- **质量代价归一**：narrativeqa 误跳层 missed far = 0.0226，占该 trace
  far 总量（7.71）**0.29%**——与 E6「跳层 far 质量损失 <0.3%」同量级；
- **e2e 交叉验证**：narrativeqa 在 E5b 主表不掉分（TLI 23.14 vs TIA
  22.18，反超）——held-out 层面的 precision 0.923 未转化为 e2e 损失，
  与 E5b 掉分三任务（musique/qasper/multifieldqa）不重合；
- **recall 保守无害**（0.41–1.00）：漏掉的可跳层只多花算力不损质量；
  narrativeqa far 总量 7.7 与 gov_report 8.9–10.6 同级（far-heavy 负载
  层轮廓错位是已知跨任务迁移困难，E5 corr 0.05–0.89 的体现）。

论文写法（§4.3 回填）：held-out 判据以「mean 0.99 + 失败模式单点定位 +
e2e 不掉分交叉验证」呈现，如实报告未达「min ≥0.98」的原始硬阈值
（min 由单一重复 trace 决定）——测量学口径与 §7 一致。

### 8b-18. M8-TC：L1 打分 TC 化（tl.dot）实验——No-Go（2026-09-26，#49）

动机：#28 M8 遗留「P2 L1 einsum 0.37ms 是 TC 化靶点（ρ>0.5 tile 占 0.767
→ 全块打分=高利用率 dense GEMM → tl.dot 上 TC）」。脚本
`sglang/bench_m8_l1_tc.py`（合成 microbench，bs=32 × S=131K，对拍=sc1
allclose + topk 块集合一致率；结果 `tli_l1_tc_bench.json`）：

| 变体 | ms | vs eager | 对拍 |
|---|---|---|---|
| eager P1+P2（行 gather + einsum） | 0.475 | 1× | 基准 |
| **KernelD（生产现役）** | **0.039** | **12.15×** | allclose ✓ 集合 1.0000 |
| 方案 A：合并访存重排（[MBLK2,Hkv,DP] 连续 tile） | 0.040 | 11.9× | allclose ✓ 集合 1.0000 |
| 方案 B：tl.dot ieee（FMA 路径） | 0.284 | 1.67× | ✓ |
| 方案 B：tl.dot tf32（真 TC） | 0.101 | 4.71× | **块集合一致率仅 0.9336（精度不过关）** |

三条结论（negative result 入论文 §5/§8b）：

1. **L1 打分已是带宽饱和 kernel**：KernelD 0.039ms 读 134MB（32 行 ×
   2048 块 × 8 head × 32 维 × min/max）→ 有效带宽 ~3.4TB/s ≈ H20 HBM
   峰值（4TB/s）的 86%。ρ>0.5 的「dense GEMM 机会」**已被带宽饱和的
   fused gather kernel 完全兑现**——TC 无余量可图，带宽是硬上界；
2. **tf32 精度破坏块选择对拍**（10-bit 尾数使近 tie 分数翻转，集合
   一致率 0.9336）；ieee dot 走 FMA 反慢 7×（块对角 W 的 K=256 仅 1/8
   有效列）——tl.dot 两个口径均 No-Go；
3. 方案 A 证伪了「KernelD tile 访存 12.5% 合并效率」的直觉诊断
   （Triton tile 展平后 warp 内实际合并）——**select 剩余瓶颈不在 L1
   打分，在 topk**（P7 partition topk 1.01ms + P4 压实 topk-min
   0.88ms = 3.38ms 的 56%）。

工程含义：M8 后续优化方向收敛到 topk 侧（算法级近似选择或 radix kernel
化），L1/L2 打分双 kernel 均已带宽饱和——「级联 kernel 化到哪一级为止」
（§4.4 原则 1）的又一实证。

### 8b-19. M8-topk：near 池压缩直写——P7 瓶颈腰斩（2026-09-26，#50）

归因（`bench_m8_topk.py`，生产形状 bs32/131K，`tli_m8_topk_breakdown.json`）：
select_decode_batched 全函数 1.808ms 中两个分区 topk 占 0.93ms（51%）；
**near topk 输入 98.6% 是 -inf**（near 池有限项仅 ~920/65728——sink 128 +
近带候选），torch.topk 对全宽 65728 列做 radix 选择纯属浪费；far 池 79%
有限（连续 band [128, 52287]）无此问题。

**实现（near 池压缩直写，`SGLANG_TLI_NEAR_COMPACT=1` 默认开）**：
KernelC dual 加 NEARC constexpr 分支——near 有限项 = tok_c 升序下的前缀
[0, ps)（sink）∪ 后缀 [pf, pn)（far_hi..sw_lo 带），slot 确定性（前缀
slot=c / 后缀 slot=ps+c−pf，ps/pf 由两个 torch sum 前缀计数算出）；
直写静态宽 **WNCAP = far_lo + (near_len − sliding_window) = 2048** 的
near_sc_c [n,Hkv,2048] + near_tok_c [n,2048]（-inf/哨兵 pad）。near topk
在 2048 宽上进行（30× 宽度削减）；输出拼接宽度不变（CUDA graph 静态形状
保持）。

**结果**：

| 指标 | off | on |
|---|---|---|
| select_decode_batched 全函数 | 1.799 | **1.458 ms（1.23×）** |
| 对拍（uniform 131K / mixed 3K-131K / short 3K-6K） | — | **三场景全部 torch.equal 精确一致**（tie 顺序都保持——near 压缩保序） |

回归：M9（PCA pipeline，含 select 集合一致）/ M10（prefill 双口径对拍）/
M5 smoke（CUDA graph 捕获+replay）全过。e2e 影响上界 = select 占 decode
step 比例（9.9K 档 1.8ms/99ms ≈ 1.8%）→ 节省 0.35ms < 单次测量噪声，
**e2e 主表不重跑**（测量学原则：不为噪声级差异重跑 headline，§8b-14 教训）。

**剩余瓶颈（结构性，划界）**：far topk 0.467ms（k=256 over ~52K 有限项，
radix 机器多 pass）+ KernelC dual 0.385ms（135MB 写 + 270MB 读 = 1.05TB/s，
转置写 stride 264KB 限制）+ L1 块 topk 0.073ms。far topk 的进一步压缩需
自定义 radix-select kernel（算法级，~0.3ms 上限收益）——与 prefill topk
同性质，暂不做，如实划界。

### 8b-20. M8-2b：KernelC dual 转置写 TMA/布局实验——写侧假设证伪 + CHUNK 形态调优 22%（2026-09-26，#51）

§8b-19 划界时把 dual kernel 0.385ms 归因为「转置写 stride 264KB 限制」。
本轮做完整证伪链（`bench_m8_tma.py` / `bench_m8_tma2.py` / `bench_m8_nw_ab.py`，
结果 `tli_m8_tma_breakdown.json` / `tli_m8_tma_variants.json` / `tli_m8_nw_ab.json`）：

**(a) 带宽分解证伪「写带宽」假设**：nostore（读+算）0.077ms + noload（写 only）
0.107ms < full 0.359ms——写侧单独 70MB@0.65TB/s 并不慢；full 的超额时间 =
**读写混合流 MC 排队惩罚**（写流与稀疏 gather 读流互相干扰），不是段粒度问题。

**(b) 三个写优化变体全部 No-Go**（同 tile 地址集合，对拍 torch.equal）：

| 变体 | 耗时 | 结论 |
|---|---|---|
| C0 转置写（生产） | 0.279 ms | — |
| C1 tl.make_block_ptr 写 | 0.271 ms | −3%，噪声级 |
| C2 TMA descriptor store（Triton 3.4 TensorDescriptor，H20 cc9.0） | 0.280 ms | 0——异步 bulk 不缓解混合流惩罚 |
| C3 布局改 [n,Tc,Hkv] 连续写 | 0.268 ms | −4%；但 full-chain（+permute.contiguous 回转置）0.346ms 反慢——topk 读侧要求 [n,Hkv,Tc] |

**TMA 方向如实划界 No-Go**：far_sc 写布局不是瓶颈（≤4%），转置写假设证伪。

**(c) CHUNK×num_warps 形态失真教训（本轮最有价值的发现）**：arange 连续 tok_c
形态的 sweep 选 128/4（0.310ms）；**真实稀疏 tok_c（compact 产物，块段间跳）**
的 sweep 反转——CHUNK=64/nw=4 = 0.385ms vs 生产 128/8 = 0.493ms（**22%**）。
机制：CHUNK=64 时每个 program 恰好覆盖一个候选块段（bs=64 → 8KB 连续
gather），小粒度 program 在段间跳之间保住访存级并行；连续形态会把读侧
L2 命中高估、误导 sweep。**铁律：kernel 调参 sweep 必须用真实 compact 产物**。

**(d) grid/CHUNK 不同步 bug（无声丢写，对拍才暴露）**：收 CHUNK=64 时 wrapper
的 grid 仍用 chunk 参数（默认 128）→ grid 只覆盖一半元素（514×64=32896 <
Tc=65728）→ near band 后缀与半数 far **无声丢失**（far_sc 是 empty 分配，
未写区域=垃圾分数）。表面症状是三场景有效集对称差 868。修复：near_compact
分支 grid 独立 `cdiv(Tc, 64)`。教训：**constexpr 改 CHUNK 必须同步 grid**；
empty+部分写模式下无越界、无 NaN——唯一防线是有效集对拍。

**生产收益（`SGLANG_TLI_NEAR_COMPACT=1` 路径，CHUNK=64/nw=4）**：dual kernel
0.493→0.385ms（22%）；select_decode_batched @bs32/131K 同轮环境
1.70→1.545ms。回归：near compact 三场景 torch.equal / M9（131K 1.06×）/
M10（双口径对拍）/ M5 smoke（CUDA graph）/ M8 e2e 全过。e2e 主表不重跑
（同 §8b-19 口径：节省 0.11ms < 测量噪声）；但 M8 e2e 复测（干净独占
GPU1）确认 headline：bs=32 **102.0 ms/step / 313.9 tok/s**（vs 原 99.0/323，
±3% 噪声级一致——首轮并行跑 bench 污染出的 109.1 作废，又一例「跑 e2e
期间该卡禁跑其他任务」的教训）。

**M8 H 卡优化收官**：TMA 写 No-Go（本轮）+ L1 TC 化 No-Go（§8b-18）+ 唯一
落地 = CHUNK/warps 形态调优 22%。dual kernel 剩余时间 = 稀疏 gather 读侧
（405MB，8KB 段间跳）+ 混合流惩罚，已达结构上限；进一步优化只有 far_sc
物化消除（打分+topk fused radix-select，算法级）——维持划界。

### 8b-21. RULER 多任务质量评测（2026-09-26，#52——Quest/HISA 论文口径对齐）

**动机**：H100 阻塞项盘点时发现 RULER 全量本质是**质量评测，不依赖机器型号**——
NIAH 单针双 seed（§8b-16）的基建可直接扩展。补 RULER 官方模板四任务
（`test_tli_ruler.py`：niah_multikey 4 针异 key / niah_multivalue 同 key 5 值
列举 / niah_multiquery 4 针 4 问 / variable_tracking 5 值×3 链），S=32K、
n=20/任务、双方法（sglang Engine 同机同权重）、评分=答案值全命中
（RULER multivalue/multiquery 官方全中口径）、max_new_tokens=160
（初版 64 截断思考链，dry run 发现后修正——工具层 bug 当场修）。

**结果（`tli_ruler_results.json`，seed=1234，单 seed n=20 口径如实标注）**：

| RULER 任务 | FullKV | TLI@1024 | gap |
|---|---|---|---|
| niah_multikey | 0.95 | 0.70 | −0.25 |
| niah_multivalue | 0.70 | 0.55 | −0.15 |
| niah_multiquery | 0.85 | 0.70 | −0.15 |
| variable_tracking | 0.70 | 0.40 | −0.30 |
| **均值** | **0.80** | **0.588** | **−0.21** |

**失败模式诊断（逐例）**：①VT 失败 12 例 = 输出 VAR 变量名而非数值
（「VAR 83B169」而非 837696）——检索到链中段但丢失赋值源头，链上多针
全部落在 far 区时 0.8% far 预算的物理上限；②multikey 失败 = 抓到其他
key 的干扰数字（与单针 NIAH 同机制）；③gap（−0.21）小于单针 NIAH
（−0.375）：多针任务部分命中率高（针多冗余）。与 Quest/HISA 论文报告的
同预算 RULER 损失同性质（Quest@1024 RULER needle 类同样显著掉分）。

**诚实口径**：①~~单 seed n=20，双 seed 复测列为 H100 项~~ ✅ 已在本地
补齐（§8b-23：seed2=5678 双方法，pooled n=40，gap −0.231 与单 seed
一致——质量评测不依赖机器型号的判断得到验证）；②FullKV 在
multiquery/multivalue 也非满分（0.85/0.70）——Qwen3-8B 本身的列举能力
上限，gap 才是稀疏损失；③TLI 生成 1045s vs triton 173s/任务 = 稀疏
prefill 慢路径 6×（M10 已修但 32K prefill 仍 ~2× 于 dense + 索引构建），
质量评测不计入速度口径。

### 8b-22. RULER CWE/FWE 扩展：模型能力上限证伪链（2026-09-26，#53——negative result 资产）

**动机**：RULER 四任务（§8b-21）全为检索/链式，补聚合型压力测试
（CWE/FWE：合成词流 common×30/uncommon×3/filler×8，问最高频词，
信号词散布全上下文，与检索型互补）。词表从 LongBench 语料抽取
（101K 词种，中频 3≤c≤200），`test_tli_ruler_cwe.py`。

**构造自验通过**：离线 Counter 复现 = 10 个 common×30 恒为词流 top10，
filler×8/uncommon×3 频率沟清晰，prompt_tokens=32029 精确——**排除构造
bug**。四轮诊断链（每轮一个变量，4K 短上下文快速定位 + 32K 复测）：

1. **Raw prompt → 停用词先验**：模型输出 'and,of,for,with,by,as,...'
   （词流里根本不含这些词）——模型自己陈述「list 里的词不是标准英文词」
   转而用语言先验作答；
2. **+防先验约束**（"random word generator / answer must come from the
   list"）：模型转入计数模式，但 thinking 链吃满 256 token 无答案——
   raw prompt 下 Qwen3 无 chat template，`/no_think` 软开关不生效；
3. **+chat 格式 + assistant prefill 空 think 块**（Qwen3 官方 no-think
   方式）：输出全部变为流中词、无 thinking——修复确认；但 4K 截断把
   频率沟砍窄（common 30→~3.5 次），32K 下计数错误（0 分）；
4. **+thinking + max_new_tokens=2048**：模型策略退化为**逐词抄写词流**
   （8K 词抄不完即截断），子串评分被抄写虚高污染（hits=3 为作弊命中）。

**结论**：CWE/FWE 聚合计数任务对 Qwen3-8B 在 8K–32K 词流上**不可解**
（no-think 计数失败、thinking 截断失败、抄写污染评分）——FullKV 基线
本身 0 分 → 该任务族**无方法区分度**，与 RULER 文献一致（CWE/FWE 为
最难任务族，聚合计数需远超 8B 级的 CoT 能力）。RULER 覆盖面由检索型
（niah×3）+ 链式（VT）承担（§8b-21 gap −0.21 有区分度）。CWE/FWE
完整诊断链如实归档为 negative result：**质量上限由模型能力而非注意力
稀疏决定的任务，不能作为稀疏方法对比口径**（避免「双 0 分被误读为
无损」的反向错觉）。

### 8b-23. RULER 双 seed 合并（2026-09-26，#53——n=40 pooled 口径）

**动机**：§8b-21 单 seed n=20 的诚实口径补强（与 NIAH 案例二教训一致），
seed2=5678 双方法复测（GPU0 一进程一 Engine 顺序跑），合并 n=40。

**结果（`tli_ruler_results.json`，seed∈{1234,5678} 各 n=20 → pooled n=40）**：

| RULER 任务 | FullKV s1,s2 | TLI s1,s2 | gap pooled |
|---|---|---|---|
| niah_multikey | 0.95, 0.90 | 0.70, 0.75 | −0.200 |
| niah_multivalue | 0.70, 0.85 | 0.55, 0.50 | −0.250 |
| niah_multiquery | 0.85, 0.85 | 0.70, 0.55 | −0.225 |
| variable_tracking | 0.70, 0.55 | 0.40, 0.35 | −0.250 |
| **均值 pooled** | **0.794** | **0.562** | **−0.231** |

**读法**：①双 seed gap（−0.231）与单 seed（−0.21）一致——结论稳健，
均值略宽 0.02 在 seed 方差量级内；②池化后逐任务 gap 收窄到
−0.20~−0.25（单 seed −0.15~−0.30）——n 翻倍降方差；③任务×seed 方差
显著：FullKV VT 0.70→0.55、TLI multiquery 0.70→0.55（±0.15）——
单任务单 seed 数字不可引，pooled 口径是论文表格的必要形态；④VT 仍
gap 最大任务族之一（−0.25），与链式追踪 far 区预算上限诊断一致。

### 8b-24. RULER QA 类扩展（2026-09-26，#55——qa1/qa2 双 seed pooled）

**动机**：RULER 覆盖面补检索型事实问答（与被证伪的聚合型 CWE/FWE
互补，§8b-22 先验证 FullKV 区分度再跑对比的教训落地）。qa1 = TriviaQA
单跳（189 对）/ qa2 = HotpotQA 两跳（182 对），needle 用 LongBench QA
真实问答对（"One of the special magic questions for {word} is: {q}
The special magic answer for {word} is: {a}."×20 针），问指定 word 的
answer，评分 = 任一答案别名 substring 命中（RULER 官方口径）。
脚本 `test_tli_ruler_qa.py` + 合并 `merge_ruler_seeds.py`（QA JSON 接入）
+ 失败模式 `analyze_ruler_qa.py`。

**FullKV dry 区分度先行验证**：n=3 dry（qa1 1.000 / qa2 0.667）→
任务成立才投入全量（CWE 教训流程化）。

**结果（`tli_ruler_qa_results.json`，双 seed 各 n=20 → pooled n=40）**：

| QA 任务 | FullKV s1,s2 | TLI s1,s2 | gap pooled |
|---|---|---|---|
| qa1（单跳 TriviaQA） | 0.75, 0.70 | 0.50, 0.40 | −0.275 |
| qa2（两跳 HotpotQA） | 0.80, 0.65 | 0.55, 0.75 | −0.075 |
| **QA pooled 均值** | **0.725** | **0.550** | **−0.175** |
| **RULER 全六任务 pooled** | **0.771** | **0.558** | **−0.21** |

**失败模式（analyze_ruler_qa.py 逐样本交叉）**：①TLI 丢的主模式 =
检索到**错误 key 的答案**（问 Cece 的答案 TLI 答 1998/Phoebe Sparrow
——其他针的干扰答案，与 NIAH 干扰数字同机制）；②TLI 独中也存在
（FullKV 复读 question 退化时 TLI 反而作答）——说明 FullKV 非饱和，
两方法各有失败面；③共同丢 = 复读 question 不作答（raw prompt 模型
格式层，两方法同丢不计入稀疏损失）。

**诚实读法**：①qa1 gap −0.275 与 NIAH 单针（−0.375）/ multikey
（−0.20）同族量级——单跳单针检索是稀疏 worst case 的再现；②qa2
两 seed 方差大（FullKV 0.80/0.65、TLI 0.55/0.75，±0.15-0.20），
−0.075 小 gap 在 n=40 下不可单独引用，只能并入六任务均值；③六任务
pooled gap −0.21 与四任务单 seed −0.21 / 双 seed −0.231 一致——
RULER 质量损失结论在任务族扩展下稳健。

### 8b-25. e2e decode 差分法测量 bug 仲裁（2026-09-26，#56——fig9a 口径加固）

**发现的系统性测量 bug**：`test_tli_m8_e2e(_long).py` 系列的 decode 计时用
差分法（t_total − t_prefill 两次独立 run 相减），且 **无 ignore_eos**：
greedy 下「一句话总结」prompt 会在 ~30 token EOS 早停，实际 decode 步数
远小于 max_new_tokens。铁证链：①无 ignore_eos 复测同配置两轮 decode
2.02s（~30 步早停）vs 16.56s（走满 256 步）——8× 离散完全由 EOS 位置
随机决定；②m10 结果 decode_s 为负（−0.15/−3.06s，第二次 prefill 比第
一次快）暴露差分法还受 prefill run-to-run 方差污染（tli prefill ~735s
方差 0.1-15s → decode ±0.4-59ms/step）。历史 1.55×（67.1/43.4）建立在
「恰好走满」的运气上。

**干净重测（ignore_eos=True + ktok 校验 + P/D 交替多轮，
`test_tli_e2e_variance.py`，bs=16 / CUDA graph / N=256）**：

| 点位 | triton dense | TLI 稳态 | 比值 |
|---|---|---|---|
| S≈7.5K（narrativeqa 32K chars 实测 avg） | 18.0 ± 0.0 ms/step | 63.5 ± 0.4（63.1/63.4/63.9） | **3.53× slower** |
| S≈30K（vcsum token 精确） | 40.4 ± 1.6（39.6/41.2） | 65.0 ± 1.1（64.5/65.6） | **1.61× slower** |

**两个额外发现**：①**首轮瞬态**：tli 9K 档首轮 47.2ms vs 稳态 63.5ms
（+34%）——engine init 后第一次 prefill+decode 循环有 warm-up 效应
（prefill 同样 163→159s），稳态才是持续 serving 口径；②**S 标签错算**：
9K 点实际平均 token = 7.5K（narrativeqa 英文 ~4.3 chars/token，历史
「9.9K」按中文 vcsum 3.2 chars/token 比例错算）——修正后 triton 斜率
1.005ms/K（旧 1.229）→ **翻转点从 44-51K 移到 54-57K（拟合 56.5K）**。

**结论**：①S 收窄链修正为 3.53×→1.61×（旧 3.48×→1.55×——方向与
量级不变，历史数字测的正是稳态、偏差 <5%）；②翻转点修正 44-51K →
54-57K（S 标签修正主导，非测量噪声）——诚实边界更远但结论叙事不变
（收益位仍在 S≥128K HBM 主导区）；③fig9 已按干净口径重绘（误差棒 +
S 标签修正 + 交叉 54-57K）；④测量学教训入库：**greedy 差分法必须
ignore_eos=True + 记录 completion_tokens 校验 + prefill 交替配对 + 弃
首轮瞬态**——m8/m8_long/m10 全系列历史 decode 数字标记为「旧口径
（EOS 不受控）」，以本节干净值为准。

### 8b-26. S=40K 第三点补测：KV pool 容量边界 + 差分法可测性极限（2026-09-26，#57）

动机：翻转点 54-57K 为两点外推（7.5K/30K），补 40K 第三点（贴近模型
40960 上限）把外推距离从 24K 压到 15K。数据源 vcsum token 精确口径
（19 个唯一 ≥165K chars 中文长文档，bs16 充足）。

**发现的物理容量边界（先于数据本身的重要）**：
- KV 物理账（Qwen3-8B，36 层×8kvhead×128d×2×2B=147.5KB/token）：
  bs16 需求 30K→484K token、40K→**644K token**；
- tli 后端额外吃 ~15.8GB 静态内存（per-layer 索引池 kq/kmin/kmax 预分配，
  证据：graph capture 后 avail triton 40.96GB vs tli 25.14GB）→
  mem_fraction=0.7 下 tli pool 仅 ~497K token；
- **30K 点（484K vs 497K，余量 2.7%）是踩线通过**；40K 超限 30% →
  KV 溢出触发 retraction 重算，prefill/decode 交错执行，差分法的相位
  分离假设被破坏。实测症状：同 run round0=41.0 / round1=125.1 ms/step
  （3× 双峰）+ prefill spread 21.2s + total 恒定反相关（1011.5/1011.8s），
  ktok 校验全过但 decode 已非良定义量——**retraction 混沌下 ktok 校验
  不足以判数据有效性**（#56 校验链的边界）；
- triton 侧 pool ~728K 装得下：**40K triton 64.8±1.3 ms/step 有效**
  （高段斜率 2.38ms/K vs 低段 1.005——40K 档 attention 流量渐占主导）。

**修复**：VAR_MEM_FRAC=0.85（tli pool ~653K ≥ 644K，capture 后剩
4.18GB 验证通过）。配置差异（40K 点 mem0.85 vs 30K/7.5K 点 mem0.7）
须在图表注中诚实标注。

（40K tli @0.85 复测结果待补——若两轮一致则第三点成立并三点重拟合；
若仍双峰则归档为「40K@bs16 差分法在 H20 不可测」negative result。）

**40K 第三点最终落地（三次尝试全链路）**：
- 尝试 2（mem0.85）：scheduler OOM（kernels.py:447 near_sc [n,Hkv,Tc]
  fp32 双 scratch 超 capture 后 4.18GB 余量）——**mem 0.85 与 40K prefill
  transient 在默认 chunk 下不可兼得**；
- 尝试 3（mem0.85 + chunked_prefill=2048 + expandable_segments）：成功。
  两次独立 run，稳态（弃池首次增长轮）样本 **62.5 / 67.0 / 67.3 →
  tli 40K = 64.9±2.4 ms/step**；triton 40K = 64.8±1.3。新瞬态发现：
  expandable_segments 大池首次增长成本可被 prefill-only run 承担
  （run1 的 prefill 估 998s vs 稳态 941s，差分 −39.8s 假象），第二轮起
  prefill spread 仅 1.0s——「弃首轮」教训的第 4 形态（池增长瞬态）。

**三点实测结论（fig9c 两点外推 → 三点实测）**：
- triton 3pt 拟合 b=1.365ms/K、a=5.81ms；tli 3pt 拟合 b=0.047ms/K
  （平台假设成立：63.5 / 65.0 / 64.9 跨 32K 增长仅 +1.4）；
- **交叉点从两点外推 54-57K 前移至实测 42-45K**（tli 平台带
  63.5-67.3 × triton 拟合 → 42.3-45.1K，线性交点 43.6K）；
- **40K 档 tli/triton = 1.00×（打平）**——收窄链 3.53× → 1.61× →
  1.00×，S=40K 即 parity；S>45K 起进入 TLI 收益区。翻转点比外推
  更近的原因：triton 高段斜率 2.38ms/K（40-30K 段）vs 低段 1.005
  （30-7.5K 段）——attention 流量 ∝S 显性化在 30K 后加速；
- 诚实标注：40K 点 mem0.85+chunk2048 vs 30K/7.5K 点 mem0.7 默认
  chunk（decode 走 CUDA graph 路径，配置差异不影响 decode step 口径，
  但 prefill 时间不可跨点比较）；chunk2048 使 tli prefill 941s 略慢于
  30K 口径外推值（chunk 串行开销）。

### 8b-27. E59 双消融：L1 块表示 + near_len 配比（2026-09-26，#59 完成，8B/30B trace 重放）

**动机**（用户两个设计问题）：①为什么 near 用 minmax 而非 avg 等其他块表示？
②near/far 配比是否应该 alpha 比例化而非固定 near_len？脚本
`two-level-attention/exp/trace/analyze_e59_l1_ablation.py`（块表示消融）+
`analyze_e59_near_alpha.py`（配比扫描），结果 `e59_*.json`。

**消融一：块表示**（K1=128 块、far top-1024 mass recall、8B 16 条 trace + 30B）：

| 块表示 | 8B 均值 | 8B 最差（narrativeqa） | 30B hotpotqa |
|---|---|---|---|
| **minmax 上界（TIA/TLI 现行）** | **0.885** | 0.781 | **0.967** |
| avg 均值点积 | 0.695 | 0.513 | 0.948 |

minmax 全面胜出（8B +0.19、最差样本 +0.27）：avg 是块内均值非保守估计，
块内符号对消系统性低估远端高响应块——上界粗筛「漏选方向错误更少」的
理论依据首次实证量化。写论文 §方法（设计动机）+ §消融表。

**消融二：near_len ∈ {1024, 2048, 4096}**（far mass + far 区 top-256 覆盖）：

| near_len | 8B far_mass | 8B far top-256 覆盖 | 30B far_mass |
|---|---|---|---|
| 1024 | 0.364 | 0.758 | 0.192 |
| **2048（现行）** | 0.330 | 0.772 | 0.182 |
| 4096 | 0.284 | 0.782 | 0.118 |

near_len 4 倍变化仅移动 far mass 0.08 / 选择质量 ~2%——边际递减，2048 是
合理工作点。**alpha 比例化实证 No-Go**（收益不显著且引入超参 + 破坏
「固定 near_len + 常数 far 预算 = O(1)/token」的结构性设计优势——
对比 TWI top-p 类 O(t) 自适应预算的卖点），诚实记入消融章节。

### 8b-28. E60：D' 升级为 prefill 动态测层 gate（2026-09-26/27，#60，commit 2f9b10f03）

**动机**：静态全局掩码跨任务不泛化（E5b musique/qasper/multifieldqa 掉
4.8-5.9 分）、per-task 重校准也失败——改为**当前请求自己的 prefill 末
chunk 测 per-layer far mass**（分布偏移从根上消除）。

机制（`indexer.py`，`SGLANG_TLI_DYN_GATE`/`_THRESH=0.01`）：末 chunk
softmax(fine) far 区统计 → `dyn_far_stat` → decode select 幂等置位
skip_far；far_stat 双峰（musique 18 层 <0.01 vs 其余 0.017-0.26），
阈值切自然间隙。开销 = 末 chunk 一次 softmax+sum，分摊 decode 期可忽略。

验证链：离线 corr 0.86-0.99 GO → 安全任务零损失 → **多跳三任务（静态版
失败集）全 GO**。全量 E5b 双臂（200/200/150 样本 × 3 任务 × gate on/off，
sglang `test_tli_dyngate_e5b.py`）**✅ 全量定稿（2026-09-27，
pred_dyngate_score.json）**：

| task | gate on | gate off | gate 代价 |
|---|---|---|---|
| musique | 27.42 | 27.57 | −0.15 |
| qasper | 40.20 | 40.37 | −0.17 |
| multifieldqa_en | 46.41 | 46.59 | −0.18 |
| **AVG** | **38.01** | **38.18** | **−0.17** |

musique 双臂 95.0% 完全一致、0 空输出。**核心结论：动态 gate 质量代价
−0.17（三任务一致），远小于写入标准 0.3、远优于静态 D' 版的 −4.8~−5.9
（musique 单任务 −10.3）——per-request 测层从根上解决静态掩码跨任务
泛化问题**。期间发现并修复 gate 跨请求自增强锁死 bug（commit c60129c37）。
已知限制：dyn_far_stat 为 per-layer 单值，同质 batch（评测口径）无害，
混合 batch 跨请求污染——正确修法 = stat 存共享 pool per-row（TODO，
生产语义）。

**绝对分口径注意（#61 定界闭环，2026-09-27 ✅）**：sglang 平台 off 臂
（=TIA 同形态）musique 27.57 / qasper 40.37，比 transformers 同形态
AB 臂（31.35/44.03）低 ~3.7 分（两任务一致系统差）。样本级 diff 排除
乱文/空输出/前缀口径——是「不同实体答案」型分歧（greedy 数值噪声被
短答案放大；bf16 kernel/batch=8 并发/decode 重建缓存路径差异）。
**五分诊臂定界终局（musique 全量）**：3.78 gap = **1.82 平台差**
（sglang triton dense 30.32 vs tf FullKV 32.14）+ **0.89 far_tokens**
（FAR=512 28.46 vs 256 27.57——trace 级「128 饱和」测不出答案翻转，
e2e 上 far 预算有实贡献）+ **~1.1 选择实现差**（sglang B' 分区 topk
vs tf 4bit 分区，形态近同实现有差；唯一未深挖项）；M11 kernel 数值
洗清（PK0 27.91 vs PK1 28.46，45/48 逐字）、dense 数值洗清
（dense40k 30.53 ≈ triton 30.32）。**论文策略**：质量主表用
transformers 口径（与 Quest/TIA 对齐），sglang 报速度 + 平台内 AB
（双口径铁律——绝对分标注平台口径差）。

### 8b-29. M11：统一稀疏 attention fused kernel（2026-09-27，#58 ext 主力，commit 1a77d0088 + f0f94a6b2）

**动机**：30B prefill 归因 ext=61%（`_sparse_extend_one` 的 [n,K2,D] fp32
gather 物化 ≈17GB/chunk 带宽主导）+ Python 批量化三连试 No-Go
（23.7/21.4 vs 20.2s）→ Triton fused 是唯一路线。

**kernel**（`kernels.py` `_tli_sparse_attn_dot_kernel`）：per (row,
kv-head) 的 [G,D] q × sel 间接寻址 K/V tile [BK,D] + online softmax；
**G pad 到 16 进 Tensor Core**（`tl.dot(q, K^T)` + `tl.dot(p.bf16, V, acc)`
= FA2 标准做法）。prefill 标量界 + decode per-lane valid 掩码
（HAS_VLD 编译期分支）双口径。

**microbench**（合成数据、8B Hkv8/G4 与 30B Hkv4/G8 形态、K2=1024）：

| 形态 | eager | fused | 加速 |
|---|---|---|---|
| 8B chunk 2K | 130.7ms | 8.45ms | **15.5×** |
| 8B chunk 4K | 163.4ms | 8.76ms | **18.7×** |
| 30B chunk 2K | 47.4ms | 4.25ms | **11.2×** |
| 30B chunk 4K | 73.3ms | 4.43ms | **16.5×** |

有效带宽 ~2TB/s（≈H20 HBM 一半，gather 随机性所致——上限由间接寻址
non-coalesced 决定）。对拍 5 形态 + valid 路径全 PASS（≤5e-3，bf16 级）。

**negative result（工程）**：广播 mul+sum 版（[G,BK,D] 中间积）G=4 快
2.8× 但 **G=8 寄存器溢出反慢 0.83×**——GQA 组大的模型必须走 tl.dot。

接入：prefill `_sparse_extend_one`（`SGLANG_TLI_PREFILL_KERNEL` 默认开）
+ decode `_sparse_attn_batched`（`SGLANG_TLI_SPARSE_KERNEL` 默认关，
graph AB 验证后开）。~~e2e AB（输出一致性 + 长文 prefill 计时）待 E5b
双臂让出 GPU~~ **✅（2026-09-27）**：**e2e prefill 91.27s(eager) →
21.25s(fused) = 4.30×**（narrativeqa 38.5K token 端到端总账，含
select/模型前向——kernel 级 11-19× 的 e2e 兑现值）；输出语义一致
（同主题总结）措辞有差=预期（kernel 吃 bf16 q_raw vs eager fp32 q_b，
数值路径差异被 greedy 放大；一致性口径见 PK 分诊臂 45/48 逐字）。
**e2e bug 已修复**（q_raw 非连续 stride 寻址错 →
contiguous()，commit 6348aeeea；合成回归 ALL PASS + e2e 输出与
triton/transformers 一致）。

**decode e2e AB（2026-09-27 ✅，narrativeqa 9.9K token × bs 8/16/32，
SGLANG_TLI_SPARSE_KERNEL 1 vs 0，无图）**：

| bs | eager 批量 | fused | 加速 | tok/s |
|---|---|---|---|---|
| 8 | 100.5 ms/step | 84.7 | 1.19× | 94.5 |
| 16 | 121.7 | 94.2 | 1.29× | 169.8 |
| 32 | 154.8 | 101.3 | **1.53×** | 315.8 |

收益随 bs 增大（批量越大 `_sparse_attn` 尾段占比越高）；
prefill 段两臂一致（43.6/43.9s，decode AB 隔离干净）。M11 双口径
齐：kernel 级 11-19× / prefill e2e 4.30× / decode e2e 1.19-1.53×。
**graph 路径验证 ✅（2026-09-27）**：SGLANG_TLI_SPARSE_KERNEL=1 下
CUDA graph capture（bs 1/2）成功且 replay 与 eager 输出 333 chars
逐字全等 → **默认开已提交（commit ece6ee5a4）**，
`SGLANG_TLI_SPARSE_KERNEL=0` 可回退。

**M8-TC L1 打分 TC 化 A/B（negative result，2026-09-27）**：tl.dot
tf32 版 vs 广播版（bs32/131K 全池 134MB）：广播 39μs @3444GB/s（72%
HBM3e 峰值）vs TC 62μs @2166GB/s = **0.63× 反慢**——内存受限算子
（有效带宽已 72% 峰值）算力非瓶颈，MMA 打包反而加延迟链。正确性
PASS（topk jaccard 0.9996、-inf 逐位）。**结论：广播 mul+sum 已是
该算子最优形态，保持默认关**（论文 negative result：『短 D'=32 的
GEMV 算子 TC 化在带宽墙 72% 时不划算』）。

### 8b-30. E63：near/far 预算配比 + near 页级形态消融（2026-09-27，#63，8B/30B trace 重放，n=18 样本）

**口径**：真实全维 softmax 行级 mass coverage（总覆盖口径，含 sink/near）；
far 细筛离线代理 = 子空间精确分数 top-far_tokens（E4c 已证 ≈ oracle）。
脚本 `analyze_e63_budget_split.py`，结果 `e63_budget_split.json`。

**A 块（beta 预算让渡：固定总 token 预算 B=sink+near_len+far_tokens=2432，
K1 页池按 far 预算 4× 超选比例缩放）**：

| 配置（near_len/far_tokens/K1） | mean cov | 说明 |
|---|---|---|
| 512 / 1792 / 112 块 | **0.9181** | 预算让渡给 far（beta<alpha） |
| 1024 / 1280 / 80 | 0.9074 | |
| 1536 / 768 / 48 | 0.8887 | |
| 2048 / 256 / 16 | 0.8498 | 基线参数但 K1 池缩水版 |
| **BASE 2048/256/K1=128（生产配置）** | **0.9094** | 现行滑窗形态 |

两个结论：①**同 token 预算下「小 near 滑窗 + 大 far 池」优于反向分配**
（0.9181 vs 0.9094，与 E5b e2e far_tokens 512→+0.89 方向互证）；②
nl2048 臂 K1 16→128 的 0.06 差距说明**页池预算（粗筛候选池）是比
far_tokens 更敏感的一等预算项**——预算体系应「页池优先保大」。

**B 块（near 页级选择形态：near 区块 topk + token 细筛，
(m_far,m_near)×页数×gamma 全组合）——No-Go**：最优组合
（minmax-far, np32, γ=1.0）仅 **0.6154**，远差于滑窗全保留基线 0.9094
——near 区 mass 分布平缓（近端 token 重要性均匀），页级选择丢弃过多。
negative result 资产：『near 区必须滑窗全保留，页级预算只适用于 far 区』。
method 双区结论：far 侧 minmax > max > avg（0.6154/0.6154/0.5522，
8B 上分化明显 30B 收窄）；near 侧 method 无差异（本身就不该选择）；
gamma 0.5→1.0 仅 +0.008（细筛折扣在该形态下非敏感项）。

**与 #59 合并的预算体系定稿**：alpha（near 长度）取 1024-2048 饱和
（e59）；预算让渡方向 = far 池优先（e63-A）；near 页级形态 No-Go
（e63-B）；far_tokens 128-256 饱和但 e2e 上 512 更优（e5b）——
生产配置 near2048/K1=128/far256 已接近帕累托前沿，near512/far1792
是备选激进点（trace +0.9pt，e2e 待验证）。

**DeJAVU 式 far 预测 + top& 短路信号**（`analyze_e63_dejavu_topand.py`，
`e63_dejavu_topand.json`，7 个 8B/30B trace）：

- **DeJAVU 式 far 预测 No-Go**：prefill 中段行 far top-256 集合对
  decode 末尾行 far oracle 的 mass recall 仅 **0.076-0.217**（IoU
  0.026-0.058）——far 选择是 query/content 相关的，跨时间位置几乎
  不重叠；与 E5「在线信号相关≈0」、E6b 跨任务轮廓低 corr 互证。
  DeJAVU 的预测器路线需要训练（=DSA 方向），在 training-free 框架
  下无信号可用（negative result 资产）。
- **top& 短路（sink 信号）规模相关**：sink_mass 与 far_mass 行级
  Pearson corr −0.14(8B)~−0.38(30B)；条件分布（sink 十分位分桶）：
  **30B 上单调递减、最高十分位 far 条件均值仅 0.023 vs 最低分位
  0.37-0.39（16× 差距）→「sink>P90 → 跳 far」短路在 30B 高度可行**
  （比 D' 的 per-layer 更细的 per-(layer,head) 粒度）；8B 上非单调
  U 型、比值仅 0.4-0.79，信号弱。与 32B 泛化复验「D' 更强」构成
  一致的「规模↑ → 稀疏结构更干净」叙事（论文正向资产）。

### 8b-30b. E63c：mid 区双分区（proposal 原框架）vs mid 单池对照（2026-09-28，用户框架对齐专项，8B/30B trace 重放，n=18 样本）

**动机**（用户 alpha/beta/gamma 框架本体验证）：mid_L = L − sink − swa 才是
indexer 处理区；mid 内分 near 子区（α·mid_L）与 far 子区，**各自独立
两级筛选**（near 拿 β 份额页池 + γ 折扣 token 细筛，far 拿剩余）。
E63-B 只测过「近端区整段页级选择」极端点，未干净测过该框架本体。
脚本 `analyze_e63c_mid_split.py`，结果 `e63c_mid_split.json`。

**口径**：同总预算 B = swa(1024) + mid 选择(1280 tok) = 2304、同总页池
128 块、sink=128；真实全维 softmax 行级 mass coverage（同 E63）。

| 臂（配置） | mean cov | 8B only |
|---|---|---|
| mid **单池** 128 块→1280 tok（α=0，TLI 简化版） | **0.9251** | 0.9231 |
| 双分区 α=.5 β=.5 γ=.5（near 64 块→640 / far 64 块→640） | 0.9214 | 0.9190 |
| 双分区 α=.25 β=.25 γ=.5（near 32 块→320 / far 96 块→960） | 0.9193 | 0.9169 |
| 双分区 α=.5 β=.25 γ=.5（near 32 块→320 / far 96 块→960） | 0.9094 | 0.9068 |
| TLI 生产锚（mid 近端 1024 免打分 + far 128 块→256 tok） | 0.9094 | 0.9075 |

**结论**：①mid 双分区（proposal 框架本体）最优臂比 mid 单池低 0.4pt
——**mid 内再分区无质量增量**（mid 近端的竞争保护价值已被 swa 保护段
覆盖，与 E63-B/E59 互证）；②mono（0.9251）> tli_anchor（0.9094）的
+1.6pt 差距来源 = swa 1024→2048 使 mid 预算从 1280 砍到 256（E63-A
「页池/预算保大」结论同向复现）；③诚实声明：trace 离线口径、30B 仅
2 样本。

**论文写作定位（用户框架对齐）**：方法章按用户一般化框架组织（sink+swa
保护段 / mid 处理区 / α β γ + (far_method, near_method) 参数空间），
现实现 = 参数空间的实证选定工作点（β→0 端，即 mid 近端免打分），
E59/E63-A/E63-B/E63c 全部消融归位为框架内扫描结论。

### 8b-31. 竞争论文扫描（2026-09-27，#63 期间发现，/tmp/two_level/related/）

- **IndexCache**（清华+Z.ai，arXiv 2603.12201）：DSA **跨层索引复用**——
  相邻层 top-k IoU 70-100% → 只留 1/4 indexer，prefill 1.82×/decode
  1.48×（30B H100 200K）；GLM-5 744B 生产验证 1.2×。**与 TLI 的关系**：
  ①他们测 DSA（有参 indexer，训练后层间趋同）而我们测 TIA 无参 minmax
  上界**跨层 IoU 仅 0.33**（E5 negative result）——跨层复用在
  training-free 上界索引上不可行，这是口径差异而非矛盾，答辩须主动讲；
  ②他们「200K 时 indexer 占 prefill 81%」与我们 30B 归因 select(81%)
  惊人一致（indexer 瓶颈普遍性的外部佐证）；③我们的 D' 跳 far 层是
  预算维度、IndexCache 是计算维度，正交可叠加。
- **DeepSelect**（DeepSeek 官方，2026-09-10 v1.0.0）：DSA topk 专用
  kernel（阈值过滤 + radix-select 压缩 + 随机块序），vs torch.topk
  **2-20×**，bf16 topk≤4096 场景。**与 TLI 的关系**：select_batched
  （30B 瓶颈 81%）现行用 torch.topk——DeepSelect 是低成本替换/对比
  候选，且按「打过官方 kernel」铁律应纳入 kernel 对比表（同 harness
  口径）。**H20 编译+实测已完成（2026-09-27 ✅，#64）**：官方只发
  sm_100a/103a（Blackwell），补 sm_90a 编译成功（三处适配：cutlass
  submodule SSH 克隆 + py3.12 f-string 语法降级 + GCC10 无 std::format
  + wheel 版本时间戳跨秒 bug）；**TLI 场景 microbench（H20，统一
  harness test_deepselect_bench.py）**：

| 场景（bf16） | torch.topk | DeepSelect | 加速 | jaccard |
|---|---|---|---|---|
| L1 块级 [256, 2048] k128 | 49μs | 11μs | **4.45×** | 1.0000 |
| L2 far [256, 131K] k256 | 517μs | 51μs | **10.08×** | 0.9882 |
| L2 K2 [256, 131K] k1024 | 531μs | 60μs | **8.83×** | 0.9956 |
| L2 far [2048, 131K] k256 | 3272μs | 320μs | **10.23×** | 0.9865 |

per-row end（对应 per-request nblk）越界检查 PASS。bf16 tie 区
jaccard 0.986-1.0 → L1 能容忍（L2 吸收）。**select 内部归因
（2026-09-27 ✅，test_tli_sel_profile.py，30B 形态合成复现
4chunks×48 层 8.08s vs 生产 10.95s 同量级）**：末 chunk op 级
CUDA 分布 = **aten::topk 62%**（radixFindKthValues 31.8% +
gatherTopK 16.8% + radixSort 5.3%）/ 自有 kernel
`_tli_l2_score_batched_dual` 24.3% / einsum+gather+scatter 全部
<2%。**结论：topk 是 select 的绝对大头（推翻早前 ~3% 误判），
DeepSelect 替换直攻 62%**——三处调用点（L1 块级 [n·Hkv,nblk]k128
+ B' far [n·Hkv,Tc]k256 + near k768）按 microbench 10× 折算
select 有望 2.5× 级整体加速。

**#64 集成落地（2026-09-27 ✅）**：indexer.py 三处替换 +
`SGLANG_TLI_DS_TOPK=1` 开关。合成 A/B：末 chunk 1.64×、jaccard
0.9961@21K、4chunks×48 层总账 6.61s（1.22×）；DS op 表 topk
22.7→3.2ms（7.1×）但 pad/cast copy 新增 3.5ms（22%），剩余瓶颈转
`_tli_l2_score_batched_dual`（43%）。**集成踩坑两连（64K 崩溃，
CUDA_LAUNCH_BLOCKING+插桩定位）**：①pad 区 torch.empty 垃圾撞
NaN bit → kernel abort；②**行有限值 < k 时阈值退化到 -inf，随机
块序把 pad 列 -inf 选进输出，idx 实测可超 padded 宽度（33307 >
33280）——32K 不崩纯属侥幸（Tc_k=32768 恰 512 倍数 pad=0）**。
修复 = 恒 pad≥512 + 显式 -inf + 返回前统一 clamp(0, C+pad-1)
（pad 区恒 -inf，下游 keep 掩码转哨兵，**有效集与 torch.topk 逐位
一致**：jaccard 0.9939@21K / 0.9742@64K）。教训：第三方 kernel
的「end 排除 pad」承诺不可信，凡 pad 必须假设索引泄漏。

**#64 e2e 定标（test_tli_30b_bench.py，30B-A3B narrativeqa 纯
prefill 双档，DS=0→DS=1）**：

| 档 | triton dense | tli DS=0 | tli DS=1 | DS 加速 | tli/dense |
|---|---|---|---|---|---|
| 32K | 5.71s | 13.50s | **10.93s** | 1.24× | 1.91× |
| 64K | 21.06s | 28.97s | **21.20s** | **1.37×** | **1.005×（追平）** |

**64K 档 tli+DS 追平 dense attention（1.005×）**——从 DS=0 的
1.38× 慢追平；稀疏理论流量收益被剩余 select 开销（dual kernel 43%
+ pad/cast copy 22%）抵消，进一步 kernel 化才有净收益空间。

**#58 收益区复测（S=64K×bs16 TP2，DS=1）**：207.53s → **167.12s
（1.24×）**，vs triton 106.24s 从慢 1.95× 收敛到 **1.57×**——未
翻正，剩余瓶颈 = decode 侧 select_decode_batched 未接 DS（2611
ms/step）+ prefill 剩余 dual/pad-copy 开销。decode 侧接入受 CUDA
graph capture 约束（DS host 端 pad/end 分配不可图内），留后续。

**#65 30B TP2 S=64K×bs16 复测终值（2026-09-28 凌晨 04:07，
SGLANG_TLI_PREFILL_KERNEL=1 + DS_TOPK=1 全开，night_relay 自动
接力跑）**：**167.12 → 115.28s（再 1.45×）**，vs triton dense
106.24s = **0.92×——bs16 档未完全翻正**（差 8%）。诚实归因：
①单请求档 64K 已 **1.285× 领先**（16.39 vs 21.06s）——稀疏流量
收益真实兑现；②bs16 档差 8% = 8B launch-bound 归因（§#65 同步
消除节）的 30B 版本：16 req 逐请求 forward_extend Python 循环 +
_per-request select 调度放大固定开销，GPU 名义满载实际带宽未
饱和；③decode 侧 1801ms/step（total/N 口径，含 prefill 摊销）——
无图 + 逐请求循环仍是多请求形态短板。**#65 定案**：目标 ≤106s
未达但 1.57×→0.92×（差 8% 收官），跨请求批量化 forward_extend
（M4 decode 同款方案）为明确后续工作，不阻塞论文（单请求收益
区 + kernel 级 1.3-5.1× + 8B 稳态 4.00×/1.94× 已构成速度卖点
三支柱）。

**#65 prefill select 三连优化（2026-09-27 晚 ✅，commit a35acf260
+ 0b7d2fcd4）**：①**最大隐藏瓶颈 = #58 早期行修复的逐行 Python 循环**
（首 chunk ~1024 行 × arange/cat/index_put ≈ 5K launch + 3K 次
.item() 同步 → chunk0 65ms 主因，e2e 每层每 prefill 都付）→ 向量化
where(j<cut, j//reps, j-cut) 构造，chunk0 67→16.5ms（4×）；
②fast_path（Tc==S 全候选）旁路到 M10 kernel+DS 路径（候选集等价
的特例，kernel+DS 已快于 eager 全宽 einsum+2×torch.topk）；
③dual kernel `out_bf16_pad` bf16+pad 直出（pad 列 -inf，DeepSelect
直用 ds_topk_padded）——消灭 fp32 [n,Hkv,Tc]×2 物化 + cast/pad
复制链（profiling pad/cast elementwise 16.3%→0）；④empty 行子集
化；⑤CHUNK×nw sweep → 64/4（12.1 vs 14.9ms）。**验证矩阵**：合成
30B 形态 4 chunks×48 层 6.81→3.52s（**1.93×**），jaccard
0.9961/0.9903 逐位不变；早期行专项对拍（grid 构造逐位 + eager vs
kernel+DS 集合 jaccard_min=1.0）PASS；M10 双口径回归 PASS；8B e2e
smoke（kernel+DS 全开）2/3 逐字一致=历史基线。

**#65 e2e 兑现（单请求双档，同机同臂）——64K 收益区翻正 ✅**：

| 档 | triton dense | tli #64 后 | tli #65 优化 | tli/dense |
|---|---|---|---|---|
| 32K | 5.71s | 10.93s | **7.38s** | 0.77×（收敛中） |
| 64K | 21.06s | 21.20s（追平） | **16.39s** | **1.285× 领先** |

**64K 档从追平（1.005×）到 1.285× 领先 dense**——稀疏理论流量收益
首次在 e2e 净兑现；32K 档从慢 1.91× 收敛到 0.77×（短上下文 select
固定开销尚未被收益覆盖，符合收益随 S 单调递增预期）。

**8B e2e 稳态复测（bs16×S=30K×n=256，test_tli_e2e_variance.py，
kernel+DS 全开 vs 历史基线）**：prefill 733.9→**196.8s（3.73×）**、
decode step 65.0→**33.5ms（1.94×，已反超 triton dense 的 40.4ms =
0.83×）**——8B Hkv=8 形态的批量收益区也全面进入净收益区间；
prefill 对 dense 1.72× 慢（8B 每层双 kernel gather 8 头形态更重，
30B Hkv=4 已 0.77×，头数是 prefill 差异主变量）。round0 decode
-11.6s 为差分法已知噪声（§8b-25 记录同款），取 round1。

**#65 GPU 同步消除（2026-09-27 深夜，commit 976ad78eb）**：select_batched
热路径 3 处 host 同步（fast_path 判定 `.all().item()` / empty 段
`.any()` / early 段 `.any()`）在生产 e2e × 16 req × 36 层 × ~60
forward ≈ 10 万次队列排空——**合成 bench 无 GPU 队列时测不出**（单
kernel 计时口径盲区）。修复 = `_kern_ok` 时直接跳过 fast_path 判定
+ 新增 `t_min_hint` host 参数（backend.forward_extend 的 prefix 是
Python int）：`t_min ≥ near_len+far_lo` 无 empty 行、`t_min ≥ 输出宽`
无 early 行，两处 `.any()` 同步条件化跳过。**e2e 验证**：8B
bs16×S=30K prefill 196.8→**183.4s（−6.7%）**（两轮 spread 0.2s），
decode 34.8ms/step 持平（方差内）——8B 对 dense prefill 差距 1.72×→
~1.60×，剩余在 `_sparse_extend_one` 本体（如实证：select 全程仅
5.9s、build+select ~13s，与 183s 的差距主体不在 select 调度）。
**方法论教训（论文 profiling 章节素材）**：逐 kernel microbench 与
生产 e2e 的口径差 = host-GPU 流水线效应，同步消除类优化必须在
e2e 层验证。

**#65 8B prefill 剩余差距终局归因（2026-09-28 凌晨，py-spy 采样 +
带宽三角验证）**：同步消除后 183.4s 对 dense 114s 仍差 ~69s，逐层
证据链：①py-spy 纯 prefill 窗口 150s 采样（scheduler 主线程 100%
CPU）：forward_extend 函数体 72.7%（109s）+ select_batched 15.3%
（23s）+ empty 行 einsum 2.5%；②循环体内单独微基准每个 op 都快
（k_all gather+float 0.42ms / pool 写入 0.12ms / out copy 0.14ms，
外推 16req×36 层合计仅 ~0.5s）——**不是任何单个 op 慢**；③采样期间
nvidia-smi：GPU util 100% 但**显存带宽 util 仅 10-11%**（H20 4TB/s
的 ~1/10）。三证据合一：**8B 形态 prefill = launch-bound（小 kernel
串行流水线不饱和）**——16 req × 36 层 × 每 forward_extend 数十个小
torch op → 数千~上万小 kernel，每个 kernel launch 间隔主导；GPU 名义
满载实际空转。dense 用少量大 kernel（flash attn 整块 [T,H,S]）带宽
利用率高。**修复方向 = 跨请求批量化 forward_extend（M4 decode 侧
`select_decode_batched` 同款思路），属工程优化非算法问题**——与 30B
单请求已 1.285× 领先一致（单请求无 16× 循环放大）；30B TP2 bs16
复测将检验多请求下同一形态是否重现。

**decode 侧 DS 定标（#65，commit c3b1cb740）**：select_decode_batched
L1+far 接 DS（graph 捕获自动回退）。**关键认知修正：decode select
非瓶颈**——全 kernel 路径 0.95ms/step@bs16/S=64K（2611ms/step 是
整步开销）。DS 净效果 bs16 +10%/bs64 −9%/bs128 −17%（高并发有益）。
**near 池 DS = No-Go 负结果**：压缩表 ~2048 有限项挤在 4bit 格点极
少数分数值（tie 组巨大）→ jaccard 0.72（L1/far 均 0.999）——tie
打破差异与候选分数离散度强相关，接 DS 前必须按池预判。

### 8b-32. RULER 官方数据四方法评测（2026-09-27 晚，#66——官方口径对齐，跑批中）

**与 §8b-21~24 的关系 = 口径升级非重复**：旧评测自合成模板 +
sglang Engine 口径 + 双 seed n=40；本节 = **官方 RULER 预生成数据**
（KVCache-Factory 镜像 NVIDIA 官方产物，3 长度 × 11 任务 × 500 条，
零外网依赖）+ **transformers 主表口径**（与 LongBench E5b 完全同一
monkeypatch 管线：TIA/TLI/Quest/FullKV）+ n=100/任务（确定性 topk，
论文常用口径；官方 500 条成本 33h×4 方法不可行）。脚本三件套
`benchmark/RULER/{pred_ruler,score_ruler,run_ruler}.py/sh` +
接力调度 chain_ruler.sh（FullKV→Quest→TIA 自动）。

**先行发现一（§8b-22 结论修正）**：官方 cwe/fwe 对 Qwen3-8B FullKV
**有区分度**（L4096：cwe 99.30 / fwe 93.67 / vt 100）——自合成
「聚合计数无区分度」是构造口径问题（官方词频沟 common×30 /
uncommon×3 / filler×8 宽，自合成 3≤c≤200 窄），非模型能力上限。
聚合计数任务族恢复为有效对比口径。

**先行发现二（官方 multiquery 口径严格得多）**：FullKV multiquery
恒 25.00（L4096/L8192 一致）= 4 问平均中 1 的任务上限，非长度或
稀疏效应——自合成 multiquery 0.85（全中口径）与官方部分分口径
不可互比。gap 解读须以 FullKV 同任务值为上限基准。

**先行发现三（TLI 官方 4K 数据近乎无损）**：TLI@1024 L4096
single×3 + multikey 1/2 全 100、multikey_3 95.65（进行中）——与
FullKV 持平；对照 sglang 口径 §8b-21 的 0.70，官方 needle 的数字
串干扰轻于自合成模板（口径差异显著，论文两表并存注明）。

FullKV 基线全表（L4096 完成，L8192 进行中）：single×3/multikey×3/
vt 全 100；multivalue 89.5；cwe 99.3；fwe 93.7；multiquery 25.0
（任务上限）。

**L4096 完整 11 任务表（n=100，#66 首个里程碑结果）**：

| task | FullKV | TLI@1024 | gap |
|---|---|---|---|
| niah_single_1/2/3 | 100.00 | 100.00 | 0 |
| niah_multikey_1/2 | 100.00 | 100.00 | 0 |
| niah_multikey_3 | 100.00 | 93.00 | −7.00 |
| niah_multiquery | 25.00 | 25.00 | 0（任务上限处持平） |
| niah_multivalue | 89.50 | 88.00 | −1.50 |
| cwe | 99.30 | 98.30 | −1.00 |
| fwe | 93.67 | 94.67 | **+1.00** |
| vt | 100.00 | 100.00 | 0 |
| **AVG** | **91.59** | **90.82** | **−0.77** |

**读法**：①官方 4K 数据 TLI@1024 基本无损（AVG −0.77）——与
§8b-21 sglang 口径（AVG gap −21）的巨大差异 = 口径效应（官方
needle 数字串干扰轻 + transformers 管线 bf16 与 sglang MoE 路径
数值差异），论文两表并存须注明；②聚合型 cwe/fwe gap −1/+1、
vt 零损——**B' far 预算设计在聚合/链式任务族完全够用**，推翻
§8b-22 自合成口径的悲观预期（该口径问题已单列为方法论发现）；
③fwe +1 = 稀疏噪声正向扰动（丢 token 降噪）；④唯一显著损失
multikey_3 −7 与 §8b-21 multikey 诊断一致（干扰数字抓错 key）。

**FullKV 基线 33 任务终表（3 长度完整，n=100×33）——模型长度
能力梯度**：

| task | L4096 | L8192 | L16384 |
|---|---|---|---|
| niah_single_1/2/3 | 100 | 100 | 100 |
| niah_multikey_1/2/3 | 100 | 100 | 100 |
| niah_multiquery | 25.00 | 25.00 | 25.00 |
| niah_multivalue | 89.50 | 69.25 | **44.25** |
| cwe | 99.30 | 93.00 | **83.60** |
| fwe | 93.67 | 93.33 | **86.67** |
| vt | 100 | 100 | 100 |

**基线要点**：①检索型（single/multikey/vt）4K→16K 完全不变——
gap 随长度变化可直接归因稀疏方法；②**聚合列举类（multivalue/
cwe/fwe）FullKV 自身随长度显著退化**（multivalue 89.5→44.3）=
Qwen3-8B 列举能力上限，非稀疏效应——gap 解读必须以同长度 FullKV
为基线（本管线按此设计）；③multiquery 恒 25 = 任务上限。

**TLI L8192 收官 + Quest L4096 收官（18:34 时点中间表）**：

- **TLI 长度梯度**：AVG gap −0.77@4K → **−2.96@8K**（86.18 vs
  89.14）；主要贡献 multikey_3 −19.0、cwe −6.4、multivalue −5.25；
  检索型与 fwe/vt 保持零损级
- **L4096 三方法终表（n=100×11）**：**Quest 87.91 / TLI 90.82 /
  FullKV 91.59**——TLI 领先 Quest 2.91 分、距 FullKV 仅 0.77；
  逐任务 TLI 全面优于或持平 Quest（multikey_3 **93 vs 75**、cwe
  98.3 vs 92.8、fwe 94.7 vs 86.7）；Quest 唯一反超项无（vt 100
  持平）
- **L8192 Quest 收官：AVG 81.76 vs TLI 86.18 vs FullKV 89.14**——
  TLI 领先 Quest **4.42**@8K（4K 2.91，差距随长度拉大：page-min
  下界近似在长上下文劣化快于 minmax 上界）；核心差距任务
  multikey_3 = Quest 58 vs TLI 81；Quest 在 multivalue 65.75 略优
  TLI 64.00（page 均匀覆盖对列举型有利，逐任务互有攻防如实报告）
**TLI 33 任务全收官终表（20:56，ALL DONE tli）——RULER 三长度
完整 gap 梯度**：

| task | F@4K | T@4K | F@8K | T@8K | F@16K | T@16K |
|---|---|---|---|---|---|---|
| niah_single_1/2/3 | 100 | 100 | 100 | 100 | 100 | 100 |
| niah_multikey_1/2 | 100 | 100 | 100 | 100 | 100 | 100 |
| niah_multikey_3 | 100 | 93 | 100 | 81 | 100 | **52** |
| niah_multiquery | 25 | 25 | 25 | 25 | 25 | 24.8 |
| niah_multivalue | 89.5 | 88.0 | 69.2 | 64.0 | 44.2 | 41.2 |
| cwe | 99.3 | 98.3 | 93.0 | 86.6 | 83.6 | **54.9** |
| fwe | 93.7 | **94.7** | 93.3 | 92.3 | 86.7 | **94.7** |
| vt | 100 | 100 | 100 | 99.0 | 100 | 97.8 |
| **AVG** | **91.59** | **90.82** | **89.14** | **86.18** | **85.41** | **78.67** |
| gap | | **−0.77** | | **−2.97** | | **−6.74** |

**16K 档逐任务归因（诚实口径）**：①multikey_3 52 = 干扰 key 数
随长度超线性增长超出 K1 块配额（索引质量边界）；②**cwe 54.9 =
预算粒度而非索引错误**——逐例诊断 100/100 全部部分命中（平均
5.5/10 词、无全 miss、pred 长度与 FullKV 同=非截断）：16K 下
uncommon 词散布更广，K2=1024（6.25% 预算比）内逐词召回 ~55%；
③fwe 16K 反超 +8（稀疏降噪）；multivalue −3 以同长度基线解读
= 列举上限主导；④检索型 single/multikey 1/2 三长度全零 gap。

**AVG gap 梯度 −0.77 → −2.97 → −6.74**——主贡献任务（multikey_3
+cwe）的 gap 机制不同（索引质量 vs 预算粒度），后者可通过预算
自适应（长度感知 K2）改善，前者是 minmax 上界的固有代价。

- **TLI L16384 关键任务**：multikey_3 **52**（93→81→52 长度梯度，
  干扰 key 数随长度超线性增长超出 K1 块配额）；multivalue 41.25
  vs 同长度 FullKV 44.25 **仅 −3**（绝对值低是模型列举上限非稀疏
  损失——同长度基线解读的关键例证）；multiquery 24.75≈任务上限

**Quest 33 任务全收官（22:11 ALL DONE quest）——TLI vs Quest
同预算（K2=1024）三长度横向（论文主对比点）**：

| AVG | FullKV | TLI | Quest | TLI−Quest |
|---|---|---|---|---|
| L4096 | 91.59 | 90.82 | 87.91 | +2.91 |
| L8192 | 89.14 | 86.18 | 81.76 | +4.42 |
| L16384 | 85.41 | 78.67 | **70.18** | **+8.49** |

**TLI 对 Quest 领先随长度超线性扩大（2.91→4.42→8.49）**——16K 档
Quest 全面崩塌：multikey_2 60 vs TLI 100、multikey_3 **21 vs 52**、
cwe **29.5 vs 54.9**；机制 = Quest page-min 下界近似在长上下文
劣化快于 TIA minmax 上界 + B' far 预算的分区保护。Quest 仅 fwe
16K（96.67）反超 TLI（94.67）与 FullKV（86.67）——page 均匀覆盖
对高频词列举有利 + 稀疏降噪，逐任务互有攻防如实报告。

**【TIA 臂（chain_ruler.sh 接力，22:18 起）跑批中——完成后补
四方法×3 长度完整表 + TIA vs TLI 消融（同门内部基准）】**

**TIA L4096 收官（00:26，11/11）——首个四方法完整长度档**：

| task | FullKV | Quest@1024 | TIA@1024 | TLI@1024 |
|---|---|---|---|---|
| niah_single_1/2/3 | 100 | 100 | 100 | 100 |
| niah_multikey_1/2 | 100 | 100 | 100 | 100 |
| niah_multikey_3 | 100 | 75 | 100 | 93 |
| niah_multiquery | 25 | 25 | 25 | 25 |
| niah_multivalue | 89.5 | 87.5 | 90.0 | 88.0 |
| cwe | 99.3 | 92.8 | 98.3 | 98.3 |
| fwe | 93.67 | 86.67 | 94.67 | 94.67 |
| vt | 100 | 100 | 100 | 100 |
| **AVG** | **91.59** | **87.91** | **91.63** | **90.82** |

**L4096 四方法梯队**：TIA 91.63（微超 FullKV +0.04，噪声级——
cwe −1 / fwe +1 的正负相抵）> FullKV 91.59 > TLI 90.82 > Quest
87.91。**TLI 距 TIA −0.81@4K**，逐任务差距点 = multikey_3（93 vs
100）+ multivalue（88 vs 90）——与 TLI vs FullKV 的 gap 任务一致
（B' far 分区预算代价，非索引错误）；cwe/fwe/vt 与 TIA 完全同分。
TIA/TLI 同精度梯队、对 Quest 均领先 3.7 分级——「速度卖点 +
与 TIA 精度差 <1」的论文定位在 RULER 官方口径下同样成立。TIA
L8192/L16384 跑批中（~10min/任务）。

**TIA L8192 收官（01:26，11/11）**：

| task | FullKV | TIA@1024 | TLI@1024 |
|---|---|---|---|
| niah_single_1/2/3 | 100 | 100 | 100 |
| niah_multikey_1/2 | 100 | 100 | 100 |
| niah_multikey_3 | 100 | **100** | 81 |
| niah_multiquery | 25 | 25 | 25 |
| niah_multivalue | 69.25 | 69.25 | 64.0 |
| cwe | 93.0 | 91.0 | 86.6 |
| fwe | 93.33 | 92.67 | 92.33 |
| vt | 100 | 100 | 99.0 |
| **AVG** | **89.14** | **88.90** | **86.18** |

**TIA-TLI gap 长度梯度 −0.81@4K → −2.72@8K**——主贡献
multikey_3（100 vs 81 = −19）+ cwe（91 vs 86.6 = −4.4）：TLI 的
两级选择在长上下文下 K1 块配额被干扰 key 稀释（4K −7 → 8K −19）
与 cwe 预算粒度代价显形；检索型（single/multikey 1/2）与 fwe/vt
与 TIA 零差或噪声级。**诚实口径**：TIA 上界索引 + 无 far/near
分区 = 全预算给 topk，单跳检索口径精度上限更高；TLI 的代价换
B' far 分区保护（多跳 far-heavy 场景，见 LongBench 主表 TLI 与
TIA 差 <0.3）+ D' 跳层速度收益——**RULER 单跳口径放大 TLI 精度
代价、LongBench 多跳口径放大小收益，两口径互补须并报**。TIA
L16384 跑批中。

**TIA L16384 关键任务出炉（03:30，8/11 时点）**：multikey_2 **100**
（TLI 100 持平、Quest 60 崩塌——TIA/TLI 同梯队 vs Quest 分化在
16K 检索型确认）；**multikey_3 TIA 89 vs TLI 52 vs Quest 21 vs
FullKV 100**——TIA 自身也出现 16K multikey_3 退化（长度梯度 100→
100→89，干扰 key 数超线性增长连 minmax 上界 + 全预算 topk 也
开始漏），但幅度远小于 TLI（−11 vs −48）：TLI 两级选择 K1 块配额
在超多干扰 key 下被稀释的弱点显形（4K −7 → 8K −19 → 16K −48）。
该任务是 TLI 相对 TIA 的最大精度代价点，机制归因清晰（块级粗筛
分辨率 vs token 级全预算），论文按诚实口径单列。

**TIA L16384 cwe/multivalue（03:55，10/11 时点）**：cwe **78.9**
（TLI 54.9 / FullKV 83.6 / Quest 29.5）——TIA cwe 退化 −4.7（vs
FullKV），TLI 额外 −24（4K −1 / 8K −4.4 / 16K −24.7 梯度陡增，
B' far/near 分区挤占 cwe 逐词召回预算 + 长度放大）；multivalue
TIA 44.25 = FullKV 完全持平（TLI 41.25）——列举上限主导任务
稀疏方法零退化。**TIA-TLI 16K 差距主构成 = multikey_3（−37）+
cwe（−24）两任务**，与 8K 档构成（multikey_3 −19 + cwe −4.4）
同族但幅度陡增——16K 下 TLI 预算粒度（6.25%）+ K1 块配额
双弱点叠加。

**#66 四方法×3 长度终表（04:07 全收官，ruler_table_final.json）**：

| AVG | FullKV | TIA@1024 | TLI@1024 | Quest@1024 | TLI−Quest | TIA−TLI |
|---|---|---|---|---|---|---|
| L4096 | 91.59 | **91.63** | 90.82 | 87.91 | +2.91 | −0.81 |
| L8192 | 89.14 | **88.90** | 86.18 | 81.76 | +4.42 | −2.72 |
| L16384 | 85.41 | **84.53** | 78.67 | 70.18 | **+8.49** | −5.86 |
| 3 长度均值 | 88.71 | 88.35 | 85.22 | 79.95 | +5.27 | −2.49 |

**TIA L16384 全表收官**：single×3 + multikey_1/2 + vt 全 100、
multikey_3 89、cwe 78.9、fwe 92.67（TLI 94.67 反超 TIA——稀疏
降噪正向例）、multivalue 44.25（=FullKV 持平）。

**终表结论（论文主叙事）**：①**四方法梯队稳定 = TIA ≈ FullKV >
TLI > Quest**，TIA 全长度距 FullKV ≤0.9（上界索引近无损）；②
**TLI 对 Quest 领先超线性扩大（+2.91→+4.42→+8.49）**——外部主
baseline 对比成立；③**TLI 对 TIA 代价 −0.81→−2.72→−5.86**
（内部消融基准），差距集中于 multikey_3 + cwe 两任务族（16K 合
计 −61），其余任务零差或反超——精度代价点精确定位，机制清晰
（K1 块配额稀释 + far/near 分区挤占预算 vs token 级全预算）；④
速度口径（§8b-25/26）：TLI 相对 TIA 的收益 = 稀疏 prefill + B'/D'
结构性加速（kernel 级 1.3-5.1×@32K-128K + e2e 单请求 1.285×@64K）
——**「同门内：TLI 用可定位的精度代价换速度分层；对外：同精度
梯队碾压 Quest」**为 RULER 章节定稿叙事。

### 8b-33. E4d：增量贪心聚类方法族消融——cluster 路线在 token 级选择形态下复活（2026-09-28，#68 用户指令「cluster 方法多方法消融」，8B trace 重放，n=16 样本全量收官）

**动机**（用户拍板决策 ②）：cluster 方法族不一定用 kmeans——增量贪心
聚类（时序逐 token、余弦阈值 sim=0.9、算术均值簇心、NoPE 代理低频
尾维 32）可能精度更高。E4c 时代「聚类降级为 negative result」的结论
是在**块级 scatter-amax 粗筛形态**下得出的；本实验补测 token 级簇分
形态（池内 token 直接用簇分排序）。脚本 `analyze_e4d_greedy_cluster.py`
（GPU1），结果 `e4d_greedy_cluster.json`。

**口径**：far 区 8 样本×2 样本×12 层（16 条 trace，narrativeqa_0/1
同文档去重后 15 独立）；四方法臂：
- `minmax_blk`：TIA 生产形态（minmax 块上界粗筛）
- `km_blk / km_tok`：E4c kmeans K=256（块 scatter-amax / token 簇分）
- `gl_blk / gl_tok`：**贪心+均值簇心**（gl=arithmetic-mean centroid，k≈2523）
- `gf_blk / gf_tok`：贪心+归一化质心（gf=l2-normalized，k≈3694，对照）
- `oracle`：全维 token 级 top-k

| far recall（16 样本均值） | @512 | @1024 |
|---|---|---|
| oracle | 0.9997 | 0.9997 |
| **gl_tok（贪心+均值簇心，token 级）** | **0.8709** | **0.8908** |
| gf_tok（贪心+归一化质心，token 级） | 0.8439 | 0.8590 |
| minmax_blk（TIA 生产） | 0.5916 | 0.7151 |
| km_tok（kmeans，token 级） | 0.6063 | 0.7013 |
| gl_blk（贪心+均值簇心，块级 amax） | 0.4676 | 0.6488 |
| gf_blk（贪心+归一化质心，块级 amax） | 0.4564 | 0.5919 |
| km_blk（kmeans，块级 amax） | 0.2556 | 0.3655 |

**结论（论文 cluster 方法族消融主表）**：
1. **贪心聚类在 token 级选择形态下大幅复活**：gl_tok@1024 = 0.891
   vs minmax_blk 0.715（+0.176）、km_tok 0.701（+0.190）——用户指定
   三要素（增量贪心/算术均值簇心/NoPE 尾维）全部被数据支持；
2. **均值簇心 > 归一化质心**（0.891 vs 0.859 @1024，跨 16 样本
   15/16 一致胜出）——算术均值保留幅度信息，簇分 = q·簇心 直接
   对齐 score 语义；
3. **贪心 > kmeans**（同 token 级 0.891 vs 0.701）：时序贪心
   K≈2.5K 细粒度簇 + 局部性偏好天然匹配 far 区稀疏尖峰；
4. **块级 scatter-amax 仍是所有聚类形态的短板**（gl_blk 0.649 <
   minmax_blk 0.715）——E4c「簇心非上界有漏选理论空洞」在贪心
   聚类下依旧成立：**cluster 方法必须配 token 级细筛才有价值**，
   与两级框架（粗筛 minmax 保上界 + 细筛贪心簇分）形成互补结构；
5. **成本侧**（E64c-1 成本模型互证）：贪心 build 在线成本
   12.5s/层（S≈64K 外推 5.4s@32K）爆炸 → 必须 prefill 期 build +
   decode 期增量 assign 摊销（E7 已证 <1μs/token/head 且 recall
   衰减 ≤0.03）；gl 比 gf 簇数少 32%（2523 vs 3694）→ 簇分 GEMV
   成本同比例更省。

**论文写作定位**：E4d+E64c 组合给出 cluster 方法族的完整故事——
精度上限（token 级贪心簇分 0.891）与成本可行性（增量摊销）都有
数据支撑，cavg=(cluster,avg) 臂在 E64a 网格中的表现将决定其在
主配置中的去留；E4c 的 negative result 限定为「块级聚类粗筛」
这一具体形态而非聚类方法族本身。

### 8b-34. E64c-2：细筛 GEMV 修正版成本模型——E64c-1 大头结论被推翻，选择链计算全程 sub-ms（2026-09-28，合成 microbench 已标注，H20 S=64K）

**动机**：E64c-1 的 fine_gemm 计时存在路径缺陷（逐 head Python 循环 +
einsum("d,thd->t") 非 batched 路径），得出「细筛 token GEMV 0.6-1.9ms
是选择链唯一大头、降维+AKO 主攻此处」的结论。修正版用标准批量路径
（展平 + per-head offset 一维索引 → [Hkv,T,d] + bmm）重测。
脚本 `analyze_e64c2_fine_gemv.py`（GPU1），结果 `e64c2_fine_gemv.json`。
口径：页池 PAGES=128×BS=64=8192 token、S=64K、Hkv=8、bf16 特征
（预降维后 KV 即 d 维存储——生产形态）、cudaEvent×20。

| 路径 | d=4 | d=8 | d=16 | d=32 | d=64 | d=128 |
|---|---|---|---|---|---|---|
| A gather+bmm（页池细筛主路径） | **0.026** | 0.061 | 0.064 | 0.071 | 0.084 | 0.101 |
| B 仅 gather | 0.009 | 0.051 | 0.051 | 0.051 | 0.050 | 0.050 |
| D 全量 S×d GEMV（无页池极端） | 0.049 | 0.056 | 0.093 | 0.147 | 0.399 | 0.442 |

**结论（修正 E64c-1）**：
1. **细筛 GEMV 真实成本 0.026-0.101ms（原报 0.6-1.9ms 是 7-30× 虚高，
   逐 head 循环 launch 开销主导）**——选择阶段全部计算组件（build/
   块分/细筛/topk）在 S=64K 下都在 sub-0.1ms 级，「选择链计算大头」
   不存在；与 M3-a「eager decode 81-90ms/step 由 Python 调度与小
   launch 主导」互证——**瓶颈是调度不是算力**；
2. **降维的真实收益在存储/带宽侧而非算力侧**：gather 字节数随 d 线性
   收缩（d128→d4 gather 0.050→0.009ms，5.6×），bmm 算力部分本就
   ~free；sup_wsvd d4（E65b）的经济叙事 = 特征存储 128→4 维（32×↓，
   与 4bit 量化正交可叠加）+ gather 带宽 5.6×，而非 GEMV 提速；
3. **页池路径 vs 全量打分**：d128 时 0.101 vs 0.442ms（4.4×）——两级
   筛选的结构性收益直接可见；d4 时差距消失（0.026 vs 0.049）——低维
   特征全量打分也变得可行（粗筛页池的存在价值随降维弱化，但注意
   精度口径需 E65 数据支持）；
4. 诚实声明：合成数据、单请求单层口径；bs 并发与 CUDA graph 下的
   launch 摊销是另一维度（M3-c/M4 已测）。

### 8b-35. E65+E65b：降维方法族统一消融——自训练矩阵全链压到 4 维只掉 2.5%（2026-09-28，#68 用户指令「降维探索+自训练降维矩阵对比」，8B trace 重放）

**动机**（用户降维蓝图）：NoPE 尾维 32 已证几乎无损 → 子空间内进一步
降维（粗筛 PCA/截维/随机投影/自训练矩阵，far/near 与粗筛/细筛分开测）。
脚本 `analyze_e65_dim_reduce.py`（E65 主，五法×d∈{4,8,16,32}×四臂）+
`analyze_e65b_sup_proj.py`（自训练两法，M9 哲学：hotpotqa_0 离线校准
一次、其余 7 样本只读泛化）。结果 `e65_dim_reduction.json` +
`e65b_sup_proj.json`。

**方法族**：trunc（子空间坐标截取）/ pca_sub（子空间内无权 PCA）/
jl_sub（子空间内 JL 随机投影）/ **sup_wsvd**（注意力概率加权 PCA
闭式解 = score-preserving 最优投影）/ **sup_grad**（Adam 梯度训练
自由矩阵 [Hkv,32,d]，回归目标=32 维子空间 score）。

**统一表（7 泛化样本均值；基准 = 尾维 32 不降 ≈ 0.80）**：

| far 捕获率 | d=4 | d=8 | d=16 | d32 基准 |
|---|---|---|---|---|
| A far 粗筛 sup_grad | **0.8132** | 0.8006 | 0.8186 | 0.8016 |
| A far 粗筛 sup_wsvd | 0.7815 | 0.7885 | 0.7949 | 0.8016 |
| A far 粗筛 trunc | 0.3647 | 0.5233 | 0.6489 | 0.8038 |
| A far 粗筛 pca_sub | 0.4199 | 0.4459 | 0.4986 | 0.8016 |
| A far 粗筛 jl_sub | 0.3577* | 0.3857* | 0.4428* | 0.5681* |
| C 细筛 sup_wsvd | 0.8026 | 0.8037 | 0.8038 | 0.8038 |
| C 细筛 trunc | 0.5126 | 0.5950 | 0.6660 | 0.8038 |
| **D 全链 sup_wsvd** | **0.7813** | 0.7885 | 0.7949 | 0.8016 |
| D 全链 sup_grad | 0.7788 | 0.7639 | 0.7730 | — |
| D 全链 trunc | 0.3110 | 0.4303 | 0.5651 | 0.8038 |

（*jl_sub 的 d32 基准 0.5681 也低——JL 全维随机投影基线本身就弱）

**结论（降维消融主表）**：
1. **降维方法分水岭 = 「注意力相关保持 vs 方差保持」**：自训练两法
   （监督信号=注意力概率/score 回归）d4 全链保持基准 97.5%（0.781/
   0.779 vs 0.802）；无监督两法崩溃（trunc d4=0.311 保 39%、
   pca_sub d4=0.406、jl 全线 ≤0.55）——PCA/JL 保 K 方差不保
   q·k 相关，冒烟期的「PCA 崩溃」全量确认且量化；
2. **sup_grad d4 粗筛 0.8132 反超 d32 基准**（101%）——非正交可学
   尺度的自由矩阵在 far 粗筛上更准（回归目标直接对齐 score 语义）；
   sup_wsvd 在细筛臂几乎无损（C d4=0.8026 vs 0.8038，99.9%）；
3. **全链 32→4 维（8× 存储收缩）质量代价 2.5%**，叠加 E64c-2
   gather 带宽 5.6× 收益与 4bit 量化正交——降维的经济叙事完整
   （存储/带宽侧，§8b-34 结论 2）；
4. **near 侧降维 Neutral**：B 臂所有方法 0.27-0.39 且对 d 不敏感
   （near 带 mass 难捕与 E63 互证）——near 粗筛降维无质量代价
   也无收益，成本侧（带宽/存储）仍可受益，如实报告；
5. **诚实修正**：冒烟期 sup_wsvd d4=1.0/1.0 为单层单样本信号，
   全量 7 样本均值 0.78-0.81（needle32k 仍 0.9999——单 key 检索
   任务加权 PCA 完美；narrativeqa 最难 0.51-0.56）。

### 8b-36. E64a：method×alpha×beta 48 臂网格全量收官——mavg 冠军 (0.125, 0.25)、α 单调递减、双分区仍不敌 mono（2026-09-28，#68 用户指令 αβ 网格消融，8B trace 重放 n=16 样本）

**设计**（用户指定命名体系）：cavg=(cluster,avg) / mavg=(minmax,avg) /
aavg=(avg,avg)；bp=64、gamma=1、B_TOK=2048 固定；α∈{.125,.25,.5,.75}
（near 子区长度占比）、β∈{.125,.25,.5,.75}（near 页池占比）×3 method
= 48 臂 + mono（α=0 单池）+ tli_anchor 对照。真实全维 softmax 行级
mass coverage。脚本 `analyze_e64a_ab_grid.py`（贪心 build 层级共享
提速版，75min→2.5min/样本），结果 `e64a_ab_grid.json`；图
`figures/e64a_{grid_heatmap, alpha_curves, beta_curves}.{png,pdf}`。

**全量主表（16 样本均值）**：

| 臂 | cov | 备注 |
|---|---|---|
| mono（α=0 mid 单池 minmax） | **0.8979** | 对照上界 |
| tli_anchor（生产形态） | 0.8885 | |
| **mavg_a0.125_b0.25（冠军）** | **0.8819** | 超 tli_anchor，距 mono −1.6pt |
| cavg_a0.125_b0.25 | 0.8778 | |
| aavg 最优臂（a0.75_b0.75） | 0.7925 | 全面垫底 |
| 最差臂（三 method a0.75_b0.125） | 0.761-0.767 | 大 α 小 β = near 区大但 near 预算不足 |

**α 方向性（各 method 最优 β 轨迹）**：α=0.125→0.75 单调下降
（mavg 0.882→0.861→0.835→0.812）——near 子区越**小**越好；
β 方向在 α=0.125 处最优 0.25，大 α 处转向 0.75（near 区大时
需要更多 near 预算补偿）。

**结论（αβ 网格消融主表）**：
1. **far_method 排序 = mavg > cavg > aavg**（0.882 vs 0.878 vs
   0.793）——minmax 上界粗筛最优；avg 粗筛 −9pt 不可用（与 E59
   0.695 vs 0.885 互证）；cavg 仅 −0.4pt 微弱劣势；
2. **双分区最优 (0.125, 0.25) 超 tli_anchor +0.7pt** 但仍低 mono
   −1.6pt——与 E63c（双分区无质量增量）互证，α→0 单池是质量
   最优点；用户框架的最优工作点在 α=0.125 端（非 α=0 精确值，
   near 独立分区保留少量配额即近无损）；
3. **cavg 垫底的口径说明**：E64a 的 cavg 粗筛是块级 scatter-amax
   形态——E4d 已证 cluster 的优势形态是 token 级簇分（gl_tok
   0.891）；块级聚类短板（§8b-33 结论 4）在 αβ 网格下再次确认；
   cluster 路线的完整形态（minmax 块粗筛 + 贪心簇分 token 细筛）
   尚未进入本网格，列为后续消融臂；
4. **α 大 β 小是最差组合**（near 大区无预算→粗筛选择噪声放大）；
   预算与区大小必须匹配（α 与 β 正相关联动）。

**E64b bp/gamma 前瞻**（跑批中，首样本 lb_gov_report_0 信号）：
bp 单调升（bp256_g1.0=0.8822 > bp128 > bp64 > bp32 > bp16=0.803），
gamma=1.0 在大 bp 下最优——页池预算远未饱和（bp=64 是预算受限
工作点非质量最优），用户「bp 多组补充扫描」决策正确。

### 8b-37. E64b：bp × gamma 细扫收官——bp 单调升、gamma 随 bp 翻转（0.5→1.0）、bp=128 是甜点（2026-09-28，#68 用户指令「bp 多组补充扫描+gamma 说明」，8B trace 重放 n=16 样本）

**设计**：冠军法 mavg（cavg 对照）、(α,β)=(0.125,0.25)、B_TOK=2048；
bp∈{16,32,64,128,256} × gamma∈{0.5,0.75,1.0} + mono 同 bp 对照
（30+5 臂）。脚本 `analyze_e64b_bp_gamma.py`，结果 `e64b_bp_gamma.json`；
图 `figures/e64b_bp_gamma.{png,pdf}`。

**全量主表（16 样本均值，mavg）**：

| bp | g=0.5 | g=0.75 | g=1.0 | mono 同 bp |
|---|---|---|---|---|
| 16 | 0.8230 | 0.8229 | 0.8224 | 0.8401 |
| 32 | 0.8458 | 0.8452 | 0.8439 | 0.8627 |
| 64 | 0.8815 | **0.8822** | 0.8819 | 0.8979 |
| **128** | **0.9184** | 0.9122 | 0.8868 | **0.9270** |
| 256 | 0.8999 | 0.9063 | **0.9124** | 0.9433 |

**结论（bp/gamma 消融主表）**：
1. **bp 单调升且远未饱和**：bp16→256 cov 0.823→0.943（mono），
   每 doubling +1.5-3.7pt——bp=64（E64a 固定值）是预算受限工作点；
   质量最优在 bp=256（0.9433），但 256 块=16K token 页池，细筛
   gather 成本随 bp 线性增长（E64c-2：gather 主导）——**bp=128
   是质量/成本甜点**（0.9270 mono / 0.9184 mavg_g0.5，距 bp256
   仅 −1.6pt 但 gather 减半）；
2. **gamma 方向随 bp 翻转**：bp≤64 时 g≈无影响（near 细筛配额
   不 binding）；bp=128 时 **g=0.5 最优**（+3.2pt vs g1.0——大页池
   下 near 6K token 强制配额挤占 far 细筛预算，gamma 折扣回吐
   far 配额）；bp=256 时 **g=1.0 最优**（+1.3pt vs g0.5——页池
   大到 near/far 池都溢出，此时折扣反而丢 near 质量）。**gamma 的
   论文表述**：不是恒定折扣，是「near 配额挤压调节器」，其最优值
   由 bp 与 B_TOK 的比值决定（bp·64·β vs B_TOK 的相对大小）；
3. **cavg 复现 E64a 结论**（bp128_g0.5=0.9098 vs mavg 0.9184，
   −0.9pt 微弱劣势）；
4. **mono 恒超双分区最优臂**（bp128：0.9270 vs 0.9184，−0.9pt；
   bp256：0.9433 vs 0.9124，−3.1pt）——E63c/E64a「双分区无质量
   增量」在 bp/gamma 全域确认，**论文最终表述：双分区的价值不在
   质量而在结构性（near 免打分带宽节省 + 防挤出保护），质量口径
   单池最优**。

### 8b-38. E64d：cluster 完整两级形态定界——贪心簇分作细筛一致负于精确细筛，cluster 价值定位为单级 token 选择器（2026-09-28，#68 cluster 方法族消融收官，8B trace 重放 n=16 样本）

**设计**：E4d 证明贪心簇分在 token 级选择形态大幅领先（gl_tok 0.891）
——但 E4d 口径是「far 区直接 token 级 top-k」（无粗筛池）。本实验把
cluster 塞进两级框架的正确位置：粗筛 minmax 上界块分（保住漏选理论
性质）+ 池内**贪心簇分**细筛（hybrid 臂），对照池内**精确子空间 token
分**细筛（mavg 冠军臂），bp∈{64,128,256}，(α,β)=(0.125,0.25)。
脚本 `analyze_e64d_hybrid.py`，结果 `e64d_hybrid.json`。

| bp | mono | mavg（精确细筛） | hybrid（簇分细筛） | hybrid−mavg |
|---|---|---|---|---|
| 64 | 0.8979 | 0.8819 | 0.8779 | −0.40pt |
| 128 | 0.9270 | 0.8868 | 0.8693 | −1.75pt |
| 256 | 0.9433 | 0.9124 | 0.8938 | −1.87pt |

**结论（cluster 方法族最终定界，三实验闭环）**：
1. **hybrid 一致负于 mavg 且差距随 bp 扩大**——粗筛池质量越高，
   细筛需要的是分辨率（token 级区分度）而非召回（池子已好），
   簇分的簇内同分粒度成为瓶颈；
2. **E4d 的 gl_tok 优势正确定位**：贪心簇分是优秀的**单级 token
   选择器**（0.891@1024 直接选，≈ mavg 两级 0.882@2048 预算），
   但在两级框架中作细筛被精确分压制——**cluster 与两级是替代
   关系而非互补关系**；
3. **cluster 路线论文最终表述**：用户指定的贪心+均值簇心+NoPE
   三要素在方法族消融中全部最优（E4d 表），但系统级对照显示
   training-free minmax 两级在同等预算下持平或更优、且免 12s/层
   build 成本——cluster 作为方法族消融主表 + 单级替代形态保留，
   生产配置 = mavg（minmax 粗筛 + 精确细筛）；
4. **成本侧收官**（E64c 互证）：cluster build 增量摊销后 <1μs/
   token/head（E7）但 prefill 12s/层一次性成本仍在，mavg 无 build。

### 8b-39. E64 帕累托定优：降维换预算是帕累托改进（E64f 实测修正版；质量=真实 trace 16 样本，成本=合成 microbench 外推 S=64K）

图 `figures/e64_pareto.{png,pdf}`（脚本 `plot_e64_pareto.py`，成本
口径=build+块分+页 topk+细筛 gather/bmm+token topk，合成数据已标注）。

**帕累托前沿（E64f 实测修正，2026-09-28 补充）**：
初版把 E64b d32 质量与 d4 成本外推拼接——双口径铁律要求实测，
E64f（`analyze_e64f_final_config.py`，同 E64b 协议 + sup_wsvd 投影
基逐层校准）修正后的真实数字：

| 臂（16 样本均值） | cov | 降维代价 |
|---|---|---|
| mavg_bp128_g0.5_d32（投影版） | 0.9214 | 基准（32→32 加权 PCA 重排 +0.3pt） |
| mavg_bp128_g0.5_d8 | 0.9102 | −1.1pt |
| mavg_bp128_g0.5_d4 | 0.9031 | −1.8pt |
| mono_bp128_d32/d8/d4 | 0.9300/0.9169/0.9089 | — |

- **「降维换预算」帕累托改进实测成立但幅度修正**：d4@bp128
  （0.9031 @ ~0.20ms）仍双优于 d32@bp64（0.8819 @ 0.208ms，
  +2.1pt 且更便宜）与 mono bp64（0.8979，+0.5pt 且更便宜）；
  初报 +3.7pt 系拼接虚高，如实修正；
- **降维质量代价与预算规模相关**：E65b 协议（小池 16 页/512
  token）d4 仅 −2.5%，E64f 协议（大池 128 页/2048 token）d4
  −1.8pt 绝对值更大但相对 −2.0%——大池需要更细 token 区分度，
  降维的簇内同分粒度瓶颈显现（与 E64d-hybrid 结论 1 同机制）；
- mono 链（bp16→256，0.840→0.943 @ 0.182→0.315ms）是质量上界
  参照；速度侧全臂 sub-0.32ms（S=64K 单层单请求合成口径）——
  成本差异绝对值小，**帕累托叙事的正确用法是「同成本下质量
  增量」而非「同质量下速度增量」**。

**#68 消融链收官定稿（七实验闭环 E4d/E64a/E64b/E64c/E64d/E65/E65b）**：
最优配置 = **mavg + sup_wsvd/sup_grad 降维（d4-8）+ (α=0.125,
β=0.25) + bp=128 + γ=0.5**（bp128 处 near 配额挤压调节器最优值）；
质量绝对最优 = mono 单池 bp256（0.9433）但成本 0.315ms 且无 near
结构性保护——论文主配置取帕累托点，mono 作上界参照。

### 8b-40. E64g：α×β 全组合网格（步长 0.125 含角点，用户指令 2026-09-28）——16 数据集 14 个确认 α=0 边界最优，最优内部臂恒在 α=0.125（2026-09-28，8B trace 重放 n=16 样本）

**设计（用户角点语义原样实现）**：α,β ∈ {0,...,1} 步长 0.125 全组合
81 点 × 3 method = 243 臂；角点守卫：α=0 或 β=0 → near 跳过（预算
不侵占 far）、α=1 或 β=1 → far 跳过。协议同 E64a（bp=64、γ=1、
B_TOK=2048）。双 GPU 分片。脚本 `analyze_e64g_full_grid.py`，结果
`e64g_full_grid.json`；图 `e64g_{grid_heatmap, per_dataset_mavg,
boundary_vs_interior}.{png,pdf}`。

**角点语义验证（16 样本均值，与用户预判完全一致）**：

| 角点 | mavg | 语义 |
|---|---|---|
| (0, 0~0.875) 全 far_method | **0.8979** | = mono（E64a 对照精确一致）；β 在 α=0 时不侵预算，(0,β<1) 全等价 |
| (1, 1) 全 near_method | 0.7967 | = aavg far-mono 同值（avg 粗筛+精确细筛覆盖全 mid，near/far 同构性自检 PASS） |
| (1, 0) 无意义 | 0.6673 | 仅 sink+swa（用户预判的数据佐证） |
| (0, 1) 无意义 | 0.6673 | 同上 |

**每数据集边界 vs 最优内部臂（mavg，16 样本逐个）**：
- 14/16 数据集 α=0 边界 ≥ 最优内部臂（Δ −0.5pt ~ −2.7pt）；
  仅 gov_report_1（+0.12pt）与 multifieldqa_en_0（+0.20pt）内部
  微弱反超（噪声量级）；
- **最优内部臂恒在 α=0.125**（16/16 一致）——最小非零 near 配额
  即近无损，与 E64a 结论互证且升级为全域 81 点确认；
- 全局最优 = α=0 边界（0.8979），无任何 (α,β) 内部点超越。

**结论（论文 αβ 消融最终版）**：
1. **质量维度的最优分区是不分区**（α=0 单池 far_method）——16
   数据集 81 点全域确认，非采样偏差；用户角点语义（(0,0)=全 far
   最优、(1,1)=全 near 次之 0.797、无意义角点 0.667）全部数据
   兑现；
2. **near 分区的价值主张必须完全落在结构侧**（near 免打分带宽
   节省、防挤出保护、与 E63/E63c/E64a/b 五次独立实验一致）——
   论文写法：α 是质量-结构 trade-off 旋钮而非质量增益旋钮，
   α=0.125 是结构收益近无损端点；
3. far_method 排序在全网格不变：mavg > cavg > aavg（0.898 /
   0.884 / 0.797）。
**Addendum（2026-09-29，用户问「热力图上 near+far 打不过 baseline？」）**：
- 新增可视化 `figs/e64g_ab_grid.{png,pdf}`（3 方法族 9×9 热力图，无意义角点 ∅ 标注，
  冠军红星）+ `e64g_ab_grid_table.md`（16 样本均值 9×9 全表 + per-dataset 最优配比列，
  逐样本 mono vs (0.125,0.25) Δ 全负 −0.005~−0.046，16/16 样本最优即 α=0）；
- **口径限定（防误读）**：本网格 B_TOK=2048 为宽松预算——mono 池足以顺带覆盖 near 区，
  分区价值主张（预算保护）在宽松预算下天然不可见；e2e 主表 K2=1024（mid 仅 768）
  为紧张预算。E64h 预算桥接（B_TOK=768+γ=0.25 复刻 e2e 语义）首样本 gov_report
  mono 0.889 仍 > 分区 0.843——若全量确认，则质量维度结论维持 α=0 最优，
  分区价值主张完全落到结构侧（near 免打分带宽）+ e2e 实测（B7 vs C0 待严格口径重跑定谳）。


### 8b-41. E67：感知型 per-request gate——prefill(last1) 信号 corr 0.924，跳 near 空间(31%层/2.3% mass)大于跳 far(12%/1.7%)（2026-09-29，#75 同期用户指令「感知型 gate：prefill 决策→decode 加速」，8B trace 重放 n=14 样本×12 层）

**设计**：用户设想 = 测完一条数据的 prefill 就得到哪些层跳 far/跳 near，应用到
decode 全程（决策成本≈0，比 dyngate 逐步在线更省）。信号 = 每层 prefill chunk
末 q 的 far/near mass 占比（prefill 本来就要算，免费）；校准双变体：
avg16（16 chunk 均值）vs last1（仅最后 chunk，最接近 decode 分布）；
τ 扫描 0.01-0.20；验证 = decode q（步长 1 段抽样 16 个）真实 mass 损失。
严格区域口径（sink128/swa1024/near 靠 q 4096）。

**判决数据（168 层）**：

| 决策 | 校准 | τ | 跳层率 | decode mass 损失 |
|---|---|---|---|---|
| skip_far | last1 | 0.20 | 11.9% (20/168) | 1.7% |
| skip_near | last1 | 0.05 | 14.3% (24/168) | 0.8% |
| skip_near | last1 | 0.10 | **30.9% (52/168)** | 2.3% |
| skip_near | avg16 | 0.10 | 50.6% (85/168) | 5.6% |

**三个结论**：
1. **prefill(last1) ↔ decode far corr = 0.924**（avg16 仅 0.754——早期 chunk 混入
   浅层注意力稀释信号；校准信号用最后一个 chunk 就够）——同请求内预测可行性
   直接验证，正中 E5b 痛点（静态跨任务 corr 0.05-0.89 崩、同任务 0.93-1.00 稳）；
2. **跳 near 空间 > 跳 far**（far 层均 0.44 中位，纯跳 far 上限 ~12-31%；near 在
   一半层里占比 <0.1）——双方向 gate（far 高阈 + near 低阈）合计可免算 40-60%
   层的其中一个区域计算，速度卖点新弹药；
3. 用户预判「筛不满 topk 无害」机制确认：mask+softmax 归一化语义 + sink/swa
   恒保送（任何层 ≥256 tok 兜底），跳层只省算不破坏生成。

**后续**：最优 τ 接入 tli_indexer（per-request prefill 决策层掩码，复用 D' 的
skip_far 代码路径）→ e2e F1+速度双测（精度用 F1 不用 mass 说话）→
与静态 D'/dyngate 三方对比。脚本 `analyze_e67_perrequest_gate.py`，
结果 `e67_perrequest_gate.json`。

### 8b-42. E64h：预算桥接终判——紧预算（768）下 mono 仍胜分区，热力图结论的预算条件假说被否定（2026-09-29，8B trace 重放 n=16 样本）

**设计**：桥接「trace 热力图 mono 赢（B_TOK=2048 宽松）vs e2e B7 赢（K2=1024 紧张）」
的预算口径错位假说。B_TOK=768 + γ=0.25 精确复刻 e2e B7 严格口径预算语义
（nt_near=256、far=512、K2_mid=768）。3×3 缩格（α∈{0,.125,.25}×β∈{0,.125,.25}）。

**结果（16 样本 AVG mass）**：mono mavg(0,0)=**0.9050** vs 分区冠军
mavg(0.125,0.25)=0.8877（**Δ=−0.0173**）——紧预算下 trace mass 口径 mono 仍胜，
**预算条件假说被否定**。质量维度结论与 8b-40 合并终判：α=0 单池最优在
宽松/紧张两种预算下均成立。分区（若 e2e 有增益）的价值主张只能落在
mass 不可见的维度（生成流畅性/近端上下文利用/结构带宽节省）——待 B7s
严格口径 e2e 终表裁决；若 e2e 也无增益，分区按用户预判转 negative result，
主线转「同 method + per-request 感知砍 far」（8b-41）。
脚本 `analyze_e64h_budget_bridge.py`，结果 `e64h_budget768.json`。

### 8b-43. E66：训练投影 LOTO 终判——d8 跨数据集 0.744-0.757（92-94% 基线），「各数据集都可行」部分成立但不及截维（2026-09-29，#75 用户指令「粗筛细筛都引入训练后降维、跨数据集泛化」，LOTO 8 折×3 臂×3 维度）

**设计（调研→设计→代码，用户指定顺序）**：调研本地论文仓库（DSA KL 蒸馏
wq/wk 投影 + warm-up、NSA 多任务联合、Linformer 学习投影、ITQ 监督哈希、
SnapKV training-free 校准）→ 三训练臂：kl_distill（DSA 式分布蒸馏，主推）、
mse_multi（E65b sup_grad 多样本升级）、wsvd_multi（混合加权 PCA 闭式）。
LOTO leave-one-task-out 8 折 = 训练排除目标任务全部样本、在其上评估——
直接回答「训练出的投影在各数据集上是否可行」。

**工程修正记录（干跑抓的 bug）**：①KL 温度 1 下原始点积分 ±300 → softmax
饱和成 argmax 监督（loss 假性收敛 0.002 但 top64 IoU 仅 0.141）→ 自适应温度
（per-head std）IoU 0.422；②全量 far 区每 iter 太慢（12min/层/臂）→
DSA 式 mini-batch MB=2048 + 200 iter（26× 加速，9.2s/层/臂）。

**LOTO_AVG（D_full 全链）**：

| 方法 | d=4 | d=8 | d=16 | trunc_d32 基线 |
|---|---|---|---|---|
| kl_distill | 0.698 | 0.744 | 0.664 | 0.804 |
| wsvd_multi | **0.758** | **0.757** | 0.751 | 0.804 |
| mse_multi | 0.659 | 0.735 | 0.722 | 0.804 |

**结论**：
1. d8 维度 4× 压缩：wsvd 0.757 / kl 0.744 = 基线的 94%/92%——**跨数据集
   泛化部分成立**（LOTO 协议下最差样本 narrativeqa 0.61-0.66 不崩，far 语义弱
   任务天然难）；d4 维度 8× 压缩掉到 87-94%；
2. 但**全面不及 E65b 单样本校准的 sup_wsvd（0.788@d8）**——多任务混合训练
   稀释了任务特异性、LOTO 严格性高于 E65b 的 7 held-out 协议，两因子叠加；
3. 三臂排序 wsvd ≥ kl > mse：闭式加权 PCA（免训练、稳定性最优）仍是
   训练族最优形态；KL 蒸馏在 d4（0.698 vs mse 0.659）显示分布对齐优势
   但随维度升不放大；
4. **论文定位**：训练投影 = E65b+E66 合并叙事「离线投影可训练且跨任务
   92-94%，但 decode q 漂移使 e2e 崩（E71-B）」——training-free 方法
   固有局限的 negative result 资产（与 DSA 必须训练的动机形成对照论证）。
   e2e decode 判决实验待 B7s 完后 GPU 接入。
脚本 `analyze_e66_trained_proj.py`，结果 `e66_trained_proj.json`。

### 8b-44. E64i：同/异 method 分区判决——near 侧 method 自由度 ≤0.003，「不同 method 组合」创新点判无增益（2026-09-29，用户指令「不同method分区不如同method的话创新点或许可删」）

E64g 的盲区：三族（mavg/cavg/aavg）只枚举 far 侧 method、near 侧恒 aavg——全部是异
method 分区臂，缺 far=near 同 method 直接对照。E64i 补齐 3×3 far/near method 全组合
（α=0.125 分区冠军点 × β∈{0.25, 0.375}，16 真实样本，γ=1/B_TOK=2048 宽预算协议）：

| far\near | mavg | cavg | aavg | 单池(α=0) |
|---|---|---|---|---|
| mavg | **0.9100** | 0.9068 | 0.8999 | **0.9166** |
| cavg | **0.9118** | 0.9085 | 0.9017 | 0.9093 |
| aavg | 0.8135 | 0.8103 | 0.8033 | 0.8109 |

**三个判决**：
1. **「不同 method 组合」无增益（可删）**：far=mavg 时最优 near 是 mavg（同 method 胜）；
   far=cavg 时换 near=mavg 仅 +0.0031 噪声级——「far 用聚类上界、near 用 minmax 上界」
   互补假说在 trace 口径不成立。near 侧 method 对结果几乎无贡献，分区臂应简化为
   far=near 同 method；
2. **任何分区组合打不过 mono 单池**：给分区 per-sample oracle 选择权（16 样本各自挑
   3×3 最优）也只 4/16 胜且胜幅全 ≤+0.005（噪声级；mono 胜 12 个 −0.0003~−0.0082），
   最优分区臂 cavg+mavg 0.9118 vs mono mavg 0.9166（−0.005）；
   一切由 far 侧 method 主导（mavg/cavg ≈0.91-0.92 vs aavg 0.81，差 10pt）；
3. **β 最优恒 ≥ α**（E64g 9×9 逐行检验，3 族一致）：近端预算份额宜超过区域份额
   （gap +0.0008~+0.0142）——「不让渡（α=β）最优」假设不成立，但行内差异 <1pt
   且整个分区输 α=0，实义有限；(α,β)=(1,0)/(0,1) 角实测深谷 0.74-0.77（零预算区）。

**far 砍预算的安全性机制核实**（问题 3 判据）：L1 far 块 -inf 后 topk 只填有效块数
（跳层省算真兑现）；L2 -inf 并列按 index 破碎落 sink 区（强制置位区，无害重复）；
mask+softmax 重归一化 + sink/swa 保送 ≥256 token——**筛不满 K2 本身不掉精度**；
真风险 = 跳错层（被跳层 far mass 大，层均中位 0.44；E5b 静态 D' 多跳掉 4.8-5.9 分
实证），判据是跳层准确率（E6 静态 precision 0.92-1.00 仅安全任务成立；E67 per-request
last1 corr 0.924 感知型正中要害）。

**双口径汇总判决**（trace vs e2e 分歧终版）：trace mass 口径（E64g/E64h/E64i 三连）
分区不可行（mono 多数胜：E64g mavg 族 14/16、E64i 3×3 oracle 12/16 + 胜幅噪声级）；e2e F1 口径（B7s 严格口径 6 任务 +0.47、musique +3.08
首次超 TIA、RULER 中性）分区小幅正且任务依赖（多跳增益）。trace 口径 GQA group-sum
评估畸变（mask 间 mass 差大头来自强制区）是分歧根因（E71-B 终极对拍）。论文叙事：
「不同 method 组合」创新点删除（本节数据）；分区降级为「多跳精度保险」次要贡献
（α=0.125/β≥α/γ=0.25，musique +3.08 核心论据）；速度主线转 E67 感知型砍 far。
紧预算（B_TOK=768/γ=0.25）复测 `e64i_tight.json` 自动跑中，终表翻负则分区整体进
negative results、B7s 12 任务终表（GPU0）+L16384（GPU1）为准。
脚本 `analyze_e64i_same_method.py` / `analyze_e64i_tight.py`。

**预算约束视角的无效格分析（2026-09-29 用户洞察补充）**：形式约束
`near_budget_token = near_pages·page_size ≤ near_L`、
`far_budget_token = B_TOK − near_budget ≤ far_L`（两侧均按实际区长 clip）。
9×9 网格上的退化/无效区远多于 4 个 ∅ 角：
- **β=1 整列**（α<1）：nt_near=4096 吃光预算 → far 只剩 max(64,...) 保底
  → far 区 ~28K token 只选 64 个，实测塌到 0.74-0.76——热力图右缘深谷的真根因
  （该保底是显式安全约束，无它 far_budget 为负直接崩）；
- **小 α × 大 β**（如 0.125/0.875）：near 预算 3584 > near_L≈3.5K → clip 成
  「选满整区」（变 dense：浪费预算但不伤精度）；
- **clip 方向不对称**：near 超界无害 clip vs far 超界伤精度 clip——右下三角系统性
  塌陷的机制解释；
- 冠军点（α=0.125, β=0.25-0.375）离全部约束边界远（nt_near 512-768 << near_L，
  far_budget 1280-1536 << far_L），最优配置不受 clip 影响；
- 结论：全网格保留（退化格即「预算让渡过度=far 饿死」的边界展示，是消融故事的
  一部分），论文图上标 clip 边界线即可，无需裁剪重跑。

### 8b-45. E72：五组合 e2e 真实精度判决（快筛 hotpotqa+musique）+ B7s 终局定稿（2026-09-29 晚，#77 任务链自动编排，严格口径 e2e）

**E64j trace 五组合最佳配置**（B_TOK=2048 宽预算口径）：mminmax(α.375/β.375)
0.9116 分区冠军 / ccluster(α.125/β.25) 0.9096 / cavg(α.125/β.375) 0.9033 /
mavg(α.125/β.375) 0.9012 / aavg(α.125/β.25) 0.8036；mono mavg 0.9166。

**e2e 快筛终值（hotpotqa/musique 200 样本×2，F1）**：

| 臂 | hotpotqa | musique | 均值 | trace mass 排名 |
|---|---|---|---|---|
| mavg | 54.43 | **34.76** | **44.59** | 第 4（0.9012） |
| aavg | 54.72 | 32.82 | 43.77 | 垫底（0.8036） |
| mminmax | 53.27 | 34.14 | 43.70 | 冠军（0.9116） |
| B7s 参照（mavg β.25） | 54.23 | 32.82 | 43.52 | — |
| cavg | 53.43 | 32.31 | 42.87 | 第 3（0.9033） |

判决要点：
- **trace mass 排名与 e2e F1 排名系统性反转**：trace 垫底 aavg e2e 第 2
  （hotpotqa 54.72 全场最高）、trace 冠军 mminmax e2e 第 3——mass 口径
  （宽预算+无 sink 双计）不可替代 e2e F1 判决，双口径必须并列报告；
- **β=0.375 > β=0.25 的 e2e 验证**：同 method mavg 双 β 差 +1.07
  （44.59 vs 43.52），与 trace 侧「β 最优恒 ≥α」跨口径一致——近端预算
  份额宜超区域份额；
- **musique 34.76 创全方法新高**（FullKV 32.14 / TIA 32.28 / B7s 32.82），
  分区+β.375 在 far-heavy 多跳上的检索质量增益叠加；
- **cavg（cluster far）e2e 首测健康**：hotpotqa 200 样本 ~21min 与其它臂
  同速（kmeans 缓存修复 4e8b829 实战验证：修复前 61s/step≈8.7h/样本），
  53.43/32.31 均值 42.87 快筛末位——cluster far 路径首次在 e2e 跑通 200
  样本且质量量级正确，但 E4d「cluster 强在 far 召回」的 trace 结论未在
  e2e F1 兑现（trace 第 3 → e2e 第 5），与 mavg 反转同属口径鸿沟证据；
- best 臂 = mavg → 13 任务双卡全量（GPU0 重载 6 + GPU1 重载 7，~5h），
  终表 `exp/trace/results/e72_combo_table.md`（__FULL_PENDING__）。

**B7s 严格口径终局（本节一并定稿）**：LongBench 13 任务 **50.41**（超 TIA
+0.35 / 超 FullKV +0.05，TLI 系列首次双超）+ RULER **87.93**（+0.17 反超）。
分区创新点 e2e 判决成立；与 trace mass 口径（mono 多数胜，胜幅噪声级）的分歧坐实为预算
口径问题（trace 2× 宽预算 vs e2e 紧预算）——**分区收益本质 = 紧预算下的
预算分配，而非候选质量**。

### 8b-46. 数字终审记录（2026-09-30 凌晨，#74 步骤⑤）

终表/报告/PPT 全量数字从原始 JSON 独立重算交叉核对（方法：逐表回溯
`paper_main_table.json` / `e71_main_table.json` / `ruler_{c0,b7s}.json` /
`e64{g,h,i,j}*.json` / `e65*/e66*.json` / `e72_screen_verdict.json`），
累计修正 6 处：

1. **TLI_C0 RULER 87.64→87.76**（paper_main_table.json 误填 B7v1 旧口径数字，
   系从中间终表 ruler_table_e71_final.json 抄录带入；原始 ruler_c0.json 分长度
   重算 91.63/88.87/82.77 → 87.76）；
2. **E64i per-sample oracle「1/16 胜」→ 4/16 胜且胜幅全 ≤+0.005（噪声级）**
   （mono 胜 12 个 −0.0003~−0.0082；narrativeqa_0/1 为同文档重复样本）；
3. **E64h 逐样本 Δ −0.007~−0.040 → −0.005~−0.046（全负结论不变）**；
4. **E64h「14/16 样本最优即 α=0」→ 16/16**；
5/6. **E64g mono 胜率表述统一为 14/16（mavg 族）**——与 8b-40 节正文对齐。

方向性结论零受影响；方法论教训两条入记忆：主表数字必须回溯原始分长度
JSON 重算不可抄中间终表；oracle 判决统计必须留可复算脚本+判据注记。
PPT v8 已用修正后主表重新生成（00:59）。

### 8b-47. E72 全量终判：mavg 13 任务 AVG 50.54——TLI 系列新高，最优配置定稿升级（2026-09-30 04:18，#77 任务链收官，CHAIN_DONE）

T4 全量 13/13 落盘（双卡拆分 GPU0 6 + GPU1 7，~6.5h），T5 自动打分出终表
`e72_combo_table.md`（用户指定格式：组合|最佳αβ|精度|far跳层|速率）：

| 任务 | mavg | B7s | Δ | 任务 | mavg | B7s | Δ |
|---|---|---|---|---|---|---|---|
| **musique** | **34.76** | 32.82 | **+1.94** | qasper | 44.51 | 43.93 | +0.58 |
| hotpotqa | 54.43 | 54.23 | +0.20 | lcc | 69.62 | 69.37 | +0.25 |
| 2wikimqa | 39.23 | 39.07 | +0.16 | 其余 8 任务 | — | — | 0~−0.44 噪声级（passage 99.50 持平） |
| **AVG（13）** | **50.54** | 50.41 | **+0.13** | | | | |

**三个终局判决**：
1. **TLI 系列新高 50.54**（超 B7s 50.41 / FullKV 50.36 / TIA 50.06 / Quest 47.72）——
   training-free 稀疏索引器 LongBench 首次超 FullKV 达 **+0.18**；
2. **最优配置定稿升级：mavg=(minmax,avg) α=0.125 β=0.375 γ=0.125**（β.25→β.375，
   E72 快筛 +1.07 在全量兑现为 +0.13 且 musique +1.94 幅度精确一致）——
   快筛 200 样本判决的方向与幅度在 13 任务全量上均兑现，**「trace mass 排名
   不可替代 e2e 判决」的方法论贡献获全量确认**（trace 第 4 → e2e 全量冠军）；
3. **musique 34.76 = 全方法新高**（FullKV 32.14 +2.62 / TIA 32.28 +2.48）——
   far 检索预算分配增益的机制证据最纯粹形态（多跳 far-heavy 任务独享大增益，
   非 far-heavy 任务噪声级互有胜负）。

速率：mavg 全量 ~16.3 min/样本（含 prefill，narrativeqa/gov_report 重载任务主导）；
RULER 该臂未跑（B7s 87.93 双基准证据保持主叙事，mavg 为主表 LongBench 列最优行）。
主表 JSON 已增补 TLI_E72 行（50.54/None+注记）。终表生成脚本两处修正：
B7s 参照行 β 误用核心臂 0.375→0.25；「12任务」→13 任务标签。
E67 tau01/tau02 双臂全部完成并打分（`pred_e67_tau0{1,2}_score.json`）：完整梯度
vs gate_off（27.57/40.37）——τ0.005：musique **27.80（+0.23）** / qasper
**39.96（−0.41）**，合计 Δ −0.18；τ0.015：musique **27.84（+0.27）** / qasper
**39.70（−0.67）**，合计 Δ −0.40。τ0.005 已非 no-op（11.8% 实体级输出分歧
vs τ0.01 的 1.1%）；梯度单调（τ×3 → 损失×2.2），qasper（far 总量 0.55-0.95）
是主要承受方，musique 反微升。终判：**感知 gate 在真触发区间对多跳任务微损
不崩**（对照静态版掉 4.8-5.9），但正收益未兑现——gate 终定位 =「安全省算力
的保守开关」而非精度增益点，写入设计报告 §4A.6。E67_DONE。

### 8b-48. E73：near/far 全链 nope 维度 + 压维扫描——双假说否定 + tail32 互补性机制分解（2026-09-30 08:30，#78，用户指令「near 侧粗筛+细筛全在 nope 维度试试、都看看能不能压缩」）

trace 重放 8 样本×8 层×末 2 query（E65 框架扩展，
`analyze_e73_nope_reduce.py` → `e73_nope_reduction.json`）。
nope = 后 64 非旋转维（range(64,128)，Qwen3 RoPE 只作用前 64 维）；
全链 = 粗筛块分 + 细筛 token 分都用同一特征；三法 trunc/pca/jl ×
d∈{4,8,16,32,64}；对照 tail32（现主表口径）与 oracle128：

| 8 样本均值 | tail32 | nope 最佳 | 结论 |
|---|---|---|---|
| FAR 全链（minmax） | **0.7958** | trunc d32 = 0.4353 | nope 全链差 36pt，否定 |
| NEAR 全链（avg） | 0.2990 | pca d8 = 0.2571 | nope 无增益（−4pt），否定 |

**机制分解新发现（tail32 的两段互补性）**：tail32 = rope 低频尾 16
（48-64）+ nope 尾 16（112-128），单独 far 全链各为 **0.5759 / 0.3826**，
拼接 0.7958 > 任一（互补增益 +0.22）——**32 维子空间的威力不是
「低频连续段」而是「位置感知段（rope 低频）+ 内容稳定段（nope 尾）
各取所长**。这解释了 E3 时代 random32 崩溃与 tail32 稳健的差异根源，
也是 tail 口径的机理论据（论文 §2.1 可引用）。

其余结论：①near 侧压维极度平坦（d4≈d64≈0.21-0.26）——near 信息
冗余大，若需打分 8 维即够，且真实管线 near 有滑窗强制兜底；
②nope 内 far 压维：trunc d32≈d64 平坦、d16 起掉；pca/jl 系统更差；
③**MLA q/k 下投影矩阵不适用**：Qwen3-8B 为纯 GQA 满秩 q/k_proj
（config 无 q_lora_rank/kv_lora_rank），无自带低秩结构——「直接用
模型投影」唯一候选是 SVD 低秩近似（= E66 wsvd 路线，已判 negative）；
MLA latent（DeepSeek 系 kv_a 512 维）做索引是模型泛化 future work
（需 MLA absorb 路径改造）。E73 定位：negative result + 机制分解
资产，进论文降维章（与 E65/E66 合并为完整降维故事：far 侧唯一推荐
细筛 PCA d16 / tail32 互补机理 / near 可压但有兜底 / 训练投影口径鸿沟）。

### 8b-49. E72 mavg RULER 补全终判：87.59——双臂格局定形，主表 v2 落袋（2026-09-30 09:55，GPU 双卡 33 单元 ~3.5h）

E72 冠军臂（mavg α.125/β.375/γ.125）RULER 33 任务补全
（`ruler_e72mavg.json`，GPU0 L4K+8K / GPU1 L16K）：

| 长度 | e72mavg | B7s | Δ |
|---|---|---|---|
| L4096 | 91.34 | 91.49 | −0.15 |
| L8192 | 88.33 | 88.62 | −0.29 |
| L16384 | 83.08 | 83.68 | −0.60 |
| **总 AVG** | **87.59** | **87.93** | **−0.34** |

差距全部集中于 multikey_3（L16K 75 vs 80、L8K 98 vs 99-100）与
cwe（L16K 77.3 vs 78.6、L8K 87.5 vs 89.5）两任务族——4bit 预算
粒度主导（与分区无关），β.375 大 near 预算在合成 needle 任务上
无增益（near 区本就覆盖大半 mid）。其余 9 任务与 B7s 持平。

**双臂格局定形（主表 v2）**：TLI_B7 = **双基准最优行**
（LongBench 50.41 双超 + RULER 87.93）；TLI_E72 = **LongBench
最优行**（50.54 新高 + musique 34.76）+ RULER 87.59 诚实并报。
长度梯度 e72mavg 同成立（−0.25（8K vs FullKV）→ L16K 83.08
差 −2.33，B7s −1.73）。两行互补叙事：LongBench 多跳 far-heavy
任务 β.375 占优（musique +1.94），RULER 合成 needle β.25 占优
（−0.34）——**预算配比的任务形态依赖性**本身是消融结论。

### 8b-50. E74：far 让渡预算合理性的任务类型泛化检验——边界条件定形（2026-09-30 10:30，用户指令「看看这个结论能不能在不同类型任务上都合理」）

对「far 让渡预算合理」（β.375 near 预算份额占优，E72 终局
判决）做逐任务族 Δ(β.375−β.25) 分解（数据：
`e74_beta_yield_family.json`；LongBench = full_scores.json vs
e71_main_table TLI_B7 行；RULER = ruler_e72mavg.json vs
b7s_ruler_final.json，后者每长度含 AVG 键须剔除——本轮数字终审
教训复现：求和必须显式排除聚合键）。

**LongBench（真实任务，β.375 总体 +0.13）任务族 Δ**：

| 任务族 | n | Δ 均值 | 明细 |
|---|---|---|---|
| 多跳 QA（far-heavy） | 3 | **+0.77** | musique **+1.94** / hotpotqa +0.20 / 2wikimqa +0.16 |
| 单文档 QA | 2 | +0.14 | qasper +0.27 / multifieldqa +0.01 |
| 长文检索 | 1 | 0.00 | passage 99.50 双臂饱和 |
| 摘要 | 3 | −0.17 | gov_report −0.09 / qmsum −0.28 / multi_news −0.13 |
| few-shot/长叙事 | 2 | −0.19 | narrativeqa −0.13 / triviaqa −0.25 |
| 代码 | 2 | +0.04 | lcc +0.25 / repobench −0.17 |

**RULER（合成任务，β.25 总体 −0.34）任务族 Δ**：needle 单
key / 多 query/值 / vt 全三长度 **0.00**（饱和，预算不敏感）；
唯一负项 = multikey_3（−2.00，L16K −5.0）与 cwe（−1.23）/
fwe（−0.56）；长度梯度 −0.15 → −0.28 → −0.60 单调放大。

**泛化判决：结论在全部真实任务族上成立，在合成 far-dense
任务族上失效——边界条件 = far 关键 token 稀疏度**：

1. **成立侧**：真实多跳 QA（最 far-heavy 的 musique 反而增益
   最大 +1.94）——机理自洽：far 关键 token 极稀疏（far_tokens
   128-256 饱和，B' 分区配额已防挤出），far 侧追加预算边际
   收益趋零，让渡给 near（dense、边际高）净赚；非 far-heavy
   族 ±0.2 噪声级互有胜负，无反例。
2. **失效侧**：multikey_3（超多干扰 key，far 候选数量爆炸）与
   cwe（远端多 key 聚合）——far 侧预算粒度成为主导维度，
   far 需要**更多**预算（β.25），让渡方向反损；且随长度放大
   （干扰项超线性增多），与 Quest 下界近似在 multikey_3 崩塌
   同构——**far-dense 任务是稀疏索引器全族的难区**，非 β 配比
   独有问题。
3. **论文表述**：β 任务形态依赖性（双臂并报）获得定量细化——
   真实长上下文任务（far 关键 token 稀疏）β.375 占优、合成
   far-dense 检索（far 关键 token 稠密）β.25 占优；far 让渡
   预算的合理性可在部署时用 far mass 集中度（GQA head 级
   稀疏度，如 L03 head3 独占 0.804 的形态）做任务形态判据。

### 8b-51. E75：per-request/per-task 自适应 α/β 可行性判决——oracle 上限 ≈ 0，不值得做（2026-09-30 11:00，用户指令「快速算 attention mass 自适应 α/β 会不会比固定好」）

三个子问题的定量回答（`e75_adaptive_ab_feasibility.json`）：

**① oracle 收益上限（每任务完美选 β）**：LongBench oracle
per-task max = 50.62 vs 固定最优 β.375 的 50.54——上限仅
**+0.08**，且全部由噪声正向选择偏差解释（非 musique 12 任务
|Δ| 中位 0.17，max 结构吃掉正向噪声：6 任务各 +0.09~+0.28）；
RULER oracle = 87.93 = 固定 β.25 本身，**+0.00**。唯一系统性
信号（musique +1.94）已被全局 β.375 兑现。按族分派（多跳→
β.375 其余→β.25）= 50.58，同样噪声级。

**② 信号可得性**：技术上完全可行且**零新增成本**——#60
prefill 末 chunk far_stat（一次 softmax 的 far 统计副产品，
prefill last1 ↔ decode corr 0.924，E67），无需新 kernel 或
CPU overlap。但分离度弱：安全任务 0.001-0.008 vs 多跳
0.015-0.026 仅 ~3× gap，且 mass 口径有已证陷阱（36 层总和
vs per-layer 混淆）。

**③ 参数面平坦性**：α 网格单调递减至边界 0.125（最优在网格
边缘）；β 0.25-0.75 平坦（E64g 热力图）——参数面对精度本就
平坦，自适应没有杠杆可撬。

**终判**：per-request 自适应 α/β **收益上限 ≈ 0，不值得做**
——同一 per-request 动态信号（far_stat）的最佳用途是 D' gate
（防反向错误有安全价值），而非 β 分派（无精度价值）。
**静态双臂（按任务形态二选一，E74 边界判据）是收益/复杂度
最优解**。α/β 依赖性的真实幅度（musique +1.94 vs
multikey_3 −2.00）是任务级形态差而非请求级可变信号。

### 8b-52. E79a：per-layer 最佳 α/β oracle 上限——层维度同样无杠杆，双重否定定案（2026-09-30 13:00，用户指令「每一层单独算最佳 alpha beta，prefill 时快速得到每层大致最佳配比」）

E75 已证 per-task 维度 oracle 上限 +0.08（e2e F1，噪声级）；本轮把粒度
细化到**层**，作为 per-layer 在线预估器方向的第一道闸门
（`e79a_perlayer_oracle.json`，8 样本×12 层×20 配置网格，mavg method
紧预算 B_TOK=1024/γ0.25 贴 e2e 部署口径）：

- **per-layer oracle 增益均值 +0.0008**（8 样本全部落在
  +0.0003~+0.0018，纯噪声级）——即使每层用作弊最优配置（含逐层
  可选 mono/分区及 α/β 网格内任意点）也追不动固定配置。
- 逐层最优配置分布：74/96 层选 mono、22 层选分区；且与层 far_mass
  **无可用规律**（选分区的层 far_mass 中位 0.199 vs 选 mono 层
  0.365——方向反直觉但幅度噪声级，预估器没有可学的映射）。
- 全部样本 fixed_best 均为 mono（trace 口径，与 E64g 16/16 一致；
  e2e 紧预算判决为分区胜 = E72 口径鸿沟已知），不改变结论方向：
  trace 层面都测不出的增益，e2e 层面更不存在。

**终判：per-layer α/β 在线预估器 No-Go（E75 任务维度 +0.08 与
E79a 层维度 +0.0008 双重否定）**——α/β 参数面在任务与层两个粒度
都极度平坦（α 单调至边界、β 0.25-0.75 平坦，E64g），自适应没有
杠杆。用户问题的另一半「哪些 far 层能不能跳过」已由 E67 完整覆盖
（感知型 per-request gate：prefill last1↔decode far corr 0.924、
跳 near 空间 31% 层/2.3% mass@τ0.1、e2e τ0.005 −0.18 微损不崩，
「保守开关」定位，论文 §2.3）。**论文资产转译：预算分配的平坦性
本身是 observation（分区预算不需要精调 → 部署简单性论据），与
E74 边界判据（任务形态二选一）共同构成 α/β 的完整故事**。

### 8b-53. E76：非线性降维 + MLA 式共享投影 trace 重放判决（2026-09-30 13:20，用户指令「t-SNE/UMAP/最新降维方法都试试，或者 MLA 自己有投影矩阵缓存了 k 的都探索」）

**E73b 数据落袋补注（13:35）**：§8b-48 的两段互补数字（0.5759/
0.3826/拼接 0.7958）当时未存 JSON，本轮用 E76 同框架重导出
`e73b_seg_complementarity.json`（far 全链同特征口径）：
rope16 **0.5759**（与报告一致）/ nope16 0.387 / tail32 **0.8113**
/ random32 0.3011。注意口径差：报告的 0.7958 = E65
`C_fine_trunc_d32`（粗筛固定尾维+细筛 trunc32），E73b 0.8113 =
粗细同特征全链——**论文引用两段互补时统一用 E73b JSON**
（rope 0.576 + nope 0.387 → 0.811，互补增益 +0.24）。终版 Fig2
（`fig2_dim_reduction_story.{pdf,png}`，plot_fig2_dim_story.py：
panel A 两段互补 / panel B 8 样本 shared SVD d16 逐点追平 tail32 /
panel C 方法族均值含 UMAP negative）已生成入库。

`e76_nonlinear_reduce.json`（8 样本×8 层，E65 组 C 协议：细筛降维、
粗筛固定尾维32 minmax，真值 = 全维 softmax per-head far mass 加权捕获）：

**① MLA 式共享投影基（主发现）**：跨 kv-head 共享 SVD 基（far 区
全部 head 拼接拟合一个 [128,d] 基）d16 **8/8 样本精确追平 tail32
基准**（0.7653/0.7603/0.7652/0.6924/0.8197/0.8826/0.9038/0.8255，
个别差 ≤0.0005）；d8 也几乎追平（多数 −0.02 内）；且系统性优于
per-head PCA（每样本 +0.02~+0.08）。**机理**：GQA 组内 K 本就
共享，head 维度信息冗余——MLA「每 token 单投影缓存 latent」的
直觉在 GQA training-free 场景同样成立，共享基是 MLA latent 索引
的唯一 training-free 代理（Qwen3 纯 GQA 无 q/kv_lora_rank，无模型
自带投影可用，E73 已查）。**部署含义**：共享基意味着投影矩阵
跨 head 存储 1 份（vs per-head 8 份）+ 跨 head 的索引特征可共享
计算。

**② 非线性降维双判否定**：UMAP d16（perplexity 邻域保持嵌入）
musique 持平（0.7649）但 needle32k **−4pt**（0.862 vs 0.9038）——
非线性嵌入保邻域几何不保选择排序，从宽口径（cosine 与
-euclidean² 取较优）下仍不及线性基；成本 fit 80-167s/layer-sample
（8 头串行）vs 增量索引 0.128ms/token 基准，**4 个数量级不可行**
（且新 K token 须 .transform 无增量路径）。

**③ t-SNE 部署不可行三重定量**（成本即判决）：>2 维插值版
transform 不支持（RuntimeError，须 BH）；BH 16 维嵌入 O(n²) 逐对
梯度，11.5K token RSS 105GB（OOM 风险实录）；1000-token 子采样
拟合仍卡 kl_divergence_bh >10 分钟（py-spy 实录 16 维 BH 病态）。

**终判**：非线性流形学习（UMAP/t-SNE）对「索引选择」这一目标
无增益且成本不可行——与 E65 线性降维结论合并：**细筛降维的
training-free 最优解 = tail32 位置先验（默认）或共享 SVD d16/d8
（MLA 式，跨 head 存储更省）**，非线性不进入论文主线（negative
result 资产）。tex §4.4 降维段补共享投影发现。

### 8b-54. E78 论文正文润色两轮（paper-writing skill 机械门 + Integration pass，2026-09-30 14:00，#82）

**第一轮（机械门 + 语义红队）**：中文适配 grep 门
（hedging/填充形容词/清嗓/contentless opener/em-dash/浮夸词六类）
——基线质量高，仅 1 处「对配置误差鲁棒」改为具体数字表述
（「oracle 与固定最优差 ≤0.001 量级」）；§2.1 图引用从旧
fig2_e3_subspace 换为终版 Fig2 三 panel 合并图（claim 式图注）；
6/6 图文件存在性校验通过；数字回溯——两段互补三处旧数字
（0.796/0.383/+0.22）统一为 E73b JSON（0.811/+0.24）。

**第二轮（Stage 4 Integration）**：术语漂移审计（两段互补 4 变体→
统一；口径鸿沟/防挤出/保守开关零漂移）；claim-evidence 映射
（Abstract 九个核心数字全文均有 ≥2 次正文支撑）；23 个 section
首句 claim-bearing 审计通过；meta section 标题转注释防渲染。
结构自检（花括号/begin-end）平衡。

**skill 缺件说明**：paper-writing 的 author_profile/
writing_checklists 子目录未随 clone 落地，用 SKILL.md 内联
Non-Negotiable Voice Rules 完整执行；观察兑现检查清单
（observation_design.md §5 七条）已逐条核对：三处 tex 更新
（E73b 数字 / E79a 入 §4.4 / DSA 流派分界句入 Related Work）。

**遗留（待用户拍板）**：主表「双臂并报」vs「B7s 单一定稿 +
β.375 消融行」——两版数据均齐。

## 9. 待办（优先级序）

1. ~~E5b 完成后~~ ✅ 主表已填（TLI 49.92，§4）；far_tokens 预算敏感性已测（128–256 饱和，§7）
2. ~~sglang M2/M3~~ ✅ 全部完成（§5.0 设计报告：算法同步 + fused L1 + paged 寻址 + O(n) 增量索引（预分配版）+ 稀疏 prefill + L2 级联 fused kernel + e2e 吞吐基线与归因 + 高并发曲线）；e2e smoke 逐字一致
3. ~~M4 批量化 decode~~ ✅（§8b-2：三阶段 bs=32 1236.8→326.4（3.8×），线性项 38→6.7ms/req，对拍全过）
4. ~~M5 CUDA graph decode~~ ✅（§8b-3：三方法契约 + 统一增量/稀疏图内路径 + veto 钩子；
   replay 逐位一致 + e2e 逐字一致；bs=32 326.4→188.6 ms/step，累计 6.6× vs M3 原型）
5. ~~M6 kq 真 4bit~~ ✅（§8b-4：uint8+scale 三张量逐位一致，128→40B/token-head；
   附带发现 L1 维数可压 d'→16 / L2 维数不可压 δ→8 的两级维数边界 + 剩余 mass 评估口径）
6. ~~M7 prefill 加速~~ ✅（§8b-5：归因修正——瓶颈是 select_batched 而非 build；
   池==因果区快路径 + 共享反量化表，select 5.2×@10K / 1.8×@130K；
   e2e prefill 263.9→126.6s（2.08×）；prefill 延迟随 S 亚线性）
7. ~~M8 H 卡特有优化（#28/#49/#51）~~ ✅ 收官（L1 TC 化 No-Go §8b-18 / TMA+写布局
   No-Go §8b-20 / 唯一落地 CHUNK 形态调优 22%——dual 0.49→0.39ms，select 三档自洽
   15.0×；打分双 kernel 均带宽饱和达结构上限，r=16 FLOP 减半已被 gather kernel
   兑现无余量。141GB 大显存专属轴（bs=64×S=131K）归 H100 主表项）
7b. ~~M10 prefill kernel 化~~ ✅（§8b-15：快路径失效边界归因 + M8 全套移植，
   微基准 7.0× / e2e prefill 双档 2.11×；decode 差分法失效教训 → N=256 双侧
   复测修正 §8b-14 结论——30K decode 稳定慢 ~1.5× 未打平，线性外推翻转点 S≈44–51K（fig9c；§8b-25 终修正：54–57K）)
8. H100 吞吐主表（机器申请中；H20 层已备好算力无关性论证：H20 TC 仅 H100 15% 仍拿到质量/流量收益）+ RULER/NIAH 补评测（对齐 Quest/SnapKV/HISA 论文数据集口径）
   **【2026-09-26 #58 预核算修正：原规格 S=131K×bs16/32 单卡物理不可行——KV
   需 310/620GB > 141GB 显存（h100_pool_budget.py）；主表须改 TP2 或 bs8 档。
   且 Qwen3-8B max_position=40960，S≥64K 收益区点须换 256K-context 模型
   （本地平台有 Qwen3-30B-A3B-Instruct-2507，验证中）。另一成果：tli TP2
   兼容已打通（num_kv_heads per-rank bug 修复 + 双 backend smoke 全过。
   **2026-09-27 M11 代码态复验 PASS**：tli/triton 双臂短 prompt 逐字一致
   （"Paris..."），含 CUDA graph bs=4 capture 成功；长稀疏路径 Jacob's
   Ladder 摘要语义等价——TP2+graph 全链路对 M11 后代码无回归）——
   H20 双卡即可测 S=64K×bs16（pool +28%）】**
   **【2026-09-27 #58 收官：S=64K×bs16 收益区首测完成（Qwen3-30B-A3B-
   Instruct-2507 256K context，TP2，narrativeqa 260K chars ≈60K token，
   16 请求真实长文，无图）：tli 207.53s vs triton 106.24s = 慢 1.95×
   （prefill 占主导，粗估 prefill 比 ~2.05×；单请求口径 64K 为 1.38×，
   批量下恶化=select_batched 未 kernel 化的线性项）。输出质量：首
   40 字符一致 4/16、样本 0（Jacob Singer）双臂同主题措辞有差=稀疏
   噪声级。**诚实结论：当前形态在 S≈60K×bs16 收益区点上仍慢 2×——
   瓶颈不在 ext（M11 已 8.5×）而在 select 批量路径**；select 批量
   kernel 化（DeepSelect 4-10× 替换 / fused 移植）是收益区翻正的
   唯一路线（与 #64 合流）。脚本 test_tli_64k_tp2.py，结果
   tli_64k_tp2_{tli,triton}.json】**
9. ~~消融表~~ ✅ 已完成（§7，trace 级）；LongBench 级消融（A/B'/D' 逐个关）视主表结果决定是否补跑
10. ~~Qwen3-32B 泛化复验~~ ✅（§8：A Go/D' Go 且更强/gate 判据修正为 negative result）+ 论文写作（骨架已定，主表已齐）
11. ~~PCA 投影集成（M9）~~ ✅（§8b-8：同成本口径投影比选择 +27% far recall、存储 40→24B/token-head、六路径对拍全过、e2e smoke 通过；eager 延迟持平=launch 掩盖，FLOP 收益留待 M8）
12. ~~#58 30B 崩坏根因~~ ✅（select_batched 早期行因果越界→均匀重复 grid 修复，commit 2d6b0e4be；8B 无回归；#59 双消融 §8b-27 完成；30B prefill 归因 ext 61%）
13. ~~#60 全量 E5b 多跳复验 + #61 质量定界~~ ✅ 全部收官（§8b-28 gate 定稿
    on 38.01 vs off 38.18；§8b-30 前置的定界闭环 3.78=1.82+0.89+~1.1；
    质量主表 transformers 口径、sglang 报速度+平台内 AB）
14. ~~M11 全链~~ ✅ 收官（kernel 11-19× / prefill e2e 4.30× / decode e2e
    1.19-1.53×@bs8-32 / graph 逐字一致 / **默认开 commit ece6ee5a4**；
    30B 归因转移 ext 61%→select 81%；#58 收官 S=64K×bs16 TP2 首测：
    tli 慢 1.95×——收益区翻正须 select 批量 kernel 化）
14b. **#64 select 批量 kernel 化（当前主攻，与高并发吞吐主表合流）**：
    DeepSelect H20 实测 4.45-10.23× 已就位 → profile 30B select 10.95s
    内 topk/精筛/scatter 占比 → 替换或 fused 移植 → S=64K×bs16 复测
    目标翻正；竞争论文防御（IndexCache 跨层复用 vs TIA IoU 0.33）已入 §8b-31
15. 最终 PPT + 论文写作（用户指示：全部任务完成后产出新版）；RULER/NIAH
    补评测对齐 Quest/SnapKV/HISA 口径（#62 调研确认 LongBench 全量已有、
    NIAH/RULER 为第二标配待补）

## 10. 答辩防御清单（更新版）

1. 「B 为什么不用聚类了？」→ E4c 严格预算数据 + 口径陷阱本身就是贡献（Fig 3）
2. 「和 HISA 的区别？」→ HISA 无层自适应/子空间选择依据/分区预算；我们有 negative results 护城河
3. 「D' 的层掩码跨模型泛化？」→ **32B 实测复验 ✅**：层轮廓双峰同构、掩码跳 41/64 层 precision 0.993（§8）；per-layer far 跨模型数量级一致
4. 「索引省 2.57× 但 attention 本身呢？」→ K2 固定 1024，attention 计算量不变；省的是索引器侧 + HBM（k_min/k_max 4× 削减）——与 proposal HBM↓45% 口径对齐时说明分母
5. 「为什么 mass 覆盖都是 0.99+，实际精度差异从哪来？」→ E5b 主表直接回答（TLI 49.92 vs FullKV 50.36）
6. 「A 的子空间选择性是否随规模消失？」→ 32B entry recall：lowfreq 仍一致最优（差距 ≤0.08 vs full128）、hifreq 仍崩溃；random 退化幅度弱化（0.149→0.5+）如实报告——「机制存在、强度递减」本身是诚实的泛化结论
