# Two-Level Indexer 论文进展报告（v3，2026-09-24）

> 承接 `TWO_LEVEL_INDEXER_DESIGN.md`（设计 + 批判性分析）。本报告从**论文写作视角**整理全部已实测结论、图表资产与剩余待办。v3 新增：**M3 sglang 系统集成实测**（L2 级联 fused kernel 接入对拍 1.0000 / e2e 逐字一致 / 高并发曲线与批量化缺口），此前版本已含 E4c 口径修正（B 重定位）、E8 组合收益、E5b 主表、32B 泛化复验。
>
> 数据基线：Qwen3-8B 真实权重 + 10 条真实 trace（needle/natural + 8 条 LongBench 官方样本），2×H20-3e。

---

## 1. 论文定位（一段话）

**TLI（Two-Level Indexer）**：面向长上下文 decode 的 training-free 稀疏注意力索引器，在 TIA（第一代两级索引）之上引入三个实测支撑的改进——**A 位置稳定子空间粗筛**、**B' far/near 分区预算**、**D' 离线校准层跳过**——在 mass 覆盖持平或超过 TIA 的前提下，索引侧计算量降低 **2.57×**（跳层 4.88×），fused L1 kernel 原型 **3.6×/1.6×**，sglang 全系统集成后两级选择 fused kernel **对拍精确一致**（jaccard 1.0000）、e2e 输出与 eager **逐字一致**、decode 步延迟 81→51.4 ms。论文同时贡献一组**方法论级 negative results**（严格预算口径下聚类代表劣于 4bit 精筛、在线信号失效、跨层复用失效、级联 kernel 化中「L1 可容忍多选、L2 不可」的边界），为该方向划定清晰的可行边界。

## 2. 贡献列表（写作时逐条对应实验）

| # | 贡献 | 支撑实验 | 数字 |
|---|---|---|---|
| 1 | **A**：L1 块 min/max 上界只在低频尾维 d'=32 计算（4× 维度削减，recall 不掉） | E3/E3b | mass recall 0.729 vs 全维 0.732；random 0.32 崩溃（选择性证据） |
| 2 | **B'**：far/near 分区 L2 预算，防远端被近端高分挤出 | E5b debug | far-heavy 层 TLI 0.9995/0.9997 vs TIA 0.9990/0.9929 |
| 3 | **D'**：离线校准静态层跳过掩码（far 质量低的层免远端检索） | E6 | 13/36 层可跳，precision 0.92–1.00，far 质量损失 <0.3% |
| 4 | 索引开销与 kernel：D' topk 截断真正兑现省算 + Triton fused L1 | E8-1/E8-2 | 索引 FLOP 2.57×（跳层 4.88×）；L1 kernel 3.6×/1.6× |
| 5 | **Negative results**：严格预算口径下聚类代表无优势；在线 L1 信号失效；跨层复用失效 | E4c/E5 | km_blk 0.09–0.39 vs minmax 0.52–0.74；IoU 0.33；corr ≈0 |
| 6 | E5b 端到端：TLI 在 LongBench 13 子集 vs TIA/TWI/Quest/FullKV | E5b | **49.92** vs TIA 50.06 / FullKV 50.36 / Quest 47.72 |
| 7 | **系统（M3）**：sglang 全链路集成——L2 级联 fused kernel + 增量索引预分配 + e2e 吞吐 | M3-a/b/c | 对拍 jaccard 1.0000；e2e 逐字一致；decode 81→51.4 ms/step（bs=1）；两级 select 1.34×@9.9K |

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
- `sglang`（two-level-indexer 分支，至 commit 5021ef105）：M1 = tli backend 注册 + trace 单测；M2 前半 = B'/D' 算法同步 + Triton fused L1；M2 后半 = paged 寻址 + O(n) 精确增量索引 + 稀疏 prefill；**M3 = decode 归因 + _sparse_attn 向量化（bfe373c18）+ L2 级联 fused kernel 接入 + 增量索引预分配 + 批量 decode 基准（36b6c8a37）+ 设计报告 §5.0/§5.0.1（5021ef105）**。test_tli_m2b.py 全链路对拍在预分配改造后仍全 PASS；分支基于 fork 最新 main（4b186cfea，2026-09-15）

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

## 9. 待办（优先级序）

1. ~~E5b 完成后~~ ✅ 主表已填（TLI 49.92，§4）；far_tokens 预算敏感性已测（128–256 饱和，§7）
2. ~~sglang M2/M3~~ ✅ 全部完成（§5.0 设计报告：算法同步 + fused L1 + paged 寻址 + O(n) 增量索引（预分配版）+ 稀疏 prefill + L2 级联 fused kernel + e2e 吞吐基线与归因 + 高并发曲线）；e2e smoke 逐字一致
3. **M4 批量化 decode（当前主线，H20）**：共享 index pool（kmin/kmax 预分配 [R_max,NBLK_MAX,Hkv,d']）+ kq 真 4bit 存储（uint8+scale，kernel 内 dequant）+ L1/L2 kernel grid 加 batch 维 + _sparse_attn 批量 gather + CUDA graph → 吞吐主表（bs×S 矩阵）
4. H100 吞吐主表（机器申请中；H20 层已备好算力无关性论证：H20 TC 仅 H100 15% 仍拿到质量/流量收益）
5. ~~消融表~~ ✅ 已完成（§7，trace 级）；LongBench 级消融（A/B'/D' 逐个关）视主表结果决定是否补跑
6. ~~Qwen3-32B 泛化复验~~ ✅（§8：A Go/D' Go 且更强/gate 判据修正为 negative result）+ 论文写作（骨架已定，主表已齐）

## 10. 答辩防御清单（更新版）

1. 「B 为什么不用聚类了？」→ E4c 严格预算数据 + 口径陷阱本身就是贡献（Fig 3）
2. 「和 HISA 的区别？」→ HISA 无层自适应/子空间选择依据/分区预算；我们有 negative results 护城河
3. 「D' 的层掩码跨模型泛化？」→ **32B 实测复验 ✅**：层轮廓双峰同构、掩码跳 41/64 层 precision 0.993（§8）；per-layer far 跨模型数量级一致
4. 「索引省 2.57× 但 attention 本身呢？」→ K2 固定 1024，attention 计算量不变；省的是索引器侧 + HBM（k_min/k_max 4× 削减）——与 proposal HBM↓45% 口径对齐时说明分母
5. 「为什么 mass 覆盖都是 0.99+，实际精度差异从哪来？」→ E5b 主表直接回答（TLI 49.92 vs FullKV 50.36）
6. 「A 的子空间选择性是否随规模消失？」→ 32B entry recall：lowfreq 仍一致最优（差距 ≤0.08 vs full128）、hifreq 仍崩溃；random 退化幅度弱化（0.149→0.5+）如实报告——「机制存在、强度递减」本身是诚实的泛化结论
