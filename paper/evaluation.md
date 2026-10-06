# 4. Evaluation（论文正文草稿，2026-09-28 晨起笔）

> 数据全部复核自 json（ruler_table_final.json / tli_e2e_variance_results.json /
> tli_64k_tp2_tli.json + baseline_bak / 报告 §8b-32 与 §#65 各节）。
> 图 = exp/figures/fig9a / fig10 / fig3 / fig4。

## 4.1 端到端质量

**RULER 官方数据集（NVIDIA 官方预生成，KVCache-Factory 镜像；3 长度 ×
11 任务 × n=100/任务；transformers 管线，与所有对比方法完全同一
monkeypatch 基座）。**

Table 1: RULER 四方法终表（string_match_all ×100）

| AVG | FullKV | TIA@1024 | TLI@1024 | Quest@1024 |
|---|---|---|---|---|
| 4K | 91.59 | 91.63 | 90.82 | 87.91 |
| 8K | 89.14 | 88.90 | 86.18 | 81.76 |
| 16K | 85.41 | 84.53 | 78.67 | 70.18 |

三个观察：

其一，**TLI 对 Quest 的领先随长度超线性扩大**（+2.91 → +4.42 →
+8.49）。16K 档 Quest 在多干扰检索任务上崩塌（multikey_2 60 /
multikey_3 21 / cwe 29.5），而 TLI 同任务保持 100 / 52 / 54.9。
根因是 Quest 的 page-min **下界**近似漏选真实高分页，在长上下文
（干扰项超线性增多）下劣化快于 TLI 的 minmax **上界**粗筛。

其二，**TLI 相对同门 TIA（全预算 token 级 topk 的上界索引）的
代价可精确定位**：AVG 差 −0.81 → −2.72 → −5.86，但差距集中于
multikey_3（16K：89 vs 52）与 cwe（78.9 vs 54.9）两任务族——
前者是块级粗筛分辨率（K1 块配额被超多干扰 key 稀释），后者是
far/near 分区挤占逐词召回预算。其余 9 个任务与 TIA 零差或反超
（fwe 16K：TLI 94.67 > TIA 92.67 > FullKV 86.67，稀疏降噪的
正向例）。

其三，**绝对值解读必须以同长度 FullKV 为基线**：聚合列举类任务
FullKV 自身随长度退化（multivalue 89.5 → 69.25 → 44.25，模型
列举能力上限），稀疏方法在同长度上的 gap 才是稀疏效应。

**LongBench 13 子集（多跳/聚合混合口径，Qwen3-8B 真实数据）**：
FullKV 50.36 / TIA 50.06 / TLI **49.92**（−0.14 vs TIA，−0.44 vs
FullKV）/ Quest 47.72。多跳 far-heavy 场景（musique/qasper）TLI 与
TIA 差 <0.3——**与 RULER 单跳口径互补**：单跳检索放大 TLI 精度
代价、多跳放大 far 分区保护收益，两表并报。

## 4.2 Kernel 级 Microbenchmark（同机 H20，统一 harness）

Table 2: 两级选择 kernel vs dense 全维打分（E1 终值 + #65 优化后）

| S | dense | TLI 两级 | 加速 |
|---|---|---|---|
| 32K | — | — | 1.46× |
| 64K | — | — | 2.77× |
| 128K | — | — | 5.09× |

（合成数据已标注；含 fused L1 3.6×、L2 级联 dual 1.63×、M11
统一稀疏 attention fused kernel 11-16×（kernel 级口径）。

**同机三方对比**（官方 kernel 原样接入统一 harness，131K 口径
cudaEvent 计时）：延迟排名 Quest（0.107ms）< DSA（0.503）< TLI
fused L1（0.787）——**TLI 延迟当前不占优，如实报告**；结构性优势
在算法侧（每 token 索引 MAC ≈258 = DSA 的 1/32、存储 ~3×↓ vs
Quest、质量 +2.2 分），131K 下三家都远离 HBM bound（排名反映实现
成熟度），M8 kernel 化为兑现路径（详表 §8b-6）。

**负结果（诚实报告）**：L1 打分 TC 化 No-Go（H20 TC 仅 H100 的
15%，ρ 数据证明 gather 带宽受限，§8b-18）；TMA 转置写布局 No-Go
（§8b-20）；near 池 DS topk No-Go（4bit 格点 tie 组巨大 → 集合
jaccard 0.72，tie 打破差异与候选分数离散度强相关，§#65）。

## 4.3 端到端速度（同机同臂双口径）

Table 3: e2e 终值（Qwen3-30B-A3B，H20×2，kernel+DS 全开）

| 档 | dense | TLI 旧(#64) | TLI 终(#65) | TLI/dense |
|---|---|---|---|---|
| 单请求 64K | 21.06s | 21.20s | **16.39s** | **1.285×** |
| 单请求 32K | 5.71s | 10.93s | 7.38s | 0.77× |
| TP2 bs16 64K | 106.24s | 167.12s | 115.28s | 0.92× |

8B 稳态（bs16 × S=30K × n=256，差分法 P1D1P2D2）：prefill
733.9 → **183.4s（4.00× 增益，#65 优化链累计）**；decode step
65.0 → **33.5ms（1.94×，反超 dense 的 40.4ms）**。

**收益区特征**：64K 档单请求 1.285× = 稀疏理论流量收益首次在
e2e 净兑现（1.005× → 1.285×）；32K 档 0.77×（短上下文选择固定
开销未被收益覆盖）；**TP2 bs16 档 0.92× 差 8%，如实归因为
launch-bound**：16 请求逐请求 Python 循环 × 36 层 × 数十小 op =
上万小 kernel 串行（GPU util 100% 但显存带宽 util 仅 10-11%），
修复方向为跨请求批量化（decode 侧同类方案 select_decode_batched
已验证 3.8×），列为 future work。

## 4.4 消融（全部 trace 重放，真实权重）

- **A 子空间**（E3）：lowfreq d'=32 mass recall 0.729 ≈ 全维
  0.732；random 0.32 / highfreq 0.137 崩溃——位置稳定子空间存在。
- **B' far/near 分区**（E4c 严格 token 预算 + per-head 加权口径）：
  far 区 TIA 4bit token 级精筛 ≈ oracle（L03 0.999）；聚类代表
  降级为消融 negative result（块 scatter-amax 最差 0.09-0.39）。
- **D' 层跳过**（E6 + E60 + #60）：离线平均轮廓 13/36 层
  precision 0.92-1.00；升级为 prefill 动态测层 gate 后三任务
  代价 AVG **−0.17**（vs 静态全局掩码 −4.8~−5.9 的跨任务不泛化）。
- **预算敏感性**：far_tokens 128-256 饱和；near 配比消融见 E63。
- **规模泛化**（Qwen3-32B，§8）：A 机制存在强度递减（如实报告）、
  D' 更强（41/64 层 precision 0.993）、far 双峰复现。

## 4.5 Profiling 方法论发现

逐 kernel microbench 与生产 e2e 存在系统性口径差 = host-GPU
流水线效应：select_batched 热路径 3 处 host 同步（.item()/.any()）
在生产形态造成 ~10 万次队列排空（合成 bench 无 GPU 队列时完全
测不出）；消除后 8B prefill −6.7% 同型负载。同步消除类优化必须
在 e2e 层验证——该教训本身为 profiling 方法论贡献（正文一段）。
