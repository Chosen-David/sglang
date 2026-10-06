# TLI: Two-Level Training-Free Sparse Attention Indexer
# ——块级上界粗筛 + far/near 分区精筛（论文完整草稿 v1，2026-09-28 晨合并）

> 底稿链：PAPER_OUTLINE.md（骨架）→ method/kernels/evaluation/intro_related
> 四章草稿（数字均从 json 复核）。合并稿供 LaTeX 转写与用户审阅修订。

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


---

> 事实锚点：报告 §5.0 设计 + E1-E8 + E4c/E63/#60/#65 各节 + fig7_architecture。
> 符号约定：S 序列长、Hkv KV 头数、d'=32 压缩子空间维、K1 块配额、K2 token 预算（1024）。

## 2.0 Overview

TLI（Two-Level Indexer）是 training-free / weight-free 的稀疏注意力
索引：**块级上界粗筛（L1）→ far/near 分区 token 级精筛（L2）**两级
级联，配合离线层跳过（D'），在每个 decode/prefill 步骤为每个 query
head 选出 K2 个 KV token。设计目标不是单点最优精度，而是**可定位的
精度代价换取结构性速度分层**——同精度梯队（vs 全预算 topk 上界索引
TIA）下实现 kernel 级 1.3-5.1× 与收益区 e2e 加速。

## 2.1 L1：块级 minmax 上界粗筛（子空间）

将 KV 缓存按块组织（block=64）。对每块每 kv-head 维护低频尾维子空间
（d'=32，Qwen3 rotate_half 的 position-stable 低频维，E3 实证：
lowfreq mass recall 0.729 ≈ 全维 0.732，random 0.32 / highfreq 0.137
崩溃）上的 min/max 上界，4bit 量化存储（uint8 + scale 三张量，128→
40B/token-head，M6）。

Query 打分用块上界的**乐观侧**：与 min 的最大距离或与 max 的最小
距离构成该块对任意 token 的分数上界——**上界保证无漏选**（对比
Quest 的 page-min 下界近似：长上下文干扰项超线性增多时下界漏选
真实高分页，RULER 16K Quest multikey_3 21 分的根因）。取 top-K1
块进入 L2（K1=128）。

**压缩表示的量-界权衡**（E4c 严格 token 预算 + per-head 加权口径）：
4bit 量化分数并列（tie）是主要质量风险——L1 层面可容忍（多选由
L2 吸收），L2 层面不可（near 池 DS No-Go：jaccard 0.72）。

## 2.2 L2：far/near 分区 token 级精筛（B'）

L2 把 K2 预算划分为两个池：

- **near 池**（滑窗近端）：最近 sw_lo 个 token 的强制覆盖——
  局部性强（H1：dense top-1024 mass 的 near 层占 0.16-0.29），
  因果滑窗语义下天然精确，无需打分。
- **far 池**（远端候选）：L1 入选块的 token 级 4bit 精筛（TIA
  同款 token 级量化打分，far 区 ≈ oracle，E4c L03 0.999）。

**B' 的价值 = 防挤出**：无分区时 far 候选与近端高分 token 在同一
池竞争，far-heavy 层（GQA head 级 far mass 极不均，L03 head3 独占
0.804）的远端关键 token 会被近端挤出。分区给 far 独立配额
（far_tokens 128-256 即饱和，E63 配比消融），近端配额扣减带
near_floor 保护防边界回退。

**聚类代表的教训（negative result）**：开题的创新点 B（kmeans 远端
聚类代表）在严格 token 预算下无一致优势（块 scatter-amax 0.09-0.39
最差）且聚类中心非上界有漏选理论空洞——B 的 lazy 维护假设仍成立
（E7 增量 assign recall 衰减 ≤0.03）但表示本身降级为消融项，
B 重定位为分区预算设计（B'）。

## 2.3 D'：离线层跳过 + 动态 gate

离线校准：trace 分析显示 far 质量呈层间轮廓（部分层 far mass 近零
——这些层的注意力几乎全部由 sink+near 覆盖）。离线平均轮廓可跳
13/36 层（precision 0.92-1.00），但**全局静态掩码跨任务不泛化**
（musique/qasper/multifieldqa 掉 4.8-5.9，E5b；per-task 重校准也
失败——长度失配 + GQA head 级 far 被平均口径低估）。

升级为 **prefill 动态测层 gate**（#60）：末 chunk 一次 softmax
far 统计 → far_stat 双峰（musique 18 层 <0.01 vs 其余 0.017-0.26）
→ 阈值切自然间隙 → decode select 幂等置位 skip_far。三任务
（静态版失败集）代价 AVG **−0.17**，开销分摊 decode 期可忽略。
已知限制：stat 为 per-layer 单值，混合 batch 跨请求污染（生产
语义应存 per-row，TODO）。

## 2.4 系统集成（sglang 生产形态）

- **paged 寻址**：选择输出逻辑位置 → req_to_token 间接 → KV pool
  槽位，与生产 KV 布局零拷贝对接。
- **O(n) 精确增量索引**：每 token 增量更新块界（增量==全量重建
  逐位一致，0.128ms 与 S 无关；预分配几何扩容）。
- **共享 index pool**：per-layer 预分配 [R, cap]（kq/kmin/kmax），
  decode 批量 select 全 eager ~15 launch 与 bs 无关。
- **CUDA graph**：capture 区域零 host 同步 + 统一增量图内路径，
  replay 逐位一致。
- **早期行修复**（#58）：首 chunk t<K2 行均匀重复 grid——非因果
  泄漏的根因修复（30B 生成崩坏的教训，8B 同 bug 耐受未崩，如实
  报告）。


---

> 事实锚点：E1/E8-2/§8b-12/13/18/19/20/29 + #65 各节 + fig10。
> 口径铁律：kernel 级 microbench 与 e2e 双层都测都诚实报告。

## 3.0 设计约束（H20 形态）

H20 的 Tensor Core 吞吐仅为 H100 的 ~15%，而 HBM 带宽同级
（4TB/s）——**选择/打分类 kernel 的第一性约束是带宽而非算力**。
ρ 实验数据（带宽利用率 vs 计算强度）显示打分 GEMV 在 d'=32 形态
下 deep memory-bound，TC 化收益无法兑现（§8b-18 No-Go 如实
报告）。因此 kernel 设计主线 = **消除中间物化 + 合并 launch**，
而非算子替换。

## 3.1 Fused L1 块打分（E8-2）

朴素实现：逐块 gather K 子空间 → 反量化 → einsum → 写回，每块
5-6 个小 kernel。Fused 版：单 Triton kernel 内完成 gather + 4bit
反量化（共享 scale 表）+ GEMV 打分 + 块级规约，输出直接为块分数。
**3.6×（32K 形态）/ 1.6×（131K）**，块 id 对拍与 eager 逐位一致。

## 3.2 L2 级联 Dual Kernel（M8-KernelC）

L2 需同时对 far/near 双池打分取 topk。Dual kernel 单 launch 完成
分区 topk：候选块展开为 token 粒度打分，**池边界即因果边界**
（far 池上界 ≤ t+1-near_len、near 池 < SW_LO——替代 eager 的
fine 矩阵因果 mask，L1 垃圾块天然落池外）。

关键工程发现（负结果资产）：
- **纯 kernel 版 No-Go**：4bit 分数并列过选（130-256/head 无人
  兜底）+ 120 轮二分串行反慢 4.5×——**L1 能容忍多选（L2 吸收）、
  L2 不能**，混合形态（Triton pass1 打分 + torch.topk + 精确
  复制）为终态。
- **bf16+pad 直出**（#65）：OUT_BF16 constexpr 分支使 far/near
  分数直接以 bf16 输出到 pad 对齐宽度（pad 列 -inf），消灭 fp32
  [n,Hkv,Tc]×2 物化 + cast/pad 复制链（profiling 中 elementwise
  16.3% → 0）。
- **CHUNK 形态调优**：64/num_warps=4 最优（12.1 vs 14.9/17.7ms，
  与 M8-2b compact 路径结论一致）。
- **TMA 转置写布局 No-Go**（§8b-20）：写侧假设证伪，实测反慢。

## 3.3 M11：统一稀疏 Attention Fused Kernel

选择完成后，稀疏 attention 本体（gather 候选 K/V + softmax +
PV）的 eager 路径需 [n, K2, D] fp32 双物化。M11 单 Triton kernel
完成 gather + online softmax + 输出直写 bf16，**kernel 级
11-16×**。工程陷阱记录：q_raw 为父张量切片 view 时行 stride ≠
H*D（实测 6144 ≠ 4096），kernel 寻址假设行连续 → 必须
contiguous 化（一次 bf16 拷贝仍远快于 eager fp32 双物化），
否则 row≥1 全部读错位置（e2e 乱文根因）。

## 3.4 DS Topk（DeepSelect 分数路径）

4bit 打分输出可用数值判别结构（DeepSelect 类）替代 torch.topk：
L1 与 far 池接入（**decode 侧净效果 bs16 +10%**），near 池保留
torch——**No-Go 教训**：near 池压缩表 ~2048 有限项挤在 4bit 格点
极少数分数值（tie 组巨大）→ 集合 jaccard 0.72。tie 打破差异与
候选分数离散度强相关，**接 DS 前必须按池预判**。

CUDA graph 约束：DS host 端 pad/end 分配不可进 capture 区域，
graph 捕获时自动回退 torch.topk（`is_current_stream_capturing`
门控）。

## 3.5 Select 调度链的消除（#65）

两个隐藏瓶颈的发现与修复（microbench 测不出、e2e 才显形的
口径盲区案例）：

- **早期行 Python 循环**（#58 修复引入）：首 chunk ~1024 行 ×
  arange/cat/index_put ≈ 5K launch + 3K 次 .item() 同步 →
  chunk0 65ms。向量化 `where(j<cut, j//reps, j-cut)` 单 op 构造
  uniform grid，4×。
- **3 处 host 同步**：fast_path 判定 `.all().item()` / empty 段
  `.any()` / early 段 `.any()` × 生产形态 ≈ 10 万次队列排空。
  修复 = kernel 路径可用时直接跳过判定 + `t_min_hint` host
  参数（forward_extend 的 prefix 是 Python int，可判「无 empty
  行 / 无 early 行」跳过同步）。8B e2e prefill −6.7%。

## 3.6 同机三方对比（统一 harness）

官方 kernel 原样接入统一计时框架（cudaEvent + 131K 口径）：
Quest 官方 decode_select_k.cuh（编译接入）、DSA FlashMLA
fp8_index（原样 import）。延迟排名 Quest < DSA < TLI（TLI 延迟
当前不占优，如实报告）——TLI 的三方优势在算法结构（索引 MAC
1/32 of DSA、存储 3×↓ vs Quest、质量 +2.2 分）。**对比铁律：
kernel 原样、harness 统一**（§8b-6 详表）。


---

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

Table 2: 两级选择 kernel 相对 dense 全维打分加速比（E1，合成数据
已标注；trace 重放口径）

| S | 32K | 64K | 128K |
|---|---|---|---|
| 加速比 | 1.46× | 2.77× | 5.09× |

（组成：fused L1 3.6×、L2 级联 dual 1.63×、M11 统一稀疏
attention fused kernel 11-16×，kernel 级口径。）

Table 2b: 同机 indexer 三方对比（官方 kernel 原样接入统一
harness，ms/层，decode 单 token 全链路，131K 口径 cudaEvent）

| indexer | S=10K | S=131K | 索引存储/token | training-free | LongBench-13 |
|---|---|---|---|---|---|
| Quest 官方（page GEMV + raft radix） | 0.066 | 0.107 | ~1KB | ✓ | 47.72 |
| DSA 官方（tilelang fp8_index + topk2048） | 0.476 | 0.503 | ~132B | ✗（1000 步 warm-up） | — |
| TLI eager（两级全链路） | 0.736 | 0.853 | ~336B | ✓ | 49.92 |
| TLI fused L1 | 0.604 | 0.787 | 同上 | ✓ | 49.92 |

三方诚实结论：Quest 索引最快（µs 级 GEMV+radix）但 1KB/token
存储（3× TLI）且 LongBench −2.2 分；DSA 与 S 无关（launch 主导）
但需训练；**TLI 延迟当前不占优（0.6-0.83ms）如实报告**——结构性
优势在算法侧：每 token 索引 MAC ≈258 = DSA 的 **1/32**、存储
~3×↓ vs Quest、D' 跳层摊销、质量 +2.2；131K 下三家都远离 HBM
bound，排名反映实现成熟度，M8 kernel 化为兑现路径。

**同机三方对比**（官方 kernel 原样接入统一 harness，131K 口径
cudaEvent 计时）：详表见 Table 2b 及其诚实结论——延迟排名 Quest
< DSA < TLI（实现成熟度），TLI 的优势在算法结构（MAC 1/32、
存储 3×↓、质量 +2.2）。

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


---

## 附录 A：实验编号总索引

全部实验 E1-E8/E59/E60/E63 + 系统里程碑 M1-M11 + 任务 #28-#67 的
脚本→结论映射见 `TWO_LEVEL_PAPER_REPORT.md` §5 与 §8b-1~32
（每项含 commit 号、对拍口径、负结果）。

## 附录 B：可复现性

- 环境：H20-3e ×2（141GB），torch 2.8.0+cu128，sglang
  two-level-indexer 分支（6+ commits）
- 数据：RULER 官方预生成（KVCache-Factory 镜像）/ LongBench v1 全量
  / Qwen3-8B、30B-A3B、32B 真实权重
- 打分：string_match_all（与官方一致）；e2e 差分法 P1D1P2D2 交替
