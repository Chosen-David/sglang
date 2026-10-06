# 2. Method（论文正文草稿，2026-09-28 晨）

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
