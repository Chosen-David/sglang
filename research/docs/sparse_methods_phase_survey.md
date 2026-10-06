# 稀疏注意力方法「索引构建 vs 稀疏检索」阶段划分调研（#134 / E108c，2026-10-06）

> 用户疑问：ClusterKV 的 kmeans 簇代表用在 decode 而非 prefill——其他论文是否也都主要用在 decode？
> 证据等级：[原文]=本地论文 PDF 提取；[代码]=本地代码逐行；[文档]=已有调研；[记忆]=论文记忆（无本地原文，标置信度）

## 对照表

| 方法 | ① 索引/统计构建 | ② 稀疏检索（top-k） | ③ prefill attention | 证据 |
|---|---|---|---|---|
| Quest | prefill 每 chunk 增量建 page min/max | **仅 decode**（QAPS 页选择） | **原论文 prefill 全量 dense**；sglang 复现版 prefill 也稀疏（公平对比增强，非原版口径） | [代码] quest_algorithm.py；[文档] quest_vs_psi_integration.md |
| MoBA | 无独立索引（chunk mean 每次 forward 现算） | **prefill+decode 都做** | 稀疏；**论文明说「MoBA is used for prefill only, full attention during generation」**（prefill 1M 加速 6.5×，主打 prefill！） | [原文] MoBA_Tech_Report.pdf；[代码] moba_efficient.py L332-371 |
| DSA (V3.2) | indexer k cache prefill 逐 token 写入（FP8 单 GEMM） | **prefill+decode 都做**（_get_topk_ragged / _get_topk_paged 双路径） | 稀疏；短序列 <2048 退 masked MHA | [原文] V3.2 论文；[代码] 官方 inference/model.py L582-602；sglang dsa_indexer.py L406-413 |
| HISA | 无独立构建（drop-in 替换 DSA indexer） | prefill+decode 都做 | 稀疏（IoU>99% 同 pattern）；**立题动机=降 DSA prefill 期 O(L²) indexer 瓶颈** | [原文] HISA_COLM_2026 README |
| ClusterKV | **prefill 期间 kmeans 聚类构建** | **decode 做簇选择**（簇代表打分→取回簇内原始 KV 精确 attention） | [记忆] prefill dense+聚类构建 | [用户口径+记忆]（本地无论文 PDF）；[代码] tli/indexer.py L259 注释 |
| SnapKV | prefill 一次性观察+压缩 | 无检索（decode 用压缩 KV） | 观察后稀疏 | [代码] snapkv_utils.py update_kv |
| PyramidKV | prefill 一次性压缩 | 无检索 | 同 SnapKV 族 | [记忆] |
| SparQ | prefill 收集每通道幅度统计 | decode 两级检索 | prefill 基本全量 | [记忆] |
| NSA | 无独立构建（三分支端到端训练） | prefill+decode | 稀疏 | [记忆]+[文档] |
| InfLLM / Landmark | prefill 建块代表 / landmark token | decode 块选择 | 块选择机制训练推理一致 | [记忆] |

## 结论（回答用户疑问）

**领域分裂为两族，无统一惯例：**

1. **Decode 加速族**（training-free KV 压缩）：Quest、ClusterKV、SnapKV、PyramidKV、SparQ——**索引/统计/聚类在 prefill 构建，top-k 检索只在 decode，prefill attention 全量 dense**。用户对 ClusterKV 的口径与此一致；我们的精度主力 baseline（Quest/ClusterKV）都属此族。
2. **Indexer 族**（稀疏原生）：DSA、NSA、MoBA、HISA——**prefill+decode 都做稀疏检索**。MoBA 甚至主打 prefill（论文原句 prefill only）；HISA 立题就是降 DSA prefill indexer 瓶颈——**prefill 期 indexer 已是公认研究方向，不是禁区**。

**ClusterKV 用 decode 是族群设计选择，非「领域默认」。**

## 对论文的启示

- **PSI prefill 做两级稀疏选择不偏离惯例**——属 indexer 族（DSA/HISA/MoBA 同族）固有属性。**切忌写「prefill dense 是领域默认」**——MoBA/DSA 原文直接反例。
- 审稿风险在**速度口径**而非惯例冲突：跨族对比 Quest（decode 加速族，prefill 只做轻量统计）时 prefill 136.0 vs 54.4s 如实报告；建议表述：「PSI 采用 indexer 族路线，prefill 稀疏检索是路线固有成本；同族（DSA）对比下 PSI 两级结构是降本（HISA 同理）；跨族（Quest）差距如实报告 + F1 增量化是既定修复（已落地 3add02624）」。
- **simgreedy 用法与 ClusterKV 惯例完全一致**：聚类资产构建（prefill/decode 持续维护）+ 簇代表打分检索（decode 主要场景）——不偏离。
