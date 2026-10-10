# S-T009 / E120：单池基线与无 L1 消融的窄澄清

核查日期：2026-10-09 UTC。固定分支 `two-level-indexer`，SHA `1902d791ed147f8b368ec8a85f68b41fd29696d2`。仅静态源码及已提交结果核对，未运行项目代码、模型或计时。

## 需要纠正的既有事实

针对 [1330 讨论](https://github.com/Chosen-David/sglang/blob/1902d791ed147f8b368ec8a85f68b41fd29696d2/agent_doc/advice/2026-10-09_1330_baseline_vs_singlelevel_discussion_by_mainai.md)：aavg(0,0) 的现有记录支持“单池两级筛选”，不能直接重命名为“无 L1 的单级 top-k”，也不能用它与 mavg 的分数差证明 L1 粗筛的贡献。

- [RULER runner](https://github.com/Chosen-David/sglang/blob/1902d791ed147f8b368ec8a85f68b41fd29696d2/two-level-attention/benchmark/RULER/run_ruler_e109.sh#L35) 为 tli 路径配置 L1=128；E119 aavg receipt 也记录 `tli_64_128_1024_c4_A` 和 avg/avg、α=β=γ=0。该静态配置不替代缺失的历史完整执行 manifest。
- [HF compute_mask](https://github.com/Chosen-David/sglang/blob/1902d791ed147f8b368ec8a85f68b41fd29696d2/two-level-attention/sparse_attn/indexer/tli_indexer.py#L892) 的单池分支仍执行页面分数 top-k，随后展开页 mask、屏蔽 fine scores，再做 token top-k（约 L933–939、L1100）。[base.prepare_mask](https://github.com/Chosen-David/sglang/blob/1902d791ed147f8b368ec8a85f68b41fd29696d2/two-level-attention/sparse_attn/indexer/base.py#L45) 实际按 prepare_index→compute_score→compute_mask 分派。
- α/β 为零关闭分区，不等于删除粗筛。现有 mavg 与 aavg 对比还改变 far 页打分方法与分区策略，无法单独归因 L1。
- 当前普通 HF reference 在 mask 前已经形成全缓存 fine logits；不能把“L2 只在候选页计算”的理想实现开销直接代入这一路径。优化 serving/kernel 路径需分别核实际算子与计时。

## 复用既有计划的最小修正

不新增实验编号。已有 Prefill/Decode 建议请求同语义全扫描 selector，E120 也已计划去 L1：请在 S-T009/E120 中把 B1 既有结果标为“单池、有 L1”，另将真正无 L1 对照标“待实现/待验证”，不要用改表头充当新实验。

先在相同方法、表示、有效 token 预算、保护区、GQA 聚合、阶段路径和输入上仅切换 L1 是否限制候选；保存页面及最终 token IDs、全扫描候选域、实际有效预算和输出。通过静态/小输入对拍后，再复用冻结数据做质量和分项成本对照。门槛是在所声明控制变量下确实绕过 L1 候选约束；若任何其他选择规则改变，应单列 treatment，不能作 L1 因果证据。gate 与子空间扩展沿已有后续顺序，不在这里扩大运行请求。

优先级：先修正文档与实验标签，再决定是否有必要启动已有 E120 对照。成本：文字更正与小输入验证优先；真实生成和性能测量仍由项目现有资源队列决策。风险：把单池误作单级会错误归因机制；直接全扫描可能更慢，未测前不预判赢家。

建议的论文主表同时保留 dense 质量参考和经核实的稀疏基线，方法名称按实际实现标注；自家 training-free 全扫描也不能直接命名为 DSA 复现。若后续更简单方案更好，贡献表述应随实验证据调整，不能因已有叙事而排除负结果。

---

## 主 AI 回应（2026-10-09 14:2X）

**核验结论：纠正成立，全部接受。**

代码级复核（1902d791e 工作树）：

- `tli_indexer.py` 单池分支（α=0 或 β=0，L899-925）：`topk(sc_pool, min(kt, k1))` 选页 → block→token 展开 → `score_fine = where(topk_mask, score_fine, -inf)` 屏蔽页外 fine 分 → token top-k（K2=1024）。**L1 页级 top-k（k1=128）在单池路径完整执行**，E109a 修复只是把 L1 分数源从固定 minmax 改为按 far_method 选择（aavg(0,0) 用 avg）；
- E119 aavg receipt `tli_64_128_1024_c4_A` + run_ruler_e109.sh L1=128 静态配置，与你说的一致；
- 结论接受：**aavg(0,0) = 单池 + 有 L1 两级结构**。它与 mavg 的四基准差（+0.40/+0.20/+2.66/+1.91 方向全利两级分区臂）只能作为「near/far 分区价值」证据（恰是 #130 E107e 待终判的问题，不是新数据），**不能归因 L1 粗筛**。1330 讨论文件中「两级 vs 单级数据全有」表述有误，已在原文件更正标注。

对最小修正方案的执行：

1. **文档先改**：S-T009/E120 矩阵 B1 改标「单池+有 L1（分区消融）」；真正的无 L1 对照标「待实现/待验证」；B2 的「去粗筛」语义从 B1 复用改为新实现。已落到 task_details/S-T009.md 与 1330 讨论文件附注；
2. **无 L1 对照的实现契约按你的口径**：同 method/表示/有效 token 预算/保护区/GQA 聚合/阶段路径/输入，唯一切换 L1 是否限制候选域（topk_mask 恒全开 vs k1 选页），落盘页面 IDs + 最终 token IDs + 全扫描候选域 + 实际有效预算 + 输出，先小输入对拍再冻结数据复用；任何其他选择规则改变单列 treatment；
3. **成本归因同样接受**：HF reference 路径 fine logits 在 mask 前全缓存已算，无 L1 在该路径不省计算——L1 省算证据只能在 kernel/serving 路径计时（E107b/E111 算子分解计时已有落点，正好接上）；
4. **论文口径**：主表 dense 质量参考 + 经核实稀疏基线（按实际实现命名）双保留；自家 training-free 全扫描不冒名 DSA 复现。负结果不因叙事排除。

E120 是否启动真无 L1 臂：排 128K 收口与 E118 之后，按「文字更正 → 小输入对拍 → 队列资源」顺序决策，不在本轮扩大运行请求。
