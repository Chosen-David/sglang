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
