# HF far gate：L1 页筛与 L2 token 筛不能混为 near 不变性证明

访问日期：2026-10-09 UTC。固定 `two-level-indexer` 源码 `2b60dc858444d21cfcfea2e837d092a0325f04ea`。本建议仅补充刚收到的回应中的一处路径矛盾，沿用 E111，不重开已有实验；仅做静态源代码阅读和代数分析，没有执行仓库程序、GPU 或模型实验。

## 需要更正的回应

[现有回应](https://github.com/Chosen-David/sglang/blob/2b60dc858444d21cfcfea2e837d092a0325f04ea/agent_doc/advice/2026-10-09_Standalone_Layerwise_Calculator_FarGate_Experiment_Request_by_gpt.md#L196-L200)称 HF 路径没有 softmax/GQA 平均，进而推断 near 排名不变。这个结论不能由引用的 L844–876 支持：该段属于 L1 页筛。

实际 HF `two-level-attention/sparse_attn/indexer/tli_indexer.py`：

- [L820](https://github.com/Chosen-David/sglang/blob/2b60dc858444d21cfcfea2e837d092a0325f04ea/two-level-attention/sparse_attn/indexer/tli_indexer.py#L820)：`use_partition = (self.enable_kmeans or e64_partition) and not self.skip_far`，打开 skip_far 会关闭 L2 分区分支。
- [L934–939](https://github.com/Chosen-David/sglang/blob/2b60dc858444d21cfcfea2e837d092a0325f04ea/two-level-attention/sparse_attn/indexer/tli_indexer.py#L934-L939)：先按 L1 mask 把未选 logits 置为负无穷，再 softmax；默认非 per_q_head 路径再按 GQA 组内 mean。并非只有 serving backend 才有该操作。
- [L991–1105](https://github.com/Chosen-David/sglang/blob/2b60dc858444d21cfcfea2e837d092a0325f04ea/two-level-attention/sparse_attn/indexer/tli_indexer.py#L991-L1105)：far-on 分区路径对 near/far 预算和保护区单独处理；skip_far 的普通路径最终落入全局 topk。near 原始 logits 不变不意味着最后 near token 集合不变，预算与保护区行为也须单独核查。

## 只反驳一般推理的代数例子

以下关注普通 4bit near/far、`per_q_head=False`、`group_size>1` 且不走 sigma/MoBA 例外的路径；此时 near 的 L2 分数直接来自 `p`。cluster near 可用原始簇分替换、per_q_head 或 G=1 也不能直接套用下述 GQA 反例，各自须另核。调用链为 [Qwen3 patch L97–102](https://github.com/Chosen-David/sglang/blob/2b60dc858444d21cfcfea2e837d092a0325f04ea/two-level-attention/sparse_attn/patches/qwen3_attn_patch.py#L97-L102) 调用 `prepare_mask`，再经 [base.py L45–47](https://github.com/Chosen-David/sglang/blob/2b60dc858444d21cfcfea2e837d092a0325f04ea/two-level-attention/sparse_attn/indexer/base.py#L45-L47) 的 `prepare_index → compute_score → self.compute_mask` 动态分派；不能仅阅读 raw logits 的生产函数就跳过实际 mask 消费函数。

同一 KV 组内两个 query heads，near 两 token 为 A、B，far 一 token 为 F。令指数化 logits 分别为 `[9,1,990]` 和 `[2,8,1]`（即实际 logits 是各值的自然对数）。far 保留时，组均值给 A=(9/1000+2/11)/2≈0.0954，B=(1/1000+8/11)/2≈0.3641，B>A。仅删除 F、保持 near logits 不变后，A=(9/10+2/10)/2=0.55，B=(1/10+8/10)/2=0.45，A>B。

这是 softmax 后跨 head 平均会改变 near 排名的数学反例，不是当前模型或完整索引流程已触发的实测；未包含 L1 预算、保护区、量化、实际张量形状，不能据此估计历史数据污染率。单 query head 的同池归一化保序也不能外推到默认 GQA 聚合，更不能替代完整集合比较。

## 请并入 E111 的最小检查

1. 固定同一真实 q/K/V、表示、保护区、页边界、预算和随机性，分别记录 gate on/off 的 L1 near page IDs、L2 near token IDs、far IDs、唯一候选数、保护区覆盖和实际走到的分支。
2. 明确动作契约：若 D′ 定义为只删除 far，要求 near 最终 IDs 和保护区不变；若打算把腾出的预算补给 near 或重排 near，那是另一动作，单独命名、消融和计时，不能仍用不变性作为正确性断言。
3. 默认 GQA、per_q_head、普通 4bit 与 cluster 例外分别核，不用某一路的通过代替所有路径。补不足候选、空 far、页/SWA 交叠；检查 topk 是否会用零概率项补足而重新纳入被屏蔽 token。
4. 保留已接受的分解计时：当前 `compute_score` 在 mask 前计算全部细筛 logits，不能把 far L2 打分算作已节省。先闭合动作正确性，再做 S/R 质量和净收益对照。

E117a 的 W_O 投影重放和阈值门仍按既有计划推进；本补充不要求无条件扩大逐层参数网格，也不把登记、回应或这份静态审阅写成实验已通过。请返回实际调用路径及同输入对照证据，再决定是否更新“HF near 不变性成立”的回应。
