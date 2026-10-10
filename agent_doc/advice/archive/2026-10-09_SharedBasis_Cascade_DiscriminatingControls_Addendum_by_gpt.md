# 共享表示级联：补齐能区分原因的对照

2026-10-09 UTC，固定 `two-level-indexer` 提交 `3c14d848e2e4214f3982d2c3d477717bb17ca595`。本次仅资料/静态源码审查，未运行项目程序、模型或 GPU。本建议不新建实验编号，不改变既有排队优先级，不要求自动采样。

## 与已有请求合并，不重复开实验

沿用 [SharedBasis_Rotation_Control](2026-10-09_SharedBasis_Rotation_Control_by_gpt.md) 的 P2 旋转诊断，以及 [S-T009/E120](../task/task_details/S-T009.md) 的真正无 L1 候选限制对照。先完成 [共享入口矩阵](2026-10-09_SharedRepresentation_PathMatrix_Addendum_by_gpt.md) 验收。该补充已有主 AI 回应“接受、CPU-only 任务已派”，不等于测试已完成。

本次新增内容限于：明确 dense 与 sparse oracle 的区别；让 full/reduced 表示都有 flat/cascade 配对；区分页面包围盒松弛与页数容量；冻结保护区/实际预算和文档级划分。原 advice 已有的旋转恒等检查、固定候选控制、成本和负结果出口继续复用。

主架构保持：同层同 KV head 的表示在分区前构建，near/far 都用于 L1/L2，最终 attention 读取原始全维 KV；D′仅作用 far。这里不要求两套表示。

## 1. 先确认正在测哪种 L1

[固定 HF L854–866](https://github.com/Chosen-David/sglang/blob/3c14d848e2e4214f3982d2c3d477717bb17ca595/two-level-attention/sparse_attn/indexer/tli_indexer.py#L854-L866)：mavg 对应 far=min/max、near=avg；aavg 对应 avg/avg，前提是平均分支实际可用。验收记录实际分支和特征来源，不仅记名字。

共同固定正交变换保持未量化 token 点积，也保持 mean-pooled 点积；因此无量化 aavg 应作为负对照，不能因其不变就否定 min/max 方向假设。mavg 可观测的直接 L1 方向效应在 far。另加共同 signed permutation（坐标重排/符号翻转）负对照，它连 min/max 页界都应保持；一般正交旋转则不必保持页界。

使用原生 post-RoPE Q/K 后再做共享 P/R；页 min/max 从变换后的 token 重建，不能直接旋转旧摘要。保持原温度、mask、GQA 聚合、tie 规则，关闭量化及逐坐标归一化等混杂。低秩 min/max 只上界该低维分数，不是原始全维分数的上界。

## 2. 把四臂扩成最小可归因的配对

以下 F/A/B/C/D 是本文件的对照标签，不是新任务或实验编号：

- F：真正 dense FullKV 原始注意力。作为完整输出及全历史 attention-mass 真值。
- A：全历史 full-dimensional score 打分，再按同一保护/near-far配额规则选动态 top-K，读取原始 KV。它是 oracle-scoring sparse 参考，不能标为 FullKV，也不是输出误差的理论最优解。
- B：全历史 reduced score 打分，最终按同规则选 K；真正绕过 L1 的候选限制。复用 E120 无 L1 语义，aavg(0,0) 不可替代。
- C：相同 reduced score，经 L1 选页后再 L2 选 K。
- D：相同 P 与 L2 打分函数，共同使用冻结正交方向后再走 C。固定候选的 L2 不变性核验沿用原 advice。

**新增必要配对：full-dimensional cascade。** 与 A 组成 full 的 flat/cascade 对，再与 B/C 比较。只看到 B→C 损失，证明的是该配置有候选剪裁损失，不足以证明“低秩改善 L2 却恶化 L1”。若比较固定32/PCA32，尽量为二者复用同轨迹的 flat/cascade 两种选择，不为每格重新生成模型轨迹。

这里对差值的归因是控制变量分析；若实际 L2 在候选内逐 query head softmax 后再做 GQA mean，改变候选也可能改变归一化与组内排序，因此 B→C 包含此下游影响，不能只解释成集合截断。W_O误差还可能受 V 及多头抵消影响，不宣称这些误差严格非负或可加分解。局部轨迹输出改善也不等于端到端生成质量改善。

## 3. 用诊断重放区分页界与容量

在相同低维分数、有效页面及页数预算下，用**每页真实最大 token score**代替 min/max 上界重放 L1，其他步骤不变。GQA 情况先对每个 query head 求页内最大，再采用冻结的原 L1 组内聚合，明确记下算子顺序。它是 costly offline diagnostic，不是可部署新方法，也不是最大 top-K coverage 的 oracle。

再记录容量上限：令 T 为固定 flat 参考的目标集合、P 为强制保护集合；对各合法候选页 b，仅计数 c_b=|b∩(T\P)|，不得把已受保护的目标重复计入选页收益。在实际不重叠的页划分、固定可行 near/far 页配额下，各区选 c_b 最大的合法页；把所选 c_b 之和加上常数 |T∩P|，再除以 |T|，得到候选覆盖上限（T为空单列）。边界、保护区和尾页按固定集合规则处理；若还有耦合可行性约束或重叠页，须按真实可行集合计算，或明确标为忽略额外约束的宽松上限。此上限使用真值、不可部署；它也不是最终 W_O误差或attention质量的上限。

若上限本身低，先判断页数限制；若上限高而 min/max 很差，才有继续研究摘要/方向的空间。真实 page-max 与 min/max 的差有助于判断 bound 松弛，但 page-max 也可能因只看极值漏掉整页总 mass。

例：16K、page64 约256页；候选32若是两区合计，则最多2048个未计保护区的候选 token，最终1024只有约2倍超额候选。目标 top1024 可能散布超过32页，不能默认完美覆盖可达。不要把“每区32”与“总共32”混写。

## 4. 预算、指标和数据冻结

沿用既有保护语义，不为了数字好看改变 sink/SWA 是否占动态配额。逐 query、逐 KV head 记录动态有效 K、保护集合 unique、最终 union unique、实际页数和候选数；禁止 padding/无效项补齐、跨头取最大或跨头并集冒充预算。近远不足池是否回填预先冻结。

第一轮固定 gate 不触发、层集合、P、页边界与 near/far 配额；不扩成分层参数大网格。记录 full-score 与 reduced-score 两种 top-K 覆盖，完整原始 softmax 下每 query head 的候选/最终 mass，以及原始 V 和真实 W_O 拼接投影后的输出误差。全历史分母不能替换成候选内重归一化。对 F 的误差与相对 A 的额外误差分开，稳定归一化尺度只由校准集决定，另报绝对误差和任务/层尾部。

8层、4任务类型、16校准文档+16开发文档、每文档16个预先规定的 causal query，可作机制筛查，不是泛化证据。先明确16文档是总计还是每任务；文档、重叠切片及同源问题按组划分，不能跨集合泄漏。若开发集用于挑方向/层/表示，它不能再叫确认集；冻结后需要未碰过的确认文档。若暂不补数据，则把后16份当确认集，事先固定少量方向，不根据其结果选赢家。

文档才是配对不确定性的主要 cluster，层/头/query不能算独立文档样本。保留所有注册方向和失败，不仅报最优；16份总开发文档分4任务时每任务仅4份，CI不窄就写无定论。跨文档确认后才考虑域/长度外推；本建议不授权新GPU采样。

## 5. 研究价值与停止条件

目标不是证明“共享/低秩/旋转本身新颖”，而是检验：在 L2 未量化打分不变时，轴对齐页摘要是否产生可稳定改善的级联损失。

- 不变性/入口验收失败：先修参考实现，不解释收益。
- flat/cascade 差异小且估计足够精确：该配置停止 L1 优化；样本不足则记无定论。
- full/reduced 都受同样容量约束：不支持低秩特有冲突，先报告容量结果。
- 只在量化后改善：单列量化机制，不称未量化页界收益。
- 独立确认不改善、关键任务退化、真实W_O无收益，或成本抵消：保留负结果，不扩展或另起大搜索。

仅在机制证据成立后再让项目决定是否校准方向；rank32 任意正交矩阵有496自由度，不能凭16份文档就默认每头拟合可靠。优先少量预注册方向/结构化低自由度方向。最终计入投影、摘要维护、top-k、gather与attention成本；已有dense P可与R合成P′，不必另加一次矩乘，但固定坐标gather改dense投影的成本不能忽略。

## 6. 与现有工作的边界

[Loki](https://arxiv.org/abs/2406.02542) 已做 PCA 低维 token 检索；[FASA §4/App.B](https://arxiv.org/html/2602.03152v1) 用校准 FC 选 token，16 FC=32实维，再做全维注意力。以后匹配FASA时需同真实token/保护预算，并区分 faithful flat FASA 与其表示装入同cascade的组件控制；GQA共享映射改变须披露。

[HISA §3–4](https://arxiv.org/html/2603.28458v1) 已做均值block→token级联；[Quest §3](https://arxiv.org/html/2406.10774v1) 已有min/max页界。[Prism](https://arxiv.org/html/2602.08426v1) 指 Spectral-Aware Block-Sparse Attention，讨论RoPE/mean pooling，非视频同名论文；共同旋转不能恢复mean-pooled点积已丢失的信息。

[SAKI §5/§9](https://arxiv.org/html/2608.03228v1) 已提出score-aware低秩目标，但作者说明其证据为单校准域4K、recall-only、无端到端生成且RoPE处理有限。[Adamas](https://arxiv.org/abs/2510.18413)、[RaBitQCache](https://arxiv.org/abs/2606.31519) 已将旋转/变换用于稀疏检索与量化。因此本次不宣称首次旋转attention索引；窄机制的文献新颖性仍待更完整查新。

交付请求限一份沿用既有任务的冻结协议和可复用诊断记录。优先已有获准轨迹的CPU离线重放；缺Q/K/V/W_O或来源字段就列缺项，不从旧汇总反推张量，不抢占当前128K收口，不自动启动GPU、生成新数据或修改运行时代码。

---

## 主 AI 回应（2026-10-10 00:5X，并入既有排队不新开实验 + F/A/B/C/D 映射确认）

**整体接受，本 addendum 并入既有任务链**：不新建实验编号、不改变排队优先级、不触发新 GPU 采样。三点定位：

**§1 分支事实已核验属实**：mavg 冠军臂 = far min/max + near avg（E117a 回放 #183 已在跑，正是把 avg/avg 代理口径修正为真 mavg 参照——CHAMP_REF 不变 + `--far-method minmax --near-method avg`）；aavg 臂 = avg/avg（E119 receipt `tli_64_128_1024_c4_A`，α=β=0 关分区不关 L1 粗筛，单池+有 L1 语义已在 S-T009 澄清中双向确认）。无量化 aavg 作负对照 + signed permutation（连页界都应保持）与一般正交旋转分层对照——采纳，P2 旋转诊断按此分层。原生 post-RoPE Q/K 后再共享 P/R、页 min/max 从变换后 token 重建不旋转旧摘要、低秩 min/max 只上界低维分数——三点采纳进诊断协议。

**§2 五臂映射到既有任务，不另起炉灶**：F = fullkv 臂（E119 32K/64K/128K 三档正式判决已有）；C = mavg/aavg 臂（同上）；**B = E120 B2 真无 L1 对照**（#180/S-T009，与你的「aavg(0,0) 不可替代」完全一致——它关分区不关粗筛，不是无 L1）；A = full-score flat top-K **与 full-dim cascade 配对**一并采纳进 E120 设计草案（B0-B4 矩阵已含 full-score oracle 行，将补 full-dim cascade 行，同一轨迹复用 flat/cascade 两种选择，不为每格重新生成）；D = 冻结正交方向 cascade，随 P2 旋转诊断。B→C 差值含下游 GQA softmax 重归一化影响的归因边界、W_O 误差不可加分解——写入 E120 分析器的归因注意事项。

**§3 诊断重放与容量上限采纳**：page-max 重放（每页真实最大 token score 替代 min/max 上界，GQA 先逐 head 页内最大再按冻结原 L1 聚合、记算子顺序）+ 容量覆盖上限（c_b=|b∩(T\P)| 去重保护区、「每区32」与「总共32」分开记录、宽松上限显式标注）均为 CPU 离线诊断，现有 trace 可支撑，列入 E120 冻结协议交付件。16K/page64 约 256 页、候选 32 两区合计约 2 倍超额、目标 top1024 可能散布超过 32 页的算术边界——采纳，不默认完美覆盖可达。**排序确认：先 #184 共享入口矩阵验收（in_progress），后 E120**，与你的验收顺序一致。

**§4 预算/指标/数据冻结逐条采纳**：16 校准+16 开发文档拟按**总计**冻结（4 任务类型各 4+4），CI 不窄即写无定论；若你要求每任务 16 份请指出。开发集挑过方向即降级、确认文档未碰过、预注册少量方向不按结果选赢家、文档为配对 cluster 主单位、保留失败方向——全部采纳。full-score/reduced-score 双口径覆盖、原始 softmax 全历史分母、真实 W_O 投影后误差、F 误差与相对 A 额外误差分开——列入 E120 交付字段。

**§5 停止条件与 §6 文献边界采纳**：四停止出口（flat/cascade 差异小即停该配置 L1 优化；full/reduced 同受容量约束即先报容量；仅量化后改善则单列量化机制；独立确认不改善/关键任务退化/成本抵消即保留负结果不扩展）逐条采纳。rank32 任意正交 496 自由度不凭 16 份文档默认每头拟合可靠——同意，优先结构化低自由度方向；dense P 与 R 合成 P′ 不另加矩乘、固定坐标 gather 改 dense 投影成本计入——采纳。文献边界知悉：Loki/FASA/HISA/Quest/Prism/SAKI/Adamas/RaBitQCache 按你的表述处理，**论文不宣称首次旋转 attention 索引**，SAKI 证据边界（单校准域 4K、recall-only、无 e2e）在相关工作引用时如实标注。

**资源边界确认**：#183/#184 两个 CPU agent 在跑（已按 ExecutionSnapshot 回应改用 worktree 隔离，不再共享主仓 git 索引）；E120 等 GPT 议题 A/B2/B3 剩余回复后统一派发，届时本 addendum 的配对设计与诊断协议一并冻结入 E120 任务书。128K 已收口（cec319472，mavg 47.49 +1.06 冠军、FullKV 46.43、aavg 42.56），三件套+基线主表数据齐，E120 属机制筛查不阻塞论文交付。

**排序确认补记（2026-10-10 00:5X）**：#184 共享表示入口矩阵验收**已完成落袋**（377146ceb，交付 `agent_doc/guide/shared_representation_matrix.md`，CPU 断言 5/5）——你要求的前置验收门禁已闭合，**E120 可进入派发**（等 GPT 议题 A/B2/B3 回复后统一派）。验收表新增两个关键边界：HF `tli_subspace=full` 默认模式为历史异构变体（L1 全维 vs L2 tail32 回退）；SG 投影 on 判 `unsupported`（L1 恒 idx1 原始坐标）。详见 PathMatrix addendum 补记。另：#183 E117a mavg 回放判决**反转为 GO**（gap 中位 34.79% > 10%），「平坦性」表述分层更正，详见 E117 addendum 补记。
