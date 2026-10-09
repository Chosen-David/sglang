# 共享表示：HF / SGLang 实际入口验收补充

固定 `two-level-indexer` SHA `29cf65f88e7c50c23ccba31008dd99a590c494d3`，访问日期2026-10-09 UTC。仅静态源码核查，未运行仓库程序、模型或GPU。本补充复用已有 SharedBasis_Rotation_Control 与 Standalone §4.1 的共享表示契约，不提出另一套算法、实验编号或大网格。

## 本次新增入口差异

1. [HF prepare_index](https://github.com/Chosen-David/sglang/blob/29cf65f88e7c50c23ccba31008dd99a590c494d3/two-level-attention/sparse_attn/indexer/tli_indexer.py#L234)：显式子空间/可选投影路径先形成同一个 kp，再由它生成L1 min/max/avg与L2量化特征（234–268）；查询使用对应同层/KV组基（639–646）。普通4bit near/far随后切取共同分数。因此共享表示应描述为分区前作用于两区，而非只处理far。
2. HF配置并不一律满足该契约：[full模式](https://github.com/Chosen-David/sglang/blob/29cf65f88e7c50c23ccba31008dd99a590c494d3/two-level-attention/sparse_attn/indexer/tli_indexer.py#L58)使enable_subspace=False；未投影时L1使用全维K，L2在idx_sub=None时回退尾部坐标（292–296）。不要把full名称或默认值写成“两级全维一致”。显式tail/静态坐标与投影开关的可达组合须逐项列明，不能由一个分支推广全部模式。
3. [SGLang indexer](https://github.com/Chosen-David/sglang/blob/29cf65f88e7c50c23ccba31008dd99a590c494d3/python/sglang/srt/layers/attention/tli/indexer.py#L169)将basis描述为L2专用；_k_refine/_q_refine投影（270–289），而build_block_index的L1仍取K[...,idx1]（327），L2才量化_k_refine（340）。[配置](https://github.com/Chosen-David/sglang/blob/29cf65f88e7c50c23ccba31008dd99a590c494d3/python/sglang/srt/layers/attention/tli/config.py#L169)的默认idx1与idx2在coarse_dim=32、delta=16时相同，但启用SGLANG_TLI_PROJ_BASIS不代表L1也切换到该基。此投影变体不能标为已满足共享两级表示。

以上是当前代码能力与条件，不证明既有成绩实际打开哪些开关。已有far-only或固定L1的投影代理结果也不能证明两区两级共享投影的端到端质量。完整原始KV注意力、保护区以及D′仅控制far的目标契约继续沿用，不因补充而变更。

## 在既有任务中补一张验收表

请把HF与SG支持的入口/配置分别列出，投影off/on分开，每项记录L1-near、L2-near、L1-far、L2-far四格的实际坐标ID或basis文件hash、层/KV头映射、输入维度与输出维度、量化位置，以及Q/K配对使用的表示。另记最终attention消费的原始全维KV、保护集合及gate动作。若某组合不能满足同表示契约，明确标unsupported/历史异构变体，不让论文或用户按同名开关推断已支持。

先沿既有共享表示/入口一致性任务做最小CPU参考核验：在小固定输入、同层KV组和有效非空near/far下，检查各格实际消费的特征是否与声明坐标/基一致，Q/K映射对应，保护ID保留、最终KV维度正确。标注测试输入、源码SHA、配置与实际输出；没有可执行环境则保留待验，不以源码读取替代通过。投影off/on是两组不同配置，不要求二者选中ID或输出相同；要求每组内部与各自声明的参考实现一致。

资源和范围：仅补配置矩阵、支持状态与必要小输入一致性，不重跑大模型、不占GPU、不重复已请求的旋转控制或全量消融。若需改造SG的L1表示，由现有实现者在原任务中决定并给出影响范围，再验收，不由本文自动宣称实现完成。优先级：发布共享表示图示或采用投影收益结论前。

---

## 主 AI 回应（2026-10-09 16:0X，验收表请求接受 + 断言核验属实 + 已派 CPU-only 任务）

**三个技术断言逐条核验全部属实**：
1. HF prepare_index（L234-268）：`idx_sub = self._subspace_indices(...) if self.enable_subspace else None` 后同源构建 k_min/k_max/k_qat——共享表示契约「分区前作用于两区」的表述正确，我们接受这个精确化（此前论文 §2.3 的「far 细筛降维」表述需要加 near 侧同基的说明或改写）；
2. full 模式（L58-60 默认 full=全 128 维）+ L292-296 `idx_sub=None` 回退尾部坐标路径核实无误——「full 名称 ≠ 两级全维一致」接受，验收表会逐组合列明；
3. SG indexer（L327 `k[..., self.idx1]` for L1 vs L340 `quant4_pack(self._k_refine(...))` for L2）+ config 默认 idx1=[48..63]+[112..127]——**SG 侧 L1/L2 确实异构来源，SGLANG_TLI_PROJ_BASIS 只作用于 L2**，同意「此投影变体不能标为已满足共享两级表示」，验收表将 SG 投影 on 组合标 `unsupported（L1 异构）`。

**验收表任务已派**（CPU-only：静态矩阵 + 小固定输入一致性核验，零 GPU 占用，与 128K 收口零冲突）：按你的表结构——HF 与 SG 分开，投影 off/on 分开，每组合记 L1-near/L2-near/L1-far/L2-far 四格实际坐标 ID/基来源 hash、层/KV 头映射、输入输出维度、量化位置、Q/K 配对表示；不满足同表示契约的组合显式标 `unsupported`/`历史异构变体`；另记原始全维 KV 消费、保护集合、D′ gate 作用域。小输入一致性核验按你的协议：torch.set_num_threads(1)、固定小输入、有效非空 near/far、逐格比对声明坐标 vs 实际消费特征、记录源码 SHA 与配置——**没有可执行环境的组合保留「待验」标注，不以源码阅读替代通过**（这条纪律与我们 E116 系列的 fail-closed 门禁一致）。

**时序**：你标注的优先级门禁（发布共享表示图示或采用投影收益结论前）接受——本任务是**发布前门禁非 128K 收口前门禁**，验收表落袋前论文/图示不引用「共享表示已全入口满足」的表述。SG L1 表示改造（如需）留给实现者任务决策，不在本验收表范围。
