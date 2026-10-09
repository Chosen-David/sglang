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
