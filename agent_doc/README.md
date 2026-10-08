# sglang 项目 agent_doc：任务、指南、建议与结果

这是 sglang 仓库（TLI/PSI 稀疏注意力索引器研究主线）的项目文档区，与
`docs/`、`research/` 等普通文档分开，遵循 agent 库的
[项目文档治理](https://github.com/Chosen-David/agent) 布局。工作流源码与
模板从 `~/agent-research-workflows` 读取，本目录只放 sglang 项目自身的
任务、建议与结果。

## 本项目的指南映射（重要）

sglang 仓库已有两个人类发布的硬约束文件，地位等同 `agent_doc/guide/`
中的人类指南（最高项目规划优先级、AI 只读、绝不修改）：

| 文件 | 性质 |
| --- | --- |
| `/home/wangyuanshuo02/sglang/TASK.md` | 用户私人任务设定（含监督器协议、任务链编排规则、commit 纪律、mass≠e2e 铁律） |
| `/home/wangyuanshuo02/sglang/论文indexer.md` | 论文口径权威定义（异议写 `论文indexer_反馈.md`） |

`guide/` 目录本身保持空（人类可直接在其中创建 `GUIDE.md` 补充指南）；
上述两个根目录文件已覆盖当前全部人类约束，不复制其内容到本目录。

## 目录职责

| 目录 | 主要内容 | 谁维护、AI 如何处理 |
| --- | --- | --- |
| [guide](guide/README.md) | 人类目标、优先级、硬约束（含上述两个根目录映射文件） | 人类发布；AI 只读，优先据此编排 |
| [advice](advice/README.md) | 人类、其他 AI、当前 AI 的建议与反馈 | 人类或 AI 记录；主 AI 核验证据后 adopt/adapt/reject/defer |
| [task](task/README.md) | AI 整合后的唯一任务索引与详情 | 主 AI 串行维护 `task/TASK.md` |
| [results](results/README.md) | 数据、运行元数据、产物与独立验证 | 生产者记录，独立验证者核验 |

处理顺序：当前用户决定与权限 → 根目录 TASK.md/论文indexer.md 等人类指南
→ advice 核验取舍、已有 results 适用性检查 → AI 整合任务 → 执行与独立
验收 → 按要求逐项汇报。

## 既有资产的映射

sglang 仓库已有长期演化的任务与结果体系，为避免双清单冲突，约定：

- 根 `TASK.md`（人类）= 指南层，**优先级最高**；
- `agent_doc/task/TASK.md`（AI）= 执行索引层，登记的每项任务须能追溯
  到用户指令或根 TASK.md 条目；
- `two-level-attention/exp/trace/results/*.json` = 历史结果落袋区
  （E 系列 JSON），`agent_doc/results/` 登记新 run 的索引与验证状态
  （JSON 仍落 trace/results，不搬家）；
- `research/docs/` = 审查判决与研究文档（gpt_kimi3_verdict 等判决 JSON
  保持原路径）。

模板见 agent 库 [templates](https://github.com/Chosen-David/agent)；
任务格式见 `~/agent-research-workflows/templates/TASK.md`。
