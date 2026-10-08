# advice：建议、反馈与取舍

这里收集人类、其他 AI（GPT 复查、Kimi3 审查等）或当前 AI 提出的建议、
方案、代码审查意见和研究反馈。保留原始来源及其版本，避免把转述当原话。

主 AI 阅读相关建议后，结合当前用户要求、根目录人类指南（TASK.md /
论文indexer.md）、实际代码和证据，逐条记录处理结果：

| 处理 | 应说明什么 |
| --- | --- |
| adopt：采用 | 依据、适用范围、对应任务与验收 |
| adapt：调整后采用 | 采用哪些部分、如何调整、为什么 |
| reject：拒绝 | 与事实、目标、约束或接口不符的具体原因 |
| defer：暂缓 | 缺少什么、影响哪些任务、何时重新评估 |

既有判决记录（GPT/Kimi3 审查 JSON）保持 `research/docs/` 原路径，本目录
只登记新建议与其取舍结论；引用原 JSON 时写明路径与 commit。

建议记录格式见 `~/agent-research-workflows/templates/advice_assessment.md`。
