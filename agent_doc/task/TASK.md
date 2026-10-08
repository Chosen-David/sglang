# sglang 项目任务索引（AI 维护）

这是 sglang 项目的唯一 AI 任务索引（agent_doc 口径），由主 AI 串行维护。
**优先级从属关系：根目录 `/home/wangyuanshuo02/sglang/TASK.md`（用户
私人设定，AI 只读）是指南层，本文件是执行索引层**——每项登记的任务须
能追溯到用户指令或根 TASK.md 条目；两者冲突时以根 TASK.md 为准。

每项任务的方法、进度、证据、阻塞与下一步写 `task_details/<任务ID>.md`，
不在此堆日志。日期是任务来源/组织日期，不自动表示完成日期。

## 2026-10-08

- [ ] [S-T001] E109 method×(α,β,γ) 海选-全量两级扫描（进行中：AVG5 6 臂
  入账，榜首 aavg(.125,.125,.375)=46.43；三机 12 卡满载推进）
  ([详情](task_details/S-T001.md))
- [x] [S-T002] GPT 复查 bug1/2/3 核实与修复（全部属实全修复，红-绿双证，
  sglang 067553e60/3a868d9bf/3b1482744）
  ([详情](task_details/S-T002.md))
- [ ] [S-T003] agent 库升级 f299566c + sglang agent_doc 布局建立
  (本文件即交付物之一；[详情](task_details/S-T003.md))

历史任务（E1-E113 全链）的原始记录位于 `two-level-attention/exp/trace/`
与 `research/docs/`，不在本索引重建；新任务自此按本格式登记。
