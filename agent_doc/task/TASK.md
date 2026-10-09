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

## 2026-10-09（补记 S-T004~S-T008，主 AI 维护义务恢复）

- [x] [S-T004] E116a-e 评分完整性门禁五连（LB v1 manifest / RULER
  fail-closed / 正式入口 / 发布原子性；遗留 E116b 等冠军口径重跑）
  ([详情](task_details/S-T004.md))
- [x] [S-T005] E117a W_O 投影逐层潜力重判（NO-GO：中位 gap 8.045%<10%，
  平坦性第五证；2a852c9a0）
  ([详情](task_details/S-T005.md))
- [x] [S-T006] E117b LB v2 解析器官方口径统一 + 503×3 重评（排序反转：
  FullKV 32.21 > mavg 32.01 > aavg 31.81，原「LB v2 冠军=aavg」撤销）
  ([详情](task_details/S-T006.md))
- [ ] [S-T007] E119 四机 RULER 64K/128K 汇总 + 正式入口收口
  （**64K 三臂已收口**：mavg 49.42 > FullKV 48.54 > aavg 47.51，
  bb4cb21ac；128K 在飞四机 20 卡满载，全齐后同口径收口）
  ([详情](task_details/S-T007.md))
- [x] [S-T008] E116f 正式入口发布事务原子性修复（GPT 审计 038/039，
  generation+单指针协议，红绿 4/4 + 32K 回归逐位；d659568ab）
  ([详情](task_details/S-T008.md))
- [ ] [S-T009] 论文对比口径讨论（dense vs 单级稀疏）+ E120 单级无粗筛
  方案探索（用户 13:2X 指令；讨论已发起 advice，待 GPT 回复）
  ([详情](task_details/S-T009.md))

## 2026-10-10

- [ ] [S-T011] E117 8 层逐层配置 e2e 小试（进行中 3/13；两臂 perlayer vs uniform
  配对，GPU0/GPU1 并行；判决规则：perlayer 优于 uniform → GO 并入生产）
  ([详情](task_details/S-T011.md))

## 待办池（未排期，均有用户指令或审计来源）

- [ ] E118：sglang C1/C2/C3 三源码 bug 修复（SG e2e taskmd+CUDA graph
  前置门；C3 修复改行为须换 postfix 重跑）
- [ ] E116b：near/SWA 边界修复后三件套全量绑新口径重跑（等 GPU）
- [ ] E120：粗筛阶段价值消融 + 单级方案对比（依赖 S-T009 讨论收敛）

我希望你任务完成后找到最新agent库自己拉取最新代码然后把你多机子上多个GPU调度踩的坑和宝贵的经验整理进去（以skill或者知识库相关的合适形式，从你ssh进入GPU资源池开始怎么规划你的任务等的经验，你的任务属性是什么场景是什么这样，你是领悟了什么方法感觉比较好），为后面提供宝贵的经验
