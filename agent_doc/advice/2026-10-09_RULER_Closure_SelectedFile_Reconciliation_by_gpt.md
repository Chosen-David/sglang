# E116c 补充：让闭包报告绑定同一份 selected-file 快照

固定源 `two-level-indexer`：`beb9525c7c5eb4963d2d5917a4f648abd408b104`；访问 2026-10-09 UTC。沿用 E116c / TL-RULER-SAMPLE-GATE-026。本次只读 JSON、源码与已发布建议；没有运行项目程序、测试或模型。以下是新闭包工件内部一致性问题，不是判断历史分数已被污染。

## 已有进展与这次窄增量

E116c 的同键多文件拒绝、显式合并、逐行 ID/答案 hash、可选 manifest/任务数门禁，以及 sources 文件 hash 输出，均已在代码中可见。仓库作者报告 11/11 测试和 FullKV 59.38 回归；本次没有独立执行，按作者报告保留。此前 0826 审计和 E116a Followup 已覆盖通用身份、输入 hash、缺任务和清单 schema，本文不重新要求整套方案，只核新增 closure 能否支持其自身结论。

## 同一个 JSON 中有一格 61 / 100 不一致

[e109_ruler32_closure_check.json](https://github.com/Chosen-David/sglang/blob/beb9525c7c5eb4963d2d5917a4f648abd408b104/two-level-attention/exp/trace/results/e109_ruler32_closure_check.json)，Git blob `69623304cb9edf4bb1a3769f65b297aef704ff10`：

- `per_task.niah_multikey_1.pred_E109_aavg_a0_b0_g0` 列出 `...-10090655.jsonl`，SHA256 `a40d81bdb3a929d5290f55b73fb0bf99df8fb9e932250acdf6983ca7edf75002`，`rows=61`。
- 同格 `cross_arm_answers_check` 声明 `rows=100`、逐行一致。
- `same_key_collision_check` 列出 `...-10090537.jsonl` 与 `...-10090655.jsonl`，对应行数 `[100,61]`，只声明 `pred_diff_count_in_common_prefix=0`。

因此 per_task 列表总行数为 3261，而跨臂检查声明总行数为 3300。这可能是 per_task 取了最新文件、跨臂检查取了最多行文件，也可能是不同时间快照；现有 artifact 无法确定。不能从这处元数据不一致反推已报 57.33 的实际来源有错，也不能把它称为已核验无污染。

另外两组碰撞是 89/100 与 54/100。共同前缀预测完全相同，不等于不同长度文件的全体均分相同。若最终选择完整100行文件，部分文件可能确实不影响最终表，但应写“所选完整文件已核验，部分文件未采用”，并给出所选hash；不能据共同前缀相同推出“任一文件得分相同”。

## 最小修订，无需先重跑 GPU

1. 从实际评分所用文件冻结 selected-file 清单，逐格绑定路径、SHA256、行数、有序样本/输入或现有可取得身份、生成配置与评分器。per_task、跨臂检查、碰撞排除及最终分数都引用同一个 selected-file ID。
2. 对上述一格明确究竟采用哪份100行文件，以及61行文件为何出现在 per_task；如果是快照间增长，分别记录时间和hash，不覆盖旧报告。重新生成一致的闭包报告并保留修订关系。
3. 保存候选全集及采用/排除原因；对短文件只比较共同前缀，并明确未覆盖尾部，不把前缀检查外推为全文件等分。若全体100行文件才是实际输入，检查它与最终表的绑定即可；证据不足再决定必要重跑。

## 新合并器的一个定向回归建议

[score_ruler.py `_ts_of` / `_resolve_group`](https://github.com/Chosen-David/sglang/blob/beb9525c7c5eb4963d2d5917a4f648abd408b104/two-level-attention/benchmark/RULER/score_ruler.py#L67-L105)把已生成的 `merged` 文件也加入候选，行数相同时按字符串比较，`merged` 大于数字时间戳。静态推理：首次生成100行 canonical 后，再加入同任务同键、也是100行但时间更新的原文件，canonical 会胜过新的数字时间戳。源码注释明确写了 merged 并列优先，可能是有意的冻结策略；它满足“重复相同输入幂等”，但与“并列取最新原始运行”的刷新含义有差异，需明确采用冻结优先还是刷新策略，不能仅据此断言设计有 bug。

请加一个小文件负例：旧100行→生成canonical→新增另100行新时间戳文件，记录应该保持冻结run还是选择新run。若同run只允许唯一不可变输入，新增冲突应拒绝；若明确采用最新策略，比较应使用原文件身份及真实来源时间，不能让派生的 `merged` 标签参与竞争。不要为了通过测试静默删旧输入；保留原始候选和结果可追溯关系。此处仅源码推断，尚未运行该回归，也不声称当前表已遭该路径影响。

优先级P1：先修订自相矛盾的闭包工件，再决定是否提升论文证据状态。通过条件：所有分支引用同一selected hash，33格计数一致，partial碰撞的结论范围准确，合并器新增等长文件情形符合明确契约。现有59.99/59.38/57.33暂保留为作者汇总，不更改数字、不宣称冠军或数据已污染。
