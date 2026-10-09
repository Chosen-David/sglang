# E116a 后续：补齐输入身份、任务全集与实际消费清单

固定源：`two-level-indexer` / `6d040d87687e65b66fc33c1ccdaf2c2e1507dd48`，访问 2026-10-09 UTC。沿用 E116a、TL-LBV1-SAMPLE-GATE-024 与 SCORER-BACKEND-025，不新建同根因编号。本建议仅静态审查新实现与 JSON；未执行仓库代码、测试或模型。以下是待运行的最小验收，不是已复现生产故障。

## 已确认的增量

`b06c181` 确实增加了 v1 ID、eval 的可选逐文件门禁、显式 scorer 后端及结果元数据；新 provenance manifest 列出三臂×13任务的文件 SHA256/行数，且提供了 lcc/repobench 双后端作者重评。这些比此前仅行数报告更完整，应保留。7/7、9/9 属仓库作者报告，本次未独立运行；旧稀疏原始预测仅列外部路径，本次未取得字节。

## 三个仍需闭合的点

### 1. ID 仍未绑定完整输入

[pred.py L431–441](https://github.com/Chosen-David/sglang/blob/6d040d87687e65b66fc33c1ccdaf2c2e1507dd48/two-level-attention/benchmark/LongBench/pred.py#L431-L441)只对全部 `answers+length` 做 digest，再拼 task/row_idx。问题或上下文改变但答案和长度不变，身份也不变。旧文件跨臂 answers 按行一致是有益检查，但答案可重复，不能单独证明是同一问题/上下文。

建议保留现有 ID 兼容历史，同时增加源数据固定revision/文件hash及每行规范化输入hash（覆盖实际用于prompt的字段），冻结prompt/tokenizer/截断配置。最小负例：同任务同答案同长度但context/question不同，必须得到不同输入身份或被同一冻结输入manifest拒绝。历史不能补证的字段标unknown，不用事后行序推断替代原始输入证据。

### 2. 逐文件通过还不是任务全集通过

[eval.py](https://github.com/Chosen-David/sglang/blob/6d040d87687e65b66fc33c1ccdaf2c2e1507dd48/two-level-attention/benchmark/LongBench/eval.py#L151-L238)只遍历实际存在的JSONL；循环结束后没有比较已验任务集合与manifest预期任务全集。缺整个任务不会进入该任务的门禁；遇到manifest外任务还会warning后跳过。`answers_sha`缺某ID键也跳过验证。两个参数缺省为空/零时，新门禁大部分未启用。

建议正式结果模式要求完整预期任务集合、每个ID必有答案hash、每个task恰好一个被manifest选中的文件；历史宽松模式显式标partial，不能共用正式验收状态。最小负例：manifest有A/B而目录只有A、目录为空、某ID缺答案hash、出现两个同task候选、未显式提供正式manifest。均应非零退出且不发布新正式结果；如允许历史模式，结果必须可机器区分。保留现有删除单行/重复行等已有测试，不重复运行模型。

### 3. 发布的 provenance 清单与 eval 输入契约不同

[e109_lbv1_manifest.json](https://github.com/Chosen-David/sglang/blob/6d040d87687e65b66fc33c1ccdaf2c2e1507dd48/two-level-attention/exp/trace/results/e109_lbv1_manifest.json)顶层是 `arms`，内含文件路径/hash/行数；eval期待 `{task:{ids,answers_sha}}`。两者各有用途，但不能把前者存在当作后者已消费通过。直接传前者时，`manifest.get(dataset)`得不到任务并走warning/continue；静态看不到正式结果发布前的schema拒绝。

建议明确区分provenance与评分identity清单，添加schema/version与转换过程的固定身份；保存实际被eval消费的identity manifest哈希、完整调用参数及结果receipt。最小负例：把provenance JSON误传`--manifest`必须立即schema失败；正确清单且全部task、ID、答案/输入hash闭合则通过；同任务多文件的选择规则需固定并列完整候选及排除理由。现有“取最多行”规则不等于按分数挑最好，但平局/多次重跑选择仍需可复核。

## 结果表述与最低成本

先补CPU门禁和清单，再从现存原始预测验证；不因此自动要求重跑模型。不能证明原始输入/配置身份时，再按既定E116b决定必要重跑。scorer显式固定已是改善，但作者双后端表不能表述为所有delta逐位相同：difflib aavg−FullKV按已报均值为−0.11，Levenshtein为−0.10；mavg按已报均值均为+0.30。保持后端分列，勿混合任务分数。

优先级P1：正式精度结论发布前完成以上三项；风险主要是把局部门禁通过误读为全任务/同输入闭包。通过条件是新负例均拒绝、完整同身份正例通过、生产调用消费的是正确schema且receipt可回查。失败则保留旧结果为探索观测并列明缺口。本建议不修改人类guide、SGLang代码或运行配置，也不声称实际数据已经错配。
