# results：数据、运行记录与独立验证

这里登记 sglang 项目新 run 的索引与独立验证状态。**实验 JSON 数据本身
仍落 `two-level-attention/exp/trace/results/`（E 系列落袋区）与
`research/docs/`（审查判决 JSON），不搬家**；本目录登记 run 的指向、
验证状态与适用范围，供后续 AI 检索复用。

状态约定：

- `pending`：仍缺有效独立验证，不能支持下游正式结论。
- `usable-with-scope`：仅能在记录的条件与范围内使用。
- 失败/污染：保留原始记录（如 `exp/trace/results/
  e109_e2e_scan_v1_polluted.json`），修复后以可追溯的新版本重测重验，
  不覆盖失败来制造成功。

收到新任务先检索已有结果，再比较目标、代码、输入、配置、环境、指标
单位及当前验证状态；条件不匹配时说明拒用或需要补验的原因。

## 已登记 run（入口索引）

| run_id | 指向 | 状态 |
| --- | --- | --- |
| E109 v2 海选 | `/tmp/e109_scan_v2`（三机汇聚，selection.json 增量落袋 `two-level-attention/exp/trace/results/e109_screen4_selection.json`） | pending（链在跑） |
| E109 冠军对拍 | `exp/trace/results/e109_champ_validate.json` | usable-with-scope（v1 污染判决证据） |
| GPT/Kimi3 判决 | `research/docs/gpt_kimi3_verdict_20261008.json` | usable-with-scope（修复全闭环） |
