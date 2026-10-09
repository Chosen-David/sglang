# TwoLevel 正式发布并发竞态与 E119 汇总门禁审查（83e7bcd）

## 审查目标与范围

- 发布前远端分支 SHA：`8bd6037b94a4bf5f2ec9351cb141875ef14e2146`；相关实现基线 SHA：`83e7bcdf2a65821e33f42f0e4c4c8dbee3290094`。其间仅新增任务索引/讨论文档，未改动本报告审查的发布器、E119 汇总脚本或结果文件，故复现仍绑定实现基线 `83e7bcdf2`。
- 重点增量：
  - `d659568ab43ad35b7bf87cb6399c8dc8161ad320`：E116f generation/兼容镜像发布协议；
  - `bb4cb21ac`：E119 RULER 64K 三臂正式结果与汇总；
  - `83e7bcdf2`：前次审查回应，仅为 advice 文档，不作为源码变化。
- 实际追踪链：`score_ruler_formal.py` 的 staging/generation → 四个固定公开镜像 → receipt 提交信号 → `analyze_e119_ruler64k_formal.py` 三臂聚合 → 已提交 result/manifest/receipt/scorer manifest。
- 本轮为 CPU/文件系统审查；未运行 GPU kernel、真实模型生成或 128K 生产任务。干净检出缺少未提交的生产 prediction JSONL 与 generation `pred_root`，因此无法在本机重跑 E119 的源/派生文件 SHA 检查；已对仓库中实际提交的结果、receipt、formal manifest 和 scorer manifest 独立复核。

## 结论摘要

| ID | 状态 | 严重度 | 结论 | 当前 E119 数据影响 |
|---|---|---:|---|---|
| `TL-RULER-CONCURRENT-PUBLISH-040` | **confirmed** | P1 | 同一 `--out` 的两个并发成功发布者可留下固定 JSON 与 receipt 的持久混合代际；两进程均返回 0。 | 未见已提交 64K 三臂受影响：三臂输出路径不同，且各自 result/manifest SHA 与 receipt 闭合。 |
| `TL-RULER-CROSSARM-IDENTITY-041` | **confirmed（验收门禁缺口）** | P2 | E119 汇总逐臂验 SHA/n，但不比较三臂样本 ID、答案、长度和源数据身份；未来错配样本仍会形成排名结论。 | 当前三份 scorer manifest 字节完全一致，formal manifest 的任务身份和源数据身份摘要也一致，因此现有排名未见由该缺口污染。 |

## 发现 1：`TL-RULER-CONCURRENT-PUBLISH-040`

### 违反的契约

`two-level-attention/benchmark/RULER/score_ruler_formal.py:584-597` 声明固定公开镜像按 JSON → MD → manifest → receipt 安装，receipt 最后落盘后“混合代际不可达”。但代码没有对规范化后的同一 `--out` 建立进程间互斥或 CAS：`run_id` 只隔离 staging/generation（约第 393-404 行），不能隔离固定 aliases。备份、四次替换、清理/回滚都可被另一个发布者交错。

### 可达触发与实际行为

触发条件：两个进程同时使用同一 `--out`，但输入产生不同结果。最小交错：

1. A 安装自己的 JSON 后暂停；
2. B 安装 JSON/MD/manifest/receipt 全部成功；
3. A 恢复并安装自己的 MD/manifest/receipt，也返回成功；
4. 最终固定 JSON 属于 B，固定 manifest/receipt 属于 A。

CPU 最小复现脚本 SHA256：

`8bfba4d996642f7c3ec965bdfa07d491e183702884951d178040dc3cda085fc6`

单次主复核原始输出（两个实际 scorer 子进程，非手写预期）：

```json
{
  "a_rc": 0,
  "b_rc": 0,
  "receipt_result_sha256": "d84fd21c588e5f067a292defef9e7346ce1ddb60a41672058e6c6fce12ad1de1",
  "actual_result_sha256": "5ed9e3c9c73b65ec3b9eaba1d215a5dcea4e1e3f6921802fc3b0fd4fb81bf80e",
  "result_matches_receipt": false,
  "manifest_matches_receipt": true,
  "generation_exists": true
}
```

独立复核在独立上下文重复运行 4 次，4/4 均为两个子进程 `rc=0` 且 `receipt.result_sha256 != sha256(固定 JSON)`。两个不可变 generation 本身均存在且各自完整；缺陷发生在仍由下游直接读取的固定兼容镜像。这也意味着并发失败路径可能用自己的备份覆盖另一个发布者刚完成的镜像，现有单写者 OSError 回滚测试不能排除此类竞态。

### 建议修复与最小重测

1. 以 `realpath(abspath(--out))` 为稳定键建立进程间独占锁；锁至少覆盖“读取/建立备份 → 四镜像安装 → SHA 终验 → 备份清理或回滚”，备份必须在获得锁后创建。若目标允许跨主机共享文件系统，必须明确并实测所选锁在该文件系统上的语义，不能把仅本机有效的锁表述为跨宿主保障。
2. 更强方案是只原子更新一个小型 generation 指针，所有消费者先读该指针并只从不可变 generation 取 result/MD/manifest/receipt；固定 aliases 仅作不具事务承诺的展示副本，且消费者不得把它们拼装成正式证据。
3. 新增双进程确定性交错测试：允许两轮串行完成，但最终四个固定 aliases 必须属于同一 `run_id`，receipt 的 result/manifest SHA 均与固定文件闭合；再覆盖一方发布中失败、异常退出、锁等待/超时和陈旧锁恢复。当前 `test_e116f_publish_atomic.py` 只有单写者故障注入，不能作为并发反证。

## 发现 2：`TL-RULER-CROSSARM-IDENTITY-041`

### 违反的契约

`two-level-attention/exp/trace/analyze_e119_ruler64k_formal.py:41-75` 对每个 arm 分别检查：receipt success、result SHA、选中的一个 cell 全部 `n=100`、该 arm 的 source/derived 文件 SHA；随后直接在第 77-108 行生成三臂排名。脚本从未比较三臂 formal/scorer manifest 中的：

- task 集合及逐 task `_id`；
- `answers_sha` 与逐行 `lengths`；
- `source_data_sha256`；
- `expect_tasks`、`min_samples`、模型/YaRN 身份和正式入口/评分脚本版本。

因此三份各自“内部闭合”但样本不同的结果仍会被输出为公平 A/B/C 排名。三臂分别调用正式入口不能弥补此缺口，因为入口的跨方法身份校验只发生在单次调用内部，而 E119 将三臂放在三个独立 root、分别运行。

### 当前数据的反证核验

本轮没有把门禁缺失误报成已发生的数据污染。对已提交证据的独立核验结果：

- 三份 `scorer.manifest.json` 字节 SHA256 完全相同：`139721c983ec4bdec54046be44bf1b3aca42e184eb34253718514f143a123637`；
- 三份 formal manifest 的 `tasks` 规范化摘要完全相同：`77124042532c72cc187444f5c3b2062708f24dea0584a1d370aa2af3fe9959a3`；
- 三份 `source_data_sha256` 规范化摘要完全相同：`05663f07d40ee482869c5e1b16ace6be1e60544fc1991c4de13b5b003e22d299`；
- 模型、YaRN 与 factor 一致，只有 arm 的 treatment 参数/方法合理不同；
- 三臂 receipt 的 result/manifest SHA 均与当前固定文件一致；重算未四舍五入平均为 `49.420909... / 48.536363... / 47.511818...`，发布值 `49.42 / 48.54 / 47.51` 正确。

所以当前 `mavg 49.42 > FullKV 48.54 > aavg 47.51` 排名的已提交身份资料能够闭合；041 是未来重跑不会自动 fail-closed 的确认缺口。

### 建议修复与最小重测

1. 汇总时从每个 receipt 定位正式 manifest/scorer manifest，先建立共同 identity digest；三臂必须逐字段比较 task、ids、answers、lengths、源数据 SHA、样本门禁、模型/YaRN、formal/scorer 实现 SHA。
2. 对允许不同的 treatment 字段建立显式白名单（如 arm 方法、alpha/beta/gamma）；不要用“忽略整个 `extra_params`”替代白名单。
3. 将共同 identity digest 和实际比较字段写入 summary；任何不一致必须非零退出且不覆盖旧 summary。
4. 负测至少分别篡改单臂一个 `_id`、`answers_sha`、length、source-data SHA 和模型身份，均应拒绝发布；另测试多 cell 输入，避免 `next(iter(d["n"]))` 静默忽略额外 cell。

## 旧发现复查与实际测试

- `TL-RULER-PUBLISH-ATOMICITY-038` / `TL-RULER-DERIVED-COMMIT-039`：在**单写者**范围内已复验修复。`python3 -m benchmark.RULER.test_e116f_publish_atomic` 实际通过 4/4：四替换位置 OSError 回滚、generation rename 失败、真实跨设备 `/dev/shm` manifest、既有 E116e 回归均通过；内部 E116e 为 12/13，D9 因本机无生产数据跳过。
- `py_compile`：`score_ruler_formal.py`、`test_e116f_publish_atomic.py`、`analyze_e119_ruler64k_formal.py` 均通过。
- E119 汇总脚本在干净检出直接运行会因未随仓库提交的 `exp/results_ruler/...` 原始预测和 generation `pred_root` 缺失而 fail closed；这不否定上述对已提交 manifest/result/receipt 的哈希核验，但表示本机未完成 production source/derived JSONL 的独立重放。

## 对已跑数据与论文结论的处理建议

- 当前 64K 三臂输出使用三个不同 `--out`，其 receipt/result/manifest 与三份共同样本身份均实际闭合，**不建议仅因 040/041 撤销现有 64K 数值**。
- 在修复 040 前，同一路径可能重叠运行的正式收口不得以“两个进程均 DONE”作为有效发布证据；发布后必须强制检查 receipt 两个 SHA 与固定 aliases，或只从 receipt 指向的 generation 消费。
- 128K 及后续多臂收口在形成结论前应补上跨臂 identity 门禁；已有 64K 的相等 manifest 可作为正例 fixture，但还需上述错配负例证明 fail-closed。

## 下一检查点

复验同 `--out` 双进程锁/CAS、异常退出与消费者 generation 解引用；随后复验 E119/128K 汇总的跨臂 identity 负测和多 cell 拒绝。GPU kernel、真实生成与 128K 完整数据仍属于未覆盖范围。
