# TwoLevel E113/E119 修复增量复审（`f5f457d`）

- 审查时间：2026-10-10 04:28–04:5X（Asia/Shanghai）
- 目标分支：`two-level-indexer`
- 上次已审查源码基线：`9fe120f7672459194727ff7559640417b7a8c7a5`
- 本次审查源码：`f5f457d4d8fa2b73d7994fcd4b61c599cc86d5b3`
- 非 advice 增量：
  - `8f5837ef0eebd6a1151cd1cd5de9d2bd90a49273`：E119 生产者 YaRN 回执、formal 消费与 128K 历史纠偏；
  - `e7e057ca2f67edee36b9e1f252ee96261a5662a0`：E113 输出路径锁及 `us_per_token` 量纲修复。
- 授权边界：只审查并提交本报告；未修改实现、测试、实验数据或其他分支。

## 结论

E113 的 056 锁生命周期和纳秒到微秒换算在静态实现上对应此前缺陷；但当前干净 CPU 环境缺 `torch`，仓库的 E113“CPU”测试未能执行到用例，故不把静态核验冒充动态通过。

E119 官方新测试在普通 Python 与 `python -O` 下均报告 10/10 通过，但存在三个未被这些测试覆盖的证据闭包问题：formal 能接受与预测字节不同代的回执；只比较 `yarn_enabled/effective_factor` 而忽略回执中其余完整配置；历史纠偏 sidecar 没有任何机器消费或入口引用。前两项会使未来正式结果把“当前同名的两个独立文件”误标成同代生产者证据，第三项使现有 128K 公开 manifest/receipt 继续向机器消费者暴露错误的精确 `2.0` 口径。

| ID | 状态 | 严重度 | 新结论 | 影响范围 |
|---|---|---:|---|---|
| `TL-E119-YARN-RECEIPT-BINDING-059` | **confirmed（formal 成功复现）** | P1，实验身份/正式证据 | 回执在生成循环前发布，且不含预测文件 SHA、行数、run ID 或完成标记；formal 分别冻结当前预测 SHA 与当前回执 SHA，却不验证二者同代。预测文件在回执之后被改写，formal 仍成功并标 `producer_receipt`。 | 尚无证据表明仓库内既有新协议结果已经受污染；但同名重跑、重复派单、生成中断后达到 `min_samples`、并发生成或事后改写时，正式结果可被错误配置标签接收。 |
| `TL-E119-YARN-RECEIPT-CLOSURE-060` | **confirmed（formal 成功复现）** | P2，配置闭包 | 回执声称保存完整 `rope_scaling`、模型 config hash、生成参数和生产脚本 hash，但消费者只保留/比较开关与 factor。把一格 `beta_fast` 改为 999、seed 改为 999、模型与脚本 hash 改为其他值，校验返回合法，整批 formal 仍成功。 | factor 本身可闭合，但不能把 `producer_receipt` 表述扩大成完整 effective-config 或同实现闭包；跨任务/跨臂配置漂移可能不被拒绝。 |
| `TL-E119-YARN-CORRECTION-DISCOVERY-061` | **confirmed（仓库检索 + 产物读回）** | P2，历史结果解释 | 三份 `.manifest.yarn_correction.json` 的 target hash 正确，但没有代码或已发布 receipt/manifest 引用、解析或强制应用它们。原 manifest 与 receipt 仍公开 `yarn_factor=2.0` 且无 provenance。 | 现有 128K 三臂分数不因此改变；但只跟随正式 receipt/manifest 的下游仍会把 2.0 当精确身份，纠偏没有成为机器可达的发布状态。 |

## 1. `TL-E119-YARN-RECEIPT-BINDING-059`

### 违反的契约与位置

`two-level-attention/benchmark/RULER/pred_ruler.py` 在本次 SHA：

- `:171-173` 直接以 `open(..., "w")` 打开并截断最终预测路径；
- `:175-207` 在任何预测行生成前把回执原子替换到最终旁挂路径；
- `:209-249` 才逐行生成并直接 flush 到最终 JSONL；结束后没有计算预测 SHA，也没有以“完成回执”提交该代际。

`two-level-attention/benchmark/RULER/yarn_receipt.py:68-95` 构造的回执没有预测内容 SHA、行数、不可变 generation/run ID 或 `status=complete`。`score_ruler_formal.py:419-451` 仅按 best-file basename 找同名回执，并分别计算其 SHA；`:566-592` 又分别记录预测 SHA 和回执 SHA。两个哈希能证明 formal 当时读到的两个字节串，却不能证明这两个字节串由同一进程、同一次 attempt 共同产生。

### 最小 CPU 复现

复现使用仓库自带 native fixture 和正式 `_formal` 子进程：先为 11 格写回执，再改写一格预测的 `pred` 字段，同时把同格回执的非 factor 配置改成另一组值。formal 返回 0 并打印 `DONE`：

```json
{
  "formal_done": true,
  "formal_returncode": 0,
  "mutated_prediction_sha_in_manifest": true,
  "mutated_receipt_sha_in_manifest": true,
  "receipt_has_finalization_marker": false,
  "receipt_has_prediction_sha": false,
  "run_identity_factor": 2.0,
  "run_identity_provenance": "producer_receipt",
  "validate_mutated_receipt": null
}
```

复现脚本 SHA256：`127841dd1d7de40cf87cf572667affcc12d9cfd274bd2bd591966d7c1457c25f`；原始日志 SHA256：`41f230060ed207e9e7f7b8924d6b4f51dedce86c81d9353f20762978022e05f8`。

这不是要求抵御恶意篡改；正常重复执行同一命名任务也可达：A/B 两进程或一次中断重跑会独立截断/写 JSONL 和替换同名回执，当前没有输出路径锁或 generation 提交点。formal 的 `min_samples` 只限制行数，不能证明终态完整或同代。

### 建议修复与最小重测

采用与 formal 发布协议一致的 generation/commit 语义：在不可变临时 generation 内生成并关闭 JSONL，计算预测 SHA256 与最终行数后，最后写 `status=complete` 回执；回执必须包含 prediction basename/SHA/行数、run/attempt ID、完整配置指纹和生产脚本身份。最终公开入口通过一次原子 pointer/rename 提交，或在“截断预测→生成→最终回执提交”全生命周期持有规范化输出路径锁。formal 必须验证回执中的预测 SHA/行数与所选 best-file 逐位一致，缺失、partial 或不一致即 fail closed。

增加真实双进程/中断测试：同输出 success/success、success/crash、crash/success；在达到 `min_samples` 后、最终行前和“预测关闭后/回执提交前”注入故障。任何可发布结果都必须是一个完整 generation，不能由 A 的预测配 B 的回执。

## 2. `TL-E119-YARN-RECEIPT-CLOSURE-060`

`yarn_receipt.py:78-95` 写入完整 `rope_scaling`、`model_path`、`model_config_sha256`、`generation_params` 和 `producer_script`。但 `validate_producer_receipt():119-161` 只对 task、context、开关、factor 和 `rope_scaling.factor` 做实质校验；`score_ruler_formal.py:442-451` 的摘要直接丢弃其余字段；`_resolve_yarn_identity():641-678` 只比较各格的开关与 factor。

同一复现中，一格回执被改成：

```text
rope_scaling.beta_fast = 999
model_config_sha256 = "a" * 64
generation_params.seed = 999
producer_script.sha256 = "b" * 64
```

`validate_producer_receipt` 仍返回 `None`，11 格 formal 仍成功并标记 `producer_receipt`。因此本修复只能证明“同一 factor 口径”，不能证明注释和 manifest 文案所暗示的完整生产配置一致。

建议为回执定义严格 schema：必需键、类型、64 位十六进制 hash、完整 rope scaling 规范和 source 枚举全部显式校验；计算 `effective_config_sha256`，formal 对同一运行中应相同的字段（模型/config、完整 rope scaling、生产脚本、seed 与共享生成参数）做跨格一致性门禁，对 task、method 等允许变化字段给出明确分组规则。新增“factor 相同但 beta/model config/producer script/seed 不同”负例，普通 Python 与 `-O` 均须拒绝。

## 3. `TL-E119-YARN-CORRECTION-DISCOVERY-061`

三份新纠偏文件的 `target_manifest_sha256` 均与目标 manifest 当前字节匹配，这是正确的；问题在于发布图没有从旧正式入口连到纠偏：

- `rg yarn_correction|correction_version` 只命中源码注释和三份纠偏 JSON，没有消费者；
- `e119_ruler128k_formal_{aavg,fullkv,mavg}.json.manifest.json` 仍是 `yarn_factor=2.0` 且无 `yarn_factor_provenance`；
- 对应 `.receipt.json` 的 `inputs.run_identity` 同样仍是 `2.0` 且没有 correction 引用。

建议不改旧字节，但发布新的、版本化的 corrected manifest/entry receipt，内容绑定旧 manifest hash 与 correction hash，并由一个稳定的最新入口或消费者解析器指向它；机器消费若发现目标旁挂 correction 却仍试图把旧 `yarn_factor` 当 effective，应显式拒绝或返回 `operator_declared_not_effective`。用三份现有 128K 产物做回归：从正式入口解析得到的 effective factor 必须是 `unknown/null`，不得再是 2.0 或未经同代证据支持的 4.0。

## E113 旧发现复查与实际执行

| 项目 | 结果 |
|---|---|
| 056 锁静态复查 | `attempt_lock_path()` 使用 `realpath(out)+.attempt.lock`；`main()` 在隔离旧产物前取 `flock`，并在覆盖执行、成功/失败发布和终检的 `finally` 中释放，机制与 056 修复目标一致。 |
| 纳秒量纲静态复查 | `e113_microbench.py:559-565` 已改为 `med_ns/(1000*T)`，无量纲 speedup 保持同单位比值。 |
| E119 官方测试 | 普通 Python 10/10 PASS；`python -O` 10/10 PASS；生成侧 U3 因环境无 `torch` SKIP。日志 SHA256：`62acc904aad4171e572e536684a317d7bf3890e349a273d73b6834ab7d993382`。 |
| E113 056 双进程测试 | 未执行到用例：`e113_attempt_lock_runner_056.py` 导入 `torch` 失败，0/6；日志 SHA256：`175d963f55c6faaa449e90647322a61c613fba2a950cd39d315893915ce4e497`。这不是锁逻辑失败证据，也不能记通过。 |
| E113 054/055/量纲测试 | 未执行到用例：模块顶层缺 `torch`，日志 SHA256：`a6d56a3f8af473f0aabdcb2e2e79182c1946e81110783cfd9a783f58dafbd545`。 |
| 语法与补丁检查 | 8 个受影响 Python 文件 `py_compile` PASS；`git diff --check` PASS。 |

E113 锁原语测试本身不需要 GPU，但当前 runner 把锁验收与 torch 张量桩绑在一起。为使并发发布门禁在干净 CPU 环境可复验，建议把 `attempt_lock_path/acquire/release` 和最小发布状态机抽成零 torch 模块，至少让锁互斥、崩溃释放与 JSON/sidecar 故障注入用例不因缺 ML 依赖全部失去覆盖；这属于可维护性建议，不是对当前锁实现动态通过或失败的结论。

## 未覆盖边界与重测范围

- 未运行 GPU、Triton kernel、真实 RULER 生成或 E113 延迟；本报告不声明精度或性能变化。
- E113 动态锁/量纲验收被缺 `torch` 阻塞，静态核验不能替代真实双进程和 GPU 冒烟。
- E119 059/060 是 CPU 端正式消费者成功复现；修复后必须重跑官方 F1–F7，并增加同代绑定、完整配置差异和并发/中断负例。
- 现有 128K 三臂分数和相对排序没有因 061 被推翻；需要收窄的是 YaRN factor 的机器可读身份和精确复现声明。
