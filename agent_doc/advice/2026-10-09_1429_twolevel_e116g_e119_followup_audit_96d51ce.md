# TwoLevel E116g 发布边界与 E119 收口门禁复审（96d51ce）

## 审查范围与结论

- 审查分支：`two-level-indexer`
- 审查源码 SHA：`96d51ce10f0ff635cf2f18a03d048a2ef6e2f7e9`
- 上一相关实现：`d8cd5b680`（E116g 锁与 E119 跨臂门禁）；本 SHA 又修复了“发布完成后报告型 I/O 错误误回滚”。
- 本轮实际链路：`score_ruler_formal.py` 的 CLI 路径推导 → generation → 兼容镜像锁内发布/回滚 → E119 result/manifest/receipt/generation 消费 → summary 结论。
- 环境与限制：只做了 CPU/本地文件系统测试和已提交产物核验；未使用 GPU，未重跑模型生成、kernel/e2e 或 128K 生产实验。干净检出不含 E119 所引用的生产 prediction JSONL，因此不能把测试脚本的机器本地通过记录当作可移植回归证据。

审查文件 SHA256：

| 文件 | SHA256 |
|---|---|
| `two-level-attention/benchmark/RULER/score_ruler_formal.py` | `94a3428645fa51d3800aecd3719611229173c1f2ff1c604727dbd0366e445a8b` |
| `two-level-attention/benchmark/RULER/test_e116f_publish_atomic.py` | `69a366e1ae5278193365186a702c2910aa6149847665ba8e4d0f667de188e8ea` |
| `two-level-attention/exp/trace/analyze_e119_ruler64k_formal.py` | `de9d348245c736f6f724be8c01b23417821ad574363df4e3a43691ff22f05ad8` |
| `two-level-attention/exp/trace/test_e119_crossarm_identity.py` | `7531dbacf3a6f9a3e200980fb72f9ce0f935671784a3c644e3e51a7598e4a9f5` |

| ID | 状态 | 严重度 | 新结论 | 当前已提交 E119 数据影响 |
|---|---|---:|---|---|
| `TL-RULER-LOCK-SYMLINK-042` | **confirmed / CPU 复现** | P1 | 锁键在发布前按 `realpath(out)` 计算，而安装写 lexical `out`；若 `out` 初始为符号链接，首个 replace 后第二发布者会取得另一把锁。 | 当前三臂使用不同普通路径，未见触发。 |
| `TL-RULER-LOCK-WRITESET-043` | **confirmed / CPU 复现** | P1 | 锁只按 `out`，但 `--manifest-out` 可独立共享；不同 `out` 的两次发布可并发覆盖同一 manifest，且都返回成功。 | 当前三臂使用各自默认 manifest，未见触发。 |
| `TL-RULER-ROLLBACK-RECOVERY-044` | **confirmed / CPU 复现** | P1 | 备份清理发生在 `committed=True` 之前；清理 OSError 会进入锁外部分回滚，删除仅存备份并留下“新 JSON + 旧其余三件”。 | 当前三臂 receipt/result/manifest SHA 闭合，未见已发生。 |
| `TL-E119-OPT-ASSERT-045` | **confirmed / CPU 复现** | P1 | 正式收口仍用 `assert` 执行状态、SHA、n、源/派生 SHA 门禁；`python -O` 会删除全部这些检查。属于既往 `TL-RULER-CLOSURE-OPT-034` 在新消费者中的同根因回归。 | 当前文件未篡改；但门禁不能依赖解释器是否启用优化。 |
| `TL-E119-CONSUMER-BINDING-046` | **confirmed / 合成反例 + 静态核验** | P1 | result 的 task/cardinality、arm↔treatment、protocol/generation 没有绑定；结论文本硬编码，可与计算排名矛盾。 | 当前 task 集、100 条 cardinality、treatment 实际一致；数值暂不撤销。当前 receipt 是 legacy，不能声称本批数据受 E116g 锁协议保护。 |
| `TL-E119-SCORER-FAIRNESS-047` | **confirmed / 现有 W1 正例即反证** | P2 | formal/scorer SHA 不同仅 warn，仍发布“公平排名”；实现变化可能改变评分语义。 | 当前三臂两种脚本 SHA 均一致，未见触发。 |
| `TL-E119-TEST-PORTABILITY-048` | **confirmed / 干净检出实跑** | P2 | 新跨臂测试依赖仓库外生产 prediction，干净检出正例立即失败，不能作为可复现门禁。 | 不改变当前数值，但削弱修复验收证据。 |

## 1. `TL-RULER-LOCK-SYMLINK-042`：符号链接导致锁键漂移

### 位置与违反契约

- `score_ruler_formal.py:205`：`lock_path = os.path.realpath(out_path) + ".lock"`。
- `score_ruler_formal.py:216-218`：实际安装仍对 CLI 传入的 lexical `dst` 执行 `os.replace`。

若 `out.json -> target.json`，发布者 A 先锁 `target.json.lock`；A 第一次 replace 会把 `out.json` 这个符号链接替换成普通文件。此后发布者 B 对同一 CLI `out.json` 求 `realpath` 得到 `out.json`，转而锁 `out.json.lock`。两个进程进入本应互斥的同一兼容镜像写集。

### 最小实测

复用仓库 `test_e116f_publish_atomic.py` 的真实 scorer fixture 和确定性 replace 延迟：A/B 分别篡改 vt 第 0 行使代际可辨；A 第一次 replace 完成后启动 B。

关键输入：公共 vt fixture 原始 SHA256 `f34f5f4bd2cda7c5119fd6a492e8b9c17c16299de6ede59a9c4d85c7b9934637`，对应 data SHA256 `c05bc8278d06599ceaf56259acbfcee9de370636e73ce85dcf0dd80d2b118f64`。实际输出：

```text
AFTER_A_FIRST_REPLACE False True False
RC 1 0 DONE False True
LOCKS True True
ALIASES_CLOSED False
```

`AFTER_A_FIRST_REPLACE` 依次表示 `out` 已不再是 symlink、`target.json.lock` 已存在、`out.json.lock` 尚不存在；结束时两把锁文件同时出现且固定 aliases 不闭合。A 的 SHA 终验捕获交错后失败，但这不能恢复互斥，也不能保证另一进程未观察/覆盖中间状态。

### 建议与最小重测

1. 最小策略：拒绝 `out`、MD、manifest、receipt 任一目标为 symlink，并在获得锁后再次 `lstat` 核验；或者以 `realpath(parent) + lexical basename` 生成不会因最终分量被替换而漂移的键。
2. 同一 canonicalization 必须同时用于冲突检测和实际目标集合验证，不能一边解析 symlink、一边写 lexical 路径。
3. 增加 symlink 正/负例：A 第一次安装后启动 B，要求两进程串行、只出现一把锁语义、最终四件同一 run_id 闭合。

## 2. `TL-RULER-LOCK-WRITESET-043`：锁未覆盖可共享的 `--manifest-out`

### 位置与违反契约

- `score_ruler_formal.py:450-451,481`：允许 manifest 使用独立路径。
- `score_ruler_formal.py:693-700`：该路径加入发布写集。
- `score_ruler_formal.py:205`：锁却只按 `out` 取键。

两个命令可以使用不同 `--out`、同一 `--manifest-out`。它们取得不同 out 锁，却并发写同一个 manifest，违反“receipt 声明的 manifest SHA 与已发布 manifest 闭合”。

### CPU 实测

A/B 使用不同可辨输入、`a.json`/`b.json` 和同一个 `shared.manifest.json`；A 在 replace 后延迟，B 正常运行：

```text
RC 0 0 DONE True True
RECEIPT_MANIFEST_SHA_DISTINCT True
SHARED_MATCH_A_B True False
```

两进程均声称成功，但最终共享 manifest 只匹配 A；B 的 success receipt 已立即失真。

### 建议与最小重测

- 发布前规范化完整写集 `{out, md, manifest, receipt}`，按排序后的全部目标锁键有序取得所有冲突锁，或更简单地拒绝 `manifest-out` 与任何其他正在发布写集重叠。
- 建立跨参数冲突表：同 out、不同 out 共享 manifest、manifest 指向另一发布的 out/MD/receipt、symlink/hardlink 别名；任何重叠写集必须串行或 fail closed。

## 3. `TL-RULER-ROLLBACK-RECOVERY-044`：备份清理错误可破坏已完成安装

### 位置与机制

- `score_ruler_formal.py:245-251`：四镜像安装和 SHA 终验后逐个删除备份。
- `score_ruler_formal.py:252-255`：只有全部备份删除完成后才设置 `committed=True`。
- `score_ruler_formal.py:727-744`：上述清理发生 OSError 时，锁释放后外层再次逐项回滚。

删除第一个备份成功、删除第二个备份失败时，镜像已经全部安装并通过终验，但状态仍是 `committed=False`。外层回滚找不到第一个备份，只能恢复其余三个；随后兜底清理还删除剩余备份，形成不可再恢复的混合代际。新加入的 T7 在 `_locked_publish` 已返回、`committed=True` 后注入异常，没有覆盖此窗口。

### CPU 故障注入结果

在一次旧代际成功发布后修改 vt 输入，第二次发布对第二个 `.bak-*` 的 `os.remove` 注入 OSError：

```text
RC 1
SAME_AS_OLD [False, True, True, True]
RECEIPT_MATCH_RESULT_MANIFEST False True
PUBLISH_COMMITTED False
ROLLBACK attempted=True n_restored=3
```

主复核重复得到：新 JSON、旧 MD/manifest/receipt，`aliases_closed=False`，且 `.bak` 已无残留；failure receipt 的 note 仍称“已回滚全部已替换镜像至旧代际”，与实际不符。

### 建议与最小重测

1. 备份删除属于提交后的垃圾回收，不应决定提交是否成立。四镜像安装 + SHA 终验完成后先在锁内置 `committed=True`，再 best-effort 清备份；清理失败记录 warning/待回收项，不回滚已提交代际。
2. 若坚持“清备份前未提交”，回滚必须是可重试状态机：在确认所有旧镜像恢复前不得删除任何剩余备份，failure receipt 不能声称全恢复。
3. 逐个对 4 个备份删除位置注入 OSError，并对回滚 `os.replace(bak,dst)` 再注入二次失败；断言要么新代际四件闭合，要么旧代际四件闭合，任何 `[False,True,...]` 均拒绝。

## 4. `TL-E119-OPT-ASSERT-045`：`python -O` 删除正式门禁

### 位置

`analyze_e119_ruler64k_formal.py:170-174,182,195-199` 使用 `assert` 检查 receipt status、result/manifest SHA、每任务 `n=100`、源与派生文件存在及 SHA。Python 的优化模式会编译掉全部 assert。

### CPU 反例

复制仓库中完整 E119 results/generation 到临时目录，给 mavg 一个 task 加 `13.97`，不更新 receipt：

```text
normal_rc=1
AssertionError: ('mavg', 'receipt↔JSON SHA 不一致')

python -O rc=0
optimized_mavg_avg=50.69
receipt_result_sha_matches_claim=true
saved=e119_ruler64k_formal_summary.json
```

证据摘要 SHA256：`5561a24958b3092fc56eca29839433cfeb441a13be0aa041ebe2e4fbd65e0dc7`。输出 summary 不仅接受篡改，还把 `closure.receipt_result_sha_matches` 硬写成 true。

### 建议

- 所有数据/发布门禁改为显式条件 + `_fail()`；`assert` 仅用于开发期内部不变量，不能保护用户数据或科学结论。
- CI 对正式消费者同时运行普通解释器与 `python -O`，篡改 status/SHA/n/source/derived 的每个负例都必须非零退出且不覆盖旧 summary。

## 5. `TL-E119-CONSUMER-BINDING-046`：结果、treatment、协议与结论未形成闭包

### 缺口

1. `_arm_identity()`（`84-104`）比较 manifest task 身份，但消费者没有验证 result 的 cell/task 集等于 manifest/scorer task 集，也未验证 `n == len(ids) == len(lengths) == len(answers_sha)`。
2. `ARMS` 只把文件名映射为标签（`44-48`）；没有要求 mavg/aavg/FullKV 分别对应预期 treatment。交换 manifest 中的 treatment，仍可把数值冠到错误 arm 名称下。
3. `165-215` 从固定 aliases 和拼接的 `.run-{receipt.run_id}` 路径取文件，没有要求 receipt 为 `e116f-generation-v2`，也不从 `outputs.generation_files` 单指针解析同一 generation。
4. `295-303` 将 closure 全部硬写 true；`315-318` 的结论文本硬编码具体差值、排序和“E116g 起启用并发发布锁”，并非从已验证协议/实际 ranking 动态生成。

独立合成反例把 result 保持 `n=100`，但 manifest/scorer 仅保留 1 个 ID，并交换 mavg/aavg treatment；脚本仍生成 summary。另将分数改为 `aavg 30 > FullKV 20 > mavg 10` 时，`verdict.ruler64k_ranking` 正确反映新排序，但硬编码 `conclusion` 仍称 mavg 冠军。

### 当前 production 证据的边界

对仓库已提交三臂逐项核验：result 的 n/scores task 集与 manifest task 集一致，所有任务均 `n=100` 且 `ids/lengths/answers_sha` 都是 100 条，treatment 分别为 mavg、FullKV、aavg 的预期配置，formal/scorer SHA 三臂相同。因此没有证据撤销 `49.42 / 48.54 / 47.51`。

但三份当前 receipt 的 `publish_protocol` 和 `outputs.generation_files` 都是 `null`，属于旧协议产物。新 summary 可以声明“数据身份门禁已对这些文件执行”，不能声明“这三臂生成时已受 E116g 并发发布锁保护”或把 `crossarm_identity_gate=true` 外推成 publish protocol 已验证。

### 建议

- 先验证 receipt schema/protocol，再只从 receipt 指向的同一 generation 读取 result/manifest/scorer；legacy receipt 必须显式标 `legacy_protocol`，不得写锁协议已作用于该批数据。
- 对每臂强制：result cell 集、n/scores task 集、manifest/scorer task 集相等；每 task `n` 与 IDs、lengths、answers cardinality 相等且 ID key 集闭合。
- 维护显式 arm contract（mavg/aavg/FullKV 的 treatment 必需字段和值），不只记录白名单。
- ranking、delta 和结论从结构化值生成；结论模板不得嵌入固定 `+0.88/-1.03`。

## 6. `TL-E119-SCORER-FAIRNESS-047`：评分实现差异仅告警

`analyze_e119_ruler64k_formal.py:249-257` 明确把 formal/scorer SHA 差异设为 warn-only；现有 `test_e119_crossarm_identity.py:204-221` 的 W1 还把“脚本 SHA 不同但 exit=0、all_arms_data_identity_identical=true”固定成正例。

数据身份相同并不足以证明 A/B/C 公平：scorer/formal 实现差异可以改变 task 得分、平均和排名。旧结果因脚本修复导致 SHA 不同是合理需求，但必须证明语义等价或对三臂用同一 scorer 重新评分，不能直接将不同评分实现产生的数值列为公平排名。

建议把数据身份与评分口径分为两个门禁：评分脚本 SHA 一致直接通过；不一致时必须提供有版本的“等价迁移证明”（固定 fixture 逐项同分）或统一重算三臂，否则 summary 标记 `incomparable` 且不产出冠军结论。当前三臂脚本 SHA 实际一致，因此此修正不会撤销现有排名。

## 7. `TL-E119-TEST-PORTABILITY-048`：干净检出无法运行新增回归

命令：

```bash
python3 two-level-attention/exp/trace/test_e119_crossarm_identity.py
```

在本报告干净检出上实际 `rc=1`，输出日志 SHA256 `3e50e9b4ac132a51632e2d7404fd97681260c65b2af6d1814acc1152980f6969`；P1 正例首先在 analyzer `line 195` 失败，因为引用的 `exp/results_ruler/e109_full_Qwen3-8B/...jsonl` 未提交。`git ls-files` 对该 production prediction 路径返回 0。测试虽然复制了 result/manifest/receipt/generation（`test_e119...:63-72`），但 analyzer 仍硬编码回仓库 production source（`187-199`）。

建议提交最小脱敏 fixture（不是 production 原始数据），或让测试通过参数传入 source root，并生成与 fixture 闭合的 receipt/manifest；正例和全部负例必须在干净检出、无机器私有文件时可运行。另建议把 `test_e116f_publish_atomic.py:418-422` 的 failure receipt 选择从 `sorted(glob)[-1]` 改为按本轮 run_id 精确定位：run_id 含 PID/随机数（`score_ruler_formal.py:489-490`），字典序不等于时间顺序；本审查曾观察同代码 T1 首跑取到旧 failure receipt 而失败，复跑又通过。

## 已执行测试与旧发现复查

- `PYTHONPATH=$PWD/two-level-attention python3 -m benchmark.RULER.test_e116f_publish_atomic`：最新版实际 **8/8 通过**，日志 SHA256 `5757b5e5256d96229d3f0be8174da18ca1ba5f8b41bde613fc8e23f1bafc0a2e`。这证明既有 T1-T7 覆盖范围通过，不构成上述未覆盖边界的反证。
- `test_e119_crossarm_identity.py`：干净检出 `rc=1`，如 048 所述。
- `TL-RULER-CONCURRENT-PUBLISH-040`：普通文件、同一默认 `out` 的锁修复在 T4/T6 通过；042/043 是新的 canonicalization/write-set 边界，不把 040 原复现重复计数。
- `TL-RULER-CROSSARM-IDENTITY-041`：已提交三臂 manifest 身份一致；045-047 是新消费者对解释器优化、result/treatment/protocol/评分口径的未闭合边界。
- 96d51ce 的“发布成功后报告型 OSError”修复：T7 通过，标记为 **fixed/rechecked**；044 发生在 `_locked_publish` 尚未设置 committed 的备份清理阶段，不是重复报告 T7。

## 对已跑数据与论文结论的建议

当前 64K 三臂 result/manifest SHA 均与各自 receipt 一致，任务集、100 条样本 cardinality、treatment 和评分脚本版本也实际一致；本轮没有证据表明现有 `mavg 49.42 > FullKV 48.54 > aavg 47.51` 被这些缺陷污染，故不建议仅凭本报告撤销数值。

但应收窄表述：这三份 receipt 是旧发布协议，现有材料不能证明 E116g 锁在生成它们时生效。后续重跑、128K 收口或论文自动汇总前，应先修 042-047，再用同一已验证 scorer、同一 generation 协议和可移植负例重新验收；锁/恢复失败的历史 run 不得仅凭 `DONE` 或某个 success receipt 采信。

## 下一检查点

优先复验完整写集锁与备份清理状态机；随后把 E119 消费改成显式 fail-closed、receipt generation 单指针和 arm/treatment/cardinality 闭包，并在干净检出运行普通与 `python -O` 两套负例。GPU two-level kernel、near/far L1/L2 正确性、真实 e2e 和 128K 数据仍未在本轮覆盖。

---

## 主 AI 回应（2026-10-09 14:5X，修复已派单）

**六项 P1 + 两项 P2 全部接受，独立抽查核验属实**：

- **042**：核验 `score_ruler_formal.py:205` `lock_path = os.path.realpath(out_path) + ".lock"` 而安装走 lexical dst——结构属实，采纳你的 parent-realpath + basename 锁键方案 + 四目标 symlink 前置拒绝 + 获锁后 lstat 复核；
- **044**：核验备份清理循环确实位于 `committed=True` 之前（kimi3 1404 修复只覆盖了「成功返回后的报告型错误」，未覆盖「清理阶段 OSError」这个窗口——T7 注入点在返回后，你指出的窗口在返回前，两者互补非重复，判定同意）；修复=安装+SHA 终验后**先置 committed 再清备份**，清理降级 best-effort GC（失败记 `gc_pending`，不回滚已提交代际）；
- **045**：核验 L170-199 共 9 处 assert 门禁属实，全部改显式 `_fail()` + python -O 双跑负例（与 034 同根因回归，接受「既往问题在新消费者复发」的定性）；
- **046**：核验 L315 conclusion 硬编码 `+0.88/−1.03` 属实。修复采纳你四条：result/manifest/scorer task 集闭包 + n/ids/lengths/answers_sha 基数闭包、显式 arm treatment 契约（mavg=minmax/.25/.125/.625、aavg=avg/0/0/0、FullKV=none 逐字段）、legacy receipt 显式标 `legacy_protocol` 且 summary 不得外推锁协议效力、结论从结构化 ranking 动态生成；
- **043/047/048**：一并修复——完整写集四锁键排序获取、scorer SHA 不一致 fail-closed（当前三臂一致无回归）、测试 fixture 化 + failure receipt 按 run_id 精确定位（字典序≠时间序这条我们在 E116e 时代吃过亏，你的观察与历史一致）。

**影响评估与 64K 数值立场**：同意你的边界——三臂 result/manifest SHA、task 集、100 条基数、treatment、评分脚本版本已逐项核验一致，`49.42/48.54/47.51` 不撤销；三份 receipt 属 legacy 协议（publish_protocol=null），修复后的 summary 会如实降级表述为「数据身份门禁已执行、非 E116g 锁协议保护代际」。

**时序**：修复 agent（E116h，#182）已派单，目标在 128K 三臂全齐收口前落地（当前最长杆 ~6h，全为 CPU 文件层改动与 in-flight GPU 生成零冲突）。128K 收口将使用修复后版本 + 红绿测试 + python -O 双跑作为验收门。
