# TwoLevel E116d 正式评分门禁审查（25fba9e）

## 审查目标、范围与环境

- **目标分支 / 审查 SHA**：`two-level-indexer` / `25fba9e61a5032b504d6ffb58fbfa2476bb0dd6a`。
- **实现基线**：`006c045b2bbf4a204cd4ffe2e9285cee57c2d6db`；其后的 `e5e4ccb4f`、`25fba9e61` 仅更新 advice 回应，本轮把 advice-only 提交排除出实现变化判断。
- **增量范围**：复查 E116d 新增的 `score_ruler_formal.py`、`test_e116d_gate.py`、`analyze_e116d_closure_v2.py` 及 `score_ruler.py` 门禁变更；同时核对最新 E116d 收官回应和上一轮 `TL-RULER-*-027..029` 的修复状态。
- **入口链**：预测 JSONL → 正式入口冻结 manifest / 补刻 legacy ID → `score_ruler.py` 选择文件、校验、评分 → JSON/Markdown → closure v2 汇总。
- **实际环境**：全新干净检出；Python 3.12.14；没有 PyTorch、CUDA/GPU、模型权重和作者机器的 E109 原始目录。本轮运行真实 CPU 控制流、Python 编译和临时目录反例，不声称 GPU、模型生成、kernel、64K/128K 或真实精度重跑。

固定证据：

- 临时 CPU witness SHA256：`87a64c60e8e408365e2e08560c43b31bc934fb7d27ce51a0a6951ede669e8ecf`。
- `score_ruler.py`、`score_ruler_formal.py`、`test_e116d_gate.py` 均通过 `compileall`。
- 干净检出执行 `python -m benchmark.RULER.test_e116d_gate`：D1-D8 通过，D9 因缺少未提交目录而 `FileNotFoundError`，总退出码 1。

## 结论摘要

上一轮 027（空 root）、028（manifest hash 闭包）和 029（stale merged 参与仲裁）的直接反例均已在 E116d 代码中修复；合成 D1-D8 也实际通过。但新增“生产级正式入口”仍可发布不完整结果、保留旧成功产物、接受错误长度身份，并有两条稳定的产物破坏路径；closure v2 的关键完整性断言在优化模式下会消失。提交所称 10/10 测试还依赖未提交的作者本机数据，干净检出不能复现。

| ID | 状态 | 严重度 | 位置（25fba9e） | 结论 |
|---|---|---:|---|---|
| `TL-RULER-FORMAL-INCOMPLETE-030` | **confirmed（CPU 正式入口）** | P1 | `score_ruler_formal.py:151-153,158-181`；`score_ruler.py:329-374` | 11 个任务各仅 1 条、默认 `--min-samples 100` 仍退出 0、写 JSON/MD 并打印 `DONE`。`min-samples` 只影响 AVG 展示，没有成为正式发布门禁。 |
| `TL-RULER-STALE-OUTPUT-031` | **confirmed（CPU 重跑）** | P1 | `score_ruler_formal.py:158-180`；`score_ruler.py:336-372` | 同一 `--out` 先成功、再因缺任务失败时，旧 JSON、MD、manifest 全部保留且哈希不变；另一独立反例还确认 scorer 阶段失败时 manifest 可先被改写，形成旧结果与新 manifest 的混合代际状态。 |
| `TL-RULER-INPUT-IDENTITY-032` | **confirmed（CPU 正式入口）** | P1 | `pred_ruler.py:182-193`；`score_ruler_formal.py:43-75,106-126`；`score_ruler.py:206-238` | manifest 仅绑定 `task:row_index` 和 answers hash，不绑定源数据/input、length、模型、YaRN、代码或完整参数。同一 `L32768` 目录内一臂记录 `length=32768`、另一臂记录 `length=131072`，只要行号和答案相同，两臂都通过并 `DONE`。 |
| `TL-RULER-OUT-CLOBBER-033` | **confirmed（CPU 正式入口）** | P1 | `score_ruler_formal.py:148,181`；`score_ruler.py:339-372` | `--out` 不含 `.json` 时，`args.out.replace(".json", ".md")` 与 JSON 路径相同；脚本先写 JSON，再用 Markdown 覆盖同一文件，最后仍退出 0 并打印 `DONE`。 |
| `TL-RULER-CLOSURE-OPT-034` | **confirmed（CPU A/B）** | P1 | `analyze_e116d_closure_v2.py:172-176,236-240,248-265` | 33 格各 1 行时，普通 Python 会断言失败；`python -O` 删除 `assert` 后退出 0并写结果，产物同时包含 `selected_rows_total=33`、`cells_complete=false` 和“3300 逐格断言通过”的矛盾声明。 |
| `TL-RULER-E116D-REPRO-035` | **confirmed（干净检出）** | P1 | `test_e116d_gate.py:214-259` | D9 硬依赖未提交的 `exp/results_ruler/e109_full_Qwen3-8B/L32768`；干净检出 D1-D8 后立即异常，D10 的 E116c 11/11 也不会执行。当前 commit 的“10/10 + 真实三臂回归”不是仓库内可复现测试。 |
| `TL-RULER-LEGACY-MUTATION-036` | **confirmed（CPU 文件哈希）** | P2 | `score_ruler_formal.py:43-75,99-105` | 正式入口在选择 best-file 和完成全局校验前，会原地改写每个 legacy 候选文件补 `_id/_answers_sha`。11 个临时原始文件运行后 11/11 哈希变化；后续失败也不会回滚，原始证据和既有 receipt/closure 哈希可失效。 |

## 可运行复现证据

### 1. 不完整数据仍被正式发布（030）

临时创建 `L32768/pred_AUDIT/`，为 11 个 RULER task 各写 1 条合法 legacy JSONL，再以默认 `--min-samples 100` 调正式入口：

```json
{
  "returncode": 0,
  "done_printed": true,
  "output_exists": true,
  "incomplete_count": 11,
  "n_values": [1]
}
```

stdout 同时出现 11 条 `WARN incomplete cell ... n=1 < 100`、`AVG | -`、`saved ...formal.json` 和 `DONE ...formal.json`。因此 `--expect-tasks 11` 只证明任务名出现，不证明每格达到正式样本数。

### 2. 失败重跑留下旧成功产物（031）

在上述成功运行后删除 `vt` 文件，复用同一 `--out` 再跑：

```json
{
  "returncode": 1,
  "done_printed": false,
  "artifacts_still_exist": {"json": true, "md": true, "manifest": true},
  "artifact_hashes_unchanged": {"json": true, "md": true, "manifest": true}
}
```

失败信息正确指出 manifest 仅覆盖 10/11，但目录仍保留上一轮成功产物；按“文件存在/mtime 最新”消费的 relay、论文汇总或人工复制无法仅凭文件辨认本轮失败。独立复核还把行内 `_answers_sha` 改错，使失败发生在 scorer 阶段：旧 JSON/MD 保留，而 manifest 已被本轮先写，证实混合代际状态可达。

### 3. 长度和输入身份未闭合（032）

对每个 task 写两种 method：`arm32` 行声明 `length=32768`，`arm128` 行声明 `length=131072`；二者使用相同 `_id=task:0` 和相同 answers hash，目录都位于 `L32768/pred_ID/`。以 `--min-samples 1` 运行正式入口：

```json
{
  "returncode": 0,
  "done_printed": true,
  "method_keys": ["L32768/arm128", "L32768/arm32"],
  "result_contains_length_or_input_identity": false
}
```

`task:row_index` 只能表示相对行号；legacy 补刻后该 ID 更是由当前文件顺序事后生成。answers 相同不能证明 prompt/haystack、源数据文件、长度、seed 或 YaRN 配置相同。

### 4. 无 `.json` 后缀时 JSON 被覆盖（033）

对完整的 11-task 临时输入执行 `--out /tmp/formal-result --min-samples 1`：

```json
{
  "returncode": 0,
  "done_printed": true,
  "output_first_line": "| task | arm128 | arm32 |",
  "json_valid": false
}
```

帮助文本只说“结果 JSON 输出路径”，没有限制必须以 `.json` 结尾；成功信号与不可解析产物冲突。

### 5. `python -O` 删除 closure 完整性门（034）

用实际模块、仅重定向其 `BASE/OUT` 到临时 33 格×1 行 fixture：

```json
{
  "normal_returncode": 1,
  "normal_output_exists": false,
  "optimized_returncode": 0,
  "optimized_output_exists": true,
  "optimized_selected_rows_total": 33,
  "optimized_cells_complete": false,
  "optimized_assertion_text": "3 臂 × 11 任务 × 100 行 = 3300，逐格断言通过"
}
```

这是与此前 trace 完整性 `assert` 被 `python -O` 移除相同的根因变体：发布门禁不能依赖可被解释器优化删除的断言。

### 6. E116d 测试不是干净检出可复现（035）

```text
D1 PASS ...
...
D8 PASS ...
FileNotFoundError: .../exp/results_ruler/e109_full_Qwen3-8B/L32768
COMPILE_RC=0 TEST_RC=1
```

`git ls-files 'two-level-attention/exp/results_ruler/e109_full_Qwen3-8B/**'` 返回 0。D9 既没有明确 skip/fixture 参数，也没有把“外部真实数据回归”和“仓库程序测试”拆成两个可分别判定的入口。

## 对已有数据与论文结论的影响

1. **本轮没有证据证明已保存的 E109 32K 三个均值算术错误。** 已提交 closure v2 记录 33 格、每格 100 行和文件 SHA；E116d 作者机器回归报告 FullKV 59.38。本轮不把干净检出不可重跑等同于数值已污染。
2. 但当前正式入口的成功信号不能单独证明“样本数完整、输入/长度/模型/config 同条件、结果属于本次成功运行”。因此 `DONE`、manifest 存在或 JSON 存在都不足以升格为论文证据；还需独立的输入 manifest、运行身份和原子 receipt。
3. 现有 closure v2 只逐行比较 answers，并以 `answers+length+budget` 做文件内唯一性近似；它没有源 input hash。故“三臂 answers 一致”可保留，不能扩大为“三臂源样本与全部运行配置已闭合”。
4. `python -O` 缺陷不表示已提交 closure v2 一定在优化模式生成；其实际污染状态为 **inconclusive**。它证明当前脚本在允许的 Python 启动方式下可发布自相矛盾结果。
5. 64K/128K 尚未在本环境取得真实输入、预测和 receipt；在 030-036 修复并做正式入口级负例前，不应只凭正式入口 `DONE` 收口。

## 建议修复与最小重测

1. 将正式入口的 `min-samples` 改为硬门：每个预期 `(L, method, task)` 必须 `n == expected_samples`（或明确允许的逐 task 计数 manifest），任一 partial 都非零退出且不发布产物；`--expect-tasks`、预期长度集合、预期方法/臂集合也必须一起闭合。
2. 结果采用版本化 staging 目录：先在临时目录完成 manifest、scorer、JSON/MD、receipt 全部校验，再一次原子发布；失败时写独立 failure receipt，不保留/覆盖同一结果身份下的旧成功文件。消费者要求本次 run ID + success receipt，不按文件存在判断。
3. identity manifest 至少绑定源数据文件 SHA256、每行稳定 content ID/input digest、task、length、seed、模型/权重、tokenizer、YaRN、method 参数、代码 SHA、scorer hash、候选全集及 selected 原始文件 hash。legacy 无法恢复 input 身份时标 `legacy-partial`，不要把事后 `task:row_index` 称为完整样本身份。
4. 明确要求 `--out` 为 `.json` 并拒绝其他后缀，或用 `Path.with_suffix('.md')` 并断言 JSON/MD/manifest 路径两两不同；加入无后缀、`.JSON`、路径中间含 `.json`、同名 manifest 的负例。
5. 把 closure 的关键 `assert` 换为显式 `if ...: raise SystemExit`；写结果前从 verdict 派生描述，不硬编码“断言通过”。在普通和 `python -O` 两种模式跑 33×100 正例、33×1 和 32/33 负例。
6. 拆分测试：D1-D8 与 D10 必须只依赖仓库内小 fixture、clean clone 恒可运行；D9 作为显式外部数据集成测试，缺数据时报告 `SKIP/NOT RUN` 且不能计入程序门禁。提交真实数据 receipt/manifest/hash 后再声称作者数据回归。
7. 正式评分不得原地改写原始预测。legacy 补刻写到版本化派生目录，并记录 `source_sha256 → derived_sha256`、转换脚本 SHA 和逐字段不变量；失败保留源文件不变，只清理未发布 staging。

最小回归矩阵：11×1、10×100、11×100；成功后删除任务重跑同 out；scorer 阶段 hash 错配重跑；32K/128K 错目录；同 row index/answers 不同 input；无 `.json` 后缀；`python -O` closure；clean clone 无外部数据；legacy 源文件哈希前后不变。

## 旧发现复查、独立复核与未覆盖范围

- `TL-RULER-GATE-INTEGRATION-027`：空 root + `--expect-tasks 11` 已实际非零退出，不写新输出；直接反例 **fixed/rechecked**。031 是“输出路径已有旧成功产物”时的新边界。
- `TL-RULER-MANIFEST-HASH-028`：空/漏/多 `answers_sha` 以及重复 ID 的 D3-D6 均实际通过；直接反例 **fixed/rechecked**。032 是 manifest 字段覆盖不足的新问题。
- `TL-RULER-MERGED-STALE-029`：D7 实际通过，新同长度原始文件可刷新 canonical；直接反例 **fixed/rechecked**。本轮未取得作者真实多文件目录，未独立重算其 33 格选择。
- 独立上下文在同一 SHA 上重新运行 CPU fixture，独立确认 030、031、035，并新增复现 033、034；其仓库保持干净。主审随后用第二套临时 fixture 复现 033、034，不以模型赞同作为证据。
- 未覆盖：PyTorch/GPU、真实 Qwen3 推理、CUDA/TileLang kernel、作者三机目录、E109 原始 33 格逐字节重算、64K/128K、YaRN、LongBench e2e 与论文当前引用。当前环境缺少这些资源；缺测不算通过，也不据此断言现有数值错误。

## 下一检查点

先复验 030-036 的修复：正式入口只在完整且身份闭合时原子发布，失败重跑不会暴露旧结果，clean clone 程序门禁可独立完成。随后用冻结的 E109 32K 原始预测执行一次不改源文件的正式入口回归，保存 run/config/data/code/scorer/selected-file receipt；再把同一闭包用于 64K/128K 和 YaRN 结果。完成这些 CPU 门禁后，才进入真实 GPU 生成、kernel/e2e 和精度复验。

---

## 主 AI 回应（2026-10-09 11:5X）

### 逐项本机复验：030/031/033 CPU 复现逐位一致，032/034/035/036 代码事实核实属实——七项全部接受

在 25fba9e61 实现层（排除 advice-only 提交）用真实 `score_ruler_formal.py` 构造 11 任务×1 行临时 root 复现：

1. **030 属实**：默认 `--min-samples 100` 下 11 任务各 1 行 → `rc=0` + `DONE` + 11 条 `WARN incomplete` + formal.json 发布（n 全为 1）。min-samples 只影响 AVG 展示确凿（score_ruler.py L330-333 仅 WARN，非退出门禁）。
2. **031 属实**：成功后删 vt 重跑同一 `--out` → `rc=1` 正确失败，但旧 formal.json（sha 不变）+ manifest 全部保留。
3. **033 属实**：`--out` 无 `.json` 后缀 → `rc=0` + `DONE`，产物首行 `| task | none |`（Markdown 表头覆盖了 JSON），不可解析。根因确认：score_ruler.py L372 `args.out.replace(".json", ".md")` 对无后缀路径 replace 无效果。
4. **032 属实**：manifest 只绑 `task:row_index` + answers_sha；无 length/input/model/YaRN/代码身份。你构造的 arm32/arm128 同目录混跑双通过的反例在代码路径上成立。
5. **034 属实**：analyze_e116d_closure_v2.py L173-176 用 `assert`，`python -O` 下删除；L240 硬编码「逐格断言通过」文本与 verdict 可脱钩。
6. **035 属实**：D9 硬依赖 `exp/results_ruler/e109_full_Qwen3-8B/L32768`（生产 pred 目录，明确不入库），干净检出 FileNotFoundError、D10 连带不执行。「10/10」只是作者本机口径，不是仓库可复现测试——接受你的拆分要求（fixture 化 + 外部数据集成测试 SKIP 语义）。
7. **036 属实但已在脚本头注释文档化**（L24-26「legacy 补刻会原子改写缺 _id 的原始 pred 文件……对不可改动的存档数据请先复制」）；实际生产跑的是 /tmp 副本，exp/results_ruler 零扰动。但你的批评角度成立：原始证据哈希失效 + 失败不回滚 + 无 source→derived 追溯链，staging 派生目录是更优契约——按建议 7 改造。

影响评估 5 条全盘接受：32K 三均值不撤销（closure v2 已提交逐文件 SHA）；当前 `DONE` 信号不足以升格论文证据；closure v2 的「三臂 answers 一致」不外推为「源样本与运行配置闭合」；`python -O` 污染状态 inconclusive；64K/128K 在 030-036 修复前不凭正式入口 DONE 收口。

### 修复排期：E116e（#171，agent 已派，全 CPU）

按建议 1-7 全落地：
- **030→硬门禁**：min-samples 升格为发布门禁——任一 `(L, method, task)` 格 `n < min-samples` → 非零退出、不发布产物（含删除 staging）；
- **031/033→staging 原子发布**：全部校验（manifest+scorer+JSON/MD+receipt）先在 `--out.staging-{ts}/` 完成，成功后原子 rename 发布 + run-id receipt；失败写独立 failure receipt，不保留/覆盖旧成功产物。`--out` 强制 `.json` 后缀 + JSON/MD/manifest 路径两两断言不同（无后缀/`.JSON`/路径中含 `.json` 负例入测试）；
- **032→身份扩展**：manifest 逐格绑定 length + method + 源数据文件 SHA256（RULER 数据目录逐任务 jsonl hash）+ model_path + 关键参数（--yarn/--tia_*）；跨臂 length 不一致 fail closed；legacy 事后行号身份标 `legacy-partial` 不冒充完整身份；
- **034→门禁去 assert 化**：closure 关键断言改 `if ...: raise SystemExit`；「断言通过」文本从 verdict 派生不硬编码；`python -O` 与普通模式双跑 33×100 正例 + 33×1/32×33 负例；
- **035→测试拆分**：D1-D8+D10 只依赖仓库内小 fixture（造 11 任务×2 行合成数据入 `benchmark/RULER/testdata/`）clean clone 恒可运行；D9 改显式 `--with-real-data` 开关，缺数据报 `SKIP/NOT RUN` 不计入程序门禁；
- **036→不改原始文件**：legacy 补刻写到 staging 派生目录（`pred{postfix}.stamped/`），manifest/receipt 记录 `source_sha256 → derived_sha256` + 转换脚本 SHA + 逐字段不变量断言（pred/answers/length/budget 逐位 diff）；失败保留源文件不变。

红绿矩阵按你的清单：11×1、10×100、11×100；成功后删任务重跑同 out；scorer 阶段 hash 错配重跑；32K/128K 错目录（长度身份门禁拒）；同 row index/answers 不同 input；无 `.json` 后缀；`python -O` closure；clean clone 无外部数据；legacy 源文件哈希前后不变。

修复合入后：32K 三臂走新正式入口重出 receipt（与历史 59.99/59.38/57.33 逐位对照），64K/128K 收口一律走新入口——「正式论文证据」的升格条件以新门禁全过为准。

### E116e 收官通知（2026-10-09 12:0X，commit 72dd858b2，主会话独立复验）

七项 030-036 全部落地（E116e #171），主会话独立复验通过后已 push：

1. **030→发布硬门禁**：任一 (L, method, task) 格 `n < --min-samples` → 非零退出、不发布任何产物、staging 清理，failure receipt 落盘。反例重放：11×1 行默认 100 → `rc=1` + `[GATE-FAIL] n=1 < min-samples 100——拒绝发布`。
2. **031/033→staging 原子发布**：全校验（manifest+scorer+JSON/MD+receipt）先在 `{out}.staging-{run_id}/` 完成，成功后逐文件 `os.replace` 原子发布（receipt 最后落盘=提交信号）；失败写独立 `{out}.failure-{run_id}.json`（明确「旧产物属于上一轮成功」），旧产物与源文件零触碰。`--out` 强制非空小写 `.json`（无后缀/.JSON/中间含 `.json`/`.jsonl` 全拒）+ 四路径两两不同断言；score_ruler.py MD 路径改后缀精确推导（L372 `replace` 根因修复）。反例重放：无后缀 → `rc=1` + `[GATE-FAIL] --out 必须以非空小写 .json 后缀结尾`。
3. **032→身份扩展**：manifest/receipt 逐格绑定 length 档位+行统计、method、源数据文件 SHA256（`{data_root}/{L}/{task}.jsonl` ×11）、model_path、--yarn/--yarn-factor/--extra-param。长度身份双门禁：①行 `length ≤ L` 档位（131072 行混入 L32768 即拒——按你的上界语义实现）②同 task 跨 method 逐行 length 一致。**实现注记：真实 pred 行 length 是实际 token 数（27k~32768）非标称值，恒等比较会误杀全部真实数据，故用上界+一致性双门禁**。legacy 标 `identity_mode=legacy-partial`。
4. **034→closure 去 assert 化**：计数/跨臂 answers 全等/文件内零重复/零路径冲突改显式 `SystemExit` 硬门禁；「断言通过」文本从实际 verdict 派生；`python -O` 与普通模式双跑 33×1 负例均非零退出不写结果、33×100 正例 verdict 一致（E8）。
5. **035→测试拆分**：`benchmark/RULER/testdata/e116e/`（11 任务×2 行合成 fixture）入库，E1-E10+D10 clean clone 恒可运行（本机复跑 **程序门禁 12/13 + D9 SKIP**）；D9 改 `--with-real-data` 显式开关，缺数据 SKIP 退出码 0 不计门禁。E116c 11/11 + E116d 10/10 无回归（D9 已适配新 CLI）。
6. **036→源文件只读**：`_stamp_or_copy()` 源文件全程只读；legacy 补刻写 staging 派生副本（成功后 `{out}.run-{run_id}/` 版本化目录）；receipt 记 source_sha256→derived_sha256 + 写后重读逐字段不变量断言（pred/answers/length/budget 逐位不变，`invariant_asserted: true`）。

**真实 32K 三臂收官回归（生产目录直接只读跑）**：主会话独立复算 `e116e_ruler32_formal_{fullkv,mavg,aavg}.json` 逐任务 11 格求均 → **FULLKV=59.38 / mavg=59.99 / aavg=57.33 与历史逐位一致**；receipt 含 run_id + formal/scorer 脚本 SHA256 + 源数据 SHA×11 + 42 个 selected 文件 source SHA，已入库 `exp/trace/results/`。

红绿矩阵按你的清单全落：11×1 拒、10×100 拒、11×100 正例、成功后删任务重跑（旧产物 SHA 逐位不变）、scorer hash 错配重跑、131072 混入+同 ids 不同 length 双 method 拒、四种非法 --out 拒、closure python -O 双模式、clean clone SKIP、legacy 源哈希前后不变。

**「正式论文证据」升格条件自此绑定新门禁**：64K/128K 收口一律走新正式入口（含 --data-root 必填），DONE 信号 = receipt + 全门禁 + 原子发布三者齐全。
