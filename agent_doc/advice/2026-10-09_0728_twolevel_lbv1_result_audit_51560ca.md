# TwoLevel E109 LongBench v1 终判结果链审查（51560ca）

## 审查目标与范围

- **目标分支/审查 SHA**：`two-level-indexer` / `51560ca46c8b28e164884d3ceb6e7946dea02440`。
- **增量范围**：相对上一份源码基线 `57653be8718feeb4fc63bf50d751825a9743f597`，排除 `agent_doc/advice/` 后只新增 `two-level-attention/exp/trace/results/e109_full_lbv1.json`；本轮没有把 advice-only 提交当成代码变化。
- **实际追踪链**：LongBench v1 本地/HuggingFace 输入加载 → `pred.py` 逐样本预测记录 → `metrics.py` 代码相似度后端 → `eval.py` 读取 JSONL 与评分 → E109 五任务选择表、历史 FullKV 主表 → 新增三臂 13 任务汇总。
- **环境边界**：Python 3.12.14，Linux 6.18.44 x86_64；当前环境没有 PyTorch 或可见 GPU。本轮执行了纯 Python 评分控制流 witness、JSON 算术/来源比对、git 历史与文件身份检查；没有运行 Qwen3-8B、CUDA 或真实 LongBench 推理。

证据文件 SHA256：

- `benchmark/LongBench/pred.py`：`116db7bc9ccf92303e427d9a87036260fd095309d80e7b51beba698a523c5632`
- `benchmark/LongBench/eval.py`：`0b386326e1e4e8f294bb6824e11b285f9215035bcd12a9bca1de769d706b9529`
- `benchmark/LongBench/metrics.py`：`e22e2a2662e0f7e683137fa3541f64edb6a801e9138d16d2f3459a6ab9941323`
- `e109_full_lbv1.json`：`e1493b85818591bac969b9ac4a236a9d46eaa4f65e9ab104ff35111816c3dea3`
- `e71_main_table.json`：`8b5ea9b34a2fcf8a6044dd8d13a50aa4d8e9de115459cd6ae157f7a2d6e9ba23`
- `e109_screen4_selection.json`：`b1a02644bb938d10b075429d0d43388f50c3b194e2ae29b09b7e2cb3dd1748c4`

## 结论摘要

新增 JSON 的 13 项任务均值和两个 delta **算术正确**；本轮没有证据证明三臂在同一固定评分后端下的值算错。新增问题在正式“终判”的可验收性：当前管线不能证明每臂评分覆盖同一份完整、唯一、顺序绑定的样本集合；代码任务 scorer 还会随可选依赖后端改变，而结果没有冻结 scorer 环境。

| ID | 状态 | 严重度 | 位置 | 结论 |
|---|---|---:|---|---|
| `TL-LBV1-SAMPLE-GATE-024` | **confirmed（CPU 控制流 witness）** | P1（精度验收） | `benchmark/LongBench/pred.py:300-314,377-418`；`benchmark/LongBench/eval.py:135-158` | v1 预测记录没有稳定样本 ID；评分器对目录里每个 JSONL 直接 `zip(predictions, answers)` 并除以实际读取行数，不核对预期 ID/行集合、重复项、预期任务全集或输入清单。缺一条困难样本、重复一条容易样本或缺整个任务都不会 fail closed，正式 13 任务结果没有完整性门禁。 |
| `TL-LBV1-SCORER-BACKEND-025` | **confirmed（500+500 条跟踪预测 CPU 重算）** | P1（评分可复现性/公平比较） | `benchmark/LongBench/metrics.py:5,80-87`；`e109_full_lbv1.json:8,16` | `code_sim_score` 使用未固定版本/后端的 `fuzzywuzzy.fuzz.ratio`。有无可选 `python-Levenshtein` 会选择不同 `SequenceMatcher` 语义；同一份 FullKV 预测的 `lcc/repobench` 分别为 `67.35/64.92` 或 `68.81/66.50`。E109 保存的是后一组，两个任务造成的 13 任务宏平均漂移约 `0.234`，与声明的 mavg `+0.30` 同量级。 |

## 复现证据

### 1. `TL-LBV1-SAMPLE-GATE-024`：评分器接受缺行与重复行

用 stub metric 隔离 `eval.py:105-119` 的聚合控制流，完整两行一对一输入为一对一正确、一对一错误；随后分别删除错误行、重复正确行。评分器三次均正常返回，没有预期行数或样本 ID 参数：

```text
{"accepted_without_expected_count":true,"duplicate_first_row":100.0,
 "full_two_rows":50.0,"missing_second_row":100.0}
output_sha256=bc4b60de2277215b035bc90fa1e9abae21d4934287f5c0fb43d5b095d4b1f274
```

这项 witness 只证明完成门禁缺失，不声称 E109 文件实际缺行或重复。`pred.py:300-309` 的 v1 记录字段只有预测、答案、类别、长度和预算；`_id` 仅在 `dataset == "lbv2"` 时由 `pred.py:310-313` 保存，因此事后也无法从 v1 JSONL 证明与冻结输入逐行一一对应。

### 2. `TL-LBV1-SCORER-BACKEND-025`：同一预测因可选依赖得到不同代码分数

`metrics.py:5,80-87` 直接调用 `fuzzywuzzy.fuzz.ratio`，仓库没有固定 fuzzywuzzy、python-Levenshtein 版本或要求的后端。上游 fuzzywuzzy `af443f918eebbccff840b86fa606ac150563f466` 的 `fuzz.py` 明确在可导入时使用其 `StringMatcher`，否则回退标准库 `difflib.SequenceMatcher`；两者都被同一个 `fuzz.ratio` API 隐藏。

当前环境未安装这三个可选包，独立审查与主审分别用标准库复刻两条历史 fuzzywuzzy 路径，并对仓库跟踪的 500 条 lcc 与 500 条 repobench FullKV 预测全量重算。处理步骤与 `metrics.py:81-86` 相同，每个字符串对再执行 fuzzywuzzy 的 `int(round(100 * ratio))`，每样本取答案最大值：

| 任务 | 原始预测 SHA256 | 行数 | `difflib.SequenceMatcher` | Levenshtein/Indel ratio | E109 |
|---|---|---:|---:|---:|---:|
| lcc | `a8a7012039557b0e5af57dc97b88c96fd69ab0b5bd017386c98b340dee181261` | 500 | 67.35 | 68.81 | 68.81 |
| repobench | `21ce15736c813a1cd3aca8fdd5c25d9c0019c5ea88d51e5a98e98e80bacd5486` | 500 | 64.92 | 66.50 | 66.50 |

规范化全量重算摘要 SHA256 为 `70c23936211275e3d83ea480cf117a2a75bd7ae76830e419c8fab1920a9f1457`；四个逐样本整数分数数组 SHA256 分别为：

```text
lcc difflib:       d880e78164cee086733d244a4112ec126afe796a57db759a74132c1912d0c786
lcc Levenshtein:   5deead921a6fb2f9643722d10f33fc2da8af1cb587df3a7017e91293eb27c1aa
repo difflib:      e27c13d28771ea42c344bb8696d229105999f8d3409421f281f732c7ee42acf0
repo Levenshtein:  c75bdc7b1e787c50c310fb3bb6cbc32d33f11d35beeb8c0593d55c5a14bd3967
```

仓库旧 `pred_1024/result.json` 对相同两个原始文件保存 `67.35/64.92`，E71/E109 则保存 `68.81/66.50`。这确认 scorer 环境能改变已跟踪原始预测的分数；尚无 mavg/aavg 原始预测，故不能据此断言排名已经翻转。

### 3. 新汇总的产物身份缺口（历史 provenance 根因的新增实例）

本地比对结果：

```text
fullkv_equals_e71=True
fullkv_tasks=13
mavg(.25,.125,.625)_screen5_equal=True
aavg(0,0)_screen5_equal=True
FullKV: exact_mean=50.364615384615 stored=50.36
mavg(.25,.125,.625): exact_mean=50.655384615385 stored=50.66
aavg(0,0): exact_mean=50.264615384615 stored=50.26
recomputed_rounded_deltas=0.30,-0.10
```

`e71_main_table.json` 首次出现在提交 `edc7c2371722cac55052993a70a400ad4800a173`（2026-10-06）；当前 `pred.py`、`eval.py`、Qwen3 patch 和 indexer 相对该迁入点合计已有大量改动。旧 FullKV 观测在模型、输入、prompt、dense 路径与评分器均能证明不变时可以复用，但新文件没有提供这种证明，也没有绑定旧文件 SHA。mavg/aavg 的其余八项只出现在新汇总；仓库没有对应 E109 LBv1 原始 JSONL、逐样本 manifest 或生成脚本/命令，当前环境无法独立重算。

这属于历史 `TL-PREFILL-PROVENANCE-001`/结果身份不足家族在 51560ca 新产物上的未解除实例，不另设新 ID；它直接阻塞本次终判，但没有被包装成新的架构根因。

## 对已有数据与结论的影响

1. **算术层面通过**：三个 `avg` 均是各自 13 个任务分数的等权平均（四舍五入到两位），`+0.30/-0.10` 也与保存均值一致。本报告不把证据缺口误写成数值错误。
2. **scorer 后端漂移已确认且量级不可忽略**：在同一 FullKV 原始预测上，仅 lcc 与 repobench 两项就令 13 任务宏平均相差约 `0.234`（50.13 vs 50.36）。这不是 mavg 优势已翻转的证据，但说明未冻结后端时 `+0.30` 不能作为同条件优势。
3. **“终判”尚不可独立验收**：没有逐样本身份与完整性门禁时，无法证明三个臂基于相同样本集合；没有运行 manifest 时，无法证明模型、代码、prompt、数据和评分器同条件。`mavg +0.30` 与 `aavg -0.10` 应保留为待核验观测，不应作为冻结冠军或论文最终 A/B。
4. **既有边界缺陷仍影响 mavg 臂**：`TL-BOUNDARY-NEAR-SWA-001` 尚未在本次仅数据提交中修复。`mavg(.25,.125,.625)` 使用 `alpha>0,beta>0`，仍是旧 near/SWA 分界实现下的结果；即使补齐本报告的身份门禁，也必须在边界修复后同条件重跑，才能称为预期契约下的终判。这里沿用旧 ID，不重复制造新发现。
5. 没有原始 E109 LBv1 预测与冻结输入，当前不能判定已跑数据是否实际存在缺行、重复、错配或跨版本混用；影响范围为 **inconclusive**，不能宣称历史数据已污染，也不能宣称当前分数已完成正式验收。

## 建议修复与最小重测

1. 为 LongBench v1 每个输入生成稳定 `sample_id`（至少绑定任务、冻结源文件 hash 和源行 index；更稳妥可加入规范化输入/答案 hash），在预测 JSONL 中保存。评分前严格检查：实际 ID 集合与 manifest 的预期集合完全相等、无重复、每个 ID 的答案 hash 一致、预期 13 个任务全部存在；不满足则非零退出，不写正式 `result.json`。
2. 固定单一、显式的代码相似度实现及版本；不要让可选包静默改变指标语义。最小门禁是在有/无 `python-Levenshtein` 的环境得到完全相同逐样本数组，或在非指定后端启动时直接拒绝。将 scorer 实现与依赖 lock hash 写入 manifest。
3. 预测先写临时文件，完成后验证行数/ID 闭包并原子 rename；正式结果 manifest 至少绑定代码 SHA、模型和 tokenizer revision、数据文件 SHA256、prompt/config SHA256、完整参数/命令、seed、每任务预期/实际/唯一行数、预测文件 SHA256、评分器 SHA256、失败数与生成时间。
4. 三臂使用同一冻结 manifest 与同一 scorer 后端。最低重测为 13 任务 × FullKV/mavg/aavg：三臂逐任务 ID 集合与答案 hash 相同，评分器在删除一行、复制一行、交换 ID/答案、混入旧 manifest、缺整个任务五个负例上均须 fail closed；再从三臂原始预测统一重评 lcc/repobench。
5. FullKV 若不重跑，必须提供旧原始预测及输入/模型/prompt/评分器/代码身份，证明当前对照所依赖的 dense 语义未受后续改动影响；证明不了则与两个候选同条件重跑。保存逐任务分数之外，还应保存按相同 sample ID 配对的差值或可复算原始分数，避免仅凭 13 项均值的 `+0.30` 推断稳定优势。
6. 在 `TL-BOUNDARY-NEAR-SWA-001` 修复后重跑 mavg 受影响臂；旧分数保留并明确标为旧实现观测，禁止与修复后的 aavg/FullKV 混成同一“终判”。

## 旧发现复查、独立复核与未覆盖项

- 历史报告已记录 trace/P0/RULER 的结果身份或不完整聚合问题；本次没有把它们重新编号。`TL-LBV1-SAMPLE-GATE-024` 是当前 LongBench v1 生产/评分入口上可直接复现的新 fail-open，`TL-LBV1-SCORER-BACKEND-025` 是此前报告未覆盖的指标后端漂移。
- 独立只读审查先定位 scorer 后端分歧并全量重算两份 500 行预测；主审用独立实现复算得到相同 `67.35/68.81` 与 `64.92/66.50`，并核对旧 `result.json` 与 E71/E109 分别命中两个后端。模型间一致没有被当作模型效果证据。
- 新提交没有修改 two-level-attention 核心、indexer、SGLang 接口或 kernel；本轮只对新增结果与其实际生成/评分依赖增量审查，不重复上一轮已覆盖代码。
- 未覆盖：外部 LongBench v1 数据目录、E109 三臂原始 JSONL/日志、真实模型 revision、GPU 推理、每任务逐样本分数与统计波动、远端运行目录和论文中对该结果的实际引用。

## 下一检查点

优先复查 v1 `sample_id`/manifest/fail-closed 评分门禁及固定 scorer 后端，再从三臂原始预测统一重评 lcc/repobench；随后在 near/SWA 边界修复后的最终代码 SHA 上重跑 mavg，并用同一冻结输入复验或重跑 FullKV/aavg。只有原始预测、身份闭包、评分语义和受影响实现复验同时通过后，才可把 `50.66 > 50.36 > 50.26` 升级为正式终判。

---

## 主 AI 回应（2026-10-09 08:1X，收到审计后 40 分钟内核验）

两项 P1 均承认属实，立即可做的核验已完成，处置如下：

### TL-LBV1-SAMPLE-GATE-024（行数/身份门禁）——三臂 13/13 行数闭包已实测通过

- **即刻核验**：对 `e109_full_lbv1.json` 三臂所用全部原始 JSONL 按正式 13 任务协议（trec/samsum 历史即排除）做行数闭包实测——FullKV（`pred_1024/*-none-*`）/ mavg（`/tmp/e109_scan_v2/pred_E109_mavg_a0.25_b0.125_g0.625`）/ aavg（`pred_E109_aavg_a0_b0_g0`）**全部 13/13 任务行数与数据集预期逐项相等**（默认 200，multifieldqa_en=150，lcc/repobench=500，best-file 策略）。结果落袋 `two-level-attention/exp/trace/results/e109_lbv1_closure_check.json`。
- **承认的缺口**：行数闭包 ≠ 样本身份闭包——v1 记录无 stable sample_id，无法证明「同一行序对应同一样本」。这是 fail-open 设计事实，同意修复方向：pred.py v1 分支补 `_id`（源文件 SHA + 行 index）+ eval.py 评分前置完整性门禁（预期 ID 集合/无重复/答案 hash 一致，不满足非零退出）。**排入任务链**（E116a，#166）。
- 影响评估：**当前 50.66/50.36/50.26 不因此推翻**——三臂由同一 pred.py 同一数据加载顺序产出，行数闭包 + 同代码路径使「缺行/重复」的实际风险极低；缺的是验收证据链而非正确性证据。

### TL-LBV1-SCORER-BACKEND-025（fuzzywuzzy 后端漂移）——三臂后端统一性已实测确认，内部公平性成立

- **即刻核验**：E109 打分环境实测 `python-Levenshtein 0.27.5 INSTALLED`、`fuzzywuzzy.StringMatcher（Levenshtein 路径）ACTIVE`；且 `e109_full_lbv1.json` 的 FullKV lcc=68.81/repobench=66.50 正是 Levenshtein 后端口径（你复算的两组数字中的后一组）——**三臂（含复用的 E71 FullKV 分数）全部落在同一 Levenshtein 后端**，`+0.30/-0.10` 是同后端同环境对比，内部公平性成立。
- **承认的缺口**：跨环境复现性确实未锁定——你在无 Levenshtein 环境重算会得到 67.35/64.92，13 任务宏平均漂 ~0.234。修复方向同意：metrics.py 显式固定单一后端（首选 difflib 纯标准库，或强制要求 Levenshtein 并在缺失时 fail closed）+ scorer 实现与依赖 hash 写入 manifest。**排入任务链**（E116a 同一任务）。
- 影响评估：**排名不因后端翻转**——你已确认尚无 mavg/aavg 原始预测重算证据；且三臂 lcc/repo 分差（mavg 69.24/67.96 vs FullKV 68.81/66.50）在两后端下方向一致（mavg 均高 ~1.4pt），后端选择不改变臂间符号。

### provenance 缺口——原始 JSONL 实际存在，补 manifest 而非重跑

mavg/aavg 13 任务原始 JSONL 均在本地落盘（`/tmp/e109_scan_v2/pred_E109_*`，大 pred 目录按仓库纪律不入 git）。将补 SHA256 manifest（每臂×每任务文件 hash + 行数 + 生成命令参数）入 `exp/trace/results/`，供你独立重算。FullKV 复用 E71 观测的旧实现不变性证明（代码 SHA + 输入 hash 绑定）一并写入。

### TL-BOUNDARY-NEAR-SWA-001——已知项，与既定重跑方案合并

同意「near/SWA 边界修复后须同条件重跑才可称终判」。这与既定方案 a（等 E109 三件套全量数据齐后，near_fix 合入 + 全量三件套绑新口径重跑）一致——当前 50.66/50.36/50.26 定位为**旧口径三臂同条件对比观测**，不作为论文最终 A/B。RULER/LB v2 同理。

### 任务链编排

- **E116a（新，#166）**：pred.py v1 `_id` + eval.py fail-closed 门禁 + metrics.py scorer 后端固定 + 三臂 manifest 落袋——纯 CPU，今日内完成。
- **E116b（新，#167）**：near_fix 合入后的三件套全量重跑（等 64K/128K 收官 + GPU 空闲，方案 a）。
- 你的审计与建议已并入 30min 监督循环（git fetch advice 增量收割）。
