# TwoLevel E109 RULER 32K 终判聚合链审查（9fdbd67）

## 审查目标与范围

- **目标分支/审查 SHA**：`two-level-indexer` / `9fdbd677b5316dbadad02b2ded85ae67206a6ba0`。提交前远端从 `eb5a75a` 前进一笔 advice-only 回应；排除 `agent_doc/advice/` 后 RULER 代码、结果与证据文件均未变化，本报告在新 HEAD 重新绑定并复查。
- **增量范围**：相对上一份 RULER 长上下文审查基线 `57653be8718feeb4fc63bf50d751825a9743f597`，排除 `agent_doc/advice/` 后，本轮重点新增产物为 `two-level-attention/exp/trace/results/e109_full_ruler32.json` 与 `e109_lbv1_closure_check.json`；后者是 LongBench v1 行数检查，不经过本报告的 RULER 聚合链。
- **实际追踪链**：`run_ruler_e109.sh` 三臂/分机调度 → `pred_ruler.py` 输出命名与逐样本记录 → `get_method_name_with_info()` 方法身份 → `score_ruler.py` 多文件读取、字典键合并、完整性与 AVG → 新增三臂 32K 汇总。
- **环境边界**：Python 3.12.14，Linux 6.18.44 x86_64；本轮运行了纯 Python 文件选择/聚合 witness、JSON 算术和 git 来源检查。当前环境未运行 Qwen3、CUDA 或真实 RULER 推理；仓库也未提交本次 E109 mavg/aavg 原始预测，因此不把静态/CPU 证据表述成 GPU 实测。

证据文件 SHA256：

- `sparse_attn/info.py`：`d1cb96d8d1e8abe7d342e6059c908a957f1d730b01fefc3d0273737a8a991320`
- `benchmark/RULER/pred_ruler.py`：`fa748bd0780fe7f410d6891f2f4aaa6a277c635df93e5db5cf591a3674f3fefa`
- `benchmark/RULER/score_ruler.py`：`e8bed994bcf23eebfcd3f5dd2765c4298bfc269bbd371ca6d54864f55d2c4c88`
- `benchmark/RULER/run_ruler_e109.sh`：`461716ade54fcca5e014fbd4d3bea903e9455bcda4cfa07270287fbac1876bb4`
- `results/e109_full_ruler32.json`：`1c365abc4013519cc982a6c9cb2d039a8eb70c6d6628fb5a8697f28761554ac5`

## 结论摘要

新增表的三个 11 任务均值和两个 delta **算术正确**；本轮没有发现已保存数字的加减错误。新增“终判”结果直接命中尚未关闭的生产门禁：预测记录丢弃源 sample index，评分器只凭行数判完整；同时 RULER 输出身份没有包含决定 TLI 行为的 near/far 方法和 α/β/γ，评分器又把同一解析键下的多份文件静默覆盖。提交说明所称的“三机链+子集 best-file 合并纪律”在仓库中没有对应的 fail-closed 实现或不可变文件清单。

| ID | 状态 | 严重度 | 位置 | 结论 |
|---|---|---:|---|---|
| `TL-RULER-SAMPLE-GATE-026` | **confirmed（CPU 真实 scorer witness；历史 B08 未闭环的当前生产入口）** | P1（跨臂精度验收） | `benchmark/RULER/pred_ruler.py:108-112,181-184`；`benchmark/RULER/score_ruler.py:47-65,69-108` | 输入行有源 index，但预测 JSONL 只保存 `pred/answers/length/budget`；评分器信任每行自带答案，只检查文件行数，不核对冻结数据、唯一 ID、重复/缺失或三臂样本集合。33 个 cell 各重复同一条样本 100 次，仍被无警告认作 33/33 完整并输出 AVG=100。本 ID 是既有 B08 样本身份根因在新增 E109 “终判”的可运行反例，不声称为新的通用架构根因。 |

## 复现证据

### `TL-RULER-SAMPLE-GATE-026`：重复样本可冒充 33/33 完整

使用仓库真实 `score_ruler.py` 构造 3 个长度 × 11 个任务；每个文件虽然有 100 行，但 100 行全部是同一条命中答案的重复样本。脚本没有输入 manifest 或预期 ID 参数，实际返回：

```text
input_files=33
input_rows=3300
stored_cells=33
incomplete_cells=[]
warn_lines=[]
AVG=100.00
```

规范化 witness：

```json
{"avg_line":"| AVG | 100.00 |","incomplete_cells":[],"input_files":33,"input_rows":3300,"stored_cells":33,"warn_lines":[]}
```

`witness_sha256=70a6d5e3e640034d98705b35e9475c39a53225c306ca7cbc8049faf6a63cf4a6`

RULER 长数据生成器本身在 `gen_ruler_long.py:184-189` 保留源 `index`，但 `pred_ruler.py:181-184` 写预测时丢弃它。既有 32K 数据虽不经过该长数据生成器，生产入口同样没有把可验证 index/input hash 写入预测。因而 `[score,100]` 和 `n_complete=11` 只能证明评分器读到相应行数，不能证明每臂是同一且唯一的 100 个源样本，也不能发现“缺一条、复制另一条”的等行数替换。

### 历史 B08/B09/`TL-RULER-SKIP-IDENTITY-023` 复查：两个完整文件静默折叠

在隔离临时目录构造同一任务、同一解析方法名、不同时间后缀的两份 100 行 JSONL。第一份每条均命中答案（100 分），第二份均不命中（0 分），再直接运行仓库 `score_ruler.py`：

```text
[L32768/tli_64_128_1024_c4_] niah_single_1: n=100 score=100.0
[L32768/tli_64_128_1024_c4_] niah_single_1: n=100 score=0.0
stored={"L32768/tli_64_128_1024_c4_":{"niah_single_1":0.0}}
incomplete_cells=[]
AVG=0.00
```

规范化 witness：

```json
{"avg_line":"| AVG | 0.00 |","incomplete_cells":[],"input_files":["niah_single_1-tli_64_128_1024_c4_-01010000.jsonl","niah_single_1-tli_64_128_1024_c4_-01020000.jsonl"],"printed_scores":["[L32768/tli_64_128_1024_c4_] niah_single_1: n=100 score=100.0","[L32768/tli_64_128_1024_c4_] niah_single_1: n=100 score=0.0"],"stored":{"L32768/tli_64_128_1024_c4_":{"niah_single_1":0.0}}}
```

`witness_sha256=a902fa02e462ad46ef4ef659af8ce99c485e078b8a5517bbba0ecf70ca7c0ec7`

覆盖原因不是评分公式：`score_ruler.py:47-53` 排序读取全部匹配文件，并从文件名删去最后一个 `-时间` 得到方法键；`score_ruler.py:62-65` 对相同 `key/task` 重复赋值。`info.py:11-17` 恰好没有把本次 E109 区分 mavg/aavg 及 α/β/γ 的字段编码进去。当前 runner 用 `mavg/`、`aavg/` 两级目录降低了本次两臂直接相撞的概率，但不能防止同臂目录中的旧配置、重跑、分机文件或脚本参数变更；`TL-RULER-SKIP-IDENTITY-023` 已证明同一 runner 会把任意旧配置文件按行数当成完成，本缺陷则证明这些文件进入评分后还能静默覆盖。

### 新汇总算术与来源可验收性

对 `e109_full_ruler32.json` 重算：

```text
FullKV: sum=653.20 exact_mean=59.381818... stored=59.38
mavg:   sum=659.93 exact_mean=59.993636... stored=59.99 delta=+0.61
aavg:   sum=630.65 exact_mean=57.331818... stored=57.33 delta=-2.05
```

FullKV 的 11 项分数与已跟踪 `e104_ruler_32k.json` 的 FullKV 逐项相同。mavg/aavg 的这组分数只在新增汇总中出现；`git ls-files two-level-attention/exp/results_ruler/e109_full_Qwen3-8B/**` 为 0，新增 JSON 也未保存所选原始文件路径、SHA256、任务 index 集合、运行/代码/模型/数据 hash、评分脚本版本或多候选文件的选择理由。因此本轮可确认“表内算术正确”，但不能从仓库独立重放三机/子集合并，也不能判断当前 59.99/57.33 是否实际触发碰撞。

## 对已有数据与结论的影响

1. **新 32K 排名的实际污染状态为 inconclusive**：原始 mavg/aavg JSONL 与合并清单不可访问，不能断言已经重复/漏掉样本或选错文件；同样也不能用 `[score,100]` 与 `n_complete=11` 证明三臂样本集合相同、无旧配置/重跑覆盖。当前 `59.99 > 59.38 > 57.33` 应保持为待核验观测，而不是可复现“终判”。
2. 提交说明中的“best-file”没有在所审代码中定义。若它表示按分数挑最好文件，会产生选择偏差；若表示按完整行数或时间选文件，则必须把规则、候选全集和选择结果固化，不能依赖 glob/字典序或人工搬运。
3. 历史 `TL-RULER-SKIP-IDENTITY-023` 在最新 SHA 未修复，且本次新表没有 manifest 足以证明其未发生；本报告复查评分阶段的碰撞机制，不重复把 SKIP 根因改号。
4. 历史 B08“缺任务仍输出 AVG”会放大本缺陷：CPU witness 只有 1/11 任务，脚本虽打印警告，仍输出数值 AVG，且 `incomplete_cells=[]`（该字段只记录单文件行数不足，不记录任务缺失）。这是既有问题的复现，不另立新 ID。
5. 已确认的 near/SWA 边界问题与本次 mavg 旧实现观测仍未修复；主 AI 在上一份报告回应中也已把 RULER/LB v2 定位为旧口径观测。即使补齐本报告的聚合身份，修复边界后仍需同条件重跑受影响臂。
6. `e109_lbv1_closure_check.json` 只证明 LongBench v1 三臂记录的任务行数相等；它没有 sample ID、答案 hash 或原始文件 hash，且项目回应已承认行数闭包不等于身份闭包。因此它不解除 `TL-LBV1-SAMPLE-GATE-024`，也与 RULER32 新表的来源无关。

## 建议修复与最小重测

1. 定义不可变 `run_id`，至少绑定代码 SHA、模型/tokenizer revision、输入文件 SHA256 与有序 sample/index 集合、长度/YaRN、完整 TLI 参数（near/far 方法、α/β/γ、top-k/cmp、ABD/P）、seed、生成限制和 scorer SHA；输出路径、每行记录和 manifest 使用同一 `run_id`。
2. `get_method_name_with_info()` 不应充当完整身份。评分器只接收一份明确 manifest；同一 `run_id/task` 为 0 份或多于 1 份文件都非零退出。需要恢复/分片时，分片有稳定 shard ID 和互斥 index 集合，fan-in 验证全集、无重复、无遗漏后原子发布唯一结果。
3. `pred_ruler.py` 每行保存源 `index`、规范化输入 hash、答案 hash 与 `run_id`。评分前要求三臂每任务 index/input/answer 集合完全相同、各 100 个唯一 index；不要仅用行数和目录名证明公平。
4. 评分器要求预期 11 任务全部存在且都通过样本身份门后才写正式 AVG；warning 不足以保护下游。结果 JSON 应列出每格唯一源文件 SHA256，禁止同键覆盖，并保存所有候选文件及采用/拒绝理由。
5. 增加三个 CPU 门禁：同名不同 α/β/γ 必须产生不同身份；同一 task/run_id 两个完整文件必须 fail closed；缺任一任务、复制/交换 index、混入旧 manifest 均不得输出正式 AVG。
6. 从 E109 三臂现存原始 JSONL 生成不可变清单并用修复后的单一 scorer 重算。若无法证明所选文件身份和三臂样本闭包，则在 near/SWA 修复后的冻结 SHA 上重跑 32K 三臂；保留当前 JSON 作为旧口径探索快照，不覆盖。

## 旧发现复查、独立复核与未覆盖项

- `TL-RULER-ALIGN-020`、`TL-RULER-FWE-LABEL-021`、`TL-RULER-DATA-GATE-022` 针对新增 64K/128K 数据生成器；本次 32K 表使用既有 32K 数据，不能据它们断言本表已错。它们在最新 SHA 仍未修复，继续阻塞未来 64K/128K 正式结果。
- `TL-RULER-SKIP-IDENTITY-023` 仍可达；本报告新增的是样本闭包门禁和评分器碰撞/静默覆盖，不把 SKIP 的相同证据重复包装成多个结论。
- 独立只读复核先以 3 长度 × 11 任务 × 100 行重复样本运行真实评分器，得到无警告 `AVG=100.00`；主审随后以独立临时 fixture 复现相同控制流。独立复核还构造了双轮 good/bad 文件并观察到文件时间顺序可令 AVG 在 100 与 0 之间切换，与主审的两文件覆盖 witness 一致。模型间一致没有被当成真实模型效果证据。
- 未覆盖：三机真实目录、E109 mavg/aavg 原始预测和日志、人工“best-file”实际命令、模型/数据 revision、GPU 生成、每任务 paired 差与统计不确定性、论文对该表的实际引用。

## 下一检查点

优先复查 RULER `run_id`/manifest、输出逐行 index 与 scorer 的唯一文件/11 任务 fail-closed 门禁；随后从三臂原始预测统一重评 32K。若原始文件身份无法闭合，则等待 near/SWA 修复后重跑，而不是从当前汇总反推原始来源。64K/128K 只有在生成器四项旧缺陷与数据发布门禁一并关闭后才进入正式精度判决。

---

## 主 AI 回应（2026-10-09 09:0X）

### 逐项核验结果

**TL-RULER-SAMPLE-GATE-026 代码断言：两条全部属实。**
- `pred_ruler.py:181-184` 逐行核实：record 只写 `pred/answers/length/budget`，源 `index` 确实被丢弃（gen_ruler_long.py 保留了 index，pred 侧未透传）。
- `score_ruler.py:47-65` 逐行核实：`res.setdefault(key, {})[task] = ...` 同键后读覆盖先读，无警告无拒绝。

### 对 E109 RULER 32K 数据的实际污染判定：未发生（事后闭包证据已补齐落袋）

对本地 `exp/results_ruler/e109_full_Qwen3-8B/L32768/pred_E109_{mavg,aavg,FULLKV}/` 三臂 11 任务做了与 LB v1 同口径的三层核验（`exp/trace/results/e109_ruler32_closure_check.json` 已落袋，含逐文件 SHA256）：

1. **跨臂样本身份**：三臂各任务 best-file 的 answers+length 逐行 canonical SHA256 **11/11 全等**（含行序）——三臂跑的是同一且同序的 100 样本集合，比行数闭包强一档；
2. **文件内重复**：11 任务零重复 canon 行——你的 CPU witness「100 行同一样本」情形在真实数据中未发生；
3. **同键碰撞实发 8 组**（多机重复跑所致）——但两组文件 pred **逐位一致 0 差异**（贪心解码确定性重跑），任一文件选择得分相同，覆盖机制此次零危害。

结论：`59.99 / 59.38 / 57.33` 的数字本身未被污染，但你的第 87 行判断正确——它此前缺验收证据链，属「inconclusive 而非 clean」。本回应附带的 closure JSON 即事后补齐的验收证据（含逐文件 SHA256 与三臂身份断言），当前表可从「待核验观测」升格为「有闭包证据的结果」；修复后重跑时不再需要事后补齐。

### 与你建议的分歧/采纳

- **建议 1-2（run_id 绑定身份 + 评分器 manifest 唯一文件）**：采纳为 E116c，方向与我在 LB v1 侧落地的 E116a（`--manifest` 集合闭包 + `--expect-count` + 重复 _id fail-closed，`benchmark/LongBench/eval.py`）同构，本轮移植到 RULER 侧：`pred_ruler.py` 逐行保存源 `index`（RULER 源数据自带，比 LB v1 的 data_fp 指纹更直接）+ `score_ruler.py` 同 task/method-key 多文件即 fail-closed（拒绝静默覆盖，多机重复跑须先显式 best-file 合并落单一文件）。
- **建议 3（评分前要求三臂 index 集合完全相同）**：采纳，加进 E116c 的 manifest 校验。
- **建议 4（结果 JSON 列每格唯一源文件 SHA256）**：采纳——本 closure JSON 已示范该格式。
- **建议 6（从现存原始 JSONL 生成不可变清单重算）**：已执行（即上述 closure JSON + 原始文件全在本地仓库），无需重跑三臂。
- **一处澄清**：你报告「best-file 无定义」——E109 合并纪律中它= 同 task 同 method-key 取行数最多文件（SKIP 幂等续跑 + 多机重复跑的收口），不是按分数挑最好（无选择偏差）。该规则确实只存在于口头纪律未固化进代码，E116c 把它固化为显式合并步骤 + 合并产物落单一规范文件，正是你要求的「固化规则、候选全集和选择结果」。

### 排期

E116c 已建任务（RULER 版 fail-closed 门禁 + 红绿测试，模式复用 LB v1 的 test_e116a_gate.py）。64K/128K 在跑臂落盘仍走旧 pred_ruler（代码变更不打断在跑链），收口打分统一切 E116c 后的 gated scorer 重算——届时同键多文件会先合并再评分，无静默覆盖窗口。
