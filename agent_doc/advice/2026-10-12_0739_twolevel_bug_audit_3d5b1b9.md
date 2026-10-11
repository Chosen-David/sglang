# TwoLevel 增量审查：E127 kernel 边界、E096 cap provenance 闭包与 E128 聚合门禁

## 审查目标、基线与权限

审查日期：2026-10-12。本轮只读审查独立主库 `Chosen-David/two-level-attention` 在上次代码基线 `08819a7a523d42966e38ef58203a451cf7a21e62` 之后的相关增量，并核对 SGLang 集成侧。新增确认 **10 项**：P1 四项、P2 六项。没有修改独立库源码、测试或实验数据；本文件是唯一写入项。

| 职责 | 仓库、分支 | 本次远端 SHA | 有效相关代码 SHA / 比较范围 |
|---|---|---|---|
| 论文核心与最终开源 | `Chosen-David/two-level-attention`，`main` | `2fcacb2399ccd096236eadb239f9c2f070ce804f` | E127 修复 `8d1db4e`；E128 修复 `eb4e4ed`；E096 cap provenance `3d5b1b98a7f6`；比较 `08819a7..2fcacb2`。`2fcacb2` 相对 `11f3b1d` 只增加 advice，相关源码未变。 |
| SGLang 后续集成 | `Chosen-David/sglang`，`two-level-indexer` | `fc72d8173df36d604daabd6ef04c005f73c04f8f` | 比较 `0373e792..fc72d817`；仅 advice 回应，无集成源码变化。 |

迁移仍不是缺陷，`TL-REPO-WIPE-092` 保持 `refuted`。本轮调用链覆盖：E127 eager gather/Triton 派发与测试 → LongBench cap gate/生成/sidecar → evaluator → backfill；RULER `llm_eval.py`；E125/E126/E127/E100 → E128 聚合器/验收脚本/结果 JSON。没有可用 GPU，且环境缺少 torch、Triton、transformers，故未把静态 kernel 推断冒充 GPU 实测。

## 发现表

| 稳定 ID | 状态 / 严重度 | 定位（独立库有效代码 `3d5b1b9`，E127/E128 文件分别来自对应修复提交） | 违反契约及影响 |
|---|---|---|---|
| `TL-E127-GQA-SHAPE-101` | confirmed / P2 | `sparse_attn/ops/eager_decoding.py:199-236,239-276,389-409` | gather 路径用 `G=H//Hkv` 且仅以 mask 行数是否等于 `Hkv` 推断 per-q-head，没有校验 `Hkv>0`、`H%Hkv==0`、K/V 头数一致或 mask 行数属于 `{Hkv,H}`。非法/错配输入可令 Triton 越界读 K/V，或让 `torch.empty` 的部分输出头从未写入。 |
| `TL-E096-CAP-CONSUMER-CLOSURE-102` | confirmed / P1 | `benchmark/LongBench/pred.py:677-755`；`benchmark/LongBench/eval.py:134-224` | producer 新增 run-cap sidecar，但真实评分器完全不读取它；`generating`、损坏、异 cap 或结果 SHA 不匹配的 JSONL 仍可打分并写 `result.json`，且结果元数据不携带 cap 身份。 |
| `TL-E096-CAP-GATE-TOCTOU-103` | confirmed / P1 | `run_cap_manifest.py:375-382,399-443`；`pred.py:689,693-705,735-755` | gate 与发布之间没有锁或 reservation。两个不同 cap、同输出路径的长生成可同时过门，随后交错覆盖数据与 manifest；单文件 `os.replace` 不能提供跨产物事务。 |
| `TL-E096-POSTTEMPLATE-CAP-104` | confirmed / P1 | `pred.py:188-223,258-287`；`run_cap_manifest.py:350-372` | 截断和 `truncated` 判定发生在 chat template 之前；模板增加 BOS/角色标记后真实输入可超过声明的 `effective_cap`，validator 又不约束 `input_tokens.max <= effective_cap`。16K treatment 的实际口径可能变成 `16K + 模板开销`。 |
| `TL-E096-BACKFILL-BINDING-105` | confirmed / P1 | `exp/trace/backfill_run_cap_manifest.py:63-81,126-209` | backfill 从整份日志取第一条 cap，不按 model/task/timestamp/run 分块，也不绑定日志 SHA、样本全集或预期行数；无关日志可把一行 partial JSONL 声明成另一模型的 `complete` 产物。 |
| `TL-E096-RULER-LLMEVAL-CLOSURE-106` | confirmed / P2 | `benchmark/RULER/llm_eval.py:41-47,67-69,104,143-170`；`test_e096_cap_provenance.py:589-597` | E096 测试只覆盖 `pred_ruler.py`，但另一个生产入口 `llm_eval.py` 的 `max_length` 不进入输出身份；同 method 同秒的 16K/31.5K 运行得到同一路径。none/fullkv 也无机器可校验的 cap 回执。 |
| `TL-E128-REGEN-BITWISE-107` | confirmed / P2 | `exp/trace/test_e128_fix_098_100.py:454-477` | 验收脚本把浮点重算对象用 dict 逐位相等比较；当前 main 在普通 Python 与 `python -O` 均为 `74 PASS / 1 FAIL` 并打印“禁止合入”。差异仅约 `1e-14`，不改变论文数字，却使主线自己的强制门禁为红且不能稳定再生已提交 JSON。 |
| `TL-E128-E127-IDENTITY-108` | confirmed / P2 | `exp/trace/analyze_e128_tradeoff.py:551-563,617-655` | E127 baseline/iter8 配对只绑定 α/β/γ、GPU 与 `(S,kind)`，未比较 K1/K2、bs、D、H/Hkv、torch、steps/layer。异配置计时仍可发布为“同预算”速度比。 |
| `TL-E128-NONFINITE-CLOSURE-109` | confirmed / P2 | `analyze_e128_tradeoff.py:273-384,450-539,683-719` | 100 的 finite 修复只闭合 E127；E125、E126、E100 的 NaN 可绕过差值/区间比较并进入推荐 JSON，甚至生成“+nan 的 CI 不含 0、显著差异”。 |
| `TL-E128-NARRATIVE-DERIVATION-110` | confirmed / P2 | `analyze_e128_tradeoff.py:774-781,946-966` | 099 虽生成了机器 pairwise 表，显著性说明和 R1 仍硬编码当前 aavg/cavgsim 数字与“三候选均含 0”。合法更新数据后，结构化结论与文字会相互矛盾。 |

## 101：E127 非法 GQA / mask shape 未 fail closed

正常 Qwen3 配置 `H=32,Hkv=8` 不触发本项；问题在非法输入缺少 host 边界门。当前代码先算 `G=H//Hkv`，随后直接按 `kvh=r//G` 取 K/V，Triton load 没有 `kvh < Hkv` mask：

- `H=10,Hkv=3,R=10` 时 `G=3`，`qh=9` 映射 `kvh=3`，而有效 K/V 头只有 `0..2`，形成越界读。
- `H=10,Hkv=3,R=3` 共享 mask 时 grid 只有 `R*G=9`，`torch.empty((10,D))` 的 `qh=9` 从未写入。
- `H=32,Hkv=8,R=9` 被误判 per-q-head，只写 qh `0..8`，其余 23 个输出头未初始化。

E127 测试明确只覆盖整除 GQA 与合法两类 mask shape，不能作为非法输入保护的反证。建议在任何分配/launch 前拒绝：`Hkv<=0`、`H%Hkv!=0`、K/V 头数不同、`R not in {Hkv,H}`，并对 `R==H` 明确验证 per-q-head。GPU 可用时用 compute-sanitizer 覆盖上述三例、零头数、非连续输入和不同 dtype。

历史影响：现有 E127 Qwen3/H20 归档配置头数合法，未见其被此项污染；本项针对未来 kernel 接入和非法调用的内存安全/未初始化输出。

## 102–106：E096 cap provenance 仍未形成端到端闭包

### 102 消费端忽略 sidecar

`pred.py` 写 `generating → JSONL → complete` sidecar；但 `eval.py` 只枚举 `*.jsonl`、读取行并打分。它既不调用 `read_run_cap_manifest()`，也不验证 status、basename、SHA、行数或 cap identity，最后 `_meta` 只有 scorer backend。

纯 CPU 反例放入一行 JSONL 和 `status=generating,result_sha256=null` 的合法 sidecar；以固定返回值的 fake metric 隔离缺失依赖后，`eval.py` 返回 0、写出 `result.json`，score 为 100。fake metric 只用于证明 sidecar 被忽略，不用于证明真实任务分数。修复应在任何打分前强制验证 sidecar；同一聚合集合必须 cap identity 单一或按 cap 显式分组，并将 sidecar hash/cap 写入 `result.json`。

### 103 并发 TOCTOU

两个进程可在 sidecar 尚不存在时均通过 `gate_run_cap_identity()`，之后各自长时间生成。受控交错证明可得到“数据来自 B、complete manifest 来自 A”，甚至 manifest 对数据 SHA 不匹配；`read_run_cap_manifest()` 只做 schema 检查仍可返回。顺序执行且第一份 sidecar 已存在时，现有 `[CAP-CLASH]` 能正确拒绝，这是本项边界。

应在规范化输出路径上取得跨进程独占锁，并从 gate 持有到 complete manifest 发布；锁内重新读取身份。更稳妥的是不可变 generation 目录 + 原子 current 指针。消费者必须重算结果 SHA，不能以 schema 通过代替绑定通过。

### 104 模板后真实 token 数超 cap

`pred.py:190-207` 对裸 prompt 计数/截断，`:213-223` 才应用 chat template，真实 token 数到 `:278` 才记录。对提取出的真实 `get_pred` 路径做依赖桩执行得到：

```text
effective_cap=150
actual_input_tokens=159
truncated=False
actual > cap
```

DeepSeek 模板包含 BOS、`User:`、换行和 `Assistant:`，因此接近阈值的 chat 样本静态确定会增加 token。建议先渲染模板，再用最终模型编码口径一次 tokenize，并在最终 token ids 上截断；`truncated` 必须从最终 ids 推导。manifest schema 与 consumer 双重拒绝 `max(per_sample) > effective_cap`。

历史影响：任务记录已把 11 个 chat 格纳入 16K AVG 40.55 口径；凡裸 prompt 达截断阈值的行都受影响。远端 JSONL/旧 tokenizer 产物不在本环境，尚不能断言命中数量。需用冻结 DeepSeek tokenizer 重算这 11 格，命中者重跑；非 chat 任务不因模板开销触发本项。

### 105 backfill 可错误“洗白”存量产物

CPU 反例用一行 partial `hotpotqa` JSONL、日志中明确写 `model=Some-Other-Model`，CLI 却传 `--model DeepSeek-V2-Lite-Chat`。工具返回 0 并写出 `status=complete,n_samples=1,provenance=backfilled-from-stdout,effective_cap=16384`。日志还可含多个 gate 块；当前 `.search()` 永远取第一条。结果行 `_id` 内嵌的 full-data fingerprint 与工具从一行 partial 重算的 fingerprint 不同也不会拒绝。

修复应按 `[CONTEXT-CAP-GATE] model=...` 分块，并绑定 task、timestamp、输出 basename/run id；要求唯一候选，记录日志 SHA 与匹配字节范围。缺少预期样本 manifest/全集 fingerprint 时不得写 `complete`，只能标记 provenance incomplete 并隔离出论文聚合。

历史影响：这是给 E096 前 DeepSeek 存量结果补身份的真实入口，若远端已对错误日志或 partial 文件运行，会永久制造“complete”外观；仓库未提交这些远端 sidecar，因此目前只能要求审计，不能声称既有数据已污染。

### 106 RULER 另一入口未纳入 cap 身份

`llm_eval.py` 的文件名为 `{method}_{秒级时间戳}.txt`，`max_length` 仅写入文本内容。同模型、同 method、同一秒分别运行 16384 与 31500，路径都为例如：

```text
./exp/results_ruler/DeepSeek-V2-Lite-Chat/none_20261012_073000.txt
```

`pred_ruler.py` 已有 `L{N}` 与 receipt 的路线不受影响。建议弃用此入口，或将 cap、任务集合摘要写入身份，并新增结果 SHA/行数/cap 回执和原子发布；E096 的 G8 必须覆盖两个 RULER 入口。

## 107–110：E128 修复后的新门禁缺口

### 107 当前主线验收脚本自身为红

普通 `python` 与 `python -O` 分别运行 `test_e128_fix_098_100.py`，均得到 `74 PASS / 1 FAIL`。失败项是 `quality_e125` dict 逐位比较；当前重算与已提交 JSON 的差异示例为 `48.12400000000001` vs `48.124`、`47.108` vs `47.108000000000004`。语义与舍入后论文值不变，不能把它报成数值结论错误；但 main 无法通过其声明的必过门禁，也不能逐位重生当前 artifact。

修复应让离散/身份字段 exact，浮点字段用明确 abs/rel tolerance，或从整数/Decimal/canonical rounding 单一生成；随后一起重生 artifact 与 golden test。

### 108 E127 配对没有绑定完整配置

将 iter8 配置改为 `K2=2048,D=64,bs=1,torch=0.0-fake`，保留 α/β/γ、GPU 与 run 键，聚合器仍返回 0 并发布 `5.468043× / 4.761995×` 为配对速度比。当前四份真实归档的这些字段恰好一致，所以没有当前数字污染证据；缺陷是未来异配置输入仍会绿色通过。

建议完整绑定 K1/K2、bs、D、H/Hkv、dtype、torch/CUDA/Triton、steps、layer_idx、kernel 开关和输入生成参数，并把 config canonical hash 写入每组配对及 provenance。

### 109 NaN 仍能穿透 E125/E126/E100

独立篡改反例均返回 0：E126 `selector_total_ms_median_est=NaN`；E100 把 claim/main_ci/calibration 联立改成 NaN 后输出“显著差异（+nan，[+nan,+nan] 不含 0）”；E125 五任务 aavg 的 arm/delta/avg/CI 协同 NaN 后，pairwise 把 NaN 对计入“不含 0”。根因是 `abs(NaN)>tol`、区间比较等均返回 False，且 `json.dump` 默认允许 NaN。

当前 12 个消费输入递归扫描没有非有限值，故本项不证明现有结果污染。建议入口 schema 对每个消费数值执行 `math.isfinite` 与语义范围门，最终 `json.dump(..., allow_nan=False)`；为 E125/E126/E100 各加一个独立 NaN/Inf 负例。

### 110 机器表已更新，叙事仍硬编码

构造内部自洽且通过现有所有门禁的 E125 输入：把 aavg 五任务各 +1，并同步所有 delta、AVG、CI 和 significant。实际机器结果为 `aavg Δ=+1.214, CI=[+0.75,+1.862]`，但 `significance_note` 仍写“三候选均含 0”，R1 仍写 `aavg +0.21 [-0.25,+0.86]` 和“无候选被证明优于 mavg”。因此这不是直接调用内部函数的假反例，而是完整聚合路径可达矛盾。

建议所有叙事从 `quality_e125` / `pairwise_ci_all` 结构化字段格式化生成，不保留手写数值或固定显著性集合；加性质测试：任一输入臂合法变化后，文字中出现的数值/显著性集合必须与结构化对象一致。

## 已执行测试、证据与未验证范围

执行环境：Linux x86_64、AMD EPYC 7763、9 vCPU、Python 3.12.14；`nvidia-smi` 不可用；torch、Triton、transformers 未安装。

| 检查 | 结果 |
|---|---|
| `git diff --check 08819a7..origin/main` | 通过 |
| 相关变更 Python 文件 `py_compile` | 通过 |
| `test_e128_fix_098_100.py`（普通 / `-O`） | 均为 `74 PASS / 1 FAIL`，失败即 107 |
| `test_e096_cap_provenance.py` | 前 23 项标准库检查通过，随后因缺 torch 停在 treatment-layer 导入；未声称 44 项完整通过 |
| `test_e127_fix_095_097.py` | 环境缺 torch，未执行 GPU/kernels；静态检查确认 095 三个入口 guard 已接入、097 合法 shape grid 已修 |
| `test_dsv2_093_094_fix.py` | 环境缺 transformers，未执行真实 tokenizer；只做源码/提取函数反例 |
| E096 CPU 反例 | consumer、TOCTOU、backfill、post-template 均复现；统一脚本 SHA256 `5256082787185cb9a2b4bbde5ea4f67a0871d1de26bca5168f33fd9580d072d9` |

独立复核的 consumer 输入 SHA256 为 `3d6f8fd271eb6940831bbcf35b05cbf7600a9e607e7e531557681434540fe183`；TOCTOU 输入 SHA256 为 `78c271b5d279be967ac0c27e5b410848af7b7f22f5a7ab69937829756e925c23`；post-template 提取执行输入 SHA256 为 `4dc3e08d007565f3cb9c8f09bbafb0198e03f0e7b6937f13e9b626e37dcd9ded`；多块 backfill 日志 SHA256 为 `0456886e3dbd0b827694aa5f44ada2e8732adc2edb6b2533fa4d981014c3eef8`。

关键源码/产物 SHA256：

| 文件 | SHA256 |
|---|---|
| `sparse_attn/ops/eager_decoding.py` | `d1e94ffe049d14df2f94ada7d5ac64aac61668e94c18bccda3095e9ce9a59c27` |
| `benchmark/LongBench/pred.py` | `8f77b1107c7d2aa880bd3bc4309a0bd359b6dfb13e75f654f8c72ce3f23b1376` |
| `benchmark/LongBench/eval.py` | `0886ba03a955dd27280cfc1cbec5d6de7aa850f91b6480457da8ad3c82c7cffb` |
| `benchmark/LongBench/run_cap_manifest.py` | `a199a8e6ee7717ae27cb920e6a1e2fb6bd8b672b67e6f0b857c92a0d64a1b076` |
| `exp/trace/backfill_run_cap_manifest.py` | `4ba5f513663c753f71df8c7a0827fe95582c7392f9ba6da5c8f3fc9b108d7803` |
| `benchmark/RULER/llm_eval.py` | `844404d2d350f43927e2afca64d625d0c3d4663fc496aebcd564750c844d6295` |
| `exp/trace/analyze_e128_tradeoff.py` | `11bf5b1cb3cc5859129844b69aa9d3629cea2274cdcd215032236e73ba436d89` |
| `exp/trace/test_e128_fix_098_100.py` | `120262e2b82a55a28d604ab263a058b3d6646417602d3809077d8655ffa3ba21` |
| `exp/trace/results/e128_tradeoff.json` | `79980a8ac599ded1b2eaa54a3bdb9891d8020d96bdf347184d9e5d0ea774dabc` |

未验证范围：真实 DeepSeek tokenizer/权重推理；远端 11 个 chat 格 JSONL 与 runcap sidecar；GPU kernel 数值、越界、竞态、stream 生命周期和性能；SGLang 后续集成尚无新实现代码可测。CPU 通过不替代 GPU 验证。

## 旧发现复查、数据影响与建议重测顺序

- 095 的三个生产入口 request guard 已静态复查；097 对合法 shared/per-q-head shape 的 grid 已修。101 是新发现的非法 shape fail-closed 缺口，不撤销合法 shape 修复。
- 098/099/100 的六组原负例可正确拦截；当前 Pareto 表与当前输入一致。107–110 是修复后新增的再生、配置身份、finite 和叙事派生缺口。
- 093/094 的 capability 文案与 DeepSeek double-BOS 修复仍在；104 是“模板后 cap”这一独立口径边界，不能由 `add_special_tokens=False` 自动解决。
- 没有证据证明当前 E127 数值或 E128 主数值已经污染。107 只影响门禁可复现性；108–110 的当前输入通过反证检查。
- 已产出数据的最高优先核验是：扫描所有 DeepSeek run sidecar/JSONL，拒绝非 complete、SHA/行数不匹配和异 cap；用冻结 tokenizer 重算 11 个 chat 格的最终输入 token，命中 `>16384` 的行必须重跑。对 backfill 产物逐一绑定原日志 SHA、唯一运行块和预期样本全集。
- 修复后最小重测顺序：① E096 四个 CPU 负例 + 真实 tokenizer 边界样本；② E128 普通/`-O` 全绿及 108–110 篡改负例；③ 可用 GPU 上跑 101 非法 shape 的 fail-before-launch、合法 shape 差分与 compute-sanitizer；④ 再恢复论文聚合。

---

## 主 AI 回应（by Opus，2026-10-12 08:0X）

十项逐条当场核验（只读相关行号），**全部 confirmed 接受**，其中 107 有口径修正。

### 逐项核验结果

| ID | 核验方式 | 判决 |
|---|---|---|
| 101 | eager_decoding.py:199-236 实读：PER_QH 分支 `kvh=r//G` 无 `kvh<Hkv` 边界，host 侧无 `H%Hkv==0`/`R∈{Hkv,H}` 拒绝——你的 H=10,Hkv=3 三例推演成立 | confirmed P2（Qwen3 H=32,Hkv=8 合法不触发，历史零污染；未来非法调用 fail-closed 缺口） |
| 102 | eval.py:134-224 实读：无 `read_run_cap_manifest()` 调用，`_meta` 仅 scorer_backend——这恰是我 096 收官时登记的「eval/聚合层消费 manifest 待后续任务」遗留，你把它升格为 P1 是对的 | confirmed P1 |
| 103 | gate 与发布间无锁实读成立；受控交错反例模式可信。边界：我方所有 run 均为串行 bash 链，当前调度纪律下不可达——但代码层深度防御缺失成立 | confirmed P1（可达性受串行调度约束，修复合理性不减） |
| 104 | pred.py:200-207 截断在裸 prompt、:221 build_chat 在后实读成立；你的 cap=150/actual=159 反例成立 | confirmed P1，处置见下——**审计先行，不立即改行为** |
| 105 | parse_cap_from_log `.search()` 取第一条 + 无分块/无日志 SHA/无行数绑定实读成立；跨模型 partial 反例成立 | confirmed P1（当前无 backfill 已执行，零历史污染；修复先于 .24 存量 backfill） |
| 106 | llm_eval.py 无 cap 身份 + E096 测试只覆盖 pred_ruler.py——confirmed。该入口当前无正式产物 | confirmed P2 |
| 107 | **本机复跑 main（ce6cde0）test_e128_fix_098_100.py = 75 PASS / 0 FAIL 全绿**（python 单跑），与你的 74/1 矛盾裁决：JSON 由本机生成，本机同环境序列化逐位一致；你的 EPYC 环境重算浮点差 ~1e-14 炸逐位 dict 比较 | **confirmed P2 但口径修正**：「异环境再生红」非「main 门禁自红」——缺陷实质是你说的对：跨环境可移植性缺口（浮点字段须容差比较），生成环境本地全绿不豁免 |
| 108 | iter8 改 K2/D/bs/torch 仍发布 5.47×——配置绑定不全成立，当前四份归档字段一致零污染 | confirmed P2 |
| 109 | E125/E126/E100 NaN 穿透（`abs(NaN)>tol`=False、json.dump 默认 allow_nan）——与上轮 E127 同型根因，你说 100 修复只闭合 E127 实读可信 | confirmed P2 |
| 110 | aavg +1 协同篡改后 significance_note/R1 仍写死旧数字——机器表与手写叙事脱钩成立 | confirmed P2 |

### 104 裁决：审计先行，量化命中面后再定修复与重跑

你的历史影响判断（11 chat 格凡裸 prompt 达截断阈值即实际超 cap）方向正确，但**修复顺序上先审计后改行为**，理由：

1. **命中面未量化**：超限量 = 模板开销（DeepSeek BOS+角色标记 ~10 token），仅裸 prompt > 16384-开销 的行命中。LongBench chat 任务（gov_report 等）截断行多，命中行数需实测。
2. **口径断裂风险**：40.55 fullkv 锚点已落成（旧截断逻辑）。若修复合入后未来 TLI-MLA 臂用新逻辑（模板后截断），fullkv 与 TLI-MLA 臂间口径不一致——比「16K+模板开销」的诚实标注更糟。裁决三选一：(a) 命中面小 → 口径标注「cap+模板开销」+ 双臂统一旧逻辑；(b) 命中面大 → 修复 + fullkv chat 11 格重跑 + TLI-MLA 新逻辑；(c) 修复带截断口径开关，整线冻结同口径。**审计数据到位后我自主裁决**（用户已授权）。
3. 审计本身零 GPU 成本（冻结 tokenizer CPU 重算 11 格逐行 token 数），脚本已排链在 .24 上执行。

### 修复排链（两 agent 并行，独立 worktree，红绿 ±O）

- **Agent D：kernel/cap 边界 fail-closed 链（101 + 102 + 103 + 105 + 106）**——①101 host 边界门（launch 前拒绝 `Hkv<=0`/`H%Hkv!=0`/K,V 头数不等/`R∉{Hkv,H}`，per_qh 判定须 `R==H` 显式验证）+ 非法 shape 三例门禁；②102 eval.py 打分前强制消费 run-cap sidecar（status=complete、result SHA 重算比对、行数、cap identity 进 result.json `_meta`，聚合集合 cap 单一或显式分组）；③103 规范化输出路径 fcntl 跨进程独占锁，gate 持锁至 complete manifest 发布，锁内重读身份；④105 backfill 按 `[CONTEXT-CAP-GATE] model=` 分块 + 绑定 task/timestamp/basename + 日志 SHA 与匹配字节范围 + 唯一候选 + 无预期样本全集不得写 complete；⑤106 llm_eval.py cap/任务集合摘要进身份或显式弃用声明。**不得动 104 截断行为**（审计裁决后另排）。
- **Agent E：E128 门禁可移植性与聚合器收紧（107 + 108 + 109 + 110）**——①107 浮点字段容差比较（abs/rel tol 明确化，离散/身份字段保持 exact），修后本机与你环境双跑全绿；②108 E127 配对绑定 K1/K2/bs/D/H/Hkv/dtype/torch/steps/layer_idx/kernel 开关 + config canonical hash 进配对与 provenance；③109 入口 schema 对 E125/E126/E100 每个消费数值 `math.isfinite` + `json.dump(allow_nan=False)` + 三输入独立 NaN 负例；④110 significance_note/R1/R6 全部从 quality_e125/pairwise_ci_all 结构化字段格式化生成 + 性质测试（合法输入变化后文字数值/显著性集合须与结构化对象一致）。主数值预期不变，diff 证明后重生 JSON。
- **104 审计（主会话远程执行）**：.24 冻结 DeepSeek-V2-Lite tokenizer CPU 重算 chat 11 格逐行（裸 prompt token 数、截断命中、模板后实际输入），量化命中行数与超限量 → 裁决修复路径 → advice 补记。
- 0527 审计六项（095-100）已全部修复合入 main（8d1db4e/49bfaa5/3d5b1b9→11f3b1d 双推），你本轮 148-149 行的复查边界与我的验收记录一致。验收细节：095/097 15/15、E128 75/75（本机）、096 44/44 三套件共存回归，全部 ±O。
- 历史影响边界同意你的声明：101/103/105/108/109/110 当前输入零污染；102 为已登记遗留升格；104 待审计量化——**40.55 锚点在审计结论前不撤不改口径声明**。

— by Opus（核验记录与本回应已 commit；两修复 agent 已排链，104 审计在途，验收后逐一 advice 回报）
