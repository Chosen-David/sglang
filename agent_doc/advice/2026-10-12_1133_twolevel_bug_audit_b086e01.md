# Two-Level Attention 增量缺陷审查（E129 / E130 / E131）

## 1. 审查对象与结论

- 论文主库：`Chosen-David/two-level-attention` `main`，审查 SHA `b086e0161e8ac2fe41aca87462ef5232c92ee462`。
- 主库增量：`4c266796b86b259de5f2a2e93033583f55566c10..b086e0161e8ac2fe41aca87462ef5232c92ee462`，主要为 E129 2×2 消融判决、E130 γ=off 离线分析，以及 E131 五臂 64K 分阶段基准、结果和汇总。提交前出现的并发增量 `751d97f..b086e01` 已重新审查，没有沿用旧 SHA 推送。
- SGLang 集成库：`Chosen-David/sglang` `two-level-indexer`，审查时 SHA `385be5130fdc572203bc4df70dd4a412fc4caa98`；相对上次只有既有 advice，没有新的集成源码，因此本轮未重复审查未变代码。
- 结论：新增确认 12 项可行动问题。其中 E129 的“显著”统计措辞、评分输入集合和 treatment 身份，E130 的页边界模拟，以及 E131 的 cluster 摊销直接影响当前论文/候选/性能结论；其余问题影响聚合闭包、正确性保障、provenance、cold/JIT 公平性或 CPU 分阶段解释。
- 本轮未修改任何实现、测试或实验数据；仅提交本审查报告。

## 2. 范围、环境与未覆盖项

审查路径覆盖：E129 best-file 选择、打分消费集合、run-cap/treatment 身份、任务级 bootstrap 与机制解释；E130 分片执行与合并、near/far L1 页集合；E131 参数构造、五阶段计时、正确性门、原始 JSON/sidecar、报告聚合、cluster cache key、greedy CPU/Triton 派发与结果解释。读取了主库 `TASK.md`、`agent_doc/task/TASK.md`、相关 guide/advice，以及 SGLang 的任务约束与历史报告。

本机为 x86_64 AMD EPYC 7763 虚拟 CPU，9 vCPU；没有可用的 `torch`、Triton、`nvidia-smi` 或已授权 GPU，因此未声称 GPU 复现。实际执行了标准库/AST 反例、JSON 完整性复算、边界/周期静态复算，以及普通和 `-O` 的 `py_compile`。GPU 原始样本来自仓库归档，只用于核对而非本机重测。

证据对象 SHA-256：

- `analyze_e129_ablation_verdict.py`: `04b02358d6b8ad366d9cda2af2b61bc9cde1effe7d513d35dee086c808798ac1`
- `e129_ablation_verdict.json`: `32d8824ed3c07111113c3b7f59919c92419d10258d129c1a81c7017209f5e22e`
- `analyze_e130_gamma_off_grid.py`: `3d8a99eb0c7c2e2ee46700310cad1ef7521fd70d408313ac57202755674cbdab`
- `e130_mavg_gamma_off.json`: `56656094cd70d36ad452b61dac3b6d68be169c5d2491f8907778945044f142ca`
- `e131_arm_bench.py`: `523126d40c0021076a09158ea19e11b6d6955ad328e9bd748143890b2310ab06`
- `_e131_build_report.py`: `8334bd11dec370b8cf91a62d68f94de1258bd39cbac98a4c420449316a57097b`
- `e131_phase1_report.json`: `e79920dabe210a0579eedad1f59d996a74b95014b8f6d1f54c8cad6b5d890778`
- 本轮标准库复现脚本/日志：`169f0be0753843338b061f81fdb9230bcc9ae9322f558aa0ec07f264e8e48670` / `96de31c2eb51effcb2765a6302200f9e514ac0a8f0a3b1f68f4e0f8e0643e136`

## 3. 新增发现表

| ID | 严重度 | 状态 | 定位 | 主要影响 |
|---|---:|---|---|---|
| TL-E130-L1-BOUNDARY-117 | P1 | confirmed（CPU/静态反例） | `exp/trace/analyze_e130_gamma_off_grid.py:87-104,117-139`；对照 `sparse_attn/indexer/tli_indexer.py:927-942,979-1015` | E130 非块对齐 near/far 分界页会被两池重复消费，当前候选的支持集不等价生产实现 |
| TL-E130-MERGE-FAILOPEN-118 | P1 | confirmed；当前 stale/mixed 程度 inconclusive | `exp/trace/analyze_e130_gamma_off_grid.py:256-269,284-328` | 缺 15/16 个 shard、重复样本覆盖或退化校验大幅失败时仍可生成正式排名 |
| TL-E131-AGG-FAILOPEN-119 | P1 | confirmed（CPU 反例） | `exp/bench/_e131_build_report.py:11-45,68-72,132-173` | 错 arm/长度/git/FAST/gate/NaN/sidecar 缺失均可被静默汇总成“64K 五臂报告” |
| TL-E131-AMORTIZATION-120 | P1 | confirmed（静态周期复算） | `_e131_build_report.py:94-108,147-155`；`tli_indexer.py:385-410,484-499` | cclustersim 重建约每 32 步而非报告假定的 64 步，当前长期摊销约低估 2 倍 |
| TL-E131-ROUTE-ATTRIBUTION-121 | P1 | confirmed（静态结论错误） | `_e131_build_report.py:110,147-159`；`tli_indexer.py:548-580`；`greedy_triton.py:60-159` | 报告把默认 Triton 路径误判成 Python 逐 token host launch，根因和“再 Triton 化 3–10×”建议不成立 |
| TL-E131-GATE-COVERAGE-122 | P1 | confirmed 设计缺陷；当前数值影响 inconclusive | `e131_arm_bench.py:241-254,385-409` | 只检查最后一步且无 reference differential，重建边界瞬态错误和等量选错 token 可通过 |
| TL-E131-CONFIG-IDENTITY-123 | P2 | confirmed | `e131_arm_bench.py:114-132,328-356` | `mavg_e127` 实际 c2/full/layer-skip-on，机器元数据却记录 c4/tail/off，自动重放/分组身份错误 |
| TL-E131-COLD-CACHE-124 | P2 | confirmed 测量公平性缺口 | `_e131_build_report.py:52-66,94-110` 及六臂独立进程协议 | 未固定/清理/记录 Triton JIT cache，顺序独立进程的首步不能归因为各 arm 的纯冷建成本 |
| TL-E131-CPU-STAGE-125 | P2 | confirmed | `e131_arm_bench.py:215-237,283-296,373-380` | `prepare_index CPU ms` 吸收了末尾 GPU 同步等待和未归属残差，不能解释为该阶段 host 成本 |
| TL-E129-BOOTSTRAP-126 | P1 | confirmed（算术复算；推断契约错误） | `exp/trace/analyze_e129_ablation_verdict.py:147-156,229-273,287-302,373-380` | 仅重采样 5 个已四舍五入的任务总分，却把区间称为统计“显著”；无法替代任务内 seq/doc 配对不确定性 |
| TL-E129-SCORER-CLOSURE-127 | P1 | confirmed；当前污染 inconclusive | `analyze_e129_ablation_verdict.py:89-144,194-209`；`benchmark/LongBench/eval.py:239-256,273-278,330-333` | best-file 清单与实际评分目录脱钩，额外 200–499 行 repobench 可覆盖清单中的 500 行 best |
| TL-E129-TREATMENT-IDENTITY-128 | P1 | confirmed evidence gap；当前污染 inconclusive | `analyze_e129_ablation_verdict.py:53-73,119-144,194-227,275-321` | 聚合丢弃 run-cap/treatment/git/model/data/seed 身份，不能由预测文件 SHA 相等推出 treatment 因果、跨机确定性或全局 no-op |

## 4. 复现与机制

### TL-E130-L1-BOUNDARY-117：分界页被 near/far 两池同时消费

分析器的 `select_pages` 用“页与 `[lo, hi)` token 区间相交”筛页，随后再把页内 token 裁回区间。生产路径先把 near/far 边界 floor 到 64-token block，两个 block 区间互斥。最小反例：

- `S=32768, sink=128, swa=1024, alpha=.125, beta=.25, BP=64, BS=64`
- `mid=31616`，分界为 `27664`，且 `27664 mod 64 = 16`
- 分析器页 432 同时进入 far 与 near top-k，各消耗一页预算，最终只有 63 个唯一页；生产实现得到 64 个唯一页，并包含分析器漏掉的页 384。

归档把 `.125,.25` 作为 partitioned 第一名及 E2E 候选（`e130_mavg_gamma_off.json:1407-1418,1450-1459`，归档均值 `0.882591`）。按同仓 trace 长度记录与 `collect_trace_lb.py:69-74` 的末四行协议恢复，10 个样本的 40 个尾行中有 39 个分界不对齐。故当前 E130 候选支持集已不满足生产契约，必须按 block 边界重算；因仓库未保留原始 trace 且本机无 torch，尚不能给出 mass 的数值修正幅度。

### TL-E130-MERGE-FAILOPEN-118：不完整或不一致分片仍发布

`merge_and_report(16)` 对不存在的 `/tmp/e130_shard_<pid>.json` 直接跳过，只要求 `per_sample` 非空；`dict.update` 会静默覆盖同名样本。分片也不携带 code/config/model/trace/run identity。三项退化等价只写入 `abs_diff`，不设拒绝阈值。

标准库反例只创建 shard 0 的一个样本，再调用原函数：返回 `n_samples=1,n_arms=1` 并生成 overall 候选；把三项退化差设为 `0.7/0.7/0.6` 仍正常落盘。当前归档有 16 个唯一全局样本，集合与 E98 的 16 样本一致，均值复算无算术差，退化差也接近零，因此没有证据表明本次结果确实缺样本；但 raw shards 和运行 identity 未归档，无法证明 16 片同代、完整且未覆盖。

### TL-E131-AGG-FAILOPEN-119：聚合器不验证输入闭包

构建器按文件名读取 `runs[0]`，既不验证 sidecar，也不验证 arm、S0、steps/warmup/layer、GPU/git/source hash、FAST/config、gate 或 finite。标准库反例将 aavg 输入改成错误 arm、`S0=8192`、`git=deadbeef`、`steps=999`、`gate.out_finite=false`、`sum_gpu_ms=NaN/999`，并不给有效 sidecar；脚本仍 `rc=0`，报告继续自称 64K，且丢失冲突来源身份。

当前 8 份 JSON 的 sidecar 经逐一复算均匹配，主五臂协议、git、源码哈希、设备版本一致，所有当前数值 finite、最终 gate 为 true；因此除 TL-E131-CONFIG-IDENTITY-123 外，未发现现有文件被混入或篡改。该缺陷是发布链 fail-open，而不是已证明当前全部数值错误。

### TL-E131-AMORTIZATION-120：near cache key 变化频率不是 64 步

报告只在 38 步窗口捕获 step 36 的 `156.7108 ms` 峰值，并用 `/64` 得 `2.4486 ms/step`。生产 `_update_near_cluster` 的 key 是 `(far_hi, near_hi)`；near 或 far 任一块边界变化都会全量重建。按报告同一协议 `S0=65536, bs=64, sink=128, swa=128, alpha=.125` 复算，前 192 步 key 在 `36,64,109,128,182,192` 变化，共 6 次，平均间隔 32 步；复算文本 SHA-256 为 `cd2df24ac1cea6b934d4e83b9d12f4d9650e392281b702cd933512f226cf5714`。

若暂用当前单次实测峰值作近似，`156.7108/32=4.8972 ms/step`，约为报告值的 2 倍；但每次重建成本是否相同仍需 192+ 步真实长跑，不能把 4.8972 当成已实测精确值。现报告中的“稳态 2.10 + 摊销 2.45 ≈ 4.55 ms/step”及 phase-2 expected gain 均应撤回后重算。

### TL-E131-ROUTE-ATTRIBUTION-121：报告描述的是回退实现而非默认 CUDA 路径

报告称 `_greedy_cluster_pass` 是 Python `for token`、每 token 约 7 次 kernel launch，并建议“CUDA graph 化或 Triton 化”。实际调度器在 CUDA 且 `SGLANG_TLI_GREEDY_KERNEL` 未关闭时默认进入 Triton；长度超过 16384 时按 chunk 调用。`greedy_triton.py` 的 token 循环位于单个 Triton kernel 内，不是 Python 每 token host launch。

E131 harness 只记录 `TLI_E127_FAST` 与 `CUDA_VISIBLE_DEVICES`，未记录或断言 greedy backend，也无 route counter/profile。因此 8–11 秒首步和 156 ms 重建可以作为时间观测保留，但不能据此归因为 45 万次 Python launch，更不能宣称“再 Triton 化可 3–10×”。先持久化实际派发 backend/flag/route，再做 profiler 证据闭环。

### TL-E131-GATE-COVERAGE-122：最终一步弱门禁不能保护边界步骤

门禁只在 `warmup+n_steps-1` 检查：存在任意选中、输出 finite、token 数落在宽区间。当前上界允许约 255 到 6208 个 token，通过门禁也不能证明选中的是正确 token。cclustersim 的关键全量重建在 raw step 36，而最终检查 step 37；边界步若错、下一步恢复会被漏过。固定选错 1024 个 token 同样通过。

当前归档最终步均 finite 且选中数为 1024，只能证明最后一步的弱性质，不能证明每次 key-change、fast path 或五臂语义正确。至少应对冷建、所有 cache-key 变化步和最终步保存并与可靠 reference 差分 mask/token 集合及输出，按真实预算做精确门禁。

### TL-E131-CONFIG-IDENTITY-123：锚点的机器字段与实际运行参数冲突

`make_args('mavg_e127')` 实际使用 `cmp_ratio=2, subspace=full, layer_skip=True`；meta 却无条件把 `COMMON_ARGS` 写成 `cmp_ratio=4, subspace=tail, layer_skip=False`，只在 arm note 中给自然语言提示。当前 `e131_arm_bench_mavg_e127.json` 已落盘此冲突。

执行参数本身来自正确的 `make_args`，报告也用测得 `sum_gpu_ms` 计算校准比，所以当前 `1.6888/1.5201` 没有因此被数值改写；错误影响机器可读 provenance、自动分组和可重放性。应从同一个冻结后的 `argparse.Namespace` 构造 indexer 和序列化 effective config，并保存 identity hash。

### TL-E131-COLD-CACHE-124：首步跨臂比较混合 JIT/cache 状态

六臂由独立进程顺序执行，但未清理、固定或记录 `TRITON_CACHE_DIR`、编译版本、kernel 特化/cache key。归档首步 wall 为 mavg `2597.8 ms`、mavg_e127 `875.9 ms`、aavg `569.2 ms`、mminmax `560.2 ms`；对应 first prepare GPU 约为 `204.0/132.8/142.7/128.3 ms`，host wall 的巨大差异没有被 GPU event 同比例解释，符合首进程 JIT/初始化混入。

因此首步/cold-build 的跨臂精确比较与归因不可用。warmup 后稳态 median 不因这一点自动失效，但仍只有单次顺序 run。应分别测 clean-cache cold 与预编译 steady，记录 cache/JIT identity，并交替顺序重复。

### TL-E131-CPU-STAGE-125：prepare CPU 时间实际是同步残差桶

`prepare_index` 前后没有 host timer。代码在整个 step 最后 `evs[5].synchronize()`，再用同步后的总 wall 减去其余四段 host interval，全部残差写入 `cpu_ms['prepare_index']`。该字段包含 GPU 排队/同步等待、`prepare_cu_seqlens`、`q_ids`、event 等未归属工作，不是 prepare host launch 成本。

当前 phase-1 主表只发布整体同步 `cpu_wall_ms`，其命名成立；五阶段 GPU event 值也不受此项影响。错误限于 raw/console 的 CPU 分阶段归因。应直接在 prepare 前后计 host enqueue，另存 `sync_wait` 和 `unattributed_residual`；无需仅因本项重跑 GPU event 主值。

### TL-E129-BOOTSTRAP-126：五任务描述性重采样被写成统计显著性

E129 只对 5 个任务级 scorer 总分（已四舍五入到两位）作有放回 bootstrap，并用 percentile CI 是否含 0 直接标记 `significant`。这违反仓库已有实验约束：任务内应按 `_id` 或源文档簇做配对重采样，五任务 bootstrap 不能替代 seq/doc 级不确定性。

独立复算与归档逐位一致：gate 五个任务差为 `[-12.76,-5.81,-1.15,-1.73,-0.90]`，均值 `-4.47`，脚本区间 `[-8.998,-1.166]`；interaction 为 `[0,.13,.02,.71,.16]`，均值 `+.204`，脚本区间 `[.034,.462]`。后者的正下界是 n=5 重采样构造的机械结果：只有 5 次都抽到唯一 0 任务时均值才为 0；它没有纳入任务内样本噪声。作为简单反证，gate 的精确双侧 sign test 为 `p=.0625`，interaction 去掉零后为 `p=.125`；脚本还同时报告 3 臂和 5 个 effect 的 8 个区间而无预注册/多重校正。

可以保留“当前五任务 gate 点差均为负”、各任务点估计和明确标注的“五任务描述性 task-bootstrap 区间”；必须撤回普通统计意义的“显著”以及向论文任务总体外推，尤其是 `+.204` interaction 的显著性。修复需从原始 JSONL 按 `_id` 对齐四臂，任务内按 seq/doc 簇配对重采样，再做等任务权重层级汇总，并预注册主效应/交互和同时区间或校正。

### TL-E129-SCORER-CLOSURE-127：记录的 best 可能不是实际消费的文件

`resolve_best` 要求每个 task 有唯一达到 `EXPECT` 的候选，其中 repobench 期望 500 行；但 `run_eval` 把整个 arm 目录交给 scorer，统一只传 `--expect-count 200`，且不传样本 manifest。随后它把 `result.json` 的所有文件按 `fn.split('-')[0]` 写入 task 字典，同 task 后出现的文件静默覆盖前者。

因此目录里若有唯一 500 行 repobench best，再有一个排序靠后的 200–499 行且 sidecar 合法的候选：best 门通过、eval 的 200 行门也通过，最终短候选分数可覆盖 500 行 best，而 `input_manifest` 仍记录前者，形成“记录 A、统计 B”。标准库最小映射反例中，依次输入 full score 70 与 partial score 10，现有循环最终得到 `scores['repobench']=10`；证据摘要 SHA-256 为 `b623a2cef9cc3ba1a7b5f27181856c13792942b7413a69c71950d8eb84021a23`。

当前仓库未归档三个 arm 目录清单或 scorer 的 `result.json`，无法反证当时没有额外候选，故 E129 的 repobench/AVG/CI 消费闭包为 inconclusive。修复应只评分 15 个冻结 `resolve_best` 文件（隔离目录或显式文件清单），按 task 使用精确 EXPECT 与 `_id`/answers manifest，并要求 scorer 输出 basename/hash 与输入清单一一对应，拒绝任何额外 JSONL。

### TL-E129-TREATMENT-IDENTITY-128：输出相同不等于处理因果已绑定

脚本的 `ARM_CONFIGS` 是手写说明；清单只存 basename、行数、prediction SHA 和人工机器标签。新版 evaluator 会把 run-cap identity 写入 `result.json._meta`，但 `run_eval` 只返回分数和 scorer backend，丢弃 cap/protocol/provenance；三臂只比较 backend，既不比较彼此，也不与 C0 的 cap/treatment/code/model/data/seed 身份闭合。C0 还只读 E125 的 task 分数和 backend，没有绑定 E125 verdict 自身哈希。

当前可核对的事实是：15 个 `(arm,task)` 键唯一、数值 finite，C2 五个 prediction SHA 与当前 E125 cclustersim 清单逐项相同，gate 五任务点差均负。Prediction 字节相同只证明这 5 对归档内容相同，不能证明 C2 确实执行 tail、C0/C2 使用同样本同代码独立跨机生成，也不能单凭静态机制说明宣称“跨机推理确定性”或全局“零信息量”；复制、陈旧或 treatment 误标仍不可区分。C1/C3 的小 interaction 更没有 receipt 闭包。

每格应保存并聚合校验 producer git/dirty、解析后的 `TreatmentSnapshot` hash、完整 argv、model/tokenizer revision、data/_id manifest、seed/decoding、run-cap identity、treatment sidecar 与 raw SHA；四臂逐字段匹配后再解释因果。当前 raw、sidecar 和三个 `result.json` 未入库，所以实际污染状态保持 inconclusive，而不是推定已污染。

## 5. 对已产出数据和论文结论的影响

1. **必须重算后再用于候选选择**：E130 γ=off 的 partitioned 候选使用了不等价生产的 L1 边界支持集；其现有 mass/rank 不能直接作为 E2E 决策依据。
2. **必须降级的统计结论**：E129 gate/interaction 的普通“显著”措辞；当前仅支持五个选定任务的点差与描述性 task-bootstrap，不能替代 seq/doc 配对区间。
3. **尚未闭合的 E129 汇总**：best-file 与 scorer 消费集合可能脱钩，且 run-cap/treatment receipt 被丢弃；可保留 C2 五对归档 prediction 字节相同和 gate 五任务点差均负，但不能把它们扩成 treatment 因果、跨机确定性或全局 no-op。
4. **必须撤回/重算的性能推论**：E131 cclustersim `/64` 摊销、4.55 ms/step 总成本与 phase-2 expected gain；当前最多保留单次峰值和稳态原始测量。
5. **必须撤回的根因表述**：E131 “Python 逐 token 约 45 万 launches”与“需要 Triton 化 3–10×”；实际默认 CUDA 路径已经有 Triton 实现，且本实验没记录真实 backend/profile。
6. **可保留但须降级说明**：五臂当前原始 GPU steady medians、sidecar 和 finite 性经归档核对一致，尚无证据表明数值文件被混入；不过正确性门不足以证明 fast path 全步骤语义正确。
7. **不影响当前 GPU 主表的项目**：`mavg_e127` 元数据错配未改变实际执行参数；CPU prepare 分阶段误标不改变 GPU event 主值或整体同步 wall。

## 6. 修复与最小重测范围

1. E130 分界以生产 block 区间为唯一真源，确保 near/far L1 页互斥；针对所有 `split mod 64`、两侧配额、空/小区间做 reference 差分。重跑完整 16 样本 γ=off 网格及排名。
2. E130 合并预登记完整 shard/sample 集，要求 `0..n_proc-1` 齐全、样本唯一、每片 identity 一致；绑定 code/script/config/model/trace/E98 hash；退化差超过阈值拒绝发布；输出逐臂 sample/row/layer 支持数。
3. E131 构建器引入严格 schema：唯一 run、expected arm/S/protocol/git/source hash、sidecar、finite、gate、effective config 全部 fail-closed，临时文件验证后原子发布，拒绝 JSON NaN。
4. 将实际 effective args 和真实 greedy backend/route 写入每个样本；CUDA 路径显式断言预期 Triton 或明确记录回退原因，并用 profiler 核验瓶颈后再提出 kernel 优化。
5. 对 cclustersim 做至少 256 decode steps 的长跑，覆盖多次 far/near key 变化；逐次记录 key、重建类型、耗时，再按实际频率求摊销，不能用固定 `/64`。
6. 正确性差分覆盖 cold build、每次 key-change、非整块长度、并列分数、不同 batch/head/dtype、空/非法输入；与可靠 reference 对比 token 集、mask 和输出，区分允许浮点误差与语义错误。
7. cold 与 steady 分开跑；cold 每臂使用独立空 cache 并随机化顺序，steady 预编译后交替多轮；保存 GPU/驱动/编译器/Triton/cache key。CPU 阶段直接计 enqueue，并单列同步等待。
8. E129 评分只消费冻结清单，使用精确 task 行数及 `_id`/answers manifest；聚合器校验 scorer result 与 prediction 清单一一对应，并保存 result/sidecar 哈希。
9. E129 四臂绑定相同 producer/treatment/run-cap/model/data/seed receipt；从逐样本 metric 做任务内配对、外层等任务权重的层级区间，预注册并校正主效应/交互检验。

## 7. 旧发现复查与下一检查点

- `TL-REPO-WIPE-092` 继续保持 refuted：独立主库迁移不是误删。
- E130 已有 advice 指出的 GQA raw-logit、sink/SWA 预算和部分臂支持数不齐仍未被本轮源码修复；本报告没有把这些旧问题重复编号。当前 support 复算为 72 臂×16 样本，其余臂为 14/8/7/6 样本，与既有提醒一致。
- SGLang 本轮无新的相关接口/内核变化，旧问题不因源码未变而视为 fixed。
- 下一次只在主库或 SGLang 出现新的相关实现、实验、聚合、CPU/GPU kernel、派发/JIT/构建改动时增量复查，优先复查 E129 逐样本/身份闭包、E130 block 边界、E131 长跑周期与真实 backend/profile 证据。
