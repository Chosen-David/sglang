# TwoLevel 增量缺陷审查：FASA/MInference 产物身份与优化模式门禁

## 审查目标与冻结范围

- 主实现库（只读）：`Chosen-David/two-level-attention` `main`
  - 上次已审代码基线：`b787a372b1e2a3760d65e1854a49ad63b17a328a`
  - 本次冻结：`8b56bbda818db524cc10815ce2561bfd3a3793a4`
  - 差分为 15 个提交、47 个文件；实质范围包括 E132c/E137 修复、FASA vendor 发布与结果报告、MInference 适配修复、LongBench/RULER 两入口和相关批测试。
- SGLang 集成库：`Chosen-David/sglang` `two-level-indexer`
  - 报告写入前远端头：`d609ec8e9b52f0efced8ceea17029de4b7163b3b`
  - 本轮远端新增仅为 advice 回应/合并，没有 two-level 集成源码变更；仅作为已授权报告落点。
- 调用链追踪：CLI 参数 → LongBench/RULER 模型加载与 vendor patch → FASA agreement / MInference pattern 实际读取 → method name / 输出路径 → 结果/回执发布；并复查旧发现 141～147 的修复状态。
- 未修改主库源码、测试、数据或实验结果；没有运行或占用 GPU。

## 本轮结论变化与新增发现

| ID | 严重度 | 状态 | 结论 |
|---|---:|---|---|
| TL-FASA-CALIBRATION-IDENTITY-142 | P1 | partial-fixed / reopened（生产函数 CPU 最小复现 + 两入口调用顺序） | 参数接线、stale 目标比较和名称 hash 已修，但未完成旧报告要求的单次冻结：`stage_fasa_agreement` 先把源 A 复制到运行时目标，FASA 实际从目标 A 构建 records；输出名却在模型加载后重新打开可变源求 SHA。源在窗口内被替换为 B 时，同一次运行形成“runtime=A、name=B”的错绑。 |
| TL-MINFERENCE-PATTERN-IDENTITY-149 | P1 | confirmed（生产命名函数 CPU 最小复现 + adapter 实际 JSON 消费链） | MInference 实际消费 pattern JSON 内容，但输出身份只编码 `basename + starting_layer`；两份不同内容、同 basename 的 pattern 得到同一 method name，且非 TLI 路径没有内容 manifest 门。 |
| TL-TLI-OPTIMIZED-CONFIG-GATE-150 | P2 | confirmed（真实 CLI 解析 + 源文件普通/优化字节码对照） | E135 的原报告 guard 已改显式 `raise`，但同一构造函数仍以 `assert` 实现 near/σ/MoBA/per-q-head 的互斥契约；`python -O` 删除这些守卫，CLI 接受的未定义组合进入 Hq/Hkv 不兼容路径。 |
| TL-FASA-RESULT-PUBLISH-151 | P1 | confirmed（冻结 Git tree 与报告声明逐路径核对） | 新报告称 FASA LongBench 16 任务和 RULER 32K 已“落盘/入库”，但冻结 HEAD 不含所称两个 JSON、逐样本预测、评分结果或回执；干净检出只能得到手录汇总表，无法重算或绑定到代码/输入。 |

## 复现与证据

### TL-FASA-CALIBRATION-IDENTITY-142（重开）：同一次运行的 runtime 与输出名来自两次独立读取

`sparse_attn/info.py:521-558` 的 `stage_fasa_agreement` 从源路径复制到仓库根的 `dimension_rankings/<model>/<dataset>_agreement.pkl`。LongBench 在 `benchmark/LongBench/pred.py:560-574`、RULER 在 `benchmark/RULER/pred_ruler.py:255-271` 先 stage，再切换 CWD 调用 `enable_fasa`。adapter 的 `exp/vendor_baselines/fasa_vendor/qwen3_fasa_patch.py:89-105` 调用官方 `constructe_adaptive_selection`，而 `fasa_utils.py:64-67` 硬编码从仓库根目标读取，因此 runtime 在模型加载阶段已经冻结为目标副本 A。

但输出路径稍后才计算：LongBench 在模型加载完成（`pred.py:631-633`）后于 `715-716` 调 `get_method_name_with_info`；RULER 同样在 patch 完成后于 `pred_ruler.py:295-297` 调用。FASA 名称函数 `sparse_attn/info.py:590-595` 经 `_fasa_agreement_sha8`（`504-518`）**重新打开源文件**，没有复用已 stage/已消费的 bytes。

使用实际 `stage_fasa_agreement` 与 `get_method_name_with_info` 的有界 CPU 复现：先写源 A 并 stage，替换源为 B，再生成 method name。

```text
source_A_sha256=7e72935da6a4a8d6ab1139dbe831b2c8186ed1a7caf5c53e52ba1f9909a6a874
staged_runtime_sha256=7e72935da6a4a8d6ab1139dbe831b2c8186ed1a7caf5c53e52ba1f9909a6a874
source_B_sha256=094deb95e66505419a1e74025c8e9810a30697d2bd6d3e5ce61b3408e53bbf10
emitted_method_name=fasa_vendor_qasper_b1024_t0.8_k16-16_topk_agr094deb95
runtime_bytes_match_source_A=true
runtime_bytes_match_name_source_B=false
```

规范化复现输出 SHA256：`e10b322fe0fef63a76ebd33f46ecbf204c8bdaaef2617ba829f15d9734fc0968`。

142 的“参数未接线、stale 目标不比较、名称不含内容 hash”已经修复，但旧报告同时明确要求“一次冻结、runtime/name 共用 snapshot”及“生成中源文件替换”回归；当前实现没有完成这部分验收。因此这不是重复登记新 ID，而是新代码复查后把稳定 ID 142 从待修状态更新为 **partial-fixed / reopened**。上述顺序是确定性错绑复现；`shutil.copy2` 无锁/非原子带来的多进程目标竞态本轮未并发实测，不把它升级为已确认的数据损坏。

**建议修复。** stage 前一次性读取源 bytes 并构造不可变 snapshot（完整 SHA、schema、model、dataset）；以临时文件 + `os.replace` 原子物化，runtime records 与输出名/回执都消费同一 snapshot。共享目标至少按规范绝对路径加锁；更稳妥的是内容寻址目标，避免不同 agreement 共用可变固定路径。回归覆盖源在 stage 后、patch 中、命名前被替换，以及两个不同源并发 stage；结果必须是同一 snapshot 或 fail closed，不能混装。

### TL-MINFERENCE-PATTERN-IDENTITY-149：pattern 内容不进入 treatment 身份

MInference adapter 在 `exp/vendor_baselines/minference_vendor/qwen3_minference_patch.py:238-248` 打开 `config_path` 并把完整 JSON 载入 `best_pattern`；该内容决定逐层逐头 sparse prefill。两入口分别在 `benchmark/LongBench/pred.py:526-538` 和 `benchmark/RULER/pred_ruler.py:221-232` 消费它。

然而 `sparse_attn/info.py:570-577` 仅取 basename 去掉 `.json` 并拼 `starting_layer`。使用生产命名函数给两份不同内容但都名为 `best_pattern.json` 的文件生成名称：

```text
config_A_sha256=a7a949a0280f8a7e080756e5c15be9497a32c148c17d734db616451651030076
config_B_sha256=dac25f7512b1d03ee008032a95ff91ce51cbc6872b1612711e78962d1cbfa58a
method_name_A=minference_vendor_best_pattern_-1
method_name_B=minference_vendor_best_pattern_-1
same_output_identity=true
```

规范化复现输出 SHA256：`cc108361a238d708ac21bae5029e7b211d4e0b8fc18cca3300d60b5befd83cd5`。LongBench 的 treatment sidecar 仅在 `method=tli` 时写（`pred.py:790-810`），RULER receipt 也只给 TLI 写 treatment manifest（`pred_ruler.py:405-417`），所以内容冲突没有第二道门。

**实际/预期。** 同 `t`/task/output-dir 重跑时，不同 pattern 可命中相同逻辑结果路径；即使时间戳不同，汇总侧看到的 method treatment 身份仍相同。预期 runtime、method name 与回执绑定同一 pattern 内容。当前新增的 MInference 跨层 cache guard 问题已经由主库 `agent_doc/advice/2026-10-13_1446_MInference_layer_cache_guard_by_gpt.md` 单独登记，本报告不重复它。

**建议修复。** 与 TLI 文件型配置相同，模型加载前一次读取 canonical JSON/原始 bytes，记录 realpath、完整 SHA256、schema/层头数量；adapter 直接消费冻结对象，method name 附内容 hash，LongBench/RULER 回执写完整身份。回归必须覆盖同 basename 不同内容、同内容不同路径、加载后源替换、畸形 JSON 和层/头数不匹配。

### TL-TLI-OPTIMIZED-CONFIG-GATE-150：同函数仍有会被 `-O` 删除的公共配置门

`sparse_attn/indexer/tli_indexer.py:203-223` 仍用 `assert` 拒绝：

- `near_select != 4bit` 与 σ/MoBA；
- `per_q_head` 与 MoBA、far/near cluster/sim_greedy、σ。

这些 flag 都是 `sparse_attn/arguments.py` 的公共 CLI；解析器本身允许单个合法值组合。例如真实解析器接受：

```text
--method tli --tli_per_q_head true --tli_far_select cluster
```

对当前**完整源文件**分别 `compile(..., optimize=0/1)` 的复现结果：

```text
parser_accepts=true
parsed_per_q_head=true
parsed_far_select=cluster
guard_literal_in_normal_bytecode=true
guard_literal_in_optimized_bytecode=false
tli_indexer.py_sha256=2aed666b6e11da33ce375fcac53fa988b2c52fbf8ec581a99a1430128214f3d7
```

规范化输出 SHA256：`2e688add7d54e1bb68db7f29e9a2c267cd42137a341d485de552f58ff4c20231`。

绕过后 `per_q_head` 保留 Hq 分数（`tli_indexer.py:1820-1827`），cluster 分数却为 Hkv（`1907-1917`）；随后拼接/散射路径会发生头维不兼容或进入未定义的选择语义。这不是“非法枚举值”问题，所有单 flag 值都通过 argparse，只有组合契约依赖可删除的 assert。

**旧发现关系。** `TL-E135-OPTIMIZED-GATE-144` 报告的 sink/SWA 边界和 sinkswa-only 互斥已经改为显式 raise，本轮复查通过；本项是同一构造函数中未纳入修复的另一组组合门，触发 flag 和下游维度均不同，故登记新 ID。

**建议修复。** 所有用户输入与 shape/组合契约统一用显式 `ValueError`，最好在 argparse 后的单一 config validator 先拒绝，再保留构造层兜底。普通与 `python -O` 对每一互斥对双跑，断言异常类型/消息一致；合法 4bit/per-q-head 与共享 kv-head 路径再做 CPU/GPU 参考差分。

### TL-FASA-RESULT-PUBLISH-151：正式结论没有可审计产物闭包

`research/docs/FASA_独立复现诊断报告_20261013_by_deepseek.md:47-82` 给出 LongBench 16 个任务逐项分数、AVG16=33.31 和 RULER 32K AVG=24.00，并在 `76-79` 声称：

- `exp/trace/results/fasa_lb_16.json + pred/result.json` 已“落盘/入库”；
- `fasa_32k.json` 已“正式落盘”。

但对冻结 Git tree 的逐路径检查为：

```text
HEAD=8b56bbda818db524cc10815ce2561bfd3a3793a4
exp/trace/results/fasa_lb_16.json tracked=false present=false
exp/trace/results/fasa_32k.json   tracked=false present=false
tracked_fasa_result_paths=[]
report_sha256=87502dffd73a9ac3aad1c685b35850da10a7114c35824003e242c393e782bf74
```

规范化检查输出 SHA256：`1f0851173acbb84c3f175607607b24d80b59375fc5149461edfa1d5cbfe2689c`。`.gitignore:16-18` 默认忽略新 `exp/` 产物，正好解释“本地存在但没有 force-add”的发布缺口。与同轮已完整提交 generation、预测、scorer manifest、receipt 的 E119 结果相比，FASA 没有逐样本预测 hash、评分器版本/hash、有效样本数、输入数据身份或 producer SHA，可编辑 Markdown 表不能替代可重算证据。

**影响边界。** 本轮没有取得报告所引用的外部机器原件，因此不判定 33.31/24.00 数值错误；报告表内 16 项均值 33.305625→33.31、RULER 六项均值 24.0，算术自洽，但“已入库”和“适配层正确/方法本质导致下降”的正式结论目前不能从默认分支复核。尤其第 142/149 类 treatment 绑定缺口使结果身份比单纯缺文件更重要：发布前必须证明实际配置、预测与评分同代绑定。

**建议修复。** 按 E119 generation/receipt 口径发布不可变结果包：逐任务预测或可访问的内容寻址对象、结果 JSON、输入/模型/pattern/agreement SHA、producer/scorer SHA、命令、环境、样本计数和完成状态；由正式聚合器重算 16-task/RULER AVG，并让报告引用 manifest。结果包未闭合前把相关表标为外部工作机 provisional，而非“入库”。

## 对已有数据和论文结论的影响

- FASA：当前报告的 LongBench/RULER 数字未被证明错误，但默认分支无法重算；在第 142 项闭环并发布同代回执前，不应把它们当作可复现正式基线，也不能仅凭预算单调性排除所有 adapter/身份问题。
- MInference：所有用相同 pattern basename、相同输出逻辑键生成的历史结果需按实际 JSON 内容重新盘点；若原件 hash 不可恢复，不能合并为同一 treatment。第 149 项不影响已单独登记的跨层 prefill 缓存语义结论。
- TLI：默认非 `-O` 运行及合法配置不受第 150 项影响；`python -O` 或设置 `PYTHONOPTIMIZE` 且使用上述未定义组合的运行需作废/重跑。没有找到证明这些组合已产出正式数据的 tracked receipt，故不声称历史主表已污染。
- SGLang 分支本轮没有相关源码变化；后续集成若复用独立库的 method name/固定 staging 目录，会继承第 142/149 项，接入前应先消费内容快照而不是字符串路径。

## 旧发现复查

- `TL-FASA-VENDOR-PUBLISH-141`：fixed/rechecked。vendor adapter、官方复用代码、转换与 smoke 文件已跟踪；本环境因缺 PyTorch 未完成 import/runtime smoke。
- `TL-FASA-CALIBRATION-IDENTITY-142`：partial-fixed / reopened。原始参数接线、stale 目标对比、名称内容 hash 已修；本轮 A/B 复现表明旧报告要求的单次快照闭包仍未完成。
- `TL-E135-PROTECTED-BUDGET-143`、`TL-E135-OPTIMIZED-GATE-144`：原报告触发条件已改显式块对齐/显式 raise，静态复查为 fixed/rechecked；第 150 项是不同 flag 组合的残余门。
- `TL-E137-PREFILL-GATE-145`：fixed/rechecked。标定只写 pending，decode 首步再冻结；sparse prefill 不消费最终 gate。
- `TL-E137-LLAMA-NOOP-146`：fixed/rechecked。注册前检测 Llama + dynamic gate 并显式拒绝。
- `TL-E137-PARAM-VALIDATION-147`：fixed/rechecked。argparse 与构造层均校验 rows/eps；CPU parser 检查覆盖边界。
- 主库当前的 MInference layer-cache guard advice 仍为 open；没有把源码未修当作关闭，也没有在本报告重复注册。
- 独立只读复核在同一冻结 SHA 上确认：142 应保留旧 ID 并重开；149/150/151 均非现有 advice 的重复，严重度分别维持 P1/P2/P1；复核同时确认 FASA 表内均值算术自洽，不能把发布闭包缺失写成数值伪造。

## 实际验证、环境与未覆盖项

- 环境：Linux 6.18.44 x86_64；Python 3.12.14；AMD EPYC 9V74，容器可见 9 vCPU；NumPy 2.3.5。
- GPU/框架：无 `nvidia-smi`；PyTorch、Triton、Transformers、Einops 均不可用。未运行 Qwen3、CUDA/Triton kernel、GPU 同步/显存/竞态检查、LongBench/RULER 生成或真实数据评分；CPU 证据不能替代 GPU 数值验证。
- 通过：12 个新增/受影响 Python 文件 `py_compile`；FASA/Minference 生产身份函数有界复现；真实 CLI 解析与完整源普通/优化编译对照；Git tree 结果发布核对；E119 批测试普通与 `-O` 均 `ALL PASS`。
- 受环境阻塞：`e132c_audits_core_gate.py`、FASA 批测试、MInference 批测试普通与 `-O` 均在导入阶段因 `ModuleNotFoundError: torch` 退出，不能记为算法失败或通过。八条批测试命令的规范化日志 SHA256：`67275330c7d54debf834c7ada658e0289237725786f8d5c18ea1142ca60f200f`。
- 未做性能声称：本轮没有测 kernel 或完整请求时延，不把代码改写、局部省算或静态分析称为提速。

关键输入 SHA256：

```text
sparse_attn/info.py                                           4ad3c184f0524112a1f52515f6d32db4f36fa309ad4a4336e4757cd8a8879203
sparse_attn/arguments.py                                      79abee3ef0850087a7d33893be92aa84a409307030fd70db226769d9f3d41f91
sparse_attn/indexer/tli_indexer.py                            2aed666b6e11da33ce375fcac53fa988b2c52fbf8ec581a99a1430128214f3d7
benchmark/LongBench/pred.py                                   b1d74de7d2b4a6d7a4b357de8b1c791f93f18a24307aba1f072056b5132e87b9
benchmark/RULER/pred_ruler.py                                 a5e22c507519dc1b4704b8f98d891b473d90a8684e1a8fbc93846f33169e4d32
exp/vendor_baselines/fasa_vendor/qwen3_fasa_patch.py           b1b3b129b843a385e0d3a88889df8c036d712b88318bea6e0b25ed2f9513e5ad
exp/vendor_baselines/fasa_vendor/fasa_utils.py                 52eb64f73ef24a05e64cee0a009b337157afb44ce2b4a66bfbe126077596569a
exp/vendor_baselines/minference_vendor/qwen3_minference_patch.py 93c92bc3892af698a57e113826592b036d5401607718414af492458b70464a5e
research/docs/FASA_独立复现诊断报告_20261013_by_deepseek.md       87502dffd73a9ac3aad1c685b35850da10a7114c35824003e242c393e782bf74
```

## 修复后重测范围与下一检查点

1. 为 FASA/MInference 建统一不可变文件快照，加入“加载后替换源”“同 basename 不同内容”“两进程并发 staging”的普通/`-O` 回归；runtime、名称、sidecar/receipt 三方完整 SHA 必须一致。
2. 将所有 CLI/shape/组合契约从 `assert` 迁到显式 validator；覆盖 near/σ/MoBA/per-q-head 全互斥矩阵，普通/`-O` 异常逐字一致，合法路径再做 reference 差分。
3. 从实际保存 FASA 原件的机器回收不可变 prediction/result/receipt 包；在干净 clone 重跑 scorer 并重算 AVG16/AVG32K，核对报告逐项数值后再标正式入库。
4. 在已授权且空闲 GPU 上记录型号/架构、驱动、PyTorch/Triton/Transformers、dtype、batch/head/sequence、warmup/JIT、同步、显存与完整请求成本；对 FASA/MInference 覆盖短/空/非整块输入、padding、长序列、不同 dtype 与非法配置。修复身份缺陷不等于证明数值/kernel 正确。
