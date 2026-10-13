# TwoLevel 增量缺陷审查：E137 动态 gate 的 prefill 因果域与跨模型接入

## 审查目标与冻结范围

- 主实现库（只读）：`Chosen-David/two-level-attention` `main`
  - 上次已审代码基线：`44a0bc1c0e96769405494bbe4c6a77dd84b87478`
  - 本次冻结：`b787a372b1e2a3760d65e1854a49ad63b17a328a`
  - 新增相关源码提交：`08cd25c`（E137 动态 D' gate）与 `ea0e4aa`（K 布局修正）；`4440402`、`6d750a0` 为结果/报告，`b787a37` 为 advice。
- SGLang 集成库：`Chosen-David/sglang` `two-level-indexer`
  - 报告写入前远端头：`73f8a69d7e5f3b89cce893bea1cbe970fc1975e9`
  - 本轮无 two-level 相关集成源码变化，仅作为授权报告落点。
- 入口追踪：LongBench/RULER 参数 → `register_patch` → Qwen3/Llama prefill patch → `calibrate_dyn_dgate` → decode/sparse-prefill `prepare_mask`；同时复查 E135 新提交网格与旧发现 `TL-E135-PROTECTED-BUDGET-143` 的关系。
- 本轮没有新增 CPU/CUDA/Triton kernel。未改主库源码、测试或结果。

## 新增发现

| ID | 严重度 | 状态 | 结论 |
|---|---:|---|---|
| TL-E137-PREFILL-GATE-145 | P1 | confirmed（调用顺序、状态消费链与纯 Python 因果对照） | Qwen 开启动态 gate 与稀疏 prefill 时，先用**完整提示尾部**决定 `skip_far`，随后同一状态被较早 prefill chunk 的 `prepare_mask` 消费；相同前缀可因未来后缀不同而得到不同的早期 prefill 路由，违反 causal prefill / “标定供 decode 使用”契约。 |
| TL-E137-LLAMA-NOOP-146 | P1 | confirmed（生产注册入口与两条模型 patch 静态可达性） | `register_patch` 对 Llama 注入相同 TLI/E137 参数与输出身份，但 Llama prefill 只 `clear()` 后走 dense attention，从不调用动态标定；`--tli_dyn_dgate` 在 Llama 上静默成为 no-op，结果仍带动态 gate 标签。 |
| TL-E137-PARAM-VALIDATION-147 | P2 | confirmed（解析器、形状公式及普通/优化模式重放） | 新增 `rows/eps` 参数无范围或有限值校验；`rows=0` 及部分负值在分配 logits 后形成 query/mask 行数不一致并崩溃，足够负的值又会以空均值得到 NaN 并静默关 gate；NaN/负 eps 同样静默关 gate，eps>1 则几乎全开 gate。 |

### TL-E137-PREFILL-GATE-145：完整提示标定状态泄漏到较早 sparse-prefill chunk

Qwen prefill 路径在 `sparse_attn/patches/qwen3_attn_patch.py:63-90` 按以下顺序执行：

1. `indexer.clear()`；
2. `calibrate_dyn_dgate(query_states, key_states)`，输入覆盖完整提示；
3. 若 `TLI_SPARSE_PREFILL=1`，调用 `sparse_prefill_attn`。

标定函数在 `sparse_attn/indexer/tli_indexer.py:316-376` 用完整序列尾部 query 与完整 K 得到一个层级布尔 `skip_far`。稀疏 prefill 随后逐 chunk 计算；从首个稀疏 chunk（默认配置下通常是整体第二个 chunk）起，`sparse_attn/ops/eager_prefill.py:147-172` 调用同一 indexer 的 `prepare_mask`，会消费该布尔值。因而当前提示的未来后缀已经参与决定较早 chunk 的 far 是否存在。

最小状态对照固定相同前缀、只改变不可见后缀，并令完整提示标定分别得到 `skip_far=True/False`：

```text
causal_pair suffix_A=true suffix_B=false
qwen_calibrate_lines=[75]
qwen_sparse_prefill_lines=[88]
sparse_prefill_prepare_mask_lines=[164]
```

该对照不声称复现模型数值，而是验证产生非因果依赖所需的生产调用顺序与共享状态全部可达。触发条件为 Qwen TLI、`--tli_dyn_dgate`、`--tli_sparse_prefill`（环境门实际开启）、提示足够长而进入稀疏 chunk、且完整提示后缀能使 gate 决定翻转；动态决定还必须未被静态 `_resolve_skip` 覆盖，例如关闭静态 layer skip 的 dynamic-only 配置。当前默认静态 gate 会覆盖动态决定是已有 E137 advice 已记录的另一项组合缺口，不构成本项反证。

这与历史 `TL-PREFILL-CAUSAL-001`（chunk 末行 query 影响同 chunk 较早行）是两条不同来源：本项是**完整提示的层级 gate**影响更早 chunk，即使修复 chunk 内共享 query，仍会存在。

**实际/预期。** 实际是 prefill 的选择集合依赖未来 token；预期是 E137 标定在 prefill 完成后冻结，仅供该请求 decode 消费，或 sparse prefill 自身使用只依赖当前可见前缀的局部决定。建议为动态状态增加明确生命周期：sparse prefill 计算期间禁用最终 decode gate，完整 prefill 结束后再一次性写入 decode snapshot。回归需构造“相同前缀、只在首个稀疏 chunk 的可见右界之后出现不同后缀”，断言分叉前位置的 prefill 输出/选择逐位相同；默认 chunk=2048 时可让两提示共享 `[0,4096)`、仅后缀不同并令完整提示 gate 翻转。同时覆盖 dense/sparse prefill、静态-only、动态-only、组合 gate。

### TL-E137-LLAMA-NOOP-146：Llama 结果身份声明动态 gate，执行路径却不标定

`sparse_attn/patches/patch.py:55-75` 对 `Qwen3Attention` 和 `LlamaAttention` 都创建相同 `TLIIndexer(args, snapshot)`；LongBench 配置仍包含 `Meta-Llama-3.1-8B-Instruct`。参数快照/哈希包含 `tli_dyn_dgate*`，`sparse_attn/info.py:543-552` 还为可读名追加 `_dyndgate`。

但两条 prefill patch 不对称：

- Qwen：`qwen3_attn_patch.py:66-75` 在 `clear()` 后调用 `calibrate_dyn_dgate`；
- Llama：`llama3_attn_patch.py:53-66` 只 `clear()` 后直接执行 dense attention，全文件无 `calibrate_dyn_dgate` 或等价调用。

`clear()` 又在 `tli_indexer.py:2283-2309` 把动态 `skip_far` 复位为 `False`。因此 Llama 动态-only 运行不会产生任何动态跳层；若静态 gate 开启，decode 只会继续使用静态集合。最小 AST/调用点检查输出 `llama_has_calibrate=false`，而 Qwen 调用行存在。

**影响。** 使用 Llama 且传入动态 gate 的后续质量/性能数据会被错误标成 E137 treatment，实质却不是动态 gate；同名/哈希能够区分参数，但不能证明参数被消费。当前新增 E137 JSON 是离线 calibration 分析，没有提交实际 LongBench/RULER E2E receipt，因此本轮未确认已有 Llama E2E 数字已污染。

**建议修复。** 二选一 fail closed：为 Llama 在 `clear()` 后、dense prefill 内按相同布局/scale/mask 契约调用标定；或在注册/冻结 treatment 时检测模型家族并拒绝 Llama 的 `--tli_dyn_dgate`。增加 Qwen/Llama 参数消费测试，receipt 记录每层标定次数、决定摘要和首个 decode 的实际决定；不能只检查文件名/参数哈希。

### TL-E137-PARAM-VALIDATION-147：非法 rows/eps 未在分配前拒绝

`sparse_attn/arguments.py:157-166` 仅使用 `type=float/int`；构造函数 `tli_indexer.py:146-148` 直接转换，没有约束。`calibrate_dyn_dgate` 的 `rows = min(config, S)` 后，query 用 `q[:, :, -rows:, :]`，mask 位置却用 `arange(S-rows, S)`：

- `rows=0` 时 Python `-0` 等于 `0`，query 实取 `S` 行，但位置数组为 0 行；
- `rows=-1` 时 query 实取 `S-1` 行，位置数组仍为空。
- `rows<=-S` 时 query 与位置数组都可为空，后续空均值得到 NaN，`NaN < eps` 恒假而静默 no-skip。

以 `S=8` 重放精确切片/位置公式：

```text
invalid_rows_shape_lengths rows=0  query_rows=8 mask_rows=0
invalid_rows_shape_lengths rows=-1 query_rows=7 mask_rows=0
```

真实实现会先创建 `[B,H,query_rows,S]` 的 float logits，随后在 causal `masked_fill` 处形状失败，长序列上还会在报错前产生显著 GPU 分配。另一方面，`r_l < eps` 对 `eps=NaN` 或负数恒假，静默关闭动态 gate；`eps>1` 对合法概率质量几乎恒真，静默大面积跳层。普通 Python 与 `python -O` 的重放结果一致，说明这不是 assert 优化模式专属问题。

**建议修复。** argparse 与构造函数均无条件校验 `rows >= 1`、`math.isfinite(eps)`、`0 <= eps <= 1`，在任何 tensor/GPU 分配前抛出 `ArgumentTypeError/ValueError`；同时校验 `H % Hkv == 0` 及支持的布局。普通模式与 `-O` 回归应覆盖 `{rows=-1,0,1,S,S+1}`、`eps={NaN,-0.1,0,0.05,1,1.1}`，合法边界再与可靠 dense reference 差分。

## 新结果对旧发现的影响

- `TL-E135-PROTECTED-BUDGET-143` 仍为 open/confirmed。新提交 `e135_sinkswa_grid.json` 包含 `sink=16`，但分析器对 token 分数先精确切片 `[sink, mid_hi)`，天然排除了生产实现里“sink 重入 top-k 并重复耗 K2_mid”的路径；其约 2.1pp 网格敏感性是**参考/trace 语义**，不能作为当前非整块生产实现的验证，也不能证明第 143 项已修复。
- `e135_sinkswa_only_verdict.json` 的 E2E 臂使用默认块对齐 `sink=128,swa=128`，不受第 143 项的非整块反例直接影响；无需因本轮发现重跑该默认点。
- E137 gate 静态/动态相互覆盖已由主库 `2026-10-13_1040_E137_gate_composition_scope_by_gpt.md` 记录，本报告不重复注册。上述第 145、146、147 项是该 advice 未覆盖的新调用域、模型接入和非法输入问题。

## 对已有数据和论文结论的影响

- 新增 E137 calibration JSON 是离线分布分析，不经过 Qwen sparse-prefill 或 Llama patch；本轮三项不推翻其中已记录的 far-mass 数值。
- 当前树未提交可绑定到 E137 动态 gate 的实际 CLI、producer receipt 或 E2E 结果，故不声称历史模型结果已经受第 145/146 项污染。未来启用 Qwen sparse prefill 的 E137 结果在修复前不可作为 causal prefill 证据；未来 Llama E137 结果在证明参数消费前不可作为动态 gate 对照。
- E135 `sink=16` 网格可继续作为 reference trace 使用，但不得写成“生产路径同语义验证”；若已有外部生产 `sink=16` 数据，仍按第 143 项建议在修复后重跑。
- SGLang 分支本轮无相关源码变更，未发现新的服务集成数据直接受影响。

## 实际验证、环境与未覆盖项

- 环境：Linux 6.18.44 x86_64；Python 3.12.14；AMD EPYC 9V74，容器可见 9 vCPU。
- 实际执行：受影响 6 个 Python 文件 `py_compile`；普通 Python 与 `python -O` 两次运行只读 AST/切片/状态复现；调用入口、模型注册、输出身份和 E135 result/analyzer 对照审查。
- 两次复现输出逐字一致。复现脚本 SHA256：`54d179a2e46e775a5c73ac1aca300d2cedb0199e9819f7373de3d7b06160a287`。
- 当前环境无 PyTorch，未运行 Qwen3/Llama 模型；无已授权可用 GPU，未运行 CUDA/Triton、显存检查、竞态检查或 LongBench/RULER。静态可达性不能替代 GPU 数值验证。
- 独立复核在同一冻结 SHA 上复查三项调用链和严重度：第 145、146 项维持 P1/confirmed，第 147 项维持 P2/confirmed；复核同时确认当前仓内未见可归因于这些缺陷的 Llama dynamic 或 Qwen P-on E2E 产物。

关键输入 SHA256：

```text
sparse_attn/arguments.py                  dada9b9d038207e1006a0d83ab50413cfc63fa148fbebb6a19dd2a4aac18199a
sparse_attn/indexer/tli_indexer.py        ea815f2b262a48f10ac1dc5648cda409b6c2f10ae57757b23705f18584e5801f
sparse_attn/ops/eager_prefill.py          4bb86dcdbe8a61feb42b204d65e4d9940f23edaa1689fd945a4b3179101cea20
sparse_attn/patches/llama3_attn_patch.py  89e25e5bd8fda9b72e5ba819d62aba8d26ede9d836a92791a88a7661b36b88fd
sparse_attn/patches/qwen3_attn_patch.py   c99ede1ab39416e879b2dc19d972d668a2e8c8242a0d74a79ca78f8d763a9ccd
```

## 修复后重测范围与下一检查点

1. 分离 `prefill-local` 与 `decode-frozen` gate 状态；用同前缀不同后缀的 Qwen sparse-prefill 对抗用例做逐位置输出/选择差分，覆盖小输入、非整块 chunk、长序列与 fp16/bf16。
2. 给 Llama 实现同契约标定或在 treatment 冻结前拒绝；以每层标定 receipt 和首 decode 实际决定证明参数被消费，再跑 Qwen/Llama 交叉模型对照。
3. 加入 rows/eps 的普通与 `-O` 非法输入矩阵；合法最小行数、rows>S、空/短 far 区分别核对无越界、无 NaN、无未初始化状态。
4. 在空闲且已授权 GPU 上记录设备/架构、驱动、PyTorch/Triton/Transformers 版本、dtype、序列/批/头配置、同步与 warmup；分别报告标定额外时延/显存、decode 收益和完整请求成本，不能用局部 gate 省算代替端到端收益。
5. 修复 `TL-E135-PROTECTED-BUDGET-143` 后重跑生产 `sink=16` 点，并把生产结果与现有 reference grid 分栏；默认 `128/128` 保留为逐位锚。
