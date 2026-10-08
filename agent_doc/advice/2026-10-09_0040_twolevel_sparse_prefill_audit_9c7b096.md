# TwoLevel 稀疏 Prefill 与 near/SWA 边界审查

## 结论摘要

审查对象为 `two-level-indexer` 分支提交
`9c7b096f723816aaa3cfc645aebfb1a40789d669`，重点复查新提交
`07a1e76cd`（稀疏 prefill）和 `bb8c6712c`（far 池为空时的控制流修复）。

本次确认两个会改变模型语义的新增 P1 问题：

1. 稀疏 prefill 用 chunk 最后一行 query 为整块所有行选 key，导致块内较早行的输出依赖未来 query，破坏自回归前缀不变性。
2. 同一共享选择集只强制保留 chunk 末行的 SWA，不能保证块内每一行自己的 128-token 局部窗口。

另外复查确认：此前 near/SWA 分界问题仍未合入当前分支，`bb8c6712c` 只修复了 far 池为空之后的控制流，没有修正使常规 `alpha=1` 真正得到空 far 池的边界公式。当前提交不宜进入 E114b 的 P-only / P+D 正式数据生产；先修复并通过前缀不变性、逐行保护集和端到端接入测试。

## 审查范围与证据绑定

| 项目 | 值 |
|---|---|
| 审查 SHA | `9c7b096f723816aaa3cfc645aebfb1a40789d669` |
| 重点源码提交 | `07a1e76cd`、`bb8c6712c` |
| 入口 | `benchmark/LongBench/pred.py` -> `patches/qwen3_attn_patch.py` -> `ops/eager_prefill.py` -> `indexer/tli_indexer.py` |
| `eager_prefill.py` SHA256 | `869bed831a92fdf0f97d4ac06b9ab468f997864717e4ee654767136b2b8a09f4` |
| `tli_indexer.py` SHA256 | `74fe5e20cb3eaadef8532842c7c94af44a323a853b961950e4becf5a6afe26cf` |
| `pred.py` SHA256 | `116db7bc9ccf92303e427d9a87036260fd095309d80e7b51beba698a523c5632` |
| `test_near_swa_boundary.py` SHA256 | `f79fc26fe34d82342b5d1cbce9c753dcca3093c5d82e6ebee35ff0b1ec4c7460` |
| 实际环境 | Python 可用；未安装 PyTorch；未发现可用 GPU |

已阅读根 `TASK.md`、`agent_doc/task/TASK.md`、相关 README、实验设计和此前 advice。旧 B09（配置身份不足）与 near/SWA 边界问题在下文标为“复查仍开放”，不重复计为新根因。

## 发现表

| ID | 状态/严重度 | 位置 | 结论 |
|---|---|---|---|
| `TL-PREFILL-CAUSAL-001` | **confirmed / P1 / 新增** | `ops/eager_prefill.py:126-148`，尤其 `138-147`；`_chunk_rows_attn:71-87` | chunk 末行 query 决定整块选择集，较早行虽然经过 key 侧 causal 交集，仍会受未来 query 内容影响。 |
| `TL-PREFILL-SWA-002` | **confirmed / P1 / 新增** | `ops/eager_prefill.py:128-147`；`indexer/tli_indexer.py:1090-1092` | 共享选择只强制 chunk 末端 SWA；较早行自己的行相对 SWA 没有被强制恢复。 |
| `TL-BOUNDARY-NEAR-SWA-001` | **confirmed / P1 / 旧发现复查仍开放** | `indexer/tli_indexer.py:335,803-806` | 动态 near 起点从序列尾减 `near_len_dyn`，没有先减 SWA；`alpha` 定义与实际 near/far 分区不一致。 |
| `TL-TEST-TMP-001` | **confirmed / P1（验收阻塞）/ 新增** | `test_near_swa_boundary.py:38,55-56` | 测试将“修复版”硬编码到未提交的 `/tmp/near_fix_v2`；干净检出不可复现，且机器残留可改变结果。 |
| `TL-PREFILL-METRIC-001` | **confirmed / P2 / 新增** | `ops/eager_prefill.py:126-147`；`indexer/base.py:45-49`；`metrics.py:24-34`；`pred.py:289-305` | `budget` 混合 chunk 末端 prefill 选择记录和逐 token decode 记录，且不是每行实际有效 mask 大小。 |
| `TL-PREFILL-PROVENANCE-001` | **confirmed / P1（数据身份）/ B09 扩展** | `pred.py:65-73,300-315,343-346,437-449` | `--tli-sparse-prefill` 不进入输出路径或记录；同参数 P-off/P-on 写同一文件并以 `w` 覆盖。非 Qwen3 模型还会静默忽略该开关。 |
| `TL-PREFILL-INTEGRATION-GAP-001` | **confirmed / P2（发布门禁缺口）/ 新增** | `test_sparse_prefill_impl.py:33-37,96-110,155-212` | 新测试直接调用 helper，并用相同末行共享选择构造参考；没有覆盖 patch 注册、Qwen forward、环境开关、cache、CLI 与 fallback。 |

## 复现与机制

### 1. 块内未来 query 泄漏

`sparse_prefill_attn` 对块 `[c0,c1)` 只调用一次 indexer：

```python
last_q = q[:, c1 - 1 : c1]
mask, _bs = indexer.prepare_mask(last_q, q_ids, k[:, :c1], cu, scale)
sel = mask[0, 0][..., :c1]
o[:, c0:c1] = _chunk_rows_attn(q[:, c0:c1], k[:, :c1], v[:, :c1], sel, ...)
```

随后 `_chunk_rows_attn` 只做 `sel & causal`。这个 causal mask 防止行 `r` 读取未来 **key/value**，但不能阻止未来行 `c1-1` 的 **query** 改变 `sel`。因此构造两条输入 A/B，使其在位置 `<=r` 完全相同、只改变同一 chunk 内 `r` 之后的 token；若末行 query 使 indexer 选中不同的历史 key，则 A/B 在行 `r` 的输出不同。这违反 causal LM 的前缀不变性，并会在多层模型中污染后续层的 K/V。

现有 T4 只断言 `prepare_mask` 接收的 K 长度不越过 chunk 末端；它没有检查“每个输出行不能依赖该行之后的 query”。T3/T4 又用同一末行 `sel` 构造参考，因此会把被测实现的错误语义复制进 oracle。

**最小验收**：固定 K/V 和前缀 query，生成两个只在行 `r+1..c1-1` 不同的输入；所有 `<=r` 的 attention 输出以及经过完整层后的 K/V 必须逐行一致。至少覆盖 chunk 首行、中行、尾行与跨 chunk 边界。

### 2. 逐行 SWA 保护集丢失

当前 indexer 对 chunk 末行强制：

```text
sink = [0,127]
endpoint_SWA = [c1-128,c1-1]
```

对 chunk 早期行只执行 `sel & key_position<=row`。以 `L=3000`、chunk `[2048,3000)`、row `2048` 为例：

```text
endpoint_SWA = [2872,2999]
row_2048_SWA = [1921,2048]
intersection  = 0 tokens
```

若用最小选择集 `sel=sink union endpoint_SWA`，causal 交集后 row 2048 只剩 128 个 sink token，它自己的 128-token SWA 全部缺失。真实 top-k 偶然命中其中若干 token 不能满足“保护集必选且不占创新预算”的契约。一个 952 行的稀疏 chunk 中，除末行外的 951 行均没有代码层面的行相对 SWA 保证。

**最小验收**：对每个稀疏 chunk 的首/中/末行，显式检查有效 mask 包含 sink 和 `[max(0,r-swa+1),r]`；保护 token 不占 mid top-k 预算。此项不能用 chunk 末行断言替代。

### 3. near/SWA 分界修复仍未合入

当前两处边界分别为：

```python
far_hi_blk = max(sink_blocks + 1, (S - near_len_dyn) // bs)
near_blks = max(sink_blocks, (kt * bs - near_len_dyn) // bs)
```

但 `near_len_dyn = alpha * mid_len` 只表示 mid 内的 near 长度，起点应相对 `S-swa_tok` 计算。用 `S=4352, sink=128, swa=128, bs=64, mid=4096` 的纯整数反例：

| alpha | 契约 near | 当前实际 near | 契约 far | 当前实际 far |
|---:|---:|---:|---:|---:|
| 0.5 | 2048 | 1920 | 2048 | 2176 |
| 1.0 | 4096 | 3968 | 0 | 128 |

`bb8c6712c` 确实把 L2 near 选择、保护区置位和 return 移出了 `far_tok_hi > far_tok_lo` 分支；该控制流修复本身未发现新的静态错误。但上游边界仍使常规 `alpha=1,beta>0` 留下一个 128-token far 区，所以该提交声称覆盖的 far-empty 路径不能按文档参数自然到达。

**最小验收**：以显式期望值测试 `alpha={0,0.5,1}`，同时断言 far/near/SWA 三段互斥、无洞、总长度守恒；`alpha=1,beta>0` 必须进入真实 far-empty 分支。测试必须针对当前检入实现，不得加载 `/tmp` 修复副本。

### 4. 测试依赖未提交 `/tmp` 状态

`test_near_swa_boundary.py` 将 `FIX_ROOT` 固定为 `/tmp/near_fix_v2`，并把仓库当前树称作“旧口径”、`/tmp` 称作“修复副本”。本次干净检出中该路径不存在。即使安装 PyTorch，测试也不能从仓库内容独立重现绿侧结果；若某台机器恰好残留不同版本，结果又不受 Git SHA 约束。

**最小验收**：删除绝对临时目录依赖，直接对当前实现断言不变量。若确需 red/green 对照，应把最小旧实现 fixture 和预期输出随测试版本化；增加一次 fresh-clone CI 执行。

### 5. budget 不能代表 P/D 阶段真实预算

prefill helper 每个稀疏 chunk 只调用一次 `prepare_mask`，metrics 因而记录 chunk 末行的选择大小；首个 dense chunk没有相应记录。实际每行的有效选择还会经 `sel & causal` 和对角兜底改变。之后同一个 metrics 容器继续记录 decode 的逐 token mask，`pred.py` 最终只写一个 `budget`。所以开启 sparse prefill 后，该字段既不是 prefill 每行平均 attend tokens，也不是纯 decode budget，不能用于 P/D 矩阵的公平比较。

**建议**：按 `prefill/decode` 分阶段记录；prefill 统计每行最终有效 mask，而非 chunk endpoint mask；另外分列 protected、mid、索引器与 dense fallback。正式报告前用小序列手工求和 oracle 核对。

### 6. 结果身份与真实接入缺口

`--tli-sparse-prefill` 只设置环境变量，不进入 `get_method_name_with_info(args)` 所形成的输出名，也不进入每条 record。相同 dataset/method/postfix/t 下 flag 关闭和开启得到同一路径，第二次 `open(...,"w")` 会覆盖第一次。该问题属于旧 B09 根因的新实例，但直接影响计划中的 E114b P-only/P+D 配对。

另外，当前只有 `qwen3_attn_patch.py` 读取 `TLI_SPARSE_PREFILL`；Llama 等已暴露的 LongBench 模型仍接受 CLI 开关，但 prefill 保持 dense 且没有告警。现有 helper 测试不会发现这一静默 no-op，也不会覆盖 register/forward/cache/transpose/fallback。

**建议**：

1. 输出目录与 manifest 绑定 stage、chunk、完整算法配置 hash、源码 SHA、输入 hash；已有路径若 fingerprint 不同则拒绝覆盖。
2. 不支持的模型在解析或 patch 注册阶段明确拒绝该 flag。
3. 记录实际 sparse call 次数和 fallback reason；请求 sparse 但调用数为 0 时任务失败。
4. 增加 tiny Qwen 或最小 patched-module 集成测试，从 CLI/env 进入真实 forward，覆盖 cache 与 dense fallback；helper 单测继续保留但不替代集成测试。

## 对既有数据与论文结论的影响

- `07a1e76cd` 默认关闭 sparse prefill，且本次差异中没有新增正式结果文件；目前没有证据表明此前以 dense prefill 运行的结果被新增的两个 prefill 问题污染。
- near/SWA 边界问题会影响 `alpha>0,beta>0` 的 E64/E109 等扫描对区域语义的解释：实际是“较窄 near、较宽 far”。现有数值不应直接删除，但不能继续按文档中的 alpha 定义解释或比较；修复后需重跑受影响配置和 winner。
- 尚未见可证明 provenance 的 E114b P-only/P+D 正式结果。若已经在未落盘 manifest 的外部环境运行，因路径覆盖和静默 no-op 风险，应视为身份不充分，修复后重跑。
- 本报告没有 GPU 数值证据，不能据此量化 perplexity、准确率或吞吐变化；P1 判定来自明确的数据依赖/保护集契约违反，而非性能猜测。

## 本次实际执行

| 检查 | 结果 |
|---|---|
| `python -m compileall -q`（本次 6 个新增/改动 Python 文件） | PASS |
| `git diff --check`（`07a1e76cd`、`bb8c6712c`） | PASS |
| near/SWA 纯整数边界反例 | PASS，复现 `alpha=1` 仍有 128-token far |
| sparse prefill 行相对 SWA 集合反例 | PASS，最小选择集中 row 2048 命中自身 SWA `0/128` |
| `python two-level-attention/test_near_swa_boundary.py` | BLOCKED：环境无 `torch`；此外源码确认依赖不存在的 `/tmp/near_fix_v2` |
| `python two-level-attention/test_sparse_prefill_impl.py` | BLOCKED：环境无 `torch` |
| GPU kernel / 完整模型 / LongBench e2e | 未执行：无 PyTorch/GPU/模型数据，不能冒充实测 |

## 修复顺序与重测范围

1. **先修因果性和逐行 SWA**：选择机制必须只使用每行可见信息；逐行 OR 自身 sink/SWA，不能仅复用 chunk 末端保护集。
2. **修 near/SWA 边界并改测试**：以 `S-swa_tok-near_len_dyn` 口径统一更新/消费两侧，删除 `/tmp` 依赖，补 `alpha=1` far-empty 回归。
3. **修 provenance/metrics/unsupported-model 行为**：在任何正式 P/D 矩阵前完成，否则结果无法可靠配对。
4. **CPU 数值层**：dense 短序列、稀疏多 chunk、前缀不变性、首/中/末行保护集、GQA/per-q-head、不同长度与非整块尾部。
5. **单卡真实模型层**：Qwen prefill-only、decode-only、both、dense baseline，同输入/seed/config；检查实际路由计数、KV cache 和逐 token logits。
6. **GPU/e2e 层**：完成精度验收后再做 kernel 与 LongBench；同步、warmup、失败样本、实际样本数和输出 fingerprint 必须落盘。不得用 helper PASS 替代该层。

本次仅提交审查报告，没有修改实现、测试、实验数据或其他目录。
