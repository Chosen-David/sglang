# TwoLevel RULER 64K/128K 数据与调度链审查（57653be）

## 审查目标与范围

- **目标分支/审查 SHA**：`two-level-indexer` / `57653be8718feeb4fc63bf50d751825a9743f597`。
- **相对上次审查范围**：`c5fe1d5b1..57653be87`，排除 `agent_doc/advice/` 后新增/修改了 `benchmark/RULER/gen_ruler_long.py`、`pred_ruler.py`、`run_ruler_e109.sh` 和 `exp/trace/results/e109_full_lbv2.json`。本轮没有把历史 advice-only 提交当作源码变化。
- **实际追踪链**：32K 源数据结构与本地 RULER 官方生成器 → 64K/128K body 扩展与语义检查 → YaRN/`logits_to_keep=1` 入口 → 三臂 shell 调度与 SKIP → `score_ruler.py` 聚合。
- **环境边界**：Python 3.12.14，Linux 6.18.44 x86_64；当前环境没有 PyTorch、Transformers 或可见 GPU。实际执行了纯 Python 控制流/语义 witness、AST 解析和 `bash -n`；没有运行 Qwen3、YaRN、CUDA、真实 32K 数据生成或 64K/128K 精度任务。

证据文件 SHA256：

- `gen_ruler_long.py`：`0772e36e8d01c8938446f0227920ede07d1d19ba8c8008074dcf6f6068d06c12`
- `pred_ruler.py`：`fa748bd0780fe7f410d6891f2f4aaa6a277c635df93e5db5cf591a3674f3fefa`
- `run_ruler_e109.sh`：`461716ade54fcca5e014fbd4d3bea903e9455bcda4cfa07270287fbac1876bb4`
- `score_ruler.py`：`e8bed994bcf23eebfcd3f5dd2765c4298bfc269bbd371ca6d54864f55d2c4c88`
- `e109_full_lbv2.json`：`d0fca3d71bed92b5341a52ef4f75aba8bf94c6cc8dc3c4e3a5f249c3971ef243`

## 结论摘要

| ID | 状态 | 严重度 | 位置 | 结论 |
|---|---|---:|---|---|
| `TL-RULER-ALIGN-020` | **confirmed（CPU 控制流 witness）** | P1（生成数据语义） | `gen_ruler_long.py:88-106` | `align_cut()` 的两个保护均失效：无换行且截断小于约 2000 字符时把 `rfind=-1` 当成近邻换行并返回 0；空格截断时又只在已截短的 `seg` 内匹配完整 needle，跨截断点的 needle 不可能被找到。代码能够在 magic-number needle 中间截断，与“绝不落在 needle 内部”的契约相反。 |
| `TL-RULER-FWE-LABEL-021` | **confirmed（算法反例）；现有数据影响 inconclusive** | P1（标签正确性） | `gen_ruler_long.py:109-123,199-219`；`lm_eval_tasks/ruler/fwe_utils.py:27-30,58-67` | FWE 的标签由全文前三高频词定义。`body × m + body 前缀` 不保持 top-3；当前检查只验证旧答案字符串仍存在/计数未下降，因此错标签可被判为 OK。短余段被 `TL-RULER-ALIGN-020` 丢弃时会暂时掩盖该问题，故两项必须一起修。 |
| `TL-RULER-DATA-GATE-022` | **confirmed（CPU 文件生命周期 witness）** | P1（验收/数据发布） | `gen_ruler_long.py:143-226` | 100 条默认数据只以固定 seed 抽 5 条验收，而且正式目标 JSONL 在验收前已用 `w` 写入。未抽中坏样本会假通过；即使抽中并返回失败，完整正式路径仍保留，可被后续调度消费。 |
| `TL-RULER-SKIP-IDENTITY-023` | **confirmed（shell 逻辑 witness）；旧根因在新脚本复发** | P1（实验身份） | `run_ruler_e109.sh:57-69`；`pred_ruler.py:146-152` | SKIP 取 `$T-*.jsonl` 的首个文件，只比较行数，不绑定当前 code/model/data/YaRN/方法参数/样本集合。旧配置已有 100 行、当前配置没有结果时仍会跳过。该根因与历史 B07/结果身份不足同类，但这是新增 RULER 长上下文入口上的可达复发。 |

## 复现证据

### 1. `TL-RULER-ALIGN-020`：换行哨兵与 needle 边界检查均错误

`body.rfind("\n", 0, cut)` 在没有换行时返回 `-1`。当 `cut < 1999` 时，`-1 > cut - 2000` 为真，代码直接返回 `j+1=0`。本地 `fwe_utils.py` 的 coded-word context 是单行空格分隔，因此这是与实际任务结构一致的条件，不是无关字符串。

CPU witness 请求在 6000 字符单行 body 后追加 1000 字符，实际追加 0：

```text
FWE_SHORT_PARTIAL {"actual_partial_chars": 0, "body_chars": 6000,
                   "estimated_tokens": 6000, "requested_partial_chars": 1000}
```

needle 保护的逻辑也不可成立：代码先取 `seg=body[:j]`，再在 `seg` 中查完整 regex match；任何跨过 `j` 的完整 needle 已被截断，`finditer(seg)` 不可能返回满足 `m.start() < j < m.end()` 的 match。使用官方格式 `One of the special magic numbers for foo is: 12345.` 的 witness 实际得到：

```text
requested_cut=2031 aligned_cut=2031
needle_span=[2006,2057] cut_inside_needle=true
prefix_tail="... One of the special magic "
complete_needles_in_prefix=0
```

规范化 witness SHA256：`84debad685ab3a775f5e83ea63d63fefe0b0b2f55ef99478592f99177e416e5b`。

### 2. `TL-RULER-FWE-LABEL-021`：答案仍出现不等于频率排序不变

构造合法 coded-word body，原始计数为 `a=600,b=550,c=500,d=450`，旧标签是 `a,b,c`。追加同一 body 的 2100 字符前缀后，计数成为 `a=1200,d=900,b=550,c=500`，正确 top-3 已变为 `a,d,b`；然而 `a,b,c` 都仍出现且计数没有下降，所以 `gen_ruler_long.py:209-213` 的现行检查返回真。

```text
orig_top3=[a,b,c]
new_top3=[a,d,b]
legacy_answer_presence_check=true
```

规范化 witness SHA256：`707b04eae99ac22fb6dab9301b443c62862ac239f1b23ea77d7d2ee26e31bdac`。这确认的是生成算法不能普遍保持 FWE 标签；由于外部 32K 源行和 Qwen3 tokenizer 当前不可访问，不能据此断言已生成的正式 64K/128K 每一行都发生翻转。

### 3. `TL-RULER-DATA-GATE-022`：失败产物仍占用正式路径

用确定性 fake tokenizer 令真实复验长度为 100、目标为 28（`tol=0.10`），`gen_task()` 明确打印 `sample_check=FAIL` 并返回 `False`，但检查目标路径得到：

```text
{"failed_output_exists": true, "failed_output_lines": 1, "gen_task_ok": false}
```

fixture SHA256：`b4b44010cc79012c0a90efd14de9437ac6408acee29417c2a76e03863b37898e`。默认 100 条时固定抽样索引仅为 `[49,97,53,5,33]`，其余 95 条的 token 长度和任务语义都不在验收闭包内。

### 4. `TL-RULER-SKIP-IDENTITY-023`：旧文件阻断当前实验

临时目录只放置 `fwe-tli_old_config-01010000.jsonl`（100 行），当前期望的 `fwe-tli_new_config-01020000.jsonl` 不存在。逐字执行脚本的 glob/head/wc 判定仍输出：

```text
SKIP:.../fwe-tli_old_config-01010000.jsonl:100
CURRENT_CONFIG_RESULT_EXISTS False
```

fixture SHA256：`885a9908c44cab81a9c50fbcb6ee2e44b931f08e9094159d55d4f33b2797768a`。此外，`get_method_name_with_info()` 并未把 TLI 的 far/near 方法和 alpha/beta/gamma 写进文件名；当前脚本靠人工 `arm` 目录区分，不能替代可校验的实验 manifest。

## 对已有数据与结论的影响

1. 仓库没有提交 64K/128K RULER 生成数据、逐样本校验清单、预测 JSONL 或运行 manifest；当前环境也不能访问脚本中的外部数据路径。因此四项代码缺陷已确认，但**正式 E109 RULER 数据是否已经受影响、影响多少仍为 inconclusive**，不能宣称历史结果已污染或已通过。
2. 若正式 FWE 行的余段小于约 2000 字符，`TL-RULER-ALIGN-020` 会丢掉余段而不是改变 top-3，主要影响目标长度；一旦单独修复该分支，`TL-RULER-FWE-LABEL-021` 可能从被掩盖变为错标签，不能分开发布。
3. NIAH 截断落在 needle 内的条件可达，但没有外部源行，无法统计实际命中数。完整 body 的重复通常仍保留至少一份完整答案，因此“旧答案仍出现”的检查不能排除新增半截 needle 对难度和模型输出的影响。
4. 新提交的 `e109_full_lbv2.json` 属于 LongBench-v2 三臂汇总，不经过本轮 RULER 生成/调度链；本轮发现不构成该 JSON 数值错误的证据。其原始分片/不可变输入与代码 hash 缺口已属于历史结果身份问题，本报告不重复升级结论。
5. 提交说明中的“6 组冒烟全过”和“token 校准 ±1%”没有随提交提供原始日志、数据 hash 或逐行清单，本轮不能独立复验；程序可解析和单个 smoke 均不能替代 11 任务 × 2 长度的语义验收。

## 建议修复与最小重测

1. `align_cut()` 只有在 `j >= 0 and j > cut-2000` 时才接受换行；needle 检查必须在完整 `body` 上取 match 区间，再判断候选截断点是否落入区间。新增三条纯函数门禁：无换行且 `cut<2000`、截断落在 magic-number needle 内、截断恰在 needle 两端。
2. FWE 不要假定任意前缀保持频率排序。最低风险方案是只用完整周期扩展 coded words，并以不参与计数的中性填充达到长度；若保留部分周期，必须重新计数全部 coded words、按与官方相同的 tie 规则计算 top-3，并与 `outputs` 严格相等。CWE 同样应验证完整 top-10 集合而非仅检查旧答案仍出现。
3. 对全部行执行真实 tokenizer 长度和任务专用语义 oracle；先写同目录临时文件，全部通过后用原子 rename 发布。完成 manifest 至少绑定源文件 SHA256、生成器 SHA、tokenizer/model revision、目标长度、任务、样本 index 集合和输出 SHA256；失败时不得留下正式文件或完成 manifest。
4. `run_ruler_e109.sh` 只允许匹配当前 manifest 的精确实验身份并校验 JSONL 行/index 闭包；不要以 `ls | head -1` 的任意旧文件作为完成条件。并发运行同一 arm/task 时还需锁或原子 claim，避免同一分钟同路径竞争写。
5. 修复后先对 11 任务 × 100 行做 CPU/tokenizer 全量生成门禁，再在受控 GPU 上分别运行 FullKV、mavg、aavg 的 32K/64K/128K。评分前核对三臂输入 SHA 完全一致、每 cell 100 个唯一 index、无旧文件混入；随后才比较精度。YaRN 和 `logits_to_keep=1` 仍需真实模型前向与峰值显存复验，本报告没有替代该门禁。

## 旧发现复查、独立复核与未覆盖项

- 历史 B07/结果身份不足在新脚本 `run_ruler_e109.sh` 上出现了新的具体入口，故记录 `TL-RULER-SKIP-IDENTITY-023` 作为复发位置；没有把同一根因包装成独立架构结论。
- `score_ruler.py` 的缺任务/不完整 AVG 问题已有历史 B08 记录，本轮没有重复编号。`pred_ruler.py` 的真实 YaRN 接口、`logits_to_keep=1` 等价性和 GPU 显存收益因缺依赖/GPU未验证，保持未决。
- 独立只读审查复跑了 FWE 短截断、needle 跨界、频率翻转、抽样/失败残留和 stale SKIP witness，结论与主审一致；其独立 needle witness 同样返回 needle 内部位置，并证明在 `seg=body[:j]` 上寻找跨界完整匹配的条件数学上不可满足。同时指出短截断 bug 可能暂时掩盖 FWE 标签 bug，要求二者一起修。模型一致意见没有被当作运行效果证据。
- 未覆盖：真实外部 32K 数据、实际 64K/128K 生成文件、Qwen3 tokenizer 逐行长度、Transformers YaRN 配置兼容性、CUDA OOM/精度、三臂完整 RULER 分数和多进程竞争实测。

## 下一检查点

优先复查 `align_cut` 与 FWE/CWE 语义 oracle 的同一修复提交，并要求全量逐行 manifest/原子发布；随后复查 RULER runner 的实验身份与并发 claim。只有这些数据门禁通过后，才值得占用 GPU 验证 YaRN 64K/128K 三臂精度与显存。
