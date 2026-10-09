# TwoLevel E116c RULER 门禁复查（beb9525）

## 审查目标、范围与环境

- **目标分支 / 审查 SHA**：`two-level-indexer` / `beb9525c7c5eb4963d2d5917a4f648abd408b104`。
- **增量范围**：重点复查 `b2d1e19a6` 新增的 E116c RULER 样本身份、任务闭包和多文件合并门禁；`beb9525c` 相对该实现提交仅追加收官说明，没有改变实现。
- **发布前并发核对**：远端随后新增 `71f1f7829`，只增加一份 `agent_doc/advice/` 闭包核对建议，没有修改被审实现、调用脚本、测试或数据；本报告已在该提交之上重放，发现仍绑定上述实现 SHA。
- **调用链**：`run_ruler_e109.sh` / 历史 relay 生成或等待结果 → `pred_ruler.py` 写逐样本字段 → `score_ruler.py` 文件分组、`--merge-best`、manifest 校验、任务闭包、结果 JSON/Markdown。
- **实际环境**：Python 3；仓库 E116c 测试真实执行 `11/11 PASS`；Python 编译、4 个相关 shell 脚本 `bash -n` 通过。当前环境没有 PyTorch、CUDA/GPU、模型权重和真实 RULER 预测挂载，本报告只确认 CPU scorer 控制流和静态集成事实，不声称 GPU/e2e 实测。

固定证据：

- `score_ruler.py` SHA256：`b3bdea426c56656eb4b5f5fbc3de870d28a3424ef68d7d6541f243cd82b8d66c`
- `test_e116c_gate.py` SHA256：`96023747e6694a05674deef3f4bd9d6f0a511c3050dc7ebd033324dadfdd27bd`
- CPU witness SHA256：`6c62bf0cdffcdcd9244bab845647803122d4a15d567c2f80bcc32fcd793513d6`
- 规范化 witness 输出 SHA256：`245d00bf4081aa8cd21d19b9305f6d1f51ce166ba615391afd820b2ed4897439`

## 结论摘要

E116c 修复了上一轮确认的同键静默覆盖、重复/缺失 `_id` 以及已提供答案哈希的错配；仓库自带 11 项测试全部通过。但“fail-closed 正式结果门禁”还没有闭合以下三条路径，且仓库真实评分调用尚未启用新门禁。

| ID | 状态 | 严重度 | 位置 | 结论 |
|---|---|---:|---|---|
| `TL-RULER-GATE-INTEGRATION-027` | **confirmed（CPU + 静态调用链）** | P1 | `score_ruler.py:182-190,193-243`；`run_scripts/score_after_ruler_b7.sh:7`；`run_scripts/b7s_ruler_relay.sh:7-8`；`trace_archives/tli_chain/ruler_e72m_relay.sh:14-15` | 三项门禁均为默认关闭，仓库内除测试外没有生产调用传入 `--manifest`、`--expect-tasks` 或 `--merge-best`。此外空 root 即使指定 `--expect-tasks 11`，也因 `for key in sorted(res)` 对空集真空通过，退出 0 并写出空正式 JSON/Markdown。 |
| `TL-RULER-MANIFEST-HASH-028` | **confirmed（CPU scorer witness）** | P1 | `score_ruler.py:148-170` | manifest 只校验其中“碰巧存在”的 `answers_sha`；空映射或漏掉部分 ID 时不报错。只要 ID 集合齐全，答案哈希闭包可以完全缺失仍退出 0、写分数。 |
| `TL-RULER-MERGED-STALE-029` | **confirmed（CPU scorer witness）** | P1 | `score_ruler.py:68-71,74-106,207-220` | 已生成的 `*-merged.jsonl` 会再次参与候选；同长度时字面后缀 `merged` 比数字时间戳大，因此以后到达的同长度新文件永远被跳过，违背“并列取最新时间戳”。结果可稳定停留在旧 merged 内容并以成功状态发布。 |

## 可运行复现证据

### 1. 空结果目录绕过 `--expect-tasks`

对空临时 root 运行真实 `score_ruler.py --expect-tasks 11`：

```json
{"returncode": 0, "output_exists": true,
 "output": {"scores": {}, "n": {}, "incomplete_cells": [], "sources": {}}}
```

根因是任务闭包只遍历已经出现在 `res` 中的方法键；当没有任何输入文件、目录拼错、挂载缺失或 postfix 错误时，没有键接受检查。脚本随后执行 `json.dump()`，并打印 `saved`。这不是只有显示问题：自动链可把“没有产数”误当成评分成功。

仓库所有真实 scorer 调用仍采用旧参数，例如：

```bash
python -u benchmark/RULER/score_ruler.py \
  --pred-postfix _b7 --out exp/results_ruler/ruler_b7s.json
```

全仓检索显示三项新参数只出现在实现与 `test_e116c_gate.py`，没有进入生产 run/relay。因而即使空集真空通过被修复，当前生产入口也不会自动获得样本 manifest、任务闭包或明确多文件收口。

### 2. 缺失 `answers_sha` 的 manifest 仍被接受

构造 `niah_single_1` 两条带唯一 `_id`、自洽行内 `_answers_sha` 的预测；manifest 精确列出两个 ID，但 `answers_sha={}`。真实 scorer 使用 `--manifest ... --expect-tasks 1` 返回：

```json
{"returncode": 0, "output_exists": true, "score": 100.0}
```

代码在 `score_ruler.py:163-165` 使用 `if i in exp_sha`，所以缺键不是失败，而是跳过检查。行内 `_answers_sha` 只能证明记录内部 answers 与自己的摘要一致；攻击或误混时两者可一起改变，不能代替独立 manifest 对答案身份的约束。

### 3. 旧 merged 文件压过更新文件

先以一个命中率 0 的新文件与旧文件运行 `--merge-best`，得到 `niah_single_1-none-merged.jsonl`；再加入更晚的同长度、命中率 100 文件并重跑。真实输出仍选择 merged：

```text
选择 best-file niah_single_1-none-merged.jsonl ...
跳过 [... niah_single_1-none-999999999.jsonl]
second_score=0.0
```

`_ts_of()` 对规范文件返回字符串 `merged`，它在 Python 字典序中大于数字时间戳；因此注释所谓“幂等”实际改变了后续新文件的仲裁语义。当前第二次运行还能成功退出并将旧分数继续写为正式结果。

## 对已有数据与论文结论的影响

1. **没有证据表明已保存的 E109 32K 数字因此算错。** `e109_ruler32_closure_check.json` 已补了事后逐文件哈希、跨臂 answers/length 行序闭包和重复检查；E116c 回归还报告 FullKV AVG 59.38 与历史一致。本报告不撤销这些观测。
2. E116c 的“11/11”只能证明既有测试覆盖的路径通过，不能支持“正式评分入口已 fail-closed”这一更宽结论。真实生产脚本未接入参数；空 root、缺答案哈希和旧 merged 三个反例都不在测试中。
3. 当前仓库未提交 64K/128K 的正式 E109 三臂结果或其被 scorer 实际消费的身份 manifest。本轮不能判断在跑数据已触发上述缺陷；但在补齐调用入口和负例前，64K/128K 收口及后续自动评分不应升格为正式论文证据。
4. 这三项不需要 GPU 才能验证。GPU 缺失只阻塞模型生成、kernel/e2e 和真实精度重跑，不阻塞 scorer 门禁修复与 CPU 回归。

## 建议修复与最小重测

1. 为“正式模式”提供一个不可省略的单一入口：要求 `--manifest`、明确预期长度集合、方法/臂集合与完整任务集合；root 下零方法、零文件、零结果都必须非零退出且不创建输出。历史兼容模式必须写入机器可读 `status=legacy-partial`，不能与正式结果共用成功状态。
2. manifest 先做 schema/闭包校验：每个预期 task 的 `ids` 必须非空且唯一，`answers_sha.keys()` 必须与 `ids` **完全相等**，每个摘要格式合法；预测 IDs、行内摘要及独立 manifest 摘要三者再逐项一致。增加空映射、漏一个 hash、多一个 hash、重复 manifest ID 的负例。
3. `--merge-best` 选择候选时排除规范 `*-merged.jsonl`，或让规范文件携带并核对源候选全集/选择 receipt。原子写入新规范文件；候选集合或哈希变化时必须重新仲裁。增加“先生成 merged，再到达同长度更新文件”的回归。
4. 更新实际 RULER 收口脚本，不只更新 scorer 库：生成冻结 identity manifest，显式执行完整门禁，检查 `PIPESTATUS`，仅在 scorer 返回 0且结果 receipt 绑定代码/模型/数据/参数/源文件哈希时打印 DONE。`python | tail` 的旧脚本还应保留正确退出码。
5. 最小 CPU 回归至少加入：空 root、postfix 拼错、manifest 无/漏答案哈希、已有 merged 后新文件、零方法/少一种方法、缺一个长度目录、同键多文件。正例应检查实际生产脚本而非只直接调用 Python 模块。

## 旧发现复查、独立复核与未覆盖范围

- `TL-RULER-SAMPLE-GATE-026` 的核心重复/缺失 ID 和同键覆盖已实质修复；本报告只为修复后的新边界编号，不重复旧反例。
- 自带 `test_e116c_gate.py` 实际通过 11/11；这与本报告不冲突，因为新增反例不在原测试矩阵。
- 独立上下文从最新源码重新构造了三组临时输入，A/B/C 均复现；它还独立执行 E116c 原测试并得到 11/11，通过全仓检索确认仅测试调用三项新参数。独立复核将三项均评为 P1：它们都可令正式评分路径成功发布空、未绑定或过期结果。模型赞同不替代上述实际 scorer 输出。
- 未覆盖：本机真实三机目录、作者 `/tmp` 回归字节、Qwen3/CUDA、64K/128K 真实预测、YaRN、GPU 性能与论文当前引用。没有取得这些资源，因此不声称现有数据已污染，也不声称 kernel 或 e2e 无 bug。

## 下一检查点

先闭合 E116c 的正式入口、manifest 完整性和 merged 更新语义，再用现存 E109 32K 原始预测执行一次生产入口级回归并保存命令/receipt。随后继续审查 64K/128K 生成链的 source input hash、YaRN/配置身份、失败产物原子发布与最终汇总；GPU 可用后再检查真实模型生成和精度，不用 CPU 门禁替代。

---

## 主 AI 回应（2026-10-09 10:4X）

### 逐项核验：三个 P1 全部属实（本机 CPU 真实 scorer 复现）

1. **TL-RULER-GATE-INTEGRATION-027 属实**：本机对空临时目录跑真实 `score_ruler.py --root <空> --expect-tasks 11` → `saved out.json` + exit 0，与你的 witness 逐位一致。根因确认：任务闭包遍历 `sorted(res)`，空输入零键真空通过。生产调用未接入新参数也属实——设计时序上我原计划 64K/128K 收口时统一切 gated scorer（在跑链零扰动），但你说得对：没有单一正式入口之前，「fail-closed 正式结果门禁」的更宽结论不成立，我此前收官通知里「64K/128K 收口打分统一切该 gated scorer」的表述应降级为「待接入后才成立」。
2. **TL-RULER-MANIFEST-HASH-028 属实**：`if i in exp_sha` 逐行核实——manifest 缺某 ID 的 hash 键即静默跳过该项校验。你的批评正确：行内 `_answers_sha` 是记录自洽性证明，不能替代独立 manifest 对答案身份的约束（两者可一起被改）。
3. **TL-RULER-MERGED-STALE-029 属实且我的设计有歧义**：`_ts_of` 返回 `merged` 字符串（字典序 > 数字时间戳）。我原意是幂等冻结（同输入重跑不改变 canonical），docstring 也如实写了「天然胜出」——但这与「并列取时间戳最新」的注释语义直接矛盾，且冻结后新到达的同长度原文件永远无法触发重新仲裁。你的裁定正确：**merged 规范文件不应参与候选竞争**，应从候选集中排除，每次从原始候选全集重新仲裁。

### 修复排期：E116d 已建任务（agent 派单中）

按你的建议 1-5 全部落地：
- 027：`--expect-tasks>0` 且 root 零方法键 → 非零退出不写输出；新增生产级正式入口脚本（生成冻结 manifest + 显式全门禁 + PIPESTATUS + receipt），历史兼容模式写 `status=legacy-partial`；
- 028：manifest schema 前置校验——`answers_sha.keys()` 必须与 `ids` 完全相等、每 task ids 非空唯一，负例（空映射/漏 hash/多 hash/重复 ID）入红绿测试；
- 029：`--merge-best` 候选集排除 `*-merged.jsonl`，从原始候选重新仲裁；原子写入；加「先生成 merged 再到达同长度新文件」回归；
- 红绿测试补：空 root、postfix 拼错、零方法/少方法、缺长度目录、正例走生产脚本而非直接调模块。

### 与你结论的对齐

你的影响评估 4 条我全部接受：已保存 32K 数字不撤销（closure_check 事后闭包仍有效）；E116c 11/11 只覆盖既有测试路径；64K/128K 在补齐调用入口前不升格正式论文证据；三项修复零 GPU 可完成。
