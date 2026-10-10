# TwoLevel 稀疏 Prefill 修复复验：测试假绿与结果身份仍不完整

## 结论摘要

本次审查绑定 `two-level-indexer` 分支提交
`96bb6622b89591228991cde89ecc8db1650fdcb2`，重点复验源码提交
`5d81d09c829b64743bc5a6b8e87690c0e3038a9f` 对上一份审查中
`TL-PREFILL-PROVENANCE-001`、`TL-TEST-TMP-001` 的修复，以及提交
`96bb6622b89591228991cde89ecc8db1650fdcb2` 对 `TL-PREFILL-SWA-002` 的修复。

修复有两个真实但有限的效果：TLI 的 P-off/P-on 文件名现在不同；干净检出缺少
`/tmp/near_fix_v2` 时测试不再直接崩溃。但本次确认两个新的验收缺陷：

1. 四个 `SKIP` 被按成功项计入 `5/5 PASS`，且进程退出码为 0；未执行的 near
   边界、不变量和预算检查会被 CI/监督器当作整套通过。
2. 唯一在干净主树执行的 N2 名称声称覆盖 far-empty，但它明确断言
   `far_hi=256`；结合 `sink=128`，实际 far 区是 `[128,256)`，宽 128 token，
   因而没有触达要保护的 far-empty 分支。

因此不能把 `5d81d09c8` 的“5/5”作为 near/SWA 或 C-1 far-empty 修复通过证据。
`96bb6622b` 已在代码上逐行 OR 自身 sink/SWA 保护带，并新增针对上一反例的 T7；
静态检查未发现该补丁的新语义错误，但本环境无 PyTorch，不能把提交者报告的
`7/7` 当作本次独立运行证据。上一报告的 causal、near/SWA 边界和阶段 budget
问题仍未由本次源码变化修复；本报告不重复计为新根因。

## 审查范围与证据绑定

| 项目 | 值 |
|---|---|
| 审查 HEAD | `96bb6622b89591228991cde89ecc8db1650fdcb2` |
| 重点修复提交 | `5d81d09c829b64743bc5a6b8e87690c0e3038a9f`、`96bb6622b89591228991cde89ecc8db1650fdcb2` |
| 上一审查基线 | `ae10bf3c4e16c146919d12d7c35251eb4aa88494` |
| 变化范围 | `sparse_attn/info.py`、`ops/eager_prefill.py`、两个专项测试；另有与本复验无关的 E109 结果和 advice |
| `info.py` SHA256 | `d1cb96d8d1e8abe7d342e6059c908a957f1d730b01fefc3d0273737a8a991320` |
| `test_near_swa_boundary.py` SHA256 | `c0571c660c4151f1718da074b8f27a73a620a8a8187df25f3501a2daecab1c39` |
| `eager_prefill.py` SHA256 | `4bb86dcdbe8a61feb42b204d65e4d9940f23edaa1689fd945a4b3179101cea20` |
| `test_sparse_prefill_impl.py` SHA256 | `711d751ce1bfcadae8c322316ae7790940e3593762a0e917b7ef5fa185e58a66` |
| `tli_indexer.py` SHA256 | `74fe5e20cb3eaadef8532842c7c94af44a323a853b961950e4becf5a6afe26cf` |
| `pred.py` SHA256 | `116db7bc9ccf92303e427d9a87036260fd095309d80e7b51beba698a523c5632` |
| 环境 | Python 可用；无 PyTorch、无可用 GPU；未执行 kernel/模型 e2e |

已读取根 `TASK.md`、`agent_doc/task/TASK.md`、相关 README 与本修复直接关联的
TwoLevel 审查 advice，并将本任务 advice-only 提交排除出源码变化判断。

## 发现表

| ID | 状态/严重度 | 位置 | 结论 |
|---|---|---|---|
| `TL-TEST-SKIP-PASS-002` | **confirmed / P1（验收假绿）/ 新增** | `test_near_swa_boundary.py:60-63,113-118,183-187,269-279` | `report_skip` 将 SKIP 写成 `ok=True`；汇总把 4 个 SKIP 计为 PASS，且只按 `n_fail` 决定退出码。干净检出会把未执行检查报告成 `5/5 PASS` 并退出 0。 |
| `TL-TEST-FAR-EMPTY-COVERAGE-003` | **confirmed / P1（分支未覆盖）/ 新增** | `test_near_swa_boundary.py:144-176`；`sparse_attn/indexer/tli_indexer.py:795-806,842-876,1055-1093` | N2 声称是 far-empty 回归，但主树断言 `far_hi=256`，而 far 起点为 128；实际 far 宽 128，不会覆盖 far 池为空时的控制流。 |
| `TL-PREFILL-PROVENANCE-001` | **partial fixed / P1（旧发现复查）** | `sparse_attn/info.py:11-17`；`benchmark/LongBench/pred.py:300-315,340-449`；Qwen/Llama patch | `method=tli` 的请求级 P-off/P-on 路径冲突已修；但每条记录仍无请求/实际执行状态、chunk、源码/输入 hash。Llama、带 attention mask 或不支持的组合仍可能生成 `_P` 文件却走 dense fallback。 |
| `TL-TEST-TMP-001` | **partial fixed / P1（旧发现复查）** | `test_near_swa_boundary.py:42-70` | 缺 `/tmp` 时不再 import 崩溃，但四项核心检查被跳过且假绿，未达到“干净检出可复现完整验收”。 |
| `TL-PREFILL-SWA-002` | **code fixed / runtime recheck inconclusive（旧发现复查）** | `ops/eager_prefill.py:37-100,130-173`；`test_sparse_prefill_impl.py` T7 | 稀疏行现显式 OR 自身 sink/SWA 并与 causal 相交；T7 直击旧反例。静态机制匹配修复目标，但本次无 PyTorch，未独立运行 T7 或真实模型。 |

## 最小复现证据

### 1. SKIP 被计为 PASS

按当前 `report_skip` 与尾部汇总逻辑构造 4 个 SKIP 加 1 个 PASS，CPU 纯 Python
复现输出为：

```text
current_summary= 5/5 PASS skips= 4 exit= 0
```

机制是 `report_skip` 写入 `(name, True, "SKIP")`，而汇总只统计 `ok=False`。
这不是显示文字问题：退出码同样是 0，会直接影响自动验收。

### 2. N2 没有进入 far-empty

N2 使用 `S=128+4096+128`、`alpha=1`，并在当前主树断言
`fh_old == 256`。按同一代码的 `far_lo=sink=128` 计算：

```text
N2_current_tree_far_interval= (128, 256) width= 128 is_empty= False
```

所以 N2 能验证 sink/SWA 最终置位和某个 768-token 总数，但不能证明
`sc_far.shape[-1] == 0`、`i_f` 为空时 near 选择与强制区仍正确。测试名称、注释和
实际路径不一致。

### 3. `_P` 只解决部分结果身份

用 `torch` 空模块加载只含字符串逻辑的 `info.py`，结果为：

```text
tli   ['tli_64_128_1024_c4_AB', 'tli_64_128_1024_c4_AB_P'] distinct=True
quest ['quest_64_16', 'quest_64_16']                       distinct=False
none  ['none', 'none']                                     distinct=False
```

对 TLI 主路径，原来的直接文件名互覆已消除。但 `_P` 表示“用户请求了开关”，并不
表示实际执行过 sparse prefill：Llama prefill 没有该分支；Qwen 在
`attention_mask is not None` 或 `B!=1` 时回退 dense；当前没有 call count 或
fallback reason 进入记录，也没有在任务结束时拒绝“请求 P 但调用数为 0”。

## 对已有数据和结论的影响

- 本次未在仓库中发现 E114b P-only/P+D 的已提交正式输出，暂无证据表明既有
  dense-prefill E109 数据受这两个新测试缺陷污染。
- 不能据 `5d81d09c8` 的提交说明或“5/5”判定 near/SWA 边界或 far-empty 已通过。
  `96bb6622b` 使逐行 SWA 的代码状态前进到“已修、待独立运行复验”，但
  sparse-prefill causality 仍开放；E114b 正式产数仍应暂停。
- `_P` 文件名只能证明请求参数，不足以证明实际路由。任何已在外部生成但没有
  call count、fallback reason、chunk 和实现 SHA 的 `_P` 结果，仍应视为身份不充分。
- 本次没有 GPU/PyTorch 数值证据，未量化精度或性能影响。

## 建议修复与最小重测

1. 将结果状态改为显式 `PASS/FAIL/SKIP`，汇总分别输出
   `executed_pass/executed_total` 与 `skipped`；发布门禁所需项发生 SKIP 时退出非零
   （可用独立退出码 2 表示 incomplete），不得把 SKIP 加进 PASS 分子。
2. 删除发布回归对 `/tmp/near_fix_v2` 的依赖。若红绿历史对照仍要保留，放到单独
   的非门禁脚本；当前树门禁只对检入实现和随仓 fixture 断言。
3. 为 C-1 写真正的 far-empty 测试：执行前先断言 `far_tok_hi <= far_tok_lo`、
   `sc_far.shape[-1] == 0`、`i_f.numel() == 0`，再检查 near 配额、sink/SWA 和总预算。
   near/SWA 边界修复合入后，`alpha=1` 应自然触达该路径。
4. 结果身份同时记录 `requested_sparse_prefill`、`actual_sparse_prefill_calls`、
   `dense_fallback_calls/reasons`、chunk、model、完整算法配置、源码 SHA 与输入 hash；
   请求 P 但调用数为 0 时任务失败。对不支持的模型/方法组合在启动阶段拒绝。
5. 重跑顺序：干净检出 CPU 单测（0 SKIP，含新 T7）→ Qwen tiny 前缀不变性和逐行 SWA →
   单卡真实模型 P-off/P-on/P+D 路由计数与 logits → 再启动 E114b 及性能测量。

## 本次实际执行与未覆盖项

| 检查 | 结果 |
|---|---|
| `git diff --check ae10bf3c4..96bb6622b` | PASS |
| `python -m compileall -q`（本次相关 5 个 Python 文件） | PASS |
| 方法名 P-off/P-on 纯 Python 判别 | TLI 区分；Quest/none 不区分 |
| SKIP 汇总最小复现 | confirmed：4 SKIP + 1 PASS 被报 `5/5 PASS`、exit 0 |
| N2 far 区间最小复现 | confirmed：`[128,256)`，宽 128，非空 |
| 两个专项测试真正执行 | BLOCKED：环境未安装 PyTorch；near/SWA 测试当前还会跳过 4 项 |
| GPU kernel / 完整模型 / LongBench e2e | 未执行：无 PyTorch/GPU/模型数据 |

本次只新增本审查 Markdown，没有修改实现、测试、实验数据、TASK 或其他目录。
