# TwoLevel 增量审查：E123 判决缺少可复验输入链，非显著被写成“持平”

审查时间：2026-10-11（Asia/Shanghai）

审查分支：`origin/two-level-indexer`

审查 HEAD：`2bb79a0d6259926743a8d22ba6f61125b3ba8c1f`

相关代码/结果区间：`db2c539f1da1c0490d7f5feba2aa9031185d1573..2bb79a0d6259926743a8d22ba6f61125b3ba8c1f`（审查判断排除 `agent_doc/advice/`）

## 审查目标、范围与环境

本轮先 fetch 远端并排除 advice-only 提交。区间内有两项相关变化：

- `c8d92cd7` 修复 082：`llm_eval.py` 在命名前冻结 treatment snapshot，并将同一对象传给命名、`register_patch` 和结果 sidecar；
- `2bb79a0d` 新增 E123 四臂五任务判决 JSON，并据此关闭 cavg 路线、维持 mavg 冠军。

审查沿“实验声明 → 已提交原始输入/身份 → scorer/聚合实现 → 汇总 JSON → 任务判决”追踪，同时复查 082 的调用点。环境为 Python 3.12.14、NumPy 2.3.5、Git 2.51.1；本机无 `torch`/CUDA/GPU。未修改实现、测试或实验数据。

## 发现表

| ID | 状态 | 严重度 | 位置 | 结论 |
|---|---|---:|---|---|
| `TL-E123-VERDICT-PROVENANCE-083` | **confirmed**（仓库对象枚举 + CPU 重算 + 独立复核） | **P2** | `agent_doc/task/task_details/S-T018.md:40,45-55`; `two-level-attention/exp/trace/results/e123_trial_verdict.json:1-107` | E123 判决只提交任务级汇总；原始预测、调度/评分/聚合脚本、逐样本配对证据、每格行数/输入身份/配置 manifest/文件哈希和运行环境均未绑定。已提交 `B=10000, seed=42` 也不足以唯一复现 `cavg_g` CI。当前仓库不能独立验证“4 臂×5 任务×200/500 行精确对账”和 treatment 同质性。 |
| `TL-E123-NONSIGNIFICANCE-EQUIVALENCE-084` | **confirmed**（统计结论与区间直接核对 + 独立复核） | **P3** | `agent_doc/task/task_details/S-T018.md:47,49,51`; `two-level-attention/exp/trace/results/e123_trial_verdict.json:3-19,106` | CI 含 0 只能说明“未检出差异/未证明胜出”，不能证明方法“持平”或不存在优于 mavg 的可能。NO-GO 作为预设换冠军门禁仍成立，但等价性措辞超出证据。 |

## 083：E123 判决无法从已提交证据独立重建

### 违反的契约

E123 的任务文档声称“四臂 5 任务 × 200/500 行精确对账后官方 scorer 配对判决”，并用结果关闭 cavg 线。可审计判决至少需要绑定：每格预测/标签或可验证摘要、样本身份与顺序、行数、实际 argv/config manifest、模型和数据版本、scorer backend/源码版本、聚合脚本及其环境。

实际提交仅包含：

1. `e123_trial_verdict.json` 中 15 个两位小数的任务分数/差值、三个均值/CI、`B=10000`、`seed=42`；
2. `S-T018.md` 对 `/tmp/e123_trial_dispatch.sh` 和 `/tmp/e123_trial/pred_{arm}/` 的路径描述。

`git ls-tree -r 2bb79a0d | rg 'e123|E123'` 只命中上述 verdict 与任务文档；提交 diff 也只新增/修改这两项。当前环境和仓库均没有 `/tmp/e123_trial*`。通用 `benchmark/LongBench/eval.py` 虽在仓库，但 verdict 没有绑定其 SHA、调用参数或 scorer backend，也没有保存 scorer 输出 `_meta`；因此“官方 scorer”“精确行数”“逐样本配对”“081 前后行为同质”都不能由远端证据复核。

### CPU 最小复现

输入：`e123_trial_verdict.json`，SHA256 `ddc88bc77ef22ac36dca25f1dd62ed9c6f31f24d541ee6ea9fe38c124a50e57b`。按 JSON 可见任务顺序 `gov_report, hotpotqa, musique, qasper, repobench`，分别对每臂用 NumPy `default_rng(42)`、任务级有放回重采样、`B=10000`、percentile 95% CI；同时穷举 `5^5=3125` 个任务重采样组合。原始输出：

```text
python=3.12.14 numpy=2.3.5
cavg_g mean=-0.242 stored=[-0.89, 0.458] rng_ci=[-0.942, 0.458] exact_ci=[-0.942, 0.458]
cavg_off mean=-2.202 stored=[-4.07, -0.714] rng_ci=[-4.07, -0.714] exact_ci=[-4.07, -0.714]
fullkv mean=0.106 stored=[-0.624, 0.848] rng_ci=[-0.624, 0.848] exact_ci=[-0.624, 0.848]
```

完整输出 SHA256：`4ba7fbd26369b99ebd7f837493c17a776153573c8d6f23e2f2e2a3b43b1b68cc`。

该复现确认 15 个 `delta = arm - mavg` 和三个均值逐项自洽；`cavg_off`、`fullkv` 的 CI 可复现，而 `cavg_g` 下界不同。独立复核进一步发现，改变未记录的任务采样顺序可恰好得到已报三个 CI，因此不能断言 `-0.890` 数值本身算错；可以确认的是：JSON 没记录任务顺序、CI 类型、RNG 实现/版本或聚合脚本，`seed=42` 不构成唯一可复验协议。

### 影响已有数据与结论

- **直接影响 E123 新判决的审计等级。** 当前只能把 15 个任务级数值视为未绑定的汇总，不能把远端 JSON 单独当作“真实预测已逐行验收”的证明。
- 若暂时信任已提交的任务级分数，NO-GO 方向未因 CI 下界差异改变：两种 `cavg_g` 区间均跨 0；`cavg_off` 仍显著为负，`fullkv` 仍跨 0。
- 但原始预测和身份记录缺失时，无法排除缺样本、错文件、重复/乱序、scorer backend 漂移或 081 前后 treatment 混装；因此不能据当前仓库证据宣布这些风险均未发生，也不应据此撤回更早数据。
- 风险集中于 E123/cavg 关闭决定和后续论文引用；不改变本轮 082 源码修复，也没有证据表明 E109/E119/E121 数值被改写。

### 修复与重测建议

1. 若原 `/tmp/e123_trial*` 仍存在，先只读封存每格预测、sidecar/argv/配置 manifest、行数、样本 ID/answers 摘要与 SHA256；若已丢失，必须在同一冻结输入和明确环境下重跑，不从任务级均值反造原始证据。
2. 将 scorer + bootstrap 聚合做成入库脚本，输入路径不得“取最新匹配文件”；显式列出 20 格文件和哈希，校验预期行数、样本身份/顺序、treatment、模型/数据/scorer 版本。
3. 结果 JSON 至少保存任务顺序、逐样本还是任务级 bootstrap、percentile/basic/BCa、RNG 名称与 NumPy 版本、脚本 SHA、完整精度输入分数和源文件哈希；同一脚本提供 `--verify-existing` 从原始证据重建 verdict。
4. 在证据补齐前，将 E123 标为“provisional / summary-only”；可以保持 mavg 作为保守生产选择，但不要把 cavg 线写成已由可复验实验最终关闭。

## 084：非显著不等于等价或“不可能优于”

`fullkv - mavg` 的区间为 `[-0.624,+0.848]`，同时容许 FullKV 较差和较好；它支持“本试验未检出差异”，不支持严格“持平”。`cavg_g - mavg` 的区间也跨 0；它支持“未达到换冠军的预设证据门槛/未证明优于”，不能推出方法事实上不能优于。

建议只改结论强度：

- `cavg_g`：`未证明优于 mavg，按预设门禁 NO-GO`；
- `fullkv`：`未检出与 mavg 的差异`；
- 若论文要声称等价/持平，须事前定义有实际意义的等价界值，并执行等价性检验（例如 TOST），而不是把“未拒绝零差异”当成等价证明。

该问题不推翻“没有足够证据更换冠军”的保守决策，但影响论文和任务文档的统计表述。

## 082 修复复查、实际测试与未覆盖项

- `llm_eval.py:57/58/93/132` 的 AST 实际调用分别为 `resolve_treatment_snapshot(args)`、`get_method_name_with_info(args, snapshot)`、`register_patch(model,args,snapshot)`、`save_results(...,snapshot)`；**082 接线 fixed/rechecked**。
- `llm_eval.py`、`patch.py`、`info.py` 执行 `compileall` 通过。文件 SHA256：`llm_eval.py=3811a84cd0e379ae8157cbef99baad34400493f1b2fe462fcf90425b61182213`，`patch.py=d46b83415f346be3f5dcc8cd534050b029339d173a562beb106b3681b706db69`。
- `test_e121_kimi3_fixes.py` 与 `test_e121_fix_081.py` 在普通 Python、`python -O` 下均于 import 阶段因 `ModuleNotFoundError: torch` 阻塞；不是测试失败，也不能声称通过。
- 未执行 GPU、真实模型、kernel、RULER/LongBench e2e；未接触外部机器或现有 GPU 作业。E123 原始数据不可用，因此没有重新评分或伪造替代数据。

## 下一检查点

下次仅在相关源码/结果证据变化时复查：优先检查 E123 是否补入可重建 evidence manifest 和聚合脚本、083 是否按原始数据复验，以及 084 的结论措辞是否降格或补等价检验。同一证据不重复报告。
