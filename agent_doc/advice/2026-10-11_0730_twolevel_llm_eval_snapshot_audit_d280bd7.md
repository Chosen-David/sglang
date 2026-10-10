# TwoLevel 增量审查：RULER `llm_eval.py` 未接入 treatment 快照

审查时间：2026-10-11（Asia/Shanghai）
审查分支：`origin/two-level-indexer`
审查代码 SHA：`d280bd72ce376cab36a6ef7b3decd25898f1e9e5`
相关代码区间：`ea439fbd17c899d7546a0f64548d93ad16e59dbc..d280bd72ce376cab36a6ef7b3decd25898f1e9e5`（排除 `agent_doc/advice/`）

## 审查目标与范围

本轮先 fetch 远端并排除 advice-only 提交。区间内存在两项相关改动：

- `12c6433b4`：修复 081，将 layer-skip / projection-basis 一次解析成 `TreatmentSnapshot`，并接入 LongBench `pred.py` 与 RULER `pred_ruler.py`；
- `d280bd72`：更新 SGLang 侧 B01/B02 测试的 B10 新边界期望。

沿入口→命名→模型 patch→运行时 indexer→结果落盘追踪新增接口及其调用者，重点复查 081 是否覆盖全部 RULER 入口，并复核 B01/B02 测试改动。未修改实现、测试或实验数据。

未覆盖：本环境无 `torch`/CUDA/GPU，未运行真实模型、kernel、RULER e2e 或 E123 数据；未验证外部未入库作业。可运行的 Python 语法编译通过。

## 发现表

| ID | 状态 | 严重度 | 位置 | 结论 |
|---|---|---:|---|---|
| `TL-E121-LLM-EVAL-SNAPSHOT-082` | **confirmed**（精确源码调用链 + 零依赖 CPU 交错复现 + 独立复核） | **P1（限直接使用该 benchmark 入口的结果）** | `two-level-attention/benchmark/RULER/llm_eval.py:47,54,82,121-128`; `sparse_attn/patches/patch.py:38-52`; `sparse_attn/indexer/tli_indexer.py:137-160,189-209` | 081 只接入 `pred_ruler.py`，遗漏同目录可直接执行的 `llm_eval.py`。该入口先按配置 A 生成文件名，之后不传 snapshot 地逐层 patch；配置文件在窗口内换成 B 时，结果名仍标 A、运行时使用 B，patch 循环中变化还可形成跨层 A/B 混代。 |

## 082 复现证据

### 违反的契约

081 的契约是：文件型 treatment 配置在模型消费前只解析一次，文件名、运行时各层和回执/sidecar 使用同一快照。`llm_eval.py` 当前却是：

1. 第 47 行 `get_method_name_with_info(args)` 在未提供 snapshot 时解析文件内容，形成名称哈希 A；
2. 第 54 行把 A 固化进 `result_filename`；
3. 第 56–80 行加载模型并先执行一次 dense 示例生成，扩大 A 到实际 patch 的时间窗口；
4. 第 82 行 `register_patch(model, args)` 仍不传 snapshot；
5. `register_patch` 因此对每层调用 `TLIIndexer(args)`；后者在 snapshot 为空时每层重新打开 layer-skip 文件或 `torch.load` projection basis；
6. 第 121/127 行用 A 名保存实际运行结果。

### 触发条件

- `--method tli`；并且
- 生效的 `tli_layer_skip_path`（含默认 mask）或 `tli_proj_basis` 在第 47 行之后、第 82 行 patch 完成之前被原地覆盖、原子替换或 symlink retarget；
- 若替换恰发生在 `register_patch` 遍历多个 attention 层期间，各层还能读取不同内容。

配置文件在整个进程内严格不可变时不触发；这也是为何普通成功运行不能排除缺陷。

### 主审最小复现

在干净 detached worktree、SHA `d280bd7` 上：

- AST 核对实际调用：`[(47, get_method_name_with_info, 1 positional), (82, register_patch, 2 positional)]`，两者均未传 snapshot；
- 用精确 `sparse_attn/info.py`（仅为缺失的 torch import 提供不参与本路径的 stub）让同一路径先保存 `{"skip":[1]}`，生成名称后替换为 `{"skip":[2]}`；
- 输出：

```text
NAME_HASH_BEFORE 790e08d4ab
RUNTIME_MANIFEST_HASH_AFTER 313ce613cc
MISMATCH_REACHABLE True
```

原始日志 SHA256：`7ab5b6ea49b469bc0e5a2c4c97b86a20642e8f094658a4cca0917c6e5bde4527`。
输入源码 SHA256：`llm_eval.py` = `4a693cfff463d02bc07a2580bf151a1556291bfa5e5bd1863907751a859f987b`，`info.py` = `d1f2ec838e2dba1ad78cc74f08cff356495d90a9098ef9f4c949100eca173c5c`。

### 独立复核

独立上下文从同一 SHA 重建精确 `info.py` 与 `register_patch` 函数体，首层构造后把 skip 从 `[1,3]` 换成 `[2,4]`，得到：

```text
name_changed=True
patched=2
per_layer_skip=[(1, 3), (2, 4)]
cross_layer_mixed=True
```

独立复核同时确认：这不是 `ea439..d280` 新引入的行为回归，而是 12c6433 快照迁移遗漏的可达调用者；现有 0533 报告只覆盖 LongBench `pred.py` 与 RULER `pred_ruler.py`，没有审查 `llm_eval.py`，因此不是旧证据重复。

## 对已跑数据与论文结论的影响

- **没有证据表明现有 E109/E119/E123 正式数据命中。** 当前正式 RULER 生成链 `pred_ruler.py` 已在第 173/188/205/312 附近解析并贯穿同一 snapshot，LongBench `pred.py` 也已接入；两者是本发现的反证边界。
- 仓库内未发现其他脚本引用 `llm_eval.py`，也未发现其落盘结果被正式聚合器消费。因此不能据此撤回现有正式分数或论文结论。
- 若曾直接运行 `llm_eval.py` 并把其 `.txt` 结果用于比较，只有在运行窗口内相应配置文件发生变化的任务需要定点重验；不能仅凭文件名哈希认定运行时 treatment。
- 缺陷主要破坏该入口的实验 provenance；发生跨层混代时还会改变实际模型行为，结果不能归属任一完整 treatment。

## 修复与重测建议

1. `llm_eval.py` 导入 `resolve_treatment_snapshot`；当 `args.method == "tli"` 时，在首次求 method name 之前只解析一次。
2. `get_method_name_with_info(args, snapshot)` 与 `register_patch(model, args, snapshot)` 必须消费同一对象；结果文件同时写 canonical manifest/内容 SHA，避免只靠短哈希。
3. 为生产 TLI 入口考虑在 `register_patch(..., snapshot=None)` 时 fail closed；若必须保留测试/库兼容路径，应提供显式 legacy 标志，避免新增入口无意旁路快照。
4. 加入 `llm_eval.py` 调用点级交错回归：resolve A 后替换为 B，断言名称、每层 skip/basis、落盘 manifest 全部仍为 A；普通 Python 与 `python -O` 双跑。
5. 对曾由该入口生成且将用于结论的结果，按实际 argv、配置内容 SHA 和作业日志筛出窗口内变更任务；仅命中者重跑。

## 旧发现复查与反证

- `TL-E121-OUTPUT-SNAPSHOT-081`：在 `pred.py` / `pred_ruler.py` 上 **fixed/rechecked**；在 `llm_eval.py` 上为本轮新确认的遗漏残余，不能把前两入口通过扩展成“全部 RULER 入口已修复”。
- `test_b01_b02_fix.py` 的 384→256、640→768、1/640→1/768 与 B10 新边界一致，本轮未发现新数值错误。
- 该测试仍硬编码 `/home/wangyuanshuo02/sglang/python`，会在该路径存在时让独立 worktree 优先导入主树；但此行早于本区间存在，`d280bd7` 没有修改它，故本轮只保留为既有测试 provenance 风险，不冒充新增缺陷。本审计环境该路径不存在，且无 torch/GPU，未复跑该 GPU 测试。

## 资源阻塞与下一检查点

本环境 `ModuleNotFoundError: torch`，无 CUDA/GPU；因此未执行 081 官方 CPU+torch 套件、B01/B02 GPU 用例或真实 RULER。已对全部相关 Python 文件执行 `compileall`，结果通过。下一次源代码变更检查优先复核 082 是否接入 `llm_eval.py` 及生产入口是否 fail closed；若源码未变，不重复报告。
