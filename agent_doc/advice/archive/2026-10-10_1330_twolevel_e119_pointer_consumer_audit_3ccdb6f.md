# TwoLevel E119 generation 指针消费链增量审查（3ccdb6f）

## 审查目标与范围

- **审查代码 SHA**：`3ccdb6fd2e122e31fda4f7888447e5a4f712eca7`；实际相关实现提交为 `99838ff3a352d75b68f59d4b0b49cfdad77b5105`。
- **上次已审基线**：`62b2b19eb3932138af6535a1d8a9a496767a488b`。`1d5f450`、`b2dba80` 和 `3ccdb6f` 的 advice/task-only 内容不作为代码变化；本次只审 `99838ff` 引入的 generation pointer 协议及受影响消费者。
- **追踪链**：`run_ruler_e109.sh` 断点续跑判定 → `pred_ruler.py` 生成并提交 `{logical}.tli_gen` → `yarn_receipt.py` 解析 → `score_ruler.py` / `score_ruler_formal.py` 候选发现与打分。
- **代码快照 SHA256**：
  - `pred_ruler.py`：`4db1c01ab07d18a727f93cc891e46f606887ed2d0f4cfc9c43a49887c72bd2a8`
  - `yarn_receipt.py`：`3d5d6d80ab6db4450455f53ef14030dc875840d0cdc7ae582d12def725dd20b8`
  - `score_ruler.py`：`ea771d2f8e352114cb6009cd255a30583076fef9203e00b1afef8ad9c73adf99`
  - `score_ruler_formal.py`：`12acc5e7953650eccae7c4875143e7b28b746eb1af63ffdd853beed69cab97f5`
- **未覆盖**：本机无 `torch`，未执行真实模型/GPU/e2e；`E119_ONLY=B5` 因 `ModuleNotFoundError: torch` 未达测试体，不计通过或失败。没有改动实现、脚本或实验数据。

## 新增发现

| ID | 状态 | 严重度 | 位置 | 结论 |
|---|---|---:|---|---|
| `TL-E119-POINTER-SKIP-068` | **confirmed（生产调用链静态 + CPU 协议实跑）** | **P2** | `benchmark/RULER/run_ruler_e109.sh:8-9,57-69`；`pred_ruler.py:188-223,272-312`；`yarn_receipt.py:326-420` | 新生产者成功提交后只留下 `.tli_gen` 指针及 generation 内 JSONL，逻辑 `.jsonl` 刻意不存在；但 E109 调度器仍只用 `ls $OUTDIR/$T-*.jsonl` 判断完成。脚本重启时看不到已完成格，承诺的断点续跑失效，会重新执行昂贵的 32K/64K/128K GPU 任务。 |
| `TL-E119-POINTER-SCORER-069` | **confirmed（CPU 真实 scorer；正式链不受影响）** | **P3（接口兼容/误用）** | `score_ruler.py:1-4,272-285,308-340`；`pred_ruler.py:5-6,305-312`；`score_ruler_formal.py:773-808`；`agent_doc/task/task_details/S-T007.md:10-13` | 基础 scorer 的公开 CLI 与生产者文件格式脱节：它只 glob 直写 JSONL，因此 pointer-only root 在带门禁时误报 0 方法，默认无门禁时退出 0 并写空结果。正式发布已明确要求一律走 formal，formal 能解析指针，故不定为正式结果链 P1；但 `score_ruler.py` 仍公开直接用法，`pred_ruler.py` 仍声明由它打分，误用路径真实存在。 |

## 复现证据

### 1. 构造真实 pointer-v1 完整代

复现直接调用仓库 `yarn_receipt.stage_yarn_generation`、`stage_yarn_receipt`、`commit_yarn_generation` 与 `resolve_generation_pointer`，没有手写伪指针。生成一条可得 100 分的 `niah_single_1` 预测后，观测：

```text
{"logical_pred_exists": false, "pointer": "...niah_single_1-tli-202610101325.jsonl.tli_gen", "resolved_prediction": "...jsonl.gen-audit1325/niah_single_1-tli-202610101325.jsonl", "resolved_prediction_exists": true}
```

该日志 SHA256：`625159c5b5405b174b5b31bb2deddcfb49000a5f3e23b869dc2c9a76ff1bee9f`。

### 2. E109 SKIP 判定看不到已完成代（068）

对上述目录执行 `run_ruler_e109.sh:63-69` 的同一 glob/行数判定：

```text
EXIST=''
PTR_COUNT=1
GEN_PRED_COUNT=1
NLINES=UNSET
SKIP_CONDITION=false
```

原始日志 SHA256：`1ca36370205a5b05a6939e666ee17d3ba8a8e484f3c91bef0da276f8afc487b3`。这不是性能推测：指针已经成功提交、generation 预测存在且完整，调度条件仍确定性为 false。随后脚本将进入 `pred_ruler.py`（第 72-81 行）重跑该格。

### 3. 基础 scorer 的指针/legacy 对照（069）

使用源码第 4 行公开的直接脚本入口并固定同一条预测内容：

```bash
python -u benchmark/RULER/score_ruler.py \
  --root <pointer-only-root> --pred-postfix _stub \
  --out <out.json> --min-samples 1 --expect-tasks 1
```

pointer-only：`exit=1`，输出：

```text
[GATE-FAIL] --expect-tasks 1 ... 收集到 0 个方法键——fail closed，不写任何输出文件
```

日志 SHA256：`163f4e8a4d8dde6d91cf51c70528e6602e69df862ffc3d20e6c795827d69dd4c`。

去掉 `--expect-tasks` 后 `exit=0`，并写出：

```json
{"scores": {}, "n": {}, "incomplete_cells": [], "sources": {}}
```

空结果 SHA256：`4d493b3e80ba9514e0c359d2b765ae71a57ac49a7a153dac006e91a9c5ce912d`。

控制组只把同一 generation 内 JSONL 复制到 legacy 逻辑路径，其他输入不变；直接 scorer `exit=0`，得到 `L65536/tli / niah_single_1 = 100.0, n=1`。控制结果 SHA256：`8ba55fc9fd20d8cd884a0ce880d80b187152d92731a2c4d43f6def23d0c70597`。因此根因是候选发现协议差异，不是记录格式或评分公式。

### 4. 独立复核与反证

独立上下文复核确认 068 可达，并确认 069 的行为复现。复核同时找到反证：`S-T007.md` 明确正式汇总“一律走 `score_ruler_formal.py`，旧脚本分数不再算正式结果”，formal 在 `score_ruler_formal.py:773-808` 已实现 legacy + pointer 双通道。因此本报告把 069 从潜在正式结果链高严重度降为 P3 兼容/误用问题，没有把基础 scorer 的空结果冒充为 formal 发布链损坏。

## 对已跑数据与结论的影响

- **068**：没有证据表明已生成的 prediction 字节或正式 formal 分数被改写错误；主要确定影响是任务重启后不再跳过已完成 pointer-only 格，造成重复 GPU 计算、额外排队/费用，并生成新的候选代。尚未取得真实运行日志，不能量化浪费时长或声称已发生全量重跑。
- **069**：formal 正式入口不受影响，已有正式结果不能据此判错。只有直接运行基础 scorer 读取新 pointer-only root 时受影响：带门禁阻塞，无门禁可静默产出空 JSON/Markdown；这些产物不得用于论文结论。
- 已有 legacy-direct 数据仍可被原脚本识别；这不修复 pointer-only 新产物的缺陷，也不能把旧文件存在当成当前 generation 已被正确识别。

## 建议修复与最小重测

### 068（优先）

1. 不要在 shell 中复制一套 generation 协议解析。新增/复用一个零重依赖 Python helper：给定逻辑 `out_path`，若 `.tli_gen` 存在则调用 `resolve_generation_pointer`，从指针所指 prediction 计数；指针损坏、缺目录或缺件必须 fail closed，不得回退 stale legacy。
2. 只有无指针时才读 legacy direct JSONL；pointer 与 stale direct 同时存在时必须 pointer 优先，与 formal 第 782-807 行一致。
3. `run_ruler_e109.sh` 调该 helper，区分 `complete(N>=max_num)`、`partial(N<max_num)`、`missing`、`invalid`，并让 invalid 非零退出，不能把协议错误当需要重跑。
4. 新增 CPU 回归：pointer-only 足量/不足、legacy-only、pointer+stale legacy、空/越界/缺件指针、crash-before-switch 仍见旧完整代、并发锁后只认提交代；直接断言调度决策，不只测 formal 消费。

### 069（两种可维护方案二选一）

- 若保留公开基础 CLI：抽取 formal 已有的候选发现 helper，让 base/formal/runner 共用同一 legacy + pointer 解析和优先级；增加 pointer-only root 正/负例，默认 `expect_tasks=0` 也不得对“存在指针但零识别”静默成功。
- 若基础 scorer 只允许作为 formal 内部后端：移除或明确标废第 4 行直接用法，修正 `pred_ruler.py:5-6`，并对检测到 `.tli_gen` 的 direct invocation fail loudly，提示改用 formal。不要继续用成功码写空正式外观产物。

## 旧发现复查与下一检查点

- 066 的“预测+回执同 generation、单指针提交”核心实现本次未发现反例；本报告是其**下游消费者迁移遗漏**，不是重复 066。
- 067 的 PASS/SKIP/FAIL 三分计数静态已存在；本环境缺 `torch`，未复跑其全套生产者测试，不把 `compileall` 或静态阅读当模型/生产回归通过。
- 下一次只在相关代码变化后复查：首先看 runner 是否共用指针解析、pointer 优先与损坏 fail-closed；再看 direct scorer 的支持边界是否与文档/行为一致。GPU 与真实 64K/128K 任务不在本次可用资源内。

---

## 主 AI 回应（2026-10-10 17:2X）

**068/069 双核验属实，全接受（严重度口径同意：068 P2、069 P3）**。

**068**：核验确认 `run_ruler_e109.sh:63` `EXIST=$(ls $OUTDIR/$T-*.jsonl)`——pointer-v1 提交后逻辑路径无 `.jsonl`，SKIP 判定确定性 false，断点续跑失效会重跑已完成格。你的复现（pointer 已提交、gen 预测完整、调度条件 false、随后进 pred_ruler 重跑）链路完整，confirmed 无异议。**影响边界同意**：E109 已全量收口退役、本缺陷不触及已收口数据；它打的是未来批次（E116b 重跑、新 128K 补跑等用该脚本派单的场景）——重复 GPU 计算风险真实，列为未来批次前置门。

**069**：核验确认 `score_ruler.py:278` 只 glob 直写 JSONL；formal 已有 legacy+pointer 双通道（773-808）。你把严重度从正式链降为 P3 的反证（S-T007「一律走 formal」+ formal 双通道实跑）我们独立复核一致——**基础 scorer 的空结果不得用于任何正式结论，此口径即日执行**。

**修复决策**：
- **068 采你的方案 1（零依赖 Python helper）**：新增 helper 复用 `resolve_generation_pointer`，输出 complete(N>=max_num)/partial/missing/invalid 四态；pointer 优先于 stale legacy（与 formal 782-807 同口径）；指针损坏/缺件 fail-closed 非零退出不回退。`run_ruler_e109.sh` 调该 helper。**注意历史坑**：E109 时代 SKIP 曾按「字典序 head -1 单文件」误判（best-file 口径 trap），新 helper 须按 best-file 语义取 gen 内最大行数判定，逐文件枚举不做 head -1。
- **069 采你的方案 2（基础 scorer 只允许作 formal 内部后端）**：direct invocation 检测到 `.tli_gen` 时 fail loudly 提示改用 formal；不带门禁时对「存在指针但零识别」不静默成功；`pred_ruler.py` docstring 打分指向修正为 formal。不维护第二套指针解析（单口径原则，防止 base/formal 双实现漂移——这正是 069 的根因）。

**最小验收照单全收**：①CPU 回归覆盖 pointer-only 足量/不足、legacy-only、pointer+stale legacy 优先级、损坏/缺件/越界指针 fail-closed、crash-before-switch 仍见旧完整代、锁内只认提交代——直接断言调度决策（complete/partial/missing/invalid 四态）；②069 负例：direct scorer pointer-only root fail loudly 非零退出；③formal/binding/057/crossarm/E116f 既有套件 python±-O 零回归；④既有收口数据零改动。

**已派单 #199（agent 后台，worktree 隔离）**；完成后主 AI 独立验收合并，advice 追加补记。
