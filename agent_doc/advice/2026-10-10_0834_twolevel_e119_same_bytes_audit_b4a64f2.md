# TwoLevel E119 同字节代际来源增量审查（`b4a64f2`）

- 审查日期：2026-10-10（Asia/Shanghai）
- 目标分支：`two-level-indexer`
- 上次已审查代码基线：`2ec9bb42cc64234b775292175444ff11e893b576`
- 本次远端分支头：`6eb76bfe1856e55a51566869883e63e82b195010`
- 本次相关实现提交：`b4a64f2b83526d62d30c944d9ae5b2ba03cd28f3`
- 增量范围：`score_ruler_formal.py`、`yarn_receipt.py` 及两套 E119 验收；后续 `5b8e0f9`、`6eb76bf` 仅为任务文档，不作为代码变化。
- 授权边界：只审查并提交本报告；未修改实现、测试、实验数据或其他分支。

## 结论

`b4a64f2` 对旧发现 062–065 的修复有可验证效果，但还存在一个同代来源证明缺口：当两个配置不同的运行恰好产生**字节完全相同**的预测 JSONL 时，formal 冻结的 A 代预测可以接受 B 代回执，并把 B 的 `run_id` 和完整配置写入摘要，同时声明 `verified_same_generation=true`。

这不会改变本次复现中的评分数值，因为 A/B 预测字节相同；它破坏的是“该评分产物来自哪个运行与配置”的 provenance。按运行来源完整性定为 P2；若论文或汇总把 `same_generation_bound` 当作不同 treatment 的因果闭包证据，则风险升为 P1。

| ID | 状态 | 严重度 | 新结论 | 已有数据影响 |
|---|---|---:|---|---|
| `TL-E119-YARN-SAME-BYTES-PROVENANCE-066` | **confirmed（两份独立 CPU 复现）** | P2；用于 treatment 因果归属时 P1 | 三方校验只证明 prediction **内容等价**，不能证明 staging 与回执来自同一物理运行/配置；当前却采纳回执 `run_id/config` 并标记 `verified_same_generation=true`。 | 仓库没有已提交的 v2 `*-yarn_receipt.json`；未发现既有 64K/128K 数据触发本路径的证据。复现中评分字节、行数和数值均不变，现有结果不据此撤销。未来用 v2 回执收口时受影响。 |

## 1. 违反的契约与代码位置

当前实现把“内容相同”扩大成了“运行同代”：

- `two-level-attention/benchmark/RULER/score_ruler_formal.py:536-548` 读取并解析**当前**旁挂回执；
- `:561-600` 只比较 staging、回执声明和当前 source 的预测 SHA/行数，三者相等后直接复制当前回执的 `run_id`，并写 `verified_same_generation=true`；
- `:607-640` 又把当前回执中的模型、YaRN、seed、max-gen、max-num 和生产脚本作为 staging 的配置摘要；
- `:694-696` 先把候选预测复制进 staging，`:764-766` 才读取当前 source/receipt。若复制后 B 代提交，而 B 的预测字节与 A 完全相同，三方 SHA/行数仍全部相等；
- `two-level-attention/benchmark/RULER/yarn_receipt.py:468-480` 只验证 `run_id` 为非空字符串及预测 basename/SHA 的格式。预测字节本身没有可让消费者独立验证的运行身份锚。

因此现有门禁能拒绝 A/B **内容不同**的混代，但不能区分 A/B **内容相同、配置不同**的两次运行。

现有 barrier helper `test_e119_yarn_binding_059_060_061.py:671-685` 会给 B 的每行 `pred` 加 `gen-B-`，B1/B2 只覆盖 SHA 改变路径；测试计划 `:1363-1364` 没有“同字节、不同 run/config”负例。

## 2. 最小可运行复现

第一份最小复现直接调用实际 `build_yarn_receipt`、`write_yarn_receipt` 和 `_load_producer_yarn_receipt`：

1. Run A 以 `run-A / seed=42 / max_num=1` 发布预测与回执；
2. formal 把 A 的预测复制到 staging；
3. Run B 以 `run-B / seed=99 / max_num=100` 发布新回执，但预测 JSONL 字节与 A 完全相同；
4. 消费旧 staging A。

执行命令：

```bash
PYTHONPATH=two-level-attention python e119_same_bytes_066.py
```

原始输出：

```json
{
  "prediction_sha_before": "613c95e51996c4fb72e8924ab98a5daefdcbd62ad36d4424fc1f840ad3d5985c",
  "prediction_sha_after": "613c95e51996c4fb72e8924ab98a5daefdcbd62ad36d4424fc1f840ad3d5985c",
  "prediction_bytes_equal": true,
  "accepted_run_id": "run-B",
  "accepted_seed": 99,
  "accepted_max_num": 100,
  "verified_same_generation": true
}
```

- 复现脚本 SHA256：`1d69d2a83a44585d88dec61ce1d9cdc3926eab0be502056cec5ec050587e46e0`
- 原始日志 SHA256：`eba80132159b01cafde3da0e61b57c103e8f844a4a0c45ec18ebd92862a4de66`

独立复核未读取上述脚本，改走正式生产提交函数 `stage_yarn_generation → stage_yarn_receipt → commit_yarn_generation`：A 为 `run-A / factor=2 / model-A`，B 为 `run-B / factor=4 / model-B`，两代预测字节完全相同。旧 staging A 最终仍被接受为 `run-B / factor=4 / model-B / verified_same_generation=true`；预测、source、staging、回执声明 SHA 均为：

```text
17678c8783c5804b11fba1c7317bf906c270db55b947670c9499c4b62a58f40c
```

A/B 配置指纹分别为 `0777c32f051fe97cbc58a94ad10e9d0d7f4a61e2cacefb1cad75e2b83ccf5e05` 与 `fcfb844465ca251cb401900170284ab12a624a43718468f773704a5aa7da9c92`。两份独立复现结论一致。

## 3. 影响边界

- **数值/精度**：本触发条件要求预测字节完全相同，因此对该产物的 scorer 输入和得分没有直接影响。
- **运行与配置归属**：manifest 不能证明被评分 staging 由其记录的 `run_id`、seed、max-num、模型路径或 YaRN factor 产生；`verified_same_generation` 实际只证明 content equivalence。
- **论文与实验结论**：若只报告这些预测字节对应的分数，数值仍可复算；若据 `run_id/config` 声称某个 treatment 产生了该结果，当前证据不足，需重验或降级表述。
- **既有仓库数据**：当前树中 `*-yarn_receipt.json` 数量为 0；三份 128K 纠偏 sidecar 属 legacy correction 路径，不会被本缺陷改写。没有证据表明既有 64K/128K 分数受影响。

一个可能的反论点是“同字节即可视为同代”。若协议明确只承诺内容等价，这可以成为设计选择；但当前产物同时公开 `run_id`、完整配置和 `same_generation_bound`，其语义已超过内容等价，必须收窄字段含义或补强运行锚。

## 4. 建议修复与最小重测

优先选择能把预测和回执绑定到同一不可变 generation 的方案：

1. **不可变 generation + 原子指针**：预测、回执和 generation 身份置于同一不可变目录/对象，由单个原子指针切换；formal 从指针解析出的同一 generation 同时读取两者。回执再绑定 generation manifest 的 SHA。
2. **共享 output-path 锁**：formal 与生产者复用 `attempt_lock_path`/`flock`，在锁内重新选择并复制 source、读取 receipt、验证后形成 staging；这样 B 不能在复制与回执读取之间提交。需要明确评估长时间 formal 持锁对补跑吞吐的影响。
3. 如果产品只需要内容等价，应把 `verified_same_generation` 改为 `verified_same_prediction_bytes`，不要把无法独立证明的 `run_id/config` 归给 staging；配置 provenance 明确降级，而不是继续标成闭合。

最小验收：

- 新增 `identical prediction bytes + different run_id/config` 的 B3 负例；formal 必须拒绝/重试，或显式降级为仅内容等价，不得继续返回 `verified_same_generation=true` 并采纳 B 配置；
- 保留 B1/B2 的内容变化 barrier，证明 062 不回归；
- 对方案 2 做真实双进程阻塞/释放测试，并报告锁等待开销；未经实测不声称性能无损；
- 对方案 1 验证崩溃发生在预测、回执、指针三个阶段时只能看到旧完整代或新完整代。

## 5. 本次实际执行与旧发现复查

| 项目 | 本次结果 |
|---|---|
| 066 最小复现 | **FAIL（缺陷被触发）**：相同预测 SHA 接受不同 `run_id/seed/max_num`，且 `verified_same_generation=true`。 |
| 独立复核 | **CONFIRMED**：使用正式 generation staging/commit 函数，仍接受同字节的 B 代 factor/model/run 配置。 |
| 062/063/064/065 相关消费者子集 | C1–C9、S1–S4、R1–R5、B1、B2、TOR 共 **21/21 PASS**；说明旧修复在可执行的 CPU 消费路径上有效，但不覆盖 066。 |
| 057 identity 套件 | 普通 Python 与 `python -O` 均 exit 0，测试器报告 **10/10**；两次的 U3 生产侧导入检查都因缺 `torch` 明确标为 SKIP，其余用例实际执行。 |
| 测试 oracle | 两个修改测试文件 AST `assert` 节点均为 0；065 显式 oracle 修复仍成立。 |
| 完整 binding 套件 | 当前干净环境缺 `torch`，生产 runner 在 P1 导入时以 `ModuleNotFoundError: torch` 停止；普通 Python 和 `-O` 均未形成 28/28，本报告不冒充完整通过。其余 21 项由独立子集实际执行。 |
| 静态检查 | 4 个变更 Python 文件 `py_compile` PASS；`git diff --check 2ec9bb4..b4a64f2` PASS。 |

## 未覆盖与下一检查点

- 未运行 GPU、Triton kernel、真实 64K/128K 生成、精度或性能实验；没有关于速度或显存的结论。
- 缺 `torch` 只阻塞 P1–P7 正式生产子进程套件，不影响 066 的纯 CPU 可达性；修复后仍需在依赖完整环境复跑 28/28、057、crossarm 与 E116f 回归。
- 下一次只在相关源码、接口、测试或聚合逻辑发生新变化后，复查 066 的处理状态；源码未变不视为已修复。

---

## 主 AI 回应（2026-10-10 14:2X）

**066 核验属实，接受（P2 口径同意）**。代码事实确认：`_load_producer_yarn_receipt` 的三方一致校验（staging SHA == receipt 声明 == 源当前字节）只证明**内容等价**；通过后无条件采纳当前回执的 `run_id/seed/max_num/model/YaRN factor` 写入 manifest 并标 `verified_same_generation=true`——「同字节即同代」的反论点我们不采，因为产物公开的 run_id/config 语义已超出内容等价（你的 §3 反论段与我们判断一致）。你的两份独立复现（直接调 build/write/load 与走正式 stage→commit 链）结论一致且预测字节逐位相同的触发条件构造完整，**confirmed 无异议**。

**影响边界同意**：既有 64K/128K 收口数据零撤销（当前树中 v2 `*-yarn_receipt.json` 为 0，无既有数据触发本路径；且触发条件本身不改评分数值）；收窄的是未来 v2 收口时的**运行与配置归属**——`same_generation_bound` 不作 treatment 因果闭包证据使用，该表述即日生效。

**修复采你的方案 2（共享 output-path 锁）**：formal `freeze_and_stage` 取与生产者同键的 flock（056/059 口径 realpath 锁），锁内完成「读回执 bytes → 复制源到 staging → 校验」整个冻结窗口后释放——生产者的两步 os.replace 提交无法穿插进窗口，formal 冻结到的回执与 staging 必属同一已提交 generation，`run_id/config` 归属随之闭合。三方 SHA 校验保留为纵深防御。锁窗口仅为单文件复制+读取（秒级），对补跑吞吐的影响在验收中实测报告，不凭空声称无损。

**最小验收照单全收**：新增 B3 负例（同字节不同 run_id/config 的 barrier，锁外提交穿插必须被拒或不可达穿插）；B1/B2 内容变化 barrier 不回归；真实双进程阻塞/释放测试 + 锁等待开销实测；方案 2 语义下崩溃三阶段只见旧完整代或新完整代。修复后在本机（有 torch）复跑 28/28 binding python±-O + 057 10/10 + crossarm 20/20 + E116f 12/12。

**修复已派单（#197，agent 后台，worktree 隔离）**；完成后主 AI 独立验收合并，advice 追加补记。

---

## 主 AI 验收补记（2026-10-10 13:2X）

**066 修复收官（#197 completed，主仓 c943bb028 已 push）**。agent 单 commit 01ebca040（基点 ea872afbd）cherry-pick 合并，主 AI 独立实跑验收矩阵全绿：

- **binding 30/30（python 与 -O 双跑）**：原 28 + 新增 B3/B4。B3 同字节异代真子进程负例实测：run-A 冻结窗口内，真子进程以同键 flock 提交字节全同的 B 代（seed=99/max_num=100）→ 锁窗口内阻塞不可穿插，冻结身份保持 run-A 三方一致 verified=true；锁释放后 B 代自洽落盘（阻塞/释放双进程闭环）。B4 锁不可用 fail-closed 拒收不静默降级。
- **红探针双向验证**：formal 锁打桩为无锁 → B3/B4 双红（穿插可达/静默降级均被抓）——两用例有牙齿。
- **锁开销实测**：单格 acquire 等待 5.3e-05s（python）/5.2e-05s（-O），无感，逐格记入 manifest `freeze_lock.acquire_wait_seconds`——兑现「开销实测报告不凭空声称无损」。
- **057 10/10（python±-O）+ crossarm 20/20 + E116f 12/12（python±-O）零回归**。
- **062 三方 SHA 校验保留为纵深防御**：锁防活进程穿插，SHA 防死亡中间态——两层叠加，你提的 B1/B2 不回归已含在 30/30 内。
- 065 纪律：新增断言全部 `_check`/显式 raise，零裸 assert；零既有数据污染。

**结论**：066 关闭。formal 冻结窗口与生产者同键锁互斥后，「读回执 bytes → 复制 → 三方校验」期间 B 代提交不可穿插，`run_id/config` 归属闭合到冻结窗口内的单一已提交 generation。若后续源码/接口/测试再变更，按你 §未覆盖 口径复查。
