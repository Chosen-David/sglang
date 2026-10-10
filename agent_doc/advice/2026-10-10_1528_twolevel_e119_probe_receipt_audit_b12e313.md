# TwoLevel E119 完成探针回执校验增量审查（b12e313）

## 审查目标与范围

- **审查代码 SHA**：`b12e313fab1e95b80a2333fa950a58aaa525916e`；本次相关实现提交为 `a76fd521f`。
- **上次已审代码基线**：`3ccdb6fd2e122e31fda4f7888447e5a4f712eca7`。其后的 advice/task-only 提交不算代码变化；本次只审新增的 pointer-aware 完成探针、E109 调度接线及正式消费器之间的契约。
- **追踪链**：`run_ruler_e109.sh` 调度 → `gen_completion_probe.py` 候选发现/完成判定 → `yarn_receipt.py` generation 指针解析 → `score_ruler_formal.py` 回执与预测绑定门禁。
- **代码快照 SHA256**：
  - `gen_completion_probe.py`：`7c1d960ba48bc953d9b35817bf9a14f5a1cdc0740820467850b06aabdf0e7406`
  - `yarn_receipt.py`：`3d5d6d80ab6db4450455f53ef14030dc875840d0cdc7ae582d12def725dd20b8`
  - `run_ruler_e109.sh`：`b376f49ed70731a9f1d9e7d2219c335d71393bb3e1177eeef773f8f72cf0f482`
  - `score_ruler_formal.py`：`12acc5e7953650eccae7c4875143e7b28b746eb1af63ffdd853beed69cab97f5`
  - `test_e119_pointer_consumer_068_069.py`：`be44ec4ede0e79657369b4990971e5848e6c7746a6a3b343f87cddd1dc22b6c0`
- **环境与未覆盖项**：Python `3.12.14`；本机没有 `torch`，因此现有 068/069 套件的生产者 fixture 在导入阶段被环境依赖阻断，未把其 8 个依赖生产者的格记为实现回归；未执行 GPU、真实模型或真实 E109 数据。没有改动实现、测试脚本或实验数据。

## 新增发现

| ID | 状态 | 严重度 | 位置 | 结论 |
|---|---|---:|---|---|
| `TL-E119-PROBE-RECEIPT-VALIDATION-070` | **confirmed（静态调用链 + 零依赖 CPU 复现 + 独立复核）** | **P2** | `benchmark/RULER/gen_completion_probe.py:147-167`；`yarn_receipt.py:407-421,592-640`；`run_ruler_e109.sh:66-91`；`score_ruler_formal.py:625-657` | 新完成探针只要求 generation 内“有预测文件且有回执文件”，随后只数预测行；它不解析回执、不校验 schema/status，也不核对回执声明的 basename、SHA256、行数。因此“预测足量 + 存在但无效/错代的回执”被判 `complete`，E109 直接 `SKIP`；同一产物到正式汇总时必被 fail-closed 拒绝。调度与正式交付的完成定义发生分裂。 |

## 违反契约与触发条件

`run_ruler_e109.sh:66-75` 明确把损坏指针/协议错误定义为 `PROBE-FAIL`，而 `88-91` 把 `complete` 直接计入 `SKIPPED`。`gen_completion_probe.py:130-132` 也声称指针损坏要 fail-closed。但实际调用的 `resolve_generation_pointer` 在 `yarn_receipt.py:420-421` 明确声明**不校验回执内容**；探针在解析后只执行 `_count_lines(gen["pred_path"])`。

可达触发条件：pointer-v1 指向的 generation 目录、预测文件和回执路径都存在，预测行数达到 `max_num`，但回执为半写/损坏 JSON、未知 schema、非 complete、或其 basename/SHA/行数与预测不一致。生产者正常原子提交不会主动制造这些状态，但事后损坏、人工改写、错误迁移或未覆盖的协议实现都可产生；既然该探针承担恢复前的 fail-closed 门，不能只验证文件存在。

实际行为：探针返回 `STATE=complete`，runner 跳过该格，并可在没有其他失败时输出 `ALL DONE`。预期行为：任何 pointer generation 的回执或同代绑定不合法都返回 `STATE=invalid`、exit 2，使 runner 计失败并阻止假完成。

## 最小 CPU 复现与原始证据

复现脚本创建 3 行 pointer generation，但将同目录回执写为合法 JSON 空对象 `{}`；随后分别调用新探针核心和正式共享回执校验器。脚本 SHA256：

```text
42662c9bd3520a5c2769016fbcb4c3afb1c10eceb623539a3ebf7dc5dfa7ed5d
```

实际输出：

```text
probe_result={"base": "vt-tli_64_128_1024_c4_-10101528.jsonl", "n": 3, "source": "pointer", "state": "complete"}
formal_receipt_error="... receipt_version=None 不在已发布协议 ('producer-yarn-config-v1', 'producer-yarn-config-v2')（存在即证据，非本协议拒收）"
```

原始输出 SHA256：`3f292cc2b82de375789ab5223d2ffdbd177afc8c52c85a57a76c2a88b9eed394`。这构成同一输入在调度入口“完整”、在正式入口“不可消费”的确定性反例。

正式入口并非只做更严格的可选审计：`score_ruler_formal.py:625-633` 解析 JSON 并调用 `validate_producer_receipt`，`643-657` 又把 v2 回执的 SHA256/行数与实际预测逐位比较。故仅在探针中补 `json.load` 或仅补 schema 校验仍不足以闭合错代/篡改场景。

## 独立复核与反证

独立上下文复核确认 070 是可行动缺陷，并同意 P2：它能让不可正式消费的 GPU 格永久被断点调度跳过并产生批脚本假完成；但正式汇总仍 fail-closed，因此没有证据支持“错误分数已发布”或 P1 数据结论污染。

反证边界也保留：当前没有发现既有 E109 generation 回执已经损坏；本机没有真实输出目录，不能据此判定历史 E109 数据受影响。现有测试 `G5`（`test_e119_pointer_consumer_068_069.py:297-336`）覆盖缺目录、路径逃逸和**缺回执**，没有覆盖“回执文件存在但 JSON/schema/绑定无效”，所以此前报告的套件通过不能反驳本发现。

## 对已跑数据与结论的影响

- **未证实已有数据受影响**：没有读取到现有 E109 真实 prediction/receipt，不能声称已有分数错误或需要全量重跑。
- **正式发布仍有保护**：若所有结果严格经过 `score_ruler_formal.py`，无效回执会被拒绝，不会静默进入正式分数。
- **确定的可靠性影响**：一旦出现上述坏态，runner 会持续 SKIP；重复启动不会自愈，正式评分又持续拒绝，形成“调度称完成、交付永远失败”的闭环断裂，需要人工定位后处理。

## 建议修复与最小重测

1. 在 `yarn_receipt.py` 抽出零重依赖的共享 `validate_committed_generation`（名称可调整），由 probe 与 formal 共用，避免第三套回执判断漂移。
2. pointer-v1 完成判定必须：读取回执单一 bytes 快照；JSON 解析失败即 invalid；执行 `validate_producer_receipt`；要求当前 v2 complete 协议；核对 `prediction_basename`、实际预测 SHA256、实际行数与回执逐位一致。任一失败都透传 `[GATE-FAIL]` 并 exit 2。
3. 只有上述绑定验证通过后，才按实际行数与 `max_num` 输出 complete/partial。legacy-direct 维持既有行数语义，不应被 pointer 回执门禁误伤。
4. 给 `test_e119_pointer_consumer_068_069.py` 增加真实提交代破坏负例：回执非 JSON、`{}`、`status != complete`、basename 错、SHA 错、lines 错，全部必须 `STATE=invalid`/`PROBE-FAIL`；合法 v2 pointer 足量仍 SKIP、legacy 三态仍通过。测试需继续在 `python` 与 `python -O` 下运行，缺依赖时不得把未执行冒充通过。
5. 修复后对现有 E109 输出做一次只读预检：逐 pointer 验证回执和预测绑定，列出 invalid 格再决定定点重跑；不要在缺真实数据时预先宣称需要全量重跑。

## 旧发现复查与下一检查点

- 068 的 pointer 候选发现、pointer 优先、best-file 与缺件 fail-closed 已接线；070 是其**完成证据闭包缺口**，不是重复报告“看不见指针”。
- 069 的 direct scorer fail-loudly 修复静态仍在，本次未发现新反例。
- 下一次只在相关代码变化后复查：优先验证 probe/formal 是否共用同一 committed-generation 校验、现有回执是否只读预检、以及新增绑定破坏负例是否覆盖 python±`-O`。
