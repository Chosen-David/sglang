# sglang two-level-indexer 每小时例行审查报告（2026-10-10_1554）

## 审查区间

- **基准 SHA**：`a32585a9bef488640d949c0687d54e3369010e5c`
- **新 origin/two-level-indexer SHA**：`04f48a51def7da358ae977b10a1e95c2ff74222f`
- **区间 commits**：
  - `a76fd521f fix(ruler): #199 068 指针感知 SKIP 探针 + 069 direct scorer fail loudly`
  - `6ea2ffc93 docs(task): S-T016 068/069 收官登记 + advice 验收补记与归档`
  - `b12e313fa E117 8层逐层配置小试判决：NO-GO`
  - `04f48a51d docs(audit): report E119 probe receipt validation gap`

## 代码变更概览

本轮变更集中在 `two-level-attention/benchmark/RULER/`：

1. 新增 `gen_completion_probe.py`：068 完成探针，pointer-v1 + legacy 双通道 SKIP 判定。
2. 修改 `score_ruler.py`：069 直接入口对 pointer root 增加 `[GATE-FAIL]` 门禁。
3. 修改 `run_ruler_e109.sh`：调度脚本改走探针四态判定（`complete/partial/missing/invalid`）。
4. 修改 `pred_ruler.py`：仅 docstring 更新，逻辑未变。
5. 新增 `test_e119_pointer_consumer_068_069.py`：068/069 红绿测试套件。
6. 文档：`agent_doc/advice/2026-10-10_1528_twolevel_e119_probe_receipt_audit_b12e313.md` 已在区间内指出 receipt 校验缺口；`04f48a51d` 即为报告该缺口的提交。

目标高优先级区域 `python/sglang/srt/layers/attention/tli/` 与 `two-level-attention/sparse_attn/` 本轮无代码变更。

## 实测验证

### 068/069 回归套件

运行 `test_e119_pointer_consumer_068_069.py`：

```bash
PYTHONPATH=$PWD python3 -m benchmark.RULER.test_e119_pointer_consumer_068_069
PYTHONPATH=$PWD python3 -O -m benchmark.RULER.test_e119_pointer_consumer_068_069
```

结果：`PASS=11 SKIP=0 FAIL=0 (total 11)`，常规与 `-O` 两种模式均通过。G1-G7、N1-N3、RP 全部 green，068/069 的已有修复契约在本轮未退化。

## 发现项

### P2：pointer-v1 完成探针不校验 generation 回执内容（TL-E119-PROBE-RECEIPT-VALIDATION-070 仍在）

**位置**：
- `two-level-attention/benchmark/RULER/gen_completion_probe.py:147-167`（pointer 分支仅 `_count_lines`）
- `two-level-attention/benchmark/RULER/yarn_receipt.py:407-421`（`resolve_generation_pointer` 明确声明不校验回执内容）

**实测证据**：

独立复现脚本（写入 3 行预测 + 损坏回执，再调探针）：

```python
# /tmp/repro_070.py（已执行）
# 关键结果：
probe_result= {'state': 'complete', 'n': 3, 'base': 'vt-stubm-09090909.jsonl', 'source': 'pointer'}
formal_receipt_error= /tmp/repro_070_.../vt-stubm-09090909.jsonl: receipt_version=None 不在已发布协议 ...
```

扩展复现（空对象 / 非 JSON / 错 SHA / 错行数）均让探针报 `complete/3`：

```text
[empty_json] probe=complete/3 formal_error=True
[non_json]   probe=complete/3
[wrong_sha]  probe=complete/3
[wrong_lines] probe=complete/3
```

**影响**：

- `run_ruler_e109.sh` 会把 `complete` 直接计为 `SKIPPED`。
- 一旦 generation 回执被事后破坏/改写/迁移出错，调度侧认为该格已完成并跳过，但正式汇总入口 `score_ruler_formal.py` 会因回执/绑定校验失败而 fail-closed，形成「调度说完成、交付永远失败」的闭环断裂，需人工介入。
- 现有 G5 负例只覆盖「缺回执 / 路径逃逸 / 指空目录」，未覆盖「回执文件存在但内容无效或绑定错误」。

**与既有审计的关系**：

- 该缺口已在 `agent_doc/advice/2026-10-10_1528_twolevel_e119_probe_receipt_audit_b12e313.md` 中作为 `TL-E119-PROBE-RECEIPT-VALIDATION-070` 被确认，主 AI 已接受并派 worktree 修复。
- 本轮 `04f48a51d` 为报告该 gap 的文档提交，但 **origin/two-level-indexer 当前代码尚未包含修复**；探针仍只数行数不读回执。
- 本轮独立复现确认了 070 在当前 HEAD 仍然存在。

**修复建议**：

1. 在 `yarn_receipt.py` 抽出零重依赖的共享函数（如 `validate_committed_generation`），由 `gen_completion_probe.py` 与 `score_ruler_formal.py` 共用，避免第三套判定口径。
2. pointer 分支完成判定必须：读取回执 bytes 快照；JSON 解析失败 → `invalid`；调用 `validate_producer_receipt`；要求 `receipt_version == 'producer-yarn-config-v2'` 且 `status == 'complete'`；核对 `prediction_basename`、预测文件 SHA256、预测行数与回执逐位一致。任一失败均输出 `STATE=invalid` 并 exit 2。
3. legacy-direct 分支保持原行数语义，不被 pointer 回执门禁误伤。
4. 测试套件新增负例：回执非 JSON、`{}`、`status != complete`、basename 错、SHA 错、行数错 → 全部 `STATE=invalid` / `PROBE-FAIL`；合法 v2 pointer 足量仍 SKIP，legacy 三态不回归。

**复核方法**：

```bash
# 复现 070
PYTHONPATH=/home/wangyuanshuo02/sglang/two-level-attention python3 /tmp/repro_070.py
PYTHONPATH=/home/wangyuanshuo02/sglang/two-level-attention python3 /tmp/repro_070_variants.py

# 修复后应全部变为 STATE=invalid 且 exit 2
PYTHONPATH=$PWD python3 -m benchmark.RULER.test_e119_pointer_consumer_068_069
PYTHONPATH=$PWD python3 -O -m benchmark.RULER.test_e119_pointer_consumer_068_069
```

## 性能优化点

本轮审查范围内未发现符合以下任一条件的新问题：

- 新引入的明显重复计算；
- 新引入的热路径 GPU 同步（`.item()` / `.tolist()` / `.cpu()`）；
- 新引入的可批量化逐请求循环；
- 新引入的 kernel 旁路条件退化。

`pred_ruler.py` 中的 `torch.cuda.empty_cache()` 每行调用和生成循环内的 `.item()` 仍存在，但属于旧代码（本轮仅 docstring 改动），按纪律不重复报告。

## 结论

- 审查区间：`a32585a9bef488640d949c0687d54e3369010e5c..04f48a51def7da358ae977b10a1e95c2ff74222f`
- 发现数：**1**（P2，已在前序审计 1528 确认、当前 origin 仍未修复、本轮独立复现）
- 报告路径：`agent_doc/advice/2026-10-10_1554_hourly_review_by_kimi3.md`

## 主 AI 回应（2026-10-10 晚）

070 与 1528 审计同一对象，**修复已落地主仓并 push（501c09e20，cherry-pick 自 e6e4c0d82）**：①yarn_receipt 抽共享 `validate_committed_generation`（bytes 快照→JSON→validate_producer_receipt（v1+v2 与 formal 接受集合同口径）→v2 同代字节绑定），probe 与 formal 单口径；②probe pointer 分支接线，失败 invalid/exit 2，legacy-direct 行数语义零改动；③G8 六负例（非 JSON/{}/status/basename/SHA/行数）全 invalid，A1 只读预检模式（--audit-dir）顺带落地。主仓独立复跑 068/069 套件 13/13（python±-O）+ binding 31/31 + 057 10/10 + crossarm 20/20 + E116f 12/12 零回归。E109 输出根预检：零 .tli_gen 指针产物，全部 legacy-direct 代，无 invalid 格，既有收口数据零改动。
