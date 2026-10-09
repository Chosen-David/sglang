# TwoLevel RULER 正式结果发布事务审计（2026-10-09 12:28）

## 审查目标、版本与范围

- **目标分支 / 远端审查头**：`two-level-indexer` / `4de147c1b39c38d2950e55712127fd05f6dd2002`。
- **新增实现来源**：本轮相对上一份审计提交 `fe4af6696daafb4af52ddbca4cddd7d580f6ac6f` 检查了 E116e RULER 正式评分、LongBench-v2 统一解析/重评分及 E117a W_O 投影分析；本文两个发现均定位到 `c64489996988e83a423e1a65ff609edde4627982` 引入的正式发布路径。
- **核心文件**：`two-level-attention/benchmark/RULER/score_ruler_formal.py`，SHA256 `c6b70a27601143c9faa8bd22218f3cf6a79ba2d47633cf17893db92665b88e43`。
- **测试文件**：`two-level-attention/benchmark/RULER/test_e116e_gate.py`，SHA256 `28acfee3926eeb8be724ad372a802ab67e7163c347c8ee439578779c64ba7013`。
- **入库 fixture 树**：`benchmark/RULER/testdata/e116e/` 共 33 个文件，逐路径+逐文件 SHA256 的规范清单哈希为 `1dd39f25be7b40c3b46b3e890cefa5c096dcdc2a884795808d11fda73fc5eb19`。
- **环境**：Python 3.12.14；CPU 文件系统故障复现，无 GPU、无模型推理、无真实 32K 原始预测重跑。

本轮先按文档命令实际执行 E116e 套件：`PYTHONPATH=$PWD python -m benchmark.RULER.test_e116e_gate`，结果 **12/13 PASS，D9 因缺真实数据显式 SKIP**。该套件覆盖发布前的 `SystemExit`，但没有覆盖发布阶段第 N 个 `os.replace` 或最后 `os.rename` 失败。

## 新发现

| ID | 状态 | 严重度 | 位置 | 结论 |
|---|---|---:|---|---|
| `TL-RULER-PUBLISH-ATOMICITY-038` | **confirmed（真实跨文件系统 EXDEV + 独立故障注入复核）** | P1（结果/清单/receipt 可混合代际） | `score_ruler_formal.py:434-448` | 四个公开文件依次 `os.replace` 不是跨文件事务；异常处理只捕获 `SystemExit`。第三次替换失败时，新 JSON/MD 已覆盖旧成功产物，旧 manifest/receipt 仍保留，无 failure receipt，直接推翻“失败不触碰旧成功产物、混合代际不可达”的契约。 |
| `TL-RULER-DERIVED-COMMIT-039` | **confirmed（独立故障注入复核）** | P1（成功 receipt 可引用不存在的证据目录） | `score_ruler_formal.py:410-415,442-448` | `status=success` receipt 在第 442 行公开，第 443 行才把 staging 改名为 receipt 声明的 `derived_dir`。最后一次 rename 失败时，成功 receipt 已对外可见但目录不存在，也没有 failure receipt。 |

## 违反的契约与可达触发

E116e 文件头和测试 E4/E5 声明：失败重跑不覆盖旧成功产物，receipt 最后发布作为提交信号，混合代际状态不可达。当前顺序实际为：

1. 覆盖公开 JSON；
2. 覆盖公开 Markdown；
3. 覆盖 manifest；
4. 覆盖 success receipt；
5. staging 改名为 `derived_dir`。

单个文件的 `os.replace` 只能保证该文件替换原子，不能保证步骤 1-4 整组原子。现实中的稳定触发不需要杀进程：`--manifest-out` 接受任意路径，若 `--out` 位于 overlay/ext4 而 manifest 位于另一挂载点（例如 tmpfs），第 3 步会得到 `EXDEV`。磁盘满、权限变化、I/O 错误或进程中断也能落入同一窗口。第 5 步的 rename 还会在 receipt 已提交后失败。

## 最小可运行复现与原始结果

### 038：不使用 monkeypatch 的跨文件系统复现

1. 把入库 `pred_root` 复制到临时目录，以默认 manifest 路径成功跑一次，形成一组旧成功产物。
2. 只把临时副本 `vt-fxm-01010000.jsonl` 第一行 `pred` 改成 `wrong`，使第二轮结果可辨；业务身份字段不改。
3. 用同一个 `--out` 重跑，但增加 `--manifest-out /dev/shm/<unique>-manifest.json`。

复现输入树哈希为 `d94ca82e12ce1c3b9df199cfb4e3c79e0170c77f035d5d0a49ed638502ffbeec`；参数/变更描述的规范 JSON 哈希为 `c55ea4d60646e23c5232b0c7bb374fc07bd48dba970b3619b0793c3094fa7a6d`。实际关键输出：

```json
{
  "first_rc": 0,
  "second_rc": 1,
  "exception": "OSError: [Errno 18] Invalid cross-device link (.../staging.../manifest.json -> /dev/shm/...-manifest.json)",
  "result_changed": true,
  "old_receipt_result_hash_matches_after": false,
  "receipt_run_id_unchanged": true,
  "custom_manifest_exists": false,
  "failure_receipts": 0,
  "staging_dirs": 1
}
```

这说明第二轮虽退出失败，却已经把公开 JSON 换成新代际；对外仍存在的旧 receipt 的 `result_sha256` 不再匹配 JSON。失败路径 `except SystemExit` 没有执行。

另用入库 fixture 对第二个 staging→公开 `os.replace` 注入 `OSError`，得到同根因结果：只有 JSON SHA 改变，MD/manifest/receipt SHA 保持旧值，failure receipt 数为 0，staging 残留 1。独立只读审查者在不同上下文复现了相同状态。

### 039：derived_dir 提交顺序复现

独立只读审查者只对最终 `os.rename(staging, run_dir)` 注入 `OSError`，其余流程正常执行。观察结果：

```json
{
  "exception": "OSError: INJECTED generation rename failure",
  "json_md_manifest_receipt_all_exist": true,
  "receipt_status": "success",
  "receipt_derived_dir_exists": false,
  "failure_receipts": 0,
  "staging_dirs": 1
}
```

因此“receipt 最后=提交信号”仍早于它所承诺的派生证据目录，消费者即使严格只信 success receipt，也可能取得不可闭合的证据集。

## 对已跑数据与论文结论的影响

- **缺陷确认，但当前已提交三臂数值未见实际污染。** 本轮独立核对仓库中的 `e116e_ruler32_formal_{fullkv,mavg,aavg}`：三份 receipt 均为 `status=success`，其 `result_sha256` 与当前 JSON 一致，`manifest_sha256` 与当前 manifest 一致。故没有证据撤回已保存的 `59.38 / 59.99 / 57.33`；本报告针对未来重跑、失败恢复和消费者一致性。
- LongBench-v2 修复的纯解析表驱动部分已在本机通过；完整第二段导入 `eval.py` 受当前环境缺少 `jieba` 阻塞，未冒充全套通过。
- E117a 本轮完成源码与已提交结果静态核对；真实 trace、模型权重和 GPU 不在当前工作区，未重跑 9103 秒的 CPU 分析，也不据此重判其 `NO-GO` 数值。

## 建议修复与最小重测

1. 不再把多个固定公开文件的逐个 `os.replace` 称为事务原子。把 JSON、MD、manifest、receipt 所需派生物全部写入同一不可变 generation 目录；目录内容完成、fsync/校验后，先在同一文件系统原子 rename 成最终 generation。
2. 最后只原子替换一个很小的 current pointer/commit receipt；消费者从该指针解析同一 generation 内的全部文件，不跨固定别名拼装。receipt 声明的 `derived_dir` 必须在提交前已经存在并校验。
3. 若支持自定义 `--manifest-out`，发布前验证所有需要 rename/replace 的源与目标位于同一文件系统；跨设备时应先在目标文件系统写临时文件再原子替换，或明确拒绝。仅扩大 `except` 到 `OSError` 不能回滚已经覆盖的前几项，不能恢复强契约。
4. 增加两条故障注入回归：
   - 第 1/2/3/4 个公开替换分别失败，断言旧 commit generation 仍完整可读、没有旧 receipt 配新结果、没有 success 提交；
   - generation rename 失败，断言不会发布 success pointer/receipt，不存在指向缺失 `derived_dir` 的提交。
5. 修复后再用入库 fixture 跑 E1-E10，并额外跑真实跨文件系统 `--manifest-out` 负例；在真实 32K 数据可用时只读重跑三臂，核对数值与新 generation 的 result/manifest/receipt hash 闭合。

## 旧发现复查、未覆盖项与下一检查点

- `TL-LBV2-PARSER-037` 的统一解析、测试和 503×3 重评分文件已出现在本轮远端；本报告不重复该已修缺陷。
- `TL-RULER-*-030..036` 的常规红绿套件本机按文档入口通过 12/13（D9 SKIP）；038/039 是发布窗口的新根因，现有 E4/E5 只覆盖发布前失败，不能作为反证。
- 未覆盖 GPU kernel 性能、真实模型精度、E117a 原始 trace/W_O 重新计算及 64K/128K 生产数据；这些资源缺口不影响本文两个 CPU 文件事务反例。
- 下一检查点：复验 generation + 单指针提交协议及两类故障注入测试；同时确认消费者只从已提交 generation 读取，不再把固定路径存在当作本轮成功。
