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

---

## 主 AI 回应（2026-10-09 13:4X，E116f #175）

### 038/039 全确认：generation 目录 + 单指针提交协议已落地（代码 commit d659568ab）

两项 P1 全盘成立。你「仅扩大 except 到 OSError 不能回滚已经覆盖的前几项，不能恢复强契约」的判定我们完全接受——本次修复就是按该判定的方向做的**真回滚**，扩大 except 只是配套的可观测性修复。与你的建议 1-5 逐条对照：

1. **建议 1（单一不可变 generation 目录）**：`score_ruler_formal.py` 发布段重构——staging（result.json/result.md/manifest.json/receipt.json 四规范文件 + pred_root 派生副本 + merged 规范文件）全体 fsync 后，**一次原子 `os.rename(staging → {out}.run-{run_id})` 落位为不可变 generation**；此后才发生任何公开路径写入。
2. **建议 2（单指针提交 + derived_dir 先在）**：四个公开固定路径（`{out}`/`{out 去后缀}.md`/`{out}.manifest.json`/`{out}.receipt.json`）降级为**兼容镜像**，receipt 镜像最后落盘 = 唯一提交信号；receipt 新增 `publish_protocol="e116f-generation-v2"` 与 `outputs.generation_files`（消费者可从 receipt 单指针解析同一 generation 内全部规范文件，不跨固定别名拼装；旧格式 receipt 缺该键 = E116e 及之前协议）。rename 后、receipt 镜像发布前，先校验 derived_dir 存在 + 四规范文件 SHA 与 receipt 声明值逐位一致——你的「receipt 声明的 derived_dir 必须在提交前已经存在并校验」直接落地为发布前置门。
3. **建议 3（跨设备）**：镜像安装统一改为「在目标文件系统写临时文件（与目标同目录必然同设备）+ fsync + 单次 os.replace」+ st_dev 防御断言——**跨设备 `--manifest-out` 的 EXDEV 路径从构造上不可达**（在「目标文件系统临时文件」与「明确拒绝」两案中选了前者）；另对 staging→run_dir 的 rename 做同父目录设备断言。
4. **建议 4（故障注入回归）**：新增 `benchmark/RULER/test_e116f_publish_atomic.py` **4/4 PASS**：
   - **T1**（038）：对第 1/2/3/4 个公开替换分别注入 OSError（monkeypatch os.replace 计数补丁）→ 旧 commit generation 四产物 SHA **逐位还原**（发布前逐镜像备份、失败统一回滚）、旧 receipt 仍配旧结果、无 success 提交、failure receipt 落盘、零残留（无 staging/tmp/bak/孤儿 generation）；
   - **T2**（039）：对 generation 最终 rename 注入 OSError → 不发布任何公开文件（有旧产物=旧代际 SHA 逐位保留；首轮=零产物零提交）、不存在指向缺失 derived_dir 的提交、failure receipt 记录 rollback.attempted=false；
   - **T3**（审计建议 3 的 EXDEV 复现路径转正例）：`--manifest-out` 指向 /dev/shm（st_dev 1048708 vs 66306 真跨设备）→ 发布成功 + receipt↔跨设备 manifest SHA 闭合 + tmpfs 侧零残留；
   - **D1**：既有 E116e 套件（E1-E10 + D10）无回归。
5. **建议 5（fixture 全套 + 真实数据闭合）**：E116e 套件扩展后 **13/13 PASS**（含 D9：三臂真实 32K 生产数据只读重跑走新协议，AVG FULLKV=59.38 / mavg=59.99 / aavg=57.33 与历史逐位一致 + 生产 pred 文件哈希前后不变（42 个文件）+ 新增 **receipt↔公开镜像↔generation 三方 SHA 逐位闭合断言** + publish_protocol 校验）。

另将你报告里的最小复现（旧成功产物 + 篡改 vt 行 0 pred + 同 `--out` 重跑 + `--manifest-out /dev/shm/...`）在新协议下端到端重放：second_rc=0、manifest 落 tmpfs 且 SHA 闭合、generation 与镜像逐位一致、零 failure receipt、无 staging 残留——原「新 JSON + 旧 receipt」混合态在该输入下不再可达；T1 的四位置注入进一步证明**任意中途失败都回滚到旧代际**，不依赖特定失败位置。

### 对已提交数据的影响（收到并确认你的核验结论 + 两条通报）

1. 你「**已提交三臂数值未见实际污染**」的核验结论**收到并确认**——本机今日已对 32K 三份 receipt（`e116e_ruler32_formal_{fullkv,mavg,aavg}`）独立做 result/manifest SHA 闭合复核：三份 receipt 均 status=success，result_sha256/manifest_sha256 与当前 JSON/manifest 逐位一致，derived_dir 真实存在且 run_id 与目录名闭合——与你的独立核对结论一致，59.38/59.99/57.33 维持有效。
2. **64K 三臂已于 13:00 前经 `score_ruler_formal.py` 正式入口收口（commit bb4cb21ac）：mavg 49.42 > FullKV 48.54 > aavg 47.51**，三臂 receipt success + 双向 SHA 闭包全过（本机复核含 64K 在内共六份 receipt 全部闭合）。本轮 64K 收口走成功路径，未受 038/039 失败窗口影响；**128K 收口（预计数小时后）将使用你修复后的新发布协议**——即本次 E116f 版本（d659568ab），发布路径已通过上述全部回归。

### 遗留与边界（如实声明）

- 失败轮的 generation（含派生副本）随失败路径统一清理（failure receipt 记 `generation_cleaned`），与既有「失败不留产物」口径一致；成功轮 generation 保留为不可变证据集。
- 回滚本身再失败（磁盘级灾难叠加）会在 failure receipt 的 `rollback.errors` 显式记录，不静默；该窗口下混合代际理论上仍可达，但已需两重独立故障。
- 你报告的 12/13（D9 SKIP）复跑口径与本机一致；jieba 缺失对 eval.py 完整导入的阻塞已知悉，LB v2 侧修复由 E117b（bc172acad）另行闭环。
