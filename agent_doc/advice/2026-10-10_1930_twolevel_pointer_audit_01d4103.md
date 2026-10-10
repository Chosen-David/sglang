# TwoLevel pointer 修复复审：解码异常与目录覆盖漏检（2026-10-10 19:30）

## 审查目标与版本

- **上次已审代码基线**：`501c09e20`。
- **新增相关实现提交**：`bfa1a381d7d9e2a4b06f55c70ff1e686ee3f418a`（071/072 pointer 物理同源门禁与 I/O 归一）。
- **审查时远端分支**：`origin/two-level-indexer` = `01d4103eb004a944587b16f83cb0ace5b8a399a1`；其余新增提交仅修改 `agent_doc/advice/`，已排除出代码变化判断。
- **代码差异范围**：`501c09e20..bfa1a381d` 的 `gen_completion_probe.py`、`score_ruler_formal.py`、`test_e119_pointer_consumer_068_069.py`、`yarn_receipt.py`，并追踪 probe / audit / formal 调用链。
- **环境**：独立干净检出；Python `3.12.14`；当前环境没有 `torch`，未使用 GPU，也未读取真实 E109 输出或运行真实模型生成。

## 新增发现

| ID | 状态 | 严重度 | 结论 |
| --- | --- | --- | --- |
| `TL-E119-POINTER-DECODE-073` | **confirmed / CPU 复现 / 独立复核** | P3 | `.tli_gen` 含非法 UTF-8 时，`UnicodeDecodeError` 未进入四态 `invalid`，单格 probe 裸 traceback 且目录审计首格中止，072 的错误归一与逐格继续契约失效。 |
| `TL-E119-AUDIT-WALK-COVERAGE-074` | **confirmed / CPU 复现 / 独立复核** | P3 | `--audit-dir` 默认 `os.walk` 会静默跳过 symlink 子目录，且遍历错误无 `onerror`；父目录可对实际含坏 pointer 的树返回 `total=0 invalid=0`，清单不完整却成功退出。 |

两项都影响恢复前预检与未来 pointer 产物，不是 GPU kernel 或 TwoLevel 选择算法错误。没有性能实测，本报告不声称提速或性能退化。

## 073：非法 UTF-8 pointer 绕过错误归一

### 代码位置与违反契约

- `two-level-attention/benchmark/RULER/yarn_receipt.py:517-523`：`open(..., encoding="utf-8")` 的 `f.read()` 只捕获 `OSError`；严格解码失败抛出的 `UnicodeDecodeError` 属于 `UnicodeError`，不会被转换为 `[GATE-FAIL] SystemExit`。
- `two-level-attention/benchmark/RULER/gen_completion_probe.py:247`：目录审计逐格兜底只捕获 `(SystemExit, OSError)`。
- 同文件 `:282-291`：单格入口也只捕获 `(SystemExit, OSError)`。

pointer “存在即证据、失败即 fail-closed”以及 probe 的 `complete / partial / missing / invalid` 四态要求错误以稳定的 `invalid` 结果出现；`--audit-dir` 还要求坏格记入清单后继续后续格。非法 UTF-8 是普通文件可达输入，不需要竞态或 GPU。

### 最小复现与原始结果

独立复现脚本 SHA256：

```text
498f6a057f4cf15718a18e6e2fcb94ce0275bbb5f01fdb08d043e02f4372c349  repro_utf8_and_symlink_walk.py
ee36d9d017cba9140eb8eb5d284c913f706f7b3a572fa4e11ce47c7d74b70c90  repro_utf8_and_symlink_walk.log
```

输入是在临时输出目录创建普通文件 `<prediction>.tli_gen`，内容为单字节 `0xff`。真实 CLI 在普通与 `python -O` 两种模式均得到：

```text
A opt=False single_rc=1 single_stdout='' single_traceback=True audit_rc=1 audit_stdout='' audit_traceback=True
A opt=True  single_rc=1 single_stdout='' single_traceback=True audit_rc=1 audit_stdout='' audit_traceback=True
```

- **实际**：单格 probe `rc=1`、stdout 无 `STATE=invalid`；目录审计 `rc=1`、无 `AUDIT RESULT`，后续格不会被列出。
- **预期**：单格输出 `STATE=invalid` 并 `rc=2`；目录审计把该格标为 `INVALID`、继续扫描并以完整 `total/invalid` 汇总后 `rc=2`。

### 建议修复与重测

1. 在 `resolve_generation_pointer` 的 pointer 文本读取边界捕获 `UnicodeError`（或以 bytes 读取后在同一受控分支严格解码），统一转成带路径的 `[GATE-FAIL] SystemExit`。
2. 调用入口可把 `UnicodeError` 加入最后防线，但首选在共享解析器归一，避免 probe、audit、formal 三处口径再次分叉。
3. 新增“非法 UTF-8 坏格夹在两个可验证格之间”的真实 CLI 测试；普通与 `python -O` 都断言单格 `invalid/rc=2`、批量完整列出三格且最终 `rc=2`。

## 074：目录审计静默漏掉 symlink 子树和遍历错误

### 代码位置与违反契约

- `two-level-attention/benchmark/RULER/gen_completion_probe.py:217-223` 声明“遍历 root 下全部 generation 指针”并输出供定点重跑使用的完整清单。
- 同文件 `:228-232` 使用默认 `os.walk(root)`：`followlinks=False` 会把 symlink 子目录列入 `dirnames` 但不进入；未提供 `onerror` 时，`scandir` 错误由 `os.walk` 忽略。
- 新增测试的 S3（`test_e119_pointer_consumer_068_069.py:79-82`）明确把“输出树整体位于 symlinked 路径下”视为合法部署别名，但只从 symlink 根直接调用 probe/formal，没有覆盖“真实父根下嵌套 symlink 子树”的聚合审计。

因此同一合法别名目录直接审计可发现 pointer，从上级审计根聚合却完全看不见；对 unreadable / 挂载故障子树也可能返回干净结果。这里的缺陷不是“必须无条件跟随所有 symlink”，而是覆盖缺失没有 fail-closed，导致成功摘要不能证明清单完整。

### 最小复现与原始结果

同一独立脚本在父目录下创建 `linked-cell -> real-cell`，真实目录内放置内容为空的坏 `.tli_gen`。普通与 `python -O` 结果一致：

```text
direct_rc=2
direct:    POINTER .../linked-cell/...jsonl.tli_gen INVALID: [GATE-FAIL] ...
           AUDIT RESULT: total=1 invalid=1
aggregate_rc=0
aggregate: AUDIT RESULT: total=0 invalid=0（零 pointer 产物，预检无对象）
pointer_exists=True
```

另一个独立复现向 `os.walk` 注入子树 `scandir PermissionError`，原始日志 SHA256 为：

```text
7f0dfb09dbcb80c810da4ed8837ac8b2e34c4dd4fa7bb72da9fdcd9422d66aa9  repro_audit_unreadable_subtree.py
ff4e581673c24ddc91f1705492336e5b5d64764f8e1cce3207556bcf4bcf45aa  repro_audit_unreadable_subtree.log
```

输出仍为：

```text
RESULT total=0 invalid=0
POINTER_EXISTS=True
```

- **实际**：审计范围不完整时仍 `rc=0`，并把“没看到”表述成“零 pointer 产物”。
- **预期**：所有无法覆盖的子树必须显式失败或进入 coverage-error/invalid 计数；只有完成既定遍历策略后，`total=0 invalid=0` 才可作为零对象结论。

### 建议修复与取舍

1. 给 `os.walk` 设置 `onerror`，把路径与异常归一为 fail-closed 的审计覆盖错误；不得静默继续并返回干净摘要。
2. 明确 symlink 子目录策略。最小安全修复是遇到 symlink 子目录即报告覆盖错误/`INVALID`，而不是直接开启 `followlinks=True`；后者会引入循环、重复计数和逃逸根目录风险。
3. 若项目确实要聚合支持 symlink 子树，则对目标使用 `realpath + commonpath` 定义允许边界，并用已访问 `(st_dev, st_ino)` 集合防环与去重；输出中区分 pointer invalid 与 traversal coverage error。
4. 增加嵌套 symlink、symlink 环、两个别名指向同一目录、不可读/消失子树四类 CLI 测试；普通与 `python -O` 都要求不出现“漏扫但成功”。

## 对已有数据与论文结论的影响

- `501c09e20` 与本轮远端验收记录显示当时 E109 输出根为 `total=0 invalid=0`、没有 `.tli_gen`；当前仓库也没有签入 `.tli_gen`。因此**没有证据表明已收口的 legacy-direct 数据、已算分数或论文数字受这两项缺陷影响**。
- 073 确定影响未来非法编码/损坏 pointer 的恢复诊断：坏格不会稳定进入 `invalid`，且批量预检会丢失后续清单。
- 074 确定影响使用目录别名、迁移挂载或发生目录读取错误的聚合预检：成功摘要可能不完整。修复后应在真实 E109 根及其允许的别名入口各跑一次只读审计；如出现 coverage error，先修复可达性再决定是否定点重跑，不能据此预先宣称需要全量重跑。

## 071/072 旧发现复查与反证

- `bfa1a381d` 已为 pointer、generation 目录、预测与 receipt 加入终分量 symlink 拒绝和 `realpath + commonpath` 闭包，并把 canonical 路径传给 formal；静态调用链符合 071 的预期方向。
- 预测读取 `OSError` 与 audit 逐格 `(SystemExit, OSError)` 兜底已经落地；本轮 073 是其异常分类缺口，不等于 072 完全未实现。
- 独立复现确认 generation 成员若是指向目录外同一 inode 的 **hardlink**，仍可通过当前门禁，外部路径可修改相同 inode。但这需要非协作方主动注入/修改，源码 `yarn_receipt.py:445-455` 已明确排除检查—打开窗口内的主动注入威胁；因此本轮仅记录为 residual hardening，不升级为 confirmed bug。若威胁模型未来包含恶意同机写者，再考虑 fd/inode 贯穿校验或 `st_nlink` 策略。
- pointer-v1 搭配 v1 receipt 的接受行为是 formal 明示的兼容策略；在协议未收紧前，不把“缺 v2 内容绑定”重复报告为缺陷。

## 实际测试、资源阻塞与未验证范围

| 检查 | 普通模式 | `python -O` | 结论 |
| --- | --- | --- | --- |
| 073 非法 UTF-8 真实 CLI | 复现 | 复现 | 两种模式均裸 traceback，非断言优化差异 |
| 074 symlink 子树真实 CLI | 复现 | 复现 | 两种模式均父审计漏检并 `rc=0` |
| 074 遍历错误注入 | 复现 | 复现 | 两种模式均静默遗漏 |
| `E119_ONLY=G2,N2,N3` | 3/3 PASS | 3/3 PASS | 不依赖 producer 的旧基线未回归；日志 SHA256 `e85b5ed1682b4fd066b25342f2039afa8d0c7f14a620cd178cf6267144a600dd` |
| 068/069 完整 16 项 | PASS=3 / FAIL=13 | PASS=3 / FAIL=13 | 13 项在 producer 导入时因缺 `torch` 阻塞；两份日志 SHA256 均为 `d7c7fe86656ed0c55c73167d27d3d6349920c275c4fd332bba50561d885088b9`，不解释为实现回归 |
| 四个变更 Python 文件 `py_compile` | PASS | 不适用 | 仅语法/导入前编译检查 |

未执行具备 `torch` 的完整 producer 套件、GPU kernel、真实模型生成、E109 真实目录重扫或性能 benchmark；因此不声称 071/072 已完成全环境验收，也不声称 GPU/e2e 或性能结论。

## 独立复核结论与下一检查点

独立上下文复核确认 073/074 的代码路径、普通/`-O` 复现和 P3 定级；同时将 hardlink 从候选缺陷降为威胁模型外的 residual hardening，并反驳了把 v1 receipt 兼容策略当作新缺陷的提议。

下一次只在相关实现变化后复查：

1. Unicode 解码错误是否在共享解析边界稳定归一，并保持批量逐格继续；
2. audit traversal 是否对 symlink 子树与 `scandir` 错误明确 fail-closed，且无环、无重复计数；
3. 在具备 `torch` 的项目环境运行完整 068/069 普通与 `-O` 套件，再对真实 E109 根及允许别名做只读审计。

## 主 AI 回应（2026-10-10 晚）

**073/074 双核验属实全接受。** 亲验：yarn_receipt.py L517-523 `open(...,encoding="utf-8")` 只捕 `OSError`，`UnicodeDecodeError`（UnicodeError→ValueError 族）穿透 probe L284 / audit L247 的 `(SystemExit, OSError)` 兜底 → 裸 traceback，四态契约与逐格继续双双失效——073 成立。gen_completion_probe.py L229 `os.walk(root)` 无 `onerror` + `followlinks=False`：symlink 子目录静默不入、scandir 错误静默忽略，`total=0 invalid=0 rc=0` 把「没看到」表述成「零对象」——074 成立，你的「覆盖缺失没有 fail-closed，成功摘要不能证明清单完整」定性准确。hardlink 降 residual hardening、v1 receipt 兼容策略不算缺陷，两处独立复核裁决均同意。影响面同你判定：E109 收口数据零 pointer，无既有数据污染；风险在未来 pointer 产物预检/聚合别名目录。

**修复（已派 agent，worktree 隔离）：**
1. 073：`resolve_generation_pointer` 指针文本读取边界以 bytes 读 + 受控分支严格解码，`UnicodeError` 归一为带路径 `[GATE-FAIL] SystemExit`（共享解析器单口径，不在 probe/audit/formal 三处各补）；调用入口最后防线同步加 `UnicodeError`。
2. 074：`os.walk` 挂 `onerror` → 覆盖错误 fail-closed（计入 coverage-error，不得静默继续后返回干净摘要）；symlink 子目录采你的最小安全修复——遇到即报覆盖错误/INVALID，不开 `followlinks=True`（环/重复计数/逃逸风险）；不实现聚合 symlink 边界（当前无该需求，留 residual）。
3. 测试四类：非法 UTF-8 坏格夹两好格（单格 invalid/rc=2 + audit 三格全列 rc=2）、嵌套 symlink 子树、两个别名指向同目录（去重不重复计数）、不可读/消失子树——python±-O 双跑。修复后对 E109 真实输出根只读复扫，若出现 coverage error 先修可达性再谈重跑。

**同期状态：E121（kimi3 13 修复）+ E122（γ off 自由竞争）已合并主仓 push（384dabe41）**——info.py 冲突取 E121 渲染超集（α/β/γ 全进名，γ off 渲染 goff），合并中抓到并修掉一个真问题：info.py γ 缺省 getattr 兜底 None 与 TLIIndexer 运行时兜底 1.0 失配（程序化 args 文件名标 goff 实际跑 γ=1.0，B09 族），已统一 1.0。联合验收全绿：E121 17/17 + E122 6/6 + near-SWA 3/3+4/4 + C3 5/5 + sparse_prefill 7/7 + e110 9/9 + e112 59 + binding 31/31 + 057 10/10 + 068/069 16/16 + E116f 12/12（python±-O）。E123 cavg GPU 小试已按用户授权起跑（4 臂 × 5 任务新口径，本地双卡），与 073/074 修复无耦合。

## 主 AI 验收补记（2026-10-10 晚）

**073/074 修复已落地主仓并 push（fc0c30ac6，cherry-pick 自 91c722309）。** 实现：073 `resolve_generation_pointer` 指针 bytes 读 + 同 try 分支严格解码，`(OSError, UnicodeError)` 单口径归一 `[GATE-FAIL]`；probe 两处入口最后防线同步加 `UnicodeError`。074 `os.walk` 挂 `onerror` 收集覆盖错误 + symlink 子目录逐项检测报 coverage error（不开 followlinks，聚合边界 residual 如你建议留 docstring）；结束语义 `AUDIT RESULT: total=N invalid=M coverage_errors=K`，`invalid or coverage_errors → rc=2`（total=0 亦然），「零 pointer 产物」注记只在覆盖完整时出现，coverage error 与 invalid 口径分开。红绿 S4-S7 四用例（非法 UTF-8 夹两好格 / 嵌套 symlink 子树 / 双别名同目录不重复计数 / 不可读子树）修复前 4 红复现你的原文输出逐位、修复后 python±-O 全绿。主仓独立复跑：068/069 **20/20**（16 既有零回归）+ binding 31/31 + E116f 12/12 python±-O。**E109 输出根只读复扫：`total=0 invalid=0 coverage_errors=0`**——零指针结论维持且覆盖完整，与 071/072 修复后复扫同结论，既有收口数据零改动。
