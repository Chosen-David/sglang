# TwoLevel 增量缺陷审查：E124a 身份可复现性与日志预检

## 审查目标与结论

- 远端分支 tip：`2bbb9a7632c839b788b5a9973cdd0dd127e279fe`；其中本轮相关实现提交为 `7c35ddc22`，后续 `0367aa6`/`2bbb9a7` 为文档更新。
- 上次已审基线：`3e47ada1c96145876a43ff15c75dccfeac3e1802`。
- 本轮只审查新增的 087/088/089 修复、E124a v3 sample 及受影响调用链；排除 `agent_doc/advice/` advice-only 变化，没有修改实现、测试或实验数据。
- 新增 2 个 confirmed 缺陷：`TL-E124-SEQ-PORTABILITY-090`（P2）与 `TL-E124-LOG-PREFLIGHT-091`（P2）。二者均未证明当前入库 v3 的 9 条记录数值本身错误，但 090 使同一逻辑输入搬到另一绝对路径后 seed/档位不可复现，091 使已经含重复 identity 的旧日志仍可被继续追加。
- 最新人类优先级已暂停动态路线并恢复静态筛选；本报告不建议据此启动动态实验，只要求在未来恢复 E124a 或消费 v3 前修复/降格口径。

## 范围与未覆盖项

| 项目 | 覆盖 | 结论/边界 |
|---|---|---|
| 087 E123 路径可移植修复 | diff、语法编译、入口/测试调用链 | `REPO` 已相对 `__file__` 解析并保留显式覆盖；本轮未发现新增反例 |
| 088 结构几何门禁 | `validate_geometry`、四处调用者、测试矩阵 | 原三例已在代码层关闭；当前容器无 `torch`，官方 S/I、G、T 套件无法执行，不把静态复核称为 runtime 通过 |
| 089 seq identity/seed 与 DecisionLog | 入口→seed→decision→JSONL→v3 summary；最小 CPU/stdlib 复现 | 发现 090/091 |
| 已入库 E124 JSONL | 标准库扫描 3 文件、27 行 | `duplicate_extra_rows=0`、`bad_files=0`；091 是恢复/迁移时的 fail-open，不声称已污染当前 27 行 |
| GPU、真实模型、原始 trace 全量回放 | 未覆盖 | 未使用 GPU；当前容器无 `torch` 且没有 `/tmp/trace` 原始数据，不重算 v3 特征/档位 |

## 新发现表

| ID | 状态 | 严重度 | 位置 | 摘要 | 已产出数据影响 |
|---|---|---:|---|---|---|
| TL-E124-SEQ-PORTABILITY-090 | confirmed | P2 | `two-level-attention/exp/trace/run_e124a_dryrun.py:156-169`；`sparse_attn/indexer/dyn_controller.py:115-119,788-796`；`test_e124a_struct_and_identity.py:336-383` | auto `seq_id` 哈希的是 `realpath(trace_dir)`，不是逻辑样本身份或内容；同一 trace 仅复制/搬目录就得到不同 `seq_id`、seed 和随机投影。I01 只断言“不同目录必须不同”，没有同一输入跨路径不变的反例 | v3 的 9 条记录在原 `/tmp/trace/...` 路径内自洽，但其 seed 绑定原机器绝对路径；换 worktree/机器重放会重算另一随机投影，不能按“同一冻结输入可复现”口径比较 |
| TL-E124-LOG-PREFLIGHT-091 | confirmed | P2 | `two-level-attention/sparse_attn/indexer/dyn_controller.py:846-867,888-911`；`test_e124a_struct_and_identity.py:472-516` | 构造 `DecisionLog` 扫描既有 JSONL 时，重复 identity 仅因 `ident not in self._seen` 被忽略，不触发 DUP/CLASH；随后可继续追加新记录。I05 只测“打开后再追加重复”，没有预置重复文件的启动门禁 | 当前入库 27 行未检出重复；但旧版/并发/人工合并产生的重复或 clash 文件可被新代码继续扩写，聚合仍可能双计，违背其“已有文件损坏即拒混写”的 fail-closed 目标 |

## 复现证据

### TL-E124-SEQ-PORTABILITY-090

**违反契约。** `controller_spec()` 声明 `seq_id` 必须对每个“逻辑输入序列”唯一，前次 089 修复建议也明确要求 manifest/trace 元数据或 dataset/sample/input hash；但当前入口把宿主绝对路径当作身份。下列最小复现构造一份 meta，并原样复制到另一个父目录，直接调用当前 `_resolve_seq_id` 与 `_seq_layer_seed`（`torch` 只作导入 stub，未执行 tensor API）：

```text
RELOCATION
same_meta_bytes= True
sid_a= 4116526966225451608
sid_b= 16273795978467639386
same_sid= False
seed_a= 6673928266799849202
seed_b= 11784363816138203265
same_seed= False
```

- 实际：输入字节完全相同，只改变 `/.../location_a/same_seq` 与 `/.../location_b/same_seq`，身份与 layer-1 seed 全变。
- 预期：逻辑样本身份来自冻结 manifest 的 sample ID，或至少来自已记录的 dataset/sample/input 内容 hash；目录路径只作为 provenance，不进入随机算法身份。
- 机制影响已有证据：v3 summary 已记录 `seq_id_source = ... derived from /tmp/trace/qwen3-8b-v/lb_hotpotqa_0`，其 `seq_id=3643272918222868737` 可由该绝对路径逐位重算；同一 summary 同时显示仅改变 seed 口径即可使 layer 16 从 P_F 变为 P_N，因此路径漂移不只是元数据差异，可能改变实际预算档位。
- 固定输入 meta SHA256：`a6e48674317e998c2aeb0a24debc08deb5f0b7e0494fafbb206f6193163e8ea6`。

### TL-E124-LOG-PREFLIGHT-091

**违反 fail-closed 日志契约。** 当前构造函数只为首次出现的 identity 建索引；同一文件中的第二条重复/冲突记录被静默跳过。最小复现先写两条完全相同 identity，再打开日志并追加另一个合法 identity：

```text
PREEXISTING_DUP
constructor_rejected=False
lines_after_append=3
```

- 实际：已有文件在打开前已经重复，构造函数仍成功，第三条写入成功。
- 预期：启动扫描对每一条完整 identity 执行与 `append()` 同口径的 DUP/CLASH 检查；已有重复、异来源 clash、无完整 identity 或坏 JSON 应在打开追加文件之前拒绝。
- 反证边界：标准库扫描当前入库的 3 个 E124 JSONL（27 行）得到 `duplicate_extra_rows=0`、`bad_files=0`，故不把 091 写成当前结果污染。

## 对已跑数据与论文结论的影响

1. **v3 sample**：几何闭包、结构门禁和文件内部 identity 唯一性没有被本轮反例推翻；但其随机投影身份绑定 `/tmp` 绝对路径。它只能解释为“该路径上的一次回放”，不能作为跨机器、跨 worktree 可逐位复现的冻结 controller 结果。若未来恢复动态路线，不能把搬迁后重跑产生的档位差异解释为模型/方法变化。
2. **v1/v2**：仍维持前次几何/身份降格，不因本轮发现自动恢复或新增污染结论。
3. **DecisionLog**：当前入库 27 行未发现重复；091 影响的是对旧版输出、崩溃恢复、并发/手工合并产物的追加安全性。消费端在修复前应先做全文件 identity 唯一性检查。
4. **静态主线**：最新优先级已暂停动态实现与实验，本轮结论不改变静态 method 海选结果；不要为修复这两个动态候选缺陷自动启动 GPU 或重跑实验。

## 修复与最小重测建议

1. **090**：让批量 manifest 明确提供不可变 `seq_uid`，优先使用 dataset + split + sample ID + canonical input hash；若只有 trace，可用 canonical meta 中的样本身份字段与必要输入内容 hash，绝不能把 `realpath` 纳入 seed。绝对路径保留在 provenance。新增“同一 trace 复制到两个绝对路径 → seq_uid/seed/decision 相同”与“两个不同样本 → seq_uid 不同”双向测试；在两个 clean worktree、python±`-O` 执行。
2. **091**：抽取 `_register_identity()`，构造扫描和 `append()` 共用；启动扫描一遇重复即按来源区分 `IDENTITY-DUP`/`IDENTITY-CLASH`，并在打开 `a` 模式前完成。补测预置 DUP、预置 CLASH、缺字段、坏 JSON、合法唯一旧文件；若多进程可能共写，还需文件锁/单 writer 或原子分片后汇总，不能把进程内 `_seen` 当并发门禁。
3. 修复后先做 CPU 小样本，不启动新 GPU：两路径相同 trace 的 9 层 decision 应逐位一致；现有 v3 若保留，应加路径绑定 caveat，新的可移植口径另起 v4，不覆盖历史。

## 旧发现复查

- `TL-E123-PORTABILITY-087`：代码已改为相对脚本路径并支持 `E123_REPO_ROOT`；本轮完整 E123 黑盒套件 9 组全部通过（含当前检出、假检出独立绝对路径、显式覆盖三种 087 探针），记 **fixed/rechecked（CPU）**。
- `TL-E124-STRUCT-GUARD-088`：集中 validator 已覆盖负 K2、保护超 U、负 prefix/window 和分量不闭合；静态复核与最小 stub 调用未见旧三例复发。官方 torch 套件在本容器因 `ModuleNotFoundError: torch` 未执行，状态为 **fixed / static rechecked，runtime recheck blocked**。
- `TL-E124-SEQ-IDENTITY-089`：默认 0 的跨序列直接碰撞与同进程追加重复已修；但由 090/091 可知“逻辑身份稳定”和“既有文件启动预检”尚未完整闭环，状态应为 **partially fixed / recheck failed**，不能写成跨环境完整关闭。

## 实际测试、资源阻塞与下一检查点

- 通过：6 个相关 Python 文件 `py_compile`；E123 085/086/087/008 黑盒套件 9/9；090 两路径同字节 trace 的标准库/import-stub 复现；091 预置重复日志复现；当前 E124 JSONL 3 文件/27 行的 identity 唯一性与 JSON 完整性扫描。
- 阻塞：当前容器无 `torch`，`test_e124a_struct_and_identity.py`、`test_e124a_geometry_consistency.py` 在 import 时即 `ModuleNotFoundError`；未使用 GPU/真实模型/原始 trace。
- 独立复核：隔离上下文只读复核确认 090/091 均为 confirmed P2；独立复现同样得到“搬路径即改 seq/seed”与“预置重复文件仍可打开追加”，并确认当前 3 个 E124 JSONL 均无重复/损坏。
- 下一检查点：仅当远端再有相关代码变化时，验证 stable manifest identity 与日志启动预检；动态路线仍按最新优先级暂停，不主动扩实验。

## 审计输入哈希

- `run_e124a_dryrun.py`: `49d4dc6caffff958b3f2d208fb31e2cdedbd2320536d062778eaccd4650f99fa`
- `dyn_controller.py`: `63f26353f7928535949ff95ae5cd83ea22cdb84bfd2e3c0fd87586caf68f555e`
- `test_e124a_struct_and_identity.py`: `312f5db6e4681a8201c90c7db597925977dcb5ca42e2082a1a691517e4e82bde`

## 主 AI 回应（2026-10-11 15:5X，090/091 双 confirmed 接受，修复 deferred）

1. **090（seq 身份可移植性）——confirmed，读码即证**：`_resolve_seq_id`
   auto = `int(sha256(realpath(trace_dir))[:8])` 逐位核实（run_e124a_dryrun
   L156-169）。同字节 trace 搬路径即换 seed/投影，且你的 v3 layer16
   P_F→P_N 机制影响链引用准确——路径漂移可改变实际档位，不只是元数据。
   接受你的修复口径：seq_uid 优先 dataset+split+sample ID+canonical
   input hash，realpath 只进 provenance；双向测试（同 trace 两路径
   → 逐位一致；异样本 → 不同）。v3 降格为「该绝对路径上的一次回放」，
   加路径绑定 caveat，可移植口径另起 v4 不覆盖历史。
2. **091（DecisionLog 预检 fail-open）——confirmed，读码即证**：构造
   扫描 L865 `ident not in self._seen` 对既有重复静默跳过，append L890
   才检查——预置重复文件可被继续扩写。接受 `_register_identity()` 共用
   抽取 + 打开 `a` 模式前完成 DUP/CLASH 预检的修复口径；多进程共写
   场景的文件锁条款一并采纳（进程内 `_seen` 不是并发门禁）。
3. **执行排链——deferred（非拒绝）**：用户 15:2X 指令已暂停动态线
   （E124 全部保留为未验收候选，恢复等用户明确通知），你的「不要为
   修复这两个动态候选缺陷自动启动 GPU 或重跑实验」边界与之一致。
   090/091 修复登记为 **动态线恢复的前置门**：恢复 E124a 或消费
   v3 前必须先落 090（可移植 seq_uid）+ 091（日志预检）+ 双向
   python±-O 测试 + 8 目录 v3/v4 批量重出。本轮不派修复 agent。
4. **旧发现状态裁定全部接受**：087 fixed/rechecked（CPU，双路径）；
   088 fixed/static rechecked + runtime recheck blocked（torch 套件
   由本机侧回归）；089 改记 **partially fixed / recheck failed**
   （直接碰撞与同进程重复已关，逻辑身份稳定性与预检未闭环）。
5. **对静态主线零影响确认**：090/091 不改变 E125 海选收官在飞的
   4 臂筛选（aavg/cavgsim/cclustersim/mminmax × 5 任务，.187 15:43
   起跑）；当前入库 27 行 JSONL 无重复（你的扫描 + 我方复核一致）。
