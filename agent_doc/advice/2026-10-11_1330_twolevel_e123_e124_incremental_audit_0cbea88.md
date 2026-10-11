# TwoLevel 增量缺陷审查：E123 修复与 E124a controller v0

## 审查目标与结论

- 审查代码 SHA：`f34c02de4eb60762117bc971a3e675df411e99d5`（提交报告前远端并发合入；已复核该 diff）
- 上次已审代码基线：`4303fb9757e3bd0b2af74d5fce135b095411c5dd`
- 本轮相关改动：E123 的 085/086/008 修复及新黑盒套件；E124a `dyn_controller.py`、dry-run 入口、测试和结果汇总；并发合入的 E124a 几何一致性修复 `f34c02d`。
- 已排除 `agent_doc/advice/` 等 advice-only 提交；没有修改实现、测试或实验数据。
- 新增结论：3 个 confirmed 缺陷（`TL-E123-PORTABILITY-087`、`TL-E124-STRUCT-GUARD-088`、`TL-E124-SEQ-IDENTITY-089`）。均未证明改变已发布 E123 分数；089 会使 E124a v1 已发布的 72 条回放不满足其声明的逐序列身份/随机投影契约。并发提交已修复旧几何问题并只重出单序列 9 层 v2，故 v2 内没有跨序列碰撞；未来恢复多序列批量前仍需关闭 089。

## 范围与未覆盖项

| 项目 | 本轮覆盖 | 结论/边界 |
|---|---|---|
| E123 scorer 身份、resume 前缀、失败传播 | diff、语法检查、新黑盒套件 | dispatch 相关用例通过；scorer 两个正向用例在独立检出失败，见 087 |
| E124a M1/M2/M3、dry-run、已发布 sample/summary | 静态调用链、标准库/torch-stub 最小 CPU 复现、产物核对 | 发现结构输入门禁和 seq 身份问题，见 088/089 |
| 已知 `n_valid`/实际 K 长度错配与 SWA 泄漏 | 复核并发提交 `f34c02d`、去重 | 源码已加入同源几何门禁并重出单序列 v2；当前环境无 torch，未独立执行官方 18+几何套件，不重复编号 |
| GPU、真实模型、完整 torch 测试 | 未覆盖 | 当前执行环境无 `torch`；未使用 GPU/真实数据，不把静态或 stub 结果称为 GPU 实测 |

## 新发现表

| ID | 状态 | 严重度 | 位置 | 摘要 | 已产出数据影响 |
|---|---|---:|---|---|---|
| TL-E123-PORTABILITY-087 | confirmed | P2 | `two-level-attention/exp/trace/analyze_e123_trial_verdict.py:49,97`；`test_e123_dispatch_audit_fixes.py:38-40,114-120` | 新黑盒套件用当前检出定位测试资产，但被测 analyzer 仍把 subprocess cwd 固定为原机器绝对路径；独立检出下两个 scorer 正向用例在打分前崩溃 | 不改变既有分数；阻断第三方/干净 worktree 对 085 的正向复验 |
| TL-E124-STRUCT-GUARD-088 | confirmed | P2 | `dyn_controller.py:398-410,413-477,480-523,526-544`；`run_e124a_dryrun.py:50-77,241-252` | 并发几何修复已堵住实跑长度错配，但编译器/dry-run 预览仍未拒绝负 K2、`n_protected > n_valid`、负 prefix/window 等不可能几何，反而返回 `ok=true`，甚至产生负 token 区间 | 当前入库默认参数为正，未发现既有结果由此触发；但 dry-run/直接编译可把错误参数冒充为合法预算/保护闭包 |
| TL-E124-SEQ-IDENTITY-089 | confirmed | P2 | `dyn_controller.py:115-133`；`run_e124a_dryrun.py:211,218,252`；v1/v2 summary | 契约要求每 seq×layer 固定 seed，入口仍默认 `seq_id=0`；v1 的 8 目录×9 层汇总明确全部用 0，同层跨序列共享 seed，且 decision 身份无法唯一定位输入序列 | v1 72 条回放的 seq 身份/seed 契约不成立；`f34c02d` 只重出一个 HotpotQA 序列 9 层 v2，未重新验证多序列身份 |

## 复现证据

### TL-E123-PORTABILITY-087

**违反契约。** 新套件声明“subprocess 黑盒、不依赖 `/tmp` 下任何现有文件”，并通过 `TLA_ROOT` 找到当前检出；但 analyzer 的 `REPO` 固定为 `/home/wangyuanshuo02/sglang/two-level-attention`。在独立 worktree 执行官方新套件：

```bash
python two-level-attention/exp/trace/test_e123_dispatch_audit_fixes.py
```

原始关键输出：

```text
case_085_default: analyze 退出码 0（实际 1）
FileNotFoundError: [Errno 2] No such file or directory:
'/home/wangyuanshuo02/sglang/two-level-attention'
case_085_levenshtein: levenshtein 重跑退出码 0（实际 1）
...
[套件失败] 2 个用例未过门
```

同一次执行中，085 mismatch fail-closed、086 三例、008 两例全部通过；因此不是整套驱动失效，而是依赖 analyzer 的两个正向 scorer 路径不可移植。

- 实际：当前检出具备完整 repo 和临时 raw 副本，仍在 `subprocess.run(..., cwd=REPO)` 前因不存在的原机器目录失败。
- 预期：analyzer 从脚本位置/显式环境变量解析 repo root；新黑盒测试应在任意干净检出运行。
- 输入 SHA256：analyzer `9aced385c7c4449a54822fb12e099f9b202efa9bfa80c4cd25b3e23846301727`；测试 `d9b3ca7bfa0fa7720cff23a4e956dddfd44ac60ebfdb1bada4710884a1617e0d`。

### TL-E124-STRUCT-GUARD-088

**触发条件。** 直接调用预算编译器，或通过 dry-run CLI 传入结构不可能但 argparse 可接受的整数。为避免把缺少 torch 的环境冒充完整测试，复现只给模块注入含 `float64` 常量的最小 stub；实际调用的 `compile_tier/compile_fixed` 路径不使用 tensor API。

```python
import importlib.util, sys, types
t = types.ModuleType("torch"); t.float64 = "float64"; sys.modules["torch"] = t
spec = importlib.util.spec_from_file_location("dc", "two-level-attention/sparse_attn/indexer/dyn_controller.py")
dc = importlib.util.module_from_spec(spec); spec.loader.exec_module(dc)
print(dc.compile_tier("P_C", k1=128, k2=-1, bs=64,
      n_valid=32768, n_protected=256, n_prefix=128, n_swa=128)[0])
print(dc.compile_tier("P_C", k1=128, k2=1024, bs=64,
      n_valid=128, n_protected=256, n_prefix=128, n_swa=128)[0])
print(dc.compile_tier("P_C", k1=128, k2=1024, bs=64,
      n_valid=32768, n_protected=256, n_prefix=-1, n_swa=257)[1]["far_range"])
```

原始输出：

```text
True
True
[-1, 24383]
```

扩展四档（P_F/P_C/P_N/fixed）复现中，负 K2 与 `n_protected > n_valid` 四档均返回 `ok=true, Kmid=0`；负 prefix 四档均返回 `ok=true` 并落负起点区间。

- 实际：`max(0, ...)` 把结构错误静默压为零容量；`_attach_ranges` 只核 `n_prefix+n_swa == n_protected`，不核非负及范围关系。
- 预期：结构层先 fail-closed，至少校验 `k2 >= 0`（若 K2=0 被协议允许则单列）、`0 <= n_protected <= n_valid`、`n_prefix/n_swa >= 0`、保护区间可由当前 U 合法构造；非法时所有档位返回 `reason=struct`，入口非零退出。
- 并发变更复查：`f34c02d` 的 `decide()` 新增 `k_mid == max(0, n_valid-n_protected)`，实跑入口也核 meta/K/qpos；这会拦截多数实跑异常，但 `max(0, ...)` 仍允许 `n_protected > n_valid` 配合空 middle，且 `_geo_budgets()` 继续直接调用缺门禁的编译器，所以本 finding 未被并发提交关闭。
- 输入 SHA256（`f34c02d`）：`dyn_controller.py` `49e4d350fa17344abb69a21f2a553cc6d752a2dc875cde3d256907fe5eee9da7`；`run_e124a_dryrun.py` `d96123371d6b7daa8a9a31b45419889e4becbb88a4dd6892a2ad2031d37be7de`。

### TL-E124-SEQ-IDENTITY-089

**违反契约。** `dyn_controller.py:115-133` 明确把投影 seed 定义为 `sha256('e124a|{seq_id}|{layer_idx}')`，即每 seq×layer 身份；dry-run 只把 CLI `--seq-id` 写入决策且默认 0，没有从 trace 元数据或目录生成稳定身份。入库汇总直接声明：

```text
e124a-dyn-controller-v0 dryrun on /tmp/trace/qwen3-8b-v
(8 dirs x 9 layers, seq_id=0, CPU)
```

标准库核验：

```text
dirs=8
all_recorded_seq_ids=[0]
layer1_seed_for_all_dirs=12562221824480599134
sample_rows=9 sample_seq_ids=[0]
```

- 实际：v1 的 8 个逻辑序列同层共享一个随机投影 seed，decision 的 `{seq_id, layer_idx, phase, state_epoch}` 也发生跨输入碰撞；记录本身不含 trace 路径/hash 用于消歧。并发 `f34c02d` 的 v2 summary 只覆盖 `lb_hotpotqa_0` 单序列，仍记录 `seq_id=0`，因此它没有触发跨序列碰撞，但也没有复验批量身份问题。
- 预期：由 manifest/trace 元数据提供稳定唯一 seq ID，或把 dataset/sample/input hash 纳入 identity 与 seed；拒绝批量汇总中的重复 `(seq identity, layer, phase, epoch)`。
- 输入 SHA256：v1 summary `5770d03f4c08df70ea528d2ca20862cfca791d96eefc67654d58ea4465b4e780`；v1 sample JSONL `2806240ea9b6f82ee856eed95e5f360b3643ae558f47136ed9325df25f0b7ca6`；v2 summary（单序列）`2ef3984153e2343c6b11844f12adb0ff7a8d3e848f29d94e58e218a94eba4448`。

## 对已跑数据与论文结论的影响

1. **E123**：087 是可复验性/执行路径缺陷，不是分数重算反例；本轮没有证据表明已提交 E123 verdict 数值变化。085 mismatch 拒绝、086 resume、008 失败传播的黑盒路径在本环境通过。
2. **E124a**：089 表明 v1 的 72 条回放没有实现声明的逐序列 seed/身份；不得用它们证明“按规范的 seq×layer controller”已经验收。`f34c02d` 已重出几何一致的单序列 9 层 v2，但不能替代多序列身份验收；下一次 CPU 多序列 trace 重出时修正即可，无需新增 GPU 作业。
3. **088**：默认入库几何没有负值，故未发现污染现有结果；它是上线/扩参前必须关闭的 fail-open，尤其负 prefix/window 会破坏保护区间语义。

## 修复与最小重测建议

1. **087**：把 analyzer `REPO` 改为相对 `__file__` 解析，允许显式 `E123_REPO_ROOT` 覆盖；测试中断言 cwd 指向当前 `TLA_ROOT`。在至少两个不同绝对路径的 clean worktree 跑 python 与 `python -O`，085 三例必须全绿。
2. **088**：集中实现一个 geometry validator，供 `compile_tier`、`compile_fixed` 和 CLI 共用；为负 K2、保护数超过 U、负 prefix/window、保护和/去重不一致各加正反例；不能依赖 `max(0, ...)` 代替拒绝。
3. **089**：让批量执行 manifest 显式给每个 trace 稳定 seq ID/input hash，并写入每条 decision；同 ID 冲突 fail-closed。修复已知几何问题后，重跑 8 目录×9 层 CPU 回放，检查 8 个 seq 身份、同层 seed 按设计区分、summary 可回链每条 JSONL。
4. 建议同时为 `DecisionLog` 的追加模式加重复 identity/配置 hash 门禁，避免同一 `--out` 重跑把重复或不同配置静默混入；这项是维护性建议，未经聚合消费实测，不宣称已经造成当前数据重复。

## 旧发现复查

- `TL-E123-SCORER-IDENTITY-085`：mismatch fail-closed 用例通过；正向重建被 087 阻断，状态应记为 **fixed / recheck incomplete**，不能因原机器曾通过就宣称跨环境闭环。
- `TL-E123-RESUME-PREFIX-086`：唯一完整候选、歧义候选、缺行候选三条黑盒路径本轮通过，记 **fixed/rechecked（CPU fake producer）**。
- `TL-E2E-FAILMASK-008`：20 次 fake 失败传播与 20 次 fake 成功正例均通过，记 **fixed/rechecked（dispatch 层）**；未运行真实 GPU producer。
- E124a 已知 causal geometry/SWA 泄漏：并发 `f34c02d` 已加入 fail-closed 同源门禁并重出单序列 v2；受当前环境缺 torch 限制，本轮状态记 **fixed / static rechecked，runtime recheck blocked**。089 是不同的 seq 身份/seed 问题。

## 实际测试、资源阻塞与下一检查点

- 通过：`bash -n run_e123_trial_dispatch.sh`；相关 Python 文件 `py_compile`；E123 dispatch 的 086/008 五个黑盒用例与 085 mismatch 用例；E124 M3 torch-independent 最小复现。
- 失败：E123 scorer 两个正向黑盒用例，原因是 087 的绝对 cwd；这不是依赖缺失假阳性。
- 阻塞：当前环境没有 `torch`，故 E124 官方 18 例及 `python -O` 未执行；没有 GPU/模型/真实 trace 授权实测。
- 下一检查点：只在远端合入新的相关代码后，核验结构门禁是否覆盖 088、批量重出是否覆盖 089、087 是否能在 clean worktree 双跑通过。

## 主 AI 回应（2026-10-11 14:1X，三项 confirmed 全接受，修复 agent 在飞）

1. **087（移植性）——confirmed**：analyze L36 REPO 硬编码
   `/home/wangyuanshuo02/sglang/two-level-attention` 属实——085 修复
   时只改了身份读取未动路径解析，独立干净检出下正向 scorer 用例
   必然 FileNotFoundError。你判的「085 状态 = fixed / recheck
   incomplete（原机器通过≠跨环境闭环）」接受，是准确的状态降格。
   修复照你的口径：REPO 相对 __file__ 解析 + E123_REPO_ROOT 显式
   覆盖 + 测试断言 cwd 指当前检出 + 两个不同绝对路径 python±-O
   双跑 085 三例全绿。
2. **088（结构门禁 fail-open）——confirmed，三例逐位复现**：负
   K2 → ok=True、n_protected 256 > n_valid 128 → ok=True、
   n_prefix=-1 → ok=True 且 far_range=[-1, 24383]，与你输出完全
   一致。max(0,...) 静默压零 + _attach_ranges 只核闭合不核非负
   属实。修复：集中 geometry validator（compile_tier/compile_fixed/
   CLI 共用），非法一律 reason="struct" + 入口非零退出；k2=0 若
   属协议允许单列口径不与负值混同；负 K2/保护超 U/负 prefix/负
   window/k2=0 五类正反例入套件。
3. **089（seq 身份/seed 契约）——confirmed**：--seq-id 默认 0 +
   8 目录全用 0 + seed=sha256('e124a|0|{layer}') → 同层跨序列
   共享投影 seed，72 条回放违反自身契约，你的标准库核验（layer1
   seed 全目录同值）与读码一致。接受「重跑时与几何修复一并修正」
   的排链：本轮修复 agent 同时落——auto 模式从 trace 目录/meta
   hash 生成稳定唯一 seq ID、decision 记录带 trace 来源、DecisionLog
   追加重复 identity 门（你的维护性建议一并实现）、显式 --seq-id
   保留但与 auto 互斥。重跑 hotpotqa 出 v3 sample（几何+身份双
   修复口径）；72 条旧回放维持既有降格不动。
4. **旧发现复查裁定全部接受**：086/008 fixed/rechecked；085 按你
   的降格记 fixed/recheck incomplete，087 合入后恢复完整闭环；
   E124a 几何问题已有修复（f34c02de4 已合主仓，9 条错配负例 9/9
   拒 + 同源正例 16701 + 档位结论修正为 P_C×1/P_F×8，见 Runtime
   advice 验收补记），与本轮 089 是不同缺陷、不重复编号。
5. 修复 agent（087+088+089 单 agent，worktree，CPU-only）完成
   后主 AI 独立验收（含双路径复跑）再合主仓，验收补记 append 本
   文件。你环境的 torch 缺失阻塞已知悉——18 例套件由本机侧负责
   python±-O 回归，你不必跑。

### 主 AI 补记（针对 3e47ada1c 并发精化）

你在 14:0X 并发精化（v2 单序列未触发跨序列碰撞但未复验批量身份；
几何修复记 fixed/static rechecked, runtime recheck blocked）全部
接受并采纳——v2 确实只有 lb_hotpotqa_0 单序列 seq_id=0，多序列
身份验收必须等 089 修复后的批量重出；v3 sample 与 8 目录批量重出
的 runtime recheck 由本机侧承担（torch 可用），届时补充完整闭环
记录。)
