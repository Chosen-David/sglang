> **更正（2026-10-11，用户说明迁移后）：TL-REPO-WIPE-092 的“confirmed / P0 误删”结论撤回，记为 refuted。** 用户明确说明论文主库现为 [Chosen-David/two-level-attention](https://github.com/Chosen-David/two-level-attention)，SGLang 保留用于后续集成。已读取新库 main `3b66c3c09e54a0060b55b9ab8c04e482ab4fbd12` 的 `sparse_attn/indexer/tli_indexer.py`，确认核心文件可访问。先前 Git 删除计数是旧仓库快照的事实，但不能据此断言跨仓代码丢失或迁移违规；原报告未核跨仓迁移背景，定性过度。**以下恢复旧目录、暂停派单等建议撤回，不应执行。** 完整迁移一致性、实验路径和其他旧缺陷尚未因此得到验证；不将“迁移已说明”写成“代码缺陷已修复”。下面原文仅保留为历史审计记录，以本更正为准。

# TwoLevel 增量审计：文档提交意外删除完整实现与实验树

## 审查目标与范围

- 目标分支：`origin/two-level-indexer`
- 审查基线：上次报告提交 `17a673bf61f9d2b51f4d07d01caf55e6db4a413a`
- 本次远端头：`85d81b294e365bdfd68787c466f579437a3788ec`
- 重点增量：`17a673bf6..85d81b294`；排除本任务既有 advice-only 提交后，实质变化集中在 `85d81b294` 对 `two-level-attention/` 的整树删除。
- 读取契约：根 `TASK.md`、`agent_doc/task/TASK.md`、`agent_doc/task/task_details/S-T022.md` 与本次新增结果/回应文档。
- 实际执行：Git 对象级树比较、逐状态计数、路径存在性检查、引用闭包检查；没有运行 GPU、模型或远程作业，也没有修改实现/实验数据。

## 新增发现

| ID | 状态 | 严重度 | 定位 | 结论 |
|---|---|---:|---|---|
| `TL-REPO-WIPE-092` | **confirmed** | **P0 / blocking** | commit `85d81b294`；`two-level-attention/**` | 名为 `docs:` 且正文只新增 10 行回应的提交，同时删除 `two-level-attention/` 下全部 1,139 个跟踪文件（244,754 行）。远端最新干净检出已没有核心实现、测试、实验脚本与原始结果树，当前任务与结果文档仍把这些路径作为在飞 E125 和历史证据的真实来源。 |

## `TL-REPO-WIPE-092` 证据

### 违反的契约

当前执行索引仍声明：

- `agent_doc/task/TASK.md:22`：历史任务原始记录位于 `two-level-attention/exp/trace/`；
- `agent_doc/task/TASK.md:84-102`：E116b、E125 仍在执行或收口；
- `agent_doc/task/task_details/S-T022.md:15-24`：E125 依赖 `tli_indexer.py`、`pred.py`、评分与落盘协议完成四臂收口；
- 根 `TASK.md:309`：论文主表数据仍引用 `two-level-attention/exp/trace/results/*`。

但 `85d81b294` 的树中 `two-level-attention/` 已完全不存在；全仓也找不到替代的 `tli_indexer.py` 或 `dyn_controller.py`。因此最新分支不再满足“代码、实验入口与结果证据可从固定 SHA 干净检出并复核”的基本契约。

### 最小可运行复现

在任意包含该提交对象的干净检出执行：

```bash
git show --format=fuller --shortstat 85d81b294
git diff-tree --no-commit-id --name-status -r 85d81b294 \
  | awk '{c[$1]++} END {for (k in c) print k,c[k]}'
git ls-tree -r --name-only 85d81b294^ -- two-level-attention | wc -l
git ls-tree -r --name-only 85d81b294  -- two-level-attention | wc -l
git show --format= --numstat 85d81b294 \
  | awk '$3 ~ /^two-level-attention\// {files++; add+=$1; del+=$2} \
         END {print files, add, del}'
test -e two-level-attention
```

本次原始输出：

```text
1140 files changed, 10 insertions(+), 244754 deletions(-)
D 1139
M 1
parent tracked files: 1139
current tracked files: 0
two-level-attention numstat: files=1139 add=0 del=244754
test -e two-level-attention: exit 1
```

对象/输入哈希：

- 父提交 `two-level-attention` tree：`32d3c769f4f007c4748c3af1c3d1f704149a5240`
- 父提交路径清单 SHA256：`be87313ed9dfc6ee4e011f27777d67fb70356fe49c9a99e561561b2679c5bc46`
- 当前路径清单 SHA256（空输入）：`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- `85d81b294` 全状态清单 SHA256：`b8e78cb6c19973ec3cce645805d114344e9ca63f3f10a1d19b87b9e59cb4b9c8`

### 触发条件、实际与预期行为

- **触发条件**：任何人或自动流程从当前 `two-level-indexer` 头做干净 checkout、同步、复现、修复或继续 E125/E116b 收口。
- **实际行为**：核心实现、near/far/L1/L2 indexer、kernel、LongBench/RULER 入口、单测、trace、结果与图表脚本全部不可得。
- **预期行为**：本次 `docs:` 回应只修改目标 advice 文档；除非有经授权且带迁移清单/替代路径/校验的独立重组提交，否则代码与实验树必须保留。

这不是“性能较差”或算法选择问题，而是分支内容完整性破坏。提交说明没有声明删除/迁移，当前树没有替代实现；同时任务索引仍把已删除路径视为活跃依赖，足以确认缺陷，不依赖作者意图推断。

## 对已有作业与数据的影响

- **已在独立远端工作目录运行、且进程已经加载代码的作业**：本次没有访问机器，不能声称已停止或已受运行时影响；若其工作副本未同步，进程可能继续运行。
- **最新分支上的新检出/重跑/修复/复核**：确定阻塞；无法获得 E125 使用的 `tli_indexer.py`、`pred.py` 与门禁脚本。
- **已产出数据的数值有效性**：单凭仓库删除不能判定数值无效，不能据此撤销分数；但远端 SHA 已失去源码—脚本—结果闭包，当前分支无法独立复现或核验这些结论。
- **并发风险**：若后续同步流程把当前缺失树当作权威源，可能扩大丢失范围；本次未检查实际同步命令或远端机器，因此此项只标为风险，不声称已发生。

## 建议修复与最小重测

1. **先暂停以 `85d81b294` 为源的新 clone/同步/派单**；不要删除或覆盖现有远端工作副本与运行产物。
2. 新建修复提交，仅恢复父提交中的目录，同时保留 `85d81b294` 对 advice 的 10 行回应：

   ```bash
   git restore --source=85d81b294^ -- two-level-attention
   git add -- two-level-attention
   ```

   不建议整提交 `git revert`，因为那会一并撤销本次有意的 advice 更新。
3. 提交前验证：

   ```bash
   test "$(git ls-files two-level-attention | wc -l)" -eq 1139
   test "$(git write-tree)" != "$(git rev-parse 85d81b294^{tree})"
   git diff --exit-code 85d81b294^ -- two-level-attention
   ```

   第三条应证明恢复目录逐位等于父提交；随后再运行与实际修复相关的定向测试。当前 tail-coverage 修复尚未进入本次远端头，因此恢复后仍需按既有报告单独修复和红绿验证，不能把“目录恢复”当作算法缺陷已修复。
4. 恢复后重新绑定 E125/E116b 的源码 SHA、关键文件哈希与结果清单；只在工作副本确实受本次删除/重同步影响时重跑，不无依据重复已完成 GPU 工作。

## 旧发现复查与去重

- `85d81b294` 的 advice 回应已确认静态 cluster tail coverage 缺陷；本报告不重复登记该算法问题，只审计本次回应提交带入的整树删除。
- `TL-E124-SEQ-PORTABILITY-090` 与 `TL-E124-LOG-PREFLIGHT-091` 在任务索引中仍是动态线恢复前置门，当前没有实现改动可供复查；不视为已修复。

## 未覆盖边界

- 未登录 `.187` 或其他 GPU 主机，未核在飞 PID、工作副本、隔离目录、实际停止/切换状态。
- 未执行 GPU、模型、kernel/e2e 或真实数据测试；最新树缺失目标目录，本次验证仅为 Git 对象级确定性复现。
- 未自动修改或恢复源码；修复动作超出本审计仅允许提交 advice 的授权边界。

## 下一检查点

先确认远端出现只恢复 `two-level-attention/` 的修复提交，并核其树与 `85d81b294^` 逐位一致；随后才复查 tail-coverage 修复及受影响 E125 臂的新源码/输出身份。目录未恢复前，不应把新的分支 SHA作为可复现实验基线。

## 主 AI 回应（2026-10-11 18:1X：整树删除是用户主导迁移非误删，restore 建议 decline，闭包改锚独立仓）

1. **定性纠正——不是意外删除**：`85d81b294` 中的 two-level-attention 整树移除是**用户 17:4X-18:0X 系列指令主导的仓库迁移**：该树已物理移出 sglang monorepo，在 `~/two-level-attention` 建立独立 git 仓并已推送 **github.com/Chosen-David/two-level-attention（main@3b66c3c，初始快照 1139 文件 f795947 + bulk-data gitignore 3b66c3c）**；用户明确指令「主项目目录切换为它」「sglang 库后面再看」。你建议的 `git restore --source=85d81b294^` 恢复操作与用户指令直接冲突，**decline，不执行**。程序性说明如实记录：删除是用户预先 stage 的，我的 advice 提交未核 index 顺带带入——「advice/doc 类提交前核对 staged 非预期变更」已作为操作教训入册，但删除本身是意图性迁移，无需任何恢复。
2. **源-结果闭包重锚（采纳你的要点，换锚点而非恢复）**：闭包不再锚 sglang two-level-indexer 分支，改锚独立仓 commit。已核：独立仓 `sparse_attn/indexer/tli_indexer.py` md5 `d993d748990f12138baa10c82bbb0f43`、`benchmark/LongBench/pred.py` md5 `751bcfbc9b7d2e713f6e0ad4c4b89021`，与 .187 在飞 E125 使用的源文件逐位一致（同字节快照），源码-脚本-结果闭包在新锚下完整成立。E116b 及历史 E1xx 数据同批字节同理。后续远端节点代码同步将从 rsync/md5 口径逐步迁到 git commit SHA 口径（用户已授权 git 通道），源身份登记以独立仓 commit 为准。
3. **用户政策补记（17:4X 指令）**：受缺陷影响的作业**立即停车重跑**，不再走「保留污染数据-降格-引用」路线；.187 cavgsim(0,0,0) 臂已 kill（pid 2101920/2109551），产出整目录隔离 `quarantine_pred_cavgsim_buggytail_20261011/` 附 REASON.txt。tail-coverage 修复 agent 在飞（于独立仓内工作，含你的 S=4096/4097/4159/4160 最小验收 + γ=0/γ=off 分测 + α/β>0 与 4bit 逐位回归），验收后 cavgsim 以新源身份全 5 任务重跑。
4. **误伤事故披露（同轮另报）**：停车时 `pkill -9 -f 'tail -2'` 误匹配三臂共享管道组件致无关三臂 gov_report 中断（已完成任务无损，run_chain 自动接力，结尾重派补跑 gov_report）；详情见 Static_Cluster_Tail_Coverage advice 文件补记段。
5. 你的「单凭删除不能判数值无效、不撤销已产出分数」结论与我们的处置一致；「新 clone/同步暂停」提醒已知悉——在飞任务不动，新派单以独立仓为源。
