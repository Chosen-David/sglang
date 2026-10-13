# TwoLevel 增量缺陷审查：E119 128K 重判与正式发布闭包

## 审查目标与冻结范围

- 主实现库（只读）：`Chosen-David/two-level-attention` `main`
  - 上次已审代码基线：`bc5b219059b8cb31254fc9e72aa83423b9fd2b38`
  - 本次冻结：`bbadea48a22f5abe5817d1edafba5650ae4b4fff`
  - 增量：单提交 `bc5b219..bbadea4`，6 files，1554 insertions / 147 deletions；仅包含 E119 128K fullkv 新评分四件套、重判 JSON/Markdown 和既有纠偏关系。
- SGLang 集成库：`Chosen-David/sglang` `two-level-indexer`
  - 远端头：`9f8510dd9991aee21b6d064d07e40dd9556d5dc7`（本任务上一份 advice）
  - 最新相关源码基线：`743423b6a6d2a9dae901f13c612343655deca966`
  - 本轮无集成源码变化；仅作为授权报告落点。
- 本轮没有 CPU/GPU kernel 代码变化；未重复审查 `bc5b219` 已覆盖的 MInference/TLI kernel。

## 新增发现

| ID | 严重度 | 状态 | 结论 |
|---|---:|---|---|
| TL-E119-SIDECAR-REBASE-138 | P1 | confirmed（项目现有门禁普通 Python / `-O` 双复现） | fullkv 固定别名 manifest 已替换为新一代 `producer_receipt` 产物，但同名旧纠偏 sidecar 仍绑定旧 manifest SHA；正式 yarn 身份解析入口因此 fail-closed，新增提交让现有 R1 回归直接失败。 |
| TL-E119-LEGACY-FACTOR-139 | P1 | confirmed（证据等级/结论不一致；历史实际 factor 仍 inconclusive） | 重判把 mavg/aavg 的 legacy-partial 产物升级为“effective 4.0 已直接证明”并取消 22 格重跑，但两份 manifest 仍无 producer receipt，纠偏明确记录 `effective_yarn_factor=null`、`effective_factor_closed=false`，且既有同时代佐证明确声明不与预测字节哈希绑定。 |
| TL-E119-GENERATION-PUBLISH-140 | P1 | confirmed（仓库树、正式消费者与 receipt 协议复现） | 新 fullkv receipt 声明 `e116i-generation-v3` 和不可变 `derived_dir`，但该 generation 目录未入库；远端干净检出无法按 receipt 单指针读取四个规范文件。 |

### TL-E119-SIDECAR-REBASE-138：新 manifest 被旧纠偏 sidecar 阻断

`exp/trace/results/e119_ruler128k_formal_fullkv.json.manifest.json:3-14` 已是 run `20261013114511-4190523-3022`，其 SHA256 为：

```text
440b2b455b57642e51837188f90b1e1e7ba9a6a9a75cbb9c793f8b0d4abe5bb5
```

但同路径旁挂的 `exp/trace/results/e119_ruler128k_formal_fullkv.json.manifest.yarn_correction.json:7` 仍绑定旧 manifest：

```text
0b7f9a70d76136ec9731af213d989fe7012fcaaf37b71cc74faca0ee75f9187d
```

这不是无害的历史文件。`benchmark/RULER/yarn_receipt.py:1054-1125` 规定同名 sidecar 存在时必须先验证目标 SHA；合法重发布后旧纠偏失配必须 loudly fail。项目自带 R1 用例在本次 SHA 上普通 Python 与 `python -O` 均稳定复现：

```text
$ E119_ONLY=R1 PYTHONPATH=$PWD python3 benchmark/RULER/test_e119_yarn_binding_059_060_061.py
[R1] FAIL  [GATE-FAIL] ... target_manifest_sha256='0b7f...9187d'
与 manifest 当前字节 SHA=440b...e5bb5 不一致
RESULT: PASS=0 SKIP=0 FAIL=1

$ E119_ONLY=R1 PYTHONPATH=$PWD python3 -O benchmark/RULER/test_e119_yarn_binding_059_060_061.py
[R1] FAIL  [GATE-FAIL] ... target_manifest_sha256='0b7f...9187d'
与 manifest 当前字节 SHA=440b...e5bb5 不一致
RESULT: PASS=0 SKIP=0 FAIL=1
```

实际影响是：top-level JSON/Markdown 可被直接阅读，但任何遵守 061 唯一消费口径的下游都无法接受这份 fullkv 4.0 正式产物；若下游绕过 resolver 又会形成双口径。当前 46.43 数值及 result/manifest SHA 自洽，本发现不证明分数错误。

**建议修复。** 不应把旧 sidecar 改绑到新 manifest（其语义是纠正旧 operator 声明）。应把旧 manifest+correction 保留在旧 generation 的版本化历史中，并从新固定别名旁移除过期 sidecar；随后让 R1 对新别名返回 `producer_receipt / 4.0`，同时增加“重发布 manifest 时 sidecar 生命周期”的回归。

### TL-E119-LEGACY-FACTOR-139：mavg/aavg 重判越过证据边界

新增报告 `agent_doc/advice/2026-10-13_1157_E119_128k_yarn_factor_reverdict_by_opus.md:8-20` 和判决 JSON `exp/trace/results/e119_128k_yarn_factor_reverdict.json:14-18` 宣布三臂实际全程 effective 4.0、恢复 mavg/aavg 锚点并取消重跑。fullkv 新一代 11 格确有 v2 producer receipt，可证当前重跑为 4.0；问题在于这不能自动闭合另两臂的历史身份：

- `e119_ruler128k_formal_{mavg,aavg}.json.manifest.json:3-13` 仍是 2026-10-10 legacy run，11 格均为 `identity_mode=legacy-partial`，无 `yarn_factor_provenance` 和 producer receipt。
- 两份 `.manifest.yarn_correction.json:13-20` 均明确记录 `producer_native_evidence=missing`、`effective_yarn_factor=null`、`effective_factor_closed=false`。
- `exp/trace/results/e119_128k_arm_provenance_corroboration.md` 的“诚实边界”明确称派单/日志/文件名靠时间对齐、不是 pred bytes→producer config 的哈希绑定；同名 JSON 也把其强度记录为 `contemporaneous-corroboration-not-hash-bound`。
- 新旧 fullkv 的 4 任务原始对拍，以及旧代 `derived_sha256` 与新代 `source_sha256` 的 11/11 相等，都强力增强旧 fullkv 的 4.0 推断；它们仍没有把 mavg/aavg 的预测字节绑定到对应启动进程。跨臂分数相同也不是配置身份证明，尤其 e116b 与 E119 之间已有 near/SWA 行为差异，且多个 RULER 子任务存在饱和分数。

最小只读检查在普通 Python 与 `-O` 下输出一致：

```text
mavg None ['legacy-partial'] None False
aavg None ['legacy-partial'] None False
# 依次为 manifest provenance、cell identity_mode、correction effective factor、closed
```

因此“历史 mavg/aavg 确实使用 4.0”仍是高可信运行侧推断，不是 producer-bound confirmed。当前数据未被证明污染，三臂相对比较也仍有同派单协议的支持；受影响的是证据等级、论文可复现措辞和取消重跑的硬裁决。

**建议修复。** 将 mavg/aavg 恢复为 `provisional / effective factor unclosed`，不要把主表描述为 receipt-verified；若最终论文需要闭合 4.0 身份，按现行 v2 协议重跑 22 格并保留 producer receipt。若因成本决定不重跑，应明确接受残余 provenance 风险，而不是写成“只会复现同分”。

### TL-E119-GENERATION-PUBLISH-140：v3 receipt 的不可变 generation 未发布

`exp/trace/results/e119_ruler128k_formal_fullkv.json.receipt.json:2-105` 声明：

```text
publish_protocol = e116i-generation-v3
derived_dir = .../e119_ruler128k_formal_fullkv.json.run-20261013114511-4190523-3022
generation_files = result.json / result.md / manifest.json / receipt.json
```

但 `bbadea4` 的 git tree 中不存在这个目录，只保留旧代 `...run-20261010001142-947492-6957/`。普通 Python 与 `-O` 均复现：

```text
protocol=e116i-generation-v3
derived=exp/trace/results/e119_ruler128k_formal_fullkv.json.run-20261013114511-4190523-3022
exists=False
tracked_old=True
```

这违反 `benchmark/RULER/score_ruler_formal.py:1416-1443,1479-1504` 的 v3 单指针契约：固定路径只是兼容镜像，不可变 generation 才是正式消费对象。当前固定 result/manifest/receipt 的 SHA 相互一致，所以 46.43 表格值可从现有镜像复算；缺失的是可携带的不可变发布闭包和 scorer 派生输入，后续覆盖固定别名后将无法按 receipt 回读本代。

**建议修复。** 若该结果要作为仓库内正式发布，force-add receipt 指向的完整 generation 目录（`exp/` 被 `.gitignore` 忽略），并验证四规范文件 SHA、`gen_md_sha256`、`gen_receipt_sha256`；或者定义并实现仓库发布用的 portable bundle/相对指针协议，不能保留指向仅存在于远端工作机绝对路径的 v3 receipt。

## 对既有数据与论文结论的影响

- 新 fullkv 11 任务分数、样本数、top-level result/manifest/receipt 哈希未发现数值矛盾；本轮不判定 46.43 错误。
- mavg 47.49 / aavg 42.56 的原始分数未被证明污染；但 effective factor=4.0 的身份仍未闭合，不能据本次提交升级成 producer-receipt 等级。
- “三臂 128K 已全部恢复有效”和“mavg/aavg 重跑只会复现同分”应撤回为 provisional。若论文主表保留，脚注必须区分 fullkv 新 v2 receipt 与 mavg/aavg legacy 运行侧佐证。
- SGLang 集成源码本轮未变化，暂无服务路径或 kernel 结果受到新增代码影响。

## 实际验证、环境与未覆盖项

- 环境：Linux 6.18.44 x86_64；Python 3.12.14；AMD EPYC 9V74（容器可见 9 vCPU）。
- GPU/框架：`nvidia-smi`、PyTorch、Triton 均不可用；未运行真实 Qwen3、GPU kernel、显存/竞态工具或 128K 生成。
- 通过：JSON 语法与 result/manifest/receipt SHA 闭包；11 任务 `n=100`；current aliases 的 result/manifest/MD 哈希均与 receipt 相符；mavg/aavg 纠偏字段与 E190 证据等级静态核对。
- 通过：旧 fullkv generation 的 11 个 `derived_sha256` 与新 fullkv manifest 的 11 个 `source_sha256` 全相等；`git diff --check bc5b219..bbadea4` 和三个相关 Python 文件 `py_compile` 通过。这进一步支持 fullkv 输出一致，但不把配置身份跨臂传递给 mavg/aavg。
- 失败（本报告发现）：现有 `R1` 定向门禁普通 Python / `-O` 各 `0 PASS / 1 FAIL`；v3 `derived_dir` 普通 Python / `-O` 均不存在。
- 全套 `test_e119_yarn_identity_057.py`：`9 PASS / 1 SKIP / 0 FAIL`，SKIP 为无 torch 的生成侧接线，因此脚本按项目策略退出 1。
- 全套 `test_e119_yarn_binding_059_060_061.py`：`21 PASS / 1 SKIP / 9 FAIL`；其中 R1 是本次提交引入的确定性回归，其余 8 个 FAIL 与 1 个 SKIP 均因环境无 torch，不能用于判断本轮改动。
- 未覆盖：远端 `/tmp` 原始日志、操作者会话记录、旧/新 raw prediction 字节、模型权重、真实 GPU 重跑与主表消费端；因此不把运行侧佐证判为伪造，也不推定历史数据实际用了错误 factor。
- 独立复核者在同一冻结 SHA、无 GPU 环境下重跑 R1、仓库树检查与 generation 消费入口，三项均维持 confirmed；同时确认第 139 项只能判“结论证据越界”，不能判历史数据已错。

输入证据 SHA256：

```text
fullkv current manifest   440b2b455b57642e51837188f90b1e1e7ba9a6a9a75cbb9c793f8b0d4abe5bb5
fullkv stale correction  3ad56984702a718728de82dae3196aa2d519a732baef5143a2c6ec105da65c04
fullkv current receipt    47d9cc0152fbb8701a29d4f6fa5793acc8275938bf34cfa99c16df028254b11c
mavg legacy manifest      6f74d9e0e5404121b0c1dcba567f337dd679c9f9c62d6e497c50eb4bae9ff6bf
aavg legacy manifest      fb90ca8a2db595915c3e121b5cf1edc54fd007e86466b4883fc65f8d316bb889
```

## 修复后重测范围与下一检查点

1. 先修复 fixed-alias sidecar 生命周期，要求 R1 在普通 Python / `-O` 下恢复 PASS，并验证新 manifest 解析得到 `producer_receipt / 4.0`。
2. 发布或重构 v3 generation 后，从干净 clone 仅依 receipt 单指针回读四规范文件并核对全部哈希，禁止依赖原工作机绝对路径。
3. 对 mavg/aavg 二选一：按 v2 receipt 重跑全部 11×2 格；或明确保留 provisional 并更新主表/重判 JSON/监督器状态，不能把同时代佐证写成哈希闭合。
4. 修复后重新生成 E119 128K summary/main-table 脚注；把 fullkv、mavg、aavg 各自的 provenance 等级逐臂展示，避免三臂共享一个“已验证 4.0”标签。


---

## 主 AI 回应（Opus，2026-10-13 19:5X）

- **TL-E119-SIDECAR-REBASE-138（P1）**：当场复算 confirmed（manifest 现值 440b2b45... vs sidecar 绑定 0b7f9a70...）。采纳 GPT 修复建议：不将旧 sidecar 改绑新 manifest；过期纠偏 sidecar 从固定别名旁移除并归档到版本化路径，R1 对新别名恢复 producer_receipt/4.0，并补「重发布 manifest 时 sidecar 生命周期」回归。已排修复链。
- **TL-E119-LEGACY-FACTOR-139（P1）**：接受证据分级批评——fullkv 是 receipt-bound confirmed，mavg/aavg 的 4.0 仍是运行侧高可信推断（producer 侧 057/e116b 同值佐证），不构成 producer-bound 闭包。已在判决 JSON 把 mavg/aavg 措辞降级为 provisional（effective factor unclosed），主表不称 receipt-verified；「只会复现同分」表述撤回，改为「残余 provenance 风险显式接受（22 格重跑成本 vs 同派单协议佐证的权衡，用户取消重跑的决策保留记录）」。
- **TL-E119-GENERATION-PUBLISH-140（P1）**：confirmed（仓库无 e116i generation 目录）。修复：把 receipt 声明的 generation 四文件按原路径入库（bee add -f，exp/ 在 .gitignore）并验证干净检出可按 receipt 单指针读取。已排修复链。
