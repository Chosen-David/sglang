# SGLang 最新代码复查：新增 bug 与修复回归

日期：2026-10-08。审查分支：`Chosen-David/sglang:two-level-indexer`。

上轮：`9049ba8759896e5efc13f287c442fb72d5e1b563`。
本轮：`15eea9adc2fcf56bbef449aaf6e915be3dbd72ae`，提交时间 2026-10-08 10:32:42 +08:00。

开始与结束均读取远端分支，返回同一本轮 SHA；新建独立只读检出，保留旧版本与旧复现证据。本轮未修改或推送 SGLang，也没有访问远端正在运行的 GPU 任务。

## 1. 结论

**最新代码仍有其他问题，而且上轮修复没有全部闭合。最紧急的是 prefill 修复引入的因果泄漏：逐行 sentinel=S_r 被全 chunk 的有效性边界接受，能把下一枚未来 token 读进当前行。**

另有评分结果结构与下游不兼容、对照组被实验组覆盖、AVG 过滤层级错误、投影 padding 污染预算与 SWA、Qwen 非 BOS token 被误删。失败退出码虽已修正，但失败 marker 仍能启动下游。

本轮得到 7 项新增或未闭合发现，另对 3 项局部修复进行了源码片段检查。42 个 Python 文件通过语法解析，32 个 shell 通过 `bash -n`。验证环境没有 PyTorch/Triton/Transformers/GPU；实际执行了源码 AST 片段、NumPy 反例、隔离 shell/scorer/完整合并脚本。**未执行完整 GPU selector/kernel 或真实模型，不能报告 e2e 精度下降数值。**

## 2. 上轮修复状态

| 上轮 ID | 本轮状态 | 依据与限制 |
|---|---|---|
| B01 无效槽位重复 token 0 | 未闭合，出现 F01 回归 | 改成了 sentinel，但 TASK.md batched 路径用 S_r，consumer 用 S；中间行泄漏，末行测试不能覆盖 |
| B02 早期行重复网格 | 局部修复通过 | 实际 grid 表达式生成 1000 个唯一 token + 24 个全局 S sentinel；未做 GPU integration |
| B03 Samsum 未赋值/缺尾 | 局部分支通过 | 实际 AST 分支拼回已有 q_input，并正确抽取新增 token；但前面的无条件首 token 删除仍在，见 F07 |
| B04 首步 EOS 不停止 | 仍在，仓库标记 deferred | 新 diff 未改生成循环 |
| B05 per-q-head 错轴 | 局部修复通过 | 实际 mask AST 已得到 `[8,4,4097]`；shared 原本二维切片轴是正确的，无须把 shared 描述成同一个错轴 bug；padding 的其他语义问题仍在，见 F06 |
| B06 runner 吞退出码 | RULER runner 退出码已修；DAG 未闭合 | 33 次 mock 失败后 exit=1；marker 与 chain 仍放行，见 F02；其他 relay 未全部改 |
| B07 旧 marker 放行 | 仍在，仓库也承认部分 deferred | 无 run-ID/commit 绑定；F02 不需要旧日志，当前失败 marker 就能触发 |
| B08 scorer 污染/不完整 AVG | 部分修复，仍不满足正式结果契约 | min-samples 有效，但缺格仍出 AVG，且新增了按长度整组过滤错误与 schema 回归，见 F03/F05 |
| B09 配置名字冲突 | 仍在，仓库标记 deferred | postfix 可能隔离某条实际扫描，但不等于所有消费者正确区分配置；见 F04 |

`research/docs/gpt_bug_verdict_20261008.json` 中的修复与 GPU 测试说明是仓库记录，本轮没有在该服务器独立复跑。本轮按实际新代码和新反例判断，不把“7 项修复”提交标题当成验收。

## 3. 新增/未闭合问题总表

| ID | 优先级 | 类型 | 触发条件 | 受控验证结果 |
|---|---|---|---|---|
| F01 | P0 | 最新修复回归 | 双池 sparse prefill；中间行有效候选不足；K2≤t<S−1 | 640 个 padding 槽位读取下一枚未来 token；数学输出 0.625，正确因果输出 0 |
| F02 | P1 | 失败协议未闭合 | runner 失败后 chain 读取其当前日志 | `ALL DONE none (FAILED=33/33)` 被放行；Quest 33 项失败后仍启动 TIA 33 项 |
| F03 | P1 | 最新 schema 回归 | 新 scorer 输出进入旧 merge/figure consumer | 实际 merge 脚本 `IndexError`；figure 的直接键读取也不兼容 |
| F04 | P1 | 新发现的旧合并 bug | C0 与 B7 原始 JSON 使用相同 method key | 输入 C0=10、B7=90，合并后两列都变成 90 |
| F05 | P1 | AVG 新过滤 bug＋残留验收缺口 | 一格部分完成，或仅少量格达到样本数 | 一格不足拖掉同长度另 10 个完整格；2/33 格也仍出 AVG=50、exit=0 |
| F06 | P2 | 新发现的投影边界/计量 bug | 投影路径 T 非 block 对齐；mask 宽是 Tpad | T=4097/Tpad=4160，真实 SWA 强制仅 65 而非 128；示例预算报 256、实际消费 193 |
| F07 | P1 | 新确认的输入协议问题 | 标准 Qwen3 tokenizer 无 BOS，使用当前 LongBench 入口 | `q_input[:,1:]` 无条件删真实首 token；实际 tokenizer/模型未在本机执行 |

P0 表示违背 causal attention 的硬语义，应先于性能/精度海选验证；P1 表示结果正确性、对照/验收或协议问题；P2 是条件性预算/边界问题，仍需在使用该组合前处理。优先级不代表已测量真实模型受损大小。

## 4. F01：逐行 sentinel 与消费端长度不同，读入未来 token

定位：

- `python/sglang/srt/layers/attention/tli/indexer.py`：`select_batched` L871–873 调入 `_select_batched_taskmd`；该方法 L1547–1552、L1569–1573 的无效 far/near 槽位使用 `S_r.view(...)`。
- 同方法：`S_r=t_c+1`；早期 identity 修复只覆盖 `t_arr < K2`，见 L1610 附近。
- `tli/backend.py`：`_sparse_extend_one` L1027–1029，`S=locs.shape[0]`、`valid=sel<S`；后续 clamp/gather；fused 与 eager 都使用这个 valid。

对 chunk 中第 r 行，因果长度为 S_r=t_r+1；当前 locs 长度却是整个 chunk 已写入的 S。对于中间行，S_r<S。新选择器把无效槽位写成 S_r，consumer 则计算 `S_r<S → True`。逻辑位置 S_r 正好是该 query 的下一枚 token，是未来位置。

这是 **标记值合法范围协议不一致**，不是浮点误差，也不是可以接受的稀疏近似。因果输出必须与所有 j>t_r 的 K/V 无关。

反例：全 chunk S=4096，query t=2047，K1=128、K2=1024、block_size=64、sink=SWA=128、(α,β,γ)=(.125,.375,.625)。实际区域函数给 near 起点 1792、SWA 起点 1920；near 只有 128 个 token，配额为 768，产生 640 个无效槽位，写为 S_r=2048。

query=0 时，完整 logits 全为 0；令 v_2048=1，所有当前/过去的 value=0。当前 consumer 接受全部 640 个未来重复槽位，输出 `640/1024=0.625`；正确因果输出为 0。改变未来 value 即改变过去输出，直接违反因果性。

本轮执行的是实际区域函数、实际 `sel_n2` 与 consumer `valid` 赋值 AST，并用 NumPy 计算 softmax 的等分情况；不是完整 GPU selector 实测。

现有 `test_b01_b02_fix.py` 的两个选择行是 t=S−1 与 t=999。末行满足 S_r=S，标记可正确屏蔽；早期行随后被 identity+全局 S 的修复覆盖。**两项都通过，仍不能排除中间行。**

建议：无效标记统一为整个 index 的全局 S，或从 selector 显式传递逐槽 valid；消费端再检查每行因果界 `0≤sel≤t_r`，不能只检查 `<S`。在常规 extend 中 `t_r=S−nq+r`，也可显式传入 query positions，避免其他调用约定漂移。先 clamp 保障 gather 安全，再用 valid 置 `-inf`；fused 与 eager 统一协议。

最小回归应包括 t=1024、2047、S−2、S−1，以及 prefix/chunked prefill。加入“只修改未来 K/V，过去输出不变”的独立因果性测试；末行候选计数不能替代它。

## 5. F02：失败 marker 仍放行下游，阶段失败不阻断下一阶段

定位：`benchmark/RULER/run_ruler.sh` L36–40；`chain_ruler.sh` L16、L39–42。

新 runner 正确返回 exit=1，但失败文字是：

```text
ALL DONE none (FAILED=33/33)
```

chain 仍用 `grep -q "ALL DONE none"`，所以本轮失败日志立即满足条件，不需要旧日志。chain 内 Quest 失败后也只累加 FAILED；打印 `ALL DONE quest (FAILED=...)` 后照样运行 TIA，最终才 exit=1。

本轮直接执行实际 runner/chain，只有 hardcoded cwd 与输入日志路径映射到隔离目录、python 替换为 exit-23 的桩：runner 的 33 次调用全部失败，exit=1；其日志使 chain 启动，Quest 与 TIA 各失败 33 次，合计 66 次下游调用，chain 最后 exit=1。未使用 GPU。

建议将状态拆成 `SUCCEEDED/FAILED`，失败不输出成功 marker；每阶段结束立即按依赖规则处理失败。若不同方法实验被定义为可独立继续，则应在 DAG 中明确独立，而不是把失败上游宣布完成。结构化状态绑定 run ID 和 frozen contract；日志字符串仅用于诊断。

## 6. F03：评分 JSON 新结构破坏合并与作图消费者

定位：`benchmark/RULER/score_ruler.py` L79；`exp/trace/run_scripts/merge_ruler_b7.py` L20–36；`merge_ruler_b7s.py` 的 `merged.update(b7s)`、`b7s_keys`；`exp/figures/fig4_partition_gain.py` L33–44。

旧 JSON 顶层是 `L4096/method -> task scores`，新 JSON 为：

```json
{"scores": {"L4096/method": {}}, "n": {}, "incomplete_cells": []}
```

merge 仍遍历顶层键并做 `key.split("/")[1]`。新键 `scores`、`n` 没有 `/`，实际完整 merge 脚本在隔离 fixture 下抛 `IndexError: list index out of range`。figure 仍以 `d["L4096/method"]` 直接读取新文件，必缺该顶层键。

建议 versioned schema 与统一 loader；在同一次变更里升级 scorer、merge、figure 与状态验收。loader 返回 `scores` 时同时保留 provenance、n 与完整性，不要取出分数后丢掉验证状态。实际执行 `score → merge → figure-input` 的 fixture 集成测试。当前已有旧图并不因此自动失效；问题是下一次用新 scorer 产物刷新终表/图时接口不兼容。

## 7. F04：C0 对照被 B7 数据覆盖，两列可能显示成同一结果

定位：`merge_ruler_b7.py` L19–23、L36–43；`merge_ruler_b7s.py` 同类直接 update 逻辑。

两套配置即便靠目录隔离，内部 method key 仍可能同为 `tli_64_128_1024_c4_A`。合并先读 base/C0，再用 B7 同键覆盖；之后 C0 与 B7 都从覆盖后的 merged 读取该键。

隔离 fixture：三个长度、全部 11 任务，base 中 C0=10，B7=90，其他 baseline 独立。实际完整合并脚本输出 `TLI_C0.AVG=90`、`TLI_B7.AVG=90`。原本 80 分的差距被抹掉。这是确定的输入输出反例，不是对真实历史表数值的推断。

触发范围：旧格式 JSON 或未来用兼容 loader 解包新 JSON 后，继续采用当前覆盖式逻辑。F03 修好之前新格式先报错；修好 F03 不能忽略 F04。

建议按实验身份分别持有 `C0_scores`、`B7_scores`；使用 `(run_id,config_hash,length,method)` 作为身份，不靠同名 method 区分臂。终表构造直接引用各臂，不先合并到同名键；重复键应显式报错。测试必须故意让实验组和对照组的数值不同，确认不会互相覆盖。

## 8. F05：AVG 过滤的是整个长度组，缺格也仍出数字

定位：`score_ruler.py` L98–108。

`res` 的键 `L/method` 对应某长度下的所有任务；代码 `all(resn[k][t] >= min_samples for t in resn[k])` 将某一任务的部分输出扩展为整个长度组无效。任务行却逐 cell 过滤，因而任务行与 AVG 采用不同集合。

完整 fixture：3 个长度×11 任务，4K 的 11 项分数=100、另两个长度=0；其中 4K 一项只有 1 样本、其他格都有 100 样本。代码把其余 10 个完整 4K 格也从 AVG 去掉，报告 AVG=0；32 个完整格的诊断均值应为 `10×100/32=31.25`。这里 31.25 是用于定位过滤 bug 的诊断值，**正式可比 AVG 应因冻结矩阵不完整而阻断**。

第二 fixture：只存在 4K 的两个任务，各 100 样本，一格本轮、一格旧轮。新 scorer 仍 exit=0、AVG=50；`incomplete_cells=[]`，缺失的 31 格没有进入该列表。虽然 stdout 有 WARN，结构化产物不能表明完整矩阵已经通过。

建议从冻结 manifest 枚举全部 expected cells，把 `missing/partial/duplicate/stale` 逐格落入机器可读状态。诊断均值严格逐 cell 过滤；正式 AVG 仅在规定集合齐全且一致时输出，否则为空并阻断正式表消费者。行数≥100 也不证明 sample ID 唯一；仍需 run/config/sample 绑定。

## 9. F06：投影 padding 移动了 SWA，并污染预算统计

定位：`sparse_attn/indexer/tli_indexer.py` 投影 prepare_index L236–259；compute_mask L988–989、L1068–1070；`sparse_attn/metrics.py` L24–30。

投影 k_qat 来自 pad_k，细筛宽度是 Tpad。compute_mask 用 `p.shape[-1]` 而非真实因果长度确定 SWA 起点，最后消费者才裁到 T。因此 B05 错轴修复只能解决尺寸崩溃，没有恢复正确的窗口和预算。

T=4097、Tpad=4160、SWA=128：代码强制 `[4032,4160)`，裁剪后真实强制区仅 `[4032,4097)` 的 65 个 token。真正最后 128 个 token 应是 `[3969,4097)`。缺失的 63 个位置可能偶然被其他池选中，但不再保证 SWA 强制语义。

更小的可辨反例用支持的 K2=256，sink=128、SWA=128、mid quota=0：执行实际两个 forced-mask 赋值和 Metrics 方法，计量报告选中 256，但实际裁剪后只有 193 个有效 token。该反例不是当前 K2=1024 全链模型测试；窗口偏移本身不依赖这个预算选择。

建议在选择前屏蔽 token≥真实长度，所有区域/预算由真实 q_ids 与 seqlens 定义；padding 只用于粗筛布局。计量使用最终消费的有效 mask，并区分槽位预算、唯一有效 token、实际读取的 page。验证 T=4096/4097/4160，不能只跑块对齐输入。

## 10. F07：Qwen 的首个真实 token 被当成 BOS 删除

定位：LongBench `pred.py` L178–182：单独 tokenize `question` 后无条件 `q_input.input_ids = q_input.input_ids[:,1:]`。

2026-10-08 核查 Qwen 官方 Qwen3-8B `tokenizer_config.json`：`add_bos_token=false`、`bos_token=null`。这没有为当前切片提供“首 token 必然是自动 BOS”的依据。标准该 tokenizer 下，首 token 是 question 内容或显式模板 token，删除会改变实际模型输入。用户服务器若使用了自定义 tokenizer revision，应另核对；本机没有执行真实 tokenizer。

本轮执行实际切片 AST，synthetic non-BOS IDs `[701,702,703]` 变成 `[702,703]`；这些数字仅是展示切片的合成 ID，不是声称 Qwen 的真实词表编码。

更稳妥的协议是完整 chat prompt 只 tokenize 一次，在 token 边界拆成 prefix 与尾部，assert 两段拼接等于规范化完整序列。分别 tokenize 字符串两段本来就可能因 BPE 边界不同而改变编码，再无条件删除一项更不能保证等价。Samsum 的新拼接只能恢复已经切过的 q_input，不能恢复在前面丢掉的 token。

这会影响规范 benchmark 输入。即使 FullKV 与 sparse 都经过同一错误预处理，也不能推出误删对两者收益差值完全抵消；须以规范输入重新验证关键对照。

来源：[Qwen 官方 tokenizer 配置](https://huggingface.co/Qwen/Qwen3-8B/raw/main/tokenizer_config.json)，检索日期 2026-10-08，网页 main 快照；服务器具体 revision 未核验。

## 11. 建议的修复与复验顺序

1. 先处理 F01，跑中间行因果性与 eager/kernel 对照；未通过前不将该 SG prefill 路径用于正式质量/速度组合。
2. 修 F03 与 F04 的产物接口和对照身份，确保新 scorer 可以进入终表，且两个不同输入保持两个不同输出。
3. 修 F02 与 F05 的依赖/结果协议。失败和不完整状态必须阻断正式报告，不只打印 WARN。重读实际启动脚本与结果 manifest 后再决定历史数据能否复用。
4. F06/F07 先做 CPU/Tokenizer 输入不变量测试，再做小模型 smoke；对受影响的参数搜索冻结版本，避免在同一扫描内无标注地改变输入协议。
5. 用修复前后各自绑定的版本保存数据。此前 GovReport trace/GQA 聚合/逐层配置消费入口等问题仍未在本提交闭合，不因局部 bug 修复推定论文协议全通过。

现有缓存路径也进行了只读追踪，包括共享 index pool、row 生命周期、增量尾块维护、decode consumer 与 prefix/chunked extend。本轮没有对这些路径运行 GPU 状态机测试；未找到足够证据的新缓存 bug，不把猜测纳入确认表。请求复用、异步扩容、多请求与 CUDA graph 仍应按实际硬件做独立回归。

当前远端活跃任务是否命中上述条件仍未知：需要启动命令、resolved config、Tokenizer revision、阶段日志与原始输出，仓库中的判决记录不能替代这些运行证据。

## 12. 证据与复现

复现脚本 `reproduce_latest_by_gpt.py` 不 import SGLang、不启动模型、不修改代码，仅用临时 fixture 执行 scorer/shell/merge 与提取的源码 AST。shell 将 cwd 与日志读取映射到临时目录，Python 调用替换为故意失败的桩；不会读写远端的真实 `/tmp` 实验日志。

证据 `evidence_latest_by_gpt.json` 保存本轮 commit、10 组检查（7 个发现、3 个局部修复）、源码 SHA256 与验证边界。42 Python/32 shell 的语法检查独立统计。任何 NumPy/mock 通过均不代表完整模型/GPU通过。

不可变源码：[本轮审查提交 15eea9a](https://github.com/Chosen-David/sglang/tree/15eea9adc2fcf56bbef449aaf6e915be3dbd72ae)。运行方式：本报告目录与 `sglang-latest-readonly` 同级，执行 `python3 reproduce_latest_by_gpt.py`。
