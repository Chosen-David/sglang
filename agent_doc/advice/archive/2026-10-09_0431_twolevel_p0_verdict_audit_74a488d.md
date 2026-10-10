# TwoLevel P0' 逐层潜力判决审查（74a488d）

## 审查目标与范围

- **审查分支/源码 SHA**：`two-level-indexer` / `74a488d48f1108c90c1d1c6c00e70aec38a9bfd3`。
- **增量范围**：相对上一份审查提交 `479de30dd7b0e349aad9f5eca9f704ab5728c16e` 新增两个提交：`dec9fb8b3` 发布 P0' 逐层潜力判决，`74a488d48` 将 E109 海选标为收官。只把这两个数据提交当作代码/结果变化；上一份 advice-only 提交不作为实现变化。
- **实际检查**：设计文档的输出失真定义、`analyze_p0p_perlayer_potential.py` 的候选回放与聚合、正式 P0' JSON、E109 最终排名、既有 near/SWA 与 trace 身份发现对新结果的影响。
- **环境与测试边界**：Python 3.12.14，Linux 6.18.44 x86_64；当前环境没有 PyTorch 或 GPU。实际执行了 Python 语法编译、JSON 解析、31 臂 AVG5 算术复算、统计中位数反例和两 head 输出投影排名反例；没有重跑真实 trace、CUDA、LongBench 或模型前向。

证据文件 SHA256：

- `analyze_p0p_perlayer_potential.py`：`bbc950fb937b6f9c101bea8c16c37aa0b3a0719e4a900c4744437e46501f9d67`
- `逐层参数求解器_数学原理与实验设计.md`：`bd873c9870740a13c6cb651aed460eff59acdeb6f06812ae970e4287e296b425`
- `p0p_perlayer_potential.json`：`55fa8416e1cb35c313ccf215de34a7c973fe4a0b4100a5957908fc2fb49761ee`
- `e109_screen4_selection.json`：`b1a02644bb938d10b075429d0d43388f50c3b194e2ae29b09b7e2cb3dd1748c4`

## 结论摘要

| ID | 状态 | 严重度 | 位置 | 结论 |
|---|---|---:|---|---|
| `TL-P0-OBJECTIVE-011` | **confirmed（源码追踪 + CPU 排名反例）** | P1 | `research/docs/逐层参数求解器_数学原理与实验设计.md:166-175,217-224`；`two-level-attention/exp/trace/analyze_p0p_perlayer_potential.py:364-427,586-614` | 设计要求先拼接所有 head 的 attention 输出误差并应用真实 `W_O`，脚本却逐 head 求平方和，未加载或应用 `W_O`。这不是“projected output MSE”，可改变候选 `c*`、各层 gap 和最终判决。 |
| `TL-P0-BASELINE-012` | **confirmed（结果交叉核对）** | P1（实验判决） | `analyze_p0p_perlayer_potential.py:65,464-467`；`results/p0p_perlayer_potential.json:42-48`；`results/e109_screen4_selection.json:235-241` | P0' 用旧领跑臂 `aavg(.125,.125,.375)` 作 G*，随后同一分支正式收官的统一冠军却是 `mavg(.25,.125,.625)`。设计协议要求同方法、同资源下与最佳统一配置比较；当前 `potential-low` 不能外推到最终 mavg 方案。 |
| `TL-P0-MEDIAN-013` | **confirmed（CPU 精确复算）** | P2 | `analyze_p0p_perlayer_potential.py:632-643`；`results/p0p_perlayer_potential.json:4945-4958` | 偶数个代表层使用排序后 `x[n//2]`，得到上中位数 7.2393%，而通常中位数应为中间两项均值 6.5132%。本次两值均低于 10%，所以不改变当前分支判决，但门槛附近可造成错误分类。 |
| `TL-P0-PREFIX-014` | **confirmed（入口追踪 + CPU 预算复算）** | P2（候选域） | `collect_trace_lb_v.py:171-177`；`analyze_p0p_perlayer_potential.py:153-199,341-361,380-404,516-529`；`results/p0p_perlayer_potential.json:4881-4897` | 候选合法性只按文档最终长度 `meta.S` 过滤，但真正回放时每个 query 使用较短前缀 `S_q=qpos+1`。140 个保留候选中有 5 个在实际早期前缀违反 `nt_near <= near_len_dyn`；其中层 35 的 `c*=(.0625,.125,.5)` 正是非法候选，故该层 `E_best/gap` 及“7 个不同 c*”不能成立。 |
| `TL-P0-TIE-015` | **confirmed（协议/实现差异；收益待测）** | P2（可维护性） | `逐层参数求解器_数学原理与实验设计.md:301-308`；`analyze_p0p_perlayer_potential.py:610-626` | 协议要求近似平局时优先稳定、容量充足、接近 G* 的候选；实现只做浮点值精确 `min(E)`，没有平局阈值、稳定性规则或记录。层 8/16/35 的前两名仅差 0.2095%/0.0262%/0.1513%，当前“参数离散性”对数值噪声敏感。 |
| `TL-TRACE-COMPLETE-004` | **open；新正式结果影响 inconclusive** | P1（旧 ID） | `collect_trace_lb_v.py:47-68,144-202`；`p0p_perlayer_potential.json:7-59` | 正式 P0' JSON 只写可变 `/tmp` 路径、样本名和 S，没有模型/tokenizer/输入/采集器/每层文件 hash。旧 trace 可跨实验复用的缺陷尚未修；本次不能确认 72.6 秒正式运行实际消费了与声明完全一致的输入闭包。 |
| `TL-BOUNDARY-NEAR-SWA-001` | **open；“海选收官”仍不可冻结** | P1（旧 ID） | `e109_screen4_selection.json:235-252`；提交 `74a488d48` | 最新提交新增最后一臂并把 E109 称作 31 臂收官，但冠军 46.95 仍是既有错误 near/SWA 分界实现下的观测。31 臂 AVG5 算术均正确，不等于修正契约下冠军成立。 |

## 复现与证据

### 1. `TL-P0-OBJECTIVE-011`：实现没有计算预注册的投影后输出误差

设计文档明确规定：

```text
Δo_l = W_O,l · concat_h(ŷ_l,h - y_l,h)
```

并特别说明“把每个 head 的误差分别求平方再相加，通常不等于真实投影后误差”。但脚本在 `eval_layer_sample()` 中形成 `y [Hkv,G,Dv]` 后，逐 head 计算 `e_id`，再用 `e_id.sum()` 累加；整个采集与分析链都没有保存、加载或应用该层 `W_O`。结果 JSON 却把目标写成 `projected output MSE`。

两 head、每 head 一维的最小反例，固定 `W_O=[1,1]`：

```text
候选 A 的 head 误差 = [ 1.0, -1.0]
候选 B 的 head 误差 = [ 0.8,  0.8]

当前脚本目标：A=1^2+(-1)^2=2.00；B=.8^2+.8^2=1.28 → 选择 B
真实投影目标：A=(1-1)^2=0.00；B=(.8+.8)^2=2.56 → 选择 A
```

因此这不是只差一个固定缩放常数：head 间经 `W_O` 的抵消/放大会改变候选排序。已保存的 `c_star`、`gap_rel` 和 `potential-low` 都必须在正确目标下重算；不能从现有 JSON 推断修正后判决仍相同。

### 2. `TL-P0-BASELINE-012`：P0' 的统一配置已被后续正式海选淘汰

正式 P0' 结果写入：

```text
method=(avg,avg), G*=aavg(.125,.125,.375)
```

但 `mavg(.25,.125,.625)` 已在 `8aa3a38b5`（02:53:49+08:00）以五任务 AVG5=46.95 登顶；P0' JSON 的生成时间为 04:14:28，提交 `dec9fb8b3` 为 04:17:40。也就是说，较强统一臂在 P0' **执行前**已经存在，并非仅被 13 分钟后的收官提交追认。`74a488d48` 的 31 臂收官排名为：

```text
1. mavg(.25,.125,.625)      AVG5=46.95
2. mavg(.125,.125,.625)     AVG5=46.82
3. mavg(.125,.375,.125)     AVG5=46.74
```

设计文档 §1、§7、§11.1 要求固定同一 method，在相同资源下比较 `G*(m)` 与逐层配置；完整系统比较还要用 `best_m G*(m)` 对 `best_m L-OUT(m)`。当前 P0' 只回答了旧 aavg 参考下的层内代理问题，不能据此停止最终 mavg 组合的逐层求解器探索。若要保留该运行，应将标题/结论限定为“aavg(.125,.125,.375) 代理下的初步负结果”，而不是当前 best method 的正式判决。

### 3. `TL-P0-MEDIAN-013`：偶数样本取了上中位数

正式 8 层 gap 为：

```text
排序后：[0, .049823, .050142, .057871, .072393, .088681, .273856, .306242]
脚本 x[8//2]：.072393
标准 median：( .057871 + .072393 ) / 2 = .065132
```

本次两者都小于 0.10，故本项单独不改变 `potential-low`。但脚本将该值直接用于 `>10%` 判决，后续若中间两层跨越门槛，会产生错误分类。建议使用 `statistics.median` 或 `torch.median` 前明确偶数定义，并为奇数、偶数、恰等门槛三种情况加单测。

### 4. `TL-P0-PREFIX-014`：最终长度合法不代表每个回放前缀合法

采集器对每个文档保存 16 个锚点和末尾 256 个 query，分析器再从中后段确定性抽取最多 6 个 query。候选域却在读取任何 `qpos` 之前，仅对每个样本的最终 `meta.S` 调用 `exclusion_reason()`；实际 `compute_mask()` 的长度则是每个 query 的 `S_q=qpos+1`。因为 `near_len_dyn` 随 `S_q` 缩短，这两个合法域并不等价。

按正式 JSON 的 8 个 `sample_S`、采集器的 `N_ANCHOR=16/N_TAIL=256` 和分析器的 6-query 抽样规则复算，在最终长度过滤保留的候选中有 5 个在真实前缀非法：

```text
(.0625,.125,.5)   passage S_q=8307/8255：nt_near=512 > near=504/500
(.0625,.125,.625) hotpot S_q=9009/8891：640 > 548/540；passage：640 > 504/500
(.0625,.25,.25)   passage S_q=8307/8255：512 > 504/500
(.0625,.5,.125)   passage S_q=8307/8255：512 > 504/500
(.0625,.625,.125) hotpot S_q=9009/8891：640 > 548/540；passage：640 > 504/500
```

复算输出的规范化 JSON SHA256 为 `c9b9e9e419d879d15fe4cf25e8575658ab7f8b38e05fd405fb550bd1867dee15`。层 35 当前 `c*=(.0625,.125,.5)` 位于第一项，因而必须在每个 `S_q` 都合法的域上重选。严格删除非法候选只会令该层 `E_best` 不降、gap 不增，所以此项本身不会把旧 `potential-low` 推向 `potential-high`；但层 35 数值、离散性统计和候选解释必须重算。

### 5. `TL-P0-TIE-015`：近似平局规则没有落到可执行判定

设计文档 §6.4 第 6 步明确要求近似平局时优先稳定、容量充足、与 G* 差异小的候选，脚本却直接执行 `min(E, key=E.get)`。正式 JSON 中层 8、16、35 的第一/第二名相对差分别仅 0.2095%、0.0262%、0.1513%，却没有预注册的近似平局阈值、重采样稳定性或 tie-break 记录；“8 层出现 7 个不同 c*”因此混合了真实结构差异与数值微差。

这不是已证明会改变 10% gap 门槛的 P1 bug，但属于值得维护的正确性保障：在看结果前冻结相对/绝对 tie 阈值和稳定性检查；将候选选择写成独立纯函数，结果 JSON 同时记录进入 tie set 的候选、规则和最终取舍。修复后的收益只能通过重复/保留 query 重跑判断，当前不宣称质量提升。

### 6. 旧发现对新增数据的影响变化

- `TL-TRACE-COMPLETE-004` 在上一报告已判 residual confirmed。本次新增的正式 JSON 没有输入 manifest 或不可变 hash，因此该缺陷从“未来风险”变成对新判决的直接证据缺口。没有外部 trace 目录，无法证明实际文件不完整或陈旧，影响状态为 **inconclusive**，不是凭静态检查断言数据已污染。
- `TL-BOUNDARY-NEAR-SWA-001` 的根因没有变化；变化是 `74a488d48` 在未重跑修正语义的情况下把 E109 标为“收官”。46.95 可保留为旧实现观测，但不能用于冻结最终 method/G*，也不能作为 P0' 最终参考。

## 对已跑数据与结论的影响

1. **P0' 的 `potential-low` 暂停使用**：当前实际目标不是预注册目标，参考 method/G* 在运行时已经过时，且候选合法域按错误长度过滤。现有 per-layer 数值可作为“未投影、逐 head MSE / 旧 aavg 参考”的探索快照保留，不能据此结束逐层方案。
2. **E109 的 31 臂表算术可复核，但“最终冠军”未冻结**：全部 AVG5 与五任务均值在四舍五入容差内一致；near/SWA 修正前后的实现语义不同，仍需重跑受影响臂。
3. **没有证据表明 median 或前缀合法域单独翻转结论**：median 修正为 6.5132% 仍低于 10%；删除非法候选只会缩小逐层相对 G* 的优势。更严重的不确定性来自目标函数和基线失配。
4. **未宣称现有 trace 已污染**：缺 manifest 使输入身份不可证，当前只能标 inconclusive；需回到真实 trace 目录核验后才能定性。

## 建议修复与最小重测

1. trace/manifest 保存每层真实 `W_O` revision 或可不可变定位的模型 checkpoint；分析端按真实 head 顺序 `concat` 后应用 `W_O`，以投影后 dense 输出计算统一 `a_l`，再重算所有候选误差。先用上述两 head 反例和一个真实小序列与模型 attention 输出逐位对拍。
2. 在选定 query 后，以所有实际 `S_q` 构造候选域；候选只要在任一消费前缀违反硬约束就剔除并记录 `sample/query/S_q/reason`。为上面 5 个候选加回归测试，特别断言层 35 旧 `c*` 不进入该运行的合法域。
3. 修复 median，并把判决统计写成显式函数；增加 8 层向量 `[0, .049823, .050142, .057871, .072393, .088681, .273856, .306242]` 的回归测试，期望 `0.065132`。
4. 预注册近似平局阈值和 deterministic tie-break；保存 tie set、首二名差、稳定性复跑及最终理由，避免把浮点微差直接解释为层间离散性。
5. 在修复 near/SWA 边界并完成受影响 E109 臂同条件重跑后，冻结最终 method 与 `G*(m)`；随后以**同 method、同 K1/K2、同 trace、同候选域**运行 P0'。若仍选择 mavg，至少以 `--far-method minmax --near-method avg --gstar 0.25,0.125,0.625` 重跑，不得沿用 aavg JSON。
6. 完成既有 trace manifest 修复：模型/config/tokenizer/input/token ids/target length/采集器 SHA、请求与实际层集合、每层 shape/dtype/S/hash；分析器要求所有 `sample × layer` 闭包齐全并把实际消费清单写入结果。
7. 重跑后先报告修正目标下的 8 层绝对误差、gap、样本/任务分项与不确定性，再决定是否进入端到端整表验证。未通过目标和输入身份门禁前，不发布新的 `potential-high/low` 总判决。

## 实际验证与未覆盖项

- `python -m py_compile analyze_p0p_perlayer_potential.py`：通过。
- 两个新增 JSON 均通过 `python -m json.tool`。
- `git diff --check 479de30dd..74a488d48`：通过。
- 31 个 E109 臂的 AVG5 均与五任务算术平均一致（允许保存值的两位小数舍入误差）。
- CPU 复算确认 median 为 0.065132；两 head 反例确认未投影目标与真实投影目标可产生相反候选排名。
- CPU 按采集/分析确定性 query 规则复算 8 个样本的实际 `S_q`，确认最终长度域中有 5 个保留候选在早期前缀非法；复算结果 SHA256 见上文。
- 独立审查上下文复核了旧 G* 的时间顺序、median、前缀合法域、近似平局协议和 E109 算术；`W_O` 目标失配由本报告主审依据设计公式与实现直接确认，未把模型间赞同当作实测证据。
- 未安装 PyTorch，未运行脚本 dry-run；无 GPU、模型权重和外部 `/tmp/trace/qwen3-8b-v`，因此未复验 72.6 秒正式运行、W_O 真值、trace 完整性或修正后判决。

## 下一检查点

优先复核 `W_O` 投影目标修复及其小序列 oracle；随后检查 near/SWA 修正后的 E109 重跑与最终 `G*(m)` 冻结，再运行绑定完整 manifest 的 P0'。若期间有新源码提交，只重验受影响路径并重新绑定审查 SHA。
