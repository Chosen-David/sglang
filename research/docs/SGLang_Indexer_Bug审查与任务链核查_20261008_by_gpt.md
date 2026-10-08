# SGLang Indexer：代码 bug、实验脚本与任务链审查

审查日期：2026-10-08。对象：`Chosen-David/sglang` 的 `two-level-indexer` 分支，提交 `9049ba8759896e5efc13f287c442fb72d5e1b563`；提交时间为 2026-10-08 09:08:47 +08:00。本轮重新读取远端分支 SHA，与本地只读检出一致。

## 1. 结论与证据边界

**确实存在代码 bug，也存在会把失败、旧结果或不完整结果带进后续实验的编排问题。目前不能把“脚本打印完成”视为论文结果已经可用。** 其中，稀疏 prefill 的无效槽位处理会改变 attention 的数学语义；LongBench Samsum 分支会报错；per-q-head 与投影组合存在长度不对齐时的 mask 维度错误；RULER runner 和 scorer 已在受控环境中复现失败误报与跨轮混合。

本轮未修改、提交或推送 SGLang。没有连接训练服务器，没有查看远端 GPU 进程、tmux、实际启动命令或运行日志，因此不能断言正在运行的某个任务已经触发这些问题。仓库中的 E109 汇总 JSON 不是当前进程状态证明，也不足以单独确认产生它的代码、输入与配置。

验证环境有 Python 和 NumPy，没有 PyTorch、Triton、Transformers 或 GPU。验证分为：实际源码片段执行、实际 shell/scorer 的隔离执行、数学与 NumPy 反例、静态调用链检查。未执行完整模型或 CUDA kernel；不报告 e2e 精度损失数值，不宣称全面无 bug。

证据文件：`reproduce_by_gpt.py`、`evidence_by_gpt.json`。其中 9 组行为/条件检查完成，41 个 Python 文件通过 AST 语法解析，32 个 shell 文件通过 `bash -n`。语法通过不代表行为正确。

## 2. 发现总表

优先级 P1 表示可能破坏结果正确性、实验可比性或下游验收；P2 表示有明确触发条件的追溯/决策问题。每项适用范围单独列出，不把条件性问题扩大为所有配置都失败。

| ID | 优先级 | 问题 | 触发范围 | 本轮证据 |
|---|---|---|---|---|
| B01 | P1 | prefill 将无效选择槽位替换为 token 0，最终 softmax 重复计权 | TASK.md 双池 prefill，某池有效候选不足配额 | 实际区域函数 + 数学/NumPy反例；GPU路径未执行 |
| B02 | P1 | 早期行 uniform grid 的重复数不均，不能普遍等价 dense | sparse prefill 中因果长度不整除 K2 的早期行 | 源码公式 + NumPy反例 |
| B03 | P1 | Samsum 后续读取未赋值的 `generated_content`；同时生成输入缺少切出的尾部 | `--task samsum` 的当前 LongBench 入口 | 实际 AST 分支执行得到 `UnboundLocalError`；输入缺尾为调用链检查 |
| B04 | P1 | 首个生成 token 为 EOS 时仍继续生成 | LongBench 手写 greedy loop，首步 EOS | 实际循环 AST + mock logits 执行 |
| B05 | P1 | per-q-head mask 切错轴，未裁掉 padding token | per-q-head + padded mask，例如投影路径、非 64 对齐长度 | 实际 mask 赋值 AST + NumPy形状失败 |
| B06 | P1 | Python 失败仍被 runner 当成功，继续打印 ALL DONE | RULER runner/若干 relay；上游命令报错 | 实际 RULER 脚本，33 次 mock Python 全失败而退出 0 |
| B07 | P1 | 旧日志完成标记即可启动新下游；守卫无 run ID 绑定 | RULER chain、E89/E90 relay、B7 系列守卫 | 源码检查 + 相同 grep 条件接受旧日志 |
| B08 | P1 | scorer 混合不同轮次，接受部分数据并输出 AVG | 同一输出目录内残留历史/部分结果 | 实际 RULER scorer + 隔离 JSONL fixture |
| B09 | P2 | 配置名字遗漏 α/β/γ 等参数，同目录同 task 同时间标签可覆盖 | 未用独立 postfix/run ID 隔离的扫描 | 实际命名函数执行，两套不同参数得到相同名字 |

此外，上一份审查已经确认 GovReport trace 丢弃正文、GQA 的 HF/SGLang 选择语义不同、P0' 中位数计算与目标定义问题。本报告在第 5 节说明这些问题如何影响现有实验链，不将其包装为本轮新发现。

## 3. 代码正确性：定位、反例与建议

### B01：无效槽位被变成真实 token，改变 attention 权重

位置：`python/sglang/srt/layers/attention/tli/indexer.py`，`_select_batched_taskmd`，L1542–1545 和 L1558–1583；消费者 `tli/backend.py`，`_sparse_extend_one`，L1012–1064。

选择器将无效 near/far 槽位转为 0，再与真实 sink、SWA token 拼接。消费者用 `locs[sel]` gather 并直接 softmax；fused 路径没有传入逐槽 `valid`，0 对 pool 容量边界而言也是合法槽位。**这避免了索引越界，但没有屏蔽无效候选。** 本轮没有将这里误报为 sentinel=S 导致的越界，因为 batched prefill 实际上已把它转成 0。

对一个原 token i，如果它出现在选择表中 m_i 次，其总 attention 权重变为：

\[
 p'_i=\frac{m_i e^{s_i}}{\sum_j m_j e^{s_j}}
      =\operatorname{softmax}_i(s_i+\log m_i).
\]

因此重复 token 等价于添加 `log(m_i)` 的 logit 偏置。不能解释为“多个零权重 padding”。这里的零是索引位置，不是 softmax 权重。

反例使用实际 `_taskmd_regions` 函数：S=4096，block_size=64，sink=SWA=128，K1=128，K2=1024，(α,β,γ)=(.125,.375,.625)。函数给出 near 起点 3584、SWA 起点 3968；near 只有 384 个 token，配额却是 768，far 配额为 0。若全部 near 候选有限，将出现 384 个 token-0 填充槽位，另有真实 sink 中的 token 0，共 385 次。

设 q=0，所有完整 attention 分数均为 0；v_0=1，其余 v_i=0。带重复槽位的输出是 `385/1024 = 0.3759765625`；同一个去重有效集合有 640 个 token，输出是 `1/640 = 0.0015625`。这是数学上的输出反例，不是实际模型精度测量。

建议修复协议：选择表保留无效标记，同时传递逐槽 `valid`；逻辑索引先 clamp 到合法位置以安全 gather，attention 分数再按 `valid` 置 `-inf`。fused kernel 已支持可选 `valid`，decode 批量消费者也已有同类协议，可以复用。**不能简单以 `sel!=0` 判有效，因为真正的 token 0 必须保留。** 根据最终消费集合统计真实 token 数，不把填充槽位计为有效预算。

### B02：早期行重复网格不普遍等价 dense

位置：同一 indexer，`_select_batched_taskmd` 尾部 L1590–1603。

代码将 K2 个槽位分配给 L 个因果 token：每个先重复 `floor(K2/L)` 次，再把余数填给最前面的 token。注释称“dense 等价”，但当 K2 不能整除 L 时，重复数 m_i 不同，沿用 B01 的公式即可看出权重改变。

反例：K2=1024，L=1000。前 24 个 token 出现两次，其他出现一次。q=0，前 24 个 v=1、其余 v=0，网格输出为 `48/1024=0.046875`，dense 输出为 `24/1000=0.024`。完整 prompt 的 S>dense_threshold 时，早期行仍可能进入 sparse prefill，不能只用短 prompt 全 dense 的 smoke test 排除这个问题。

建议：早期行使用真正 dense attention，或每个合法 token 出现一次，剩余槽位使用有效性掩码；分别覆盖 L=1、64、65、1000、1023、1024、1025 以及 chunked/prefix prefill。

### B03：Samsum 分支报错，而且生成输入缺尾

位置：`two-level-attention/benchmark/LongBench/pred.py`，L165–184、L186–201、L243。

Samsum 分支赋值给 `output=model.generate(...)`，却与其他分支共用 `tokenizer.decode(generated_content, ...)`；该分支没有给 `generated_content` 赋值。实际 AST 分支与 decode 语句经 mock model 执行，得到 `UnboundLocalError`。

另一个同分支问题：前面已经把 prompt 尾部切入 `question/q_input`，Samsum 的 `model.generate(**input)` 只收到剩下的 prefix，没有把 q_input 接回去。即便只修未赋值变量，也仍未恢复完整输入。

当前默认 13 任务列表不包含 Samsum，所以这不能用来否定已经完成的那 13 个任务；但若补 few-shot 摘要类别或扩成更完整任务集，会触发。建议所有任务共享一个经过测试的生成入口，Samsum 的换行停止策略作为显式参数，输入始终对应完整规范化 token 序列。

### B04：首步 EOS 没有停止

位置：同一文件 L218–233。

代码先加入首个 token，之后进入循环；EOS 判断只在下一次模型调用之后执行。源代码循环接受 mock logits `EOS → A → EOS` 时，生成列表为 `[EOS,A,EOS]`，调用模型 3 次。常规首步停止语义下应仅有 `[EOS]`，调用一次；跳过特殊 token 的 decode 还能留下不应生成的 A。

建议首步与后续步共享停止判断，并按 model generation_config 处理 EOS 列表。若某任务刻意设置 min_new_tokens，应明确抑制终止 token 或定义停止协议，而不是忽略已输出的 EOS。与 dense baseline 使用相同停止协议；再核对对比库中的 generate 行为。

### B05：per-q-head mask 的裁剪轴错误

位置：`two-level-attention/sparse_attn/ops/eager_decoding.py` L27–41；投影 padded 来源：`sparse_attn/indexer/tli_indexer.py` L236–259。

per-q-head 的 repeat 结果是 `[Hkv,G,Tpad]`，代码 `[:, :eos_k-bos_k]` 裁剪的是 G 轴；需要裁剪最后的 token 轴。普通未 padding、block_size=1 的路径可能碰巧尺寸相等，因而掩盖错误。投影路径从 `pad_k` 生成 k_qat，非块对齐长度能产生 Tpad>T。

源代码 mask 赋值在 NumPy repeat 替身下执行：H=32、Hkv=8、G=4、T=4097、Tpad=4160。实际得到 mask `[8,4,4160]`，完整 score 是 `[8,4,4097]`，`where` 无法广播。

建议按 `[..., :T]` 裁剪并 assert 形状；padding 位置在选择器内也必须显式无效，不能只在消费者尾部裁剪后仍用含 padding 的预算/区域边界。GPU 回归应覆盖 shared/per-q-head × 投影/无投影 × T=4096/4097/4160，不能只测整齐的 4096。

## 4. 实验脚本与任务链

### B06：失败被吞掉，完成标记不可信

位置：`benchmark/RULER/run_ruler.sh` L5、L16–25；`chain_ruler.sh` 的 Quest/TIA 两段；`exp/trace/run_scripts/relay_gpu0_e89e90.sh` 等。

RULER runner 没有失败传播机制。`python ... 2>&1 | tail -2` 的默认退出码来自 tail，而不是 Python；外层循环也不检查失败。甚至 `cd` 失败后仍继续运行。

隔离复现直接执行原版 `run_ruler.sh none 0 1`，仅在临时 PATH 中把 python 替换为打印错误、退出 23 的桩。结果：33 个 Python 调用全部失败，shell 仍退出 0，并打印 `ALL DONE none`。没有调用任何模型或 GPU。

建议 runner 采用可靠 pipeline 失败检查，启动时检查 cwd、依赖、输入与配置；每个节点失败写结构化 `failed` 状态，禁止生成 success marker。只添加 `set -euo pipefail` 还不足以完成数据验收：需要检查每个输出满足结果契约。

### B07：旧 marker 会放行新任务，文件行数也不是验收

位置：`benchmark/RULER/chain_ruler.sh` L9；E89/E90 relay 等待 `/tmp/e87_gpu0.log`；`b7s_guard.sh` L3–17、`b7s_ruler_relay.sh`、`b7_final_relay.sh`。

grep 只识别字符串，没有绑定本轮 run ID、上游 PID、commit、参数或成功契约。旧日志里存在 `ALL DONE none` 就能满足当前条件。等待也缺少上游失败/退出/超时的处理，既可能提前放行，也可能无限等待。

B7 LongBench 守卫使用 glob 后 `head -1`，可能读到旧文件；`wc -l >= n` 不能验证 sample ID 唯一、来源一致或内容有效。随后调用的 `run_e71_eval.py` 又按每个任务各自的 `sorted[-1]` 挑最新文件，守卫检查的文件与评分使用的文件并未绑定。

注意：此处 **`repobench` 前缀不是 bug**。当前 LongBench writer 有意把 `repobench-p` 截成 `repobench`，scorer 也有对应键；已排除“守卫因该前缀必然卡住”的误报。问题是文件版本选择与验收协议。

建议用结构化节点状态而不是 grep 日志；下游必须引用同一冻结结果 manifest。日志用于诊断，不能替代 DAG 依赖状态。

### B08：新旧结果混合、不完整样本仍出 AVG

位置：`benchmark/RULER/score_ruler.py` L33–57、L64–79；相关风险还在 LongBench `eval.py` 的全目录扫描和 `run_e71_eval.py` 的逐任务最新文件策略。

RULER scorer 按目录 glob 扫所有轮次，method key 丢弃时间标签，同一任务按排序后的最后文件覆盖。它不验证任务/长度清单、样本数或 run ID。缺任务仍出 AVG，AVG 是现有 cell 的平均，因此不同方法可能在不同任务/长度集合上被比较。

直接执行实际 scorer 的 fixture：任务 1 旧轮有 3 条错误记录，新轮只有 1 条正确记录；任务 2 只有旧轮的 3 条错误记录；其他任务和长度均缺失。scorer 返回 0，并输出两个任务的 `100/0`、AVG=50。它既接受新轮部分输出，也混入旧轮另一任务。

这证明“会接受污染”的机制，**没有证明你的某个既有分数一定被污染**。已有数据应保留，只有补齐代码、配置、样本与输出的追溯后才能决定哪些结果可继续用、哪些要重跑。

### B09：输出名字不唯一

位置：`sparse_attn/info.py` L11–13；LongBench `pred.py` L369–381；E90 relay 未传 `--t`。

TLI 名字只含 block size、K1、K2、cmp ratio 与 ABD 开关，遗漏 α/β/γ、near/far method、投影 basis、具体子空间和 per-q-head 等。实测两组参数 (.125,.375,.625) 与 (.875,.875,.5) 都得到 `tli_64_128_1024_c4_A`。如果 output directory、task、`--t` 也相同，writer 的 `open(...,"w")` 会覆盖结果。

单独的 postfix 能规避部分冲突，因此不能仅凭这一点断言 E109 扫描已覆盖；但当前仓库未找到与 E109 汇总 JSON 对应的受版本管理 launch runner，无法核实其隔离方式。E90 relay 使用固定默认 `--t`，相同子空间重跑会复用同一路径。

建议输出目录使用 UUID/run ID，配置规范化后生成 hash，完整参数保存在 manifest；不要只依赖人为 postfix 或分钟级时间标签。

## 5. 当前实验口径：哪些可用，哪些仍缺证据

| 项目 | 源码实际情况 | 对论文的影响 |
|---|---|---|
| Qwen3 LongBench 默认 runner | 13 个任务；512 与 2048 启用，1024 预测段被注释 | 是 13 任务子集，不是 LongBench v1 全 21 任务；eval 脚本仍评分 1024 目录，可能评分旧结果 |
| RULER runner | 11 任务 × 4K/8K/16K；默认每格前 100 样本 | 不是所有 RULER 任务/长度/样本；结果需明示子集，不得写完整 RULER |
| GQA 的 HF/SG 对拍 | HF L2 为每 q-head softmax 后 mean；SG 为组内 q 求和后 raw score topk | 数学上不是同一 selector；G=4 测试仅记录差异，不计失败。HF 精度与 SG 速度不能直接拼成同一实现的质量-速度结论 |
| Prefill 阶段 | HF patch 的初始长 prefix 走 dense，切出的尾部逐 token sparse；SG 长序列 extend 可全 sparse | 必须分清 prefill/decode 工作负载；同名算法不代表相同阶段或相同成本 |
| GovReport trace | `collect_trace_lb_v.py` L38 模板没有 `{context}`；L116 从头截到 target_len | 不同正文能产生相同输入。相关 trace 不足以支撑摘要任务、层校准与子空间迁移结论 |
| Qwen 的 `nope` 标签 | HF 人为把后半维称 NoPE，但模型 rotary_dim=head_dim | 这不等同真实无旋转维度；真 NoPE/混合应在真实架构上验证 |
| 逐层 α/β/γ | SG 各层 indexer 接收同一 global profile；未见逐层求解配置消费入口 | P0' 脚本是潜力检查，不是已部署的逐层参数求解器 |
| 首四层 dense | 当前相关 patch/backend 没有与所述要求对应的统一 guard | 若论文协议需要首四层 dense，必须确保精度与速度入口都执行，不能把 skip-far 当 dense |
| P0' 目标 | pre-W_O 的 head 输出误差，默认与指定冠军对比；不是同代理目标下固定参数全局最优 | 可以探索，但不能声称接近真实逐层 e2e oracle，亦不能据此量化严格收益 |
| P0' 中位数 | L632–643 偶数个 gap 取上中位点 | [0,0,0,.09,.11,.2,.3,.4] 标准中位数 .10，代码 .11，跨过 `>10%` 门槛 |

P0' 的 `far_budget=0 && far_frac>.25` 剔除规则依赖特定历史经验。应标为启发式与适用范围，而不是“数学上必崩”；它会把用户希望研究的某些层 far-skip 候选提前排除。缺失层也应阻断完整结论，不能只 warn 后继续输出总体 verdict。

对于当前正在跑的远端任务，尚缺：真实启动命令、resolved config、commit、依赖版本、GPU/并行设置、节点状态/退出码、输入 sample ID 清单、预测文件和 stderr。只有这些证据能确认它实际使用哪一个入口、是否触发本报告中的条件、数据是否已通过验收。

## 6. 建议的任务链与阻断顺序

先修正确性与数据协议，再扩方法海选。大量重复跑海选不能弥补输入错误或 selection 语义不一致。

| 节点 | 输入 | 验收/输出 | 下游放行条件 |
|---|---|---|---|
| G0 冻结运行契约 | commit、模型/Tokenizer revision、完整 config、basis/solver artifact hash、任务/长度/样本清单 | run_id + config_hash + expected_cells + sample IDs | 预检通过、无同 ID 活跃 writer |
| G1 选择器/attention 正确性 | 合法、短序列、非对齐、池不足、GQA、投影输入 | indices + valid + 唯一集合 + reference output | 正确处理无效槽位；无需误报所有变体都相等，但差异须命名清楚 |
| G2 生成与 prompt 回归 | 完整 token 序列、chat template、EOS、Samsum | 输入序列一致；输出/停止与基准协议一致 | 不丢 question，不用空正文校准，不续写首步 EOS |
| G3 受控 smoke | 小样本 dense 与目标 sparse；同模型/输入/权限 | raw predictions、退出码、stderr、sample IDs、真实预算 | 完整有效输出，无 OOM/NaN/缺样本；不能仅看完成文字 |
| G4 海选 | 开发集、冻结搜索预算、可行候选、全局 fixed 与逐层方法 | 同成本定义下的质量/速度/缓存 Pareto | 前序版本有效；不能因少读候选而虚报节省 |
| G5 独立结果验收 | 实际执行代码、配置、原始数据与契约 | usable-with-scope 或 failed/incomplete/stale | 生产者自报与 JSON pass 不能替代验收 |
| G6 冻结优胜者后终评 | 未用于搜索的测试样本/任务；全量或明确子集 | 正式表、每任务指标、误差/置信区间、成本 | 所有比较方法使用同一集合与协议；缺 cell 阻断正式 AVG |

G1 建议至少覆盖：G=1/4；S=1000/1023/1024/1025/2048/2049/4096/4097/8192；single-pool/partition；far=0/near 不足；不同 chunk/prefix；shared/per-q-head；投影/无投影；kernel/eager；index incremental rebuild 对照。先用小张量与数学 reference 验证，再做模型 smoke，避免为确认局部 bug 消耗整套 LongBench 成本。

G3/G4 记录有效选中 token 数、候选槽位数、page 数、投影与索引缓存字节、prefill/decode/求解器时间；后续生产与验收日志以短结构化摘要和证据引用传递，避免重复把全日志塞进 Agent 上下文。预算/token 节省必须用测量支撑。

修复后的数据处理建议：保留历史 raw data，标注受影响的 commit/入口/配置；B01/B02 重点重测 SG 长 prefill 的输出与质量，B03/B05 重点补边界和组合回归；B06–B09 重建 manifest 后判断哪些历史结果可追溯复用。GovReport trace 必须用含正文的新版本重采；不要从错误 trace 向逐层求解器继续传递结论。

## 7. 本轮实际检查与未检查内容

- 已完成：9 组受控检查；实际 host 区域函数、命名函数、Samsum/生成循环 AST、mask 赋值 AST；真实 RULER shell 和 scorer 的隔离运行；41 Python/32 shell 语法检查；远端分支 SHA 回读；SGLang 工作树保持 clean。
- 未完成：PyTorch/GPU 全调用链、CUDA graph、多请求/多 GPU 真机回归、真实模型 e2e、远端活跃进程与正在产出的文件审查。没有把 NumPy 反例冒充 kernel 实测。
- 结果：可确认应修问题与触发条件，尚不能确认你的全部既有数据失效，也不能批准当前任务链为完整正确。应先按版本和 manifest 确认影响范围。

不可变源码入口：[被审查提交](https://github.com/Chosen-David/sglang/tree/9049ba8759896e5efc13f287c442fb72d5e1b563)。

复现命令：在与只读仓库同级的本报告目录执行 `python3 reproduce_by_gpt.py`。脚本不 import SGLang、不启动模型、不修改仓库；依赖 NumPy，只写自身目录下的 `evidence_by_gpt.json`。该文件保存源码 SHA256、各反例数值与验证等级。
