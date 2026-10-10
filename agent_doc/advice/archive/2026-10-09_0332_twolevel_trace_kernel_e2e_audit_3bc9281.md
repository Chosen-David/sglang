# TwoLevel trace、kernel 与 e2e 验收审计（3bc9281）

## 审查目标与范围

- **审查源码基线**：`3bc92811c7682d5393ab2c736692c2b3dc8b33a6`（`origin/two-level-indexer`，2026-10-09 03:32 CST 拉取并在独立干净 worktree 复核）。
- **增量范围**：重点复查 `f26197dfc..3bc92811c` 对 0233 报告三项修复，并轮换审查 kernel microbench 与 E71/E110 e2e 执行链。`8aa3a38b5` 是数据提交，`f3b524153` 是 advice-only 提交；均未当作核心源码变化。
- **实际检查**：trace 采集完成门禁、幂等身份、L1/L2 kernel 输入/计时/正确性口径、归档 JSON 与论文/TASK 声明、e2e shell 失败传播、最新 E109 选择结果受既有 near/SWA 边界问题的影响。
- **环境**：Python 3.12.14，Linux 6.18.44 x86_64；当前环境无 PyTorch、模型权重和可用 GPU，因此未运行 CUDA kernel、LongBench 或真实模型。完成了 Python 编译、Shell 语法、纯 Python/Bash 最小反例和静态调用链核对。

## 结论摘要

| ID | 状态 | 严重度 | 位置 | 结论 |
|---|---|---:|---|---|
| `TL-TRACE-ASSERT-OPT-007` | **confirmed（CPU stub A/B）** | P1 | `two-level-attention/exp/trace/collect_trace_lb_v.py:132-135,189-202` | 三个生产完成门禁使用 `assert`；`python -O` 会删除检查，零 hook 命中仍写出声明完整的 `meta.json`。 |
| `TL-KBENCH-CONTRACT-009` | **confirmed（源码/归档/论文绑定）** | P1 | `two-level-attention/benchmark/efficiency/benchmark_mha_kernel.py:59-82,132-204`；`paper/TLI_paper.tex:341-343`；`TASK.md:390-398` | 论文/TASK 的 1.46×/2.77×/5.09× 来自随机、全维 BF16 L1 microbench；脚本既不是 tail32-4bit L1，也不是 trace replay，且无正确性 oracle。现有数值不能证明论文所称生产配置。 |
| `TL-E2E-FAILMASK-008` | **confirmed（Bash 反例）** | P1 验收缺陷 | `two-level-attention/exp/trace/e110_ccluster_e2e_draft.sh:74-86,89-111`；同类 `run_e71.sh:15-25,31-49`、`run_scripts/run_e71_b7_full.sh:8-19` | Python 通过 `2>&1 | tail -2` 执行且无 `pipefail`/`PIPESTATUS`；任务失败可被 `tail` 的 0 覆盖，worker 与总脚本仍打印 DONE 并返回成功。 |
| `TL-BENCH-FLOPS-010` | **confirmed（CPU 复算）** | P2 | `benchmark_mha_kernel.py:121-125,200-204,237-247` | QK 与 OV 把 MAC 数写成 FLOP，漏掉乘加 ×2；归档 dense 和 sparse-final-attention GFLOPs/TFLOPs 均精确低报 2 倍。延迟与 speedup 不受影响。 |
| `TL-TRACE-COMPLETE-004` | **partial fixed / residual confirmed** | P1 | `collect_trace_lb_v.py:47-68,144-202` | 缺层、缺文件、损坏文件在普通模式已拒绝；但跳过判据仍未绑定模型、tokenizer、输入、target length 或实现 hash，也未核对 shape/dtype/S。变更实验仍可复用旧 trace。 |
| `TL-BOUNDARY-NEAR-SWA-001` | **open；新增结果受影响** | P1（旧 ID） | `sparse_attn/indexer/tli_indexer.py:795-806`；`exp/trace/results/e109_screen4_selection.json:235-241` | 最新宣称冠军的 E109 α=.25 臂仍运行在既有 near/SWA 错分边界上；分数可保留为旧实现观测，不能冻结为修正契约下冠军。 |

## 复现与证据

### 1. `TL-TRACE-ASSERT-OPT-007`：优化解释器移除数据完整性门禁

采集器用 `assert layer_set`、`assert not bad_layers` 与 `assert not missing` 承担输入合法性和“所有请求层均命中 hook 后才发布 meta”的生产门禁。Python 文档语义下，`-O` 会移除这些语句。

以 fake model/tokenizer/torch 加载原模块，令模型 forward 成功但 **0 次 hook 命中**；除解释器是否使用 `-O` 外输入完全相同，目标层为 `[1]`：

```text
普通 Python：AssertionError；logical_rc=1；meta_exists=False
python -O  ：saved 0 layers；meta_exists=True；meta.layers=[1]
```

`-O` 路径写出的 meta 摘要：

```json
{"S": 3, "n_layers": 2, "prompt": "lb_hotpotqa_0", "layers": [1], "with_v": true}
```

这会重新产生“没有 layer 文件、完成标记却声明完整”的原故障。采集器文件 SHA256：`ba0cc94a2b71bbc31a77a80ecde636e038de4077bc2a370d0b3cc96e54eabd17`。

### 2. `TL-TRACE-COMPLETE-004`：幂等跳过仍不具实验身份

`_sample_complete(pdir, layer_set)` 在构造文本和 tokenization 之前调用，只接收路径与层集合；meta 也只记录 `S/n_layers/prompt/layers/with_v/question/answer`。以同层文件和可回读五键为条件，存储身份为 `old-model/target_len=32768/input_sha=old`、请求身份为 `new-model/target_len=1024/input_sha=new` 时，抽取原函数控制流的 CPU stub 仍返回 `true`。

因此本次修复只覆盖“物理文件存在/可读/五键齐全”，没有覆盖“文件属于当前实验”。模型、tokenizer、输入内容、截断长度或实现改变后，旧 trace 仍可能被静默跳过并污染分析。

### 3. `TL-KBENCH-CONTRACT-009`：归档 speedup 与论文所称配置不一致

归档 `two-level-attention/exp/results_efficiency/benchmark_mha_20260923_124217.json` 是 `TASK.md:392-398` 三个 speedup 的数据源；`paper/TLI_paper.tex:343` 又把这组三个精确数值描述为“trace 重放口径”和“生产配置（tail32-4bit L1）”。但生成脚本：

- 用未固定 seed 的 `torch.randn` 分别生成 q/k/v、L1 q/min/max、L2 q/k，彼此不是同一语义输入派生；
- `level1_q` 与 `level1_k_min/max` 的最后一维都是 `dim_k=128`、dtype 为 BF16，源码没有 tail32 或 4-bit 路径；
- 计时链包含 L1、L2 与最终 sparse attention，但没有 `allclose`、reference selection、索引范围/去重检查或输出正确性断言；
- 归档 JSON 只有 `configs` 与 `timestamp`，未保存 seed、设备/驱动/CUDA/PyTorch、源码/输入 hash 和逐次原始时延。

因此当前 latency/speedup 只能标为“旧版全维 BF16、随机输入 microbench 的观测”；它不等于 trace replay，也不能直接证明当前 tail32-4bit 生产选择链。此结论不否定 cudaEvent 计时本身：脚本有 warmup、事件计时与 `torch.cuda.synchronize()`。

文件 SHA256：

- 脚本：`a4d77b2cea8bef9130262dbe91fb3823e093dc522ae6b026c5d6b993e279048f`
- 归档 JSON：`211018db0af52cd112551ba18cb5167a0418a52818d0de38fd7f8c070b44df20`

### 4. `TL-BENCH-FLOPS-010`：MAC/FLOP 混用

脚本对 QK 和 OV 都计算 `M*N*K`，但标签写作 FLOPs。若采用标准“一次乘法 + 一次加法 = 2 FLOP”口径，应乘 2。CPU 复算 32K 档：

```text
dense  archived 0.134217728 GFLOP -> standard 0.268435456 GFLOP
sparse archived 0.004194304 GFLOP -> standard 0.008388608 GFLOP
```

此外 sparse wall time 是 L1+L2+attention 总和，而其 FLOP 分子只计最终 attention；即使补 ×2，当前 sparse TFLOPS 也不能解释为全链硬件利用率。该错误不改变保存的 ms、speedup 或 throughput ratio。

### 5. `TL-E2E-FAILMASK-008`：管道吞掉任务失败

E110/E71 都采用：

```bash
python ... 2>&1 | tail -2
```

脚本未启用 `set -o pipefail`，也未读取 `${PIPESTATUS[0]}`。纯 Bash 反例让两个等价 Python 子进程分别退出 23/29，两个 worker 仍打印 DONE，`wait` 返回 0，最终继续打印 `E110_LOCAL_DONE`。`bash -n` 只能证明语法有效，不能发现此退出码语义错误。

E110 文件明确仍是 draft，未见正式数据受影响；当前 E71 归档的 13 个任务均有 AVG，未发现缺项直接证据。但失败历史与重跑次数未持久化，不能据现存汇总反推所有原任务都一次成功。

### 6. 既有 near/SWA 缺陷对新增 E109 冠军的影响

`tli_indexer.py:795-806` 虽用 `mid_len=S-sink-swa` 计算 near 长度，却用 `(S-near_len_dyn)//bs` 算 near 起点，没有先减去 SWA。按常用反例 `S=4352,sink=128,swa=128,bs=64,alpha=.25`：契约 near 为 1024 token，当前实现实际 near 为 896，另 128 token 被错划入 far；32K 同样少一个 128-token SWA 区间。

`E109_mavg_a0.25_b0.125_g0.625` 的 AVG5=46.95 可保留为该旧实现下观测，但修复边界后应重跑 winner 及全部 `alpha>0,beta>0` 相关臂。selection JSON 的 30 臂五项字段及 AVG5 算术均正确；没有发现新的缺样本假成功证据。其 SHA256 为 `f78852989c240c0a6bc76a8c1c6e7a8b26e5a448acbb68cdd0099f363113ac52`。

## 对已跑数据与论文结论的影响

1. **Kernel 速度主张需降级并重跑**：1.46×/2.77×/5.09× 的延迟值可作为旧脚本观测保留，但当前证据不能支撑“tail32-4bit L1 / trace replay / 当前生产配置”的绑定。FLOP/TFLOPS 字段还需整体修正口径。
2. **Trace 数据存在跨实验陈旧复用风险**：3bc 修复了普通模式下缺层/损坏文件，但 `-O` 可绕过完成门禁，且同层旧数据仍可跨模型、输入和 target length 被跳过。由此产出的分析应先核验目录身份与层文件闭包。
3. **E110 无正式结果受影响证据；E71 没有现存缺项证据**：但脚本完成状态不可靠，不能把 DONE 当作任务成功证据。
4. **E109 新冠军未冻结**：受开放的 near/SWA 边界缺陷影响，边界修正前后不是同一配置语义。

## 建议修复与最小重测

1. 将三处 `assert` 改为显式 `ValueError`/`RuntimeError`；增加普通与 `python -O` 两条测试，均要求非法层/零 hook 命中非零退出且不写 meta。
2. 为 trace 建 manifest：模型与 revision、config/tokenizer hash、输入行/格式化文本/token ids hash、target length、采集器 SHA、层集合、每层 shape/dtype/S/hash。跳过前验证完整依赖闭包；写入采用临时文件 + 原子 rename，meta 最后发布。
3. 从当前真实 dispatcher/config 驱动 kernel benchmark，使用固定 trace/input snapshot，并实际走 tail32+4bit L1；每一级用独立 reference 验证索引、范围、tie/重复与 sparse attention 输出，正确性通过后再计时。
4. 固定 seed，保存逐次 paired latency、Git/实现/input hash、GPU/驱动/CUDA/PyTorch 与完整配置；旧 JSON 显式标为 legacy full-dim BF16 random microbench。
5. 明确 MAC/FLOP 定义；采用 FLOP 时补 ×2。分别报告组件 FLOPs/latency 与全链 wall time，避免用只含最终 attention 的分子解释整条选择链 TFLOPS。
6. e2e 脚本启用严格模式与 `pipefail`，或显式检查 `${PIPESTATUS[0]}`；每任务持久化 rc、输出行数、输入/实现 hash 与 failure manifest。并发汇总逐 PID 检查，任一失败则总脚本非零且不得发布 DONE。
7. 先修复并验证 near/SWA 边界，再重跑 E109 winner 与受影响臂；保留旧结果但标注实现契约，禁止跨实现直接排名。

## 旧发现复查

- `TL-TRACE-LOGITS-005`：**静态 fixed/rechecked**。forward 传入 `logits_to_keep=1`，消费者只读末 token；无 GPU/权重，峰值显存与数值等价未实测。
- `TL-DEBUG-FAR-EMPTY-006`：**静态 fixed/rechecked**。空 `i_f` 使用 `-1` 哨兵，并新增 debug+far-empty 测试；因无 PyTorch 未执行测试。
- `TL-TRACE-COMPLETE-004`：普通解释器下缺层/缺文件/损坏文件已补强，但因 `assert` 与身份未绑定仅判定 **partial fixed**。
- `TL-BOUNDARY-NEAR-SWA-001`：仍 open；本次不重复造新 ID，只新增 E109 结果影响说明。

## 实际验证与未覆盖项

- `python -m compileall -q`：采集器、indexer、边界测试、kernel benchmark 均通过。
- `git diff --check f26197dfc..3bc92811c`：通过。
- `bash -n`：所查 E71/E110 脚本通过；退出码反例确认 fail masking。
- trace 零 hook stub 在普通 Python 与 `python -O` 做同输入 A/B，结果如上。
- 未执行 PyTorch 单测、CUDA kernel、真实模型 trace、LongBench、GPU 性能或数值精度实验；因此没有声称性能提升，也没有据静态检查声称其余路径无 bug。

## 下一检查点

修复后优先复验：① 普通/`-O` trace 完成与身份闭包；② production tail32-4bit trace-driven kernel correctness-before-timing；③ e2e 单任务失败、双 worker 一失败、恢复重跑与汇总拒收；④ near/SWA 边界修复后的 E109 同条件重跑。
