# Prefill/Decode 双阶段 Indexer 口径矩阵实验设计（E114b，2026-10-08 rev2）

> 状态：设计稿 rev2。触发条件：**E109 海选收官（AVG5 五任务全齐 + 选举落袋）**。
> rev2 变更：①按用户指令完整吸收 GPT 建议 `agent_doc/advice/TwoLevel_Prefill_Decode_实验流程建议_by_gpt.md` 八项关键要求（§0 对照表给出落点）；②实现现状更新——实现 agent 稀疏 prefill **已落码**（工作树未提交：`pred.py` L71/L345-346 `--tli-sparse-prefill` + env `TLI_SPARSE_PREFILL`；`qwen3_attn_patch.py` L59-77 prefill 分支门控；`sparse_attn/ops/eager_prefill.py` 149 行 `sparse_prefill_attn`（chunk=2048 共享选择）；`test_sparse_prefill_impl.py` 在写）。调研基线 two-level-attention @ 0d39b6a32。
> **融合原则（冲突裁决）**：以用户既有指令为主线——「decode-only 决冠军 → 四格矩阵看精度损失 → 定叙事」；GPT 文档补协议细则（候选保留、复筛、归因量、分母、判读、前置门、叙事锚点）。GPT 中与主线重叠的要求视为细则化（如「不预设 decode-only 胜出」= 用户三分支叙事本义，非推翻主线）；GPT 的 serving 侧细则（batch 维度、服务端 TTFT、多卡）归 #140 sglang 臂，不进本 HF 矩阵；GPT 的长度三档初筛已被 E109 海选覆盖，不重做。

---

## 0. GPT 八项要求吸收对照表

| # | GPT 要求（出处） | 本设计落点 | 与用户主线的融合 |
|---|---|---|---|
| 1 | Decode 初筛保留 2~3 Pareto 候选，四格矩阵对冠军+1 对照跑（§5.1） | §3.1 | 用户「decode 决冠军」细化为「决 Pareto 前沿，冠军+对照进矩阵」；并列时选实现简单者 |
| 2 | Prefill 独立小规模复筛（§5.2） | §3.2 | 最小落地：1 个「decode 弱但构建便宜」method，hq/mu n=50 P-only 冒烟，无潜力即停 |
| 3 | 四模式真实独立，自身轨迹 KV，首 token 边界记录（§4.1） | §1（边界）+ §4.1 | P-only/P+D 全部新跑不复用 dense KV 落盘（本设计原則，GPT 确认） |
| 4 | 归因量 Δ_P/Δ_D/Δ_PD + 描述性交互量 I（§6.1） | §4.3 | 新增归因小节；分任务配对、原始分数不相加 |
| 5 | 性能四层分母 S_idx/S_P/S_D/S_req；D-only 建索引成本归位；固定工作量与自然生成分开（§8） | §6 | HF 口径如实写：decode 索引在 decode 期构建、prefill 段无索引成本；自然生成为主，固定工作量并入 #140 |
| 6 | 正确性前置门：因果断言/退化参考/补齐规则/实际稀疏日志（阶段 A） | §5.1 | 与 #161 池不足审计交叉（既有设计 §8 风险 2 升级为前置门） |
| 7 | ε_j 容忍度预设：开发 1pp 筛选值 + 正式 bootstrap CI（§7.2） | §5.2 | 用户 0.3 = AVG5 决赛口径实用阈值，与 1pp 开发筛选值分层共存；「不显著≠等效」入纪律 |
| 8 | 论文定位预设规则表（§11.2） | §7 | GPT 表格原文引用作三分支判据锚点 |

---

## 1. 口径基础：HF harness 下的「阶段」定义与首 token 边界

`benchmark/LongBench/pred.py` 生成循环（L234-264）天然分三段，这是本实验「阶段」的物理定义：

| 段 | 代码位置 | patch 分支判定 | 现状口径 | 稀疏 prefill 开启后 |
|---|---|---|---|---|
| ① context prefill | L235-239 附近，context 一次多 token forward | `get_seq_length == input_shape[1]`（qwen3_attn_patch.py L50） | dense（仅 observe_prefill_q 旁路） | `TLI_SPARSE_PREFILL=1` 时走 chunk 化稀疏 |
| ② question-stage 逐 token | question 循环，q_input 每 token 一次 forward | else 分支（decode 路径） | sparse（走 indexer） | 不变（sparse） |
| ③ generation 逐 token | 生成循环 | else 分支 | sparse | 不变 |

**首 token 边界（GPT §4.1 要求显式记录）**：第一个生成 token = 最后一个 question-stage forward 的 `output.logits[:, -1, :].argmax`（pred.py 手写循环；① context prefill 的 logits 从不被采样消费）。因此「模型内 TTFT」在本 harness = ① + ② 段耗时（不是 ① 单独）；TPOT 分母 = 输出 token 数 − 1。GPT 通用表述「首 token 来自最后一次 prefill forward」在本 harness 的对应物是 ② 段末（② 走 decode/sparse 路径）——论文写作时如实说明，避免与 sglang serving（chunked prefill + decode）混淆；必要时用 sglang 路径补一条验证臂。

**四格定义**：「decode-only」= ①dense + ②③sparse；「prefill-only」= ①sparse + ②③dense；「both」= 全 sparse；Baseline = 全 dense（`--method none`）。

---

## 2. Flag 方案与实现 agent env 门控的收敛

### 2.1 目标接口：单变量 `TLI_SPARSE_STAGES`（env）≡ `--tli_sparse_stages`（argparse），四值 `{decode, prefill, both, none}`，默认 `decode`

取舍理由（rev1 保留）：

- 本实验自由度就是 2×2 阶段开关；多变量组合会产生第二个 FullKV 入口（「PREFILL=0 且 decode 强制 dense」与 `--method none` 语义重叠），审稿与自查都易踩「两个 baseline 入口行为不一致」的坑（B09 命名冲突教训同族）。
- **E109 隔离纪律要求默认逐位不变**：`default=decode` = 现行为（prefill dense + decode sparse），所有在跑链零扰动。

### 2.2 与已落码 `TLI_SPARSE_PREFILL` 的收敛（兼容实现 agent env 门控现状）

实现 agent 已落码的形态：`pred.py` `--tli-sparse-prefill`（store_true）→ 模型加载前设 env `TLI_SPARSE_PREFILL=1`；patch 层 prefill 分支逐 forward 读 env（+ B=1、attention_mask None 守卫）→ `sparse_prefill_attn`；decode 分支无门控（恒 sparse）。**该形态天然覆盖 `decode`（默认）与 `both`（env=1）两值，缺 `prefill`（decode 转 dense）与 `none`。**

收敛方案（在已落码之上做别名层，不推翻其实现）：

1. **优先级链**：`--tli_sparse_stages` 显式 > env `TLI_SPARSE_STAGES` > env `TLI_SPARSE_PREFILL=1`（别名，解析为 `both`）> 默认 `decode`。
2. **patch 层**：新增模块级 `_resolve_stages()`（读 argparse 属性或两个 env，按上述优先级归一为 `{'prefill','decode'}` 集合）；prefill 分支的 env 判定（L64）收敛为 `'prefill' in _resolve_stages(...)`——`TLI_SPARSE_PREFILL=1` 经别名层仍解析为含 prefill，**实现 agent 现有 env 门控行为逐位保留**；decode 分支前置守卫 `'decode' not in stages` 时走 dense 回退（§2.3）。
3. **pred.py 层**：新增 `--tli_sparse_stages {decode,prefill,both,none}`；`--tli-sparse-prefill` 保留为兼容别名（映射 `both`），避免破坏实现 agent 的调用形态与在写单测。
4. `stages=none` 须与 `--method none` 的 FullKV 做 T0b 逐位对拍后才允许出现在任何链上（第二入口一致性门）。

### 2.3 prefill-only 的 decode-dense 最小改法

改动点在 `sparse_attn/patches/qwen3_attn_patch.py`（约 +12 行，rev1 方案保留）：

1. decode 分支（else 入口）前置守卫：`'decode' not in stages` → 复用 prefill 分支同款 `attention_interface(self, query_states, key_states, value_states, attention_mask, ..., scaling=self.scaling, sliding_window=self.sliding_window, **kwargs)` 调用形态，`attn_weights=None` 后 reshape 返回。要点：decode 时 q_len=1，`attention_interface`（eager/sdpa/fa）原生支持；无 padding 时 `attention_mask` 为 None，无 B05 类切轴风险；**必须绕开** else 分支现有 transpose/unpad/`prepare_cu_seqlens` 逻辑（那是 eager_decoding_attn 的 [1,tq,H,D] 布局约定）；`indexer.clear()` 维持只在 prefill 分支调用。
2. `llama3_attn_patch.py` 同款改动后置（E109/E114 全 Qwen3 臂，Llama 不在本矩阵）。

**回归保护单测（红-绿纪律，随实现一并交）**：
- T0a：`stages=decode`（默认）输出与 HEAD 逐位一致；
- T0b：`stages=none` 输出与 `--method none` 不注册 patch 的 FullKV 逐位一致（q_len=1 dense 路径 + sliding_window 传递正确性）；
- T1：`stages=prefill` 时 decode 段不触 indexer（断言 `prepare_mask` 调用次数为 0），prefill 段触 chunk 化选择（次数 = ⌈S/2048⌉·层数量级）。

---

## 3. 候选保留与 Prefill 独立复筛（GPT §5 吸收）

### 3.1 Decode 初筛保留 Pareto 前沿 2~3 候选，不单冠军垄断

E109 海选收官时（AVG5 五任务全齐 + 选举落袋），按以下规则保留候选：

- **保留维度**：质量最高者（AVG5 第一）、均衡者（分任务无短板：hq/mu/代码/摘要四族均不显著掉）、更快者（构建开销低或 kernel 化程度高——本项目中「快」的 proxy = L1 侧构建代价（minmax 上界块统计 < 聚类迭代）与 E113b Triton kernel 覆盖度（sim_greedy 已 kernel 化））。
- **并列判据（GPT §5.1）**：候选间 AVG5 差落在噪声内（E100 实测五任务 n=200 下单任务 CI 半宽 ±1~3 分量级 → AVG5 分辨力 ~±1）先视为并列；重复测量后仍并列，**选实现更简单者**。
- **四格矩阵执行口径**：**冠军 + 1 个代表对照**进首轮四格（对照优先选与冠军 method 族不同者，以覆盖「method 族 × 阶段」交互；两候选双卡并行每卡一臂，墙钟不翻倍）。第 3 候选仅在第 2 候选四格结果与冠军出现 method 族级分歧时补跑。
- 超参：全部沿用各候选海选原值 (α,β,γ)×method；`--tli_enable_layer_skip false`（与海选同口径，D' 层跳过是正交消融不混入）。

### 3.2 Prefill 独立小规模复筛（GPT §5.2 最小落地）

目的：防 decode 排名偏差淘汰 prefill 特性候选（构建便宜的 method 在 prefill 段的 chunk 级重复选择中有结构性优势——每 chunk 一次选择，构建代价被放大 ⌈S/2048⌉·层数 倍）。

- **复筛对象（1 个）**：从 E109 数据里挑「decode 排名靠后但构建最便宜」的 method。首选 **mminmax 系**（L1 只需块 kmin/kmax 在线统计，构建代价最低；decode 侧精度排名靠后最可能被淘汰）。kmeans 系（cavgkm/cclusterkm）构建反而贵（聚类迭代），仅在 mminmax 出现正信号时才考虑追加。
- **协议**：hq/mu n=50 P-only 冒烟（`stages=prefill`），与 FullKV 同 50 样本配对（FullKV pred_1024 取前 50 行对拍）。
- **停止判据（诚实标注噪声）**：n=50 下 avg2 噪声 ~±2-3pp，1pp 筛选值不可分辨——「无潜力即停」= avg2 损失 ≥3pp（显著超噪声）；「有潜力」= 损失 ≲1pp 量级 → 并入四格候选池（候选总量 ≤3，GPT §5.2 约束）。
- 无潜力即停，不为「凑候选」扩测（GPT §10 停止条件）。

---

## 4. 四格矩阵：精度协议、模式独立性与归因量

### 4.1 四模式真实独立（GPT §4.1 铁律）

| 格 | prefill(①) | decode(②③) | 数据来源 | 新增 GPU 成本 |
|---|---|---|---|---|
| FullKV | dense | dense | **复用** `exp/results_longbench/Qwen3-8B/pred_1024/*-none-*.jsonl`（五任务 200/200/200/200/500 全齐，同 pred.py 同 config-path 口径） | 0 |
| decode-only | dense | sparse | **复用** E109 各候选海选五任务 pred（跑分 JSON 现成；2 候选均有） | 0 |
| prefill-only | sparse | dense | 新跑 | ~8-11 GPU·h/候选 |
| both | sparse | sparse | 新跑 | ~8-11 GPU·h/候选 |

- **独立性铁律**：P-only/P+D 一律自身轨迹生成（fresh KV 由稀疏 prefill 逐层产生并持续更新），**不得复用 Baseline 落盘的 dense KV 做精度评测**——否则掩盖稀疏 prefill 对隐藏状态与后续层 KV 的影响。固定 dense Q/K/V 回放仅限 TLI_DEBUG 局部诊断，不进精度数字。
- D-only 的 decode 索引在 ② 段首个 forward 经 `prepare_mask → prepare_index` 构建（详见 §6 归位），① 段无索引成本——P-only 的 decode 转 dense 后索引完全不构建，两者状态路径真实不同。
- 首轮 2 候选 × {prefill-only, both} ≈ **32-44 GPU·h，本地双卡并行（每卡一候选）墙钟 ~10-11h**（gov_report ~2.9h 瓶颈）。§3.2 复筛冒烟另计 ~1-2 GPU·h。
- 决赛口径（分支 A 触发时）：both vs decode-only 在 **13 任务 LongBench 全量**重跑（n=200/任务，AVG 的 CI 收窄到 ~±1，与 E100 同判读力），预留 ~2 臂 × 20-30 GPU·h 备用。
- 样本口径对齐海选：qasper/hotpotqa/musique/gov_report 各 200，repobench-p 500；固定样本清单（所有格同一份，GPT §3.3）。

### 4.2 与实现 agent 代码的对齐点

`sparse_prefill_attn` 已实现的语义（eager_prefill.py 模块注释）：chunk=2048 末行 q 代表选择 + chunk 内共享 + 行因果下三角交集（行 r 只可见 [0, r]）；首 chunk（起始 < K2）与短行 dense；sink [0,128) 强制保留；仅 B=1 且 attention_mask None 的质量路径口径，其余回退 dense。**矩阵臂全部满足质量路径前提（LongBench 单样本 B=1 无 padding）；回退 dense 的 fallback 比例日志见 §5.1。**

### 4.3 归因量定义（GPT §6.1）

对同一候选同配置（子空间、层掩码、预算规则一致，仅改 P/D 开关；预算依赖历史长度时保持同一函数形式），对越高越好的分任务分数定义：

```
Δ_P  = A_0 − A_P        （Baseline → P-only：稀疏 prefill 改变上下文表示后，完整 decode 保留多少质量）
Δ_D  = A_0 − A_D        （Baseline → D-only：dense prefill 后稀疏 decode 的影响）
Δ_PD = A_0 − A_PD       （Baseline → P+D：双阶段联合损失）
I    = Δ_PD − Δ_P − Δ_D （描述性交互量：I>0 表示该分数尺度上联合损失超过两项之和）
```

- **I 标注纪律**：固定实验配置下的描述性交互量，不是一般误差传播定律或机制证明（GPT 原文口径）。
- **报告纪律**：分任务配对报告（每任务逐样本配对差 + bootstrap CI）；**不同任务的原始分数不相加**（AVG5/AVG13 只作汇总排序，正式判读看分任务配对区间）；关键任务（musique far 检索多跳，E85f/E74 判据预测最敏感）的明显退化不能被平均分掩盖。
- 归因表落袋：`exp/trace/results/e114b_stage_matrix.json`，每候选 × 每格 × 每任务 {score, Δ, CI} + I。

---

## 5. 判读标准与正确性前置门

### 5.1 正确性前置门（GPT 阶段 A；先过门再跑大规模精度，防「绕过稀疏路径」的假冠军）

实现 agent 单测 `test_sparse_prefill_impl.py` 之上，矩阵臂放全量前须过：

1. **因果无泄漏断言**：chunk 共享选择下早期行的选集 ⊆ 该行因果可见集（sel ≤ row_pos，实现已按「选集 ∩ 行因果下三角」编写——单测须用随机样本断言每行有效选集上界；子空间在线统计/块摘要/L1/L2 打分是逐 forward 局部量，不跨 chunk 泄露，但**共享选择的早期行因果 clamp 必须显式断言**，参照 sglang S1 修复的 `sel<S-nq+r+1` 模式）。
2. **退化参考单测（好单测，GPT §4.2）**：预算覆盖全部有效历史（K2 ≥ S）时稀疏 prefill 输出 ≡ full-attention 参考（实现的首 chunk dense 已覆盖短序列等价性；补 K2≥S 中长序列一例）。
3. **候选不足/重复补齐规则**：near 池不足（高 γ 低 β 候选 < topk）与 prefill 共享选择的交互——**等 #161 池不足审计（`research/docs/pool_starvation_audit_20261008.md`，尚未产出）落地后交叉核对，其结论必须先于矩阵臂放全量**；sink/window 与动态候选重叠只计一次的语义核对。
4. **执行路径日志**：各格实际稀疏层数、稀疏 chunk 数 / dense 回退 chunk 数（fallback 比例）、实际选中 token 数——用于验证「真的在跑稀疏路径」（GPT：若大部分 query 回退 dense，「精度不掉」不能算验证通过）。正式计时轮关闭重型 profiler。
5. T0a/T0b/T1 回归单测（§2.3）全绿是任何矩阵臂上 GPU 的前置条件。

### 5.2 ε_j 容忍度分层（GPT §7.2 + 用户口径融合）

- **开发筛选值 ε_dev = 主指标绝对降 1 个百分点**（GPT 建议，可调整的筛选值不是最终标准）。用于 §3.2 复筛冒烟与四格矩阵期分任务方向判读。
- **用户 0.3 阈值 = AVG5/AVG13 决赛口径的「不损」实用阈值**（分支 A 触发条件：13 任务全量 both vs decode-only |ΔAVG|<0.3）——两者分层共存，文档与论文均按此表述，不得混用。
- **正式判读 = 配对 bootstrap 95% CI（E100 管线现成，B=10000 逐样本）**：只有损失 CI 上界 ≤ 预设容忍度才可声称「在该范围内保持质量」；区间过宽 = 证据不足；**「差异不显著」≠「等效」**；多次随机生成的题目内输出相关，不当独立样本扩大 n。
- 海选五任务矩阵（n=200/任务）判读力 ~±1~3 分/任务：只做方向判定（Δ 与任务族结构，如 far 检索任务 mu 是否系统性掉分）；决赛走 13 任务全量。
- ε_j 在正式测试前预设，不从结果倒推（GPT §7.2 纪律）。

---

## 6. 性能计量：四层分母与阶段归位（GPT §8 吸收）

### 6.1 四层分母定义

| 层级 | 计时范围 | 分母参照 | 本项目首轮落法 |
|---|---|---|---|
| S_idx = T_idx,ref / T_idx,ours | 在线打分/粗筛/细筛/Top-k/去重 | 同语义全扫描/单级 selector | **FullKV 无 selector → S_idx = N/A**（不能拿 full attention 时间除 Indexer 时间充数）；首轮报告选择开销绝对量与占比（§6.3），形式比值留给全扫描细筛对照消融（GPT 消融 1） |
| S_P = T_P,ref / T_P,ours | 完整 prefill 段（chunked 则累加全部 chunk） | 同模型同设置 FullKV | 计时副本四格实测 |
| S_D = TPOT_ref / TPOT_ours | decode 步（模型内 TPOT） | 同上 | TPOT = 生成段耗时/(输出 token 数−1)（首 token 在 ② 段末产生，见 §1 边界）；长生成不用初始位置单步 TPOT 代表全程，逐 token 段内报告均值+尾段 |
| S_req = T_req,ref / T_req,ours | 整请求（含在线建索引） | 同上 | 整请求 wall time 直接实测，不拼接 S_P 与 S_D |

### 6.2 成本归位（GPT §8.1 铁律：每请求构建/聚类不能伪装为离线成本）

- **D-only**：decode 索引在 ② 段首个 forward 构建（`prepare_mask → prepare_index` 全量建索引），成本计入该格的 decode/整请求耗时；**① 段（prefill）无索引成本**（仅 static_pair 模式的 observe_prefill_q 旁路统计——no-op 级，如实写）。
- **P-only / both**：prefill 段每 chunk 一次选择的成本计入 prefill 段；both 的 decode 增量选择成本计入 decode 段。
- 离线共享校准开销（PCA basis 等）单列，不进请求耗时。
- 各比值必须在相同 workload 上计算；不因某配置 OOM 悄悄降 batch 后继续算同表加速比（本矩阵 B=1 固定，天然满足）。

### 6.3 计时实现与固定工作量/自然生成分开（与 #140 合并排程）

1. **自然生成为主（质量+耗时同轮）**：pred.py 生成循环天然三段（§1）。计时不在生产码上改——**在 `exp/trace/` 下建带计时副本**（`pred_staged_timing.py`，cp + 在 ①②③ 段边界插 `torch.cuda.synchronize()` + `time.perf_counter()`），逐样本落 JSON：`{task, sample_id, ctx_tokens, gen_tokens, prefill_s, question_s, gen_s, total_s, sparse_chunk_n, fallback_chunk_n}`。遵守 e135 审计计时纪律（真实 token 数、EOS、timestamp 全落盘，事后中位数/复算）。四格 × 2 候选 × n=32 ≈ ~8 GPU·h，复用矩阵臂 GPU 排队窗口续跑。**报告真实输出长度、截断率与总耗时**（自然生成协议，GPT §8.3——防止「少生成导致假加速」）。
2. **Indexer 选择开销旁路**：与实现 agent 合并时顺带加 env `TLI_PROFILE_SELECT=1`（默认关）——`prepare_mask`/`sparse_prefill_attn` 选择段前后 synchronize+perf_counter 累积，计时副本末尾汇总（选择段占 prefill/decode 各自百分比）。不改默认行为。备选零侵入：TLI_DEBUG 离线重放单测层测选择开销（只作归因旁证不进 e2e 数字）。
3. **固定工作量测速并入 #140**（sglang 三臂 e2e 分段重测：F1-F3 修复后、TP2、≥3 次重复取中位数、固定输入/输出长度协议）——本 HF 矩阵不单独做固定工作量臂，论文速度表两口径并列（HF 自然生成 wall time + sglang serving kernel 级），互不替代。
4. **噪声最低要求（GPT §8.5）**：可控空闲 GPU、预热稳定后取数、≥5 独立重复轮（计时副本 n=32 × 重复）、交错安排基线与候选、报告中位数与轮次级加速比区间；小于波动范围的差异判「未决」重测；不用 profiler 内数字作最终延迟。并行精度评测可以，延迟基准避免同卡干扰。
5. 排程：E109 冠军出炉 → §3.2 复筛冒烟（~1h）→ 五任务矩阵双卡（~10h，每卡一候选）→ 计时副本同窗口续跑 → both 判「不损」后 13 任务全量 + #140 sglang 机器同/次窗口。

---

## 7. 论文叙事三分支（GPT §11.2 预设规则表作判据锚点）

**GPT §11.2 表格原文（判据锚点，测试结果 → 建议主张 → 必须保留的边界）**：

| 测试结果 | 建议主张 | 必须保留的边界 |
|---|---|---|
| P、D 都有可靠净收益，P+D 质量达标 | 双阶段高效 Indexer／稀疏 attention | 分别给 S_P、S_D 与整请求收益 |
| 只有 D 稳定获益 | Decode 为核心 | 报告 prefill 结果与开销／质量原因 |
| 只有 P 稳定获益 | Prefill 为核心 | 报告 decode 不划算的长度与 batch 范围 |
| 两阶段需不同预算／策略 | 按阶段配置的统一框架 | 规则在开发集固定，报告组合精度和切换成本 |
| 只在长上下文获益 | 有长度阈值的方法 | 短上下文回退及阈值附近测量 |
| 只有 Indexer 快，模型收益不足 | Indexer 算子／选择器优化 | 不宣称模型或整请求 E2E 加速 |
| 精度区间太宽或质量下降超过阈值 | 暂无充分保精度证据 | 扩充针对性验证或调整方法，不能称无损 |

本设计三分支与上表映射（用户主线「四格矩阵看精度损失 → 定叙事」= GPT「最终定位由质量和净收益共同决定，不预设 decode-only 或双阶段必胜」）：

- **分支 A（both 不损：13 任务全量 |ΔAVG|<0.3 于 decode-only）→ GPT 行 1「双阶段」**：方法节新增「prefill chunk 化共享选择」小节；主表/速度表用 both 臂（含 S_P 与 TTFT 收益，四层分母分开给）；同口径陈述锚 MoBA/Quest/ClusterKV（Fig.12/13 显式画 prefill 段）与 HISA（TTFT/TPOT）；加分句：长输出/短上下文场景 prefill 占比上升，双阶段是部署完备性要求（Quest §3.1 decode 占比 86%+ 数字反用作边界说明）。
- **分支 B（decode-only 损失小、prefill 稀疏损失大）→ GPT 行 2「Decode 为核心」**：主表 decode-only；prefill-only/both 降为「精度-速度 Pareto 边界」消融行，**prefill 结果与开销/质量原因如实报告**（GPT 边界列要求）；先例引 Quest（主加速图只计 decode）/SparQ/FASA/Double Sparsity；机理解释落在早期行共享选择的因果可见集受限（§8 风险 2）。
- **分支 C（prefill-only 显著差）** → 负结果资产 + GPT 行 7 边界纪律：HF harness 下 ②③ 段已占计算大头，decode 侧选择收益兑现最充分；「阶段维度消融」小节支撑「为什么 decode 优先」的设计决策；同时反证 both 臂的正交性（若 both≈decode-only，prefill 稀疏不伤 decode 侧选择质量得到隔离证明）。
- **即便最终选 decode-only，也保留 P-only 与 P+D 的有界验证结果**（GPT §11.2 收尾纪律：它们解释适用范围并证明未未经测试就排除另一阶段）——本矩阵设计天然满足（四格全跑）。
- 若两阶段最适预算/策略不同（GPT 行 4）：超出本轮范围，记 future work（阶段差异化配置组合须再完整测 P+D 精度，两个单阶段候选各自合格不保证组合合格）。
- 写作路由按用户既定路由表：起草 → scientific-paper-generation skill；修订/风格门 → paper-writing skill。

---

## 8. 风险与依赖

1. **prefill 稀疏 O(S²/chunk) 重建拖慢 gov_report（最大风险）**：E107a 已实锤 TLI prefill 每 chunk 全量重建是 sglang 路径最大单项根因（对 Quest 输 2.5× 主因）；HF 稀疏 prefill 同款（chunk=2048、每 chunk 全量重建索引，eager_prefill.py 已注明「质量路径可接受」）。gov_report（S~10-15K → 5-8 chunk × O(S) 重建 × 36 层）prefill 段 wall time 可能不降反升。缓解：①放全量前 gov_report n=8 冒烟测 prefill 段 wall time（>30% 拖慢触发预案 b）；②预案 b = prefill-only/both 先跑 hq/mu/qasper/repobench 四任务，gov_report 标注「prefill 段成本待 F1 增量化后复测」；③论文如实报 prefill 稀疏的当前成本边界（GPT 行 6「只有 Indexer 快」的表述纪律同样适用反向情形：prefill 段变慢不掩饰）。
2. **早期行选集 ∩ 因果的池不足交互（#161 缺陷族交汇）**：chunk 共享选择早期行因果可见 token 少于 chunk 末行；near 池不足（高 γ 低 β 候选 < topk）与之叠加。§5.1 前置门 1/3 已设断言与审计交叉，**#161 结论先于矩阵臂放全量**。
3. **E109 在跑链隔离**：新 flag 默认 `decode`（逐位=现行为；实现 agent 已按 default-off 落码，门控在 B=1+mask None 守卫下其余情形回退 dense，在跑链零扰动已满足）；矩阵臂用独立 `pred_postfix`（建议 `_E114b_STM_` 前缀）+ 独立 OUTROOT，SKIP 幂等协议天然隔离；`--tli_sparse_stages` 合入主树须过 T0a 后才允许出现在任何链加载路径。
4. **decode-dense 路径语义**：q_len=1 走 `attention_interface` 时 sliding_window 参数照传（Qwen3 SWA 配置时行为须与 FullKV 对拍，T0b 覆盖）；bf16 eager 精度与 FullKV 逐位一致是 prefill-only 格 baseline 有效性前提。
5. **判读力风险**：§5.2——海选口径噪声 ±1~3 分/任务，决赛必须 13 任务全量；musique（far 检索多跳）是 prefill 稀疏预期最敏感任务，矩阵期重点盯分任务配对 CI 而非只看 AVG。
6. **长输出任务边界（GPT §7.1 对照）**：本矩阵 LongBench 短/中输出为主，GPT 建议的 MATH500/AIME 长生成类不在本轮——论文泛化主张相应限定（受控检索诊断由 RULER 补），不做未经测试的通用性声明。

---

## 9. 执行 checklist（依赖排序）

| # | 步骤 | 前置条件 | 资源 | 预估 |
|---|---|---|---|---|
| 0 | E109 收官：AVG5 全齐 + **Pareto 2~3 候选选举落袋**（§3.1 规则）＋实现 agent 稀疏 prefill 合并（报告+default-off 代码入主树） | 四机链收官 | — | 触发门 |
| 1 | `--tli_sparse_stages` 别名层收敛（§2.2，兼容已落码 env 门控）+ T0a/T0b/T1 单测红-绿 | 0 | CPU 级 | ~1h |
| 2 | 正确性前置门 §5.1 全项（因果断言/K2≥S 退化单测/补齐规则）＋#161 池不足审计交叉核对 | 1 | CPU 级 | ~0.5-1h |
| 3 | §3.2 prefill 复筛冒烟：mminmax 系 hq/mu n=50 P-only vs FullKV 同样本配对 | 1 | 1 GPU | ~1-2h |
| 4 | 冒烟：冠军臂 × {prefill, both} × qasper n=8 + gov_report n=8（正确性 + prefill 段 wall time 健康度，风险 1 缓解①） | 1,2 | 1 GPU | ~0.5h |
| 5 | 五任务矩阵：2 候选 × {prefill-only, both}（200/500 样本，双卡每卡一候选；复用 FullKV/decode-only） | 3,4 | 本地双卡 | ~10-11h 墙钟 |
| 6 | 分段计时副本：4 格 × 2 候选 × n=32 自然生成（§6.3 协议）+ TLI_PROFILE_SELECT 汇总 | 5 | 同窗口 | ~8 GPU·h |
| 7 | 判读落袋：分任务 Δ_P/Δ_D/Δ_PD/I + 配对 bootstrap CI → `e114b_stage_matrix.json` + commit push | 5 | CPU | ~0.5h |
| 8 | （条件）both≈decode-only → 13 任务全量 both（+prefill-only 对照视分支）＋#140 三臂重测同/次窗口排程 | 7 | 双卡+sglang 机 | ~40-60 GPU·h |
| 9 | 论文对应小节起草（分支 A/B/C 按 GPT §11.2 锚点 + §7 映射路由 skill） | 7 | CPU | — |

GPU 总预算：复筛 ~1-2 + 首轮矩阵 ~32-44 + 计时 ~8 + 全量备选 ~40-60 GPU·h。
