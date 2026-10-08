# Prefill/Decode 双阶段 Indexer 口径矩阵实验设计（E114b，2026-10-08）

> 状态：设计稿（只读调研产出，未跑 GPU、未改生产代码）。触发条件：**E109 海选冠军出炉（AVG5 五任务全齐 + 选举落袋）**。
> 调研基线：two-level-attention @ 0d39b6a32（C3 修复后）；实现 agent 的稀疏 prefill（TLI_SPARSE_PREFILL）**尚未落码**（全仓 grep 零痕迹、无专属 worktree，`research/docs/sparse_prefill_impl_20261008.md` 尚不存在——本设计按其既定语义「chunk=2048 共享选择」编写，落码后按 §1.3 收敛）。

---

## 0. 口径基础：HF harness 下的「阶段」到底指什么

`benchmark/LongBench/pred.py` 的生成循环（L234-264）天然分三段，这是本实验「阶段」的物理定义：

| 段 | 代码位置 | patch 分支判定 | 现状口径 |
|---|---|---|---|
| ① context prefill | L235-239，context 一次多 token forward | `get_seq_length == input_shape[1]`（qwen3_attn_patch.py L50） | **dense**（仅 observe_prefill_q 旁路） |
| ② question-stage 逐 token | L241-247，q_input 每 token 一次 forward | else 分支（L68） | **sparse**（走 indexer） |
| ③ generation 逐 token | L241-264 | else 分支 | **sparse** |

即：**「decode-only」现状 = ①dense + ②③sparse；「prefill-only」= ①sparse + ②③dense；「both」= 全 sparse**。注意 ② 在 serving 语义上属 incremental prefill，但本 harness 中走 decode 路径——论文写作时须如实说明该口径（§4 风险），避免与 sglang serving（chunked prefill + decode）混淆；必要时用 sglang 路径补一条验证臂。

---

## 1. Flag 方案

### 1.1 结论：单变量 `TLI_SPARSE_STAGES`（env）≡ `--tli_sparse_stages`（argparse），四值 `{decode, prefill, both, none}`，默认 `decode`

取舍理由：

- **本实验的自由度本来就是 2×2 阶段开关**，多变量组合（`TLI_SPARSE_PREFILL` × `TLI_DENSE_DECODE`）会产生 4 组合，其中「PREFILL=0 且 DENSE_DECODE=1」= 全 dense，与 `--method none`（不注册 patch）语义重叠，是第二个 FullKV 入口——审稿与自查都易踩「两个 baseline 入口行为不一致」的坑（本仓 B09 命名冲突教训同族）。
- **E109 隔离纪律要求默认逐位不变**：`default=decode` = 现行为（prefill dense + decode sparse），所有在跑链零扰动；不新增 `default=None` 之类的悬空分支。
- 与既有 `--tli_*` argparse 族对齐（E109 链全部命令行传参，见 daemon 脚本 `/tmp/e109_gpu_daemon_local.sh` 调用形态），同时保留 env 形态给 patch 层无 args 上下文时使用。

优先级（防双开关打架）：`--tli_sparse_stages` 显式 > env `TLI_SPARSE_STAGES` > 别名（见 §1.3）> 默认 `decode`。

### 1.2 prefill-only 的 decode-dense 最小改法（只写方案不改码）

改动点全部在 `sparse_attn/patches/qwen3_attn_patch.py`（约 +12 行）：

1. **新 helper `_resolve_stages(args)`**（模块级）：读 `getattr(args, 'tli_sparse_stages', None)` 或 `os.environ.get('TLI_SPARSE_STAGES')`，归一为 `{'prefill','decode'}` 集合；未设 → `{'decode'}`。
2. **decode 分支（L68 else 入口）前置守卫**：`if 'decode' not in _resolve_stages(self.indexer.args):` → 直接复用 prefill 分支同款调用形态（L57-67 的 `attention_interface(self, query_states, key_states, value_states, attention_mask, ..., scaling=self.scaling, sliding_window=self.sliding_window, **kwargs)`），`attn_weights = None`，然后 reshape 返回。要点：
   - decode 时 q_len=1，`attention_interface`（eager/sdpa/fa）均原生支持；`attention_mask` 无 padding 时为 None，无 B05 类切轴风险；
   - **必须绕开** else 分支现有的 transpose/unpad/`prepare_cu_seqlens` 逻辑（那是 eager_decoding_attn 的 [1,tq,H,D] 布局约定，dense 路径不需要）；
   - `indexer.clear()` 维持只在 prefill 分支调用（每次新请求仍重置索引态，C3 修复后含 `_last_q`）。
3. **prefill 分支（L50-67）**：`'prefill' in stages` 时改走实现 agent 的 chunk 化稀疏 prefill 路径（其 `TLI_SPARSE_PREFILL` 门控点收敛为 stages 判定，见 §1.3）；`observe_prefill_q` 旁路无条件保留（与 static_pair 正交）。
4. `llama3_attn_patch.py` 同款改动可后置（当前 E109/E114 全 Qwen3 臂，Llama 不在本矩阵）。

**回归保护单测（红-绿纪律，随实现一并交）**：
- T0a：`stages=decode`（默认）输出与 HEAD 逐位一致；
- T0b：`stages=none` 输出与「`--method none` 不注册 patch」的 FullKV 逐位一致（q_len=1 dense 路径正确性）；
- T1：`stages=prefill` 时 decode 段不触 indexer（断言 `prepare_mask` 调用次数为 0），prefill 段触 chunk 化选择（次数 = ⌈S/2048⌉·层数量级）。

### 1.3 与实现 agent `TLI_SPARSE_PREFILL` 的收敛

实现 agent 尚未合并，两阶段收敛路径：

- **若其先落码**（`TLI_SPARSE_PREFILL` env bool，prefill 分支门控）：在其基础上做别名层——`TLI_SPARSE_STAGES` 显式时覆盖一切；未设 STAGES 但 `TLI_SPARSE_PREFILL=1` 时解析为 `both`；两者全缺省 = `decode`。`TLI_DENSE_DECODE` 不再需要（`prefill` 值即其语义）。落地时读 `research/docs/sparse_prefill_impl_20261008.md`，把其「prefill 分支改造点」与本文 §1.2 第 3 点对齐，避免两套 prefill 分支。
- **若本设计先行实现**：prefill 分支留 `NotImplementedError` 占位，等 agent 的 chunk=2048 共享选择实现插入（接口：接收 q/k/v + chunk 选择的块级 mask，返回 attn_output）。

---

## 2. 四格矩阵：精度协议与 GPU 成本

样本口径对齐海选：qasper/hotpotqa/musique/gov_report 各 200，repobench-p 500（EXPN 同 daemon 脚本）；判读主指标 = AVG5（五任务均分）+ 分任务 Δ 表。

| 格 | prefill | decode | 数据来源 | 新增 GPU 成本 |
|---|---|---|---|---|
| FullKV | dense | dense | **复用** `exp/results_longbench/Qwen3-8B/pred_1024/*-none-*.jsonl`（五任务行数已核：200/200/200/200/500 全齐，同 pred.py 同 config-path 口径） | 0 |
| decode-only | dense | sparse | **复用** E109 冠军臂海选五任务 pred（跑分 JSON 现成） | 0 |
| prefill-only | sparse | dense | 新跑 | ~8-11 GPU·h（gov_report ~2.9h 瓶颈 + qasper ~2h + hq/mu 各 ~1.5h + repobench ~1.5h） |
| both | sparse | sparse | 新跑 | ~8-11 GPU·h（prefill 段替换 dense，wall time 见 §5 风险 1） |

- **首轮合计 ~16-22 GPU·h，本地双卡 ~10h 墙钟**（与 E109 收官后的空闲窗口契合）。
- **判读阈值纪律（诚实标注）**：|ΔAVG5|<0.3 视为「不损」是用户定的实用阈值，但 E100 bootstrap 实测五任务 n=200 下单任务 CI 半宽 ±1~3 分（musique ±2.9、hotpotqa ±3.5 量级）——**0.3 阈值在海选口径下低于噪声分辨力**。因此：海选五任务矩阵只做方向判定（Δ 与任务族结构，如 far 检索任务 mu/passage 是否系统性掉分）；**决赛口径 = both vs decode-only 在 13 任务 LongBench 全量重跑**（n=200/任务，13 任务 AVG 的 CI 收窄到 ~±1，与 E100 同判读力），预留 ~2 臂 × 20-30 GPU·h 备用预算（双卡 1-2 天）。
- 超参：全部沿用 E109 冠军臂 (α,β,γ)×method 组合原值，`--tli_enable_layer_skip false`（与海选同口径，D' 层跳过是正交消融不混入）。

---

## 3. 分段计时设计（与 #140 合并排程）

三层口径铁律（kernel microbench 与 e2e 两层都测都诚实报告）在本矩阵的落法：

1. **Harness 级 prefill/decode 分段（零生产码改动）**：pred.py 生成循环天然三段（§0）。计时不改 `benchmark/LongBench/pred.py`（生产码），而是**在 `exp/trace/` 下建带计时的副本脚本**（如 `pred_staged_timing.py`，cp + 在 L235 前后 / L241 循环前后 / L251 循环前后插 `torch.cuda.synchronize()` + `time.perf_counter()`），逐样本落 JSON：`{task, sample_id, ctx_tokens, gen_tokens, prefill_s, question_s, gen_s, total_s}`。遵守 e135 审计的计时纪律：真实 token 数、EOS、timestamp、prompt token 数全落盘，事后可中位数/复算。四格各跑一次计时副本（可复用矩阵臂的同一 GPU 排队窗口，sample 子集 n=32 即可，~1 GPU·h/格）。
2. **Indexer 选择开销旁路**：现有 `TLI_DEBUG`（tli_indexer.py L194 等多处）可 dump 选择中间量但不计时。方案：与实现 agent 合并时顺带加 env `TLI_PROFILE_SELECT=1`（默认关）——在 `prepare_mask` 前后 `synchronize+perf_counter` 累积到 metrics，pred 副本脚本末尾汇总输出（选择段占 prefill/decode 各自的百分比）。**不改默认行为、在跑链不受影响**。备选（零侵入）：TLI_DEBUG 重放类脚本离线单测层单独测选择开销（不进 e2e 数字，只作归因旁证）。
3. **与 #140（三臂 e2e 分段重测）合并**：#140 是 **sglang serving 路径**（F1-F3 修复后、TP2、93.9/121.1/61.6 ms/step 口径，e135 P1 纪律：≥3 次重复取中位数）；本矩阵是 **HF two-level 路径**——两口径并列进论文速度表（HF 端到端 wall time + serving kernel 级），互不替代。排程建议：E109 冠军出炉 → 本矩阵五任务跑本地双卡（~10h）→ 同窗口或紧随 #140 走 sglang 机器 → both 判「不损」后 13 任务全量两口径收尾。**一次 GPU 排队窗口做完 §2 矩阵 + §3.1 计时副本**（计时副本插在矩阵臂跑完后同 GPU 续跑，无需二次排队）。

---

## 4. 论文叙事三分支预案

同口径正当性素材（内部知识 + 已核调研文档 `research/docs/稀疏Attention两阶段测速调研_20261006.md`）：MoBA chunk 化 prefill 共享选择 + decode 双阶段；NSA 两阶段（但其 prefill 侧只有 training forward 实测）；Quest prefill/decode 都选择（原论文主加速图仍只计 decode，§3.1 给 decode 占比 86%+）；DSA/LongCat/HISA 有 TTFT/TPOT 两阶段系统级证据。

- **分支 A（both 不损，|ΔAVG|<0.3 于 13 任务全量口径）** → 双阶段主张：方法节新增「prefill chunk 化共享选择」小节；主表/速度表用 both 臂（含 TTFT 收益）；同口径陈述锚 MoBA/Quest/ClusterKV（Fig.12/13 显式画 prefill 段）与 HISA（TTFT/TPOT）；加分句：长输出/短上下文场景 prefill 占比上升，双阶段是部署完备性要求（Quest 86% 数字反用作边界说明）。
- **分支 B（decode-only 损失小、prefill 稀疏损失大）** → 「decode 是索引器质量收益所在，prefill 稀疏化为速度可选项」：主表 decode-only；prefill-only/both 降为「精度-速度 Pareto 边界」消融行，prefill 精度边界如实报；同口径先例引 Quest（主加速图只计 decode）/SparQ/FASA/Double Sparsity（调研已核：这条后训练 KV 检索路线主要优化 decode）；机理解释落在早期行共享选择的因果可见集受限（§5 风险 2）。
- **分支 C（prefill-only 显著差）** → 负结果资产：HF harness 下 ②③ 段（question+generation 全走 decode 路径）已占计算大头，decode 侧选择收益兑现最充分；作为「阶段维度消融」小节支撑「为什么 decode 优先」的设计决策；同时反证 both 臂的「prefill 稀疏不伤 decode 侧选择质量」（若 both≈decode-only，则 prefill 稀疏的正交性得到隔离证明）。
- 写作路由按用户既定路由表：起草 → scientific-paper-generation skill；修订/风格门 → paper-writing skill。

---

## 5. 风险与依赖

1. **prefill 稀疏 O(S²/chunk) 重建拖慢 gov_report（最大风险）**：E107a 已实锤 TLI prefill 每 chunk 全量重建是 sglang 路径最大单项根因（对 Quest 输 2.5× 的主因）；HF 稀疏 prefill 若同款（chunk=2048 共享选择、每 chunk 全量重建索引），gov_report（S~10-15K → 5-8 chunk × O(S) 重建 × 36 层）prefill 段 wall time 可能不降反升，并拖慢矩阵臂。缓解：①放全量前先 gov_report n=8 冒烟测 prefill 段 wall time（>30% 拖慢即触发预案 b）；②预案 b = prefill-only/both 先跑 hq/mu/qasper/repobench 四任务，gov_report 标注「prefill 段成本待 F1 增量化后复测」（F1 是 sglang 路径已验证资产可平移）；③论文如实报 prefill 稀疏的当前成本边界。
2. **早期行选集 ∩ 因果的池不足交互（#161 缺陷族交汇）**：chunk 共享选择 = chunk 内早期行的块选择由代表 q 代做，早期行因果可见 token 少于 chunk 末行 → 选集可能越出早期行因果界（sglang S1 哨兵/因果泄漏族 + T1 far 池溢出族的同构风险在 HF 路径重现）。设计验收断言：随机 3 样本上稀疏 prefill 臂每行有效选集 ⊆ 该行因果可见集（sel < row_pos+1 语义，参照 backend.py S1 修复的 `sel<S-nq+r+1` 模式）；near 池不足（高 γ 低 β 时候选 < topk）与 prefill 共享选择的交互，等 `research/docs/pool_starvation_audit_20261008.md`（#161，尚未产出）落地后交叉核对——**该审计的结论必须先于矩阵臂放全量**。
3. **E109 在跑链隔离**：新 flag 默认 `decode`（逐位=现行为）；不动任何在跑脚本；矩阵臂用独立 `pred_postfix`（建议 `_E114b_STM_` 前缀）+ 独立 OUTROOT（如 `~/tmp_e114b_stage_matrix`），SKIP 幂等协议天然隔离；稀疏 prefill 分支 default-off 合入主树须过 T0a 单测后才允许出现在任何链的加载路径上。
4. **decode-dense 路径语义**：q_len=1 走 `attention_interface` 时 sliding_window 参数照传（Qwen3 有 SWA 配置时行为须与 FullKV `--method none` 对拍，T0b 单测覆盖）；bf16 eager 精度与 FullKV 逐位一致性是「prefill-only 格」的 baseline 有效性前提。
5. **判读力风险**：见 §2——海选口径 0.3 阈值低于噪声分辨力，决赛必须 13 任务全量；musique（far 检索多跳）预期是 prefill 稀疏最敏感任务（E85f/E74 判据可预测），五任务矩阵期重点盯。

---

## 6. 执行 checklist（依赖排序）

| # | 步骤 | 前置条件 | 资源 | 预估 |
|---|---|---|---|---|
| 0 | E109 冠军出炉（AVG5 全齐+选举）＋实现 agent 稀疏 prefill 合并（报告+default-off 代码入主树） | 四机链收官 | — | 触发门 |
| 1 | `--tli_sparse_stages` flag 收敛实现 + T0a/T0b/T1 单测红-绿 | 0 | CPU 级 | ~1h（实现 agent 或主会话派工） |
| 2 | #161 pool starvation 审计交叉核对（早期行因果断言 + 池不足交互） | 1 | CPU 级 | ~0.5h |
| 3 | 冒烟：冠军臂 × {prefill, both} × qasper n=8 + gov_report n=8（正确性 + prefill 段 wall time 健康度） | 1,2 | 1 GPU | ~0.5h |
| 4 | 五任务矩阵：prefill-only + both 两臂（200/500 样本，复用 FullKV/decode-only） | 3 | 本地双卡 | ~10h 墙钟 |
| 5 | 分段计时副本四格（n=32，同窗口续跑）+ TLI_PROFILE_SELECT 汇总 | 4 | 同窗口 | ~4 GPU·h |
| 6 | 判读落袋：AVG5+分任务 Δ 表 → `exp/trace/results/e114b_stage_matrix.json` + commit push | 4 | CPU | ~0.5h |
| 7 | （条件）both≈decode-only → 13 任务全量 both（+prefill-only 对照视分支）＋#140 三臂重测同/次窗口排程 | 6 | 双卡+sglang 机 | ~40-60 GPU·h |
| 8 | 论文对应小节起草（分支 A/B/C 按判读路由 skill） | 6 | CPU | — |

GPU 总预算：首轮 ~16-22 GPU·h + 计时 ~4 GPU·h + 全量备选 ~40-60 GPU·h。
