# GPT 审查报告（sglang_twolevel_audit_by_gpt）逐项核查与实测验证_by_kimi3

核查日期：2026-10-08。核查对象：`agent_doc/advice/sglang_twolevel_audit_by_gpt.md`
（审查固定 SHA `ee256310b2e8898d9ed3607da9c325b872fc10ff`）。

## 核查环境与基准

- 本地 HEAD：`0bf18338bf1dd79ed6ce5059e0a3208f1a64022b`（`two-level-indexer` 分支），
  比审查 SHA 多 2 个提交（E109 入账 + LB v2 接入）。`git diff ee25631..HEAD --stat`
  确认**被审查的全部 6 个文件零改动**，审查的行号与结论在当前 HEAD 仍然适用。
- 与 GPT 审查环境的关键差异：**本机有 torch 2.8.0+cu128 与 2×H20 GPU**（GPT 审查
  环境无 torch，只有纯 Python 算术见证）。因此本次核查把 GPT 的「算术/控制流见证」
  全部升级为**真实 torch 执行**：直接调用仓库真实源码类/方法（不复制其逻辑），或对
  源码逐字做 AST/行级提取后 exec。
- 实测方法分三档（均强于 GPT 的 paraphrase 见证）：
  1. **真实方法调用**：`TLIIndexer.select` / `select_decode_batched` /
     two-level `TLIIndexer._near_token_score` 直接跑（C1/C2/C3）；
  2. **逐字 exec 源码块**：按行号/AST 从 `.py` 文件提取原语句执行（C3 合并块、T1/T2/T3）；
  3. **GPU 对照**：H20 上跑 fused triton kernel 验证「kernel 已修」的对比结论（C1 对照）。

## 总结：六项主结论全部属实，均有实测证据

| ID | GPT 审查结论 | 核查裁决 | 实测关键数字 |
|---|---|---|---|
| C1/P1 | eager select() near 不足时返回 -inf 下标当真 token | **属实（真实 torch 复现）** | 576 个 -inf 下标被返回，全部 < S 通过消费端 valid |
| C1 对照 | fused kernel（S-T002 bug1）已修、不计为未修复 | **属实（H20 GPU 实测）** | 同场景 fused 路径返回 576 个哨兵 S |
| C2/P1 | graph decode 短行 sink∩swa 重复计权，0.75≠0.6 | **属实（真实 torch 复现）** | 96 位置重复；sparse=0.75 vs dense=0.6 |
| C3/P1 | near 簇分缺 softmax_scale，top-k 边界排名翻转 | **属实（真实 torch 复现）** | 放大 11.31×；实际选簇段、对齐后应选尾段 |
| T1/P1 | E98 mass 扫描把池外 -inf 槽位 scatter 为 True | **属实（真实 torch 复现）** | 网格点 (α.5,β.875,γ0) 过约束；1536 个池外位置 |
| T2/P1 | E104 不同任务集 AVG 直接相减可虚构优势 | **属实（逐字 exec 复现）** | 真实差 0，脚本输出 delta.AVG=+50 |
| T3/P2 | KVCache baseline 复用只看行数，改配置静默 SKIP | **属实（逐字 exec 复现）** | max-capacity 1024→2048 仍 SKIP |

**对「分析本身有没有 bug」的回答：未发现 GPT 审查的事实性错误。** 有 4 处表述
可以加强或升级（见末节），其中 2 处经本次实测从「存疑/待核实」升级为「已确认」。

---

## 逐项核查记录

### C1（P1）：默认 eager decode 把 -inf top-k 槽位当真实 token —— 属实

**源码核对**（当前 HEAD 与审查 SHA 一致）：

- `python/sglang/srt/layers/attention/tli/indexer.py:654-662`：`i_f`/`i_n` 直接取
  `torch.topk(...).indices` 拼接返回，不按分数过滤、不写哨兵；`:662` 的未分区
  返回同样无过滤。与审查引用逐字一致。
- 可达链：`backend.py:614-635`（单稀疏请求或 `SGLANG_TLI_BATCH_SELECT=0` 走
  per-request eager）、`config.py:62`（`use_l2_kernel=False` 默认）、
  `indexer.py:589-590`（`q_agg=max` 旁路 fused）均属实。
- 消费端 `backend.py:983` `valid = sel < seq_v` 只看上界，不查分数，属实。
- 旁证：`indexer.py:1795-1796` 注释自认「eager 路径此处会选出 -inf 垃圾实位置，
  批量版采用更安全的哨兵口径」。

**实测**（直接调用真实 `TLIIndexer.select`，默认 profile，S=16384，构造 L1 选中
far 块 2..129 + 强制滑窗块，near 合法池恰 192 token、配额 768）：

```json
{"sel_shape": [1, 1024], "near_quota": 768, "near_valid_pool_size": 192,
 "far_invalid": 0, "near_invalid_indices_returned": 576,
 "near_selected_with_minus_inf_score": 576,
 "all_returned_indices_below_S_pass_consumer_valid": true}
```

复刻源码 `fine` 矩阵 gather 选中槽位分数，576 个确为 `-inf`；且全部下标 `< S`，
通过消费端唯一 validity 检查。GPT 的「至少 576」精确成立（本构造恰为 576）。

**GPU 对照（验证「kernel 已修、勿重复报bug」）**：同构造切 `use_l2_kernel=True`
在 H20 上跑 fused `tli_l2_partition_topk`：

```json
{"path": "fused tli_l2_partition_topk", "sel_shape": [1, 1024],
 "sentinel_S_slots": 576, "below_S_slots": 448}
```

fused 路径正确填 576 个哨兵 S。审查「eager 未修 / kernel 已修」的对比定性准确。
另核实 `test_b1_b2_fix.py:35-66` 确实只直接测 fused kernel，不覆盖 eager 分支，
审查「测试为什么漏掉」的判断属实。

### C2（P1）：TASK.md graph decode 短行 sink∩swa 重复计权 —— 属实

**源码核对**：

- `indexer.py:2468-2495`（`_select_decode_taskmd` 内）：`sink_out`=[0,sink_tok) 与
  `forced_out`=swa 窗独立拼接，无去重、无短行 identity 回退。属实。
- 对照 `indexer.py:1666-1677`（`_select_batched_taskmd`，prefill 批量）：有
  identity+哨兵修复；decode 版没有。属实。
- 路由链：`decode_cuda_graph_runner.py:716-718` 调 `veto_cuda_graph`；
  `backend.py:397-414` 只 veto `token_budget < L ≤ dense_threshold`（S=160 不触发）；
  `backend.py:653-689`（`_forward_decode_graph`）统一走 `select_decode_batched`；
  `indexer.py:1817-1818` 转发 `_select_decode_taskmd`。全链属实。
- eager 非图路径短行走 dense（`backend.py:560`），故本 bug 为 **graph 路径特有**，
  与审查的条件限定一致。

**实测**（env `ALPHA=.125 BETA=.125 GAMMA=.375` 进 taskmd 分区模式，直接调用真实
`select_decode_batched`，S=160）：

```json
{"sel_shape": [1, 1, 536], "valid_lanes": 256, "unique_positions": 160,
 "n_duplicated_positions": 96,
 "duplicated_range_matches_sink_swa_overlap": true,
 "mid_lanes_all_sentinel": 280,
 "dense_output": 0.6000000238418579, "graph_sparse_output": 0.75}
```

256 条 valid lane 只覆盖 160 个位置，重复的恰为 sink∩swa=[32,128) 共 96 个；
按消费端 `backend.py:1019-1022` 的 softmax 语义（Q=0→均匀、重叠区 V=1）算得
sparse 输出 0.75 ≠ dense 0.6，与 GPT 见证数字一致（本次为真实 torch 执行，非
纯算术）。

**补充精化（审查未点明）**：重复仅在 **128 < S < 256**（`sink_tok+swa_tok=256`）
窗口发生——S≤128 时 sink 越界槽位转哨兵、S≥256 时 sink 与 swa 不重叠。这解释了
`test_tli_m5.py:317-359` 为何漏掉：它只测 S=600/1500，S=600 时两区本就不重叠，
且其 mass-coverage 指标对重复槽位不敏感。审查对测试缺口的定性属实。

### C3（P1）：near 簇分未乘 softmax_scale，混合池 top-k 排名翻转 —— 属实

**源码核对**：

- `two-level-attention/sparse_attn/indexer/tli_indexer.py:605-606`：`_last_q` 缓存
  的是**未缩放** q；`:640,678`：`score_fine` 用 `q * softmax_scale`；`:722-728`
  `_near_token_score` 用未缩放 q 与簇心点积（只修正 GQA mean，无 scale）；
  `:1048-1060` 把簇分直接覆盖进 `sf_g` 切片后同池 top-k。注释声称「量纲一致」，
  实际相差 `softmax_scale` 倍。属实。
- 可达性：`:432` `near_hi = (S - swa) // bs * bs` 块对齐下取整，消费端
  `swa_lo_tok = S - swa` 为精确位置，非对齐步必然产生未覆盖回退尾段 → 混合池。
  属实。

**实测**（`object.__new__` 绕过 `__init__`，**直接调用真实方法**
`TLIIndexer._near_token_score`；再**逐字 exec 源码 1048-1060 行**合并块；构造
簇覆盖 token 原始点积 1.0、尾段 token 原始点积 2.0）：

```json
{"softmax_scale": 0.08838834764831845,
 "cluster_score_as_coded": 0.9999998807907104,
 "tail_score_sf_g": 0.1767766922712326,
 "cluster_score_scale_aligned": 0.0883883386850357,
 "inflation_ratio": 11.313708305358887,
 "top1_as_coded": "covered_cluster_token",
 "top1_scale_aligned": "tail_token"}
```

实际代码比较 1.0 vs 0.1768（偏选簇段），统一 scale 后应比较 0.0884 vs 0.1768
（应选尾段）：top-1 排名翻转，放大倍数恰为 √128≈11.31。与 GPT 的
`cluster_scale_witness.json` 数字完全一致，本次为真实 torch 张量执行。

### T1（P1）：E98 mass 扫描把 coarse pool 外 -inf 槽位当有效 token —— 属实

**实测**（逐字 exec 源码 106-123 行预算/约束块 + AST 提取 `select_sub` 函数体
exec，真实 torch topk）：

```json
{"grid_point": {"mid_len": 8192, "alpha": 0.5, "beta": 0.875, "gamma": 0.0},
 "constraint_block_skipped": false,
 "nb_near": 56, "nt_near": 0, "nt_far": 2048, "near_L": 4096, "far_L": 4096,
 "far_pool_capacity_tokens": 512, "far_selected_total": 2048,
 "selected_with_minus_inf_score_outside_pool": 1536}
```

该网格点通过脚本全部约束（far 约束只查 `nt_far > far_L` 即 2048>4096，不查池
容量），far 池仅 8 页×64=512 token 却 topk 2048，实测恰 1536 个池外 -inf 位置被
`scatter True` 计入 cov_mass。与 GPT 数字一致。

### T2（P1）：E104 不同任务集 AVG 直接相减虚构优势 —— 属实

**实测**（逐字 exec 源码 48-49 行 AVG 赋值与 54-57 行 delta 循环；构造两臂真实
结果完全相同 easy=100/hard=0、仅完成进度不同）：

```json
{"main_AVG_over_1_completed_task": 100.0,
 "ref_AVG_over_2_completed_tasks": 50.0,
 "reported_delta_AVG": 50.0,
 "true_paired_delta_on_common_tasks": 0.0}
```

共同任务真实差值为 0，脚本输出 `delta.AVG=+50`。属实（输出 note 仍写
「11 任务×n=100」，与审查引用一致）。

### T3（P2）：KVCache baseline 复用不含实验参数 —— 属实

**实测**（逐字 exec 源码 168-179 行复用块，伪造 200 行旧输出后以不同
`max_capacity` 调用）：

```json
{"rerun_with_max_capacity_2048": "SKIPPED", "fresh_task": null}
```

复用键只含 `method/output_suffix/task` + `行数>=n`；`max_capacity/window_size/
kernel_size/pooling/sink_guard/split_question/attn_impl` 均不参与。1024 跑出的旧
输出使 2048 实验在 `get_pred` 前直接 `SKIP ... return`，属实（确定控制流缺陷，
本次为真实执行源码语句）。

---

## 次级观察的核查

| GPT 次级观察 | 核查结果 |
|---|---|
| `benchmark/LongBench/pred.py` 无条件删 question 首 token，「待配置核实」 | **升级为已确认 bug**。当前文件行号已漂移到 `pred.py:204-207`（lbv2 提交所致），逻辑不变；`pred_kvcf.py:87-89` 同款。实测 Qwen3-8B tokenizer（实验脚本实际 MODEL_PATH）：`add_bos_token=False`、`bos=None`，`tokenizer('Question: What...')` 首 token 是真实内容 `'Question'`，`[:, 1:]` 后变 `': What is the answer?'`——**删除的是真实 prompt 内容，不是 BOS**。所有 qasper/hotpotqa 等 split-question 任务的 question stage 输入均被污染 |
| EOS 首 token 仍继续 decode（pred.py / pred_kvcf.py:99-109） | 静态属实：`generated_content` 先收首 token，循环只检查第 2 个及以后的 EOS |
| `run_scripts/e89_moba_smoke.sh` 无 `set -e` | **无法本地核实**：该文件不在当前 git 树（GPT 审的是导出快照，可能已删除） |
| E110 `test_e110_ccluster.py` 硬编码旧树路径，「可能实际比较同一份代码」 | **属实且比审查表述更强**：该测试文件当前就位于主树 `two-level-attention/test_e110_ccluster.py`，`:41-42` 的 `REPO=dirname(__file__)` 与 `MAIN_TREE=/home/.../two-level-attention` **恒相等**——在本树运行时「新旧」必然加载同一份源码，跨版本回归证明力为零（非「可能」） |
| E110 T4/T5 与 E113b T3 配置使 far_budget=0，未验证活跃 far 配额 | 静态核算属实（β=.25,γ=.5 → nt_near=min(1024,768)=768=K2_mid，far_budget=0） |
| decode_cuda_graph_runner.py:710-718 引擎路由（主控复核补充） | 属实，`:716-718` 即 veto 钩子调用 |

## 对 GPT 审查文档本身的评价（找「分析的 bug」）

1. **未发现事实性错误**：六项主结论的问题定位、可达链、触发条件、 witness 数字
   全部经真实 torch/GPU 执行复现；对 S-T002 已修三项的排除、对 E109 不受旧脚本
   污染的克制推断、对自己环境限制（无 torch）的声明都准确。
2. **可加强的表述（非错误）**：
   - C2 的「short prompts」可精确为 **128 < S < 256** 的触发窗口（本次实测推导）；
   - E110 硬编码路径的「可能比较同一份代码」应改为「在当前树必然比较同一份代码」；
   - pred.py 删首 token 从「待配置核实」升级为「已确认删除真实内容」（见上表）；
   - 审查给出的 C2 修复回归长度表（129/160/255/256/600/1024）合理，建议把 255/256
     边界也纳入（窗口右端）。
3. **证据等级差异**：GPT 的五项数值见证是纯 Python 算术/AST 赋值级；本次已将
   C1/C2/C3/T1/T2/T3 全部升级为直接执行仓库真实源码（方法调用或逐字 exec），
   并补了 H20 GPU 上的 fused 对照。结论方向未变，证据强度显著提高。

## 复现实测脚本说明

全部脚本在 torch 2.8.0+cu128 / H20 上运行通过（断言全绿）。要点：

- **C1**：`TLIIndexer(TLIProfile(), head_dim=128)`，index 中 kmax[2:130]=+1000/
  kmin[2:130]=-1000（L1 上界严格胜出任一 q），kq 全零；`select(..., use_l1_kernel=
  False, use_l2_kernel=False)`，统计返回下标中不属于合法池的数量并 gather `fine`
  验证 -inf。对照组同构造 `use_l2_kernel=True`（GPU）。
- **C2**：env 三变量进 taskmd；pool_l 全零（s2=0），S_cap=256/NBLK_CAP=4，
  `select_decode_batched(pool_l, [0], [160], q)`；按 `valid = sel < S` 统计重复
  lane，并按 `backend.py:1019-1022` 语义算 Q=0/V∈{0,1} 见证输出。
- **C3**：`object.__new__(TLIIndexer)` + 手工簇心（c=q/(q·q)，原始点积=1）调用
  真实 `_near_token_score`；`open().read().splitlines()[1047:1060]` 逐字 exec 合并块。
- **T1**：`ast.parse` 提取 `select_sub` 的 FunctionDef 编译执行（闭包变量作全局
  注入）；约束块按行号提取、`continue→raise` 模拟。
- **T2/T3**：按行号提取源码语句逐字 exec（T3 的 `return` 包进函数改为返回值）。

注：GPT 报告附带的 `core_witnesses.json` 风格脚本依赖其导出快照目录
（`sglang-source`），本机不存在；本核查全部脚本直接作用于当前 git 工作树源码。
