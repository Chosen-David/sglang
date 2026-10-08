# SGLang TLI 全链路深度审查：新发现 Bug 清单

审查日期：2026-10-08。审查方式：5 路并行深审（HF indexer / ops+patches / LongBench 生成打分链 / SG 端口 / 编排脚本）+ 主会话对全部关键发现逐一亲自复核（tokenizer 实测、代码逐行对照、文件名实证）。全程只读，未修改任何文件（E109 海选正在双卡运行）。

审查对象：`two-level-indexer` 分支 commit `15eea9adc`（GPT 审查 9 bug 修复版）+ `/tmp/e109_*.sh` 编排脚本 + `/tmp/e109_scan_v2` 数据实况。

**总判决：在跑的 E109 数据本身未被污染（臂间排名可信），但存在 4 个 P1、12 个 P2 及一批 P3；其中两个"倒计时事故"需要立即安排补救。**

---

## 0. 需要立即知情的两件事（不打断实验，但要安排补救）

### N1【P1】screen4 链收官时会用"退化结果"覆写已落袋的正确文件（已亲验）

`/tmp/e109_screen4_chain.sh` 末尾内嵌打分段：

- L108：`fs = sorted(glob.glob(f"{d}/{t}-tli_*.jsonl"))`，t 含 `"repobench-p"`——但 `pred.py:356` 用 `dataset.split("-")[0]` 作文件前缀，实际文件名是 `repobench-tli_*`（`ls` 实证）→ repobench-p **永远匹配不到 → 永远 None → `len(vals)==5` 永不成立 → 全臂无 AVG5**；
- L123：结果写入 `exp/trace/results/e109_screen4_selection.json`——正是 `/tmp/e109_score_v2.py`（其 L54 已正确处理前缀截断）落袋正确结果的**同一个文件**。

即链跑完那一刻，正确的 selection json 会被"全臂无 AVG5"的退化版覆盖。

**补救**：链收官后重跑一次 `python3 /tmp/e109_score_v2.py` 即可恢复。
**注意**：绝不能编辑运行中的 `/tmp/e109_screen4_chain.sh`——bash 边读边执行，编辑会把链搞坏。

### N2【P2】E109 原始数据在 /tmp，systemd 清理定时器 active

`/tmp/e109_scan_v2` 下 49 臂原始 jsonl。`systemd-tmpfiles-clean.timer` active，默认 10 天超龄清理；10 月 7 日完成的批次约 10 月 17 日进入危险区（relatime 挂载下读取未必刷新 atime）。评分 json 已落袋仓库，但**原始预测不可重建**（49 臂 × 双卡 GPU 天）。

**补救**：尽快把 `pred_E109_*` 拷入 `exp/results_longbench/`（只拷贝，不影响在跑臂）。

---

## 1. 影响 E109 数据口径的（系统性、全臂公平，不污染排名）

### R01【P2】`pred.py:182` q_input 无条件丢首 token（已亲验）

`q_input.input_ids = q_input.input_ids[:, 1:]` 是 Quest 上游为 Llama/Mistral 去 BOS 的设计。**Qwen3 tokenizer 不加 BOS**——实测：

```
tok('Question: what is this?').input_ids[0,:3] = [14582, 25, 1128]
tok.decode(14582)        → 'Question'
tok.decode(ids[0,1:])    → ': what is this?'   ← "Question" 整个词被删除
```

每个样本的 question 窗口丢第一个真实 token：hotpotqa/qasper 丢 `"Question"`，gov_report 丢 `"Now"`（`rfind("Now,")` 切点），repobench-p 丢的是紧邻 `Next line of code:` 之前的代码 token（下一行补全任务的最关键上下文）。`pred.py:167` `q_pos = max(len(prompt)-100, q_pos)` + `:169` 死判断保证 split 对每个任务每个样本都发生。

**影响**：所有臂（含 FullKV/Quest baseline）共用同一 pred.py → **臂间排名公平有效**；但绝对分系统性微偏低，与官方 LongBench 公布值/外部 baseline 不可直接比。
**修法**（海选收官后，当前绝不能动）：按 tokenizer 是否自动加 BOS 决定切不切首 token，而非无条件 `[:, 1:]`。修复后全量阶段分数与海选分数存在口径差，论文表格只能用修复后重跑的数。

### R03【P2】hotpotqa 为 200/300 子集

本地数据 `/home/wangyuanshuo02/datasets/LongBench/data/hotpotqa.jsonl` = 200 行（官方 300）；qasper/musique/gov_report=200、repobench-p=500 恰为官方全量。pred.py 无采样/截断逻辑，n 完全来自预切片数据。臂间一致；对外报告须注明 "hotpotqa 200/300 子集"。联网后可用 `_id` 对拍确认是前缀切片还是随机子集。

---

## 2. SG 端口（python/sglang/.../tli/）：B01 修复不完备 + 一个新 P1

> 均不影响 E109 HF 数据；全部为 E112/E113 SG e2e 前的阻塞/准阻塞项。

### S1【P1】taskmd e64 批量 prefill：哨兵 S_r 与消费端全局 S 不匹配（已亲验）

B01 修复在 taskmd 路径用 **per-row** `S_r = t_r+1` 作哨兵：

```1547:1551:python/sglang/srt/layers/attention/tli/indexer.py
                sel_f2 = torch.where(
                    ~keep_f2, S_r.view(-1, 1, 1), i_f2
                )
```

消费端却用**全局** `S` 判有效：

```1027:1029:python/sglang/srt/layers/attention/tli/backend.py
        S = locs.shape[0]  # 请求序列长（哨兵界）
        valid = sel < S
        sel_g = sel.clamp(max=S - 1)
```

chunk 内行 `r < nq-1` 有 `S_r < S` → 哨兵被判**有效** → gather 出位置 `t_r+1`（该行 query 的下一个 token，KV 已写池、真实 logit）→ 每个无效槽位以完整权重重复计入同一**未来 token**（因果越界 + B01 重复计权从 token 0 平移到 token `t_r+1`）。高 γ 配置（如 (.125,.375,.625)）下 near 缺口对 `S_r ∈ [1024, 6400)` 的每一行都成立——**每个 chunk 的绝大多数行**带数十~数百个指向未来 token 的重复槽位。

单测没抓到的原因：`test_b01_b02_fix.py` 两个场景都是单行 `t_arr`，单行时 `S_r == S` 恰好被屏蔽——单行测试在构造上无法暴露 S/S_r 混用。

**修法**：e64 三处 `S_r.view(...)` 改全局 `S`（最小改动，与 B02 grid 口径一致）；或消费端改 per-row `valid = sel <= t_r`（更严格，可同时兜住 S3）。**必须补 ≥2 行、含池不足行的 e64 对拍测试。**

### S2【P1】decode 批量单池 topk：swa `+inf` 强制把哨兵 lane 复活（已亲验）

```1975:1978:python/sglang/srt/layers/attention/tli/indexer.py
            causal = tok <= t_t.view(-1, 1)
            valid = tok < S_cap
            sc = s2.masked_fill(~(valid & causal).unsqueeze(1), float("-inf"))
            sc = sc.masked_fill((tok >= sw_lo_t.view(-1, 1)).unsqueeze(1), float("inf"))
```

哨兵 lane 的 `tok = S_cap`（未 clamp）≥ `sw_lo_t` 恒成立 → 第二行 `masked_fill` 把刚打成 `-inf` 的哨兵**全部复活为 `+inf`**。topk `k=min(1024, Tc)` 时数万个 +inf 哨兵与 128 个真实滑窗 token 并列，真实候选被 tie 完全挤出；后续 `sc_g == -inf → SENT` 的转换对 +inf 不生效。净效果：退化成"只看滑窗"，最坏 0 个有效 lane → eager softmax 全 -inf → **NaN**（fused kernel `l_i=0` → 0/0 NaN）。

**触发**：① D' skip_far 层的批量 decode（n≥2 或 CUDA graph——图内每步都走这条路）；② taskmd 单池臂（α=0 或 β=0，计划内配置）的批量 decode/graph。per-request 路径用全宽 fine 矩阵，无此问题——**n==1 正常、n≥2 崩**。
**修法**：第二处 masked_fill 条件改为 `valid & (tok >= sw_lo_t)`（causal 由 `sw_lo ≤ t` 蕴含）。

### S3【P2】per-request `_select_taskmd` e64 池不足无 keep 掩码

`indexer.py:797-806`：decode n==1 走 per-request 路径，e64 下 far/near 池有限项 < 配额时，topk 选出的 `-inf` 槽位 = 池内真实位置（< S）→ 消费端判有效 → 以真实 logit 计入。而批量版用 `keep = sc > -inf` 转哨兵屏蔽——**n==1 与 n≥2 decode 语义不一致**，bs=1/bs≥2 A/B 对拍在池不足行发散；与 HF 参考的"垃圾位权重 0"也不等价（SG 消费端重算 q·k 真实 logit）。触发：e64 taskmd、decode n==1、α 小且 S ≲ sink+swa+nt_near/α。

### SG 侧 P3

| 编号 | 位置 | 问题 |
|---|---|---|
| S4 | `indexer.py:1060` + `kernels.py:457` | `near_len=0` 配置下 clamp 哨兵（S-1）落入 far 池 → 末 token 重复计权（默认 near_len=2048 及 decode 侧安全） |
| S5 | `indexer.py:1318` | `_select_batched_taskmd` 未使用 `t_min_hint`，early 检查保留每 chunk GPU 同步（测速口径污染，taskmd 正是要测速的臂） |
| S6 | `backend.py:191` vs `indexer.py:180` | PCA basis 实际秩与 `proj_rank` 不校验，`.pt` 文件 r 不符时 pool 槽宽与 indexer 维度静默错位 |
| S7 | `indexer.py:1178` | empty 行 fallback 宽度补齐用 0 填充（B01 原版语义；当前不可达，建议防御性改哨兵） |
| S8 | `config.py:99-103` | 注释仍写"输出哨兵转 0"，B01 前过期描述 |

### SG 侧审查通过项（明确结论）

**B02（identity grid + 哨兵尾垫）：完备。** **B01 在非 taskmd 路径：完备**（clamp 顺序、decode 哨兵协议 S_cap 统一、CUDA graph 静态宽度、empty/early 特判均闭环）。池管理/侧流并发（阶段 A0 前置、ev_kv/ev_done 覆盖、realloc 安全）、kernels.py 边界（grid/CHUNK 一致、online softmax NaN guard、L1 topk 边界）、图/急切混跑自愈逻辑均通过。

---

## 3. HF 框架（two-level-attention/sparse_attn）：走廊内自洽，走廊外脆

### F1【P2】layer_skip × α/β 分区 → attention sink 永久丢失（已亲验）

`tli_indexer.py:812`：`use_partition = (enable_kmeans or e64_partition) and not self.skip_far`——skip 层落到 L1071 legacy 全序列 topk，该路径**没有** L1068-1069 的 sink/swa 正交强制；而 sink 块的 p 已被 L1 gate 成 0、未 gate 的 near 块（β=0.25 时 2048 token > K2=1024）全部 p>0 → **skip 层永远选不到 sink**，且 γ/β 分区预算成死参数、swa 以 p=1.0 占用预算（与非 skip 层口径不一致）。与 L800 设计注释直接矛盾。

**触发**：`--tli_enable_layer_skip true` + α>0,β>0 + 当前层 ∈ skip 集合。E109 `layer_skip=false` 不受影响；**开 layer_skip 实验前必修**（skip 层补 sink/swa 正交强制，或走分区路径仅 far 池置空）。

### F2【P2】qwen3 patch 分支判定的三个走廊外失败（机制已核验）

`patches/qwen3_attn_patch.py`（llama3 patch 同构）：

- **use_cache=False 崩溃**：L41 对 None 有守卫，L50 `past_key_values.get_seq_length(...)` 无守卫 → AttributeError（训练/PPL/logprob 打分形态）；
- **chunked prefill 静默输出零**：分支条件 `get_seq_length == q_len` 把"带非空 past 的多 token 前向"误判进 decode 分支，而 `eager_decoding.py` 只算 `q[0]`/`o[0]` → token 1..tq−1 的 attention 输出为 0，经 o_proj 后被残差掩盖，**无任何报错**；batch>1 则崩溃；
- **attention_mask≠None 分支 dead-broken**（L71-73 + `patches/utils.py`）：按 2D bool padding mask 假设实现，但 HF 实际只传 4D float causal mask 或 None，任何真实调用下都不正确。

E109（batch=1、逐 token、sdpa、mask=None）不触发。建议实验收官后在 decode 分支入口加 `assert q_len == 1 and batch == 1 and past_key_values is not None` 硬守卫。

### F3【P2】GLM-4 patch 从未注册：静默跑 dense 却标记为稀疏方法

`patches/patch.py:17-31` 只认 `Qwen3Attention`/`LlamaAttention`；`glm4_moe_lite_attn_patch.py` 存在但从未被 import/注册，且本体也不含 indexer 调用（半成品）。GLM-4 模型跑 method≠none → 零模块匹配 → 静默不 patch → **产出实际是 dense 的结果却被打上稀疏方法标签**。

### F4【P2】σ（sigma_select）静默退化

`sigma_select≠none` + `enable_kmeans=false` + α=β=0 时 `use_partition=False` → σ 分支不进，静默退化为默认两级 topk——**实验跑的不是声明的方法**，无 assert；D' 跳层命中时 σ 臂同样静默退化。moba × σ 组合下 σ 也是死参数（moba 提前 return）。

### HF 侧 P3

| 编号 | 位置 | 问题 |
|---|---|---|
| F5 | `tli_indexer.py:788-795` vs `982/988` | far/near 分区边界按 padded 长度（kt·bs）计量，与真实 S 偏差 ≤63 token（亚块级噪声，预算总量与因果性不受影响） |
| F6 | `tli_indexer.py:1037` | far 区不足时余额被 near 吸收，near 可超 γ 配额 nt_near（E109 网格内不可达） |
| F7 | `tli_indexer.py:165/239` + clear | `_basis` 不在 clear() 重置：proj_basis × static_pair 同开时跨请求残留（两者均未开） |
| F8 | `tli_indexer.py:605` + clear | `_last_q` 不在 clear() 重置（现调用序下无害） |
| F9 | `tli_indexer.py` 全程 | batch>1 时拼接序列被当单序列索引（跨序列泄漏；端到端形状已对不上，非静默） |
| F10 | `metrics.py:43` | `add_k_delta = (k_max + k_min).abs()` 疑似应为极差（死代码，无生产者） |
| F11 | `info.py:12` vs `tli_indexer.py:63-67` | ab 串用 flag 原值而非生效值（subspace=full 时 indexer 强制关 subspace，名字仍含 "A"）——B09 修复时须一并校正取值来源 |
| F12 | `eager_decoding.py:45-48` | 消费端无全 False 行防线（当前由 indexer 契约兜底，契约内不可达） |

---

## 4. 编排脚本（/tmp/e109_*.sh）

### F1【P1】screen4_chain 内嵌打分 repobench-p glob 永不匹配 + 收官覆写好文件
见 N1（已亲验）。

### F2【P1】full13_chain 同款 glob 错误，且 AVG 缺臂不拦截

`/tmp/e109_full13_chain.sh:113` 同根因（repobench-p 必然 None）；且其 AVG 按"非 None 任务数"平均（`:125`），与 screen4 的"5 任务齐全才出 AVG5"门不同——**12 任务均值被标为"13 任务全量终判"**。full13 启动前必修。

### F3【P2】repobench-p 的 SKIP/断点续跑判定永不命中（4 处）

`screen4_chain.sh:67`、`full13_chain.sh:48`、`full13_remote.sh:26`、`screen4_remote.sh:44` 均为 `ls "$OUTDIR"/${t}-tli_*.jsonl`（t=repobench-p 永不匹配）→ SKIP 恒假。链重挂时每臂全量重跑 repobench-p（500 样本，实测 13~90 分钟/臂）；旧链未死就重挂会**同名文件双写**。（旧脚本 `/tmp/b7_split_guard.sh:4` 的 EXPECT 用的就是正确前缀 `repobench:500`——口径曾已知，screen4/full13 编写时回退了。）

### F4【P2】full13_chain 与 screen4_chain 挂同一 marker → 双卡超订

`full13_chain.sh:20` 与 `screen4_chain.sh:12` 都等 `grep "E109_CHAMP_VALIDATE_DONE"` → 两者同时放行各起 GPU0+GPU1 两个 worker → 每卡 2× Qwen3-8B → OOM 或严重降速。screen4 还要跑很多小时，此时挂 full13 必撞。

### F5【P2】远程 screen4_remote 没有 γ 悬崖快筛名单 → 在跑本地刻意跳过的坍塌臂

`/tmp/e109_screen4_remote.sh:25-34` 无本地 `:45-52` 的 12 模式坍塌名单。sim 臂 ~5h/任务；实证 `pred_E109_cavgsim_a0.125_b0.25_g0.75/qasper-*.jsonl` 已存在而本地链日志无任何 SCREEN4 cavgsim 行——**浪费已实际发生**（数十 GPU 小时）。

### F6【P2】本地与远程两条 screen4 链臂集重叠

两侧各自 `ls` 各自的 `/tmp/e109_scan_v2`，SKIP 只在各自文件系统生效；本地链走到某 sim 臂时若文件尚未 rsync 回来，同臂同任务两边各算一遍（~5h/任务）。

### F7【P2】全链路 python 调用无退出码检查

所有 pred 调用 `2>&1 | tail -2`（管道退出码=tail 的 0），无 PIPESTATUS；`champ_validate.sh` 的 run_arm 连 SKIP/行数检查都没有；DONE marker 无条件 echo。失败 → 无输出文件（好在 pred.py 一次性写出不留残文件）→ marker 照样亮 → 缺臂记 None。screen4 侧表现为"无 AVG5"，full13 侧被 F2 放大成"缺臂也算 AVG"。当前实测 0 次 Traceback/OOM。

### F8-F10【P3】

- `e109_collect.sh`：v1 遗留，等永不出现的 `E109_LOCAL_DONE`（v2 标记是 `E109_V2_LOCAL_DONE`）→ 永远 hang；若误执行会把 v1 污染数据重新落袋；
- 所有 marker 等待环无超时（生产端死掉 → 下游永久静默挂起）；
- `/tmp/b7_split_guard.sh:4` EXPECT 清单漏 triviaqa，仅靠 GPU1 任务顺序兜底。

### 编排侧审查通过项

tag 解析对全部 49 臂逐一验证无误；cavgsim/cclustersim 的 sim=0.9 与实际运行一致；坍塌臂快筛名单对 prescore.json 全量比对无漏网（本地侧）；双卡 round-robin 无同文件双写；pred.py 一次性写出（崩溃不留 partial）；marker 字符串逐字核对一致；E110 草案与 E109 口径（任务/n/K1/K2/cmp/模型/subspace 默认值/far·near method 默认值）全部一致。

---

## 5. 审查过且确认干净的维度（与发现同等重要）

- **HF aavg 主路径**（E109 在跑口径）：预算算术（K2_mid/nt_near/far_budget/nb_far）逐项与设计一致；因果性（所有选择池上界 ≤ S、单位置 decode 无未来泄漏、chunked prefill 撞显式 assert 而非静默错）；块对齐尾巴无死区；GQA 组内 mean/sum 口径与注释一致；clear() 聚类状态重置完备（除 F7/F8）；全 -inf 行不可达（nan_to_num 兜底到位）。
- **RoPE/q_norm/k_norm** 与 HF Qwen3 原版逐行一致；GQA `(h g)` 布局两侧一致。
- **chat template**：Qwen3 硬编码模板与 `apply_chat_template(enable_thinking=False)` **逐字符相等**（实测）；裸文本任务排除清单与官方 LongBench 一致。
- **超长截断**：中间截断与官方一致；max_length=31500 不越界。
- **metric 实现**与官方 LongBench 逐函数一致；repobench-p→repobench 的 scorer 键三方一致（eval.py / run_e71_eval.py / e109_score_v2.py）。
- **e=0 语义**与官方不带 --e 一致（仅 CLI 形态不同，迁移命令需改写）。
- **生成循环**（除已知 B04）：KV cache 每步恰一次 update、无重复/遗漏；输出写入在全部样本完成后（崩溃不留 partial）；每臂独立 postfix 目录无碰撞。

---

## 6. 行动优先级总表

| 时序 | 动作 | 对应发现 |
|---|---|---|
| **现在**（不动在跑链） | ① 备份 `/tmp/e109_scan_v2` 原始 jsonl 到仓库目录；② 记下链收官后必须重跑 `python3 /tmp/e109_score_v2.py` | N2、N1 |
| E109 收官后 | 修 R01（q_input 丢首 token，全量阶段需重跑）；修 B04/B09（计划内缓期）；full13 启动前修编排 F2/F3/F4 | R01、编排 |
| SG e2e（E112/E113）前 | 修 S1/S2/S3；补 ≥2 行含池不足行的 e64 对拍测试（单行测试构造上抓不到 S/S_r 混用） | S1-S3 |
| layer_skip 实验前 | 修 HF indexer F1（sink 丢失） | F1 |
| GLM 实验前 | 注册并完成 GLM patch，或显式拒绝 | F3 |
| 顺手 | patch 加 use_cache/batch/chunk 硬 assert；σ 组合守卫；SG P3 若干 | F2/F4、S4-S8 |

---

## 7. 证据边界

- 主会话亲验：N1（glob+覆写路径）、R01（tokenizer 实测）、HF F1（L812/L1068/L1071 对照）、S1（indexer L1547 vs backend L1028）、S2（L1977-1978 逐行）。
- 子代理发现附代码引用、主会话抽样复核机制：其余各条。
- 未做：GPU 全链路回归、真实模型 e2e 对拍（E109 占卡中）；hotpotqa 子集性质（前缀 vs 随机）需联网对拍 `_id`。
- 与 GPT 两份审查（`SGLang_Indexer_Bug审查与任务链核查_20261008_by_gpt.md`、`SGLang_最新代码复查_新增Bug与修复回归_20261008_by_gpt.md`）的关系：本报告全部为新发现或其修复的回归核验，不重复 B01-B09 本体。
