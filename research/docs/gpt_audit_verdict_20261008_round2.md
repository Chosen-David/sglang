# GPT 审查报告六项核验判决（round2，只读核验 + CPU 复现）

核验日期：2026-10-08。核验 agent：只读核验，未修改任何生产代码，未跑 GPU。
- 审查报告：`/home/wangyuanshuo02/sglang/agent_doc/advice/sglang_twolevel_audit_by_gpt.md`（审查 commit ee256310b）
- 核验基线：sglang 仓 HEAD c7b1087e7（tli/indexer.py 最新提交 067553e60，**工作区无未提交改动**）；two-level 仓 HEAD c7b1087e7（tli_indexer.py 最新提交 54d93a4c6，工作区干净）——即核验的是**比报告审查版本更新的代码**，结论对最新代码成立。
- CPU 复现脚本与产出：`/tmp/verify_gpt_audit/`（t1_enumerate.py / t1_pollution_map.py / c1_c2_witness.py + 3 个 JSON）

## 六项 verdict 总表

| ID | verdict | 一句话判决 | E109 在跑链 | 论文已引数据 |
|---|---|---|---|---|
| C1 | **属实**（静态+数值复现） | sglang eager per-request `select()` 三处 topk 均 -inf 直通，最新代码仍存活 | **零影响**（E109 走 two-level HF patch） | 历史 sglang serving 单请求精度/延迟数据须标注 |
| C2 | **属实**（静态+反例复现 0.6 vs 0.75） | `_select_decode_taskmd` 缺短行 identity 修复，sink/swa 拼接无去重 | **零影响**（同上） | **零已落袋数据**（taskmd+graph+短序列路径从未跑过 e2e 数据） |
| C3 | **属实**（静态+数值复现，放大 11.31×） | near 簇分未乘 softmax_scale，cluster 与 sim_greedy 两路径同中 | **部分臂在产污染数据**（ccluster 分区臂） | E105 cavg 13 任务零污染（near=4bit）；E109 prescore #3/#11 臂受污染 |
| T1 | **属实**（静态+全网格枚举） | far 池溢出点占每样本 6.1-8.8%，报告反例 1536 精确复现 | 零影响（mass 离线网格） | 核心数字全干净；fig9 两曲线低 γ 段 + fig10 一 bar + election 一 mass 值虚高 |
| T2 | **脚本缺陷属实，实际未污染** | AVG 部分收割口径缺陷存在，但落袋 JSON 六臂 11/11 任务全齐 | 零影响 | e104_ruler_32k.json 当时结论有效；记档高危 |
| T3 | **静态缺陷属实，未踩坑** | 断点复用无参数指纹，但历史调用参数集均唯一 | 零影响 | E81/M6/E104 数据有效；记档高危 |

---

## C1 — sglang eager per-request 选择器 -inf 槽位泄漏：**属实**

### 证据（最新代码仍存活）
- `python/sglang/srt/layers/attention/tli/indexer.py` L637-662：`fine` 对非候选/非因果位置填 -inf；L656 `i_f = torch.topk(far_f, k2_far).indices + far_tok_lo`、L660 `i_n = torch.topk(near_f, min(k2_near, S)).indices`、L662 整行 `torch.topk(fine, p.token_budget)`——**三处 topk 输出均无 -inf 过滤、无哨兵转换**，原始索引直接返回。
- 消费端 `backend.py` L983 `valid = sel < seq_v`：-inf 槽位的 indices 全在 [0,S) → 全部判有效，L1019-1022 对其重算真实 attention logits。消费端拿不到选择器内部分数。
- 可达性确认：`config.py` L74 `use_batch_select` 默认 **True** 但 `backend.py` L614 仅 `len(sparse_rows) >= 2` 才走批量路径 → **n==1 恒走 per-request `select()`**；`SGLANG_TLI_BATCH_SELECT=0` 时全部走此路径。`config.py` L62 `use_l2_kernel` 默认 **False** → fused kernel 旁路（已修的 bug1 路径）不生效，eager 路径是**默认实际路径**。
- 与已修 bug 的边界：S-T002 bug1 修复在 `kernels.tli_l2_partition_topk`（fused）；S3 修复（e4e005df7）在 `_select_taskmd`（taskmd per-request）。**`select()`（非 taskmd 的 B'/TIA 口径）是独立的第三处，至今未修**——报告表述准确。

### 数值复现（/tmp/verify_gpt_audit/c1_c2_c3_witness.json）
默认参数（budget=1024, far_tokens=256, sliding_window=128）S=16384：near 配额 768，构造 near 有效候选 192（含 forced +inf 滑窗）→ 至少 **576 个 -inf 槽位**作为合法 token 进入 attention。复现值与报告一致。

### 污染面
- **E109 海选：零影响**。E109 走 `/home/wangyuanshuo02/sglang/two-level-attention/sparse_attn/patches/qwen3_attn_patch.py` 的 HF patch——prefill 走 dense 分支（`get_seq_length == input_shape[1]`）、decode 走 `prepare_mask`（two-level 权威实现，topk_mask+全长 softmax+p[...,-swa:]=1.0 语义，无此结构）。**不经 sglang 任何代码**。
- 历史 sglang serving 数据（E87/E89/E90/E102）：bs≥2 批量路径哨兵口径安全；**bs=1 的单请求精度/延迟数据走了 C1 路径**。触发条件 = near 池有效候选 < 768（L1 top-128 块全落 far 区时成立），实际触发频率无法事后判定（无中间量存档）。处理建议与 B01/B02 同款：标注「潜在受 C1 影响口径」。

### 修复建议
`select()` 内 gather 各分支 topk values，`-inf` 选中项转哨兵 S（与 `_select_taskmd` L679-680 已有约定一致：下游 `valid = sel < seq_len` 统一屏蔽），含 L662 未分区整行 topk 出口。回归：near-starvation 构造跨 eager/fused/batched 三路径断言每个输出属于该 head 有效池。

### 对在跑链的动作判断
**无需立即动作**（E109 不经过此路径）。

---

## C2 — sglang CUDA graph 短序列 sink/swa 重复计权：**属实**

### 证据
- `indexer.py` L2466-2494（`_select_decode_taskmd` 尾部）：`sink_out`（静态宽 sink_tok=128，位置 [0,128)）与 `forced_out`（静态宽 swa_tok=128，位置 [swa_lo, S)）直接 `torch.cat` 拼接——**无去重、无短行 identity 替换**。
- 对照组确认：`_select_batched_taskmd` L1661-1677 已有 B02 修复（early 行 t<K2 用 identity grid + 哨兵尾垫）；**`_select_decode_taskmd` 无同款修复**——报告对照定位准确。
- 路由链确认：`backend.py` L397-414 `veto_cuda_graph` 仅 veto `token_budget(1024) < L ≤ dense_threshold(2048)` 的行；S=160 不触发 veto；`backend.py` L653-689 `_forward_decode_graph` 把所有行（含短真实行）统一送 `select_decode_batched` → L1817-1818 taskmd 分支 → `_select_decode_taskmd`。
- 区域计算确认（源码 `_taskmd_regions` L214-266 手算）：S=160 时 `near_blks = far_hi_blk = far_lo_blk = 2` → far/near L1 双池空 → 输出仅 sink+forced 256 lanes。

### 数值复现
S=160：sink [0,128) + swa [32,160)，重叠 96 位置各出现 2 次；256 lanes 覆盖 160 唯一位置。Q=0 + V=1-on-overlap 反例：dense 输出 **96/160 = 0.6**，graph 稀疏输出 **192/256 = 0.75**。与报告逐位一致。重复位置均 < S → 消费端 valid 全过 → 两条 lane 各自计权（softmax 内重复计权实锤）。

### 污染面
- **E109 海选：零影响**（不经 sglang）。
- **已落袋数据：零污染**。触发需要同时满足：taskmd 模式（E112 #149 引入）+ CUDA graph 开启 + 批内含短真实行（S ≤ 1024 或 1024<S 但 S 长到 L1 双池空）。E112 的 59/59 对拍走 eager per-request（`_select_taskmd`）；E87/E89/E90/E102 早于 taskmd 引入（走非 taskmd 路径，swa 是池内 +inf 强制而非正交拼接，topk 天然唯一）。**该缺陷是「未踩但真实可达的部署缺陷」**。

### 修复建议
真实短行（S ≤ token_budget）整行 identity [0,S) + 哨兵尾垫（对齐 `_select_batched_taskmd` 的 B02 修复）；或 veto TASK.md 短行回 eager dense。仅 masking 重叠不足（独立配额下不能保证短行保留全部因果 token）。回归长度点 129/160/255/256/600/1024 + 非常数 V。

### 对在跑链的动作判断
**无需立即动作**（不影响任何在跑实验；属 sglang serving 部署路径修复项）。

---

## C3 — two-level near 簇分未乘 softmax_scale：**属实（三项里唯一影响 E109 在跑数据）**

### 证据（静态链路完整确认）
- `sparse_attn/indexer/tli_indexer.py` L605-606：`compute_score` 缓存 `self._last_q = q.squeeze().float()`——**未乘 softmax_scale**（对比 L640/678 `b_q_full = (q_sq[0] * softmax_scale)`：`score_fine` 是缩放后的点积）。
- L712-728 `_near_token_score(q)`：用该未缩放 q 与 near 簇心算点积，仅做了 GQA mean 聚合对齐，**未应用 softmax_scale**。
- L1048-1061 混合池：`sf_g = score_fine 的 group-mean`（**已缩放**）作为回退尾段分，`near_tok_score`（**未缩放**）覆盖簇段，两者进同一个 `near_p` 池 topk（L1061）。L1042-1043 注释声称「两源均为 group-mean 原始点积分（量纲一致）」——**与实现不符**，实为未缩放 vs 缩放混合。
- **修复面确认（核验点①）**：`near_select=cluster`（kmeans）与 `sim_greedy` **两路径都受影响**——两者都在 `_update_near_cluster`（L421-466）写 `_km_near_centroids`，都经 L820-824 `near_tok_score = self._near_token_score(q_last)` 消费。far 侧 `_far_token_score` 虽同样未缩放，但 far 池纯簇分 homogeneous（L992-1011 far topk 只用簇分）→ far 排序不受影响；**混合只发生在 near 池**。触发常态化：`_km_near_hi` 块对齐（(S-swa)//bs·bs）而消费端 `swa_lo_tok` 是精确 token 位置 → 非对齐 decode step 恒有 [near_hi, swa_lo) 回退段（宽 0-63 token）与簇覆盖段同池竞争。
- 数值复现：D=128 时簇段正分被放大 **11.31×**（1/scale）；反例（簇段原始分 1.0 vs 尾段 2.0）实际比较 1.0 vs 0.177 → 错选簇段；统一量纲后应选尾段。与报告一致。

### 污染面（核验点②，精确到臂与数据文件）
**受影响臂 = near_select ∈ {cluster, sim_greedy} 且 α>0（分区）的 e2e 数据**。运行参数核实（/tmp/e109_scan_local.sh L86-88、/tmp/e109_scan_remote.sh L48-59、/tmp/e109_screen4_chain.sh L24-27）：
- `cavgkm`/`cavgsim`（far_select=cluster/sim_greedy + **near_select 默认 4bit**）→ near 池纯细筛分 homogeneous → **不受影响**；
- `cclusterkm`/`cclustersim`（**--tli_near_select cluster/sim_greedy**）→ 受影响；
- (0,0) 单池点（near 区不激活，`near_len_dyn=swa_tok`，`Tn<256` 不建簇）→ **不受影响**（核验点③确认）；
- 全部 aavg/mavg/mminmax 臂（near_select=4bit）→ 不受影响。

E109 v2 已落袋受污染数据（/tmp/e109_scan_v2/pred_E109_*，hq+mu 两任务）：

| 臂 | prescore 排名（48 完整臂 hq/mu avg2） | 状态 | 污染判定 |
|---|---|---|---|
| cclustersim_a0.125_b0.25_g0.125 | **#3（44.46）** | 五任务补齐中（远程 .251） | **受污染** |
| cclustersim_a0.125_b0.25_g0.25 | #11（43.58） | hq/mu 落袋 | **受污染** |
| cclustersim_a0.5_b0.125_g0.75 | #29（35.97） | 坍塌臂 | 受污染（本就垫底） |
| cclusterkm_a0.125_b0.25_g0.75 | #47（27.34） | γ 悬崖坍塌臂 | 受污染（本就垫底） |
| cclustersim_a0.125_b0.25_g0.75 | #48（27.23） | 坍塌臂 | 受污染（本就垫底） |

**零污染确认**：
- AVG5 已入账 9 臂（e109_screen4_selection.json：aavg×4 + cavgsim×2 + mminmax×2 + aavg(.875,.125,.375)=45.86）**全部 near=4bit，零污染**；
- prescore 榜首 mavg(.125,.375,.125) 44.59、次席 mavg(.25,.125,.625) 44.58、cavgsim/cavgkm/aavg 单池点——零污染；
- E105 cavg 13 任务全量（far=cluster + near=4bit）零污染（E105 的 v1 污染是另一独立问题）；
- 远程 sim 臂 10 个中仅上表 cclustersim 分区 4 点受污染，cavgsim×5 与 (0,0) 点零污染。

**对 kmeans vs sim_greedy 决胜的影响**：两方（cclusterkm vs cclustersim）共享同一缺陷口径 → 内部对比「相对公平」（都错但错得一致）；但与 near=4bit 组合的绝对比较（AVG5 排名、海选选举）存在系统性偏置——簇段被放大 11.31× → 回退段（near 末端最靠近 query 的 ≤63 token）几乎必然被簇段挤出 near topk。方向 = ccluster 分区臂的精度被系统性压低（欠选最高信息量的回退段 token），**cclustersim(#3, 44.46) 的真实竞争力可能被低估或高估（方向为压低），其与头部 aavg/mavg 臂的排序不可信**。

### 附加边界注记（核验中发现，报告未提）
`clear()`（L1078-1099）不清 `_last_q`。当前 e2e 主路径安全（two-level patch prefill dense + decode 每步 `compute_score` 先更新 `_last_q`，q shape [1,1,H,D] 满足缓存条件）；但任何「prefill 期间调用 compute_mask 且簇已构建」的路径（如 TLI_DEBUG 重放类脚本）会用上一请求 decode 的 stale q 打 near 簇分。修复 C3 时顺手在 clear() 重置 `_last_q = None`。

### 修复建议
`_near_token_score` 补乘实际 `softmax_scale`（调用接口已有 scale，勿硬编码 sqrt(128)）；建议同时给 `_far_token_score` 补齐（当前无害但规范）；`clear()` 重置 `_last_q`。回归：非对齐 S + topk 边界两分数 + G=1/G>1，覆盖 kmeans 与 sim_greedy 两路径。

### 对在跑链的动作判断
**建议立即修复（唯一需要立即动作的项）**。理由：
1. 远程 .251 的 cclustersim 分区臂五任务数据正在产生（prescore #3 是海选头部竞争者），继续跑 = 继续落袋污染数据；
2. 修复面一处（two-level `tli_indexer.py`），运行中 python 进程已加载旧模块不受磁盘修改影响，**链上下一个臂的新进程自动加载新代码**（与 bash 链字节偏移坑不同，python 模块修改对新进程是安全的）；
3. 修复后在 selection.json/落袋 JSON 标注代码版本分界，上表 5 个受污染臂（有效竞争者实际只有 #3/#11 两个非坍塌臂 ×2 任务）修复后重跑，成本 ~4 卡时。
若主 AI 决策暂不修复：则 ccluster 分区臂数据继续按「带 C3 缺陷口径」落袋并全程标注，kmeans-vs-sim 决胜与 ccluster-vs-其他组合对比结论降级为「同一缺陷口径下的相对比较」。

---

## T1 — E98 mass 网格 far 池溢出：**属实（报告数值全部精确复现 + 全网格定量扩展）**

### 证据
- `exp/trace/analyze_e98_abg_full_grid.py` L98-101（`select_sub`）：`ts = tok_score.masked_fill(~pool, -inf)` 后 `it = torch.topk(ts, min(n_tokens, mid_len))`，**未检查选中 score 有限性**，`cand.scatter_(1, it + SINK, True)` 把全部下标（含 -inf 槽位）scatter 为 True。far 侧调用 L129 传 `n_pages = BP - nb_near`（far 池仅 (64-nb_near) 页 × 64 token）而 `n_tokens = nt_far` 可达 2048——**无「far 池容量 ≥ nt_far」约束**（现有约束只有 near_L ≥ nb_near·BS 和 nt_far ≤ far_L）。
- near 侧核验：nt_near = nb_near·BS·γ ≤ nb_near·BS 恒成立 → **near 侧永不溢出，缺陷仅 far 侧**。

### 定量复现（/tmp/verify_gpt_audit/t1_grid_overflow.json + t1_pollution_per_sample.json）
- 报告反例（mid_len=8192, a=.5, b=.875, g=0）：far_pool=512 vs nt_far=2048 → **溢出 1536，逐位精确复现**。
- 全网格枚举（对齐脚本 GRID/COMBOS/BS/BP/B_TOK=2048 与约束逻辑）：溢出点占比随 mid_len 5.7%→8.8%（4096 时 0%——far_L 约束先行过滤）；溢出量 192/中位 640/最大 1536。
- **实际落袋数据污染映射**（从 e98_abg_full_grid.json per_sample 反推各样本 mid_len）：16 样本中每样本 41-63 个点受污染（654-720 点中 6.1-8.8%），**污染点结构 = β ∈ {0.625, 0.75, 0.875} 且低 γ（β 越高允许的 γ 上界越高：β=.625 时 γ≤.125，β=.875 时 γ≤.375）**，对 5 个 method 组合同构（预算逻辑与 method 无关）。

### 对论文已引数据的影响（逐点判定）
**核心引用点全部干净**：
- E98 冠军 mass 0.8801（mavg .125/.375/.625）：β=.375 不在污染 β 集 → 干净；
- 分区最优 0.8822（mavg .125/.25/.75）、mass 冠军 0.896（mminmax .875/.875/.5）、单池 0.89794（α=0 → nb_near=0 → far 池全容量 4096 ≥ 2048）、fig11 双口径对照、部署参照臂 0.8728（β=.25）、γ 截断坍缩 54.16/35.57（e2e 网格，不经此脚本）——全部干净；
- E98 mass→e2e 选举入口（mass top-3 候选）：election 文件排名臂全部 β≤.875 且 γ≥.5 → 干净，**选举链路未受污染**。

**受污染的图件/数值（须修图或标注）**：
- **fig8 热图：零污染**——切片 γ=0.75，污染需要低 γ，恰好全部避开（幸运而非设计）；
- **fig9 γ 扫描：2/5 曲线低 γ 段失真**——mminmax 曲线（best α.875/β.875）的 γ=0/.125/.25/.375 四点（0.6779/0.8848/0.8928/0.8946）与 aavg 曲线（best α.875/β.875）同样四点（0.6665/0.7833/0.7948/0.7964）mass 虚高；mavg/cavg/ccluster 三曲线（β=.25/.375）干净，**mavg 平坦区叙事（卖点）不受影响**；
- **fig10 单池对照：aavg bar 虚高**——aavg 分区最优 (α.875,β.875,γ.375) mass=0.7964 本身是污染点；但其即使虚高仍显著低于其他组合（0.88+），「aavg mass 弱」方向性结论不变；
- election JSON 中 aavg(.875,.875,.375) 的 mass 0.7964 同点虚高（该臂 e2e 值 54.4/33.4 不受影响）。

### 修复建议
`select_sub` 补 far 容量约束 `(BP-nb_near)*BS >= nt_far`（或 topk 后按 score>-inf 过滤再 scatter）；重算受影响点后重出 fig9/fig10。重算成本：CPU 网格受影响点 63 点 × 5 组合 × 16 样本，可增量重跑。

### 对在跑链的动作判断
**无需立即动作**（离线 mass 脚本，不影响 e2e/E109；修复排 paper-figure 管线批次）。

---

## T2 — E104 AVG 部分收割不同任务集相减：**脚本缺陷属实，实际数据未受影响**

### 证据
- `exp/trace/analyze_e104_ruler_32k.py` L38-40：无完整 pred 的任务记 None；L48-49 `AVG = 非空任务均值`（部分收割口径）；L54-57/L62-69：只要两边 AVG 非 None 就直接产出 delta.AVG——**任务交集/完整性未纳入判断**。静态缺陷成立（CPU 反例 A={easy:100} vs B={easy:100,hard:0} → delta.AVG=+50 的口径缺陷可直接推出）。
- **实际落袋数据核查**（exp/trace/results/e104_ruler_32k.json）：六臂（main_g0625/ref_g0125/fullkv/quest/tia/single_pool）**全部 11/11 任务完整、零 None cell** → 当时的 −2.15/+1.40/+13.15 三个 AVG delta 均在同任务全集上相减，**结论有效**。

### 污染面与建议
未踩坑（部分收割路径存在但落袋数据恰为完整收官）。记档：后续任何用该脚本模式的中途收割对比，必须显式交集配对或完整集门槛。无需动作。

---

## T3 — pred_kvcf.py 断点复用无参数指纹：**静态缺陷属实，调用历史未踩坑**

### 证据
- `exp/trace/pred_kvcf.py` L168-179：复用条件 = `out_dir = {OUT_BASE}/{method}{output_suffix}` 下同 task 文件行数 ≥ args.n——**max_capacity / window_size / kernel_size / pooling / sink_guard / split_question / attn_impl / 模型路径均不参与指纹**。报告的反例（同 suffix 改 max_capacity 2048 被静默 SKIP）控制流成立。
- **调用历史核查**（/tmp/m6_sinkguard_full.sh、m6_pyramidkv_gpu1.sh、m6_smoke.sh、relay_gpu0/gpu1_e89e90.sh、kvcf_gpu0/gpu1*.log）：
  - E81 baseline（snapkv/h2o/pyramidkv/streamingllm）：全部 max_capacity=1024 默认参数，每 method+task 单一参数集；
  - M6 sinkguard 臂：`--sink-guard 128 --output-suffix _sinkguard` 与 baseline 无 suffix 目录隔离——**当时正是意识到了这个坑**（脚本 L144-145 注释「sink-guard 臂目录隔离，避免覆盖原 baseline 文件」）；
  - 冒烟 n=20 行数 < 200 不触发跳过（脚本 L170-171 注释确认该防误触发逻辑被显式考虑）；
  - 全部 kvcf 日志 0 次 SKIP 触发。
- 结论：**未踩但高危**（报告的 P2 定级恰当）。未来任何 kmeans/预算参数扫描若复用此脚本须先加指纹。

### 修复建议
输出目录/文件名加入 config hash（method+capacity+window+kernel+pooling+sink_guard+split+n+model），或输出 manifest；仅同指纹完整输出可复用。无需立即动作。

---

## 对 E109 在跑海选的总污染面结论

1. **AVG5 已入账 9 臂与 prescore 头部（mavg×2/cavgsim/aavg 单池）全部零污染**——海选当前排名与已收割判决可信。
2. **唯一在产污染 = C3 × ccluster 分区臂**：远程 .251 的 cclustersim(.125,.25,.125)（prescore #3）五任务数据 + 未来 cclusterkm 决胜臂数据。建议立即修复 C3（一处、新进程自动生效），5 个已落袋污染臂（有效竞争者 2 个）修复后重跑；若不修则全程标注「同一缺陷口径下的相对比较」。
3. C1/C2 是 sglang serving 部署路径缺陷（历史单请求精度数据标注 + 未来部署修复项），与 E109/论文主表（two-level 路径）无关。
4. T1 污染限于 E98 离线 mass 网格的 β≥.625 低 γ 区；论文全部核心 mass 数字与主结论（冠军/平坦性/单池最优/排序反转）干净，fig9 两曲线低 γ 段 + fig10 aavg bar 须在修复后重算重出。
5. T2/T3 均未污染任何已落袋数据，记档防复发。

## 复现资产
- `/tmp/verify_gpt_audit/c1_c2_witness.py` → c1_c2_c3_witness.json（C1 576 / C2 0.6 vs 0.75 / C3 11.31× 三 witness）
- `/tmp/verify_gpt_audit/t1_enumerate.py` → t1_grid_overflow.json（网格溢出枚举，报告反例 1536 精确复现）
- `/tmp/verify_gpt_audit/t1_pollution_map.py` → t1_pollution_per_sample.json（16 样本实际污染点映射，fig9/fig10/选举点逐点判定依据）
