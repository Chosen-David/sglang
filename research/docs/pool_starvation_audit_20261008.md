# 候选池不足（pool starvation）缺陷族专项审计 — tli_indexer.py

- 日期：2026-10-08
- 审计对象：`/home/wangyuanshuo02/sglang/two-level-attention/sparse_attn/indexer/tli_indexer.py`（主树 HEAD，含 C3 scale 修复）与 `/tmp/near_fix/sparse_attn/indexer/tli_indexer.py`（near-SWA 边界修复副本，未合入，diff 仅两组 hunk：`_maybe_build_kmeans` 与 `compute_mask` 的 near_base 分支化）
- 权威口径：`/home/wangyuanshuo02/sglang/TASK.md`（引用行号均为该文件）
- 方法：CPU only、`torch.set_num_threads(1)`、`sys.dont_write_bytecode`、types.ModuleType 双版本并行加载；每个判决均有数值复现（探针脚本 `/tmp/pool_audit/probe_*.py`，场景 S=4352/seed=31/2 kv-head/4 q-head/D=128/block=64/K1=128/K2=1024 除注明外）
- 纪律：只读审计，未改任何生产 `.py`

---

## 1. 执行摘要（三判决表）

| 项 | 判决 | 一句话 |
|---|---|---|
| **A：mavg a.125/b.375/g.625 mid=384/512 短缺** | **合法行为（TASK.md 严格 γ 语义 + 池独立），N5 断言本身过时须改** | mid=512 = far 0（γ 饱和 far_budget=0，γ 悬崖合法坍缩）+ near 512（near 池候选=α·mid_L=512 < 配额 768，池内截断）；主树 384 = 512 − 128（N1 边界 bug 再偷 128，这部分不合法）；该配置本身违反 TASK.md L145-147 约束（near_budget_token 1920 > near_L 512），属 L216-219「没有意义」配置集 |
| **B：N2 α=1 far_hi≠192** | **far_hi=192 断言其实通过；FAIL 真因是另一个未知的 P1 控制流洞（C-1）** | near_fix 建簇侧 far_hi=192 正确；FAIL 在 `mask_new[..., :128].all()`——far token 池空（near_blks==sink_blocks）时 `compute_mask` L999 的 `if far_tok_hi > far_tok_lo:` 把 near 池选择与 sink/swa 强制整体旁路，静默落入 L1081 单池兜底：sink 不保证（实测 False）、mid 预算失守（实测 896>768）、短序列退化全注意力。实现 bug，测试期望（sink 必强制）是对的 |
| **C：缺陷族全量审计** | **two-level 路径无 sglang S1/S2/S3 式哨兵/-inf 直通；1 个新 P1（C-1 控制流洞）+ 1 个 P1 待合入（N1 已修）+ 5 个 P2** | 全部 topk k>len 已 min() clamp、k=0 安全、索引恒为真实位置（全长 bool mask 语义结构免疫）；风险集中在「池空/池饿时的静默退化」而非越界 |

**严重级最高的三个发现**：
1. **C-1（P1）**：far token 池空 → 整个 L2 分区块（含 near 池、sink/swa 强制、K2_mid 预算）被跳过，落入单池兜底（tli_indexer.py L999-L1080 缩进结构）。主树被旧边界 bug 掩盖（α=1 时 near_blks=4 far 非空），near_fix 合入后 α=1/短序列必然触发。
2. **N1（P1，near_fix 已修待合入）**：near 左界从 kt·bs 而非 swa 起点前推，所有 e64 分区臂 near 系统性少 128、far 多 128。
3. **C-4（P2）**：L2 topk 从 softmax 零分位选 token 绕过 L1 门控（β≥~0.914 才触发；实测 β=.9375/γ=0 时 far 有限候选 512、实选 768，256 个未粗筛 token 入 mask）。E98/E109 网格步长（γ≥0.125、β≤0.875、K2=1024）实测不可达，历史数据零污染。

**near_fix 合入建议：条件 GO**（详见 §6——必须与 C-1 修复同 commit，且须拍板 E109 数据口径重跑时机）。

---

## 2. A 根因判决：mavg a.125/b.375/g.625 @S=4352 的 mid=384/512

### 2.1 精确分解（数值实测，probe_a.py）

| 版本 | far 选中 | near 选中 | mid 总 | near_blks | far 区宽 | near 区宽 | nt_near | far_budget |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 主树 | **0** | **384** | **384** | 60 | 3712 | 384 | 768 | 0 |
| near_fix | **0** | **512** | **512** | 58 | 3584 | 512 | 768 | 0 |

中间量（TLI_DEBUG 实录，near_fix）：`kt=68 near_blks=58 swa_lo_blk=66 nb_far=80 nb_near=48`；`K2=1024 K2_mid=768 sink_tok=128 swa_tok=128 nt_near=768 far_budget=0 k2_far=0`。

L1 层：far 池选 80 块、near 池选 48 块，但 near 池块区间 [near_blks, swa_lo_blk) 主树仅 6 块 / near_fix 8 块，nb_near=48 ≥ 两者 → **near 区全部块进 L1**（无 L1 饥饿）。短缺全部发生在 L2。

### 2.2 逐层根因

1. **far=0 的根因：γ 饱和（合法）**。`nt_near = min(nb_near·bs·γ, K2_mid) = min(48·64·0.625=1920, 768) = 768` → `far_budget = max(0, 768−768) = 0`（L1008/L1026-1027）。这是已知 γ 悬崖：阈值 γ* = K2_mid/(β·K1·bs) = 768/3072 = **0.25**，γ=0.625 远超 → far 零召回。与 S 无关（K2 固定），即 E98 冠军臂 v2 崩盘（-17.7）的同一机理。
2. **near=512 的根因：near 池候选不足 + 池内截断（合法）**。near 区宽 = near_L = α·mid_L = 0.125·4096 = 512（near_fix 正确口径）；`k2_near = K2_mid − k2_far = 768`，但 L1071 `min(k2_near, near_p.shape[-1])` 把 k 截到池宽 512 → near 全选 512。剩余 256 预算**不回补 far**（far 已按名义 far_budget=0 选完）。
3. **主树 384 中有 128 是 N1 bug 贡献（不合法部分）**：主树 near 左界从 kt·bs=4352 前推 → near_blks=60 → near 区宽 384 = 512−128，被 swa 边界 bug 偷走。

### 2.3 TASK.md 对齐核查（逐条）

| TASK.md 行号 | 内容 | 核查结果 |
|---|---|---|
| L137 | `near_L = alpha * mid_L` | near_fix 的 near 区宽 512 = 0.125·4096 ✓；主树 384 ✗（N1 bug） |
| L55-57（示例） | near 邻接 swa、far 邻接 sink | near_fix 从 swa 起点前推 ✓ |
| L138-139 | near_budget_page_topk = K1·β = 48；far = 80 | L1 实现一致（L844-847）✓ |
| L142/L152 | near_budget_token = nb·page·γ = 48·64·0.625 = **1920** | 实现 clamp 到 K2_mid=768（L1008）——γ 严格化决策（E109a，去 64 保底）的既定口径 |
| L143 | far_budget_token = budget_token − near_budget_token = 768−1920 < 0 | 实现 max(0,·)=0 ✓（同上既定口径） |
| **L145-147** | **约束 near_budget_token ≤ near_L** | **1920 ≤ 512 违反** → 本配置属 L216-219「不满足约束……没有意义」配置集 |
| L154-155 | near/far 分别在自己的分区内筛选自己的 token | near 只能在自己 512 token 里选 → 输出 512；剩余预算不跨池给 far ✓ |

### 2.4 判决

**合法行为**。两个独立理由：①γ 悬崖坍缩（far_budget=0）是 E109a γ 严格化的既定语义，E105/E109 全部坍塌臂数据按此口径落袋；②near 池候选 < 配额时池内截断、剩余不跨池转移，符合 L154-155 池独立语义——且该配置本身违反 L145 约束，属「没有意义」集，扫描端理应过滤（等效配置裁剪已做的正是这件事）。

**N5 断言过时，须改**：「mid 总选恒 = K2_mid」只在两池候选 ≥ 各自配额时成立。应改为分段断言：
- 饱食配置（如 ccluster .5/.25/.5：near 宽 2048 ≥ 768，实测 mid=768 ✓；健康 γ 臂 .125/.375/.125：384+384=768 ✓）→ mid == K2_mid；
- 饥饿配置（mavg .125/.375/.625）→ **期望值 512**（near_fix）= far_budget(0) + min(k2_near, near 区宽)。

**附带口径发现（非 bug，须论文/表述注意）**：mid < K2_mid 的饥饿只在**小 S** 咬人（near 区宽 = α·mid 随 S 增长；S=32K 时该臂 near 宽 ≈4064 ≥ 768，mid 守恒 768、全来自 near）；γ 悬崖 far=0 在**所有 S** 咬人。两个现象须分开表述。

---

## 3. B 根因判决：N2 α=1

### 3.1 事实链（probe_n2b.py 实测）

- `fh_old = 256`（断言 ✓，主树 α=1 far 仍留 128——N1 bug 实锤）
- `fh_new = 192`（**断言 ✓ 通过**——near_fix 建簇侧 `far_hi_blk = max(sink_blocks+1, (S−swa−near_len_dyn)//bs) = max(3, 2) = 3` 正确）
- `mask_new[..., :128].all() = False`（**FAIL 点**）
- `mask_new[..., S-128:].all() = True`；mask 总/token = 1024（非分区口径的 1024 全预算）

### 3.2 真因：compute_mask L999 的控制流洞（新发现，P1）

精确缩进核查（主树与 near_fix 该段相同，near_fix 未触及）：**L1045-L1080（near 池选择 k2_near、i_n topk、topk_mask scatter、sink/swa 强制置位、return）全部嵌在 L999 `if far_tok_hi > far_tok_lo:` 之内**。far token 池空（`far_tok_hi == far_tok_lo`，即 near_blks == far_lo_blk == sink_blocks）时整块跳过，落入 L1081 单池兜底 `topk(p, K2)`。

后果（全部数值实测）：
| 场景 | 触发 | 实测后果 |
|---|---|---|
| α=1（4bit/sim_greedy/cluster 三 method 均复现） | near_blks=2=sink_blocks | mid=896 > K2_mid=768（**预算失守**）；sink 强制不保证（sink_all=False，靠 p 概率侥幸入选的 token 不齐） |
| 短序列 S=200/300/320（e64 臂） | mid ≤ 1 块使 near_blks 触底 | `topk(p, min(S,1024)) = S` → **全注意力**（预算语义完全旁路；S=320 主树旧边界则 mid=0 预算全空转） |
| 主树为何从未触发 | 旧边界 α=1 时 near_blks=4（far 非空）；E109 网格 α≤0.875、S 大 | 洞被旧 bug 掩盖——**near_fix 合入会把暴露面从「理论可达」变成「α=1/短序列必然触发」** |

### 3.3 判决

**实现 bug，测试期望正确**。N2 测试对 far_hi=192 的期望（建簇侧 sink+1 块保底）与对 sink 强制的期望都合理；FAIL 不是 `_km_far_hi` 计算链的问题（那部分 near_fix 已对），而是消费端控制流洞。修复点见 §5-1。测试本身唯一要改的是注释归因（「far_hi 建簇侧为何不是 192」的提问方向错了——它就是 192）。

---

## 4. C 候选池不足缺陷族全量清单

行号均指主树 `sparse_attn/indexer/tli_indexer.py`（near_fix 该两处行号 +2~+9）。

| # | 缺陷 | 行号 | 触发条件 | 后果 | 严重级 | 测试覆盖 |
|---|---|---|---|---|---|---|
| C-1 | **far token 池空 → L2 分区选择整体旁路落单池兜底**（near 池/sink 强制/K2_mid 预算全跳过） | L999 guard 包裹 L1000-1080 | near_blks == sink_blocks：α=1、短序列 mid≤1 块、α·mid 舍入 < bs | sink 不保证、mid 超预算（896>768）、短序列全注意力、零日志静默 | **P1** | N2 捕捉（断言归因错） |
| C-2 | near 左界从 kt·bs 而非 swa 起点前推（N1） | L797、L335 | 所有 e64 分区臂 | near 系统性少 128、far 多 128；α=1 far 仍留 128 | **P1** | N1 红绿对拍 ✓（near_fix 已修） |
| C-3 | near 簇分未乘 softmax_scale | L611-614（已修） | near_select=cluster/sim_greedy 分区臂 | 簇段放大 ~11.3× 霸榜 | P0 | test_c3_near_scale_fix 5/5 ✓（HEAD 已闭环） |
| C-4 | **L2 topk 从 softmax 零分位选 token 绕过 L1 门控**（p 非 L1 位是 0.0 非 -inf，L928-929） | L1031-1033（far）、L1071（near） | k2_far > nb_far·bs（β≥~0.914 且 far_budget 大）；near 侧理论上 β<0.03 才可达（不可达） | 位置合法无 OOB，但两级筛选契约破坏，mid 构成含未粗筛 token；实测 β=.9375/γ=0：far 有限 512、实选 768（256 绕过）。**E98/E109 网格（γ≥.125, β≤.875, K2=1024）实测不可达 → 历史数据零污染**；K2=2048 且 β≥.875 且 γ≤~.107 的未来扫描须防 | **P2** | 无 |
| C-5 | **far→near 单向剩余配额转移**（选择顺序副作用非设计） | L1045 `k2_near = K2_mid − k2_far` | far 池候选不足（far 宽 < far_budget，α 大） | 实测 a.875/b.125/g.125：near 名义 128 实得 256；与「剩余预算不跨池转移」表述冲突（只挡 near→far）；γ 作为「严格配额」的表述在此失效 | **P2** | 无 |
| C-6 | 主树短序列 far 池吞 swa 块（near_blks > swa_lo_blk 时 far 上界未 clamp） | L803、L804 | near_len_dyn < swa_tok（极短序列/极小 α·mid） | far topk 花 far_budget 选 p=1.0 的 swa token（反正被强制，预算白花）；实测 S=320 主树 mid=0 | P2 | 无（near_fix 顺带修复：near_base=swa_lo ⇒ near_blks ≤ swa_lo_blk−1） |
| C-7 | 建簇侧 far_hi 用真实 S、消费侧 near_blks 用 pad 后 kt·bs 的边界漂移 | L335 vs L803/L990 | S 非块对齐（实测 S=4353：建簇 2176 vs 消费 2240） | far 区 ≤1 块条带簇分数不可选；near 簇右缘 ≤ bs−1 token 回退细筛（C3 修复设计内） | P2/info | 无 |
| C-8 | far 区过小时 cluster/sim_greedy 静默退化 4bit 路径 | L357-359（Tfar<far_clusters·4→centroids=None）、L392-395 | far 区 < 1024 token（kmeans）/ 128（greedy） | far_method=cluster 臂静默变 4bit 精筛——扫描数据口径须标注 | P2 | 无 |
| C-9 | 建簇侧 `max(sink_blocks+1,…)` 与消费侧 `max(sink_blocks,…)` 的 1 块差 | L335 vs L803 | far 区空 | 保底簇 [128,192) 建而不用（浪费一次建簇，无害）；即 N2 期望 192 的出处 | info | N2 |
| C-10 | （阴性发现）哨兵/-inf 槽位直通 | 全文 | — | **two-level 路径无 sglang S1/S2/S3 式缺陷**：L1 score_coarse 全有限（b_score 覆盖，-inf 仅 skip_far 且 k 已 clamp L848-853/897-901）；全部 topk k>len 已 min() clamp（L859/869/902/1017/1031/1071）；k=0 安全返回空；i_f/i_n 恒为真实位置进全长 bool mask（结构免疫哨兵泄漏） | — | 既有 6 组回归 |

另记一个观测性小缺陷：L1036 `far_p_finite` 调试口径误导（p 无 -inf，恒等于池宽，不是 L1 门控后的有限数）——本次审计 C-4 的真实有限数须从 nb_far·bs 推导。

---

## 5. 修复建议（只写不改，全部须按「修复改行为 → 验证 → 重测」纪律走）

| # | 修复点 | 精确位置与改法 | 风险分级 |
|---|---|---|---|
| 1 | **C-1 控制流洞** | L998 `nt_near = 0` 后加 `k2_far = 0`；把 L1042-L1080（near 池选择 + sink/swa 强制 + return）移出 `if far_tok_hi > far_tok_lo:`（guard 只保留 far 选择段 L1000-1041）；L1073 `i_n = i_f[..., :0]` 依赖 i_f——far 空时初始化空 i_f（如 `i_f = torch.zeros_like(p[..., :0], dtype=torch.long)` 形状对齐 scatter）。修后 α=1 → far 0 + near 768、sink/swa 强制恢复；短序列 → mid = min(K2_mid, near 宽) 而非全注意力 | **低风险可直接修**（不触任何在跑网格臂的行为：α≤0.875 大 S 下 far 池恒非空，逐位不变可程序化断言） |
| 2 | **N5 断言更新** | test_near_swa_boundary.py N5：分段断言——ccluster .5/.25/.5 期望 768；mavg .125/.375/.625 期望 **512**（并注明 TASK.md L145 约束违反 = 饥饿语义依据） | 低风险可直接修（测试文件） |
| 3 | **N2 注释归因修正** | test_near_swa_boundary.py N2 docstring：FAIL 真因是消费端控制流洞非建簇侧；far_hi=192 断言保留 | 低风险（测试文件）；**断言本身与 §5-1 修复是红绿对**——先修 1 再跑 N2 应转绿 |
| 4 | **C-4 零分槽位** | L1031 的 `far_tok_hi - far_tok_lo` 改为「L1 门控后有限 token 数」（等价 `min(nb_far, far 区块数)·bs`）；near 侧 L1071 同理 clamp 到 `min(nb_near, near 区块数)·bs`。**须用户/主 AI 拍板**：改变 β≥.914 臂行为（历史网格未触及，但语义选择「严格两级契约」vs「预算优先」是口径问题） | 须拍板 |
| 5 | **C-5 单向转移口径** | 二选一须拍板：①严格配额——k2_near 改 `min(nt_near, …)`（far 饿死时 mid < K2_mid）；②充分利用对称化——far 在 near 之后补选剩余。当前实现是「事实上的③单向转移」。影响论文 γ 语义表述（γ 是 near 配额下界而非精确值，当 far 池饿时） | 须拍板 |
| 6 | C-7 边界漂移 | `_maybe_build_kmeans` 建簇基准从真实 S 对齐到 pad 后 kt·bs（与消费端同源），或接受 ≤1 块覆盖损失并在代码注释注明 | 低风险可直接修（或仅注记） |
| 7 | C-8 静默退化 | far 区过小退化 4bit 时打一行日志；扫描结果 JSON 标注 `far_degraded: true` | 低风险可直接修 |
| 8 | 观测性 | L1036 far_p_finite 改打 L1 门控后的真实有限数 | 低风险 |

---

## 6. near_fix 合入主树 GO/NO-GO

**条件 GO**，三个前置条件：

1. **必须与 C-1 修复同一 commit 合入**。理由：near_fix 把 α=1/短序列从「旧 bug 掩盖下不可达」变成「必然触发 C-1 洞」（sink 失守 + 预算失守）。单独合入 near_fix = 修一个 bug 放大另一个 bug 的暴露面。合入后 N2（sink 断言）应转绿，形成完整红绿对。
2. **N5 断言先改**（§5-2），否则合入后测试套仍红。
3. **E109 数据口径须用户拍板**：near_fix 改变**所有 e64 分区臂**的边界（near 区 +128 token）→ 已落袋 v2 数据（海选头部 0.01 级密集）与主树旧口径绑定。两个选项：
   - **a（推荐）**：等 E109 海选收官、冠军定档后，与全量三件套（LB v1 16 任务 + LB v2 + RULER）绑定新口径重跑——单臂 128/768 = 1/6 预算构成位移在海选噪声内，不值得中途全重跑；
   - **b**：立即合入 + 头部臂（AVG5 前 4）换 postfix 重跑验证 delta（~6-8 GPU·h）。
   两种都遵循「修复会改行为 → 换 postfix 防同臂目录新旧混写」纪律。

near_fix 两组 hunk 本身质量：N1/N3/N4 红绿对拍 + 本次 ccluster/mavg 边界数值验证（far_hi 2304→2176 / near 1920→2048；(0,0) 与老逻辑逐位不变）全部通过，且顺带修复 C-6。**合入内容 = near_fix hunk + C-1 修复 + 测试 N2/N5 更新，一个 commit。**

---

## 附：证据脚本索引（可复现）

- `/tmp/pool_audit/probe_n2.py / probe_n2b.py`：N2 真因（fh=192 通过、sink_all=False、单池兜底 1024/token）
- `/tmp/pool_audit/probe_a.py / probe_a2.py`：A 分解（far 0 / near 384-512、ccluster 768 守恒、健康 γ 768、α=1 mid=896、短序列全选）
- `/tmp/pool_audit/probe_dbg.py`：TLI_DEBUG L1/L2 中间量 + C-4 零分槽位实锤（far_finite=512 vs 实选 768）
- `/tmp/pool_audit/probe_edge.py`：非对齐 S=4353 / 短序列 / α=1 三 method 冒烟
- `/tmp/pool_audit/probe_transfer.py`：C-5 单向转移实锤（near 名义 128 实得 256）
