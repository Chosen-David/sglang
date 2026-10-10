# 20×H20 多机多卡编排建议：BeeGFS 单一协同面（v2，取代手工 rsync 方案）

日期：2026-10-08（v2 修订：吸取"手工 sync 不可持续"反馈 + 通道实测结果）。作者标识：by_kimi3（建议类文档，待主 AI 核验后 adopt/adapt/defer）。
数据依据：screen4 链日志实测速率；`/tmp/e109_scan_v2` 实况（14:14 快照）；远程通道实测（rssh/rrsync/mount/git 探测，见 §2）。

---

## 1. 资源盘点

| 主机 | GPU | 现状 | 备注 |
|---|---|---|---|
| 本地（sglang 主树） | 2×H20 | screen4 链在跑 | 打分/落袋/commit 在此机 |
| wys_8xh20（10.238.9.187） | 8×H20 | 待接入 | **本机尚无 ssh 包装脚本**；BeeGFS 挂载待验证 |
| wys2-8xh20（33.32.33.24） | 8×H20 | 远程 screen4 曾在跑 | F5：若仍在跑坍塌 sim 臂，先杀；BeeGFS 挂载待验证 |
| 2xh20-8100（10.238.139.251） | 2×H20 | 远程 sim 臂在跑 | ssh 通道已通（实测）；BeeGFS 挂载待验证 |

## 2. 通信现状实测（v2 新增）：为什么是手工 sync，以及它坏在哪

| 通道 | 现状机制 | 实测问题 |
|---|---|---|
| 控制 | `/tmp/rssh.sh`（sshpass+ssh，密码硬编码） | **只配置了 .251 一台**；另两台 8 卡机本机无通道 |
| 数据 | `/tmp/rrsync.sh`（sshpass+rsync 拉回） | 拉模式靠人记；远程 /tmp 不在备份纪律内（N2） |
| 代码 | **无任何同步**：远程 `~/two-level-attention` 是独立 git repo（HEAD `018fbb4`，与主树不同 lineage），文件年代混杂（pred.py 5 月版 / tli_indexer.py 10-07 版） | **已漂移**——E109 未出事只因 aavg 路径对修复惰性；E110/full13 必踩 |
| 协调 | marker 日志 grep（本地/跨机 ssh） | B07/F02 脆弱协议：不绑定 run、失败 marker 也放行 |
| **模型/共享数据** | **BeeGFS FUSE 真共享**（`beegfs-fuse`，mount 表实证） | ✅ 唯一健康通道——全链协同应建立在它之上 |

## 3. 目标架构：BeeGFS 单一协同面（消灭手工 rsync）

原则：**代码、结果、调度、日志四类信息全部走 BeeGFS 共享目录；ssh 只用于一次性启动 worker 和应急 kill，不再承担任何日常同步。**

```
/mnt/dolphinfs/ssd_pool/docker/user/hadoop-scale-llm/tli_runs/<stage>/
├── code/                 # 冻结代码快照（git archive + sha256 manifest）
├── results/pred_<tag>/   # 各 runner 直写（先写 *.tmp 再 rename，原子发布）
├── manifest/             # 调度面：jobs/ pending → running → done/failed
│   ├── pending/<tag>__<task>
│   ├── running/<tag>__<task>   (含 host/gpu/pid/start_ts)
│   ├── done/<tag>__<task>      (含 rc、行数、耗时)
│   └── failed/<tag>__<task>    (含 rc、stderr 尾巴)
└── logs/<tag>__<task>.log
```

四个机制要点：

1. **代码单源**：`git archive` 主树（或 E110 worktree）到 `code/`，附 `sha256sum` 清单；各机 `PYTHONPATH` 指向它。**任何机发现 hash 不符拒绝启动**——代码漂移从"靠记忆"变成"启动期硬失败"。
2. **结果直写**：runner 的 `--output-dir` 直接指向 BeeGFS `results/`；`pred.py` 已是"全部样本完成后一次性写出"，再改成"写 `.tmp` + rename"即原子发布（E109 在跑链不改，见 §6）。**从此没有结果回收这一步**——打分机读的就是同一目录。
3. **调度用原子 claim**：worker 循环 `mkdir manifest/running/<job>`（mkdir 在网络 FS 上原子）——抢到即跑，写完状态目录即释放。**天然消除 F6 双跑**，且任意主机随时 `ls manifest/` 得全局进度，marker grep 协议整体退役（B07/F02 连带治愈）。
4. **日志直写** BeeGFS `logs/`：故障排查不用登机器。

实现量估计：一个 ~150 行的 `dispatch.py`（枚举缺口 → mkdir claim → 调 pred → 写状态）+ 每机一条启动命令。不需要常驻守护、不需要数据库。

## 4. 调度与分片（静态大包 + 动态 claim 兜底）

纯静态分片的缺陷：各机速率不可预知（sim 臂方差大），快的机闲置。改为**两层**：

- **预分片**（避免长尾集中）：sim 臂 gov_report 按臂均匀预分到 4 台主机（写进各机 `prefilter`），防止最后一台机被 5 小时的 job 卡住；
- **动态 claim**：其余 job 全部进 `manifest/pending/`，各机 worker 自由 claim——快的多吃，天然负载均衡。

角色分配（预分片只针对长尾）：

| 主机 | 预分片（长尾 job） | 动态池角色 |
|---|---|---|
| 本地 2 卡 | 不动，跑完当前链 | 收官后并入 |
| wys 8 卡 | sim gov ×2 | 主力吃 mminmax/mavg gov |
| wys2 8 卡 | sim gov ×2 | 主力吃 cluster/sim 臂 |
| 2xh20 2 卡 | E110 smoke → E110 全量 | 独立工作包 |

## 5. 速率与工作量（v1 数据，未变）

- 实测：qasper ~13min / repobench-p ~93min / gov_report ~165-180min（正常臂，n=200/500）；sim 臂 1.4-8×；
- screen4 剩 93 job ≈ 180±30 GPU·h；全链（E110+full13+RULER+SG）≈ 300±50 GPU·h；
- 20 卡无重复、无闲置时：screen4 ~10-14h 收官，全链 ~2-3 天。

## 6. 迁移路径（不动在跑链）

| 阶段 | 做法 |
|---|---|
| **E109 screen4 在跑链** | 完全不动（本地 /tmp + 链自带断点）。收官时**一次性** rsync 远程产出回本地 + 重跑 `e109_score_v2.py`（N1）——这是最后一次手工 sync |
| **E110 / full13 / RULER** | 直接生在 §3 架构里：BeeGFS code 快照 + 结果直写 + manifest 调度 |
| **SG e2e/测速** | 与 HF 链无依赖，2xh20 或任意空闲卡插空（先合 S1/S2/S3 修复 + L2424 漏点 + 多行对拍测试） |

## 7. 前置检查清单（启动前逐项过）

1. 三台远程机 `mount | grep bgfuse` 验证 BeeGFS 已挂载且路径一致；
2. 在 BeeGFS 目标目录做写权限与小文件 rename 测试（quota/权限）；
3. wys/wys2 的 ssh 包装脚本落地（只用于启动 worker 与应急）；
4. 杀 wys2 上仍在跑的坍塌 sim 臂（F5：cavgsim/cclustersim 的 g0.75 族，~5h/job 纯白烧）；
5. `code/` 快照 sha256 清单生成并在各机首启时校验；
6. full13 启动前修编排 F2（repobench-p glob + AVG 缺臂）与 F4（marker 冲突）——在新架构里 F4 自然消解（调度不看 marker），F2 由打分脚本的统一前缀函数消解。

## 8. 时间线（按 §3 架构落地顺利）

| 时点 | 里程碑 |
|---|---|
| 10-08 16:00 | §7 检查过完；`dispatch.py` + BeeGFS 目录骨架就绪 |
| 10-08 17:00 | wys/wys2 worker 上线吃动态池；2xh20 跑 E110 smoke |
| 10-09 04:00-08:00 | screen4 收官 → N1 打分修复 → AVG5 排名定稿 |
| 10-09 12:00 | E110 全量收官 → cluster 对决判决（新架构下结果已在共享盘，直接打分） |
| 10-09 晚 ~ 10-10 | full13 冠军臂（新架构首批全程） |
| 10-10 ~ 10-11 | RULER + SG e2e 穿插收官 |

## 9. 验收口径

- 调度面健康：`manifest/failed/` 为空或逐项有人工签注；无同一 job 的 done×2（claim 失效证据）；
- 结果面：每臂 5 任务行数齐 + `e109_screen4_selection.json` 无 None cell；
- 代码面：各机启动日志的 code sha256 与快照清单一致；
- 收尾：BeeGFS `results/` 全量 rsync 一份进仓库 `exp/results_longbench_e109_v2_backup/` 落袋（共享盘 ≠ 备份纪律）。

## 10. 与 v1 的差异说明

v1 方案是"静态分片 + 各机 /tmp + rsync 回收"，v2 基于两条实测修正：① 代码同步缺失已造成远程副本漂移（§2），手工 sync 不可持续；② BeeGFS 是全机真共享层，协同面应整体建立在它上面，rsync 降级为"收官落袋"的一次性动作。v1 的速率模型、工作量账目、工程纪律（N1/F5/前缀/不编辑在跑脚本）全部保留有效。

---

## 11. v3：基于文献与实测的六个进一步优化（2026-10-08 追加）

文献基线：Hyperband（Li et al., JMLR 2018, arXiv:1603.06560）、ASHA（Li et al., MLSys 2020, arXiv:1810.05934）、MapReduce backup tasks（Dean & Ghemawat, OSDI'04）、LPT 调度界（Graham 1969）、lease 模式（Chubby 式）。负载实测：远程卡为 **H20-3e 143GB**（nvidia-smi 实测）。

### O1（收益最大）GPU 多路复用：短 job bin-packing

H20-3e 143GB ÷（Qwen3-8B bf16 ~16GB + 32K KV ~10GB + 开销）≈ **3-4 个短任务进程/卡**。当前 1 job/卡对 qasper/musique/hotpotqa/repo（占 GPU·h ~40%）浪费巨大。规则：短任务池 3/卡、gov_report 与 sim 臂独占。预期省总时长 25-30%。前置：小规模验证 OOM 与降速比（2/3/4 路各测一臂）。等效卡数 20 → ~40-60。

### O2 ASHA 式逐轮早停（面向 full13/RULER/未来扫描）

把扫描切 rung：便宜任务组合（musique+qasper ≈34min/臂）跑全部臂 → 配对 CI 剪枝 → 幸存臂才碰 gov_report（180min）。现有 γ 悬崖快筛即 rung-0 的手工版，ASHA 将其形式化。**纪律：剪枝门用配对置信区间（top-2 差 0.01 时全保留），不硬切排名**——与本席配对检验证据一致（同族差异多为噪声）。预期省未来扫描 40-60% GPU·h。

### O3 长尾投机执行（backup tasks）

动态池见底只剩 straggler 时，同一 job 允许第二 worker 冗余 claim，先完成者赢（rename 原子，后到者见 done 即弃）。配合 LPT（按估时降序投 job）。把 sim gov（3-5h）长尾砍掉一半量级，用收尾闲置卡，零成本。

### O4 Lease + 心跳自愈（修 v2 的真实漏洞）

mkdir-claim 的失败模式：worker 死在中途 → job 永卡 running/。修法：running 目录内置 heartbeat 文件周期 touch；任意 worker 兼任 janitor 回收超期 lease 回 pending。重跑幂等（行数 SKIP + 同名覆盖 + 崩溃无 partial）→ 回收零风险。

### O5 长 job 任务内断点

sim gov 3-5h 崩溃全损（pred.py 一次性写出）。新阶段改逐样本 append+flush（pred_ruler.py L125 已是此模式）+ 状态文件记完成数 → 崩溃损失 ≤1 样本，支持抢占/迁移。

### O6 在线增量打分

scorer 改 watcher：done 落地即更新部分表——末 job 完成时排名立读（消灭收官延迟），且 O2 的 rung 决策可飞行中执行。

### 确认不做（避免过度工程）

- **Ray/Kubeflow/Slurm 全家桶**：300 job × 20 卡，~200 行 dispatcher + BeeGFS 即最优复杂度；Pollux/Gavel 类 gang scheduler 面向多卡训练联合调度，与本负载（ embarrassingly parallel 评测）不匹配；
- **worker 常驻模型（省加载）**：方向对但需重构 pred.py 为可调用 harness；仅当实测模型加载占短 job >5% 时启动，记 backlog。

### 更新后的落地顺序

dispatch.py（v2 骨架）→ O4 lease（骨架自带，必须先有）→ O1 多路复用验证（当天可测）→ O5 append 写出（E110 起生效）→ O6 watcher 打分 → O3 投机（池见底逻辑）→ O2 ASHA rung 协议（full13 设计时冻结）。

---

## 12. v3.1：与 agent 库 cluster_planning 模块的对表与取舍（2026-10-08 追加）

对表对象：`~/agent-research-workflows`（Chosen-David/agent）新增 `agent_runtime/cluster_planning.py`（`plan_wave` + `verify_local_artifact`）与 `workflows/cluster_orchestration_workflow.md`。该模块是**纯建议式波次放置规划器**：不预留、不 SSH、不启动，输出 advisory JSON，由可信执行端原子准入。

| 库的设计 | 取舍 | 理由 |
|---|---|---|
| inventory 快照纪律（≤300s、observed_at、实测链路、快照≠租约） | **adopt** | dispatcher 启动前采集 per-host GPU 快照带时间戳，替代静态假设 |
| verified_task_ids 只来自独立验收，worker 自报不算 | **adopt** | 与 B06/B07/F02 教训同构；manifest `done/` 经 scorer 验证后才喂下游 |
| `verify_local_artifact`（流式 SHA-256 + 字节上限 + 变更检测） | **adopt（直接复用）** | 代码快照/数据集的启动校验，省得自写 |
| 灰度验收协议（单卡→同机两卡→两机组→网络中断） | **adopt** | 作为 dispatcher 自身的上线 rollout 协议 |
| "规划竞争同卡 → 可信端原子准入整组" | **架构互证** | 与 mkdir-claim 独立收敛到同一模式，不改 |
| gang 放置（多机同构组 + 双向 RDMA 阈值） | **defer** | 当前负载为单卡独立 job；仅未来 SG TP>1 测试可能启用 |
| 整卡分配粒度（Whole GPUs only，禁 MIG） | **reject（对本负载）** | O1 多路复用（3-4 job/卡）是最大收益点，粒度留在我们的 dispatcher 层 |
| 后端迁移（Slurm/Ray/MultiKueue/SkyPilot） | **reject（对本规模）** | 20 卡 300 job 的 BeeGFS manifest 已是最优复杂度；库自身亦注明大规模才需专门后端 |
| 点对点传输规划 + hash cache | **defer（兜底）** | 有 BeeGFS 真共享则无需传输；某机未挂载时启用该模型 |

结论：库提供"纪律与原语"（快照、验收、校验、灰度），不提供"我们的调度器"；BeeGFS manifest 执行层维持自建，四处糙点用库的原语补齐。
