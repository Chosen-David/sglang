# E113：method 组合 × kernel 设计——实证篇（Triton 原型 + SEG-GREEDY 仿真 + microbench）

> 定位：本文档是 **kernel 化设计文档的实证篇**，与主树设计篇
> `/home/wangyuanshuo02/sglang/research/docs/sim_greedy_kernel_design.md`（#150，CUDA C++ persistent
> kernel V1 主线）互补：该文档给「应该怎么设计」，本文档给「已落地的 Triton 原型逐位等价实证 +
> 方案2 界验证仿真否决 + microbench 数据」（速度数据已于 **2026-10-10 按 050 修复版在空闲
> GPU 上重跑闭包**，正式证据为 `results/e113_microbench_v2.json`，见 §2.3 的 v2 注记；旧
> `results/e113_microbench.json` 降级为**实现身份未闭合的历史观测**存档，保留不动），其结论
> 直接校准设计篇的 V1/V2 决策门。
>
> 关联：#128（overlap kernel 设计）/ #146（E109 v2 远程 sim 臂，13h/臂 直接动因）/ #147（E110
> ccluster e2e）/ #150（E113 设计篇）。全部代码与数据在本 worktree
> `exp/trace/{e113_greedy_triton.py, e113_seg_greedy_sim.py, e113_microbench.py,
> test_e113_greedy_triton.py, test_e113_microbench_identity.py}` + `exp/trace/results/e113_*.json`。
>
> **E113b 更新（e2e 集成完成）**：方案1 kernel 已收编进生产路径
> `two-level-attention/sparse_attn/indexer/greedy_triton.py`（kernel 本体，单一事实源），
> `tli_indexer.py` 的 `_greedy_cluster_pass` 改为调度器（env 开关
> `SGLANG_TLI_GREEDY_KERNEL`，默认 1=kernel，0=Python 循环回退），far 冷启动/
> far decode 增量/near 重建三路径全接；主树 exp/trace 的 e113_* 脚本与 JSON
> 同步落袋（e113_greedy_triton.py 为转发入口）。集成对拍见
> `two-level-attention/test_e113b_kernel_integration.py`。

---

## 0. 结论速览

| 问题 | 判决（正确性列全部实测支撑；速度列为 **v2 正式实测**——050 修复版重跑、身份闭包，见 §2.3） |
|---|---|
| **方案1 Triton 单 kernel 顺序化** | **GO（立即可用）**：与 Python 参考 100% 逐位一致（T2：8 seed × 3 sim × 393216 token 位 0 mismatch；v2 重跑 10/10 case `assign_mismatch=0` 复证），增量续跑 ≡ 全量重放（T3，E110 铁律过）；**2.0-64.9 μs/token**（随 K 扫描量；v2 实测），生产口径（K̄≈2.3K@T=16K）~8.3 μs/token，vs Python 循环 **3.7-211× 加速**（v2 实测；legacy 旧值 3.4-51×，身份未闭合且疑受当时在跑扫描干扰，见 §2.3 新旧对照） |
| **方案2 段式推测 + Cauchy-Schwarz 界验证（SEG-GREEDY）** | **NO-GO（仿真否决）**：算法本身决策级精确（assign_mismatch=0 全 case），但**可证明跳过率仅 0.29-2.9%**、候选集 28-86% 活簇——界在单例密集的贪心态上太松，加速无来源 |
| **方案3 终态路线** | Triton 原型 = **立即止血**（13h/臂 → 估 ~0.6-1.3h/臂，v2 口径）；CUDA persistent V1（设计篇主线，2-3 μs/token 目标）= 终态——Triton v2 实测 ~8.3 μs/token（生产 K̄ 口径）证实了设计篇「8 program/78 SM + 每 token 归约延迟」的担忧（与 V1 差距收窄到 ~3-4×），但 Triton 版零语义风险、当天可上线 |
| method 矩阵 | §4 全表：规则稠密段（块统计/GEMM/kmeans）→ cuBLAS/Triton 已有或低风险；**唯一 launch-bound 段就是 sim_greedy 贪心链**，本原型已解决 |

---

## 1. 根因复述（为什么 13h/臂）

sim_greedy 增量贪心聚类的参考实现 `_greedy_cluster_pass`（tli_indexer.py）是**逐 token Python
循环**：每步 bmm / masked_fill / argmax / gather / 2×index_add / where ≈ 10 次 CUDA op，解释器 +
launch 开销 ~79-84 μs/token（E108 probe 实测），而每 token 真实 GPU 计算（K̄×dd 点积 + argmax ≈
2300×32 MAC）仅 ~2-3 μs——**launch-bound 比例 ~30:1**。E109 v2 远程 10 个 sim 慢臂 × 双任务的
贪心段直接把臂 wall 推到 ~13h。

贪心语义（不可动的铁律，详见设计篇 §3）：

```
for i in 0..T-1:                       # token 维：串行（贪心定义）
    cos = (sums[live]·x_i)/(‖sums[live]‖·‖x_i‖)   # K 维可并行
    a = argmax(cos)；upd = cos[a] ≥ sim            # 全 K 精确 argmax + 最小索引 tie-break
    sums[a] += x_i（归并）或新建簇 k_live          # running-mean 时序耦合
```

并行维度解剖：token 串行（语义）/ K 可并行 / head(8) 可并行 / layer 在 e2e 关键路径上串行。
**打破 token 时序 = 换语义**，所以只有两条正路：①把串行循环塞进单 kernel 消灭 launch 开销（方案1）；
②用可证界的批量推测跳过大部分重算（方案2）。

---

## 2. 方案1：Triton 单 kernel 顺序化（GO，已实证）

### 2.1 设计（`exp/trace/e113_greedy_triton.py`）

- **grid=(H,)**：每 program 独占一个 kv-head 的全部贪心状态（sums/cnt/sq/k_live），无跨
  program 竞争、无原子操作；8 head = 8 条独立链天然并发。
- **token 循环在 kernel 内**：每 token 把 K 分 tile 扫描 sums 行（L2 流量），`cos = dot/(norm·xn+EPS)`，
  tile 内 `tl.argmax`（first-occurrence）+ 跨 tile **严格 >** 比较——早 tile 赢并列 = 全局
  first-occurrence，对齐 `torch.argmax` 最小索引 tie-break（铁律 3）。
- **决策与更新 branchless**：`upd = best_cos >= SIM`；`dst = upd ? best_k : k_live`；
  `sums[dst] = upd ? row+x : x`（新建簇槽必为零槽，与参考实现「零权重 index_add 落零槽」语义一致）；
  `sq` 增量维护 `sq[dst] += 2·dot_a + ‖x‖²`，`dot_a` 从扫描期 dot 直接取（规避 -inf×0=NaN 污染 sq 的
  E108 实测坑）；归并时 per-token `+= x` 与参考实现**天然同累加序**（铁律 2 fp 纪律不触碰）。
- **容量**：wrapper 保证 K ≥ max(k_live)+T（最坏每 token 新建），F.pad 扩容点与参考一致。
- **调优（E113 实测，GPU1 空闲，T=16384；μs/token 绝对值为 legacy 观测，见 §2.3）**：BT=128/nw=8 → 41.0 μs/token；
  **BT=512/nw=4 → 9.8 μs/token（4.2×）**；BT=1024 持平、K=16K 单例臂 512 略优 → 默认 BT=512/nw=4。
  BT 不改语义：dot 归约沿 dd 维（行内 32 元素与 BT 无关），跨 tile 严格 > 链与 BT 无关——调参后
  单测 5/5 逐位全同复验过。chunk 单 launch 越大越快（16384 vs 8192：9.8 vs 17.1 μs/token，
  launch 间隙效应），默认 chunk=16384。v2 重跑（§2.3）绝对值更快（T=16384 结构簇臂
  7.5 μs/token），但 BT/chunk 的**相对优劣结论不受影响**（身份闭包前后同一 kernel 源码）。

### 2.2 语义等价实证（`test_e113_greedy_triton.py`，5/5 PASS）

| 测试 | 内容 | 结果 |
|---|---|---|
| T1 | 冷启动全量对拍：3 组 (T,H,dd,sim)，assign 逐位 + cnt/sq/簇心 allclose | PASS（含簇合并路径，簇数< T 鉴别力门过） |
| T2 | 8 seed × 3 sim 多尺寸逐位一致率 | **393216 token 位 100% 一致，0 mismatch** |
| T3 | 分三段增量续跑 + chunk 构建 vs 全量重放（E110 铁律） | assign 逐位一致，簇数 96/4096 |
| T4 | sim=0.99 全新建簇路径 + 手工向量 running-mean 语义 | PASS（[0,0,0,1] 手工向量判决正确） |
| T5 | 多 head 独立性（同数据复制 4 head → 同 assign） | PASS |

> 教训记录：T1 初版用纯高斯随机数据 FAIL——32 维随机向量近正交（cos≈0），全部单例簇、归并路径
> 零覆盖。**贪心类单测必须用 base+noise 混合簇结构数据**（同 base 的 token 间 cos≈0.9+），
> 否则测试无鉴别力。

### 2.3 性能（microbench v2：正式证据 `results/e113_microbench_v2.json`）

> **v2 注记（2026-10-10 01:45，GPT 审计 `TL-E113-BENCH-PROVENANCE-050` 方案 4 验收完成）**：
> 旧版四缺陷（Python reference 硬编码仓库外绝对路径 / manifest 无身份 / 无 correctness 门 /
> singleton 无固定 seed）已按 050 修复（commit 1d17c3dba），并于 2026-10-10 在**空闲 GPU0
> （无并发扫描的无干扰窗口）**用修复版重跑 10 case，产出 v2 正式证据：
> - **身份闭包**：干净 worktree checkout `6bdb7eb3b`（双侧 git dirty=False、same_root=True），
>   tli_indexer.py SHA `c5dfc215…` / greedy_triton.py SHA `a7c38986…`、torch 2.8.0+cu128 /
>   triton 3.4.0 / CUDA 12.8 / driver 550.127.08 / H20-3e、generator seed=7、逐 case 输入
>   SHA256、逐次原始延迟、输出内容 SHA + `.sha256` sidecar 全落 manifest；
> - **correctness fail-closed**：10/10 case `assign_mismatch=0`（assign 逐位 + k_live + cnt/sq/
>   簇心 sums 全过）；GPU 侧注入验收（篡改 assignment 一位 → 非零退出 + failure.json 落盘 +
>   性能 JSON 不发布）PASS，可复现脚本 `exp/trace/e113_failclosed_inject.py`；
> - **噪声检查**：Triton 臂 rep=3 样本内极差 ≤0.65%（9/10 case；首 case 7.4% 对应待机
>   345 MHz→1980 MHz 升频过渡，<10% 不构成顺序漂移）；运行窗口 GPU 温度 33→38°C、SM clock
>   稳定 1980 MHz、无降频漂移。
>
> **050 历史注记**（背景，保留）：旧 `results/e113_microbench.json`（2026-10-07 14:31 产生）的
> 旧版脚本把 Python reference 固定为仓库外绝对路径动态导入且 manifest 无身份——运行时无法
> 证明双侧实现同源，**降级为「实现身份未闭合的历史观测」存档，保留不动**；其数值本身无算错
> 证据（10 case `assign_mismatch` 均为 0）。

**v2 实测表**（10 case 全表见 v2 JSON；下表为 legacy 表同五格 + 新旧对照）：

| 数据臂 | T | K_live | Python μs/token | Triton μs/token | 加速比（v2） | 加速比（legacy 旧值） |
|---|---|---|---|---|---|---|
| 结构簇 C/N=0.125 | 8192 | 1024 | 239.7 | **3.46** | 69.3× | 49.8× |
| 结构簇 C/N=0.125 | 16384 | 2048 | 238.7 | **7.47** | 32.0× | 15.5× |
| 结构簇 C/N=0.125 | 32768 | 4094 | 237.4 | **14.57** | 16.3× | 7.6× |
| 单例极端 C/N=1.0 | 8192 | 8192 | 238.7 | **16.56** | 14.4× | 11.4× |
| 单例极端 C/N=1.0 | 32768 | 32768 | 237.1 | **64.85** | 3.7× | 3.4× |

（v2 全量 10 case 加速比 3.7-211.2×；T=1024/4096 结构簇臂高达 120-211×——小 T 时 Python 循环
launch 开销占比最大，Triton 单 kernel 优势最极端。）

**新旧差异如实并报**：Triton 侧 v2 普遍快 2-4.5×（如 T=16384 结构簇 16.9→7.5 μs/token），
Python 侧基本一致（240-283 → 237-240 μs/token），差异集中在 Triton 侧。可能原因：① 旧值产生于
2026-10-07 14:31，正值四机 20 卡 E109 全量扫描满载期，旧 manifest 不记录并发/占用状态（这正是
050 指出的可复现性缺口），若同卡或同机存在在跑负载，Triton 计时段会被抬高；② v2 在确认
GPU0 空闲（1 MiB / 0%）且无任何在跑扫描进程的窗口执行。旧值身份未闭合，**正式引用以 v2 为准**；
若引用 legacy 值须注明其身份边界。

- **每 token 延迟 ∝ K 扫描量**（v2：结构簇臂 ~3.6 ns/簇行·head、单例臂 ~2.0 ns）：K=1K →
  2.0-3.0 μs；K=2K → 7.5-8.9 μs；K=32K → 64.9 μs。
- 生产口径（E108：sim0.9 跨层 C/N≈0.2776 → T=16K 时 K̄≈2.3K / K_max≈4.6K）→ **~8.3 μs/token
  （K̄）/ ~17 μs（K_max 尾段）**（v2 校准，legacy 估 17/32 μs 偏保守）。
- Python 基线两口径如实并报：本机 `_greedy_cluster_pass` v2 实测 236.7-240.2 μs/token（T≥4096
  稳态；首 case 436.6 含进程冷启动膨胀，不采用）；E108 probe 口径 79-84 μs/token（实现略异）。
  保守取 E108 口径，生产 K̄ 下加速 **~10×**；本机口径 **~28×**。
- **microbench ≠ e2e**：上表为贪心段 kernel microbench 层证据，不表述为 e2e 加速（e2e 层
  另测，双口径纪律）。

### 2.4 e2e 量级推算（保守，用 E108 口径换算；v2 校准）

> 注：本节推算以 §2.3 的 **v2 正式速度数据**为输入（身份闭包后口径）。推算值为 microbench
> 层换算的量级估计，非 e2e 实测。

- 贪心段（现状 ~5h/任务，16K token 主导）：8.3 μs/token → 0.136 s/层 × 36 层 ≈ **4.9 s/样本** →
  200 样本 ≈ 16 min/任务（原 2.7h prefill 贪心段 → vs E108 口径 Python 79-84 μs/token **~10×**）；
  32K 任务 ×2（带宽/延迟而非 launch）。
- ccluster_sim 臂（far+near 双池 + 2 任务，现状 ~13h）→ **估 ~0.6-1.3h/臂**（v2 口径；legacy
  推算为 1-2.5h/臂，偏保守）。
- 与设计篇 V1 CUDA 目标（1.2-1.7 s/样本@16K，2-3 μs/token）差距 **~3-4×**（8.3 vs 2-3 μs/token）：
  差距来源 = ①8 program 只占 78 SM 中的 8 个（SM 利用率 10%）；②每 token 跨 tile 归约串行依赖链。
  这正是设计篇 §5.1「为什么不是 Triton」论断的实测注脚——**Triton 版不是终态最优，但它是零语义
  风险、当天可上线的止血方案**，且为 V1 铺平了 Kcap 预分配/live-mask 状态改造（两版共享同一套
  状态布局）。

---

## 3. 方案2：段式推测 + 界验证（NO-GO，仿真否决）

### 3.1 算法（`exp/trace/e113_seg_greedy_sim.py`）

段首冻结簇和向量快照 s⁰_k，段内批量 GEMM `D_j(k) = x_j·s⁰_k` 推测每个 token 的决策，再用区间界
证明哪些推测与精确时序语义必然一致（免重算）：

- 段内漂移 `d_k = s_k − s⁰_k`（join 时增量维护），δ_k = ‖d_k‖；
- Cauchy–Schwarz：`|dot_j(k) − D_j(k)| = |x_j·d_k| ≤ ‖x_j‖·δ_k`；
- 三角不等式：`‖s_k‖ ∈ [n⁰_k−δ_k, n⁰_k+δ_k]` ⇒ cos 上下界
  `UB_k = (D+xn·δ)/((n⁰−δ)·xn+EPS)`、`LB_k = (D−xn·δ)/((n⁰+δ)·xn+EPS)`；
- 判据：`a = argmax_k UB_k`；若 `LB_a > max_{k≠a} UB_k` 且 > 段内新建簇精确 cos → a 是真
  argmax（**精确判定，非近似**），阈值决策只需 a 的单行精确 cos；不可证明 → 候选集
  `C = {k: UB_k ≥ LB_a} ∪ 新建簇` 精确重算（真 argmax 必在 C 内）。
- **决策级等价**：界只用于跳过重算，不可证明 token 走精确路径——整个算法与逐 token 精确贪心
  决策级等价，不是近似变体。bound 加 1e-6 保守余量（放大 UB / 缩小 LB）防 fp 误差误判。

### 3.2 仿真结果（T=8192，真实 trace hotpotqa L18 tail32 mid 区 + 合成结构数据）

| case | sim | 簇数 | assign_mismatch | **可证明跳过率** | 候选集/活簇 | 段重启 |
|---|---|---|---|---|---|---|
| trace hotpotqa L18 | 0.8 | 1056 | **0** | 0.49% | 0.28 | 0 |
| trace hotpotqa L18 | 0.9 | 4785 | **0** | **0.29%** | 0.32 | 18 |
| synthetic 混合簇 | 0.8 | 1024 | **0** | 2.89% | 0.86 | 2 |
| synthetic 混合簇 | 0.9 | 1024 | **0** | 2.89% | 0.86 | 2 |

（`results/e113_seg_greedy_sim.json`；skip/exact 为 head-token 计数口径）

### 3.3 否决理由

1. **可证明跳过率 0.29-2.89%**：99.7%+ 的 token 落入「不可证明」路径，须候选集精确重算；
2. **候选集 = 28-86% 活簇**：界太松。机理：贪心态大量**单例簇**（cnt=1，sums=归一化后的 x），
   一次 join 就让 δ_k 从 0 跳到 O(1)（簇心转向半程），n⁰/δ 比崩坏 → 界宽度覆盖几乎所有
   候选。Cauchy-Schwarz 界在「低计数簇 + 高维漂移」的贪心态上是结构性失效，不是调参问题；
3. PyTorch 仿真版本身比 Python 循环参考慢 ~100×（段内逐 token 界计算的开销），即便 kernel 化
   批量 GEMM 段（~11 GFLOP@T=8192 trace，TC 上 <1ms）很便宜，「不可证明重算段」没有可跳过的
   余量——**方案2 的加速来源在贪心态上不存在**；
4. 唯一价值：证明了「推测+界验证」框架在贪心聚类上的**正确性形式**（决策级等价的构造性证明，
   mismatch=0 全 case）——若未来簇态变为高计数（如 kmeans 后处理态），该框架可复用。

---

## 4. 任务B：全 method 矩阵 kernel 方案表

5 个 method 组合 × 3 阶段。列「现状」为 two-level-attention 主树实现口径；★ = E113 新增/改造。
后端选择总纲（与设计篇 §4 一致）：**规则稠密计算 → cuBLAS/Triton；时序控制流状态机（贪心链）→
先 Triton（本原型，已实证）→ 终态 CUDA persistent（设计篇 V1）**。

| method 组合 | prefill 构建 kernel | decode 增量 kernel | L1/L2 选择 kernel | 预估瓶颈 | CPU/GPU 协同点 |
|---|---|---|---|---|---|
| **mavg** (minmax, avg) | 块 k_min/k_max 统计（torch aminmax，可 Triton fuse）+ far avg 池 running-mean（分段 GEMM） | 块滑动统计 O(1) 更新 + far 池 EMA | L1: q·k_minmax GEMM（cuBLAS TC）；L2: q·k_avg GEMV | L1 GEMM 段批量化（#128 Ov-1/Ov-2） | 块统计采集期可 CPU；L1 GEMM 与 L2 4bit 段跨流（#128 双池独立） |
| **aavg** (avg, avg) | 双侧 avg 池 running-mean，prefill = 分段 GEMM 直接算块均 | 池均值增量（加入/淘汰两端口 O(dd)） | L1: q·k_avg GEMM；L2 同 | 最规则——全 cuBLAS，几乎无自研 kernel | 同上；最快迁移到 TC |
| **mminmax** (minmax, minmax) | 同 mavg 的块统计 | 同 mavg | L1 同 mavg；L2: q·kminmax GEMV（4bit 界） | L2 段 GEMV 带宽 | 同 mavg |
| **cavg** (cluster, avg) | far = kmeans（`gpu_kmeans` GEMM 距离 ×20 iter Lloyd，**已 GPU**）+ near avg 池 | kmeans 每 64 步重建（批头化 GEMM）+ near 池增量 | L1: q·簇心 GEMM（簇分 group-mean 聚合后） | kmeans 重建频率 × GEMM 量 | kmeans 采集期 CPU 可并行（离线 dump 重放） |
| **ccluster** (cluster, cluster) | far kmeans + near mean-cluster（E110） | far 增量 + near 每 64 步重建 O(Tn) | 同 cavg，L2 也走簇心 | **near 重建段**（Tn≈α·mid） | near 重建可侧流 overlap（E107f） |
| **cavg_sim / ccluster_sim** ★ | **sim_greedy 贪心链 = 本 E113 Triton 原型**（`greedy_pass_triton`，逐位等价已实证） | 同一 kernel WARM 模式（far 续跑 T_seg=64 / near 重建 T_seg=Tn）——E110 增量≡重放语义原样 | 簇分数 scatter-max（Triton 小 kernel 或并入贪心 epilogue） | ~~Python 循环 30:1 launch-bound~~ → **已消除**；剩 SM 占用率（8/78） | 贪心段 GPU 化后 CPU 侧仅剩数据搬运，双流 overlap 直通 #128 |

要点：
- **5 个组合里唯一 launch-bound 的段就是 sim_greedy**（其余都是规则 GEMM/reduction，cuBLAS/torch
  已高效或 #128 已覆盖）——本原型落掉这块，全 method 矩阵的 GPU 化路径就齐了；
- decode 阶段贪心段开销（kernel 化后，沿用设计篇 §5.5 口径，v2 校准 8.3 μs/token）：far 续跑
  每 64 步 64×8.3μs×36 层 ≈ 19 ms → 0.3 ms/步摊销；near 重建（ccluster_sim，Tn≈1.9K）
  1.9K×8.3μs×36 层 ≈ 0.57 s → **~9 ms/步摊销，重——须 E107f 侧流 overlap 或 V1（2-3μs/token
  后降到 ~1.5-3ms/步）**。

---

## 5. roofline 账本（H20-3e：78 SM / L2 63MB / HBM ~4TB/s / FFMA fp32 ~44 TFLOPS）

生产口径：T=16K、dd=32、sim=0.9、K̄≈2.3K / K_max≈4.6K（E108 C/N=0.2776）。

| 项 | 量级 | 判读 |
|---|---|---|
| 点积 FLOPs（2·dd·ΣK_t） | 0.70 TFLOP/样本（36 层×8 头） | @FFMA 44T = **16 ms 理想下界** |
| Triton 实测（K̄ 臂，v2） | 0.136 s/层 → 4.9 s/样本 | 距下界 ~300×（16 ms 为**每样本 36 层合计**理想下界；legacy 表「~17×」系层/样本口径混用，v2 校正）：**延迟主导非算力**（8 program/78 SM + 跨 tile 归约串行链） |
| sums 读（L2 流量） | 4.8 GB/head @16K → 38 GB/层 | @L2 ~8TB/s ≈ 5 ms/层——**非瓶颈** |
| sums 驻留 | K_max×dd×4 = 0.59MB/head → 4.7MB/层 | **L2 完全驻留**（shared 228KB 放不下全 K） |
| 每 token 延迟拆解 | K=2K：4 tile×(load 64KB + 归约) ≈ 8.3μs（v2 结构簇臂实测 7.5μs@K=2K） | tile 间串行依赖链 = 主延迟源 |
| vs Python 现状 | 79-84 μs/token（E108）/ 237-240（本机 v2 实测） | **~10×（保守）/ ~28×（本机口径）** |
| CUDA V1 目标（设计篇） | 2-3 μs/token → 1.2-1.7 s/样本 | Triton 与之差 ~3-4×（v2 校准）= V1 立项依据 |

- **Triton 版 e2e 形态**：hotpotqa 200 样本贪心段 2.7h → **~16 min**（v2 口径推算；legacy 推算
  30-35 min 偏保守）；ccluster_sim 13h/臂 → **估 ~0.6-1.3h/臂**。E109 v2 后续补臂（0.85/0.95 sim、
  13 任务全量）从「过夜不可行」变「当天可跑」。（推算值为 §2.4 换算，非 e2e 实测。）
- V1 后（设计篇口径）：贪心段 ≤1.7 s/样本 → 单臂 ≤1h，13 任务全量过夜可跑。

---

## 6. 与 #128 overlap 框架的衔接

1. **贪心段独占一个 CUDA stream 即可**：`greedy_pass_triton` 是单 kernel 自包含状态机（无中间
   同步点），与 far/near 双池独立 buffer 的 #128 双流设计正交——贪心链跑在侧流，主流跑
   attention/前向，E107f 侧流异步口径直接挂；
2. **状态布局对齐 V1**：本原型的 sums/cnt/sq/k_live 布局（[H,K,dd] fp32 连续 + Kcap 预分配）与
   设计篇 V1 的 persistent kernel 状态完全同构——**Triton → CUDA V1 迁移只换 kernel 本体，状态
   与 wrapper 语义不变**，单测（T1-T5）可直接复用作 V1 的回归门（门 3）；
3. **E111 layer gate 正交**：skip 层零构建，贪心 kernel 不被调用即可，无交互；
4. **E107 F1 chunk 化**：`greedy_build_triton` 的 chunk 参数 = WARM 模式入口（chunk 边界即续跑
   点），与 F1「prefill 每 chunk 增量」共用同一语义论证（增量≡重放，T3 已证）。

---

## 7. 交付物清单与后续动作

| 交付物 | 路径（本 worktree） | 状态 |
|---|---|---|
| Triton 原型 kernel | `exp/trace/e113_greedy_triton.py` | 5/5 单测逐位全过；BT=512/nw=4 调优默认 |
| 对拍单测 | `exp/trace/test_e113_greedy_triton.py` | T1-T5 全 PASS（含增量≡重放铁律） |
| SEG-GREEDY 仿真 | `exp/trace/e113_seg_greedy_sim.py` + `results/e113_seg_greedy_sim.json` | 决策级等价实证 + NO-GO 判决落袋 |
| microbench | `exp/trace/e113_microbench.py`（050 修复版）+ `results/e113_microbench_v2.json`（**v2 正式证据**）+ `results/e113_microbench.json`（**legacy：实现身份未闭合的历史观测存档，保留不动**） | v2 已于 2026-10-10 空闲 GPU0 重跑：10/10 case 0 mismatch、身份闭包（checkout 6bdb7eb3b dirty=False、双侧同 checkout、impl SHA/seed/逐次延迟/内容 SHA+sidecar）、噪声检查过（温度 33→38°C 无降频、样本内极差 ≤0.65%）；GPU 侧 fail-closed 注入验收 PASS（`e113_failclosed_inject.py`）；红绿单测 `test_e113_microbench_identity.py` 4/4 |
| 本设计文档 | `research/docs/e113_method_kernel_design.md` | 本文 |

**后续动作**（对齐设计篇 §9 计划，本文档补充实证校准）：

1. **P0.5（新增，本实证的直接推论）**：把 `greedy_pass_triton` 经 rsync 增量推送远程，替换
   E109 v2 后续 sim 臂的 `_greedy_cluster_pass` 调用（接口同签名，含 Kcap 预分配改造）——门 1
   dump 重放对拍 + 门 3 单测（本 worktree T1-T5 直接复用）过了即可上线。预期 13h/臂 →
   ~0.6-1.3h/臂（v2 口径）；
2. V1 CUDA persistent kernel（设计篇 P2）判据更新：Triton v2 实测 ~8.3 μs/token（生产 K̄ 口径）
   把贪心段压到 e2e ~10% 以内——**V1 优先级可降为 P2 后置**（先 C1/C2 采完 E109/E110 数据再
   投入），除非 32K 长任务臂（narrativeqa/gov_report，Triton v2 下 14.6-64.9 μs/token）成为
   臂内瓶颈；
3. 方案2 档案化：正确性框架（决策级等价构造）留档，高计数簇态（kmeans 后处理）场景可复用；
   贪心态 NO-GO 结论入 negative result 资产。

---

## 8. 数据源与复现

- Triton kernel 与单测：`exp/trace/e113_greedy_triton.py` / `test_e113_greedy_triton.py`
  （`CUDA_VISIBLE_DEVICES=<idle> python3 test_e113_greedy_triton.py`，<1 min）
- SEG-GREEDY：`CUDA_VISIBLE_DEVICES=<idle> python3 e113_seg_greedy_sim.py --T 8192`（~15 min，
  真实 trace /tmp/trace/qwen3-8b）
- microbench（050 修复版）：`CUDA_VISIBLE_DEVICES=<idle> python3 e113_microbench.py`（~2 min，
  Python 臂占大头；`--wait` 可轮询等空闲）。默认双侧实现都从**当前 checkout** 加载；跨版本
  A/B 须显式 `--ref-root`/`--tri-root`，两侧 git SHA/dirty 与实现文件 SHA256 全落 manifest；
  correctness 门 fail-closed（任一超容差非零退出、不发布性能 JSON）；发布为临时文件 +
  fsync + 原子 replace，manifest 含 torch/triton/CUDA/driver 版本、逐次原始延迟与输出内容
  SHA（+ `.sha256` sidecar）。**v2 正式证据**（`results/e113_microbench_v2.json` +
  sidecar，2026-10-10）产生方式：`git worktree add --detach /tmp/e113_clean_wt <HEAD>` 干净
  checkout（保证 manifest dirty=False）后从 worktree 运行、`--out` 指回主树 results 目录——
  延迟测量须在空闲 GPU 无并发扫描窗口执行（硬纪律）。fail-closed 注入验收复现：
  `CUDA_VISIBLE_DEVICES=<idle> python3 e113_failclosed_inject.py`（退出 0 = 门生效）。
  已落袋的 `results/e113_microbench.json` 是修复前旧协议产物（**实现身份未闭合的历史观测**，
  保留不动）；CPU-only 红绿单测 `test_e113_microbench_identity.py`（4/4）。
- 参考实现：`two-level-attention/sparse_attn/indexer/tli_indexer.py::_greedy_cluster_pass_python`
  （050：microbench 默认从当前 checkout 同源加载，不再引用仓库外绝对路径主树；
  `sys.dont_write_bytecode` 防落盘）
- E108 基线：e108_sim_greedy_probe.json（84/79 μs/token、C/N=0.2776、fp 边界注记）
- 设计篇对照：`/home/wangyuanshuo02/sglang/research/docs/sim_greedy_kernel_design.md`（#150）
