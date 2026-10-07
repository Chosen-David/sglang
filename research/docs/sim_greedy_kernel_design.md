# E113：sim_greedy / cluster 方法 × kernel 设计文档
## （含「数据采集期 CPU overlap 策略」与「最终 GPU kernel 实现」的严格两分）

> 交付背景：2026-10-07 用户澄清指令（最高优先级，覆盖此前任务书中 CPU 协同的表述）：
> 1. **CPU 只是数据采集期的 overlap 手段**（临时抢时间），不代表论文最终实现用 CPU——本文档严格区分 §7「数据采集期 CPU overlap 策略（临时）」与 §5「最终 kernel 实现（GPU）」两节；
> 2. **最终实现 prefill 与 decode 两阶段都上 GPU kernel**，不是只做 decode 增量；
> 3. **不限定 Triton**：CUDA C++ / Triton / cuBLAS 哪个快用哪个，逐方案给出选择理由（§4 表 + §5 论证）；
> 4. 附加探索点：**数据采集期本身的 CPU/GPU 协同加速**——远程 sim 臂慢的并行化评估（§7）。
> 其余要求不变：method×kernel 方案总表（§4）、语义等价铁律（§3）、roofline 账本（§6）、microbench 备用脚本（§8）。
>
> 关联任务：#150（E113）/ 上游：#128（overlap kernel 设计，`overlap_kernel_design.md`，本文是其 sim_greedy 特化篇）/ #132（E108 mass probe）/ #147（E110 ccluster e2e 实现）/ #146（E109 v2 远程 sim 臂）。

---

## 0. 结论速览（先给判决）

| 问题 | 判决 |
|---|---|
| sim_greedy 慢的根因 | **不是算力不足，是 Python 逐步循环的 launch/解释器开销**：每 token ~30 个 CUDA op 合计 237-269μs（E108 probe 253μs 与 E113 bench 双实现互证），真实 GPU 计算每 token 仅 ~2-3μs（开销比 **~100:1**） |
| 最终实现（GPU） | **V1 = CUDA C++ persistent kernel**（贪心链整链单 kernel，8 head 链并发）：prefill 全量/chunk 增量 + decode 续跑/near 重建共用一个 kernel 的四种模式；预算 ~122s/样本（hotpotqa 均值）→ **1.2-1.7s/样本（~70-100×）** |
| 为什么不是 Triton | 贪心链是「token 串行 + 全 K 精确 argmax + 动态控制流（归并/新建/扩容）」的状态机——Triton 无 grid 级同步、单 program 单 block 串行循环无法占满 SM、动态长度 while 编译保守；**规则 GEMM/reduction 部分才用 Triton/cuBLAS** |
| 数据采集期 CPU 加速（实测判决） | **C2 torch 进程池 NO-GO**：实测 CPU 单线程 torch 循环 825-1483μs/token，比 GPU Python 循环还慢 3-6 倍（torch 逐 op 调度开销在 CPU 上同样致命），8 worker 并行也追不平；**C1b（CUDA Graph 重放+图内 unroll×8）实测 2.3-2.5×、assign 逐位一致**，13h→~5.5h，是唯一零语义风险的即用手段；根治 = V1 kernel |
| 语义等价 | 铁律 = **决策等价**（assign 序列一致）；C1b 实测 assign diff=0、k_live 一致（含 unroll pad 通胀扣除）；归约平局必须复刻 torch.argmax 最小索引 tie-break；决策路径禁 TF32/fp16 |

---

## 1. 现状实况与瓶颈测量（全部实测，标注出处）

### 1.1 代码现状

- 贪心实现：`two-level-attention/sparse_attn/indexer/tli_indexer.py::TliIndexer._greedy_cluster_pass`（L448-500）——`for i in range(T)` 逐步循环，每步 ~10 个 CUDA op（bmm / masked_fill / argmax / gather / 2×index_add_ / where）。
- 状态：`sums[H,K,dd] / cnt[H,K] / sq[H,K] / k_live[H]`，K 容量按需扩容（上界 T）。
- 语义：token i 与「i 时刻的簇心（running mean）」余弦相似度最大的活簇比较，cos ≥ sim 归并否则新建簇；**decode 期 far_hi 块对齐前移时对新增段续跑 ≡ 全量重放（逐位一致，已证）**。
- 两个消费端：far 区（`_update_far_greedy`，增量续跑）与 near 区（`_update_near_cluster`，ccluster 时按 (far_hi,near_hi) 块界全量重建，每 64 decode 步一次）。

### 1.2 生产口径（本文所有账本用这套参数）

| 参数 | 值 | 出处 |
|---|---|---|
| 模型 | Qwen3-8B：36 层 × 8 kv-head（Hkv=8） | E108 probe / tli_indexer |
| 聚类维度 dd | **32**（`--tli_sim_dims` 默认 subspace → tail32 = 48:64+112:128，cmp_ratio=4） | `_sim_dims_indices`：「'full' 落到 tail 分支」 |
| sim | 0.9（E109 v2 远程臂注册表口径） | `/tmp/e109_scan_remote.sh` |
| 簇数规模 | C_over_N ≈ 0.2776（sim0.9，跨层池化）→ T=16K 时 K_max≈4.6K / K̄≈2.3K | e108_sim_greedy_probe.json |
| 远程臂 | hotpotqa+musique n=200，(0,0) 单池 + 冠军位 γ 扫 + 轴采样，双卡 5 臂×2 | 同上脚本 |

### 1.3 瓶颈定位：launch-bound，不是算力-bound

三条独立证据：

1. **E108 GPU probe**（本机 H20）：`build_s` 是**每层单链**计时（probe L239-241，每层每 sim 调一次 `greedy_cluster_assign`）——gov_report S=32768：8.28s/单链 = **253μs/token**；hotpotqa S=16957：4.02s = **237μs/token**。
2. **E113 microbench 互证**（本 bench 独立重写的参考循环）：T=4K/8K/16K 均为 **255-269μs/token**——与 probe 双实现一致（`research/bench/e113_sim_greedy_bench.json`）。而每 token 真实 GPU 计算（K̄×dd 点积 + argmax ≈ 2300×32 MAC）≈ **2-3μs**。开销比 **~100:1**。
3. **远程实况**（2026-10-07 13:34 rssh 实测）：21 核 load 2.7（18 核闲置），两个 sim 臂进程各 **94% 单核**（Python 解释器瓶颈特征），GPU util 54-55%（被 attention 段拉高，贪心段 GPU 近乎空转）。

> 口径勘误记录：早期版本曾把 build_s 误当 3 层合计（得出 84μs/token）——已按 probe 源码勘正为每层单链。修正后远程 wall 推算：hotpotqa 均值 T≈13K → 36 层 × ~3.4s ≈ 122s/样本 × 200 ≈ **6-8h/任务**（中位更短的样本拉低均值，与记忆「贪心 ~5h/任务」同量级）；ccluster 臂 far+near 双池 + 两任务 ≈ **~13h/臂** ✓。

> 口径注记：远程 e2e 贪心实际在 **GPU tensor 上跑 Python 循环**（k 在 GPU，sparse_attn/benchmark 目录 grep 无 set_num_threads）；`torch.set_num_threads(1)` 单线程限制是 E108 probe / E110 单测的 **CPU 计算路径**约定。两条路径共享同一根因：**token 串行 + 每步 launch/解释器开销 + head/layer 并行维度未被利用**。§7 的 CPU 化方案里 `set_num_threads(1)` 语义变为「每 worker 单线程，并行度来自 worker 数」。

---

## 2. 贪心算法的并行性解剖（kernel 设计的出发点）

逐步语义（不可动）：

```
for i in 0..T-1:                       # token 维：串行（数据依赖）
    cos = (sums[live] · x_i) / (‖sums[live]‖·‖x_i‖)   # [K_i] —— 可全并行
    a = argmax(cos)；upd = cos[a] ≥ sim                # 全 K 精确 argmax —— 语义硬约束
    sums[a] += x_i（或新建簇）                          # 每步只改 1 行 [dd] —— 稀疏更新
    assign[:, i] = a
```

| 维度 | 可并行？ | 说明 |
|---|---|---|
| token | **否** | 语义即串行决策过程（greedy 的定义）；任何打破时序的方案 = 换语义，违反铁律 |
| K（簇） | **是** | 每步 cos/argmax 是 K 上的归约——K̄≈2.3K、K_max≈4.6K，归约并行度充足 |
| head（8） | **是** | 8 条独立链（k_live/sums 各自独立）；GPU bmm 已隐式向量化，CPU 可进程并行（§7-C2） |
| layer（36） | **e2e 否 / 离线是** | **关键路径事实**：k_{l+1} 依赖 layer l 的 attention 输出，attention 又依赖 layer l 的 mask（贪心结果）→ 在线管线 layer 间严格串行；离线 dump-replay 可全并行（§7-C3） |
| sample | e2e 否 / 跨臂、离线是 | benchmark batch=1 |

**关键结构洞察（V2 变体的基础）**：每步只有 1 行 sums 变化 → 「step i 对未变行的 cos」可以批量预计算成 GEMM（块内用 touched-cluster 修正补齐），把占大头的点积 FLOPs 从 GEMV 串行流改造成 TC/FFMA GEMM——但 **argmax 的 K 级归约仍是每 token 串行下界**。

---

## 3. 语义等价铁律（kernel 化的硬门）

参考实现 = HEAD `_greedy_cluster_pass`（fp32、单成员 `+= x` 更新序、增量 sq 维护）。

1. **判决层级**：
   - **决策等价（硬门）**：assign 序列逐 token 一致。生产口径 sim=0.9 必须 100%；允许的偏差上限 = E108 已测 fp 边界基线（sim≥0.85 逐位全同；sim=0.80 有 0.03% = 35/126440 边界差——那是参考实现自身换 norm 维护方式的基线，不是给 kernel 的豁免额度）。
   - 数值等价（软门）：sums/centroid 相对差 < 1e-5。
2. **fp 纪律**：决策路径（cos/argmax/阈值比较）**禁 TF32 / fp16 / bf16**，必须 FFMA fp32 逐序累加；CUDA 编译选项 `-ftz=false` 与否需对拍确认。V2 的块级 GEMM 改变累加次序 → 必须过 §3.3 验证，失败则回退 V1（V1 的 per-token `+= x` 与参考实现天然同序）。
3. **argmax tie-break**：`torch.argmax` 平局取**最小索引**；CUDA 树状归约必须显式实现 `(val, idx)` 字典序比较，否则决策等价在平局处静默破裂（余弦值大量重复时——如零范数、重复 token——必然触发）。
4. **归并/新建分支**：`m >= sim` 的等值边界、`dot_a` 从 dot 直接 gather（规避 -inf×0=NaN 污染 sq 的 E108 实测坑）必须原样保留。
5. **验证流程（三道门）**：
   - 门 1：dump k 离线重放对拍——36 层 × 8 头 × 多任务多 T，assign 逐位 diff 率 = 0（sim0.9）；
   - 门 2：e2e pred 两臂（cavg_sim / ccluster_sim）输出与参考臂 `torch.allclose` + 打分一致；
   - 门 3：单测回归——`test_e110_ccluster.py` T7 逐位保护套件扩展 kernel 路径，进 CI 约定（commit 前必跑）。

---

## 4. method × kernel 方案总表（含 CUDA/Triton/cuBLAS 选择理由）

范围：5 个 method 组合（mavg/cavg/aavg/ccluster/mminmax）在 L1 粗筛构建 + 打分、L2 细筛的 kernel 映射。标 ★ 的为 E113 新增/改造项；其余为现状保留或 #128 已覆盖。

| # | 计算段 | method 归属 | 现状 | 方案 | 后端选择与理由 | 阶段 |
|---|---|---|---|---|---|---|
| 1 | 块统计 k_min/k_max | minmax（mavg/mminmax/cavg 的 far 门） | torch amin/amax（已高效） | 保留；如做 fused 可并 avg | **Triton**（简单 tile reduction，无控制流，重写收益小） | prefill |
| 2 | 块统计 k_avg | avg（aavg/mavg/cavg） | torch mean | 同上 | **Triton** | prefill |
| 3 | L1 打分 q·k_minmax / q·k_avg（GEMM） | 全部 | torch bmm/matmul | cuBLAS 路径保持；批量化见 #128 Ov-1/Ov-2 | **cuBLAS/cublasLt**（规则大 GEMM，TC 效率天花板，自写无益） | prefill+decode |
| 4 | kmeans 建簇（cluster 臂） | cavg/ccluster(kmeans) | per-head Python 循环调 `_gpu_kmeans`（niter×[assign GEMM + scatter mean]） | ★ 批头化 + fused 迭代 | **Triton**（assign 步 = 规则 GEMM，TC 友好；scatter-mean 用 atomic_add tile；无动态控制流）+ cuBLAS 做 assign GEMM。**选 Triton 不选 CUDA**：kmeans 每轮是稠密规则计算，Triton tile 抽象正好，CUDA 手写收效低 | prefill；decode 每 64 步重建 |
| 5 | **sim_greedy 贪心链** | cavg_sim / ccluster_sim | Python 逐步循环（launch-bound 30:1） | ★ **V1 persistent kernel**（§5.3），V2 mega-step 变体（§5.4） | **CUDA C++**（§5.1 论证：串行状态机 + 全 K 精确 argmax + 动态分支 + 跨步状态驻留，Triton 无 grid sync、动态 while 保守、单 program 占不满 SM） | **prefill + decode 两阶段** |
| 6 | k_qat 4bit 量化 + L2 细筛 GEMV | 全部（4bit 侧） | 已 fused（`tli_sparse_gather_attn_dot`） | 不动（E107 已优化） | 现有 fused kernel | decode |
| 7 | top-k 选择 | 全部 | torch topk | 保持；如成瓶颈换 cub segmented radix sort | **CUB**（设备级 radix select 成熟） | decode |
| 8 | 簇分数 scatter-max（assign→块分） | cluster/sim 臂 | torch scatter_reduce | ★ 并入 #5/§4-4 的 epilogue | **Triton**（独立小 kernel）或并入 V1 epilogue | prefill+decode |

**一句话总纲：规则稠密计算（GEMM/reduction）→ cuBLAS/Triton；时序控制流状态机（贪心链）→ CUDA C++。**

---

## 5. 最终实现：GPU kernel 设计（prefill + decode 两阶段全 GPU）

### 5.1 后端论证：为什么贪心链选 CUDA C++ 而非 Triton

1. **控制流**：每 token 一次「argmax → 阈值比较 → 归并/新建」分支，且新建簇改变 k_live 形状语义（容量内滑动写指针）。Triton 的 masked 张量抽象表达「动态活簇集合上的 argmax」极其别扭（要把 K 维 pad 到 Kcap 并全程带 live mask，反复全量 masked 归约）。
2. **同步模型**：链内每步需要 block 级两轮归约（sum-of-products → argmax）。Triton program = 单 block，可以表达，但 **8 条 head 链 = 8 个 program = 8 个 block 占 78 个 SM 中的 8 个**，且无法在一个 program 内做 warp 级 handoff 优化；CUDA 可以把 8 链放进 1 个 block 的 8 warp 组做 warp-specialization，或 1 链 1 block + 跨块 grid sync（cooperative groups）自由选择。
3. **串行长循环**：T=16K 步的循环，Triton 每步的边界检查/掩码生成的指令开销显著；CUDA 手写归约内循环可把每 token 压到 ~2-3μs 的延迟下界。
4. **反例（诚实记录）**：若未来把贪心改成 V2 纯 GEMM 批式（§5.4）且块内串行部分足够薄，Triton 表达块级 GEMM 会更省事——**V2 若立项再评估 Triton**，V1 先 CUDA。

### 5.2 两阶段统一：一个 kernel，两种模式

```
__global__ void tli_greedy_chain_kernel(
    const float* __restrict__ x,      // [T_seg, H, dd] 段内 token（fp32，已取 dim 子空间）
    float* sums, int* cnt, float* sq, // [H, Kcap, dd] / [H,Kcap] / [H,Kcap]  跨调用驻留（冷启动置零）
    int* k_live,                      // [H]
    int* assign_out,                  // [H, T_seg]（追加偏移 assign_base）
    int T_seg, int dd, int Kcap, float sim, int mode /* COLD | WARM */);
```

- **prefill-COLD**：跨请求 clear 后首 chunk，状态置零全量建簇。
- **prefill-WARM（chunk 增量，F1 口径）**：每 chunk 追加段续跑，状态从上 chunk 延续——与现有「增量续跑 ≡ 全量重放」的已证语义逐位一致。
- **decode-WARM（far 续跑）**：far_hi 每推进一块（64 token）触发一次，T_seg=64，代价 = 段长而非全 far 重放（现有语义原样）。
- **decode-REBUILD（near 侧，ccluster_sim）**：near_hi/far_hi 块界变（每 64 步）→ 全量重建 near 链（T_seg = Tn ≈ α·mid）；**注意 near 重建不能改成「左缘删除增量」**——被删 token 曾影响后续 assign（贪心时序耦合），删除后从零重放 ≠ 状态删行，语义不等价，违反铁律。正确做法是把重建跑快（§6）+ 侧流 overlap。
- 集成点：E107f 侧流异步（索引构建不占 decode 关键路径）/ E111 layer gate（skip 层零构建，正交）/ E107 F1 chunk 化（WARM 模式入口）。

### 5.3 V1：persistent chain kernel（主线方案）

- **布局**：1 head 链 = 1 block（256-512 线程）；8 链 8 block 并发（若单链延迟仍高于 BW 下界，V1b 升级为 1 block 8 warp 组分持 8 链做 ILP 交织，把归约延迟摊给 8 链）。
- **每 token 步**：
  1. `xi` 广播到 shared（dd=32 → 128B）；
  2. 点积 `dot[k] = sums[k]·xi`：K 维分片到线程，warp shuffle 归约 → **这是 L2 延迟主导段**（sums 行在 L2，~200-300 cycle）；
  3. `argmax(cos)`：`(val,idx)` 字典序树归约（铁律 3 的 tie-break）；
  4. 归并/新建：单行 `sums[a] += xi`、`cnt[a] += 1`、`sq[a] += 2·dot_a + ‖xi‖²`（与参考同序）；
  5. `assign[h][i] = ...`、`k_live[h] += !upd`。
- **容量**：Kcap 按 T_max 预分配（far 区 ≤ 32K → Kcap=32K × dd32 × 4B = 4MB/head，8 头 32MB——显存无压力），规避 Python 版的逐步 F.pad 扩容。
- **NaN 坑原样规避**：`dot_a` 从 dot 直接 gather（§3.4）。

### 5.4 V2：mega-step GEMM 变体（可选二阶段，先验证再立项）

- 每块 B=256 token：① 大 GEMM `D = X_blk × Sums^T`（[B,K]，FFMA fp32）一次性算基线 cos；② 块内串行只处理「touched 簇修正」：维护块内簇和 c_j（每 token O(≤B×dd)），`cos_i,j = D_i,j + x_i·c_j` 对 touched 子集修正后 argmax；③ 块间 sums 定期物化。
- 收益上限的诚实评估：点积 FLOPs 转入高效 GEMM（V1 中它本就不是瓶颈——**V1 瓶颈是每 token 归约延迟**），argmax K 级归约仍是串行下界 → **预期仅再 1.5-2.5×**，且改变 fp 累加次序触碰铁律 2（须门 1 全过）。
- **判决：V1 落地实测后（microbench §8）再决定 V2 是否立项；若 V1 已达 1.2-1.7s/样本（贪心段 < e2e 10%），V2 直接 NO-GO。**

### 5.5 decode 阶段开销账（GPU kernel 化后）

| 段 | 频率 | 单次代价（V1 估算） | 摊销/decode 步 |
|---|---|---|---|
| far 续跑 | 每 64 步 | 64 token × ~2.5μs × 36 层 ≈ 5.8ms | **0.09ms/步**（可忽略） |
| near 重建（ccluster_sim，α=0.125，Tn≈1.9K） | 每 64 步 | 1.9K × 2.5μs × 36 层 ≈ 171ms | **2.7ms/步 —— 重，须侧流 overlap 或回落 near=4bit（cavg_sim 口径）** |
| kmeans near 重建（ccluster_kmeans） | 每 64 步 | 向量化，与 cavg 同量级 | 与现状一致 |

---

## 6. roofline 账本（T=16K 样本、dd=32、sim0.9 → K̄≈2.3K / K_max≈4.6K；H20-3e：78 SM / L2 63MB / HBM ~4TB/s / FP32 FFMA ~44 TFLOPS——TC 148 TFLOPS 但决策路径禁用）

| 项 | 每 head 链 | 每 layer（8 头） | 每 sample（36 层） | 备注 |
|---|---|---|---|---|
| 点积 FLOPs（2·dd·ΣK_t） | 2.4 GFLOP | 19 GFLOP | **0.70 TFLOP** | @FFMA 44T = 16ms 理想下界 |
| sums 读（L2 流量） | 4.8 GB | 38 GB | 1.38 TB | @L2 ~8TB/s ≈ 0.17s —— **非瓶颈** |
| sums 驻留 | K_max×dd×4 = 0.59MB | 4.7MB | — | **L2 完全驻留**（shared 228KB 放不下全 K，放弃 shared 驻留方案） |
| 归约延迟（串行下界） | 16K 步 × 2-3μs = 32-48ms | 32-48ms（8 头并发） | **1.2-1.7s** | **V1 主瓶颈：延迟而非带宽** |
| vs 现状 Python 循环 | ~3.4s/层（13K 均值） | — | ~122s | **kernel 化 ~70-100×** |
| 现状 launch 开销占比 | 237-269μs/token vs 2-3μs 计算 | — | — | ~100:1（E108 probe 与 E113 bench 双实现互证），kernel 化收益来源 |

- 32K 任务（gov_report/narrativeqa）FLOPs/延迟 ×4 → ~5-7s/样本，200 样本 ≈ 20min。
- **V1 后 e2e 形态**：hotpotqa 200 样本贪心段 ~6-8h → **5-7min**；单条完整 sim e2e 臂（含模型前向/attention/打分）估计 **≤1h**，13 任务全量在单卡从「不可行」变「过夜可跑」。
- 以上延迟项（每 token 2-3μs）为设计估算，**以 §8 microbench 实测为准**；若实测高于 5μs/token，触发 V1b（8 链 ILP 交织）或 V2 复议。

---

## 7. 数据采集期 CPU/GPU 协同加速（**临时策略**——只为抢出 E109/E110 sim 臂数据，不进论文实现叙事；最终实现 = §5 GPU kernel）

### 7.1 并行维度再确认（§2 表的采集期特化）

- **在线 e2e（pred 管线）**：贪心在关键路径（layer l mask → layer l attention → k_{l+1}）→ **layer 维不可并行**；可并行的只有 head（8）与「跨臂/跨卡」。CPU 化只能按 head 并行，不能幻想 36 层并行。
- **离线 mass/召回采集（dump 重放）**：k 预先 dump → layer × head × sample 三维全并行（现有 analyze_* 重放管线直接挂 multiprocessing）。

### 7.2 方案判决（C1b/C1/C2/C3——全部带 E113 bench 实测数据）

| 方案 | 做法 | 实测/判决 | 13h/臂 → |
|---|---|---|---|
| **C1b：CUDA Graph 重放 + 图内 unroll×8（GPU，采用）** | 循环体固定 shape（Kcap 预分配 + live mask）→ `torch.cuda.CUDAGraph` 捕获 **8 个 token 步**（摊薄每图重放开销）→ 重放 T/8 次；token 喂入用 device 索引 `index_select`（graph 内安全），assign 用 device 偏移 `index_copy_` | **实测 2.3-2.5×，assign diff=0、k_live 一致**（含 unroll pad 零 token 的 k_live 通胀扣除）；graph 捕获坑：cublas handle 须捕获前侧流预热、捕获会执行一步污染状态须复位 | **~5.5h** |
| C1（同上但 unroll=1） | 单 token 步捕获重放 | 实测 1.45-2.5×——unroll 前后每 token 成本几乎不变（~105μs），说明瓶颈是**图内 ~30 个小 kernel 的执行开销**而非 replay 派发；继续提速只剩 op 融合一条路（torch.compile 未验证） | ~6h |
| **C2：CPU torch 进程池（NO-GO，实测否决）** | 每层 8 head 切 8 进程（每进程 `set_num_threads(1)`）| **实测 CPU 单线程 torch 循环 825μs/token（T=4K）→ 1483μs/token（T=8K），比 GPU Python 循环（255μs）慢 3-6 倍**——torch 逐 op 调度开销在 CPU 上同样致命，8 worker 并行（÷8 后仍 100-185μs/token·头）追不平 GPU 循环；要翻身需 C/C++ 扩展重写链（~5-10μs/token 可期），但那是 kernel 级工程量，直接被 V1 支配 → **不做** | 否决 |
| **C3：离线 dump-replay（仅 mass/召回侧）** | k dump → 离线 layer×sample 全并行 | mass 侧数据分钟级；**不适用 e2e 精度**（贪心必须在环内） | 仅 mass 口径 |

- **执行判决：数据采集期加速 = C1b 一条路**（bench 脚本即原型，§8）；C2 否决、C3 仅 mass 侧。C1b 后贪心段 ~5.5h/臂仍嫌重，若 0.85/0.95 sim 补跑量大则直接上 V1（V1 本就是终态，P1 阶段顺路完成）。
- **约束（用户指令）**：不动在跑实验——E109 v2 远程 10 慢臂照跑；C1b 作为**后续配置**（0.85/0.95 sim 补跑、13 任务全量、E110 剩余臂）生效，通过 rsync 增量推送（注意双端 origin 不同源，走 rsync 不走 git pull）。

### 7.3 为什么 CPU 不是终态（论文叙事边界，防止口径混淆）

- **实测判决（比预期更干脆）**：CPU torch 循环单线程 825-1483μs/token，比 GPU Python 循环慢 3-6 倍——torch 的逐 op 调度开销在 CPU 上同样致命；8 worker head 并行后每头仍 ~100-185μs/token，追不平 GPU。要竞争需 C/C++ 扩展重写贪心链，工程量与 V1 kernel 同级却被其性能支配（V1 目标 2-3μs/token）→ 直接做 V1；
- 论文对比口径铁律（kernel 级 microbench + e2e 两层）要求**同机同口径 kernel 对比**——CPU 贪心无法进 kernel 对比表；
- 论文中 CPU 只允许出现在「数据采集工程手段」脚注，不得进 §4.x 硬件效率小节（#128 已定的叙事边界）。

---

## 8. microbench 备用脚本（已落地并跑通，实测结果入档）

- 路径：`research/bench/bench_e113_sim_greedy.py`；结果：`research/bench/e113_sim_greedy_bench.json`（T ∈ {4K,8K,16K} × sim0.9 × dd32，合成数据已标注 synthetic；`--dump` 接真实 k 待 kernel 联调时补）。
- 四臂：A reference（HEAD 原样语义基线）/ B cudagraph（C1b：图内 unroll×8 重放）/ C2 cpu（单线程对照）/ D fused（V1 kernel TODO 占位）。

### 8.1 实测结果（2026-10-07，本机 H20-3e，GPU 空闲时段）

| 臂 | 每 token 开销 | vs 基线 | assign diff | k_live |
|---|---|---|---|---|
| A reference（GPU Python 循环） | 255-269μs（T=4K/8K/16K 稳定） | 1× | — | — |
| B C1b CUDA Graph（unroll=8） | ~105-118μs | **2.3-2.5×** | **0.0（逐位一致）** | 一致（pad 通胀扣除后） |
| C2 CPU torch 单线程 | 825μs（T=4K）→ 1483μs（T=8K） | **0.17-0.31×（更慢）** | 0.0（vs GPU 参考） | — |
| V1 fused kernel | 目标 2-3μs | 目标 ~100× | 目标 0 | — |

### 8.2 三条实测教训（写 kernel 前必读）

1. **bench_arm 状态污染**：贪心链原地 `index_add_` 改 state——warmup 与 timed 共享同一 state tensor 时，timed 轮从 warmup 轮残留的 k_live/sums 续跑，产出「污染轨迹」，曾造成 assign 对比假 diff 1.6%。**每次计时调用必须全新 state**（bench 内已改为 args 工厂）。
2. **graph 捕获两坑**：①cublas handle 不能在 capture 内首次创建——捕获前须侧流 warmup 一步；②捕获本身会执行一步并推进 device 索引/污染 static 状态——捕获后必须全部复位再 replay。
3. **xi_n 归约形状**：参考实现 `x.norm(dim=-1)` 是全量 [T,H] 一次算好；图内若逐 token [H,dd] 现算，归约树序不同→fp 边界翻转级联（须预计算后按 pos 索引，与参考同源）。
4. **unroll 收益封顶的归因**：unroll 1→8 每 token 成本几乎不变（~105μs），证明瓶颈是图内 ~30 个小 kernel 各 ~3-7μs 的执行开销而非 replay 派发——torch op 粒度融合（torch.compile）理论可再压，但上限仍远离 V1 的单 kernel；C1b 定位=零语义风险的过渡手段。

- 判读标准（V1 落地后复跑本脚本）：V1 每 token 延迟 ≤3μs → §6 账本成立，V2 NO-GO；>5μs → 触发 V1b/V2 复议。

---

## 9. 实现计划（分阶段，验收口径）

| 阶段 | 内容 | 验收 | 依赖 |
|---|---|---|---|
| P0 ✅ | microbench 基线（参考实现 + C1b 图重放原型 + CPU 对照）跑通 | **已完成**：JSON 落袋，C1b 2.3-2.5×、assign diff=0；C2 实测否决 | 无 |
| P1 | C1b 进 indexer（Kcap 预分配 + WARM/COLD 模式改造），E109 后续臂生效 | 门 1 对拍 0 diff + 门 3 单测 | P0 |
| P2 | V1 CUDA persistent kernel（prefill COLD/WARM + decode WARM/REBUILD 四模式） | 门 1/2/3 全过；贪心段 ≤1.7s/样本@16K | P1 的状态改造 |
| P3 | 集成 E107f 侧流 + E111 gate 正交验证；E110 ccluster_sim 臂全量重测 | e2e 延迟三段式对比（E107 口径）落袋 | P2 + E109 收官 |
| P4 | V2 复议（按 §5.4 判决条件） | microbench 数据说话 | P2 实测 |

## 10. 数据源清单

- 贪心语义与状态机：`two-level-attention/sparse_attn/indexer/tli_indexer.py` L353-516（`_update_far_greedy` / `_update_near_cluster` / `_greedy_cluster_pass` / `_sim_dims_indices`）
- 每 token 开销实测（双实现互证）：e108_sim_greedy_probe.json `build_s`（**每层单链**口径，probe L239-241：253μs/237μs per token）+ `research/bench/e113_sim_greedy_bench.json`（255-269μs per token）
- C1b/C2 实测：同上 bench JSON（C1b 2.3-2.5× & assign diff=0；CPU torch 825-1483μs/token 否决）
- C_over_N 簇数规模：e108_sim_greedy_probe.json（sim0.9 → 0.2776）
- 远程实况：2026-10-07 13:34 rssh 实测（21 核 / 2×94% 单核进程 / GPU 54-55% / 臂配置 /tmp/e109_scan_remote.sh）
- 远程臂 wall：记忆 round3（贪心 ~5h/任务，ccluster 臂 ~13h）
- 硬件：本机 torch 查询（H20-3e 78 SM / L2 62.9MB / 141GB）
- fp 边界基线：e108_sim_greedy_probe.json `assign_fp_note`（sim≥0.85 逐位全同）
- 上游设计：overlap_kernel_design.md（#128，Ov-1/Ov-2/Ov-3 与 F1-F6 衔接）
