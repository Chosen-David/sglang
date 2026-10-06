# TLI/PSI method 组合异构 overlap kernel 设计

> 任务：E108a 设计文档（2026-10-06）。为两级稀疏注意力索引器的 method 组合
> （mavg / cavg / aavg / mminmax）设计 kernel 级 overlap 方案：far/near 两区
> 各自的表示法（cluster / minmax / avg）分置到不同硬件单元（TensorCore /
> CUDA core / CPU / 多 stream）并发执行，把 indexer 开销从 Σt_i 压向
> max(分组耗时)。
>
> **文档性质**：纯设计（不实现、不跑 GPU）。所有实测数字标注 JSON 来源；
> 所有预测数字显式标注「估算」并给出假设。

---

## 0. 结论速览（先给判决）

1. **overlap 自由度真实存在**，其架构来源有两个，均已核实于代码：
   - **双池独立**：B' 分区让 far_sc / near_sc 是两个独立 buffer、独立配额、
     独立 top-K（`indexer.py` select_decode_batched L1249-1263；kernels.py
     KernelC 双池直写）——far topk 与 near topk 之间**零数据依赖**；
   - **打分原语异构**：far=cluster 是 GEMM（TC 友好）、far=minmax 是区间
     算术归约（CUDA core）、near=avg 是 GEMV（两者皆可）。
2. **但实测收益分层明显（诚实判决）**：
   - **主精度臂 mavg（50.78 那条）**：decode 侧 overlap 估算收益
     **~15-30% 选择链**（需要 Ov-3 分块流水才拿得到；纯双池 topk 并行
     只有 ~3%，因 near_compact 已把 near topk 宽度砍 30×）；
   - **速度臂 cavg（far=cluster）**：overlap 结构收益最大（cluster GEMM
     ∥ avg GEMV 天然异构分置），但 e2e 精度落后 mavg 1.50（43.58 vs
     45.08，ClusterKV 口径）——**速度潜力与精度代价在同一臂上**，论文
     必须分开表述；
   - **prefill 侧**才是 overlap 的主战场（M10 慢路径 1042ms/调用@末
     chunk），但**前置依赖 F1 增量化**（#126），两者正交可叠加。
3. **H20-3e 的 TC 稀缺（~15% H100 吞吐）使「TC 只喂真 GEMM 段」成为
   设计铁律**：L1 打分 TC 化已实测 2.58× **变慢**（内存受限段，见
   §6.1）——TC/CUDA-core 分置在 H20 上不是锦上添花，而是资源约束下的
   必需；但它只在 compute-bound 段兑现（kmeans build、大 K_c cluster
   打分），内存受限段的并发收益来自延迟隐藏而非带宽。
4. **CPU 侧 overlap 边界已实测**：kmeans 质心 / KV 数据搬 CPU 是 NO-GO
   （PCIe 55GB/s vs GPU gather 213GB/s，往返 57× 惩罚，
   cpu_gather_overlap_bench.json）；可行的是 gate 统计（标量级）与
   copy engine 并发（实测 −25%）。

---

## 1. 动机与现状证据

### 1.1 为什么转向 overlap

论文速度主张经 round4/round5 收窄为三层：kernel 选择链 / DSA 1/32 计算
量 / 336B vs 1KB 存储；e2e 层诚实判决为「Quest e2e 全形状占优、PSI 唯一
兑现点 = decode 增量索引开销 32 vs 59.5 ms/step（1.85×）」。要扩大速度
卖点，需要一个**架构级**的新杠杆：单级索引（Quest/DSA）的整链是同构串
行计算、无旁路可拆；PSI 的双池独立配额 + 异构打分原语天然可拆——
「开销从 Σ 压到 max」是 PSI 独有的硬件效率自由度，且不触碰精度路径
（精度由配置决定，overlap 只改调度不改数值）。

### 1.2 现状计时证据（全部来自已有 JSON）

**kernel 级全链对比**（kernel_comparison_indexers.json，decode 单 token
indexer 全链路 per-layer-call，H20-3e，S=131K）：

| 索引器 | total_ms | 分解 | 备注 |
|---|---|---|---|
| quest_official | 0.107 | score 0.094 + select_k 0.013 | 官方 decode_select_k，fp16 页界 |
| dsa_official | 0.503 | fp8_index 0.407 + topk 0.096 | 官方 tilelang，单 GEMM 串行链 |
| tli eager | 0.853 | — | 真实 trace |
| tli fusedL1 | 0.787 | — | M8 kernel 化后 |

**PSI decode 选择链分段**（tli_m8_topk_breakdown.json，生产形状
n=32 / S=131K / Tc=66048 / Hkv=8 / K1=128，near_compact 关闭口径）：

| 段 | ms | 资源画像 |
|---|---|---|
| L1 打分 kernel（KernelD 广播版） | 0.0634 | 内存受限（读 kmin/kmax pool 128MB） |
| L1 块 topk（K1=128 over NBLK=2048） | 0.0732 | CUDA core 选择/排序 |
| tli_compact（onehot→候选位置） | 0.0503 | CUDA core cumsum+展开 |
| L2 dual kernel（4bit 反量化+GEMV+双池直写） | 0.3854 | 内存受限（读 0.404GB + 写 0.069GB，有效 ~1.32TB/s vs 3TB/s 理想，见 tli_m8_tma_breakdown.json） |
| far topk（W_far=256 over Tc） | 0.4674 | CUDA core，compute-bound |
| near topk（W_near=1024 over Tc） | 0.4606 | 同上；**有限项占比仅 1.36%** |
| 全函数 | 1.8081 | — |

**关键结构事实**：far topk 与 near topk **互不依赖**（各自读 far_sc /
near_sc），是链上最大的可并行对（合计 0.928ms = 全链 51%）。但生产配置
`use_near_compact=True`（config.py L91-94）已把 near topk 宽度从 Tc=66048
砍到 WNCAP=2048（30×↓）——**生产口径下 near topk ≈ 0.02-0.05ms（估算：
0.4606 × 2048/66048 + 固定开销）**，纯双池 topk 并行的真实收益远小于
bench 口径的 25%。这是本设计最重要的诚实边界。

**prefill 侧结构证据**（tli_sel_profile.json，S≈21K chunk 化 prefill）：
`_tli_l2_score_batched_dual_kernel` 占 CUDA 时间 **52.46%**，两个 topk
kernel 合计 ~20.2%——选择链是 prefill 索引侧的主导成本，且 M10 慢路径
单次调用 1042ms@末 chunk（config.py L99-103 注释）。

**e2e 层**（c3_3arm_tp2_{tli,quest}.json，TP2 / bs16 / 64K / n_decode=64）：
PSI decode 93.9 vs Quest 121.1 vs dense 61.6 ms/step；prefill 136.0 vs
54.4s。增量索引开销口径：PSI 32 vs Quest 59.5 ms/step（E102，1.85×）。

**CPU/总线边界**（cpu_gather_overlap_bench.json，bs32/S131K/K2=1024，
KV 选中集 134MB）：GPU gather 213GB/s vs PCIe pinned 55GB/s；CPU 往返
57× 惩罚；**copy engine 与 GEMM 并发：serial 10.672ms → overlap
7.981ms（−25%）**——后者是 Ov-C 的实测背书。

---

## 2. 现状计算链与数据依赖图

### 2.1 依赖图（select_decode_batched，M8 生产路径）

```
q ──┬─→ [L1 打分 KernelD] ─→ sc1 ─→ [L1 块 topk] ─→ onehot ─→ [tli_compact] ─→ tok_c
    │                                                                  │
    └─→ [q2 refine（GEMV/投影，~20μs 级）] ──┐                        │
                                              ↓                        ↓
pool(kq 4bit) ────────────────→ [L2 dual kernel] ─→ far_sc ─→ [far topk] ─→ [far gather/where] ─┐
                                              └─→ near_sc ─→ [near topk] → [near gather/where] ─┤→ [merge/cat] ─→ sel
                                                        （滑窗/sink 强制段：纯索引算术，独立）──┘
```

依赖关键点（overlap 可行性的根据）：

- **far topk 与 near topk 零依赖**：far_sc / near_sc 是 KernelC 直写的
  两张独立分数表（kernels.py `_tli_l2_score_batched_dual_kernel`）；
- **滑窗/sink 强制段零依赖**：f_pos / f_pad 是纯索引算术
  （bench_m8_topk.py tail() L129-131），可与任何打分段并发；
- **q2 refine 与 L1 零依赖**：两者都只读 q（L1216 vs L1055）；
- **kmeans build 与整条 select 链零依赖**（cavg 臂）：build 只读 prefill
  KV（indexer.py L255-264），select 读质心——跨 chunk 时间上可完全
  隐藏；
- **L2 依赖 L1**（tok_c 由 L1 块选择展开）：链内不可并行，只能跨
  tile/chunk 流水（Ov-3）。

### 2.2 method 组合的异构原语

| 组合 | far 侧原语 | near 侧原语 | 硬件亲和 |
|---|---|---|---|
| mavg（主精度臂） | minmax 块界上界（区间算术归约） | avg 均值代表 GEMV | 双双 CUDA core；TC 化已实测 2.58× 慢（§6.1） |
| cavg（速度臂/ClusterKV 口径） | cluster 质心距离 **GEMM**（gpu_kmeans：x@c.T） | avg GEMV | **far→TC / near→CUDA core 天然分置** |
| aavg | avg GEMV | avg GEMV | 同构，仅可 stream 错峰 |
| mminmax | minmax 归约 | minmax 归约 | 同构双归约 |

---

## 3. overlap 机会矩阵

计算段 × 硬件单元 × 数据依赖（✓=可行且有收益，△=可行但收益存疑，
✗=不可行/负收益）：

| 计算段 | TC (MMA) | CUDA core | CPU | 可并行的无依赖对象 | 输入来源 |
|---|---|---|---|---|---|
| L1 界打分 | △（实测 2.58× 慢，内存受限） | ✓ 现状最优 | ✗ | q2 refine、kmeans build、gate 统计、下一 tile 的 L1 | q + kmin/kmax pool |
| L1 块 topk | ✗ | ✓ | ✗ | （下游全依赖它） | sc1 |
| tli_compact | ✗ | ✓ | ✗ | q2 refine、滑窗段 | onehot |
| L2 dual 打分 | △（带宽主导） | ✓ 现状 | ✗（57×） | kmeans build、build(chunk i+1)、L1(下一 tile) | q2 + kq pool + tok_c |
| far topk | ✗ | ✓ | ✗ | **near topk、滑窗段、（分块流水下）上一 chunk 的 L2 打分** | far_sc |
| near topk | ✗ | ✓ | ✗ | **far topk、滑窗段** | near_sc |
| 尾段 merge | ✗ | ✓（小 op） | ✗ | —（依赖两池结果） | i_f, i_n |
| kmeans build（cavg） | **✓ 唯一真 TC 段**（[Tfar,32]×[32,256] GEMM ×20 iter，compute-bound） | △ | ✗（实测 NO-GO） | **整条 select 主链（异步 stream 排后台）** | kfar_sub（prefill KV） |
| cluster far 打分（cavg） | △（[n·Hkv,32]×[32,256] 小 GEMM，μs 级，TC 无所谓） | ✓ | ✗ | **near avg GEMV（异构分置主对象）** | q2 + centroids |
| 反量化 | — | ✓ 已寄存器融合进 L2 dual | ✗ | — | kq_q/sc/mn |
| gate 统计（D'） | ✗ | △ | **✓（标量级数据，pinned + copy engine）** | **全部段（完全异步）** | prefill 末 chunk |

矩阵的三个读法：

1. **链内可并行对**只有（far topk, near topk）一对大的 + 若干小 op——
   decode 链的纯 stream 并行收益有限（§6.2）；
2. **链外可隐藏对象**（kmeans build / gate 统计 / 下一 chunk 的 build）
   是时间维度上的免费午餐——这才是 Ov-2/Ov-3 的主体；
3. **TC 在本架构中的正确位置**只有一个：kmeans build 的 20 轮 GEMM 迭代
   （compute-bound）；其余段全部内存或排序受限，TC 化是负资产（已有
   实测背书）。

---

## 4. 三个 overlap 层次设计

### Ov-1：双池 stream 并行（最浅层，decode + prefill 通用）

**设计**：把 far 分支（far topk → far gather/where）与 near 分支（near
topk → near gather/where）放两条 stream，滑窗/sink 强制段放第三条（或
与 near 同 stream），event join 后小 merge。双 stream 编译点：

```
s_main: L1 打分 → L1 topk → compact → L2 dual ──┬─ fork event e
s_far :  wait(e) → far_topk → far gather/where ──┤
s_near:  wait(e) → near_topk + 滑窗段 → near gather/where ─┤ join → merge/cat
```

**关键实现选择——优先「合并单 kernel」而非双 stream**：far/near topk
合并为一个 launch（把 far_sc/near_sc 行拼接做 batched topk，或写双出口
kernel）可以拿到同等的 SM 利用率提升，且**没有 event 同步开销、没有
CUDA graph 多流 capture 复杂度**（M5 graph replay 逐位一致性是 decode
路径的硬约束，torch.cuda.graph 的 fork/join capture 可录但验证成本高）。
双 stream 版本只在与链外对象（kmeans/gate）组合时才必要。

**组合语义对四臂的普适性**：far_sc/near_sc 独立在所有 method 组合下
成立（KernelC 双池直写与 method 无关——method 决定的是分数怎么算出来，
池结构不变），所以 Ov-1 对 mavg/cavg/aavg/mminmax 全部适用。

### Ov-2：异构硬件分置（method 组合维度）

**cavg 臂（结构最典型）**：

```
stream_TC   : kmeans build（20 轮 [Tfar,32]×[32,256] GEMM，后台异步，prefill 一次）
主链 decode : cluster far 打分（小 GEMM，TC 或 CUDA core 皆可，μs 级）
              ∥  near avg GEMV（读 near 带 kq pool，CUDA core/带宽）
              → max(T_far_cluster, T_near_avg) + merge
```

cavg 的 far 侧把「token 级 4bit 精筛 + 大宽度 topk」整体替换为「质心
GEMM + 小宽度 topk（K_c=256）」——far 侧成本从 (0.385×far 份额 +
0.467) ms 级坍缩到 μs 级（估算：[256, 32]×[32, 256] GEMM + topk over
256，<0.05ms）。**这是 Σ→max 最激进的一格**，但精度代价已知
（e2e 43.58 vs mavg 45.08，落后 1.50）——论文只能作为
「架构允许的速度上限 + 精度-速度 Pareto 前沿点」呈现。

**mavg 臂（主精度臂）**：far=minmax 归约 + near=avg GEMV 都是带宽/
归约型，**没有 TC 可分置**——overlap 手段退化为 Ov-1（topk 并行，收益
~3%，§6.2）+ Ov-3（分块流水，收益 15-30%，估算）。诚实结论：**主精度
臂的 overlap 红利主要在时域（流水）不在异构（硬件分置）**。

**aavg / mminmax**：同构原语，只能 stream 错峰（两池带宽共享，收益
≈ 延迟隐藏），列为最低优先级。

### Ov-3：L1→L2 跨 tile / 跨 chunk software pipelining（最深层）

两个正交的流水轴：

**(a) prefill 跨 chunk 流水（首选，收益最大）**：

```
chunk i:   [build(i):   quant4_pack + kmin/kmax ...........] → [select(i): L1→L2→topk→merge]
chunk i+1:              [build(i+1) ......................] ∥ select(i)   ← 后台 stream
```

build 只依赖 KV（已就绪），select 只依赖 q——chunk i+1 的 build 与
chunk i 的 select 零依赖，可完全隐藏。参照 FlashAttention 的 cp.async
双缓冲结构（producer/consumer 双 buffer 轮转），build 写 buffer B[i%2]
时 select 读 buffer B[(i+1)%2]。**与 F1 增量化的关系**：F1 把 build 从
O(S)/chunk 压到 O(chunk)，残余 build 仍可被流水隐藏——正交可叠加
（F1 砍总量，Ov-3 砍时序）。

**(b) 链内 tile 级流水（decode / prefill 的 select 内部）**：

q 行切成 tile（如 8 请求一组），L1(tile j+1) 与 L2+topk(tile j) 双
stream 交错——L1 读 kmin/kmax（128MB）与 L2 读 kq 4bit pool 是不同
内存区域，带宽可部分叠加（估算：两段带宽利用率均 ~44-50%，并发后
HBM 聚合带宽有望提升至 ~1.5-1.8TB/s 量级，**假设**：两段访问模式不
互相破坏 L2 cache 局部性——需实测）。far topk（compute-bound）与
dual kernel（bandwidth-bound）资源错峰，是链内最互补的流水对：
**score 分块产出 far_sc chunk k+1 ∥ topk 消费 far_sc chunk k（partial
topk 归并）**，把 0.385+0.467=0.852ms 的串行段压向 ~0.5-0.6ms
（估算，假设 topk partial merge 开销 <0.05ms）。

**tie 风险**：分块 partial topk 的并列值行为与整体 topk 可能不同——
与 DS topk 替换（#64）同类问题，已有 tie 容忍口径（jaccard 0.986-1.0
+ 有效集一致）可复用为验收标准。

### Ov-C：CPU 侧 overlap（附层）

- **kmeans 质心更新 / KV 搬 CPU：NO-GO（已实测）**。cpu_gather_overlap
  _bench.json：PCIe pinned 55GB/s vs GPU gather 有效 213GB/s，CPU 往返
  57× 惩罚。质心更新需读全部 far token 子空间（~126K token × 32 维 ×
  8 head fp32 ≈ 129MB/层），单趟 D2H 就 2.3ms+（估算）而 GPU GEMM 全程
  20 iter 是 ms 级——数据搬运成本吞掉全部收益。**结论：kmeans 留 GPU
  （TC stream），CPU 只做控制面**。
- **gate 统计（D' far_stat）CPU 化：可行**。prefill 末 chunk 的
  per-layer far mass 是标量级归约（KB 级数据），异步 stream + pinned
  memory D2H，与 GPU 计算完全并发；copy engine ∥ GEMM 的并发性已有
  实测背书（serial 10.672 → overlap 7.981ms，−25%）。CPU 拿到 far_stat
  后做阈值判断置 skip_far，下一 decode 步生效——时序上天然错开一步，
  无同步开销。
- **kmeans 质心 host 镜像 prefetch**（可选）：质心 [Hkv, 256, 32] fp32
  ≈ 4MB/层，D2H 0.07ms（估算 @55GB/s），供 CPU 侧诊断/gate 用，异步
  完全隐藏。

---

## 5. 与已有 kernel 的衔接（复用清单）

| 件 | 状态 | 在 overlap 方案中的角色 |
|---|---|---|
| `tli_l1_score_batched_dot`（M8-TC tf32 MMA） | 已在（`SGLANG_TLI_L1TC_KERNEL=1`） | L1 段直接复用；但 l1_tc_bench 实测比广播版慢 2.58×——**overlap 方案中 L1 默认仍走广播版 KernelD**，TC 版作为大 DP（≥64）消融口径保留 |
| `tli_l1_score_batched`（KernelD 广播版） | 已在，生产默认 | Ov-1/Ov-3 的 L1 段原样复用 |
| `tli_l2_score_batched_dual`（KernelC 双池直写） | 已在，生产默认 | **overlap 的结构基础**——far_sc/near_sc 独立 buffer 已经存在，stream 拆分零改动 kernel 本体 |
| `tli_compact` | 已在 | 复用 |
| `tli_sparse_gather_attn_dot`（消费端） | 已在 | 不动（merge 输出口径不变） |
| `gpu_kmeans`（GEMM 距离） | 已在（indexer.py L141） | Ov-2 的 TC 段：迁到独立 stream + 可选 tilelang/CUTLASS TC 化（现 PyTorch einsum 已走 TC，改动仅 stream 编排） |
| **新增件 1：双池合并 topk kernel** | 待写 | Ov-1 首选实现（单 launch，替代双 stream） |
| **新增件 2：partial topk 归并 kernel** | 待写 | Ov-3(b) 分块流水的 topk 消费端 |
| **新增件 3：build/select 双 buffer 轮转编排** | 待写（backend 层 Python，非 kernel） | Ov-3(a) prefill 跨 chunk 流水 |
| **新增件 4：gate 统计异步采集** | 待写（小改动） | Ov-C |

新增件合计约 2 个 Triton kernel + 2 段 Python 编排——**工程量远小于
M8 本体**，且每一件都可独立开关回退（沿用 `SGLANG_TLI_*` env 开关
惯例，config.py 增 4 项）。

---

## 6. H20-3e 定量预期

### 6.1 硬件约束先行：TC 稀缺改变了设计目标

H20-3e：78 SM，TC 吞吐 ~15% H100，HBM ~3TB/s（tma_m8_breakdown 的
ideal 口径）。两个实测判决确立设计铁律：

- **判决 A（L1 TC 化负收益）**：tli_l1_tc_bench.json——KernelD 广播版
  0.0391ms vs tl.dot tf32 0.1008ms（**2.58× 慢**）。原因：L1 打分读
  128MB pool，带宽主导，TC 化不省流量只添 MMA 排布开销。**推论：内存
  受限段上 TC 无用，「cluster 上 TC + 归约上 CUDA core」的收益不在
  压缩单段耗时，而在避免多段挤占同一资源**——若把 kmeans GEMM 与
  minmax 归约都堆到 CUDA core，78 SM 的 ALU 排队；分置后 GEMM 吃 TC、
  归约吃 ALU，**互不争抢才并发得起来**。H20 的 TC 稀缺意味着这个分置
  不是「锦上添花」而是「必需」：全挤 TC 会饱和（15% 吞吐装不下全部
  GEMM），全挤 CUDA core 也会与 topk/归约争抢。
- **判决 B（带宽是共享资源）**：两个内存受限段并发（如 L1 读 pool ∥
  L2 读 pool）不减少 HBM 总流量，收益仅剩延迟隐藏与 cache 错峰——
  估算时对这类并发取保守系数（η≈0.3-0.5）。

### 6.2 decode 侧临界路径估算

串行基线（生产口径，near_compact 开；分项来自 tli_m8_topk_breakdown
.json，near topk 按宽度比折算——**估算**）：

```
T_serial ≈ 0.063(L1) + 0.073(L1 topk) + 0.050(compact) + 0.385(dual)
         + 0.467(far topk) + ~0.03(near topk, 折算) + ~0.15(尾段/merge)
         ≈ 1.22 ms / 层调用 @bs32/131K
```

各 overlap 层次叠加后的临界路径（**全部为估算，假设逐条列出**）：

| 方案 | 公式 | 估算值 | 假设 |
|---|---|---|---|
| Ov-1 双池 topk 并行（生产口径） | T − T_near | ~1.19ms（−3%） | near_compact 已砍宽度，收益封顶 |
| Ov-1（near_compact 关的 bench 口径） | T − min(T_far, T_near) | ~1.35/1.81ms（−25%） | 完美并发 η=1，event 开销 ~20μs |
| Ov-3(b) far 分块流水（score∥topk） | T − (T_dual_far + T_far_topk) + max(·) | **~0.85-1.0ms（−18~30%）** | partial topk 归并 <0.05ms；带宽并发 η≈0.5；tie 行为对拍通过 |
| Ov-2 cavg 臂（far 侧坍缩） | T − T_dual_far − T_far_topk + T_cluster(μs 级) | **~0.4-0.5ms（−60%）** | 质心打分 [256,32]×[32,256] 忽略不计；**精度 −1.50 已知** |
| 全叠加（mavg：Ov-1+Ov-3b） | — | **~0.8-0.95ms（−22~34%）** | 上述假设交集 |

**换算到 e2e 口径（估算）**：decode 增量索引开销 32ms/step（36 层 ×
~0.89ms/层，E102 口径）。mavg 主臂全叠加省 22-34% → **32 → ~21-25
ms/step**，对 Quest 59.5 的优势从 1.85× 扩到 **~2.4-2.8×**（假设：
overhead 与选择链成线性比、TP2 形状不变、graph replay 兼容）。
cavg 速度臂 → 索引开销 ~12-15ms/step 量级（估算），但精度代价同臂。

### 6.3 prefill 侧估算

M10 慢路径 1042ms/调用@末 chunk 是基线锚点。分层数（无逐段 JSON，
结构依据 sel_profile：dual 52.5% + topk 20%）：

```
T_chunk(overlap) ≈ max(T_build(i+1), T_select(i)) + ε   vs   T_build + T_select
```

- **前置**：F1 增量化先把 T_build 砍 ~7/8（64K/8K chunk 口径，#126）；
  残余 build 与 select 流水后，chunk 临界路径 ≈ T_select（假设 build
  残余 < select，F1 后大概率成立——**需 F1 落地后实测确认**）；
- select 内部再叠 Ov-3(b) 分块流水（dual 52.5% 与 topk 20% 错峰）：
  粗估 prefill 索引侧总时长再降 **15-30%（估算）**；
- e2e 锚点：prefill 136.0s vs Quest 54.4s（c3_3arm_tp2）——索引侧只
  是差距的一部分（#126 归因 F1-F4 全链），overlap 单独不足以翻盘，
  **必须与 F1/F2/F3/F4 打组合拳**（§9）。

---

## 7. 差异化叙事（修正版，重要）

**废弃的错误说法**：「learned router 无法复制 / Quest 是 learned 所以慢」
——不成立。sglang 的 Quest 复现臂是 **training-free 页界索引**
（#126 调查 0.1：改造基底 = TLI backend，差异只在索引与选择），kernel
级 microbench 它还是全场最快（0.107 vs 0.787ms）。

**正确的差异化**（写进论文与本文档的口径）：

1. **单级索引整链同构、无旁路可拆**：Quest 的选择链 = 单一 GEMV 序列
   （page 分数 einsum → topk），所有段读同一份页界数据、做同一种
   运算——stream 拆分后各段争抢同一资源（GEMV 带宽 + topk ALU 串行
   依赖），**Σ 无法压成 max**；DSA（真 learned）的 indexer 本身就是
   单条 GEMM 串行链（fp8_index 0.407ms 一整段），结构上没有可并发
   的独立分支。
2. **PSI 双池独立配额 + 异构打分原语才有分置自由度**：far_sc/near_sc
   独立 buffer + 独立 top-K（B' 的防挤出设计**同时**是并行度来源）；
   far 侧原语可换（minmax 归约 ↔ cluster GEMM）而 near 侧不动——
   method 组合维 = 硬件分置维。这是「架构级硬件效率优势」的准确表述。
3. **诚实边界**：kernel 级 Quest 仍最快；PSI 的已实测速度优势在
   （a）decode 增量索引开销 1.85×（32 vs 59.5ms/step，源于 4bit 子空间
   直读 pool + CUDA graph + 无全宽物化），（b）1/32 计算量、（c）336B
   存储。overlap 是**设计潜力**：把 PSI 多段链的 Σ 压向 max，方向是
   缩小与 Quest 单链的 kernel 差距 + 扩大 e2e 增量口径优势——论文
   措辞用 enables/allows 级，未实测前不写数字主张。

---

## 8. 实现计划（分阶段，验收口径与风险）

统一验收框架：复用 `test_c3_3arm_bench.py` / `test_c3_3arm_tp2.py` 三段
计时（prefill / decode_ms_per_step / tok_per_s），加两类对拍：
（i）**逐位对拍**——overlap 只改调度不改数值，sel 输出须与串行版逐位
一致（哨兵口径，M5 graph replay 同标准）；（ii）**tie 容忍对拍**——
分块 partial topk / 合并 topk 用 jaccard ≥0.986 + 有效集一致（复用 #64
DS topk 口径）。

| 阶段 | 内容 | 工程量 | 验收口径 | 主要风险 |
|---|---|---|---|---|
| **P0（前置，#127 范畴，非本设计）** | F1 增量化 + F3 惰性化 + F2 去物化 | — | 见 #126 报告 | 无冲突（§9 正交性） |
| **P1：Ov-1 合并双池 topk kernel** | 单 launch 处理 far_sc+near_sc 两表（行拼接 batched 或双出口）；env `SGLANG_TLI_MERGED_TOPK` | 小（1 个 Triton kernel） | decode 选择链分段计时：far+near topk 合计 0.49ms → 单 kernel 目标 ~0.3-0.35ms（估算）；逐位/tie 对拍 | topk 归并逻辑正确性；小宽度下收益被 launch 开销吃掉（near_compact 生产口径收益 ~3%——若实测 <5% 如实记 negative result） |
| **P2：Ov-C gate 统计异步** | pinned + 异步 stream + copy engine | 很小 | far_stat 获取零阻塞 GPU 主链；D' 开关行为不变 | 时序错一步的语义确认（下步生效） |
| **P3：Ov-3(a) prefill 跨 chunk 流水** | build(i+1) ∥ select(i) 双 buffer 轮转（backend 层） | 中 | bench_m10_prefill / 30K e2e prefill 段对比（F1 后基线）；chunk 临界路径 ≈ max(·) | F1 未落地时收益被 O(S) build 淹没——**严格排在 F1 后**；双 buffer 显存 +1 chunk 索引（kq 40B/token-head × 8K chunk ≈ 2.6MB/层/请求，可忽略） |
| **P4：Ov-3(b) far 分块流水** | dual kernel 分块产出 + partial topk 归并消费 | 中大 | decode 选择链 −15-30%（估算区间，低于 10% 如实记 negative）；tie 对拍 | partial topk tie 行为；CUDA graph capture 兼容（分块轮转的静态形状化，复用 M5 静态宽度手法） |
| **P5：Ov-2 cavg 异构分置** | kmeans build 后台 TC stream ∥ 主链；cluster 打分 ∥ near GEMV | 中 | cavg 臂 decode 选择链 −60%（估算）；质心逐位一致（seed 固定）；**精度口径不动**（43.58 已知，不重跑 13 任务除非用户拍板） | 收益大但绑在精度落后臂上——只作 Pareto 前沿点 |
| **P6（可选）：双 stream 版 Ov-1** | 仅当 P1 合并 kernel 证实 SM 争抢时才做 | 中 | graph fork/join capture replay 逐位一致 | 多流 capture 复杂度（M5 硬约束） |

顺序理由：P1/P2 低风险先行拿到「overlap 已落地」事实；P3 依赖 F1；
P4 是主精度臂最大收益项但工程最重；P5 锦上添花且自带精度注脚。

---

## 9. 与 #126 修复建议（F1-F6）的正交性自查

| #126 项 | 与 overlap 的关系 |
|---|---|
| F1 prefill 索引增量化 | **正交可叠加，且是 P3 的前置**：F1 砍 build 工作总量（O(S)→O(chunk)），Ov-3(a) 压残余段的时序（max 化）。不冲突 |
| F2 消除全量 fp32 物化 | 正交：F2 砍流量，overlap 不依赖流量假设（除 §6.1 判决 B 的保守系数会因 F2 改善） |
| F3 kq_f 反量化表惰性化 | 正交：F3 删纯浪费，overlap 分置的是有效工作。不冲突 |
| F4 prefill 跨请求批量化 | **互补相乘**：F4 提供请求维并行度，双池维提供池间并行度——批量化后 n 大，SM 打满更容易，overlap 系数 η 上修 |
| F5 host 同步消除 | 互补：F5 是 overlap 的必要条件之一（流上有 .item() 同步则 stream 并发失效） |
| F6 kq 4bit 构建流水化 | 正交（build 侧） |

**自查结论**：零冲突；实现顺序建议 F1/F3/F5（低难度大收益）→ P1/P2 →
F2/F4 → P3/P4 → P5。每个 overlap 件均有 env 开关可独立回退，不引入
任何精度路径改动（数值逐位不变或 tie 容忍口径内）。

---

## 10. 论文转译段落草稿（中/英，scope 安全版）

### 中文草稿（§4.x 硬件效率小节）

> **双池结构与选择链的并发自由度。** PSI 的两级选择链由多个可独立调度的
> 计算段组成：块上界粗筛、far/near 双池精筛、以及两个配额独立的 top-K
> 选择。由于远端池与近端池在数据流上互不依赖（独立输入缓冲、独立预算、
> 独立 top-K），且两侧打分原语异构（聚类距离为 GEMM、块界上界为区间
> 归约、均值代表为 GEMV），该结构**允许**将选择链开销从各段之和压缩为
> 分置并发后的分组最大值：远端侧的 GEMM 打分可分置到 Tensor Core 流，
> 近端侧的归约与 top-K 留在 CUDA core 流并发执行。作为对照，单级页界
> 索引（Quest）的选择链是单一同构 GEMV 序列，段间共享同一数据依赖链、
> 无可并发旁路；learned 索引器（DSA）本身即为单条 GEMM 串行链。在
> Tensor Core 吞吐仅为 H100 约 15% 的 H20 部署上，这一分置自由度尤具
> 意义：计算段的资源错峰使索引链不至于与注意力主计算争抢同一硬件单元。
> 本节为架构层设计分析；已实测兑现的速度优势仍以 §4.4 的 decode 增量
> 索引开销口径为准（32 vs 59.5 ms/step），overlap 压缩比的实测验证
> 留作后续工作。

### English draft (§4.x hardware-efficiency subsection)

> **Dual-pool structure and scheduling freedom of the selection chain.**
> The PSI selection chain decomposes into independently schedulable
> segments: block-bound coarse scoring, far/near dual-pool refinement, and
> two quota-independent top-K selections. Because the far and near pools
> share no data dependency (separate score buffers, separate budgets,
> separate top-K) and their scoring primitives are heterogeneous—cluster
> distances are GEMMs, block-bound scores are interval reductions, and
> mean representatives are GEMVs—the chain *allows* its cost to be
> compressed from the sum of segment latencies toward the maximum over
> hardware-partitioned groups: far-side GEMM scoring can be placed on a
> Tensor Core stream while near-side reductions and top-K run concurrently
> on CUDA cores. By contrast, a single-level page-bound index (Quest) is
> one homogeneous GEMV sequence with no concurrent bypass, and a learned
> indexer (DSA) is a single serialized GEMM chain. On H20, where Tensor
> Core throughput is roughly 15% of H100's, this partition freedom is
> particularly relevant: staggering segments across units keeps the index
> chain from contending with attention for the same hardware. This
> subsection is a design-level analysis; measured speed advantages remain
> those of §4.4 (decode incremental indexing overhead, 32 vs 59.5
> ms/step), with empirical validation of the overlap compression ratio
> left to future work.

措辞自查：只用 enables/allows/「允许」级；唯一数字主张（32 vs 59.5）
是已实测的既有口径；对 Quest/DSA 的对照限于结构性陈述不贬损其单链
速度（kernel 级 Quest 最快的事实由 §4.4 如实报告）；精度零涉及。

---

## 11. 数据源清单（本文全部实测数字的出处）

| 数字 | JSON / 代码 |
|---|---|
| quest 0.107 / dsa 0.503 / tli eager 0.853 / fusedL1 0.787 ms @131K | research/bench/kernel_comparison_indexers.json |
| 选择链分段 1.8081ms 全表（l1 0.0634 / dual 0.3854 / far_topk 0.4674 / near_topk 0.4606 / finite 0.7939/0.0136） | research/results/tli_m8_topk_breakdown.json（bench_m8_topk.py：n=32/S131K/Tc=66048，near_compact 关口径） |
| dual kernel 带宽账（0.3592ms / 0.404GB 读 / ideal@3TB/s 157.7μs） | research/results/tli_m8_tma_breakdown.json |
| L1 TC 化 2.58× 慢（0.0391 vs 0.1008ms） | research/results/tli_l1_tc_bench.json |
| CPU 边界（213 vs 55GB/s / 57× / −25% 并发） | research/bench/cpu_gather_overlap_bench.json |
| prefill 结构（dual 52.46% / topk ~20%） | research/results/tli_sel_profile.json |
| e2e 三臂（93.9/121.1/61.6 ms/step；136.0/54.4s） | research/results/c3_3arm_tp2_{tli,quest}.json + E102（记忆归档） |
| decode 增量索引开销 32 vs 59.5 ms/step | E102 判决（round4 记忆归档；增量化差分口径） |
| cavg 精度 43.58 vs mavg 45.08 | ClusterKV 显式化（2026-10-06 轮询第 4 轮 B 线，记忆归档） |
| M10 慢路径 1042ms@末 chunk / WNCAP=2048 / DS topk 4.4-10.2× / near 有限项 ~920/65728 | python/sglang/srt/layers/attention/tli/config.py 代码注释 |
| H20 78 SM / TC ~15% H100 | kernels.py 头注释与本机规格（用户给定） |

---

## 12. 自查清单（写完复核）

- [x] 原始任务（overlap 设计文档一个文件）完成：机会矩阵 / 三层次 /
      CPU 附层 / 衔接 / H20 定量 / 修正叙事 / 实现计划 / 论文草稿全覆盖
- [x] 数字只用 JSON/代码注释实测值，预测全部标「估算」+ 假设
- [x] 不 overclaim：主臂收益标估算区间、cavg 精度代价同臂披露、
      论文段落 enables/allows 级、kernel 级 Quest 最快如实保留
- [x] 与 #126 F1-F6 零冲突且正交关系逐项说明（§9）
- [x] 修正版差异化叙事（Quest=training-free 同构单链 / DSA=GEMM 串行链）
      全文一致，无「learned 无法复制」残留
- [x] method 组合四臂（mavg/cavg/aavg/mminmax）逐一讨论 overlap 可能性
- [x] CUDA graph 兼容性（M5 逐位一致硬约束）作为风险项显式列出
