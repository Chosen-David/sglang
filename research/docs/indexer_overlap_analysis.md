# Indexer 方法 Overlap 潜力分析（Prefill + Decode 双阶段）

> 调研日期：2026-10-06  
> 数据来源：`two-level-attention/sparse_attn/indexer/` 全部 indexer 实现 + `exp/trace/probe_e108_sim_greedy.py` 实测  
> 目标：分析每种 indexer method 的算子构成，评估其在 prefill 和 decode 两个阶段与 GPU Attention（Tensor Core）实现 kernel 级 overlap 的可行性

---

## 1. 两阶段执行模型

### 1.1 当前调用链（`qwen3_attn_patch.py`）

**Prefill 阶段**（`past_key_values.get_seq_length == input_shape[1]`）：
```
qwen3_attn_forward
  ├─ indexer.clear()                    ← 重置索引状态
  ├─ indexer.observe_prefill_q(q)       ← 旁路采集（仅 E85f 静态 pair）
  └─ attention_interface(...)           ← dense attention（TC GEMM 全序列）
       └─ 不调用 prepare_mask！不做 sparse 选择！
```

**Decode 阶段**（`else` 分支）：
```
qwen3_attn_forward
  ├─ indexer.prepare_mask(q, q_ids, k, cu_seqlens_k)
  │    ├─ prepare_index(k, ...)        # 构建索引（k_min/k_max/k_qat/centroids）
  │    │    └─ _maybe_build_kmeans()   # ← 第一个 decode step 才触发！
  │    ├─ compute_score(q, ..., idx)   # q 对索引打分
  │    └─ compute_mask(q_ids, scores)  # topk → bool mask
  └─ eager_decoding_attn(q, k, v, mask) ← sparse attention（TC GEMM）
```

### 1.2 关键发现：索引 build 的触发时机

| 阶段 | indexer 调用 | 索引 build | 阻塞点 |
|------|-------------|-----------|--------|
| Prefill | `clear()` + `observe_prefill_q()` | ❌ 不 build | 无（dense） |
| Decode step 1 | `prepare_mask()` | ✅ **全量 build** | **同步阻塞** |
| Decode step 2+ | `prepare_mask()` | 复用缓存 | topk 同步 |

**核心瓶颈**：第一个 decode step 的 `prepare_index` 会 build 全量索引（30K tokens），这是同步阻塞的：
- k-means：20 迭代 × 30K tokens → 实测 **61s/step**（代码注释）
- sim_greedy：30K tokens 单遍 → E108 实测 **5-27s**

### 1.3 两阶段 Overlap 的本质

**Prefill 阶段**：GPU 做 dense attention（TC GEMM，O(S²)），CPU 空闲 → **可利用 CPU 异步 build 索引**

**Decode 阶段**：GPU 做 sparse attention（TC GEMM），indexer 每步计算 mask → **可利用 CPU 异步增量更新索引**

统一目标：**索引 build/更新的时间 < Attention 时间 → 完全隐藏，零气泡**

---

## 2. Prefill 阶段 Overlap 分析

### 2.1 Prefill 的计算特征

| 特征 | 值 | 说明 |
|------|---|------|
| Attention 复杂度 | O(S²) | S=30K → 每层 ~7.4e12 FLOPs |
| 每层时间 | ~50-150ms | TC GEMM + MLP |
| 36 层总时间 | **~2-5s** | 含内存访问、kernel launch |
| CPU 状态 | **空闲** | 当前无 indexer 调用 |

### 2.2 Prefill 阶段的 Overlap 机会

当前 prefill 是 dense attention，indexer 不被调用。但这意味着 **CPU 完全空闲**，可以用来异步 build 索引：

```
时间轴 ──────────────────────────────────────────►

GPU (TC):  [L0 dense attn] [L1 dense attn] [L2 dense attn] ...
                ~100ms          ~100ms          ~100ms
CPU:       [idle]            [build L0 idx]  [build L1 idx]  ...
                              ↑ 每层 prefill 完成后
                                k 数据可用，CPU 开始 build
```

**关键问题**：每层 CPU build 时间 vs 每层 GPU prefill 时间

### 2.3 各 Method 的 Prefill Build 时间估算

#### minmax/avg/4bit

```
每层 build 操作：
  k_min/k_max:  [30K, 8, 32] → amin/amax over BS=64 → [469, 8, 32]
  k_avg:        [30K, 8, 32] → mean over BS=64 → [469, 8, 32]
  4bit quant:   [30K, 8, 32] → elementwise → [30K, 8, 32]

FLOPs: ~30K × 8 × 32 × 3 ops = ~23M FLOPs
CPU 时间: ~2-3ms/层
```

**Overlap 评级：★★★★★（完美）**
- CPU build ~3ms << GPU prefill ~100ms → **完全隐藏**
- 且 prepare_index 是纯 elementwise/reduction，无 GEMM 依赖

#### sim_greedy（增量贪心聚类）

```
每层 build 操作（30K tokens 全量）：
  for i in range(30000):
    bmm: [8, K, 32] × [8, 32, 1] → 8×K×32 FLOPs
    K=1200 (sim=0.80): 307K FLOPs/step
    30K steps: 9.2 GFLOPs

CPU 单核 ~10 GFLOPS → ~1s/层
36 层串行: ~36s
```

**Overlap 评级：★★★☆☆（部分可行）**
- CPU build ~1s/层 >> GPU prefill ~100ms/层 → **CPU 是瓶颈，无法完全隐藏**
- 但可以**多线程**（8 线程 → ~125ms/层）接近 GPU 速度
- 或**降低 sim 阈值**（sim=0.80 → K~1200，比 sim=0.90 的 K~4700 少 4 倍）

**优化路径**：
1. **多线程 CPU**：8 线程并行 → 每层 ~125ms ≈ GPU prefill 速度
2. **NumPy/BLAS 优化**：`bmm` 用 BLAS 加速 → 每步 ~30μs → 30K steps → ~0.9s/层（单线程）
3. **C++ 实现**：消除 Python 循环开销 → 理论 ~50ms/层

#### k-means

```
每层 build 操作（30K tokens，20 迭代）：
  for iter in range(20):
    x @ c.T: [30K, 32] × [32, K] → 30K×32×K = ~30M FLOPs (K=256)
    argmax:  [30K, K] → reduction
    index_add: scatter
    
  20 迭代 × (GEMM + argmax + index_add) = 20 × 4 kernel launches
  实测: 61s/step（代码注释）
```

**Overlap 评级：★☆☆☆☆（不可行）**
- 每层 build ~200ms（20 迭代 × 10ms）≈ GPU prefill 速度
- 但 36 层总 build ~7.2s >> prefill 总时间 ~2-5s
- **迭代特性导致无法分段**：必须等 20 次迭代全部完成才能用

### 2.4 Prefill 阶段 Overlap 方案

**方案 P1：CPU 异步 build（推荐 minmax/avg/4bit）**

```
GPU: [L0 prefill] → [L1 prefill] → [L2 prefill] → ...
CPU:     ↓            ↓              ↓
      [build L0]   [build L1]    [build L2]
      ~3ms          ~3ms          ~3ms
      ↑ 完全隐藏在 L1/L2/L3 的 prefill 内
```

**方案 P2：流水线 build（sim_greedy 优化后）**

```
GPU: [L0 prefill] → [L1 prefill] → [L2 prefill] → ...
CPU:     ↓            ↓              ↓
      [build L0]   [build L1]    [build L2]
      ~125ms        ~125ms        ~125ms
      ↑ 与下一层 prefill 并行，prefill 完成后
        索引已 build 到 L35
```

**方案 P3：Prefill 后集中 build（所有 method）**

```
GPU: [L0 prefill] → [L1 prefill] → ... → [L35 prefill] → [decode step 1]
CPU:                                               ↓
                                              [build all layers]
                                              ~36s (sim_greedy)
                                              ↑ 用户感知延迟
```

---

## 3. Decode 阶段 Overlap 分析

### 3.1 Decode 的计算特征

| 特征 | 值 | 说明 |
|------|---|------|
| 每步新增 token | 1 | decode 每步只生成 1 token |
| Attention 复杂度 | O(S) | 每步 Q[1,H,D] × K[S,H,D] → ~134M FLOPs |
| 每层时间 | ~200μs | TC GEMM |
| 36 层总时间 | **~7ms** | 36 层串行 |
| indexer 调用 | 每层每步 | prepare_index + compute_score + compute_mask |

### 3.2 Decode 阶段的 Overlap 机会

**Decode step 1（索引已 build 后）**：
```
GPU: [L0 attn] → [L1 attn] → [L2 attn] → ...
CPU:   ↓          ↓          ↓
    [L1 update] [L2 update] [L3 update]  ← 增量更新 1 token
    ~128μs      ~128μs      ~128μs
    ↑ 完全隐藏在下一层 Attention 内
```

**Decode step 2+**：
```
GPU: [L0 attn] → [L1 attn] → ...
CPU:   ↓          ↓
    [L1 update] [L2 update]  ← 增量更新 1 token
    ~128μs      ~128μs
```

### 3.3 各 Method 的 Decode Update 时间

| Method | 每步操作 | FLOPs | CPU 时间 | GPU 时间 | Overlap? |
|--------|---------|-------|---------|---------|---------|
| **minmax/avg** | 增量 amin/amax/mean | ~8×32×1 = 256 | ~1μs | ~10μs | ✅ 完美 |
| **4bit** | 1 token quant | ~8×32×3 = 768 | ~1μs | ~5μs | ✅ 完美 |
| **sim_greedy** | bmm + argmax + index_add | ~307K | ~128μs | ~10μs | ✅ CPU 可行 |
| **k-means** | 全量重算（20 迭代） | ~600M | ~200ms | ~200ms | ❌ 不可行 |

---

## 4. 各 Method 算子分解与 Overlap 评级

### 4.1 minmax（块 min/max 上界粗筛 + 4bit token 精筛）

**实现**：`tia_indexer.py` / `tli_indexer.py`（`far_method="minmax"`）

| 阶段 | 算子 | 形状 | TC 友好？ | 同步点 | Prefill Build | Decode Update |
|------|------|------|-----------|--------|--------------|--------------|
| prepare_index: k_min/k_max | `amin/amax` over block | [T/BS, Hkv, d'] | ❌ reduction | 0 | ~2ms | ~1μs |
| prepare_index: 4bit quant | `round + clamp` | [T, Hkv, d'] | ❌ elementwise | 0 | ~3ms | ~1μs |
| compute_score: coarse | `einsum(q·k_min)` | [H, T/BS] | ✅ GEMM | 0 | — | — |
| compute_score: fine | `einsum(q·k_qat)` | [H, T] | ✅ GEMM | 0 | — | — |
| compute_mask: L1 topk | `topk + scatter` | [Hkv, T/BS] | ❌ topk | 1 | — | — |
| compute_mask: L2 topk | `topk + scatter` | [Hkv, T] | ❌ topk | 1 | — | — |

**Overlap 评级：★★★★★（双阶段最优）**

- **Prefill**：CPU build ~3ms << GPU prefill ~100ms → 完全隐藏
- **Decode**：增量更新 ~2μs << GPU attention ~200μs → 完全隐藏
- **唯一阻塞**：两次 topk（GPU kernel launch），但输入小（T/BS ~ 几百）

### 4.2 avg（块均值粗筛）

**实现**：`tli_indexer.py`（`far_method="avg"` / `near_method="avg"`）

| 阶段 | 算子 | 与 minmax 差异 |
|------|------|---------------|
| prepare_index: k_avg | `mean` over block | 比 min/max 多一次 sum |
| compute_score: avg | `einsum(q·k_avg)` | 无需 clamp 分裂 |

**Overlap 评级：★★★★★（与 minmax 同级）**

- 算子结构与 minmax 几乎一致
- mean reduction 同样无 GEMM 依赖，适合 CPU
- compute_score 比 minmax 更简单（无需 clamp(≤0)/clamp(≥0) 拆分）

### 4.3 Quest（块 min/max 单级选择）

**实现**：`quest_indexer.py`

| 阶段 | 算子 | TC 友好？ | 同步点 |
|------|------|-----------|--------|
| prepare_index | `amin/amax` | ❌ | 0 |
| compute_score | `einsum + stack+amax` | ✅ GEMM | 0 |
| compute_mask | `sort + scatter` | ❌ sort | 1 |

**Overlap 评级：★★★★★（最优，但精度最低）**

- **单级选择**，只有一次 sort，同步点最少
- 无 4bit quant（无 L2 精筛），prepare_index 最轻
- **但**：精度最低（LongBench 47.72 vs TLI 50.54）

### 4.4 k-means（迭代式聚类）⭐ 反面教材

**实现**：`tli_indexer.py` `_gpu_kmeans()` + `probe_e108_sim_greedy.py` `kmeans_cluster()`

```python
for _ in range(niter):           # ← 迭代！20 次
    a = (x @ c.T).argmax(dim=1)  # GEMM + argmax
    cnt = torch.bincount(a, ...) # reduction
    sums = torch.zeros(...).index_add_(0, a, x)  # scatter
    c[ne] = sums[ne] / cnt[ne]   # elementwise
```

| 阶段 | 算子 | TC 友好？ | 同步点 | 迭代次数 |
|------|------|-----------|--------|----------|
| assign | `x @ c.T` | ✅ GEMM | 1 | 20 |
| assign | `argmax` | ❌ | 1 | 20 |
| update | `bincount + index_add` | ❌ scatter | 1 | 20 |
| update | `c = sums / cnt` | ❌ | 1 | 20 |

**20 迭代 × 4 同步点/迭代 = 80 次强制同步**

**Overlap 评级：★☆☆☆☆（双阶段均不可行）**

| 阶段 | 问题 | 结论 |
|------|------|------|
| Prefill | 20 迭代 × 30K tokens，每层 ~200ms，36 层 ~7.2s > prefill 总时间 | ❌ 无法 overlap |
| Decode step 1 | 全量 build 61s（实测），同步阻塞 | ❌ 不可用 |
| Decode step 2+ | far 区不变可复用，但 step 1 已阻塞 61s | ❌ 首步致命 |

**根本问题**：迭代内每步的数据依赖（`c` 更新后才能算下一个 `assign`）使得 CPU 预计算不可能——你无法在 Layer n 的 prefill 期间预算 Layer n+1 的 kmeans，因为 kmeans 需要完整 k 序列且迭代收敛不可分段。

### 4.5 sim_greedy（增量贪心聚类）⭐ 核心分析

**实现**：`probe_e108_sim_greedy.py` `greedy_cluster_assign()` + `analyze_e4d_greedy_cluster.py`

```python
for i in range(T):                          # ← 逐 token，单遍！
    xi = x[i]                               # [Hkv, d']
    cos = bmm(sums, xi) / (norm * xi_n)     # [Hkv, K]  ← 小 GEMM
    a = cos.argmax(-1)                       # [Hkv]     ← 小 argmax
    upd = (cos.gather(...) >= sim)          # [Hkv]     ← 阈值判断
    sums.index_add_(0, flat, xi * w)        # scatter add（增量更新簇心）
```

#### 4.5.1 算子分解（单 token 步）

| 算子 | 形状 | TC 友好？ | 同步点 | 说明 |
|------|------|-----------|--------|------|
| `bmm(sums, xi)` | [Hkv, K, d']×[Hkv, d', 1] | ✅ 小 GEMM | 0 | K≤T, d'=32 |
| `norm = sq.sqrt()` | [Hkv, K] | ❌ | 0 | 增量维护 sq 避免全量 |
| `cos = dot / norm` | [Hkv, K] | ❌ | 0 | elementwise |
| `masked_fill` | [Hkv, K] | ❌ | 0 | 掩盖死簇 |
| `argmax(-1)` | [Hkv] | ❌ | **0** | 极小（K 维） |
| `gather + >= sim` | [Hkv] | ❌ | 0 | 阈值判断 |
| `index_add_` (×3) | scatter | ❌ | **0** | 增量更新 sums/cnt/sq |
| `k_live += (~upd).long()` | [Hkv] | ❌ | 0 | 簇计数器 |

**关键发现：每步同步点 = 0**

#### 4.5.2 与 k-means 的本质区别

| 特性 | k-means | sim_greedy |
|------|---------|------------|
| 迭代 | 20 次 | **1 遍** |
| 每步数据依赖 | 依赖上一次迭代的 `c` | 仅依赖当前 `sums`（增量） |
| 同步点/步 | 4 | 0 |
| 总同步点 | 80 | 0 |
| 可分段？ | ❌（迭代不可拆） | ✅（每 token 独立） |
| 增量更新？ | ❌（全量重算） | ✅（`index_add_`） |

#### 4.5.3 Overlap 评级：★★★★★（双阶段最优）

**Prefill 阶段**：

| 方案 | 每层 build 时间 | 与 prefill 比较 | 结论 |
|------|---------------|----------------|------|
| Python 单线程 | ~1s | >> 100ms | ❌ 太慢 |
| Python 多线程（8线程） | ~125ms | ≈ 100ms | ✅ 接近 |
| NumPy/BLAS 优化 | ~0.9s | >> 100ms | ❌ 太慢 |
| **C++ 实现** | **~50ms** | **< 100ms** | **✅ 可隐藏** |

**Decode 阶段**：

| 操作 | FLOPs | CPU 时间 | GPU Attention | Overlap? |
|------|-------|---------|-------------|---------|
| 增量更新 1 token | ~307K | ~128μs | ~200μs | ✅ 完美 |

#### 4.5.4 实测数据支撑（E108 探针）

| sim | mass | C/N | C_mean | build_s（全量 30K） |
|-----|------|-----|--------|-------------------|
| 0.80 | 0.9021 | 0.064 | 1174 | 5.15 |
| 0.85 | 0.9052 | 0.122 | 2245 | 5.07 |
| **0.90** | **0.9092** | **0.252** | **4729** | **27.12** ⚠️ |
| 0.92 | 0.9106 | 0.344 | 6507 | 5.03 |
| 0.94 | 0.9118 | 0.461 | 8808 | 5.14 |
| 0.96 | 0.9150 | 0.587 | 11362 | 5.12 |
| 0.98 | 0.9181 | 0.700 | 13710 | 5.04 |
| 0.99 | 0.9249 | 0.739 | 14539 | 5.05 |
| **锚点** | | | | |
| full_fine | 0.9398 | — | — | — |
| minmax_mono | 0.9092 | — | — | — |
| cavg(sim0.9) | 0.8896 | — | — | — |
| kmeans512 | 0.9255 | 0.031 | — | — |
| kmeans1024 | 0.9301 | 0.061 | — | — |

**关键发现**：
1. sim=0.90 的 `build_s=27.12s` 异常高（其他 sim ~5s）——这是 **Python for 循环 + GPU kernel launch** 的开销（K=4729 时 bmm 每步 launch 一次 GPU kernel，T~30K 步 → 30K launches）。**这正是 CPU 异步化的动机**
2. sim=0.80 时 C/N=6.4%（簇数极少），mass=0.9021 ≈ minmax_mono(0.9092)，build 快 5×
3. kmeans 在相近 C/N 下质量更高（K=512, C/N=3.1%, mass=0.9255），但**无法 overlap**

### 4.6 4bit（token 级 min/max 量化精筛）

**实现**：`tia_indexer.py` `min_max_per_token_quant()`

| 阶段 | 算子 | TC 友好？ | 同步点 |
|------|------|-----------|--------|
| quant | `amax/amin + round/clamp` | ❌ | 0 |
| score | `einsum` | ✅ GEMM | 0 |
| topk | `topk + scatter` | ❌ | 1 |

**Overlap 评级：★★★★★（双阶段完美）**

- quant 是逐 token 独立 elementwise，完美适合 CPU 预计算
- L2 score 是纯 GEMM，TC 友好
- **4bit 本身不是独立 method**，是 minmax/avg 的 L2 精筛组件

### 4.7 TWI（Twilight，cumsum top-p 选择）

**实现**：`twi_indexer.py`

| 阶段 | 算子 | TC 友好？ | 同步点 |
|------|------|-----------|--------|
| prepare_index | 同 TIA | ❌ | 0 |
| compute_score | 同 TIA | ✅ | 0 |
| compute_mask: sort | `sort` 全序列 | ❌ | 1 |
| compute_mask: cumsum | `cumsum` | ❌ | 1 |
| compute_mask: topp | `scatter` | ❌ | 1 |

**Overlap 评级：★★★☆☆（中）**

- sort 是 O(T log T)，比 topk 更重
- cumsum 引入额外同步
- 但 prepare_index 部分仍可 overlap

---

## 5. Method 组合的 Overlap 矩阵

基于 E64j 的五种组合定义（far_method × near_method）：

| 组合 | far 粗筛 | near 粗筛 | L2 精筛 | Prefill Overlap | Decode Overlap | 瓶颈 |
|------|----------|-----------|---------|----------------|---------------|------|
| **mminmax** | minmax | minmax | 4bit | ★★★★★ | ★★★★★ | topk |
| **mavg** | minmax | avg | 4bit | ★★★★★ | ★★★★★ | topk |
| **cavg** | cluster(greedy) | avg | 4bit | ★★★★☆ | ★★★★★ | prefill build |
| **ccluster** | cluster | cluster | cluster | ★★★★☆ | ★★★★★ | prefill build |
| **aavg** | avg | avg | 4bit | ★★★★★ | ★★★★★ | topk |

---

## 6. 统一 Overlap 方案

### 6.1 方案 A：CPU 异步（推荐 minmax/avg/4bit）

```
┌─────────────────────────────────────────────────────────┐
│ Prefill 阶段                                             │
│  GPU: [L0 dense] → [L1 dense] → [L2 dense] → ...        │
│  CPU:    ↓           ↓           ↓                       │
│       [build L0]  [build L1]  [build L2]  ← ~3ms/层     │
│       ↑ 完全隐藏在下一层 prefill 内                        │
├─────────────────────────────────────────────────────────┤
│ Decode 阶段                                              │
│  GPU: [L0 attn] → [L1 attn] → [L2 attn] → ...           │
│  CPU:    ↓          ↓          ↓                        │
│       [L1 update] [L2 update] [L3 update] ← ~2μs/层     │
│       ↑ 完全隐藏在下一层 Attention 内                      │
└─────────────────────────────────────────────────────────┘
```

**适用**：minmax/avg/4bit（build/update 时间 << Attention 时间）

### 6.2 方案 B：C++ 异步 build + CPU 增量（推荐 sim_greedy）

```
┌─────────────────────────────────────────────────────────┐
│ Prefill 阶段                                             │
│  GPU: [L0 dense] → [L1 dense] → [L2 dense] → ...        │
│  CPU:    ↓           ↓           ↓                       │
│       [build L0]  [build L1]  [build L2]  ← ~50ms/层    │
│       ↑ C++ 实现，与下一层 prefill 并行                    │
│       prefill 完成后，索引已 build 到 L35                  │
├─────────────────────────────────────────────────────────┤
│ Decode 阶段                                              │
│  GPU: [L0 attn] → [L1 attn] → [L2 attn] → ...           │
│  CPU:    ↓          ↓          ↓                        │
│       [L1 update] [L2 update] [L3 update] ← ~128μs/层   │
│       ↑ 完全隐藏在下一层 Attention 内                      │
└─────────────────────────────────────────────────────────┘
```

**适用**：sim_greedy（C++ build ~50ms/层 ≈ prefill 速度；decode update ~128μs << 200μs）

### 6.3 方案 C：跨层双缓冲（通用）

```
Layer n Attention (GPU TC):
  ┌───────────────────────────────────┐
  │  Q·K^T GEMM + softmax + Q·V^T     │  ← ~200μs (decode) / ~100ms (prefill)
  └───────────────────────────────────┘
Layer n+1 indexer (CPU, 并行):
  ┌───────────────────────────────────┐
  │  增量更新/全量 build                │  ← ~128μs (decode) / ~50ms (prefill)
  └───────────────────────────────────┘
  ↑ 完全隐藏在 Layer n Attention 内
```

**适用**：所有 method 除 k-means 外

---

## 7. 结论与推荐

### 7.1 双阶段 Overlap 可行性排序

| 排名 | Method | Prefill | Decode | 综合 | 核心优势 | 核心劣势 |
|------|--------|---------|--------|------|---------|---------|
| 1 | **minmax/4bit** | ★★★★★ | ★★★★★ | ★★★★★ | build/update 极快，完美隐藏 | 精度中等 |
| 2 | **avg (mavg/aavg)** | ★★★★★ | ★★★★★ | ★★★★★ | 与 minmax 同级 | 同上 |
| 3 | **sim_greedy** | ★★★★☆ | ★★★★★ | ★★★★☆ | decode 完美，prefill 需 C++ 优化 | prefill build 较慢 |
| 4 | **Quest** | ★★★★★ | ★★★★★ | ★★★★★ | 最简单 | 精度最低 |
| 5 | **TWI** | ★★★★☆ | ★★★☆☆ | ★★★☆☆ | sort+cumsum 额外开销 | top-p 比 topk 更重 |
| 6 | **k-means** | ★☆☆☆☆ | ★☆☆☆☆ | ★☆☆☆☆ | 质量最高 | **双阶段均不可行** |

### 7.2 推荐路径

1. **短期（minmax/avg/4bit）**：直接用方案 A（CPU 异步），prefill build ~3ms + decode update ~2μs，完美隐藏
2. **中期（sim_greedy）**：实现 C++ build 模块（~50ms/层），方案 B，decode 增量更新 ~128μs
3. **长期（k-means）**：放弃——迭代特性使其在双阶段均无法 overlap，质量优势被延迟抵消

### 7.3 k-means 的不可替代性分析

k-means 在 E108 中质量最高（K=1024, mass=0.9301 vs sim_greedy 0.9092），但它**双阶段均无法 overlap**：

| 阶段 | k-means 问题 | 后果 |
|------|------------|------|
| Prefill | 每层 ~200ms × 36 层 = 7.2s > prefill 总时间 | 索引 build 比 prefill 还慢 |
| Decode step 1 | 全量 build 61s（实测） | 第一个 token 延迟 61s，不可用 |

**最终判决**：
- 追求**极致速度** → minmax/avg/4bit（双阶段完美 overlap，精度中等）
- 追求**精度+速度平衡** → sim_greedy（decode 完美，prefill 需 C++ 优化，精度 ≥ minmax）
- 追求**极致精度** → k-means（但无 overlap，总延迟高，仅适合离线场景）

---

## 8. 附录：opt4 设计提案 vs cluster_sim_greedy（E108 探针）

> 用户 2026-10-06 提出的 opt4 设计（cluster_sim_greedy 的改进版），与 E108 探针中已实现的 sim_greedy 的对比。

### 8.1 核心结论：opt4 是 sim_greedy 的改进提案，尚未实现

代码库中 `probe_e108_sim_greedy.py` 已实现 `greedy_cluster_assign`（逐 token 增量聚类），但 opt4 设计的 `ClusterState.extend`（chunk 级）、`triton_medoid_mqa_logits`（Triton kernel）、leader 选举、`_compute_cap`（定长截断）**均未实现**。

### 8.2 opt4 vs cluster_sim_greedy（E108 探针）逐阶段对比

#### 阶段 1：增量聚类

| 特性 | opt4 设计 | cluster_sim_greedy（E108） | 差异分析 |
|------|----------|--------------------------|---------|
| **步进粒度** | chunk 2048 token 批量处理 | 逐 token 单步 | opt4 更粗粒度，减少循环开销 |
| **聚类维度** | nope 子维（屏蔽 RoPE） | tail32 子空间（低频尾维 32） | opt4 保留 RoPE 用于打分，sim_greedy 打分也在子空间 |
| **归入已有簇** | 与所有已有簇心算余弦 ≥0.9 → 归入最相似 | 相同（余弦 ≥ sim → 归入最相似） | 相同 |
| **新开簇** | leader 选举（未匹配 token 间余弦，无相似前驱者为 leader） | 直接新开簇（无 leader 选举） | opt4 更精确，但增加 O(n²) 开销 |
| **非 leader 吸附** | 吸附到最近 leader（余弦 ≥0.9） | 无（直接新开） | opt4 减少簇数量 |
| **强制兜底** | 达容量上界 → 强制归入全局最近簇 | K_MAX 上限 → 强制归入最优簇 | 相同思路 |
| **簇心累加** | `centers_raw_sum`（全维 128） | `sums`（tail32 子空间 32 维） | opt4 全维，sim_greedy 子空间 |
| **执行设备** | 全程 GPU 向量运算，无 CPU 同步 | GPU（PyTorch），但有 Python 循环 | opt4 更彻底 GPU 化 |

#### 阶段 2：打分

| 特性 | opt4 设计 | cluster_sim_greedy（E108） | 差异分析 |
|------|----------|--------------------------|---------|
| **打分 kernel** | `triton_medoid_mqa_logits`（Triton） | PyTorch `einsum` | opt4 用 Triton 加速 |
| **簇心维度** | 全维 128（`centers_raw_sum / cnt`） | tail32 子空间（`sums / cnt`） | opt4 信息更完整 |
| **query 维度** | 全维 128（保留 RoPE） | tail32 子空间（屏蔽 RoPE） | opt4 保留位置信息 |
| **打分函数** | `Σ_h w_h · relu(q_h · centroid_g)` | `qsub · cent`（点积） | opt4 有 relu + per-head 权重 |
| **RoPE 处理** | 聚类时屏蔽，打分时保留 | 聚类+打分都在 tail32（无 RoPE） | opt4 打分更精确 |

#### 阶段 3：选簇

| 特性 | opt4 设计 | cluster_sim_greedy（E108） | 差异分析 |
|------|----------|--------------------------|---------|
| **可见性** | 因果可见簇大小 M[i,g]（`searchsorted`） | 无（全 mid 可见） | opt4 考虑 causal |
| **排序键** | `rel = where(M>0, s_g, -inf)` | 直接 `cs.gather(assign)` | opt4 排除不可见簇 |
| **预算控制** | 定长 token 预算，累加 M 直到用尽 | rep/whole 两种模式 topk | opt4 定长 = FA3 行数恒定 |
| **末簇截断** | `_compute_cap`（cap = min(M, budget - cum)） | 无（topk 自然截断） | opt4 保证精确预算 |
| **输出** | 定长 selected keys → grouped-GEMM | 不定长 token indices | opt4 利于 kernel 融合 |

### 8.3 关键差异总结

| 维度 | opt4 | cluster_sim_greedy | 影响 |
|------|------|-------------------|------|
| **聚类粒度** | chunk 2048 | 逐 token | opt4 CPU 预计算更高效 |
| **聚类维度** | nope（屏蔽 RoPE） | tail32（低频尾维） | 子空间选择不同 |
| **打分维度** | 全维 128（保留 RoPE） | tail32（32 维） | opt4 打分更精确但更重 |
| **Leader 选举** | 有 | 无 | opt4 簇质量更高但增加开销 |
| **打分 kernel** | Triton | PyTorch einsum | opt4 更快，TC 友好 |
| **选簇** | 定长 + causal + _compute_cap | rep/whole topk | opt4 保证 FA3 行数恒定 |
| **Prefill** | 用 opt4 | 离线 build（未 e2e） | opt4 统一双阶段 |
| **Decode** | 用 opt4 | 未 e2e | opt4 统一双阶段 |

### 8.4 opt4 的 Overlap 潜力（相比 sim_greedy）

| 方面 | opt4 改进 | Overlap 影响 |
|------|----------|------------|
| **chunk 级增量** | 2048 token 批量处理，减少循环开销 | CPU 预计算更高效，prefill 更易隐藏 |
| **Triton kernel** | `triton_medoid_mqa_logits` 替代 PyTorch einsum | 更接近 TC 峰值，打分阶段更快 |
| **定长选簇** | `_compute_cap` 保证 FA3 行数恒定 | 利于 kernel 融合，减少 launch 开销 |
| **Leader 选举** | 减少簇数量，降低 K | 减少打分阶段 GEMM 大小 |
| **全维打分** | 128 维 vs 32 维 | GEMM 更大但 Triton 优化可能抵消 |

### 8.5 待验证问题

1. **leader 选举开销**：未匹配 token 间 O(n²) 余弦计算在 chunk 2048 内的实际耗时
2. **全维打分成本**：128 维 GEMM 是否被 Triton kernel 优化抵消（vs 32 维）
3. **定长截断正确性**：`_compute_cap` 在 causal 场景下的边界条件
4. **chunk 级 vs 逐 token**：chunk 内 leader 选举的精度损失是否可接受
5. **prefill 集成**：opt4 要求 prefill 也 sparse，与现有 dense prefill 的兼容性
