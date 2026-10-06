# Two-Level Indexer 设计报告：批判性可行性分析 + 代码级设计

> 基于开题 proposal（Subspace- & Distance-Aware Two-Level Indexer，21 页 PPT）+ 相关工作代码核实 + 本机实测。
> 撰写：2026-09-23。本报告遵循「不要全信 proposal，用代码和实际数据说话」的原则。
> 配套调研：`INDEXER_RESEARCH.md`（DSA 总调研）。

---

## 0. 关键背景事实（先于一切分析）

对 proposal 做批判前，先确立四个**代码核实过的事实**，它们改变了整个课题的定位：

### 事实 1：导师仓库已有第一代 two-level indexer 的完整实现和真实数据

`/home/wangyuanshuo02/two-level-attention/`（git.sankuai.com/~huyuxuan09/two-level-attention）：

- **TIA**（`sparse_attn/indexer/tia_indexer.py`）：
  - Level-1：block 内 `k_min/k_max` 区间算术上界 `q⁺·k_max + q⁻·k_min`，GQA 组内头平均后 block-level topk；
  - Level-2：**4-bit per-token 量化的部分维度 k**（`delta = 64 // cmp_ratio`，取每 64 维半段的尾部 delta 维）上算 token 级分数，softmax 后 group-mean 再 topk；
  - sliding window 128 强制包含。
- **TWI**（`twi_indexer.py`）：Level-2 换成 **top-p 自适应预算**（cumsum ≤ topp），无固定 topk。
- **TileLang kernel**（`tls_attn/ops/mha_indexer_level1.py / level2.py`）：
  - Level-1 fwd：`Q_pos/Q_neg` 两个 fragment 分别与 `K_max/K_min` 做 GEMM，per-block 归约 → score；
  - topk kernel：共享内存 **bitonic 全排序**（`block_TK2` 元素，每 KV head 一个 CTA）；
  - Level-2 fwd：按 Level-1 的 block_indices **gather** K 块 → GEMM → online softmax 变体（cummax rescale）；
  - Level-2 topk：同样 bitonic，滑窗块直接置 +inf。
- **真实 LongBench 数据**（`exp/results_longbench/`，Qwen3-8B/14B/32B @budget 1024）：
  - Qwen3-32B 上 TIA 全面优于 Quest（triviaqa 70.78 vs **46.75**，musique 35.71 vs 29.76），平均接近 FullKV；
  - TWI（top-p 0.95，平均预算 ~1000–1800）与 TIA 同精度档。

**含义：proposal 不是从零开题，而是 TIA/TWI 的下一代设计。三个创新点都必须相对 TIA/TWI（而非相对 DSA）讲清增量。**

### 事实 2：HISA (COLM 2026) 已把「two-level」这个框架本身做掉了

HISA：block 级 coarse → token 级 refine 的层次索引，与 DSA top-k 的 IoU >99%，indexer 2×@32K / 4×@128K 加速。proposal slide 6 自己也承认「论文主线不应只是 two-level」。**novelty 必须全部压在 A/B/C 三个具体机制上。**

### 事实 3：proposal 的主实验载体是「DSA trace replay」，不依赖 671B 权重

slide 15：开发阶段 = DSA trace replay / 单层 microbench。即：拿到（或构造）indexer 的 q/k 表示，重放打分与选择过程。这使得 selector-fidelity 实验可以在单卡上完成——**本机 2×H20-3e（78 SM / cc9.0 / 139GB）足够跑 E1–E4 全部 microbench**。DeepSeek-V3.2 完整 e2e 属于「资源允许」项，可用 OpenDSA（DeepSeek-V2-Lite-16B 可训 indexer）替代。

### 事实 4：TIA 已隐式做了「子空间」，只是没点破

TIA 的 `indices = range(64-delta, 64) + range(128-delta, 128)`：Qwen3 是全维 rotate_half RoPE（head_dim 128，前后半各 64 维配对旋转，频率随维内 index 递减），**取每半的尾部 delta 维 = 低频维度**。低频维度旋转慢 → block 内值域稳定 → min/max 上界紧。这正是 proposal 创新点 A「子空间保留」的物理机制，但 proposal 的表述（RoPE 64d / noPE 64d 分割）是 **DSA/MLA 语境**（DSA indexer k_s 确实是部分 RoPE：64d noPE + 64d RoPE，non-interleaved）。**两套语境不能混用，见 §2 的批判。**

---

## 1. 总体判断（TL;DR）

| 创新点 | 一句话判定 | 最大风险 | 决策（2026-09-23 实测后） |
|---|---|---|---|
| A 子空间粗筛 | **Go（实测证实）**：低频子空间 d'=32 的 mass recall 0.729 ≈ 全维 0.732，随机维坍缩到 0.32 | 叙事依赖 DSA 语境，换 Qwen3 需改讲法 | **主线保留**；完整 TIA pipeline 覆盖率 0.997–1.000 |
| B 距离条件化混合代表 | **重定位（E4c 修正）**：E4b「kmeans 4–10× 占优」含整簇超选 bug（budget=512 实取 2769 tok）；严格预算下 km_blk 0.09–0.39 / km_tok 0.45–0.79 均无一致优势，且 far 区 TIA 4bit token 级精筛 ≈ oracle（L03 0.999） | 簇中心分数系统性低估簇内尖峰 token | **B 重定位为 far/near 分区 L2 预算**（防远端被近端高分挤出，far-heavy 层反超 TIA）；聚类降级为消融 + negative result 素材 |
| C TC×CC 异构 kernel | fused/级联方向正确，但「TC×CC」标签在本机 H20 上验证力弱 | H20 的 TC 只有 H100 的 ~15%，异构结论可能反向 | 主线改为 fused+级联 topk，TC×CC 降级为可选章节 |
| D' 层自适应级联跳过（**本报告新增**） | **Go（实测证实）**：跨 prompt 平均轮廓预测可跳层 precision 1.00，far 质量损失 ≤0.3% | 轮廓跨任务泛化有限（同任务 corr 0.93+，跨任务 0.05–0.89） | **新增创新点**：静态层跳过掩码，13/36 层免远端检索 |
| E' 跨层复用 / 增量 topk / 在线 L1 信号（探索后裁剪） | **No-Go（实测否定）**：跨层 IoU 0.33、decode churn 22%、L1 信号与 far 质量相关 ≈0 | — | **写入论文 negative results 章节做消防护城河** |

proposal 的 Go/No-Go 设计（Recall≥99% @4× candidate、HBM↓45%、kernel ≥1.3×、packing<15%）**本身是严谨的**，值得肯定：每个创新点都带 No-Go 降级路径，且明确标注所有图为 synthetic。真正的问题是：(1) 部分数字的口径没写清（HBM↓45% 的分母）；(2) 硬件假设（A100+H100/H200）与本机现实（H20）错位；(3) 对 TIA 已有实现的增量没有显式声明——这在开题答辩里会被导师直接戳。

**实测总览（全部为 2026-09-23 本机 2×H20 + Qwen3-8B 真实权重，10 条 trace：needle32k / natural32k / 8 条真实 LongBench 样本（hotpotqa、narrativeqa、passage_retrieval_en、gov_report 各 2 条，官方模板 32K 截断），脚本与数据见 `two-level-attention/exp/trace/`）**：

| 实验 | 结论 | 关键数字 |
|---|---|---|
| E1 kernel 复跑 | 两级 kernel 加速随长度增长 | 1.46×@32K / 2.77×@64K / 5.09×@128K |
| H1 位置分解（10 trace） | **proposal 的「近端集中」叙事在 mass 口径下不成立**，真正的质量大头是 sink（前 64 token，0.37–0.71） | dense top-1024 的 mass 覆盖 = **1.0000（全部 trace）**——TIA@1024≈FullKV 的根本原因 |
| E3/E3b 子空间 | 低频 d'=32 ≈ 全维 d'=128 | mass recall 0.729 vs 0.732；完整两级 pipeline（含 4bit L2 + 滑窗）mass 覆盖 **0.9967–1.0000** |
| E4/E4b 远端代表 | ~~kmeans 中心全面占优~~ **E4c 修正：E4b 含整簇超选 bug** | E4b km256@512=0.781 实际用了 2769 tok（5.4× 超预算）；严格预算下（E4c）km_blk 0.09–0.39 最差 / km_tok 0.45–0.79 与 minmax 块级（0.52–0.74）互有胜负；far 区 4bit token 级精筛 ≈ oracle |
| E5 附加 idea 淘汰 | 跨层复用/增量 topk 均死 | 跨层 IoU 0.33；decode 步间 churn 22% |
| E6 层跳过（D'） | 离线轮廓静态跳过可行 | 13/36 层可跳，precision 1.00，漏 far 质量 ≤0.0226（<0.3%）；在线 L1 信号判定不可行（相关 ≈0） |
| E4c/E5b TLI 端到端 | TLI(A+B'+D') 质量持平或超过 TIA | 全 36 层 mass diff<0.001；far-heavy 层反超（L03 0.9995 vs 0.9990、L05 0.9997 vs 0.9929） |
| E8-1 索引开销 | D' 真正兑现计算节省（topk 截断到有效块数） | 索引 FLOP 跳层 4.88× / 非跳层 1.27× / 全层平均 **2.57×** |
| E8-2 fused kernel 原型 | Triton 单 launch L1（子空间+D'剔除+in-kernel top-K1） | S=128K：跳层 3.60× / 非跳层 1.64× vs eager；对拍块 id 完全一致 |

---

## 2. 创新点 A：Subspace-Preserving Coarse Index —— 批判性分析

### 2.1 proposal 声称什么

- DSA 的 index 表示是 RoPE 64d + noPE 64d 混在 128d cache 里；
- Stage-1 coarse 只 load 32/48/64d 子空间做打分；
- 目标：4× candidate 下 Recall@K ≥ 99%，Stage-1 HBM bytes ↓≈45%（slide 8 已自我修正为「不宣称总 cache 减半，只减 Stage-1 traffic」——这个修正是对的，见 2.3）。

### 2.2 技术机制分析（为什么可能成立）

**上界紧致性论证（proposal 没写透的加分点）**：TIA/HISA 式区间算术 coarse 分数的 gap：

```
gap(block) = Σ_d |q_d| · (k_max_d − k_min_d)
```

维度 d′ 越少，gap 累加项越少 → 上界越紧 → coarse 排序与 exact 排序的 rank 相关性越高。**子空间裁剪不仅省带宽，还可能直接提升 Level-1 recall**。这是创新点 A 最值得写进论文的 mechanism claim，且用 trace replay 一天能验证。

**位置稳定性论证**：noPE 子空间（DSA）或低频维度（Qwen3 rotate_half 尾部 delta 维）的值跨位置稳定，block 聚合（min/max 或均值）不引入大的表示误差；高频/RoPE 维度的块聚合几乎无意义（TIA 取尾维正是这个原因）。

### 2.3 批判点

1. **语境错配风险（最大）**：proposal 的 RoPE/noPE 分割叙事只在 DSA（部分 RoPE，non-interleaved）下成立。若实验主体换成 Qwen3（全维 rotate_half），分割对象变成「高频/低频维度」，公式不同、motivation 要重写。**决策**：selector-fidelity 实验按 proposal 用 DSA trace replay（q/k 表示可用官方 `/tmp/papers/DeepSeek-V3.2/inference/model.py` 的 indexer 前向 + 真实 hidden state 构造）；LongBench 精度实验用 Qwen3（导师 pipeline 现成），叙事上把 A 表述为「position-stable subspace」（涵盖 noPE 与低频两种实例化）。两套实验共用同一个抽象：`d' 维子空间投影矩阵 P`。
2. **HBM↓45% 的分母问题**：Stage-1 的 HBM 流量 = coarse index 读取 + q 投影 + score buffer 写。以 128K/bs=64/H_I=64/FP8 为例：coarse index = 2048 blocks × 64 heads × 128d × 1B ≈ 16.8MB/层。d' 64→32 只砍 index 读取这一项。若 score buffer（fp32，2048×64）与 q（64×128×1B）占比可观，总流量降不到 45%。**必须先算流量模型再承诺数字**。测量口径写死为：`Stage-1 kernel 的 DRAM bytes（ncu 实测）`。
3. **worse-layer P5 recall**：proposal 自己要求 worst-layer 不崩塌——粗筛的误差在不同层分布不均（部分层 top-k 更分散），子空间裁剪可能恰好伤到 worst layer。这是真实的，处理方式是把 P5 而非 avg 写进 Go/No-Go（proposal 已这么做，好评）。

### 2.4 代码级设计

**算法线（导师仓库，新增 `sparse_attn/indexer/sdai_indexer.py`）**：

```python
class SDAIIndexer(Indexer):
    """Subspace & Distance-Aware Indexer (proposal 创新点 A+B 的算法载体)"""
    def __init__(self, args):
        self.subspace_dim = args.sdai_coarse_dim      # d' ∈ {32,48,64,96,128} 扫描
        self.subspace_idx = build_subspace_index(     # 统一抽象两种语境
            layout=args.rope_layout,                  # 'dsa_partial' | 'qwen_rothalf'
            d=self.dim_k, d_prime=self.subspace_dim)
        self.near_blocks = args.sdai_near_blocks      # 创新点 B：近端块数
        self.far_clusters = args.sdai_far_clusters    # 创新点 B：远端聚类数 K_c

    def prepare_index(self, k, cu_seqlens_k):
        # k: [1, S, H_kv, 128] (GQA，RoPE 已施加后)
        # A: 子空间粗筛索引 —— 只对 subspace_idx 维度算 block min/max
        k_sub = k[..., self.subspace_idx]             # [1,S,H,d']
        k_coarse = rearrange(pad(k_sub), 'b (t bs) h d -> b t bs h d', bs=block)
        return {"k_min": k_coarse.amin(2), "k_max": k_coarse.amax(2),  # d' 维
                # B: 远端聚类中心（见 §3.4）
                "far_centroids": ..., "far_assign": ...}
```

`compute_score` 与 TIA 完全同构（`q⁺·k_max + q⁻·k_min`），只是 einsum 的 d 维换成 d'。`compute_mask` 的 Level-1 topk 不变。**相对 TIA 的 diff 只有 prepare_index 里的维度选择**——实现量极小，扫描 d' 的一维实验一晚上能跑完。

**系统线（sglang，见 §5）**：粗筛 kernel 的 `dim_k` 参数从 128 换成 d'（TileLang kernel 的 `dim_k` 本来就是编译期常量，直接换即可）；k_min/k_max pool 的分配尺寸同步改。

### 2.5 实测验证（E3 / E3b，2026-09-23 完成，10 条真实 trace）

**子空间维度扫描（`analyze_e3.py`，needle32k + natural32k，块 32/64，d' ∈ {32,64,96,128}，模式 lowfreq/random/highfreq）**：

| 模式（blk=32, d'=32） | mass recall | entry recall | 说明 |
|---|---|---|---|
| lowfreq（每半尾 16 维） | **0.729** | 0.271 | ≈ 全维 d'=128 的 0.732 |
| random（随机 32 维） | 0.320 | 0.149 | 崩溃 |
| highfreq（每半头 16 维） | 0.137 | 0.050 | 彻底崩溃 |
| 全维 d'=128（参照） | 0.732 | 0.303 | 任何模式 d'=128 等价 |

**机制结论（可写进论文的 claim）**：(1) 块聚合打分的判别力几乎全部来自低频尾维——d'=32 lowfreq 与 d'=128 的 mass recall 差距 <0.5%；(2) 上界紧致性论证得到间接支持——d' 减半 entry recall 也不降（0.271 vs 0.303，gap 变紧的效应抵消了信息减少）；(3) 随机/高频维坍缩证明这不是「任意 32 维都行」，position-stable subspace 是唯一正确的抽象。

**完整两级 pipeline 端到端覆盖率（`analyze_e3b.py`，全部 10 条 trace，TIA 原语义：L1 块上界 d'=32 → K1=128 块 → L2 4bit 部分维 delta=16×2 → K2=1024 + 滑窗 128 强制）**：

| trace | dense top-1024 覆盖 | 两级 pipeline 覆盖 | 保持率 |
|---|---|---|---|
| needle32k / lb_hotpotqa×2 | 1.0000 | **1.0000 / 0.9998** | ≈100% |
| lb_gov_report×2 | 1.0000 | 0.9979 / 0.9998 | >99.8% |
| lb_narrativeqa×2 | 1.0000 | 0.9971 / 0.9971 | 99.7% |
| lb_passage_retrieval×2 | 1.0000 | 0.9982 / 0.9967 | >99.7% |
| natural32k | 1.0000 | 0.9969 | 99.7% |

**解释了导师 LongBench 数据里 TIA@1024 ≈ FullKV 的根本原因**：dense top-1024 在 mass 口径下覆盖率本来就是 1.0000（全部 10 条 trace，无一例外）——质量集中在极少数高分数 token 上，entry-level 的 recall 数字（0.27–0.50）会严重低估实际精度。**论文必须用 mass coverage 作为主指标、entry recall 作为辅助**，否则审稿人会问「为什么 recall 只有 0.3 精度却不掉」。

---

## 3. 创新点 B：Local Avg + Remote Cluster Hybrid —— 批判性分析

### 3.1 proposal 声称什么

近处 token 用小块摘要，远处用语义聚类中心，自适应预算 `C = C_near + C_far`，Stage-2 统一 token 级 exact refine。proposal 自己标注了 prior-art 风险（ClusterKV/NSA/Double-P）。

### 3.2 批判点

1. **与 ClusterKV 的边界必须写死**：ClusterKV 是「k-means 聚类后整簇取/丢」——聚类即选择，丢 token 直接影响精度。本设计聚类中心只是**远端粗筛代表**，最终选择由 Stage-2 exact token refine 决定。精度上界由 refine 保证，聚类只影响「远端关键 token 是否进入候选」的 recall。这个区分成立，但论文里必须显式对比 ClusterKV 的 recall-latency 曲线（正好接上 sparse-bench 的 KVCache-Factory 计划——ClusterKV 是我们本来就缺的方法）。
2. **聚类的 build/update 是全案最大的工程风险**：128K 上下文、8 KV head、128d，远端 ~120K token。prefill 后跑 GPU KMeans（50 iter GEMM，~200 GFLOP/层）单层毫秒级，但 ×36 层 ×每请求在线执行不可接受。proposal slide 14 的方案是 lazy maintenance（增量 assign + 惰性 merge），**这是没有验证过的核心假设**。备选：只在 prefill 做一次聚类 + chunk 追加 token 的增量 assign（新 token 找最近中心，中心滚动更新），解码期不重聚类。Go/No-Go 实验必须单独测 build/update overhead（proposal 也这么要求了，好评）。
3. **H1 是 B 的前置条件**：近/远异构表示的收益取决于 top-k 距离分布是否双峰（近邻集中 + 少量远邻关键 token）。若分布平坦（比如 needle 型任务关键 token 均匀散布），近端预算分配就是浪费。H1 实验必须用 LongBench/RULER 真实 trace（而非合成高斯）统计 dense top-k 距离直方图——**本机用 Qwen3-8B（beegfs 权重已定位：`/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B`）+ 导师 pred.py 的 patch 机制 hook compute_mask 即可采集**。
4. **聚类代表分数的一个理论坑**：聚类中心的 q·c 是簇内 token 的均值近似，不是上界。远端某个关键 token 的分数可能远高于其中心 → 漏选。TIA 的 min/max 是严格上界（不会漏，只会多选）。**混合设计里远端用聚类中心替代 min/max 是用「更紧的均值」换「失去上界保证」**——安全做法是远端分数 = max(中心分数 − margin, 块内 max 近似)，margin 由 trace 校准。这一点 proposal 完全没讨论，是答辩时最容易被问穿的洞。

### 3.3 判定

算法 novelty 三者最高（distance-conditioned representative + adaptive budget + exact refine 的组合确实没人做过 DSA-specific 版本）。原依赖的两个未验证假设已实测解决：H1 双峰性 → 修正为 sink/near/far 三层分解（§4A.1，真实形态对 B 更有利——far 质量虽然总量小但集中在少数层，混合代表的用武之地明确）；lazy 聚类开销 → E4 全量 build 540ms/层，必须走 prefill 一次 + 增量 assign 路线（E7 待测分摊开销）。**决策（2026-09-23）：Go，B 保持主线**。降级路径不变：退化为辅助优化（近端 summary 保留、远端回退 min/max 块界）。

### 3.4 代码级设计（算法线）

```python
# SDAIIndexer 内，创新点 B 部分
def prepare_index(self, k, cu_seqlens_k):
    S = k.shape[1]
    near_len = min(S, self.near_blocks * block)      # 近端：小块（block=16）滚动 min/max
    far_k = k[:, S - near_len:, ...]                  # 近端部分
    far_k = k[:, :S - near_len, ...]                  # 远端部分
    # 远端聚类：prefill 后一次 + 增量 assign
    # centroids: [K_c, H, d'], assign: [T_far] → cluster id
    centroids = self._kmeans(far_k[..., self.subspace_idx], K_c=self.far_clusters)
    return {"near_min": ..., "near_max": ...,          # 近端块级 min/max（同 TIA）
            "far_centroids": centroids, "far_assign": assign}

def _kmeans(self, x, K_c, niter=20):
    # GPU mini-batch kmeans：GEMM 距离 + argmin + bincount segment-sum 更新
    # （直接复用 KMeans/CPU 调研的 GEMM 技巧，cuda 化）
    ...

def compute_score(self, q, q_ids, index_dict, softmax_scale):
    score_near = q⁺·near_max + q⁻·near_min           # 上界，不漏
    score_far  = q @ far_centroids                    # 均值近似（紧但有漏的风险）
    # 自适应预算：按两类分数分布的相对质量分配 C_near / C_far
    # 简版：固定比例；进阶：按 top 分数和的占比 softmax 分配
```

增量更新（decode 每步新 token）：近端块滚动更新 O(1)；远端新 token `assign = argmin ‖k_new − c_j‖`（一次 GEMV），中心做 EMA 滚动更新；**不做全局重聚类**。`clear()` 时释放。

### 3.5 实测验证（E4 / E4b，2026-09-23 完成）——**Go 决策依据**

远端 = [64, t−2048)，gt = dense top-1024 中的远端 token，预算 C_far tokens/head（`analyze_e4b.py`，Pareto 扫描）：

**needle32k（C_far=512 时差距最大档）**：

| 策略 | recall@512 | recall@2048 | mass@512 | mass@2048 |
|---|---|---|---|---|
| block mean | 0.071 | 0.187 | 0.029 | 0.157 |
| block minmax（TIA） | 0.031 | 0.141 | 0.616 | 0.900 |
| kmeans K_c=256 | **0.451** | 0.700 | 0.924 | 0.948 |
| kmeans K_c=1024 | 0.437 | 0.710 | **0.935** | 0.958 |
| exact oracle | 0.715 | 1.000 | 0.948 | 0.963 |

**真实 LongBench（lb_passage_retrieval_en_1，代表性 retrieval 型任务）**：km256 recall@512=0.548 vs minmax 0.111（**4.9×**）；km1024 mass@512=0.875 vs minmax 0.409；km256 mass@1024=0.862 vs minmax 0.529。

**结论**：
1. **kmeans 中心在远端候选质量上全面占优**——recall 达 minmax 的 4–10×，mass 也持平或更高（低预算档大幅领先）。minmax 的 recall 崩溃原因正是 §3.2 第 4 点：区间上界系统性偏向「块内方差大」的块，排序能力差——实测直接证实了这个理论预判，也说明「非上界」的理论空洞在实践中没有造成 mass 损失（kmeans 的 mass 反而更高）。
2. **exact oracle 距离 kmeans 仍有距离**（recall 0.72 vs 0.45 @512），说明聚类粒度是真实的精度瓶颈，K_c 与增量维护值得继续投入。
3. **build 开销 ~540ms/层（36 层全量在线执行不可接受）**——必须按 §3.4 的「prefill 一次 + 增量 assign」落地；D'（§4A 层跳过）可以再砍掉 ~36% 层的聚类开销，两个创新点形成组合。
4. 10 条 trace 的 Pareto 曲线形状一致（`results/e4b_pareto.json`），结论跨任务稳健。

### 3.6 增量维护实测（E7，2026-09-23 完成）——工程风险排除

`analyze_e7.py`：模拟「prefill 时刻只用 far 前半 token 全量聚类（K_c=256），far 增长一倍后不重聚类、新 token 增量 assign 到旧中心」，在最终位置对比 fresh（全量聚类）：

| trace | fresh recall@512 | stale recall@512 | fresh mass@1024 | stale mass@1024 |
|---|---|---|---|---|
| lb_hotpotqa_1 | 0.490 | 0.468（−0.02） | 0.843 | 0.879（**+0.04**） |
| lb_passage_retrieval_en_0 | 0.567 | 0.559（−0.01） | 0.813 | 0.871（**+0.06**） |
| needle32k | 0.453 | 0.436（−0.02） | 0.928 | 0.920（−0.01） |
| natural32k | 0.438 | 0.421（−0.02） | 0.650 | 0.625（−0.03） |

**结论**：上下文翻倍不重聚类，recall 衰减仅 1–3 个点，mass 持平或**反而上升**（旧中心覆盖半径更大→同预算纳入更多远端 token）；增量 assign 是单次 GEMV，开销 <1μs/token/head。**B 的 lazy 维护假设（proposal slide 14，未验证）首次得到实测支持**——540ms/层的全量 build 只需在 prefill 做一次，decode 期成本可忽略。

### 3.7 E4c 严格预算修正（2026-09-23 晚）——**B 创新点重定位的关键依据**

E5b 端到端实现中发现 E4b 的 budget 语义有 bug：整簇贪心展开时 `taken` 检查在取簇**之前**，最后一簇整簇纳入后实际 token 数远超预算（实测 hotpotqa L03：budget=512 实取 **2769 tok，5.4× 超选**）。同时 E4b 用 head 平均分布 pm 评估所有 head 的候选，而 GQA 下 far mass 跨 kv-head 极度不均（L03 实测 [0.003, 0.306, 0.001, **0.804**, 0, 0.001, 0.004, 0.117]，head3 独占 0.804）——平均口径进一步虚高。

`analyze_e4c.py`（e4c_strict_budget.json）以**严格 token 预算 + per-head 加权口径**重测（10 trace × 36 层平均，far 捕获）：

| 策略 @512 / @1024 | 捕获范围 | 判定 |
|---|---|---|
| minmax 块级（TIA L1 语义） | 0.52–0.74 / 0.66–0.85 | 真实基线（块级无超选） |
| **km_blk**（B 原设计：块分数=簇分 scatter-amax） | **0.09–0.39 / 0.24–0.44** | **最差**——块内 1 个高分簇成员拉整块，63 个无关 token 稀释预算 |
| km_tok（簇分数 token 级 topk） | 0.45–0.79 / 0.54–0.84 | 互有胜负：needle +0.24、hotpotqa +0.05，narrativeqa −0.13、gov_report −0.02 |
| oracle（真实 q·k topk） | ≈1.0 | — |

**决定性发现**：TIA 的 L2 是全序列 token 级 4bit 精筛，far 区质量 ≈ oracle（L03 mass 0.999）——**用任何簇分数替换 4bit token 级精筛都是净损失**。

**B 重定位**（`tli_indexer.py` 实现）：
- far 区保留 TIA 4bit token 级精筛；B 的贡献 = **far/near 分区 L2 预算**（far 池独立 top-K2_far，防远端被近端高分挤出）
- 效果：TLI(A+B'+D') 全 36 层 mass diff<0.001，且 far-heavy 层反超 TIA（L03 0.9995 vs 0.9990、L05 0.9997 vs 0.9929——挤出现象真实存在，分区有效）
- 聚类路径保留为 `--tli_far_select cluster` 消融 + 论文 negative results 素材（「budget 口径必须严格 token 级」本身是方法论贡献）
- E7 的增量维护结论不受影响（其对象是聚类数据结构本身），但不再进入主线推理路径

### 3.8 E8 组合收益实测（2026-09-23 晚）

**E8-1 索引开销模型**（`analyze_e8_1.py`，360 层-trace 样本）：

| 维度 | 数值 |
|---|---|
| TIA 平均 L2 候选 | 8320 tok/step |
| TLI 平均 L2 候选（D' topk 截断到有效块数） | 6148 tok/step（跳层仅 ~2240） |
| 索引 FLOP 加速比 | 跳层 **4.88×** / 非跳层 1.27×（A 子空间贡献）/ 全层平均 **2.57×** |

注：D' 的省算依赖「topk 截断到有效块数」实现（`tli_indexer.py` compute_mask）——否则 topk 会用 -inf far 块填满 K1，far token 照样进 L2 白算。attention 主计算量不变（K2 固定），节省的是索引器侧。

**E8-2 fused 级联 topk kernel 原型**（`e8_2_fused_topk.py`，Triton，合成数据 kernel 级 microbench，规模对齐真实口径 S=128K/nblk=2048/Hkv=8/d'=32/K1=128）：

单 launch L1 kernel（子空间区间算术 + D' 块剔除 + 当前块强制 + in-kernel 阈值二分 top-K1）：
- 跳层场景（far 全剔除）：**3.60×** vs eager PyTorch（0.034 vs 0.123 ms）
- 非跳层场景：**1.64×**（0.075 ms）
- 正确性：跳层场景块 id 34/34 完全一致；非跳层分数和差 0.36%（阈值二分的并列截断，真 fused 版用 warp 级 bitonic 精确截断）

**E8-2 下半场：L2 级联 fused kernel**（`e8_2_l2_cascade.py`，同规模合成数据）：
单 launch/head：候选 token（选中块展开 ~8384 个）的 4bit 精筛分数全驻寄存器 → far/near 两池各做阈值二分 topk → scatter 写 token mask。
- 正确性：与 eager 分区语义对拍 **4096/4096**（仅 5 个阈值并列多选，交 L2 后不影响语义）
- 延迟：eager 1.897 ms → triton 1.165 ms = **1.63×**
- 完整两级 = L1 kernel + 小 compaction（2048 元素）+ L2 kernel，共 2 launches（vs eager ~15 kernels）

### 3.9 E59 双消融（2026-09-26，8B/30B trace 重放，回应两个设计问题）

**消融一：L1 块表示 minmax vs avg**（`analyze_e59_l1_ablation.py`，K1=128 块、far top-1024 mass recall 口径，8B 16 条 trace + 30B hotpotqa）：

| 块表示 | 8B 均值 | 8B 最差 | 30B hotpotqa |
|---|---|---|---|
| minmax 上界（现行） | **0.885** | 0.781 (narrativeqa) | **0.967** |
| avg 均值点积 | 0.695 | 0.513 (narrativeqa) | 0.948 |

minmax 全面胜出（8B 平均 +0.19，最差样本 +0.27）：avg 是块内均值的非保守估计，块内符号对消使远端高响应块被系统性低估——这正是 TIA 采用 minmax 上界的理论依据（上界粗筛在「漏选」方向上错误更少）。**结论：现行 minmax 设计实证最优，avg 记入消融表。**

**消融二：near_len 配比敏感性**（`analyze_e59_near_alpha.py`，near_len ∈ {1024, 2048, 4096}，far 区 mass 与 far 区 top-256 覆盖率）：

| near_len | 8B far_mass 均值 | 8B far top-256 覆盖 | 30B far_mass |
|---|---|---|---|
| 1024 | 0.364 | 0.758 | 0.192 |
| 2048（现行） | 0.330 | 0.772 | 0.182 |
| 4096 | 0.284 | 0.782 | 0.118 |

near_len 4 倍变化（1024→4096）far mass 仅降 0.08、far 选择质量仅升 ~2%——**边际递减明显，固定 near_len=2048 是合理工作点**。用户提议的 alpha 比例化（near 预算随总预算按比例分）实证上收益不显著且引入额外超参；更重要的是：**固定 near_len + 常数 far 预算 = 每 token 成本 O(1)（与序列长度 t 无关），这是 TLI 相对 sliding-window+自适应 budget 方法（如 TWI top-p）的结构性设计优势**——alpha 化会引入 O(t) 依赖破坏该性质，记入论文设计动机段。

---

## 4. 创新点 C：TC×CC 异构 Kernel —— 批判性分析

### 4.1 现状（导师 TileLang kernel 的实测短板）

已有 4 个独立 kernel launch（L1-fwd / L1-topk / L2-fwd / L2-topk），合成数据实测（`exp/results_efficiency/`，本机复跑见 §6 E1）：

| seqlen | dense MHA | sparse(两级) kernel | 加速 |
|---|---|---|---|
| 32K | 0.855 ms | 0.660 ms | 1.30× |
| 64K | 1.707 ms | 0.690 ms | 2.47× |
| 128K | 3.407 ms | 0.744 ms | 4.58× |

**32K 只有 1.30×**——低于 HISA 声称的 2×@32K。原因：decode 单 token 时两级 kernel 计算量小，4 次 launch + 中间 score buffer 的显存往返主导延迟。**这正是 proposal 创新点 C 要解决的真问题，方向判断正确。**

### 4.2 批判点

1. **H20 硬件特性使「TC×CC」标签的验证力打折（最大环境风险）**：H20-3e 为 78 SM（H100 的 59%），BF16 TC 算力 ≈ 148 TFLOPS（H100 SXM 的 ~15%），FP32 CC 算力 ~44 TFLOPS。**TC 与 CC 的算力比与 H100 完全不同**，在 H20 上调出的「TC lane / CC lane 配比」到 H100 上可能反向。proposal 写「A100 + H100/H200（如可用）」——本机没有。**对策**：(a) 论文 kernel 章节主线改为 **fused + 级联 topk**（下面第 2 点），这在任何硬件上都成立；(b) TC×CC 异构做成 persistent kernel / warp specialization 的可选段落，用 cost model（proposal slide 12 本来就规划了 runtime cost model）+ 本机实测标注清楚硬件；(c) 如后续拿到 H100 机器再补主表。
2. **bitonic 全排序是明显可砍的开销**：L1-topk 对 `T/bs` 个块分数做 bitonic **全排序**（O(n log²n)），但只需要 top-128。级联方案：L1-fwd kernel 内每 tile 先留局部 top-k（block-level 预筛，共享内存里小规模 bitonic 只排 tile 内元素）→ 全局合并只处理 `n_tiles × local_k` 个元素。128K 时全局排序元素数从 131072 降到 ~2048。**这本质是把「two-level」思想下沉到 kernel 内部，和创新点 A/B 形成贯穿全文的叙事线——这是设计报告最重要的写作建议。**
3. **L2 kernel 的两遍循环可以优化**：现有实现先存 cummax 再第二遍 rescale（online softmax 变体），可以直接在 fragment 里维护 running max + 单遍 rescale（FlashAttention 式），省一遍 score buffer 读写。
4. **与 Libra 的区分**（proposal 自己点名）：Libra 是通用 TC+CC 异构执行；我们的 novelty 必须落在 indexer-specific 的映射——候选 queue 的 shared-memory staging、coarse tile 与 irregular gather 的 producer-consumer 流水。写论文时用 ncu 的 TC active / SM active / stall reasons 对比图支撑（proposal slide 13 已规划，口径对）。

### 4.3 代码级设计（kernel 线，导师仓库 `tls_attn/ops/` 扩展）

```python
# 新文件 mha_indexer_fused.py
@tl.jit(...)
def mha_indexer_fused_kernel(..., local_topk: int):
    # 一个 CTA per (batch, kv_head)，persistent 循环扫所有 tile：
    with T.Kernel(batch, num_kv_heads, threads=256) as (bx, by):
        # 粗筛阶段（TC lane）：GEMM(Q_pos, K_max) + GEMM(Q_neg, K_min)
        for i_s in T.Pipelined(loop_range, num_stages=num_stages):
            T.gemm(...)                    # 子空间 d' 维（创新点 A）
            T.reduce_sum(acc_s, acc_o, dim=0, clear=True)
            tile_topk(acc_o, scores_shared, local_topk)   # 级联：tile 内局部 top-k
            merge_into_global(scores_shared, indices_shared)  # 维护全局 top-K 堆
        # 全局 bitonic 只在最后对 local_topk × n_tiles 个元素排一次
        # （与现有 mha_indexer_topk_kernel 的 bitonic_sort macro 复用）
```

L1 的 fwd+topk 融合成一个 launch；L2 的 gather+GEMM+softmax+topk 同理融合。四 launch → 两 launch。tile 的 `dim_k` 用 d'（创新点 A 直接受益：GEMM 规模缩小）。

**H20 实测预期**：launch 开销 ~5μs×2 + score buffer 往返消除，32K 下预估 1.30×→1.5×+（E1 之后可精确验证）。

---

## 4A. 新创新点 D'：层自适应级联跳过（Layer-Adaptive Cascade Skip）

> 本节为实测驱动的新增创新点（非 proposal 原有）。来源：H1 完整分解发现层间 far 质量异质性极大，且该异质性跨 prompt 可预测。

### 4A.1 实测发现（`analyze_h1_full.py` + `analyze_e6.py`，10 条 trace）

| trace | sink | near | far（均值） | far>0.02 的层 | 单层 far 最大 |
|---|---|---|---|---|---|
| lb_hotpotqa×2 | 0.69/0.71 | 0.21–0.24 | **0.016–0.018** | **4–6/36** | 0.15–0.21 |
| lb_passage_retrieval×2 | 0.68/0.69 | 0.19 | 0.021–0.027 | **5/36** | 0.25–0.29 |
| needle32k | 0.68 | 0.23 | 0.028 | 13/36 | 0.17 |
| natural32k | 0.65 | 0.16 | 0.085 | 20/36 | 0.39 |
| lb_gov_report×2 | 0.37–0.39 | 0.28 | **0.249–0.294** | 22–23/36 | 0.82–0.87 |
| lb_narrativeqa×2 | 0.43 | 0.29 | 0.214 | 24/36 | 0.70 |

两个关键观察：
1. **far 质量的层轮廓高度任务依赖**：hotpotqa 平均 far 只有 gov_report 的 1/18；同任务两个不同样本的层轮廓相关性 **0.93–1.00**，不同任务间 0.05–0.89。
2. **远端检索开销花在了大量「远端本来就没质量」的层上**：hotpotqa 32/36 层 far<0.02，但这些层的远端聚类/L2 精筛照跑不误。

### 4A.2 机制设计：离线 profile → 静态层跳过掩码

```python
# SDAIIndexer.__init__ 加载（校准阶段产出，非在线）
self.layer_far_skip: torch.BoolTensor [n_layers]   # True = 该层跳过远端检索
# 校准：N 个代表性 prompt 上跑 dense far-mass 统计（trace 收集器已有），
# 取跨 prompt 平均轮廓 avg[i]；layer_far_skip[i] = avg[i] < τ (τ=0.02)
# 推理时跳过层 = 只保留 sink(64) + 滑窗(128) + 近端(2048)，C_far 全部省掉
```

**实测效果（E6）**：用 10 条 trace 的平均轮廓（13 层 < 0.02）预测每个 prompt 的可跳层——

| 指标 | 结果 |
|---|---|
| precision | **0.92–1.00**（跳过的层确实没 far 质量） |
| 漏掉的 far 质量 | ≤0.0226（相对全层 far 总量 <0.3%） |
| recall | 0.41–1.00（保守：hotpotqa 实际可跳 30+ 层，我们只跳 13） |

### 4A.3 与在线信号方案的对比（重要的 negative result）

在线 L1 块上界信号（far 块 top-32 分数和 / 全块分数和）与真实 far 质量**几乎零相关**（相关系数 −0.33~0.13，10 条 trace），用其判定会把 gov_report 的 8.4/8.9 far 质量直接漏掉。**结论：层跳过决策必须用离线 profile，不能用在线启发式**——这本身就是论文里一个干净的消融实验。

### 4A.4 开销收益账

- 跳过层省掉：远端聚类 build（E4 实测 540ms/层的 36%）+ L1 远端块打分 + L2 远端候选精筛；
- 跳过层仍做：sink/滑窗/近端（这些是质量大头，sink 单独占 0.37–0.71）；
- 与 TWI（top-p 自适应预算）互补：TWI 管 token 预算的层内自适应，D' 管层的「远端关/不关」二值决策，正交可叠加。

### 4A.5 风险与防御

1. **分布偏移**：校准 profile 与实际 workload 任务不一致时 recall 下降（如用 hotpotqa+needle 校准、gov_report 来了只跳 13/实际需要 0 层）——**此时质量不损失**（保守方向的错误只是少省点算力）；反向错误（把 far 重要的层跳了）由 precision 1.00 + margin 阈值防护。
2. **写进论文的口径**：D' 是「calibration-based static layer scheduling」，与 Mixture-of-Attention 层异质性研究（如 NSA 的 head 选择、SBA）对话，但那些是训练后的路由，D' 是 training-free 校准——增量声明要写清楚。

### 4A.6 D' 升级：prefill 动态测层 → decode 动态跳 far（E60，2026-09-26，commit 2f9b10f03）

E5b 已证明静态全局掩码跨任务不泛化（musique/qasper/multifieldqa_en 掉 4.8-5.9 分），per-task 重校准也失败。升级方案：**不依赖校准集，在当前请求自己的 prefill 末 chunk 上测 per-layer far mass**——分布偏移问题从根上消除（测的就是推理时的真实分布）。

机制（`indexer.py`，env `SGLANG_TLI_DYN_GATE` / `SGLANG_TLI_DYN_GATE_THRESH=0.01`）：
1. `select_batched` 末 chunk（r1==Nq）对真实 fine 分数矩阵 softmax 后统计 far 区 mass（行×Hkv 平均）→ 存 `self.dyn_far_stat`；
2. decode `select()` 开头幂等置位：`skip_far = dyn_far_stat < thresh`；
3. far_stat 双峰分布（musique 18 层 <0.01 vs 其余 0.017-0.26），阈值 0.01 切在自然间隙上。

验证（`test_tli_8b_dyngate.py` / `_mh.py`，LongBench 官方模板口径）：
- **离线 corr GO**：prefill far_stat 与 decode 真实 far 质量相关 0.86-0.99；
- **安全任务零损失**（gov_report/narrativeqa，far_stat 低 → 跳层，输出与 gate-off 语义一致）；
- **多跳三任务 GO**（musique/qasper/multifieldqa_en——静态版正是在这批任务崩的）：gate-on 输出与 gate-off（= TIA 精度基线）语义等价，静态版失败模式（掉 4.8-5.9 分）未复现；
- **全量 E5b 双臂定稿（2026-09-27，`pred_dyngate_score.json`，sglang 版）**：musique/qasper/multifieldqa_en（200/200/150 样本）gate-on AVG **38.01** vs gate-off **38.18**——**dyngate 代价 −0.17（噪声级），动态 gate e2e 无损成立**；vs E5b transformers 主表 TIA 参考 43.17 的绝对差为 sglang↔transformers 推理栈口径差（同臂内部对比不受影响，`sglang_triton_dense_musique` 30.32 对照臂同性质）。已知限制：dyn_far_stat 跨请求污染（indexer per-layer 单值，最后写入者覆盖同批全部请求的 decode 决策）——同质 batch 无害，混合 batch 须迁到共享 index pool per-row（TODO）。
- **E67 升 τ 真判决（tau01/tau02 双臂，2026-09-30，`pred_e67_tau0{1,2}_score.json`）**：far-heavy 多跳双任务（musique/qasper 各 200 样本）vs gate_off 对照（27.57/40.37）完整梯度：

  | τ | 触发程度 | musique | qasper | 合计 Δ |
  |---|---|---|---|---|
  | 0.005 | 11.8% 实体级输出分歧 | 27.80（+0.23） | 39.96（−0.41） | −0.18 |
  | 0.015 | 安全/多跳边界 | 27.84（+0.27） | 39.70（−0.67） | −0.40 |

  τ0.005 已非 no-op（vs τ0.01 的 1.1%）；梯度单调（τ×3 → 损失×2.2），musique 反微升、qasper 单调掉——qasper far 总量大（0.55-0.95）是主要承受方。结论：**感知 gate 在真触发区间（τ≤0.015）对多跳任务微损不崩**，与静态版掉 4.8-5.9 分形成对照——动态 per-request 信号确实防住了反向错误，但正收益（省算力换精度）未兑现，gate 终定位 =「安全省算力的保守开关」而非精度增益点。τ 口径注：sglang dyn_far_thresh = per-layer far mass（行×Hkv 平均），E67 trace 侧 0.1/0.2 的「占比」口径不可直搬。
- **证据强度警示（2026-09-29 监督轮复核）**：逐样本比对 on/off 两臂输出，550 样本中仅 6 条（1.1%）不同——τ=0.01 下 gate 触发率极低（E67 trace 级数据同阈值跳层率 ~12%、far mass 损失 1.7%），「−0.17 无损」实为**近 no-op 无损**，不能作为「gate 有效且无损」的强证据，只能证明「低触发率下无害」。真正有信息量的判决须升 τ（E67：τ≥0.2 才有可观跳层空间；跳 near 空间更大 31% 层/2.3% mass@τ0.1）→ 已排入 E67 e2e（B7s GPU 空闲后，per-request last1 信号 corr 0.924 版本，F1+速度双测）。另：早期诊断日志（00:40 版本）出现的 `far_stat=nan` 为 nan 防护提交（2f9b10f03, 01:05）之前的旧代码，scored run（07:27 修复后）不受影响。

设计权衡（写论文时明确）：动态测层的开销 = prefill 末 chunk 一次 softmax+sum（O(S·Hkv)，与一次 L2 打分同量级，分摊到整个 decode 期可忽略）；收益 = 跳层层的 far 检索（L1 topk + L2 gather + attention far 部分）全部省掉。

---

## 5. sglang 集成设计（新分支 `two-level-indexer`，**M1 已落地 2026-09-23**）

### 5.0 当前进度（M3-b/c 已落地 2026-09-24，commit 36b6c8a37）

已落地（sglang 分支 `two-level-indexer`，基于 fork 最新 main 4b186cfea）：
- **M1**：`tli/{config,indexer,backend}.py` + 注册名 **`tli`**（attention_registry.py + choices.py）；needle32k 真实 trace mass coverage 1.0000，算法移植正确性闭环。
- **M2 前半（2026-09-23 晚）**：
  - `indexer.select` 同步 E4c 修正算法：**B' far/near 分区 top-K2**（far_tokens=256，near_floor 保护防 far 预算吃光近端）+ **D' topk 截断到有效块数**；
  - `tli/kernels.py`——**Triton fused L1 kernel**（`tli_l1_topk`：单 launch 子空间区间算术 + D' 剔除 + 当前块强制 + in-kernel 阈值二分 top-K1），E8-2 原型生产化；
  - 单测 `tli/test_tli_m2.py`（真实 trace hotpotqa L03）：B' 分区 mass **0.99952**（与 two-level-attention 侧 0.9995 一致）；fused kernel 与 eager 块选择 **127/129 一致**（并列截断差异）。
- **M2 后半（e282de8fc）**：paged 寻址（req_to_token 间接）+ O(n) 精确增量索引 + 稀疏 prefill（select_batched）；test_tli_m2b 全链路对拍 PASS。
- **M3-a/b/c（2026-09-24，bfe373c18 + 36b6c8a37）**：
  - 吞吐基线与归因（详见 §5.0.1）；`_sparse_attn` 向量化（Hkv 循环 → 单次 flat gather + 批量 einsum）；
  - **L2 级联 fused kernel 接入**（`tli_l2_partition_topk`）：混合形态 = Triton pass1（gather+GEMV 打分写 far/near 双池 scratch）+ torch.topk 精选 + 滑窗 `arange` 精确复制（近端配额扣减 F=t+1-sw_lo）；**池边界即因果边界**（far 池上界 ≤ t+1-near_len、near 池 < SW_LO，替代 eager fine 矩阵因果 mask，L1 垃圾块天然落池外）；哨兵 S pad 约定。对拍 jaccard **1.0000**（S=9891/131072 × t=末位/回退/中段全过）+ e2e 输出与 eager **逐字一致**。纯 kernel 版 No-Go：4bit 分数并列在终选级过选 130-256/head 无下游兜底 + 120 轮二分串行反慢 4.5×——**L1 能容忍多选（L2 吸收）、L2 不能**（级联 kernel 化通用教训）；
  - **增量索引预分配**（几何扩容 buffer 替代每步 torch.cat 的 O(S) 全量拷贝——S=131K 时 kq 134MB×36 层 ≈ 4.8GB/步纯 memcpy）；增量==全量重建仍逐位一致；update 延迟 0.128ms 与 S 无关。

#### 5.0.1 M3 实测结论（2026-09-24，Qwen3-8B 真实 narrativeqa/LongBench 上下文，H20）

**decode 单步延迟（bs=1，tli vs triton，均 disable_cuda_graph）**：tli eager 84–90 → 向量化+L1k 59–66 ms/step（1.4×）；**与 S 无关** → 开销是 launch 数主导而非算力/带宽。select 微基准（L1+L2 双 kernel vs eager）：9.9K 1.34× / 131K 1.05×（小 S 收益=砍 launch 数，大 S 时 eager L2 的 topk/gather 本身已占大头，kernel 只省 fine 矩阵与 [Tc,Hkv,nd2] 中间量）。

**M3-c 高并发曲线（bs=1/8/16/32 × S≈9.9K token，64 步 decode）**：

| bs | triton ms/step | triton tok/s | tli ms/step | tli tok/s |
|---|---|---|---|---|
| 1 | 10.0 | 99.9 | 51.4 | 19.5 |
| 8 | 16.4 | 486.8 | 369.7 | 21.6 |
| 16 | 19.0 | 842.3 | 666.4 | 24.0 |
| 32 | 32.4 | 986.5 | 1236.8 | 25.9 |

**结构性发现：tli 的 `forward_decode` 是逐请求 Python 循环**——每请求 ~38ms/step（36 层 × select 0.39 + increment 0.13 + sparse_attn ~0.3 + Python 调度），随 bs **线性放大**；triton 全批量化仅 3.2×（10→32.4ms）涨幅。**要拿高并发吞吐主表（§硬件叙事的 H100+大 batch 展示位），必须把选择/增量/前向批量化**：

1. **批量化重设计（下一里程碑，与 TileLang kernel 接入合并）**：per-layer 共享 index pool（kmin/kmax 预分配 `[R_max, NBLK_MAX, Hkv, d']`，~2.4GB@32req/2048blk 可容）+ L1/L2 kernel grid 加 batch 维（`grid=(bs, Hkv)`，per-request nblk/stride 进 metadata 张量）；kq（4bit 精筛缓存，134MB@131K/req）不可共享预分配 → **须改真 4bit 存储**（uint8 + per-token scale，kernel 内 dequant，16.8MB@131K）——与 TIA 的 kpool 设计合流；
2. _sparse_attn 批量化可先行（独立于索引重设计）：req_to_token 本身是 2D `[max_req, max_S]`，`pool_pos = gather(r2t[reqs], 1, sel.view(bs,-1))` 一次 gather 出全部请求的槽位，flat 索引 + 批量 einsum `bhgd,bhkd->bhgk`，~8 launch 总量（现 bs×6）；
3. CUDA graph 未支持（tli 路径 disable_cuda_graph=True），批量化后捕获才有意义。

**逐 token 输出正确性**：tli eager vs L1+L2 双 kernel 在 32K 字符真实上下文上 **64 token 逐字一致**（test_tli_l2_e2e_smoke.py）。

prototype 边界（M3 剩余待做）：
- 批量化 decode（§5.0.1 上述重设计）+ 接导师 TileLang 两级 kernel（`tls_attn/ops/`）；
- CUDA graph / MTP / speculative 未支持（参照 qwen_sparse_attn_backend.py 补）；
- 模型侧无需改动（全局 `--attention-backend tli` 即生效）。

### 5.1 为什么照抄 qsa/ 的结构

sglang 里 `layers/attention/qsa/` 是 **Qwen4-Exp 的 training-free、weight-free 稀疏索引器**（`QSAIndexer`：average_pool 压缩 key → 块级选择 → token topk → 稀疏 attention），结构与我们需求完全同构，且已解决 paged KV gather、CUDA graph（`graph_metadata.py`）、prefill/decode 双路径（`qsa_mqa_prefill/qsa_mqa_decode`）、128MB logits budget 的分块等所有工程难点。**集成 = 把 TIA/SDAI 的两级逻辑填进 qsa 的骨架。**

### 5.2 目录与文件映射

```
python/sglang/srt/layers/attention/tli/          # two-level indexer
├── config.py        # is_tli_profile / TLIProfile 解析 server_args
├── indexer.py       # TLIIndexer：两级选择（A: 子空间 d'；B: 近/远混合）
├── kernel.py        # average_pool → k_min/k_max 计算（子空间维）+ 级联 topk triton kernel
├── metadata.py      # TLIIndexerMetadata：topk 结果缓存（参照 dsa_indexer_metadata.py）
├── sparse_attn.py   # 候选 gather + 稀疏 GQA 前向（复用 qsa_sparse_fa2_cu_seqlens_triton）
├── backend.py       # TLISparseAttnBackend(AttentionBackend)：init_forward_metadata
│                    #   / forward_decode / forward_extend（prefill 路径）
└── glue.py          # 挂接 qwen3 模型：monkeypatch 或参照 qwen_sparse_attn_backend 的注册方式
```

关键接入点：
- **decode**（`forward_decode`）：单 token q → Level-1 子空间块粗筛（k_min/k_max pool 从 paged KV 维护，preshuffle 布局参照 `dsa/kpool_fp8_index.py`）→ Level-2 候选 token 精筛 → topk → `qsa_mqa_decode` 式稀疏前向。
- **prefill**（`forward_extend`）：chunked prefill 下对每个 chunk 的 q 行做 Level-1（128MB logits budget 分块，照抄 `_qsa_prefill_row_chunk_size`），Level-2 只对候选块做（`qsa_mqa_prefill` 的稀疏 varlen FA2 路径）。短序列（<2048）退 dense——与 DSA 的 `SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD` 同策略。
- **近/远混合**（创新点 B）：`init_forward_metadata` 里按 seq len 切近/远，近端块直接全选（预算 C_near），远端走聚类中心分数 → C_far = budget − C_near；decode 增量：新 token 进近端窗，滑出窗的 token 做一次聚类 assign。

### 5.3 分支计划

```
sglang: main → branch two-level-indexer
  M1: tli/{config,indexer,metadata} + eager 版两级选择（纯 PyTorch，对拍导师 TIA）
  M2: kernel.py triton 化（L1 fused + 级联 topk）+ sparse_attn 复用
  M3: backend + glue，Qwen3-8B 上 bench_serving e2e
```

**注意**：算法探索期（E1–E4）在导师 two-level-attention 仓库做（pipeline 现成、迭代快）；sglang 分支承担「系统级 e2e 数字」（论文 system 章节）。两条线共用同一套 index 数据结构定义（tensor 布局写进本报告 §5.4 后即冻结）。

### 5.4 冻结的 tensor 布局（两仓库共享）

```
k_min/k_max pool : [num_blocks, H_kv, d']        bf16（d' = 子空间维）
far_centroids    : [K_c, H_kv, d']               bf16
far_assign       : [T_far]                       int32
L1 topk 结果     : [B, H_kv, K1]                 int32（块 id）
L2 topk 结果     : [B, H_kv, K2]                 int32（token id）→ 喂稀疏 attention
```

---

## 6. 实验计划与数据对拍（本机 H20 立即可执行）

### E1 — 复跑 kernel benchmark（**已完成，H20 实测 2026-09-23**）

`two-level-attention/benchmark/efficiency/benchmark_mha_kernel.py`，batch=4/4heads/1kv_head/128d，warmup 10 × measure 100：

| seqlen | dense MHA | 两级 SparseMHA | **本机 H20 加速** | 导师原机器（参考） |
|---|---|---|---|---|
| 32K | 0.584 ms* | 0.400 ms* | **1.46×** | 1.30× |
| 64K | 1.949 ms | 0.703 ms | **2.77×** | 2.47× |
| 128K | 3.906 ms | 0.767 ms | **5.09×** | 4.58× |

（*32K 的分项数字取自输出日志汇总；完整 JSON 存 `exp/results_efficiency/benchmark_mha_20260923_124217.json`）

**对拍结论**：
1. proposal `pseudodata_kernel.csv`（naive 2.8ms / fused 2.05 / overlap 1.62 @128K）与本机两级 pipeline 总延迟 0.767ms **不同口径**——proposal 的数字明显把更多内容计入（可能是完整 indexer 含投影/量化/resize，或不同硬件假设）。**正式实验前必须先冻结 kernel latency 的测量口径**（建议：分项计时 L1-fwd / L1-topk / L2-fwd / L2-topk + 总和，四段式报告），否则两套数字没法对拍。
2. proposal `pseudodata_context.csv` 的趋势（加速随长度增长）方向正确，本机实测 1.46×→5.09× 甚至好于其 kernel 级预期。
3. proposal `pseudodata_e2e.csv`（1.04×→1.24×）仍待 sglang M3 后测——e2e 分母含模型权重读取/MoE/通信，attention 加速 5× 稀释到 e2e 1.2× 是合理量级（可先用 Amdahl 定律粗算：若 attention 占 decode 时间 30%，5× attention 加速 → e2e ≈ 1/(0.7+0.3/5) = 1.30×；proposal 的 1.24× 反而略保守）。
4. **32K 只有 1.46× 仍是短板**（HISA 声称 2×@32K），印证创新点 C 的 fused/级联方向是真问题。

### E2 — H1 位置偏斜验证（**已完成，2026-09-23**）

- 权重：`/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B`（5 分片完整）
- 方法：`collect_trace.py` hook `ALL_ATTENTION_FUNCTIONS["sdpa"]`，needle32k + natural32k + 真实 LongBench（hope clone 官方数据集，hotpotqa/narrativeqa/passage_retrieval_en/gov_report 各 2 条最长样本，官方模板 32K 截断）共 **10 条 trace × 36 层**，每层存 K [S,8,128] + 尾部 256 q + 16 锚点 q。
- **结果（`analyze_h1_full.py`，见 §4A.1 表）**：
  1. proposal 的 H1「近端集中」在 **mass 口径下不成立**——near(<2048) 只占 0.16–0.29，真正的偏斜是 sink（前 64 token 占 0.37–0.71）。proposal slide 4 的叙事需要改写为「sink + 近端 + 远端三层分解」；
  2. dense top-1024 mass 覆盖 = **1.0000（10/10 条 trace）**——这是 TIA@1024≈FullKV 的根本原因，也是论文换用 mass coverage 主指标的依据；
  3. 层异质性真实且跨 prompt 可预测（同任务 corr 0.93–1.00）→ 催生创新点 D'（§4A）。

### E3 — H2 子空间冗余验证（**已完成，2026-09-23**）

- `analyze_e3.py`（子空间扫描）+ `analyze_e3b.py`（完整两级 pipeline 覆盖率），结果见 §2.5。**Go**。
- proposal `pseudodata_dimension.csv` 的真实替换数据已备齐（`results/e3_subspace_recall_v2.json`、`results/e3b_pipeline_mass_cov.json`）。

### E4 — 创新点 B 代表质量 microbench（**已完成，2026-09-23**）

- `analyze_e4.py`（C_far=2048 单点）+ `analyze_e4b.py`（预算 {512,1024,2048,4096} × 5 策略 Pareto），结果见 §3.5。**Go**。
- proposal `pseudodata_pareto.csv`（Hybrid 99.48%@0.49）的真实替换数据已备齐（`results/e4b_pareto.json`）。

### E5 — 附加创新点探索（**已完成，2026-09-23，全部 No-Go → negative results 章节**）

- `analyze_e5.py`：(1) 跨层 top-k 复用（IndexCache 式）——相邻层 top-1024 IoU 仅 **0.33**，skip8 更低，复用无价值；(2) decode 步间增量 topk——相邻步 churn **22%**（每步 1/5 候选换新），增量维护比全量重算更贵。
- 在线 L1 信号层跳过（D' 的在线版本）——与真实 far 质量相关 ≈0（见 §4A.3），否定在线启发式。
- **这些否定结果本身是论文资产**：证明「层级稀疏的捷径（跨层复用/增量维护/在线信号）在这个 workload 上全部走不通」，为离线 profile 路线（D'）和两级设计（A/B）提供了反衬。

### E5b — 精度与 e2e（待做，依赖 sglang 集成 M3）

- 精度：导师 LongBench/RULER pipeline 直接加 `--method sdai`；对拍 Qwen3-32B 已有 TIA/TWI/Quest/none 四组真实数据（`exp/results_longbench/*/result.json`）——**这是现成的论文 baseline 主表**；
- e2e：sglang M3 后 `bench_serving`，对拍 `pseudodata_e2e.csv` 与 `pseudodata_index_overhead.csv`（8%→4.1%）。

### E6 — 创新点 D' 层跳过（**已完成，2026-09-23**）

- `analyze_e6.py`：离线平均轮廓静态跳过 13/36 层，precision 0.92–1.00、far 质量损失 <0.3%（见 §4A.2）；在线 L1 信号版本否定（见 §4A.3）。

### E7 — kmeans 增量维护/staleness（**已完成，2026-09-23**）

- `analyze_e7.py`：前半聚类 + 后半增量 assign vs 全量聚类——recall 衰减 ≤0.03、mass 持平或上升、增量开销 <1μs/token/head（详见 §3.6）。**创新点 B 工程风险排除：prefill 一次全量聚类 + decode 增量 assign 即可**。

### E8 — 待做清单（按优先级）

1. D' + B 组合收益测量：跳过 13 层后聚类总 build 开销从 540×36 降到 540×23 ms/请求；
2. 级联 topk fused kernel 原型（创新点 C 主线）+ E1 对拍；
3. sglang tli/ 集成（§5）后跑 E5b 精度/e2e。

### E64 — 用户一般化框架接入（2026-09-27~28，transformers 侧 tli_indexer.py 全参数化）

E64a 48 臂网格（method×alpha×beta）+ E64b 冠军细扫（bp/gamma）+ E64c 速度成本模型 + E64d 可视化（详见 #69-#72 完成记录）。关键产出：α/β/γ 三参数框架（α=near 区占 mid 长度比、β=near 块预算占 K1 比、γ=near 细筛 token 折扣）+ sup_wsvd 投影基（注意力加权 PCA top-r，full-D [36,Hkv,128,8] 存储尾维 32 子空间 scatter）。**重大口径发现：E64 trace 系 B_TOK=2048 vs 主表 K2=1024 错位**——γ 按总预算等比缩放 0.5→0.25，E64 绝对质量数字（0.9300 mono / 0.9102）是 2× 预算口径，论文必须标注。

### E71 — 主表重跑 + B（d8 投影）e2e 崩坏终局定案（2026-09-28，**B 降级 negative result，C0 为主推**）

transformers 侧 tli_indexer.py 接入 E64 框架（--tli_alpha/beta/gamma/proj_basis；L1 双池 far=minmax 上界/near=avg；L2 γ 预算；投影基惰性提取 [Hkv,32,r]）。C 冒烟 hotpotqa **F1=54.93 > TIA 53.89 > FullKV 53.48**（纯 A 单池 bp128 K2=1024 超基线）。

**B 配置（bp128+sup_wsvd d8+α.125/β.25/γ.25）e2e 崩坏，逐项修复均无效**：
- B0（γ0.5）F1=13.51 → 口径错位（γ 预算吃光 K2，far 仅保底 64）；
- B2（γ0.25 缩放修正后）仍 21.55；B3（投影+单池）47.73 vs C0 54.93——投影本身掉 7.2 分；
- bf16 三环节（k/特征/分数）trace 复测全部无损；fp32 score_fine 修复（8 维点积幅度 ~O(0.5)，bf16 折叠 top-K 边界 gap）后 B4 冒烟仍全错。

**终极对拍定案（/tmp/tli_ref_diff.py，hook e2e 全链路 q/k/mask vs trace 参考实现）三连证据**：
1. mass_ref(d8 连续投影)=0.2753 ≈ mass_ref2(e2e 同构 softmax+group-mean 细筛打分、无量化)=0.2751——细筛打分口径与 4bit 量化都不是分歧源；
2. **C0 判决臂：ref(32 维连续直取) mass 也只有 0.3028**，而 e2e C0 mask=0.9941、IoU 仅 0.64——trace 参考口径（group-sum q 点积选择+连续 minmax 池+排除 sink）**在 e2e decode 上系统性不成立**，无论 32 维/d8、连续/量化，mass 恒 ~0.28-0.31；
3. decode 第一步（q 未漂移）所有 mask mass=1.0 → 第二步起参考全崩——**注意力加权 PCA 基对 decode q 分布漂移脆弱（分布外适用域问题），位置稳定尾维子空间恒稳**；e2e 高 mass 主要由 sink/swa/当前块强制区贡献，GQA group-sum 评估口径本身有畸变。

**论文叙事**：离线投影口径无损（0.9466）vs e2e 崩（47.73）的口径鸿沟本身是 negative result——training-free 投影基没有在线适应，decode q 漂移即失效；与「位置稳定子空间」的鲁棒性形成对照，反向支撑创新点 A 的子空间选择依据。C0 已放量 12 任务（双 GPU，postfix _c0）。

**对拍工程坑（复现必读）**：e2e 的 k 带 batch 维 [1,S,Hkv,D]（trace 是 3D）；compute_mask 的 L2 是 softmax(score_fine)→group-mean→topk 而 trace 是 group-sum 点积直接 topk（GQA 下 group-sum 分数与各 head 真实分布相关性低）；prefill q 时间维 >1 须 decode-only 守卫；hook 用 runpy.run_path 跑 pred.py + monkeypatch load_dataset 截样本。

### 实验纪律（对 proposal 数据观的继承与强化）

proposal 的做法值得肯定：所有图标注 synthetic/expected、Go/No-Go 前置、要求替换为实测。我们的强化：(1) 每张对拍图同时报告本机 H20 与（如可用）H100 数字——异构结论必须带硬件标注；(2) selector-fidelity 与 end-quality 双口径分开报告，不允许用 recall 替代下游精度；(3) 聚类类实验必须含 build/update overhead 单列，不许只报 query latency。

---

## 7. 对 proposal 的修改建议汇总（开题答辩防御清单）

1. 增量声明补一段「相对 TIA/TWI（本组第一代）」的显式对比表，否则「two-level」会被 HISA+TIA 双重 prior-art 夹击。
2. 创新点 A 的叙事改为「position-stable subspace」，把 DSA noPE 与 Qwen3 低频维统一为一个抽象（`subspace projection P`），避免语境错配。**实测已备好支撑数据**（§2.5：lowfreq d'=32 ≈ 全维，random/highfreq 崩溃）。
3. 创新点 B 补「聚类中心非上界」的漏选分析 + margin 校准方案（§3.2 第 4 点）——理论空洞仍在，但实测表明 mass 口径下 kmeans 反而优于 minmax（§3.5），可转守为攻。
4. 创新点 C 主线改为 fused + 级联 topk（硬件无关的收益），TC×CC 降级为 persistent kernel 可选章节并绑定 cost model；硬件章节明确 H20 实测 + H100 如可用。
5. HBM↓45% 改为「Stage-1 kernel DRAM bytes（ncu 实测口径）↓X%」，X 由流量模型先算出再承诺。
6. 实验资源行更新：本机 2×H20-3e 可完成 E1–E4 与全部 selector-fidelity 实验；DeepSeek-V3.2 e2e 属集群项，备选 OpenDSA-16B（2×H20 可跑，正好接 TransArch 里的 OpenDSA 代码）。
7. **主指标从 entry Recall@K 换成 mass coverage**：dense top-1024 的 mass 覆盖实测恒等于 1.0000，entry recall（0.27–0.50）严重低估实际精度；两口径并列报告（§2.5）。
8. **H1 假设改写**：位置偏斜的真实形态是「sink(64 tok) + near(2048) + far」三层而非双峰；sink 占 0.37–0.71 是远超 proposal 预期的大头。
9. **新增创新点 D'（层自适应级联跳过）写进正文**（§4A）：离线校准 profile → 静态层掩码，13/36 层免远端检索，far 质量损失 <0.3%；附在线信号失败的 negative result 做消融。
10. **negative results 单独成节**（E5）：跨层复用 IoU 0.33、decode churn 22%、在线 L1 信号相关 ≈0——三个「想抄近路」的方向全部实测否定，凸显两级设计 + 离线 profile 的必要性。

---

## 附：资源与代码索引（本机已核实）

| 资源 | 位置 |
|---|---|
| GPU | 2×H20-3e（78 SM, cc9.0, 139GB），tilelang 0.1.7.post3 可用 |
| Qwen3-8B 权重 | `/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B` |
| 导师第一代实现 | `/home/wangyuanshuo02/two-level-attention`（TIA/TWI + TileLang kernel + 三模型 LongBench 真实数据） |
| proposal 材料 | `/tmp/two_level/indexer_proposal/`（21 页 PPT + 7 组伪数据 CSV + drawio） |
| 相关工作代码 | `/tmp/two_level/related/{TransArch(HISA/OpenDSA), IndexCache, DeepSelect}` |
| DSA 官方教学实现 | `/tmp/papers/DeepSeek-V3.2/inference/{model.py, kernel.py}` |
| sglang 集成参照 | `sglang/python/sglang/srt/layers/attention/qsa/`（training-free indexer 完整骨架） |
| Qwen3-8B 真实 trace（10 条 × 36 层） | `/tmp/trace/qwen3-8b/`（needle32k、natural32k、lb_*×8），采集脚本 `two-level-attention/exp/trace/collect_trace{,_lb}.py` |
| 实测结果 JSON | `two-level-attention/exp/trace/results/`（h1_full / e3_subspace_recall_v2 / e3b_pipeline_mass_cov / e4_representatives / e4b_pareto / e5_reuse_churn / e6_layer_skip） |
| LongBench 官方数据集 | `/home/wangyuanshuo02/datasets/LongBench/`（hope clone，21 子集 × 200 行） |
