# DSA Lightning Indexer 科研调研笔记

> 目标：优化 DeepSeek Sparse Attention (DSA) 的 indexer，对比 baseline DSA 评测：
> ① 精度基本不掉 ② indexer 本身的 kernel 加速 ③ end-to-end 吞吐提升
> 调研时间：2026-09-21。环境备注：arXiv/WebSearch 不可达，调研走 GitHub git clone
> （`git@github.com` SSH 可用，HTTPS 不稳定）。

## 1. DSA 论文核心（DeepSeek-V3.2-Exp 技术报告，本地 `/tmp/papers/DeepSeek-V3.2/DeepSeek_V_3_2.pdf`）

### 1.1 Lightning Indexer 公式

```
I_{t,s} = Σ_{j=1..H_I}  w_{t,j} · ReLU( q_{t,j} · k_s )
```

- `H_I` = indexer 头数（64）；每头维度 `d_I` = 128，FP8 量化
- `q_{t,j}, w_{t,j}` 由 query hidden state 投影得到（w 即 head gate）
- `k_s` 由前面 token 的 hidden state 投影，**部分 RoPE**（注意：non-interleaved 布局，
  与 MLA 的 interleaved 不同——官方 2025.11 修过此处 bug，实现时高危点）
- ReLU 激活是为了 throughput（非 softmax）
- Top-k Selector 按 I_{t,s} 选 top-2048（128K 上下文下稀疏度 ≈ 1.6%）

### 1.2 架构耦合（为什么 Qwen 不能直接用）

- DSA 实例化在 **MLA 的 MQA 模式**之上：latent 向量（kv_lora 512 维）被一个 query token
  的全部 head 共享；indexer 作用在 latent 空间，细筛 attention 是 MQA
- 671B 超参：`index_n_heads=64, index_head_dim=128, index_topk=2048,
  q_lora_rank=1536, kv_lora_rank=512`（config_671B_v3.2.json）
- indexer 是**有参数的子模块**（wq_b/wk_b 投影 + head gates），必须训练：
  - warm-up：lr=1e-3，仅 1000 步 × 16 seq × 128K = 2.1B token
  - 训练目标：KL(p_t,: ‖ Softmax(I_t,:))，p 为 MLA 注意力分布的 L1 归一化
  - 之后 sparse 训练阶段全参数继续训练适配稀疏模式

### 1.3 论文实验口径（我们的评测模板）

| 维度 | 论文做法 | 我们的对应 |
|---|---|---|
| 精度 | 16 个 benchmark 对比 V3.1-Terminus（MMLU-Pro/GPQA/HLE/LiveCodeBench/AIME/SWE...）| 同口径；注意 worst-case：HLE 21.7→19.8、HMMT 86.1→83.6 有回落 |
| 成本/吞吐 | H800 集群每百万 token 成本，prefill 与 decode 分列，token position 0K→128K 曲线 | sglang bench_serving，扫 8K→128K 序列长度 |
| indexer 单独加速 | 论文未给（**增量贡献点**）| ncu profile MQA logits + topk kernel，对比 DeepGEMM PR200 CUDA 实现 |
| （建议补充）topk 扫描 | 无 | 1024/2048/4096 的稀疏度-精度-速度三重曲线 |

短序列（<2048）DSA 自动退回 masked dense MHA（`SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD` 可调）。

## 2. 相关论文地图

### 赛道 A：可训练稀疏注意力（indexer 家族）

| 论文 | 出处 | 核心思想 | 备注 |
|---|---|---|---|
| NSA (Native Sparse Attention) | arXiv 2502.11089, 2025.02 | 压缩+选择+滑窗三分支；论证 KV 须跨 query head 共享（kernel 级约束）| DSA 直接前驱；有 0.5B/8B 预训练开源代码 |
| MoBA (Mixture of Block Attention) | arXiv 2502.13189, Moonshot/Kimi 2025.02 | block 级无参数 gating 的 MoE 式稀疏；全↔稀疏无缝切换 | 本地 `/tmp/papers/MoBA`；README 明说需继续训练，非 drop-in |
| DSA (DeepSeek-V3.2-Exp) | DeepSeek-AI 2025.09（技术报告，非 arXiv）| token 级细粒度 + lightning indexer | 我们的 baseline |
| 跟进家族 | 2025-2026 | GLM-5.x Dsa、Longcat-Flash、Dots3、HYV4、DeepSeek-V4（FP4 indexer）| 见 sglang `model_config.py:144` 白名单——indexer 已是行业趋势 |

### 赛道 B：免训练 KV 选择（推理侧近亲，Related Work 必引）

Quest（ICML 2024，query-aware page 级选择）、H2O、SnapKV、PyramidKV、StreamingLLM。
分界线：它们不训练、逐请求打分；indexer 家族训练打分器、离线固化。

## 3. SGLang 实现现状（已发表技术 = 必须对比的 baseline）

核心文件（`/home/wangyuanshuo02/sglang/python/sglang/srt/layers/attention/dsa/`）：

- `dsa_indexer.py`：Indexer 类（调度层），`_get_topk_paged`（decode 路径）、
  `_get_topk_ragged`（prefill 路径）、`_forward_cuda_k_only`
- `dsa_indexer_kpool.py` / `kpool_fp8_index.py`：FP8 index k cache 池
- `dsa_indexer_metadata.py`：topk 结果缓存
- `paged_mqa_logits_backend.py`：MQA logits kernel 调度（aiter/cutedsl/deepgemm 多后端）
- `dsa_backend_kpool.py`：细筛稀疏 attention backend
- CUDA kernel：`kernels/jit/csrc/deepseek_v32/indexer_k.cuh`（DeepSeek JAX 参考的 CUDA 移植）

已实现的优化（写论文时作为"已知技术"逐条对比）：

1. Index Cache 跨层复用（官方称 negligible accuracy cost）
2. skip_topk 层模式：GLM-5.2 用 61 位 pattern `FFSFSSS...` 决定哪些层跑 indexer
3. 短序列自动退 dense MHA（阈值 2048）
4. FP8 → FP4 index 分数缓存（DeepSeek-V4）
5. chunked MQA logits 的 SM budget 管理、preshuffle paged 布局
6. decode 时 indexer 与 q_b_proj 的 alt_stream 重叠执行
7. MTP precompute 与 speculative decoding 的浅层结合（`dsa_backend_mtp_precompute.py`）

## 4. 开放研究机会（观察到的 gap）

- **decode 增量计算**：每步新 token 只需追加一列 index 分数，topk 可增量维护
  （当前每次重算 MQA logits + 全量 topk）
- **动态自适应 topk**：按 I 分布熵调整 k，简单上下文少取 token
- **topk 本身的开销**：128K 里选 top-2048，排序/选择代价可观，有专用 kernel 空间
  （bitonic / radix topk，或直接在 MQA logits kernel 里 fused）
- **indexer × speculative decoding 深度结合**：draft token 的 topk 复用/预测
- **跨层共享度分析**：Index Cache 只是经验性的"相邻层结果相近"，缺乏理论刻画

## 5. 实验资源与约束

| 资源 | 位置 | 用途 |
|---|---|---|
| sglang 完整实现 | `/home/wangyuanshuo02/sglang` | e2e 基准 |
| 官方教学级实现 | `/tmp/papers/DeepSeek-V3.2/inference/{model.py, kernel.py}` | 入门精读（比 sglang 干净）|
| DSA 论文 PDF | `/tmp/papers/DeepSeek-V3.2/DeepSeek_V_3_2.pdf` | 6 页技术报告 |
| MoBA 论文+代码 | `/tmp/papers/MoBA` | 竞品路线 |
| NSA 训练代码 | github fastnlp/NativeSparseAttention | 小模型自训 |
| DeepGEMM CUDA kernels | github deepseek-ai/DeepGEMM PR #200 | indexer logits kernel baseline |

**硬件约束（关键）**：本机仅 Xeon CPU 无 GPU。671B e2e 实验需 8×H200 级集群。
分级路径：
- kernel 级研究（indexer 加速）：合成数据 + 单卡 A100/H100 即可，不需要跑大模型
- 精度 + e2e：需集群，或用 NSA 设置自训 0.5B/8B 小模型

## 6. 建议切入顺序

1. 精读官方 `inference/model.py` 的 indexer 前向（配合论文公式）
2. 决定论文类型：kernel/systems 向（indexer 加速）还是算法向（训练/选择策略）
3. kernel 向 → 复现 DeepGEMM indexer logits 做公平 baseline → 找 gap（见 §4）
4. 算法向 → NSA 0.5B 自训复现 → 改 indexer 设计
