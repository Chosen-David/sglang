# 3. Kernels（论文正文草稿，2026-09-28 晨）

> 事实锚点：E1/E8-2/§8b-12/13/18/19/20/29 + #65 各节 + fig10。
> 口径铁律：kernel 级 microbench 与 e2e 双层都测都诚实报告。

## 3.0 设计约束（H20 形态）

H20 的 Tensor Core 吞吐仅为 H100 的 ~15%，而 HBM 带宽同级
（4TB/s）——**选择/打分类 kernel 的第一性约束是带宽而非算力**。
ρ 实验数据（带宽利用率 vs 计算强度）显示打分 GEMV 在 d'=32 形态
下 deep memory-bound，TC 化收益无法兑现（§8b-18 No-Go 如实
报告）。因此 kernel 设计主线 = **消除中间物化 + 合并 launch**，
而非算子替换。

## 3.1 Fused L1 块打分（E8-2）

朴素实现：逐块 gather K 子空间 → 反量化 → einsum → 写回，每块
5-6 个小 kernel。Fused 版：单 Triton kernel 内完成 gather + 4bit
反量化（共享 scale 表）+ GEMV 打分 + 块级规约，输出直接为块分数。
**3.6×（32K 形态）/ 1.6×（131K）**，块 id 对拍与 eager 逐位一致。

## 3.2 L2 级联 Dual Kernel（M8-KernelC）

L2 需同时对 far/near 双池打分取 topk。Dual kernel 单 launch 完成
分区 topk：候选块展开为 token 粒度打分，**池边界即因果边界**
（far 池上界 ≤ t+1-near_len、near 池 < SW_LO——替代 eager 的
fine 矩阵因果 mask，L1 垃圾块天然落池外）。

关键工程发现（负结果资产）：
- **纯 kernel 版 No-Go**：4bit 分数并列过选（130-256/head 无人
  兜底）+ 120 轮二分串行反慢 4.5×——**L1 能容忍多选（L2 吸收）、
  L2 不能**，混合形态（Triton pass1 打分 + torch.topk + 精确
  复制）为终态。
- **bf16+pad 直出**（#65）：OUT_BF16 constexpr 分支使 far/near
  分数直接以 bf16 输出到 pad 对齐宽度（pad 列 -inf），消灭 fp32
  [n,Hkv,Tc]×2 物化 + cast/pad 复制链（profiling 中 elementwise
  16.3% → 0）。
- **CHUNK 形态调优**：64/num_warps=4 最优（12.1 vs 14.9/17.7ms，
  与 M8-2b compact 路径结论一致）。
- **TMA 转置写布局 No-Go**（§8b-20）：写侧假设证伪，实测反慢。

## 3.3 M11：统一稀疏 Attention Fused Kernel

选择完成后，稀疏 attention 本体（gather 候选 K/V + softmax +
PV）的 eager 路径需 [n, K2, D] fp32 双物化。M11 单 Triton kernel
完成 gather + online softmax + 输出直写 bf16，**kernel 级
11-16×**。工程陷阱记录：q_raw 为父张量切片 view 时行 stride ≠
H*D（实测 6144 ≠ 4096），kernel 寻址假设行连续 → 必须
contiguous 化（一次 bf16 拷贝仍远快于 eager fp32 双物化），
否则 row≥1 全部读错位置（e2e 乱文根因）。

## 3.4 DS Topk（DeepSelect 分数路径）

4bit 打分输出可用数值判别结构（DeepSelect 类）替代 torch.topk：
L1 与 far 池接入（**decode 侧净效果 bs16 +10%**），near 池保留
torch——**No-Go 教训**：near 池压缩表 ~2048 有限项挤在 4bit 格点
极少数分数值（tie 组巨大）→ 集合 jaccard 0.72。tie 打破差异与
候选分数离散度强相关，**接 DS 前必须按池预判**。

CUDA graph 约束：DS host 端 pad/end 分配不可进 capture 区域，
graph 捕获时自动回退 torch.topk（`is_current_stream_capturing`
门控）。

## 3.5 Select 调度链的消除（#65）

两个隐藏瓶颈的发现与修复（microbench 测不出、e2e 才显形的
口径盲区案例）：

- **早期行 Python 循环**（#58 修复引入）：首 chunk ~1024 行 ×
  arange/cat/index_put ≈ 5K launch + 3K 次 .item() 同步 →
  chunk0 65ms。向量化 `where(j<cut, j//reps, j-cut)` 单 op 构造
  uniform grid，4×。
- **3 处 host 同步**：fast_path 判定 `.all().item()` / empty 段
  `.any()` / early 段 `.any()` × 生产形态 ≈ 10 万次队列排空。
  修复 = kernel 路径可用时直接跳过判定 + `t_min_hint` host
  参数（forward_extend 的 prefix 是 Python int，可判「无 empty
  行 / 无 early 行」跳过同步）。8B e2e prefill −6.7%。

## 3.6 同机三方对比（统一 harness）

官方 kernel 原样接入统一计时框架（cudaEvent + 131K 口径）：
Quest 官方 decode_select_k.cuh（编译接入）、DSA FlashMLA
fp8_index（原样 import）。延迟排名 Quest < DSA < TLI（TLI 延迟
当前不占优，如实报告）——TLI 的三方优势在算法结构（索引 MAC
1/32 of DSA、存储 3×↓ vs Quest、质量 +2.2 分）。**对比铁律：
kernel 原样、harness 统一**（§8b-6 详表）。
