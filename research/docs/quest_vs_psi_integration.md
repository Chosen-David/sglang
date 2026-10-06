# Quest vs TLI/PSI 集成机制逐项差距调查（#126 / E107a，2026-10-06）

> 调查 agent 只读产物落袋。目的：定位 Quest e2e 快（TP2 62.1 vs 142.0s、单请求
> 64K prefill 54.4 vs 136.0s）的集成层根因，为 #127 e2e 修复提供设计输入。

## 0. 关键背景事实（影响全部结论）

1. **sglang 的 Quest backend 不是原版 Quest 移植，而是 2026-10-04 为审稿 C3 修复
   新建的对照臂**（`quest/backend.py` docstring L1-23：「改造基底 = TLI backend
   ……差异只在索引与选择」）。它与 TLI backend 共享同一套 req_to_token/pool 接线、
   `_sparse_attn_batched`/`_dense_extend_one`/`_sparse_extend_one` 消费端，**连
   fused attention kernel 都用同一个 `tli_sparse_gather_attn_dot`**（
   `quest/backend.py` L35/L514-517/L572-575）。因此两臂 e2e 差距可以完全归因于
   「索引构建 + 选择」两段，attention 消费端是受控变量。
2. **MoBA backend 是 Quest backend 的子类**（`moba/backend.py` L78：
   `MoBASparseAttnBackend(QuestSparseAttnBackend)`），零改动继承消费端与
   CUDA graph veto，只换索引（chunk-mean ksum）与选择。
3. 路径：`python/sglang/srt/layers/attention/{quest,tli,moba}/{backend.py, indexer.py, config.py}`（tli 另有 kernels.py）。

## 1. 逐项差距表

| # | 维度 | Quest 做法 | TLI 做法 | 代码证据 |
|---|---|---|---|---|
| ① | prefill 索引构建数据类型 | `k_buf[locs]` 保持 bf16，page min/max 在 bf16 上算 | `k_buf[locs].float()` 全量 fp32 物化（64K 单请求单层 ≈128-256MB） | q/backend.py:390 vs t/backend.py:654 |
| **②** | **chunked prefill 索引增量化** | **增量**：`S_st==prefix` 时页边界结合律合并 O(nq) | **每 chunk 全量重建**（O(S)/chunk，整请求 O(S²/chunk)） | q/backend.py:383-387 + _update_row_full L420-456 vs t/backend.py:654-655 无条件全量 |
| ③ | 索引内容 | 仅 page 级 kmin/kmax（2 键，bf16，8B/token/head） | 额外对全部 S token 做 quant4_pack（4bit 格点+fp32 scale），写 pool 5 键（44B/token/head） | q/backend.py:391-394 vs t/indexer.py:244 + t/backend.py:660-664 |
| ④ | prefill 选择结构 | 单级 page top-16：界张量小 cast → 2 einsum（恒等式）→ 1 次 topk，零同步 | 两级：L1 topk K1=128 → onehot 并集 → sel_mask → fast/slow/kernel 三路径，eager 兜底有 `[n,Hkv,S]` fp32 fine 物化 + 全宽 topk ×2 | q/indexer.py:233-297 vs t/indexer.py:514-903 |
| ⑤ | 反量化表 | 无对应物 | select_batched **每次调用**无条件构建全宽 fp32 反量化表 kq_f（[S,Hkv,nd2]），M10 kernel 路径也白付 | t/indexer.py:583 |
| ⑥ | host 同步 | 静态宽度零同步 | eager 慢路径 `int(sel_mask.sum(1).max().item())` 等多处 | t/indexer.py:607,779,805,853,887 |
| ⑦ | prefill Python for 循环 | **同样存在（未修复）** q/backend.py:362 | t/backend.py:640 | **非两臂差异**——Quest 循环体轻一个数量级才是差距来源 |
| ⑧ | prefill 稀疏 attention 消费端 | fused kernel | fused kernel（同一个） | 无差异（受控变量） |
| ⑨ | dense 短序列路径 | fp32 物化 | 同 | 无差异 |
| ⑩ | decode CUDA graph | 无（恒 veto，q/backend.py:184-193） | 有（M5 图内统一稀疏+增量零同步 replay） | TLI decode 优势构成项 |
| ⑪ | decode 选择 kernel 化 | PyTorch：全宽 fp32 行物化 ×2 + 全维 D=128 einsum | M8 fused：直读 pool 子空间 d'=32 + 4bit 寄存器反量化 | decode 索引开销 32 vs 59.5 ms/step=1.85× 的直接来源 |
| ⑫ | 索引存储 | 8B/token/head | 44B/token/head | Quest 反而更省（论文已分开报告） |

## 2. 一句话结论

sglang 的 Quest 臂是 TLI backend 的**减法 fork**——用「单级 page 界 + bf16 索引 +
chunk 增量合并 + 零同步静态宽度选择」替换了 TLI 的「fp32 全量物化 + 每 chunk
全量重建 + token 级 4bit 缓存 + 两级 fine 矩阵选择」。**TLI 的 kernel 化投入全部
押在 decode 侧（M5/M8/M10），prefill 侧的 build 段与 select 兜底段还是
correctness-first 原型残留**；Quest 算法本身只需 page 级界，结构性绕开这些成本。
这解释了 E102 三段式判决：decode 反转（PSI 赢 1.85×）+ prefill 惨败（输 2.5×）。

## 3. 修复建议清单（#127 输入，按预期收益排序）

| # | 修复 | 预期收益 | 难度 | 参照 |
|---|---|---|---|---|
| **F1** | **prefill 索引 chunk 增量化**：forward_extend 加 `S_st==prefix` 增量分支（页边界结合律合并 kmin/kmax/kq） | **最大单项**——消除每 chunk O(S) 全量重建（64K/8K chunk 少做 ~7/8 build 工作 ×36 层×16 请求）；decode 侧 update_block_index（t/indexer.py:268-350 增量+尾块修正已验证）可移植 | 中 | q/backend.py:383-396,420-456 |
| **F2** | 消除全量 fp32 物化：`.float()` 前先切子空间；kmin/kmax 直接 bf16 算（min/max 对 bf16 精确）；quant4_pack 输入口径须对拍（4bit 格点逐位性是现有不变式） | 带宽与显存峰值 2×+ | 低-中 | t/backend.py:654 |
| **F3** | kq_f 反量化表惰性化：只在 fast path/empty-row fallback 实际用到时构建 | 每 select 调用省 ~32MB/层/调用，kernel 路径下纯白付 | 低 | t/indexer.py:583 |
| **F4** | prefill 跨请求批量化（for 循环消除）：ragged build + batched select + batched attn | bs=16 摊薄固定项；Quest 也没做，做了即双方共同改进（E106 待办「TP2 重测」前置） | 高 | t/backend.py:640 |
| **F5** | 消残余 host 同步（.item()/bool() hint 化） | eager/异常路径队列排空 | 低 | t/indexer.py:607,805 |
| F6 | kq 4bit 构建流水化/降精度（与 F2 合并考虑） | — | 中 | t/indexer.py:244 |

**不要移植的**（Quest 侧劣势，论文如实报告）：无 CUDA graph、全维 D=128 打分、
每步 fp32 行物化——这是 PSI decode 1.85× 优势的来源，保留为对照卖点。

F1+F2+F3 三项移植即可对齐 Quest 的 prefill 主差距，且不触碰 decode 已验证的
1.85× 优势路径。
