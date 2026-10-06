# E72 五组合 e2e 真实精度对比（Qwen3-8B LongBench, F1 口径）

| method 组合 | 最佳 α/β | 数据集精度 | far 跳层 | 速率 |
|---|---|---|---|---|
| mavg=(minmax,avg) | α=0.125 β=0.375 γ=0.125 | hotpotqa 54.43 / musique 34.76；13任务全量 AVG 50.54 | 未启用（E5b：静态跨任务掉分→需 per-task 校准；E6 静态版可跳 13/36 层 far 损失<0.3%） | ~16.3 min/样本(e2e 含 prefill) |
| aavg=(avg,avg) | α=0.125 β=0.25 γ=0.125 | hotpotqa 54.72 / musique 32.82 | 未启用（E5b：静态跨任务掉分→需 per-task 校准；E6 静态版可跳 13/36 层 far 损失<0.3%） | ~9.4 min/样本(e2e 含 prefill) |
| mminmax=(minmax,minmax) | α=0.375 β=0.375 γ=0.125 | hotpotqa 53.27 / musique 34.14 | 未启用（E5b：静态跨任务掉分→需 per-task 校准；E6 静态版可跳 13/36 层 far 损失<0.3%） | ~9.5 min/样本(e2e 含 prefill) |
| mavg=(minmax,avg)[B7s] | α=0.125 β=0.25 γ=0.125 | hotpotqa 54.23 / musique 32.82 | 未启用（E5b：静态跨任务掉分→需 per-task 校准；E6 静态版可跳 13/36 层 far 损失<0.3%） | ~16.3 min/样本(e2e 含 prefill) |
| cavg=(cluster,avg) | α=0.125 β=0.375 γ=0.125 | hotpotqa 53.43 / musique 32.31 | 未启用（E5b：静态跨任务掉分→需 per-task 校准；E6 静态版可跳 13/36 层 far 损失<0.3%） | ~11.6 min/样本(e2e 含 prefill) |

**best arm: mavg**

- trace 口径（E64j mass，B_TOK=2048 宽预算）：mono mavg 0.9166 > mminmax 0.9116 > ccluster 0.9096 > cavg 0.9033 > mavg 0.9012 > aavg 0.8036——与 e2e F1 排名不一致，快筛判决以 e2e 为准（trace mass 排名不可替代 e2e 判决）
- kernel 速度资产（与 method 无关的公共开销）：L1 fused 3.6×/L2 级联 1.63×（E8-2）
- ccluster 仅 trace 评估；如 trace 显著最优需补 e2e near-cluster 参数化