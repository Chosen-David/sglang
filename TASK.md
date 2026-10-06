注意：此文件不允许被claude以及其他AI修改，作为本人记录探索进度和设定任务的工具，这点应该写入记忆避免后续误动该文件，自动对于任务编排任务链（先评估各个任务的耗时优先排耗时短可以出结果的任务链），当存在任务链的时候设定监督器每隔1h监督一次直到任务链全部完成。本文件产出的可视化结果都放在/home/wangyuanshuo02/sglang/ref/figs下面

提示：从10.1晚上12:00开始，运行一个每隔6h自动运行的论文优化轮询监督器，对于已经完成的论文对照一篇顶会论文（不局限于本地，你也可以去网上搜和查找最新的顶会论文）进行润色和逐句对照，不断问自己：
1. 我的表述是不是可以优化和更合理更贴近论文
2. 我的架构图能不能更加精致就像这些顶会论文一样
3. 我的数据可视化之类的能不能更加美观，能不能像这些顶会论文一样把可视化表现的十分精美
4. 启动一个读者的agent，让他读你生成的论文PDF来看有没有从基础（比如可视化文字拥挤，架构图线条错位，文字表述混乱等问题）到高级（这个图和文字的表述真的对的上吗，比如架构图是不是能真的画的反映出你的设计了呢，这个可视化或者表的结论是不是和文字表述一样呢），然后给出读者意见，然后你再针对读者意见迭代新的版本（必要的话写代码补充实验）
5. 完成读者审稿后启动一个审稿人agent，让他扮演审稿人以批判的视角结合最新的indexer的论文来看这篇论文的这些创新点并且提出批判性的意见（模型选的怎么样，数据集合理吗，结论合理吗，实验设计的合理吗，是不是有没有分析到的点，难点分析的合理吗等意见）。然后你针对这些意见不断迭代新的版本（必要的话写代码补充实验）
6. check问题是不是都收尾了，然后保留这次论文优化的结果为独立的文件夹（里面包含源文件以及中英两版本各自的PDF），然后等待下次轮询优化继续，每次完整走完这几步，这个轮询监督器只能我手动kill。


提示：有些非性能敏感性的可以并行的任务可以起CPU探索和跑出相关数据也行如果适合在CPU上进行的话。本机有两张GPU也可以每张跑一个探索这样（可以先判断会不会OOM还是说双卡DP可以更快，总之如果只是需要精度数据的话，怎么快怎么来）。每次完成一个任务链的任务记得commit并且push代码避免丢失工作内容，还有pull agent库的最新agent进行agent升级

提示：如果发现bug，fix完代码后记得先验证对不对，是不是fix对了，然后再开始执行编排好的任务链，重测之前错误的数据。如果是代码编写和文件表述上存在偏离先从性能精度各个方面评估一下除非你的方法全面优于我的文件内容表述（可以提出你的idea和我商量）否则按照bug处理

提示：论文放在/home/wangyuanshuo02/sglang/paper下面，记得里面的图不一定真的都需要，然后对于需要的图一定要美化，参考其他顶会论文的作图美学，使用画图（比如架构图等其他原理图，然后记得留draw.io源文件），图表格制作的skill来尽量美观的表述。论文的表述也调用相关的skill来做

提示：不能光看atten mass，不一定代表e2e精度，记得写入记忆，记得实测后再下结论，反正你就老老实实测试就行。不要临时起意偷懒不测试完我这里列的全部


探索试点配置： EXPLORE_DATA = 等待论文调研结果出来看看其他论文用的什么方法
avg_method: 也即粗筛使用k_avg，然后q和k_avg算分作为粗筛分数然后对于选中部分细筛
minmax_method: 粗筛使用k_minmax，然后拆分q为q_neg, q_pos和k_minmax算分作为粗筛分数然后对于选中部分细筛
cluster_kmeans_method: 粗筛分块用kmenas方法，每个簇用算术平均得到代表k，然后q和聚类的代表k算分排序直到召回到mid_token个，参考clusterkv的论文实现
cluster_sim_greedy :配置sim（按照dump的轻量数据确定一个值，然后在多个数据集都可以生效那种 - 待确定），对于k按照增量贪心聚类（余弦相似度达到sim的就聚类到一起）每个簇用算术平均得到代表k，然后q和聚类代表k算分排序直到召回到mid_token个
cavg_method: (far_method, near_method) = (cluster, avg)
mavg_method: (far_method, near_method) = (minmax, avg)
aavg_method: (far_method, near_method) = (avg, avg) 
ccluster_method: (far_method, near_method) = (cluster, cluster) 
mminmax_method: (far_method, near_method) = (minmax, minmax)

探索完每个组合method的最佳配置后，在这个最佳配置上对比这几种method组合的得分看谁最高呢，注意不是atten mass的分析，是需要真正的去跑去测试数据集的精度来比较的，然后出一个表格和对应的可视化图

method（是什么method组合，如果是avg这种非组合method就直接写avg这种） ｜ 最佳的alpha beta配置分别是多少（如果是avg这种非组合method就填 - ） ｜ 测了哪些数据集是什么以及这些数据集的精度是什么 ｜ 哪些层far可以跳过噪声范围内不影响精度并且这里已经跳过了  ｜ 性能速率怎么样


到找到最佳配置后针对这个best_method进行降维探索，training-free或者引入训练，最后得到和其他论文的开源工作比较看看能不能打得过（我们的best_method和其他人工作对比），形成表格和可视化图：

方法 ｜ 测了哪些数据集和这些数据集的精度 ｜ 性能效率是多少
ours(我们的best_method+降维) ｜  测的精度是多少 ｜ 我们的性能效率
其他论文的方法 | |



用于测试让渡策略或者砍预算策略

bp: budget_page_topk的简写
abg: (alpha, beta ,gama) = 什么数值
ab: 不配置gama，仅考虑alpha beta数值是多少

# 概述

除去sink和swa区域（sink是最初始的区域sink_L个token，swa区域是当前算的q的滑窗区域swa_L个token），剩下的是mid区域的mid_L个token（然后near是mid区域中靠近q的，far是离目标q远的其余部分），值得注意的是sink和swa和我们的工作是正交的不作为我们的创新点，我们核心关注的是mid区域（这两个区域的保护token是必选的而且自动从预算里面排除的，比如我细筛topk2048的话这里就是2048 - sink_token - swa_token这两个固定预算，剩下的预算mid_token是给我们的可以创新的点，也即从mid_L里面筛选mid_token个，sink和swa的token从粗筛到细筛阶段都不会碰到这里面的token的，他们直接保送直接计算了）

举个例子，比如我sink_L=1, swa_L=2, near_L=3, far_L=4，然后组成为
k0, k1, k2, ..., (k9, q9)
其中q10是当前算的q，则k9,k8是swa区域，k0是sink区域，near区域是k5,k6,k7，far区域是k1,k2,k3,k4


先对于mid区域分成near和far两个区域，然后找子维度空间（比如仅仅用nope维度），然后对于子空间降维，降维后分别区域粗筛+细筛得到最终结果

值得注意的是，两个区域都是需要完整的粗筛和细筛，对于自己掌控的near_L,用near_method粗筛 near_budget_page_topk个page
然后对于自己掌控的far_L,用far_method粗筛 far_budget_page_topk个page（参数alpha控制负责的长度，beta控制多少个page）
然后对于所有 budget_page_topk个page（far的和near的）,qk细筛精细算细筛，near区域细筛near_token个，far区域细筛far_token个（参数gama控制，也即最终筛出的token，注意扫描的时候需要去除明显不合理的情况，比如near_L>=near_budget_page_topk*page_size>=near_token）这个图我需要你扫描出来，找出各个method的最优alpha,beta, gama组合，需要数据，注意不只是atten mass，还需要实际数据的精度

还有一种方法的探索，就是far区域或者near区域不再用两级筛选，直接用top-sigma，也即直接细筛得分达到sink token的 qk得分的max - sigma之后就直接选中该token
需要探索near far区域不同method下，一种两级筛选+top-sigma的混合方法，或者两种都是top-sigma的方法，注意不只是atten mass，还需要实际数据的精度


补齐所有的实验和数据，并且给我结论，最后需要看到这种表格

method组合（比如aavg）｜ (alpha, beta, gama) ｜ 精度咋样，速度咋样




methdo组合（比如avg+top-sigma）｜ （alpha, beta, gama） | 精度咋样，速度咋样


还有关于子空间选择器的创新点的探索，怎么选择？为什么这样设计可以work为什么其他空间的信息可以丢掉，关于降维目前各个方法对比的数据需要给我


“把预算花在哪”(near/far分区 + alpha/beta配比策略 + D'跳过far区域策略) 
“谁来做粗筛”(方法矩阵) 
“在哪个子空间做”(降维探索)，

更进一步的，能不能每一层单独的算最佳alpha beta，等你prefill进去的时候，快速得到每一层的大致最佳alpha和beta配比以及哪些far层能不能跳过（或者prefill不跳decode跳，或者直接两阶段都用上，因为不同层的也可能不一样最佳配比。然后消融时候证明你的预估器是准确的也即和真实最佳精度差不多）看看争取替换全层固定比例。然后这个alpha beta参数计算器应该被证明是几乎不影响性能的，或者可以overlap或者很轻量这样，看看能不能编入任务链探索设计一下，具体到每层。然后测试性能和精度

observation怎么写？ 为什么near far分区弄不同的method组合有效？你需要解释为什么比单一的method有效，需要结合实际的分析，可以调研其他论文的observation怎么做的
你的alpha, beta配比上，为什么预算值得让渡，之前的方法浪费预算了吗还是怎么了，调研设计一下这个创新点的observation怎么写
你的子空间和降维上，怎么写你的观察
注意你的observation不是单纯列一下数据，然后告诉别人你的方法好，还应该进一步延伸出更多的规律，为什么好，是不是各个模型通用的

选择器可以尽量上，我感觉正是因为接近所以说明预测器算的准，说明比暴力枚举好呀，而且你暴力枚举不能面对新的变化的数据集啊，而且这样ablation环节也有的写了说明选择器选的和枚举后的等效而且在占用总耗时很低百分比的情况下性能提升巨大。如果现在的预测器算的每层的alpha和beta不准得分析原因，比如第一次chunk时候dump atten mass来近似划分alpha beta这样？你是说每层的话分alpha beta意义不大？可是你dump每层的atten mass的话明明层之间的atten mass分布和最终选择token的分布明明还是有很大的区别的我的意思是理论上每层固定的alpha beta感觉并不是可以最优适应不同层的情况    

你的工作对比其他论文工作的精度和性能提升怎么样，在哪些数据集上比的，记得拉取他们的开源代码真实的对比延迟和精度在各个数据集上，最终论文应该和其他人的工作比


最终near区域的method为：
far区域的method为：






实现精度 打过baseline
速度 打过baseline
end2end 打过baseline

其中baseline应该包含：Quest，clusterKV，MoBA这些论文的数据以及真的拉取他们的源码跑出baseline

消融：
你的论文的消融实验，各个子创新点如果被消除会有什么影响，method组合上或许可以列出你暴力枚举的各个method？（参考其他论文的消融实验）
far, near分区方法确实有用
method上，确实far, near按照这样配置最优
alpha, beta确实按照论文设置收益最大
alpha beta的设计上能不能设计一个alpha beta快速计算器得到精确的alpha beta值，这样就可以变成动态感知型的了？然后消融一下看看你的alpha beta感知器是不是真的有效呢

kernel的设计上，为什么这样设计可以实现性能的极致优化
降维上：各种降维方法怎么样，自己训练的降维矩阵怎么样对比然后哪种method哪种降维方法最优

# 实验

你的模型用的什么？我的意思是参考其他indexer的论文的实验部分，他们的创新点是哪些，论文实验部分是用的什么模型在哪些数据集上实验的，结果怎么样，你应该调研实验部分放入/home/wangyuanshuo02/sglang/ref/paper下面给我看，并且产出如下的表格





# near和far的配比问题



比如 目标q，他先前对应的L个token，除去skink_L（最开始的token）和swa_L（离q最近的滑窗token）已经被保护起来必须选了，剩下的mid_L才是真正indexer两级筛选需要考虑的

alpha代表 near_method处理的长度占比： near_L = alpha * mid_L，剩下的作为far_L = mid_L - near_L 用far_method处理
beta代表 near_method的预算占比： near_budget_page_topk = budget_page_topk * beta，
剩下的是far_method的预算 far_budget_page_topk = budget_page_topk - near_budget_page_topk

进一步的，
near_budget_token = near_budget_page_topk * page_size
far_budget_token = budget_token - near_budget_token

注意约束：

near_budget_token <= near_L
far_budget_token <= far_L

其中budget_page_topk代表粗筛阶段筛选TOPK budget_page_topk个block进入细筛

更近一步？设置细筛也是单独细筛这样？ near_budget_token = near_budget_page_topk * page_size * gama 和 far_budget_token = budget_token（比如我细筛topk2048的话这里就是2048 - sink_token - swa_token这两个固定预算） - near_budget_token

near和far分别在自己的分区里面筛选自己的token




## alpha=beta的设置

最简单的方法是alpa beta同预算，在 (far_method, near_method) = (cluster, avg) 下测试得到结果为：



## alpha < beta的设置

考虑一种情况，也即far部分的token虽然可能数量多但是需要的预算far_budget_page_topk并不一点要很多才能达到情况这种情况下
### 预算让渡策略
可以控制beta>alpha也即把far的部分预算让渡给near区域（可以作为新的method编入任务链）



far让渡给near区域

near让渡给far区域


### 直接砍far预算
还有一种性能做法是直接取消far部分的预算，这样可以提升速度（比如可能某些层的far部分预算取消只需要算near的话精度已经很高了）

砍预算的话目前是有D' gate策略，但是待探索一个新的动态gate而不是静态的跳过far部分

这个或许可以做成感知型的？也即测完一条数据后的prefill阶段就得到哪些层可以跳过far或者跳过near不算。然后应用到decode上进行加速



### 同method的测试方法呢
之前测试的 (far_method, near_method) = (cluster, avg) ,但是现在预算不同的情况下是不是可能精度有·提升哪怕是同method？
可以先dump一组数据测试一下结果（可以拿 EXPLORE_DATA 的测试看看效果呢）
我的意思是 EXPLORE_DATA下
(far_method, near_method) = (avg, avg) 对比 avg_method
(far_method, near_method) = (max, max) 对比 minmax_method



此外你需要设计实验说明找到哪个配比哪个配置才是精度，速度（需要考虑上ako优化后的极致性能版本）综合考量的最优解：

测试以下method组合

aavg
mminmax
ccluster_sim_greedy
cavg
mavg


# (alpha, beta，gama)不同配比下的 


比如alpha beta gama以0.125为步长从0到1的不同组合配置都需要测试得到不同数据集下的精度数据这种，其中
ab 1,1 全部near_method
ab 0, 0 全部far_method
ab 1, 0 没意义因为near负责很多但是没有选择权利
ab 0, 1 没意义因为far负责很多但是没有选择权利

此外不满足约束
near_budget_token <= near_L
far_budget_token <= far_L
的数值也是没有意义的


搞出9*9的数据然后找到设计合适的可视化图进行可视化，atten mass的图（CPU就可以采集），精度图

看完测完后分析，
1. 是不是不同method组合反而不如单method （比如(alpha beta)= (1,0)这种最大）？ 以此判断需不需要不同method组合
2. 同method组合里，不让渡预算反而更好（也即alpha始终=beta的时候值最大）？ 以此判断需不需要让渡预算
3. 某些层far分区直接被砍掉后（等于说只在near_L区域粗筛near_budget_token个然后在这里进行topk2048）是不是会掉精度？（这个在前两项探索完后找到最佳配置（method1, method2, alpha, beta）后在这个配置下进一步探索砍预算策略，其中method1和method2可以相同也可以不同） 以此判断值不值得砍预算
4. alpha, beta如果值得上的话，能不能有快速计算alpha,beta的方法比如快速算下atten mass（需要看atten mass的评估方法是不是合理，和你实际测出的最佳alpha beta是不是精度差异不大），比如写个轻量小kernel或者CPU端overlap一下？

最好全面一点包括atten mass图的采集和实际数据集跑出精度结果，
通过这些判断near far分区的方法是不是真的可行，如果可行的话具体应该怎么设计，不可行的话原因是什么    




# 关于cluster的方法

多探索不同的方法做消融，不一定是kmeans，比如增量贪心聚类可能更好，配置设置成sim = 0.9， sim_dims设置成nope维度或者被压缩后的维度，算术均值簇心这样或许精度上更高



# 降维探索

nope维度进行 - 探索结论 可用几乎无损
能不能对于nope进一步降低维度？粗筛上，PCA?其他降维度方法？

能不能细筛也可以降低维度？（性能优化重要一步）

需要数据表格和可视化图 - 待数据

比如near_method能不能降维，far_method能不能降维这种(或者自己训练一个轻量的投影矩阵？)


## 关于training free的探索

### PCA降维

目前可以降维到多少，性能提升多少

### t-sne降维

### umap降维

### 其他最新论文的降维方法

待探索

### 其他降维方法


## 关于训练一个轻量投影矩阵

你可以粗筛细筛都引入训练后降维试试呢，注意你训练出的应该在各个数据集上都可行，记得参考其他人相关论文的投影矩阵训练方法，先调研再设计再写代码 


### sup_wsvd（注意力概率加权 PCA — 闭式解）
一个 analytical 闭式解


```
k_0, k_1, ..., k_{t-near_len-1}  ← 当前 q_t 视角下的 far 区 token 序列
                                  (sink / near 段已排除)
q_t                               ← 当前第 t 步的 query vector
```
sup_wsvd

step1: 校准集收集（离线）

拿到 calibration trace（如 hotpotqa_0），里面有完整的 q/k 表示。取所有 far 区的 (q, K) 对。

Step 2: 计算注意力权重 A
对校准集里每一个 far 区位置 t，算完整的 softmax 概率：

```
score_i = q_t · k_i / √D      for i ∈ far
p_i = exp(score_i) / Σ_j exp(score_j)
```

p_i ≡ "第 i 个 far key 对当前 q 的重要程度"


### 其他训练方法


# 实验数据

## 方法组合 × (α, β, γ) — LongBench 13 任务全量 e2e + 速度

数据来源：`two-level-attention/exp/trace/results/e71_main_table.json`, `e98_full_13tasks.json`, `e98_best_election.json`

| 方法组合 | 全称 | α | β | γ | LongBench AVG | HotpotQA | Musique |
|---------|------|:-:|:-:|:-:|:--:|:--:|:--:|
| **mavg** | (minmax, avg) 预算让渡 | 0.125 | 0.375 | 0.625 | **50.78** | **55.44** | 34.71 |
| **mavg** | (minmax, avg) E72 最优 | 0.125 | 0.375 | 0.125 | 50.54 | 54.43 | **34.76** |
| **mavg** | (minmax, avg) B7s 基线 | 0.125 | 0.25 | 0.125 | 50.41 | 54.23 | 32.82 |
| **aavg** | (avg, avg) 同方法分区 | — | — | — | 50.54 | 54.72 | 32.82 |
| **mminmax** | (minmax, minmax) | 0.375 | 0.375 | — | ~50.3 | 53.27 | 34.14 |
| **cavg** | (cluster, avg) | 0.125 | 0.375 | — | ~49.9 | 53.43 | 32.31 |
| **C0** | 单池 minmax (不分区) | — | — | — | 50.16 | 54.93 | 29.74 |
| **TLI_old** | 无分区无让渡 | — | — | — | 46.83 | 53.96 | 21.07 |
| FullKV | 全注意力 baseline | — | — | — | 50.36 | 53.48 | 32.14 |
| Quest | 对比方法 | — | — | — | 47.72 | 45.74 | 27.25 |
| TIA | 对比方法 | — | — | — | 50.06 | 53.89 | 32.28 |
| TWI | 对比方法 | — | — | — | 47.86 | 48.98 | 23.60 |
| MoBA | E89 复现臂 | — | — | — | 49.19 | 51.02 | 29.74 |

除了mavg之外其他的method组合没有扫描全量，还有对比方法上缺少clusterKV等其他开源论文方法的数据呢

## Top-σ 混合臂 — Screen 精度 + MACs/Token

数据来源：`e87_e2e_screen.json`, `e87c_signear8_tail.json`, `final_table_e97.json`

> ⚠️ **数据缺口**：Top-σ 仅测了 HotpotQA + Musique (2/13 任务)，未跑全量 LongBench

| 方法 | 混合模式 | σ | HotpotQA | Musique | MACs/Token | 选中 Token | 数据来源 |
|------|---------|---|:--:|:--:|:--:|:--:|------|
| twolvl | baseline B7s tail32 | — | 54.23 | 32.82 | 76,892 | 2,048 | E71 |
| twolvl | E90 full128 | — | 54.72 | 32.82 | — | — | E90 |
| **near_σ8** | `--tli_sigma_select near` | **8** | **55.12** | 31.83 | 130,255 | 1,301 | E87 e2e |
| near_σ2 | near 区 σ + far 两级 | 2 | 54.40 | 31.14 | — | — | E87 e2e |
| near_σ32 | near 区 σ + far 两级 | 32 | 54.92 | 31.58 | — | — | E87 e2e |
| near_σ8 tail | tail32 口径 | 8 | 55.08 | 31.19 | — | — | E87c |
| far_σ8 | far 区 σ + near 两级 | 8 | 54.31 | 30.09 | 635,809 | 1,973 | E87 e2e |
| far_σ32 | far 区 σ + near 两级 | 32 | 52.48 | — | 635,809 | 2,287 | E87 e2e |
| mid_σ8 | 全区 σ 不分区 | 8 | 53.50 | 31.09 | 689,172 | 1,225 | E87 e2e |
| mid_σ32 | 全区 σ 不分区 | 32 | 53.48 | — | 689,172 | 1,574 | E87 e2e |
| rope64 | 子空间 e2e 最优口径 | — | **55.98** | 32.96 | — | — | E90 |

**Top-σ 维度效应分析**（E87c 判决）：

| 臂对比 | hq delta | mu delta | 结论 |
|--------|:--:|:--:|------|
| twolvl full128 − tail32 | +0.49 | 0.00 | 维度效应大：+0.49 纯来自 4× 细筛维 |
| near_σ8 full − tail | +0.04 | −0.64 | 维度效应极小：top-σ 锚与打分维度弱耦合 |
| near_σ8 tail vs twolvl tail | **+0.85** | −1.63 | σ 选择净增益确认，mB 仍在 |

**Top-σ 结论**：
- HotpotQA 上 near_σ8 有净增益 (+0.85)，但 Musique（多跳 far 检索任务）全线下降 (−1.63)
- Top-σ 维度效应远小于两级筛选，天然对降维鲁棒
- ⚠️ 13 任务未全量，不能纳入主表，需补跑


只测试了这两个任务是吗，然后就取消这个方法了是吗

## E98 全部 e2e 精度网格 (hq+μ/2 排序，n=200)

hq+μ/2 是什么意思，为什么这里又来一个表格，上面那个表格不是有数据了吗

数据来源：`e98_e2e_grid.json`

| 排名 | 配置 | mass | hq | μ | hq+μ/2 | vs B7s |
|:--:|------|:-:|:-:|:-:|:--:|:--:|
| 🥇 | mavg α.125/β.375/γ.625 | .880 | 55.44 | 34.71 | **45.08** | **+1.56** |
| 🥈 | mavg α.125/β.25/γ.75 | .882 | 54.16 | 35.57 | 44.86 | +1.34 |
| 🥉 | mavg α.125/β.25/γ.375 | .881 | 54.16 | 35.57 | 44.86 | +1.34 |
| 4 | mminmax α.875/β.875/γ.5 | .896 | 54.40 | 33.40 | 43.90 | +0.38 |
| 5 | aavg α.875/β.875/γ.375 | .796 | 54.40 | 33.40 | 43.90 | +0.38 |
| 6 | mminmax α.75/β.75/γ.5 | .895 | 54.38 | 33.24 | 43.81 | +0.29 |
| 7 | aavg α.75/β.75/γ.5 | .794 | 54.38 | 33.24 | 43.81 | +0.29 |
| 8 | aavg α.875/β.75/γ.5 | .796 | 54.13 | 33.05 | 43.59 | +0.07 |
| 9 | cavg α.125/β.25/γ.75 | .879 | 54.13 | 33.03 | 43.58 | +0.06 |
| 10 | mminmax α.625/β.625/γ.5 | .895 | 53.88 | 33.04 | 43.46 | −0.06 |
| 11 | cavg α.125/β.25/γ.375 | .877 | 54.13 | 32.50 | 43.31 | −0.21 |
| 12 | cavg α.125/β.375/γ.625 | .879 | 53.43 | 32.20 | 42.81 | −0.71 |
| ref | mavg α.125/β.375/γ.125 | .873 | 54.43 | 34.76 | 44.59 | +1.07 |
| B7s | mavg α.125/β.25/γ.125 | .882 | 54.23 | 32.82 | 43.52 | — |

**关键发现**：Mass vs E2e 倒置 — mass 最好的 (mminmax .896) e2e 排第 4；mass .880 的 mavg 反而 e2e 最优 (45.08)

## Kernel 速度 benchmark

数据来源：`exp/results_efficiency/benchmark_mha_20260923_124217.json` (batch=4, GQA 4:1, dim=128)

| SeqLen | MHA (ms) | Sparse MHA (ms) | 加速比 | 吞吐提升 |
|:------:|:--------:|:---------------:|:------:|:-------:|
| 32K | 0.98 | 0.67 | **1.46×** | +46% |
| 65K | 1.95 | 0.70 | **2.77×** | +177% |
| 128K | 3.91 | 0.77 | **5.09×** | +409% |

## Mass Coverage 数据 (trace-level, CPU 采集)

数据来源：`e87_topsigma_merged.json`

| Task | twolvl | far_σ8 | near_σ8 | mid_σ8 | far_σ32 | near_σ32 | mid_σ32 |
|------|--------|--------|---------|--------|---------|----------|---------|
| gov_report | 0.823 | 0.691 | 0.831 | 0.699 | 0.719 | 0.844 | 0.740 |
| hotpotqa | 0.907 | 0.760 | 0.862 | 0.715 | 0.769 | 0.865 | 0.726 |
| musique | 0.888 | 0.724 | 0.869 | 0.705 | 0.735 | 0.871 | 0.718 |
| narrativeqa | 0.831 | 0.681 | 0.825 | 0.675 | 0.719 | 0.828 | 0.715 |
| passage_retrieval | 0.921 | 0.771 | 0.893 | 0.743 | 0.782 | 0.896 | 0.756 |
| qasper | 0.887 | 0.769 | 0.853 | 0.734 | 0.778 | 0.855 | 0.745 |

规律：near_σ ≈ twolvl（near_σ32 微超 twolvl），far_σ/mid_σ 远低于 twolvl — far 区 top-σ 噪音穿透严重

## Sigma 校准扩展 (e87b)

数据来源：`e87b_sigma_calibration_s0.json`

| Task | noise_k1 | noise_k2 | mid_σ256 | near_σ256 |
|------|:--:|:--:|:--:|:--:|
| gov_report | 0.932 | 0.937 | 0.913 | 0.842 |
| hotpotqa | 0.963 | 0.966 | 0.909 | 0.853 |
| multifieldqa | 0.965 | 0.967 | 0.896 | 0.817 |
| musique | 0.957 | 0.960 | 0.898 | 0.856 |
| narrativeqa | 0.923 | 0.930 | 0.904 | 0.858 |
| passage_retrieval | 0.961 | 0.967 | 0.917 | 0.846 |

Noise floor (k=1 random) 全在 0.92–0.97 区间，验证 top-σ 在 far/mid 区的 mass 大量来自噪声穿透

## 分区 verdict (E88)

数据来源：`e88_partition_verdict.json`

| 预算 | mono mass | part mass | delta | oracle | 分区胜率 |
|:--:|:--:|:--:|:--:|:--:|:--:|
| 256 | 0.8233 | 0.8206 | −0.0027 | 0.8262 | 1/16 |
| 512 | 0.8549 | 0.8520 | −0.0029 | 0.8574 | 0/16 |
| 768 | 0.8754 | 0.8716 | −0.0038 | 0.8772 | 1/16 |
| 1024 | 0.8902 | 0.8866 | −0.0037 | 0.8915 | 1/16 |
| 2048 | 0.9271 | 0.9236 | −0.0035 | 0.9265 | 0/16 |

结论：同预算下 mono mass 略优于 partition，**分区 e2e 增益来自 β 让渡 (预算不均价) 而非分区本身**




## 最终结论

1. **mavg 全面最优**（minmax 粗筛 far + avg 粗筛 near + β 让渡），γ 细化让渡 (+1.56) > αβ 让渡 (+1.07)
2. **β 让渡有显著增益**：β .25 → .375 带来 HotpotQA +1.28
3. **Mass ≠ e2e 反复验证**：mminmax mass .896 最高但 e2e 排第 4
4. **Top-σ 在 HotpotQA 有净增益 (+0.85)，Musique 全线下降** — far 检索仍需两级
5. **Top-σ 维度效应极小**，天然对降维鲁棒
6. **分区本身 mass 略负 (−0.003)**，e2e 增益来自 β 让渡非分区
7. **速度随 seq 长度线性增长**：128K 处 5.09×