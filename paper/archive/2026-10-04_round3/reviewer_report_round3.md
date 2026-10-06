# PSI 论文 round3 审稿人级批判报告（E98 数据全量重写后首次）

- **审读对象**：`/home/wangyuanshuo02/sglang/paper/TLI_paper.tex`（中文 19 页）+ `TLI_paper_en.tex`（英文 20 页）
- **审读日期**：2026-10-04
- **方法**：tex 源逐行通读 + pypdfium2 文本层抽验（tofu=0，关键数字 50.78/0.787/0.166/45.08/87.93 等全部落袋）；表格算术全量复算；本会话无图像识别，图仅查 caption 与正文引用一致性。外部先验工作检索不可达（连接黑洞），novelty 判断基于内部知识。
- **前提**：round1/round2 审稿意见针对旧版（主表 50.54 臂），本轮针对重写后版本（50.78 臂、双口径网格选举叙事），旧意见不重复收割，只收割新叙事带来的新问题。
- **总体印象**：重写后诚实度与可验算性显著提升（表格算术全部自洽、负结果入册、口径注齐全），但换主表臂引入了三个新的结构性弱点：**被评测系统与被描述系统的 L1 口径不一致**、**选举偏差直接暴露在 headline 数字上**、**速度卖点在新叙事下反而更突出地失守**。

---

## 一、CRITICAL（reject 级）

### C1. 被评测系统 ≠ 被描述系统 ≠ 被测速系统：主表臂 L1 是全维 128 无量化，方法节卖的是 tail32-4bit

- **位置**：方法 §3.2（中文 p7-8）vs 实验 §4.2 首段（中文 p11）vs §3.4/§4.4（kernel 与存储主张）
- **审稿人原话模拟**：*"The Method section (§3.2) defines L1 as a minmax upper bound over a 4bit-quantized d'=32 subspace (40B/token-head), and every speed and storage claim — the 336B/token index, the fused 4bit-dequant L1 kernel (3.6×), the 1/32 MAC argument vs DSA — is predicated on that design. Yet §4.2 states the main-table arm computes the L1 bound on the full 128 dimensions without quantization. So the quality tables (50.78) are produced by a system that is roughly 4× more expensive in L1 than the system whose speed you benchmark and whose storage advantage you claim. The paired calibration you offer covers only two tasks on the γ=0.125 reference arm, where musique drops by 1.54 under tail32 — the same order as your +0.42 headline. Please rerun the full 13-task main table with the production configuration."*
- **根因**：主表臂沿用了 E87c 校准后的全维 L1 口径（L1 上界全维 fp），而方法/实现/速度/存储四处的叙述全部绑定 tail32-4bit 设计。E87c 双臂校准（hq 54.83/54.43、mu 33.22/34.76）本意是证明「不敏感」，但 musique −1.54 的方向反转 + 只在 2 任务参照臂上做，撑不起「主表结论对 L1 维度不敏感」这句话——若把 −1.54/2≈−0.5 的净效应外推到 13 任务，+0.42 可能直接归零甚至翻负。
- **危害链**：质量数字（全维 L1 系统）、kernel microbench 0.787ms 与 1.46×→5.09×（tail32-4bit 系统）、存储 336B/token（tail32-4bit 系统）三者分属两个不同系统，审稿人一旦发现（几乎必然发现，§4.2 自己写了），全部速度与存储主张的可信度被连带质疑。
- **修复建议**：**补实验可解（优先级最高，约一次 13 任务全量过夜跑）**——用生产形态（tail32-4bit L1）重跑 LongBench 13 任务主表；若 AVG 掉幅 ≤0.2 则主表换数 + 保留全维臂为「质量上界参照」（类似 rope64 的处理）；若掉幅大则这是比论文更重要的工程发现，必须如实处理。写作上无论结果如何，§4.2 首段应显式加一句「方法节描述的生产配置与主表臂的 L1 口径差异如下，成对校准为……」的自我暴露式表述（现有写法已经接近，但「不敏感」结论措辞需降级为「2 任务成对校准，方向任务间反转」）。

### C2. +0.42 vs FullKV 的统计站不住：无重复实验无 CI，且 83% 的差分来自两个选举任务（in-sample）

- **位置**：摘要（p1）、贡献 #2（p4）、§4.2 LongBench 段（中文 p12-13）、消融「双口径参数网格」（p14）
- **审稿人原话模拟**：*"No repeated runs, no confidence intervals anywhere. Worse: the main-table configuration was elected by measuring 13 arms on exactly hotpotqa and musique, and the headline +0.42 margin decomposes as musique +2.57 and hotpotqa +1.96 — the two election tasks contribute 4.53 of the 5.46 total margin (83%). The remaining 11 held-out tasks net +0.93/11 ≈ +0.08, which is at the noise floor your own paper reports (median |Δ| = 0.17 across configurations). The expected maximum of 13 noisy draws on 2 tasks fully explains the +0.49 election gain. As written, the central accuracy claim is a multiple-comparison artifact."*
- **根因**：选举协议（2 任务筛 13 臂）与主表验证（13 任务）在 hotpotqa/musique 上重叠，没有任何 held-out 纪律；论文自己的 E75 噪声分析（12 非 musique 任务 |Δ| 中位 0.17）给出了现成的噪声地板，但没有被用来审视 headline 本身。
- **次生问题**：13 臂里选 max 再全量验证，论文虽然反复强调「选举只认端到端实测」的方法论自觉，但没有做任何对抗自身选择偏差的动作（无 held-out、无 Bonferroni/置换检验、无 seed 重复），方法论叙事反而把聚光灯打到了这里。
- **修复建议**：① **补实验可解**：≥3 个解码 seed 的主表重复，报 per-task 均值±std 与配对显著性（Wilcoxon 或 per-task bootstrap over samples，LongBench 每任务 200 条样本量足够做 per-task bootstrap CI——不需要重跑推理，只需对已有逐样本分数做重采样，**写作+轻量分析可解**）；② **写作可解（最低成本）**：在 §4.2 增加一行诚实分解——「+0.42 中 4.53 来自两个选举任务，held-out 11 任务净 +0.08，与参数面平坦性一致；headline 定位应读作『不输 FullKV』而非『超越』」，把摘要「超出 FullKV 0.42」降级为「与 FullKV 持平（+0.42，其中 held-out 任务 +0.08）」——这与用户铁律「老老实实测试如实报告」同向；③ 选举协议补一个声明：hotpotqa/musique 的主表数字为 in-sample，或改用 2 任务选举 + 11 任务确认的双段表述。
- **注意**：2wikimqa +1.03 与 narrativeqa +2.48（相对旧臂）是真实的 out-of-sample 正向信号，rebuttal 时应主打这两个。

### C3. 速度卖点三层失守，且缺 vs Quest/MoBA 的 e2e 层对比（审稿铁律缺口）

- **位置**：摘要末段（p1）、§4.4 kernel microbench（中文 p16）、§4.5 e2e 速度（p16）
- **审稿人原话模拟**：*"The paper positions speed as the primary selling point, yet: (i) the indexer kernel is third of three at 131K — 7.3× slower than Quest (0.787ms vs 0.107ms) — and the excuse 'implementation maturity, not algorithmic ceiling' is exactly what reviewers are trained not to accept; (ii) e2e speed exceeds dense only in the single-request 64K shape (1.285×), while 32K is 0.77× and the production-relevant TP2 bs16 shape is 0.92×; (iii) the 8B prefill is 1.60× slower than dense. There is no e2e throughput comparison against Quest or MoBA serving pipelines — the kernel-level three-way comparison is not complemented by an e2e-level three-way comparison, so the claimed speed advantage is never demonstrated against any competing sparse method end-to-end."*
- **根因**：跨请求批量化（已验证 3.8× 方向）尚未落地，导致生产形态的 launch-bound 短板直接进表；同时 e2e 对比只做了 vs dense，Quest/MoBA 只有 kernel 层与质量层对比——用户自己的铁律（kernel 与 e2e 两层对其他稀疏方法都必须测必须报）在 vs Quest 上只兑现了一半。
- **修复建议**：① **补实验可解（关键路径）**：落地跨请求批量化的 forward_extend kernel 化后，补 TP2 bs16 64K 的 e2e 数字（若 ≥1.2× 则速度叙事从「有形状边界」升级为「生产兑现」）；② **补实验可解**：Quest@1024 接入同一 sglang 管线跑 64K 单请求 e2e（Quest 官方有 sglang 集成路径），给出 PSI vs Quest 的 e2e 加速比——哪怕 kernel 慢 7.3× 但 e2e 赢（Quest 的 metadata 构建在 prefill 侧更贵），速度故事就成立；③ **写作可解**：摘要速度句重排为「先报 e2e 兑现的形状（64K 1.285×）、再报复杂度与存储主张、kernel 延迟第三如实报告」，删掉「PSI 的速度优势在……而非延迟」这种防守句式改为主动叙事；「131K 下三家都远离 HBM bound」这句保留（是好句子）。
- **不可解部分**：0.107ms vs 0.787ms 的 7.3× 差距短期只能靠批量化摊薄，若最终 e2e 仍不敌 Quest，速度卖点须彻底让位于存储/复杂度卖点（336B/token vs 1KB/token、1/32 MAC）——这是结构性风险，需在 Limitations 中提前定调。

### C4. PSI 在 RULER 上输给自家第一代 TIA（−0.42 总分、16K −0.85），论文从未明说；且双表两套配置

- **位置**：RULER 终表（中文 p11-12）、摘要（p1）、Limitations（p17）
- **审稿人原话模拟**：*"In Table 2, TIA@1024 scores 88.35 overall and 84.53 at 16K, versus 87.93 and 83.68 for PSI — the proposed full system is dominated by its own first-generation ablation on one of the two benchmarks. The text never states this; the narrative jumps to 'PSI approaches FullKV (88.71)' and 'leads Quest'. Moreover, the 'PSI' column in the two main tables is not even the same configuration (RULER: β=0.25/γ=0.125; LongBench: β=0.375/γ=0.625), so cross-benchmark statements like 'quality vs Quest +3.06 LongBench / +7.98 RULER' silently splice two different systems."*
- **根因**：分区让渡在 far-dense 合成任务上失效（multikey_3 TIA 89 vs PSI 80），这是已知机理（E74 判决），但正文选择了只对标 FullKV 与 Quest 的表述策略；「一组固定配置覆盖 12/13 任务」的部署简单性卖点（p9）与双基准双配置事实存在直接张力（Limitations 认了，正文卖点句没软化）。
- **修复建议**：① **写作可解（必做）**：§4.2 RULER 段加一句显式承认——「PSI 全系统在 RULER 总分低于同管线无分区的 TIA（87.93 vs 88.35），损失集中于 far-dense 任务（multikey_3 −9）；这是让渡机理在合成 far-dense 族的已知边界（E74），TIA 的全池形态恰在该族占优」——主动交代比被审稿人从表里抓出来好一个量级；② **补实验可解（强烈建议）**：用主表臂配置（β=0.375/γ=0.625）重跑 RULER-33，补齐「单一配置跨双基准」的空格（Limitations 自己承认未验证）——若主表臂 RULER ≥87.5，双配置问题整体消失，部署简单性卖点完整兑现；③ **写作可解**：两表 caption 已注明配置差异（做得对），但正文所有跨基准合并句（如 §4.4 的 +3.06/+7.98）须加「（两基准各自最优臂口径）」限定。

### C5. 最近邻 HISA 只引不比；Related Work 漏掉两级/分区形态的更早先例（InfLLM、Landmark Attention）——novelty 声明无实验支撑

- **位置**：引言 P3（p2）、相关工作「两级结构」（中文 p18）、贡献列表（p3-4）
- **审稿人原话模拟**：*"The paper concedes the two-level cascade form to HISA (COLM 2026) and claims its increment is the systematic characterization of two orthogonal dimensions (upper bound vs lower bound; partition vs single pool). But HISA is never run. A block-summary-then-refine hierarchy also appears in Landmark Attention (ICLR 2024) and InfLLM (2024, per-block units with two-level selection), neither of which is cited. And 'partition vs single pool' is precisely the sink+window+retrieval decomposition used by StreamingLLM-style hybrids and KV-compression methods with window retention. What remains as defensible novelty is the RoPE rotation-pair subspace mechanism and the offline-vs-e2e protocol-gap study — which is an empirical-analysis contribution, not a system contribution."*
- **根因**：novelty 定位实际压在「降维自由度刻画 + 口径鸿沟」上（这确实是全文最强、最独有的部分），但标题、摘要、贡献排序仍以系统（两级+分区）为第一身份，把最容易被攻击的部位放在了最前面。
- **修复建议**：① **补实验可解（若 HISA 有可用实现）**：HISA 接入统一 monkeypatch 管线跑 LongBench 13 任务——若 PSI 胜出则 novelty 主张闭环，若接近则「正交维度刻画」的定位反而更稳；② **写作可解（最低成本、必做）**：Related Work「两级结构」段补 Landmark Attention 与 InfLLM 两条 bibitem 及一句定位（Landmark 用块均值摘要、InfLLM 用块单元向量两级选取，均无上界保证与分区预算的刻画）；③ **写作可解**：贡献 #1（降维自由度）与贡献 #4（平坦性/口径鸿沟）在摘要中前置——把论文身份从「新索引器」调到「training-free 稀疏索引的设计空间系统刻画 + 一个实例」，这正是数据实际支撑的身份。

---

## 二、MAJOR（weak reject 级）

### M1. 质量评测长度与 131K 动机脱节

- **位置**：引言 P1（131K 带宽瓶颈叙事，p1）vs §4.2（RULER 止步 16K，LongBench 中位 ~9K token）
- **审稿人原话模拟**：*"The motivating bottleneck is a 131K-context decode, the kernel microbench runs at 131K, but every quality number tops out at 16K (RULER) or ~9K median (LongBench). The partition gain itself 'turns positive only at 16K' — so the regime where your method starts winning is exactly where your quality evaluation stops."*
- **根因**：RULER 32K/64K/131K 未跑（算力与时间）；「分区增益 16K 才转正 +0.91」的长度梯度外推到 131K 是论文最重要的隐含主张，但无数据点支撑。
- **修复建议**：**补实验可解**——RULER 32K（甚至 65K）一档、FullKV/PSI/TIA/Quest 四列即可，n=50 也行；若 32K 分区增益继续放大（+2 以上），全文最强卖点就有了第二数据点。若算力不允许，**写作可解**：在 Limitations 把「质量证据长度 ≤16K」与「速度证据长度 131K」的错位显式承认。

### M2. 质量在 8B、速度在 30B-A3B，同一模型上无 speed-accuracy 联合证据；8B prefill 慢于 dense 1.60×

- **位置**：§4.5（p16）
- **审稿人原话模拟**：*"Accuracy is established on Qwen3-8B and throughput on Qwen3-30B-A3B. No single configuration of model + method shows both the quality parity and the speedup simultaneously. And on the 8B model the prefill is 1.60× slower than dense — a regression in the prefill-heavy serving mix."*
- **根因**：8B 的 8 kv-head 使 gather 形态重（文中已归因）；但审稿人要求「同一模型上同时给精度与速度」的联合证据是常规标准。
- **修复建议**：**补实验可解**——30B-A3B 上跑一个 LongBench 子集（hotpotqa/musique/2wikimqa 三任务即可）质量对照，与现有 30B 速度数字拼成同模型双口径；或反过来在 8B 上报 64K 单请求 e2e。**写作可解**：把「8B prefill 1.60× 慢、kv-head 数是主变量」的归因升格为一个小表（8 头 vs 4 头同型对照），把弱点转化为「形态依赖已定位」的卖点。

### M3. kv-head 组内共享选择的近似代价未消融

- **位置**：贡献 #2（p4）、§4.4 存储主张（p16）
- **审稿人原话模拟**：*"PSI's 3× storage advantage over Quest rests on indexing 8 kv-heads instead of 32 query-heads, i.e., all 4 query-heads in a group share one selection. The quality cost of this sharing is never isolated. Given that you elsewhere show far mass is extremely uneven across heads, why is intra-group sharing free?"*
- **根因**：设计叙述用「跨 head 不均」论证 kv-head 级索引的必要性，但组内共享 vs 逐 query-head 选择的质量差没有单独消融臂。
- **修复建议**：**补实验可解（轻量）**——trace 重放口径做 per-q-head vs per-kv-head 选择的 mass 对照即可（不必 e2e 全量），若差 <0.005 则一句话封死；若差大则存储主张须重新表述。**写作可解（临时）**：至少在 §3.2 注明「组内共享的近似代价经 trace 重放量化为 X」——目前是空白。

### M4. mass 网格的 γ 截断坍缩自伤「3600 配置」叙事；13 臂精选标准 ad hoc

- **位置**：消融「双口径参数网格」（中文 p14）、摘要（p1）
- **审稿人原话模拟**：*"The offline grid 'scans 3600 valid configurations', but your own truncation analysis shows the γ dimension collapses under the e2e budget — γ arms that differ offline score bit-identically at e2e. So the effective configuration count at deployment scale is far smaller, and the 13-arm e2e curation ('deduplicated by effective budget, selected from quality-optimal configurations') is an undocumented ad hoc procedure. How were exactly these 13 chosen? Where is the list?"*
- **根因**：γ 截断是真实工程教训且已如实入册（值得表扬），但「3600 配置」的表述放在摘要里先声夺人、坍缩教训放在 p14 才揭底，叙述顺序制造了期望落差；13 臂清单全文未列表。
- **修复建议**：**写作可解**——① 摘要中「3600 个合法配置」改为「3600 个合法配置（其中 γ 维在部署预算下坍缩，有效配置见 §4.3）」或干脆改为「按端到端有效预算去重后的配置网格」；② 附录补一个 13 臂配置全表（(α,β,γ)×方法组合×双任务分），一页解决，rebuttal 期也用得上。

### M5. MoBA 与 Quest 的复现无「与原论文报告量级对齐」的声明

- **位置**：§4.2（p12-13，MoBA 统一 harness）、全文（Quest 无对齐声明）
- **审稿人原话模拟**：*"MoBA is rerun in your own harness with numbers well below its published results; Quest's RULER 16K collapse to 70.18 also differs markedly from the near-lossless behavior Quest reports at comparable budgets. Neither reproduction is validated against the original paper's reported magnitudes on any overlapping setting. How do I know the baselines are not mis-configured?"*
- **根因**：用户自己的纪律（baseline 复现精度须与其论文报告量级一致）未在论文中落实为声明句；尤其 MoBA 是自建 harness（官方 wrapper decode 回退 full-attention 不可比的理由成立，但复现数字因此更需要对齐证据）。
- **修复建议**：**写作可解（必做）**——① MoBA 段加一句「复现数字与官方论文在 [某重叠设置] 的报告量级对照为 X，偏差归因为 sink/swa 保送口径差异」；Quest 同理（Quest 原论文 RULER/LongBench 有公开数字，挑一个重叠预算点对齐）；② 若实在无重叠设置，注明「无重叠口径，复现协议见附录」并把预算/页大小配置全部入附录 B。

### M6. musique +3.08 vs TIA 与主表算术不符（陈旧数字残留）

- **位置**：消融「分区」段（中文 p14，英文 p15 同）
- **审稿人原话模拟**：*"The partition ablation claims 'musique +3.08 overtaking TIA', but Table 3 gives 34.71 vs 32.28 = +2.43. Which is it?"*
- **根因**：+3.08 是旧主表臂（E71/E72 世代）口径的陈旧数字，E98 换臂后未同步。全文数字我复算了一致，唯此一处打架——恰恰说明重写时逐句对照旧数据的流程有一处漏网。
- **修复建议**：**写作可解（必改，双语两处）**——改为 +2.43 或直接改写为「musique 34.71 对 TIA 32.28（+2.43）」；顺手全文再跑一遍「表数字 vs 正文数字」的机械对账（本报告已做了一轮，只发现这一处）。

### M7. 「首次」类 overclaim

- **位置**：§4.5（中文 p16）：「稀疏理论流量收益**首次**在 e2e 净兑现」
- **审稿人原话模拟**：*"'The first net cash-out of the theoretical sparse-traffic win at e2e' — Quest, MoBA, and HISA all report wall-clock e2e speedups in their papers. This claim is false as stated."*
- **根因**：想强调的是「本管线首次跨过盈亏平衡」，但句面写成了领域首次。
- **修复建议**：**写作可解**——删「首次」，改为「在本管线上首次跨过 e2e 盈亏平衡点」或「64K 单请求形态下净兑现」。同类扫描：摘要「为全部方法（含 FullKV）最高」是对的单任务事实，保留。

### M8. 双口径叙事的复杂度风险：自证预言式的指标信任外溢

- **位置**：§2.2 + §4.3 整段（占全文约 15% 篇幅）、fig5/fig11
- **审稿人原话模拟**：*"The paper spends a full subsection and two figures establishing that its own offline metric (mass coverage) does not predict e2e accuracy (ρ=0.166), then continues to use the same metric family to support the mechanism claims (rotation pairs, partition mechanics, gate correlations). Why should the reader trust the trace-based evidence for the mechanism if the trace-based evidence failed for configuration ranking? And is 'mass' a self-defined metric that no other paper uses — making the flatness and reversal claims unfalsifiable by outsiders?"*
- **根因**：这是「为证明不需精调花一整节」风险的深层形态——论文自己证明了离线口径会主动误导，却没有给出「哪些 trace 结论仍然可信」的完整边界（§4.3 末段给了「崩塌级可测、可用级不可排序」的二分，这是好的，但不够前置）。
- **修复建议**：**写作可解**——① 把「trace 可靠性二分」（能检测崩塌、不能排序可用级）从 §4.3 末段**前置到 §2 开头的方法论声明处**（现有一句「系统性 negative results 是本文观察方法论的一部分」，在其后加二分原则），让读者带着边界读全部 trace 证据；② 每处依赖 trace 证据的机制 claim（旋转对三重判决、gate corr 0.924、GQA 不均 0.804）补一个「该结论是否经 e2e 侧验证」的标注——五臂消融已覆盖子空间排序（e2e 同向），gate 有差分 e2e，GQA 不均只有 trace——不均衡处如实标注；③ mass 指标的「自造」质疑由 §4.1 的形式化定义（C 与 R_far 的分母明确）基本挡住，保留即可，但建议补一句与 SparQ/Quest 论文中类似 recall 口径的对照命名。

---

## 三、MINOR

1. **bib 卫生**（中文 p19）：`nsa`/`clusterkv` 两条 bibitem 正文零引用（thebibliography 环境仍会打印成「悬空参考文献」）；英文版 HISA 与 SparQ 条目缺作者（中文版 HISA 有完整作者列表，两版不一致）；SparQ/DSA/MoBA/ClusterKV 四条带「投稿前核对」TODO 注释——投稿前必须清零。英文版 bibliography 比中文版少 nsa/clusterkv（这反而正确），两版条目集应统一。
2. **+8.40 vs +7.98 双数字混用**（相关工作 p18 vs §4.4 p16）：相关工作用 TIA−Quest=+8.40 论证「上界家族」，microbench 用 PSI−Quest=+7.98——两个口径都对但相邻章节出现两个「上界 vs Quest」数字，审稿人会当成不一致。建议 Related Work 统一用 PSI 口径或明写「TIA 口径」。
3. **摘要超长**（中文 ~340 字单段、嵌套 4 层数字密度）：MoBA 四句式骨架被填成了数字清单，15+ 个数字。建议砍到 8 个以内，把「排序反转 ρ=0.166、亏 1.18、3600 配置、E87c 校准数字」这一层细节下沉正文。§4.2 LongBench 单段 600+ 字同理（MoBA/StreamLLM/β 交叉表三件事挤一段）。
4. **日期未随重写更新**：两版 `\date{2026-09-30}`，重写于 10-04。
5. **「12/13 任务一组固定配置覆盖」**（§3.3 p9）与 C4 的双配置事实直接张力——Limitations 已认，但正文这句卖点句未加「LongBench 口径」限定。
6. **β 交叉表粒度薄**（tab:betacross，2×2 四格承载「对角占优」结论）：四格差 ≤0.35 恰在自身噪声地板上，建议 caption 补 per-task bootstrap 的 CI 或至少注明 n。
7. **RULER 表内「PSI@1024」简称**：caption 已注明非主表臂配置，但表头简称与 LongBench 表头相同，交叉阅读时易误读为同一系统——建议表头改「PSI@1024(参照臂)」。

---

## 四、Top-3 reject 理由预测与 rebuttal 策略

| # | 审稿人 reject 理由（预测原文） | 对应问题 | rebuttal 可解性 |
|---|---|---|---|
| R1 | "The headline +0.42 over FullKV is a multiple-comparison artifact from electing on the same two tasks that drive the margin, measured once without any variance estimate — and the quality numbers come from a different L1 configuration than the system whose speed and storage are claimed." | C1+C2 | **高**：tail32-4bit L1 的 13 任务重跑（一夜）+ 逐样本 bootstrap CI（对已有分数重采样，无需重跑推理）+ held-out 分解句。若 tail32 重跑掉幅 ≤0.2，R1 整体消解 |
| R2 | "A paper whose stated selling point is speed is third of three in kernel latency, slower than dense in two of three e2e shapes, and never compares e2e throughput against any competing sparse method." | C3 | **中**：跨请求批量化落地 + TP2 e2e 重测 + Quest e2e 接入是硬前提；若批量化后仍不敌 Quest e2e，只能改卖点定位（存储/复杂度），属结构性风险 |
| R3 | "The system novelty over HISA is asserted, not measured — the closest prior work is never run, earlier two-level precedents are uncited, and the proposed system loses to its own ablation (TIA) on one of two benchmarks." | C4+C5 | **中高**：HISA 复现（若实现可得）+ 主表臂配置重跑 RULER（统一配置，双表同一系统）+ Related Work 补 InfLLM/Landmark + 贡献排序前移（#1/#4 升格）——其中三项写作可解，RULER 统一重跑约半天 GPU |

---

## 五、整体预测与提升优先级

**当前形态预测：weak reject（4/10 分位）**。数字可验算性、负结果入册、双口径诚实这三点是全场强项（我复算了 LongBench 全部 10 列×13 行、RULER 三档+16K 逐任务、8胜5负、+0.42/+0.62/+0.17/+0.91/+13.50 全部自洽——这在投稿论文里罕见），但 C1（系统口径不一致）与 C2（选举偏差）是任何称职审稿人都会独立抓到的 reject 级问题，且二者叠加在同一个 headline 数字上。

**修复后上限**：C1 重跑 + C2 统计 + C4 统一配置三件事（合计约 1.5 天 GPU + 1 天写作）完成后可达 borderline accept（6/10）；R2 的 e2e 批量化是唯一不确定项，决定最终落在 borderline 还是 accept。

**优先级排序**（投入产出比序）：

1. **tail32-4bit L1 主表 13 任务全量重跑**（C1）——一次性消解最大单点风险，无论结果好坏都必须做
2. **逐样本 bootstrap CI + held-out 分解句**（C2）——对已有逐样本分数重采样即可，几乎零 GPU 成本
3. **主表臂配置重跑 RULER-33**（C4）——统一双表系统 + 兑现「一组配置」卖点，半天 GPU
4. **RULER 32K 一档补跑**（M1）——分区增益长度梯度的第二数据点，全文最强卖点的补强
5. **musique +3.08→+2.43 等 mechanical 对账 + 摘要瘦身 + 「首次」删除**（M6/M7/m3）——纯写作，半天
6. **MoBA/Quest 复现对齐声明 + 13 臂配置附录表 + trace 可靠性二分前置**（M5/M4/M8）——纯写作，一天
7. **跨请求批量化 e2e + Quest e2e 接入**（C3）——工程量大，放最后但若成了全文升级一档

---

## 附：本轮复算通过项（审稿人也会算的部分，已确认无懈可击）

- LongBench 表 10 列总分全部复算通过（50.36/47.72/50.06/50.16/50.78/27.70/14.87/28.82/14.20/49.19）
- RULER 三档 AVG 与 16K 逐任务行（含被省略的 single_1–3/multikey_1=100、multiquery=25 的加权）全部复算通过
- 8胜5负、musique +2.57/+4.97、+0.42/+0.62/+0.72/+3.06/+1.59/−1.17、PSI-Quest 长度梯度 +3.58/+6.86/+13.50、分区 4K/8K/16K −0.14/−0.25/+0.91、速度比 1.285×/1.94×/1.21×/4.00× 全部与表内数字一致
- 唯一算术打架：musique vs TIA 正文 +3.08 vs 表算 +2.43（M6）
- 双语版内容对齐抽查通过（主表、E87c 校准、fig11 caption、Limitations 关键句两版一致）
- PDF 文本层：中文 19 页/英文 20 页，replacement chars=0，抽查 20 个关键数字短语全部落袋
