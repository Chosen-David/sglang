我已完整读完英文版全文（303 行）并逐段核对中文版（308 行）与图表资产（figures/ 目录、fig7 PDF 存在）。以下为结构化审稿意见。所有引用均给出英文版行号（`TLI_paper_en.tex:L`）。

---

# 审稿意见：TLI: A Training-Free Two-Level Sparse Attention Indexer

## 1. 总评

**倾向：weak reject（borderline 以下）**。论文的诚实性纪律（负结果入册、双口径报告、Limitations 自曝噪声边缘）显著高于平均水准，但核心正面主张的净证据量太薄——「首个超 FullKV」建立在自认噪声级 +0.18 且由单一任务 musique 独撑的 LongBench 总分上；速度卖点在三方同机 kernel 对比中排名垫底（0.787ms vs Quest 0.107ms）；分区消融的全部数字引用一个从未在文中出现的「单池消融臂」；且存在多处可被逐字揪住的数字自相矛盾。当前形态投 NeurIPS/ICLR 大概率被「主张超出证据」类意见击穿。

## 2. 按审稿轴打分（1-5）

| 轴 | 分数 | 一句话理由 |
|---|---|---|
| Novelty | 3.0 | 两级形态归 HISA、上界是下界的对偶、sink/SWA 是已知组件；真正的增量是 far/near 分区 + 降维自由度的系统刻画，属于「组合与刻画」级而非机制级创新 |
| Soundness | 2.5 | 主表数字自洽（RULER 三长度均值验算全部吻合），但消融证据链断裂（单池臂无表）、同一量两个值（+3.08 vs +5.02）、逐 benchmark 换 β 与「不需精调」直接矛盾 |
| Evaluation | 2.5 | 双口径方法论是亮点，但 e2e 仅 Qwen3-8B 一个模型族、零显著性检验、速度侧只报赢的形状、主表内含三个已知失效的 baseline |
| Significance | 2.5 | LongBench +0.18（自认噪声边缘）、RULER −0.78（输给 FullKV）、kernel 延迟三方垫底——三个 headline 维度两个半不成立 |
| Clarity | 3.0 | MoBA 式结构工整、叙事流畅，但三种 recall 口径（0.729/0.811/0.804）无定义表、大量浮空数字无锚点、TIA 全文无引用无展开 |

## 3. Major Concerns

### M1. 「首个超 FullKV」主张的证据量与其强度不匹配，且论文自相承认

**证据**：Abstract L17「TLI scores 50.54 and becomes the first training-free sparse indexer to exceed FullKV (50.36)」；Limitations L237 自认「the margin of the first LongBench win over FullKV sits at the edge of noise (+0.18)」。

**为什么 reviewer 会揪住**：这是全文最强的优先权主张（"the first"），但（a）论文自己在 Limitations 承认噪声边缘，Abstract 却不带任何限定地上头条——同一文档内两套口径；（b）无任何显著性检验（LongBench 每任务 n=200，musique F1 类指标任务间方差大，0.18 的总分差完全在重采样噪声内）；（c）净胜结构经不起分解：musique +2.62（L186 表）摊到 13 任务贡献 +0.20，其余 12 任务是「$0$ to $-0.44$」（L191）——即 headline 完全由**单一任务**扛着，而其余任务净贡献为负。RULER 上则是 −0.78 输给 FullKV（L171）。所以「首超」= 一个 benchmark 上一个任务扛着一个自认噪声级的总分差。

**修改建议**：① Abstract 措辞降级为「matches FullKV within noise (50.54 vs 50.36) while musique exceeds it by +2.62」；② 补 bootstrap 置信区间（按任务重采样即可，成本一天）；③ 把「musique 单点扛全局」写进 §3.1 而不是留给读者自己算。

**位置**：Abstract L17、§3.1 L191、Limitations L237。

### M2. 分区消融的全部数字引用一个「从未出现的单池臂」，且同一量在文中给出两个不同值

**证据**：贡献 #2（L47）「musique +3.08 over the same-budget single-pool arm」；§3.1（L191）「musique 34.76 ... (+2.62 over FullKV, **+5.02** over the same-budget single-pool arm)」——同一个「same-budget single-pool arm」上的同一任务增益，两处相差 1.94 分。此外「LongBench +0.25 and RULER +0.17 over the single pool」（L47）、长度梯度「−0.14 → −0.25 → +0.91, all measured against the same-configuration single-pool arm」（L176）、multikey_3 +5.0 / cwe +5.4（L176）——**该单池臂在两张主表和任何消融表中都没有一列自己的数字**。读者反推 50.54−0.25=50.29、34.76−5.02=29.74，文中任何表都找不到这两个值；主表里唯一的单池方法 TIA 是 50.06/32.28，与反推值不符。

**为什么 reviewer 会揪住**：这是论文核心创新点（贡献 #2）的全部定量证据，而证据本体不可见。+3.08 与 +5.02 的矛盾强烈暗示两处引用了不同配置跑的不同对照（正文从未区分），审稿人会直接质疑消融的口径管理。

**修改建议**：必须补一张「partition vs single-pool（同配置、逐任务、双 benchmark）」完整消融表，写明单池臂每格绝对分数与配置；统一 +3.08/+5.02（若来自不同 TLI 配置，逐处标注配置名）。

**位置**：L47、L176、L191、L198。

### M3. 「不需精调 / 一组固定配置」与「两个 benchmark 用不同 β」直接矛盾，构成 post-hoc 调参嫌疑

**证据**：§2.3 L135「\textbf{the budget partition is insensitive to configuration error and needs no tuning} ... one fixed TLI configuration covers 12/13 tasks」；§3.1 L191「LongBench multi-hop tasks favor $\beta{=}0.375$ (the 50.54 row), while RULER synthetic needle tasks favor $\beta{=}0.25$ (the 87.93 row), so both rows are reported and practitioners select by task profile」；RULER 表 caption L163 也写「$\beta{=}0.25$」，而 §2.3 L135 的配置是「$\beta{=}0.375$」。

**为什么 reviewer 会揪住**：两张 headline 表各自使用对该 benchmark 更有利的 β，选择依据是看到了两个 benchmark 的结果之后——这正是「平坦性=部署简单」卖点最怕的镜像指控（测试集调参）。论文把它包装成「诚实并报」，但审稿人读到的是：如果参数面真的平坦到不需精调，为什么不在两个 benchmark 上用同一个 β 出两行数字？

**修改建议**：补一张 2×2 交叉表（{β=0.25, 0.375} × {LongBench, RULER}）证明差异确实小（如记忆中 β 在 0.25–0.75 平坦，这张表应该能救回来）；若差异不小，则删掉「needs no tuning / one fixed configuration」的措辞，改为「β 的一次二选即可由粗粒度任务形态先验决定」。

**位置**：L135、L163、L176、L191。

### M4. RULER 上 vs 最近内部基线 TIA 是净输（−0.42），最难检索任务上输 −9~−14，分区「长度梯度」叙事被自己的 Limitations 拆台

**证据**：主表 L168-171：TLI 16K 83.68 vs TIA 84.53，总体 87.93 vs 88.35——TLI 相对最直接的结构对照（同为上界+4bit、无分区）在 RULER **三长度全输或全平**（4K 91.49<91.63、8K 88.62<88.90、16K 83.68<84.53）。Limitations L237「residual cost concentrated in multikey\_3 (16K: 75--80 versus 89 for TIA)」——最难的检索任务上分区版比第一代单池版**低 9~14 分**。同时 L176 声称「the 16K partition gain concentrates in multikey\_3 (+5.0)」——即 vs 那个不可见的单池消融臂 +5.0，vs 主表里的 TIA 是 −9。两个「单池」参照系在 multikey_3 上相差约 15 分，论文没有任何解释（两者都被描述为 4bit 上界单池，L160）。

**为什么 reviewer 会揪住**：审稿人一定会问「你的分区到底帮不帮最需要 far 检索的任务？」——答案取决于选哪个对照，而文中两个对照给出相反结论。且 Abstract 的「grows **monotonically** with length ($-0.14 \rightarrow -0.25 \rightarrow +0.91$)」（L17）本身算术上就不是单调（−0.14→−0.25 是下降），4K/8K 分区是**负增益**，「随长度单调增长」是修辞不是事实。

**修改建议**：① 把梯度改写为诚实版本：「partition costs ≤0.25 at ≤8K and pays back +0.91 at 16K」；② 解释 TIA 与单池消融臂在 multikey_3 上的 15 分差（若 TIA 是老管线不同打分口径，必须写明，否则消融臂可信度崩塌）；③ 在正文正面呈现 vs TIA 的 multikey_3 劣势而非藏在 Limitations。

**位置**：L17、L168-171、L176、L237。

### M5. 速度卖点：headline 加速比全部对 dense，同机三方对比垫底；e2e 只报赢的形状

**证据**：Abstract L17「kernel-level speedup of 1.46× at 32K to 5.09× at 128K over dense scoring, and 1.285× end-to-end for a single 64K request」；§3.3 L208 同机三方：「Quest 0.107ms, DSA 0.503ms, and TLI 0.787ms。The latency ranking is Quest $<$ DSA $<$ TLI」——TLI 比 Quest 慢 **7.35×**。§3.4 L212：「the 32K shape reaches **0.77×**」「TP2 bs16 64K shape reaches **0.92×**」——e2e 只在「单请求 64K」这一个形状赢。同句「decode step 65.0 → 33.5ms (1.94×, overtaking dense at 40.4ms)」中 1.94× 的基线 65.0ms 是什么从未定义（dense 是 40.4ms，vs dense 只有 1.21×），「prefill 733.9 → 183.4s (4.00×)」同样不说明 733.9 是谁的 prefill、dense prefill 多少。

**为什么 reviewer 会揪住**：稀疏索引论文的速度主张天然要过「vs 竞品 indexer」这一关。论文引以为傲的诚实报告（「we report as measured」「ranks third」）值得肯定，但 Abstract 的数字选择暴露了选择性呈现：对 dense 的 5.09× 上头条、对 Quest 的 7.35× 落后与 0.77×/0.92× 两个输的 e2e 形状全部不进 Abstract（却宣称「all accuracy costs and negative results are reported」）。「≈258 MACs / DSA 的 1/32」「index storage 3× below Quest」「quality +2.2 points」三个支撑量中两个无任何数字出处，「+2.2」的基准（vs Quest? 哪个 benchmark?）未定义——RULER 差是 +7.98、LongBench 是 +2.82，都对不上 2.2。

**修改建议**：① Abstract 速度句改写为同时含竞品对比与 e2e 形状边界的版本；② 65.0ms/733.9s 的对照身份逐处标注（建议：TLI-eager → TLI-optimized → dense 三列小表）；③ MACs/存储/质量三量给出可核查的推导或测量；④ 0.77×@32K 至少在 Abstract 的限定语里出现一次。

**位置**：L17、L208、L212。

### M6. 主表收录三个论文自己判定为「失效」的 baseline，属于用坏数字撑场面

**证据**：表 2 L185：SnapKV 27.70 / H2O 14.87 / PyramidKV 28.82（FullKV 50.36）；caption L179 自注「without sink-guard adaptation」；正文 L191/L102/L224 三处解释其崩坏机理是「emit EOS as their first token」的实现性失效，并称这是「fairness correction」。Quest 论文与这三家的原始论文在各自模型上都不至于 mus祭 0.63 分（H2O musique 0.63）。

**为什么 reviewer 会揪住**：论文明知失效原因（无 sink 保送）且自称做了「公平性修正」的洞察，却仍然把未修正的数字放进主表与 50.54 同列——这在审稿人眼里是明知故犯的弱化对照。要么给这三家加 sink 保送重跑（有修正方案却不用，难辞其咎），要么把这三行移入「adaptation pitfalls」小节单独呈现。

**修改建议**：主表替换为「SnapKV/H2O/PyramidKV + sink guard」的修正版数字；失效版数字作为 §3.1 末的适配性发现单独报告（那本身是有价值的观察）。

**位置**：L102、L179-191、L224。

### M7. 消融与机理段的三套 recall 口径（0.729/0.732、0.811、0.804/0.766）无协议定义，tail32 基线在两处相差 0.007 无解释

**证据**：贡献 #1 L46 与 §2.3 L109 与 §4.2 L196 反复使用「mass coverage of the full-dimensional top-1024 at 0.729, versus 0.732 for full-dimensional scoring itself」；表 tab:select L127 给 tail32 = 0.811（「far full-chain mass recall, mean over 8 samples」）；§4.2 L196 PCA 段「PCA at $d{=}16$ is near-lossless (0.766 versus 0.804)」——0.804 是谁的 recall、什么口径？若是 tail32，为何与表中 0.811 不一致（样本集不同？8 vs 16 样本？）。更基础的问题：「full-dimensional scoring itself」cover 自己的 top-1024 应为 1.0，0.732 意味着该指标另有定义（经 L1 粗筛链？带 4bit 量化？），文中从未给出任何一种口径的形式化定义。32B 段又出现第三种口径「entry-recall protocol」（L237）。

**为什么 reviewer 会揪住**：贡献 #1 整段（三重判决、四象限、饱和宽度）是论文最像「科学」的部分，但其所有数字的可比性依赖读者自行脑补三套协议的关系。0.729 与 0.811 都被称为 tail32 的「mass recall」，数值差 8pt，不解释清楚会被当成 cherry-pick 各取所需。

**修改建议**：在 §2.3 或附录加一张「协议定义表」：每个口径的候选池、参考集（dense top-1024? far 区全链?）、样本集大小、是否量化，并解释 0.804 vs 0.811 与 0.729 vs 0.811 的口径关系。

**位置**：L46、L109-111、L119-133、L196、L237。

### M8. 「上界 vs 下界是质量分化根源（+7.98）」是混杂归因；贡献 #4 五臂反转与 132 点网格零定量支撑

**证据**：Related Work L224「the score representation takes an upper bound (versus a lower bound, **the root of** the long-context quality split, +7.98 on the overall RULER average)」。+7.98 = TLI 87.93 − Quest 79.95（L171），但 TLI 与 Quest 同时差异于：sink 强制保留、4bit 子空间降维、两级级联、分区、（Quest 侧是否带 sink 未说明）。隔离变量应为 TIA（同为上界）vs Quest = +8.40，或同管线单池臂 vs Quest。把 TLI−Quest 的全部差归因于「上界」是因果跳跃。同类问题：贡献 #4（L49）「Trace-replay mass rankings and e2e F1 rankings systematically reverse across a five-arm comparison (the configuration ranked 4th in replay is the e2e champion)」——五臂的 replay 分数与 e2e 分数**一张表都没有**，这是方法论贡献却只有一句定性描述；§2.3 L135「A grid of 132 configuration points」同样无图无表。

**修改建议**：① +7.98 的归因改为「the upper-bound family (TIA/TLI) leads the lower-bound Quest by ~8 points」并补 TIA-vs-Quest 的隔离数字；② 五臂反转必须出表（臂名、replay 排名、e2e 分数），这是贡献 #4 的全部证据；③ 132 点网格至少给一张热图（figures/ 里 e64g 系列现成）。

**位置**：L49、L135、L224。

### M9. 单一模型族 e2e + 30B 因果 bug 披露引发的「8B 数据是否曾被污染」疑问

**证据**：全部 e2e 质量结论基于 Qwen3-8B（L160「All experiments use the real weights of Qwen3-8B」）；32B 只做了子空间机制检查（L237），30B-A3B 只做了速度（L212）。且 L100 披露「the 8B model tolerated the same bug without collapsing」——8B 在非因果泄漏存在时表面不崩。

**为什么 reviewer 会揪住**：(a) 论文的 sink 观察高度依赖 Qwen3 的 sink 权重特性（末层 0.592，L102），TLI 的强制保留区设计是否在其他模型族（Llama/Mistral 类 sink 行为弱得多）同样必要且同样有效，零证据；(b) 「8B 耐受了同 bug 未崩」这句话的言外之意是存在一个 8B 曾带 bug 跑出、表面正常的时期——审稿人会要求明确声明**所有报告的 8B 数字均在 bug 修复后产出**，当前文本没有这句保证。

**修改建议**：① 明确一句「all reported runs were produced after the causality fix」；② 至少补一个非 Qwen3 模型族（如 Llama-3.1-8B）的 LongBench 抽样验证，否则把「generalization beyond Qwen3 untested」写进 Limitations。

**位置**：L100、L160、L237。

### M10. Gate（贡献 #3）没有任何量化的收益数字

**证据**：贡献 #3 L48 与 §2.3 L137 给出的是 τ 梯度的**代价**（musique +0.23 / qasper −0.41），「saves compute safely」——但全文没有任何一处给出 gate 省了多少计算（跳过 X% 层的 far 检索 → decode 提速多少 ms 或多少 %）。一个只报代价不报收益的组件列为贡献 #3，审稿人会问「删掉它论文损失什么」。

**修改建议**：补一行 gate 开/关的 decode latency 差分（离线轮廓版已知 13/36 层可跳，哪怕给个上界估计）。

**位置**：L48、L137、L200。

## 4. Minor Concerns

1. **TIA 零引用零展开**：出现在两张主表（L166/L183）与正文十余处，bibliography 无条目，缩写全称从未给出。作为「最直接的结构消融参考」这是硬伤（若为未公开系统，须注明并给技术报告或附录描述）。
2. **「trade small wins and losses within noise (0 to −0.44)」（L191）**：区间 [0, −0.44] 内不存在 win，「wins and losses」措辞与区间自相矛盾。
3. **「grows monotonically with length」**（Abstract L17）：−0.14→−0.25 非单调，见 M4；中文版 L17 同病。
4. **超参不完整**：$sw_{lo}$、$|I_{\mathrm{sink}}|$ 的取值全文未给（K1=128、α/β/γ 给了），预算公式 L98 无法代入复算；F「saturating at 128--256」（L90）与公式 F=max(64, …) 的关系未说明。
5. **「musique ... gains the most at +1.94」（L191）**：+1.94 相对什么（β=0.375 vs 0.25？）未定义；同段 multikey_3 −2.00 同样无锚点。
6. **「cross-request batching already validated at 3.8×」（L212）**：3.8× 是什么的 3.8×，无上下文。
7. **«0.128ms independent of sequence length»（L147）**：每 token 每层每 head 还是全局？量纲不明。
8. **LongBench 主表只有 2 行**（Overall + musique，L185-186）：13 任务 7 方法主张下只展示总分+旗舰任务，逐任务表（或至少 win/loss 分布图）是必须的；旗舰任务外的 12 任务被折叠成一句「$0$ to $-0.44$」。
9. **RULER 无逐任务表**：Quest multikey_3=21（Abstract L17）、multivalue 89.5→69.25→44.25（L176）、cwe +5.4 等所有逐任务证据均无表可查。
10. **图 2 caption（L115）**：「(c) linear beats nonlinear, shared beats per-head」——正文相应段落（L196）从未给出 per-head vs shared 的对比数字，caption 超出正文证据。
11. **SparQ 条目无作者**（L284-285），DSA/MoBA/HISA 条目同样作者缺失且标注 TODO（仅中文版有 TODO 注释，英文版连 TODO 都没有直接裸条目）；HISA「arXiv:2603.28458」我无法核实存在性。
12. **Appendix A「companion experiment log shipped with the source code」（L243）**：无 URL、无 commit hash、无匿名仓库链接——复现性声明形同虚设；Appendix B 有硬件与分支名但同样无 code link。
13. **`Quest@1024` 与 TLI@1024 的预算对齐声明**：正文说 matched budget（L160 附近语境），但 Quest 原始实现的 page 大小、是否含 sink 等适配细节未写；given M6 的 sink 敏感性发现，Quest 是否被给了 sink 保送必须写明（否则 +7.98 的对比同样受 sink 混杂影响）。
14. **§3.4「the first net cash-out of the theoretical sparse-traffic win at e2e」（L212）**：优先权措辞（“the first”）无文献支撑，建议删除。
15. **「~258 MACs per token (1/32 of DSA)」（L208）**：推导过程未给（64 头×128 维 FP8 的 DSA 每 token MAC 数如何得出 258×32）。
16. **中英数字核对结论**：抽查 20 处关键数字（50.54/50.36/34.76/32.14/87.93/88.71/−0.14→−0.25→+0.91/+7.98/0.729/0.732/0.811/0.567/0.473/0.849/0.865/3.8pt/1.285×/16.39s/21.06s/0.107/0.503/0.787ms/65.0→33.5ms/13.51/+5.02/+3.08）两版逐位一致，未发现中英漂移；M2 的 +3.08/+5.02 矛盾两版同在（中文版 L46 vs L176）。
17. **写作**：「the first ... to exceed」「direct evidence」「guarantees no miss」等强断言密度偏高，与 Limitations 的克制语气不一致；建议全文做一轮 claim-strength 校准。

## 5. 审稿人最可能的 Reject 理由 Top-3 与防御建议

**R1：「Headline 质量主张（首超 FullKV）证据不足且自认噪声级，速度主张在竞品对比中不成立——两个卖点都站不住。」**
防御：(a) 把卖点从「首个超 FullKV」移到「musique 类 far-heavy 任务上 +2.62 超 FullKV、+5.02 超单池」这一可防守的单点主张，并补显著性检验；(b) 速度叙事从「我们快」改锚为「我们的算法复杂度/存储结构性占优（MAC 1/32、存储 3×）+ 质量占优（+2.82 LongBench），延迟差距是工程成熟度问题且有已验证的修复路径（3.8× 批量化）」——先把 M5 要求的 65.0ms 基线澄清和交叉表补齐，否则这条防御也无从谈起。

**R2：「核心创新点（分区）的消融证据不可见：单池臂无表、+3.08 与 +5.02 自相矛盾、vs TIA 在 RULER 净输且最难任务输 9-14 分。」**
防御：这是最可修复的一条。补齐 M2 的完整 partition-vs-single-pool 逐任务双 benchmark 表 + M4 的 TIA/单池臂差异解释（若 TIA 走老管线口径，一句话+一个脚注就能拆掉「两个对照打架」的指控）+ 修正「monotonically」措辞。修完后分区故事（紧预算下的预算分配机理 + 16K 收益区 + musique 机理证据）其实是全文最扎实的部分。

**R3：「公平性问题打包出现：主表含三个已知失效的 baseline、逐 benchmark 换 β、Quest 侧 sink 适配未说明、+7.98 的混杂归因。」**
防御：逐项做减法——(a) sink-guard 修正版 KV 压缩 baseline 重跑（论文自己有修正方案，工作量可控）；(b) β 交叉表证明平坦性后统一单配置出主表（或明确二选先验）；(c) Quest 适配细节写进 §3.1 协议段；(d) +7.98 归因降级为家族级对比。这四件事做完，诚实报告反而会从「被怀疑的选择性诚实」变成加分项——论文目前最大的悲剧性在于：它拥有的诚实纪律（Limitations 段落、负结果资产、如实报第三名）恰恰因为正面主张的修辞强度不匹配而失去了被相信的资格。

---

**一句话总结**：这是一篇实验纪律罕见地好、但主张修辞系统性超前于证据的论文——把 Abstract 和贡献列表的口径降到 Limitations 已达到的诚实水位、补上三张缺失的表（单池消融、五臂反转、β 交叉），它就值得一个 borderline/weak accept；以当前形态投顶会，R1+R2+R3 三连击大概率把它打回 weak reject。