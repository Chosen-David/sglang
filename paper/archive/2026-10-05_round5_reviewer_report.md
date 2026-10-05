# PSI 论文 round5 对抗性审稿报告（round4 全修订版）

- **审读对象**：`/home/wangyuanshuo02/sglang/paper/TLI_paper.tex`（中文 21 页）/ `TLI_paper_en.tex`（英文 23 页），round4 修订版
- **审读日期**：2026-10-05
- **方法**：双语 tex 逐行通读；全部关键数字对照落袋 JSON 复算（e98_full_13tasks / e100_bootstrap_ci / e100_tail_ci / e101_ruler_g0625 / c3_3arm_e2e / e103_kvhead_ablation / e104_ruler_32k / e98_e2e_grid / e98_abg_full_grid）；LongBench 表 10 列×13 行全量复算；编译日志核查（CN 21 页 2 处 Overfull、EN 23 页 0 处，5 个引用图文件全部存在）。本会话无图像识别，图仅查 caption 与正文一致性。外部检索不可达，novelty 判断基于内部知识。
- **前提**：round3 判 weak reject 4/10（5C+8M+7MINOR）。round4 声明修复 C1–C5 + M4 + E103 + RULER 32K 补点 + 读者快修一处。本轮任务：逐项验证修复质量 + 新鲜视角找残留/新问题。
- **总体印象**：round4 的核心修复（C1/C2/C3/M4/E103）是**真实且高质量的**——不是修辞级粉饰，而是补了实验、换了口径、删了过度主张，全部数字与落袋 JSON 逐位对上。但存在三类问题：**① C5 完全未修（InfLLM/Landmark 零引用、HISA 仍只引不比）**；**② C4 修复文本引入了两处与落袋数据直接矛盾的事实性错误**（「其余 8 任务逐位持平」与「niah_single 系全族全满」）；**③ 速度叙事收窄后仍残留一句与自家数据矛盾的「兑现形状」主张**。对一个以「双口径诚实报告」为核心身份的论文，可被 JSON 证伪的表述是最致命的部位。

---

## 一、round3 修复逐项验证表

| 项 | 判定 | 证据 |
|---|---|---|
| **C1** 被评测系统≠描述系统 | **修到位**（残留一个隐式拼接，见新 MAJOR-6） | §4.2 首段（CN L213 / EN L216）自我暴露式表述完整：主表臂全维 L1、tail32-L1 13 任务配对全量 50.53 vs 50.78（配对差 −0.25，CI [−0.57,+0.06] 含 0）、tail 口径对 FullKV +0.16 CI 含 0、hotpotqa −1.81 方向性结构、E87c 旧校准 54.83/54.43、33.22/34.76 相容性——**全部与 e100_tail_ci.json / e100_tail_full.json 逐位一致**。「持平口径在两种 L1 下都成立」的桥接成立 |
| **C2** 选举偏差无 CI | **修到位** | 摘要（CN L17）、贡献#2（L46）、§4.2（L253）、Limitations（L371）四处全部转持平口径：+0.42 CI [−0.18,+1.02]、sign test 8/13 p=0.58、held-out 11 任务 +0.08 [−0.46,+0.60]、musique 单任务 CI [−1.29,+6.37]——**全部与 e100_bootstrap_ci.json 一致**（本人独立复算 held-out 11 任务净差 = 0.079 ≈ +0.08 ✓、8 胜 5 负清点 ✓、sign test p=0.5811 ✓）。in-sample 分解句（「+0.42 的差分主要来自两个 in-sample 选举任务，全量主张应读作持平」）正是 round3 要求的表述 |
| **C3** 速度三层失守 | **基本修到位**（残留一句矛盾主张，见新 MAJOR-4/5） | 旧数字全删验证：1.285×/0.77×/0.92×/30B 1.30×/musique +3.08/「首次」全部 grep 零残留（此前匹配均为 0.804/0.924/60.77 假阳性）。三臂 e2e 数字与 c3_3arm_e2e.json 逐位一致（14.9/23.6/33.3、62.1/106.1/142.0、136.0/54.4、61.6/93.9/121.1、32 vs 59.5ms/step=1.85×）。速度主张收窄为三层且摘要明写「Quest 全形状 e2e 占优」——诚实判决到位 |
| **C4** RULER 输 TIA 未明说 | **部分修** | 已做：E101 主臂重扫 85.83（与 e101_ruler_g0625.json 全部逐位一致：89.77/86.28/81.45、cwe 82.0/68.6/55.5、fwe −3.0/−1.67/0.0）+ 双臂双表 Limitations 如实 + 16K 逐任务表 + 32K 六臂行。**未做**：①「PSI 参照臂 87.93 总分仍低于自家 TIA 88.35」这一 round3 明确要求显式承认的事实，正文与 Limitations **至今一句未提**（grep 88.35 只出现在表格行）；② §4.4「质量 vs Quest LongB +3.06 / RULER +7.98」跨基准拼接两套臂配置仍无「两基准各自最优臂口径」限定（round3 修复建议③）；③ 修复文本自身引入两处数据矛盾（新 C-N1 / MAJOR-2），见下节 |
| **C5** HISA 只引不比 + 漏 InfLLM/Landmark | **未修** | 双语全文 grep InfLLM/Landmark 零匹配；Related Work（CN L356–364 / EN L361–369）结构与 round3 审读时完全相同；HISA 仍然只引不比。**round4 声称修复的 5C 中此项没有兑现** |
| **M4** γ 截断 3600 叙事自伤 | **修到位** | 摘要加「（其中 γ 维在部署预算下经截断坍缩去重）」；tab:e2egrid 13 臂全表入文——**本人逐位核对 e98_e2e_grid.json 全部 13 行（组合×αβγ×mass×hq×mu）零误差**；「3600 有效配置」与 e98_abg_full_grid.json 实测 720×5=3600 一致（9×9×9=729/组合，合法性过滤后 720）；γ 坍缩对（0.375/0.75 逐位同分 54.16/35.57）表内可见；快修「重放第 4（五组合快筛口径）vs 网格口径第 2 并报」双语一致到位（CN L339 / EN L343） |
| **E103** kv-head 共享消融 | **修到位** | 贡献#2 + 消融节分区段双语四处，数字与 e103_kvhead_ablation.json 一致（B 臂 53.79/34.51，Δ −1.65/−0.20；C 臂 28.37/12.00；mid=0 机理 + caveat）。残留：仅 hotpotqa/musique 两任务（恰为两个选举任务），无 CI——见 MINOR-8 |
| **读者快修**（重放名次口径） | **修到位** | CN L339 / EN L343：「重放质量排名第 4 的组合（……五组合快筛重放口径）……网格重放口径下同一组合列四组合第 2（与快筛名次不同，反转结论一致）」——口径限定准确 |

**复算通过项（审稿人也会算的部分）**：LongBench 表 10 列总分全部复算通过（50.36/47.72/50.06/50.16/50.78/27.70/14.87/28.82/14.20/49.19，最大偏差 0.005 属舍入）；8 胜 5 负清点 ✓；held-out 净差复算 ✓；RULER 长度梯度 +3.58/+6.86/+13.50/+13.15 ✓；分区梯度 −0.14/−0.25/+0.91 ✓；+0.62/+0.17/+0.72/+3.06/+1.59/+2.43(musique vs TIA)/+4.97 ✓；16K 分区增益 multikey_3 +5.0 / cwe +5.4 ✓；e2e 网格四组合均值 45.08/43.90/43.90/43.58 与 44.86/44.59 ✓；mass 四组合排序 0.896>0.882>0.879>0.796 ✓；32K cwe −18.2 占 77%（−18.2/−23.65）✓；双臂差 −2.15 逐位一致 ✓；TP2 prefill 1.33×（136.0/102.25）✓；8B 稳态 1.94×/4.00×/1.21×/1.60× 口径自洽 ✓。**除下节列出的矛盾点外，本轮未发现任何新的算术打架。**

---

## 二、新发现问题

### CRITICAL

#### C-N1. 「其余 8 任务与参照臂逐位持平」与落袋 JSON 直接矛盾（C4 修复文本引入的数据失实）

- **位置**：CN L251（§4.2 RULER 段）、CN L216（tab:ruler caption「其余任务与参照臂逐位持平」）、CN L371（Limitations「其余 8 任务逐位持平」）；EN L254 / L219 / L376 三处同文
- **原文摘录**：「伤害集中于词表聚合任务 cwe（82.0/68.6/55.5……）与 fwe（−3.0/−1.67/0.0），**其余 8 任务——含全部 single/multikey needle——与参照臂逐位持平**」
- **反证**（e101_ruler_g0625.json `delta_vs_b25_g0125`，逐长度）：
  - multikey_3：4K **0.0** / 8K **−3.0** / 16K **−1.0**（非零，且属「multikey needle」）
  - multivalue：4K **−0.5** / 8K **−1.0** / 16K **−0.5**（非零）
  - 复算闭合：8K 总 Δ−2.34×11=−25.74 = cwe −20.0 + fwe −1.67 + multikey_3 −3.0 + multivalue −1.0（差值属舍入）✓——即总账里明确含这两任务的损失
- **三重内部不自洽**：①「逐位持平」在本文用语中一贯指逐位相同（如「γ 截断坍缩臂逐位同分」「配对逐位校验」），而 multikey_3 差至 −3.0；② 11 任务 − cwe − fwe = 9 ≠「其余 8」（无论把 multikey_3 还是 multivalue 划入伤害侧，都同时使「含全部 single/multikey needle」或「逐位」之一为假）；③ 该句在正文/表注/Limitations 双语共 **6 处**重复。
- **危害**：这是 round4 修复 C4 时新引入的表述，恰好落在论文赖以立身的「如实报告」身份上——审稿人（或 artifact 评估）用配套 JSON 一对即穿。
- **修改建议**：改为「其余 9 任务差 ≤3 分：全部 single/multikey\_1/2/multiquery/vt 逐位相同，multikey\_3 至 −3.0（8K）、multivalue 至 −1.0 属边界级损失」；Limitations 同步。

### MAJOR

#### M-N1. 32K 行「TIA」列与「PSI-单池」列 11 任务逐位全同——两列疑为同一系统，标签与 4K–16K 行的 TIA 不可比

- **位置**：tab:ruler（CN L221–227 / EN L224–227）；e104_ruler_32k.json
- **证据**：32K 档 TIA 与 single_pool 两臂 **11 个任务×逐位完全相同**（60.77；vt 31.8、multikey_3 16、cwe 83.5、fwe 12.67 全同）——两个名义上不同的系统在非天花板任务上逐位一致的概率实际为零。而 8K/16K 行两列明显有别（88.90 vs 88.87、84.53 vs 82.77），4K 行同为 91.63（近天花板可用巧合解释）。E104 备注明「budget: …subspace full(默认)」且单池臂为「tli α=0/β=0/γ=1 (C0 defaults)」——32K 的「TIA」臂极可能是同一 tli 后端同配置的复跑（全维 L1 单池），而非 4K–16K 行所用的历史 TIA（4bit 子空间量化，cmp_ratio 4）。
- **危害**：若成立，32K 行的「TIA@1024」是错标签的同池臂：① 表内两列冗余；② 「上界家族在 32K 保持质量」的旁证被削弱（真 TIA 的 32K 表现实际未知）；③ 与 caption「TIA 是第一代……4bit 子空间量化」的系统描述冲突。
- **修改建议**：核实 32K TIA 臂的实际运行命令与子系统；若确为单池同构，表内删除该列或改标「单池（复用）」，并补一次真 TIA 的 32K 跑（11 任务×n=100，半天 GPU）；至少在 caption 注明两列在该档同构。

#### M-N2. tab:ruler caption「niah_single 系全族全满校验管线」/ EN "niah_single tasks scoring full marks across all methods" 与 e104 JSON 矛盾

- **位置**：CN L216 / EN L219
- **反证**：e104_ruler_32k.json 32K 档 niah_single_1/2/3 = Quest 98/91/95、六臂中多数为 99、仅 FullKV 全 100。**没有任何一系任务「全族全满」**。
- **修改建议**：改为「niah_single 系全族 ≥91 的管线健康检查（仅 FullKV 满分）」或直接删去该短语。双语两处。

#### M-N3. 「高并发 decode 是 PSI 在 e2e 的兑现形状」与同段自家数据矛盾（C3 收窄后残留的唯一过度主张）

- **位置**：CN L347 / EN L351（§4.5）
- **原文摘录**：「decode 段排序反转——**dense 61.6 < PSI 93.9 < Quest 121.1 ms/step**……高并发 decode 是 PSI 在 e2e 的兑现形状。」
- **矛盾**：同句刚给出 dense decode 最快、PSI 慢于 dense 32.3ms/step；PSI 在 decode 的相对优势**只存在于对 Quest 的增量口径**（59.5 vs 32ms/step）。任何 e2e 形状下 PSI 都没有净赢（total 输 Quest 与 dense、decode 输 dense）。「兑现形状」按字面读是「PSI 在该形状净兑现带宽收益」——数据不支持。
- **修改建议**：改为「decode 段是 PSI 相对 Quest 的唯一 e2e 优势形状（93.9 对 121.1 ms/step，索引开销 1.85×），但仍慢于 dense；配合已验证的跨请求批量化（3.8×）是通往净兑现的路径」。双语两处。摘要中「PSI 在 e2e 的兑现点是 decode 段」同理收窄为「对 Quest 的相对优势点」。

#### M-N4. Quest prefill 比 dense 快约 2×——三臂 prefill 路径疑非计算等价，PSI prefill 损失归因被混杂

- **位置**：CN L347 / EN L351；c3_3arm_e2e.json
- **证据**：单请求 64K prefill：Quest 8.07s vs dense 20.86s；TP2：Quest 54.43s vs dense 102.25s。一个 prefill 期仍需全量注意力的稀疏方法，prefill 应 ≈ dense + 索引开销；Quest 反而快 ~1.9×，说明 Quest backend 的 prefill 路径与 dense Triton 不计算等价（稀疏 prefill、不同 kernel 或不同 chunked-prefill 配置）。论文把 PSI 的 136s 归因于「两级索引逐 chunk 构建成本 vs Quest 一级索引」，但该归因在 Quest≠dense 这一混杂下不成立。
- **修改建议**：① 核实 Quest sglang backend 的 prefill 路径（是否稀疏 prefill/不同 kernel）；② 在 §4.5 加一句解释 Quest prefill 快于 dense 的机理，或明示「三臂 prefill 路径非计算等价，prefill 段比较读作 backend 间对照」；③ 若 Quest 用了稀疏 prefill，PSI 的 prefill 劣势叙述需相应改写（PSI 当前是 dense prefill + 索引构建）。

#### M-N5. 「免调参」（贡献#4）与「双臂按任务形态二选一」（部署口径）的叙事冲突未软化，round4 修复使其更刺眼

- **位置**：贡献#4（CN L48 / EN L49）「部署只需一组默认配置……参数不需精调本身构成部署优势」；§3.3（CN L177 / EN L180）「PSI 一组固定配置覆盖 12/13 任务」；摘要（L17）与 Limitations（L371）「双臂并报：真实任务形态取主臂、合成 needle 形态取参照臂」
- **矛盾链**：双基准最优臂分属两套 (β,γ)（0.375/0.625 vs 0.25/0.125），选错臂在 RULER 损 2.10——这正是一种两点式调参。「按任务形态二选一」（§4.2「使用者按任务形态选择」）本身承认了形态级配置依赖。§3.3 的「覆盖 12/13 任务」是 LongBench 口径却无限定语，与 Limitations 的双臂事实并读即见冲突（round3 MINOR-5 已指出，round4 双臂叙事落地后该句更需软化）。
- **修改建议**：贡献#4 收窄为「参数面在任务形态**内部**高度平坦（四粒度 oracle ≤0.08/0.0008/0/0.21–0.49），形态间只需一次二选一（β/γ 两点），对照 DSA 千步 warm-up 仍是免调参量级」；§3.3「一组固定配置覆盖 12/13 任务」加「（LongBench 真实任务口径；合成 needle 形态取参照臂，见 §4.2/Limitations）」。

#### M-N6. 速度/存储主张的系统归属拼接仍为隐式（C1 残留）

- **位置**：摘要（CN L17「在精度持平的前提下，选择链相对 dense 打分取得 1.46×→5.09×」）；§3.4 实现（L186–191，336B/token、fused 4bit 反量化 3.6×）；§4.4（L343，0.787ms、258 MAC、336B vs 1KB）
- **问题**：主表 50.78 出自全维 L1 系统；336B/token、258 MAC、1/32 DSA、0.787ms 全部属 tail32-4bit 系统。round4 用「两 L1 口径质量均持平（50.78 与 50.53，CI 均含 0）」完成了统计桥接，但摘要与 §3.4/§4.4 没有一句明示「速度与存储数字属于生产配置（tail32-4bit），其质量经 50.53 臂验证为同精度持平」。审稿人仍可复读 round3 C1 的原话（"quality tables from a 4× more expensive L1 than the system whose speed you benchmark"）——现在有了桥，但桥没架在句子层面。
- **修改建议**：§4.4 首句加「本节 kernel 与存储数字对应生产配置（tail32-4bit L1），其 13 任务质量为 50.53（对 FullKV +0.16，CI 含 0，§4.2）」；摘要速度句加「（生产配置口径，质量经 §4.2 双口径验证持平）」或直接以 tail 口径为速度侧锚。

#### M-N7.（round3 遗留未修，本轮复核确认）C5：InfLLM / Landmark Attention 零引用、HISA 零实验对比

- **位置**：Related Work 全节（CN L356–364 / EN L361–369）
- **证据**：双语全文 grep InfLLM/Landmark 零匹配。round4 声称「C5 已修复」，实际 Related Work 与 round3 审读版逐字相同。novelty 定位（「上界 vs 下界、分区 vs 全池两正交维度的系统刻画」）仍无最近邻实验支撑。
- **修改建议**：① 必做（零 GPU）：补 Landmark Attention（ICLR 2024）与 InfLLM（2024）两条 bibitem + 一句定位（块摘要/块单元两级选取，均无上界保证与分区预算刻画）；② 强烈建议：HISA 若有实现，接入统一 monkeypatch 管线跑 LongBench 13 任务——这是 novelty 主张闭环的最后一块。

#### M-N8.（round3 遗留未修）M2：质量在 8B、速度在 30B-A3B，同一模型+配置无 speed-accuracy 联合证据

- round4 的 e2e 三臂全部在 30B-A3B 上跑，但没有补任何 30B 质量数字；质量侧全部 8B。「精度持平 + kernel 加速」的联合主张仍跨两个模型拼装。修复成本已给出过（30B 上 3 任务 LongBench 子集，~半天 GPU）。

#### M-N9.（round3 遗留未修）M5：MoBA/Quest 复现无「与原论文报告量级对齐」声明

- 双语全文无任何 baseline 对齐声明句。Quest RULER 16K 70.18 与其论文报告的近无损行为差异巨大，MoBA 复现 49.19 远低于官方数字——两者都需要一句「与原论文在 [重叠设置] 的对照/无重叠口径的协议说明」。这是用户自设的 baseline 纪律，论文文本至今未兑现。

### MINOR

1. **摘要超载进一步恶化**（CN L17 / EN L17）：round3 m3 已批（~15 个数字），round4 修复又塞入 CI、sign test、三臂 e2e、双臂并报、三层速度收窄——现约 25+ 个数字、~500 字单段，MoBA 四句式骨架已被撑破。审稿人第一印象受损。建议砍至 8 个数字：50.78/持平 CI、musique 34.71、Quest 崩塌 21、kernel 1.46→5.09、Quest e2e 62.1 vs 142.0、decode 1.85×、RULER 87.93/85.83 双臂、+0.91@16K。
2. **\date 仍为 2026-09-30**（CN L8 / EN L7），论文实质重写于 10-04/10-05。
3. **bib 卫生**（round3 MINOR-1 未修）：CN 版 nsa/clusterkv 两条 bibitem 正文零引用（L435/L439 悬空打印）；EN 版无这两条但 HISA 条目缺作者（CN 版有完整作者列表）；两版条目集不一致；SparQ/DSA/MoBA/ClusterKV 四条「投稿前核对」TODO 仍在。
4. **+8.40（Related Work，TIA 口径）与 +7.98（§4.4，PSI 口径）双数字并存**（CN L358 vs L343 / EN L363 vs L347）——round3 MINOR-2 未修，相邻章节两个「上界家族 vs Quest」数字，审稿人会当不一致读。建议 Related Work 明写「TIA 口径」或统一 PSI 口径。
5. **tab:betacross 的 RULER β=0.375 格 87.58 与 e101_ruler_g0625.json old_refs 的 87.59 不符**（CN L290）：0.01 誊写差，须对源（exp/results_ruler/ruler_e72mavg.json）核定后统一。
6. **32K 行粗体 60.78 标记 PSI 列最优**（tab:ruler L224）：在全族退化区（多数任务 4–35 分）用粗体标「最优」与正文「绝对值以同长度 FullKV 为基线」的读法相悖；且主臂 32K 总分 58.63（低于 FullKV 59.38）未在表或正文出现，只在 cwe 语境露出 61.9——粗体标记对读者有轻度误导。建议 32K 行去粗体或全行斜体示退化区。
7. **musique「为全部方法（含 FullKV）最高」在摘要无限定**（CN L17）：单任务 CI [−1.29,+6.37] 含 0（正文有披露，摘要无）。持平口径的纪律应对自己最有利的数字同样适用；建议摘要加「（单任务 CI 亦含 0）」或删「最高」改「34.71（对 FullKV +2.57）」。
8. **E103 消融仅 hotpotqa/musique 两任务**（恰为两个 in-sample 选举任务），无 CI、无 held-out——「共享不降反升」的结论强度受限于 n=2 任务。建议补 2 个非选举任务（如 2wikimqa/narrativeqa）。
9. **4bit 量化后 min/max 是否仍严格上界未说明**（§3.2 L126）：若量化对 max 向下取整，式 (3) 的不等式在量化口径下不严格成立，「无漏选」保证缺一句 rounding 方向的说明（保守取整则成立）。
10. **bootstrap 聚合方式的口径说明**：CI 采用「任务内重采样逐样本配对差→任务均值→13 任务等权」，属条件于任务集的推断（把 13 任务当固定基准口径，与 LongBench 官方聚合一致，可辩护）；但与 sign test（任务为抽样单元）并列时两种推断单元不同，建议 §4.2 加半句「CI 为条件于该 13 任务基准的口径，任务集层面的不确定性由 sign test 补充」。
11. CN 版 2 处 Overfull hbox（编译日志）；TP2 PSI 侧历史 +23% 回归（141.95 vs 115.28s，c3 JSON regression_note「待归因」）未在论文标注——三臂同协议内部公平，但 artifact 评估者对照历史数字会追问，建议 Limitations 或附录一句归因状态。

---

## 三、若第一次读：Top-3 reject 理由预测

| # | 审稿人 reject 理由（预测原话） | 对应问题 | rebuttal 可解性 |
|---|---|---|---|
| R1 | *"After all honesty caveats, in what regime does PSI actually win? Accuracy is parity (CI contains 0), e2e latency loses to Quest in every shape and to dense in decode, the indexer kernel is third of three, and the prefill is 1.33–1.60× slower than dense. The surviving claims are a kernel-internal speedup over dense scoring, a storage number, and a MAC count — none demonstrated as a net end-to-end win over any baseline."* | C3 收窄后的结构性剩余 + M-N3 | **中**：跨请求批量化（3.8× 已验证）+ prefill 索引构建 kernel 化落地后补 TP2 e2e 重测是唯一硬路径；若最终仍不敌 Quest，只能彻底转「设计空间刻画 + 方法论」身份（写作可解，见第四节） |
| R2 | *"The mechanism story is Qwen3-layout-specific (rotate_half), all evidence is one model family, quality stops at 32K while the motivation is 131K, accuracy is on 8B and speed on 30B-A3B — and the closest prior two-level work (HISA) is never run while earlier precedents (InfLLM, Landmark) are uncited."* | C5 未修 + M2 + M1 半修 | **中高**：三个写作项（引用、限定、贡献重排）+ 30B 三任务质量 + HISA 复现，合计 ~1 天 GPU + 1 天写作 |
| R3 | *"For a paper whose identity is honest reporting, the text still contains statements falsifiable by its own JSONs: eight tasks 'match bit-for-bit' while the paired deltas show −1 to −3 on two of them; 'niah_single full marks across all methods' while Quest scores 91–98; 'high-concurrency decode is the cash-out shape' while dense wins decode; and the 32K TIA column is bit-identical to the single-pool column."* | C-N1 + M-N1/2/3 | **高**：全部写作级，1 天清零——但这恰恰是最不该留给审稿人的把柄 |

**关于「诚实是加分还是减分」**：round4 的诚实化（CI、held-out 分解、三臂如实、负结果入册）本身是**加分项**——本轮复算确认这些数字没有一个虚报，这在投稿论文里罕见，审稿人会感知到。但目前的摘要把「诚实」写成了「让步清单」：约 40% 篇幅在陈述自己不赢什么。正确的姿态是把诚实转为**主动的方法论主张**（「我们给出 training-free 稀疏索引设计空间的系统刻画 + 离线代理不可信的协议鸿沟定案」），把负结果组织成发现而非辩护。当前形态是「诚实的弱系统论文」，一步之遥是「强实证分析论文」——差别主要在贡献排序与摘要叙事，不在数据。

---

## 四、Accept 路径建议（按投入产出比排序）

1. **清零全部事实性失实**（C-N1、M-N1/2/3、MINOR-5，纯写作+核实，~1 天）：这是当前最高优先级——论文的全部 credibility 押在「与 JSON 可对账」上，6 处「逐位持平」、2 处「全满」、2 处「兑现形状」、32K TIA 列标签，每一处都是审稿人一次 grep 就能引爆的点。
2. **C5 兑现**（写作半天 + HISA 复现视实现可得性）：补 InfLLM/Landmark 引用与定位句；HISA 接入统一管线跑 13 任务。同时把贡献 #1（降维自由度刻画）与 #4（平坦性/协议鸿沟）在摘要前置——论文的实际身份是「设计空间系统刻画 + 一个实例」，数据支撑的正是这个身份。
3. **一次净 e2e 赢的形状**（工程量最大，决定上限）：跨请求批量化落地 + prefill 索引构建 kernel 化后重测 TP2 bs16 64K；目标至少追平 dense（106s），理想是逼近 Quest（62s）。做不到则接受 R1 的判决，全面转实证分析定位。
4. **30B-A3B 三任务质量**（~半天 GPU）：闭合 speed-accuracy 联合证据（M-N8）。
5. **免调参叙事收窄**（M-N5，纯写作）：贡献#4 与 §3.3「12/13」句加形态限定，把「双臂二选一」表述为形态级一点选择而非调参。
6. **baseline 对齐声明**（M-N9，纯写作）：MoBA/Quest 各一句与原论文量级的对照或无重叠口径说明。
7. 摘要瘦身至 ≤8 数字（MINOR-1）、日期/bib/TODO 清零（MINOR-2/3）。

---

## 五、总评分与判决

**5/10，weak reject（较 round3 的 4/10 提升 1 分，已站上 borderline 门槛之下沿）。**

- **加分**：C1/C2/C3/M4/E103 五项修复全部经落袋 JSON 独立复算验证为真修（换实验、换口径、删过度主张，非修辞粉饰）；表格算术全场复算自洽；负结果与 CI 纪律在投稿论文中罕见。
- **扣分**：① C5 声称修复实际未修（novelty 支撑缺口原样保留）；② C4 修复文本引入 6 处可被自家 JSON 证伪的失实表述（C-N1 为最重）；③ 速度叙事收窄后仍留一句与数据矛盾的「兑现形状」；④ Quest prefill 2× 快于 dense 的协议混杂未解释；⑤ 免调参 vs 双臂的叙事冲突在修复后更刺眼；⑥ R1（无可证明的净赢形状）是结构性剩余。

**判决**：round4 把论文从「会被 3 个独立审稿人各抓一个 reject 点」的状态修到了「只剩 1 个结构性问题 + 一批可一天清零的写作级失实」。写作级修复（路径 1/2/5/6/7，合计约 2 天、近乎零 GPU）完成后可达 6/10 borderline accept；路径 3（净 e2e 赢形状）是通向 7+/accept 的唯一硬门槛，其成败也决定论文最终以「系统」还是「实证刻画」身份被接收。

---

## 附：本轮复算通过项（供 rebuttal 备用）

- LongBench 10 列×13 行总分全量复算（最大偏差 0.005，舍入级）；8 胜 5 负清点；held-out 11 任务净差独立复算 = +0.079
- e100_bootstrap_ci / e100_tail_ci：CI、sign test、musique/hotpotqa 单任务 CI、E87c 校准数字全部一致
- e101_ruler_g0625：85.83、89.77/86.28/81.45、cwe/fwe 逐长度 delta 一致（唯「其余任务持平」表述与 delta 表矛盾，见 C-N1）
- e104_ruler_32k：六臂 AVG、cwe −18.2/77%、双臂差 −2.15、PSI−Quest +13.15 一致（唯 TIA=单池逐位全同与 niah_single 全满两处存疑，见 M-N1/2）
- c3_3arm_e2e：三臂全部延迟数字、decode 增量口径 32/59.5ms、1.85×、TP2 prefill 1.33× 一致
- e98_e2e_grid：tab:e2egrid 13 行逐位零误差；四组合 e2e 均值与 mass 排序复算一致；γ 坍缩对可见
- e98_abg_full_grid：3600 = 720 配置×5 组合，与「9×9×9 过滤后 3600」表述算术自洽
- e103_kvhead_ablation：三臂数字与机理表述一致
- 编译：CN 21 页（2 Overfull）/ EN 23 页（0 Overfull）；5 个引用图文件全部存在；双语 5 处关键段（摘要/§4.2 首段/RULER 双臂判决/e2e 速度/Limitations）语义与数字对齐抽查通过
