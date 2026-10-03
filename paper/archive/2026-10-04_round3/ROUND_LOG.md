# Round3 归档日志（2026-10-04 06:50）

## 本轮主线：E98 全链 → 论文全量重写（#95/#97/#102/#103/#104）

### 数据链（two-level-attention，分支 dev-tli-e89）
- E98 mass 全网格（3600 有效臂，7fdd431）→ e2e v2 网格 13 臂（γ 截断坍缩 v1→v2 教训，d78dc4a）
- 选举：BEST = mavg α.125/β.375/γ.625（e2e avg 45.08；mass→e2e 排序反转终局确认）
- 13 任务全量：**AVG 50.78**（vs TLI_E72 50.54 +0.24、vs FullKV 50.36 +0.42；db1838d）
- E99 预测器：NO-GO 四重否定（6366ca7）→ 「不需精调→部署简单」论文卖点
- M6 sinkguard：snapkv 30.01 / h2o 17.02 / pyramidkv 32.35（17cc4d1）
- 可视化：fig5 六 panel + fig11 双口径对照图（2134279）+ CODEMAP/README/重组提案（6309806）

### 论文链（sglang，分支 two-level-indexer）
- 中文版全量重写（4ce82731a）：主表换臂 50.78、§4.3 新增「双口径参数网格与配置选举」段、fig11 入正文 fig:reversal
- 英文版同步（101116991）：19 项全对齐
- 读者 agent 审读：正文数字体系零打架；图件层 3 CRITICAL + 3 MAJOR
- 快修（d177228ac + two-level fe2afb0）：tie 名次 bug（修改竞赛排名）、E72/E98/E1 代号清零、「双任务均居首」→Pareto 口径、+0.49 统一

### 终态
- CN 19 页 / EN 20 页，编译干净 tofu=0，旧臂数字 5 处均合法参照语境
- MINOR 未修（留人工）：图内希腊字母文本层（mathtext 无 ToUnicode）、图内 mavg 缩写 vs 正文全称
- 待用户：新版 PDF 审阅、重组提案拍板、手动触发轮询监督优化

### 本目录文件
- TLI_paper.tex / TLI_paper.pdf（中文版快照，d177228ac）
- TLI_paper_en.tex / TLI_paper_en.pdf（英文版快照，d177228ac）
- reorg_proposal_2026-10-04.md（重组提案副本）
