# 2026-10-01 第二轮论文优化循环归档（round2）

## 本轮六步完成情况

| 步骤 | 状态 | 产出 |
|---|---|---|
| ① 顶会逐句对照润色 | ✅ | 中文版全量重写收尾（#87）+ MoBA 句式骨架双语对齐 |
| ② 架构图精进 | ✅（round1 已重画 fig7 v3） | 本轮无图改动，图资产沿用 round1 |
| ③ 可视化精进 | ✅ | fig2 三 panel 已含 E76/E85 数据；UMAP 呈现问题遗留 round3（见下） |
| ④ 读者 agent 审读 | ✅ | reader_report.md（round1 C1/C2/C3 全解决；新 2 CRITICAL / 5 MAJOR / 6 MINOR） |
| ⑤ 审稿人 agent 批判 | ✅ | reviewer_report.md（borderline；M1/M2/M5/M7 已修，NM1-NM5 新发现） |
| ⑥ 收尾归档 | ✅ | 本目录 + 双语 PDF 重编译（中 13 页 / 英 14 页，零错误零越界） |

## 本轮主攻（round1 三 CRITICAL + M3/M5/M9）

1. **C1 单池消融臂不可见 → 已解决**：表 3 扩为 13 任务 × 8 方法列 + TLI-单池列（musique 29.74 总分 50.16）；表 2 RULER 加单池列（91.63/88.87/82.77/87.76）
2. **C2 mass 指标未形式化 → 已解决**：§3.1 新增「评测指标」小节，C（选择覆盖）与 R_far（far 区召回）双形式定义 + 0.811/0.804 批次差异注记 + GQA kv-head 级加权口径，双语落地
3. **C3 LongBench 主表仅 2 行 → 已解决**：13 行全表（8 列均值程序化验算逐位吻合，最大偏差 0.005）
4. **M3 β 交叉表 → 已落地**：β∈{0.25,0.375}×{LongBench,RULER} 四格表（对角占优、四格差 ≤0.35，与参数面平坦性互证）
5. **M5 速度句 → 已落地**：Abstract 加 Quest 0.107ms vs TLI 0.787ms + 0.77×/0.92× 形状边界；8B 稳态 733.9s/65.0ms 身份标注（TLI 自身 kernel 化前）
6. **M9 单模型族 → 已落地**：Limitations「规模与模型族」段扩写（rotate_half vs interleaved 布局依赖、RoPE-free 待检验、MLA future work）
7. **E85f 终判并入**：static pair 重放 +3.8pt / e2e −0.50（50.04 vs 50.54，musique −5.27）六处一致——protocol gap 第三案成立

## 双 agent 快修（round2 新发现，双语 14+1 项）

- **C-1/NM1**：正文「LongBench +0.25」与表算 +0.38 打架 → 双语三处统一为 +0.38 = 50.54−50.16（含可验算口径注明）
- **M4**：E85f 内部代号清出 Limitations（→「静态 pair 臂类实证」/ "static-pair-arm-type evidence"）
- **NM3**：0.732 口径注明（同管线 d'=128 臂含 4bit 量化与级联损失），双语三处
- **M1**：0.77× 双义 →「30B prefill 耗时为 dense 的 0.77 倍（1.30× 加速）」
- **M2**：mass 指标术语统一（表头→far 区召回 R_far）
- **M3/读者 m**：PCA 口径（共享 SVD 基 + per-head 0.752 见图）
- **NM5**：存储口径（336B/token vs Quest 约 1KB/token 全 head 汇总，摘录+正文双语）
- **m1**：+0.95 修正；**m2**：0.804 双义限定；musique +2.62/其余 −0.30 分解进正文；首超优先权措辞删除；「升级路径」标签撤下（closed path）
- 英文版 14 项镜像同步完成

## 验证

- 双语编译零错误：中 13 页（xelatex）/ 英 14 页（pdflatex）；Overfull 仅 2 处亚毫米级（0.26pt/3.54pt）；英文 0 处
- 逐页文本层：替换符 0 / 坏引用 0 / 代码代残留 0（双语 grep E85f/+0.25/full-dimensional scoring itself 全空）
- 程序化排版：文本对象边界框越界 0（1in 边距 + 6pt 容差）
- 关键修复数字抽验入 PDF：+0.38=50.54、336B/token、0.752、+0.95、1.30×、静态 pair 臂类 等全过

## 遗留（round3 优先级，来自双 agent）

1. **C-2 图 2(c) UMAP 柱**（0.813 高于 tail32 0.802，与 caption 矛盾；UMAP 仅 2 样本不可横比）→ 改图或图注加限
2. **M5 RULER per-task 表**（multikey_3 +5.0 / cwe +5.4 / Quest 21 / TIA 89 无表可查）
3. **NM4 竞品 e2e 速度对照**（Quest 无 e2e 速度数据）
4. **M6 sink-guard 修正 baseline 重跑**（或移出主表）
5. **M10 gate 收益量化**（只报代价不报收益）
6. **NM2 TIA 预算结构脚注**（中文版已加，需核实数据源；英文未同步）
7. TIA 零引用（minor）、SparQ 等 bib 作者缺失（minor）

## 文件清单

- TLI_paper.tex / TLI_paper_en.tex：round2 终版双语源（快修 15+14 项后）
- TLI_paper.pdf（13 页）/ TLI_paper_en.pdf（14 页）：重编译终版
- reader_report.md / reviewer_report.md：双 agent 完整意见
- git：d08c7f5f3（round2 主攻）+ 8a77d5caa（M3/M9 补充）+ 本轮快修提交
