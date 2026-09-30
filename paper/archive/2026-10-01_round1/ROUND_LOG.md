# 2026-10-01 首轮论文优化循环归档

## 本轮六步完成情况

| 步骤 | 状态 | 产出 |
|---|---|---|
| ① 顶会逐句对照 | ✅（上轮 8b-57） | 预算符号形式化（MoBA §2.2 对照）、tab:longbench 双语落盘 |
| ② 架构图精进 | ✅ | fig7 重画（见下） |
| ③ 可视化精进 | ✅ | fig2 panel(a) 重画为 E85b 旋转对三判决九 bar 图 |
| ④ 读者 agent 审读 | ✅ | Top-5 清单全部落实（见下） |
| ⑤ 审稿人 agent 批判 | ✅ | reviewer_report.md（weak reject 倾向，9 条 Major，M1/M2/M4/M8 已快修） |
| ⑥ 收尾归档 | ✅ | 本目录（PDF 待有 LaTeX 环境的机器编译） |

## 读者 agent Top-5 落实明细

1. 「16K 反超 FullKV (+0.91)」基线张冠李戴 → 双语改「vs 单池消融臂」真实口径
2. 内部代号全清（B7s/TLI_E72/TLI_B7/L03/mavg/M8/P1D1P2D2）→ 语义描述
3. TIA 定义 + 参考文献：中文版 13 cite + 17 bibitem；英文版 18 cite + bibliography + L1 构造式
4. fig7 重画 + fig2 caption/panel(a) 对齐
5. musique delta 基线标注 + mass recall 双口径命名

## 审稿快修（M1/M2/M4/M8，双语十处）

- M1：「首个超 FullKV」→「与 FullKV 持平（+0.18 噪声级边缘）」（Abstract + 正文段）
- M2：+3.08 vs +5.02 矛盾 → 统一主表口径 +5.02，单池臂数字 29.74 可见
- M4：「随长度单调增长」→「16K 才转正（4K/8K 为 −0.14/−0.25）」
- M8：+7.98 混杂归因 → 家族级 +8.40（TIA vs Quest 隔离数字）

## 审稿遗留（下轮待办，按优先级）

- **M2 完整版**：补 partition vs single-pool 逐任务双 benchmark 消融表（单池臂 C0 逐格绝对分数）
- **M3**：β∈{0.25,0.375} × {LongBench,RULER} 2×2 交叉表（数据已有：E72 mavg β.375 / B7s β.25 双臂 13 任务全跑过，需打分回填）
- **M5**：Abstract 速度句改写（含 Quest 对比与 0.77×/0.92× e2e 形状边界）+ 65.0ms/733.9s 对照身份标注
- **M6**：SnapKV/H2O/PyramidKV sink-guard 修正版重跑（或移入适配性发现小节）
- **M7**：三套 recall 口径（0.729/0.811/0.804）协议定义表
- **M9**：单模型族局限的正面承认与跨模型族讨论

## E81 终值（本轮回填）

SnapKV 27.70 / H2O 14.87（13 任务全量）/ PyramidKV 28.82

## 文件清单

- TLI_paper.tex / TLI_paper_en.tex：本轮终版双语源（sglang 070a8d09a + 快修提交）
- figures_snapshot/：fig7/fig2 重画版 + 全部图资产
- reviewer_report.md：审稿人完整意见（12984 字符）
- 编译说明：中文版须 xelatex（CJK）；英文版 pdflatex 即可；本机无 LaTeX 环境
