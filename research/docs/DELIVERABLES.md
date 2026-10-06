# TLI 项目交付物索引（2026-09-28 晨，#67 自主部分收口）

> 全部实验/优化/评测主线收官（#28-#66），论文草稿与终局 PPT 已交付。
> 数字口径统一 = 同步消除后最新 run（8B prefill 183.4s/4.00×、decode 33.5ms/1.94×）。

## 论文文稿（paper/）
| 文件 | 内容 | 状态 |
|---|---|---|
| `paper/TLI_paper_full_draft.md` | 合并成稿（Abstract→Intro→Method→Kernels→Eval→Related→Limitations→附录 A/B，~390 行） | 数字终审 ✅ |
| `paper/method.md` | 2.0-2.4（Overview/L1 上界/L2 分区/D' gate/系统集成） | ✅ |
| `paper/kernels.md` | 3.0-3.6（H20 约束/fused L1/dual 级联/M11/DS/同步消除/三方对比） | 事实修正版 ✅ |
| `paper/evaluation.md` | 4.1-4.5（RULER 终表/e2e 终值/消融/方法论） | 数字终审版 ✅ |
| `paper/intro_related.md` | Abstract/Intro 四贡献/Related 四类/Limitations 三侧 | ✅ |
| `PAPER_OUTLINE.md` | 6 节骨架+图锚+写作顺序 | ✅ |

## PPT
| 文件 | 说明 |
|---|---|
| `TLI_progress_v7.pptx` | 终局数据 5 页（三支柱/RULER 终表/e2e/优化链/定稿叙事）；可复现 `exp/figures/make_ppt_v7.py`（pptx 环境 `~/sglang/.venv_pptx/bin/python3`）；数字与合并稿 23 项一致性终验 PASS |

## LaTeX
| 文件 | 说明 |
|---|---|
| `paper/TLI_paper.tex` | 合并稿转写（24 section + 4 table + 6 figure 环境：fig7 架构/fig2 子空间/fig10 kernel/fig3 预算/fig9 e2e 边界/fig4 层跳过）；`paper/figures` 软链图源；本机无 xelatex，Overleaf 直接编译（ctex 中文） |

## 核心数据资产（论文表格的 json 权威源）
| 数据 | 文件 |
|---|---|
| RULER 四方法终表 | `two-level-attention/exp/results_ruler/ruler_table_final.json` |
| 8B e2e 稳态 | `sglang/tli_e2e_variance_results.json` |
| 30B TP2 64K | `sglang/tli_64k_tp2_tli.json`（旧基线备份 `/tmp/tli_64k_tp2_tli_baseline_bak.json`） |
| 三方 kernel 对比 | `sglang/kernel_comparison_indexers.json` |
| 实验全索引 | `sglang/TWO_LEVEL_PAPER_REPORT.md` §5 + §8b-1~32 |

## 终局数字速查（对外表述统一口径）
- RULER AVG：TLI 90.82/86.18/78.67 vs Quest 87.91/81.76/70.18（+2.91/+4.42/+8.49）vs TIA 91.63/88.90/84.53
- e2e：30B 单请求 64K **1.285×**；8B 稳态 prefill **4.00×**/decode **1.94×**；TP2 bs16 0.92×（诚实：launch-bound）
- kernel 级：两级选择 1.46×→5.09×（32K→128K）；M11 fused 11-16×；三方对比 TLI 延迟不占优（0.787 vs Quest 0.107ms）如实报告，优势=MAC 1/32+存储 3×↓+质量 +2.2

## 答辩讲稿
| 文件 | 说明 |
|---|---|
| `paper/defense_script.md` | v7 五页对齐版答辩讲稿（每页口播稿 + 预期提问应答 + 5 条高危问题预案，12-15 分钟设计）——供用户润色成答辩版 |

## 剩余（需用户参与）
1. 审阅修订 `TLI_paper_full_draft.md`（或直接改 `paper/TLI_paper.tex`；结构完整、数字已核，需语言润色与导师口径对齐）
2. LaTeX 编译校对（转写已完成 `paper/TLI_paper.tex`；注意：勿复发「三方对比最快」笔误——诚实结论是延迟不占优）
3. 答辩版 PPT（v7 基础上扩讲稿）
