# research/ —— TLI/PSI 论文研究资产（从根目录归档，2026-10-10 整理）

sglang fork 之上叠加的研究文件归档区。根目录只保留 sglang 原生结构 + 两个在用 bench 入口
（`test_c3_3arm_bench.py` / `test_c3_3arm_tp2.py`，E102/E106 五臂 e2e 延迟用）。

## 结构

| 目录 | 内容 | 备注 |
|---|---|---|
| `docs/` | 研究文档：TWO_LEVEL_INDEXER_DESIGN（批判分析+设计报告）、INDEXER_RESEARCH（DSA 调研）、TASK/PAPER_OUTLINE/DELIVERABLES、CPU+TC+CC.txt（开题素材） | 论文indexer.md（用户私人，AI 不得修改）仍在 `~/sglang/` 根，未动 |
| `bench/` | kernel 级 microbench：M8 系列（L1 TC/L2 kernel/TMA/topk/phase）、DSA vs TLI indexer 对比、MoBA router、Quest score、kernel_comparison | 统一 harness 口径（DSA tilelang fp8_index 原样 / Quest 官方 kernel 摘编） |
| `tests/` | sglang 侧驱动/诊断脚本：test_tli_8b/30b 系列（E5b/E67 dyngate/M5-M11 e2e）、test_quest_*、debug_m9 | 第一代（导师 TIA）与 M 系列；E98 之后质量实验在 two-level-attention（迁移中见任务链 #124） |
| `results/` | 结果 JSON：c3_3arm_tp2 三臂（E102）、tli_m5-m11 e2e/breakdown、kernel bench 结果、pca_basis 校准基（.pt）、RULER 早期结果、research_pack | e105/e106 新结果继续落 two-level-attention/exp/trace/results/（迁移后改此处） |
| `pred_archive/` | 早期 e2e 预测目录（pred_dyngate_*/pred_e5b_*/pred_e67_*） | 第一代 dyngate/E5b/E67 时期，未入 git |
| `figures_m8/` | M8 kernel 实验图 | |
| `indexer_proposal/` | 开题 PPT 提取（21 页 proposal + drawio 图） | |
| `ppt/` | TLI_progress_v4-v11 PPT + 生成脚本 | |

## 迁移进行中

two-level-attention 仓（transformers 层质量实验 + E 系列数据）将物理并入本仓（任务链 #124），
届时 `research/two-level/` 承接其 `sparse_attn/` + `exp/`，落袋路径与 CODEMAP 统一改指向。
