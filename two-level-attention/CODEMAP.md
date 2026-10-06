# CODEMAP — 实验代号 → 脚本 → 数据 → 结果 → 论文引用

> 本表由 code-organization skill 维护（2026-10-04 首建，覆盖 round3 E98/M6/E99 任务链）。
> 增量更新约定：每轮任务链收官后只追加新行，不重排历史行；每行至少一个可核对锚点（脚本/JSON 路径或 commit）。
> E1–E90 历史实验映射为 backlog（详见 sglang 项目记忆 history_two_level.md），后续轮次增量补齐。

## round3 收官轮（E98 全链 + M6 + E99 + E97 终表 + 可视化）

| 代号 | 一句话结论 | 脚本 | 数据 | 结果 JSON | 论文引用 |
|---|---|---|---|---|---|
| E98 mass 全网格 | α×β×γ 9³×5 method 组合 3600 有效臂 mass：mass 冠军 mminmax_a.875_b.875_g.5 (0.896)，全局 top 为单池角点 mavg_a0_b0 (0.8979) | `exp/trace/analyze_e98_abg_full_grid.py`（跑批 /tmp/e98_abg_grid.sh，16 shard 分进程，不入库） | collect_trace_lb 真实 trace 16 样本 CPU 重放 | `exp/trace/results/e98_abg_full_grid.json`（commit 7fdd431） | 图 fig8/9/10（`exp/trace/plot_e98_abg_grid.py` → sglang/ref/figs）+ `exp/figures/fig5_abg_scan.py` panel (a)-(c) |
| E98 e2e 网格 | mass top-3 γ≥0.5 臂在 e2e K2=1024 下 nt_near 截断坍缩为同配置 → v2 按 (α,β,nt_near) 去重重选 13 臂；e2e 冠军 mavg_a.125_b.375_g.625 avg 45.08 | `/tmp/e98_e2e_grid_v2.sh`（v1 坍缩教训写进 JSON note；/tmp 脚本不入库） | LongBench hotpotqa/musique n=200，pred 输出 /tmp/e98_e2e/pred_* | `exp/trace/results/e98_e2e_grid.json`（commit d78dc4a） | fig5 panel (d) 散点 + fig11 panel (b) |
| E98 选举 | 只认 e2e 实测：best = mavg a.125/b.375/g.625（hq 55.44/mu 34.71 avg 45.08）；mass 冠军 e2e 第 4 档（反转如实报告）；ref_gain +0.49；ccluster 无 e2e 路径不参与 | `exp/trace/elect_e98_best.py` | 输入 e98_e2e_grid.json + /tmp/e98_e2e_runs_v2.json | `exp/trace/results/e98_best_election.json`（commit d78dc4a；脚本预置 2979ee5） | fig11（两 panel 名次标签即选举结论） |
| E98 13 任务全量 | best 臂 LongBench 全量 **AVG 50.78 新主表臂**（vs TLI_E72 50.54 +0.24、FullKV 50.36 +0.42），13/13 任务完整 | `/tmp/e98_full_13tasks.sh`（等选举 JSON 自动双卡启动，含打分 heredoc；不入库） | /tmp/e98_full/pred_E98BEST_mavg_a0.125_b0.375_g0.625/（n=200，multifieldqa_en 全集 150） | `exp/trace/results/e98_full_13tasks.json`（commit db1838d） | 论文主表（#97 全量重写的数据底座） |
| E99 预测器 | **NO-GO 四重否定**：mass_total↔e2e Spearman 0.166（p=0.59，bootstrap CI 含 0）；oracle 增益 +0.21 噪声级；与 E75 (+0.08)/E79a (+0.0008)/E79b (0) 先验一致 | `exp/trace/analyze_e99_predictor.py` | 只读 E98 三个落袋 JSON（n=13 臂小样本口径） | `exp/trace/results/e99_predictor_probe.json`（commit 6366ca7） | 负结果资产（暂无正文引用） |
| E97 终表 | method×(α,β,γ)×精度（mass+e2e）×速度 终表合成，#97 论文数据底座 | `exp/trace/make_final_table_e97.py` | e64g_full_grid / e88_partition_verdict / e87_topsigma_merged / e71_main_table / e72_screen_verdict 等 | `exp/trace/results/final_table_e97.json` + `.md` | 论文主表底座（缺臂自动标 MISSING） |
| M6 sinkguard | sink 保送（不占预算，PSI 严格口径）三方法 13 任务：snapkv 30.01 (+2.31) / h2o 17.02 (+2.15) / pyramidkv 32.35 (+3.53)——「首 token EOS」崩坏修复必要非充分，距 FullKV −18~−20 分 | 实现 `exp/trace/kvcf_qwen3.py` + `exp/trace/pred_kvcf.py --sink-guard`；跑批 /tmp/m6_sinkguard_full.sh、/tmp/m6_pyramidkv_gpu1.sh、/tmp/repair_pyramidkv_mfq.sh（均不入库） | `exp/results_longbench/Qwen3-8B/pred_kvcf/{snapkv,h2o,pyramidkv}_sinkguard/` | `exp/trace/results/m6_sinkguard_verdict.json`（commit 7611f29，第一臂）+ `m6_sinkguard_full.json`（commit 17cc4d1，三方法全量）；打分为监督器会话内联（无独立脚本） | 审稿 M6「有药不吃」闭环（论文 Limitations / 审稿意见归档） |
| E98 可视化 | fig5 (a)-(d)：α/β 网格热图+γ 扫描+oracle 平坦性+mass↔e2e 散点；fig11：四组合 mass 最优 vs e2e 最优双口径对照（排序反转直观化） | `exp/figures/fig5_abg_scan.py` + `exp/figures/fig11_e98_mass_vs_e2e.py`（无代号副本 fig11_mass_vs_e2e.pdf 供论文引用名） | E98 三个落袋 JSON 数据快照（脚本头注明「勿手改数字」） | —（图资产 pdf/png） | `sglang/paper/TLI_paper.tex` L82 (fig5_abg_scan.pdf)、L306 (fig11_mass_vs_e2e.pdf)；英文版 `TLI_paper_en.tex` L84、L310 |

## 口径速查（本轮踩坑固化）

- **mass vs e2e 预算口径差**：mass 网格 BP=64/B_TOK=2048；e2e K1=128/K2=1024。跨口径选臂必须按目标口径的有效配置去重（γ 截断坍缩实证存于 e98_e2e_grid.json `v1_fix` 字段）。
- **任务集口径**：E71/E72/E98 主表 13 任务含 multi_news、无 trec/samsum（与 E81/M6 口径不同）；任务键集从 `e71_main_table.json` 取，勿手写。
- **键名映射**：repobench-p（文件/目录名）↔ repobench（eval.py 键名）；multifieldqa_en 全集 150 条非截断。
- **method 组合名映射**：mavg=(minmax,avg)、mminmax=(minmax,minmax)、aavg=(avg,avg)、cavg=(cluster,avg)、ccluster=(cluster,cluster)；pred.py 只认全名（`--tli_far_method minmax --tli_near_method avg`），JSON tag 用组合名。

## 资产归档（2026-10-06 #144，E135 审计 P0）

- **per-sample raw 证据已从 /tmp 落袋**：E98 主表臂/E100 tail/E98 e2e 网格/E103/E90/E87/E89/MoBA → `exp/results_longbench/Qwen3-8B/`（pred_E98BEST_*、pred_E100TAIL_*、e98_e2e_grid/、e103_perqhead/、e90_subspace/、e87_topsigma_e2e/、e89_moba/）；E101/E104 RULER → `exp/results_ruler/Qwen3-8B/{L4096,L8192,L16384,L32768}/`
- **跑批脚本/日志/ledger/mass 分片** → `exp/trace_archives/`（e98/e100/e101/e103/e104/e87/e89/m6/tli_chain/e135 + MANIFEST.md + md5 清单）
- **trace dump（68GB）** → 本机 `~/.archive/trace-dumps/{qwen3-8b,qwen3-30b,qwen3-32b}`（不入 git，md5 见 exp/trace_archives/trace_dumps_md5/）
- 查 raw 数据时先读 `exp/trace_archives/MANIFEST.md`（资产→原 /tmp 路径→去向→支撑论文数字）
