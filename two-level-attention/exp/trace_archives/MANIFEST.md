# /tmp 实验资产归档清单（任务 #144，2026-10-06）

> 拷贝归档（cp -a，保留 mtime），/tmp 原文件未删除。仓库文件 md5 见 `md5_repo_archives.txt`；trace dump 逐文件 md5 见 `trace_dumps_md5/`（与 `~/.archive/trace-dumps/md5_*.txt` 逐位同源，抽样三处已核对一致）。
>
> **gitignore 注意**：sglang 根 .gitignore 全局忽略 `*.jsonl` / `*.log` / `*.png` / `*.pdf`，本次归档的全部 jsonl 与 log 均以 `git add -f` 强制入库。既有 `pred_1024/` 此前也只有 result.json 进了 git（85 个 per-sample jsonl 仅存在于工作区）；本次顺带强制入库其中 FullKV 臂（`*-none-*.jsonl` 13 文件）——E135 审计 +0.42 CI 判决的 PSI/FullKV 配对至此才真正双双落 git。

## 一、per-sample 原始记录 → 入 git 仓库

## 一、per-sample 原始记录 → 入 git 仓库

| 资产 | 原路径（/tmp） | 体积 | 去向（exp/results_…/Qwen3-8B/） | 文件数 | 支撑的论文/判决数字 |
|---|---|---|---|---|---|
| E98 主表臂 13 任务 | /tmp/e98_full/pred_E98BEST_mavg_a0.125_b0.375_g0.625 | 3.0M | results_longbench/pred_E98BEST_mavg_a0.125_b0.375_g0.625 | 13 jsonl | 主表 50.78、+0.42 CI、sign test（E135 A 组全部 verified） |
| E100 tail32 L1 臂 13 任务 | /tmp/e100_tail/pred_E100TAIL_mavg_a0.125_b0.375_g0.625 | 3.0M | results_longbench/pred_E100TAIL_mavg_a0.125_b0.375_g0.625 | 13 jsonl | tail 口径 50.53、−0.25 CI |
| E98 e2e 网格 15 臂（hq/mu） | /tmp/e98_e2e/pred_{mavg,mminmax,aavg,cavg}_* | 904K | results_longbench/e98_e2e_grid/ | 15 目录 30 jsonl | e2e 网格判决、γ 截断坍缩实证 54.16/35.57、fig5(d)/fig11 |
| E103 kv-head 消融 B/C 臂 | /tmp/e103_perqhead/{pred_B,pred_C} | 124K | results_longbench/e103_perqhead/ | 4 jsonl | E103 hq 53.79 vs 55.44 |
| E90 子空间五臂（hq/mu） | /tmp/e90_sub/pred_sub{full,rope,nope,random,highfreq} | 304K | results_longbench/e90_subspace/ | 5 目录 10 jsonl | E90 排序 rope64 55.98 > full128 > tail32 > … |
| E87 top-σ e2e 臂 | /tmp/e87_e2e/pred_sig{near,mid,far}_{2,8,32}、/tmp/e87c_tail | 532K | results_longbench/e87_topsigma_e2e/ | 16 jsonl | 消融 σ8 hq 55.12 弹性臂 |
| E89 MoBA baseline 13 任务 | /tmp/e89_moba/pred_moba | 3.0M | results_longbench/e89_moba/ | 13 jsonl | MoBA 49.19 全量（观察章） |
| E101 RULER 主臂三长度 | /tmp/e101_ruler/{L4096,L8192,L16384}/pred_E101RULER_* | 872K | results_ruler/{L4096,L8192,L16384}/pred_E101RULER_mavg_a0.125_b0.375_g0.625 | 3×11 jsonl | RULER 85.83 |
| E104 RULER 32K 六臂 | /tmp/e104_ruler_32k/L32768/{pred_E104MAIN,REF,B_C0,B_FULLKV,B_QUEST,B_TIA} | 2.1M | results_ruler/L32768/ | 6 目录 66 jsonl | 单池 60.77 ≈ 分区 60.78 |

## 二、trace dump（attention 逐层 K/V dump，不可再生）→ ~/.archive/trace-dumps/（不入 git）

| 资产 | 原路径（/tmp/trace/） | 体积 | 文件数 | 归档路径 | md5 清单 |
|---|---|---|---|---|---|
| Qwen3-8B 16 LB 样本 + needle32k + natural32k | /tmp/trace/qwen3-8b | 27G | 592 | ~/.archive/trace-dumps/qwen3-8b | trace_dumps_md5/md5_qwen3-8b.txt |
| Qwen3-30B hotpotqa 2 样本 | /tmp/trace/qwen3-30b | 1.8G | 98 | ~/.archive/trace-dumps/qwen3-30b | trace_dumps_md5/md5_qwen3-30b.txt |
| Qwen3-32B 14 LB 样本目录 | /tmp/trace/qwen3-32b | 41G | 910 | ~/.archive/trace-dumps/qwen3-32b | trace_dumps_md5/md5_qwen3-32b.txt |

生成脚本（在库，可部分再生但需 GPU 数小时）：`exp/trace/collect_trace_lb.py`、`exp/trace/collect_trace_task.py`。E64–E90/E98/E105/E108 全系 mass 实验的数据源。

## 三、脚本 / 日志 / ledger / 分片 → exp/trace_archives/e<NN>/（入 git）

| 子目录 | 内容 | 原路径前缀 | 体积 |
|---|---|---|---|
| e98/ | mass 网格跑批+16 分片、e2e 网格 v1/v2+runs ledger v1/v2、13 任务全量+16 进程日志、e99 stdout（43 文件） | /tmp/e98_* | 3.2M |
| e100/ | tail 全量脚本+日志+修订模板（3 文件） | /tmp/e100_* | 200K |
| e101/ | RULER 跑批+核对（3 文件） | /tmp/e101_* | 232K |
| e103/ | kv-head 消融全链：双卡脚本/日志/mask dump pt×2/打分/单测/冒烟（16 文件） | /tmp/e103_* | 536K |
| e104/ | RULER 32K 两轮跑批+生成器调用脚本（6 文件） | /tmp/e104_*, /tmp/ruler_gen_32k.sh | 436K |
| e87/ | top-σ 筛臂双卡+σ 校准+E87c tail 校准（11 文件） | /tmp/e87* | 192K |
| e89/ | MoBA 全量日志+冒烟脚本（2 文件） | /tmp/e89_* | 32K |
| m6/ | sinkguard 三方法跑批+修复（9 文件） | /tmp/m6_*, /tmp/repair_* | 200K |
| tli_chain/ | E71/E72 编排链全目录 39 文件（b7s_ruler_final.json、E64J/E67/E72 ledger、orchestrator 日志） | /tmp/tli_chain | 420K |
| e135/ | 审计脚本+复算中间产物（2 文件） | /tmp/e135_* | 12K |

## 四、跳过项与理由

| 资产 | 理由 |
|---|---|
| /tmp/e105_full、/tmp/e105_cavg_remote、/tmp/e105_* | E105 实验在跑仍在增殖，等完成后另行归档 |
| /tmp/e107_pool_baseline.sh/.log | 等 ALL_E105_ARMS_DONE 接力（#129 未执行） |
| /tmp/papers/（4.5M）、/tmp/two_level/（564M） | 外部仓库 clone，可再 |
| /tmp/ruler_official（1.2M） | RULER 官方生成器 clone（github 可再）；调用脚本 e104/ruler_gen_32k.sh 已入库 |
| /tmp/ruler_32k_data、/tmp/ruler_32k_raw（各 125M） | RULER 合成输入数据，可由 ruler_official + ruler_gen_32k.sh 再生；E104 per-sample 已含 input/pred/answers |
| /tmp 其余 tli_* / b5-b7 / 30b / c3 等散件 | E1–E72 时代过程日志，判决均已落袋 exp/trace/results/（E135 审计范围外）；如需可后补 |
