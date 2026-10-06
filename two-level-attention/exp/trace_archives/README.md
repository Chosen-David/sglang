# /tmp 实验资产归档（E135 审计 P0 动作，任务 #144）

**归档日期**：2026-10-06　|　**动机**：E135 EXP0 审计（`exp/trace/results/e135_exp0_audit.json`）发现论文质量主张依赖的 per-sample raw 证据此前只存于 /tmp（容器重启即失），本目录将其落袋为持久副本。

**归档规则**（体积纪律）：
- per-sample pred 数据 → 按既有约定入 `exp/results_longbench/Qwen3-8B/` 与 `exp/results_ruler/Qwen3-8B/`（与 pred_1024/pred_e72_mavg 等同层）
- 生成脚本 / 运行日志 / runs ledger / mass 网格分片（<50MB 小文件）→ 本目录 `e<NN>/` 子目录
- trace dump（68GB，不可再生）→ 本机 NVMe 持久区 `~/.archive/trace-dumps/`（不入 git），逐文件 md5 清单见本目录 `trace_dumps_md5/`
- 仓库内全部归档文件的 md5 → `md5_repo_archives.txt`
- 完整资产清单（原路径 / 体积 / 去向 / 理由）→ `MANIFEST.md`

**目录内容**：
- `e98/`：mass 全网格跑批脚本+16 分片 JSON、e2e 网格 v1/v2 脚本+日志+runs ledger（e98_e2e_runs.json / _v2.json）、13 任务全量脚本+日志+16 进程日志、e99 stdout
- `e100/`：tail32 L1 臂 13 任务全量脚本+日志+论文修订模板
- `e101/`：RULER γ.625 主臂三长度跑批脚本+日志+核对脚本
- `e103/`：kv-head 共享消融双卡脚本+日志+mask dump+打分脚本+单测+冒烟
- `e104/`：RULER 32K 六臂跑批脚本+日志+两轮（main/5arms）+RULER 生成器调用脚本
- `e87/`：top-σ 筛臂双卡脚本+日志+σ 校准日志+E87c tail 校准脚本与分数
- `e89/`：MoBA baseline 全量日志+冒烟脚本
- `m6/`：sinkguard 三方法（snapkv/h2o/pyramidkv）跑批脚本+日志+修复脚本
- `tli_chain/`：E71/E72 时代编排链全目录（含审计点名的 b7s_ruler_final.json、E64J/E67/E72 链 ledger 与 orchestrator 日志）
- `e135/`：审计脚本 e135_audit_recompute.py 与复算中间产物 JSON

**对应关系**：各 eNN 的判决 JSON 在 `exp/trace/results/`、分析脚本在 `exp/trace/`（见仓库根 CODEMAP.md）；本目录只保存跑批 shell、日志、ledger 与 per-sample 原始记录。

**跳过未归档**（理由见 MANIFEST.md）：/tmp/e105_full、/tmp/e105_cavg_remote（实验在跑仍在增殖）、/tmp/e107_pool_baseline（等 E105 接力）、/tmp/papers 与 /tmp/two_level 与 /tmp/ruler_official（外部 clone 可再生）、/tmp/ruler_32k_data 与 /tmp/ruler_32k_raw（RULER 合成数据，可由 ruler_gen_32k.sh + ruler_official 再生）。

**纪律**：本归档只拷贝不删除，/tmp 原文件保留，删除永远留给用户。
