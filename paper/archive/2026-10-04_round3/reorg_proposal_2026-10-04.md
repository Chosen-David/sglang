# 重组提案 — 2026-10-04（round3 E98/M6/E99 收官后 code-organization 审计）

> 状态：**PROPOSE_AND_WAIT，未执行任何移动/删除**。所有建议动作等用户拍板后逐项执行（git mv 保留历史，一次一个逻辑组一个 commit，message 写明动机与回滚方式）。
> 边界：本仓库跑批脚本（含 /tmp 接力脚本）按绝对路径引用 `exp/trace/` 下脚本，移动任何脚本前必须 grep 全仓 + /tmp/*.sh 引用点并确认无运行中管线。

## 一、孤儿脚本（无结果落盘且无引用；grep 证据 = 全仓 *.py/*.sh/*.md 及 /tmp/*.sh 引用计数 0）

| # | 文件 | 证据 | 建议动作 | 风险 | 回滚 |
|---|---|---|---|---|---|
| O1 | `exp/trace/debug_tli_e5b.py` | 仅 b0a732d（E1-E8 期）提交；无对应 results JSON；全仓引用 0 | 归档到 `exp/trace/archive/`（git mv） | 极低（无引用） | `git revert <hash>` |
| O2 | `exp/trace/diag_l03_far.py` | 同 O1 | 同 O1 | 极低 | 同上 |
| O3 | `exp/trace/analyze_e71_debug.py` | 仅 343651e 提交；无 JSON；引用 0 | 同 O1 | 极低 | 同上 |
| O4 | `exp/trace/analyze_e71_swa_fix_dryrun.py` | 仅 e9acd1e；是「区域口径修复」（sink/swa 正交）的干跑验证记录，有历史证据价值 | **保留**或归档（二选一由用户拍板；建议保留——口径修复是论文叙事的一部分） | — | — |
| O5 | `tmp.py`（仓库根） | tilelang/torch 版本探针（4 行 print）；git tracked；引用 0 | git rm（删除留给用户拍板）或移 exp/trace/archive/ | 极低 | git revert |
| O6 | `exp/trace/analyze_e87b_sigma_calibration.py` | 脚本主 OUT `e87b_sigma_calibration.json` 不存在，仅 `_s0` shard 落盘（E87b 只跑了一个 shard，主 JSON 从未 merge） | 二选一：① 判决已由 E87/E88 后续收割 → 归档脚本 + `_s0` JSON；② 若需完整 sigma 校准数据 → 补跑 merge（约数分钟 CPU） | 低（merge 后可能改变 E87b 引用口径） | git revert |

## 二、重复实现 / 疑似重复（同功能多版本并存）

| # | 文件组 | 现状 | 建议 |
|---|---|---|---|
| D1 | `exp/figures/fig7_architecture.py` | **脚本名与产物名漂移**：脚本输出 `fig7_psi_architecture.pdf/png`（且 fig7_psi_architecture.py 曾存在后被改名，git status 有过渡痕迹）；论文正文引用的是 `fig7_tli_architecture.pdf`（tex L114）——同一架构图三个名字 | 用户拍板统一命名（建议产物定名 fig7_tli_architecture.pdf 对齐论文，脚本改名 + 改输出名 + grep 引用） |
| D2 | `exp/trace/plot_e64g.py` vs `exp/trace/plot_e64g_heat.py` | 同为 E64g 可视化（前者 3 图含 per-dataset panel，后者热图 + md 表），功能重叠 | 建议保留 `plot_e64g.py`（图更全），`plot_e64g_heat.py` 归档——以「哪个产物被报告/论文引用」为准，用户确认后执行 |
| D3 | `exp/figures/make_figures.py`（套件入口）vs 各 `plot_*.py` / `make_figN.py`（单图脚本） | 历史层积：早期统一入口与后期单图脚本并存，入口已不能覆盖新图 | backlog（低优先级）：归档 make_figures.py 或补 README 说明各自分工 |

## 三、命名漂移（同一概念多个名字，高危混淆点）

| # | 现象 | 建议 |
|---|---|---|
| N1 | **figN 双代编号并存**：旧代 `fig5_e8_speedup` / `fig8_m3_system` / `fig9_e2e_boundaries` / `fig10_m8_kernels`（make_fig8/9/10.py 产物）与新代论文正文 `fig5_abg_scan` / `fig11_mass_vs_e2e`（fig5/fig11_*.py 产物）同目录同号系；且 E98 的 fig8/9/10 由 `exp/trace/plot_e98_abg_grid.py` 生成到 **sglang/ref/figs/**，与 exp/figures 的旧代 fig8/9/10 同号不同物 | 高危混淆点。建议（须先 grep sglang/paper、报告 md 全部引用点）：旧代图 git mv 到 `exp/figures/legacy/`（或改名带实验代号前缀，如 `fig5_e8_speedup` → `legacy_fig5_e8_speedup`）。论文 tex 已核实的引用（fig1_h1_decomposition/fig2_dim_reduction_story/fig5_abg_scan/fig7_tli_architecture/fig11_mass_vs_e2e）不受影响 |
| N2 | method 组合名（mavg/mminmax/aavg/cavg/ccluster）vs pred.py 参数全名（minmax/avg/cluster）两套表示 | 已在 CODEMAP.md「口径速查」固化映射表，不改代码 |
| N3 | repobench-p（文件/目录名）vs repobench（eval 键名） | 已在 CODEMAP.md 与 exp/trace/results/README 固化；后续打分脚本建议显式注释 |

## 四、待补落袋（非孤儿，结果只在报告无 JSON）

| # | 文件 | 现状 | 建议 |
|---|---|---|---|
| B1 | `exp/trace/e8_2_fused_topk.py` / `e8_2_l2_cascade.py` | E8-2 kernel 级 microbench 原型（论文速度口径素材），结果只在报告正文，results/ 无 JSON（仅 `e8_1_index_flop.json`） | 建议补 JSON 落袋（kernel 对拍 + 时延），保 microbench/e2e 双口径铁律可审计 |

## 五、执行清单（授权后逐项）

1. 用户对 O1-O6、D1-D3、N1、B1 逐项拍板（删/留/归档/改名/补跑）；
2. 每个逻辑组一个 commit：`组织：X→Y，动机，回滚=git revert <hash>`；
3. 移动后 grep 旧路径引用并同步修复，目标脚本 `--help`/干跑最小验证；
4. CODEMAP.md 增量更新受影响行。

**本轮已执行项：0 个移动、0 个删除**（仅新增文档：CODEMAP.md / exp/trace/results/README.md / 本提案）。
