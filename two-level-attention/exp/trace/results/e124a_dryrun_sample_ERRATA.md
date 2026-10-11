# ERRATA：e124a_dryrun_sample 几何错配说明（2026-10-11）

本目录（`e124a_dryrun_sample/`）与 `../e124a_dryrun_summary.json` 为 E124a A1
首版交付，**含几何不一致缺陷，已降格为「trace 特征回放 + 给定几何预算预览」，
不支持因果/保护/容量闭包验收**（GPT A1 验收反馈裁定）。原文件为已入库历史，
保留不动、不回改。

## 错配机制（三条，GPT 指认 1 条 + 主 AI 核实加深 2 条）

1. `run_e124a_dryrun.py --n-valid` 旧默认**硬编码 32768**（非 None），
   `n_valid = args.n_valid if args.n_valid else int(S)` 的 meta.S 回退分支
   永不触发——trace `lb_hotpotqa_0/meta.json` 实际 S=16957 被静默覆盖。
2. mid 切片 `k_all[128: 32768-128]` 在 S=16957 时被 Python **静默钳制**为
   `k_all[128:16957]`（16829 行）——SWA 尾段 128 token **未被切除**，混进
   特征 middle。
3. 于是同一决策记录里特征几何 16829（含 SWA）与预算几何 32512
   （按 n_valid=32768 编译，near_range=[16384,32640] 越出实际序列长 16957）
   两套口径混用。

## 证据

- 9 条 `per_seq_layer_decisions_lb_hotpotqa_0.jsonl` 记录全部
  `features.n_mid=16829` ≠ `n_valid-n_protected=32512`；
  `summary.geometry.n_valid=32768` ≠ trace meta.S=16957。

## 修复与正例

- 修复：`--n-valid` 默认改 None（meta.S 回退生效）+ 显式值与 trace 实际
  K 长度双向核对 + 切片前防钳制断言 + decide() 消费侧同源绑定
  （k_mid 行数 == max(0, n_valid−n_protected)，不一致 fail-closed）+
  qpos 因果核对入记录。三档规则与 M1/M2/M3 数值语义零改动。
- 正例（同源记录，n_mid=16701=16957−256）：见
  `../e124a_dryrun_v2_sample/` 与 `../e124a_dryrun_v2_summary.json`。
- 红绿套件：`exp/trace/test_e124a_geometry_consistency.py`
  （本目录 9 条错配记录作拒绝负例；显式 `--n-valid 32768` 重放 →
  `[E124A-ABORT][GEOM-MISMATCH]` 非零退出）。

依据：`agent_doc/advice/2026-10-11_Runtime_Sequence_Layerwise_Execution_Plan_by_gpt.md`
「A1 验收反馈（2026-10-11 UTC）」节及主 AI 回应。
