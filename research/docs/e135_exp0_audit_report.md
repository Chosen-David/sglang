# E135 / EXP0：raw 数据可复算性审计报告

**日期**：2026-10-06　|　**审计模式**：只读（零 GPU 推理，仅 CPU/I/O 级验证）　|　**审计 JSON**：`two-level-attention/exp/trace/results/e135_exp0_audit.json`

**回应对象**：`research/docs/PSI_实验设计审稿意见与补实验计划_20261006.md` EXP0 节——「验证当前论文分数、区间和速度能否追到相同实现与真实样本；能从 immutable 输入与 raw 记录重算主要表格及配对区间，再决定最小重跑范围。缺失继续标 reported/unverified；不由汇总反推方差，不凭文件存在判为复现成功」。

**总判决：21 项审计项，15 verified / 6 summary_only / 0 missing。质量侧（分数与区间）全部可从 per-sample raw 记录 bit 级复算；计时侧字段自洽但测量协议缺陷未补齐，维持 reported/unverified。**

---

## 一、质量数据（A 组）：全部核心主张可关闭

方法：独立审计脚本从 per-sample jsonl（`pred` + `answers`）按 `benchmark/LongBench/eval` 官方 scorer 口径逐例重打分 → 任务均值 → 13 任务等权 AVG；bootstrap 按记录的 B=10000 / seed=20261004 / 原 rng 消耗顺序复跑。**不由汇总反推，全部从逐例原始记录重算。**

| 审计对象 | 论文值 | 重算结果 | 判决 |
|---|---|---|---|
| E98 主表臂 13 任务 | **50.78** | 50.78（13/13 任务逐位一致） | ✅ verified |
| FullKV 13 任务 | **50.36** | 50.36（13/13 逐位） | ✅ verified |
| +0.42 的 95% CI | **[-0.18, +1.02]** 含 0 | 逐位复现 | ✅ verified |
| held-out 11 任务 | +0.08 [-0.46, +0.60] | 逐位复现 | ✅ verified |
| musique / hotpotqa 单任务 CI | [-1.29,+6.37] / [-1.44,+5.57] | 逐位复现 | ✅ verified |
| sign test | 8 胜 5 负 p=0.5811 | 复现 | ✅ verified |
| E100 tail32 L1 臂 | 50.53，vs full **-0.25** [-0.57,+0.06] | 13/13 逐位 + CI 逐位复现 | ✅ verified |
| E98 mass 全网格 | 3600 有效臂 | 3600/3600 mean 可由 per_sample 复算（mismatch=0） | ✅ verified |
| γ 截断坍缩实证 | **54.16 / 35.57**（g0.75≡g0.375） | 两臂逐位复现（n=200） | ✅ verified |
| E101 RULER 主臂 | **85.83** | 11 任务×3 长度逐位复现 | ✅ verified |
| E104 RULER 32K | 单池 **60.77** ≈ 分区 60.78 | 6/6 臂逐位复现 | ✅ verified |
| E103 kv-head 消融 | B 臂 hq **53.79** vs 55.44 | B/C 四分数逐位复现 | ✅ verified |
| RULER 87.93 参照臂 | 87.93 | 仅 summary 级可核（三长度均值 91.49/88.62/83.68 验算通过）；**per-sample 未定位** | ⚠️ summary_only |

**审计锚点修正**：任务书对照值「musique 34.71/32.28」中的 32.28 实为论文主表 **TIA 列**（TLI_paper_en.tex L268）；FullKV musique = **32.14**，diff = 2.57。论文主表本身自洽。

**措辞风险提示**：摘要「musique 34.71 为含 FullKV 在内所有方法最高」在主表方法列内成立（外部基线最高 TIA 32.28），但内部 TLI 变体（γ.125 参照臂 34.76、E100 tail 臂 35.04）高于 34.71——建议限定为「主表各方法列中最高」。

---

## 二、计时数据（B 组）：字段可核、协议不可复算

### 字段核对（全部一致）

| 臂 | total_s | prefill_s | 判决 |
|---|---|---|---|
| Quest（TP2 bs16 64K） | 62.06 | 54.43 | 字段 ✓ |
| dense triton | 106.13 | 102.25 | 字段 ✓ |
| PSI | 141.95 | 136.03 | 字段 ✓ |

派生算术自洽（decode=total−prefill、ms/step 按 63 有效步、tok/s=bs×64/total 全部重算通过；PSI 臂 ms/step 93.9 vs 舍入重算 94.0 为舍入路径差异，非错误）。

### 口径核对

生成脚本 `sglang/test_c3_3arm_tp2.py` 在库，确认两遍相减法：pass1（max_new_tokens=1）= 纯 prefill；pass2（max_new=64）= total；decode = 差值 ÷ 63 步；`disable_radix_cache` 保证两遍独立 prefill；cuda graph 双段 disabled；单次 hello 预热。

### 审稿人指出的口径缺陷——**全部属实，如实列为 unverified/须重测**

1. **单次运行、无重复**：无中位数、无方差（TP2 三臂各只跑一遍）
2. **无 EOS / 实际生成 token 数**：`n_decode=64` 是请求值非实测值
3. **无每步 timestamp、无墙钟日期 / GPU 状态**
4. 生成文本仅存 80 字符截断；prompt token 数未记录（仅 chars=260000）

单请求三臂（`c3_3arm_e2e.json`）可核性高一档：JSON 内保存 pass1/pass2 原始计时各 2 次（min 规则 + 差值算术复算 9/9 通过），但 EOS 缺口同在。旧 `tli_64k_tp2_*.json`（total-only 115.28/106.24）已被新口径取代，论文不得再引用。

另：E102 如实记录的 TLI 侧 +23% 回归（115.28→141.95，归因未结）重测时须一并归因。

---

## 三、harness 完整性（C 组）：全部 verified

- **selector 代码**：`python/sglang/srt/layers/attention/tli/`（backend/indexer/config/kernels + 5 测试）与 `quest/`（backend/indexer/config + 测试）全部 commit 在库
- **打分链**：`run_e71_eval.py` 在库（方法→pred 目录映射显式）；scorer = 在库的 `benchmark/LongBench/eval.py`（本审计逐例重打分与其逐字一致）；模型/数据路径记录在生成脚本（Qwen3-8B=/mnt/dolphinfs/...、dataset=datasets/LongBench/data）
- **commit 绑定**：论文速度数字绑定 sglang `0a9fade`（当前 HEAD `657c4bfcd` 的祖先）；期间触及质量路径的仅 `3add02624`（#127 F1-F3），质量语义 bit-exact 由 `test_incremental_prefill.py` 背书（torch.equal 级 6 项验收）。LongBench 质量数字由 two-level 仓（transformers+sparse_attn，非 sglang）产出，数据 commit `db1838d` 后质量路径仅两处加法式改动（E103 开关、RULER 32K CLI），工作区该路径干净

---

## 四、最大风险与最小重跑范围

### 🔴 存储脆弱性（本次审计最大发现）

FullKV per-sample 已入库（pred_1024，持久），但以下**全部只在 /tmp（机器重启即失）**：

> E98BEST 13 任务、E100TAIL 13 任务、e2e 网格 16 臂、E101/E104/E103 per-sample、b7s_ruler_final.json、mass 网格 16 分片、runs ledger v1/v2、全部生成脚本与运行日志、RULER 官方生成器 clone

**今天全部 verified 的判决，都建立在 /tmp 还活着之上。**

### 最小重跑建议（按优先级）

1. **P0（零 GPU，立即）**：上述 /tmp 资产迁入 two-level 仓库归档目录——这是把 15 项 verified 判决从「暂时可复算」变成「持久可复算」的唯一动作
2. **P1（GPU ~1-2h，关审稿 C3）**：TP2 三臂重测——≥3 次重复取中位数 + 实际 token 数/EOS + timestamp + prompt token 数；顺带归因 TLI +23% 回归
3. **P2（GPU ~2-3h，可选）**：RULER γ.125 参照臂 87.93 重跑留 per-sample（仅当审稿人要求该数字逐例追溯）
4. **质量主表：无需重跑**——已从 immutable per-sample 输入 bit 级复算

### 主张关闭状态

- **可关闭**：50.78 / 50.36 / +0.42 及全部 CI / tail32 口径判决 / RULER 85.83 / 32K 六臂 / E103 / γ 坍缩实证
- **维持 reported/unverified**：TP2 三臂 e2e 计时（协议缺陷未补齐前）；RULER 87.93（逐例级）；失败运行登记（无正式登记表，选择顺序靠 /tmp ledger 部分可追溯）

---

*审计脚本：`/tmp/e135_audit_recompute.py`（audit 专用，未触碰任何实验代码）；复算中间产物 `/tmp/e135_audit_recompute.json`。两仓 commit 见本报告提交记录。*
