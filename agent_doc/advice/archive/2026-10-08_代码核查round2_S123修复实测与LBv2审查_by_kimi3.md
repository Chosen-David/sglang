# 代码核查 Round 2：S1/S2/S3 修复实测验证 + LB v2 新代码审查

日期：2026-10-08。作者标识：by_kimi3。核查基线：`two-level-indexer` @ `0bf18338b`（工作树干净）。
性质：修复回归验证 + 新变更审查。**本轮全部关键结论均有实测证据**（GPU 运行时测试 / 张量级复现 / 功能测试），未修改任何文件。

---

## 1. 结论总表

| 项 | 判决 | 证据等级 |
|---|---|---|
| S1（taskmd e64 哨兵 S_r/全局 S 混用，因果泄漏） | **修复有效，已实测** | GPU 运行时多行场景 |
| S2（swa +inf 复活哨兵，两位点） | **修复有效，两位点均在** | 源码 + 张量级机制复现 |
| S3（per-request `_select_taskmd` 池不足无 keep 掩码） | **修复有效，已实测** | GPU 运行时 |
| 消费端 row_bound 纵深防御 | **语义精确（含 chunked prefill）** | 调用链数学核对 |
| B01/B02 既有修复 | 回归通过 | 官方测试 3/3 复跑 |
| LB v2 新代码（`75268f877`） | **无 bug**；1 个部署缺口（§4） | 9 组功能测试 + 审查 |
| **本轮新发现 bug** | **0**（已修复链路未见新引入问题） | — |

## 2. S1/S2/S3 修复的实测证据（`e4e005df7`）

### 2.1 S1：多行 e64 场景（单行测试构造上抓不到的场景，本次补测）

配置 (α,β,γ)=(.125,.375,.625)，S=4096，`t_arr=[2000, 4095]`（行 0 near 池仅 145 token < 配额 768 → 大量无效槽位）。GPU 实测 `select_batched` 输出：

```
行0: 哨兵(S=4096)=4984  因果越界有效位(>2000)=0  值恰为2001(修复前哨兵)=0
各行有效位 ≤ t_r 检查: PASS
```

修复前，行 0 的无效槽位会写成 `S_r=2001` 并被消费端 `valid=sel<S` 接受（读入未来 token）；实测现在全部为全局 S 并被屏蔽，**因果泄漏 0**。

消费端 row_bound 一致性：`backend.py` 调用点 `t_arr = arange(prefix, S)`（L849）→ `row_bound[r] = S-nq+r+1 = t_r+1` 逐行相等；chunked prefill 每个 chunk 自带 (prefix,S)，同式成立。fused kernel 路径 `valid` 传递已确认（L1044-1047）。

### 2.2 S2：两位点 + 机制复现

- 位点 1（`select_decode_batched` skip_far，L2049-2052）与位点 2（`_select_decode_taskmd` 单池，L2506-2509）源码均带 `& valid`（上一轮只修了位点 1，`e4e005df7` 补齐位点 2）。
- 张量级复现（哨兵 lane `tok=S_cap`）：旧条件 `(tok>=sw_lo)` 下哨兵被打成 -inf 后**复活为 +inf=True**（bug 机制成立）；新条件 `(tok>=sw_lo)&valid` 下哨兵**保持 -inf=True**（修复有效）。

### 2.3 S3：per-request 池不足

同一配置 t=2000（decode n==1 路径）实测 `_select_taskmd`：`哨兵=6144，越界有效位=0` —— n==1 与 n≥2 decode 语义现已统一。

### 2.4 既有 B01/B02 回归

`test_b01_b02_fix.py` GPU 复跑 3/3 PASS（near 池不足哨兵 384/head、token0 恰 1 次、early 行 identity grid、消费端 1/640）。

## 3. LB v2 新代码审查（`75268f877`）

### 3.1 逐项核查（无 bug）

| 检查点 | 结果 |
|---|---|
| `extract_choice_letter`（pred.py）vs `lbv2_choice_score`（eval.py）双侧一致性 | 9 组样例逐值一致 ✓ |
| 解析优先级：官方式 `(X)` > 裸 `X` > 首字母兜底 | "A is wrong. The correct answer is (B)" → B ✓（官方式压过杂散字母） |
| 大小写 / 星号剥除 / 空串 / 无匹配 | 全过 ✓ |
| 加载分支：目录/文件两种 `--dataset-path`、缺文件抛清晰 FileNotFoundError | 审查通过 |
| 字段归一化 `answers=[answer]`、`all_classes=[A-D]`、`record["_id"]` | 审查通过 |
| `dataset2prompt/maxlen`：官方 0shot 模板、128 | 已就位 ✓ |
| scorer 注册与 v1 回归（preds 字段集不变） | 审查通过（commit 自述 hotpotqa 回归过） |

### 3.2 一个适用的既有 quirk（非新问题）

R01（q_input 无条件 `[:, 1:]` 砍首 token）对 lbv2 同样生效：`q_pos = rfind("Choices:")` → question 窗口第一个 token（"Choices" 的一部分）被砍。与 v1 全任务同性质——全臂一致、内部公平，但应并入 R01 修复时的统一处理（按 tokenizer 是否加 BOS 决定切不切），不要把 lbv2 漏掉。

## 4. 部署缺口（非代码 bug，需行动）

**LongBench-v2 的 `data.json`（503 题）不在 `~/datasets/` 任何位置**（`find` 全盘确认）。lbv2 分支的 FileNotFoundError 是按设计的清晰报错，但 `lbv2` 任务实际跑不起来——commit 自述的"冒烟 8 样本 accuracy=50%"用的数据路径已不可考（疑似 /tmp 被清理）。

**行动**：把官方 LongBench-v2 `data.json` 放到 `~/datasets/LongBench/data/lbv2.json`（或运行时显式 `--dataset-path` 指向它）。这是"全量三件套"口径的前置。

## 5. 仍未闭合项（沿前轮审查，状态更新）

| 项 | 状态 |
|---|---|
| R01（q_input 丢首 token，含 lbv2） | 未修（计划内缓到 E109 收官） |
| B04（首步 EOS）/ B09（命名） | 未修（计划内缓期） |
| HF F1（layer_skip × 分区丢 sink） | 未修（layer_skip 实验前必修） |
| 编排 F1（screen4 内嵌打分 repobench-p glob → 收官覆写） | **值守中**：链收官后必须重跑 `e109_score_v2.py` |
| SG fused L2 错误池（`tli_l2_partition_topk` 无 keep 掩码）/ q_agg=max decode 未生效 / PCA basis 不适配 TP | 未修（SG e2e 前处理） |
| RULER 协议近似 / C0·B7·B7s 脚本吞失败 / F02 失败 marker 放行 | 未修（RULER 阶段前） |
| **SG 修复后的 GPU e2e 对拍**（F01/S1 类 bug 的端到端验证） | **仍缺**——本轮验证了 selector 输出层，未做 SG 全链路 e2e；E112/E113 前必须补 |

## 6. 证据附录

| 证据 | 位置 |
|---|---|
| S1/S3 GPU 运行时测试脚本（可复跑） | 本会话临时脚本（多行 e64 + per-request 场景；建议固化进 `test_b01_b02_fix.py` 同款目录，覆盖单行测试盲区） |
| B01/B02 官方测试输出 | `test_b01_b02_fix.py` 复跑 3/3 PASS（本会话） |
| S2 张量级复现 | 旧/新条件对照（哨兵复活 vs 保持 -inf） |
| lbv2 功能测试 | 9 组样例 + 双侧一致性断言（全 PASS） |
| 消费端一致性 | `backend.py:849` `t_arr=arange(prefix,S)` ↔ `_sparse_extend_one` row_bound 数学恒等 |

**建议**：把 §6 的多行 e64 测试固化成 `test_s1_multirow_fix.py`（单行 `test_b01_b02_fix.py` 在构造上抓不到 S1 类回归——本轮已实测补齐）。
