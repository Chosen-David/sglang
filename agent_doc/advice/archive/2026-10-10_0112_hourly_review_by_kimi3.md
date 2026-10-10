# kimi3 每小时代码审查报告 — 2026-10-10_0112

**审查分支**: `two-level-indexer`
**唯一远端**: `github.com:Chosen-David/sglang`
**审查区间**: `9a59067fcd62cbfac9361ecb986f0bc66cb8d2ef..7675618a58c48e844b2d5a840829a7f2d3e62620`
**本轮时间标签**: 2026-10-10_0112

## 1. 审查范围说明

- 新提交 9 个（`7675618a5` 至 `ce1174865`），全部集中在审计/收口/测试脚本：
  - `two-level-attention/exp/trace/analyze_e117a_mavg_ref.py`（新建）
  - `two-level-attention/exp/trace/analyze_e119_ruler128k_formal.py`（新建）
  - `two-level-attention/exp/trace/analyze_e119_ruler64k_formal.py`（E116i 046③ 两残留修复）
  - `two-level-attention/exp/trace/e113_microbench.py`（E113 050 身份闭包修复）
  - `two-level-attention/exp/trace/test_e113_microbench_identity.py`（新建）
  - `two-level-attention/exp/trace/test_e119_crossarm_identity.py`（E116i 046③ 新增红绿用例）
  - `two-level-attention/exp/trace/testdata/e119_min/gen_e119_min_fixture.py`（128k 档位参数化）
- 核心路径 `python/sglang/srt/layers/attention/tli/` 与 `two-level-attention/sparse_attn/` **无代码变更**。
- 工作树未提交改动中，跳过 `paper/`、`ref/figs/`、`exp/results/` 等数据产物；仅 `research/docs/SGLang_TLI深度审查_新发现Bug清单_20261008_by_kimi3.md` 有实质追加，其中记录了本轮需报告的源码 bug。

## 2. 实测确认的发现

### Bug 1：near 区起点未扣除 SWA，α 扫描语义偏移（B10）

**状态**：源码未修，双实现同构，有实测证据。
**影响路径**：
- `two-level-attention/sparse_attn/indexer/tli_indexer.py:796-806`
- `python/sglang/srt/layers/attention/tli/indexer.py:235-242`

#### 问题描述

根据 `TASK.md` L53/L137 的权威定义：
- `mid = S − sink − swa`
- `near_L = α × mid_L`，区间应为 `[S − swa − near_L, S − swa)`

但当前双实现中：
1. `near_len_dyn = max(bs, int(α × mid_len))` —— 已从 mid 计算，正确；
2. `near_blks = max(sink_blocks, (S − near_len_dyn) // bs)` —— **从完整序列末尾 `S` 往前推，未扣除 `swa`**；
3. `swa_lo_blk = max(near_blks, kt − swa_tok//bs)` —— 正确地把 near 池上界切到 SWA 起点。

结果：near 池被 SWA 扣了两次（`mid_len` 已扣一次，上界再扣一次），实际 near POOL = `near_len_dyn − swa_tok`。

#### 实测证据

执行 `/tmp/verify_alpha_near2.py`（仓库外既有验证脚本，直接复用）：

```text
场景: S=4416, sink=128, swa=192, bs=64, α=0.5
mid_L (spec) = 4096
spec near_L = 2048 tokens (32 blocks)
代码 near_blks = (4416 − 2048)//64 = 37 → near 起点 = 2368
swa_lo_tok = 4416 − 192 = 4224
实际 near POOL = 4224 − 2368 = 1856 tokens (29 blocks)
差距 = 192 tokens = swa_tok
✓ 指控成立: near POOL = near_L − swa_tok (被扣了两次 swa)
```

执行 `/tmp/verify_alpha_impact.py` 得到偏差表（节选）：

| S | α 标称 | eff_α（实际生效） | 相对偏差 |
|---|---|---|---|
| 4096 | 0.125 | 0.085 | **+32.2%** |
| 4096 | 0.500 | 0.458 | +8.5% |
| 8192 | 0.125 | 0.106 | +15.4% |
| 16384 | 0.125 | 0.116 | +7.6% |
| 32768 | 0.500 | 0.495 | +1.0% |

规律：**序列越短、α 越小，偏差越大**。E109 最优 α=0.125 在 4k-8k 任务上实际生效 α≈0.085-0.106。

#### 影响

- **不污染 E109 已落袋数据的相对结论**：双实现（HF 权威实现 + SG 生产实现）完全同构，所有臂承受相同偏移，内部排序和最优 α 选取保持一致。
- **影响论文叙事与可复现性**：若论文写 "near_L = α × mid_L"，读者按 spec 复现会得到与代码不同的 near 区。
- **修复后最优 α 会小幅平移**，需在 E109 收官后补一组 α 扫描验证平移幅度。

#### 修复建议

双实现同步修改一行：

```python
# tli_indexer.py L803
near_blks = max(self.sink_blocks, (kt * bs - swa_tok - near_len_dyn) // bs)

# python/sglang/srt/layers/attention/tli/indexer.py L239
near_blks = max(p.sink_blocks, (S - swa_tok - near_len_dyn) // bs)
```

修后 near POOL = `(S − swa) − (S − swa − near_len_dyn) = near_len_dyn = α × mid_L`，与 spec 一致。

#### 复核方法

1. 重新跑 `/tmp/verify_alpha_near2.py` 与 `/tmp/verify_alpha_impact.py`，确认 `eff_α == α`；
2. 在修复后的分支上跑 `test_e113_microbench_identity.py` 与 `test_e119_crossarm_identity.py 64k/128k`，确保分区相关单测/收口测试仍通过；
3. 重跑至少一组 E109 α 扫描（如 4k/8k），观察最优 α 是否平移、相对排序是否保持稳定。

## 3. 已验证无异常的新提交

以下新提交/脚本经实测均按设计工作，未发现新的 bug 或指标污染：

| 脚本/提交 | 验证动作 | 结果 |
|---|---|---|
| `analyze_e117a_mavg_ref.py` | `python3 analyze_e117a_mavg_ref.py --dry-run` | patch 红绿自检通过，干跑输出正常 |
| `analyze_e119_ruler128k_formal.py` | 对真实 128K 产物跑汇总 | exit=0，三臂身份/脚本 SHA/发布协议闭包全过，输出 `mavg 47.49 > FullKV 46.43 > aavg 42.56` |
| `analyze_e119_ruler64k_formal.py`（E116i 046③ 修复后） | 对真实 64K 产物跑汇总 | exit=0，legacy 协议路径正常 |
| `test_e113_microbench_identity.py` | 直接执行 | 4/4 PASS |
| `test_e119_crossarm_identity.py` | 分别执行 `64k` 与 `128k` | 16/16 PASS（两档） |

**046③ 两残留修复验证**：N10（v2 缺必需角色）与 N11（读交换 bytes 快照绑定）在红绿单测中均 fail-closed 拒收；N11c 控制组（完整代际更新先于消费）正确通过，证明修复不误伤合法更新。

## 4. 未报告项

- **C1/C2/C3 等已确认未修旧问题**：本轮无相关源码变更，按纪律不重复报告。
- **B04/B09**：工作树文档中再次确认，但属既有缓期问题，无新提交触及相关代码，不重复报告。
- **性能优化点**：本轮变更均为审计/收口/测试脚本（非热路径），未发现需给出量级依据的性能退化或可优化点。

## 5. 结论

本轮审查区间发现 **1 个源码 bug**（B10，near 起点未扣 SWA），已附实测证据、影响评估与修复方案。新提交的消费者闭包修复（046③）与身份闭包修复（050）经单测/真实数据汇总验证均按设计生效。

---
*报告生成：kimi3，2026-10-10_0112*
*复核入口：执行 `/tmp/verify_alpha_near2.py`、`/tmp/verify_alpha_impact.py` 及 `test_e119_crossarm_identity.py [64k|128k]`*
