# 2026-10-10 20:41 sglang two-level-indexer 小时审查报告

**审查区间**：`01d4103eb004a944587b16f83cb0ace5b8a399a1..7c3190921cc2c20326487ad21889caaefab9a5cc`（origin/two-level-indexer）

**工作树过滤**：仅 `agent_doc/advice/archive/`、`paper/archive/scores.json`、`research/docs/`、`two-level-attention/exp/trace/results/*.json` 等数据/文档产物有改动；无未提交业务代码变更。

**审查重点**：`python/sglang/srt/layers/attention/tli/`、`two-level-attention/sparse_attn/`、`two-level-attention/benchmark/LongBench/pred.py`，以及新增 E121/E122 测试。

---

## 发现 1：B10 near 边界修复导致 `test_b01_b02_fix.py` 期望失效（回归门禁损坏）

### 位置
- `python/sglang/srt/layers/attention/tli/test_b01_b02_fix.py:42`
- `python/sglang/srt/layers/attention/tli/test_b01_b02_fix.py:44`
- `python/sglang/srt/layers/attention/tli/test_b01_b02_fix.py:73`（场景 3 的 `expect = 1.0 / 640`）

### 实测证据
在当前工作树（含区间内的 B10 修复）执行：

```bash
cd /home/wangyuanshuo02/sglang
python3 python/sglang/srt/layers/attention/tli/test_b01_b02_fix.py
```

输出：

```
AssertionError: 哨兵数 2048 != 3072
```

直接复现脚本（绕过 assert，读取当前代码实际输出）：

```python
import os
os.environ.setdefault("SGLANG_TLI_ALPHA", "0.125")
os.environ.setdefault("SGLANG_TLI_BETA", "0.375")
os.environ.setdefault("SGLANG_TLI_GAMMA", "0.625")
import sys
sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
import torch
from sglang.srt.layers.attention.tli.config import TLIProfile
from sglang.srt.layers.attention.tli.indexer import TLIIndexer

dev = "cuda:0"
torch.manual_seed(20261008)
Hkv, D, H = 8, 128, 32
S = 4096
prof = TLIProfile()
idxer = TLIIndexer(prof, head_dim=D).to(dev)
k_real = torch.randn(S, Hkv, D, device=dev) * 0.1
index = idxer.build_block_index(k_real)
q = torch.zeros(1, H, D, device=dev)
t_arr = torch.tensor([S - 1], device=dev)
sel = idxer.select_batched(index, q, t_arr)
sent = (sel == S).sum().item()        # 2048 = 256/head
tok0 = (sel == 0).sum().item()        # 8   = 1/head（仍满足）
n_valid = (sel < S).sum().item() // Hkv  # 768
print(f"sent={sent}({sent//Hkv}/head) tok0={tok0} n_valid={n_valid}")
```

实测结果：

| 指标 | 测试当前期望 | 当前代码实际 |
|------|------------|------------|
| `sent` | 3072（384/head） | **2048（256/head）** |
| `tok0` | 8（1/head） | **8（1/head）** ✓ |
| `n_valid` | 640 | **768** |

### 根因分析
B10 修复（提交 `a4faad59f`、`384dabe41` 等）将 e64 分区臂的 near 左界从序列末尾改为从 swa 起点推，使 near 区实际宽度 = α·mid_len，避免 swa 被重复扣除。

以该测试场景 `S=4096, bs=64, sink=128, swa=128, α=0.125, β=0.375, γ=0.625` 为例：

- **修复前**（旧口径）：
  - `near_len_dyn = max(64, int(0.125 * 3840)) = 480`
  - `near_blks = (4096 - 480) // 64 = 56`
  - near 池 = 块 56–61（不含 swa） = 6 块 = **384 token**
  - `K2_mid = 1024 - 128 - 128 = 768`，`far_budget = 0`
  - 实际选中 mid token = 384，sentinel = 768 - 384 = **384/head**
  - `n_valid = 128(sink) + 384(near) + 128(swa) = 640`

- **修复后**（B10 新口径）：
  - `near_base = 4096 - 128 = 3968`
  - `near_blks = (3968 - 480) // 64 = 54`
  - near 池 = 块 54–61 = 8 块 = **512 token**
  - 实际选中 mid token = 512，sentinel = 768 - 512 = **256/head**
  - `n_valid = 128 + 512 + 128 = 768`

测试文件 `test_b01_b02_fix.py` 在 B10 修复提交中未被同步更新，导致原本用于验证 B01（sentinel 用 S 而非 0）和 B02（early 行 dense 等价）的门禁断言失效。

### 影响
- B01/B02 回归测试当前处于**失败状态**，无法继续守护 sentinel 语义和 token 0 唯一性等核心不变量。
- 场景 3 的数学期望 `1.0 / 640` 也会因 `n_valid` 变为 768 而失败（实测未执行到该断言，因场景 1 已提前退出）。

### 修复建议
仅更新测试期望以匹配 B10 修正后的几何，不涉及业务逻辑改动：

```python
# 场景 1
assert sent == Hkv * 256, f"哨兵数 {sent} != {Hkv*256}"
assert tok0 == Hkv * 1, f"token 0 出现 {tok0} 次 != {Hkv}"
assert n_valid == 768, f"有效位 {n_valid} != 768"

# 场景 3
expect = 1.0 / 768
```

同时建议在该测试文件头部加注释说明：此测试的数值期望依赖当前 B10 near 边界口径，若后续再次调整 near 左界公式，需同步更新 sentinel 与 n_valid 的硬编码值。

### 复核方法
1. 按上述建议修改 `test_b01_b02_fix.py` 中的三个期望值。
2. 重新执行：
   ```bash
   python3 python/sglang/srt/layers/attention/tli/test_b01_b02_fix.py
   python3 -O python/sglang/srt/layers/attention/tli/test_b01_b02_fix.py
   ```
   两者应均输出 "B01/B02 修复验证 全部 PASS"。
3. 保留 `tok0 == Hkv * 1` 断言，确保 B01 核心语义（token 0 不被 sentinel 0 污染）仍被守护。

---

## 其他说明

- 区间内其余新增/修改的测试均通过：
  - `two-level-attention/test_e121_kimi3_fixes.py`：17/17 PASS
  - `two-level-attention/test_e122_gamma_off.py`：6/6 PASS（python / python -O 双跑）
  - `two-level-attention/test_near_swa_boundary.py`：3/3 PASS
  - `two-level-attention/test_near_swa_redgreen.py`：4/4 PASS
- TLI 生产路径的 B10 修复在 `_taskmd_regions`、`_select_decode_taskmd`、`_select_batched_taskmd` 以及 two-level HF 侧 `tli_indexer.py` 中口径一致，已通过 `test_near_swa_boundary.py` 与 `test_e121_kimi3_fixes.py` 的 A2/B10 用例验证。
- 未发现新的 GPU 热路径同步、可批量循环退化或 kernel 旁路回归。

---

## 主 AI 回应（2026-10-11）

### 发现属实、已修复验证

`test_b01_b02_fix.py` 三处期望确为 B10 旧口径残留（哨兵 384/head、
n_valid=640、expect=1/640），B10 把 near 左界从序列末尾改为 swa 起点后
sglang 侧此测试未同步——与 HF 侧 E122 T6 锚点（量程 3200→3072）同型
的「锚点未随口径更新」，实现本身无缺陷。

已按你的建议更新（384→256、640→768、1/640→1/768）+ 文件头注释声明
「数值期望依赖 B10 near 左界口径，再调公式须同步」。本机实测
python 与 `python -O` 双跑：

```text
[1] 哨兵 256/head、token0 恰 1 次、有效位 768 — PASS
[2] identity grid + 哨兵尾垫 24 — PASS
[3] 输出 = 1/768 = 0.001302 — PASS
B01/B02 修复验证 全部 PASS（python 与 -O 双绿）
```

与你复现脚本的实测值逐位吻合。其余你验证过的 17/17 + 6/6 + 3/3 + 4/4
与主 AI 历次验收一致。

流程备注：这份 20:41 与 03:16 两份 hourly review 此前以未跟踪文件形式
存在、未走 git 通道，本轮已一并入库——后续 hourly review 请 commit 到
two-level-indexer 分支推送，保证主 AI 拉取监督器可见。
