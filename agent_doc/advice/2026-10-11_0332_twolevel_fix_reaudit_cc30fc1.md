# TwoLevel 增量缺陷复审：075–078 修复批次

审查时间：2026-10-11 03:32–04:20（Asia/Shanghai）

目标分支：`two-level-indexer`

审查代码 SHA：`cc30fc11d4dc20b6c5780db12e645ad7654ade3e`

上次已审代码基线：`c7d6d02ec5ef99a22d27cec3f1d5b598f6929677`

相关源码提交：`d8fff2f9635d6ed8e7a9cd20ced5c19993566b43`（075）、
`e5dfc19e67121ae2384222c6c292bdade2cf8004`（076/077/078）

## 审查目标与范围

本轮只审查相对上次基线新增的相关实现、接口和验收脚本，排除
`agent_doc/advice/` 下的报告提交。实际追踪：

- 075：`prepare_index → compute_score → compute_mask → eager decode` 的
  `valid_length`、padding、near/far、SWA 和 MoBA 路径；
- 076：TLI 参数解析/父类状态 → canonical manifest/hash → LongBench
  sidecar 写入门 → RULER generation/receipt；
- 077/078：generation pointer resolver → 单格 probe → 目录 audit；
- 新增 E121/E119 验收脚本的普通 Python/`-O` 口径。

未运行 GPU、真实模型、真实数据集或 e2e 精度；当前审查环境为 Python
3.12.14，无 `torch`、`pytest`、CUDA 设备。没有把 CPU/stub 结果称为
GPU 或真实 workload 验收。

## 新增发现

| ID | 状态 | 严重度 | 结论 |
|---|---|---:|---|
| `TL-E121-OUTPUT-ID-079` | **confirmed / CPU 调用复现 / 静态可达** | **P1** | 076 新 manifest 仍不是有效 treatment 的单射：继承自 TIA 的 `tia_enable_async_topk` 未入 manifest/hash；`tli_proj_basis` 与 `tli_layer_skip_path` 只记路径、不记文件内容。同步/异步或同路径不同内容会得到完全相同 method name/sidecar，LongBench 写入门会放行覆盖，RULER 会复用同一逻辑指针。 |
| `TL-E121-TEST-SKIP-080` | **confirmed / CPU 源码级结果回放** | **P2** | 075 新验收脚本把未执行的 T3 基线逐位对照和 T5 GPU 冒烟记录为 `PASS`；汇总没有 `SKIP` 状态且在两项均未执行时仍返回 0。另有一个关键手算 oracle 使用裸 `assert`，在 `python -O` 下被删除，违反仓库已确立的 065/067 验收纪律。 |

## 079：canonical manifest 仍遗漏有效行为与文件内容身份

### 违反的契约与调用链

`sparse_attn/info.py:22-52` 声明 manifest 覆盖“全部输出相关参数”，但字段
集合没有 `tia_enable_async_topk`。TLI 继承 TIA：

- `sparse_attn/indexer/tia_indexer.py:13-19` 把
  `args.tia_enable_async_topk` 保存到 `self.enable_async`；
- `sparse_attn/indexer/tli_indexer.py:1000-1007` 在开关打开时缓存当前
  L1 mask，并在下一步用 `prev_mask` 替换当前 mask；关闭时使用当前 mask。

因此该开关不是性能提示，而会改变进入细筛的候选集合。同步与异步运行仍经
`info.py:62-84,214-216` 得到相同 JSON、hash 和 method name。

文件型配置也只有词法路径身份：

- `info.py:49` 的 `tli_proj_basis` 仅保存路径，注释称内容由
  argv/receipt 承载；但 LongBench 的 sidecar 就是这份 manifest
  （`benchmark/LongBench/pred.py:488-504`），没有 basis 内容摘要；
- `info.py:51` 的 `tli_layer_skip_path` 同样只保存路径，文件中的
  `skip` 集合会在 `TLIIndexer.__init__` 读取并直接改变各层 far 路径；
- RULER v2 producer receipt 的 `generation_params`
  （`benchmark/RULER/pred_ruler.py:302-306`）只含
  `max_gen/max_num/seed/method/pred_postfix/t`，也没有这两个文件的内容
  digest。现有注释所称的 receipt 纵深不能识别“同路径、内容已变”。

### 最小可运行复现

在无 torch 环境以仅供导入的空模块 stub 加载 `info.py`，构造除
`tia_enable_async_topk` 外完全相同的两个 namespace；再把同一路径的
投影基字节从 `basis-A` 改为 `basis-B`。复现脚本 SHA256：
`de05af1dddaa61a9084d724f773e5dbe396b225a38703a4a199bdd97ce8f2c89`。

```python
a = cfg(async_topk=False)
b = cfg(async_topk=True)
print(get_method_name_with_info(a) == get_method_name_with_info(b))
print(get_treatment_manifest_json(a) == get_treatment_manifest_json(b))

p.write_bytes(b"basis-A")
m1 = get_treatment_manifest_json(cfg(proj=str(p)))
h1 = sha256(p.read_bytes()).hexdigest()
p.write_bytes(b"basis-B")
m2 = get_treatment_manifest_json(cfg(proj=str(p)))
h2 = sha256(p.read_bytes()).hexdigest()
print(h1 != h2, m1 == m2)
```

原始输出：

```text
079-async-name-equal True
079-async-manifest-equal True
079-basis-bytes-differ True
079-basis-manifest-equal True
```

`info.py` 输入 SHA256：
`bc7a59b50a3805c26365fda044e30d5da9c571b35aaf2700aea1e80bdcd0231c`。

### 实际/预期与影响

- 实际：不同同步语义或不同投影/层掩码内容共享逻辑路径；LongBench 看到
  相同 sidecar 会允许 `open(..., "w")` 覆盖，RULER 会在相同
  `{out}.tli_gen` 下切换 generation。
- 预期：任何会改变选择结果的有效配置都必须进入 canonical 身份；文件型
  配置必须绑定实际读取内容，而非只绑定可复用的路径字符串。
- 已有数据：不能据此断言已污染。需用保存 argv 定点筛
  `method=tli + tia_enable_async_topk` 的同目录重复运行，以及
  projection/layer-skip 文件在运行间内容发生变化但路径不变的格。
  已知 E123 四臂未启用 projection basis，且按独立 arm 目录隔离；本轮没有
  新证据推翻先前“E123 不命中 075/076 触发条件”的结论。

### 修复与重测建议

1. 把 `tia_enable_async_topk` 的**生效布尔值**加入 canonical manifest。
2. 在模型加载/输出路径确定前解析 `tli_proj_basis` 和
   `tli_layer_skip_path`，记录规范路径、SHA256、必要时 shape/schema；
   文件缺失或读取失败应 fail closed，不用 `default=str` 隐去身份问题。
3. LongBench sidecar、RULER generation receipt 与 method hash 共用同一份
   resolved manifest；不要让不同入口各自维护字段子集。
4. 增加自动覆盖测试：遍历 argparse 注册的所有 TLI/TIA 输出相关参数，
   对每个可独立改变行为的字段做 mutation test；另测同路径内容变更必须改
   hash/拒绝覆盖。对 async 需至少两 decode 步，证明第 2 步候选与同步路径
   可不同。

## 080：075 验收脚本把 SKIP 算作 PASS，且 `-O` 删除 oracle

### 代码与复现

`test_e121_proj_pad_fix.py:270-274` 在未提供 `E121_BASE_ROOT` 时调用：

```python
report("T3 非投影路径与基线逐位一致", True, "SKIP（...）")
```

`test_e121_proj_pad_fix.py:335-338` 在无 CUDA 时同样调用
`report(..., True, "SKIP（无 cuda）")`。但 `report` 只有
`PASS/FAIL` 两态，`main:353-364` 只统计这两态，`FAIL=0` 即退出 0。
从已提交源文件 AST 提取并执行原 `report` 后回放两个环境缺口：

```text
PASS  T3 baseline  -- SKIP
PASS  T5 GPU  -- SKIP
080-replayed-summary 2 PASS / 0 FAIL rc 0
```

源文件 SHA256：
`5c2496b0a046dc4b4581c123e85398a5cb519d3a64204a143c01b9df3214bf78`。

此外 `test_e121_proj_pad_fix.py:203-204` 用裸 `assert` 固定
`(mid, near, far_hi)=(5889,736,5248)`。在 `python -O` 下该语句消失；
而后续 K2 与预期边界又由同一个 `geom_expect` 结果派生，oracle 若回归可
出现自洽假绿。这与仓库在 065/067 中明确采用“显式检查、PASS/SKIP/FAIL
三分、SKIP 非零退出”的纪律冲突。

### 影响边界

- 这不会直接改变预测数据，也不能据此否定 075 实现修复；T1/T2 的 CPU
  几何和 padding/SWA 检查仍提供实质覆盖。
- 它会夸大验收覆盖：默认命令没有 baseline checkout 时把 T3 算作 PASS；
  无 GPU 机器还会把 T5 算作 PASS，并返回 0。提交说明中的默认
  “23/23”包含至少一个未执行的 T3；只有明确提供有效
  `E121_BASE_ROOT` 且 CUDA 实际可用的运行，才同时覆盖这两项。
  在“无 baseline + CPU-only”条件下正确摘要应为
  `21 PASS / 2 SKIP / 0 FAIL` 且非零退出，而不是
  `23 PASS / 0 FAIL`、退出 0；提供 baseline 后 T3 从 1 条 SKIP 展开为
  6 条实测，因此完整矩阵总数才会从 23 变为 28。
- 本环境无 torch，因此没有声称实际执行 075 tensor 套件；以上结论来自
  已提交控制流与原 `report` 函数的可运行结果回放。

### 修复与重测建议

1. 令 `report` 接收 `PASS/SKIP/FAIL` 三态，T3/T5 缺资源返回
   `SKIP`；汇总显式打印三类，任一 SKIP/FAIL 都不打全绿且非零退出。
2. 把第 203 行裸 `assert` 改为始终执行的显式检查；补一个 oracle
   元测试，故意破坏 `geom_expect` 后普通 Python 与 `-O` 都必须红。
3. CI 至少分为 CPU 必跑（含真实 baseline worktree）与 GPU 必跑两个 job；
   若 GPU job 不在当前环境执行，应保持 pending/skip，不由 CPU job代签。

## 旧发现复查

- **075**：实现已把 `valid_length` 从 index 透传到 score，并在 mask 前
  将 padding 置 `-inf`；near/far、SWA、σ、γ=off、ccluster 与 MoBA
  相关位置均改用真实 `S`。静态复查未发现新的同根实现缺口，但因无 torch/
  GPU 未重跑 tensor/kernel；075 保持“实现已修、GPU/e2e 依赖既有实测”
  状态，不用 080 的测试口径缺陷倒推实现失败。
- **076**：原报告列出的 far/near method/select、subspace、per-q-head 等
  六配置碰撞已修；新增 E119 套件在本环境用 import-only torch stub 实际
  运行，普通 Python 与 `-O` 均为 `PASS=9 FAIL=0`。079 是新 manifest
  对父类开关和文件内容身份的残余，不把 076 全部修复成果撤销。
- **077/078**：NUL/control pointer 与 dangling symlink fixtures 在上述
  两次 9/9 运行中均按预期 fail closed/计 coverage error；本轮未发现回归。

## 实际执行与未验证范围

| 检查 | 结果 | 边界 |
|---|---|---|
| 相关 Python `compileall` | 通过 | 仅语法/字节码编译 |
| E119 076/077/078 suite（torch import-only stub） | 普通 Python 9/9；`-O` 9/9 | 被测路径不调用 torch；不能外推 tensor/GPU |
| 079 canonical identity 最小复现 | async 两配置同名同 manifest；同路径 basis 字节改变仍同 manifest | 纯身份函数 CPU 调用；异步输出差异由生产分支静态可达 |
| 080 SKIP 汇总回放 | 两项 SKIP → `2 PASS / 0 FAIL`，等价 rc=0 | 执行已提交的原 `report`；未导入完整 torch 套件 |
| GPU kernel/e2e/真实精度 | 未执行 | 当前环境无 torch/CUDA/真实数据 |

独立上下文复核重新追踪了 manifest→父类异步状态→TLI 第二步 mask、
projection 文件加载→LongBench sidecar，以及 T3/T5→汇总/退出码两条链，
确认 079/080 均可达并同意 P1/P2 分级。它同时给出反证边界：TIA（非 TLI）
可读名本身含 `_async`，async 第一个 `compute_mask` 与同步路径相同；
079 从第二步起触发；T3/T5 在资源齐备时会真实执行，不是假 PASS。独立模型
一致只作交叉检查，以上代码路径、CPU 调用输出与控制流回放才是结论依据。

本报告只记录新增审查证据、影响边界和修复/重测建议；未修改实现、实验
脚本、实验数据、任务索引或其他分支。
