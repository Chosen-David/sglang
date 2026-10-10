# TwoLevel 增量审查：079 修复仍未冻结运行时文件身份

审查时间：2026-10-11 05:33–06:22（Asia/Shanghai）

目标分支：`two-level-indexer`

审查 SHA：`81c49ca438a61744cc84e7b2dcff33cffccf4228`

相关实现提交：`ba8495da64fd6aa13ff1d6e465c0dddca2de76ce`

上次已审代码基线：`cc30fc11d4dc20b6c5780db12e645ad7654ade3e`

## 目标、范围与环境

远端轻量检查排除 advice-only 提交后，发现 `ba8495d` 修改了 079 的
canonical treatment identity 和 080 的验收三态。因此本轮只追踪：

1. `args → register_patch → TLIIndexer` 实际加载投影基/层跳过掩码；
2. LongBench 的生成、method name、sidecar 写门；
3. RULER 的 method name、generation、receipt 与正式校验；
4. 079/080 新验收脚本及 076–078 回归。

环境为 Python 3.12.14、无 `torch`/CUDA；工作树是 blob-less sparse
checkout。执行了纯 Python 身份/回执反例和 import-only torch stub 回归，
未运行 tensor、GPU、真实模型、LongBench/RULER e2e 或真实数据。本文不把
静态调用链或 stub 称为硬件实测。

## 新增发现

| ID | 状态 | 严重度 | 结论 |
|---|---|---:|---|
| `TL-E121-OUTPUT-SNAPSHOT-081` | **confirmed / CPU exact-code 复现 / 独立复核** | **P1** | 079 将文件内容加入 manifest，但没有把同一份已解析快照交给运行时、名称和回执。LongBench 可出现 `runtime A / name B / sidecar C`，RULER 可出现 `runtime A / basename B / receipt C`，且现有校验接受。默认 D′ 掩码还完全未进入 manifest：运行时读取 tracked `DEFAULT_MASK`，manifest 却记录 `null`。 |

## 081：文件内容被反复打开，声明身份不等于实际 treatment

### 违反的契约

上一轮 079 的明确修复要求是：在模型加载/输出路径确定前解析
`tli_proj_basis` 与 `tli_layer_skip_path`，由 LongBench sidecar、RULER
receipt 和 method hash 共用同一份 resolved manifest。`ba8495d` 只共用了
**函数**，没有共用一次冻结的**值/已解析对象**：

- `sparse_attn/info.py:86-139` 每次求 manifest 都重新打开路径；basis 甚至
  先 `open(...).read()` 求 SHA（`:109-117`），再由另一次
  `torch.load(path)` 取 shape（`:118-121`）。一次 manifest 内部也可混代。
- `sparse_attn/patches/patch.py:29-43` 对每层分别创建 `TLIIndexer`；
  `sparse_attn/indexer/tli_indexer.py:122-130` 每层重新读取 skip JSON，
  `:158-167` 每层重新 `torch.load` basis。文件在 patch 循环中变化时，单个
  模型本身可混装多代文件内容。

### LongBench 可达链

`benchmark/LongBench/pred.py:382-384` 先加载/patch 模型，实际文件已被各层
消费；`:459-470` 完成整套生成；直到 `:472` 才重新读取路径计算
method name，`:501` 又重新读取一次形成 sidecar，`:506` 起覆盖输出。

触发方式：运行时加载 A 后，同一路径被原地更新、原子替换或 symlink
retarget 为 B/C。生成结果来自 A，却按 B/C 命名并声明。如果目标位置已有 B
sidecar，`gate_output_treatment_identity` 还会正常放行，随后以 A 的结果
覆盖 B 身份的输出。这正是 079 试图修复的“同路径、内容变化”场景，只是
变化发生在一次长运行内部而非两次函数调用之间。

### RULER 可达链与未闭合校验

`benchmark/RULER/pred_ruler.py:164-175` 先 patch/load A，`:190` 重读并以 B
生成 method name / `out_path`，`:239-280` 运行生成，`:297-299` 再读 C 写入
receipt，`:323` 提交。现有门禁没有把三者闭合：

- `benchmark/RULER/yarn_receipt.py:755-768` 只校验 receipt 内
  `sha256(json)==声明 sha256`；
- `:845-850` 只要求 `prediction_basename` 等于物理预测文件名；
- `:974-979` 明确不把 treatment manifest 放入 effective config 指纹，并
  假设 basename 的 `_h<10hex>` 与 manifest 已经绑定，但代码未比较二者。

因此 basename 的 B hash 与 receipt 的 C hash 不同仍能通过 schema/formal
前置校验；“receipt 内部自洽”不能证明它描述了实际运行。

### 默认 D′ 掩码遗漏

`sparse_attn/arguments.py:60,65` 默认 `tli_enable_layer_skip=True` 且
`tli_layer_skip_path=None`。运行时在
`sparse_attn/indexer/tli_indexer.py:25-27,122-127` 把 None 替换为 tracked
`DEFAULT_MASK` 并读取其 `skip` 集合；但 `sparse_attn/info.py:150-153` 只在
路径显式非 None 时解析文件，所以默认 manifest 始终为 `null`。

当前默认 mask（Git object）SHA256 为
`9902254a10361cf209af70b862fb3fb842f6697de391888324bd156dd29f4b23`，
大小 1223 bytes；它自 `ea872af` 加入后截至审查 SHA 没有修改。这说明当前
提交历史里没有“默认 mask 已变”的命中证据，但 canonical 身份并未绑定
实际生效内容。若默认文件跨运行变化、部署包漏带该文件或 sparse checkout
未展开该路径，输出名仍显示 D 且 manifest 仍是 null；运行时则可能使用新
集合或在 `:128-130` 静默关闭 D′。

反向也证明实现按“argv 是否声明”而非“实际是否消费”解析：显式给出
`tli_layer_skip_path` 但 `tli_enable_layer_skip=False` 时，079 仍强制文件
存在并求 hash，而运行时不会读取它。

## 最小复现与原始输出

主审复现脚本：`/tmp/repro_tli081.py`，SHA256
`ac22c8da7660cf9e90f9b21f9d543f56f03f6df9ecb522656193f70c55478856`。
它直接调用当前 `info.py`、`build_yarn_receipt` 与
`_validate_common_schema`；由于环境无 torch，层掩码的运行时加载使用
`tli_indexer.py:122-127` 的同一 `json.load(...)["skip"]` 表达式。输入在
同一路径从 `[1,3]` 替换为 `[2,4]`：

```text
runtime_skip_ids = [1, 3]
current_skip_ids = [2, 4]
LongBench name hash = 23a8cf4785       # 声明 B，而运行时为 A
RULER basename hash(before) = 9817f32a45
RULER receipt hash(after)   = 23a8cf4785
ruler_schema_error = null
receipt_accepted_despite_name_manifest_mismatch = true
```

临时目录 realpath 会进入 manifest，因此每次复跑的具体 hash 可不同；关键
不变量是运行时集合、basename hash、receipt hash 三者可不同且校验返回
`None`。独立上下文另以 `git show 81c49ca:<path>` 直接执行精确 `info.py` 与
`yarn_receipt.py`（torch 只作 import/load/shape stub）：脚本
`/tmp/tl081_exact_code_repro.py` SHA256
`cfd54f5d2e7050db533fbe7a5cb3eda794638326e2ba987eaff9c3b5ccdb79ac`
（4854 bytes），原始日志 `/tmp/tl081_exact_code_repro.log` SHA256
`ae1475b4c5789932c8c53a8ca2b1a2c655f2dedcc7a12db21f9b49c7a13635fb`
（761 bytes）。结果为 LongBench `runtime_A_sha=559aead08264` 且
`gate_passed=True`；RULER basename B hash=`a935784086`、receipt C
hash=`3671ba7209`、A/B/C 全异，真实
`validate_producer_receipt(receipt, basename_b)` 仍返回 `None`。独立复核
还确认 register_patch 循环可形成跨层混代。

默认路径 CPU 调用结果：

```json
{
  "args_path": null,
  "enable_layer_skip": true,
  "manifest_path": null,
  "method_name": "tli_64_128_1024_c2_BDa0_b0_g1_hb8b76a0b5e"
}
```

其中可读名含 `D`，但 manifest 没有默认 mask 内容。

## 影响已有数据与论文结论

- **没有证据证明既有数值已经错误**。tracked 默认 mask 在当前可见历史中
  未变；已知 E109 runner 显式关闭 layer skip 且无 projection，未命中。
- E123 已知未启用 projection/async，但 mavg/cavg TLI 臂默认启用 D′；因此
  它们的数值不因“文件稳定”自动失效，却缺少默认 mask 的身份闭包。应核对
  实际生产 checkout 中该文件存在、运行开始/结束 hash 均为上述值，再决定
  是否补标签或重跑；不能仅凭 `tli_layer_skip_path=None` 声称未使用文件。
- 若任何已跑格在 patch→生成→提交窗口内更新过显式 basis/skip 路径，或各
  机器 checkout 漏带/内容不同，则 receipt/sidecar 不能证明 treatment，需
  按保存 argv、作业日志和实际文件 hash 定点重跑。不能从当前仓库反推当时
  文件内容，因此本轮不撤销具体论文数值。

## 修复与最小重测建议

1. 在模型构造/patch 前只解析一次 effective treatment：将默认
   `DEFAULT_MASK` 也展开；从同一批 bytes 同时计算 SHA、解析 JSON/tensor、
   验 schema，得到不可变 resolved manifest + 已解析 basis/skip 对象。
2. 将已解析对象注入每层 `TLIIndexer`，禁止各层重新按路径打开；这样同时
   消除 patch 循环跨层混代。若必须保留路径，至少以已打开 fd/bytes 快照
   加载，而不是 hash 一次、`torch.load(path)` 再读一次。
3. LongBench 在 `load_model_and_tokenizer/register_patch` 前冻结 manifest；
   输出名、sidecar 和运行时都消费同一对象。RULER 同理，receipt 禁止生成后
   重读路径。
4. receipt 校验显式重算 `sha256(treatment_manifest.json)[:10]`，并要求等于
   `prediction_basename` 的最终 `_h<10hex>`；新 producer 的 TLI 回执应要求
   manifest 必存，不能只做可选的内部自洽检查。
5. 新增确定性交错测试：A 在 patch 时加载，随后原子替换为 B，再生成/落盘；
   预期要么运行前冻结 A 且所有身份均为 A，要么 fail closed，禁止 A/B/C
   混装。再覆盖 patch 多层中途替换、默认 mask 缺失/损坏/变更、显式关闭
   D′、symlink retarget，以及普通 Python/`-O`。
6. 对 E123 TLI 臂只做定点 provenance 清点：记录每个生产 checkout 的默认
   mask 存在性/字节 SHA 与启动日志；确认稳定则无需因 081 无条件重跑。

## 旧发现复查、测试与未覆盖项

- **079**：`tli_enable_async_topk` 已加入 manifest，显式路径的跨调用内容
  变化也会改变 hash；这两部分保持 fixed/rechecked。081 是“未冻结实际
  消费快照 + 默认有效文件遗漏”的新残余，不撤销全部 079 成果。
- **080**：源码已改为 PASS/SKIP/FAIL 三态、SKIP 非零退出，并将裸 assert
  替换为显式 oracle + 破坏元测试；本轮静态复核未见回归。因无 torch/CUDA
  未重跑 075/079 tensor 套件，状态为实现已修、硬件覆盖沿用既有证据。
- `python -m compileall -q` 对本次 079/080 相关 Python 文件通过；
  `git diff --check cc30fc1..ba8495d` 通过。
- 076/077/078 套件使用 import-only torch stub，普通 Python 与 `-O` 均为
  `PASS=9 FAIL=0`；该结果只覆盖命名、写门和 pointer 路径。
- 未执行新 079 的 R1–R5（需要真实 torch，尤其 R4 tensor 行为），未执行
  baseline、GPU kernel、真实模型、e2e 精度或真实 E123 文件清点。

本报告只新增审查证据与修复/重测建议；未修改实现、测试脚本、实验数据、
任务索引或其他分支。
