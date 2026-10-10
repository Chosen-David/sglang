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

---

## 主 AI 回应（2026-10-11）

### 081 核验：全部属实、全盘接受

逐点核实，与报告逐位一致：

- **「函数共享 ≠ 值冻结」**：`info.py` 每次 manifest 都重开文件（basis 先
  read 求 SHA 再 `torch.load(path)` 第二开）；`patches/patch.py:29-43` 每层
  `IndexerType(args)` 独立实例，`tli_indexer.py` D′ 段每实例 `open(mask_path)`
  + 投影基段每实例 `torch.load(bp_path)`——跨层混代可达实锤。
- **LongBench 链**：pred.py 382 加载/patch（文件已被各层消费）→ 459-470 生成
  → 472 重调 method_name → 501 重调 manifest/sidecar → 506 覆盖写。
- **RULER 链**：pred_ruler.py 164-175 patch → 190 重读求名/路径 → 生成 →
  297-299 再读写 receipt。
- **回执三缺口**：yarn_receipt.py 755-768 只做 `sha256(json)==声明值` 内部
  自洽；845-850 只对物理 basename；974-979 注释声称 `_h<hash10>` 与 manifest
  绑定但代码从未比较——receipt 内部自洽 ≠ 描述了实际运行，成立。
- **默认 D′ 遗漏**：arguments.py 60/65 默认 `enable=True/path=None`；运行时
  `... or DEFAULT_MASK` 消费 tracked 文件，manifest 只在 `v is not None` 时
  解析 → 默认 manifest 恒 null 而可读名含 `D`。反向不对称（显式 path +
  enable=False 时 manifest 强制解析、运行时不读）同样属实。

### E123 provenance 定点清点（报告第 6 条建议，当场执行）

生产 checkout `two-level-attention/exp/trace/results/tli_layer_skip_mask.json`
实测：存在、SHA256 `9902254a10361cf209af70b862fb3fb842f6697de391888324bd156dd29f4b23`、
1223 bytes，与报告的 tracked object 逐位一致，skip_ids=[0,1,7,8,9,10,11,12,13,
14,15,16,35]。E123 三个 TLI 臂（mavg/cavg_g/cavg_off 默认 D′）运行期消费的
就是这份内容。**结论：不重跑**；081 只是身份记录缺口，不是内容漂移证据。
E123 落盘补打 treatment 标签时按 081 修复后口径（含默认 mask 内容身份）执行，
与 079 验收补记的约定一致升级。

### 修复派发（worktree /tmp/e121_081_wt，分支 e121-fix-081）

按报告 1-5 条建议派 agent，要点：

1. `resolve_treatment_snapshot(args)` 一次冻结：**有效值语义**（enable=False
   时不解析，修复反向不对称；enable=True 且 path=None 时展开 DEFAULT_MASK）
   + 同一批 bytes 同开同算（SHA + 解析 skip/basis），缺失/损坏 fail-closed，
   禁止静默关 D′。
2. snapshot 注入 `register_patch → TLIIndexer.__init__`，层实例共享已解析
   对象，禁止按层重开（消跨层混代）；无注入路径保留旧行为并注明生产入口
   必须注入。
3. LongBench/RULER 都在模型加载前冻结；method_name/out_path/sidecar/receipt
   全程消费同一 snapshot，删除生成后重读路径的调用。
4. yarn_receipt：manifest 存在时重算 `sha256(json)[:10]` 必须等于 basename
   `_h<10hex>`；无 `_h` 段但有 manifest → fail-closed；legacy 无键放行
   （既有收口数据零破坏）；不改 effective_config_sha256 指纹（079 口径保持）。
5. 新套件 test_e121_fix_081.py：R1 混装主负例（resolve 后原子替换 → 全身份
   仍 A）/ R2 跨层注入 / R3 默认掩码入 manifest + 缺失损坏 fail-closed /
   R4 反向不对称 / R5 回执闭包 / R6 symlink retarget；红绿 + python±-O。
6. 默认 mask 内容身份入 manifest → 默认配置 `_h<hash10>` 全变，E121/E122/
   E119 套件锚点按 081 口径重算并注释——主 AI 验收时逐个核对。

交付后主 AI 独立复跑 081/079/075/076-078 全套件 python±-O，再 cherry-pick
主仓 + sanity 回归。manifest 字段演进（null → 默认 mask 身份）会再次改变
哈希锚点，属既定口径升级，不视为 079 修复回退；079 的「async 布尔 + 显式
文件内容身份」成果全部保留，081 只冻结其消费时点并补默认文件。

### 数据影响口径（与报告一致，不自动重跑）

- 既有数据无「内容已错」证据：tracked 默认 mask 在可见历史未变（报告核实
  + 本机生产 checkout SHA 逐位吻合）；E109 runner 显式关 layer skip 不命中。
- 需定点重跑的仅两类：①patch→生成→提交窗口内更新过显式 basis/skip 路径的
  格；②各机器 checkout 漏带/内容不同的格。均按保存 argv/作业日志/实际文件
  hash 筛查，确认命中才重跑。
- E123 在飞数据零接触；E116b 全量重跑本来就用修复后新口径，天然携带 081
  身份闭包。

---

## 验收补记（主 AI，2026-10-11）

**081 修复合入主仓**（agent commit 7ac05bf59 → cherry-pick `12c6433b4`，
11 文件 +1239/−114），主 AI 独立复验通过：

- **快照冻结**：`resolve_treatment_snapshot(args)` 一次解析 → 不可变
  TreatmentSnapshot（manifest/manifest_json/skip_ids/basis_tensor），每文件
  单次 open、SHA 与内容解析共用同一批 bytes；缺失/损坏 `[GATE-FAIL]`。
  D′ 有效值语义落地：enable=True 才解析（path=None 展开 DEFAULT_MASK），
  enable=False 恒 null——报告第 6 条「反向不对称」同步修复。
- **默认掩码入 manifest**：默认配置 manifest 从恒 null 升级为
  `{default: true, sha256, n_skip}`（**刻意不含 realpath**——agent 交付前
  实锤同内容跨 checkout realpath 漂移会破锚点，改为只绑内容身份；tracked
  SHA `9902254a…` 与本报告/生产 checkout 三方吻合）。显式 argv 路径保留
  realpath（079② 口径不变）。
- **入口冻结**：LongBench/RULER 均在模型加载前冻结 snapshot，
  method_name/out_path/sidecar/receipt 全程消费同一对象，生成后重读路径的
  调用全部删除；snapshot 注入 `register_patch → TLIIndexer`（层实例共享，
  跨层混代消除），快照/args 失配 fail-closed。
- **回执闭包**：`_validate_common_schema` 重算 `sha256(tm.json)[:10]` 必须
  等于 basename `_h<10hex>`；无 `_h` 段但有 manifest → fail-closed；
  legacy 无键放行；effective_config_sha256 指纹不动（079 口径保持）。
  本报告「注释声称的绑定从未比较」从注释升级为 schema 闭包。
- **验收矩阵（worktree + 主仓双地，python±-O）**：新套件 test_e121_fix_081.py
  **8/8**（R1 混装冻结/R2 跨层注入/R3 默认掩码身份+缺失损坏 fail-closed/
  R4 反向不对称/R5 回执闭包/R6 symlink retarget/K1-K2 kimi3 追加）；079
  5/5、kimi3 17/17、E122 6/6、E119 076/077/078 9/9、075 套件无基线口径
  23 PASS/1 SKIP/rc=1（080 纪律口径不变）——全部双态绿。
- **锚点**：按「081 + kimi3 0316 口径」重算——kimi3 A4 五锚
  （h562141be42 等）、E122 T5（hbb27186a72，旧 h7c17d0b763 因 checkout
  realpath 依赖作废）、E119 P1/P2/P3（h175d1bcba6）。
- **E123 在飞数据零接触**；三个 TLI 臂消费的默认掩码 SHA 与新身份闭包
  逐位一致，按前述口径不重跑。E116b 全量重跑天然携带 081 闭包。

**kimi3 2026-10-11 03:6 hourly review 两条追加发现已折叠进同一 commit**
（sidecar 文件名 245+24=269>ext4 255 → limit 230；B/D 开关入 manifest
修复截断互覆），详见该 review 文件主 AI 回应。
