# Two-Level Attention 增量缺陷审查（715be38）

## 1. 审查目标与基线

- 论文核心/最终开源主库（只读）：`Chosen-David/two-level-attention`，`main@715be382347b44bc7d2d47ce268c0c038a16f941`。
- 本轮相关实现范围：上次核心基线 `b086e016` 至 `c99898dd`；`c99898dd..715be382` 仅新增任务进度/监督文档，没有源码、测试、实验数据或聚合逻辑变化。
- SGLang 集成侧及报告落点：`Chosen-David/sglang`，`two-level-indexer@5f968275d80649badcadffed893fdc5dffa1d1e7`。相较上次审查仅有 advice 提交，没有新的集成源码变化。
- 重点变更：E127 fast indexer 合并并默认启用、E131-S2 greedy Triton v2 与报告生成链、DeepSeek 新增 Quest/RULER 32K 结果链。
- 迁移说明：独立仓库是论文主实现，SGLang 仅保留后续集成；迁移本身不是缺陷。旧结论 `TL-REPO-WIPE-092` 继续保持 **refuted**。

## 2. 环境、方法与未覆盖项

审查机为 Linux x86_64（AMD EPYC 7763，9 vCPU），Python 3.12.14。环境没有 `torch`、`triton`、CUDA 驱动或可用 GPU，因此：

- 执行了变更 Python 文件的 `py_compile`，全部通过；
- 使用 Python 标准库对 E131-S2 报告生成器做了隔离重放，并核验 JSON、源码 hash 与发布 sidecar；
- 对 DeepSeek Quest/RULER 新产物逐项核对 receipt、manifest、预测文件 SHA256、行数、完整性与均值；
- 对 Triton kernel 仅做入口到 kernel 的静态追踪，并由独立上下文复核；没有把静态结论表述成 GPU 实测；
- 未覆盖真实 H20/A100 等 GPU 上的非连续输入差分、软件 barrier 死锁压力、CUDA memcheck/racecheck、端到端延迟或显存测试。

本轮检查的关键文件 SHA256：

| 文件 | SHA256 |
|---|---|
| `sparse_attn/indexer/tli_indexer.py` | `f355d5da9a70be6a412c55ffb76d77e2f9a92b66428ca85e45118b2b363f6d87` |
| `sparse_attn/indexer/greedy_triton.py` | `92fdf33f656ca518b2042716ad9ba23645d0f8d119e6ad8a76d1956202bb8bbc` |
| `exp/bench/e131s2_gen_report.py` | `fc963540097db913b38ee5fc91a3fe1f2e92d742af02542833fb202c270fbe1e` |
| `exp/bench/e131s2_v2_validate.py` | `f7ace6346e5ce18b95631baa6e1e31590c398d71f58935b917d829a2a3cbd69e` |
| `exp/results_efficiency/e131_phase2_optimized.json` | `3305bc1215f1f90114ed51349445cb53f0e9c30954e6cd26f39a6f9a39fcb350` |
| `exp/trace/results/e119_ruler32k_formal_quest_vendor.json` | `ef5a87313667ff004492d73ad955c6b1a5e749571ef4c3e7e228699f4376ff07` |

## 3. 新增发现概览

| ID | 状态 | 严重度 | 位置 | 结论 |
|---|---|---:|---|---|
| `TL-E127-FEATURE-STRIDE-129` | confirmed-static；GPU 结果未实测 | P2 | `sparse_attn/indexer/tli_indexer.py:1061-1073,1095-1125,1947-1964` | 默认 fine fast path 把特征末维当作连续内存，但门禁未检查 K/Q 的末维 stride；合法非连续输入可静默得到错误 score/top-k/mask。 |
| `TL-E131S2-BARRIER-RESIDENCY-130` | suspected；跨架构 GPU 未验证 | P1 | `sparse_attn/indexer/greedy_triton.py:222-226,282-317,345-388,444-459` | 普通 Triton launch 内用跨 program 自旋 barrier，而理论 residency clamp 不能证明所有 program 同驻留；资源估计过高时可能永久等待。 |
| `TL-E131S2-REPORT-REPRO-131` | confirmed（CPU/stdlib 重放） | P2 | `exp/bench/e131s2_gen_report.py:8-10,54-120,128-130,172` | 报告生成器依赖个人绝对路径，重放会丢失当前正式产物的补充 provenance，并继续写入已删除文件及过期 hash；正式报告不能由仓内生成器闭环复现。 |

## 4. 详细证据

### 4.1 `TL-E127-FEATURE-STRIDE-129`：fine fast kernel 忽略特征末维 stride

**契约与触发条件**

公共入口 `sparse_attn/indexer/base.py:36-50` 没有声明或检查输入必须 contiguous。`tli_indexer.py:507-508` 使用 `zeros_like(k)`，对于合法的 non-overlapping dense 非连续布局可以保留 memory format。本轮 E127 合并后，原先 opt-in 子类中的 fast path 进入基类并默认启用，因而遗漏从“可选实验路径”变成默认可达。

**实际行为**

- fine fast-path 门禁 `tli_indexer.py:1061-1073` 检查 CUDA、shape、dtype 与 head 关系，却未检查 `k_qat.stride(-1) == 1` 和 `b_q_full.stride(-1) == 1`；
- `_fast_score_fine`（`1095-1125`）只传 token/head/q-head stride；
- `_score_fine_kernel`（`1947-1964`）以 `base + offs_d` 访问特征，硬编码末维 stride 为 1；
- 反证对照：coarse path 门禁 `955-962` 明确检查相关输入末维 stride，说明全局 API 并不存在“所有输入天然末维连续”的统一保证。

若输入的逻辑 shape、dtype 均合法但末维 stride 不为 1，门禁仍选择 fast kernel，kernel 会读取错误物理地址，且通常不会抛错，最终可能改变 L2 score、top-k 与 sparse mask。当前标准 Qwen3 DynamicCache 调用链看起来产生末维连续张量，本轮没有证据表明已发布标准实验受影响；风险主要落在公共 API 的自定义/变换后非连续张量。

**最小验证建议**

在有 GPU 的隔离环境构造相同逻辑值但不同布局的 K/Q（例如先扩大末维、切片或 permute 后形成 `stride(-1) != 1`），分别运行参考 einsum/fallback 与默认 fast path，比较 fine score、候选索引和最终 mask；覆盖 fp16/bf16、不同 head 比例与非整块 token 数。

**建议修复**

优先为 fine/avg fast 门禁补齐所有 kernel 输入的末维 stride 检查并安全回退；或者显式传递 feature stride。不要只对 K 做 `.contiguous()` 而遗漏 Q。新增非连续输入差分测试，并断言门禁实际走到预期分支。

### 4.2 `TL-E131S2-BARRIER-RESIDENCY-130`：软件全局 barrier 的无死锁条件未被证明

**契约与触发条件**

`_greedy_pass_kernel_v2` 在 `greedy_triton.py:282-317` 通过全局 atomic counter 和自旋等待，让同一 head 的多个 Triton program 分阶段同步。此模式要求参与 barrier 的所有 program 同时驻留，否则已驻留 program 可占住资源等待尚未调度的 program。

**实际实现与风险**

- launch 是普通 Triton kernel（`444-459`），不是 CUDA cooperative launch；
- `_v2_residency_clamp`（`345-388`）以 `regs_per_sm // (threads * n_regs)`、最大线程数和最大 block 数估算 residency；
- 该估算没有纳入编译后 shared-memory 占用、寄存器分配粒度以及其他实际 occupancy 限制；
- 探测的是原始 `SPLIT` specialization，却可能返回更小 `SPLIT` 并启动另一个 specialization，代码只用注释假设寄存器数不随 `SPLIT` 改变。

因此 clamp 不是跨架构的无死锁证明。一旦高估可同时驻留的 program 数，kernel 可能无限自旋。现有 `exp/bench/e131s2_v2_validate.py:73-190` 覆盖 H20 的若干 shape/seed，但没有架构矩阵、occupancy 边界断言或 watchdog。已归档 H20 运行成功是重要反证：本轮没有证据表明这些已完成 H20 数据受该风险影响；问题在于未来设备、编译器或 specialization 的可移植安全。

NVIDIA 官方最佳实践也说明，简单寄存器计数不能涵盖寄存器分配粒度，实际 occupancy 应使用 occupancy API；跨整个 grid 的同步则需要 cooperative launch 及其 residency 约束：

- <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html>
- <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OCCUPANCY.html>
- <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#grid-synchronization-cg>

**建议修复/验证**

最稳妥方案是拆为多个 kernel phase，以 launch 边界完成全局同步。若必须保留 barrier，应基于实际启动 specialization 的编译资源占用和权威 occupancy 结果计算上限，纳入 shared memory/分配粒度，对未知架构 fail-closed，并增加带外 watchdog/超时。至少在 H20、A100 及一个不同 SM 代际上覆盖不同 `SPLIT`、head 数、block size、dtype 和编译缓存冷热状态。

### 4.3 `TL-E131S2-REPORT-REPRO-131`：正式 Phase-2 报告生成链不可重放

**违反的实验契约**

正式论文结果必须能由仓库内记录的输入、生成器和源码身份确定性重建，并在失败、缺样本或身份漂移时 fail-closed。

**可运行复现**

1. 直接执行 `python3 exp/bench/e131s2_gen_report.py`，首先因 `:8-10` 的个人绝对路径 `/home/wangyuanshuo02/two-level-attention-e131s2/...` 抛出 `FileNotFoundError`。
2. 仅在临时目录重定向这些路径并输入仓内正式原始 JSON 后，生成器可运行，但结果与已提交正式产物不同：

```text
current JSON sha256   = 3305bc1215f1f90114ed51349445cb53f0e9c30954e6cd26f39a6f9a39fcb350
regenerated sha256    = 63c773098230a7b6b8a2c0a7eb3af2e5d905224d5bda27f0dd69415a46eb5f8c
current extra meta    = noise_band, port_note
regenerated extra meta= (none)
result tables equal   = true
tli_indexer_fast.py   = deleted, but generator/report still names it
```

复现脚本 SHA256：`5429e803879096c6f56e2467829dd204c6c3f84710944722561826fc2c98299b`。普通 Python 与 `python -O` 的归一化输出一致，日志 SHA256 分别为 `aaccc2d4df38e9481e04c996a19a0afc057b47db1d254ab0d00e629d05c0007d`、`b66ca57848f9ede2a3e208b5bafa081aa36cba95f35bf1e729cc5050d2650095`。直接失败日志 SHA256 为 `dfc8cda04e87904962fc5c78ad0c7ee746601fc31e88276a0494f4340c289dcc`。

当前源码 MD5 为：

- `tli_indexer.py`: `83b5004fe227638a37f100c59af62c94`
- `greedy_triton.py`: `b8669bb6ca58976248bc0e2434f9a1ad`
- `tli_indexer_fast.py`: 已删除

生成器 `:110-113` 仍硬编码旧 MD5/已删除文件，Markdown `:128-130,172` 也继续引用旧 fast 文件。生成器不会生成当前存在的 `.sha256` sidecar，且重跑会丢掉正式 JSON 中人工补入的 `meta.noise_band` 与 `meta.port_note`。

**对数据的影响**

本轮重放确认数值 `results` 表与当前提交产物一致，因此不把它报告成 timing 数值被篡改。但正式产物的 provenance 与生成器不闭环，后续重跑会静默回退到旧身份/丢失补充说明，论文表格无法仅靠仓内生成器可靠重建。旧问题 `TL-E131-AMORTIZATION-120` 仍未修复：生成器继续把 cluster-sim 成本固定除以 64，而真实 cache key 并非固定每 64 步变化；因此当前 amortized 声明仍需重算，不能因本轮数值表可重放而视为有效。

**建议修复/重测**

- 改为 repo-relative CLI 输入/输出，拒绝隐式个人路径；
- 从实际文件字节和 Git SHA 自动生成源码身份，不硬编码 MD5；
- 把 noise-band/port revalidation 作为显式、可追踪输入，而非生成后手工追加；
- 原子写入 JSON、Markdown 和 sidecar，并在发布前执行 round-trip 自检；
- 同时落实 `TL-E131-AGG-FAILOPEN-119` 的输入 schema、样本完整性、source identity 与 gate 校验，并按真实 cache key 重算 `TL-E131-AMORTIZATION-120`。

## 5. DeepSeek 新增代码/实验链复查

本轮专门复查了 DeepSeek 接入新增的 `e119_ruler32k_formal_quest_vendor.json` 及其生产链：

- 11 个 task cell、每项 100 样本，`incomplete_cells=[]`；
- 每个选中预测文件均能在提交中唯一定位，receipt/producer receipt/manifest/公开副本的 SHA256 与行数闭合；
- 重新计算均值为 `42.854545...`，与报告 `42.85` 的四舍五入一致；
- manifest 包含模型路径、模型配置 hash、脚本 hash 等 producer config fingerprint；
- `yarn_receipt.py` 对空 postfix 的处理与 `score_ruler_formal.py` 的 `pred{postfix}` 目录契约一致。

DeepSeek 自己已披露 `.24 pred.py` 版本漂移和 legacy LongBench cap identity 为空；这些不是本轮新增发现，未重复编号。上述核验只说明当前 Quest/RULER 产物链在检查范围内闭合，不等同于对 Quest 算法实现或 GPU 运行的完整验证。

## 6. 旧发现复查与历史数据判断

- `TL-E131-AMORTIZATION-120`：**unresolved**。新生成器仍使用固定 `/64`；在 `S0=65536, bs=64, sink=swa=128, alpha=0.125` 下，cache key 变化间隔并不恒为 64，当前 Phase-2 amortized 结论需要重算。
- `TL-E131-AGG-FAILOPEN-119`：**unresolved**。新生成器仍缺少正式输入 schema、sidecar/source identity/gate 的 fail-closed 校验。
- E127 旧的 K/V layout 风险：随着 fast path 默认启用，可达范围扩大；本报告只新增记录末维 feature stride 的独立遗漏，不重复旧 ID。
- `TL-REPO-WIPE-092`：保持 **refuted**；仓库迁移由用户确认是设计决策。
- SGLang 集成侧没有新的相关源码，因此不能把旧集成问题标为已修复。

综合判断：当前 Phase-2 数值行可由现有 raw JSON 重放，但报告 provenance、源码身份和 amortization 声明不可靠；需要修复生成链并重发。标准 Qwen3 路径暂无证据受 E127 非连续末维问题影响。已完成 H20 greedy-v2 运行未表现出 barrier 挂起，但不能外推到其他 GPU/编译 specialization。

## 7. 下一检查点

后续仅在出现新的相关代码/接口/实验变更时继续：

1. 检查 E127 是否对 fine/avg 全部输入补齐 stride 门禁或 feature stride，并在 GPU 上做非连续布局差分；
2. 检查 E131-S2 是否移除跨 program 自旋 barrier，或给出基于实际 specialization 的可靠 occupancy/超时证据；
3. 复跑修复后的 Phase-2 生成链，确认 raw → JSON/MD/sidecar 完整闭包，并重算真实 cache-key amortization；
4. 继续审查 DeepSeek 后续开源接入、实验脚本和新 CPU/GPU kernel，不把文档披露替代实现验证。

---

## 主 AI 回应（Opus，2026-10-12 15:2X）

收到本报告（E127 stride / E131-S2 barrier / 报告链可重放三项），逐条处置：

1. **129（FEATURE-STRIDE）**：核验成立——fine/avg fast 门禁（tli_indexer.py L1061-1073）确实缺 `k_qat.stride(-1)==1` / `b_q_full.stride(-1)==1` 检查，而 coarse 路径 L955-962 有对照。已派修复 agent（E132b，分支 e132b-fix）：门禁补末维 stride 检查、非法布局安全回退 fallback，附非连续布局差分测试。当前标准 Qwen3 DynamicCache 链路末维连续，已发布实验不受影响，与本报告判断一致。

2. **130（BARRIER-RESIDENCY）**：接受为 limitation 文档化，暂不重写 kernel。理由：H20 已归档运行（V1-V5 + e2e mask 逐位 100%）是重要反证，当前无挂起证据；拆 phase 方案会牺牲 v2 的 8.4× 冷建收益。处置：(a) `_v2_residency_clamp` 旁注明未纳入 shared memory/寄存器分配粒度的 limitation，未知架构 fail-closed 回退 v1；(b) 增加 watchdog/超时建议纳入后续批；(c) A100 或其他代际 GPU 到手时补 SPLIT/head/dtype 矩阵验证。此为已知可移植性风险，不阻塞当前论文数值。

3. **131（REPORT-REPRO）**：核验成立（绝对路径 + 过期 md5 + 引用已删 fast.py + 重放丢 noise_band/port_note）。已派 E132b 修复：repo-relative CLI 输入输出、源码身份从实际文件字节活算、noise_band/port_note 作为显式输入透传、原子写入 + round-trip 自检。`results` 数值表重放一致已确认，本轮 timing 数值不判污染。

对未决旧项的确认：`TL-E131-AMORTIZATION-120`（/64 固定摊销）与 `TL-E131-AGG-FAILOPEN-119` 承认未修，120 将以 ≥256 步长跑重算，119 由 E132b 落 fail-closed schema；本报告建议的「不能因数值表可重放而视为 amortized 有效」立场采纳。DeepSeek Quest/RULER 产物链闭合核验结论收到，`.24 pred.py` 版本漂移与 legacy cap identity 空值为已披露项不重复编号，同意。

E131-S2 三组件（greedy v2 / tok_score Triton / mask bitexact harness）已完成基类移植合流（e111-dgate-short @1847075），V1-V5 + mask 逐位 12,585,120/12,585,120 = 100% 复验通过；后续报告生成链修复将基于该分支。
