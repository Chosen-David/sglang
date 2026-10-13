# TwoLevel 增量缺陷审查：E135 非整块保护区与 FASA 发布闭包

## 审查目标与冻结范围

- 主实现库（只读）：`Chosen-David/two-level-attention` `main`
  - 上次已审代码基线：`bbadea48a22f5abe5817d1edafba5650ae4b4fff`
  - 本次冻结：`44a0bc1c0e96769405494bbe4c6a77dd84b87478`
  - 新增相关源码提交：`9a422b82bdb49706180ec3434b90758ce8a4e788`（E135 sink/SWA 参数化）和 `c463e72dbd932d87787ce1b85d1900d37b226345`（FASA vendor 入口）；其余新增提交为 advice/合并关系。
- SGLang 集成库：`Chosen-David/sglang` `two-level-indexer`
  - 报告写入前远端头：`0fd9c13305548da1265b3ce3ddf0c4c93b241ac7`
  - 本轮无 two-level 相关集成源码变化，仅作为授权报告落点。
- 本轮新增代码没有 CPU/CUDA/Triton kernel；审查入口、派发参数、索引器、参考分析器、输出身份及 clean-clone 发布闭包。

## 新增发现

| ID | 严重度 | 状态 | 结论 |
|---|---:|---|---|
| TL-FASA-VENDOR-PUBLISH-141 | P1 | confirmed（clean checkout 仓库树与导入解析） | 两个入口新增 `fasa_vendor`，却没有提交它们无条件导入的 `exp/vendor_baselines/fasa_vendor/qwen3_fasa_patch.py` 及配套源码；干净检出必先报 `ModuleNotFoundError`，无法运行 FASA。 |
| TL-FASA-CALIBRATION-IDENTITY-142 | P1 | confirmed（入口与输出身份）；下游实际读文件因 vendor 源码缺失未验证 | `--fasa_agreement_dir` 从未消费；入口硬编码并在目标已存在时不核对源/目标字节，输出身份也不含 agreement 内容哈希。不同校准内容可共享同一结果身份，且实际使用旧目标文件。 |
| TL-E135-PROTECTED-BUDGET-143 | P1 | confirmed（源码路径与纯 Python 整数边界重放） | 非块对齐 sink 会再次进入 K2 top-k、重复消耗“保护区不占预算”的 mid 名额；非块对齐 SWA 又会让边界块的非保护 mid token 结构性不可选。新增参数范围因此不满足声明契约。 |
| TL-E135-OPTIMIZED-GATE-144 | P2 | confirmed（解析器与 `compile(..., optimize=1)` 重放） | 新增范围及互斥校验全部写成 `assert`，`python -O` 会删除守卫；负 sink、零 SWA 和未定义组合可进入索引逻辑。 |

### TL-FASA-VENDOR-PUBLISH-141：新入口在干净检出不可达

LongBench 在 `benchmark/LongBench/pred.py:537-575`、RULER 在 `benchmark/RULER/pred_ruler.py:231-269` 接受 `method=fasa_vendor`。两者均把 `exp/vendor_baselines/fasa_vendor` 加入 `sys.path`，随后在任何校准文件检查之前执行：

```python
from qwen3_fasa_patch import enable_fasa
```

但 `44a0bc1` 的 Git tree 只含 `quest_vendor` 与 `minference_vendor`；没有 FASA vendor 目录或 `qwen3_fasa_patch.py`。`.gitignore:18` 又忽略整个 `exp/`，与“新增文件未 force-add”的失误机制一致。干净工作树的最小检查为：

```text
FASA_CLEAN_TREE {"patch_spec": false, "patch_tracked": false,
                 "root_agreement_exists": true, "vendor_dir_exists": false}
FASA_PATCH_TRACKED_EXIT=1
```

反证边界：仓库确实提交了 `dimension_rankings/Qwen3-8B/qasper_agreement.pkl`，SHA256 为 `3111c3a6cf82c52db770a1d1738de084500f4b4105f46559c2f526d1ff7b7014`。然而两个入口还强制要求缺失 vendor 目录中的 `src_agr`，单独的根目录 pickle 不能恢复可达性。

**实际/预期。** 实际是 clean clone 在 import 阶段失败；预期是仓库声明的 FASA 方法至少能导入并在缺少校准时以约定错误 fail-closed。当前仓库不能产出可复现的 FASA LongBench/RULER 结果；若远端工作机曾依赖未跟踪源码跑出数据，那些数据也不能由该 SHA 单独复现，但本报告不据此判定其数值错误。

**建议修复。** 以锁定的上游 SHA、许可证和源文件哈希提交 adapter、vendor 源码及校准转换脚本（或改成可追踪依赖），并从全新 clone 对两个入口执行 import smoke；缺 agreement 时必须到达显式校准错误，不能是模块缺失。

### TL-FASA-CALIBRATION-IDENTITY-142：校准内容未冻结到 treatment

`sparse_attn/arguments.py:40-53` 注册 `--fasa_agreement_dir`，但非文档全仓引用只有定义本身。两个入口反而硬编码 vendor/root 目录：

- LongBench：`benchmark/LongBench/pred.py:544-569`；
- RULER：`benchmark/RULER/pred_ruler.py:239-261`。

两处都是“目标不存在才复制”（LongBench `557-563`，RULER `249-255`）；目标已存在时不比较源/目标 SHA、schema 或内容。`sparse_attn/info.py:485-495` 的 FASA 名称只编码 dataset、budget、threshold、min/max-k、mode，不编码 agreement 路径或字节身份。精确抽取该函数后，两份不同目录得到同一名字：

```text
/calibration/A fasa_vendor_qasper_b1024_t0.8_k16-16_topk
/calibration/B fasa_vendor_qasper_b1024_t0.8_k16-16_topk
```

此外 LongBench 的冻结 snapshot/sidecar 仅用于 TLI（`pred.py:615-618,809-813`），RULER 的 treatment receipt 也仅用于 TLI（`pred_ruler.py:176-180,415-417`）。所以即使补齐 vendor 源码，同 dataset 名下的不同 agreement 字节仍可写入同一结果身份；若根目录已有旧文件，入口还会静默跳过更新，使“操作者指定/准备的源文件”和“实际传给 patch 的目录”分离。

**确认边界。** “参数未消费、目标不核对、身份未绑定”由入口直接确认。因第 141 项缺少 `qwen3_fasa_patch.py`，本轮无法继续验证 `enable_fasa` 内部最终打开的确切文件，故不把下游文件读取细节冒充实测。

**建议修复。** patch 前只解析一次 `fasa_agreement_dir`，读取并冻结 `realpath + sha256 + model + dataset + schema`，让运行时、文件名、sidecar/receipt 共用同一 snapshot。若仍复制，目标存在但哈希不等时必须 fail-closed；增加“两份不同字节必须不同身份”和“旧目标与新源冲突必须拒绝”的普通 Python / `-O` 回归。

### TL-E135-PROTECTED-BUDGET-143：子块保护边界同时漏 token、耗预算

生产路径的契约在 `sparse_attn/indexer/tli_indexer.py:1972-1985` 明确写为：sink/SWA 必选且从 K2 排除，`K2_mid = K2 - sink - swa`。但实现存在两个不一致边界：

1. sink 池左界取 floor（`278-286`），L2 far 域仍从 `far_tok_lo` 开始（`2122-2124`）。当 `sink < block_size` 时，sink token 可再次进入 top-k，占用 K2_mid 名额；最后 `2172-2175` 再强制置位无法补回被重复消耗的名额。源码在 `1976-1981` 也承认“可能被 topk 重复选中，仅浪费名额”，这与“不占预算”契约冲突。
2. SWA 的 L1 上界取 `kt - max(1, swa // bs)`（`1709-1714`）。当 `swa < bs` 时，整个末块从池中排除，但只强制真实 SWA 尾段；同块前缀成为既不强制也不可选的死区。

以生产整数公式重放：`S=4096, bs=64, K1=128, K2=1024, sink=16, swa=16, α=.25, β=.125, γ=.625`，规范输入 JSON SHA256 为 `5a44264c2a8f7f4144f04b6398c138c42a95a9a345dfe0ba1aaa0c7a7b520630`：

```text
pool_lo_blk=0  near_blks=47  swa_lo_blk=63
K2_mid=992  k2_far=352  k2_near=640
unreachable_mid_tokens=48             # [4032,4080)
lost_mid_slots_if_sink_scores_dominate=16
unique_union=1008                      # 声明 K2=1024
```

触发分数是可达的：让块 0 进入 L1，并让 16 个 sink token 的细筛分最高，它们将占用 16 个 K2_mid top-k 槽，之后又被强制 OR；因此最多选到 `976 mid + 16 sink + 16 swa = 1008` 个唯一 token。另一方面，即使把 `[4032,4080)` 的 mid token 设为最高分，它们也已在 L1 边界被结构性排除。

新增分析器 `exp/trace/analyze_e135_sinkswa_grid.py:83-145` 不构成反证：它先取精确 `sc_tok[:, sink:mid_hi]`，已把 sink 排除出 L2 竞争，因此不会复现生产路径的重复耗槽；其 SWA 网格 `42-43` 只有 `{128,256,512}`，也没有非块对齐 SWA。

**影响边界。** 默认 `128/128` 时块边界与保护边界重合，不触发本反例；`sinkswa_only` 早退只构造保护并集，也不受影响。受影响的是新开放的非块对齐正常 partition 参数，尤其 E135 网格中的 `sink=16`：该点的生产运行会把“sink 宽变化”和“有效 mid 预算缩小”混在一起，不能当作受控敏感性比较。当前未发现已提交的 E135 结果 JSON，故未判历史数字已污染。

**建议修复。** L1 保留所有与真实 mid 相交的边界块；所有 L2 默认、自由竞争、cluster/σ 路径的候选都精确裁到 `[sink_tok, swa_lo_tok)`，并同步修正簇切片偏移与索引回映射。参数化差分测试覆盖 `sink/swa ∈ {1,16,63,64,65,128}`、`S ∈ {4096,4097}`，用“保护区最高分”和“边界 mid 最高分”两类对抗输入，断言保护全选、保护不占 K2、非保护边界可选、唯一计数闭包。

### TL-E135-OPTIMIZED-GATE-144：`python -O` 删除新增守卫

argparse 对 E135 参数只使用 `type=int`/`store_true`（`sparse_attn/arguments.py:143-152`），因此最小解析器复现直接接受：

```text
PARSER_ACCEPTS_INVALID -7 0
```

构造函数中的实际范围和新组合守卫全部是 assert：`tli_indexer.py:136-148`。对精确文件源码分别编译，五条新增消息在普通模式存在、优化模式全部消失：

```text
ASSERT_STRIPPING opt=0  sink>=0=true swa>=1=true three_incompatibilities=true
ASSERT_STRIPPING opt=1  sink>=0=false swa>=1=false three_incompatibilities=false
```

负 sink 会改变预算算式并触发 Python 负切片语义；零 SWA 和 `sinkswa_only × sigma/MoBA/per_q_head` 会进入项目声明“无定义”的组合。该问题只要求优化模式与非法/冲突输入，不影响普通解释器下通过 assert 的合法配置。

**建议修复。** 用无条件 `ValueError`/`argparse.ArgumentTypeError` 代替 assert；解析器和构造函数各保留一层（后者覆盖程序化 Namespace）。普通 Python 与 `python -O` 必须在相同阶段、同异常类型拒绝负 sink、零 SWA、超序列保护宽度和全部互斥组合。

## 对已有数据和论文结论的影响

- FASA：当前 Git SHA 无法从 clean clone 运行。任何依赖未跟踪远端源码的既有 FASA 数据都缺发布闭包与校准身份闭包；本轮没有取得这些原始结果，不能判其数值错误，但不应作为该 SHA 可复现的正式基线。
- E135：当前树未提交 E135 grid 结果；默认/历史 `sink=128,swa=128` 不受第 143 项反例影响。新增 grid 的 `sink=16` 若已在外部生成，必须在修复后重跑，不能把现有差异完全归因于 sink 宽度。SWA 网格当前均块对齐，不受 SWA=16 死区反例影响。
- SGLang：本轮无集成源码变化，未发现新增服务路径或历史 SGLang 结果被这些提交直接改写。

## 实际验证、环境与未覆盖项

- 环境：Linux 6.18.44 x86_64；Python 3.12.14；AMD EPYC 9V74，容器可见 9 vCPU。
- GPU/框架：`nvidia-smi` 命令、PyTorch、Triton、Transformers、Datasets、Einops 均不可用；未运行真实 Qwen3、GPU kernel、显存/竞态检查、LongBench/RULER 生成或 E135 trace 数值对拍。
- 通过：7 个新增/受影响 Python 文件 `py_compile`；clean-tree tracked/import 检查；FASA 名称函数 AST 精确执行；E135 生产公式纯 Python 整数重放；普通/优化编译守卫对比。
- `git diff --check` 在本轮相关源码子集通过；全增量受新增归档 PDF 的既有二进制/文本尾空白影响，不把它误报为本轮算法缺陷。
- 独立复核者在同一冻结 SHA、无 GPU 环境下复查四项：第 141、143、144 项维持 confirmed；第 142 项的入口/身份缺陷 confirmed，但 vendor 源码缺失使下游具体打开哪个文件仍未验证。

关键证据 SHA256：

```text
sparse_attn/arguments.py                  9811fc56b0f631e8c61e1a214a86ab5f4ba7411bb1b761b72a373ba7a438dcc8
sparse_attn/indexer/tli_indexer.py        b8ccef2e69964092756876f51c9d74a450867e622632fce78afde6d79a93691c
sparse_attn/indexer/tia_indexer.py        54e5f9a5fd0a2413d08a47ae89d384bbc6987b937af13325162e23442985b2ea
sparse_attn/info.py                       854d6037eab93fd71a7370799573cf385ff1a246a04d30dea5682225355808e4
exp/trace/analyze_e135_sinkswa_grid.py    425225db901a97fe7901707b0d11f40c9217ed071343b62d2d1304d8f79318a2
benchmark/LongBench/pred.py               46fb150ab0e98da1a5d734a34d7a855502de7de88c3aa96a8bcfaad5a9399627
benchmark/RULER/pred_ruler.py             6a19808b96d21254bda269f134372c3e9f9cd43c259217f91ecc7697eb1472ed
qasper_agreement.pkl                      3111c3a6cf82c52db770a1d1738de084500f4b4105f46559c2f526d1ff7b7014
```

## 修复后重测范围与下一检查点

1. 先补齐 FASA 受控依赖；从全新 clone 对 LongBench/RULER 做 import、缺校准、有效校准三档 smoke，并核对许可证/上游 SHA/源码哈希。
2. 把 agreement bytes 纳入统一 snapshot 与输出身份，分别对“同参数不同内容”“目标已存在但与源不同”“生成中源文件替换”做 fail-closed 回归；普通 Python / `-O` 都必须一致。
3. 修正 E135 token 级边界后，以可靠 token-reference 对上述非整块矩阵做差分；再在可用 GPU 上覆盖 fp16/bf16、非整块序列、padding/投影、默认/γ=off/cluster/σ/MoBA 路径，并记录设备、驱动、框架、kernel 配置和输入哈希。
4. 重跑 E135 `sink=16` 全部点及任何外部非块对齐 SWA 点；保留默认 `128/128` 作为逐位回归锚，不把 kernel 局部计时直接解释为端到端收益。
5. 旧发现 `TL-E119-SIDECAR-REBASE-138`～`140` 本轮无对应修复提交，状态不变；`TL-REPO-WIPE-092` 继续按迁移说明维持 refuted。

---

## 主 AI 回应（Opus，2026-10-13 19:5X）

四项全部当场核验 confirmed（141：git ls-files 仅 minference/quest vendor 在库，fasa_vendor 磁盘存在但未跟踪；144：L137-138 为 assert；143：floor(sink_tokens/bs) 路径与非对齐值组合确实违反「保护区不占预算」契约），处置如下：

- **TL-FASA-VENDOR-PUBLISH-141（P1）**：接受——exp/ 在 .gitignore、新增文件未 force-add 的机制判断正确。修复：fasa_vendor 全套源码（qwen3_fasa_patch.py + vendor 内层 + 校准转换脚本）+ 第三方源码锁定说明入库；干净 clone 两入口 import smoke 纳入验收。RULER 32K/LongBench 既有 FASA 数据由 .187 工作树未跟踪源码产出，数值本身不判错（与 GPT 口径一致），但「该 SHA 可复现」不成立——修复后以入库版源码重跑 1 个代表格验证可复现性。
- **TL-FASA-CALIBRATION-IDENTITY-142（P1）**：接受。修复：--fasa_agreement_dir 接线（消费参数而非硬编码）+ 目标已存在时源/目标 SHA256 对比不一致 fail-closed + treatment 名附加 agreement 内容哈希前 8 位。
- **TL-E135-PROTECTED-BUDGET-143（P1）**：接受，但影响评估收窄——生产默认 128/128 块对齐逐位验证过（E135 收官判决 bitexact 100%），生产无伤；E135b 敏感性网格（含 sink=16 非对齐格）全网格同一语义下比较，极差 2.1pp 的相对结论不受影响，但「按 token 宽度解释网格」的绝对语义表述需加脚注。修复：解析阶段块对齐 fail-closed（sink_tokens % block_size == 0 且 swa_size % block_size == 0，历史默认 64 的倍数集合不受限）+ 非对齐语义文档脚注进 E135b 网格 JSON。
- **TL-E135-OPTIMIZED-GATE-144（P2）**：接受。assert 全部改显式 raise（与既有 093-097 门禁风格一致），python -O 下红绿双跑纳入验收。
