# TwoLevel 增量缺陷审查：MInference prefill 边界与投影 L2 契约

## 审查目标与冻结范围

- 主实现库（只读）：`Chosen-David/two-level-attention` `main`
  - 上次已审代码基线：`e7fcc50c359c0297dde0136933d17ebc13ef580c`
  - 本次冻结：`bc5b219059b8cb31254fc9e72aa83423b9fd2b38`
  - 增量：`e7fcc50c..bc5b219`，130 files，40512 insertions / 225 deletions；本轮实质审查集中于新增 MInference Qwen3 适配、vendored sparse kernel、E133b/c 紧凑子空间和 E134 `l2_subspace`。
- SGLang 集成库：`Chosen-David/sglang` `two-level-indexer`
  - 冻结：`743423b6a6d2a9dae901f13c612343655deca966`
  - 相对上次基线无相关源码变化；仅作为本报告授权落点和后续集成影响边界。
- 迁移口径：独立库是论文核心实现；未把 SGLang 中旧子树迁出视为删除缺陷，`TL-REPO-WIPE-092` 维持 refuted。

## 新增发现

| ID | 严重度 | 状态 | 结论 |
|---|---:|---|---|
| TL-MINF-PADMASK-135 | P1 | confirmed（静态控制流 + 有界参考模型；GPU 未实测） | Qwen3 MInference 稀疏 prefill 接收 `attention_mask`，但稀疏层和浅层 flash 两条路径均不消费非平凡 mask；batch padding/额外约束会改变有效 token 输出。 |
| TL-MINF-CHUNKCACHE-136 | P1 | confirmed（静态控制流；GPU 未实测） | 任意 `q_len>1` 都被判为 sparse prefill；已有 prefix cache 时 `K_len>Q_len`，下游 sparse kernel 仍按 `Q_len==K_len` 的方阵协议生成 seqlens、索引和 causal 坐标。 |
| TL-E134-PROJ-L2FULL-137 | P2 | confirmed（CLI 可达 + 静态控制流） | `--tli_proj_basis` 与 `--tli_l2_subspace full` 可同时传入，但投影分支仍以 rank-r 投影特征执行 L2，和 CLI 声明的“L2 全 128 维”不符。 |

### TL-MINF-PADMASK-135：稀疏 prefill 丢失 padding mask

**违反的契约。** `Qwen3Attention.forward` 的 `attention_mask` 必须约束有效 key；改变被 mask 的 K/V 不应改变有效 query 输出。适配器在 `exp/vendor_baselines/minference_vendor/qwen3_minference_patch.py:38-45` 接收 mask，在 `:91-99` 仅将其放进 `prefill_kwargs`。但 `minference/modules/minference_forward.py:631-665` 只读取 `attn_forward_config/layer_idx/num_hidden_layers`，稀疏层直接调用 `minference_prefill_kernel`，浅层直接调用只有 `causal=` 的 `flash_attn_func`，均未读取 `prefill_kwargs["attention_mask"]`。

kernel 侧也无法隐式恢复 padding：`minference/ops/pit_sparse_flash_attention_v2.py:195-224` 从 `query.shape[2]` 构造单元素 `seqlens=[context_size]`。B>1 时它既没有每样本有效长度，`seqlens` 元素数也不等于 batch 大小。纯 causal mask 由 kernel 自带 causal 规则覆盖，不属于本缺陷；触发条件是 left/right padding 或其他非平凡约束。

**最小有界证据。** 参考 softmax 取 2 个 key，将第 2 个 key mask。只改变被 mask 的 V：

```text
input_sha256 = 5129994ee9e03734701d4bfdbcef8f2a64e1ce061dcfe4a4e04c00a17a5796f8
dense_masked(base) = [2.0, -1.0]
dense_masked(alt)  = [2.0, -1.0]
mask_ignored(base) = [51.0, 49.5]
mask_ignored(alt)  = [-49.0, 24.5]
```

这不是生产 kernel 数值实测，而是对“消费 mask / 忽略 mask”两种控制流的确定性判别模型；源码 AST 复核 `minference_prefill_forward` 的 `prefill_kwargs` 访问仅有 6 处，均不是 `attention_mask`。

**修复建议。** 最小安全修复是在 adapter fail-closed：只允许 B=1 且 mask 为 `None` 或全有效；完整支持需把 padding 转为每样本 seqlens/unpad 表示，并让 pattern selection、稀疏层和浅层 flash 都消费同一有效域。回归至少覆盖 B=2 左/右 padding、不同有效长度；分别对比逐样本无 padding 参考，并验证改变 pad token 的 K/V 不改变有效 token 输出。

### TL-MINF-CHUNKCACHE-136：chunked prefill 违反 kernel 方阵前提

`qwen3_minference_patch.py:53-66` 仅按 `q_len==1` 区分 decode；所有 `q_len>1` 均走稀疏 prefill。`:76-80` 在已有 cache 时先追加本 chunk K/V，此时 `K_len=past_len+q_len>q_len`，`:97-99` 仍调用 MInference prefill。

下游 `minference_forward.py:592-629` 用 `q_len` 生成 vertical/slash 索引；`pit_sparse_flash_attention_v2.py:195-224` 和 `:244-257` 又把 `query.shape[2]` 同时作为 `context_size` 与 K/V seqlens，并对 Q/K/V 加同一个由 Q 长度导出的 pad。其 causal 坐标从 0 开始，也没有 prefix offset。因此 cached prefix 会被截断/错误定位，或由后端因 Q/K 非方阵协议报错。

**修复建议。** 在 kernel 显式支持独立 `q_len/kv_len`、query 绝对 offset 和矩形 causal 之前，检测“更新前 cache 已有 token 且 `q_len>1`”并回退保存的原生 dense forward；不要先修改 cache 再回退。回归用 prefix P + chunk C（C>1），对拍原生 Qwen3 的 one-shot 与 chunked dense 输出/cache，并覆盖不同 P/C、B、dtype。

### TL-E134-PROJ-L2FULL-137：`full` 被静默降为投影维 L2

`sparse_attn/arguments.py:93-102` 独立接受：

```text
--method tli --tli_subspace tail --tli_proj_basis basis.pt --tli_l2_subspace full
```

本机 argparse 最小验证成功解析为 `tail / basis.pt / full`。构造函数只验证枚举和投影基约束，没有拒绝此组合。非投影路径在 `tli_indexer.py:630-637` 对 `l2_subspace=full` 正确量化原始 128 维 K；但投影路径在 `:572-580` 无条件量化 `kp`，评分侧 `:1058-1065` 生成投影 `b_q`，`:1199-1205` 又无条件令 `b_q_full=b_q`。`:1214-1221` 的全维 Q 分支仅在无投影时可达。

实际运行因而是“投影 L1 + 投影 L2”，manifest 却可记录 `l2_subspace=full`。修复可二选一：实现“投影 L1 + 原始 128 维 L2”，或在解析/构造阶段对 `proj_basis != None && l2_subspace == full` fail-closed。必须补三组合差分测试：无投影 sub+full、投影 sub+same、投影 sub+full；后者若支持，断言 `k_qat.shape[-1]==128` 且分数对拍全维参考。

## 对既有数据的影响

- **MInference 已产 LongBench/RULER/pattern 搜索：当前证据不支持判作污染。** LongBench `pred.py:345-357` 每样本 B=1，一次 `past=None` prefill，之后 question/decode 均逐 token；Samsum 的 generate 路径传全 1 mask。RULER `pred_ruler.py:303-333` 同样逐行 B=1、一次无历史 prefix 的 prefill 后逐 token decode。pattern 搜索也是 B=1、无历史 cache。它们没有触发 padding batch 或 chunked cached prefill。本结论不外推到 batch serving、prefix cache、speculative/multi-token continuation。
- **E134 子空间矩阵：当前证据不支持判作污染。** 扫描 `exp/trace/results/e134_l2full` 的 5 个 manifest，唯一组合为 `tli_l2_subspace=full, tli_proj_basis=null`；因此其 sub+full 臂走的是正确的非投影路径。
- SGLang `743423b6` 未引入本轮新增代码，尚无当前集成产物受影响；未来接入这些入口时应携带上述 fail-closed 门禁。

## 实际验证、环境与未覆盖项

- 环境：Linux x86_64；Python 3.12.14；NumPy 2.3.5；GCC 13.3.0。
- GPU/框架：`nvidia-smi`、PyTorch、Triton 均不可用；未运行 CUDA/Triton kernel、compute-sanitizer、racecheck，也未声称 GPU 对拍或性能结论。
- 通过：新增参数组合 argparse 可达性；`minference_prefill_forward` AST 数据流检查；padding-mask NumPy 判别模型；E134 5 个 manifest 闭包扫描；相关 Python 目录 `compileall`（仅 `dyn_controller.py` 既有无效转义 `SyntaxWarning`）。
- 现有 `test_e132a_fix_111_116.py`：9 PASS / 6 FAIL / 2 SKIP；6 个失败均因子进程环境缺 `jieba` 导致 `eval.py` 提前退出或未形成预期锁等待，2 个 SKIP 因无 PyTorch。该结果不用于支持或反驳本报告三项发现。
- 未覆盖：真实 Qwen3 权重、非连续 tensor、fp16/bf16 数值、不同 GPU 架构、真实 batch serving/prefix cache，以及修复后的端到端质量与性能。

源码证据 SHA256：

```text
qwen3_minference_patch.py       2f35ce276daaf6385bb7d1e8f4dbeb1e1cdfba392a41e1a884f4a92632e55571
minference_forward.py           8a3759b6b604ee1aaa2720a3e95bc6001e20492dd558139e17d00f791e23404b
pit_sparse_flash_attention_v2.py c847875b0391df052d8171c294eac25949f2a0e5bc9e3cc2d65e59bbe9d4e8a8
tli_indexer.py                  245b556ac80e808a70adae2ee2073b38edf05ebf9c8c48cfa3a2390fdf58fa7b
arguments.py                    29405d498419cad6c4c601635038a3867fdf1495d7fc8bc2abdcf95e45e0427f
best_pattern_qwen3_8b.json      17ad18b70e164a1ba9ff29636ab121bd537b2334a7bef137aa8398aa51e5c7fc
```

## 旧发现复查与下一重测范围

- E131-S2 跨 program spin barrier/residency 风险已有独立报告和门禁，本轮未重复编号或重复结论。
- 独立复核者在相同 SHA、无 GPU 环境下逐项反证，三项均维持 confirmed；并确认既有 advice 无同证据重复项。
- 修复后优先顺序：先做 MInference fail-closed + B=2 padding/reference 对拍，再做 cached chunk P/C 对拍；随后补 `proj_basis × l2_subspace` 组合门禁。只有 GPU 差分、sanitizer 和完整请求复测通过后，才能扩展 MInference 的支持范围或更新性能/质量结论。

---

## 主 AI 回应（Opus，2026-10-13 19:5X）

三项全部当场核验 confirmed，处置如下：

- **TL-MINF-PADMASK-135 / TL-MINF-CHUNKCACHE-136（P1）**：接受。MInference vendor 为 B 轨对手 baseline，与 GPT 判定一致——我方 LB/RULER harness 全程 B=1 + 单次无历史 prefill，既有数据不判污染。修复采纳最小安全方案：adapter 层 fail-closed（非平凡 mask / 已有 cache 且 q_len>1 时显式拒绝或回退原生 dense），完整 seqlens/unpad 支持不做（vendor baseline 无此需求）。已排修复链（agent）。
- **TL-E134-PROJ-L2FULL-137（P2）**：接受。任何实验均未使用 proj_basis × l2_subspace=full 组合（e134_l2full manifest 全为 proj_basis=null），无数据污染。修复选 fail-closed 方案：解析/构造阶段拒绝 `proj_basis != None && l2_subspace == full`（实现「投影 L1 + 全维 L2」需另铺 kernel 路径，收益不明不预做）。已排修复链。
- 修复优先顺序同意 GPT 建议（MInference fail-closed 先于 chunk P/C 对拍）；红绿门 + python -O 双跑纳入验收。
