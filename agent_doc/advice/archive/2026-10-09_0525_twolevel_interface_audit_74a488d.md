# TwoLevel TLS/稀疏接口边界审查（74a488d）

## 审查目标与范围

- **目标分支最新 HEAD**：`two-level-indexer` / `29a2d980b39a9585155e99676ef92cf206b4f7b1`。
- **实际源码 SHA**：`74a488d48f1108c90c1d1c6c00e70aec38a9bfd3`。最新 HEAD 相对该 SHA 仅新增 `agent_doc/advice/2026-10-09_0431_twolevel_p0_verdict_audit_74a488d.md`，没有实现、脚本或数据变化；本报告把 advice-only 提交排除出代码变化判断。
- **本轮轮换覆盖**：`tls_attn` 普通/KVO SFA 的 cache→L1→L2→decode 接口，`sparse_attn` Qwen3/Llama patch 的无缓存和 padding 分支，以及 calibration recorder 的首次写入与汇总。
- **环境边界**：Python 3.12.14，Linux 6.18.44 x86_64；没有 PyTorch、Transformers、TileLang 或可见 GPU。实际执行了 Python 集合/异常/shape witness 和受影响文件的 `py_compile`；没有运行 CUDA kernel、模型前向、真实 padding batch 或精度任务。

证据文件 SHA256：

- `tls_attn/patches/qwen3_sfa_patch.py`：`986c1c3f9edf99da2ffb49062d518b64ca07341be0433e12bbf8314f841a3d96`
- `tls_attn/patches/kvo_qwen3_sfa_patch.py`：`8524c9f2899d892b082416c323e00e6c8b11224653e9756a656a6bb8fb4bbe57`
- `tls_attn/ops/mha_indexer_level1.py`：`28e39d10fa0c6709b6c4493426659dcd9f67c8b4e48a0f879b3c071cb4405459`
- `tls_attn/ops/mha_indexer_level2.py`：`4efc4393405eaf5a126eb152291dd6b30da9dd4d7292851908c35da3b0b1b676`
- `sparse_attn/calibration_patches/recorder.py`：`14dd62130e182c8aeee31515181ddbcfa29759fdf0eed4e92dc9de9f14c4b781`
- `sparse_attn/patches/qwen3_attn_patch.py`：`f0599ae294c8c40f7ebdc97945a2ff9dc9618cd90dcba96334fcd4b14a58b634`
- `sparse_attn/patches/llama3_attn_patch.py`：`b8280f1e886e092914c0a677624f86c994444b04264667d0b53175f06734ff0b`
- `sparse_attn/patches/utils.py`：`c4d8312f19f463f4c9af1636d74caf7ef8e5f3fa9aa4e74c3bc78b8271af8ede`

## 结论摘要

| ID | 状态 | 严重度 | 位置 | 结论 |
|---|---|---:|---|---|
| `TL-TLS-L1-GAP-016` | **confirmed（源码闭环 + CPU 集合 witness；GPU 未跑）** | P1 | `tls_attn/patches/qwen3_sfa_patch.py:76-83`；`tls_attn/ops/mha_indexer_level1.py:79-104`；`tls_attn/ops/mha_indexer_level2.py:82-110` | 普通 SFA 把 token 长度向下取整为 L1 块数；当长度不是块大小的整数倍且超过窗口时，L1 far 与 L2 强制窗口之间永久漏一个完整块。默认 `B=64,W=3,S=257` 时 token 64–127 无法进入最终候选。KVO 同位置已使用 ceil，不存在该缺口。 |
| `TL-CAL-RECORDER-017` | **confirmed（Python 控制流 witness；Tensor 汇总未跑）** | P1（实验链） | `sparse_attn/calibration_patches/recorder.py:25-27,47-57`；`calibration_patches/qwen3_attn_patch.py:39-51` | recorder 第一次 `update()` 就对不存在的 `self.layer_dict[layer_idx]` 取值，必然 `KeyError`；即使修正，`get_record()` 最后也错误地 `torch.stack(stride_id)`，只堆最后一层，而不是 `results`。当前 calibration patch 不能完成首轮采集/多层汇总。 |
| `TL-PATCH-NOCACHE-018` | **confirmed（Python 控制流；真实模型未跑）** | P2（兼容性） | `sparse_attn/patches/qwen3_attn_patch.py:43-52`；`sparse_attn/patches/llama3_attn_patch.py:41-50` | 签名允许 `past_key_values=None`，更新也显式保护 None，但随后无条件调用 `past_key_values.get_seq_length(...)`；标准 `use_cache=False` 前向会在进入 attention 前抛 `AttributeError`。 |
| `TL-PAD-UNPAD-019` | **confirmed（shape 契约证明；PyTorch/GPU 未跑）** | P1（padding decode） | `sparse_attn/patches/qwen3_attn_patch.py:87-103`；`llama3_attn_patch.py:63-79`；`patches/utils.py:12-18` | decode 时 query 是 `[B,1,H,D]`，KV/mask 是 `[B,K,...]`/`[B,K]`；代码却用同一完整 KV mask 索引 query、key、value。`K>1` 时布尔 mask 与 query 前两维不匹配；常见 0/1 整型 mask 还会被当高级整数索引。padding batch 因此会异常或产生错误 packed query。 |

## 复现与机制证据

### 1. `TL-TLS-L1-GAP-016`：普通 SFA 在未对齐长度下漏一个完整块

记当前 token 长度为 `S`、块大小为 `B`、强制滑窗块数为 `W`：

1. 普通 patch 在 `qwen3_sfa_patch.py:77` 传给 L1 的长度是 `n=floor(S/B)`。
2. L1 top-k 在 `mha_indexer_level1.py:89-97` 只允许块 `i < n-W`。
3. L2 在 `mha_indexer_level2.py:88-93` 强制加入从 `floor((S-1)/B)-W+1` 开始的 `W` 个块。
4. 当 `S % B != 0` 时，`floor(S/B)=floor((S-1)/B)=m`。L1 最后可选块是 `m-W-1`，L2 第一个强制块是 `m-W+1`，所以块 `m-W` 没有任何入口。

默认 `B=64,W=3,S=257` 的最小清晰反例：L1 只允许 block 0，L2 强制 block 2/3/4；block 1（token 64–127）既不能在 L1 被选中，也不在 L2 滑窗内。`kvo_qwen3_sfa_patch.py:77` 使用 `(lengths+B-1)//B`，正好把 block 1 纳入 L1 可选域，是同仓实现反证。

CPU witness 的规范化输入 SHA256 为 `55959217920a4cb585771ddb939c137e0eb68f7918f7dcc5e472fab4c0dc6e02`，核心输出：

```text
S=192 L1=[]     near=[0,1,2] missing=[]
S=193 L1=[]     near=[1,2,3] missing=[0]
S=255 L1=[]     near=[1,2,3] missing=[0]
S=256 L1=[0]    near=[1,2,3] missing=[]
S=257 L1=[0]    near=[2,3,4] missing=[1]
S=319 L1=[0]    near=[2,3,4] missing=[1]
S=320 L1=[0,1]  near=[2,3,4] missing=[]
S=321 L1=[0,1]  near=[3,4,5] missing=[2]
```

因此在每个 64-token 周期中，除整块边界外的 63 个 decode 长度都存在一个移动的完整块盲区；这不是 top-k 预算主动淘汰，而是该块从候选全集中消失。

### 2. `TL-CAL-RECORDER-017`：首次写入和最终汇总均不可用

`layer_dict` 初始化为空字典。首次循环执行：

```python
if (layer_idx, stride) not in self.layer_dict[layer_idx]:
```

Python 会先求值右侧 `self.layer_dict[layer_idx]`，因此还没来得及创建 `(layer_idx,stride)` 就抛 `KeyError`；纯 Python 同语义 witness 实际输出 `KeyError(0)`。调用链由 `register_calibration_patch()` 把闭包 patch 到 Qwen3/Llama attention，首次 dense forward 在 `recorder.update()` 可达。

即使把首次写入改成检查 tuple key，`get_record()` 仍有两个汇总口径问题：更新覆盖 stride 16–64（含 64），读取只遍历 16–63；最后一行对最后一次循环遗留的 `stride_id` 做 `torch.stack`，而不是对逐层 `results` 做 stack。第一处会遗漏 stride 64，第二处会破坏 `[num_layers,h]` 汇总或直接触发张量维度错误。

### 3. `TL-PATCH-NOCACHE-018`：可选 cache 只保护了 update，没有保护分支判断

两个 patch 都先执行 `if past_key_values is not None: update(...)`，说明 None 是预期输入；但下一段立即对同一对象调用 `get_seq_length`。纯 Python witness 实际输出：

```text
AttributeError("'NoneType' object has no attribute 'get_seq_length'")
```

建议显式区分无缓存 dense 前向、首次 cache prefill 和已有 cache decode，不要用一个无保护的 cache 长度表达式同时判三种状态。

### 4. `TL-PAD-UNPAD-019`：KV mask 不能直接用于单-token query

decode 的转置后 shape 为 query `[B,1,H,D]`、KV `[B,K,Hkv,D]`，而 `attention_mask` 被 `prepare_cu_seqlens_from_mask()` 当 `[B,K]` 使用。随后 `(unpad_tensor(x, attention_mask) for x in (query,key,value))` 又把 `[B,K]` mask 原样用于 query。CPU shape witness 固定 `B=2,Q=1,K=5`，得到 query 前缀 `(2,1)` 与 mask `(2,5)` 不兼容。

修复不能只写 `.bool()`：还需要为 KV 和 query 分别构造 mask。KV 使用 `[B,K]` 有效位；单-token decode 的 query 应为每个仍活跃 batch 一项，并保持与 `cu_seqlens_k`、`query_position_ids`、最终 repad/reshape 的 batch 顺序一致。

## 对已有数据与结论的影响

1. `TL-TLS-L1-GAP-016` 只直接影响 `MHAGenerator(enable_sfa=True, enable_offloading=False)` 的 TLS 普通 SFA 路径。当前 `two-level-attention/main.py:38-45` 示例明确使用 `enable_sfa=False`，E109/P0' 走另一条 `sparse_attn` 实验链；没有证据据此宣布现有 E109/P0' 数据已污染。
2. 任何已经使用普通 TLS SFA 且 prompt/decode 长度未固定在 64 的整倍数上的精度/e2e 结果都需要重测。盲区会随长度移动，不能只重跑一个边界点。
3. calibration recorder 在首个更新即失败，所以本仓当前实现无法自行生成一份成功的该格式校准结果；若仓外存在结果，应核对生成代码 SHA，不能假设由此版本产生。当前没有发现绑定该 recorder SHA 的正式结果，故不声称历史校准数据错误。
4. 无缓存和 padding 两项是条件路径缺陷。当前已审结果多为 B=1、cache decode，不能据此外推到 `use_cache=False` 或真实 padding batch；反过来，这些结果也不能证明两条路径可用。

## 建议修复与最小重测

1. 普通 SFA 把 L1 块数改为与 KVO 一致的 ceil；同时保证 L1 cache 容量按 ceil 分配或断言 `max_cache_len % B == 0`。新增纯函数 oracle：对 `S=192/193/255/256/257/319/320/321`，断言 `L1 可选块 ∪ L2 强制块` 覆盖全部合法块且无越界。随后用 GPU 对拍 `S=256/257/319/320` 的 level1 indices、level2 indices 和最终 attention 输出。
2. recorder 用 tuple key 初始化，例如 `setdefault((layer_idx,stride), [])`；读取 stride 与写入闭区间一致，最终 `torch.stack(results)`。最小测试须覆盖首次 update、两层、stride 64 可胜出、两次 clear/update，以及 Qwen GQA group 聚合。
3. 为 Qwen3/Llama patch 增加 `past_key_values=None` 的 dense oracle，并与未 patch 模型在 `use_cache=False` 下逐位对拍；cache prefill/decode 保持原门禁。
4. padding decode 分离 query/KV mask，显式转 bool；以 `B=2`、左右 padding 长度不同、`K>1,Q=1` 做 eager 与 patch 输出对拍，同时检查 `cu_seqlens`、query batch 顺序、最终 reshape 和每条序列的候选均不串样。
5. 修复后先跑接口单测，再跑一个真实 Qwen3 小模型的 dense/SFA、cache/no-cache、B=1/B=2 padding 组合；这些程序和 shape 证据不能替代 GPU kernel 与模型精度验收。

## 旧发现复查与未覆盖项

- 既有 `TL-BOUNDARY-NEAR-SWA-001`、trace、kernel bench 和 P0' 发现状态未改变；本报告四项均为此前 advice 未记录的新路径/新证据，没有复写旧 ID。
- 受影响文件全部通过 `python -m py_compile`；这只证明语法可解析，不能证明 Tensor shape、kernel 或数值正确。
- 独立审查上下文复核了 `TL-TLS-L1-GAP-016` 的 floor/ceil 推导，并以 `S=257` 给出同一缺块反例；主审另外核对了实际调用链和旧 advice 去重。模型赞同未作为效果证据。
- 未验证 TileLang 越界、GPU kernel 实际 indices、PyTorch padding 异常文本、修复后速度/精度、真实 LongBench/RULER 或多 GPU。没有 GPU/依赖不等于这些路径通过。

## 下一检查点

优先复查 `TL-TLS-L1-GAP-016` 的 ceil 修复及跨 64-token 边界 GPU oracle；随后验证 padding batch 与 no-cache dense 对拍。若 calibration recorder 被修复，再审其生成结果的 input/model/code hash、层闭包、stride 64 和汇总 shape，避免把“能运行”直接当作校准结论可靠。
