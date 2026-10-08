# TwoLevel trace 完整性、显存与 far-empty 调试路径审查

## 结论摘要

- **审查对象**：`two-level-indexer` 源码提交 `f26197dfc1509583abb3fc1976fcf933b398bafc`。
- **增量范围**：从上次源码基线 `96bb6622b` 到 `f26197dfc`；其中 `4577908be` 为 advice-only 提交，已从源码变化判断中排除。本次同时轮换检查 `collect_trace_lb_v.py` 的完整调用链及新 far-empty 门禁触达的调试路径。
- **新增结果**：2 项已确认缺陷、1 项已确认的调试路径缺陷；均未自动修改实现或测试脚本。
- **实测边界**：当前隔离环境无 `torch`、无 `nvidia-smi`，因此不能声称 GPU/模型实跑。已完成 Python 语法编译、差异检查、纯 Python 合约复现与源码可达性核验；所有“确认”均明确标注证据类型。

## 审查范围与证据身份

| 项目 | 值 |
|---|---|
| 分支/审查 SHA | `two-level-indexer` / `f26197dfc1509583abb3fc1976fcf933b398bafc` |
| 主要文件 | `two-level-attention/exp/trace/collect_trace_lb_v.py`（SHA256 `1f2ce3731d33af02b23d56ad13c13a2c2b16d6e65066bc62eaaaf9db27a0cddc`） |
| 关联核心文件 | `two-level-attention/sparse_attn/indexer/tli_indexer.py`（SHA256 `74fe5e20cb3eaadef8532842c7c94af44a323a853b961950e4becf5a6afe26cf`） |
| 新门禁文件 | `two-level-attention/test_near_swa_boundary.py`（SHA256 `c14adb945c34b5f64c3f6ada7868cf00a5ad08f243314efc5e54d7a9cbab9a9a`） |
| 外部接口基线 | HuggingFace Transformers `v4.56.2`，`modeling_qwen3.py` blob `81b16c4ee6b6eb30acc531bf8cddd31d07065ac1` |
| 独立复核 | 另一独立审查上下文用官方 PyPI `transformers==4.56.2` wheel 逐行复核；wheel SHA256 `79c03d0e85b26cb573c109ff9eafa96f3c8d4febfd8a0774e8bba32702dd6dde`，三项分类一致 |
| 纯 Python 复现输入 | `audit-evidence-0233.py`，SHA256 `043993e006fb0eee3b61eb74ce509ab44bcbc23b958b092249e84683bf734d79`（仅本地临时证据，未提交） |

未覆盖：真实 Qwen3-8B 权重、LongBench 私有挂载、CUDA kernel、GPU 峰值显存、实际 trace 输出目录及外部已跑数据。仓库中未发现 `collect_trace_lb_v.py` 产出的受版本控制 `meta.json` 或 `layerNN.pt`，故外部产物影响须在原运行宿主复核。

## 新发现表

| ID | 状态 | 严重度 | 位置 | 违反的契约 | 影响 |
|---|---|---:|---|---|---|
| `TL-TRACE-COMPLETE-004` | **confirmed（控制流/纯 Python 合约复现）** | P1 | `collect_trace_lb_v.py:97,114-116,122-165` | 请求的层集合必须全部落盘并经校验后，才能发布完成标记；不能让“请求层”冒充“已采集层” | 缺层产物仍写 `meta.json`，下次因文件存在直接 skip；下游可能把不完整 trace 当成完整样本 |
| `TL-TRACE-LOGITS-005` | **confirmed（官方接口源码 + 算术复现；OOM 未实测）** | P1（可运行性） | `collect_trace_lb_v.py:153-169` | trace 采集只消费最后一个 token 的预测，不应为所有 32K token 构造词表 logits | 默认 Qwen3 forward 的 `logits_to_keep=0` 产生全序列 logits；Qwen3 默认词表 151,936 时 bf16 张量约 9.27 GiB，可能使本应可跑的采集 OOM |
| `TL-DEBUG-FAR-EMPTY-006` | **confirmed（静态可达；异常栈未实跑）** | P2 | `tli_indexer.py:859,881-887`；触发配置见 `test_near_swa_boundary.py:133-174` | far-empty 是受支持并已进入门禁的边界，开启诊断不应改变算法可执行性 | `i_f` 在 far-empty 时为空；`TLI_DEBUG=1 && layer_idx==1` 无空保护调用 `i_f.min()/max()`，会在调试/对拍流程中终止 |

## 复现与证据

### TL-TRACE-COMPLETE-004：缺层仍被标成完成

实际控制流：

1. CLI 的 `--layers` 在第 97 行直接形成 `layer_set`，没有验证 `0 <= layer < n_layers`。
2. hook 只把真正命中的层写入 `stored`（第 122-147 行）。
3. 落盘前没有断言 `set(stored) == layer_set`；第 160 行却把 `sorted(layer_set)` 写入 `meta.json`，不是实际 `stored` 层。
4. 下次第 114-116 行只检查 `meta.json` 是否存在就跳过，既不核对层文件集合，也不核对 model、target length、请求层、输入身份或文件可读性/哈希。

最小判别输入为请求 `{1,999}`、实际命中 `{1}`。纯 Python 合约复现输出：

```text
stored_count=1
meta_layers=[1, 999]
missing_layers=[999]
next_run_action=skip_when_meta_exists
```

这不是只针对非法层号：attention 后端/层编号接口漂移、hook 未触发某一层时，同一验收空洞也会出现。另一个不依赖接口漂移的反例是：同一 `--out` 先以 `--layers 1` 跑完，再以 `--layers 1,4` 运行；旧 `meta.json` 会让第二次直接 skip，`layer04.pt` 永远不会由普通重跑补采，直至人工删除或修复完成标记。默认层集在当前 Qwen3-8B 配置下看起来合法，但合法默认值不能替代产物完整性检查。

下游也未形成第二道硬门禁：`analyze_p0p_perlayer_potential.py:508-514` 以 meta 存在选择样本，`547-553` 对缺层仅 warning/continue，之后仍可用剩余层计算 median/verdict；因此不完整 trace 不只会被缓存，还可能进入部分样本/部分层结论。

**建议修复**：

1. 模型加载后立即校验层号范围、非空与去重；非法输入在任何样本执行前失败。
2. forward 后、任何 `.pt`/`meta.json` 发布前断言 `missing = layer_set - stored.keys()` 为空，并核验每层 `k/v/q/qpos/S` 形状与 dtype。
3. 先写唯一临时目录，全部层保存并 `torch.load` 回读通过后，原子发布完成 manifest；manifest 记录 `requested_layers`、`stored_layers`、模型/config/tokenizer/脚本 hash、每层文件 hash。
4. skip 条件改为“manifest 与当前输入身份匹配且依赖闭包完整”，不是仅看 `meta.json` 存在。
5. 消费侧把任何预注册 `(sample, layer)` 缺失升级为 hard fail，不允许以部分层继续发布总 verdict。

**最小验收**：合法默认层全齐；越界层在 forward 前失败；模拟 hook 缺一层时不得产生完成 manifest；截断一个 `.pt` 后重跑不得 skip；完整样本二次运行才允许幂等跳过。

### TL-TRACE-LOGITS-005：仅用末 token，却构造全序列 logits

当前第 154 行调用：

```python
out = model(torch.tensor([ids], device=ns.device))
```

第 168 行只消费 `out.logits[0, -1]`。而固定版本 Transformers `v4.56.2` 的 [`Qwen3ForCausalLM.forward`](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen3/modeling_qwen3.py#L442-L500) 默认 `logits_to_keep=0`，随后使用 `slice(-logits_to_keep, None)`；整数 0 等价于全序列 slice，并对全部 hidden states 执行 `lm_head`。官方 Qwen3 配置默认 `vocab_size=151936`。

对脚本默认 `target_len=32768` 的确定性算术复现：

```text
full_logits_elements=4978638848
bf16_gib=9.2734375
last_token_bf16_mib=0.2897949
waste_ratio=32768
```

这是对“产生不必要的大张量”与其大小的确认，不是一次已观察到的 OOM。真实峰值还取决于模型实现、allocator、GPU、KV/attention 临时量和卸载行为。

**建议修复**：对固定的 Transformers 4.56.2/Qwen3 接口传 `logits_to_keep=1`；若只为触发 hooks 且预测 token 不是必需产物，可进一步避免 LM head，但须保持接口兼容并明确验收。不要只在 forward 后切 `out.logits[:, -1]`，因为大张量已被计算。

**最小验收**：固定同一模型、输入、dtype 与 GPU，旧/新两版最后 token argmax 相同；用 `torch.cuda.reset_peak_memory_stats()`/同步后测峰值；记录旧版是否 OOM、新版是否完成以及实际峰值，不能把 9.27 GiB 理论值写成实测节省。

### TL-DEBUG-FAR-EMPTY-006：far-empty 调试打印访问空张量

新 N6 已冻结自然可达条件：`S=192, bs=64, sink_blocks=2, swa=128, alpha=1`，由同一公式得到 `near_blks==sink_blocks==2`，因此 `sc_far.shape[-1]==0`，第 859 行 `topk(..., k=0)` 的 `i_f` 为空。第 882-883 行已正确保护空 `i_n`，但第 886 行对 `i_f.min()/max()` 没有同样保护。

N6 的 `run_mask` 把 `layer_idx` 固定为 3，而生产调试/对拍脚本（例如 `analyze_r1c_canonical.py:123-132`）显式用 `layer_idx=1` 并开启 `TLI_DEBUG`，所以现有 N6 不覆盖这个调试分支。当前判断使用字符串真值，连 `TLI_DEBUG=0` 也会进入该分支；这不是本次主要缺陷，但修复时宜显式解析布尔值。

**建议修复**：像 `i_n` 一样先计算 `i_f_min/i_f_max = -1`（或明确的 `None`）作为空池哨兵，再打印；同时把 N6 增加一个 `layer_idx=1 + TLI_DEBUG=1` 的无异常断言并核对 `i_f` 空哨兵。

**影响边界**：默认不开 `TLI_DEBUG` 的推理不受这一打印缺陷影响；但诊断/公式对拍会在最需要检查 far-empty 时崩溃，造成该边界不可观测。

## 对已有数据与结论的影响

- `TL-TRACE-COMPLETE-004` 可能污染所有由 `collect_trace_lb_v.py` 生成、仅凭 `meta.json` 判完成的外部 trace。应逐样本比较 manifest 声明层与实际 `layerNN.pt`，逐文件回读并检查 `S`/shape/dtype/hash；任何缺层、不可读或身份不匹配样本均隔离后重采，依赖这些 trace 的 P0' 分析重跑。由于下游当前允许缺层后继续，已有 verdict 也要反查其实际输入层集合。
- 仓库内没有发现该脚本的受版本控制 `meta.json`/`layerNN.pt`，无法据此宣称现有外部数据已受影响或未受影响。
- `TL-TRACE-LOGITS-005` 本身不改变已成功生成 trace 的数值语义；主要风险是 OOM/无法完成与资源浪费。若曾出现失败，应检查是否留下无 manifest 的临时层文件；修复后同输入重采并核对最后 token。
- `TL-DEBUG-FAR-EMPTY-006` 不影响关闭 debug 的已有精度结果，但会使 far-empty 调试证据缺失或运行中止，相关“已对拍”结论需确认当时是否真的触达该配置。

## 旧发现复查与本次实际测试

- `f26197d` 已把上次的 SKIP 伪通过拆出门禁，并新增短序列 N6；静态检查确认 N6 公式确实产生 far-empty，且门禁汇总不再接受 SKIP。
- `91359a1` 已把 `gov_report` 的 `{context}` 恢复到模板中；本次未发现该五行修复引入新的模板缺陷。
- `python -m compileall -q`：本次相关四个 Python 文件通过。
- `git diff --check 96bb6622b..f26197d`：通过。
- `python two-level-attention/test_near_swa_boundary.py` 与 `test_near_swa_redgreen.py`：均在 import 阶段以 `ModuleNotFoundError: No module named 'torch'` 退出；这不是测试失败证据，也不能记为门禁通过。
- 独立复核使用固定版本 wheel 交叉检查 Qwen3 forward、配置默认值、collector 控制流和 debug 条件，三项均得到相同结论；同样未执行 GPU，不把一致意见当成性能实测。
- GPU、Qwen3 权重、LongBench 挂载不可用；未执行 kernel/e2e/精度或峰值显存实测。

## 建议修复与重测顺序

1. 先修 `TL-TRACE-COMPLETE-004`：它决定数据能否被可信消费；清点并隔离旧 trace 后再运行 P0' 分析。
2. 同一小补丁中传 `logits_to_keep=1`，以一个 1K 输入先做末 token 等价和峰值显存 A/B，再扩到 32K；保留实际 GPU/软件/模型/config/hash。
3. 给 `i_f` 空池调试打印加保护，将 N6 在 `layer_idx=1/TLI_DEBUG=1` 下执行；再运行现有 near/SWA、C-1、E110 与 prefill 非退化门禁。
4. 最后用至少一个真实 LongBench 样本走完整链：输入身份 → 9 层 K/Q/V → 完整 manifest → 下游分析读取；故意删除一层验证消费者必须拒绝。

## 下一检查点

修复提交出现后，优先复核：完成 manifest 是否绑定实际文件而非请求集合、`logits_to_keep=1` 是否在固定版本上等价且降低实测峰值、far-empty debug 是否可运行；随后轮换到 kernel/e2e 汇总脚本的失败样本保留与结果身份链。
