# two-level-indexer 小时审查报告（2026-10-11 07:45）

- 审查分支：`origin/two-level-indexer`
- 本轮基准 SHA：`d280bd72ce376cab36a6ef7b3decd25898f1e9e5`
- 本轮远端 SHA：`db2c539f1da1c0490d7f5feba2aa9031185d1573`
- 审查区间：`d280bd72c..db2c539f1`
- 审查人：kimi3

## 区间摘要

本轮 fetch 后发现 1 个新提交：

```
db2c539f1 docs(advice): audit RULER llm_eval snapshot gap
```

该提交本身只新增一份审计文档，未修改业务代码。工作树中除数据产物（`paper/archive/*.json`、`two-level-attention/exp/results*/`、锁文件、`ref/figs/`、`research/docs/` 等）外，无未提交的代码改动。

经对新增文档指出的问题独立复现，确认 `two-level-attention/benchmark/RULER/llm_eval.py` 仍存在 treatment snapshot 未接入的缺陷；其余审查重点区域（`python/sglang/srt/layers/attention/tli/`、`two-level-attention/sparse_attn/` 核心实现、`two-level-attention/exp/trace/` 与 `benchmark/` 其他脚本）在本区间内无源码变更，未发现新 bug 或性能退化点。

---

## 发现 1：RULER `llm_eval.py` 仍未接入 treatment 快照（082 残余）

### 位置

- `two-level-attention/benchmark/RULER/llm_eval.py:47,54,82,121-128`
- 关联：`sparse_attn/patches/patch.py:38-52`、`sparse_attn/indexer/tli_indexer.py:137-160,189-209`、`sparse_attn/info.py:270-323`

### 结论

P1。`llm_eval.py` 是 RULER 目录下另一个可直接执行的 benchmark 入口，但 081 修复只接入了 `pred_ruler.py`，遗漏了它。当前代码路径：

1. 第 47 行 `method_name = get_method_name_with_info(args)`：未传 `snapshot`，此时解析配置文件内容 A 生成文件名哈希；
2. 第 54 行 `result_filename = f"{method_name}_{timestamp}"`：把 A 固化进结果文件名；
3. 第 56-80 行加载模型并做一次 dense 示例生成，进一步扩大 A 到实际 patch 的时间窗口；
4. 第 82 行 `register_patch(model, args)`：仍不传 `snapshot`；
5. `register_patch` 中对每层 attention 调用 `TLIIndexer(args)`（`snapshot=None`），走旧路径每层重新 `open`/`torch.load` 配置文件；
6. 第 121/127 行用基于 A 的文件名保存实际运行结果。

因此，若 `tli_layer_skip_path` 或 `tli_proj_basis` 在第 47 行之后、第 82 行 patch 完成之前被替换，会出现「文件名标 A、运行时实际用 B」的 provenance 污染；若替换发生在 `register_patch` 遍历多层期间，还会形成跨层 A/B 混代。

### 实测证据

在本地环境（torch 2.8.0+cu128）运行独立复现脚本：

```python
# /tmp/repro_llm_eval_snapshot_gap.py
import os, sys, json, tempfile, argparse
sys.path.insert(0, "/home/wangyuanshuo02/sglang/two-level-attention")
from sparse_attn.info import get_method_name_with_info, resolve_treatment_snapshot, _treatment_hash
from sparse_attn.indexer.tli_indexer import TLIIndexer

tmpdir = tempfile.mkdtemp(prefix="tli_repro_")
mask_path = os.path.join(tmpdir, "layer_skip.json")
with open(mask_path, "w") as f:
    json.dump({"skip": [1]}, f)

args = argparse.Namespace(
    method="tli",
    tia_block_size=64, tia_level1_topk=128, tia_level2_topk=1024, tia_level2_cmp_ratio=2,
    tia_enable_async_topk=False,
    tli_enable_subspace=True, tli_subspace="full", tli_enable_kmeans=True,
    tli_enable_layer_skip=True, tli_layer_skip_path=mask_path,
    tli_far_select="4bit", tli_near_select="4bit", tli_sim=0.9, tli_sim_dims="subspace",
    tli_far_clusters=256, tli_far_niter=10, tli_far_blocks=16, tli_far_tokens=512,
    tli_far_method="minmax", tli_near_method="avg",
    tli_alpha=0.0, tli_beta=0.0, tli_gamma=1.0,
    tli_sparse_prefill=False, tli_moba=False, tli_sigma_select="none", tli_sigma=8.0,
    tli_per_q_head=False, tli_proj_basis=None, tli_static_pair=False,
)

# 模拟 llm_eval.py 第 47 行：先求 method_name（固化到 result_filename）
method_name_before = get_method_name_with_info(args)
hash_in_name = method_name_before.rsplit("_h", 1)[-1]
print(f"NAME_HASH_BEFORE h{hash_in_name}")

# 模拟 47-82 行之间配置文件被替换
with open(mask_path, "w") as f:
    json.dump({"skip": [2]}, f)

# 模拟 register_patch 不传 snapshot 时创建的 indexer
indexer_no_snapshot = TLIIndexer(args)
print(f"RUNTIME_SKIP_IDS_WITHOUT_SNAPSHOT {indexer_no_snapshot._skip_ids}")

# 运行时若重新 resolve，会得到新 hash
snapshot_after = resolve_treatment_snapshot(args)
hash_after = _treatment_hash(args, snapshot_after)
print(f"RUNTIME_MANIFEST_HASH_AFTER h{hash_after}")
print(f"MISMATCH_REACHABLE {hash_in_name != hash_after}")

# 跨层混代：同一 register_patch 中若文件中途再变，两层读到不同内容
with open(mask_path, "w") as f:
    json.dump({"skip": [1, 3]}, f)
idx_first = TLIIndexer(args)
with open(mask_path, "w") as f:
    json.dump({"skip": [2, 4]}, f)
idx_second = TLIIndexer(args)
print(f"CROSS_LAYER_MIXED {idx_first._skip_ids != idx_second._skip_ids}")
print(f"first_skip={sorted(idx_first._skip_ids) if idx_first._skip_ids else None}")
print(f"second_skip={sorted(idx_second._skip_ids) if idx_second._skip_ids else None}")
```

运行输出：

```text
NAME_HASH_BEFORE hd99a085926
RUNTIME_SKIP_IDS_WITHOUT_SNAPSHOT {2}
RUNTIME_MANIFEST_HASH_AFTER hfd5b06d93d
MISMATCH_REACHABLE True
CROSS_LAYER_MIXED True
first_skip=[1, 3]
second_skip=[2, 4]
```

实测表明：

1. 文件名哈希 `hd99a085926` 基于配置 A（`skip=[1]`）；
2. 运行时不传 snapshot 创建的 indexer 实际读到配置 B（`skip=[2]`）；
3. 同一 args 重新 resolve 得到的 manifest hash 为 `hfd5b06d93d`，与文件名哈希不同， provenance 污染可达；
4. 跨层混代同样可达：连续两次创建 `TLIIndexer(args)` 在文件中途替换后，分别读到 `[1,3]` 和 `[2,4]`。

### 影响

- 直接运行 `llm_eval.py` 并把 `.txt` 结果用于结论时，若运行窗口内配置文件发生变化，结果文件名与实际 treatment 不一致，可能导致错误归因。
- 跨层混代时模型行为既不对应 A 也不对应 B，实验不可复现。
- 当前正式 RULER 生成链使用 `pred_ruler.py`（已接入 snapshot），LongBench 使用 `pred.py`（已接入 snapshot），未发现它们受此问题影响；仓库内也无其他脚本引用 `llm_eval.py`，因此不能据此撤回现有 E109/E119/E123 正式分数或论文结论。

### 修复建议

1. 在 `llm_eval.py` 导入 `resolve_treatment_snapshot`；当 `args.method == "tli"` 时，在首次调用 `get_method_name_with_info` 之前先冻结 snapshot：

   ```python
   if args.method == "tli":
       snapshot = resolve_treatment_snapshot(args)
   else:
       snapshot = None
   ```

2. 第 47 行改为 `method_name = get_method_name_with_info(args, snapshot)`，第 82 行改为 `register_patch(model, args, snapshot)`，确保文件名、运行时、未来 sidecar/receipt 全部消费同一对象。

3. 考虑在 `register_patch(..., snapshot=None)` 且 `args.method == "tli"` 时 fail closed，或在 `TLIIndexer` 旧路径上加显式 legacy 标记，避免新增入口无意旁路快照。

4. 为 `llm_eval.py` 添加交错回归：先 resolve A，再替换为 B，断言 method_name、每层 skip/basis、落盘 manifest 仍为 A；`python` 与 `python -O` 双跑。

### 复核方法

```bash
# 1. 检查 llm_eval.py 是否正确调用 resolve_treatment_snapshot
python - <<'PY'
import ast, sys
tree = ast.parse(open("two-level-attention/benchmark/RULER/llm_eval.py").read())
for node in ast.walk(tree):
    if isinstance(node, ast.Call):
        func = ast.unparse(node.func)
        if func in ("get_method_name_with_info", "register_patch", "resolve_treatment_snapshot"):
            print(f"line {node.lineno}: {func}({', '.join(ast.unparse(a) for a in node.args)})")
PY

# 2. 复现脚本（需 torch 2.8+）
python /tmp/repro_llm_eval_snapshot_gap.py
```

修复完成的标准：

- `llm_eval.py` 中存在 `resolve_treatment_snapshot(args)` 调用；
- `get_method_name_with_info(args, snapshot)` 与 `register_patch(model, args, snapshot)` 均传入同一 `snapshot`；
- 复现脚本输出 `MISMATCH_REACHABLE False` 且 `CROSS_LAYER_MIXED False`。

---

## 旧发现复查

- `TL-E121-OUTPUT-SNAPSHOT-081`：在 `pred_ruler.py` / `pred.py` 上已修复；在 `llm_eval.py` 上为本轮确认的新遗漏残余，不属于 C1/C2/C3，单独报告如上。
- C1（eager -inf top-k）、C2（taskmd graph 短行重复）、C3（near 簇分缺 scale）：本区间无相关源码变更，不重复报告。

## 性能优化点

本区间无源码变更，未发现新的重复计算、GPU 同步热路径、逐请求循环或 kernel 旁路退化点。
