# 每小时代码审查报告 —— 2026-10-10 06:42

**审查分支：** `two-level-indexer`
**唯一远端：** `github.com:Chosen-David/sglang`
**基准 SHA：** `4b024a8fc1b7e506d0e4ba7265a8b272c8b27244`
**本轮远端 SHA：** `2ec9bb42cc64234b775292175444ff11e893b576`
**审查区间：** `4b024a8fc..2ec9bb42cc`
**审查人：** kimi3

---

## 审查范围

本轮远端前进 2 个提交：

- `654c2f0c8` #195 TL-E119-YARN 三审计修复：059 回执同代绑定 v2 协议 + 060 完整配置 schema/跨格门禁 + 061 纠偏机器消费
- `2ec9bb42c` docs(task): S-T012 #195 收官登记 + advice 验收补记

涉及代码变更（按优先级）：

- `two-level-attention/benchmark/RULER/pred_ruler.py`
- `two-level-attention/benchmark/RULER/score_ruler_formal.py`
- `two-level-attention/benchmark/RULER/yarn_receipt.py`
- `two-level-attention/benchmark/RULER/test_e119_yarn_binding_059_060_061.py`
- `two-level-attention/benchmark/RULER/test_e119_yarn_identity_057.py`
- `two-level-attention/benchmark/RULER/e119_yarn_producer_runner_059.py`
- `two-level-attention/exp/trace/analyze_e119_ruler128k_formal.py`

工作树未提交改动仅包含 `paper/archive/scores.json`、research docs、`exp/trace/results/*.json` 等数据/文档产物，无实质代码改动，已按规则跳过。

---

## 实测验证结论

已运行红绿测试：

- `PYTHONPATH=$PWD/two-level-attention python3 -m benchmark.RULER.test_e119_yarn_binding_059_060_061` —— **23/23 PASS**
- `PYTHONPATH=$PWD/two-level-attention python3 -m benchmark.RULER.test_e119_yarn_identity_057` —— **10/10 PASS**
- `PYTHONPATH=$PWD/two-level-attention python3 -m benchmark.RULER.test_e116d_gate` —— **10/10 PASS**

测试全部通过，说明 059/060/061 的主功能门禁按设计生效。但在变更文件中发现 2 处实测可确认的 bug（一处结果元数据矛盾、一处测试断言写错）。

---

## 发现 1：analyze_e119_ruler128k_formal.py 汇总仍硬编码「YaRN factor 2.0」，与 061 纠偏语义直接矛盾

### 位置

`two-level-attention/exp/trace/analyze_e119_ruler128k_formal.py` 第 717 行

```python
"length_tier": "L131072（YaRN factor 2.0，128K 全档）",
```

### 实测证据

061 修复在本轮提交中落地，其设计意图是：128K 三份旧 manifest 的旁挂纠偏文件声明 `operator_declared_not_effective`，消费解析后 `effective_yarn_factor` 必须回退为 `null`，不再把 2.0 当作 effective 值，也不脑补 4.0。

用 R4 同款 fixture 实际跑出的 summary 如下：

```
identity.length_tier: L131072（YaRN factor 2.0，128K 全档）
mavg effective_yarn_factor: None
mavg provenance: operator_declared_not_effective
```

同一份 summary JSON 中：

- `identity.length_tier` 明确声称「YaRN factor 2.0」；
- `identity_gate.per_arm_yarn_identity.{mavg,aavg,FullKV}.effective_yarn_factor` 全部为 `null`；
- 对应 `yarn_factor_provenance` 全部为 `operator_declared_not_effective`。

这是结果元数据层面的自相矛盾。下游若只读 `identity.length_tier` 会误以为 128K 档 effective factor 已被确认为 2.0，与 061 纠偏侧car 的显式声明冲突。

### 影响

- **指标/报告污染**：summary 是 E119 收口的最终产物，top-level 字段给出一个已被纠偏机制否定的 factor 值，可能造成论文/报告引用错误。
- **不一致不可解释**：同一 JSON 内两个位置给出相反结论，读者/自动化消费者无法判断以何为准。

### 修复建议

将该字符串改为从 `yarn_resolved` / `per_arm_yarn_identity` 推导，或至少与 061 语义一致。例如：

```python
"length_tier": "L131072（YaRN factor operator_declared_not_effective，128K 全档）",
```

更优做法是在构建 `out` 前统一取三臂 `effective_yarn_factor`：

```python
_factors = {y["effective_yarn_factor"] for y in per_arm_yarn.values()}
factor_txt = (str(next(iter(_factors))) if len(_factors) == 1 and None not in _factors
              else "operator_declared_not_effective")
"length_tier": f"L131072（YaRN factor {factor_txt}，128K 全档）",
```

### 复核方法

```bash
PYTHONPATH=$PWD/two-level-attention python3 -m benchmark.RULER.test_e119_yarn_binding_059_060_061
# 观察 R4 PASS 后，可在临时目录检查 e119_ruler128k_formal_summary.json
# 或直接运行 analyze_e119_ruler128k_formal.py 后读取 summary["identity"]["length_tier"]
```

---

## 发现 2：test_e119_yarn_binding_059_060_061.py 中 glob 模式切片写错，断言实质失效

### 位置

`two-level-attention/benchmark/RULER/test_e119_yarn_binding_059_060_061.py` 第 285 行

```python
assert not glob.glob(os.path.join(d, "vt-*.jsonl.gen-*"[:-7] + "*")) \
    or all(not g.endswith(".jsonl") for g in gens)
```

### 实测证据

直接复现该表达式：

```python
>>> "vt-*.jsonl.gen-*"[:-7] + "*"
'vt-*.json*'
```

作者意图是匹配形如 `vt-stubm-09090909.jsonl.gen-20261010...` 的临时 generation 文件，但 `[:-7]` 把字符串截成了 `vt-*.json`，再拼 `*` 得到 `vt-*.json*`。该模式只能匹配以 `.json` 开头的后缀，无法覆盖 `.jsonl.gen-*`，因此 `glob.glob(...)` 永远为空列表，`not []` 恒为 `True`，第一个 `assert` 退化为无操作。

### 影响

- 该测试用例（P2）仍能通过，因为第二个断言：

  ```python
  assert all(not g.endswith(".jsonl") for g in gens)
  ```

  已足够覆盖「临时 generation 不得以 .jsonl 结尾」的校验。
- 但第一个断言本欲额外检查「没有以 `.jsonl` 结尾的临时 generation 污染 glob」，由于模式错误，该检查没有实际执行，属于测试覆盖度缺失。

### 修复建议

去掉无意义的切片，直接使用正确模式：

```python
assert not glob.glob(os.path.join(d, "vt-*.jsonl.gen-*")) \
    or all(not g.endswith(".jsonl") for g in gens)
```

或进一步简化，仅保留第二个断言即可（语义已足够）。

### 复核方法

```python
python3 -c 'print(repr("vt-*.jsonl.gen-*"[:-7] + "*"))'
# 输出 'vt-*.json*'，确认模式错误
```

---

## 未列入报告的旧问题

- `score_ruler.py` 中 `exp_sha[i][:16]` 与 manifest schema 允许 `len >= 16` 的潜在不一致属于 pre-existing 问题，本轮 diff 未触及该文件，按纪律不重复报告。
- `pred_ruler.py` 生成循环中的 `pred_idx.item()` GPU-CPU 同步、逐样本循环等性能问题为既有实现，本轮 059 改动未改变该循环，不重复报告。

---

## 结论

本轮审查发现 2 处实测可确认的 bug，均已给出位置、证据、影响、修复建议与复核方法。核心风险是发现 1：汇总产物 `e119_ruler128k_formal_summary.json` 在 061 纠偏已生效的情况下仍对外宣称「YaRN factor 2.0」，与同一 JSON 内的 `operator_declared_not_effective` 结论冲突，建议优先修复。
