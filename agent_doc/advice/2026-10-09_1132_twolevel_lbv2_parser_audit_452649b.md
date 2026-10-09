# TwoLevel LongBench-v2 评分解析审计（2026-10-09 11:32）

## 审查目标、基线与范围

- **目标分支 / 远端审查头**：`two-level-indexer` / `452649b9ae80b312eb1d7ba82a4ddbd19a17779b`。
- **实现层**：本轮远端相对上一份审计提交 `496a94a37` 只新增了对 `TL-RULER-*-030..036` 的确认与修复排期，没有源码变化；LongBench-v2 解析实现当前仍来自 `75268f8773292f13f8add3dbdd13082bf1a58d78`，后续 E116a 修改未改变该解析逻辑。
- **入口链**：LongBench-v2 `data.json` → `pred.py` 生成原始文本及 `pred_choice` → `eval.py` 重新从原始 `pred` 解析选项 → `e109_full_lbv2.json` 三臂汇总。
- **实际检查文件及 SHA256**：
  - `two-level-attention/benchmark/LongBench/pred.py`：`4bc423acf77b5dfaf45bb1978b2f2438e114baa7c50521517062f9af96d98307`
  - `two-level-attention/benchmark/LongBench/eval.py`：`ec68b7c5940d6804fb41bd25b603ee6759077fa483683fc5a0a3739766964eee`
  - `two-level-attention/exp/trace/results/e109_full_lbv2.json`：`d0fca3d71bed92b5341a52ef4f75aba8bf94c6cc8dc3c4e3a5f249c3971ef243`
- **官方参照**：2026-10-09 重新克隆 `THUDM/LongBench`，HEAD 为 `2e00731f8d0bff23dc4325161044d0ed8af94c1e`；官方 [`pred.py:58-68`](https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/pred.py#L58-L68) 只接受 `The correct answer is (X)` / `The correct answer is X` 两种格式，否则返回 `None`。
- 本轮仅执行源码静态追踪和 CPU 纯函数反例；未运行模型、GPU、503 条真实预测或性能测试，也未修改实现、测试与实验数据。

## 新发现

| ID | 状态 | 严重度 | 位置 | 结论 |
|---|---|---:|---|---|
| `TL-LBV2-PARSER-037` | **confirmed（CPU 执行真实函数 + 官方实现对照）** | P1（精度口径/三臂排序） | `benchmark/LongBench/pred.py:121-139,301-314`；`benchmark/LongBench/eval.py:20-42` | 本地兜底使用大小写不敏感的 `\b([ABCD])\b` 并取全文第一个独立字母；英文冠词 `a` 会被当成选项 A，甚至在文本后面明确写 `option C` 时仍先返回 A。第二条 `The correct answer is ([A-D])` 同样启用 `IGNORECASE`，也会把 `The correct answer is a difficult choice...` 中的冠词 `a` 判成 A。该行为既不等于回答语义，也不同于官方评分器，能同时制造假阳性和假阴性。 |

## 可运行复现证据

复现通过 AST 直接加载当前仓库两个真实函数和官方 `extract_answer()`，不复制或改写其判断逻辑。核心输入与输出如下（fixture SHA256：`8278c366dcd635e737bd62baa3a4da69e1e0ef5db54b2054ccad54eccaf8b684`）：

```json
[
  {
    "text": "The correct answer is a difficult choice; I lean C.",
    "local": "A",
    "official": null
  },
  {
    "text": "This is a hard call; option C is best.",
    "local": "A",
    "official": null
  },
  {
    "text": "The correct answer is (C).",
    "local": "C",
    "official": "C"
  },
  {
    "text": "The correct answer is C.",
    "local": "C",
    "official": "C"
  }
]
```

另一个定向反例把第一条文本分别配真值 A/C：当前 `lbv2_choice_score()` 对真值 A 给 `1.0`，对真值 C 给 `0.0`；官方解析均为 `None`。因此这不是“只放宽格式但不改变正确性”的差异。

独立只读复核在 Python 3.12.14 下另构造了更小的假阳性：

```json
{"ground_truth":"A","prediction":"I cannot provide a definitive answer."}
```

fixture SHA256 为 `feef377374db105d4e6cd4514f9e1f6444188f4e3ab04eb40094cc72fdee23e9`；实测 `current_extract='A'`、`current_score=1.0`，官方 `extract_answer=None`、得分 `0.0`。独立复核同时确认：两处本地函数彼此一致，根因是它们共同偏离官方，而不是 `pred.py` / `eval.py` 漂移；Markdown `*` 与真值大小写也不是本反例根因。

正例仍与官方一致，说明问题集中在额外兜底，而不是两条官方格式本身。仓库内没有覆盖 `extract_choice_letter` / `lbv2_choice_score` 的专门测试。

## 对已有数据与论文结论的影响

1. `e109_full_lbv2.json` 报告 503 题，aavg / FullKV / mavg 分别为 `32.60 / 32.21 / 32.01`，恰好对应四舍五入后的 `164 / 162 / 161` 道正确题。单题权重为 `100/503 = 0.198807` 个百分点；两题为 `0.397614` 个百分点，已与 aavg 相对 FullKV 的 `+0.39` 同量级；一题也与 FullKV 相对 mavg 的 `0.20` 同量级。
2. **本轮不能确认这 503 条实际输出中出现了多少解析分歧。** 三臂原始 JSONL / 分片未提交且当前环境没有副本，汇总 JSON 也没有保存逐题原始文本、官方解析重评分或解析器 hash。当前解析是官方解析的宽松超集，若命中额外兜底，只会虚增对应臂的绝对正确数；但三臂虚增数量可能不同。因此现有三臂排序影响状态是 **inconclusive**，不能据本反例直接改写三项分数，也不能继续把小于两题的差值当成稳健结论。
3. 该缺陷只影响 LongBench-v2 四选一解析；本轮没有证据表明 LongBench-v1、RULER、kernel 数值或 two-level attention 选择结果因此错误。

## 建议修复与最小重测

1. 正式口径直接复用官方两条大小写敏感模式；无匹配记 `None/0`。若确实要支持额外格式，必须把它命名为自定义 scorer，并采用无歧义的锚定规则（例如完整 `option (X)` / `answer: X` / 最后一行单独选项），禁止用大小写不敏感的全文首个独立 `[A-D]`。
2. 消除 `pred.py` 与 `eval.py` 两份手写正则：放到一个无重依赖的纯函数模块；生成端可以保存 `pred_choice` 作为观察值，但正式评分端应从冻结的原始文本使用同一版本解析，并把 scorer 版本/源码 SHA 写入 receipt。
3. 增加表驱动测试：两条官方正例；小写/星号；冠词 `a`；`option C`；多个候选字母；解释中先提 A 后最终答 C；拒答；空输出。每例同时校验生成端和评分端，官方模式下与固定官方实现逐项一致。
4. 从三臂原始 503 条 JSONL 统一重评：保存 `_id → raw_pred → old_choice → official_choice → answer → old/new judge` 的差异表，检查三臂 `_id` 集合完全相等；重新计算分数和置信区间/逐题配对差异。若原始文本已丢失，不能从现有三项汇总反推修复后分数，须在最终修复 SHA 与同一冻结输入上重跑。

最小通过标准：上述全部反例不再把冠词 `a` 识别为 A；官方格式正例保持；503 条三臂原始文本均可按同一 scorer 重放；新的三臂排序报告绑定数据、代码与 scorer hash，并单独报告发生解析变化的题数。若变化题数达到 1-2 条，应撤回旧的“小幅领先/落后”终判并以重评分结果替代。

## 旧发现复查、资源阻塞与下一检查点

- 最新远端只确认并排期修复上一轮 `TL-RULER-*-030..036`，未出现可复验的修复源码；本报告不重复这些旧发现。
- 既有 `e109_full_lbv2.json` 已明确 31.5K 内部口径，未冒充官方 120K 榜单；本轮不把截断选择重复记录为新 bug。
- 阻塞项：三臂原始 JSONL/分片、真实模型/GPU 与冻结 LongBench-v2 输入在当前环境不可用；因此没有真实 503 条重评分、精度重跑或 GPU 验收。
- 下一检查点：先复验 `TL-LBV2-PARSER-037` 的统一解析修复，并要求旧三臂原始文本重放；同时等待 E116e 修复提交后按上一报告矩阵复验 RULER 030-036。只有 scorer 身份和逐题结果闭合后，才适合引用 LongBench-v2 小于两个样本量级的差值。
