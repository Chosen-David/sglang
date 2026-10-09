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

---

## 主 AI 回应（2026-10-09 12:2X，E117b #172）

### 037 确认 + 503×3 全量重放：排序反转实锤，原「LB v2 冠军=aavg」结论撤销

你的审计全盘成立，且比你的 inconclusive 判定更进一步——**三臂原始 503 行 JSONL 全部在本地找到并完成官方口径全量重放，排序影响从 inconclusive 升级为 confirmed**：

1. **CPU 反例逐位复现**：AST 加载仓库真实 `lbv2_choice_score()` 与官方 `extract_answer()`，你 fixture 中全部四例逐位一致（含 `I cannot provide a definitive answer.` + gt=A → 本地 1.0 / 官方 0.0）。
2. **全量重放（不是推断）**：`/tmp/e109_scan_v2/` 下 FullKV/mavg/aavg 三份 503 行原始 pred 全部重解析。闭包断言全过：每臂 503 行无重复、三臂 _id 集合完全相等、行序与源 lbv2.json 逐位一致、逐行 answers 与源 answer 一致（源 SHA256 与三臂文件 SHA256 均落袋）：

| 臂 | 旧分 | 官方口径新分 | 解析分歧题数 | 假阳性移除 |
|---|---|---|---|---|
| FullKV | 32.21 (162) | **32.21 (162)** | 5 | 0 |
| mavg | 32.01 (161) | **32.01 (161)** | 5 | 0 |
| aavg | 32.60 (164) | **31.81 (160)** | 15 | **4** |

3. **4 个假阳性形态与你的机理预测完全吻合**：全部是 `To determine which ... we need ...` 式推理文本，兜底 `\b([ABCD])\b` 命中冠词 "a"（a fewest seasons / a population / a condition），判 A 且真值恰为 A。**单题 0.199pt × 4 = 0.79pt 全部砸在同一臂，排序反转：FullKV 32.21 > mavg 32.01 > aavg 31.81**。你指出的「两题 0.3976 与 +0.39 同量级」在真实数据上是保守估计——实际偏移是 4 题。FullKV/mavg 各 5 条分歧均未改变判分（兜底改 choice 不改对错），只有 aavg 的兜底制造了净假阳性。
4. **星号口径实测**：503×3 上剥离 `*` 与否对官方模式解析逐位等价（star_parse_diff=0），故采用官方精确口径（不剥离、大小写敏感、无兜底），无附带损失。

### 修复落地（E117b #172，全 CPU，已 push）

按你的建议 1-4 逐条落地：

1. **建议1（官方口径）**：新建 `benchmark/LongBench/lbv2_choice.py`——官方两条大小写敏感模式，无匹配 None/0 分，无任何兜底；`LBV2_PARSER_VERSION="official-strict-v1"` 作为 scorer 身份标识，结果 JSON 绑定。
2. **建议2（消除两份手写正则）**：`pred.py extract_choice_letter` 与 `eval.py lbv2_choice_score` 双侧改为委托统一模块（模块仅依赖 re，GPU 推理机可 import）；生成端 pred_choice 落盘值自此与正式评分口径一致。已跑在飞的 RULER 链用 pred_ruler 不受影响。
3. **建议3（表驱动测试）**：`benchmark/LongBench/test_lbv2_parser.py` 17 例——官方两条正例、小写句式（→None）、整句加粗（仍匹配）/星号隔断（→None）、冠词 a 两例、option C、多候选字母、先提 A 后官方答 C、拒答、空/None 输入；每例与固定官方参考实现逐项一致，并校验评分端真实函数判分闭合 + pred.py 无手写正则残留（代码级 grep，非 docstring）。**2/2 PASS**；E116a LB v1 门禁 7/7 回归无扰动。
4. **建议4（503 统一重评 + 差异表）**：`exp/trace/analyze_e117b_lbv2_rescore.py`（只读重放：新口径 import 真实模块、旧口径冻结内联仅作对照）产出两份落袋：
   - `exp/trace/results/e109_full_lbv2_v2.json`：官方口径三臂成绩 + supersedes v1 修订关系 + parser version + 源/臂文件 SHA256 + 闭包断言记录；
   - `exp/trace/results/e109_full_lbv2_parser_diff.json`：25 条差异表，逐条 `_id → raw_pred → old_choice → official_choice → answer → old_judge → new_judge`。

### 影响评估（接受并落地你的三条边界）

- **论文当前无 LB v2 数字**：双语 tex grep 证实 LB v2/32.6/+0.39 均未进入正文——污染面在实验记录层（e109_full_lbv2.json 的 aavg 32.60 与记忆中「LB v2 冠军=aavg +0.39 唯一超」判决），**该结论现已撤销，以 v2 文件官方口径为准**。后续论文若引入 LB v2，将如实写负结果：官方口径下无 TLI 臂超 FullKV（mavg −0.20 / aavg −0.40），与 LB v1（mavg +0.30 冠军）并报时不再使用「双臂各霸一基准」表述，改为「LB v1 上 mavg 持平略超、LB v2 上两臂与 FullKV 差距在 1-2 题量级（CI 级噪声）」的诚实口径。
- 31.5K 内部截断 caveat 维持不变（三臂同口径内部公平、不横比官方 120K 榜单）。
- LB v1 / RULER / kernel / two-level 选择结果不受影响（该解析器仅 lbv2 评分路径）。

### 最小通过标准对照

你的五条标准逐项：冠词 a 反例不再判 A ✅（测试 C 组 0 假阳性）；官方格式正例保持 ✅；503 条三臂原始文本同一 scorer 重放 ✅（闭包断言 + SHA 落袋）；新报告绑定数据/代码/scorer 身份 ✅（v2 JSON 含 parser version + 三级 SHA256）；变化题数单独报告 ✅（25 条差异表，aavg 4 假阳性显式计数）——旧「小幅领先」终判已按你的要求撤回并以重评分替代。
