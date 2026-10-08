# TwoLevel E100 配对置信区间审查：FullKV 缺少样本身份校验

## 审查结论

- **审查分支/HEAD**：`two-level-indexer` / `cc680d049acdbf20fe87ff89d6d71f3362fafb7f`
- **实际实现基线**：`b02849052b44e9195d51b11e6b2b6da5fe1947a0`。其后的 `3048c7004`、`cc680d049` 仅新增 `agent_doc/advice/` 文档，本轮没有把 advice-only 提交当成源码更新。
- **新增发现**：1 个已确认的统计脚本缺陷 `TL-AUD-CI-PAIR-001`。E100 脚本对 tail/full 做了逐位答案校验，却在计算 tail/FullKV 的 paired bootstrap 前只校验长度；未来输入若同长度但顺序不同，会静默产生无效的配对置信区间。
- **当前数据影响**：仓库当前归档的 13 个 tail/FullKV 文件共各 3150 行，`answers` 逐任务、逐行完全相同，因此**没有发现已发布的 `+0.16` 和 `[-0.43, 0.76]` 被该缺陷改变的证据**。但 `answers` 并非唯一 example ID，且结果 JSON 没有绑定原始输入哈希，故当前样本身份仍属 **inconclusive**，不能反向宣称现有 CI 已被充分验证。缺陷影响的是脚本的 fail-closed 保证和后续重跑可靠性，也不应表述为现有论文数值已错。
- **改动范围**：本轮只新增本审查 Markdown；未修改实现、测试脚本或实验数据。

## 范围与环境

本轮沿 `E100 tail32 -> FullKV 原始 jsonl -> 逐样本 scorer -> paired bootstrap -> 论文结论` 检查：

- `two-level-attention/exp/trace/analyze_e100_tail_ci.py`
- `two-level-attention/exp/trace/results/e100_tail_ci.json`
- `two-level-attention/exp/results_longbench/Qwen3-8B/pred_E100TAIL_mavg_a0.125_b0.375_g0.625/`
- `two-level-attention/exp/results_longbench/Qwen3-8B/pred_1024/`
- `paper/TLI_paper.tex:213`

脚本 SHA256：`4f577126821cf98f31f399a12fb24e8912ef9e991608e095d01f59116f5a75ac`。结果 JSON SHA256：`258a17baae887e3141f6d1ad362963a4418d8178f92f723993b049458cc2fd98`。本轮对仓库归档目录应用脚本相同的 glob 选择规则，所得 26 个可见输入文件的“路径 + 文件内容”清单 SHA256 为 `75b6c480cecf51dcffd6e9c183c1f9a7f8edc746b35126adcae3f2e723584c63`。原脚本硬编码 tail/full 到 `/tmp`、ROOT 到 `/home/wangyuanshuo02`，结果 JSON 又未记录输入路径与哈希，因此该清单不能冒充原执行瞬间输入的不可变绑定。

当前环境没有 PyTorch/CUDA，且完整分析脚本导入时缺少 `jieba`，所以本轮没有冒充 GPU 或完整 scorer 重跑。完成了真实 jsonl 身份核对、Python 语法编译、shell 语法检查和独立 NumPy 反例。

## 新增发现

| ID | 状态 | 严重度 | 位置 | 违反的契约 | 当前数据影响 |
|---|---|---:|---|---|---|
| `TL-AUD-CI-PAIR-001` | **confirmed（静态追踪 + CPU 反例）** | 中 | `two-level-attention/exp/trace/analyze_e100_tail_ci.py:72-90, 118-123, 174-186` | “逐样本配对 bootstrap”要求两臂的同一数组位置代表同一示例。脚本只对 tail/full 执行 `answers` 逐位断言；FullKV 只返回分数并校验长度，然后直接 `tail - fkv`。 | 当前归档文件未发现 `answers` 序列差异，但没有稳定 example ID，现有 CI 的真实样本身份验证仍为 inconclusive；若后续 FullKV 文件被重排、换版本或 glob 选中另一份同长度文件，均值仍可看似正确，但 CI 会静默失真。 |

### 根因与可达路径

1. `load_task()` 在第 81--83 行读取 tail/full 的 `answers` 并断言完全相等。
2. `load_fkv()` 在第 87--90 行只返回 `per_sample_scores()`，丢弃样本身份。
3. 主循环第 121 行只断言 `len(tail) == len(fkv)`，第 123 行直接计算 `tail - fkv`。
4. 输出第 180 行的 `pairing` 元数据只说明 tail/full 已校验；但同一结果文件第 186 行又输出 `tail_vs_fullkv_ci`。这部分 CI 实际没有相同级别的配对保障。

`sorted(glob)[-1]` 还允许脚本在目录出现新的同长度 FullKV 文件时自动切换输入。只要新文件样本顺序变化，长度断言不会失败，脚本仍会输出正式 verdict。

### 最小 CPU 反例

以下反例保持 tail 与 FullKV 的边际分数及均值完全不变，仅重排 FullKV；按脚本相同的逐位置作差和 percentile bootstrap，区间随错误配对变化：

```python
import numpy as np

tail = np.array([1., 1., 0., 0.])
fkv_a = np.array([0., 0., 1., 1.])
fkv_b = np.array([1., 0., 1., 0.])  # 同一组分数，仅顺序改变

def ci(fkv):
    rng = np.random.default_rng(20261004)
    d = tail - fkv
    idx = rng.integers(0, len(d), size=(10000, len(d)))
    return d.mean(), np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
```

实际输出：

```text
aligned_or_order_A       mean=0.0, CI=[-1.0, 1.0]
same_marginal_reordered_B mean=0.0, CI=[-0.75, 0.75]
tail/fkv_a/fkv_b means    0.5 / 0.5 / 0.5
```

这证明宏平均差不受排列影响并不能证明 paired CI 有效；协方差和区间取决于正确配对。

## 对仓库现有 E100 归档数据的核验

对仓库归档目录应用脚本的 `sorted(glob)[-1]` 规则后，对 13 个任务执行了真实 `answers` 列表逐位比较：

- 13/13 任务 `answers_equal=True`；
- tail 与 FullKV 各 3150 行；
- 每个任务长度一致，包含 10 个 200 样本任务、`multifieldqa_en` 150 样本、`lcc`/`repobench` 各 500 样本；
- 输入清单 SHA256 如上。

这排除了可由 `answers` 序列观察到的明显乱序，但不是严格 example-ID 证明：例如 `passage_retrieval_en` 的 200 行只有 30 种不同 `answers` 列表，`musique` 的 200 行只有 149 种，同答案组内重排不会被发现。加之原执行输入来自未绑定哈希的绝对路径，本轮只能判定“未发现现有 CI 受影响的证据”，不能判定“现有 CI 已确认不受影响”。`paper/TLI_paper.tex:213` 的数值本轮没有撤回证据，但应在加入稳定样本 ID/输入哈希后重验。需要修复的是生成链：它目前依赖未被程序验证的隐含顺序约定。

## 建议修复与最小重测

1. 让 `per_sample_scores()` 或新的加载函数同时返回 `scores`、`answers`，最好再返回稳定 example ID；FullKV 与 tail 同样执行逐位身份断言。
2. 若数据格式没有稳定 ID，至少断言完整 `answers` 列表一致；更稳妥的是保存规范化样本身份哈希，避免不同问题恰好共享相同答案时误配。
3. 在输出 JSON 的 `method.pairing` 中分别记录 `tail_vs_full`、`tail_vs_fullkv` 的校验状态，并写入实际文件路径、SHA256 与样本数，不只写自然语言总述。
4. 加两个无 GPU 单测：同序输入通过；同长度但重排输入必须 fail closed。再以当前 26 个文件重跑，验收 `+0.16 [-0.43, 0.76]` 在修复后逐位不变。

建议只增加身份校验，不改变 scorer、bootstrap 种子或统计口径，便于把“安全校验修复”与“数值方法变化”隔离。

## 本轮实际检查与未覆盖项

- `python -m compileall`：`analyze_e100_tail_ci.py`、`analyze_e100_bootstrap_ci.py` 通过。
- `bash -n`：`exp/trace_archives/e100/e100_tail_full.sh` 通过。
- 独立复核：另一审查上下文用脚本真实加载函数与内存 jsonl 重放，确认“FullKV 同长度重排仍通过”并复现两组不同 CI；同时指出 `answers` 非唯一，已据此把当前数据影响收紧为 inconclusive。
- 完整 E100 脚本重跑：因当前环境缺 `jieba` 阻塞；未把该依赖失败算作算法失败。
- GPU kernel/e2e：本轮未运行，当前环境无 PyTorch/CUDA；没有据此声称 kernel/e2e 无缺陷。
- 上轮 `TL-AUD-Q4-FP16-001` 仍是 suspected：缺 PyTorch/CUDA，未获得足以升级或否定它的新证据，本报告不重复展开。

下一轮应轮换检查 kernel benchmark 的 warmup/同步/失败样本落盘，以及 e2e 汇总是否严格绑定实现 SHA、配置和原始文件哈希。
