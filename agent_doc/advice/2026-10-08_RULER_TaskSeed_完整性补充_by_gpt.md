# RULER 汇总补充：任务内 seed 不对称仍会进入正式均值

日期：2026-10-08。状态：CPU 反例已复现；建议补充既有回归，不涉及实现修改或 GPU 实验。

## 与已有审查的区别

固定源码：[`3048c7004764f22cb5838faa0c8dd0fef921f065`](https://github.com/Chosen-David/sglang/tree/3048c7004764f22cb5838faa0c8dd0fef921f065)。

已有 [TL-AUD-MERGE-001](https://github.com/Chosen-David/sglang/blob/3048c7004764f22cb5838faa0c8dd0fef921f065/agent_doc/advice/2026-10-08_2245_twolevel_bug_audit_b028490.md#L71-L81) 的修复建议与回归测试矩阵针对缺任务、缺整侧方法及按位置串组。本补充保留四个 RULER、两个 QA 任务及全部八个 method×seed tag，分组位置也正确；仅删掉一个 task 在 TLI seed 2 下的记录。所有 tag availability 仍为 True，该 task 的两方法也都非空，却比较了不同 seed 集合。

这沿用既有 [T2 配对完整性原则](https://github.com/Chosen-David/sglang/blob/3048c7004764f22cb5838faa0c8dd0fef921f065/agent_doc/advice/sglang_twolevel_audit_by_gpt.md)，新增的是 `merge_ruler_seeds.py` 的具体可执行反例与 task×method×seed 回归，不重复提出通用实验流程。

## 根因与最小复现

[`research/results/merge_ruler_seeds.py`](https://github.com/Chosen-David/sglang/blob/3048c7004764f22cb5838faa0c8dd0fef921f065/research/results/merge_ruler_seeds.py#L29-L65)：L31–34 只检查 tag 是否存在；L39–47 聚合时丢掉 seed 身份，只保留 `(score,n)`；L57–61 只要求两侧非空，分别池化后相减。

原脚本 SHA256：`dfa0cf60116e68379f71eef6794af9359609bf2b386c0e074f9959c0168735df`。下面脚本以冻结的原脚本路径作为唯一参数，只替换两个输入路径，保留全部汇总算法；fixture 分数均为合成数据，n=20。

```python
import hashlib, json, re, sys, tempfile
from pathlib import Path

source = Path(sys.argv[1]).read_text()
assert hashlib.sha256(source.encode()).hexdigest() == 'dfa0cf60116e68379f71eef6794af9359609bf2b386c0e074f9959c0168735df'
def fixture(prefix, tasks):
    return {prefix + '_' + method + suffix:
            {task: {'score': 0.5, 'n': 20} for task in tasks}
            for method in ['triton', 'tli'] for suffix in ['', '_seed2']}
ruler = fixture('ruler', ['r1', 'r2', 'r3', 'r4'])
qa = fixture('qa', ['qa1', 'qa2'])
ruler['ruler_triton']['r1']['score'] = 0.1
ruler['ruler_triton_seed2']['r1']['score'] = 0.9
ruler['ruler_tli']['r1']['score'] = 0.2
del ruler['ruler_tli_seed2']['r1']
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    for name, obj in [('ruler.json', ruler), ('qa.json', qa)]:
        (root / name).write_text(json.dumps(obj))
    for var, name in [('RES', 'ruler.json'), ('RES_QA', 'qa.json')]:
        source, count = re.subn(r'^' + var + r' = .*$',
                               var + ' = ' + repr(str(root / name)),
                               source, count=1, flags=re.M)
        assert count == 1
    exec(compile(source, 'merge_paths_only.py', 'exec'), {'__name__': '__main__'})
```

本轮实际在 Python 3 CPU 环境执行，退出码 0；全部 tag availability 为 True。原脚本输出摘录：

```text
r1                     0.10,0.90        0.20         -0.300  (pooled 0.500 vs 0.200)
RULER 四任务均值 pooled: FullKV 0.500 / TLI 0.425 / gap -0.075
QA 两任务均值 pooled: FullKV 0.500 / TLI 0.500 / gap +0.000
```

r1 的输出确实显示一侧只有一个 seed，但不会触发 incomplete 或阻止普通分组均值。两种方法共同拥有的 seed 1 上，差值为 `0.2−0.1=+0.1`；脚本却将它与 FullKV 两 seed 均值相减，得到 `−0.3`。这里的 +0.1 仅是共同 seed 的 partial 比较，不能替代缺失的双 seed 结果。

## 修复验收建议与影响边界

在已有 TL-AUD-MERGE-001 回归中追加：

1. 保留 task、method、seed 身份，逐任务检查预期 seed 集合。正式双 seed 汇总缺任意必需单元时应失败，并列出缺项；不能仅检查 tag 存在或两方法非空。
2. 探索模式若允许部分比较，显式使用同一 seed 交集，报告 seed 列表及每侧 n。样本 ID/输入身份仍须一致；不得把单 seed partial 标为双 seed n=40。
3. 测试本例；再测试两侧各一个但 seed 不同（不能比较）、完整同 seed（原值不变），以及相同 seed 但样本集合/n 不匹配（拒绝默认配对或采用明确的共同样本协议）。

当前保存的 `tli_ruler_results.json` 与 `tli_ruler_qa_results.json` 共 24 个非 dry 的 task×method×seed 记录均完整，每条 n=20，保存的正确性标记均值与分数一致；12 组方法配对的答案列表及 prompt-token 数顺序也一致。本缺陷未在这些已保存记录中触发，不能据此宣布历史结果被污染。该核对不是完整输入身份验证或原始指标重评分；输出截断及缺少运行配置的问题仍需保留原有限制。本补充没有重跑模型、推断缺失实验分数或验证修复实现。
