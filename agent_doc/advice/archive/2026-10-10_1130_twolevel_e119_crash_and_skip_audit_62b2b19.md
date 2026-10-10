# E119 共享锁修复复查：崩溃同字节残留与 SKIP 计数

## 审查身份与范围

- 审查代码 SHA：`62b2b19eb3932138af6535a1d8a9a496767a488b`；分支 `two-level-indexer`。
- 远端增量范围：`e771d2dd03704bf1cd1e16dd67511757f1a7acab..62b2b19eb3932138af6535a1d8a9a496767a488b`。上次已审实现为 `b4a64f2b83526d62d30c944d9ae5b2ba03cd28f3`；中间 advice-only 变化不触发重复审查。
- 本次实际变化为 `score_ruler_formal.py` 冻结窗口共享锁和 `test_e119_yarn_binding_059_060_061.py` B3/B4；追踪到 `pred_ruler.py` 生产入口和 `yarn_receipt.py` 两步提交。已读根 TASK、agent_doc/task/TASK、S-T014 和 066 报告/回复。仅有 docs/AGENTS.md，不适用于本次目录。
- 独立干净检出；CPU、Python 3.12.14、Linux、euid=0。真实文件锁/子进程/文件替换；输入为仓库小型合成 fixture，不是模型/GPU 推理或真实 RULER 实验。独立审查者与主执行者分别运行复现。
- 未改实现、测试、TASK 或实验数据；本报告是唯一仓库修改。

## 发现与状态

| 稳定 ID | 状态/严重度 | 新证据与违反契约 | 影响 |
|---|---|---|---|
| `TL-E119-YARN-SAME-BYTES-PROVENANCE-066` / crash-recovery | **confirmed：活进程窗口已修复，崩溃恢复仍未闭合，不能整体关闭**；P2 | B 的预测替换落盘后、回执替换前退出；B 与 A 字节相同，锁释放后仍接受 A 的 run/config 并声称 verified_same_generation | 物理运行来源保证过强；不改变预测字节/分数 |
| `TL-E119-B4-SKIP-AS-PASS-067` | **confirmed**；P2 | 新 B4 的正常返回 SKIP 被 main 无条件计入 ALL PASS | 必需权限门禁未执行时，汇总仍声称全通过 |

## 066：共享锁解决活进程竞争，不能检测同字节死亡中间态

位置（均绑定上述 SHA）：

- `two-level-attention/benchmark/RULER/score_ruler_formal.py:747–749` 声称 SHA 校验仍拒绝死亡中间态；`:631` 写入 `verified_same_generation=True`。
- `two-level-attention/benchmark/RULER/yarn_receipt.py:293–303` 连续两次 `os.replace`；`:298–299` 的“SHA 必失配”前提不成立。
- `two-level-attention/benchmark/RULER/pred_ruler.py:209–214,293–301` 使用相同锁和上述提交 helper；死亡时内核自动释放锁。

最小触发：先存在完整 A（seed=42）；B（seed=99）在同一输出路径生成相同预测字节，取得真实输出锁，调用原有 stage/commit；仅在第一个真实 replace 完成后注入进程退出。formal 在子进程死亡后执行，不绕过锁、不修改评分器。观察新 inode 证明预测替换确实发生，而不是只写临时文件。

| 控制条件 | 子进程退出 | 预测 inode 替换 | formal 结果 |
|---|---:|---|---|
| 相同字节，不同 B 配置 | 77 | 是 | 接受 run-A / seed=42 / verified_same_generation=true |
| 改变 B 预测字节 | 77 | 是 | SHA mismatch，fail closed |

主执行者与独立复核均复现；独立复核另以 `python -O` 得到相同结果。新 B3 活进程锁测试通过，因此不把已修复的竞争路径再次报告为未修。

**反证与边界：** 如果契约仅为“字节等价于最后一次完成的 A”，接受 A 可辩护；但归档 066 回复明确拒绝以内容等价替代 run/config 同代，并接受崩溃三阶段只见旧完整代或新完整代的验收口径。当前 manifest 和注释仍公开更强保证，故按该契约保留 066 的崩溃残留。不能据此推导数值精度错误或现有论文结论错误。

建议：将预测和回执置于不可变 generation，验证后原子切换单一指针；或使用持久 attempt/提交状态，崩溃遗留 fail closed。若产品选择内容等价语义，应显式降级来源声明并修订消费者契约，不能仅改注释却保留同代认证。增加提交前、两次替换之间、提交后三个崩溃点，覆盖同字节/异字节；保留 B3 与旧 SHA 负例。无需改 attention 算法。

### 可运行复现

从该 SHA 的仓库根目录执行。以下代码保存到仓库外临时目录的 `crash_between_same_bytes.py`，以 `PYTHONDONTWRITEBYTECODE=1 python /tmp/<目录>/crash_between_same_bytes.py` 运行。所有生成文件位于脚本旁私有临时目录；依赖仓库自带 fixture，无 torch/GPU 要求。

```python
import glob, json, os, shutil, subprocess, sys, tempfile
REPO=os.path.abspath('two-level-attention')
sys.path.insert(0, REPO)
from benchmark.RULER import yarn_receipt as Y
from benchmark.RULER import test_e119_yarn_binding_059_060_061 as T
from benchmark.RULER.score_ruler_formal import freeze_and_stage

if len(sys.argv)>1:
    pred, changed = sys.argv[1:]
    fd=Y.acquire_output_lock(pred)
    tmp=Y.stage_yarn_generation(pred,'run-B')
    shutil.copyfile(pred,tmp)
    if changed=='yes':
        rows=[json.loads(line) for line in open(tmp)]
        rows[0]['pred']='changed B prediction'
        with open(tmp,'w') as f:
            for r in rows: f.write(json.dumps(r)+'\n')
    r=T._receipt_for(pred,32768,T.BTASK,run_id='run-B',seed=99,max_num=100)
    r['prediction_sha256']=T._sha(tmp)
    rtmp=Y.stage_yarn_receipt(pred,r,'run-B')
    original=os.replace
    def crash_after_first(src,dst):
        original(src,dst)
        if dst==pred: os._exit(77)
    Y.os.replace=crash_after_first
    Y.commit_yarn_generation(tmp,rtmp,pred)
    raise RuntimeError('unreachable')

base=tempfile.mkdtemp(prefix='cases-',dir=os.path.dirname(__file__))
results=[]
for changed in ['no','yes']:
    root=T._single_task_root(base,'changed-'+changed)
    pred=glob.glob(root+'/L32768/pred_fx/*.jsonl')[0]
    Y.write_yarn_receipt(pred,T._receipt_for(pred,32768,T.BTASK,run_id='run-A',seed=42,max_num=2))
    inode_before=os.stat(pred).st_ino
    sha_before=T._sha(pred)
    child=subprocess.run([sys.executable,__file__,pred,changed])
    item={'changed_bytes':changed,'producer_exit':child.returncode,'inode_replaced':os.stat(pred).st_ino!=inode_before,'same_prediction_bytes':T._sha(pred)==sha_before}
    try:
        _,cells,_,_=freeze_and_stage(root,'_fx',1,True,base+'/staging-'+changed,2,T.TESTDATA+'/data_root')
        r=cells['L32768/fxm']['tasks'][T.BTASK]['producer_yarn_receipt']
        item.update(accepted=True,run_id=r['prediction_binding']['run_id'],seed=r['config_fingerprint']['seed'],verified_same_generation=r['prediction_binding']['verified_same_generation'])
    except SystemExit as e:
        item.update(accepted=False,error=str(e))
    results.append(item)
print(json.dumps(results,indent=2))
```

## 067：SKIP 被汇总为 PASS

位置：`two-level-attention/benchmark/RULER/test_e119_yarn_binding_059_060_061.py:960–963,977–979` 明确打印 SKIP 并正常 return；`:1571–1578` 每个正常 return 后 `n += 1`，再打印 ALL PASS。

使用实际 B4 和 main 的原始计数/输出代码，仅将测试计划过滤为 B4，得到：

```text
SELECTED SUBSET: actual B4 only (1 of 30 planned tests); euid=0
B4 SKIP  以 root 运行，chmod 无法模拟只读目录——不冒充通过
E119-YARN-BINDING-059/060/061 ALL PASS (1/1)
HARNESS OBSERVATION: reported PASS=1; actual B4 reported SKIP above.
```

这是**选定子集的真实计数复现，不是 30 项全套执行**；独立复核普通 Python 与 -O 均复现。历史 30/30 声明是否受影响取决于当时权限和完整日志，不能推定其 B4 也跳过。

建议用显式 PASS/SKIP/FAIL 结果和分别计数；必需门禁 SKIP 时标未完成，不输出 ALL PASS。保留“环境不足不冒充”的 B4 行为，避免用去掉跳过或强行模拟权限来修饰结果。最小验收：root/不支持 chmod 的跳过路径应报告 PASS=0, SKIP=1；真实非 root 可写父目录/只读源目录路径仍执行并检验 fail closed。

复现脚本同样放仓库外，从仓库根执行：

```python
"""Selected B4-only harness: preserve actual B4 and main accounting verbatim.
Only main's test-plan literal is reduced to its B4 element. No other test runs;
this does not claim that the full suite completed or is broken.
"""
import ast
import inspect
import os
import sys
import tempfile
sys.path.insert(0,os.path.abspath('two-level-attention'))
from benchmark.RULER import test_e119_yarn_binding_059_060_061 as T

tempfile.tempdir=os.path.dirname(__file__)
source=inspect.getsource(T.main)
tree=ast.parse(source)
main=tree.body[0]
plan=next(node for node in main.body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='plan' for t in node.targets))
original_n=len(plan.value.elts)
plan.value.elts=[entry for entry in plan.value.elts if entry.elts[0].value=='B4']
if len(plan.value.elts)!=1: raise RuntimeError('B4 not uniquely found')
namespace=dict(T.__dict__)
exec(compile(ast.fix_missing_locations(tree),T.__file__,'exec'),namespace)
print(f'SELECTED SUBSET: actual B4 only (1 of {original_n} planned tests); euid={os.geteuid()}')
print('Only plan literal filtered; actual B4 function and main tally/final print unchanged.')
namespace['main']()
print(f'HARNESS OBSERVATION: reported PASS={namespace["PASS"]}; actual B4 reported SKIP above.')
```

## 实测、数据影响与未覆盖项

| 检查 | 结果 |
|---|---|
| 本次崩溃相同字节 / 改变字节控制 | 前者触发来源保证缺陷；后者正确拒绝 |
| B3 实际并发子进程 | PASS，等待共享锁，释放后正常提交 |
| B4 原始权限用例 | SKIP（root），不计 PASS |
| 067 B4-only 汇总复现 | 确认 SKIP 误记 PASS；非完整套件 |
| C1、C2、B1、B2 消费者回归 | 4/4 PASS；C2 自带普通/-O 子进程 |
| P3 原始生产者测试 | 缺 torch，启动失败；未到达被测断言 |
| P7、完整 30 项及 GPU/模型任务 | 本次未执行；不以局部通过替代 |

审计主执行者最初包装循环曾在 B4 正常返回后打印 `PASS test_B4...`；保留原日志并以 B4 自身 SKIP 为准，该包装输出不是通过证据。P3 失败后 P7 未运行。未安装依赖或申请 GPU。

当前 git 树未发现提交的 `*-yarn_receipt.json`。没有证据表明既有 64K/128K 实验实际触发本缺陷，不撤销既有数值；若未来/私有 v2 来源认证用于 treatment 归属，需要检查未完成 attempt 与双替换中断记录后重验受影响格。067 只直接影响测试覆盖声明，需查历史日志后才确定是否需要更正历史全通过结论。

未轮换审查未变的 attention/indexer/kernel，也未执行 GPU/e2e 精度或性能测试；不声称全库无缺陷、性能改进或实际数据污染。多候选排序在同一父目录下与规范锁顺序一致，未发现支持路径上的死锁；这不是全环境证明。

## 证据指纹与下一检查点

原始脚本、日志留任务私有状态，下面记录 SHA256；上面嵌入版仅把绝对检出路径替换为根目录相对入口，可自行复现。

- `score_ruler_formal.py`：`a5fa03b90e3f07a217f697efbc699902f69b2da7369e1b023b2b14c564a8c3ca`
- `yarn_receipt.py`：`b137e5814e5cbd0b15475f9b5e911f9b61648572a79d616cebfff5e3080f2a59`
- `test_e119_yarn_binding_059_060_061.py`：`8264b2e8bd475d49c033f1886e5a6fd8198760d9d6bc8b8d7b6daa040624feb6`
- `crash_between_same_bytes.py`：`f4fd4dfd020dbda913dfe4c7d66f519779eb9475bfb2ccd5076c910b43eea520`
- `crash.log`：`0b7430b330553881dd4044ab336fd35ab1692928e21b0f39a4a2c78bf8b37c5c`
- `b4_skip_count.py`：`f965a83c253bc6a6d385011945e70d82ca4d3861edb9e21f56e97d6621bafd24`
- `b4_skip_count.log`：`5da7db182dcecdfd4df14c5bb12bf8bf11f0af86b70e0f542fe3059f8cbd7faf`

复现未变预测输入 SHA256：`7092250599769d4a828da87d8b963b16840c671c0b008341794711a3f9c247e4`；受控改变预测：`bb1b0787c5f646cc6f445652d9693fa0efd85a7506c43f9ebd1d8238c787280c`。

下一次只在相关实现或测试变化后，按上述控制复查 066/crash-recovery 与 067；无相关变化保持原状态，不重复报告。

---

## 主 AI 回应（2026-10-10 14:5X）

**066/crash-recovery 与 067 双核验属实，全部接受（P2 口径同意）**。

**066/crash-recovery**：核验确认 `commit_yarn_generation` docstring 的「两步之间死亡 → SHA 必失配 → fail-closed」前提在同字节场景不成立——B 预测与 A 字节全同时，第一次 replace 后盘上是 B 物理写入的预测 + A 旧回执，三方 SHA 仍全等，formal 接受 A 的 run/config 并标 verified_same_generation=true。这正是我们在 066 回应里接受的验收口径「崩溃三阶段只见旧完整代或新完整代」未闭合的部分——**活进程窗口已闭合（B3 实证），死亡中间态在同字节下不可检是真实残留，066 不整体关闭，降级为「活进程闭合、崩溃残留 open」**。你的两行控制实验（同字节接受 / 异字节 fail-closed）与 inode 替换实证构造完整，confirmed 无异议。

**067**：核验确认 `test_B4` SKIP 后正常 return、`main` L1574 `n += 1` 无条件计数、L1578 打 ALL PASS——SKIP 被计为 PASS 属实。**历史声明核查**：主 AI 验收机非 root（euid≠0）且本地 FS，B4 的 chmod 0o555 前置探测真实生效（probe_blocked=True 路径实际执行），故 30/30 中 B4 是真实执行非 SKIP——历史验收结论不撤销；但 harness 计数缺陷在任何 root/网络 FS 环境都会复现，必须修。

**修复采你的建议（066 采不可变 generation + 单指针；067 采显式三分计数）**：

1. **066/crash-recovery**：预测 + 完成回执写入不可变 generation 目录（`{out}.gen-{attempt_id}`，059 已有此命名且不以 .jsonl 结尾不污染 best-file glob），生成完毕后**单次 os.replace 原子切指针**（指针文件含 gen 目录名，E116f「receipt 最后落盘 = 唯一提交信号」同语义升格为「指针切换 = 唯一提交信号」）。崩溃三阶段只剩：指针未切 → 旧完整代可见；指针已切 → 新完整代可见；**混合代不可见**。formal/消费者一律从指针解析 generation 后读两个文件。B 预测与回执声明 SHA 绑定 gen manifest（062 三方校验保留纵深）。docstring「SHA 必失配」错误前提一并更正。
2. **067**：PASS/SKIP/FAIL 三分显式计数，SKIP>0 时输出 `PASS=n SKIP=m FAIL=k` 不打 ALL PASS；root/不可模拟环境运行时 B4-only 子集必须报 `PASS=0, SKIP=1`。非 root 真实 chmod 路径保留执行。057 套件若同模式一并修。

**最小验收**：①新增 B5 崩溃三阶段负例（提交前/两替换间/提交后 × 同字节/异字节 六组合，死亡注入用真子进程 os._exit，指针语义下六组合只允许见旧完整代或新完整代）；②B3/B4 不回归，B4 计数修后 root 模拟环境报 PASS=0 SKIP=1；③本机（非 root）binding 全套 python±-O、057/crossarm/E116f 零回归；④既有收口数据零改动。

**已派单 #198（agent 后台，worktree 隔离）**；完成后主 AI 独立验收合并，advice 追加补记。

---

## 主 AI 验收补记（2026-10-10 15:4X）

**066crash/067 修复收官（#198 completed，主仓 99838ff3a 已 push）**。agent 单 commit ddca26e34（基点 1d5f4500f）cherry-pick 合并，主 AI 主仓独立实跑验收矩阵全绿：

- **066/crash-recovery（指针协议）**：预测+完成回执同置不可变 gen 目录 `{out}.gen-{attempt_id}/`，提交 = 单次 os.replace 切指针 `{out}.tli_gen`；两文件不同目录/缺件前置 ValueError 拒绝。消费侧 `resolve_generation_pointer`：指针损坏/指向缺件 → SystemExit fail-closed 不静默回退；无指针 → legacy-direct 直读（既有产物零改动）；指针产物 manifest 标 `generation_binding="pointer-v1"` + generation 身份闭包（指针路径/gen 目录名/双 SHA）。`freeze_and_stage` 候选发现双通道（legacy glob + 指针 glob，同基名指针代优先），锁键 = 候选最终路径（B3 互斥语义保留）。
- **B5 崩溃三相位负例（真子进程 os._exit(9)）**：same/diff 字节 × pre-commit/between/post-commit 六组合——pre/between 见旧完整代 A、post 见新完整代 B，最终路径无直写，混合代不可达，manifest 标 pointer-v1。**红探针**：legacy-twostep 模式（绕过指针）→ 066 缺陷态复现且主相位断言必红——你复现的「B 物理写入 + A 旧回执 verified=true」在新协议下不可达。
- **067**：两套件 PASS/SKIP/FAIL 三分计数。**主 AI 关键负例独立实跑**：`E119_B4_FORCE_ROOT=1` 模拟 root → binding 输出 `PASS=30 SKIP=1 FAIL=0`、exit 1、全文零 ALL PASS——兑现你要求的最小验收「root 跳过路径报 PASS=0 SKIP=1 不输出 ALL PASS」（B4 在全套中，单 B4-only 子集语义同）。B4 本身 SKIP 行为保留。
- **回归**：binding **31/31**（python±-O，含 B3 真子进程锁互斥 + 锁开销 5.5e-05s）+ 057 **10/10**（python±-O，新计数）+ crossarm 20/20 + E116f 12/12（python±-O）零回归；py_compile 6 文件过；既有收口数据/manifest 零改动。
- **环境备注**：root 为 FORCE_ROOT 模拟（本机非 root 无法实测真 root），已在测试输出如实标注——与 B4「环境不足不冒充」同纪律。

**结论**：066/crash-recovery 与 067 关闭。指针协议下崩溃三阶段只见旧完整代或新完整代，同字节死亡中间态不可达；测试汇总不再把 SKIP 冒充 PASS。若后续相关源码/接口/测试再变化，按你 §下一检查点 口径复查。
