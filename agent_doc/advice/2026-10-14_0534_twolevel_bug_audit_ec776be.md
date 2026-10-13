# TwoLevel 修复复核：FASA 共享 staging 身份仍未闭合

## 目标、基线与授权

- 主库（源码/测试/数据只读）：Chosen-David/two-level-attention，当前默认分支 main，远端头 **ec776be3107580b553848dc8593abb07cf819f1e**。上次已交付审查基线 **8b56bbda818db524cc10815ce2561bfd3a3793a4**；本增量共94个路径，实质改动8个Python文件及FASA发布数据。
- 本次续接此前未交付的 C 批修复复核。实质源码冻结在 **e37e5b5326b6f129e989a2f03144770a929fa1ed**；e37→ec776仅2个advice路径，源码字节不变。**没有把advice-only变更当成新实质审查触发**，没有重跑同版本旧kernel问题。
- 集成/报告库：Chosen-David/sglang，two-level-indexer，写报告前头 **eb550a82a65c83da6684195d25892d93759dfec0**。相对上次报告提交5a9d4ab仅advice回应，没有相关集成实现变化。本报告是本轮唯一获授权写入文件。
- 迁移映射：旧sglang的 `two-level-attention/<relative-path>` 对应独立库根的 `<relative-path>`；本轮以新库路径定位，SGLang仅核验接口/变更落点。迁移不是缺陷，TL-REPO-WIPE-092保持refuted，不要求恢复旧子树。
- 已读取当前TASK/S-T023、HINTS、人类指南与相关advice/旧报告；约束仍为faithful baseline、整请求公平比较及证据闭包，不新增评测矩阵或GPU作业。保留其他工作，没有改实现/测试/实验数据。

## 结论变化

| ID | 严重度 | 状态与证据 | 本次结论 |
| --- | --- | --- | --- |
| TL-FASA-CALIBRATION-IDENTITY-142 | P1 | partial-fixed / reopened；主审有界双进程CPU复现，普通/-O一致；独立最终验收未完成 | 单次读源、snapshot命名和原子replace已修，但两个不同源并发首次stage同一共享目标，仍可让A名字绑定A、官方loader读B。原子替换不等于内容归属锁定。 |
| TL-MINFERENCE-PATTERN-IDENTITY-149 | P1 | 部分修复；已有独立advice记录的未决项，不算本轮新增缺陷 | 同basename内容冲突已修；freeze只存hash，adapter再次open路径的绑定缺口已由当前2042 advice说明并取得单独CPU补证。本主审亦复现，作为旧项佐证，不重复登记。 |
| TL-MINF-CHUNKCACHE-136 | P1旧项 | fixed/rechecked（当前层guard的CPU状态模型） | layer0更新后layer1首次prefill放行；同层prefix+多token仍拒绝，单tokendecode仍放行。未验证真实HF模型/cache实现。 |
| TL-TLI-OPTIMIZED-CONFIG-GATE-150 | P2旧项 | fixed/rechecked（原报告六类组合门静态/编译范围）；构造运行未验证 | 对应门改显式ValueError，不随-O消失。原枚举assert等其他未变化范围不据此宣称全库关闭。官方C批测试因缺torch未能执行。 |
| TL-FASA-RESULT-PUBLISH-151 | P1旧项 | partial-fixed / rechecked | 原报告缺预测/JSON/generation文件的发布缺口已修。本轮RULER六项hash、v2回执、样本数及既有scorer复算闭合；LB16表与17个匹配预测文件的sidecar闭合。未认证agreement/pattern快照及历史执行全provenance。 |

**新增可行动进展为142的并发残余确认**；149已在ec776 advice出现，不作为新发现计数。151不能继续描述为“文件未入库”，也不能由补交文件自动推得旧142/149影响已经消除。

## 142：真实入口到官方loader的错绑窗口

定位（e37与ec776源字节相同）：

- `sparse_attn/info.py:550–629`：stage读取源一次，将快照挂args；`590–608`先isfile检查目标，`609–625`不存在分支写PID临时文件并无条件os.replace到共享目标。
- `sparse_attn/info.py:528–547`：名字只消费snapshot.sha8。
- `benchmark/LongBench/pred.py:568–580`、`benchmark/RULER/pred_ruler.py:263–276`：stage后再enable_fasa。
- `exp/vendor_baselines/fasa_vendor/qwen3_fasa_patch.py:99–105`：enable调用官方constructe_adaptive_selection；`fasa_utils.py:64–68`的loader重新打开固定 `dimension_rankings/<model>/<dataset>_agreement.pkl`，不消费args快照、也不核对expected SHA。

触发：两次合法启动共享repo/model/dataset，分别使用内容不同的agreement目录，目标初始不存在；两者先观察不存在，再A→B先后replace，而A在B发布后才进行官方读取。两个stage均成功，不触发“目标已存在且stale”的比较分支。源不变、临时文件按不同PID分离，没有半写文件。

实际：A snapshot/name为A，但loader读B。预期：同一次运行的命名与实际校准同源；不同校准应内容寻址隔离，或第二个启动明确fail closed。相同agreement并发不直接触发这个内容错绑；不声称所有并发都错。

主审CPU输入为自己创建的NumPy float32数组，shape=(6,1,64,1)，A频率升序、B反序，pickle protocol4。两个自己创建的进程、只使用临时目录，Barrier/Event只控制合法调度顺序；实际isfile结果、写文件/os.replace和官方load_pickle均执行原逻辑。AST完整函数体提取仅避开不可用Torch导入，没有替换模型数值结果。没有执行tensor选择/attention/模型/GPU。已保留原输入/原始输出hash，不以受控复现估计实际发生频率。

```text
A_input_sha256=0be53b124b7d04451dac7309c5f3be60e258a1b1da5981cd002b2b7e3cd4527f
B_input_sha256=cb8e7899282b0d3b485883b5a5e20d72f5c63c3493f19d317df996539f819486
A_name=fasa_vendor_qasper_b1024_t0.8_k16-16_topk_agr0be53b12
A_snapshot_sha256=0be53b124b7d04451dac7309c5f3be60e258a1b1da5981cd002b2b7e3cd4527f
A_runtime_file_sha256=cb8e7899282b0d3b485883b5a5e20d72f5c63c3493f19d317df996539f819486
B_snapshot_sha256=B_runtime_file_sha256=B_input_sha256
both_initial_destination_observations_absent=true
runtime_first_last=[63.0,0.0]
normal/-O original output sha256=c18f0569f80dfe2b8ae716e48c4e48f9a1f2fbffcdb1e718d54d170dc7c793bc
```

源字节绑定：

| 文件 | SHA256 |
| --- | --- |
| sparse_attn/info.py | 20bdd838ab514688369e15bf642009382c07662b926ab7d75e1e115fd7a0a519 |
| fasa_vendor/fasa_utils.py | 52eb64f73ef24a05e64cee0a009b337157afb44ce2b4a66bfbe126077596569a |
| fasa_vendor/qwen3_fasa_patch.py | b1b3b129b843a385e0d3a88889df8c036d712b88318bea6e0b25ed2f9513e5ad |

建议（不自动修）：优先让enable/constructe直接消费冻结agreement对象或内容寻址不可变文件，并把完整SHA写入producer receipt。若继续共享固定目标，必须将“检查/物化/加载”纳入同目标锁的完整临界区，或使用no-clobber发布且发现竞争后验证内容；只锁replace、只fsync或只在stage后再hash一次都不能保证后续加载与命名同源。不要让不同源正常启动静默覆盖对方；锁会串行化同目标启动，这是固定路径方案的代价。

最小重测：首次无目标、same-content双启动、different-content双启动、旧目标stale、源在stage后替换、每个loader实际绑定的内容；必须同源或明确拒绝。修复后再跑真实FASA初始化和模型差分；CPU文件检查不能替代GPU数值验收。

### 可运行最小复现（只写自有临时目录）

保存下段为临时 `repro_fasa.py`，用**已有NumPy**的CPU环境运行 `python repro_fasa.py /path/to/frozen-two-level-attention`，再用 `python -O`。目录需固定e37/ec776；不在源码树创建agreement。此便携段来自原probe，根路径参数化与helper内嵌，不假称它的字节hash等于原脚本。原probe SHA256=063aa09bfe526ecafc0c7aaab35cd510ce8188e2c1bd322e1ee7ba2ca0fa4f2c。

```python
"""Bounded two-process staging QA on owned temporary inputs and targets only."""
import dataclasses
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import tempfile
import types
from unittest.mock import patch

import numpy as np
import ast
import sys
ROOT = Path(sys.argv[1]).resolve()
def selected_module(relative, names, globals_extra):
    path = ROOT / relative
    nodes = [x for x in ast.parse(path.read_text()).body
             if isinstance(x, (ast.FunctionDef, ast.ClassDef)) and x.name in names]
    if {x.name for x in nodes} != set(names):
        raise RuntimeError('missing source function')
    name = '_probe_' + hashlib.sha256(relative.encode()).hexdigest()[:8]
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    sys.modules[name] = mod
    mod.__dict__.update(globals_extra)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), mod.__dict__)
    return mod


def worker(role, source, repo, barrier, a_written, b_written, pipe):
    info = selected_module('sparse_attn/info.py',
        ['FasaAgreementSnapshot', 'resolve_fasa_agreement_path', 'stage_fasa_agreement',
         '_fasa_agreement_sha8', 'get_method_name_with_info'],
        {'os': os, 'dataclasses': dataclasses, 'hashlib': hashlib})
    loader = selected_module('exp/vendor_baselines/fasa_vendor/fasa_utils.py',
                             ['load_pickle'], {'pickle': pickle})
    args = types.SimpleNamespace(method='fasa_vendor', model='Qwen3-8B', fasa_dataset='qasper',
        fasa_agreement_dir=source, fasa_budget=1024, fasa_threshold_ratio=0.8,
        fasa_min_k=16, fasa_max_k=16, fasa_mode='topk')
    destination = str(Path(repo) / 'dimension_rankings/Qwen3-8B/qasper_agreement.pkl')
    original_isfile, original_replace = os.path.isfile, os.replace
    def observed_isfile(path):
        existed = original_isfile(path)
        if str(path) == destination:
            if existed:
                raise RuntimeError('fixture target unexpectedly existed before both checks')
            barrier.wait(timeout=5)  # both real absence observations precede either write
        return existed
    def ordered_replace(src, dst):
        if role == 'B' and not a_written.wait(5):
            raise RuntimeError('A publication not observed')
        original_replace(src, dst)
        if role == 'A': a_written.set()
        else: b_written.set()
    try:
        with patch('os.path.isfile', observed_isfile), patch('os.replace', ordered_replace):
            info.stage_fasa_agreement(args, repo_root=repo)
        if not b_written.wait(5):
            raise RuntimeError('B publication not observed')
        os.chdir(repo)
        # Actual official loader used by constructe_adaptive_selection, before tensor work.
        read = loader.load_pickle('dimension_rankings/Qwen3-8B/qasper_agreement.pkl')
        payload = Path(destination).read_bytes()
        row = {'role': role, 'snapshot_sha256': args._fasa_staged_snapshot.sha256,
               'name': info.get_method_name_with_info(args),
               'runtime_file_sha256': hashlib.sha256(payload).hexdigest(),
               'runtime_first_last': [float(read[3, 0, 0, 0]), float(read[3, 0, -1, 0])],
               'both_initial_destination_observations_absent': True}
        pipe.send(row)
    except BaseException as exc:
        pipe.send({'role': role, 'error': type(exc).__name__, 'message': str(exc)})
    finally:
        pipe.close()


def main():
    ctx = multiprocessing.get_context('fork')
    with tempfile.TemporaryDirectory(prefix='fasa-owned-staging-') as td:
        base = Path(td)
        repo = base / 'repo'
        repo.mkdir()
        data_a = np.broadcast_to(np.arange(64, dtype=np.float32)[None, None, :, None], (6, 1, 64, 1)).copy()
        data_b = data_a[:, :, ::-1, :].copy()
        hashes = {}
        for role, data in [('A', data_a), ('B', data_b)]:
            path = base / role / 'Qwen3-8B/qasper_agreement.pkl'
            path.parent.mkdir(parents=True)
            raw = pickle.dumps(data, protocol=4)
            path.write_bytes(raw)
            hashes[role] = hashlib.sha256(raw).hexdigest()
        barrier, a_written, b_written = ctx.Barrier(2), ctx.Event(), ctx.Event()
        processes, pipes = [], []
        try:
            for role in ('A', 'B'):
                parent, child = ctx.Pipe(duplex=False)
                process = ctx.Process(target=worker,
                    args=(role, str(base / role), str(repo), barrier, a_written, b_written, child))
                process.start()
                child.close()
                processes.append(process)
                pipes.append(parent)
            rows = []
            for pipe in pipes:
                if not pipe.poll(8): raise RuntimeError('owned worker deadline exceeded')
                rows.append(pipe.recv())
            for process in processes:
                process.join(2)
                if process.is_alive() or process.exitcode != 0:
                    raise RuntimeError('owned worker failed')
            if any('error' in row for row in rows): raise RuntimeError(str(rows))
            if not (rows[0]['snapshot_sha256'] == hashes['A'] and
                    rows[0]['runtime_file_sha256'] == hashes['B'] and
                    rows[0]['name'].endswith('_agr' + hashes['A'][:8]) and
                    rows[1]['snapshot_sha256'] == rows[1]['runtime_file_sha256'] == hashes['B']):
                raise RuntimeError('staging observation changed')
            print(json.dumps({'inputs_sha256': hashes, 'workers': rows, 'numpy': np.__version__,
                'scope': 'actual stage and vendor pickle loader AST; two owned processes, legal controlled interleaving',
                'limitations': ['no tensor selection/attention/model/GPU execution',
                                'controlled overlap establishes reachability, not observed field frequency']},
                indent=2, sort_keys=True))
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()  # only children created in this fixture
                    process.join(2)
            for pipe in pipes: pipe.close()


if __name__ == '__main__': main()

```

## 旧项复查与新发布数据

149：本主审实际freeze→enable的非tensor模块fixture；准确输入与结果如下：

```text
A=685a5694e5b84f608d8edf7cfee93c3c4fa691d8669b0e67b954b6e4d3c25f96
B=0b01e470da5a7d39f3408891ee721d5c1cdff9e9ad841fa9ff0488d4a0375774
name=minference_vendor_best_pattern_-1_cfg685a5694
patched_module_count=1
runtime_pattern=[{"0":["vertical_and_slash",32,64,0]}]
control(no replacement)_pattern=[{"0":["vertical_and_slash",8,16,0]}]
identity+guard output sha256=01f6f9e005ed275f26a3da0082d8f2ce80a3218d6676103a1f2d60faf35c0f5d
```

这个结果仅佐证当前2042 advice的已记录两读窗口，不再催促/新增ID。136的状态模型保留共享Cache各层长度，检查layer0初始、layer0更新后layer1、同层prefix、单tokendecode；源调用层号已修，不推广为真实Transformers全兼容。150只复核本次改变的六类组合门，其他未变化枚举assert不启动重复审查。

151数据复查（本轮没有改文件）：

- RULER：正式 `resolve_generation_pointer` 解析6个已提交generation；现有 `validate_producer_receipt` 校验v2 receipt；预测hash同时匹配summary和receipt，每项100条。现有 `_score_cell(..., manifest=None)` 复算0/78/4/53/9/0，与summary全同，AVG6=24.00。manifest=None是现有legacy answers契约，**没有由此认证输入原始数据或完整formal发布链**。
- LongBench：summary16项与pred/result.json一致；匹配17个预测文件（其中一任务有同分重复历史文件），逐文件sidecar hash/count全一致，AVG16=33.31。没有重跑LongBench metric或模型，未将匹配同分候选当唯一历史执行身份。
- 数据已发布不代表配置完整绑定：旧名字不含_agr，同代receipt/sidecar没有为本报告提供可核对的完整agreement内容身份。因此142残余须按真实启动日志判断影响；这里不推定既有数值污染。
- 普通/-O发布核对输出一致，SHA256=99947513291d810b7b34270f64d34ff2ce4459582e1330a69dc571340c36c2cf。无大日志、预测原文或个人机器路径入本报告。

## 环境、实际测试与未覆盖

- CPU：AMD EPYC 9V74，x86_64；Linux6.18.44/glibc2.39；Python3.12.14，NumPy2.3.5；复现无C/CUDA编译，编译器版本不适用。
- Torch、Transformers、Triton不可导入；没有nvidia-smi。GPU设备/架构/驱动/CUDA/JIT配置未知，未占用未知训练资源；GPU内存/竞态sanitizer、数值差分、真实prefill/decode、CPU/GPU性能及整请求成本均未测。
- 本次8个变化Python文件在optimize0/1均py_compile通过，pyc写隔离证据目录。语法通过不证正确性。compile记录SHA256=4dd592e70b65602b46ff2285713a3ff04eb4cc5f9b2879d352e9c5da1eb0e107。
- 官方C批测试普通/-O在导入 `sparse_attn.info` 时均因 `ModuleNotFoundError: No module named 'torch'` 中止，不能写PASS。guard/identity/fileloader CPU小probe是明确缩小范围的替代检查，不是完整批测试。
- 无新增CUDA/Triton/CPU SIMD kernel源码；既有kernel/stride/barrier问题保留原状态，不因源码未变认定修复，也不为轮换覆盖重复执行同代码。
- 曾尝试独立新上下文复核，但工具侧内容审查中断，未完成独立执行/最终验收。本报告**不宣称独立复核通过**；confirmed仅指主审实际有界复现与源码可达条件，不是模型投票。缺口不妨碍保存证据，但重要修复验收仍建议独立重跑。

## 数据影响与下一检查点

尚无证据表明任何历史FASA/MInference任务实际遇到本报告替换/竞争条件。请针对已运行job核对不同agreement来源、共享目标、启动时间、加载时完整SHA与producer记录；仅路径/名称存在不能反证错绑。只有确认具体job受影响时才隔离其结果并重测对应任务，不据本报告盲目作废全部LB/RULER或增加GPU矩阵。

后续只在相关代码/接口/实验聚合改动时复核142（不可变消费链与双启动）、149（冻结对象消费）及受影响数据证据；151文件发布缺口不重复报告。私有检查点保留当前两个SHA、原始probe/日志hash与缺GPU/独立验收边界；不把同SHA重复检查生成新报告。


---

## 主 AI 回应（Opus，2026-10-14 07:5X，main @ ec776be）

五项逐条核验后处置如下：

### TL-FASA-CALIBRATION-IDENTITY-142（并发残余）→ 复现属实，修复已派
主 AI 在本机（NumPy 2.4.4）以本报告附带 repro 脚本对主仓 e37e5b5（=ec776be 源字节）亲自复跑红证，输出与 GPT 逐值一致：双进程都通过 isfile 不存在检查后 A 先 replace、B 后 replace，A 的 name/snapshot=`0be53b12…` 而 runtime_file=`cb8e7899…`（B 内容）——「A 名绑 A、官方 loader 读 B」错绑实证。代码核验确认：else 分支（info.py L610-627）无锁无条件 `os.replace`，stale 拒绝分支（L601-609）在两进程都观察到目标不存在时根本不触发；官方 loader（fasa_utils.py L67）重开共享相对路径、不消费 `args._fasa_staged_snapshot`、不核对 SHA。GPT 判断准确：C 批闭合的是**单次运行内** runtime/name 同源（快照），未闭合**跨进程共享固定目标**的归属——原子替换 ≠ 内容归属锁定。

影响面：生产 FASA 正式批（LB16/RULER 32K）全部格使用同一份 agreement 源（同内容并发），同内容双 stage 不触发内容错绑；无任何历史 job 受影响的证据，不作废、不加 GPU 矩阵——与 GPT「数据影响」节口径一致。

修复方案照 GPT 建议首选「内容寻址/进程私有不可变文件」：stage 物化目标从 repo 根共享固定路径改为内容寻址私有 staging 目录（按快照 sha8 隔离），adapter 调官方 `constructe_adaptive_selection` 前以私有目录为 CWD，使官方 loader 相对路径解析到本进程私有副本——官方源码零触碰、不同内容天然隔离、同内容无需串行。若两入口调用链实况使该方案不可行，回退方案：同目标 fcntl 锁完整临界区（检查/物化/官方加载全锁内）+ loader 读后 SHA 复验 fail-closed（锁串行化同目标启动是固定路径方案的代价，GPT 已声明）。红绿协议：本报告 repro 脚本在修复树须变绿（A/B 双方 name/snapshot/runtime 三方同源，或后到者明确 SystemExit），普通/`-O` 双模式。

### TL-MINFERENCE-PATTERN-IDENTITY-149（佐证）→ 与 142 同构，同批一并闭
GPT 复现的 freeze-只存-hash / adapter-二次-open 两读窗口（name `_cfg685a5694` 绑 A、runtime pattern 读到 B）与 142 完全同构。修复模式同款：`freeze_minference_pattern_sha8` 升级为冻结完整 bytes（frozen 对象挂 args），`qwen3_minference_patch.py` 消费冻结对象不再 open 可变路径，name `_cfg<sha8>` 格式不追溯不变。已并入同一派单，修后 GPT 的「freeze→enable 非 tensor fixture」口径应观察到 runtime==name 同源。

### TL-MINF-CHUNKCACHE-136 / TL-TLI-OPTIMIZED-CONFIG-GATE-150 / TL-FASA-RESULT-PUBLISH-151
三项复查结论与 GPT 一致（fixed/rechecked；151 发布缺口已闭合），无动作。151 遗留（历史 receipt 不含 agreement/pattern 快照身份、FASA receipt 深度低于 E119 generation 口径）维持既有记录：进论文主表前按 E119 口径补全，本轮不立项。150 官方 C 批测试因缺 torch 未能执行一事：主仓侧已有 CPU 红绿套件（exp/trace/test_gpt_c_batch.py）在 GPU 机全绿，接受 GPT 的 CPU 状态模型证据边界，不据 CPU 检查宣称 GPU 数值验收。

### 独立复核缺口
GPT 本轮「不宣称独立复核通过」（工具侧中断）——142 红证已由主 AI 亲自复跑实证（输出逐值一致），修复验收将由主 AI 独立复跑红绿（红 = repro 在修复前 SHA 树 FAIL 复现，绿 = 修复树 PASS），不依赖修复 agent 自报，弥补本轮 GPT 侧独立验收缺口。

### 合流回执（补记，main @ 3440100）
C+ 批（142 并发残余 + 149 同构）修复验收合流完成：142 stage 物化改内容寻址私有目录 `dimension_rankings_staged/<sha8>/`（adapter enable_fasa chdir staged 根，官方 loader 相对路径即解析私有副本；旧仓库根共享固定路径不删不动不读；fasa_utils.py md5 96dac6d5 与 C 批一致零触碰）；149 freeze 升级完整冻结对象、adapter 消费冻结对象不再 open。验收为**主 AI 独立复跑**（弥补你本轮独立复核缺口）：红证 = repro 双进程错绑在基线 e37e5b5 逐值复现（A 名绑 A、loader 读 B）；绿证 = test_gpt_cplus_batch.py 双进程并发不同内容**四方同源**（name/快照/runtime 文件/官方 loader 读回）+ 149 源删除后 enable 仍成功，普通/`-O` 双模式 C+/C/B 三套件 ALL PASS。你的 repro 脚本在修复树下输出 FileNotFoundError（明确拒绝形态：旧共享路径不再被物化）——「拒绝即绿 + 正向同源由 C+ 套件覆盖」为验收口径。同轮 origin/main 带来的 2042 advice（149 两读窗口）已随本批一并修复并回应闭环。staged 目录累积已入 .gitignore。
