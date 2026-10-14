# TwoLevel 增量审查：内容寻址与测试状态

## 目标、权限与基线

- 主库 `Chosen-David/two-level-attention` 默认分支main：`404feb17cf14d536dec943aa433d6d21d610d172`，上次已审查`ec776be3107580b553848dc8593abb07cf819f1e`。
- 集成库 `Chosen-David/sglang` two-level-indexer：`eb55f6fba3f381b121955e9e3437026944e2f752`，相比上次`5bc2e74e6848f20b4a541d4282e234e8f74281a2`仅既有报告补记，无集成源码变化。
- 主库9个路径变化（其中advice仅回应）；审查7个Python文件及.gitignore、入口→stage/freeze→adapter→vendor loader→回归成功判定；代码迁移/同字节搬迁不触发本轮问题。跨仓映射仍为旧`sglang/two-level-attention/`→新独立库根，SGLang无新接口差异。
- 主库实现/测试/数据完全只读；本次唯一交付为SGLang `agent_doc/advice/` 此Markdown。未修改guide/旧报告、未推主库main。私有检查点与有界CPU证据在任务隔离目录。

## 发现表

| ID | 状态 | 级别 | 定位（主库404feb1） | 违反契约及触发 |
|---|---|---|---|---|
| TL-FASA-CALIBRATION-IDENTITY-142 | reopened / confirmed（字节生命周期），数值影响inconclusive | P2条件残余；原普通并发P1已缓解 | `sparse_attn/info.py:614–654`；消费`exp/vendor_baselines/fasa_vendor/qwen3_fasa_patch.py:116–124`；LB `pred.py:575`、RULER `pred_ruler.py:273` | 新目录仅SHA256前8hex。真实前缀碰撞、同model/dataset、同时首次stage均通过isfile=False时，A原子写后被B替换；A快照指A，官方loader路径现为B。不同内容并非“天然隔离/错绑结构性不可能”。 |
| TL-CPLUS-SKIP-SUCCESS-152 | confirmed（原控制流CPU复现） | P2验收口径 | `exp/trace/test_gpt_cplus_batch.py:333–340,397–404` | 149 adapter依赖导入异常打印SKIP并return，main不汇总跳过状态，仍返回0、打印142/149组合RESULT: ALL PASS；缺项被组合成功声明覆盖。 |

## 142：实证、反证与修复建议

输入是合法标准库pickle字节生命周期fixture（dict），**不是合法FASA张量矩阵**。CPU从实际固定源码AST提取stage及依赖，不执行torch模块或GPU；真实hash无mock，50,126个有界候选找到：

```
A = pickle.dumps({'calibration_fixture': 6832}, protocol=4)
B = pickle.dumps({'calibration_fixture': 50125}, protocol=4)
A sha256 = 5a88577ae9ed21bfdd8dbe3f5712ec2ea7d24d5f91eac0b8c23b6d2e810dae17
B sha256 = 5a88577a70b00e564ff845a50ac700cbe1ba9a6b6de3251bd5a6b1208797c892
shared directory = dimension_rankings_staged/5a88577a/
```

两个受控子进程只在os.replace前放置调度barrier，确保双方已通过不存在分支；不改变hash、isfile或源内容。A先replace，B后replace，A在B完成后读取自己snapshot.dst_path：实际hash=B、source/hash=A、matched=false，B matched=true。普通Python与`-O`一致；独立上下文另行复跑确认。此证据确认staging接口字节归属缺陷，不冒充真实校准矩阵/模型输出实测。

反证：两个普通不同前缀输入均matched=true；**顺序**真实碰撞在第二次stage被全量SHA拒绝（独立SystemExit含142），不报顺序静默错误；同内容幂等和既有stale拒绝CPU控制通过。条件很窄，不将普通并发修复判作完全无效；不以32bit摘要碰撞本身推断既有作业发生过该事件。

建议至少用完整SHA256作为staged目录，保留全量校验，并考虑独占创建/写后复核；短sha8只作显示标识，不能单独承担文件隔离保证。内存参考仅把目录段改完整SHA，同一对碰撞输入2/2字节归属正确（非源码修复、非模型测试）。重测不同内容/同内容/同前缀不同全hash的顺序与受控并发、损坏目标、失败清理、两入口真实loader和合法校准矩阵；如发现实际碰撞作业，仅重跑对应校准身份及其受影响预测，不要求无证据的全量历史重跑。

## 152：实证、反证与修复建议

原test149实际依赖导入失败时，其except返回普通None，main无条件输出成功。本环境裸运行整套件**先在顶层import torch失败**，因此绝不称整套原生套件跑过或通过。

生产者额外CPU源执行只省略info模块未被这些测试使用的torch import及相应suite import，四项142实际函数执行通过；149保留真实依赖导入，得到：

```
[CPLUS-149] SKIP: minference_vendor 依赖不可导入（No module named 'torch'）
GPT-CPLUS-BATCH(142-concurrent/149-adapter) RESULT: ALL PASS
main_return 0 optimized 0
# -O 同型，optimized 1
```

独立上下文以原test149+main AST、四个142 no-op专门隔离组合成功分支，也得到SKIP+ALL PASS+0；这是控制流复现，不是完整套件实跑。可达真实环境还包括torch可用但transformers/vendor/.so缺失或导入报错。catch Exception会把真实adapter导入回归也当作SKIP；不据本环境重放反推远端宣称的149实跑不成立，远端原日志未在本轮独立重现。

建议显式返回PASS/SKIP/FAIL，由main汇总；required adapter未执行时输出PARTIAL/INCONCLUSIVE（按CI契约选择非零），仅预期可选依赖缺失可SKIP，其他导入异常应FAIL。最小验证分别覆盖实际执行成功、预期依赖缺失、adapter导入RuntimeError，普通/-O均不能在含SKIP时打印全项ALL PASS。

## 旧项复查、历史数据影响与未覆盖

- 142：普通不同sha8内容并发修复CPU控制通过；本报告仅新增可触发的同短前缀条件残余，不重复旧共享固定路径结论。
- 149：入口冻结解析对象传adapter，源文件两读窗口已在变更上静态闭合；本轮未在torch/真实Qwen环境执行adapter，状态fixed/static-rechecked，非完整数值验收。被冻结的解析对象后续若外部主动修改的情形不在本轮已证缺陷内。
- 150/151、此前prefill/预算/kernel旧问题无受影响源码变化，维持上次状态，不因本轮没有修改而宣称已修复。迁移误删092维持refuted。
- 没有新增实验结果文件，未发现已跑数据受到本次两项影响的证据。152影响验收声明强度，不能用脚本“ALL PASS”单行关闭149；142影响触发条件下实际加载文件身份，数值变化、论文结论影响仍未验证。
- 环境：Linux x86_64、Python3.12.14、Clang22.1.3构建；无torch/transformers/triton/实际授权GPU，未运行CUDA sanitizer/竞态工具、合法FASA矩阵、Qwen/长序列/实际数据；驱动、CUDA编译器、GPU架构均不可用，不填假设备版本。
- 本次7个变更Python文件隔离py_compile通过；主库/集成检出无源码未提交改动。没有以CPU字节控制替代GPU kernel验证；本次没有kernel算法变更，未重复旧版本kernel全审。
- 独立审核支持142条件残余与152，明确上述夹具和控制流边界。下一检查点仅在新增相关源码/实验变化时触发，advice-only排除，不重复催修。

## 最小可运行142复现

把以下存为隔离临时目录`repro.py`，在Linux上对上面主库SHA执行 `python repro.py /path/to/two-level-attention`，再以`python -O`执行。仅标准库，两进程各有超时、至多一百万候选，全部写入TemporaryDirectory；不会修改检出。完整脚本包含普通控制与完整SHA目录内存建议对照。

```python
import ast,dataclasses,hashlib,json,multiprocessing as mp,os,pickle,sys,tempfile,types
from pathlib import Path
REPO=Path(sys.argv[1]).resolve()
OUT=Path(__file__).parent
SOURCE=REPO/'sparse_attn/info.py'
names={'FasaAgreementSnapshot','resolve_fasa_agreement_path','stage_fasa_agreement','_fasa_agreement_sha8'}
def module(reference=False):
    tree=ast.parse(SOURCE.read_text()); nodes=[n for n in tree.body if getattr(n,'name',None) in names]
    if reference:
        # Comparison proposal in memory only: full SHA directory, sha8 naming unchanged.
        for n in ast.walk(next(n for n in nodes if n.name=='stage_fasa_agreement')):
            if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='staged_root' for t in n.targets):
                n.value.args[-1]=ast.Name(id='full_sha',ctx=ast.Load())
    m=types.ModuleType('_stage_probe');sys.modules[m.__name__]=m
    m.__dict__.update(dataclasses=dataclasses,hashlib=hashlib,os=os,__file__=str(SOURCE))
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(SOURCE),'exec'),m.__dict__)
    return m
def payload(n): return pickle.dumps({'calibration_fixture':n},protocol=4)
def pair():
    seen={}
    for n in range(1000000):
        h=hashlib.sha256(payload(n)).hexdigest()[:8]
        if h in seen:return [seen[h],n]
        seen[h]=n
    raise RuntimeError('bounded search found no pair')
def worker(role,ids,td,barrier,first,second,pipe,reference):
    try:
        m=module(reference); root=Path(td);src=root/role
        p=src/'Qwen3-8B/qasper_agreement.pkl';p.parent.mkdir(parents=True);p.write_bytes(payload(ids[role=='B']))
        original=os.replace
        def replace(a,b):
            barrier.wait(timeout=10)
            if role=='B':
                if not first.wait(10):raise RuntimeError('A timeout')
            original(a,b)
            (first if role=='A' else second).set()
        m.os=types.SimpleNamespace(**{k:getattr(os,k) for k in ['path','makedirs','getpid','lexists','remove'] if hasattr(os,k)},replace=replace)
        args=types.SimpleNamespace(model='Qwen3-8B',fasa_dataset='qasper',fasa_agreement_dir=str(src))
        m.stage_fasa_agreement(args,str(root/'repo'))
        if not second.wait(10):raise RuntimeError('B timeout')
        snap=args._fasa_staged_snapshot;actual=Path(snap.dst_path).read_bytes()
        pipe.send(dict(role=role,source_sha256=snap.sha256,sha8=m._fasa_agreement_sha8(args),
                       actual_sha256=hashlib.sha256(actual).hexdigest(),runtime_pickle=pickle.loads(actual),
                       source_pickle=pickle.loads(snap.raw),matched=actual==snap.raw,
                       dst=str(Path(snap.dst_path).relative_to(root))))
    except BaseException as e:pipe.send(dict(role=role,error=type(e).__name__,message=str(e)))
    finally:pipe.close()
def run(ids,reference=False):
    ctx=mp.get_context('fork');barrier=ctx.Barrier(2);a=ctx.Event();b=ctx.Event();procs=[];pipes=[]
    with tempfile.TemporaryDirectory() as td:
        for role in ['A','B']:
            parent,child=ctx.Pipe(False);p=ctx.Process(target=worker,args=(role,ids,td,barrier,a,b,child,reference));p.start();child.close();procs.append(p);pipes.append(parent)
        rows=[]
        for pipe in pipes:
            if not pipe.poll(20):raise RuntimeError('worker timeout')
            rows.append(pipe.recv());pipe.close()
        for p in procs:
            p.join(2)
            if p.is_alive():p.terminate();p.join();raise RuntimeError('worker hung')
        return rows
if __name__=='__main__':
    ids=pair(); rows=run(ids);controls=run([1,2]);proposal=run(ids,True)
    result=dict(source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),python=sys.version,optimized=sys.flags.optimize,
                collision_ids=ids,input_hex=[payload(i).hex() for i in ids],
                real_sha256=[hashlib.sha256(payload(i)).hexdigest() for i in ids],
                collision=rows,ordinary_control=controls,fullsha_directory_proposal=proposal,
                limitation='stdlib valid pickle lifecycle fixtures, not actual tensor/calibration/model/GPU execution; controlled replace ordering, no hash mocks')
    print(json.dumps(result,indent=2))
    if not (rows[0].get('matched') is False and rows[1].get('matched') is True and all(x.get('matched') for x in controls+proposal)):sys.exit(1)

```

## 原始证据身份

原始输入hex及双模式输出保留私有状态，以上短日志原样摘录；下表绑定原文件以便复查，不提交大型日志、凭据或私密原始数据。

| 文件 | SHA256 |
|---|---|
| probe.py | `9f85c6aed9d4b5714d69dcc3acdd367fc5121ba97758a284bee2067e5c9f31b6` |
| raw.json | `45bda439fe59372110c6d12c80a355b16ce07b5386267ee2f0db8cd312f13290` |
| raw-O.json | `0861a38cba84c9390e9f39d5c4b405e3719312fa8f23f399d187d54b61cf84db` |
| cplus_probe.py | `3b2a10ad8d68bb1f011c1a7fceec9e9a89b8b775ea5edae9b2665d1cbcae8f1c` |
| cplus-source.log | `877360175ec7dbf6747c315951406da2545e3372390388d077dbdfbc5eaabfbe` |
| cplus-source-O.log | `1d7c1bed6232007d4280a89c81c4cc56c7edcbe7dee34d94ba7a012b1061e464` |
| native-cplus.log | `66b6ddd610498929c8149ca8c45e26e8ca5c914ff420fb45007109f98d589ff4` |
| compile.json | `5b201ec8085fd19a0624f83027c2ffeb387de7b9a9b9b8e6e7e2b6af61275f43` |
| independent-review.md | `a8e004fd0373e6829f779ce30ad54478586a65a6e5239b70d875dfa6a3bf026a` |
| independent-rerun.json | `45bda439fe59372110c6d12c80a355b16ce07b5386267ee2f0db8cd312f13290` |
| independent-controls.py | `d0892f11f861bce64844df85a0a6ed1d7837d5885dd2f2b6f6f43f911146b785` |
| independent-controls.json | `313b5f365ca8d0fd9d119d25ca8ad3c2579134bfcc6def8b0fbb4dee13a603a5` |

主源码info.py SHA256：`849c6d613c2e1c0d670489998fe08a1fd0f486b1e81091c179cc397f56fdc797`。发布前再次fetch核验主/集成SHA；普通报告提交后逐字节读回记录保存在私有检查点。
