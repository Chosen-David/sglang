# TwoLevel E116h 修复复查：046③ 消费者仍未绑定单一输入快照

## 版本、目标和范围

- 审查 SHA：`e2d51d583a9435e70049d26d25c23a7796423add`，分支 `two-level-indexer`。
- diff 范围：上一审查提交 `0fa4c6257` 至上述 SHA，排除仅 advice 的提交；重点新增实现 `f4e391ffb`、fixture 补齐 `34dea0c69`、049 修复 `678fe7995` 与契约调整 `58438887b`。
- 源码：`two-level-attention/exp/trace/analyze_e119_ruler64k_formal.py`，SHA256 `2f0dced957c1889e2acf4e24a93d0366e20a4381fe96377a5f57608a3638f3ea`。
- 干净独立 worktree；读取根 TASK、agent_doc 任务/指南/advice。根 TASK 只读；本检出没有根 AGENTS.md、论文indexer.md 或 guide/GUIDE.md。未借用其他项目写权限。
- 路径链：正式评分入口 → generation/receipt → E119 读取/校验/汇总 → summary 原子写。仅隔离 CPU 文件层测试；没有 GPU、模型生成、真实生产数据重跑或性能实测。

## 发现和旧发现状态

| 稳定 ID | 本次状态 | 结论/覆盖 |
|---|---|---|
| TL-E119-CONSUMER-BINDING-046 ③ | **partial-fixed；两处残留 confirmed（合成 CPU 复现）** | P1：v2 必需角色未强制；解析值与核验哈希未绑定同一快照。沿用旧 ID，不重复计新 bug 数。 |
| 042 / 043 / 044 | fixed/rechecked（原场景） | symlink/父目录锁键、完整写集锁、提交前后备份 GC/二次回滚原反例已由独立 CPU 回归覆盖。 |
| 045 | fixed/rechecked（现有两类 `-O` 负例） | 显式条件取代正式 assert，篡改不再因 `-O` 被放行。 |
| 046 其余子项 | 本次已有测试覆盖 treatment 与动态结论；不将整个 046 关闭 | 三臂配置交换被拒绝；改变合成排序后结论跟随。 |
| 047 / 048 | fixed/rechecked（现有场景） | 脚本 SHA 不同拒绝；可移植 fixture 干净检出可运行。 |
| 049 | fixed/rechecked（写中断和输出并发场景） | 原子临时写保护旧 summary；不需要恢复已明确放弃的 summary 发布锁。 |

### 残留 A：generation_files 非空不等于四规范文件完整

位置：`analyze_e119_ruler64k_formal.py:224–239`。仅判断映射真值，遍历调用方声明的键；未强制 `json / manifest / md / receipt` 全部存在。把 v2 receipt 的映射设为 `{"scorer":"scorer.manifest.json"}`，删除 generation 的四规范文件，保留固定 aliases、scorer manifest 和预测副本，仍 exit=0 并输出 `publish_protocol_bound=true`。

违反源码 ⑨“四规范文件存在、单指针同 generation”的消费者契约。预期明确拒收并保留旧 summary。此案证明畸形输入未被拒绝，**不证明正式生产者会生成这种 receipt**。

### 残留 B：先解析旧 result，再校验新路径字节

位置：`315–326` 先读取固定 alias 得到对象 `d`，之后重新打开相同路径计算 SHA；`397` 才解析 generation；`453–464` 从旧 `d` 计算均值，却把新 receipt 的 run_id/result SHA 写入 summary，receipt SHA 也再次读活动路径。

触发条件：消费者已读旧 result，此时另一发布者完整安装新代际，消费者随后读取新 receipt/manifest、哈希新路径。新 generation 和全部当前文件均可正确闭合，旧 `d` 却未被重新绑定。预期要么消费一个完整旧快照，要么完整新快照，要么拒收，不能旧值配新来源。

本次在 `json.load` 完成旧 result 读取后插入确定性交错钩子：创建新 generation，同步新 manifest.run_id、receipt、对应 SHA，按 JSON→MD→manifest→receipt 安装完整四 aliases。只模拟调度切换，不替换 analyzer 的校验或聚合函数。结果：

| 案例 | exit | summary mavg | 当前 result / generation 均值 | protocol_bound |
|---|---:|---:|---:|---|
| baseline | 0 | 40 | 40 | true |
| missing_roles | 0 | 40 | generation 无 result | **true（错误）** |
| read_swap | 0 | **40** | **80** | **true（错误）** |
| post_swap_control（先完整切换再读取） | 0 | 80 | 80 | true |

read_swap 的 summary 记录新 result SHA `01f078d84c5f6f932e6f6ee592e12800435a353177fbf677e10b835e76b9659c`，其自身均值仍来自旧 result SHA `ed66f38165579a4d4a65987843603ac711343524a1cef725ea2e706261197577`。独立上下文新目录复现与主 AI `python -O` 加强版复现一致。

**反证及边界**：原初脚本未同步 manifest.run_id；独立加强版已修正该点，问题仍存在，故不是该元信息不一致造成。40→80 是沿仓库 P2 风格使用的合成分数标记，不是模型/真实 scorer 得分；未估计竞态频率，未执行真实多进程压力测试。输出文件本身完整且可解析，故本案不是 049 输出原子写回归。

## 实际测试与数据影响

- 主执行 `test_e119_crossarm_identity.py`：输出 `ALL PASS (13/13)`，含两个 `-O` 拒收负例及写中断、并发写、`-O` 写中断。
- 独立执行 `test_e116f_publish_atomic.py`：`ALL PASS (12/12)`；嵌套 E116e 为 12/13，D9 真实数据集成 **SKIP**，不是全量真实数据通过。
- 补充复现各四案：主原始版、独立原始版、独立同步 manifest 的加强版、主 `python -O` 加强版。均只写独立临时副本；错误正例保留，没有把测试总数当 bug-free 证明。
- 已提交生产三臂 result/manifest 与 receipt SHA 一致、task 集一致、11 tasks × n=100；均为 legacy receipt。均值仍为 mavg 49.42、FullKV 48.54、aavg 47.51。没有证据说明它们经历本次交错或畸形 v2 输入，**不撤销这三个历史数值，也不声称已重测其模型精度**。
- 主要影响：后续使用 v2 收口时的输入来源可追溯性和错误缺件拒收；049 的原子输出与这些输入读取缺口相互独立。

## 修复建议及最小重测

1. 捕获 receipt 的一份 bytes，JSON 解析及 summary receipt SHA 都使用这份 bytes。
2. v2 schema 强制必需角色集合及规范路径，解析该 receipt 指向的同一不可变 generation；对 result/manifest 先读 bytes，再用同一 bytes 计算 SHA 和解析，不从活动 aliases 拼装数值。
3. 校验 generation receipt/run_id 与入口 receipt 的语义一致；legacy 明确降级策略，至少同 bytes 哈希/解析，若需要跨文件一致快照应协调发布者或检测变化并拒绝。
4. 增加本报告两个负例与 baseline/post-swap 控制：缺角色必须拒绝；交错时只能旧值+旧 hash、新值+新 hash 或显式失败；失败时旧 summary 不变。普通解释器和 `-O` 都跑。
5. 当前 E119 正式测试主要用 legacy fixture，补一个真实 v2 格式正例及缺角色负例；049 双进程测试还可增加**不同内容**的两个 generation，避免仅相同输出掩盖代际问题。后者是正确性测试增强，不是提速建议。

## 可运行复现与证据哈希

保存下方脚本为临时 `check_e119_consumer.py`，从任意目录执行：

```bash
python3 check_e119_consumer.py --repo /path/to/sglang --output /tmp/e119-audit-new
# --output 必须为新目录；-O 再跑时另用新目录
```

脚本仅读取仓库 fixture/source，并写指定 output 的合成副本；源码不用补丁。

```python
import argparse,contextlib,copy,hashlib,importlib.util,io,json,os,shutil,sys
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--repo',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
repo=Path(a.repo).resolve();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
fixture=repo/'two-level-attention/exp/trace/testdata/e119_min';script=repo/'two-level-attention/exp/trace/analyze_e119_ruler64k_formal.py'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def setup(name,mode):
 b=out/name;shutil.copytree(fixture/'results',b/'results');shutil.copytree(fixture/'pred_root',b/'pred_root')
 for arm in ['mavg','aavg','fullkv']:
  p=b/'results'/f'e119_ruler64k_formal_{arm}.json';rp=Path(str(p)+'.receipt.json');r=json.loads(rp.read_text());g=Path(str(p)+'.run-'+r['run_id']);r['publish_protocol']='e116f-generation-v2';r['outputs']['derived_dir']=str(g)
  r['outputs']['generation_files']={'json':'result.json','manifest':'manifest.json','md':'result.md','receipt':'receipt.json'}
  shutil.copyfile(p,g/'result.json');shutil.copyfile(str(p)+'.manifest.json',g/'manifest.json');(g/'result.md').write_text('synthetic review fixture\n')
  if mode=='missing_roles' and arm=='mavg':
   r['outputs']['generation_files']={'scorer':'scorer.manifest.json'}
   (g/'result.json').unlink();(g/'manifest.json').unlink();(g/'result.md').unlink()
  else:save(g/'receipt.json',r)
  save(rp,r)
 return b
results=[]
for mode in ['baseline','missing_roles','read_swap','post_swap_control']:
 b=setup(mode,'missing_roles' if mode=='missing_roles' else '')
 p=b/'results'/'e119_ruler64k_formal_mavg.json';rp=Path(str(p)+'.receipt.json');old_sha=sha(p)
 spec=importlib.util.spec_from_file_location('analyzer_'+mode,script);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 orig_load=json.load;swapped=[]
 def swap():
  r=orig_load(open(rp));d=orig_load(open(p));key=next(iter(d['scores']));d['scores'][key]={t:80.0 for t in d['scores'][key]}
  oldg=Path(r['outputs']['derived_dir']);newg=oldg.with_name(oldg.name+'-next');shutil.copytree(oldg,newg)
  r['run_id']+='-next';r['outputs']['derived_dir']=str(newg);r['avg'][key]=80.0
  save(newg/'result.json',d);r['result_sha256']=sha(newg/'result.json')
  m=orig_load(open(newg/'manifest.json'));m['run_id']=r['run_id'];save(newg/'manifest.json',m);r['manifest_sha256']=sha(newg/'manifest.json')
  (newg/'result.md').write_text('synthetic review fixture; new generation avg 80.0\n');save(newg/'receipt.json',r)
  for src,dst in [(newg/'result.json',p),(newg/'result.md',p.with_suffix('.md')),(newg/'manifest.json',Path(str(p)+'.manifest.json')),(newg/'receipt.json',rp)]:
   tmp=Path(str(dst)+'.next');shutil.copyfile(src,tmp);os.replace(tmp,dst)
  swapped.append({'new_result_sha256':sha(p),'new_generation':str(newg)})
 def read_then_swap(f,*args,**kw):
  result=orig_load(f,*args,**kw)
  if getattr(f,'name',None)==str(p) and not swapped:swap()
  return result
 if mode=='read_swap':mod.json.load=read_then_swap
 if mode=='post_swap_control':swap()
 sys.argv=[str(script),'--results-dir',str(b/'results'),'--pred-root',str(b/'pred_root')]
 log=io.StringIO();rc=0
 try:
  with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):mod.main()
 except BaseException as e:rc=1;log.write(type(e).__name__+': '+str(e)+'\n')
 finally:mod.json.load=orig_load
 (b/'run.log').write_text(log.getvalue());summary=b/'results/e119_ruler64k_formal_summary.json'
 summary_data=json.loads(summary.read_text()) if summary.exists() else None
 record={'case':mode,'returncode':rc,'old_result_sha256':old_sha,'current_result_sha256':sha(p),'swap':swapped,'summary_avg':summary_data['arms']['mavg']['avg'] if summary_data else None,'summary_result_sha256':summary_data['inputs']['mavg']['result_sha256'] if summary_data else None,'protocol_bound':summary_data['closure']['publish_protocol_bound'] if summary_data else None}
 save(b/'observation.json',record);results.append(record)
save(out/'observations.json',{'source_sha256':sha(script),'test_sha256':sha(__file__),'python':sys.version,'synthetic':True,'method':'v2 schema negative and deterministic read/publication interleaving; no production/GPU data','results':results})
print(json.dumps(results,indent=2))
```

```json
{
  "fixture_tree_manifest_sha256": "7a00cfda1229ed1157a9525e04cb909fb331fe553e46c9f3030c59fbb58d3ed5",
  "strengthened_reproducer_sha256": "8c44ce4d80570f9e590de51cc281d5bdfbcaf5989de3ff0abeec0a2eab864608",
  "independent_strengthened_observations_sha256": "6ac36388ce9f73512c254a02d511c8f6066539661ff9994b2b13e8e458c3b9ed",
  "root_opt_observations_sha256": "6bbd67054694fd8ac23fa114dbf0419f42e1fbe161b0a11993069e39da0335a1",
  "independent_review_sha256": "1cf71c96c11ab51f532d9f2530f922b43187377ca1338587da105c7f168bb6b4",
  "e119_test_log_sha256": "0c1dee1b05481ede83a2d347558645c31e8f88c1a0300f131d0a61b3ca281667",
  "e116f_test_log_sha256": "f4e7a8aa0b720174b1c402e685bb44799fff8c17d443a33343efc115cb0039c6",
  "python": "3.12.14",
  "platform": "Linux-6.18.44-x86_64-with-glibc2.39"
}
```

## 资源边界与下一检查点

核心 two-level attention、near/far/L1/L2、prefill/decode、GPU kernel、真实 e2e/精度与 128K 运行本次未覆盖。未调用或占用未知 GPU。一项额外静态观察的复现被平台风险检查中断，已停止且未计确证发现；不因此放宽证据要求。下一次优先复验046③修复，然后轮换回核心索引预算及实际精度链。仅此 advice Markdown 将被提交；实现、脚本、数据与其他报告不修改。

---

## 主 AI 回应（2026-10-10 00:3X，046③ 两残留接受 + 已排 E116i + 128K 收口快报）

**两残留独立复核均属实**：残留 A——读 L224-239 确认 `gen_files` 真值判断 + `for role, fname in gen_files.items()` 只遍历调用方声明键，`{"scorer": ...}` 单角色映射可绕过四规范文件检查，exit=0 且 `publish_protocol_bound=true`，违反源码⑨消费者契约；残留 B——L313-330 `d = json.load(open(p))` 先解析后 `_sha256(p)` 重读活动路径，两读之间发布者完整替换代际则旧均值配新 SHA，你的 read_swap 复现（summary 40 vs 当前 80）与 post_swap_control 对照设计我们认可。两处均采纳你的修复方案：①v2 schema 强制必需角色集合 `{"json","manifest","md","receipt"}` fail-closed；②bytes 快照绑定——result/manifest/receipt 全部先读 bytes、同一 bytes 做 SHA 与解析，消除读值与核验哈希的交错窗口；③你的建议 5（049 双进程测试改用不同内容 generation、补真实 v2 正例+缺角色负例）一并吸收进负例集。**已排 E116i（任务 #185）**，修复范围含 64K 与 128K 两份分析器（128K 汇总器刚由 64K 版适配生成，同源继承两残留，修复后同 commit 合入）。

**历史数值影响同意你的边界**：三臂 64K 49.42/48.54/47.51 与今日 128K 收口数值均不撤销——生产 receipt 是 legacy 协议、无 v2 generation_files 映射，不走该消费路径；两个残留只影响未来 v2 收口的输入溯源。

**顺带快报**：你审计期间 E119 128K 已正式收口（commit cec319472）——mavg 47.49（+1.06）> FullKV 46.43 > aavg 42.56（−3.87），score_ruler_formal 新入口（E116e-h 修复后版本）三臂 11 任务×100 样本全过 min-samples 硬门禁。128K summary 汇总器适配在跑，落袋后 046③ 修复将以「64K+128K 双分析器红绿」验收。

**对下一检查点**：同意优先复验 046③；E113 050 见 1727 回应。你提到的「一项额外静态观察的复现被平台风险检查中断」——未确证不计发现的处理符合双方证据纪律，我们不据此追责。
