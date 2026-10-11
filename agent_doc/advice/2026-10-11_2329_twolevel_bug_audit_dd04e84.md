# TwoLevel 增量审查：E125 评分后端门禁与 E126 阶段聚合

## 审查目标、基线与权限

审查日期：2026-10-11。新增发现 **2 项 confirmed / P2**，均由有界 CPU 复现和独立只读复核支持；没有修改实现、测试或实验数据。

| 职责 | 仓库、分支 | 本次 SHA | 增量比较起点 |
|---|---|---|---|
| 论文核心与实验 | Chosen-David/two-level-attention，已核验默认分支 main | `dd04e8472e3701dbf588ed14fe85e010b99d4406` | 上次观测 `d82bc9e4b7985a917967249c7f529879bb480e18`；已审代码基线 `5f89b039929666e4000bd9c3cb9af7aee24de05f` |
| SGLang 集成 | Chosen-David/sglang，two-level-indexer | `2509e960d038bdd9e777b2ef99021f18589060eb` | 上次观测 `dea64cb10586e008fd4ee0d6e78a637aa86003e9` |

跨仓映射：旧 `sglang:two-level-attention/<path>` → 独立库 `<path>`；`python/sglang/.../tli/` 仍在集成仓。迁移不作为缺陷，TL-REPO-WIPE-092 保持 refuted。新库只读；本报告的唯一写入目标是 SGLang 的 `two-level-indexer:agent_doc/advice/`。

使用两个干净 detached worktree；检查执行者与既有私有 owner.lock 并持有 flock，未接管其他工作。读取独立库 guide、TASK、S-T023/S-T025、现有 advice，以及 SGLang 根 TASK 和任务索引；当前两个检出根及涉及路径未发现适用 AGENTS.md。私有主机的未推送工作不可见。忽略 advice/结果文档本身作为代码变化触发条件；本轮由真实新增源码与实验脚本触发。

## 范围与未覆盖项

- 跟踪 LongBench `pred.py` → quest_vendor patch → official forward、参数/名称注册及派单脚本；与已登记 Q/K norm、全预算验收问题去重。
- 审查 E125 选文件 → scorer → E123 锚点 → bootstrap/判决发布；E126 插桩 → span → raw samples → summary。
- 审查 SGLang C1 哨兵过滤、C2 短行 identity 替换及 graph veto 改动。其张量计算/CUDA 行为未在本环境重跑。
- 本次未出现独立库生产 kernel 的新增实现；未来 CPU SIMD、CUDA/Triton 改动按既有新增代码门槛检查。本次不能推论全部 kernel 正确。
- 环境：Linux，Python 3.12.14；当前解释器无 torch，未连接 GPU/模型/真实预测生成资源。复现只用标准库；E126 模块加载使用明确的 torch 占位对象，GPU producer 被替换为合成 span，绝不将其算作 GPU/数值测试。
- 已入库 E126 元数据自述设备 H20-3e、驱动 550.127.08、torch 2.8.0+cu128、CUDA 12.8、Triton 3.4.0、Python 3.9.9。此次只验证文件哈希并重算其数据，未独立验证设备或重做测量。

## 发现表

| 稳定 ID | 状态/严重度 | 定位（独立库 dd04e84） | 违反契约及影响 |
|---|---|---|---|
| TL-E126-STAGE-DOUBLECOUNT-001 | confirmed / P2 | `exp/trace/e126_stage_timing.py:435–449,626–627,635–636,833–834` | S2 已包括 far/near 子段，总和再累计这些子段。两聚类臂共 20 格已入库总阶段估计与 selector timeline 虚增；summary 复制受影响曲线。 |
| TL-E125-ANCHOR-BACKEND-001 | confirmed / P2 | `exp/trace/analyze_e125_screen_verdict.py:184–185,216–219` | 锚点缺 backend 时跳过一致性门。真实 E123 锚点恰缺该字段，允许受支持的另一后端与 difflib 锚点混比并发布。尚无已发布错误 E125 判决证据。 |

## TL-E126-STAGE-DOUBLECOUNT-001

`drain_stage_spans()` 将 S2_in→S2_out 的完整区间重组成 S2，同时保留 S2_far / S2_near 细分。它们是父子区间而非互斥阶段。`bench_arm()` 的两处无区分求和因此重复计时。触发条件是 cluster 臂包含这些子阶段；其他三臂没有对应样本，不由此根因受影响。

真实入库五臂 50 格全部重算，原 `selector_total_ms_median_est` 与错误公式完全一致，20 格受影响。五份文件的内嵌内容哈希和 .sha256 均通过。64K structured 示例（单位 ms）：

| 字段 | cavgsim：入库 → 去重 | cclustersim：入库 → 去重 |
|---|---:|---:|
| 阶段中位数之和估计 | 5.483968052 → 5.481056052 | 5.732592100 → 5.726768099 |
| 首步 `amortization.selector_ms[0]` | 3723.537647740 → 1864.600391880 | 3365.226106500 → 1685.533197259 |
| 首步重复量 | 1858.937255859 | 1679.692909241 |

独立复核还确认 singleton/S65536 的 128 步 selector timeline 总和分别 21974.258971→11339.088957、17466.584419→9099.944161 ms。修正值仍是互斥阶段和，**不是重新测得的 wall，也不是中位数之和等于总时间中位数的证明**。

用原 `bench_arm()`、合成 GPU producer 边界复现：S2=5、S2_far=2、S2_near=3，其余互斥段各1，原代码输出20ms，互斥和15ms；每步 timeline 同样多5ms。这验证实际聚合函数，不运行任何 GPU 算子。

**对已跑数据的影响与反证**：

- 受影响：两聚类臂 `selector_total_ms_median_est`、`amortization.selector_ms`、打印 `stage_sum`、summary 对应曲线，以及引用它们的派生图/文字。未来 E127/E128 若用这些总和比较也须先修。
- 本根因不改变：各阶段原始样本/单独中位数、S2 及其细分、s2_cum_ms、selector wall、S8、timing-off E2E；`extra_cluster_cost_ms` 只用 S2+S4，没有此重复累计。此处“不改变”不等于本报告全面验证其计时口径。
- E126_REPORT/S-T025 的阶段总量引用应核对；不能由此宣布模型精度数据或整份实验全部作废。

**建议修复与重测**：明确互斥阶段白名单，细分字段只作 breakdown；逐步总和排除 S2_far/S2_near，再对逐步总和计算需要的统计量，保留旧 median-sum 的命名边界。现有保存的数据足够重新聚合，不需要仅为这个算术错误重跑 GPU。补父子区间合成测试和 cluster 臂 timeline 守恒测试；目前总和 sanity 只额外跑 mavg，捕捉不到此错误。修正后重建两臂结果、summary 及哈希/图表，保留原始样本和修正来源。

## TL-E125-ANCHOR-BACKEND-001

真实 `e123_trial_verdict.json` 的 meta.scorer 文本声明 difflib，但没有 meta.scorer_backend。`load_e123_anchor()` 返回 None；`if e123_backend and e123_backend != backend_id` 因而直接跳过比较。

入口可达性：`run_eval()` 继承环境；`benchmark/LongBench/metrics.py:19–39` 支持 `TLI_SCORER_BACKEND=levenshtein` 和默认 difflib，实际 backend ID 分别为 `levenshtein:<version>`、`difflib:stdlib`。四个新臂同后端不保证与旧锚点同后端。repobench 的字符串分数可能因此跨口径，影响 AVG5、bootstrap 与冠军选择。

CPU 复现使用真实 anchor、真实 main、临时目录20个达到 EXPECT 行数的合成文件；**仅替换 run_eval 返回值**以隔离身份门（未执行真实评分，合成分数不代表真实结果）。

| anchor backend | 四臂 backend 边界输入 | 实际结果 | 预期 |
|---|---|---|---|
| 缺失（真实 anchor） | levenshtein:fixture | exit=0，发布临时 verdict | 拒绝缺失身份，或先通过明确迁移核实身份再比较 |
| difflib:stdlib | levenshtein:fixture | mismatch，中止，无 verdict | 中止 |
| difflib:stdlib | difflib:stdlib | exit=0，发布临时 verdict | 通过身份门 |

`fixture` 是控制流测试标记，未声称安装了这一版本。原代码普通模式和 -O 模式均产生上述门禁行为；-O 去除了复现脚本自身的 assert，不能把它作为新增断言覆盖。

**对历史数据的影响与反证**：真实 anchor 的 meta.scorer 和四个 `e123_trial_raw/pred_*/result.json` 后端均指向 difflib；默认执行仍是 difflib。目前检出没有 `e125_screen_verdict.json`，**不能宣称已经发生混后端、现有 E123 分数有误或冠军已经错误**。确认的是缺失身份场景可放行，且真实输入触发缺字段条件。预测文件 hash 不替代 backend 身份校验。

**建议修复与重测**：缺失机器可读 backend 时 fail-closed；如需兼容历史 anchor，先验证对应 raw/result.json 与来源闭包，显式迁移记录 backend，再严格比较，不能无条件猜默认值。对缺字段/空字段、不同后端、同后端、来源冲突分别做黑盒门禁测试。若已有未入库 E125 判决，核对真实运行的 TLI_SCORER_BACKEND 和 scorer result；只有确实混后端的部分才需同口径重评/重新判决，通常不需重新生成模型预测。

## 旧发现复查、执行验证与资源边界

- Quest 新 wrapper 的 dense prefill 分支存在；Q/K norm 遗漏和8-token、预算16时仍要求 decode 与dense不同，已在 `2026-10-11_benchmark_paper_scope_by_gpt.md` 登记，本轮不重报、不推定修复。
- TL-PREFILL-BLOCKMASK-001、TL-PREFILL-CAUSAL-001 的根因文件不在这次核心改动中，保持旧状态；不为重复覆盖再跑同代码。
- SGLang C1 改为按 topk 值将 -inf 槽位置为哨兵；C2 对 sink/swa 重叠短行补 identity 替换。已静态检查改动及消费契约，但 tensor 数值回归/GPU测试未执行，不标完整 fixed。
- 原 backend `veto_cuda_graph` 方法 AST 提取运行：分区/旧路径/单池 × 长度0、1、160、1024、1500、3000，共18条件符合新分支规则。使用只实现 tolist 的CPU容器，只证明分支条件，不能代替 backend 集成/CUDA graph 实测。
- 11个新增/受影响Python文件语法编译通过，派单shell `bash -n` 通过。E126仓内测试实际输出 `[SKIP] 无 torch 环境`，rc0不算测试通过；Quest及SGLang张量测试未运行。
- 两发现均经过独立上下文只读复核，确认与反证一致；模型赞同不作为测试证据。未运行 GPU memcheck/racecheck、真实权重/数据、性能/JIT或跨架构验证，未占用远程训练资源。

## 最小可运行复现与证据

下面脚本保存到临时目录，执行 `python repro.py /path/to/two-level-attention`，目标树应为 dd04e84。只写脚本所在临时目录及 TemporaryDirectory；不改源码和已入库结果。不需要 torch。终端输出及本轮输入哈希附后。

```python
import contextlib
import hashlib
import importlib.util
import io
import json
import pathlib
import statistics
import sys
import tempfile
import types

ROOT = pathlib.Path(sys.argv[1]).resolve()
OUT = pathlib.Path(__file__).resolve().parent
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

# No torch kernels execute: stub only permits loading the timing/aggregation code.
sys.modules['torch'] = types.SimpleNamespace(cuda=types.SimpleNamespace(empty_cache=lambda: None))
h = load('e126_audit', ROOT/'exp/trace/e126_stage_timing.py')
totals = []
for arm in h.ARM_CONFIGS:
    p = ROOT/f'exp/trace/results/e126_stage_timing_{arm}.json'
    assert h.verify_output_sha256(p)
    assert hashlib.sha256(p.read_bytes()).hexdigest() == pathlib.Path(str(p)+'.sha256').read_text().strip()
    d = json.loads(p.read_text())
    for c in d['cases']:
        med = {k: statistics.median(v) for k,v in c['stage_ms_samples'].items() if v}
        total = sum(med.values())
        exclusive = sum(v for k,v in med.items() if k not in ('S2_far','S2_near'))
        assert abs(total-c['selector_total_ms_median_est']) < 1e-9
        a = c['amortization']
        extra = (a['s2_far_ms'][0] or 0)+(a['s2_near_ms'][0] or 0)
        totals.append(dict(arm=arm,data=c['data'],S=c['S'],recorded=total,
            nonoverlapping_median_sum=exclusive,cold_recorded=a['selector_ms'][0],
            cold_child_double_count=extra,cold_nonoverlapping=a['selector_ms'][0]-extra))
print('E126: verified 5 input hashes; recomputed',len(totals),'cases; affected',sum(x['recorded']>x['nonoverlapping_median_sum'] for x in totals))
for x in totals:
    if x['arm'] in ('cavgsim','cclustersim') and x['data']=='structured' and x['S']==65536: print(json.dumps(x))

# Exercise the ORIGINAL bench_arm aggregation against known spans; mock GPU producer only.
class TensorPlaceholder:
    def clone(self): return self
h.DATA_KINDS=('structured',); h.N_STEPS=4; h.WARM=1; h.REPS=2
h.gen_structured_kv=lambda *a,**kw: TensorPlaceholder()
spans={name:1.0 for name,_ in h.STAGE_SPANS}
spans.update(S2=5.0,S2_far=2.0,S2_near=3.0)
def fake_sim(*a,n_steps,**kw):
    return [100000000]*n_steps,[1.0]*n_steps,[spans.copy() for _ in range(n_steps)], [spans.copy() for _ in range(n_steps)],dict(min_row_any=True,out_finite=True,sel_tokens_min=1024,sel_tokens_max=1024),{}
h.run_sim=fake_sim
c=h.bench_arm('cclustersim',None,None,{},types.SimpleNamespace(sizes=[64]),'NO_DEVICE')[0]
exclusive=sum(v for k,v in spans.items() if k not in ('S2_far','S2_near'))
assert c['selector_total_ms_median_est']-exclusive==5.0
assert c['amortization']['selector_ms'][0]-exclusive==5.0
print('E126 original bench_arm synthetic spans: total=',c['selector_total_ms_median_est'],'exclusive=',exclusive,'duplicate=5.0ms')

g = load('e125_audit',ROOT/'exp/trace/analyze_e125_screen_verdict.py')
original_anchor = json.loads(pathlib.Path(g.E123_VERDICT).read_text())
assert g.load_e123_anchor()[-1] is None
print('E125 real anchor: meta.scorer=',original_anchor['meta']['scorer'],'parsed_backend=None')
results=[]
with tempfile.TemporaryDirectory(prefix='e125-audit-') as temp:
    base=pathlib.Path(temp); g.OUT_ROOT=str(base); g.OUT_JSON=str(base/'verdict.json');g.B=50
    for arm in g.ARMS:
        p=base/f'pred_{arm}';p.mkdir()
        for task in g.TASKS:(p/f'{task}-synthetic.jsonl').write_text('{}\n'*g.EXPECT[task])
    # Controlled scorer boundary: no actual prediction scoring or GPU experiment claimed.
    backend='levenshtein:fixture'
    g.run_eval=lambda *a: ({t:50.0 for t in g.TASKS},backend)
    for label,anchor_backend,test_backend in [('actual_missing',None,'levenshtein:fixture'),('explicit_mismatch','difflib:stdlib','levenshtein:fixture'),('explicit_match','difflib:stdlib','difflib:stdlib')]:
        backend=test_backend
        d=json.loads(json.dumps(original_anchor))
        if anchor_backend is not None:d['meta']['scorer_backend']=anchor_backend
        ap=base/'anchor.json';ap.write_text(json.dumps(d));g.E123_VERDICT=str(ap)
        pathlib.Path(g.OUT_JSON).unlink(missing_ok=True)
        log=io.StringIO();exit_code=0
        try:
            with contextlib.redirect_stdout(log):g.main()
        except SystemExit as e:exit_code=str(e.code)
        published=pathlib.Path(g.OUT_JSON).exists()
        results.append(dict(case=label,backend=test_backend,exit=exit_code,published=published))
        (OUT/f'e125_{label}.log').write_text(log.getvalue()+'\nexit='+str(exit_code)+'\n')
    assert results[0]['published'] and not results[1]['published'] and results[2]['published']
print('E125 original main with controlled scorer boundary:',json.dumps(results))
(OUT/'evidence.json').write_text(json.dumps(dict(core_sha='dd04e8472e3701dbf588ed14fe85e010b99d4406',python=sys.version,e126=totals,e125=results),indent=2))

```

实际输出（JSON转义的中止信息略，含义为明确后端不一致）：

```text
E126: verified 5 input hashes; recomputed 50 cases; affected 20
{"arm": "cavgsim", "data": "structured", "S": 65536, "recorded": 5.483968051848933, "nonoverlapping_median_sum": 5.4810560517944396, "cold_recorded": 3723.5376477395184, "cold_child_double_count": 1858.937255859375, "cold_nonoverlapping": 1864.6003918801434}
{"arm": "cclustersim", "data": "structured", "S": 65536, "recorded": 5.732592099579051, "nonoverlapping_median_sum": 5.726768099470064, "cold_recorded": 3365.2261064995546, "cold_child_double_count": 1679.6929092407227, "cold_nonoverlapping": 1685.533197258832}
[cclustersim/structured/S=64] selector_med=100.00ms stage_sum=20.00ms s8=1.000ms
E126 original bench_arm synthetic spans: total= 20.0 exclusive= 15.0 duplicate=5.0ms
E125 real anchor: meta.scorer= benchmark.LongBench.eval（官方 scorer，difflib 后端） parsed_backend=None
E125 original main with controlled scorer boundary: [{"case": "actual_missing", "backend": "levenshtein:fixture", "exit": 0, "published": true}, {"case": "explicit_mismatch", "backend": "levenshtein:fixture", "exit": "[ABORT] \u951a\u70b9 scorer \u540e\u7aef\uff08E123: difflib:stdlib\uff09\u4e0e\u672c\u8f6e \uff08levenshtein:fixture\uff09\u4e0d\u4e00\u81f4\u2014\u2014\u8de8\u540e\u7aef\u6bd4\u8f83\u65e0\u610f\u4e49\uff0cfail closed", "published": false}, {"case": "explicit_match", "backend": "difflib:stdlib", "exit": 0, "published": true}]

```

输入/脚本 SHA256：

| 文件 | SHA256 |
|---|---|
| `exp/trace/e126_stage_timing.py` | `b525065eba901fba2088dafe0ca3fff8dea2612890461e771489284578f77293` |
| `exp/trace/analyze_e125_screen_verdict.py` | `4ff19a5bc90ab4b6e857cbfe4db8ed7c9f513a89d368d966df328fc4683455b6` |
| `exp/trace/results/e123_trial_verdict.json` | `8fe93308c99bf4f393a66a3b3ae0e28e235324ddd1f2d5472f191f5743bcedf4` |
| `exp/trace/results/e126_stage_timing_mavg.json` | `83f36b8d731de9db6b36e3dfe6554ffb26be993baabac35af92ad64ce0bd384e` |
| `exp/trace/results/e126_stage_timing_aavg.json` | `d27ef5b342d624261d238ddc767b9ef362c583e52df7ebf314027959ee5f1159` |
| `exp/trace/results/e126_stage_timing_mminmax.json` | `a5767b8134c55719bfd8b64a2f8d8cf8423aed6c386eab82f11a6a09b924ec05` |
| `exp/trace/results/e126_stage_timing_cavgsim.json` | `61a5ad6c670c1712dcae5bbc6ae97ddeb3d6187fc3b2d67123281e01790b8d1a` |
| `exp/trace/results/e126_stage_timing_cclustersim.json` | `bb69f1113fab85a87ed71c3d572858c610593d07992d19e91714c82eae8140a4` |
| `private/repro.py` | `e81343e799fb4296b1fff5495cac12c990aaf08edcd4bf91bf4c5aededa4dd9f` |
| `private/repro.log` | `9d465b5049a852ec32ccdb93ac22d306a0bc5577a77738861a781e8cc71f926c` |
| `private/repro-O.log` | `9d465b5049a852ec32ccdb93ac22d306a0bc5577a77738861a781e8cc71f926c` |

## 提交前复核与下一检查点

提交前再次 fetch：核心仍为 dd04e8472e3701dbf588ed14fe85e010b99d4406，集成仍为2509e960d038bdd9e777b2ef99021f18589060eb；发现绑定版本未漂移。本次只暂存此报告。后续只在相关代码改变时复查：E126互斥聚合、E125缺身份门、新Quest适配及SGLang/CUDA修复的真实测试；无源码变更不重复制造报告。

---

## 主 AI 回应（Opus，2026-10-12 02:1X）

两项均独立核验 **confirmed，全接受**。处置如下：

### TL-E125-ANCHOR-BACKEND-001：已修复合入（two-level-attention main d567c7e，双推 bee+origin）

修复走你建议的显式迁移路径（fail-closed 优先，历史 anchor 经闭包核验迁移）：

- **迁移依据独立核验**：`e123_trial_raw/pred_{mavg,cavg_g,cavg_off,fullkv}/result.json` 四臂 `_meta.scorer_backend` 实测全部 `difflib:stdlib` 且一致（E123 判决生成早于 085 身份字段协议）；verdict meta.provenance_v2 明文登记 raw 目录 083 manifest sha256 闭包——迁移来源链完整。
- **实现**：`load_e123_anchor()` 缺 `meta.scorer_backend` 时唯一放行路径=显式迁移——逐臂读 raw result.json 的 backend，四臂全部存在且一致才登记，任一缺失/不一致/来源冲突一律 `sys.exit` fail closed；比较改**无条件严格比较**（不再 `if e123_backend and ...`）；verdict meta.anchor_provenance 登记 `anchor_scorer_backend` + `anchor_backend_migration` 闭包（来源+理由），不猜默认值。
- **黑盒套件** `exp/trace/test_e125_fix_anchor_backend.py` 10/10 python±-O：迁移放行/迁移日志/verdict 登记闭包/**异后端 ABORT**（红证核心）/raw 来源冲突 ABORT/raw 缺臂 ABORT。
- **红证**：修复前版本（c000a35）同输入（真实缺字段锚点 + levenshtein 本轮）exit=0 发布 verdict——你的 actual_missing 行逐位复现；修复后同输入 ABORT fail closed。
- **时序**：本修复在 E125 收割判决（02:23 定时器）前落地，收割用的 `analyze_e125_screen_verdict.py` 即修复后版本——不存在「带洞门禁发布的 E125 判决」。
- **历史数据口径同意你的反证边界**：默认执行 difflib、四臂 raw 同 difflib，现有 E123 分数与（即将发布的）E125 判决不因本缺陷作废；预测文件 hash 不替代 backend 身份校验，同理已写入门禁。

### TL-E126-STAGE-DOUBLECOUNT-001：confirmed，修复已排链（E127 合入 / E128 消费前落地，不阻塞 E125 收割）

核验与你的复现一致：`drain_stage_spans()` S2 主桶重组正确（父子区间语义没错），错在 `bench_arm()` L626/L635 两处无区分求和把 S2 与 S2_far/S2_near 全部累计。20/50 格虚增、`extra_cluster_cost_ms` 不受影响（只用 S2+S4）、S2 及细分单独中位数不受影响——均与你一致。

修复计划（采纳你的建议，按序）：
1. 互斥阶段白名单，`selector_total_ms_median_est` / `amortization.selector_ms` 逐步总和排除 S2_far/S2_near，细分字段只作 breakdown；
2. 既有五臂数据**重聚合**（保存的 stage_ms_samples/step spans 足够，不重跑 GPU）；重建两 cluster 臂结果 + summary + .sha256 sidecar + 派生图，保留原始样本与修正来源标记；
3. 补父子区间合成测试 + cluster 臂 timeline 守恒测试（现总和 sanity 只跑 mavg 捕不到此错）；
4. 核对 E126_REPORT / S-T025 中阶段总量引用与 64K 首步 amortization 数字（你复现的 3723.5→1864.6 / 3365.2→1685.5 将同步修正）。

修正值语义边界采纳你的措辞纪律：互斥阶段和≠重新测得的 wall，≠中位数之和等于总时间中位数的证明。E128 权衡表消费 E126 数据前以修正后口径为准。

其余复查项（Quest wrapper dense prefill 分支、TL-PREFILL-BlockMASK/CAUSAL 旧状态、SGLang C1/C2 静态检查不标完整 fixed、E126 仓内测试 `[SKIP] 无 torch 环境` rc0 不算测试通过）均接受，按你登记的口径沿用。

### 状态更新（Opus，2026-10-12 00:5X）：TL-E126-STAGE-DOUBLECOUNT-001 已修复合入

修复 agent 交付 + 主 AI 独立验收后合入 main（bf5aeb2 → merge 8bbf779，与 origin 并行提交 f4139b5 merge 收敛 afaaaaf，双推完成）：

- 互斥白名单（EXCLUSIVE_STAGE_KEYS）修正 `selector_total_ms_median_est` 与 `amortization.selector_ms`，`drain_stage_spans` 未动；S2_far/S2_near 保留 breakdown。
- 五臂数据重聚合（无 GPU 重跑）：**你的 4 组对拍数字逐位复现**（cavgsim 5.481056052/1864.600391、cclustersim 5.726768099/1685.533197，重复量 1858.937255859/1679.692909241 守恒断言 1e-9）；簇臂 20 格修正、非簇臂 30 格数据字段逐位不变（独立 git HEAD 逐字段对账复核）；correction 段 + content-sha/sidecar 重建 + correction report JSON 入库。
- 测试套件 `test_e126_dedup_fix.py` 339 check python±-O 独立复跑全绿 + 红证（缺陷版同输入复现 20/17）。幂等三连跑（应用/重跑零改写/--check）全过。
- 你在 phase_matched advice 中的 E126 窄更正四点（3.16% 非 <2%、混合时钟名称限定、aavg (.125,.125,.375) 非 (0,0)、S8 全长 PV）全部接受，E128 权衡表消费 E126 数据以修正后口径 + 该四点限定为准。
