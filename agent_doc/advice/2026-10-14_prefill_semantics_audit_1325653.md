# Prefill 接入语义与双流审查（2026-10-14）

## 结论与范围

回应用户“prefill 阶段也采用两级筛选和 near+far+sink+swa”：当前接入确实调用两级 indexer，但每 chunk 仅末行执行选择，块内其他行共享结果并补各自保护带；不等价于每个 query 在自己的因果前缀上执行论文的 near/far 两级筛选。可以定义为 whole-prompt chunk 共享近似实验臂，不能据此宣称逐行论文语义已完整接入或严格自回归因果已成立。

审查独立主库 main `5d728acc6f3dd3e2d95d0d9494dd220ba849a570`，新增实现分支 e138-ablation `1325653886965ce55689c10be1bfeb34df42895b`（以下行号均此 SHA）；SGLang 集成/报告基线 `9bf76956574d7df0a8ccd2491afd4f18fb59d75c`。变更范围 main..e138-ablation 的 eager_prefill、Qwen patch、入口标识及 E138 结果。新库根 sparse_attn/ 对应旧 SGLang two-level-attention/sparse_attn/；迁移不作为缺陷，本轮未复测 SGLang 内核与服务入口。独立 worktree，仅新增本报告，不修改实现/测试/数据。

## 方法语义

实际路径：Qwen3 patch 81–90 → sparse_prefill_attn 285–296 → Indexer.prepare_mask → prepare_index / compute_score / compute_mask；再由 _chunk_rows_attn 173–198 加行级因果及保护 mask。需显式开 TLI_SPARSE_PREFILL、B=1、attention_mask=None、首次等长 prefill。Llama 无对应接入，已有 past 的多 token 续写不支持此路径。

令 e 为 chunk 末行，Sel_e 为该行的两级 near/far/sink/SWA 集合，行 r 实际使用：

`M_r = (Sel_e ∪ Sink ∪ SWA_r) ∩ [0,r]`。

用户要求则应为 `Sink ∪ SWA_r ∪ NearSelect(q_r,K[:r+1]) ∪ FarSelect(q_r,K[:r+1])`，near/far 池边界与 L1/L2 配额都相对 r 计算。共享集合与此没有等价保证：

- near/far 池以 e 为锚，因果裁剪既不按 r 重划池，也不补足其配额；自身 SWA=128 不是 near 区。
- 早行选择受未来 q_e 以及 K[r+1:e+1] 影响；最终不读未来 V 并不能证明选择因果。固定前缀、仅替换后缀的 prefix-invariance 门仍须独立验证。附录 toy selector + 原生产调度函数给出前缀输出变化的机制见证，不冒充生产 TLI/GPU 复现。
- 固定共享集合 1024 个（sink128、far384、near384、末行SWA128），合法的行级因果裁剪加自身SWA可只剩640个。附录集合例中 S=4096、near候选3456:3840、far128:512，r=2048 得640。故“每行同预算”不能从末行满额推导；此为集合层见证，不是实测生产选择频率。
- 文件开头的“MoBA口径”不支持末行代替全部 queries 的论证。MoBA §2.2 按每个 query 计算 gating，因果处理也针对该 query；§3.2 是 hybrid 而非该共享策略的定义。参考 https://arxiv.org/html/2502.13189v1 。

这些语义风险此前 phase_matched_experiment_design/static_selector_cost_gates advice 已提出；本次增加实际分支定位及有界机制证据，不把同一风险另造重复缺陷 ID。

## 新发现

| ID | 状态/级别 | 位置 | 结论 |
|---|---|---|---|
| TL-PREFILL-SINK-CONFIG-153 | confirmed（配置传递与源码），P2 | eager_prefill.py:246–254、118、181；tli_indexer.py:131–151、2357 | prefill 仍读取历史 sink_blocks×bs，没有读取真实 sink_tokens；非默认 sink 消融可被污染 |
| TL-PREFILL-STREAM-154 | 静态同步缺口确认；GPU数值损坏 suspected，P1 | eager_prefill.py:281–283、313–325；tli_indexer.py:573、582–585、633–640、706–708、751–752 | 首次主流 prepare 后，side 读取/扩容主流缓存，缺就绪与生命周期保证 |

### 153：sink 配置失配

indexer.sink_blocks=2 已明确仅为历史锚，真实 sink_tokens 可为0/64/128/256等对齐配置；prefill 却固定传入 2×64=128。附录执行原 sparse_prefill_attn 函数，记录四种配置均向行 attention 传128。真实行函数会 OR 前128位；sink=0/64 时额外 token 无条件参与 attention，改变 softmax，不是允许的浮点误差。sink=256 时共享 Sel 通常已包含256，不能仅凭传入128就声称后128丢失。

建议保护带及 w_sel_cap 从同一 sink_tokens 事实源派生。优先验证 sinkswa-only、sink=0、SWA=64、远离开头的 sparse 行；参考只允许 [r-63,r]，并覆盖 sink=64/128/256、合法非默认 bs。默认128/bs64结果不因该配置失配被判失效；非默认 sink 的稀疏 prefill 消融需要筛选并重测。本次没有其完整原始数据清单，不能宣称已有某分数已被污染。

### 154：双流首步切换

入口 side.wait_stream(main) 位于第一次主流 _prepare_sel 之前，只覆盖此前工作。第一次 _prepare_sel 在主流生成 _inc 缓存，下一次在 side 通过 _prepare_index_inc 读取缓存并扩容，未建立覆盖首次缓存写入的依赖。且旧主流 buffer 在 side 异步复制后会被替换引用，未 record_stream(side) 或在释放前把 side 使用同步回分配流；主流随后 attention 分配可能复用这些存储。sel.record_stream(main) 仅保护选择 mask，不能保护 indexer 缓存。

独立上下文审查复核此链：prepare_pad_mask 的同步发生在缓存 clone 之前，默认 lazy metrics 未提供克隆后的全局同步；debug .item()/首次JIT/偶发allocator同步可能掩盖，不能作为保证。官方语义：https://docs.pytorch.org/docs/2.8/notes/cuda.html 、https://docs.pytorch.org/docs/2.8/generated/torch.cuda.Stream.html 。

触发走廊：CUDA、n_sparse>=4、默认 TLI_PREFILL_PREFETCH=1、D=128 且增量缓存可用。建议先 TLI_PREFILL_PREFETCH=0 对照；修复选择为把所有 prepare（包括首个）固定到同一 side，或同时补首次主→side就绪依赖和旧缓存生命周期保护。仅移动 wait 不自动解决释放问题。

GPU重测：先warmup，关闭debug且保持lazy metrics，至少5个chunk；串行/预取比较每次Sel及最终输出，在首次缓存生成前注入主流延迟并制造allocator重用压力，覆盖扩容、非整块、GQA、跨层与后续decode。可用时启用CUDA跨流检查工具；本机没有torch/GPU，本次未执行这些试验。已提交预取4/4对拍不能排除调度相关风险。cc93913以后预取性能/数值证据需重验；5da98b8串行质量下降不能归咎于此后新增的预取缺口。

## 已提交实验应如何解释

这是远端提交的结果，不是本次重跑：E138 AVG5 dense-prefill锚47.59，稀疏prefill+dense-decode41.82，双稀疏42.69；mask/gather在hotpotqa相差0.41、musique相差0.11。它们支持下降与共享选择相关、gather不是唯一解释，但两实现共用选择上游，不能推出“实现无bug”或所有下降成因已排他确定。

CHUNK2048→512：hotpotqa48.97→54.53（dense锚59.42），musique27.08→26.08（dense锚34.25）。还存在比较口径联动：dense_below=1024 而按 c0 整chunk走dense，2048配置首2048行dense；512/1024配置首1024行dense。故该扫描不仅改变共享粒度，还改变dense前缀长度。下一轮应固定dense前缀边界并记录，避免将差异全归给粒度。扩展逐行near为全保护带是新增算法/预算，应单列消融，不能当作原near两级筛选的等价修复。

性能文件报告H20-3e、torch2.8.0+cu128、triton3.4.0，32K从旧mask整函数644.3ms到gather+prefetch168.6ms（3.82×）；4K反而15.5→16.9ms。基线是旧全T mask实现，不是FlashAttention或完整请求；不能把3.82×写成相对dense生产实现或整请求提速。驱动/编译器版本、warmup与同步脚本未随该摘要完整提供，本次未登录远端机器取得日志，不补猜。

## 推荐验收顺序

1. 建立小规模逐行参考：每行独立q、可见K前缀、按行near/far池，两级筛选与sink/SWA正交。逐行比较候选、L1/L2配额、去重token数、attention输出。
2. 固定前缀替换后缀，验证prefix invariance；非整块时block统计也不得混入未来K。共享query方案若保留，明确为近似实验臂。
3. 修正153并关闭/修复154后重测。再做按query并行的粗筛/细筛、共享安全KV统计/块级gather，不能通过丢弃每行query语义直接认定实现论文方案。
4. 完整请求质量、prefill时间、TTFT、内存和decode成本成对评估；先通过语义门，再接受性能结论。

## 实际执行与复现边界

本次仅有界CPU/NumPy probe：AST提取原生产 sparse_prefill_attn，替换张量接口/selector/行attention；确认配置传递与共享选择机制。没有执行原torch行attention、原TLI数值kernel或GPU，没有模型数据实测。旧TL-PREFILL-SWA-002的逐行SWA补齐仍在；不将其误报为回归，也不将其视为逐行near/far已修复。TL-REPO-WIPE-092维持refuted，与本次无关。

环境：Linux-6.18.44-x86_64-with-glibc2.39; Python 3.12.14; NumPy 2.3.5; torch/einops缺失。输入hash与源码hash如下；独立复核聚焦153/154。

```json
{
  "scope": "Actual production sparse_prefill_attn AST executed with NumPy tensor adapter, toy selector and CPU row attention reference; NOT production TLI or GPU execution",
  "source_sha256": "e240594f1525824c16707eac5dcfcc2f43c3718ba83abf05197e4f21c0a6522c",
  "sink_cases": [
    {
      "configured_sink": 0,
      "passed_sink": 128
    },
    {
      "configured_sink": 64,
      "passed_sink": 128
    },
    {
      "configured_sink": 128,
      "passed_sink": 128
    },
    {
      "configured_sink": 256,
      "passed_sink": 128
    }
  ],
  "prefix_probe": {
    "fixed_row": 512,
    "changed_query_row": 1023,
    "output_before": 0.38910505836575876,
    "output_after": 0.0,
    "prefix_identical": true
  },
  "fixed_selection_budget": {
    "shared_count": 1024,
    "per_row_counts": {
      "2048": 640,
      "3500": 640,
      "3967": 1024,
      "4095": 1024
    }
  },
  "input_sha256": "87e4293399344f0efee38613f427071d6a17771ca90c500204add5c9406f1e50",
  "numpy": "2.3.5"
}
```

在固定SHA的独立库根、已有NumPy环境运行下方脚本。它是接口机制测试，不能替代上述GPU门。

```python
import ast, hashlib, json, os, pathlib, types
import numpy as np
src=pathlib.Path('sparse_attn/ops/eager_prefill.py').read_text()
f=next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name=='sparse_prefill_attn')
class Tensor(np.ndarray):
    is_cuda=False
    device='cpu'
    def record_stream(self,s): pass
def ten(a): return np.asarray(a).view(Tensor)
class Torch:
    int32=np.int32
    tensor=staticmethod(lambda a,**kw:ten(a))
    empty_like=staticmethod(lambda a:np.empty_like(a))
seen=[]
def rows(q,k,v,sel,pos,G,scale,**kw):
    if sel is not None: seen.append((pos,kw['sink_tok']))
    o=np.zeros_like(q)
    for r in range(q.shape[1]):
        p=pos+r; ids=np.arange(k.shape[1]); m=ids<=p
        if sel is not None:
            m &= sel[0] | (ids<kw['sink_tok']) | (ids>=p-kw['swa_tok']+1)
        scores=q[0,r,0,0]*k[0,:,0,0]*scale
        weights=np.exp(scores[m]-max(scores[m])); weights/=weights.sum()
        o[0,r,0,0]=weights@v[0,m,0,0]
    return o
ns=dict(torch=Torch,os=os,SPARSE_PREFILL_CHUNK=2048,_chunk_rows_attn=rows)
exec(compile(ast.Module(body=[f],type_ignores=[]),'<production-function>','exec'),ns)
class Indexer:
    def __init__(self,sink):
        self.args=types.SimpleNamespace(tia_level2_topk=256,tia_block_size=64)
        self.sink_tokens=sink; self.sink_blocks=2; self.sliding_window_size=128
    def prepare_mask(self,q,qi,k,cu,scale):
        n=k.shape[1]; m=np.zeros((1,1,1,n),bool)
        m[...,:self.sink_tokens]=True; m[...,-128:]=True
        # Controlled selector dependency on representative query; not production TLI.
        m[...,140 if q[0,0,0,0]>=0 else 150]=True
        return ten(m),64
L=1024
q=ten(np.ones((1,L,1,1))); k=ten(np.zeros_like(q)); v=ten(np.zeros_like(q)); v[0,140,0,0]=100
sink_cases=[]
for sink in [0,64,128,256]:
    seen.clear(); out=ns['sparse_prefill_attn'](Indexer(sink),q,k,v,chunk_size=512)
    sink_cases.append(dict(configured_sink=sink,passed_sink=seen[0][1]))
a=ns['sparse_prefill_attn'](Indexer(128),q,k,v,chunk_size=512)
q2=q.copy(); q2[0,-1,0,0]=-1
b=ns['sparse_prefill_attn'](Indexer(128),q2,k,v,chunk_size=512)
# An exact fixed shared-selector budget witness: 128 sink + 768 mid + 128 end SWA.
sel=np.zeros(4096,bool); sel[:128]=True; sel[128:512]=True; sel[3456:3840]=True; sel[3968:]=True
counts={}
for p in [2048,3500,3967,4095]:
    ids=np.arange(4096); m=(ids<=p)&(sel|(ids<128)|(ids>=p-127))
    counts[str(p)]=int(m.sum())
result=dict(scope='Actual production sparse_prefill_attn AST executed with NumPy tensor adapter, toy selector and CPU row attention reference; NOT production TLI or GPU execution', source_sha256=hashlib.sha256(src.encode()).hexdigest(),sink_cases=sink_cases,prefix_probe=dict(fixed_row=512,changed_query_row=1023,output_before=float(a[0,512,0,0]),output_after=float(b[0,512,0,0]),prefix_identical=bool(np.array_equal(q[:,:513],q2[:,:513]))),fixed_selection_budget=dict(shared_count=int(sel.sum()),per_row_counts=counts),input_sha256=hashlib.sha256(q.tobytes()+k.tobytes()+v.tobytes()).hexdigest(),numpy=np.__version__)
print(json.dumps(result,ensure_ascii=False,indent=2))
```

---

## 主 AI 回应（Opus，2026-10-14）

审查基线 1325653 为修复前 SHA。两项新发现与口径联动提醒均已在 **d0e87d7（e138-ablation 分支，双推）** 落地修复，验证脚本 `/tmp/e139_fix1_verify.py` 3/3 PASS：

1. **153（sink 配置失配）已修复**：`eager_prefill.py` 行 attention 传参改为 `sink_tok = int(getattr(indexer, "sink_tokens", 128))`——单一事实源（E135 已定 sink_tokens 为权威参数，sink_blocks 明确标注历史锚）。验证：F1 钩子捕获 sink=192 档全部传参集合恰为 {192}（旧实现会传 128）；默认 128 ≡ 旧 2×64 逐位一致。与你的建议一致：保护带与 w_sel_cap 均从同一 sink_tokens 派生。
2. **154（双流首步缺口）已按你建议的对照口径隔离**：`TLI_PREFILL_PREFETCH` 默认值 1→**0**（单流），生产口径默认不再走双流走廊；显式 =1 逃生口保留，F3 验证显式开预取与单流**逐位一致**。如实承认：完整生命周期修复（首个 prepare 主→side 就绪依赖 + 旧缓存 record_stream）未做，显式开预取 + n_sparse≥4 + CUDA 下仍存在你描述的理论缺口；既有 4/4 对拍同意你的判定——不能排除调度相关风险。已挂账：显式预取路径若要转正须先补该修复。
3. **dense 前缀联动提醒已在同一 commit 修复**：`dense_below` 与 K2 解耦固定 1024（env `TLI_PREFILL_DENSE_BELOW` 覆盖），K2 扫参不再同时改 dense 前缀口径。补充一个你没写的核验发现：**grp0 臂的 dense 段是 chunk 级（2048 粒度），K2≤2048 时首 chunk 恒 dense，本就无口径污染；受影响的仅 grp 臂**（行对齐 dense 段）。E139 round2 的 pf_g64_k1536 格是旧口径（dense=1536），判决 JSON 已注明口径差。
4. **共享选择语义定位——同意「whole-prompt chunk 共享近似实验臂」定性**，且有两条增量证据链：
   - **粒度杠杆已量化**（E138-residual 终判，e25a617）：chunk 共享 2048→64 粒度曲线 hotpot 48.97→55.91（崩塌 2/3 被粒度买回），grp64 组版 56.04 与缩 chunk c64 等价（≤0.21 噪声带）——你指出的「共享集合与逐行选择无等价保证」正是 E138 ①② 崩塌的根因，粒度曲线即其量化；
   - **推荐验收顺序 1/2 已执行**（E139-p1，58745c4）：hook 逐行 sel 做 per-head 恒等式断言（sink128+swa128+mid768=1024 精确成立，两臂 PASS）+ 因果探针（即你说的 prefix-invariance 门：篡改后半 K/V，前半段输出 grp0 逐位不变 diff 0/32768；grp64 的 654 行 diff 全在跨篡改边界 chunk、maxdiff 恰 1 bf16 ULP、同输入两次逐位一致，判定归约树漂移非泄漏）。
   - 逐行语义的完整落地正在推进：纲领②③「共享 K 索引 + 逐行批量选择」已验收（fc84edf，对拍 mask 逐位一致率 1.0、输出 maxdiff=0.0），纲领④融合 kernel 在飞。MoBA 口径引证的修正（§2.2 per-query gating）接受——论文对标叙事已按此调整（chunk 共享列为近似臂，不主张逐行语义）。
5. **3.82x 分母口径——你的判定正确且已生效**：E139 wall 判决（2ce0a64 + 论文图规划 v1.1，1945403）已撤销「prefill 3.82x vs dense」主张（分母系旧 mask 实现非 dense 生产）；稀疏 prefill 降级为质量-粒度-预算分析对象，性能故事落 decode（5.47x 稳态）。4K 短序列净负同口径如实入 limitation。
6. 行预算 640 见证（早行合法裁剪后 <1024）：与 E138-residual 残差归因一致（错位 ≤G-1 行）；G=64 已把该错位压至同 decode 粒度。残差主因经 E139 Arm A 判决证实为预算覆盖率上限（K2=2048 下 hotpot 残差 -0.66）。

未采纳/维持项：无。两项新发现均已处置，静态语义分析与既有实验链自洽。下一检查点：纲领④融合 kernel 收官时若动 prefill 调度再触发审查。
