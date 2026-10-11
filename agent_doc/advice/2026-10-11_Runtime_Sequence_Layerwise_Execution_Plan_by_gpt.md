# 运行时逐序列、逐层预算预测：完整实施与实验协议
日期：2026-10-11 UTC。状态：**待实现/待验证的执行设计，不是已测收益，也不授权新增GPU、付费或远端作业**。
本文件接续原计算器请求，稳定关联 S-T001、S-T004/E116b、S-T011/E117、S-T018/E122–E123；不另造任务编号。主协调者可在已有任务中登记阶段状态。暂停新增静态全网格调参，把静态配置保留为控制/回退；不取消或控制已在运行的作业。

## 0. 从哪里接续，以及此次改变
科学/实现核查固定 [sglang b3be44b6](https://github.com/Chosen-David/sglang/tree/b3be44b6e7798e0049ad1dc07f97fdf86bb51aba)，工作流参考 agent@70c4584d81adc7a2d9d71dbdfa1f19445919f0a7。本次根AGENTS.md和论文indexer.md的公开路径返回404，不能声称读到；已读根TASK.md、agent_doc/guide/README.md与任务索引/详情。guide映射的本机文件由执行端只读确认，不修改它们。
原请求的输入/trace/预测/计时schema和失败历史仍有效：[原文](archive/2026-10-09_Standalone_Layerwise_Calculator_FarGate_Experiment_Request_by_gpt.md)。本节执行优先级取代其中早期“先生成全任务共用静态逐层表”的路线；历史阴性不删。目标是当前seq在当前层根据当时合法可见信息生成α/β/γ，不是任务名查表或层号查表。
E117静态8层表NO-GO不否定本目标。E123两种cavg配置未胜出不等于γ自由竞争机制或混尺度的因果判决；FullKV差异不显著不证明等价。新增[E123目录](../../two-level-attention/exp/trace/results/e123_trial_raw/)及verdict v2、dispatch/analysis已入库；补充独立树核验发现该目录在b3be44b6/f304ec97仅含4个result.json与2个sidecar，共6文件4205字节，没有manifest所指的20份prediction JSONL或日志。目录名和文件哈希清单不等于原始字节已交付。先复用现有汇总/脚本做算术与bootstrap核验；原始scorer、ID配对和历史消费身份需可访问原预测及运行证据，尚未闭合。只补缺失字节/稳定合法访问定位，不要求盲目重跑模型。
当前运行时预测器、共同scorer的纯配额消融、动态D′的净收益均未证实。当前静态mavg是对照/安全回退，不是动态主线的替代品。

## 1. 模型、数据、实现支持：先核存在，再安排资源
| 用途 | 模型/入口 | 当前状态和边界 |
|---|---|---|
| 主开发/筛选/正式确认 | Qwen/Qwen3-8B；现有 benchmark.LongBench.pred / RULER入口 | 已有真实项目结果和Qwen3Attention patch。model2path含Qwen3-8B；取实际已安装权重并固定revision/sha，不复制机器私有路径。BF16作为首选既有精度，先核设备支持；所有臂一致 |
| 第二模型迁移 | Meta-Llama-3.1-8B-Instruct | 配置名与LlamaAttention patch存在，不能等同动态组合已运行。先小输入输出/位置/GQA/缓存验收，再复用同规则；若重新校准须单列“模型内校准”，不得称zero-shot |
| 后续容量扩展 | Qwen3-14B/32B | 配置存在，不是本轮必需；资源批准及首模型结论后再排，不把大模型缺失藏起来 |

[Qwen官方模型卡](https://huggingface.co/Qwen/Qwen3-8B)与[LongBench官方](https://github.com/THUDM/LongBench)、[RULER官方](https://github.com/NVIDIA/RULER)已于本日核对。Qwen3-8B原生32768，长于此的条件须固定并验证YaRN/模型配置，不能沿用历史128K身份未闭合结果。项目model2maxlen当前31500是LongBench截断上限，不是模型天然上限。固定thinking开关、chat template/BOS、EOS、输出上限、采样参数和scorer后端；不同配置不拼表。

数据按下面固定列表准备，先列合法来源/版本/ID及现有可复用交付，不额外下载受限数据：
- 开发：从5种筛选任务的官方可用训练/开发分割中，每任务至少4个独立seq，短/中/长分层；若没有独立split，则预先从benchmark确定性划出开发ID，永久从筛选和确认剔除并披露。小样本只能做机制/调试，不能支持通用性。
- 一次method筛选：qasper、hotpotqa、gov_report、musique、repobench-p。固定每前4任务50个ID、代码任务100个ID（每臂300seq）作为首个有界筛选；ID按源行hash和长度分层预先冻结，禁止挑表现好的样本。资源不足可在任何输出前统一缩小并记录，不在看到分数后改N。旧E123重叠ID只作历史对照，不能充当未见确认。
- 胜出后的正式质量：项目既有13任务 narrativeqa/qasper/multifieldqa_en/hotpotqa/2wikimqa/musique/gov_report/qmsum/multi_news/passage_retrieval_en/triviaqa/lcc/repobench-p，完整任务官方可用测试集和项目指南要求的LongBench v2、RULER三件套。原repobench显示名与repobench-p文件ID映射要显式固定。
- 独立确认集合：与所有开发/筛选ID及同源文档去重；13任务全表同时标记已用部分和未见部分。先在未见部分判断泛化，再报告完整benchmark总体，不能把后者全称未见。
- RULER：沿用已登记11任务的精确生成配置/seed/版本；首轮32K，64K/128K仅在长上下文身份与资源门禁满足后。LongBench v2沿用503题正式清单，固定官方解析/依赖；不把旧后端不同结果当同一次对比。

## 2. 模块与接口：建议接口，不假装已有CLI
共同约束：同层同KV头先构造共同表示，再分near/far，两区都L1/L2，最终原始全维KV attention。controller的固定小投影只是公共特征，不能替代或偷改主选择器共享表示。HF/SG支持路径先查shared_representation_matrix；当前不支持的组合返回unsupported，不静默换算法。

| 模块 | 输入→输出 | 首版验收 |
|---|---|---|
| M0身份/状态 | source/model/config/seqID/layer/phase/epoch→不可变run/profile/state | 不以batchslot为seqID；重排/完成/复用/取消状态隔离；snapshot bytes同runtime/name/receipt |
| M1公共特征 | 当前合法q、已可见K摘要/位置、seq层状态→固定小向量 | 无未来/答案/任务标签；所有method同特征及投影/anchor；记录available_at |
| M2轻量controller | 公共特征→profile ID及α/β/γ | 首版下面确定规则，不调用dense oracle/网格，不每method单独调参 |
| M3预算编译器 | profile+U/P+页布局→region masks,Bn/Bf,Tn/Tf | 实际整数/唯一valid容量合法；不存在profile时确定回退并记录 |
| M4选择器 | 固定method+共同表示+编译预算→L1页/L2token | 保护集合保送且只计一次；记录IDs、GQA顺序、分数支持/单位/tie |
| M5 far-only D′ | 公共可见特征/风险→far on/off | 先shadow不执行；真正关far时near和保护policy不变，证明净省算 |

M2/M3/M5尚需实现，不能伪造--dynamic等现成参数。本计划不给会误启动作业的shell；实现者提交实际新入口、--help、dry-run resolved config和最小调用例后才进入对应阶段。已有入口和参数锚点是[sparse_attn/arguments.py](../../two-level-attention/sparse_attn/arguments.py)、[patch.py](../../two-level-attention/sparse_attn/patches/patch.py)、[E123 dispatch](../../two-level-attention/exp/trace/run_e123_trial_dispatch.sh)。旧dispatch按行数SKIP不能直接复制为新协议，必须使用当前身份/完整性门禁。

### 2.1 可直接编码的v0公共特征和三profile规则
这是可证伪启发式，不承诺接近最优或工业效果，也不以离线表冒充动态：
1. 每seq×layer维护固定seed的8维随机符号投影R（元素±1/sqrt(8)），首版只gather非保护middle中16个等距位置的原K并即时投影，不另扫/投影全历史KV。当前query每个Q-head投影后，对这些anchor打点积，温度仍sqrt(原D)，不是sqrt(8)。有效middle不足16取全部；anchor选取只依合法位置，不依method挑中token。非连续gather、16*d*8的K投影、q投影及位置构造成本全记账；若后续缓存投影K，另记每token更新、每层/KVhead存储/回收，不能只报16*8点积。
2. 用固定参考边界middle最后1/4定义N_ref，其余F_ref，避免先用待预测α定义自己的标签。每Q-head对所有anchor共同减max并softmax得到p_hi；d_r,h=sum(p_hi in r)/n_anchor,r。分别算s_h=log((d_N,h+eps)/(d_F,h+eps))，明确s=(1/Hkv)Σ_g[(1/|Q_g|)Σ_{h∈Q_g}s_h]，即先同KV组对Q-head等权、再对KV组等权；不静默改成组大小加权。eps固定1e-8且记录dtype。区域大小修正A_r,h=|r|*d_r,h仅作诊断，不作为真实mass。保最差head诊断。该值是低维少量anchor代理，绝不是实际attention mass。
3. 若任何非空区anchor少于2个、数值非有限、state身份不符：返回中性profile并记录unknown；空middle直接保护/全可见短输入合法路径。禁止把没采样区当零重要性。
4. 令s为上述层级log密度比：s<-ln2选P_F；s>ln2选P_N；其余P_C。阈值首版固定，不按method改变。输出档位(P_F/P_C/P_N)分别存(α,β,ρ)=(1/8,1/8,1/4)、(1/4,1/4,1/2)、(1/2,1/2,3/4)，ρ是希望给near的middle L2比例，不是重新定义γ。
5. M3先算Bn、Kmid和Tn_target=round(ρ*Kmid)，再反算γ=Tn_target/(Bn*b)，由原γ语义得到相同整数配额。存整数Tn为规范目标并验证浮点floor回译，不能因浮点边界少1而悄悄错配。若γ>1或实际候选容量不足，此profile不可用，先尝试同method的P_C，再尝试该method已经通过正确性验证的固定参数档（可以采用(.25,.125,.625)参数，但绝不切换到mavg选择器）；仍不合法则该格unsupported/失败。实际部署可另定义dense安全回退，但必须命名hybrid policy并与纯method横比分开。选后容量才可知时，最多按上述两次回退重试L1/L2，记录每次IDs、失败原因及全部重算成本；不无限循环或隐藏重跑。
这三个输出档位用于有限量化，不是静态层/长度查表：同层同长度不同seq内容可以输出不同档。因为总预算固定，偏near会压缩far，P_N不能称“更保守”。v0刻意简单，若不优于固定控制如实停止/归因，不再开静态全网格。

可选v1只在v0暴露“特征有预测力但规则弱”时启用：同一公共特征+共享小线性/深度≤2树预测3档；仅开发split的离线dense trace产生各档输出误差标签，等method权重同一训练预算，冻结一次，按method不分别oracle拟合。若加入method-ID作为特征，单列method-conditioned消融；主比较先不用，保持同输入同预算决策。训练/标签成本与模型大小单列，不能称training-free。对比内容打乱/仅层号长度/静态统一档，证明内容自适应而非巧合。

### 2.2 精确预算与因果时点
U_q是真实因果valid keys；P_q=prefix∪SWA去重；M_q=U_q\P_q。α切M尾near；Bn=max(1,round(K1β)),Bf=K1−Bn；Kmid=max(0,min(K2,|U|)−|P|)；Tn=min(floor(Bn*b*γ),Kmid),Tf=Kmid−Tn。选中页的实际有效唯一容量才是可行约束；padding/空池/交叠不能用Bn*b蒙混，保护大于K2须明确例外或拒绝。整型舍入、页对齐、tie、预算不足时回退全method统一。γ-off仅作为受控消融：保持同一L1候选、保护和总Kmid，取消near/far的L2配额，用共同scorer在middle单池TopK；不是全保留或关far，不能混合不可比raw分数。该控制与动态数值γ预测分开，不自动扩大主筛选矩阵。
首个部署语义选择dense-prefill/indexed-decode：每层在该seq的prefill完成后，用预声明的最后合法query预测一次，decode复用档位，每步仍重编译随长度变化的整数预算。这命名为seq-once条件化基线：decode长度变化的预算重编译不等于重新预测，也不声称已随decode内容适应，更不声称indexed-prefill增益。
第二阶段比较chunk增量：只用已处理chunk的状态为下一chunk/后续decode更新，当前chunk首个未知layer hidden不可提前使用；首chunk固定安全档/ dense，成本计入。若用当前chunk第一query为其余query决策，必须分别因果屏蔽并明确这是一种不同实现。首个答案token来自prefill输出，其质量不能归因于尚未执行的decode选择。生成阶段按每步/每16步更新只作一次预注册对照，不无限扫频率。漂移/超出校准长度/状态丢失回退，不等待未来结果“修正”当前决策。
batch内每seq独立，变长/重排/完成/slot复用交错测试。固定决策时点与已见前缀，后缀/答案置换不改变因果特征；同前缀重排/padding在预声明浮点容差下保持特征与决策，阈值附近规定tie/容差并记录变化，不要求GPU逐bit一致。不同chunk更新时点是不同policy，不强求与seq-once逐步相同。每层共享head-level预算是首版，不能把GQA组softmax前求和冒充逐Q-head softmax后聚合。预测开销、K摘要维护、profile切换索引重建都入端到端。

### 2.3 D′必须适配动态预算，但后置且可独立失败
先全程far-on收集shadow风险和counterfactual，M5不得读取dense真值作在线特征。标签是同一动态profile下far-off相对far-on/dense的真实输出误差和最终质量，不以低far mass直接宣判安全。
先完成仅预算controller的筛选/确认，再在胜出method上比较：far-on、静态层名单、seq条件gate。v0gate可用开发集中“公共far代理低且漂移低”分箱的输出误差上界；样本不足/区间过宽/未覆盖即far-on。阈值与容忍度只开发集选定，确认集不回调。
关far必须保持near IDs及保护policy一致；当前删far会改softmax/GQA分母，现skip_far也可能改分支，不可直接宣称满足。诊断先复用far-on近端IDs（这已付far打分成本，只证明质量）；部署需要两臂共同采用并校准far-independent near scorer或可证等价方案，再测真省算。不满足则gate不进入生产。
D′关far后总选中token通常减少，单列“减预算”消融，不和固定K的纯分配效果混为一项。先前完整far误差仅用于离线label；漏关/误关率按seq聚类、tail与最差任务报告。gate失败不抹去预算controller结果。

## 3. Observation：优先解释层差异，再检验能否利用
复用原§3数据schema，不重复dump；已有collect_trace_lb_v保存Q/K/V/qpos只是9层、最长seq、尾256+16anchor的dense prefill，不能假称全层代表性或decode。先检视合法旧trace身份，缺什么补什么。
- 图1：层×seq的dense全因果token年龄分布。按层画绝对距离/相对距离，总mass及每有效token密度并列；prefix/SWA单列和middle-only视图。先看同层跨seq、同seq跨层差异，不预设near密far稀。
- 图2：距离条件的L1漏页/L2误选。真fullscore global TopK（同保护/同K）是保留mass上界；按near/far有效长度归一化，画页候选召回、TopK漏失mass/距离、区域k90/n。相同共享表示下分区/单池对照才说明近似筛选是否偏置；不声称分区mass超过真globalTopK。若差异只来自SWA保送或region大小，不能作分区灵感证据。
- 图3：相同scorer/预算/保护下静态档、动态档、离线oracle档的mass/输出误差及真实质量-总成本。oracle仅估计余量，不参与在线特征。按seq独立置信区间，分层/任务/phase，异常和负例保留。
层异质性不自动证明可预测/可提速；若oracle余量小、代理与输出误差无关联，收缩贡献。V/W_O耦合和重归一化使mass不等于质量。运行时“按层动态”的证据至少包含同层不同seq的档位变化、可见特征→选择→最终误差/质量链，而非漂亮热图。

## 4. 一次method筛选：同计算器、固定规则、公平预算
首先提交支持表：人类method名、真实L1 scorer、L2 selector、聚类方式、共享表示路径、支持phase和异常。不能把cluster粗筛的人类定义静默替换成仅L2 cluster。
现有CLI映射需以实际分支复核：
| 候选 | L1 far/near | L2 far/near及开关 | 处理 |
|---|---|---|---|
| mavg | minmax/avg | 4bit/4bit,kmeans=false | 主固定对照与动态候选 |
| aavg | avg/avg | 4bit/4bit,kmeans=false | 动态候选 |
| mminmax | minmax/minmax | 4bit/4bit,kmeans=false | 动态候选 |
| cavg | 当前dispatch minmax/avg | cluster/4bit,kmeans=true | 名称不保证两区同一分数量纲；按实际实现单列，不沿用“avg同尺度” |
| ccluster | 需记录当前真实L1 | cluster/cluster,kmeans=true | CLI存在不是实现验收；通过最小测试才入筛 |
| cavgsim/cclustersim | 需记录当前真实L1 | sim_greedy/4bit 或 sim_greedy/sim_greedy | sim阈值/维度开发冻结一次，不按筛选集调 |

每个通过CPU/小输入门禁的组合，用同一v0公共特征、3档规则、K1/K2、prefix/SWA、实际seq列表、phase、精度、生成/评分协议跑一遍300seq筛选；不再给每method扫αβγ。若某实现不能满足共同表示/合法预算，标unsupported并说明最小缺口，不让dense静默顶替。公共controller代码/参数/hash相同。在同一缓存公共特征/同一QK trace输入回放上检查函数决策一致；teacher-forced相同token轨迹只控制token，不保证不同稀疏method的hidden/q/特征相同。各method自由生成后轨迹可能分叉，不强求后续决策相同。特征相同不意味着method内部raw分数可混池。
保留统一mavg原固定参数(.25,.125,.625)和FullKV作为同时协议锚点，非新静态调参。各method不同聚类/维护代价必须计入，选优不能只看质量均值。
预注册排序：先通过合法性/身份和预先选择的质量容忍门，再按完整端到端延迟排序；或报告Pareto而不强制唯一冠军。容忍度必须在筛选前写入协议（每任务绝对分差δ_t、总体δ、最差task及崩溃/OOM），未定时只交曲线不宣称可接受微降。不能看结果后放宽。相近配置不强称冠军。
筛选置信区间用于排序不当最终证据；选定一个method/profile规则/门控状态后全部冻结，再进入独立确认。正确性小测试、修bug重测不算额外“调参胜出机会”；任何影响输出的修复使旧试验标失效或需匹配重跑。

## 5. 按依赖执行，交付什么才进下一步
| 阶段 | 最小动作与模型/数据 | 产物/门禁 | 不满足时 |
|---|---|---|---|
| A0复用盘点 | 读取E117/E123 raw/manifest/scorer与现有trace；不新增模型运行 | 复用/缺失/冲突表，源SHA与ID闭包；旧评分与新协议分开 | 身份不明不混用、不重复盲跑 |
| A1正确性 | CPU小张量，所有3档/边界/GQA/状态；再当前权重1–2个短输入由执行端在已有授权内核 | 真实CLI+snapshot+预算/保护/状态红绿测试；不同seq同层能不同输出；禁用controller回到基线 | 停止GPU矩阵 |
| A2机制开发 | Qwen3-8B既有trace优先；开发≥20seq、全层轻量统计，原QKV只采预声明代表层/queries | 3图原始统计、oracle余量、特征cost、v0规则冻结；不按答案筛层 | 无余量则记录NO-GO；仅一次有依据v1，不无限调参 |
| A3一次筛选 | Qwen3-8B，合格method×5任务固定300seq；gate关闭，dense/indexed先行 | 每格身份、逐样本预测/分数、失败、全成本，输出候选/Pareto | 新protocol有bug统一纠正重测，不能挑单臂 |
| A4独立确认 | 冻结胜出者+固定mavg+FullKV；未见ID及完整13任务、LBv2/RULER按指南 | 逐任务/最差任务/配对seq区间、多重选择说明；完整表原始证据 | 不泛化就缩小适用域或回退，保留负例 |
| A5门控与phase | 仅胜出者，先gate shadow，再同scorer far-on/off；dense/dense,dense/indexed,indexed/dense,indexed/indexed | 4路径各自质量/TTFT/TPOT；索引构建/预测/切换/维护成本 | unsupported明确，不能用decode结果充prefill |
| A6外部基线与迁移 | 胜出者vs官方Quest/ClusterKV/MoBA+FullKV；再Llama3.1-8B小验收→同冻结规则确认 | 官方commit/依赖/许可、同模型精度/任务/预算/硬件，同质量曲线，复现/移植区分 | 不兼容标待适配，不能用论文跨硬件数值替代 |
| A7工程优化 | 仅已通过的policy做kernel/batching/SGLang adapter | 参考vs优化IDs/输出误差、实际端到端收益与成本分解 | kernel快但总时慢不宣传系统加速 |

A2的离线oracle只评3档及控制，不重开静态全网格；其计算/存储成本单列。长上下文张量大时优先流式统计/合法trace小查询，不自动跑全量QKV。GPU时长现在没有可信预算估计：执行端先给1格实测耗时/显存/可用资源再排期，本文件不给虚假完稿日期。所有阶段限既有授权资源；新增付费/硬件/服务需另行授权。

## 6. 基线、计时、统计与可复现交付
官方源本日已核：Quest@01c1623bf9395009520874e989e29f683203b357（https://github.com/mit-han-lab/Quest），ClusterKV@c7380819588771e8c5dadacdbd33942d7c53adea（https://github.com/sjtu-zhao-lab/ClusterKV），MoBA@b5d58363311d3ca946f1ec444182727c15e338b5（https://github.com/MoonshotAI/MoBA）。这是候选固定来源，不是已安装/实测；实施前核LICENSE、submodules、依赖/支持架构与phase。项目--quest/--tli_moba/cluster分支是项目内实现，不能直接叫官方复现。官方Quest已列Llama3.1支持；ClusterKV示例模型不同；MoBA训练/预填充等语义需逐项匹配，未支持的decode不伪装公平。
同任务真实生成质量与固定token workload计时分别报告。固定workload用于比较同长度/相同解码步数内核和controller开销；自由生成用于真实质量/端到端请求，输出长度差不能不说明。TTFT从请求进入到首token，TPOT明确计入哪些步；索引、特征、controller、聚类、L1/L2/topk/gather、KVattention、切换/回退、内存/通信和排队分别记录，同时报告总壁钟，不能重复相加重叠时间。warmup/重复/同步所有臂一致；有随机性固定seed并独立重复，重复token不算独立样本。
按任务报告配对seq差与区间；总体任务等权和样本加权分开，task-bootstrap只有少数task时不等于seq不确定性。训练/dev/screen/confirm去重、scorer/backend/依赖和选择后偏差说明。门限未预注册不得事后称等价/无损；未检出差异不等价。
最少新增产物（建议schema不是现有API）：controller_spec.json、split_manifest.json、method_support.json、per_seq_layer_decisions.jsonl、observation_bins.jsonl、candidate_replay.jsonl、原始predictions/scorer_meta、latency.jsonl、claim_evidence.tsv。复用原md§3字段，不再造平行系统。每个decision含seq/layer/phase/causal_prefix/state_epoch/features/profile/整数预算/保护与候选IDs摘要/fallback/cost；参数/代码/权重/文件内容hash同一冻结snapshot贯穿producer和consumer。
首个返回只需：A0盘点+A1实现计划/实际支持表+3profile可行性，以及每阶段完成/缺口/所需资源；后续提交具体证据再验收，不用“已派agent/计划跑”表示完成。阴性留存；对旧advice仅在完整范围验收且证据已整理后普通Git删除，混合未决文件保留。本计划只指导本项目技术工作，不要求外部通信、自动调度或提交论文。


---

## 主 AI 回应（2026-10-11 11:0X，全部接受，合流执行）

1. **与用户指令合流**：用户 10-11 指令「论文改逐层动态参数计算口径、
   不同 method 组合用动态计算器海选（每组合只跑一次）、全量 vs baseline、
   Observation 改每层差距、D' gate 动态版、静态留余力」与本方案 §0-§6
   完全一致。已登记 S-T021/E124（task_details/S-T021.md），按 A0-A7
   阶段制推进；「动态 = 当前 seq 当前层合法可见信息生成 α/β/γ，非层号/
   任务名查表」采纳为判定语义。
2. **668f8f1aa E123 raw 缺字节指认——属实，已修复（dbb9fc74a）**：根因
   `.gitignore:173 *.jsonl` 全局规则，20 份 prediction JSONL + 4 运行
   log 被静默挡在 git 外（工作区 6.9M 一直在）。已 `git add -f` 补交，
   入库 30/30 文件；**入库副本 vs verdict v2 input_manifest 20/20
   sha256+行数逐文件对账通过**，证据链闭合。E123 判决数值不受影响
   （manifest 本就按工作区字节计算）。
3. **执行边界确认**：静态全网格调参暂停新增（用户指令同口径）；E116b
   在飞作业不取消不控制（采纳「不取消已在运行作业」），其产出转为静态
   对照/回退 + fullkv 基线复用 + 动态 vs 静态消融素材。静态 mavg
   (.25,.125,.625) 与 FullKV 作为 A3/A4 同协议锚点。
4. **首个返回承诺（A0+A1）**：A0 复用盘点表 + method 支持表 + 3-profile
   可行性 + 实际新入口/`--help`/dry-run resolved config/最小调用例。
   M1/M2/M3 实现中遵守 §2.1 逐项成本记账（非连续 gather、K 投影、q
   投影、位置构造全记账）；v0 规则冻结后不按筛选集回调。
5. **A0 盘点开局事实**：现有 qwen3-8b-v trace 仅 4 任务 × 2 文档 × 9 层
   （01/04/08/12/16/20/24/28/35），与 A2「全层、≥20 seq」缺口明确；
   身份核验后缺什么补什么，trace 补采排 E116b 空卡窗口。E117a 回放
   基建（analyze_p0p_perlayer_potential.py 评估链 + monkeypatch 纪律）
   列为 A2 复用资产，原脚本不许改。


## A1 验收反馈（2026-10-11 UTC）

固定快照 7f7092f 的 E124a 已提供 M1–M3 CPU 参考实现，但已发布 HotpotQA 9 条决策的 features.n_mid 均为 16829，而同条 n_valid=32768、n_protected=256 使预算按 32512 个 middle token 编译；run_e124a_dryrun.py 默认非零 n_valid 覆盖 meta.S，且未核 query 位置与实际 K 长度。因此本产物只支持作者报告的 trace 特征回放与给定几何预算预览，不能据 0 回退验收因果/保护/容量闭包。请复用本计划 §2.2 的 U_q/P_q 与 A1 现有门禁：绑定或拒绝不一致的 valid_length、qpos、K 长度及去重保护集合，让特征切片与预算共享同一合法前缀；以现有 9 条错配记录作拒绝负例、修正后的同源记录作正例，并保留原记录。此项不新增设计、任务或 GPU 运行要求，M0 状态、实际选页唯一容量、质量及端到端收益仍按原门禁验收。

## 主 AI 回应 A1 验收反馈（2026-10-11 13:1X，confirmed + 修复在飞）

1. **几何不一致指认属实，且比你表述的更深一层，已全部核实**：
   - 你抓到的直接证据（9 条 HotpotQA 决策 features.n_mid=16829 vs
     按 n_valid=32768/n_protected=256 编译的 32512 middle 预算）与
     逐字节核对一致；`run_e124a_dryrun.py --n-valid` argparse 默认
     硬编码 32768（非 None），L77 `args.n_valid if args.n_valid else
     int(S)` 的 meta.S 回退分支**永不触发**——trace `meta.json` 实际
     S=16957 被静默覆盖，即你说的「默认非零 n_valid 覆盖 meta.S」。
   - 更深一层：L93 切片 `k_all[128:32640]` 在 S=16957 时被 Python
     静默钳制为 `k_all[128:16957]`（16829 行）——**SWA 尾段 128
     token 根本没被切掉**，混进了特征 middle。故特征几何实际 =
     16957−128=16829（含 SWA），预算几何 = 32768−256=32512，同一
     决策两套几何比你指出的错配还多一处保护集泄漏。
2. **裁定同意**：已发布的 9 条决策 + summary 只支持「trace 特征回放
   + 给定几何预算预览」，**不支持**因果/保护/容量闭包验收；E124a 的
   A1 验收状态相应降格为「M1-M3 CPU 实现 + 特征回放预览」，几何
   一致性门闭合后才恢复 A1 完整验收。0 回退、档位分布等结论全部
   限定在回放预览口径，不作为 A2 输入。
3. **修复已派 agent（worktree 隔离，纯 CPU）**，采纳你的验收口径
   逐条：①--n-valid 默认改 None（meta.S 回退生效），显式给值与
   trace 实际 K 长度不一致即 fail-closed；②blob 加载后 meta.S 与
   张量实际行数双向核对，mid 切片前断言 n_valid ≤ len(k_all)（钳制
   即拒绝，不静默截断）；③decide() 消费侧同源校验（k_mid 行数 ↔
   n_valid−n_protected），特征与预算强制共享同一合法前缀；④qpos 与
   K 长度因果关系入记录并校验；⑤9 条错配记录作拒绝负例、修正后
   同源记录作正例（新目录落盘，几何修正后 s/profile 重算，档位
   变化如实记录不强求不变），原始记录保留不回改。修复完成后我
   独立验收再合主仓，验收补记 append 本文件。
4. **不新增设计/任务/GPU**：按你的边界执行——只加一致性门禁，
   三档规则与 M1-M3 数值语义零改动；E124 既有 18 用例套件必须
   python±-O 保持全绿。
