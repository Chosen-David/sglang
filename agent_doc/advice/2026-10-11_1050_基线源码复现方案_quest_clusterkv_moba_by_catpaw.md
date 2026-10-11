# 基线源码复现方案：Quest / ClusterKV / MoBA 接入 two-level-attention

- 日期：2026-10-11 10:50
- 作者：catpaw（应用户指令调研，供其他 AI 借鉴与评判）
- 状态：调研完成，待评审；执行需等 S-T019 让出 GPU 资源
- 关联：GUIDE.md「baseline 应该包含 Quest、clusterKV、MoBA 这些论文的数据以及真的拉取他们的源码跑出 baseline」；TASK.md 待办池 E120

---

## 0. TL;DR 结论表

| 基线 | 官方源码状态 | 摘取难度 | 精度对比路径 | 性能对比路径 | 最大风险 |
|---|---|---|---|---|---|
| **Quest** | ✅ 已克隆 `sparse-bench/third_party/quest` @ `01c1623b`（2025-07-10） | 中（精度路径纯 PyTorch 可直接摘；kernel 路径带 libraft/CUDA 重依赖） | ① 官方 HF monkeypatch（仅 llama/mistral，需移植 Qwen3）；② 我们 harness 已有 `quest` indexer 臂 | 官方 CUDA kernel（page 化 gather + fused decode）vs 我们的 kernel，`benchmark/efficiency/` 同台面 | 官方 kernel 编译链重（libraft、cmake≥3.26、flash-attn 2.6.3），与现环境 torch 2.8/cu128 兼容性未验证 |
| **MoBA** | ✅ 已下载 `sparse-bench/third_party/MoBA`（master tarball，2026-10-11） | **低**（`moba/` 仅 4 个 Python 文件，纯 PyTorch + flash-attn，无自研 CUDA） | 官方代码可直接包一层跑我们的 LongBench/RULER 数据；**口径警告见 §3.2** | 官方 `moba_attn_varlen`（flash-attn varlen 组合）即其性能实现，可与我们的 prefill 路径同数据对拍 | **官方自述需要续训，非 training-free drop-in**——论文口径必须写清 |
| **ClusterKV** | ❌ **官方仓库地址未确认**（本环境搜索引擎全被墙，已试渠道见 §3.3） | —（拿不到码） | **Plan B**：我们 harness 的 `cavg`/`ccluster` 臂本身就是「参考 ClusterKV 论文实现」的口径（GUIDE.md 明确定义） | 同左 | 若确实从未开源，论文里对比口径只能引用其论文数字 + 自实现口径，需措辞严谨 |

**一句话**：Quest 和 MoBA 的官方源码已经在我手里，随时可以摘；ClusterKV 官方代码大概率未开源（或地址待网络畅通后确认），精度对比用我们已有的 ClusterKV 口径臂兜底。

---

## 1. 现状盘点（我们已经有什么，避免重复造轮子）

### 1.1 我们自己的 harness 血缘与基线接口

- `two-level-attention/benchmark/LongBench/pred.py` **第 1 行注明**：`Modified from: https://github.com/mit-han-lab/Quest/blob/main/evaluation/LongBench/pred.py`——我们的 LongBench 入口本身就是 Quest 评测脚本的血统，数据流、任务列表、打分脚本（`eval.py`）与官方口径同源。
- 稀疏方法统一抽象在 `sparse_attn/indexer/base.py`，三接口契约：

```python
class Indexer:
    def prepare_index(self, k, cu_seqlens_k): ...       # 建索引（如 page 的 k_min/k_max）
    def compute_score(self, q, q_ids, index_dict, softmax_scale): ...  # 粗筛打分
    def compute_mask(self, q_ids, score_dict): ...      # 出选择 mask
    # prepare_mask() 串起三者 + metrics.add_select_result 记录选择统计
```

- 已注册的方法（`sparse_attn/indexer/__init__.py` + `arguments.py`）：`--method ∈ {quest, tia, twi, tli, none}`。**Quest 已经是 harness 内 indexer 臂**（`quest_indexer.py`：page 级 k_min/k_max、query 符号展开打分、top-k page 展开——严格 Quest 论文口径的公平重实现，decode 路径）。
- **MoBA 已是 harness 内 E89 臂**：`tli_indexer.py --tli_moba`（全维 chunk-mean gate 选 top-K 块全展开，`sparse_attn/ops/eager_prefill.py` 实现 MoBA 论文 §3.2 TopK Gating 的 chunk 共享选择语义），E89 已有 13 任务精度数据（49.19）。
- patch 注入点：`sparse_attn/patches/patch.py register_patch(model, args, snapshot)`，当前支持 `Qwen3Attention` / `LlamaAttention` 两类模块的 forward 替换。
- 评测三件套口径（S-T004/S-T019 固化）：LongBench 官方 `eval.py`、RULER `score_ruler` direct CLI、LB v2 官方 choice 抽取（E117b）。**任何新基线臂必须复用同一打分链，禁止自带 scorer**。

### 1.2 sparse-bench 已有资产（`~/sparse-bench/`，2026-09-21 建）

`third_party/` 已克隆：KVCache-Factory @ `68cd9551`（PyramidKV 团队的统一底座，集成 FullKV/SnapKV/H2O/PyramidKV/Quest 等 16 方法）、SnapKV、PyramidKV、quest、streaming-llm、Scissorhands、vllm、FlexLLMGen。**本次新增 MoBA**（tarball 解压，无 git 元数据——网络受限下 codeload 是唯一稳定通道，见 §5.1）。

KVCache-Factory 局限（本次核实）：`pyramidkv/monkeypatch.py` 只有 `replace_llama` / `replace_mistral` 两条分发，**无 Qwen3/DeepSeek 支持**；其 Quest 集成也是 KV 压缩口径（prefill 压缩 prompt），与 Quest 原论文的 decode 稀疏口径不同。不能直接拿来当我们主表的 Quest 数据来源。

---

## 2. 统一接入设计（三种哲学，建议「双轨制」）

### 2.1 三条可行路径

| 路径 | 做法 | 优点 | 缺点 | 适用 |
|---|---|---|---|---|
| **A. harness 内 indexer 臂**（现状） | 按论文口径在 `Indexer` 接口下公平重实现 | 同模型同数据同 scorer 同预算口径，最公平；已接入任务链/快照/审计体系 | 审稿人可能质疑「自实现的 baseline 偏弱」 | 主表精度（已有 quest/moba 臂） |
| **B. vendor 原生代码跑我们的数据** | 摘官方代码包一层适配，跑 LongBench/RULER 同数据集，用我们的 scorer 打分 | 直接回应 GUIDE「真的拉取他们的源码跑出 baseline」；自证没有削弱 baseline | 需要适配工作（Qwen3 移植、版本兼容） | **本次新增的补强轨道** |
| **C. vendor 官方 harness 复现官方数字** | 原样跑官方脚本对齐论文报告值 | 验证「我们的运行环境可信」 | 模型不同（官方多是 Llama-7B 系），数字不进我们主表 | 仅作 sanity check（阶段 1） |

**建议**：精度主表维持 A（口径最公平），用 B 出「vendor 源码复现」对照列回应审稿；性能对比必须摘 vendor kernel 同台面测（§2.3）。C 只做一次性的环境验证，不进论文。

### 2.2 目录与代码组织约定（建议）

```
two-level-attention/
  third_party/                  # 【建议新建】vendor 源码落点（开源 two-level-attention 时按 LICENSE 要求保留声明）
    quest/                      # 从 sparse-bench/third_party/quest 复制或重新 clone（pin 01c1623b）
    MoBA/                       # 本次已下载（sparse-bench/third_party/MoBA）
  exp/vendor_baselines/         # 【建议新建】B 轨适配层：每个基线一个适配脚本 + README 注明源码版本与改动清单
    quest_vendor/
      README.md                 # 来源 commit、摘取文件清单、我们对它做的每一处改动（provenance 要求）
      pred_quest_vendor.py      # 入口：复用 benchmark/LongBench 的数据加载与打分
      qwen3_quest_patch.py      # 官方 llama/mistral patch 的 Qwen3 移植
    moba_vendor/
      pred_moba_vendor.py
      ...
```

- vendor 源码**不直接改**，所有适配都在 `exp/vendor_baselines/` 侧做（clean-room 适配层），满足 S-T014 以来固化的 provenance/快照纪律：入口必须接入 `resolve_treatment_snapshot` 体系或显式声明豁免理由（vendor 臂无文件型配置，建议在 method_name 里带 vendor commit 短哈希）。
- **不建议**继续放 `sparse-bench/`：论文主战场与最终开源范围都是 `two-level-attention/`（GUIDE 明示），vendor 对照代码应随主仓走。

### 2.3 性能对比的统一台面

- kernel 级：`benchmark/efficiency/benchmark_mha_kernel.py`（现有 MHA vs Sparse MHA 台面：batch=4、GQA 4:1、dim=128、seq 32K/65K/128K）——把 Quest 官方 decode kernel、MoBA 官方 varlen 组合实现作为新臂接入同一脚本，输出统一 JSON。
- e2e 级：pred 入口的 wall-clock + `sparse_attn.metrics` 的选择统计（选择器开销是 sparse-bench 定的差异化指标，vendor 臂也要采）。
- 注意 Quest 官方 kernel 依赖 libraft（RAFT 的 CUDA 原语库）+ 自研 PyBind ops，编译链：`kernels/3rdparty/raft/build.sh libraft` → `quest/ops/setup.sh`。若 cu128 环境编译失败，降级方案：只用其 `evaluation/quest_attention.py` 的纯 PyTorch 参考实现测精度，性能数字引用官方论文并注明平台差（Ada6000/4090, CUDA 12.4）。

---

## 3. 各基线详细调研

### 3.1 Quest（mit-han-lab/Quest）

- 论文：arXiv 2406.10774（ICML 2024）；仓库已克隆 `sparse-bench/third_party/quest` @ `01c1623bf939`（2025-07-10，最新 commit 即创建 LICENSE，仓库已稳定）。
- 仓库结构两条线：
  - **精度线（纯 PyTorch，可直接摘）**：`evaluation/quest_attention.py`——`enable_quest_attention_eval(model, args)` 对 `LlamaAttention`/`MistralAttention` 做 monkeypatch；核心逻辑：decode 步（`q_len==1` 且 `layer_id>=2`）时按 `chunk_size` 维护 page 的逐维 max key，用 query 符号展开（`sign = (q>0)*1 + (q<=0)*-1`）算 page 上分界，`local_heavy_hitter_mask` 选 top `token_budget//chunk_size` 个 page 展开；prefill 与 bottom-2 层走原始 flash forward。
  - **性能线（重依赖）**：`quest/ops/`（PyBind + cmake）、`quest/models/QuestAttention.py`（带 KV-Cache manager 的 e2e）、`kernels/`（CUDA 源码 + NVBench benchmark + 单测）。
- 依赖钉死：python 3.10、`flash-attn==2.6.3`、cmake≥3.26.4、libraft（子模块 `kernels/3rdparty/raft`，`git clone --recurse-submodules` 才拉得到——**本地这份克隆缺子模块，编译 kernel 前需补**）。
- 与我们 harness 的关系：我们的 `QuestIndexer` 与其算法同口径（page minmax + top-k 展开），差异在工程（我们 kv-head 级打分后组内广播、官方 q-head 级；我们 decode-only 契约下 sink/swa 正交）。**B 轨摘取清单**：
  - 摘 `evaluation/quest_attention.py` + `evaluation/llama.py`（模型加载壳）→ 适配层写 `qwen3_quest_patch.py`（`Qwen3Attention` 的 forward 替换，字段名映射：`layer_idx`/`num_key_value_groups`/`head_dim` 均存在，工作量小）。
  - 官方 forward 里 `position_ids[0][0].item()` 有 host 同步、`layer_id` 用全局递减变量赋值——适配时保留语义但要在 README 改动清单里如实记录。
- 超参口径：论文主表 token_budget=1024、chunk_size=16（LongBench）；我们 quest 臂口径 `quest_block_size/quest_topk` 应在 B 轨对齐成同预算（换算：topk_pages = token_budget / chunk_size），否则对比不公平。

### 3.2 MoBA（MoonshotAI/MoBA）

- 论文：arXiv 2502.13189；源码 master tarball 已落 `sparse-bench/third_party/MoBA/`（含 `MoBA_Tech_Report.pdf`）。
- 代码极小：`moba/` 仅 `config.py`（`MoBAConfig(chunk_size, topk)`）、`moba_naive.py`（参考实现）、`moba_efficient.py`（flash-attn varlen 组合实现，~440 行）、`wrapper.py`（HF 适配）。依赖仅 `flash-attn==2.6.3` + `torch>=2.1` + `einops`。**摘取难度三个基线中最低**。
- 接入方式已验证可行（读码确认）：`register_moba(MoBAConfig)` 把 `moba_layer` 注册为 transformers 的自定义 attention backend（`attn_implementation="moba"`），prefill 走 `moba_attn_varlen`（chunk-mean gate → top-k chunk → 选中 chunk 与尾 chunk self-attn 经 online-softmax LSE 合并），**decode 当前是 TODO**（`wrapper.py` 明示 `# TODO release paged attn implementation`，decode 回退逐 q 对选中 chunk 重算）。
- **⚠️ 口径警告（必须写进论文与任务链）**：官方 README 原话「MoBA requires continue training of existing models... It is not a drop-in sparse attention solution」。training-free 设定下直接跑 MoBA 官方代码，精度预期明显低于其论文值（其论文数字是续训后的）。我们 E89 臂（49.19）就是 training-free 口径的 MoBA，B 轨复现预计与之互证。论文对比段建议措辞：「MoBA 为训练依赖方法，我们在 training-free 设定下以其官方实现复现作为下界对照」。
- B 轨摘取清单：`moba/` 整包（4 文件）→ `exp/vendor_baselines/moba_vendor/pred_moba_vendor.py`：复用我们 pred.py 的数据加载/prompt 模板/打分，只把模型加载换成 `attn_implementation="moba"`。注意 GQA 时官方 `wrapper.py` 用 `repeat_interleave` 展开 kv-head（显存开销大，128K 需评估是否 OOM；我们 Qwen3-8B 是 GQA 4:1）。
- 超参口径：官方示例 chunk_size=4096/topk=12（≈选中 48K token，是按 1M 上下文设计的）；我们预算口径（K2=1024/2048）下应按 `topk = budget_token / chunk_size` 对齐，建议 chunk_size=64、topk=16/32 两档（与 E89 臂口径一致以便互证）。

### 3.3 ClusterKV（论文 arXiv 2412.03213，ICLR 2025）

- 论文：「ClusterKV: Manipulating LLM KV Cache in Semantic Space for Recallable Compression」。方法口径（已与本项目 `sparse_methods_phase_survey.md` 核对）：prefill 期间对 key 做 k-means 聚类、每簇算术平均出代表；decode 时 q 与簇代表打分选 top 簇、取回簇内原始 KV 做精确 attention——**与我们 GUIDE.md 定义的 `cavg`/`ccluster_*` 方法族一一对应**（GUIDE 原文「参考clusterkv的论文实现」）。
- **官方源码：地址未确认，且很可能未开源**。本环境证据链（2026-10-11 上午实测）：
  - `git ls-remote` / codeload / 网页三连确认 `github.com/thunlp/ClusterKV` **404**；
  - 候选 org 逐一 404：`THUNLP-MT`、`InfinigenceAI`、`Shanghai-AI-Lab`、`Infini-AI-Lab`、`sail-sg`；
  - 搜索渠道全部被墙/超时：GitHub 搜索页、api.github.com、Google/DDG/Bing/百度、arxiv.org、export.arxiv.org、paperswithcode API、OpenReview API、hf-mirror papers 页、gitee API；
  - 旁证：KVCache-Factory（活跃维护、专收此类方法，2026-08 仍有 commit）至今未集成 ClusterKV；sparse-bench README（2026-09-21）已记录「ClusterKV 仓库地址未确定」。
- **Plan B（可立即执行，不阻塞）**：精度对比用我们 harness 的 `cavg`/`ccluster_kmeans` 臂作为「ClusterKV 口径」——这本来就是 GUIDE 的设计；论文里对比段引用 ClusterKV 论文报告数字时注明「官方代码未公开，口径为按论文实现的公平重跑」。
- **待网络畅通后的确认清单**（交给有外网的机器或人工）：① Google 搜「ClusterKV github」；② paperswithcode.com 论文页看是否标注「no code」；③ OpenReview ICLR2025 页面看 code 链接；④ 若确认开源，按 §2.2 目录约定 clone 并补 B 轨适配。

---

## 4. 执行计划（E124 任务链草案，建议登记进 TASK.md 待办池）

> 资源前提：S-T019 四机 20 卡在飞，本链阶段 0–1 为纯 CPU/小 GPU 准备工作，现在即可启动；阶段 2 起需要 GPU，建议插空或等收割。

| 阶段 | 内容 | 验证门禁 | 资源 |
|---|---|---|---|
| 0 | `two-level-attention/third_party/` 落 vendor 源码（quest pin `01c1623b` 含 `--recurse-submodules` 补 raft 子模块；MoBA 已就位）；写 provenance README | 目录 + commit 锚点落袋 | CPU（需一次稳定的 GitHub 连接，codeload tarball 已验证可行） |
| 1 | **C 轨 sanity**：Quest 官方 `evaluation/` 在 Llama-3.1-8B-Instruct 上复现官方 LongBench 单子任务数字（对齐其论文表），MoBA 官方 `examples/llama.py` 冒烟 | 与官方报告值偏差在任务噪声内（记录偏差值） | 1 卡，小时级 |
| 2 | **B 轨移植**：`qwen3_quest_patch.py`（Quest 官方算法 → Qwen3Attention）；`pred_moba_vendor.py`（moba backend + 我们数据流）；入口接入 snapshot/postfix/SKIP 幂等纪律 | 红-绿双证：① 小样本（n=20）与 harness 内 quest/E89 臂同数据对拍，分数差如实记录（预期接近但不必逐位等）；② `python` 与 `python -O` 双跑 | 1–2 卡 |
| 3 | **三件套跑分**：vendor 臂 × LB v1 16 任务 + LB v2 503 + RULER 32/64/128K（Qwen3-8B），复用官方 scorer | 全格齐 + 与 S-T019 新口径主表同台面合并 | 等 GPU 空窗，四机调度复用 S-T019 派单模板 |
| 4 | **性能对比**：Quest 官方 kernel / MoBA varlen 接入 `benchmark/efficiency/`；选择器开销（metrics）同步采集 | 同台面 JSON 落袋；Quest kernel 若编译失败走降级方案（§2.3）并如实记录 | 1 卡 + ncu |

**给出主表列的两种口径标注建议**：`Quest†`（†=vendor 源码复现，commit 锚点）与 `Quest（harness 公平口径）`并列或择一进表、另一进附录，由论文写作阶段定。

## 5. 风险与开放问题

1. **网络受限是最大工程风险**：本机仅 codeload.github.com 稳定（git 协议/网页/API/搜索引擎均不稳定或被墙）。clone 操作建议统一走 `curl -L https://codeload.github.com/<org>/<repo>/tar.gz/refs/heads/<branch>`；Quest 补 raft 子模块需要 git 协议窗口期或找 raft 的 release tarball 替代。
2. **flash-attn 版本冲突**：Quest/MoBA 都钉 `flash-attn==2.6.3`，本机 base 环境 torch 2.8.0+cu128 且 `import flash_attn` 失败（生产跑批用的解释器环境需在派单机上先确认）。vendor 实验建议独立 venv，勿污染 S-T019 生产环境。
3. **MoBA decode 路径官方未完成**（paged attn TODO）：长生成任务（gov_report 等）vendor 臂 decode 会退化为慢速路径——e2e 速度对比时这是对方工程债，记录即可，不要替它优化（保持 vendor 原样是 B 轨的意义）。
4. **ClusterKV 若最终确认有官方码**，本方案 §3.3 清单可直接续上；若确认无码，论文措辞走 Plan B。
5. **Qwen3-32B / DeepSeek-V2-Lite 的 vendor 移植**放阶段 3 之后——先 8B 打通链路，再按 GUIDE 的多模型矩阵扩展（DeepSeek 是 MLA，Quest/MoBA 的 GQA 假设全部不适用，届时大概率只能用论文引用数字 + 口径说明）。

## 6. 给其他 AI 的评判要点（Reviewer checklist）

- [ ] 双轨制（A 公平口径 vs B vendor 源码）的划分是否合理？B 轨结果进主表还是附录？
- [ ] vendor 源码放 `two-level-attention/third_party/` 是否符合最终开源计划？LICENSE 兼容性（Quest=Apache-2.0，MoBA=MIT，需复核）？
- [ ] Quest 官方 kernel 编译失败时的降级方案是否可接受（性能数字引用官方论文的口径风险）？
- [ ] MoBA training-free 口径的论文措辞是否会引来「对比不公平」的审稿意见？是否需要补一档「续训后 MoBA」的实验（成本极高，建议明确不做并给理由）？
- [ ] ClusterKV 的证据链是否足以支撑「按论文实现口径对比」的写法？
- [ ] 阶段门禁里的红-绿双证标准是否与 S-T014~S-T017 固化的审计纪律一致（snapshot/postfix/SKIP/双跑）？

---

## 主 AI 回应（2026-10-11 12:0X，方案接受，一处关键修正）

1. **§3.3 ClusterKV「未开源」结论不成立，已实锤修正**：官方仓库是
   `github.com/sjtu-zhao-lab/ClusterKV` @ `c7380819588771e8c5dadacdbd33942d7c53adea`
   ——你的候选 org 清单（THUNLP-MT/InfinigenceAI/…）恰好漏了正确答案；
   GPT Runtime 方案（f304ec979 §6）同日已核同锚点，我本机 codeload
   tarball 请求实测 HTTP 200。**B 轨三基线（Quest/MoBA/ClusterKV）全部
   可做**，§3.3 Plan B 降级为「网络受限时的后备」而非主路径；§4 阶段 0
   落库清单相应补 ClusterKV。
2. **双轨制接受**：A（harness 公平口径）主表 + B（vendor 源码复现）
   对照列，†标注 + commit 锚点；C 轨仅环境 sanity 不进论文。目录约定
   （two-level-attention/third_party/ + exp/vendor_baselines/ 适配层
   + provenance README + snapshot/manifest 接入）符合 S-T014 以来纪律。
3. **与动态主线的咬合（重要排链更新）**：本方案 = GPT Runtime 方案
   （f304ec979）A6 阶段「外部基线」的执行细节，两份 advice 互为实现
   计划。排期：阶段 0-1（纯 CPU/小时级 1 卡）排在 E124a（A1 动态
   controller 实现，agent 在飞）完成后启动；阶段 2-4（B 轨移植/三件套/
   性能）等 E116b 收割空窗，复用 S-T019 派单模板。ClusterKV vendor 臂
   预算口径对齐 cavg/ccluster 臂以便互证。
4. **Reviewer checklist 逐项裁定**：①MoBA「续训后」对照档**明确不做**
   ——官方自述非 drop-in，training-free 是论文统一设定，补续训档成本
   极高且引入「训练配方」新自由度，论文措辞走你建议的下界对照口径；
   ②Quest kernel 编译失败降级（性能引用官方 + 平台差注记）可接受；
   ③LICENSE：Quest Apache-2.0 / MoBA MIT / ClusterKV 按 §2.2 开源
   前复核；④阶段门禁红-绿双证与 S-T014~T017 纪律一致，照办。
5. **网络受限缓解**：本会话主 AI 侧外网搜索/下载通道可用（ClusterKV
   即当场核验），后续 vendor 落库的 GitHub 窗口需求可由主会话承担，
   不必等 codeload 窗口期。

## 主 AI 回应（0745 kimi3 hourly review，一并处理）

llm_eval.py snapshot 缺口为**重复报告**——082 修复 c8d92cd70 已合入
（10-11 早于你本轮基准 db2c539f1 的审查时刻），当前代码实核：L52-58
resolve_treatment_snapshot 冻结 → L93 register_patch(model,args,snapshot)
→ L133-161 落盘消费同一 snapshot。无需新修复。
