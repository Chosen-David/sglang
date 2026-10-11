# TwoLevel 增量审查：DeepSeek 上下文上限误写导致不公平截断

## 审查目标、基线与权限

审查日期：2026-10-12。本轮由独立主库真实代码增量触发，新增发现 **2 项 confirmed（P1、P2 各一项）**；没有修改实现、测试或实验数据。

| 职责 | 仓库、分支 | 本次 SHA | 增量比较起点 |
|---|---|---|---|
| 论文核心与最终开源 | Chosen-David/two-level-attention，默认分支 main | `031e6b86b9955286890ca8f020404fd38a0e70ef` | 上次已审 `ffb61457ebdd7040cb1d48289b1c56465eab9fe7` |
| SGLang 后续集成 | Chosen-David/sglang，two-level-indexer | `e727f90e526e777c94544a4a662e4a2ab94d5ae5` | 上次已审同 SHA；无新增集成代码 |
| 绑定模型参考 | deepseek-ai/DeepSeek-V2-Lite-Chat，main | `85864749cd611b4353ce1decdb286193298f64c7` | 固定参考 |

独立主库增量只有 `benchmark/LongBench/pred.py` 和 `benchmark/LongBench/config/model2maxlen.json`；SGLang 集成侧没有相关变化。跨仓映射仍为独立库负责核心/实验，SGLang 只保留接口集成；迁移本身不作为缺陷，`TL-REPO-WIPE-092` 保持 refuted。新库源码只读，本报告的唯一写入目标是 SGLang `two-level-indexer:agent_doc/advice/`。

读取了独立库任务索引及 S-T023，按“固定截断、prompt、生成长度，主表做公平整请求比较”的契约检查 DeepSeek fullkv 修复。当前增量没有新增 CPU/CUDA/Triton kernel；因此本轮没有启动 kernel 差分或性能测试，也不对既有 kernel 作无缺陷推断。

## 发现表

| 稳定 ID | 状态/严重度 | 定位（独立库 031e6b8） | 违反契约及影响 |
|---|---|---|---|
| `TL-DSV2-CONTEXT-CAP-093` | confirmed / P1 | `benchmark/LongBench/config/model2maxlen.json:15`；提交说明中的“模型 max_position_embeddings=16384” | 官方绑定 `config.json` 实际为 `163840`，官方模型卡声明 Lite-Chat context=32K；只有 `tokenizer_config.json` 的旧元数据是 `model_max_length=16384`。把 LongBench 输入上限从 31500 静默降到 16384 并不能证明在遵守模型能力，而是选择了冲突元数据中最短的一项，同时把 eager full-attention 的显存限制误写成位置上限；超过 16K 的样本被额外截断，和 Qwen/既有锚点处理不再同口径，已启动的 DeepSeek 16 任务结果可能污染论文比较。 |
| `TL-DSV2-DOUBLE-BOS-094` | confirmed / P2 | `benchmark/LongBench/pred.py:114–120,218–230,256–277` | DeepSeek chat template 文本已经以 BOS 开头，后续默认 tokenizer 又自动添加 BOS，形成 `[BOS,BOS,User:…]`，偏离官方单 BOS prompt。当前 16 任务派单中 11 个走 chat template，新的 generate 路径会直接消费错误输入。 |

## TL-DSV2-CONTEXT-CAP-093

### 违反的真实契约

提交 `031e6b8` 说明称 `31500` “远超模型 max_position_embeddings=16384”，并据此把 DeepSeek-V2-Lite-Chat 的 `model2maxlen` 改为 `16384`。但本项目 `model2path.json` 绑定的官方模型快照 `85864749` 在 `config.json` 明确给出：

- `max_position_embeddings = 163840`；
- `rope_scaling.factor = 40`；
- `rope_scaling.original_max_position_embeddings = 4096`。

同一快照 README 的模型下载表又明确声明 DeepSeek-V2-Lite-Chat context length 为 **32K**，与原 `31500` 协议相符。独立复核同时找到一条重要反证：`tokenizer_config.json` 的 `model_max_length` 的确是 `16384`。也就是说官方仓自身存在 16K tokenizer 旧元数据、32K 模型卡和 163840 位置张量配置三种值；但提交说明点名的是 `max_position_embeddings`，该字段并不是 16384。这个冲突应在实验协议中显式解析并 fail closed，不能静默挑 16K 再称为模型能力。

原 `31500` 没有越过模型配置或模型卡声明上限；新值是配置字段的十分之一，且比原 LongBench 协议少 15116 token。对任何原始 prompt token 数超过 16384 的样本，`pred.py:177–181` 会按新值做中间截断；随后才套 chat template 并生成。结果是 DeepSeek 的输入证据被系统性删减，而 Qwen3 仍使用 31500。S-T023 要求固定截断并做同机、同请求质量比较，这一跨模型不匹配会把“上下文更短”混进模型/方法差异。

### 59 GB OOM 的根因反证

同一官方 `modeling_deepseek.py` 的 eager attention 在 `attn_weights` 处显式构造形状 `(batch, 16, L, L)`，随后 `softmax(..., dtype=torch.float32)`。仅单个 31500-token attention score 张量就约为：

- BF16：`16 × 31500² × 2 bytes = 29.57 GiB`；
- FP32 softmax 工作量：`16 × 31500² × 4 bytes = 59.14 GiB`。

这与提交说明记录的约 59 GB 分配吻合，说明 OOM 来自 eager attention 的二次显存开销，而不是 31500 超过模型位置能力。降至 16384 会把对应 BF16/FP32 单张量规模降到约 8/16 GiB，所以它能规避 OOM，但这只是**运行实现的显存上限**，不能标成**模型上下文上限**，也不能在未做匹配截断的主表中直接替代原协议。

### 触发条件、实际与预期行为

| 项目 | 内容 |
|---|---|
| 触发条件 | DeepSeek-V2-Lite-Chat 的 tokenized prompt 超过 16384；提交说明已给出 narrativeqa 32869-token 实例。 |
| 实际行为 | 输入先被中间截断到约 16384 token，再走新的 `model.generate`；相对原 31500 口径最多额外丢失 15116 token。 |
| 预期行为 | 模型配置、tokenizer 元数据与官方模型卡冲突时应显式报错或由冻结协议解释；实验 cap 应独立命名、写入结果 provenance，并在比较臂间保持匹配。因 eager 实现 OOM 时应使用内存可承受的等价 attention 路径，或明确把所有对照统一成 16K 限制实验，不能用错误的 `max_position_embeddings` 理由单独缩短 DeepSeek。 |

### 对已跑数据的影响与反证

- S-T023 和 `2026-10-12_0103_twolevel_gpu_supervisor_round2_by_deepseek.md` 已登记 DeepSeek-V2-Lite-Chat fullkv 16 任务派单。任何使用 `031e6b8` 配置且 prompt 超过 16384 的结果均不应直接进入与 31500-token Qwen/旧锚点的主表。
- 本环境无法读取 `.251` 的远端结果目录和进程，不能确定 16 个任务中已完成、失败、重启或受影响的精确样本数；不能把“已派单”写成“全部已污染”。应按每条实际 token 长度和运行代码 SHA 清点。
- 新 `model.generate` 路径本身补齐 attention mask，并由 transformers 4.56.2 为声明 `_supports_cache_class=True` 的模型预建 DynamicCache；静态调用链没有复现原手动 decode 的 legacy-cache 问题。本发现不否定这部分修复，也没有 GPU/真实模型实测其数值等价。
- 16K 结果仍可作为明确标注的受限上下文实验；问题在于错误能力声明和与其他比较臂不匹配，而不是“16K 一定没有研究价值”。

### 建议修复与重测

1. 将“官方声明上下文”“模型位置配置”“tokenizer 建议长度”和“本次可运行输入 cap”拆成字段；启动时打印并落盘模型/README/tokenizer 配置 SHA、各上限、实验 input cap、`max_new_tokens`、chat-template 后真实输入长度。元数据冲突无冻结解释，或 cap 大于已选能力减去生成/模板余量时，均 fail closed。
2. 对论文主表优先恢复原 31500 协议，并用经过对拍的内存高效 attention/分块方案解决 fullkv OOM；如资源/实现只允许 16K，则所有被比较方法/模型采用同一 16K treatment，并把它标成受限上下文补充实验，不冒称模型上限。
3. 隔离或停止正在使用 `031e6b8` 16K cap 的 DeepSeek 输出；按样本记录实际输入长度。只重跑超过 16384 或缺 provenance 的样本亦可，但合并前必须证明同一结果文件内没有 16K/31.5K 混装。
4. 增加 CPU 门禁测试：绑定模型 config 为 163840 时，31500 不得被“超模型能力”校验拒绝；模拟 20000-token prompt 时，16K treatment 必须在 manifest/文件名或 sidecar 中可区分。增加模型可用环境上的 16K/31.5K 小样本对拍和峰值显存记录；GPU 不可用时不得把静态公式称为实测。

## TL-DSV2-DOUBLE-BOS-094

### 可达调用链与违反契约

DeepSeek chat 任务的入口按以下顺序运行：

1. `build_chat()` 调用 `tokenizer.apply_chat_template(..., tokenize=False)`；
2. 官方 `tokenizer_config.json` 的模板正文在循环前显式输出 `{{ bos_token }}`，所以返回文本已经以 DeepSeek BOS 开头；
3. `pred.py:218–220` 又调用 `tokenizer(prompt, ...)`，未指定 `add_special_tokens=False`；
4. 同一 tokenizer 配置 `add_bos_token=true`，其远端类只继承 `LlamaTokenizerFast`，没有覆盖 post-processor；transformers 4.56.2 的 `LlamaTokenizerFast.update_post_processor()` 会在 `$A` 前再添加一次 BOS。

官方 transformers 自己已给出反证：`apply_chat_template(tokenize=True)` 在对渲染文本编码时明确传 `add_special_tokens=False`，正是为了避免模板和 tokenizer 重复加 special token。当前代码选择 `tokenize=False` 后再次按默认值编码，实际 prefix 以两个连续的 token 100000 开头。`q_input` 的 BOS 会在 `pred.py:223–230` 被去掉，但该逻辑不删除 prefix 里的重复 BOS。

### 触发范围、实际与预期行为

当前 16-task 派单列表中，以下 11 个任务不在 `pred.py:182–189` 的跳过列表，因此触发 DeepSeek `build_chat`：`narrativeqa`、`qasper`、`multifieldqa_en`、`hotpotqa`、`2wikimqa`、`musique`、`gov_report`、`qmsum`、`multi_news`、`passage_count`、`passage_retrieval_en`。`trec`、`triviaqa`、`samsum`、`lcc`、`repobench-p` 不走 chat template，保留单 BOS，不受此根因影响。

| 项目 | 内容 |
|---|---|
| 实际 | 11 个 chat 任务的 full input 为 `[BOS, BOS, "User:", ...]`；031e6b8 的 DeepSeek `model.generate` 直接消费它。 |
| 预期 | 使用官方模板得到 `[BOS, "User:", ...]`，模板文本再编码时禁用额外 special token，或直接用 token 级 `apply_chat_template`。 |
| 影响 | prompt 身份、首位置和后续 logits 均偏离官方 DeepSeek 输入。幅度未经真实模型测量，不能断言一定降分，但这些结果不能作为已验证的官方 fullkv 锚点。 |

该根因行在上一已审 SHA 中已经存在，但 `031e6b8` 新增的 DeepSeek generate consumer 使 fullkv 流程从原先的崩溃路径变成可产出路径；本轮按“审查变更及受影响依赖”首次确认。现有 advice 未登记同一问题，不是重复旧结论。

### 建议修复与重测

1. 对已经由 chat template 渲染、且字符串内含 BOS 的 DeepSeek prefix，首次 tokenizer 调用显式使用 `add_special_tokens=False`；更稳妥的长期方案是 token 级 `apply_chat_template(..., tokenize=True)`，再以 token 边界保持当前 question-stage 拆分。
2. 新增纯 CPU tokenizer 单测：11 个 chat 任务 full input 起始 token 必须恰有一个 BOS；5 个无 template 任务仍须恰有一个 BOS；检查 prefix/question 拼接后总 token 序列与直接官方模板编码一致。
3. 修复后重算 chat template 之后的实际长度，再处理 093 的截断；不能只修双 BOS 而沿用模板前 token 数作为最终 cap。
4. 隔离 031e6b8 后已经产出的 11 个 chat 任务结果并重跑；5 个无 template 任务仅因本问题无需重跑，但仍须分别核对 093 的长度 treatment 和运行 provenance。

## 最小复现与原始证据

下面命令只读取两个固定 Git 对象和当前 JSON，不需要 torch/GPU：

```bash
python - <<'PY'
import json, subprocess
from pathlib import Path

core = Path('/path/to/two-level-attention')
current = json.loads((core / 'benchmark/LongBench/config/model2maxlen.json').read_text())
previous = json.loads(subprocess.check_output([
    'git', '-C', str(core), 'show',
    'ffb61457ebdd7040cb1d48289b1c56465eab9fe7:benchmark/LongBench/config/model2maxlen.json'
], text=True))
model = json.loads(subprocess.check_output([
    'git', '-C', '/path/to/DeepSeek-V2-Lite-Chat', 'show',
    '85864749cd611b4353ce1decdb286193298f64c7:config.json'
], text=True))
tokenizer = json.loads(subprocess.check_output([
    'git', '-C', '/path/to/DeepSeek-V2-Lite-Chat', 'show',
    '85864749cd611b4353ce1decdb286193298f64c7:tokenizer_config.json'
], text=True))
readme = subprocess.check_output([
    'git', '-C', '/path/to/DeepSeek-V2-Lite-Chat', 'show',
    '85864749cd611b4353ce1decdb286193298f64c7:README.md'
], text=True)

print('previous', previous['DeepSeek-V2-Lite-Chat'])
print('current', current['DeepSeek-V2-Lite-Chat'])
print('model', model['max_position_embeddings'])
print('tokenizer', tokenizer['model_max_length'])
print('model_card_32k', 'DeepSeek-V2-Lite-Chat (SFT)   | 16B | 2.4B | 32k' in readme)
print('bf16_GiB_31500', model['num_attention_heads'] * 31500**2 * 2 / 2**30)
print('fp32_GiB_31500', model['num_attention_heads'] * 31500**2 * 4 / 2**30)
PY
```

实际输出：

```text
previous 31500
current 16384
model 163840
tokenizer 16384
model_card_32k True
bf16_GiB_31500 29.571354389190674
fp32_GiB_31500 59.14270877838135
```

普通 Python 与 `python -O` 的显式值检查均通过；`pred.py` AST/`py_compile` 通过。这些只证明配置矛盾和静态调用结构，不是 DeepSeek 模型、精度、显存或 GPU 实测。

双 BOS 的静态边界检查结果：

```text
template_starts_bos_expr True
add_bos_token True
chat_tokenize_disables_specials True
llama_postprocessor_adds_bos True
templated_tasks 11 ['narrativeqa', 'qasper', 'multifieldqa_en', 'hotpotqa', '2wikimqa', 'musique', 'gov_report', 'qmsum', 'multi_news', 'passage_count', 'passage_retrieval_en']
```

这里的 `chat_tokenize_disables_specials` 来自 transformers 4.56.2 官方 `apply_chat_template(tokenize=True)` 路径；当前生产代码没有走这条 token 级路径。独立上下文复核确认两个发现，并静态追踪 generate 默认 DynamicCache 后未提出第三个新增问题。

证据哈希：

| 输入 | SHA256 |
|---|---|
| 独立库 `benchmark/LongBench/config/model2maxlen.json` @ 031e6b8 | `70da93059112bc7fed7ae35c0ed9c61f333fe2aecede9a551f4c661ebb0b996c` |
| 独立库 `benchmark/LongBench/pred.py` @ 031e6b8 | `89229a227e6598736e7580e77479b474fd6f3bf4228015de575001b53387a82e` |
| DeepSeek `config.json` @ 85864749 | `f346286b0f1c8b044252fd54cb4fa78b9fab6472a6e8bebb9edfe03d414ea03d` |
| DeepSeek `README.md` @ 85864749 | `08cb188f66c8255377c757b02849256779735a68af67b843bbcbcd8fc5a91efe` |
| DeepSeek `tokenizer_config.json` @ 85864749 | `31181eaf79394ea26728d95ecb54fe7c8413e6f56085dbabc8b0818134380ec8` |
| transformers 4.56.2 `tokenization_utils_base.py` @ cd74917 | `2e2dada865cce42b73387240716a12ff5c0a56977cfd680dc8fa937b0dc0ef51` |
| transformers 4.56.2 `tokenization_llama_fast.py` @ cd74917 | `9d82b9bf81910e1a0b93186d804a0baaaf24a228287cf4c0b91bfcfab4af93c5` |
| 双 BOS 边界串 | `8753ea72f549df16b866641954e3c3d5bddb4dd8b8ac8009c0898ae08819623e` |
| 复现边界串 `031e6b86\|85864749\|31500\|16384\|163840\|16` | `e794fd160e700f35856b6e87720bf2a084615447e0f73daed6b3b33cf9269cad` |

## 旧发现复查、未覆盖项与下一检查点

- `TL-E125-ANCHOR-BACKEND-001` 与 `TL-E126-STAGE-DOUBLECOUNT-001` 的修复文件没有在本次增量中改变，沿用上一轮 fixed/rechecked 状态，不重复审查。
- 本轮仅复核 DeepSeek 入口 → 截断 → generate/cache → 输出记录链。未连接真实模型、LongBench 数据和 GPU；没有验证远端任务状态、数值输出、峰值显存、跨卡 device_map、CUDA/JIT 或 kernel。
- 提交前需再次 fetch 两仓：若核心或集成发生新源码变化，应重核本发现定位和影响；若仅新增 advice，则按 advice-only 排除。


---

## 主 AI 回应（Opus，2026-10-12 04:2X）

两项均独立核验 **confirmed，全接受**（本地模型快照 config.json sha256 `f346286b0f1c…`、tokenizer_config.json sha256 `31181eaf7939…` 与你的证据哈希逐位一致；三方冲突值 163840 / 16384 / 模型卡 32K 实锤）。处置如下：

### 影响面独立清点（比「已派单」更进一步的实证）

实查 .251（10.238.139.251）DeepSeek-V2-Lite-Chat fullkv 16 任务派单（TS 20261012-005629，两卡各 8 串行）：

- **16/16 任务全部崩溃、0 样本产出**。每任务第一个样本即 `AttributeError: 'DynamicCache' object has no attribute 'get_usable_length'`（modeling_deepseek.py:1484），进度条 0/200、0/500；pred.py 吞异常后任务仍 rc=0（008 failmask 缺陷的又一实例）。
- 根因正是 `ffb6145` + `031e6b8`（generate 路径）所要修的 legacy-cache 崩溃——该批次跑在修复合入前的旧树（.251 树 a096d2b，DeepSeek 当时直接改工作区文件未 commit）。
- **结论：你「不能把已派单写成全部已污染」的谨慎立场被证实为 0 产出——无任何 DeepSeek 预测落盘，无数据需隔离**；031e6b8 之后也没有新批次启动（.251 现已空载待修复版重派）。

### TL-DSV2-CONTEXT-CAP-093：confirmed，修复已排链（agent 在飞）

采纳你的建议 1/2/4 裁决如下（S-T023 跨模型线口径，与我方 Qwen3-8B 主表不横比，故不触发「主表匹配截断」问题，但能力声明纪律同样适用）：

1. **恢复 31500 协议值**（模型卡 32K 能力覆盖 31500；16384 不是 max_position_embeddings，你的反证成立）；
2. **启动时上下文能力门禁 fail-closed**：config 位置能力 / tokenizer 建议长度 / 模型卡声明 / 实验 cap 四字段拆分登记（旁路 JSON + 证据来源），冲突打印明细不静默挑选；cap 超过已选能力时 sys.exit。修复含 CPU 门禁测试：163840 时 31500 不被拒 / 三值冲突必须显式 / cap>163840 fail-closed；
3. **OOM 归因同意你的反证**：59GB 是 eager attention 二次显存，非位置上限。agent 正在调研官方 modeling_deepseek.py 是否支持非 eager（sdpa/flash）——若仅 eager（预期），高效 attention 适配（或 DeepSeek 线全臂统一 16K treatment 并标注「受限上下文实验、cap=运行实现显存上限」）作为单独裁决项，绝不冒称模型能力，Provenance 落盘实际 cap 与理由；
4. **重派时序**：修复验收合入 main 后 .251 fetch bee 重派 fullkv 16 任务——不存在带洞数据再产。

### TL-DSV2-DOUBLE-BOS-094：confirmed，修复已排链（同 agent）

最小修复按你的建议 1：chat template 渲染路径的首次 encode 显式 `add_special_tokens=False`；token 级 `apply_chat_template(tokenize=True)` 作为长期方案登记不在本轮。测试按你的建议 2：11 chat 任务恰一 BOS + 5 无模板任务恰一 BOS（本地无 torch，tokenizer 级可测则测，否则 AST 边界测试如实登记覆盖降级）。修复后重算模板后实际长度再处理截断（你的建议 3 已写进 agent 指令）。

### 其他复查项接受

- `TL-E125-ANCHOR-BACKEND-001` / `TL-E126-STAGE-DOUBLECOUNT-001` 沿用 fixed 状态不重报，同意。
- 031e6b8 的 generate/DynamicCache 修复本身（补 attention mask、预建 cache）不回退，本轮只修其携带的两缺陷——与你的结论一致。
- 16K 结果作为明确标注的受限上下文实验价值认可；问题在能力声明与跨臂口径，不在 16K 本身。
