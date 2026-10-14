# DeepSeek-V2-Lite-Chat（MLA）对手 baseline 适配调研报告

> 调研日期：2026-10-14
> 执行：deepseek 轨（Explore agent 一手源码核实 + 记忆）
> 背景：用户 2026-10-14 拍板「最终论文对手 baseline 全部补齐，LLaMA 放最后」。DeepSeek-V2-Lite-Chat
> 是扩展模型实验（验证扩展到 nope+rope 的 MLA + 跨家族验证），需判断 Quest/MInference/ClusterKV/MoBA
> 官方源码是否支持 MLA，据此定 DeepSeek 臂走 vendor 还是标注不可适配。

## 铁律前提
「一律用官方开源源码，绝不语义复现」——因此必须先确认官方源码本身是否支持 MLA，才能决定路径：
- 官方支持 → vendor 官方适配（接口胶水，不碰选择逻辑）
- 官方物理不可适配 → fullkv 锚点 + 标注方法不支持（不是偷懒不复现）

## 一、DeepSeek-V2-Lite-Chat 关键维度实测值（一手证据）

来源：`/home/wangyuanshuo02/models/huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/config.json`
+ `~/.cache/huggingface/modules/transformers_modules/DeepSeek-V2-Lite-Chat/modeling_deepseek.py`

| 项 | 值 |
|---|---|
| model_type / 类 | `deepseek_v2` / `DeepseekV2ForCausalLM`（trust_remote_code） |
| hidden_size / num_hidden_layers | 2048 / 27 |
| num_attention_heads == num_key_value_heads | 16 == 16（名义值，MLA 下无传统 per-head KV） |
| q_lora_rank | **null**（Lite 版 Q 无低秩压缩，直接 q_proj） |
| qk_nope_head_dim / qk_rope_head_dim | 128 / 64 → **q_head_dim = 192** |
| kv_lora_rank | **512** |
| v_head_dim | **128** |
| rope_scaling | yarn，factor=40，original_max_position=4096，mscale_all_dim=0.707 |

**K/V 实际形状（HF eager 路径，modeling_deepseek.py:829-867）**：
- KV 投影无 `k_proj`/`v_proj`，只有 `kv_a_proj_with_mqa`（2048→512+64=576）和 `kv_b_proj`（512→16×256）
- **K cache：(bsz, 16, L, 192)**（nope 128 + rope 64，RoPE 只作用于后 64 维）
- **V cache：(bsz, 16, L, 128)** —— **K head_dim (192) ≠ V head_dim (128)**，任何「单 head_dim」假设代码的硬卡点

## 二、Quest：官方不支持 MLA，且非纯形状适配可救

本地仓库：`third_party/quest/`。全仓库 grep `deepseek|MLA|latent|kv_lora` **零匹配**。

### 路径 A：纯 PyTorch 前向（LongBench eval 实际用的 `evaluation/quest_attention.py`）
1. **patch 只认 Llama/Mistral 类**（行 256）：`isinstance(module, (LlamaAttention, MistralAttention))`——
   `DeepseekV2Attention` 不匹配 → patch 静默不生效（跑 dense）。
2. **无 k_proj/v_proj**（行 100-108）：`self.k_proj(hidden_states).view(..., num_key_value_heads, head_dim)`——
   MLA 无此属性 → AttributeError。
3. **GQA repeat_kv 假设**（行 139-140）：结构假设是标准 GQA（K/V 同 head_dim、kv heads 可 expand 到 q heads）。

### 路径 B：Quest CUDA kernel 路径（`quest/models/QuestAttention.py` + `quest/utils/`）
1. **单 head_dim 假设，K/V 共用**（`quest/utils/kv_cache.py:20-23`）：KvPool 缓冲
   `(num_layers, capacity, 2, page_size, num_heads, head_dim)`——MLA 的 192/128 不一致直接无法表达。
2. **MHA-only CUDA 路径**（`quest/models/llama.py:540-543` + `controller.py:122-129`）：qo_heads 和 kv_heads
   传同一个数，连 GQA 都不支持。
3. **整头 RoPE**（`QuestAttention.py:99` `apply_rope_in_place`）：对整个 head_dim 做 RoPE——MLA 只有 64/192 维做。
4. **核心选择元数据深绑 per-head 独立 K**（page 级 per-head K 的逐维 max 符号量化 + paged metadata cache）：
   MLA 压缩形态下 K 不是 per-head 张量，选择逻辑输入语义不存在。

### 结论
**官方 Quest 物理不可适配**：不是形状问题，选择逻辑与 KV 表示本身深度耦合 MHA 结构，「纯形状适配不碰
选择逻辑」不可达 → **不走 vendor，不建议语义复现**。

## 三、MInference：本地无官方源码，无法源码级核实（诚实标注）

- `third_party/` 无 MInference 目录（违反「先拉官方仓库」纪律一环，需补拉）。
- 当前 python 环境无 `minference` 包；此前是官方 pip 包走 vendor（`e119_ruler128k_formal_minference_vendor.md`，
  Qwen3-8B，AVG 49.53），只跑过 MHA/GQA 的 Qwen3。
- 机制性判断（待源码核实）：patch 机制按 HF 模型 attention 类 monkey-patch forward；论文/README 声明支持
  列表（Llama/Mistral/Phi/Gemma/Qwen/GLM 等）均为 MHA/GQA；pattern 搜索建立在全维 QK^T 相似度图聚类上。
  MLA 障碍：(a) 注册表是否有 `DeepseekV2Attention` 分支是硬门；(b) nope/rope 分段 + yarn mscale 修改的
  softmax_scale 需处理。
- **待网络恢复后 `git clone microsoft/MInference` → `grep -ri "deepseek\|DeepseekV2\|mla"` 才能下正式结论**。

## 四、其余 third_party 对 MLA 的立场

| 仓库 | MLA 支持 | 证据 |
|---|---|---|
| **ClusterKV** | 否，llama-only | 只有 `llama.py`；`ClusterKVAttention.py:36-37` 假设 k_proj/v_proj 标准 GQA；pred.py 直接 `LlamaForCausalLM.from_pretrained` |
| **MoBA** | 否，但卡点最小 | `moba_efficient.py:289-303` 纯张量级 `[seqlen, head, head_dim]`（单 head_dim 变量）；选择逻辑是块级相似度门控，不依赖 per-head KV cache 内部结构 |
| **sparq-gpt-fast** | 否 | `model.py` 是 gpt-fast 的 GPT 家族，与 HF MLA 体系无关 |

## 五、对策建议（DeepSeek MLA 臂）

1. **Quest → 定性「官方不支持，方法-架构不匹配」**。DeepSeek 臂 Quest 位置 = **fullkv 锚点 + 标注
   「Quest 官方仅验证 MHA/GQA（Llama/Mistral），官方源码不支持 MLA」**，引 Quest 论文实验模型表佐证。
   与铁律不冲突：Qwen3 轨已有 md5 级 vendor，DeepSeek 臂是官方源码物理不可适配。
2. **MInference → 先补动作再决策**：有网机器 clone + grep；有 DeepseekV2Attention 分支 → vendor；无 →
   同 Quest 结论。当前 minference_vendor 数据不得外推到 MLA。
3. **MoBA** 是灰色地带里最可行的稀疏对手（纯块级几何门控、无 KV 内部结构假设，卡点仅 K/V 单 head_dim
   一处）——理论上可薄适配，但官方零 MLA 代码，任何改动需逐点登记「接口胶水」并经用户裁决是否越界。
4. **ClusterKV**：同 Quest 结论（llama-only + k_proj/v_proj 结构假设）。
5. **已有可复用资产**：DeepSeek-V2 fullkv 16K-cap 路径已修复跑通（model.generate 协议，
   LBv1 16 任务 AVG 40.55，16K 受限上下文 treatment），可直接作 MLA 臂 dense 锚点。

## 六、待办
- [ ] 网络恢复后：clone microsoft/MInference 到 third_party/ + grep deepseek 核实。
- [ ] MoBA MLA 薄适配是否越界 → 用户裁决。
- [ ] DeepSeek 臂主表：fullkv 锚点（已完成）+ 标注 Quest/ClusterKV 不可适配 + MInference 待核实。
