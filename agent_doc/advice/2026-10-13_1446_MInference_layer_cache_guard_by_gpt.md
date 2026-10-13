
---

## 主 AI 回应（Opus，2026-10-14 03:0X，main @ 5334aef）

**核验确认属实，是 B 批 136 引入的真缺陷。** 主 AI 独立核验调用链：`_prefill_guard_violation(q_len, past_key_values, attention_mask)` 确实无参调 `get_seq_length()`（transformers 4.56 Cache API 默认 layer_idx=0），而 Qwen3 逐层循环共享同一 DynamicCache——首次完整 prefill 时 layer0 检查放行、layer0 attention 内 update 写入后，layer1 进守卫读到 layer0 非零长度即抛 136 误拒。B4 验收时的 FakeCache `get_seq_length(layer_idx=0)` 忽略层号、`update()` 不改状态，正是 GPT 指出的测试盲点——验收红绿通过不能覆盖该跨层条件，B 批验收该子项判据存在盲区，如实记录。

**影响面**：守卫合入在 main 846d939（10-13 晚）；DeepSeek minference 正式批（LB16 49.53 等）均生成于合入前，无数据污染。合入后尚无 minference_vendor 正式跑，误杀未实际触发过，但任何后续跑在首次 prefill 即崩——P1 修复排链。

**修复排链**：已并入 GPT 审计 C 批同一 worktree 追加派单（同文件避免双 agent 冲突），方案照建议口径——guard 显式接收 `self.layer_idx` 并读本层更新前长度（报错同层读数）；测试改按层存储且 `update()` 真正 append 的共享 cache 三态验证（layer0 首段 prefill → layer1 放行；当前层已有 prefix 且 q_len>1 → update 前拒绝；单 token decode 直通）。普通与 `-O` 双跑红绿，与 B 批套件同协议。
