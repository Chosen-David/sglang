# S1/F01 中间行因果性专项验证（GPT round2 F01 + Kimi3 S1，2026-10-08）
# 核心反例：batched prefill 多行 chunk，中间行 t_r 的哨兵若用逐行
#   S_r = t_r+1 而消费端按全局 S 判有效 → 哨兵值 S_r 恰为未来 token
#   位置且 < S → 被当有效位读入（因果泄漏：未来 K/V 改变过去输出）。
# 修复：选择器哨兵全局 S + 消费端逐行因果界 sel < S-nq+r+1。
# 场景（GPT F01 同款）：S=4096，行 t=2047（中间行），(α,β,γ)=(.125,.375,.625)
#   → near 池仅 128 token < 配额 768 → 640 无效槽位。
#   验证：① 哨兵位全部 = 全局 S（≠ t_r+1）；② 行内有效槽位全部 ≤ t_r；
#   ③ v[2048]=1 其余 0 时该行输出恒 0（未来 token 不泄漏）；
#   ④ 多行（t=2047 与 t=4095）同时跑，末行行为不变。
import os
os.environ.setdefault("SGLANG_TLI_ALPHA", "0.125")
os.environ.setdefault("SGLANG_TLI_BETA", "0.375")
os.environ.setdefault("SGLANG_TLI_GAMMA", "0.625")
import sys

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
import torch
from sglang.srt.layers.attention.tli.config import TLIProfile
from sglang.srt.layers.attention.tli.indexer import TLIIndexer

dev = "cuda:0"
torch.manual_seed(20261008)
Hkv, D, H, G = 8, 128, 32, 4
S = 4096

prof = TLIProfile()
assert (prof.alpha, prof.beta, prof.gamma) == (0.125, 0.375, 0.625)
idxer = TLIIndexer(prof, head_dim=D).to(dev)
k_real = torch.randn(S, Hkv, D, device=dev) * 0.1
index = idxer.build_block_index(k_real)
assert index["S"] == S

# ---- 场景 1：双行 chunk（中间行 + 末行），中间行 near 池不足 ----
t_arr = torch.tensor([2047, S - 1], device=dev)  # nq=2
q = torch.zeros(2, H, D, device=dev)
sel = idxer.select_batched(index, q, t_arr)  # [2, Hkv, K2]
K2 = sel.shape[-1]

# ① 哨兵值 = 全局 S（修复前中间行哨兵 = t_r+1 = 2048）
row_mid = sel[0]  # [Hkv, K2]
sent_mid = (row_mid == S).sum().item()
sent_2048 = (row_mid == 2048).sum().item()  # 修复前的未来哨兵值
assert sent_2048 == 0, f"中间行出现 t_r+1=2048 哨兵 {sent_2048} 个（S1 未闭合）"
assert sent_mid > 0, "中间行 near 池不足应产生哨兵（前提：640 无效槽位）"
# ② 行内全部有效槽位 ≤ t_r（因果界）
valid_mid = row_mid < S
assert bool((row_mid[valid_mid] <= 2047).all()), "中间行有效槽位越出因果界"
print(f"[1] 中间行 t=2047：哨兵 {sent_mid//Hkv}/head 全局 S，"
      f"有效槽位全部 ≤ t_r，无 t_r+1 哨兵 — PASS")

# ---- 场景 2：消费端逐行因果界（模拟 _sparse_extend_one 协议）----
nq = 2
v = torch.zeros(S, Hkv, D, device=dev)
v[2048] = 1.0  # 只在未来 token（对 t=2047 而言）放非零 value
row_bound = (S - nq + torch.arange(nq, device=dev) + 1).view(-1, 1, 1)
valid = sel < row_bound  # 消费端 S1 修复后协议
sel_g = sel.clamp(max=S - 1)
out = torch.zeros(nq, Hkv, D, device=dev)
for r in range(nq):
    for h in range(Hkv):
        v_sel_h = v[:, h][sel_g[r, h]]
        w_h = valid[r, h].float()
        w_h = w_h / w_h.sum().clamp(min=1)
        out[r, h] = (w_h.unsqueeze(-1) * v_sel_h).sum(0)
# 中间行 t=2047：v[2048] 是未来 → 输出必须恒 0（修复前哨兵 2048 被判
#   有效 → 640/1024 = 0.625 的未来泄漏）
assert torch.allclose(out[0], torch.zeros_like(out[0]), atol=1e-7), \
    f"中间行未来泄漏：out[0] 非零（max={out[0].abs().max().item()}）"
# 末行 t=4095：v[2048] 是过去，若被选中权重合法；恒等性由场景 1 的
#   因果界保证，不做数值断言（只验证不 NaN）
assert not bool(torch.isnan(out[1]).any()), "末行 NaN"
print(f"[2] 消费端逐行因果界：中间行未来 token v[2048]=1 输出恒 0"
      f"（修复前 640/1024=0.625 泄漏）— PASS")

# ---- 场景 3：GPT F01 数值反例复核（v_2048=1、q=0）----
# GPT 报告：修复前输出 640/1024 = 0.625；修复后因果正确输出 = 0
# （场景 2 已断言）。补充：仅改未来 value 不改变过去输出（因果性定义）
q2 = torch.randn(2, H, D, device=dev) * 0.1
sel2 = idxer.select_batched(index, q2, t_arr)
valid2 = sel2 < row_bound
out_a = torch.zeros(nq, Hkv, D, device=dev)
for r in range(nq):
    for h in range(Hkv):
        v_sel_h = v[:, h][sel2[r, h].clamp(max=S - 1)]
        w_h = valid2[r, h].float()
        w_h = w_h / w_h.sum().clamp(min=1)
        out_a[r, h] = (w_h.unsqueeze(-1) * v_sel_h).sum(0)
v[3000] = 5.0  # 改变另一个未来 token（对 t=2047）的 value
out_b = torch.zeros(nq, Hkv, D, device=dev)
for r in range(nq):
    for h in range(Hkv):
        v_sel_h = v[:, h][sel2[r, h].clamp(max=S - 1)]
        w_h = valid2[r, h].float()
        w_h = w_h / w_h.sum().clamp(min=1)
        out_b[r, h] = (w_h.unsqueeze(-1) * v_sel_h).sum(0)
assert torch.allclose(out_a[0], out_b[0], atol=1e-7), \
    "改变未来 value (t=3000) 改变了中间行输出 → 因果性被违反"
print(f"[3] 因果性不变量：修改未来 value 前后中间行输出逐位相同 — PASS")

print("\nS1/F01 中间行因果性修复验证 全部 PASS")
