# B01/B02 修复专项验证（GPT 审查 2026-10-08 反例场景）
# 场景 1（B01）：S=4096, (α,β,γ)=(.125,.375,.625) → near 区仅 384 token
#   配额 768 → 384 个无效槽位。修复前转 0（token 0 出现 385 次）；
#   修复后哨兵 S（valid=False），token 0 有效位恰 1 次。
# 场景 2（B02）：early 行 t=999（L=1000 < K2=1024）→ 修复前前 24 token
#   重复 2 次；修复后每 token 恰一次 + 哨兵尾垫（真 dense 等价）。
# 场景 3（消费端数学）：q=0 等权 softmax + v_0=1 其余 0 →
#   修复后输出 = 1/有效位数（去重语义），≠ 修复前 385/1024=0.376。
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

# ---- 场景 1：晚期行 near 池不足 ----
q = torch.zeros(1, H, D, device=dev)  # q=0 → 分数全等，验证槽位语义
t_arr = torch.tensor([S - 1], device=dev)
sel = idxer.select_batched(index, q, t_arr)  # [1, Hkv, K2]
K2 = sel.shape[-1]
sent = (sel == S).sum().item()
tok0 = (sel == 0).sum().item()
n_valid = (sel < S).sum().item() // Hkv  # 每 head 独立 → 除 Hkv
# 预期：γ 悬崖配置 far_budget_cap=0 → far 池宽 0（v2 严格语义）；
#   near 384 哨兵/head（GPT 反例 invalid_zero_slots=384 同款场景）
assert sent == Hkv * 384, f"哨兵数 {sent} != {Hkv*384}"
assert tok0 == Hkv * 1, f"token 0 出现 {tok0} 次 != {Hkv}（修复前 385×Hkv）"
assert n_valid == 640, f"有效位 {n_valid} != 640"
print(f"[1] B01 near 池不足：哨兵 {sent//Hkv}/head（原 0 填充），"
      f"token0 有效位恰 1 次（原 385），有效位 640 — PASS")

# ---- 场景 2：early 行 identity grid（B02）----
q2 = torch.zeros(1, H, D, device=dev)
t2 = torch.tensor([999], device=dev)  # L=1000 < K2=1024
sel2 = idxer.select_batched(index, q2, t2)
row = sel2[0, 0]  # [K2]
valid2 = row < S
vals = row[valid2]
# 每个合法 token [0, 1000) 恰出现一次
assert sorted(vals.tolist()) == list(range(1000)), "identity grid 非 dense 等价"
assert (~valid2).sum().item() == K2 - 1000, "哨兵尾垫数错误"
print(f"[2] B02 early 行：每 token 恰一次（原前 24 token 重复 2 次），"
      f"哨兵尾垫 {K2-1000} — PASS")

# ---- 场景 3：消费端 valid softmax 数学（q=0 等权 + v_0=1）----
v = torch.zeros(S, Hkv, D, device=dev)
v[0] = 1.0  # GPT 反例：只有 token 0 的 value 非零
valid = sel < S  # [1, Hkv, K2] 消费端协议
sel_g = sel.clamp(max=S - 1)
out = torch.zeros(1, Hkv, D, device=dev)
for h in range(Hkv):
    v_sel_h = v[:, h][sel_g[0, h]]  # [K2, D] gather 模拟
    # softmax 前 -inf 屏蔽（等权分数 → softmax 只依赖 valid）
    w_h = valid[0, h].float()
    w_h = w_h / w_h.sum()
    out[0, h] = (w_h.unsqueeze(-1) * v_sel_h).sum(0)
expect = 1.0 / 640  # 去重语义
assert torch.allclose(out, torch.full_like(out, expect), atol=1e-6), \
    f"消费端输出 {out[0,0,:3].tolist()} != 1/640={expect}"
print(f"[3] 消费端 valid softmax：输出 = 1/640 = {expect:.6f}"
      f"（修复前 385/1024 = {385/1024:.6f}）— PASS")

print("\nB01/B02 修复验证 全部 PASS")
