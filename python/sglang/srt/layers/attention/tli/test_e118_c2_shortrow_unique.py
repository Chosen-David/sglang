# E118 C2 修复验证（GPT 审计 sglang_twolevel_audit C2/P1，基线 ee256310b）
# 红绿口径：
#   红（修复前）：_select_decode_taskmd e64 分支短真实行（S < sink+swa）
#     的 sink 区 [0,sink) 与 swa 区 [S-swa,S) 重叠，两区直接拼接 →
#     重叠 token 重复计权（审计 S=160 反例：96 token 双计，256 lane 覆盖
#     160 token，图内稀疏输出 ≠ dense）；veto_cuda_graph 对 e64
#     taskmd 短行仍按「S ≤ token_budget 等价 dense」豁免。
#   绿（修复后）：重叠行输出 identity [0,S) + 哨兵尾垫（唯一 + dense
#     等价，branch-free 保 CUDA graph 可录制）；e64 taskmd 短真实行
#     （1 < S ≤ dense_threshold）veto 回 eager（γ 配额切分使「等价
#     dense」前提不成立，见 backend.py veto 注释重推导）。
# 运行方式：同 test_e118_c1_sentinel_fix.py（E118_SGLANG_ROOT 指向
#   修复前基点树跑红；缺省按本文件所在树跑绿）。
import os
import sys

_ROOT = os.environ.get("E118_SGLANG_ROOT")
if _ROOT:
    sys.path.insert(0, _ROOT)
else:
    sys.path.insert(
        0,
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
        ),
    )

import torch

from sglang.srt.layers.attention.tli.backend import TLISparseAttnBackend
from sglang.srt.layers.attention.tli.config import TLIProfile
from sglang.srt.layers.attention.tli.indexer import TLIIndexer

dev = "cuda:0"
torch.manual_seed(20261011)
Hkv, D, G = 2, 128, 1
H = Hkv * G
S_CAP, NBLK_CAP = 1024, 128


def make_taskmd_profile() -> TLIProfile:
    """e64 分区臂（审计口径 α/β/γ = .125/.125/.375，minmax/minmax）。"""
    p = TLIProfile()
    p.taskmd = True
    p.alpha, p.beta, p.gamma = 0.125, 0.125, 0.375
    p.far_method, p.near_method = "minmax", "minmax"
    return p


def build_pool(idxer: TLIIndexer, lens: list[int]):
    """按行长度 build 全量索引并拷入共享 pool 张量（M5 布局）。"""
    nd2 = idxer.nd2
    d1 = idxer.profile.coarse_dim
    pool = {
        "kq_q": torch.zeros(len(lens), S_CAP, Hkv, nd2, dtype=torch.uint8, device=dev),
        "kq_sc": torch.zeros(len(lens), S_CAP, Hkv, dtype=torch.float32, device=dev),
        "kq_mn": torch.zeros(len(lens), S_CAP, Hkv, dtype=torch.float32, device=dev),
        "kmin": torch.zeros(len(lens), NBLK_CAP, Hkv, d1, dtype=torch.float32, device=dev),
        "kmax": torch.zeros(len(lens), NBLK_CAP, Hkv, d1, dtype=torch.float32, device=dev),
        "S": [0] * len(lens),
    }
    for r, S in enumerate(lens):
        k = torch.randn(S, Hkv, D, device=dev) * 0.1
        idx = idxer.build_block_index(k)
        nblk = idx["nblk"]
        assert nblk <= NBLK_CAP and S <= S_CAP
        pool["kq_q"][r, :S] = idx["kq_q"]
        pool["kq_sc"][r, :S] = idx["kq_sc"]
        pool["kq_mn"][r, :S] = idx["kq_mn"]
        pool["kmin"][r, :nblk] = idx["kmin"]
        pool["kmax"][r, :nblk] = idx["kmax"]
        pool["S"][r] = S
    return pool


# ================= 场景 1：短行 identity + 长行不受扰 =================
prof = make_taskmd_profile()
idxer = TLIIndexer(prof, head_dim=D).to(dev)
lens = [160, 1024]  # 行 0：S=160 审计反例（sink [0,128) ∩ swa [32,160)）
pool = build_pool(idxer, lens)
q = torch.randn(len(lens), H, D, device=dev) * 0.1
sel = idxer._select_decode_taskmd(
    pool, torch.arange(len(lens), device=dev),
    torch.tensor(lens, device=dev), q,
)  # [n, Hkv, K2]
K2 = sel.shape[-1]
SENT = S_CAP
assert K2 == prof.token_budget, f"输出恒宽 {K2} != token_budget"

# ---- 行 0（S=160）：唯一 + dense 等价（= identity [0,160)）----
for h in range(Hkv):
    lanes = sel[0, h].tolist()
    valid = sorted(x for x in lanes if x < SENT)
    dup = len(valid) - len(set(valid))
    assert dup == 0, f"[短行 head{h}] {dup} 个重复槽位（sink∩swa 重复计权）"
    assert valid == list(range(160)), (
        f"[短行 head{h}] 有效集 != identity [0,160)（覆盖 {len(valid)}"
        f" 唯一 {len(set(valid))}，前 8 {valid[:8]}）——图内输出非 dense 等价"
    )
    n_sent = sum(x == SENT for x in lanes)
    assert n_sent == K2 - 160, f"[短行 head{h}] 哨兵 {n_sent} != {K2-160}"
print(f"[1] S=160 短行：每 token 恰一次（identity + 哨兵尾垫 {K2-160}）— PASS")

# ---- 行 1（S=1024 ≥ sink+swa=256）：不触发 identity，分区选择保持唯一 ----
for h in range(Hkv):
    lanes = sel[1, h].tolist()
    valid = sorted(x for x in lanes if x < SENT)
    dup = len(valid) - len(set(valid))
    assert dup == 0, f"[长行 head{h}] {dup} 个重复槽位"
    # γ 配额切分：384 far + 384 near(实际 near 区 128 满选+哨兵) +
    # 128 sink + 128 swa → 有效 768 < 1024（非 dense，e64 语义保持）
    assert len(valid) == 768, (
        f"[长行 head{h}] 有效 {len(valid)} != 768（e64 分区口径被 identity 修复波及？）"
    )
print(f"[1] S=1024 长行：identity 修复不波及（分区选择 {768} 有效 lane，唯一）— PASS")

# ---- 混批复核：两行同一调用内互不影响（branch-free per-row where）----
assert sel.shape == (2, Hkv, K2)
print(f"[1] 混批静态宽 [2, {Hkv}, {K2}] 保持（CUDA graph 可录制前提）— PASS")

# ================= 场景 2：veto_cuda_graph 豁免重推导 =================
class _FB:
    def __init__(self, lens):
        self.seq_lens_cpu = torch.tensor(lens)


class _BE:
    pass


def veto(prof_, lens):
    be = _BE()
    be.profile = prof_
    be.dense_threshold = prof_.dense_threshold
    return TLISparseAttnBackend.veto_cuda_graph(be, _FB(lens))


# e64 taskmd：短真实行 160（S ≤ token_budget，旧豁免区）→ 必须 veto
#（γ 配额切分破坏「等价 dense」前提；eager 该区间走 _dense_attn）
assert veto(prof, [160]) is True, (
    "e64 taskmd S=160 未 veto——图内 γ 稀疏 ≠ eager dense（C2 关联豁免失效）"
)
# e64 taskmd：token_budget < S ≤ dense_threshold → 沿既有 veto
assert veto(prof, [1500]) is True
# e64 taskmd：全长行（> dense_threshold）→ 不 veto（图内稀疏属既知口径）
assert veto(prof, [3000]) is False
# e64 taskmd：pad 行（seq_len=1）不触发 veto（哨兵行语义安全）
assert veto(prof, [1]) is False
print("[2] e64 taskmd：短真实行全部 veto 回 eager，pad/长行不受影响 — PASS")

# 旧 B'（taskmd=False）：原豁免语义逐位不变
prof_b = TLIProfile()
prof_b.taskmd = False
assert veto(prof_b, [160]) is False, "旧 B' S=160 应保持豁免（数学等价 dense）"
assert veto(prof_b, [1500]) is True
assert veto(prof_b, [3000]) is False
print("[2] 旧 B' 路径：S ≤ token_budget 豁免逐位不变 — PASS")

# taskmd 单池（α=0，e64=False）：整行 topk k ≥ S，等价 dense 同旧 B' → 豁免保持
prof_sp = make_taskmd_profile()
prof_sp.alpha, prof_sp.beta = 0.0, 0.0
assert veto(prof_sp, [160]) is False, "taskmd 单池 S=160 应保持豁免"
print("[2] taskmd 单池：豁免保持 — PASS")

print("E118 C2 全部场景 PASS：短行唯一 + dense 等价，veto 豁免按 e64 重推导")
