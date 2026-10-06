# F1/F2/F3 单测（#127）：prefill 索引 chunk 增量化 vs 全量重建逐位一致。
#
# 验收口径（torch.equal 级，硬验收）：
#   1. indexer 级：build_block_index + update_block_index 增量链 ==
#      一次性 build_block_index 全量（5 键逐位；含块边界对齐/不对齐 chunk、
#      pool S 容量扩容触发）
#   2. backend 级：forward_extend 逐 chunk 走 F1 增量分支（spy 断言分支
#      选择），pool 行 5 键与全量重建逐位一致
#   3. S_st != prefix（跳变/branch miss）→ 回退全量重建且位级一致
#   4. kmeans 消融臂（far_kmeans）→ 回退全量重建
#   5. 增量 vs 强制全量（S=-1 复位）两 backend 的 forward_extend 输出
#      torch.equal（同索引位 → 同选择 → 同 attention）
#   6. F3：kernel 路径无 empty 行的 chunk，kq_f 反量化表零构建
#      （kq_unpack spy 计数 == 0）
# 运行：CUDA_VISIBLE_DEVICES=1 python test_incremental_prefill.py
import sys

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")

import torch

import sglang.srt.layers.attention.tli.indexer as tli_idx_mod
from sglang.srt.layers.attention.tli.config import TLIProfile
from sglang.srt.layers.attention.tli.indexer import TLIIndexer
from sglang.srt.layers.attention.tli.backend import TLISparseAttnBackend

dev = "cuda:0"

torch.manual_seed(20261006)
Hkv, D, H = 8, 128, 32  # Hkv/nd2 二次幂 → select_batched 走 M10 kernel 路径
G = H // Hkv
# chunk 划分：2200 首 chunk 跨 dense_threshold(2048)（走全量）；
# 64/512 与 block_size=64 对齐（开新块分支）；100/157/33/134/900 不对齐
# （尾块结合律合并分支）；累计 4100 > 初始 S_cap 4096（触发 _grow_pool_s）
CHUNKS = [2200, 100, 64, 157, 512, 33, 134, 900]
S_TOTAL = sum(CHUNKS)
assert S_TOTAL == 4100, S_TOTAL

k_real = (torch.randn(S_TOTAL, Hkv, D, device=dev) * 0.7).float()
v_real = (torch.randn(S_TOTAL, Hkv, D, device=dev) * 0.05).float()
qs = [(torch.randn(c, H, D, device=dev) * 0.5).float() for c in CHUNKS]

prof = TLIProfile()


def ref_index(S: int) -> dict:
    return TLIIndexer(prof, head_dim=D).to(dev).build_block_index(k_real[:S])


def assert_pool_row_bitexact(pool_l: dict, row: int, S: int, tag: str) -> None:
    """pool 行 5 键 vs 全量重建逐位一致（torch.equal 硬验收）。"""
    ref = ref_index(S)
    nblk = ref["nblk"]
    assert pool_l["S"][row] == S, f"{tag}: S 簿记 {pool_l['S'][row]} != {S}"
    assert torch.equal(pool_l["kq_q"][row, :S], ref["kq_q"][:S]), f"{tag}: kq_q 不逐位一致"
    assert torch.equal(pool_l["kq_sc"][row, :S], ref["kq_sc"][:S]), f"{tag}: kq_sc 不逐位一致"
    assert torch.equal(pool_l["kq_mn"][row, :S], ref["kq_mn"][:S]), f"{tag}: kq_mn 不逐位一致"
    assert torch.equal(pool_l["kmin"][row, :nblk], ref["kmin"]), f"{tag}: kmin 不逐位一致（尾块精确界被破坏）"
    assert torch.equal(pool_l["kmax"][row, :nblk], ref["kmax"]), f"{tag}: kmax 不逐位一致"


# ================= 1. indexer 级：增量链 == 全量（5 键逐位） =================
idxer = TLIIndexer(prof, head_dim=D).to(dev)
inc = None
cur = 0
for ci, c in enumerate(CHUNKS):
    if inc is None:
        inc = idxer.build_block_index(k_real[:c])
    else:
        inc = idxer.update_block_index(inc, k_real[cur : cur + c])
    cur += c
    ref = ref_index(cur)
    assert inc["S"] == cur == ref["S"] and inc["nblk"] == ref["nblk"]
    for key in ("kq_q", "kq_sc", "kq_mn"):
        assert torch.equal(inc[key][:cur], ref[key][:cur]), f"[1] chunk{ci} {key} 增量 != 全量"
    assert torch.equal(inc["kmin"][: ref["nblk"]], ref["kmin"]), f"[1] chunk{ci} kmin 增量 != 全量"
    assert torch.equal(inc["kmax"][: ref["nblk"]], ref["kmax"]), f"[1] chunk{ci} kmax 增量 != 全量"
print(f"[1] indexer 级增量链（{len(CHUNKS)} chunk，对齐/不对齐/扩容全覆盖）5 键逐位一致 PASS")


# ================= 2-6. backend 级（mock pool + req_to_token 间接寻址） =================
class FakePool:
    def __init__(self, k_pool, v_pool):
        self.k_pool, self.v_pool = k_pool, v_pool

    def get_kv_buffer(self, layer_id):
        return (self.k_pool, self.v_pool)

    def set_kv_buffer(self, layer, locs, k, v):
        raise RuntimeError("save_kv_cache=False 下不应写 KV")


class FakeRTP:
    def __init__(self, req_to_token):
        self.req_to_token = req_to_token


class FakeMC:
    head_dim = D
    num_key_value_heads = Hkv
    num_hidden_layers = 36


class FakeLayer:
    layer_id = 3
    scaling = D**-0.5


REQ = 7
POOL_N = S_TOTAL + 64
perm = torch.randperm(POOL_N, device=dev)
k_pool = torch.zeros(POOL_N, Hkv, D, device=dev)
v_pool = torch.zeros(POOL_N, Hkv, D, device=dev)
k_pool[perm[:S_TOTAL]] = k_real
v_pool[perm[:S_TOTAL]] = v_real
req_to_token = torch.zeros(8, POOL_N, dtype=torch.long, device=dev)
req_to_token[REQ, :S_TOTAL] = perm[:S_TOTAL]


def make_backend() -> TLISparseAttnBackend:
    pool = FakePool(k_pool, v_pool)

    class FR:
        model_config = FakeMC()
        device = dev
        token_to_kv_pool = pool
        req_to_token_pool = FakeRTP(req_to_token)

    return TLISparseAttnBackend(runner=FR())


def run_chunk(be: TLISparseAttnBackend, ci: int) -> torch.Tensor:
    prefix = sum(CHUNKS[:ci])
    nq = CHUNKS[ci]

    class FakeFB:
        req_pool_indices = torch.tensor([REQ], device=dev)
        extend_prefix_lens = torch.tensor([prefix], device=dev)
        extend_seq_lens = torch.tensor([nq], device=dev)

    out = be.forward_extend(
        qs[ci].reshape(nq, H * D),
        k_real[prefix : prefix + nq].reshape(nq, Hkv * D),
        v_real[prefix : prefix + nq].reshape(nq, Hkv * D),
        FakeLayer(), FakeFB(), save_kv_cache=False,
    )
    return out.reshape(nq, H, D)


def install_branch_spy(be: TLISparseAttnBackend) -> dict:
    """计数 build_block_index / update_block_index 调用（分支选择断言）。"""
    idx = be._get_indexer(3)
    spy = {"build": 0, "update": 0}
    _ob, _ou = idx.build_block_index, idx.update_block_index

    def _b(k):
        spy["build"] += 1
        return _ob(k)

    def _u(index, k_new):
        spy["update"] += 1
        return _ou(index, k_new)

    idx.build_block_index = _b
    idx.update_block_index = _u
    return spy


# ---- 2. forward_extend 逐 chunk 增量：pool 行位级一致 + 分支断言 ----
be = make_backend()
spy = install_branch_spy(be)
pool_l = be._get_pool(3)
row = None
outs_inc = []
s_cap0 = pool_l["S_cap"]
for ci in range(len(CHUNKS)):
    outs_inc.append(run_chunk(be, ci).clone())
    S = sum(CHUNKS[: ci + 1])
    if row is None:
        row = pool_l["row_of"][REQ]
    assert_pool_row_bitexact(pool_l, row, S, f"[2] chunk{ci}")
    if ci == 0:
        assert spy == {"build": 1, "update": 0}, f"[2] 首 chunk 应全量重建: {spy}"
    else:
        assert spy["update"] == ci and spy["build"] == 1, (
            f"[2] chunk{ci} 应走 F1 增量分支: {spy}"
        )
assert pool_l["S_cap"] > s_cap0, "[2] 末 chunk（S=4100>4096）应触发 pool S 扩容"
print(f"[2] forward_extend 增量分支：{len(CHUNKS)} chunk 全部 5 键逐位一致，"
      f"首 chunk 全量 + 后续 {len(CHUNKS) - 1} chunk 增量（spy={spy}），扩容触发 PASS")

# ---- 3. S_st != prefix（跳变/branch miss）→ 回退全量且位级一致 ----
be3 = make_backend()
spy3 = install_branch_spy(be3)
run_chunk(be3, 0)
pool_l3 = be3._get_pool(3)
row3 = pool_l3["row_of"][REQ]
prefix1 = CHUNKS[0]
pool_l3["S"][row3] = prefix1 - 7  # 模拟 S 跳变（≠ prefix）
run_chunk(be3, 1)
assert spy3["build"] == 2 and spy3["update"] == 0, f"[3] 跳变应回退全量重建: {spy3}"
assert_pool_row_bitexact(pool_l3, row3, prefix1 + CHUNKS[1], "[3] 跳变回退")
print("[3] S_st != prefix 跳变 → 回退全量重建且位级一致 PASS")

# ---- 4. kmeans 消融臂（far_kmeans）→ 回退全量重建 ----
be4 = make_backend()
spy4 = install_branch_spy(be4)
run_chunk(be4, 0)
be4.profile.far_kmeans = True  # cavg 臂：聚类索引无法增量维护
run_chunk(be4, 1)
assert spy4["build"] == 2 and spy4["update"] == 0, f"[4] kmeans 应回退全量重建: {spy4}"
assert_pool_row_bitexact(be4._get_pool(3), be4._get_pool(3)["row_of"][REQ],
                         CHUNKS[0] + CHUNKS[1], "[4] kmeans 回退")
print("[4] far_kmeans（cavg 消融臂）→ 回退全量重建且位级一致 PASS")

# ---- 5. 增量 vs 强制全量：forward_extend 输出 torch.equal ----
be5 = make_backend()
outs_full = []
for ci in range(len(CHUNKS)):
    pl = be5._get_pool(3)
    r = pl["row_of"].get(REQ)
    if r is not None and ci > 0:
        pl["S"][r] = -1  # 强制下一 chunk 走全量重建（对照组）
    outs_full.append(run_chunk(be5, ci).clone())
for ci in range(len(CHUNKS)):
    assert torch.equal(outs_inc[ci], outs_full[ci]), (
        f"[5] chunk{ci} 增量与全量重建的 forward_extend 输出不逐位一致"
    )
print(f"[5] 增量 vs 强制全量：{len(CHUNKS)} chunk 输出 torch.equal PASS")

# ---- 6. F3：kernel 路径无 empty 行的 chunk，kq_f 反量化表零构建 ----
be6 = make_backend()
run_chunk(be6, 0)
run_chunk(be6, 1)  # prefix=2200 ≥ near_len(2048)+far_lo(128) → _no_empty
unpack_spy = {"n": 0}
_orig_unpack = tli_idx_mod.kq_unpack


def _unpack_spy(*a, **kw):
    unpack_spy["n"] += 1
    return _orig_unpack(*a, **kw)


tli_idx_mod.kq_unpack = _unpack_spy
try:
    run_chunk(be6, 2)  # prefix=2300，kernel 慢路径无 empty 行 → kq_f 应零构建
finally:
    tli_idx_mod.kq_unpack = _orig_unpack
assert unpack_spy["n"] == 0, (
    f"[6] F3 失效：kernel 路径无 empty 行仍构建了 kq_f（{unpack_spy['n']} 次）"
)
# 对照：有 empty 行的 chunk（首 chunk，t < near_len+far_lo 的早段行存在）
# 允许构建（兜底路径语义要求）
print("[6] F3 惰性反量化表：kernel 路径无 empty 行时 kq_unpack 零调用 PASS")

print("\n全部 6 组验收 PASS：F1/F2/F3 增量 prefill 与全量重建逐位一致")
