# F5（#129）e2e 异步化验收单测：P1 host 同步消除 + P2 侧流索引构建。
#
# 验收口径：
#   [1] kernel 慢路径（默认 profile，8 chunk 增量链）：改造后 vs 改造前
#       （HEAD 快照 /tmp/tli_head）forward_extend 输出 torch.equal
#   [2] eager 慢路径（use_prefill_kernel=False + S>K1*bs*nblk 边界，
#       nblk>K1 强制慢路径）：vs HEAD 逐位一致——覆盖三处 F5 消除点
#       （Tc 静态上界 / empty 无条件向量 topk / early 无条件向量修复）
#   [3] skip_far 臂（eager）：vs HEAD 逐位一致——覆盖 rank 截断复刻旧
#       动态宽度（K1 不收缩 + device 标量截断）；附 select() 单行对拍
#       （decode 侧同一 F5 改动的位级验收）
#   [4] 侧流开 vs 关：kernel + eager 两场景输出 torch.equal（P2 验收）
#   [5] host 同步计数：monkeypatch torch.Tensor.{item,__bool__,__int__,
#       tolist}——eager 中段 chunk HEAD(6) vs 改造后(2)（残余=fast_path
#       豁免 1 + req tolist 1）；kernel 中段 chunk HEAD(3) vs 改造后(1)
#   [6] 侧流计数 == 同步执行计数（侧流/event 本身零新增 host 同步）
#
# 运行：CUDA_VISIBLE_DEVICES=1 python test_async_prefill.py
import os
import shutil
import subprocess
import sys

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
sys.path.insert(0, "/tmp")

import torch

from sglang.srt.layers.attention.tli.config import TLIProfile
from sglang.srt.layers.attention.tli.backend import TLISparseAttnBackend

dev = "cuda:0"
SGLANG_ROOT = "/home/wangyuanshuo02/sglang"
TILI_DIR = "python/sglang/srt/layers/attention/tli"


def build_head_ref() -> None:
    """从 git HEAD 抽取改造前快照到 /tmp/tli_head（相对导入改写）。

    HEAD 三文件（backend/config/indexer）即 F5 改造前基线；kernels.py
    F5 零改动，一并抽取保持包自包含。tli 内部绝对导入改写为相对导入，
    sglang 运行时依赖（base_attn_backend）保持绝对（未改动）。
    """
    if os.path.exists("/tmp/tli_head/backend.py"):
        return
    os.makedirs("/tmp/tli_head", exist_ok=True)
    for f in ("__init__.py", "config.py", "indexer.py", "kernels.py", "backend.py"):
        blob = subprocess.check_output(
            ["git", "-C", SGLANG_ROOT, "show", f"HEAD:{TILI_DIR}/{f}"]
        )
        with open(f"/tmp/tli_head/{f}", "wb") as fp:
            fp.write(blob)
    for f in ("backend.py", "indexer.py"):
        path = f"/tmp/tli_head/{f}"
        with open(path) as fp:
            src = fp.read()
        src = src.replace("from sglang.srt.layers.attention.tli.", "from .")
        with open(path, "w") as fp:
            fp.write(src)


build_head_ref()
import tli_head.backend as head_backend_mod  # noqa: E402

HeadBackend = head_backend_mod.TLISparseAttnBackend

torch.manual_seed(20261007)
Hkv, D, H = 8, 128, 32
G = H // Hkv


# ---------------- mock 基建（对齐 test_incremental_prefill.py） ----------------
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


class FakeFB:
    """CPU 镜像 + device 张量双提供：改造后读镜像，HEAD 读 device 兜底。"""

    def __init__(self, req, prefix, nq):
        self.req_pool_indices = torch.tensor([req], device=dev)
        self.extend_prefix_lens = torch.tensor([prefix], device=dev)
        self.extend_seq_lens = torch.tensor([nq], device=dev)
        self.extend_prefix_lens_cpu = [prefix]
        self.extend_seq_lens_cpu = [nq]


REQ = 7


def run_prefill(be, k_real, v_real, qs, chunks, layer=FakeLayer()):
    """逐 chunk 走 forward_extend（增量链），返回各 chunk 输出 clone。"""
    outs = []
    prefix = 0
    for ci, c in enumerate(chunks):
        fb = FakeFB(REQ, prefix, c)
        out = be.forward_extend(
            qs[ci].reshape(c, H * D),
            k_real[prefix : prefix + c].reshape(c, Hkv * D),
            v_real[prefix : prefix + c].reshape(c, Hkv * D),
            layer, fb, save_kv_cache=False,
        )
        outs.append(out.reshape(c, H, D).clone())
        prefix += c
    return outs


def make_tensors(chunks, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    S = sum(chunks)
    k_real = (torch.randn(S, Hkv, D, generator=g) * 0.7).to(dev).float()
    v_real = (torch.randn(S, Hkv, D, generator=g) * 0.05).to(dev).float()
    qs = [
        (torch.randn(c, H, D, generator=g) * 0.5).to(dev).float()
        for c in chunks
    ]
    return k_real, v_real, qs


def make_backend(cls, pool_n, k_real, v_real, req_to_token):
    pool = FakePool(
        torch.zeros(pool_n, Hkv, D, device=dev),
        torch.zeros(pool_n, Hkv, D, device=dev),
    )
    pool.k_pool[: k_real.shape[0]] = k_real
    pool.v_pool[: v_real.shape[0]] = v_real

    class FR:
        model_config = FakeMC()
        device = dev
        token_to_kv_pool = pool
        req_to_token_pool = FakeRTP(req_to_token)

    return cls(runner=FR())


def req_to_token_for(S_total):
    rt = torch.zeros(8, S_total + 64, dtype=torch.long, device=dev)
    rt[REQ, :S_total] = torch.arange(S_total, device=dev)
    return rt


# ---------------- 场景定义 ----------------
# A：kernel 慢路径（默认 profile；Hkv=8/nd2=32 二次幂 + n≥2 → _kern_ok）。
#    首 chunk 2200 跨 dense_threshold；含对齐/不对齐 chunk 与 S 扩容。
CHUNKS_A = [2200, 100, 64, 157, 512, 33, 134, 900]
# B：eager 慢路径（use_prefill_kernel=False；S=9400 > K1*bs=8192 → nblk=147
#    > K1=128，并集 < 全块，fast_path 判定 False → 慢路径必然触发）。
CHUNKS_B = [2400, 2200, 2600, 2200]


def scenario(cls, chunks, seed, use_prefill_kernel=True, skip_far=False):
    k_real, v_real, qs = make_tensors(chunks, seed)
    be = make_backend(cls, sum(chunks) + 64, k_real, v_real, req_to_token_for(sum(chunks)))
    if not use_prefill_kernel:
        be.profile.use_prefill_kernel = False
    if skip_far:
        be._get_indexer(3).skip_far = True
    outs = run_prefill(be, k_real, v_real, qs, chunks)
    return be, outs


# ================= [1] kernel 慢路径：vs HEAD 逐位一致 =================
be_m, outs_A_m = scenario(TLISparseAttnBackend, CHUNKS_A, seed=11)
be_h, outs_A_h = scenario(HeadBackend, CHUNKS_A, seed=11)
for ci in range(len(CHUNKS_A)):
    assert torch.equal(outs_A_m[ci], outs_A_h[ci]), f"[1] chunk{ci} kernel 路径 != HEAD"
assert be_m._side_stream is not None, "[1] 默认配置应创建侧流"
print(f"[1] kernel 慢路径：{len(CHUNKS_A)} chunk vs HEAD 逐位一致 PASS"
      f"（侧流默认开，req 行 S={be_m._get_pool(3)['S'][be_m._get_pool(3)['row_of'][REQ]]}）")

# ================= [2] eager 慢路径：vs HEAD 逐位一致 =================
be_m2, outs_B_m = scenario(
    TLISparseAttnBackend, CHUNKS_B, seed=22, use_prefill_kernel=False
)
be_h2, outs_B_h = scenario(HeadBackend, CHUNKS_B, seed=22, use_prefill_kernel=False)
for ci in range(len(CHUNKS_B)):
    assert torch.equal(outs_B_m[ci], outs_B_h[ci]), f"[2] chunk{ci} eager 路径 != HEAD"
print(f"[2] eager 慢路径（S=9400>nblk 边界，Tc 静态上界/empty/early 三处 F5 点）："
      f"{len(CHUNKS_B)} chunk vs HEAD 逐位一致 PASS")

# ================= [3] skip_far 臂：vs HEAD 逐位一致 =================
be_m3, outs_C_m = scenario(
    TLISparseAttnBackend, CHUNKS_B, seed=33, use_prefill_kernel=False, skip_far=True
)
be_h3, outs_C_h = scenario(
    HeadBackend, CHUNKS_B, seed=33, use_prefill_kernel=False, skip_far=True
)
for ci in range(len(CHUNKS_B)):
    assert torch.equal(outs_C_m[ci], outs_C_h[ci]), f"[3] chunk{ci} skip_far != HEAD"
# select() 单行对拍（decode 侧同一 F5 改动：K1 静态 + device rank 截断）
k_real, v_real, qs = make_tensors(CHUNKS_B, seed=33)
idx_m = be_m3._get_indexer(3).build_block_index(k_real.float())
idx_h = be_h3._get_indexer(3).build_block_index(k_real.float())
sel_m = be_m3._get_indexer(3).select(idx_m, qs[-1][-1:].float(), sum(CHUNKS_B) - 1)
sel_h = be_h3._get_indexer(3).select(idx_h, qs[-1][-1:].float(), sum(CHUNKS_B) - 1)
assert torch.equal(sel_m, sel_h), "[3] select() skip_far 单行 != HEAD"
print(f"[3] skip_far 臂（rank 截断复刻旧动态宽度）：{len(CHUNKS_B)} chunk + "
      f"select() 单行 vs HEAD 逐位一致 PASS")


# ================= [4] 侧流开 vs 关：输出 torch.equal =================
def scenario_side(side: bool, chunks, seed, use_prefill_kernel=True):
    k_real, v_real, qs = make_tensors(chunks, seed)
    be = make_backend(
        TLISparseAttnBackend, sum(chunks) + 64, k_real, v_real,
        req_to_token_for(sum(chunks)),
    )
    be.profile.use_side_stream = side
    if not use_prefill_kernel:
        be.profile.use_prefill_kernel = False
    return run_prefill(be, k_real, v_real, qs, chunks)


for tag, chunks, seed, upk in (
    ("kernel", CHUNKS_A, 11, True),
    ("eager", CHUNKS_B, 22, False),
):
    outs_on = scenario_side(True, chunks, seed, use_prefill_kernel=upk)
    outs_off = scenario_side(False, chunks, seed, use_prefill_kernel=upk)
    for ci in range(len(chunks)):
        assert torch.equal(outs_on[ci], outs_off[ci]), (
            f"[4] {tag} chunk{ci} 侧流开/关输出不一致"
        )
print("[4] 侧流开 vs 关：kernel + eager 两场景全部 chunk 输出 torch.equal PASS")


# ================= [5]/[6] host 同步计数 =================
class SyncCounter:
    """monkeypatch 计数四类 host 同步入口（CUDA 张量上的
    item/__bool__/int/tolist；场景内张量全在 CUDA，计数即同步数）。"""

    NAMES = ("item", "__bool__", "__int__", "tolist")

    def __enter__(self):
        self.n = 0
        self.orig = {}
        for name in self.NAMES:
            orig = getattr(torch.Tensor, name)
            self.orig[name] = orig

            def wrap(a, *args, _o=orig, **kw):
                self.n += 1
                return _o(a, *args, **kw)

            setattr(torch.Tensor, name, wrap)
        return self

    def __exit__(self, *a):
        for name, orig in self.orig.items():
            setattr(torch.Tensor, name, orig)


def counted_prefill(cls, chunks, seed, use_prefill_kernel, side, warm_pass=True):
    """跑两遍（第一遍触发 triton 编译/autotune），计数第二遍各 chunk
    forward_extend 的 host 同步数，返回 per-chunk 计数列表。"""
    k_real, v_real, qs = make_tensors(chunks, seed)
    be = make_backend(
        cls, sum(chunks) + 64, k_real, v_real, req_to_token_for(sum(chunks))
    )
    if not use_prefill_kernel:
        be.profile.use_prefill_kernel = False
    if cls is not TLISparseAttnBackend:
        side = False  # HEAD 无侧流
    else:
        be.profile.use_side_stream = side
    if warm_pass:
        run_prefill(be, k_real, v_real, qs, chunks)
    counts = []
    prefix = 0
    for ci, c in enumerate(chunks):
        torch.cuda.synchronize()
        fb = FakeFB(REQ, prefix, c)
        with SyncCounter() as sc:
            be.forward_extend(
                qs[ci].reshape(c, H * D),
                k_real[prefix : prefix + c].reshape(c, Hkv * D),
                v_real[prefix : prefix + c].reshape(c, Hkv * D),
                FakeLayer(), fb, save_kv_cache=False,
            )
        counts.append(sc.n)
        prefix += c
    return counts


# [5] eager 慢路径计数：select_batched 的 row_chunk=64 循环内每 row-chunk
# 各触发一次 fast_path 豁免同步（HEAD 另有 Tc/empty 两处 → 3 次/row-chunk）。
# 改造后中段 chunk 残余 = ceil(nq/64) 次 fast_path 豁免 + 1 次 req tolist。
cnt_h_B = counted_prefill(HeadBackend, CHUNKS_B, 22, use_prefill_kernel=False, side=False)
cnt_m_B = counted_prefill(
    TLISparseAttnBackend, CHUNKS_B, 22, use_prefill_kernel=False, side=True
)
mid_h, mid_m = cnt_h_B[1], cnt_m_B[1]
n_rc = (CHUNKS_B[1] + 64 - 1) // 64  # chunk1 的 row-chunk 数
print(f"[5] eager 慢路径中段 chunk（chunk1，prefix=2400，{n_rc} row-chunk）host 同步计数："
      f"HEAD={mid_h} → 改造后={mid_m}（全 chunk HEAD={cnt_h_B} vs 改造后={cnt_m_B}）")
assert mid_m < mid_h, f"[5] 同步未减少: {mid_m} vs {mid_h}"
assert mid_m == n_rc + 1, (
    f"[5] 改造后残余同步超预期（{n_rc} 次 fast_path 豁免 + 1 次 req tolist）: {mid_m}"
)
assert mid_h >= 2 * n_rc + 3, f"[5] HEAD 基线计数异常（应 ≥2×row-chunk+3）: {mid_h}"

# kernel 路径计数：HEAD 3（cumsum tolist + int×2）vs 改造后 1（req tolist）
cnt_h_A = counted_prefill(HeadBackend, CHUNKS_A, 11, use_prefill_kernel=True, side=False)
cnt_m_A = counted_prefill(
    TLISparseAttnBackend, CHUNKS_A, 11, use_prefill_kernel=True, side=True
)
mid_h, mid_m = cnt_h_A[1], cnt_m_A[1]
print(f"[5] kernel 路径中段 chunk（chunk1，nq=100 < row_chunk）host 同步计数："
      f"HEAD={mid_h} → 改造后={mid_m}"
      f"（全 chunk HEAD={cnt_h_A} vs 改造后={cnt_m_A}）")
assert mid_m < mid_h and mid_m == 1, (
    f"[5] kernel 路径改造后残余同步超预期（仅 req tolist 1）: {mid_m}"
)
print("[5] host 同步计数验收 PASS（两路径均严格减少且达预期残余）")

# [6] 侧流开/关计数一致（侧流与 event 零新增 host 同步）
cnt_off_B = counted_prefill(
    TLISparseAttnBackend, CHUNKS_B, 22, use_prefill_kernel=False, side=False
)
cnt_off_A = counted_prefill(
    TLISparseAttnBackend, CHUNKS_A, 11, use_prefill_kernel=True, side=False
)
assert cnt_off_B == cnt_m_B, f"[6] eager 侧流计数漂移: {cnt_off_B} vs {cnt_m_B}"
assert cnt_off_A == cnt_m_A, f"[6] kernel 侧流计数漂移: {cnt_off_A} vs {cnt_m_A}"
print("[6] 侧流开/关 host 同步计数逐 chunk 一致 PASS")

print("\n全部 6 组验收 PASS：F5 P1（host 同步消除）+ P2（侧流索引构建）"
      "位级一致性与同步消除双口径闭环")
