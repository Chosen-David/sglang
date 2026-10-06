# M8-KernelC/D 基准：双池直写 + L1 fused vs 全 eager。
# 口径同 bench_m8_phase.py（合成数据、生产形状 n=32/S_cap=131072/Hkv=8/K1=128/
# budget=1024）——kernel 级 microbench，数值合成仅用于归因，输出对拍保证语义。
# 对拍判据：C 档 vs B 档须 torch.equal（topk 输入逐位一致）；含 D 档（L1 kernel
# 归约 1e-7 级）与全 eager 按既有口径 jaccard 1.0（并列排序/哨兵 lane 可不同）。
import os
import sys

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
import time

import torch

os.environ.setdefault("SGLANG_TLI_L2B_KERNEL", "1")
os.environ.setdefault("SGLANG_TLI_COMPACT_KERNEL", "1")
os.environ.setdefault("SGLANG_TLI_L1B_KERNEL", "1")

from sglang.srt.layers.attention.tli.config import TLIProfile
from sglang.srt.layers.attention.tli.indexer import TLIIndexer

dev = "cuda:0"
torch.manual_seed(0)

n, R = 32, 32
S_cap, Hkv, D = 131072, 8, 128

# 四份 profile（env 在构造时读取）：full(A+B+C+D) / C 档 / B 档 / 全 eager
os.environ["SGLANG_TLI_L2D_KERNEL"] = "1"
os.environ["SGLANG_TLI_L1B_KERNEL"] = "1"
prof_full = TLIProfile()
os.environ["SGLANG_TLI_L1B_KERNEL"] = "0"
prof_c = TLIProfile()
os.environ["SGLANG_TLI_L1B_KERNEL"] = "1"
os.environ["SGLANG_TLI_L2D_KERNEL"] = "0"
prof_b = TLIProfile()
os.environ["SGLANG_TLI_L2B_KERNEL"] = "0"
os.environ["SGLANG_TLI_COMPACT_KERNEL"] = "0"
os.environ["SGLANG_TLI_L1B_KERNEL"] = "0"
prof_e = TLIProfile()
os.environ["SGLANG_TLI_L2B_KERNEL"] = "1"
os.environ["SGLANG_TLI_COMPACT_KERNEL"] = "1"
os.environ["SGLANG_TLI_L2D_KERNEL"] = "1"
os.environ["SGLANG_TLI_L1B_KERNEL"] = "1"

idx_full = TLIIndexer(prof_full, head_dim=D).to(dev)
idx_c = TLIIndexer(prof_c, head_dim=D).to(dev)
idx_b = TLIIndexer(prof_b, head_dim=D).to(dev)
idx_e = TLIIndexer(prof_e, head_dim=D).to(dev)

nd2, d1, bs = idx_full.nd2, prof_full.coarse_dim, prof_full.block_size
NBLK_CAP = S_cap // bs

pool = {
    "kq_q": torch.randint(0, 16, (R, S_cap, Hkv, nd2), dtype=torch.uint8, device=dev),
    "kq_sc": torch.rand(R, S_cap, Hkv, device=dev) * 0.01,
    "kq_mn": (torch.rand(R, S_cap, Hkv, device=dev) - 0.5) * 0.1,
    "kmin": (torch.rand(R, NBLK_CAP, Hkv, d1, device=dev) - 0.5) * 0.2 - 0.1,
    "kmax": (torch.rand(R, NBLK_CAP, Hkv, d1, device=dev) - 0.5) * 0.2 + 0.1,
}
rows = torch.arange(n, device=dev)
q = torch.randn(n, Hkv * 4, D, device=dev) * 0.3

def run(idxer, S_list):
    return idxer.select_decode_batched(pool, rows, S_list, q)

def jaccard_sets(a, b, sent):
    ok = True
    for i in range(a.shape[0]):
        for h in range(a.shape[1]):
            sa = set(a[i, h].tolist()); sa.discard(sent)
            sb = set(b[i, h].tolist()); sb.discard(sent)
            if sa != sb:
                ok = False
    return ok

def bench(fn, reps=20, warmup=4):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2] * 1e3

# ---- 场景 1：uniform S = S_cap（131K 满行）----
S_list = [S_cap] * n
out_f = run(idx_full, S_list)
out_c = run(idx_c, S_list)
out_b = run(idx_b, S_list)
out_e = run(idx_e, S_list)
torch.cuda.synchronize()
assert torch.equal(out_c, out_b), "C 档 vs B 档输出不一致（应逐位相等）！"
print("[uniform 131K] C vs B: torch.equal PASS")
assert jaccard_sets(out_f, out_e, S_cap), "full vs eager 有效集不一致！"
print("[uniform 131K] full vs eager: 有效集一致 PASS")
ndiff = (out_f != out_c).sum().item()
print(f"[uniform 131K] full(D) vs C: 逐元素差 {ndiff}（L1 1e-7 归约级 tie 翻转预期内）")

# ---- 场景 2：mixed S（短行边界：far 池空/近池小/滑窗截断）----
g = torch.Generator(device=dev).manual_seed(7)
S_mixed = [int(x) for x in torch.randint(3000, S_cap + 1, (n,), generator=g, device=dev)]
out_f2 = run(idx_full, S_mixed)
out_c2 = run(idx_c, S_mixed)
out_e2 = run(idx_e, S_mixed)
torch.cuda.synchronize()
out_b2 = run(idx_b, S_mixed)
assert torch.equal(out_c2, out_b2), "mixed: C 档 vs B 档不一致！"
print("[mixed 3K-131K] C vs B: torch.equal PASS")
assert jaccard_sets(out_f2, out_e2, S_cap), "mixed: full vs eager 不一致！"
print("[mixed 3K-131K] full vs eager: 有效集一致 PASS")
assert jaccard_sets(out_f2, out_c2, S_cap), "mixed: full vs C 不一致！"
print("[mixed 3K-131K] full vs C: 有效集一致 PASS")

# ---- 计时 ----
t_f = bench(lambda: run(idx_full, S_list))
t_c = bench(lambda: run(idx_c, S_list))
t_b = bench(lambda: run(idx_b, S_list))
t_e = bench(lambda: run(idx_e, S_list), reps=10)
print(f"\nselect_decode_batched 全函数（bs=32 / S=131K，中位 20 reps）：")
print(f"  全 eager                     : {t_e:8.3f} ms")
print(f"  B 档（A+B）                  : {t_b:8.3f} ms  ({t_e/t_b:.2f}×)")
print(f"  C 档（+双池直写）            : {t_c:8.3f} ms  ({t_e/t_c:.2f}×，增量 {t_b/t_c:.2f}×)")
print(f"  full（+L1 fused，A+B+C+D）   : {t_f:8.3f} ms  ({t_e/t_f:.2f}×，增量 {t_c/t_f:.2f}×)")
