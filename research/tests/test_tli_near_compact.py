# M8-topk near 池压缩对拍测试：SGLANG_TLI_NEAR_COMPACT on/off 输出有效集一致 + 计时
# 判据：①uniform 131K 与 mixed 3K-131K 两场景，最终输出 [n,Hkv,W] 的
#   有效集（去哨兵）逐 (a,h) 一致（tie 翻转容忍对称差 ≤2，与 M10 判据同款）；
# ②输出形状一致（拼接宽度静态不变）；③计时对比。
import os
import sys

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
import time

import torch

for k in ["L2B", "COMPACT", "L1B", "L2D"]:
    os.environ.setdefault(f"SGLANG_TLI_{k}_KERNEL", "1")

from sglang.srt.layers.attention.tli.config import TLIProfile
from sglang.srt.layers.attention.tli.indexer import TLIIndexer

dev = "cuda:0"
torch.manual_seed(0)
n, R = 32, 32
S_cap, Hkv, D = 131072, 8, 128

os.environ["SGLANG_TLI_NEAR_COMPACT"] = "1"
prof_on = TLIProfile()
os.environ["SGLANG_TLI_NEAR_COMPACT"] = "0"
prof_off = TLIProfile()
os.environ["SGLANG_TLI_NEAR_COMPACT"] = "1"

idx_on = TLIIndexer(prof_on, head_dim=D).to(dev)
idx_off = TLIIndexer(prof_off, head_dim=D).to(dev)

nd2, d1, bs = idx_on.nd2, prof_on.coarse_dim, prof_on.block_size
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


def set_diff(a, b, sent):
    """逐 (a,h) 有效集对称差；返回 (max_diff, n_mismatch_rows)"""
    md, nmis = 0, 0
    for i in range(a.shape[0]):
        for h in range(a.shape[1]):
            sa = set(a[i, h].tolist()); sa.discard(sent)
            sb = set(b[i, h].tolist()); sb.discard(sent)
            d = len(sa ^ sb)
            md = max(md, d)
            if d:
                nmis += 1
    return md, nmis


def bench(fn, reps=30, warmup=6):
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


ok = True
for tag, S_list in [
    ("uniform 131K", [S_cap] * n),
    ("mixed 3K-131K", [int(x) for x in torch.randint(3000, S_cap + 1, (n,),
                            generator=torch.Generator().manual_seed(7))]),
    ("short 3K-6K", [int(x) for x in torch.randint(3000, 6000, (n,),
                         generator=torch.Generator().manual_seed(9))]),
]:
    out_on = run(idx_on, S_list)
    out_off = run(idx_off, S_list)
    torch.cuda.synchronize()
    same_shape = out_on.shape == out_off.shape
    md, nmis = set_diff(out_on, out_off, S_cap)
    exact = torch.equal(out_on, out_off)
    status = "PASS" if (same_shape and md <= 2) else "FAIL"
    if status == "FAIL":
        ok = False
    print(f"[{tag}] shape同={same_shape} exact={exact} 有效集对称差max={md} "
          f"(不一致行 {nmis}/{n*Hkv}) → {status}")

t_on = bench(lambda: run(idx_on, [S_cap] * n))
t_off = bench(lambda: run(idx_off, [S_cap] * n))
print(f"\nselect_decode_batched @bs32/131K: off={t_off:.3f}ms → on={t_on:.3f}ms "
      f"({t_off/t_on:.2f}×)")

print("\nALL PASS" if ok else "\nFAIL")
sys.exit(0 if ok else 1)
