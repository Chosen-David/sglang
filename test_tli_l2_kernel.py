# M3-b: L2 级联 fused kernel 对拍（select kernel 路径 vs eager 路径）
# 合成数据 kernel 级 microbench（口径与 E8-2 原型一致；真实模型验证在 e2e）
# 运行：CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/home/wangyuanshuo02/sglang/python \
#   python test_tli_l2_kernel.py
import sys

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
import torch

from sglang.srt.layers.attention.tli.config import TLIProfile
from sglang.srt.layers.attention.tli.indexer import TLIIndexer


def main():
    torch.manual_seed(0)
    dev = "cuda:0"
    Hkv, G, D = 8, 4, 128
    H = Hkv * G
    p = TLIProfile()
    p.dense_threshold = 2048
    idx = TLIIndexer(p, head_dim=D).to(dev)

    ok_all = True
    for S in (9891, 131072):
        k = torch.randn(S, Hkv, D, device=dev) * 0.3
        index = idx.build_block_index(k)
        for t in ([S - 1, S - 200, 6000] if S < 20000 else [S - 1, S - 500, 60000]):
            q = torch.randn(1, H, D, device=dev) * 0.3
            sel_e = idx.select(index, q, t)  # eager L1 + eager L2
            # 隔离 L2：同 eager L1 候选池，仅 L2 走 kernel
            sel_k = idx.select(index, q, t, use_l2_kernel=True)
            S_loc = index["S"]
            jacs, sizes = [], []
            for h in range(Hkv):
                e = set(sel_e[h].tolist())
                kk = set(x for x in sel_k[h].tolist() if x < S_loc)
                jacs.append(len(e & kk) / max(1, len(e | kk)))
                sizes.append((len(e), len(kk)))
            j = sum(jacs) / len(jacs)
            status = "OK" if j > 0.98 else "FAIL"
            if j <= 0.98:
                ok_all = False
            print(
                f"S={S:>6} t={t:>6}: mean Jaccard(L2 eager, L2 kernel) = {j:.4f} "
                f"[{status}] sizes(head0)={sizes[0]}"
            )

        # 延迟对比（同候选池，仅 L2 阶段差异）
        import time

        q = torch.randn(1, H, D, device=dev) * 0.3
        t = S - 1

        def bench(fn, n=50):
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n):
                fn()
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / n * 1e3

        te = bench(lambda: idx.select(index, q, t))
        tk = bench(lambda: idx.select(index, q, t, use_l2_kernel=True))
        print(f"S={S:>6} select 延迟: eager={te:.3f} ms, +L2 fused={tk:.3f} ms, "
              f"L2 增量 {te - tk:+.3f} ms\n")
    print("PASS" if ok_all else "FAIL")


if __name__ == "__main__":
    main()
