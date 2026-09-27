# DeepSelect（DeepSeek 官方 topk kernel）vs torch.topk 对拍+计时
# 场景对齐 TLI select_batched 生产形态：
#   L1 块级：[n*Hkv, NBLK] topk=K1=128（131K/64=2048 块）
#   L2 token 级：[n*Hkv, S] topk=256（far_tokens）/1024（K2 decode）
# 兼测 per-row end（对应 per-request 上下文长度，select_batched 原生需求）
# 用法：CUDA_VISIBLE_DEVICES=x PYTHONPATH=/home/wangyuanshuo02/.local/pylibs python3 test_deepselect_bench.py
import sys

import torch

sys.path.insert(0, "/home/wangyuanshuo02/.local/pylibs")


def bench(fn, iters=50, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    import deep_select
    print("[info] stride req:", deep_select.get_stride_requirement())
    dev = "cuda:0"
    torch.manual_seed(0)
    S = 131072
    NBLK = S // 64
    cases = [
        # (名称, 行数, 列数, topk)
        ("L1_blk  b32x8head NBLK2048 k128", 32 * 8, NBLK, 128),
        ("L2_far  b32x8head S131K   k256", 32 * 8, S, 256),
        ("L2_K2   b32x8head S131K   k1024", 32 * 8, S, 1024),
        ("L2_far  b256x8head S131K  k256", 256 * 8, S, 256),
    ]
    for name, rows, cols, k in cases:
        x = (torch.randn(rows, cols, device=dev) * 3).to(torch.bfloat16)
        # torch.topk 基线（同 bf16 输入，生产口径 torch.topk 在 fp32 上）
        t_torch = bench(lambda: torch.topk(x, k, dim=-1))
        t_ds = bench(lambda: deep_select.topk(x, k, return_value=False, indices_type=torch.int32))
        # 正确性：DS 索引集合 vs torch 索引集合（bf16 tie 区允许少量差异）
        _, idx_t = torch.topk(x.float(), k, dim=-1)
        _, idx_d = deep_select.topk(x, k, sorted_index=False, indices_type=torch.int64)
        jac = []
        for i in range(min(rows, 64)):
            jac.append(len(set(idx_t[i].tolist()) & set(idx_d[i].tolist())) / k)
        bytes_ = rows * cols * 2
        print(f"[{name}] torch={t_torch*1000:.0f}us DS={t_ds*1000:.0f}us "
              f"加速={t_torch/t_ds:.2f}x  DS带宽={bytes_/(t_ds/1e3)/1e9:.0f}GB/s  "
              f"jaccard={sum(jac)/len(jac):.4f}")
    # per-row end（对应 select_batched 的 per-request nblk_t）
    rows, cols, k = 32 * 8, S, 256
    x = (torch.randn(rows, cols, device=dev) * 3).to(torch.bfloat16)
    end = torch.randint(cols - 4096, cols + 1, (rows,), device=dev, dtype=torch.int32)
    _, idx = deep_select.topk(x, k, end=end, return_value=False, indices_type=torch.int64)
    ok = True
    for i in range(min(rows, 16)):
        m = idx[i] < end[i]
        if not bool(m.all()):
            ok = False
    print(f"[per-row end] 越界检查 {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
