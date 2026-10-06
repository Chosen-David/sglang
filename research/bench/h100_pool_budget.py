# H100 主表 pool 容量预核算（#57 教训前置化：S=40K 踩坑 3 次才定位
# retraction 混沌；H100 主表是 headline 实验，上机前必须先算账）
#
# 账目模型（H20 实测标定，H100 同架构显存 141GB 同底座）：
# - KV/token = L × Hkv × D × 2(KV) × 2B(bf16) = 147456 B/token（Qwen3-8B）
# - tli 额外静态：per-layer 索引池 kq/kmin/kmax 预分配 ~15.8GB（H20 实测
#   capture avail 差：triton 40.96GB vs tli 25.14GB @ mem0.7）
# - 权重+激活+graph 基线 ~28GB（mem0.7 capture 前 avail 40.96GB 的反推）
# - pool 上限 ≈ (总显存 × mem_frac − 权重/激活/graph − tli 索引池) / KV_per_token
#
# 用法：python3 h100_pool_budget.py [--gpu-mem 141] [--model qwen3-8b|qwen3-32b]
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-mem", type=float, default=141.0, help="GPU 显存 GB")
    ap.add_argument("--model", default="qwen3-8b",
                    choices=["qwen3-8b", "qwen3-32b"])
    ap.add_argument("--backend", default="both", choices=["triton", "tli", "both"])
    args = ap.parse_args()

    # 模型 KV 几何（L, Hkv, D）与权重大小
    if args.model == "qwen3-8b":
        L, Hkv, D, W = 36, 8, 128, 16.0
    else:
        L, Hkv, D, W = 64, 8, 128, 64.0
    kv_per_tok = L * Hkv * D * 2 * 2  # B/token

    # tli 索引池静态开销：H20 实测 ~15.8GB（8B）。粗略 ∝ L（per-layer pool）。
    # 8B: 15.8/36 = 0.439 GB/层；32B: 0.439×64 = 28.1GB（须实测修正）
    tli_idx_gb = 0.439 * L

    # 激活/graph 基线（graph capture 后）：H20 8B 实测 ~15GB（avail 25.14@0.7 反推：
    # 141×0.7=98.7 − pool − 权重16 − 索引15.8 ≈ pool 48.6 → capture avail 25.14
    # 与 prefill transient 预留有关；保守取 12GB 作激活+graph+reserved）
    act_graph_gb = 12.0

    print(f"== {args.model} @ {args.gpu_mem}GB H100 主表预算 ==")
    print(f"KV/token = {kv_per_tok} B；权重 {W}GB；tli 索引池 ~{tli_idx_gb:.1f}GB\n")

    scenarios = [
        ("bs=16 × S=131K", 16, 131000),
        ("bs=32 × S=131K", 32, 131000),
        ("bs=16 × S=64K", 16, 64000),
        ("bs=32 × S=64K", 32, 64000),
    ]
    for name, bs, S in scenarios:
        need_tok = bs * (S + 256)
        need_gb = need_tok * kv_per_tok / 1e9
        print(f"{name}: KV 需求 {need_tok/1000:.0f}K token = {need_gb:.1f}GB")
        for be in (("triton", "tli") if args.backend == "both" else (args.backend,)):
            extra = tli_idx_gb if be == "tli" else 0.0
            for mf in (0.85, 0.90, 0.92):
                pool_gb = args.gpu_mem * mf - W - act_graph_gb - extra
                pool_tok = pool_gb * 1e9 / kv_per_tok
                margin = (pool_tok - need_tok) / need_tok * 100
                flag = "OK" if margin > 5 else ("TIGHT" if margin > 0 else "OVERFLOW")
                print(f"  [{be:6s} mf={mf:.2f}] pool {pool_tok/1000:.0f}K tok"
                      f" → 余量 {margin:+.1f}%  {flag}")
        print()


if __name__ == "__main__":
    main()
