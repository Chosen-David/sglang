# E8-1: D'+B' 组合收益实测（索引计算量维度）
# 对比 TIA vs TLI(A+B'+D') 在 decode 单步的索引开销模型：
#   L1: 块 min/max 上界计算（块数 × d' 维 einsum）
#   L2: 4bit 部分维 token 精筛（进入 L2 的 token 数 × d' 维 einsum）
#   D' 收益: 跳层（13/36）的 far 区从 L1 剔除 → L2 候选从 top-128 块(8192 tok) 降到 sink+near(~2240 tok)
#   A 收益: L1 的 min/max 维度从 128 → 32（4×）
#   B' 收益: far/near 分区（质量保证，不省计算）
# 用真实 trace 的层掩码和序列长度计算逐层 FLOP，输出加速比
import glob
import json
import os

import torch

TRACE = "/tmp/trace/qwen3-8b"
OUT = "/home/wangyuanshuo02/two-level-attention/exp/trace/results"
MASK = json.load(open(f"{OUT}/tli_layer_skip_mask.json"))
SKIP = set(MASK["skip"])
D = 128
D_SUB = 32          # A：子空间维数（cmp_ratio=4）
BLK = 64
K1 = 128            # TIA L1 topk 块数
SINK_BLK = 2
NEAR_TOK = 2048


def main():
    stats = []
    for pdir in sorted(glob.glob(f"{TRACE}/*")):
        name = os.path.basename(pdir)
        if not name.startswith(("needle", "natural", "lb_")):
            continue
        for lf in sorted(glob.glob(f"{pdir}/layer*.pt")):
            d = torch.load(lf, map_location="cpu")
            S = d["S"]
            layer_id = int(lf.split("layer")[1].split(".")[0])
            nblk = (S + BLK - 1) // BLK
            skip = layer_id in SKIP
            # ---- TIA 单步索引 FLOP（每 Hkv head）----
            # L1: 块分数 = q·(k_min,k_max) 区间算术 → 2 × nblk × 128（全维）
            # L2: 进入 L2 的 token = K1 块 × 64 + 滑窗，4bit 分数 q·k_qat → tok × 32（部分维）
            tia_l1 = 2 * nblk * D
            tia_l2_tok = K1 * BLK + 128  # top-128 块 + 滑窗冗余
            tia_l2 = tia_l2_tok * D_SUB
            tia = tia_l1 + tia_l2
            # ---- TLI 单步索引 FLOP ----
            # A: L1 子空间 d'=32 → 2 × nblk × 32；若跳层，far 块剔除 → nblk_eff
            nblk_eff = SINK_BLK + (NEAR_TOK // BLK) + 1 if skip else nblk
            tli_l1 = 2 * nblk_eff * D_SUB
            # L2 候选：跳层 → L1 topk 只有 sink+near（~2240 tok）；
            #          非跳层 → 与 TIA 同（K1 块）+ B' 分区不省计算
            tli_l2_tok = min(K1, nblk_eff) * BLK + 128 if not skip else (SINK_BLK + NEAR_TOK // BLK) * BLK + 128
            tli_l2 = tli_l2_tok * D_SUB
            tli = tli_l1 + tli_l2
            stats.append({
                "name": name, "layer": layer_id, "S": S, "skip": skip,
                "tia_flop": tia, "tli_flop": tli,
                "tia_l2_tok": tia_l2_tok, "tli_l2_tok": tli_l2_tok,
                "speedup": tia / tli,
            })
            del d
    # 汇总
    import statistics as st
    all_sp = [s["speedup"] for s in stats]
    skip_sp = [s["speedup"] for s in stats if s["skip"]]
    noskip_sp = [s["speedup"] for s in stats if not s["skip"]]
    print(f"样本: {len(stats)} 层-trace（跳层 {len(skip_sp)}, 非跳层 {len(noskip_sp)}）")
    print(f"TIA 平均 L2 候选: {st.mean([s['tia_l2_tok'] for s in stats]):.0f} tok")
    print(f"TLI 平均 L2 候选: {st.mean([s['tli_l2_tok'] for s in stats]):.0f} tok")
    print(f"索引 FLOP 加速比: 全层平均 {st.mean(all_sp):.2f}× | 跳层 {st.mean(skip_sp):.2f}× | 非跳层 {st.mean(noskip_sp):.2f}×")
    json.dump(stats, open(f"{OUT}/e8_1_index_flop.json", "w"), indent=1)
    print(f"saved -> {OUT}/e8_1_index_flop.json")


if __name__ == "__main__":
    main()
