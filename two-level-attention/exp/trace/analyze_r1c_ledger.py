# -*- coding: utf-8 -*-
# R1c-EXP1b：canonical 配置的真实 token/bytes 账本。
# 口径（最重要）：本账本是 **bytes 读写口径**（算法必需的逻辑读/存量 bytes，
#   per request per layer，kv-head 级 = 8 heads）；论文 §4 的 1.46×@32K→5.09×@128K
#   是 **kernel 时间口径**（选择链 vs dense 打分，trace 重放，生产 tail32-4bit L1 系统）
#   ——两者分开标注，不得混用。
# 逻辑口径 vs 原型实现差异（诚实标注）：原型 eager 路径物化全宽 k_qat
#   （zeros_like 全 128 维）并对全序列算细筛分——账本按「indexed 系统必需」
#   计算（L2 只读 L1 选中块的 kq 条目）；该差异即 E107a 归因的 prefill/原型残留。
# 配置源：exp/trace/results/r1c_canonical_config.json（EXP1a 锁定，唯一配置源）。
import json

REPO = "/home/wangyuanshuo02/sglang/two-level-attention"
CFG = json.load(open(f"{REPO}/exp/trace/results/r1c_canonical_config.json"))
OUT = f"{REPO}/exp/trace/results/r1c_token_bytes_ledger.json"

# ---- 从 canonical JSON 取参数（不重复硬编码） ----
M = CFG["model"]
H_KV, D, N_LAYER = M["n_kv_heads"], M["head_dim"], M["n_layers"]
ELEM = 2  # bf16/fp16 均 2 B
KV_PER_TOK_PER_HEAD = D * ELEM            # 256 B（K 或 V 单侧）
KV_PER_TOK_LAYER = 2 * H_KV * KV_PER_TOK_PER_HEAD   # 4096 B（K+V，8 heads）
DENSE_K_PER_TOK_LAYER = H_KV * KV_PER_TOK_PER_HEAD  # 2048 B

BS = CFG["block_layout"]["block_size_B"]["value"]
SINK = CFG["block_layout"]["sink"]["value_tokens"]
SWA = CFG["block_layout"]["swa"]["value_tokens"]
K1 = CFG["budgets"]["K1_L1_blocks"]["value"]
K2 = CFG["budgets"]["K2_B_TOK_tokens"]["value"]
BETA = CFG["budgets"]["beta"]["value"]
GAMMA = CFG["budgets"]["gamma"]["value"]

# ---- 存储布局（canonical 主表臂 / 生产 M6 / 论文口径 / 理想 packed，全部显式） ----
L1_CANON_B_PER_BLOCK_HEAD = 2 * D * ELEM          # min+max 全 128 维 bf16 = 512 B
L2_PROD_B_PER_TOK_HEAD = 32 * 1 + 2 * 4           # uint8 grid[32] + fp32 sc + fp32 mn = 40 B（M6）
L2_PACKED_B_PER_TOK_HEAD = 32 // 2 + 2 * 4        # 真 nibble-packed 16B + fp32 双 scale = 24 B
L1_PROD_B_PER_BLOCK_HEAD = 2 * 32 * 4             # 生产 tail32-L1：kmin/kmax fp32 [nblk,Hkv,32] = 256 B
L1_PAPER_B_PER_BLOCK_HEAD = 2 * 32 * 2            # 论文 336B 推导隐含的 fp16 bounds = 128 B
L1_4BIT_B_PER_BLOCK_HEAD = 2 * 32 // 2            # 论文 §2.2 蓝图 4bit L1（packed min/max）= 32 B


def row_from_cfg(S):
    """从 canonical JSON 的 per_context_table 取该 S 的推导行（已过原型代码对拍）。"""
    for r in CFG["budget_derivation"]["per_context_table"]:
        if r["S"] == S:
            return r
    raise KeyError(S)


def ledger_row(S):
    r = row_from_cfg(S)
    kt = r["n_blocks_padded_kt"]
    n_cand_tok = r["L1_candidate_tokens=(i_f+i_n)*64"]   # L2 只读 L1 选中块
    total_sel = r["total_selected_tokens"]

    # ---- 存量（per request per layer） ----
    l1_store = kt * H_KV * L1_CANON_B_PER_BLOCK_HEAD
    l2_store = S * H_KV * L2_PROD_B_PER_TOK_HEAD
    idx_store = l1_store + l2_store
    dense_kv_store = S * KV_PER_TOK_LAYER

    # ---- 每 decode 步读（per request per layer，逻辑必需口径） ----
    l1_read = kt * H_KV * L1_CANON_B_PER_BLOCK_HEAD       # L1 全扫（含 sink/swa 4 块，可省 <1%，见注）
    l2_read = n_cand_tok * H_KV * L2_PROD_B_PER_TOK_HEAD  # L2 只读被选中块的 kq
    kv_sel_read = total_sel * KV_PER_TOK_LAYER            # 选中 KV（K+V）读取
    dense_read = S * KV_PER_TOK_LAYER

    return {
        "S": S,
        "n_blocks": kt,
        "L1_candidate_tokens": n_cand_tok,
        "total_selected_tokens": total_sel,
        "storage": {
            "L1_index_B": l1_store,
            "L2_index_B": l2_store,
            "index_total_B": idx_store,
            "dense_KV_B": dense_kv_store,
            "index_vs_dense_KV": round(idx_store / dense_kv_store, 4),
            "index_vs_dense_K_only": round(idx_store / (S * DENSE_K_PER_TOK_LAYER), 4),
            "index_B_per_token_per_layer": round(idx_store / S, 2),
            "derivation": (
                f"L1 = {kt}块×8head×512B(min+max全128维bf16) = {l1_store}B；"
                f"L2 = {S}tok×8head×40B(tail32 uint8 grid+2×fp32 scale) = {l2_store}B；"
                f"dense KV = {S}×(8head×(128K+128V)×2B) = {dense_kv_store}B"),
        },
        "per_decode_step_read": {
            "L1_full_scan_B": l1_read,
            "L2_selected_blocks_only_B": l2_read,
            "selected_KV_B": kv_sel_read,
            "PSI_total_read_B": l1_read + l2_read + kv_sel_read,
            "dense_full_KV_read_B": dense_read,
            "index_overhead_vs_dense": round((l1_read + l2_read) / dense_read, 4),
            "total_read_vs_dense": round((l1_read + l2_read + kv_sel_read) / dense_read, 4),
            "KV_read_reduction_factor": round(dense_read / (l1_read + l2_read + kv_sel_read), 2),
            "derivation": (
                f"L1全扫 = {kt}×8×512B；L2 = {n_cand_tok}cand×8×40B"
                f"（i_f+i_n = {r['L1_far_blocks_selected_i_f']}+{r['L1_near_blocks_selected_i_n']} 块）；"
                f"选中KV = {total_sel}tok×4096B；dense = {S}×4096B"),
        },
    }


def main():
    contexts = [r["S"] for r in CFG["budget_derivation"]["per_context_table"]]
    curve = [ledger_row(S) for S in contexts]

    # 汇总口径（每请求全 36 层，131K 点）
    S131 = 131072
    r131 = ledger_row(S131)

    ledger = {
        "exp": "R1c-EXP1b",
        "date": "2026-10-07",
        "task": "#137 R1c：真实 token/bytes 账本（canonical 配置逐项推导，可复算）",
        "config_source": "exp/trace/results/r1c_canonical_config.json（EXP1a 锁定，预算表已过原型代码对拍）",
        "protocol_clarification": {
            "本账本": "bytes 读写口径：算法必需的逻辑读 bytes（per request per layer，kv-head 级 8 heads）；"
                      "L1 全扫、L2 只读 L1 选中块、选中 KV 全读",
            "论文§4 kernel 口径": "1.46×@32K / 2.77×@64K / 5.09×@128K 是选择链相对 dense 全维打分的 **时间** 加速"
                                  "（trace 重放，生产 tail32-4bit L1 系统）——与 bytes 口径分属两套度量，不可互相换算",
            "原型实现差异": "原型 eager 路径物化全宽 k_qat（zeros_like 全 128 维，256B/token-head）并对全序列算"
                            "细筛分；账本按 indexed 系统必需口径计（E107a 已归因为 correctness-first 原型残留，"
                            "kernel 化路径 M6/M8 即为本口径）",
        },
        "units": {
            "per": "per request per layer（除非显式标注 ×36）",
            "kv_heads": H_KV, "head_dim": D, "kv_elem_bytes": ELEM,
            "dense_KV_bytes_per_token_per_layer": KV_PER_TOK_LAYER,
            "dense_K_bytes_per_token_per_layer": DENSE_K_PER_TOK_LAYER,
            "n_layers": N_LAYER,
        },
        "storage_layouts": {
            "canonical_main_arm": {
                "L1": f"{L1_CANON_B_PER_BLOCK_HEAD} B/block/kv-head（min+max 全 128 维 bf16）"
                      f"→ 摊 {L1_CANON_B_PER_BLOCK_HEAD // BS} B/token/kv-head",
                "L2": f"{L2_PROD_B_PER_TOK_HEAD} B/token/kv-head（tail32：uint8 grid 32B + fp32 sc/mn 8B，"
                      f"生产 M6 布局，sglang tli/indexer.py quant4_pack L120-134）",
                "total_per_token_per_layer": f"{(L1_CANON_B_PER_BLOCK_HEAD // BS + L2_PROD_B_PER_TOK_HEAD) * H_KV} B/token"
                                             f"（=64B L1 摊 + 320B L2，8 heads 汇总）",
                "vs_dense_KV": f"{(L1_CANON_B_PER_BLOCK_HEAD // BS + L2_PROD_B_PER_TOK_HEAD) * H_KV / KV_PER_TOK_LAYER:.4f}"
                               f"（384/4096 = 9.4%）",
            },
            "production_tail32_L1": {
                "L1": f"{L1_PROD_B_PER_BLOCK_HEAD} B/block/kv-head（tail32 kmin/kmax fp32）"
                      f"→ 摊 4 B/token/kv-head → 32 B/token",
                "L2": "40 B/token/kv-head（同上）→ 320 B/token",
                "total_per_token_per_layer": "352 B/token",
                "note": "论文 §4「336B/token」= 320(kq) + 16(L1 bounds 按 fp16 假设)；"
                        "生产代码 kmin/kmax 实为 fp32 → 352B。336 与 352 差在 L1 bounds 精度假设，"
                        "round2 审稿 NM5 已flag 过 40B/token-head vs Quest 8B/token-head 的口径张力——"
                        "本账本数字为精确复算值",
            },
            "ideal_packed_variants": {
                "L2_true_nibble_packed": f"{L2_PACKED_B_PER_TOK_HEAD} B/token/kv-head（16B nibble + 8B fp32 scale）"
                                          f"→ 192 B/token；若 scale 用 fp16 则 20B → 160 B/token",
                "L1_paper_4bit_blueprint": f"{L1_4BIT_B_PER_BLOCK_HEAD} B/block/kv-head（tail32 4bit packed min/max）"
                                            f"→ 0.5 B/token/kv-head → 4 B/token（论文 §2.2 蓝图，未在生产实现）",
            },
            "prototype_materialization_artifact":
                "原型 k_qat = zeros_like(k) 全 128 维 bf16 = 256 B/token-kv-head 字面占用"
                "（非逻辑索引 bytes，仅尾 32 维非零）——引用账本数字时须排除该口径",
        },
        "per_decode_step_read_protocol": {
            "L1_full_scan": "读全部块的 min/max（含 sink/swa 共 4 块——严格逻辑可跳过，"
                            "32K 时占比 4/512 < 1%，账本按全扫保守计）",
            "L2_selected_blocks_only": "只读 L1 选中块（i_f+i_n 块）的 kq 4bit 条目；"
                                        "sink/swa 强制区不经 L2（直接置位）",
            "selected_KV": "最终选中的 K2=1024 token 的 K+V 全维读取",
            "excluded": "mask/topk 索引元数据 O(K1+K2) 整数读写、q 单 token 读——量级 <0.1%，不计",
        },
        "context_curve": curve,
        "summary": {
            "index_overhead_vs_dense": {
                "4096": curve[0]["per_decode_step_read"]["index_overhead_vs_dense"],
                "8192": curve[1]["per_decode_step_read"]["index_overhead_vs_dense"],
                "16384": curve[2]["per_decode_step_read"]["index_overhead_vs_dense"],
                "32768": curve[3]["per_decode_step_read"]["index_overhead_vs_dense"],
                "65536": curve[4]["per_decode_step_read"]["index_overhead_vs_dense"],
                "131072": curve[5]["per_decode_step_read"]["index_overhead_vs_dense"],
            },
            "at_32K": {
                "index_read_B": curve[3]["per_decode_step_read"]["L1_full_scan_B"]
                                + curve[3]["per_decode_step_read"]["L2_selected_blocks_only_B"],
                "dense_read_B": curve[3]["per_decode_step_read"]["dense_full_KV_read_B"],
                "PSI_total_read_B": curve[3]["per_decode_step_read"]["PSI_total_read_B"],
                "total_read_vs_dense": curve[3]["per_decode_step_read"]["total_read_vs_dense"],
                "KV_read_reduction_factor": curve[3]["per_decode_step_read"]["KV_read_reduction_factor"],
            },
            "at_128K": {
                "index_read_B": curve[5]["per_decode_step_read"]["L1_full_scan_B"]
                                + curve[5]["per_decode_step_read"]["L2_selected_blocks_only_B"],
                "dense_read_B": curve[5]["per_decode_step_read"]["dense_full_KV_read_B"],
                "PSI_total_read_B": curve[5]["per_decode_step_read"]["PSI_total_read_B"],
                "total_read_vs_dense": curve[5]["per_decode_step_read"]["total_read_vs_dense"],
                "KV_read_reduction_factor": curve[5]["per_decode_step_read"]["KV_read_reduction_factor"],
            },
            "per_request_36_layers_at_128K": {
                "index_storage_B": r131["storage"]["index_total_B"] * N_LAYER,
                "index_storage_GB": round(r131["storage"]["index_total_B"] * N_LAYER / 2**30, 3),
                "dense_KV_storage_B": r131["storage"]["dense_KV_B"] * N_LAYER,
                "dense_KV_storage_GB": round(r131["storage"]["dense_KV_B"] * N_LAYER / 2**30, 2),
                "index_vs_dense_KV": r131["storage"]["index_vs_dense_KV"],
            },
            "asymptote": "L1 线性于 S（64B/token）、L2 常数（8192 cand × 40B × 8 = 2.62MB/步）、"
                         "dense 线性（4096B/token）→ overhead 比例随 S 收敛到 64/4096 = 1.56%"
                         "（L2 项趋于 0），总量读比例收敛到 (64+0+4096·1024/S)/4096",
        },
        "budget_table_source": "r1c_canonical_config.json budget_derivation.per_context_table"
                               "（S=4K 时 near 区只有 6 块 → 实选 704 < K2=1024，如实入表）",
    }
    with open(OUT, "w") as f:
        json.dump(ledger, f, ensure_ascii=False, indent=1)
    print(f"written {OUT}")
    for c in curve:
        pr = c["per_decode_step_read"]
        print(f"S={c['S']:>7}: overhead={pr['index_overhead_vs_dense']:.4f} "
              f"total_read={pr['total_read_vs_dense']:.4f} "
              f"reduction={pr['KV_read_reduction_factor']}x "
              f"L1={pr['L1_full_scan_B']/2**20:.2f}MB L2={pr['L2_selected_blocks_only_B']/2**20:.2f}MB "
              f"KVsel={pr['selected_KV_B']/2**20:.2f}MB dense={pr['dense_full_KV_read_B']/2**20:.1f}MB")


if __name__ == "__main__":
    main()
