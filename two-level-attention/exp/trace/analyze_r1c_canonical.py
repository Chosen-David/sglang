# -*- coding: utf-8 -*-
# R1c-EXP1a：论文主配置（canonical）机器可读锁定。
# 目标：把 PSI/mavg 主表臂的全部参数锁成 r1c_canonical_config.json，
#   作为后续 EXP2/4/5 计时实验的唯一配置源。每个参数标注来源
#   （论文节号 / 实验代号 / 代码行）。
# 验证：用原型 TLIIndexer（sparse_attn，产出 50.78 主表的同一代码路径）
#   在 GPU1 合成数据上实跑 prepare_mask，TLI_DEBUG 捕获池参数与选中数，
#   与闭式公式逐项对拍（S=4K..128K 六档）。
# 口径注意：
#   - 主表臂（50.78）= 原型管线（two-level-attention/sparse_attn + HF transformers
#     monkeypatch）产出；sglang 生产路径（tli/）存储布局不同（tail32-4bit L1），
#     差异在 quantization/storage 口径节显式标注。
#   - 运行口径：E98BEST 13 任务全量脚本的实际 flags（/tmp/e98_full_13tasks.sh）。
import contextlib
import io
import json
import os
import sys

import torch

REPO = "/home/wangyuanshuo02/sglang/two-level-attention"
sys.path.insert(0, REPO)
OUT = f"{REPO}/exp/trace/results/r1c_canonical_config.json"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")  # 不占 GPU0
DEV = "cuda:0" if (torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES") != "") else "cpu"
# CUDA_VISIBLE_DEVICES 已在进程外设定时不覆盖

# ---------------- canonical 参数（与 E98BEST 运行 flags 逐项对应） ----------------
BS = 64            # tia_block_size
K1 = 128           # tia_level1_topk（L1 块预算）
K2 = 1024          # tia_level2_topk（B_TOK：最终选中 token 总预算）
CMP = 4            # tia_level2_cmp_ratio → delta = 64//4 = 16 → tail32
ALPHA = 0.125      # tli_alpha：near 区长 = int(α·L_mid)
BETA = 0.375       # tli_beta：near 块预算 = round(β·K1)
GAMMA = 0.625      # tli_gamma：near 细筛配额 = min(int(nb_near·B·γ), K2_mid)
SINK_BLOCKS = 2    # sink = 前 2 块 = 128 token（tli_indexer.py L62 硬编码）
SWA = 128          # sliding_window_size（tia_indexer.py L16 硬编码 = 128）
FAR_METHOD = "minmax"   # tli_far_method：far 池 L1 分数源 = 块 minmax 上界
NEAR_METHOD = "avg"     # tli_near_method：near 池 L1 分数源 = 块均值
SINK = SINK_BLOCKS * BS  # 128

# 模型常量（Qwen3-8B config.json）
N_LAYER, H_Q, H_KV, D = 36, 32, 8, 128
G = H_Q // H_KV  # GQA group = 4

CONTEXTS = [4096, 8192, 16384, 32768, 65536, 131072]


def budget_row(S):
    """闭式复算 compute_mask 的 E64 分区预算（与 tli_indexer.py L460-696 逐行对应）。

    代码语义要点：
      - kt·bs 用 pad 后长度（prepare_pad_mask 补到块边界）；
      - near_len_dyn = max(bs, int(α·mid_len))，int() 向零截断；
      - near_blks = max(sink_blocks, (kt·bs − near_len_dyn)//bs)；
      - swa_lo_blk = max(near_blks, kt − max(1, swa//bs))，swa 块完全排除出双池；
      - L1：nb_near = max(1, round(K1·β))，nb_far = K1 − nb_near；
        far 池块区间 [sink_blocks, near_blks)，near 池块区间 [near_blks, swa_lo_blk)；
      - L2：K2 = min(S, level2_topk)；K2_mid = max(0, K2 − sink − swa)；
        nt_near = min(int(nb_near·bs·γ), K2_mid)；far_budget = max(64, K2_mid − nt_near)；
        k2_far = min(far_budget, far 区 token 数, K2_mid)；k2_near = K2_mid − k2_far；
        near 实选 = min(k2_near, near 区有限 token = i_n 块 × bs)；
      - sink/swa 强制置位不占 topk 竞争（严格口径 2026-09-29）。
    """
    kt = (S + BS - 1) // BS
    S_pad = kt * BS
    mid_len = max(0, S_pad - SINK - SWA)
    near_len_dyn = max(BS, int(ALPHA * mid_len))
    near_blks = max(SINK_BLOCKS, (S_pad - near_len_dyn) // BS)
    swa_lo_blk = max(near_blks, kt - max(1, SWA // BS))
    far_blocks_avail = near_blks - SINK_BLOCKS
    near_blocks_avail = max(0, swa_lo_blk - near_blks)
    nb_near = max(1, int(round(K1 * BETA)))
    nb_far = max(1, K1 - nb_near)
    i_f = min(nb_far, far_blocks_avail)
    i_n = min(nb_near, near_blocks_avail)
    K2_eff = min(S, K2)
    K2_mid = max(0, K2_eff - SINK - SWA)
    nt_near = min(int(nb_near * BS * GAMMA), K2_mid)
    far_budget = max(64, K2_mid - nt_near)
    far_tokens_avail = far_blocks_avail * BS
    k2_far = min(far_budget, far_tokens_avail, K2_mid) if far_tokens_avail > 0 else 0
    k2_near = max(0, K2_mid - k2_far)
    near_finite = i_n * BS  # near 池有限 token = L1 near 块 × bs（-inf 不入选）
    near_sel = min(k2_near, near_finite)
    swa_tok_eff = min(SWA, S)
    total = SINK + swa_tok_eff + k2_far + near_sel
    return {
        "S": S, "n_blocks_padded_kt": kt, "S_padded": S_pad,
        "L_mid": mid_len, "ell_near=int(a*L_mid)": near_len_dyn,
        "near_blks": near_blks, "swa_lo_blk": swa_lo_blk,
        "far_blocks_avail": far_blocks_avail, "near_blocks_avail": near_blocks_avail,
        "nb_near=round(b*K1)": nb_near, "nb_far=K1-nb_near": nb_far,
        "L1_far_blocks_selected_i_f": i_f, "L1_near_blocks_selected_i_n": i_n,
        "L1_candidate_tokens=(i_f+i_n)*64": (i_f + i_n) * BS,
        "K2_eff=min(S,1024)": K2_eff, "K2_mid=K2-sink-swa": K2_mid,
        "nt_near=min(int(nb_near*64*g),K2_mid)": nt_near,
        "far_budget=max(64,K2_mid-nt_near)": far_budget,
        "k2_far": k2_far, "k2_near=K2_mid-k2_far": k2_near,
        "near_selected=min(k2_near,i_n*64)": near_sel,
        "sink_tokens": SINK, "swa_tokens": swa_tok_eff,
        "total_selected_tokens": total,
    }


def formula_check():
    """原型代码实测对拍：合成 k/q 跑 prepare_mask，解析 TLI_DEBUG 池参数。"""
    from types import SimpleNamespace
    from sparse_attn.indexer.tli_indexer import TLIIndexer

    args = SimpleNamespace(
        tia_block_size=BS, tia_level1_topk=K1, tia_level2_topk=K2,
        tia_level2_cmp_ratio=CMP, tia_enable_async_topk=False,
        tli_alpha=ALPHA, tli_beta=BETA, tli_gamma=GAMMA,
        tli_far_method=FAR_METHOD, tli_near_method=NEAR_METHOD,
        tli_enable_kmeans=False, tli_enable_layer_skip=False,
        # 未传 flags → 默认：tli_subspace="full"（L1 全维）、tli_far_select="4bit"
    )
    torch.manual_seed(0)
    results = {}
    for S in CONTEXTS:
        indexer = TLIIndexer(args)
        indexer.layer_idx = 1  # TLI_DEBUG 打印条件
        k = torch.randn(1, S, H_KV, D, device=DEV, dtype=torch.float32)
        q = torch.randn(1, 1, H_Q, D, device=DEV, dtype=torch.float32)
        cu = torch.tensor([0, S], dtype=torch.long)
        q_ids = torch.tensor([S - 1])
        os.environ["TLI_DEBUG"] = "1"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            mask, blk = indexer.prepare_mask(q, q_ids, k, cu)
        os.environ.pop("TLI_DEBUG", None)
        # 解析 [L1dbg]/[L2dbg] 行的 key=value 对（pshape=[...] 含空格，须过滤）
        def _parse(ln):
            return dict(kv.split("=", 1) for kv in ln.split("]")[1].split()
                        if "=" in kv)
        l1 = l2 = None
        for ln in buf.getvalue().splitlines():
            if ln.startswith("[L1dbg]"):
                l1 = _parse(ln)
            if ln.startswith("[L2dbg]"):
                l2 = _parse(ln)
        m = mask[0, 0]  # [Hkv, T] kv-head 级
        per_head = m.sum(-1)
        sink_sel = int(m[..., :SINK].sum(-1).max())
        swa_sel = int(m[..., max(0, S - SWA):].sum(-1).max())
        far_hi_tok = int(l1["near_blks"]) * BS
        far_sel = int(m[..., SINK:far_hi_tok].sum(-1).max())
        near_sel_code = int(per_head.max()) - sink_sel - swa_sel - far_sel
        row = budget_row(S)
        checks = {
            "kt": int(l1["kt"]) == row["n_blocks_padded_kt"],
            "near_blks": int(l1["near_blks"]) == row["near_blks"],
            "swa_lo_blk": int(l1["swa_lo_blk"]) == row["swa_lo_blk"],
            "nb_far": int(l1["nb_far"]) == row["nb_far=K1-nb_near"],
            "nb_near": int(l1["nb_near"]) == row["nb_near=round(b*K1)"],
            "K2": int(l2["K2"]) == row["K2_eff=min(S,1024)"],
            "K2_mid": int(l2["K2_mid"]) == row["K2_mid=K2-sink-swa"],
            "nt_near": int(l2["nt_near"]) == row["nt_near=min(int(nb_near*64*g),K2_mid)"],
            "far_budget": int(l2["far_budget"]) == row["far_budget=max(64,K2_mid-nt_near)"],
            "k2_far": int(l2["k2_far"]) == row["k2_far"],
            "total_selected": int(per_head.max()) == row["total_selected_tokens"],
            "far_sel_tokens": far_sel == row["k2_far"],
            "near_sel_tokens": near_sel_code == row["near_selected=min(k2_near,i_n*64)"],
        }
        results[S] = {
            "code_L1dbg": l1, "code_L2dbg": {k: l2[k] for k in
                ("K2", "K2_mid", "sink_tok", "swa_tok", "nt_near", "far_budget",
                 "k2_far")},
            "code_selected_per_kvhead_max": int(per_head.max()),
            "code_selected_per_kvhead_min": int(per_head.min()),
            "formula_total": row["total_selected_tokens"],
            "checks_all_pass": all(checks.values()),
            "checks": checks,
        }
        del k, q, mask
        if DEV != "cpu":
            torch.cuda.empty_cache()
    return results


def main():
    verif = {}
    try:
        verif = formula_check()
    except Exception as e:  # 验证失败不静默——写入 JSON 供人查
        verif = {"error": repr(e)}

    # 论文 §2.2 公式 vs 代码语义的一致性（canonical γ=0.625 下两者同值）
    gamma_note = (
        "论文 §2.2 的 F = max(64, K2_mid − ⌊γ·⌈ℓ_near/B⌉B⌋) 与代码 "
        "nt_near = min(⌊nb_near·B·γ⌋, K2_mid) 在低 γ 臂数值不同（γ=0.125, S=32K："
        "论文 F=256 vs 代码 far_budget=384），但 canonical γ=0.625 下两者同坍缩到 "
        "far 保底 64——γ 截断坍缩（E98 网格教训）：nb_near·B·γ = 48·64·0.625 = 1920 "
        "> K2_mid=768 恒成立，near 池吃满全部 mid 预算，far 恒为 64。"
    )

    cfg = {
        "exp": "R1c-EXP1a",
        "date": "2026-10-07",
        "task": "#137 R1c：EXP1 canonical 配置锁定（后续 EXP2/4/5 计时实验的唯一配置源）",
        "canonical_arm": {
            "name": "PSI/mavg（E98 双口径网格 e2e 选举臂）",
            "run_flags": ("--method tli --tia_level1_topk 128 --tia_level2_topk 1024 "
                          "--tia_level2_cmp_ratio 4 --tli_enable_kmeans false "
                          "--tli_enable_layer_skip false --tli_alpha 0.125 "
                          "--tli_beta 0.375 --tli_gamma 0.625 "
                          "--tli_far_method minmax --tli_near_method avg"),
            "e2e_LongBench_13task_AVG": 50.78,
            "e2e_source": "e105_three_arm_verdict.json / e135_audit_recompute.json（E102/E98 全量）",
            "FullKV_ref": 50.36,
            "vs_FullKV": "+0.42（95% CI [-0.18,+1.02] 含 0，持平口径）",
            "method_semantics": ("mavg = (far=minmax 块上界, near=avg 块均值)；"
                                 "cavg/mminmax/aavg 同预算复测 50.35/50.29/50.29，"
                                 "mavg 全局最优（E105 终判）"),
        },
        "model": {
            "name": "Qwen3-8B",
            "path": "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B",
            "n_layers": N_LAYER, "n_q_heads": H_Q, "n_kv_heads": H_KV,
            "head_dim": D, "gqa_group_size": G,
            "kv_dtype": "bf16（pred.py L289 torch_dtype=bfloat16；与 fp16 同为 2 B/elem，"
                        "本账本 bytes 口径两者等价）",
            "kv_bytes_per_token_per_layer": H_KV * D * 2 * 2,  # K+V 各 2048B
            "rope_theta": 1000000.0,
            "source": "模型 config.json（本脚本已实读核实）",
        },
        "block_layout": {
            "block_size_B": {"value": BS, "source": "--tia_block_size 默认 64（arguments.py L5）"},
            "sink": {"value_tokens": SINK, "value_blocks": SINK_BLOCKS,
                     "source": "tli_indexer.py L62 硬编码 sink_blocks=2（前 2 块）"},
            "swa": {"value_tokens": SWA,
                    "source": "tia_indexer.py L16 硬编码 sliding_window_size=128"},
            "region_definition": (
                "区域权威定义（2026-09-29 用户澄清，tli_indexer.py L463-479）："
                "sink/swa 为正交强制区（不进创新管线、不占配额，最终 mask 直接置位）；"
                "mid = 除 sink/swa；near = mid 靠 q 的 α 比例（int 截断）；far = mid 其余。"
                "注意 R1b trace 重放用的 SWA=1024 是 far 区分析的窗口口径"
                "（r1b_cascade_decomposition.json config），非部署值；部署 swa=128。"),
        },
        "budgets": {
            "K1_L1_blocks": {"value": K1, "source": "--tia_level1_topk 128（E98BEST 运行 flags）"},
            "K2_B_TOK_tokens": {"value": K2, "source": "--tia_level2_topk 1024（E4c 严格预算：最终 mask 选中 token 总数上限）"},
            "cmp_ratio": {"value": CMP, "source": "--tia_level2_cmp_ratio 4 → delta=64//4=16 → tail32"},
            "alpha": {"value": ALPHA, "source": "E98 e2e 网格选举（e98_best_election.json）；near 区长 = int(α·L_mid)"},
            "beta": {"value": BETA, "source": "E98 e2e 网格选举；near L1 块预算 = round(β·K1) = 48 块"},
            "gamma": {"value": GAMMA, "source": "E98 e2e 网格选举；near 细筛配额折扣（γ 截断坍缩见 budget_notes）"},
            "far_L1_score": {"value": FAR_METHOD, "source": "--tli_far_method minmax：far 池 L1 分数 = 块逐维 min/max 上界的乐观侧求和（q_d 符号取端值）"},
            "near_L1_score": {"value": NEAR_METHOD, "source": "--tli_near_method avg：near 池 L1 分数 = 块均值点积（全 128 维块均值 k_avg，tli_indexer.py L219）"},
            "gqa_aggregation": {"value": "kv-head 级共享（组内 mean 聚合）",
                                "source": "E103 默认口径：32 q-head → 8 kv-head 组内 mean，索引量省 4×（per_q_head=False 默认）"},
        },
        "subspace": {
            "L1": {"value": "full 128 维（无子空间、无量化）",
                   "source": "主表臂未传 --tli_subspace → 默认 full（tli_indexer.py L39-44）；"
                             "k_min/k_max 在 pad 后全 128 维上逐块 amin/amax（L213-217）。"
                             "论文 §4.2：tail32-L1 同预算消融臂 50.53，CI 含 0"},
            "L2": {"value": "tail32 = 维 [48..63] ∪ [112..127]",
                   "source": "cmp_ratio=4 → delta=16 → 硬编码尾部（tia_indexer.py L48-52）；"
                             "r1b_pair_map.json：= 旋转对 pair_id 48..63（16 个最低频完整 RoPE 旋转对，"
                             "pair j=(j,j+64)，inv_freq 单调递减于 j）"},
            "note": "L1 全维 / L2 tail32 的「双口径」= 论文 §4.2 明示的主表口径（审稿 C1 修复后的如实标注）",
        },
        "quantization": {
            "L1": {"value": "无量化（主表臂）",
                   "detail": "k_min/k_max 以 KV dtype（bf16，2 B/elem）存储：2×128×2 = 512 B/block/kv-head。"
                             "生产口径（sglang tli/，tail32-4bit L1）：kmin/kmax fp32 [nblk,Hkv,32] = 256 B/block/kv-head——"
                             "两口径均为「主表 L1 全维无量化 / 生产 L1 tail32 无量化 fp32」，论文 §2.2 的「L1 4bit」仅指 M6 蓝图的 packed 形态"},
            "L2": {"value": "per-token min-max 16 级 4bit 量化（tail32 上）",
                   "detail": "原型 min_max_per_token_quant（tia_indexer.py L21-30）：scale=(max−min)/15，zero=−min，"
                             "clamp(round((x+zero)/scale),0,15)·scale−zero；"
                             "生产 M6 布局（sglang tli/indexer.py quant4_pack L120-134）：uint8 grid[32] + fp32 sc + fp32 mn "
                             "= 40 B/token-kv-head（与论文 §2.2 40B/token-head 一致）"},
        },
        "d_prime_disabled_paths": {
            "kmeans": "off（--tli_enable_kmeans false；far_select=4bit 默认，聚类已降级消融 E4c/E108）",
            "layer_skip_D_prime": "off（--tli_enable_layer_skip false；主表臂不用跳层；D' gate 机制在但未启用）",
            "top_sigma": "off（默认 tli_sigma_select=none）",
            "static_pair": "off（默认 tli_static_pair=False，E85f e2e 判决 −0.50）",
            "per_q_head": "off（E103 默认共享口径）",
            "async_topk": "off（tia_enable_async_topk=False；不影响选择语义）",
        },
        "budget_derivation": {
            "formulas": {
                "n_blocks_kt": "kt = ceil(S/64)（pad 到块边界，L1 用 pad 后长度 kt·64）",
                "L_mid": "L_mid = kt·64 − 128(sink) − 128(swa)",
                "ell_near": "ℓ_near = max(64, int(0.125·L_mid))（int 向零截断）",
                "near_blks": "near_blks = max(2, (kt·64 − ℓ_near)//64)",
                "swa_lo_blk": "swa_lo_blk = max(near_blks, kt − 2)（swa=128 → 2 块，完全排除出双池）",
                "L1_pools": "nb_near = round(128·0.375) = 48；nb_far = 128 − 48 = 80；"
                            "far 池块区间 [2, near_blks)，near 池块区间 [near_blks, swa_lo_blk)；"
                            "实选 i_f = min(80, far 区块数)，i_n = min(48, near 区块数)",
                "K2_mid": "K2 = min(S, 1024)；K2_mid = max(0, K2 − 128 − 128) = 768（S≥1024）",
                "nt_near": "nt_near = min(int(48·64·0.625), 768) = min(1920, 768) = 768",
                "far_budget": "far_budget = max(64, 768 − 768) = 64（far 恒为保底 64——γ 截断坍缩）",
                "k2_pools": "k2_far = 64；k2_near = 768 − 64 = 704；near 实选 = min(704, i_n·64)",
                "total_selected": "total = sink 128 + swa 128 + far 64 + near（=1024 封顶，短上下文时小于 1024）",
            },
            "per_context_table": [budget_row(S) for S in CONTEXTS],
            "budget_notes": [
                gamma_note,
                "γ=0.625 的真实语义：near 池吃满 K2_mid=768，far 恒保底 64 token——"
                "E98 e2e 选举臂是 near 主导臂（mass 网格上该臂并非 mass 最优，"
                "mass→e2e 排序反转的实证，fig11）",
                "K2=1024 含 sink+swa 各 128（严格口径：固定区不进 topk 竞争但计入总量）",
            ],
        },
        "verification_vs_prototype_code": {
            "protocol": ("GPU1 合成数据（seed=0 randn k/q）实跑 sparse_attn TLIIndexer.prepare_mask，"
                         "TLI_DEBUG 捕获 [L1dbg]/[L2dbg] 池参数与最终 mask 选中数，"
                         "与闭式公式逐项对拍；device=" + DEV),
            "results": verif,
        },
        "implementation_identity": {
            "main_table_pipeline": "two-level-attention/sparse_attn（HF transformers monkeypatch，"
                                   "benchmark/LongBench/pred.py；prefill dense + decode 两级选择）",
            "production_pipeline": "sglang python/sglang/srt/layers/attention/tli/（paged 寻址、"
                                   "增量索引、quant4_pack 4bit 存储；存储/口径差异见 quantization 节）",
            "note": "EXP2/4/5 计时实验凡引用本 JSON，须注明用哪条管线",
        },
    }
    with open(OUT, "w") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=1)
    ok = all(v.get("checks_all_pass") for v in verif.values()) if verif and "error" not in verif else False
    print(f"written {OUT}")
    print("verification all pass:", ok)
    for S, v in (verif.items() if "error" not in verif else []):
        print(f"  S={S}: pass={v['checks_all_pass']} total_code={v['code_selected_per_kvhead_max']} "
              f"total_formula={v['formula_total']}")


if __name__ == "__main__":
    main()
