import torch

def get_method_name_with_info(args):
    if args.method == "quest":
        return f"quest_{args.quest_block_size}_{args.quest_topk}"
    elif args.method == "twi":
        return f"twi_{args.twi_block_size}_{args.twi_level1_topk}_{args.twi_level2_topp}"
    elif args.method == "tia":
        async_info = "_async" if args.tia_enable_async_topk else ""
        return f"tia_{args.tia_block_size}_{args.tia_level1_topk}_{args.tia_level2_topk}_c{args.tia_level2_cmp_ratio}{async_info}"
    elif args.method == "tli":
        # 【F11 修复（kimi3 清单 2026-10-08）】ab 用「生效值」而非 flag 原值：
        # tli_subspace=full 时 TLIIndexer.__init__ 内部强制 enable_subspace=False
        # → 不写 "A"（原按 flag 原值写 "A"，默认 full 口径下文件名虚标子空间，
        # 与实际 treatment 不符——历史 ARM_CONTRACT 串 tli_64_128_1024_c4_A 的
        # "A" 即此虚标，既有文件名不追溯改动）。
        subspace = getattr(args, "tli_subspace", "full")
        eff_subspace = getattr(args, "tli_enable_subspace", True)
        if subspace == "full":
            eff_subspace = False          # 与 TLIIndexer.__init__ 同源
        elif subspace in ("rope", "nope"):
            eff_subspace = True
        ab = ("A" if eff_subspace else "") + ("B" if args.tli_enable_kmeans else "") + ("D" if args.tli_enable_layer_skip else "")
        # 【B09 修复（kimi3 清单 2026-10-08）】α/β/γ 是 treatment 的一部分，
        # 必须进输出文件名（a0.25_b0.125_g0.625 风格），否则不同分区配置
        # 共享同 method_name → 同路径 w 模式互相覆盖 / 同名混装不可区分。
        # γ 缺省/显式 None 渲染 "off"（argparse 路径默认 1.0 不会是 None）。
        alpha = getattr(args, "tli_alpha", 0.0)
        beta = getattr(args, "tli_beta", 0.0)
        gamma = getattr(args, "tli_gamma", None)
        g_str = "off" if gamma is None else f"{gamma:g}"
        # 【10-09 TL-PREFILL-PROVENANCE-001 修复】稀疏 prefill 开关必须进输出文件名，
        # 否则 P-off/P-on 同路径 w 模式互相覆盖（GPT 审计 B09 族新实例）。默认不传
        # flag 时后缀为空 → 在跑链文件名逐位不变。
        p = "_P" if getattr(args, "tli_sparse_prefill", False) else ""
        return (f"tli_{args.tia_block_size}_{args.tia_level1_topk}_"
                f"{args.tia_level2_topk}_c{args.tia_level2_cmp_ratio}_"
                f"{ab}a{alpha:g}_b{beta:g}_g{g_str}{p}")
    elif args.method == "none":
        return "none"
    else:
        raise ValueError
