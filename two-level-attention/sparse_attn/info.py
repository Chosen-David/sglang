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
        ab = ("A" if args.tli_enable_subspace else "") + ("B" if args.tli_enable_kmeans else "") + ("D" if args.tli_enable_layer_skip else "")
        # 【10-09 TL-PREFILL-PROVENANCE-001 修复】稀疏 prefill 开关必须进输出文件名，
        # 否则 P-off/P-on 同路径 w 模式互相覆盖（GPT 审计 B09 族新实例）。默认不传
        # flag 时后缀为空 → 在跑链文件名逐位不变。
        p = "_P" if getattr(args, "tli_sparse_prefill", False) else ""
        return f"tli_{args.tia_block_size}_{args.tia_level1_topk}_{args.tia_level2_topk}_c{args.tia_level2_cmp_ratio}_{ab}{p}"
    elif args.method == "none":
        return "none"
    else:
        raise ValueError
