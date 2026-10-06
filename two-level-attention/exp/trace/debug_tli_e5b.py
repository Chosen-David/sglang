# TLI 逐层 debug：找 0.954 vs 0.999 的质量差距来源
import sys
import argparse
import torch
import glob

sys.path.insert(0, "/home/wangyuanshuo02/two-level-attention")

parser = argparse.ArgumentParser()
from sparse_attn.arguments import add_sparse_attn_args
add_sparse_attn_args(parser)

TRACE = "/tmp/trace/qwen3-8b/lb_hotpotqa_0"
layer_files = sorted(glob.glob(f"{TRACE}/layer*.pt"))


def run(tag, extra):
    args = parser.parse_args([
        "--method", "tli", "--tia_level2_topk", "1024", "--tia_level2_cmp_ratio", "4",
    ] + extra)
    from sparse_attn.indexer import indexer_type_dict
    TLI = indexer_type_dict["tli"]
    TIA = indexer_type_dict["tia"]
    print(f"\n=== {tag} ===")
    for lf in layer_files:
        d = torch.load(lf, map_location="cuda:0")
        k, q, qpos, S = d["k"].float(), d["q"].float(), d["qpos"].cuda(), d["S"]
        t = qpos[-1].item()
        Hkv, H, D = k.shape[1], q.shape[1], k.shape[-1]
        G = H // Hkv
        qg = q[-1:].reshape(1, Hkv, G, D)
        s = torch.einsum("bhgd,chd->bhgc", qg, k).sum(-2) * (D ** -0.5)
        s = s.masked_fill(torch.arange(S, device="cuda").view(1, 1, S) > t, float("-inf"))
        p = torch.softmax(s, dim=-1)
        total = p.sum().item()
        pm = p.mean(dim=(0, 1))
        far_mass = pm[128:t - 2048].sum().item()
        layer_id = int(lf.split("layer")[1].split(".")[0])
        k_in = k.unsqueeze(0).to(torch.bfloat16)
        q_in = q[-1:].reshape(1, 1, H, D).to(torch.bfloat16)
        cu = torch.tensor([0, S], device="cuda", dtype=torch.int32)
        covs = {}
        for name, cls in [("tli", TLI), ("tia", TIA)]:
            idx = cls(args)
            idx.layer_idx = layer_id
            mask, _ = idx.prepare_mask(q_in, torch.tensor([t], device="cuda"), k_in, cu)
            covs[name] = p[0][mask[0, 0]].sum().item() / total
            idx.clear()
        diff = covs["tia"] - covs["tli"]
        flag = " <<<" if diff > 0.01 else ""
        print(f"L{layer_id:02d} far={far_mass:.3f} TIA={covs['tia']:.4f} TLI={covs['tli']:.4f} diff={diff:+.4f}{flag}")
        del k, q, s, p, d
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # 全开（A+B+D'）
    run("A+B+D' (far_blocks=16)", [])
    # 只 A（子空间），关 B 和 D'
    run("A only", ["--tli_enable_kmeans", "false", "--tli_enable_layer_skip", "false"])
    # A + D'
    run("A+D'", ["--tli_enable_kmeans", "false"])
    # A + B
    run("A+B (far_blocks=32)", ["--tli_enable_layer_skip", "false", "--tli_far_blocks", "32"])
