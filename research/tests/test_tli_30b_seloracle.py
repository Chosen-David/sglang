# #58 sel 内容终局二分：用真实 K 算全因果分数，构造两种 oracle sel：
#   mean    —— per-(row, kv-head) 用组内 G 个 q-head 平均分布的 top-K2（理想共享集合）
#   perhead —— G 个 q-head 各自 top-(K2//G) 的并集（不足按 mean 分数补齐/超截断）
# 输出对照 dense-oracle（" Jacob"）：
#   mean 好 + perhead 好 → TLI 打分实现失真（修复=校准/打分）
#   mean 崩 + perhead 好 → GQA 组内分化根因定稿（修复=per-q-head sel）
#   都崩                  → K2 预算根本不足（30B 需要更大预算/不同架构）
# 用法：SEL_ORACLE=mean|perhead CUDA_VISIBLE_DEVICES=0 python3 test_tli_30b_seloracle.py
import json
import os
import sys
import warnings

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
warnings.filterwarnings("ignore")

import torch

from sglang.srt.layers.attention.tli.indexer import TLIIndexer

_orig_build = TLIIndexer.build_block_index
_orig_select = TLIIndexer.select_batched
MODE = os.environ.get("SEL_ORACLE", "mean")
DUMP_N = 0


def probed_build(self, k):
    index = _orig_build(self, k)
    index["_k_all"] = k.detach()  # [S, Hkv, D] fp32 真实 K
    return index


TLIIndexer.build_block_index = probed_build


def uniform_repeat(vals, K2):
    """[m] 位置均匀重复到长度 K2（softmax 数学等价 dense 归一化）。

    m=0 → 全 0；m<=K2 时每位置重复 K2//m 或 +1 次（差 ≤1，
    重复权重比 ≤ 1+1/(K2//m)，K2=1024/m≤1024 时偏差 ≤0.1%）。
    """
    m = vals.numel()
    if m == 0:
        return torch.zeros(K2, dtype=torch.long, device=vals.device)
    reps = torch.full((m,), K2 // m, dtype=torch.long, device=vals.device)
    rem = K2 - (K2 // m) * m
    if rem > 0:
        reps[:rem] += 1
    return vals.repeat_interleave(reps)


def oracle_select(self, index, q, t_arr, row_chunk=64):
    global DUMP_N
    sel = _orig_select(self, index, q, t_arr, row_chunk)
    k_all = index.get("_k_all")
    if k_all is None:
        return sel
    S = index["S"]
    n, Hkv, K2 = sel.shape
    H = q.shape[1]
    G = H // Hkv
    D = k_all.shape[-1]
    dev = sel.device
    pos = torch.arange(S, device=dev)
    k_e = k_all.repeat_interleave(G, dim=1)  # [S, H, D]
    new = sel.clone()
    k_per_head = max(4, K2 // G)
    for r0 in range(0, n, 512):
        r1 = min(r0 + 512, n)
        sc = torch.einsum("nhd,shd->nhs", q[r0:r1].float(), k_e) * (D**-0.5)
        for ri in range(r1 - r0):
            t_r = int(t_arr[r0 + ri])
            causal = pos <= t_r
            sc_r = sc[ri].masked_fill(~causal.unsqueeze(0), float("-inf")).view(Hkv, G, S)
            pm = sc_r.mean(1)  # [Hkv, S] 组内平均分布（未 softmax，单调等价）
            if MODE == "mean":
                idx = torch.topk(pm, min(K2, t_r + 1), dim=-1).indices  # [Hkv, m]
                row = torch.stack([uniform_repeat(idx[h], K2) for h in range(Hkv)])
                new[r0 + ri] = row
            else:  # perhead
                row = torch.zeros(Hkv, K2, dtype=torch.long, device=dev)
                for h in range(Hkv):
                    i_top = torch.topk(sc_r[h], k_per_head, dim=-1).indices  # [G, k]
                    u = torch.unique(i_top.flatten())
                    if u.numel() > K2:
                        # 超预算：按 mean 分数截断
                        sc_u = pm[h, u]
                        keep = torch.topk(sc_u, K2).indices
                        u = u[keep]
                    elif u.numel() < K2:
                        # 不足：按 mean 分数补其余因果位置
                        rest_mask = causal.clone()
                        rest_mask[u] = False
                        cand = rest_mask.nonzero().squeeze(-1)
                        n_add = min(K2 - u.numel(), cand.numel())
                        if n_add > 0:
                            sc_c = pm[h, cand]
                            add = cand[torch.topk(sc_c, n_add).indices]
                            u = torch.cat([u, add])
                    u_sorted = torch.sort(u).values
                    row[h] = uniform_repeat(u_sorted, K2)
                new[r0 + ri] = row
    if DUMP_N < 3:
        last = new[-1]
        print(f"[seloracle-{MODE}#{DUMP_N}] S={S} K2={K2} last-row uniq="
              f"{torch.unique(last).numel()}/{K2} max={int(last.max())}", flush=True)
        DUMP_N += 1
    return new


TLIIndexer.select_batched = oracle_select


def main():
    from sglang import Engine

    MODEL = "/home/wangyuanshuo02/models/Qwen3-30B-A3B"
    eng = Engine(
        model_path=MODEL,
        attention_backend="tli",
        dtype="bfloat16",
        device="cuda",
        tp_size=1,
        mem_fraction_static=0.55,
        trust_remote_code=True,
        disable_radix_cache=True,
        watchdog_timeout=1800,
        cuda_graph_config={"decode": {"backend": "disabled"}, "prefill": {"backend": "disabled"}},
    )
    rows = [json.loads(l) for l in open(
        "/home/wangyuanshuo02/datasets/LongBench/data/narrativeqa.jsonl")]
    ctx = sorted(rows, key=lambda r: -len(r.get("context", "")))[0]["context"][:32000]
    prompt = ctx + "\n\nSummarize the above text in one sentence:"
    outs = eng.generate([prompt], sampling_params={"temperature": 0.0, "max_new_tokens": 64})
    print(f"[seloracle-{MODE}]: {json.dumps([o['text'][:150] for o in outs], ensure_ascii=False)}")
    eng.shutdown()
    print("DONE")


if __name__ == "__main__":
    main()
