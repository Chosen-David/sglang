# #58 sel 内容审计（不改 sel 只统计）：全行的行级因果越界（sel[r] > t_arr[r]）
# 与重复度（K2 - unique）。_sparse_extend_one 无掩码，越界=泄漏本 chunk 未来
# token、重复=softmax 加权偏置。max_new_tokens=1 快速诊断。
import json
import sys
import warnings

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
warnings.filterwarnings("ignore")

import torch

from sglang.srt.layers.attention.tli.indexer import TLIIndexer

_orig_select = TLIIndexer.select_batched
DUMP_N = 0


def audit_select(self, index, q, t_arr, row_chunk=64):
    global DUMP_N
    sel = _orig_select(self, index, q, t_arr, row_chunk)
    S = index["S"]
    n, Hkv, K2 = sel.shape
    t_dev = torch.as_tensor(t_arr, device=sel.device).long().view(n, 1, 1)
    # 行级因果越界：sel[r,h,k] > t_arr[r]（t_r 含本位置，合法值域 [0, t_r]）
    over = sel > t_dev
    # 越界程度：超出的距离分布
    over_amt = (sel - t_dev).clamp(min=0).float()
    # 分位近似：kthvalue（quantile 对 33M 元素超限）
    ov_flat = over_amt.flatten()
    k99 = max(1, int(ov_flat.numel() * 0.99))
    p99 = float(ov_flat.kthvalue(k99).values)
    # 重复度：per (row, head) 唯一数
    uniq_per_rowh = torch.tensor([
        torch.unique(sel[r, h]).numel()
        for r in range(0, n, max(1, n // 8))
        for h in range(Hkv)
    ], dtype=torch.float)
    if DUMP_N < 8:
        print(
            f"[audit#{DUMP_N}] S={S} n={n} K2={K2} | "
            f"caus_over={int(over.sum())}/{n * Hkv * K2} "
            f"rows_with_over={int((over.any(dim=(1, 2))).sum())}/{n} "
            f"over_max={int(over_amt.max())} over_p99={p99:.0f} over_mean={float(ov_flat.mean()):.1f} | "
            f"uniq_per_rowh mean={float(uniq_per_rowh.mean()):.1f} min={float(uniq_per_rowh.min())} / {K2}",
            flush=True,
        )
        DUMP_N += 1
    return sel


TLIIndexer.select_batched = audit_select


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
    outs = eng.generate([prompt], sampling_params={"temperature": 0.0, "max_new_tokens": 1})
    print(f"[audit]: {json.dumps([o['text'][:80] for o in outs], ensure_ascii=False)}")
    eng.shutdown()
    print("DONE")


if __name__ == "__main__":
    main()
