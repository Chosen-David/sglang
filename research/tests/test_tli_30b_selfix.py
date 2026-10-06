# #58 修复验证：select_batched 输出后处理——早期行（t_r < K2）的 sel 含
# 未来位置（行级因果越界，审计 caus_over=2095104 全来自前 K2-1 行），
# _sparse_extend_one 无掩码直接 softmax → 早期行泄漏未来 token 逐层传播。
# 修复：t_r < K2 的行整行替换为 [0, t_r] 均匀重复 grid（数学等价 dense 行）。
# t_r >= K2 的行不动（审计显示无越界）。64 token 语义级验证。
import json
import sys
import warnings

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
warnings.filterwarnings("ignore")

import torch

from sglang.srt.layers.attention.tli.indexer import TLIIndexer

_orig_select = TLIIndexer.select_batched
DUMP_N = 0


def fixed_select(self, index, q, t_arr, row_chunk=64):
    global DUMP_N
    sel = _orig_select(self, index, q, t_arr, row_chunk)
    S = index["S"]
    n, Hkv, K2 = sel.shape
    dev = sel.device
    t_list = t_arr.tolist() if torch.is_tensor(t_arr) else list(t_arr)
    n_fix = 0
    for r in range(n):
        t_r = int(t_list[r])
        if t_r >= K2:
            continue
        n_fix += 1
        # [0, t_r] 均匀重复到 K2（softmax 数学等价 dense 行）
        reps = K2 // (t_r + 1)
        rem = K2 - reps * (t_r + 1)
        base = torch.arange(t_r + 1, device=dev).repeat_interleave(reps)
        if rem > 0:
            extra = torch.arange(rem, device=dev)
            grid = torch.cat([base, extra])
        else:
            grid = base
        sel[r] = grid.view(1, K2).expand(Hkv, K2)
    if DUMP_N < 4:
        print(f"[selfix#{DUMP_N}] S={S} n={n} K2={K2} fixed_rows={n_fix}", flush=True)
        DUMP_N += 1
    return sel


TLIIndexer.select_batched = fixed_select


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
    print(f"[selfix]: {json.dumps([o['text'][:150] for o in outs], ensure_ascii=False)}")
    eng.shutdown()
    print("DONE")


if __name__ == "__main__":
    main()
