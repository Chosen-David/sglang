# 30B 稀疏 prefill bug 终极 dump（#58）：patch _sparse_extend_one 入口，
# 打印实际收到的 sel（行/列切片）与输出统计。TLI vs uniform 对比。
# 若 backend 收到的 sel 与 probe 截获的一致而输出仍崩 → 消费端数值 bug；
# 若收到的 sel 不一致 → 传递路径被改写。
import json
import sys
import warnings

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
warnings.filterwarnings("ignore")

import torch

from sglang.srt.layers.attention.tli.indexer import TLIIndexer
from sglang.srt.layers.attention.tli.backend import TLISparseAttnBackend

_orig_select = TLIIndexer.select_batched
MODE = "tli"  # tli | uniform
DUMP_N = 0


def probed_select(self, index, q, t_arr, row_chunk=64):
    sel = _orig_select(self, index, q, t_arr, row_chunk)
    if MODE == "uniform":
        S = index["S"]
        n, Hkv, K2 = sel.shape
        t_last = int(t_arr[-1])
        grid = torch.linspace(0, max(0, t_last - 1), K2, device=sel.device).long()
        grid = torch.sort(grid).values
        sel = grid.view(1, 1, K2).expand(n, Hkv, K2).contiguous()
    return sel


TLIIndexer.select_batched = probed_select

_orig_ext = TLISparseAttnBackend._sparse_extend_one


def probed_ext(self, q_b, sel, locs, pool, layer_id, Hkv, G):
    global DUMP_N
    out = _orig_ext(self, q_b, sel, locs, pool, layer_id, Hkv, G)
    if DUMP_N < 12:
        n, H, D = q_b.shape[0], q_b.shape[1], self.head_dim
        s_last = sel[-1, 0]  # 最后一行 head0
        print(
            f"[ext#{DUMP_N}] L{layer_id} nq={n} sel: dtype={sel.dtype} "
            f"min={int(s_last.min())} max={int(s_last.max())} "
            f"n_inf={(s_last == float('inf')).sum()} head0[:24]={s_last[:24].tolist()}",
            flush=True,
        )
        print(
            f"[ext#{DUMP_N}] out: absmean={float(out[-1].abs().mean()):.4f} "
            f"absmax={float(out[-1].abs().max()):.2f} "
            f"nan={int(torch.isnan(out).sum())}",
            flush=True,
        )
        DUMP_N += 1
    return out


TLISparseAttnBackend._sparse_extend_one = probed_ext


def main():
    from sglang import Engine

    MODEL = "/home/wangyuanshuo02/models/Qwen3-30B-A3B"
    eng = Engine(
        model_path=MODEL,
        attention_backend="tli",
        dtype="bfloat16",
        device="cuda",
        tp_size=1,
        mem_fraction_static=0.7,
        trust_remote_code=True,
        disable_radix_cache=True,
        watchdog_timeout=1800,
        cuda_graph_config={"decode": {"backend": "full", "bs": [4]}, "prefill": {"backend": "disabled"}},
    )
    rows = [json.loads(l) for l in open(
        "/home/wangyuanshuo02/datasets/LongBench/data/narrativeqa.jsonl")]
    ctx = sorted(rows, key=lambda r: -len(r.get("context", "")))[0]["context"][:32000]
    prompt = ctx + "\n\nSummarize the above text in one sentence:"
    outs = eng.generate([prompt], sampling_params={"temperature": 0.0, "max_new_tokens": 1})
    print(f"[dump-{MODE}]: {json.dumps([o['text'][:150] for o in outs], ensure_ascii=False)}")
    eng.shutdown()
    print("DONE")


if __name__ == "__main__":
    main()
