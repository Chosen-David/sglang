# #58 30B prefill 性能归因：分计时 build_block_index vs select_batched
# vs _sparse_extend_one（48 层 × 4 chunks，21K token prompt）。
# 用法：CUDA_VISIBLE_DEVICES=0 python3 test_tli_30b_attribution.py
import json
import os
import sys
import time
import warnings

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
warnings.filterwarnings("ignore")

import torch

from sglang.srt.layers.attention.tli.indexer import TLIIndexer
from sglang.srt.layers.attention.tli.backend import TLISparseAttnBackend

ATTR_FILE = "/tmp/tli_30b_attr_parts.json"
T_BUILD = [0.0]
T_SELECT = [0.0]
T_EXT = [0.0]
N_BUILD = [0]
N_SELECT = [0]
N_EXT = [0]
print("[attribution] patches installed (module level)", flush=True)


def _dump_parts():
    """按 pid 落盘分计时（scheduler 子进程执行体，主进程读）"""
    try:
        d = {}
        if os.path.exists(ATTR_FILE):
            d = json.load(open(ATTR_FILE))
        d[str(os.getpid())] = {
            "build_s": T_BUILD[0], "select_s": T_SELECT[0], "ext_s": T_EXT[0],
            "n_build": N_BUILD[0], "n_select": N_SELECT[0], "n_ext": N_EXT[0],
        }
        json.dump(d, open(ATTR_FILE, "w"))
    except Exception as e:
        print(f"[attr-dump] failed: {e}", flush=True)


_orig_build = TLIIndexer.build_block_index


def probed_build(self, k):
    global N_BUILD
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    index = _orig_build(self, k)
    torch.cuda.synchronize()
    T_BUILD[0] += time.perf_counter() - t0
    N_BUILD[0] += 1
    return index


TLIIndexer.build_block_index = probed_build

_orig_select = TLIIndexer.select_batched


def probed_select(self, index, q, t_arr, row_chunk=64):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    sel = _orig_select(self, index, q, t_arr, row_chunk)
    torch.cuda.synchronize()
    T_SELECT[0] += time.perf_counter() - t0
    N_SELECT[0] += 1
    return sel


TLIIndexer.select_batched = probed_select

_orig_ext = TLISparseAttnBackend._sparse_extend_one


def probed_ext(self, q_b, sel, locs, pool, layer_id, Hkv, G):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = _orig_ext(self, q_b, sel, locs, pool, layer_id, Hkv, G)
    torch.cuda.synchronize()
    T_EXT[0] += time.perf_counter() - t0
    N_EXT[0] += 1
    _dump_parts()
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
        mem_fraction_static=0.55,
        trust_remote_code=True,
        disable_radix_cache=True,
        watchdog_timeout=1800,
        cuda_graph_config={"decode": {"backend": "disabled"}, "prefill": {"backend": "disabled"}},
    )
    rows = [json.loads(l) for l in open(
        "/home/wangyuanshuo02/datasets/LongBench/data/narrativeqa.jsonl")]
    ctx = sorted(rows, key=lambda r: -len(r.get("context", "")))[0]["context"][:85000]
    prompt = ctx + "\n\nSummarize the above text in one sentence:"
    # 预热
    eng.generate(["hello"], sampling_params={"temperature": 0.0, "max_new_tokens": 1})
    if os.path.exists(ATTR_FILE):
        os.remove(ATTR_FILE)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outs = eng.generate([prompt], sampling_params={"temperature": 0.0, "max_new_tokens": 1})
    torch.cuda.synchronize()
    total = time.perf_counter() - t0
    # 汇总子进程落盘的分计时
    import time as _t
    parts = {}
    for _ in range(30):  # 等子进程 flush
        if os.path.exists(ATTR_FILE):
            parts = json.load(open(ATTR_FILE))
            break
        _t.sleep(1)
    b = sum(p["build_s"] for p in parts.values())
    s = sum(p["select_s"] for p in parts.values())
    e = sum(p["ext_s"] for p in parts.values())
    nb = sum(p["n_build"] for p in parts.values())
    ns = sum(p["n_select"] for p in parts.values())
    ne = sum(p["n_ext"] for p in parts.values())
    attn_total = b + s + e
    print(f"[attribution] total={total:.2f}s | build={b:.2f}s({nb}) "
          f"select={s:.2f}s({ns}) ext={e:.2f}s({ne}) "
          f"| attn_sum={attn_total:.2f}s ({attn_total/total*100:.1f}%) "
          f"| non-attn={total-attn_total:.2f}s", flush=True)
    res = {
        "total_s": round(total, 2),
        "build_s": round(b, 2),
        "select_s": round(s, 2),
        "ext_s": round(e, 2),
        "out": outs[0]["text"][:60],
    }
    json.dump(res, open("/tmp/tli_30b_attribution.json", "w"), indent=1)
    eng.shutdown()
    print("DONE")


if __name__ == "__main__":
    main()
