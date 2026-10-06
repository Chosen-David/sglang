# M11 e2e bug：dump 首个 _sparse_extend_one 调用的输入（q/sel/locs/k/v）
# 供离线复现 kernel 与 eager 的 diff
import os
import sys
import warnings

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
warnings.filterwarnings("ignore")

import torch

from sglang.srt.layers.attention.tli.backend import TLISparseAttnBackend

_orig = TLISparseAttnBackend._sparse_extend_one
DONE = [False]


def probed(self, q_b, sel, locs, pool, layer_id, Hkv, G, q_raw=None):
    if not DONE[0]:
        k_buf, v_buf = pool.get_kv_buffer(layer_id)
        torch.save({
            "q_b": q_b[:16].cpu(), "q_raw": q_raw[:16].cpu(),
            "sel": sel[:16].cpu(), "locs": locs.cpu(),
            "k_buf": k_buf[:50000].cpu(), "v_buf": v_buf[:50000].cpu(),
            "Hkv": Hkv, "G": G, "layer_id": layer_id,
        }, "/tmp/m11_dump.pt")
        print(f"[m11dump] saved layer={layer_id} nq={q_b.shape[0]} "
              f"sel[1,:1,:8]={sel[1,0,:8].tolist()}", flush=True)
        DONE[0] = True
    return _orig(self, q_b, sel, locs, pool, layer_id, Hkv, G, q_raw=q_raw)


TLISparseAttnBackend._sparse_extend_one = probed


def main():
    from test_dyngate_diag2 import MODEL, _TPL, MIND_TPL
    from sglang import Engine
    import json

    row = json.loads(open(
        "/home/wangyuanshuo02/datasets/LongBench/data/musique.jsonl").readline())
    prompt = MIND_TPL.format(p=_TPL["musique"].format(
        **{k: row.get(k, "") for k in ("context", "input")}))
    eng = Engine(model_path=MODEL, attention_backend="tli", dtype="bfloat16",
                 device="cuda", tp_size=1, mem_fraction_static=0.7,
                 trust_remote_code=True, disable_radix_cache=True,
                 disable_cuda_graph=True, watchdog_timeout=1800)
    eng.generate(["warmup"], sampling_params={"temperature": 0.0, "max_new_tokens": 1})
    eng.generate([prompt], sampling_params={"temperature": 0.0, "max_new_tokens": 1})
    eng.shutdown()


if __name__ == "__main__":
    main()
