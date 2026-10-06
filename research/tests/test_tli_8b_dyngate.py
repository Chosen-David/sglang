# #60 动态 gate 验证（8B）：SGLANG_TLI_DYN_GATE=1 下 prefill 统计
# per-layer far mass → decode 动态 skip_far。打印每层统计值与跳层决策，
# 输出与 gate-off 对照（安全任务 narrativeqa 应不掉语义）。
import json
import os
import sys
import warnings

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
warnings.filterwarnings("ignore")

import torch

from sglang.srt.layers.attention.tli.indexer import TLIIndexer

_orig_select = TLIIndexer.select
SEEN = set()


def probed_select(self, index, q, t, tail_k=None, use_l1_kernel=False, use_l2_kernel=False):
    out = _orig_select(self, index, q, t, tail_k, use_l1_kernel, use_l2_kernel)
    # 置位发生在 _orig_select 开头——打印须在调用后
    key = id(self)
    if key not in SEEN:
        SEEN.add(key)
        print(f"[dyn] far_stat={self.dyn_far_stat} skip_far={self.skip_far}",
              flush=True)
    return out


TLIIndexer.select = probed_select


def main():
    from sglang import Engine

    MODEL = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B"
    eng = Engine(
        model_path=MODEL,
        attention_backend="tli",
        dtype="bfloat16",
        device="cuda",
        tp_size=1,
        mem_fraction_static=0.8,
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
    print(f"[dyngate-{os.environ.get('SGLANG_TLI_DYN_GATE', '0')}]: "
          f"{json.dumps([o['text'][:150] for o in outs], ensure_ascii=False)}")
    eng.shutdown()
    print("DONE")


if __name__ == "__main__":
    main()
