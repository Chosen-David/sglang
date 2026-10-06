# #60 动态 gate 多跳复验：musique/qasper/multifieldqa_en（E5b 静态掩码
# 掉 4.8-5.9 分的同批任务）。核心验证点：多跳任务 far mass 高 → 动态 gate
# 不跳层 → 输出与 gate-off 一致（静态版正是在这批任务崩的）。
# 用法：TASK=musique GATE=1 CUDA_VISIBLE_DEVICES=x python3 test_tli_8b_dyngate_mh.py
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
    key = id(self)
    if key not in SEEN:
        SEEN.add(key)
        print(f"[dyn] far_stat={self.dyn_far_stat} skip_far={self.skip_far}", flush=True)
    return out


TLIIndexer.select = probed_select

TASK = os.environ.get("TASK", "musique")
GATE = os.environ.get("GATE", "1") == "1"

_TPL = json.load(open(
    "/home/wangyuanshuo02/two-level-attention/benchmark/LongBench/config/dataset2prompt.json"))


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
        f"/home/wangyuanshuo02/datasets/LongBench/data/{TASK}.jsonl")]
    r = sorted(rows, key=lambda x: -len(x.get("context", "")))[0]
    ctx = r["context"][:32000]
    q = r["input"]
    # LongBench 官方模板（E5b 同口径）
    prompt = _TPL[TASK].format(context=ctx, input=q)
    outs = eng.generate([prompt], sampling_params={"temperature": 0.0, "max_new_tokens": 64})
    print(f"[mh-{TASK}-gate{int(GATE)}]: "
          f"{json.dumps([o['text'][:150] for o in outs], ensure_ascii=False)}")
    print(f"[q-{TASK}] {q[:120]!r}")
    eng.shutdown()
    print("DONE")


if __name__ == "__main__":
    main()
