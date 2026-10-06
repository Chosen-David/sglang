# 干跑验证：swa 排除出双池修复（2026-09-29 用户澄清的严格口径）
# 检查项：①L1dbg i_n_blk_max < swa_lo_blk（near 池不含 swa 块）
#         ②L2dbg far_hi = near_blks*bs（far 细筛池不吞 near 区）
#         ③总预算仍恒定（K2=1024 语义）
import runpy
import sys

sys.path.insert(0, "/home/wangyuanshuo02/two-level-attention")

sys.argv = [
    "pred", "--model", "Qwen3-8B",
    "--model_path", "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B",
    "--task", "hotpotqa", "--method", "tli", "--e", "0", "--t", "dbg",
    "--dataset-path", "/home/wangyuanshuo02/datasets/LongBench/data",
    "--config-path", "benchmark/LongBench/config",
    "--output-dir", "/tmp/tli_dbg_out",
    "--tia_level1_topk", "128", "--tia_level2_topk", "1024", "--tia_level2_cmp_ratio", "4",
    "--tli_enable_kmeans", "false", "--tli_enable_layer_skip", "false",
    "--tli_alpha", "0.125", "--tli_beta", "0.25", "--tli_gamma", "0.125",
    "--pred_postfix", "_dbg",
]

import os

os.environ["TLI_DEBUG"] = "1"
os.makedirs("/tmp/tli_dbg_out/pred_dbg", exist_ok=True)
os.chdir("/home/wangyuanshuo02/two-level-attention")

import datasets as hf_datasets

_orig = hf_datasets.load_dataset


def _load_1(*a, **kw):
    ds = _orig(*a, **kw)
    try:
        return ds.select(range(1))
    except Exception:
        return ds


hf_datasets.load_dataset = _load_1
runpy.run_path("/home/wangyuanshuo02/two-level-attention/benchmark/LongBench/pred.py", run_name="__main__")
