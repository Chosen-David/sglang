# #58 终局隔离实验：_sparse_extend_one 换成 dense causal oracle（完全忽略
# sel，用 locs 全量 K/V + 正确因果 mask）。区别于 dense_threshold 实验之处：
# 本实验仍执行 build_block_index / select_batched 全链路（含共享 pool 写入），
# 只替换最后的 gather+softmax 消费端。
#   输出好  → sel 内容问题（尽管 truecov cov 0.94）→ 转 per-head 选择/budget
#   输出崩  → build/select 的副作用（pool 污染/k_buf gather/dtype）——sel 无辜
# 用法：CUDA_VISIBLE_DEVICES=0 python3 test_tli_30b_denseoracle.py
import json
import sys
import warnings

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
warnings.filterwarnings("ignore")

import torch

from sglang.srt.layers.attention.tli.backend import TLISparseAttnBackend

_orig_ext = TLISparseAttnBackend._sparse_extend_one
DUMP_N = 0


def oracle_ext(self, q_b, sel, locs, pool, layer_id, Hkv, G):
    global DUMP_N
    if DUMP_N < 6:
        print(f"[oracle-ext#{DUMP_N}] L{layer_id} nq={q_b.shape[0]} S={locs.shape[0]} "
              f"sel[max]={int(sel.max())} dtype={q_b.dtype}", flush=True)
        DUMP_N += 1
    # ---- dense causal oracle（复用 _dense_extend_one 逻辑 + 行分块控显存）----
    k_buf, v_buf = pool.get_kv_buffer(layer_id)
    nq, H, D = q_b.shape[0], q_b.shape[1], self.head_dim
    S = locs.shape[0]
    out = torch.empty(nq, H * D, device=q_b.device, dtype=q_b.dtype)
    q_g = q_b.reshape(nq, Hkv, G, D)
    k_e = k_buf[locs].float().transpose(0, 1)  # [Hkv, S, D]
    v_e = v_buf[locs].float().transpose(0, 1)
    pos = torch.arange(S, device=q_b.device)
    row_chunk = 256  # [256, Hkv, G, S] fp32 峰值 ≈ 1GB @ S=32K
    for r0 in range(0, nq, row_chunk):
        r1 = min(r0 + row_chunk, nq)
        att = torch.einsum("ahgd,hsd->ahgs", q_g[r0:r1], k_e) * (D**-0.5)
        qpos = (S - nq + torch.arange(r0, r1, device=q_b.device)).view(r1 - r0, 1)
        causal = pos.view(1, S) <= qpos
        att = att.masked_fill(~causal.view(r1 - r0, 1, 1, S), float("-inf"))
        att = torch.softmax(att, dim=-1)
        o = torch.einsum("ahgs,hsd->ahgd", att, v_e)
        out[r0:r1] = o.reshape(r1 - r0, -1).to(q_b.dtype)
    return out


TLISparseAttnBackend._sparse_extend_one = oracle_ext


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
    print(f"[oracle-dense]: {json.dumps([o['text'][:150] for o in outs], ensure_ascii=False)}")
    eng.shutdown()
    print("DONE")


if __name__ == "__main__":
    main()
