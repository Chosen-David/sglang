# M11 e2e bug 对拍定位：同一 _sparse_extend_one 输入，eager vs fused kernel
# 输出张量级 diff（scheduler 子进程顶层 patch，spawn re-import 生效）
import json
import os
import sys
import warnings

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
warnings.filterwarnings("ignore")

import torch

from sglang.srt.layers.attention.tli.backend import TLISparseAttnBackend
from sglang.srt.layers.attention.tli.kernels import tli_sparse_gather_attn_dot

_orig = TLISparseAttnBackend._sparse_extend_one
DIFFS = []
N = [0]


def probed(self, q_b, sel, locs, pool, layer_id, Hkv, G, q_raw=None):
    out_e = _orig(self, q_b, sel, locs, pool, layer_id, Hkv, G, q_raw=None)  # eager
    k_buf, v_buf = pool.get_kv_buffer(layer_id)
    pool_sel = locs[sel]
    out_k = tli_sparse_gather_attn_dot(
        q_raw, pool_sel, k_buf, v_buf, G, S_loc=k_buf.shape[0]).view(
        q_b.shape[0], -1)
    out_k2 = tli_sparse_gather_attn_dot(
        q_raw, pool_sel, k_buf, v_buf, G, S_loc=k_buf.shape[0]).view(
        q_b.shape[0], -1)
    dd = (out_k.float() - out_k2.float()).abs().max().item()
    if dd > 1e-5:
        print(f"[m11race] layer={layer_id} kernel-run1-vs-run2 diff={dd:.4f}", flush=True)
    # 现场 mini-eager（与 _orig 同输入同语义）：分辨 _orig 是否吃到不同输入
    D_ = q_raw.shape[-1]
    out_m = torch.empty_like(q_raw)
    for h in range(Hkv):
        q_h = q_raw[:, h*G:(h+1)*G].float()
        pp = pool_sel[:, h].long()
        k_sel = k_buf[pp, h].float()
        v_sel = v_buf[pp, h].float()
        att = torch.einsum("ngd,nkd->ngk", q_h, k_sel) * (D_**-0.5)
        att = torch.softmax(att, dim=-1)
        out_m[:, h*G:(h+1)*G] = torch.einsum("ngk,nkd->ngd", att, v_sel).to(q_raw.dtype)
    dm = (out_e.float() - out_m.float().view(out_e.shape)).abs().max().item()
    if N[0] < 3:
        print(f"[m11mini] layer={layer_id} _orig-vs-mini-eager diff={dm:.5f} "
              f"mini-vs-kernel={(out_m.float().view(out_e.shape)-out_k.float()).abs().max().item():.5f}", flush=True)
    # 终极：现场 tensors 的属性全打印（找非 contiguous/怪 stride）
    if N[0] == 0:
        print(f"[m11attr] q_raw {tuple(q_raw.shape)} stride={q_raw.stride()} contig={q_raw.is_contiguous()} "
              f"ptr_align={q_raw.data_ptr() % 16}", flush=True)
        print(f"[m11attr] sel {tuple(sel.shape)} stride={sel.stride()} dtype={sel.dtype} "
              f"ptr_align={sel.data_ptr() % 16}", flush=True)
        print(f"[m11attr] pool_sel {tuple(pool_sel.shape)} stride={pool_sel.stride()} dtype={pool_sel.dtype} "
              f"ptr_align={pool_sel.data_ptr() % 16}", flush=True)
        print(f"[m11attr] k_buf {tuple(k_buf.shape)} stride={k_buf.stride()} contig={k_buf.is_contiguous()} "
              f"ptr_align={k_buf.data_ptr() % 16}", flush=True)
        print(f"[m11attr] out_k {tuple(out_k.shape)} stride={out_k.stride()}", flush=True)
    d = (out_e.float() - out_k.float()).abs().max().item()
    nq = q_b.shape[0]
    DIFFS.append((layer_id, nq, d))
    if layer_id == 0 and N[0] == 0:
        torch.save({
            "q_b": q_b[:16].cpu(), "q_raw": q_raw[:16].cpu(),
            "sel": sel[:16].cpu(), "locs": locs.cpu(),
            "k_buf": k_buf.cpu().clone(), "v_buf": v_buf.cpu().clone(),
            "out_e": out_e[:16].cpu(), "out_k": out_k[:16].cpu(),
            "Hkv": Hkv, "G": G, "S_loc": k_buf.shape[0],
        }, "/tmp/m11_dump2.pt")
        print(f"[m11dump2] layer0 first-call saved, d={d:.4f}", flush=True)
    N[0] += 1
    if N[0] <= 5 or d > 0.05:
        print(f"[m11dbg] layer={layer_id} nq={nq} maxdiff={d:.4f}", flush=True)
        if d > 0.05 and N[0] <= 10:
            # 定位首分歧：行级 diff
            rd = (out_e.float() - out_k.float()).abs().amax(-1)
            bad = (rd > 0.05).nonzero().flatten()
            print(f"  bad rows: {bad[:10].tolist()} / {nq}, "
                  f"rowdiff max={rd.max():.3f}", flush=True)
    return out_e  # 始终返回 eager（隔离 M11 bug，不影响生成）


TLISparseAttnBackend._sparse_extend_one = probed


def main():
    from test_dyngate_diag2 import MODEL, _TPL, MIND_TPL
    from sglang import Engine

    row = json.loads(open(
        "/home/wangyuanshuo02/datasets/LongBench/data/musique.jsonl").readline())
    prompt = MIND_TPL.format(p=_TPL["musique"].format(
        **{k: row.get(k, "") for k in ("context", "input")}))
    eng = Engine(model_path=MODEL, attention_backend="tli", dtype="bfloat16",
                 device="cuda", tp_size=1, mem_fraction_static=0.7,
                 trust_remote_code=True, disable_radix_cache=True,
                 disable_cuda_graph=True, watchdog_timeout=1800)
    eng.generate(["warmup"], sampling_params={"temperature": 0.0, "max_new_tokens": 1})
    o = eng.generate([prompt], sampling_params={"temperature": 0.0, "max_new_tokens": 32})
    print("[eager-out]", repr(o[0]["text"][:120]), flush=True)
    eng.shutdown()
    big = [x for x in DIFFS if x[2] > 0.05]
    print(f"[m11dbg] total calls={N[0]} bigdiff={len(big)} "
          f"max={max((x[2] for x in DIFFS), default=0):.4f}")


if __name__ == "__main__":
    main()
