# 30B 稀疏 prefill bug 探针（#58 第三层隔离）：
# monkeypatch TLIIndexer.select_batched —— ①打印 sel 结构统计
# （唯一 token 数 / 因果越界数 / 值域），②用「均匀网格采样」替换 sel
# （同形状同宽度）。若输出恢复 → sel 内容损坏；仍垃圾 → 注意力 gather 侧损坏。
# 用法：CUDA_VISIBLE_DEVICES=0 python3 test_tli_8b_probe.py
import json
import sys
import warnings

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
warnings.filterwarnings("ignore")

import torch

from sglang.srt.layers.attention.tli.indexer import TLIIndexer, kq_unpack

_orig_select = TLIIndexer.select_batched
MODE = "audit"  # uniform | stats | audit（audit=对拍 sel vs 全量打分 top-k）


def probed_select(self, index, q, t_arr, row_chunk=64):
    sel = _orig_select(self, index, q, t_arr, row_chunk)
    S = index["S"]
    n, Hkv, K2 = sel.shape
    last = sel[-1]  # [Hkv, K2] 最后一行（决定首 token）
    uniq = torch.unique(last).numel()
    caus_bad = int((last >= S).sum())
    near_n = int((last >= S - 2048).sum())  # 近端+滑窗区占比
    print(
        f"[probe] S={S} sel={tuple(sel.shape)} "
        f"last-row: uniq={uniq}/{K2} caus_bad={caus_bad} "
        f"near2048={near_n} min={int(last.min())} max={int(last.max())}",
        flush=True,
    )
    if MODE == "audit":
        # 决定性对拍：同一 q2/kq_f 全量打分的 top-K2 vs sel（最后一行）
        p = self.profile
        q2 = self._q_refine(q[-1:].float()).reshape(1, Hkv, q.shape[1] // Hkv, self.nd2).sum(2)
        kq_f = kq_unpack(index["kq_q"][:S], index["kq_sc"][:S], index["kq_mn"][:S])
        k_absmax = float(kq_f.abs().max())
        print(f"[audit] kq_f absmax={k_absmax:.3g} sc_max={float(index['kq_sc'][:S].max()):.3g} "
              f"q2_absmax={float(q2.abs().max()):.3g}", flush=True)
        fine_full = torch.einsum("ahd,shd->ahs", q2, kq_f)[0]  # [Hkv, S]
        t_last = int(t_arr[-1])
        causal = torch.zeros(S, dtype=torch.bool, device=sel.device)
        causal[: t_last + 1] = True
        fine_full = fine_full.masked_fill(~causal.unsqueeze(0), float("-inf"))
        top = torch.topk(fine_full, min(K2, S), dim=-1).indices  # [Hkv, K2']
        # mass coverage：sel 集合捕获的 fine_full 分数质量（per head）
        sc_sorted = torch.sort(fine_full, dim=-1, descending=True).values
        total = torch.clamp(sc_sorted[:, :1024].sum(-1), min=1e-9)  # top-1024 总分
        for h in range(Hkv):
            s_h = torch.unique(last[h])
            cov = fine_full[h, s_h].sum() / total[h]
            ov = len(set(s_h.tolist()) & set(top[h].tolist())) / max(1, len(s_h))
            print(f"[audit] h{h}: mass_cov(top1024)={float(cov):.4f} "
                  f"iou_vs_topk={ov:.3f} sel_uniq={len(s_h)}", flush=True)
    if MODE == "uniform":
        # 均匀网格：覆盖 [0, t_last) 的 K2 个升序位置（全部因果）
        t_last = int(t_arr[-1])
        grid = torch.linspace(0, max(0, t_last - 1), K2, device=sel.device).long()
        grid = torch.sort(grid).values.clamp(max=max(0, t_last - 1))
        sel = grid.view(1, 1, K2).expand(n, Hkv, K2).contiguous()
        print(f"[probe] uniform grid: t_last={t_last} K2={K2} -> replaced", flush=True)
    return sel


TLIIndexer.select_batched = probed_select


def main():
    from sglang import Engine

    MODEL = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B"
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
    print(f"[probe-{MODE}]: {json.dumps([o['text'][:150] for o in outs], ensure_ascii=False)}")
    eng.shutdown()
    print("DONE")


if __name__ == "__main__":
    main()
