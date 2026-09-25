# M9：PCA 投影降维集成对拍测试（真实 trace，hotpotqa 校准基迁移口径）
# [1] 质量：select() 全链路 far recall —— PCA(r=16) vs 选择(δ=16→nd2=32)
#     vs 选择(δ=8→nd2=16，同维数对照，预期崩溃 0.378)
# [2] 存储口径：kq 24 B/token-head（16 grid + 8 scale）vs 40 B
# [3] 增量一致性：PCA update_block_index == 全量 rebuild（kq/kmin/kmax 逐位）
# [4] 批量一致性：select_decode_batched（pool flat 索引 nd2=16）vs per-request
#     select 集合一致；实选数 == token_budget
# [5] 延迟：select() per-call（trace S≈9.9K + 合成 131K microbench，后者已标注）
import json
import sys

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
import torch

from sglang.srt.layers.attention.tli.config import TLIProfile
from sglang.srt.layers.attention.tli.indexer import TLIIndexer, kq_unpack

dev = "cuda:0"
B = torch.load("/home/wangyuanshuo02/sglang/tli_pca_basis_r16.pt", map_location=dev)  # [36,8,128,16]
TRACES = [
    ("hotpotqa", "/tmp/trace/qwen3-8b/lb_hotpotqa_0"),
    ("musique", "/tmp/trace/qwen3-8b/lb_musique_0"),
    ("gov_report", "/tmp/trace/qwen3-8b/lb_gov_report_0"),
]
LAYERS = [3, 5, 10, 20, 33]
results = {"rows": [], "latency": {}, "storage": {}}

prof_sel = TLIProfile()          # δ=16 → nd2=32（当前生产配置）
prof_s8 = TLIProfile()           # δ=8 → nd2=16（同维选择对照）
prof_s8.delta = 8
idxer_sel = TLIIndexer(prof_sel, head_dim=128).to(dev)
idxer_s8 = TLIIndexer(prof_s8, head_dim=128).to(dev)

print(f"{'task':>12s} {'L':>3s} {'t':>6s} | {'sel32':>6s} {'sel16':>6s} {'pca16':>6s}")
for tname, TDIR in TRACES:
    for layer in LAYERS:
        d = torch.load(f"{TDIR}/layer{layer:02d}.pt", map_location=dev)
        k_real = d["k"].float()
        q_real = d["q"].float()
        qpos = d["qpos"].cuda()
        S, Hkv, D = k_real.shape
        H = q_real.shape[1]
        G = H // Hkv
        idxer_pca = TLIIndexer(prof_sel, head_dim=128, basis=B[layer]).to(dev)
        for t in [S // 2, S - 1]:
            in_range = torch.nonzero(qpos <= t).squeeze(1)
            qi = int(in_range[-1])
            q_t = q_real[qi : qi + 1]
            # oracle：dense 全维 GQA-sum far top-k2_far
            qg = q_t.reshape(1, Hkv, G, D)
            s_full = torch.einsum("bhgd,chd->bhgc", qg, k_real[: t + 1]).sum(-2)
            far_lo, near_len = 128, 2048
            far_hi = max(far_lo, t + 1 - near_len)
            k2_far = min(prof_sel.far_tokens, max(0, far_hi - far_lo),
                         max(0, prof_sel.token_budget - (prof_sel.sliding_window + far_lo)))
            if k2_far <= 0:
                continue
            oracle = torch.topk(s_full[0, :, far_lo:far_hi], k2_far, dim=-1).indices + far_lo
            oh = torch.zeros(Hkv, t + 1, device=dev)
            oh.scatter_(1, oracle, 1.0)

            def far_rec(idxer):
                index = idxer.build_block_index(k_real[: t + 1])
                sel = idxer.select(index, q_t, t)  # [Hkv, K2] far+near 拼接
                m = torch.zeros(Hkv, t + 1, device=dev)
                m.scatter_(1, sel[:, :k2_far], 1.0)
                return ((oh * m).sum(1) / k2_far).mean().item()

            rec = {
                "sel32": far_rec(idxer_sel),
                "sel16": far_rec(idxer_s8),
                "pca16": far_rec(idxer_pca),
            }
            results["rows"].append({"task": tname, "layer": layer, "t": t,
                                    **{k: round(v, 4) for k, v in rec.items()}})
            print(f"{tname:>12s} L{layer:02d} {t:6d} | " +
                  " ".join(f"{rec[k]:6.4f}" for k in ["sel32", "sel16", "pca16"]))
        del d, k_real, q_real
        torch.cuda.empty_cache()

n = len(results["rows"])
print(f"\n==== [1] far recall mean（{n} 行，pipeline 全链路口径）====")
for k in ["sel32", "sel16", "pca16"]:
    v = sum(r[k] for r in results["rows"]) / n
    print(f"{k:>6s}: {v:.4f}")
    results["latency"][k + "_rec"] = round(v, 4)

# ================= 2. 存储口径 =================
d = torch.load(f"{TRACES[0][1]}/layer03.pt", map_location=dev)
k_real = d["k"].float()
S, Hkv, D = k_real.shape
idxer_pca3 = TLIIndexer(prof_sel, head_dim=128, basis=B[3]).to(dev)
idx_sel = idxer_sel.build_block_index(k_real)
idx_pca = idxer_pca3.build_block_index(k_real)
b_sel = (idx_sel["kq_q"][0].numel() + idx_sel["kq_sc"][0].numel() * 8) // Hkv
b_pca = (idx_pca["kq_q"][0].numel() + idx_pca["kq_sc"][0].numel() * 8) // Hkv
assert idx_pca["kq_q"].shape[2] == 16
results["storage"] = {"select_B_per_tok_head": b_sel, "pca_B_per_tok_head": b_pca,
                      "ratio": round(b_sel / b_pca, 3)}
print(f"[2] kq 存储: 选择 {b_sel}B → PCA {b_pca}B /token-head（{b_sel / b_pca:.2f}×）；"
      f"basis 常驻 {B.numel() * 4 / 1e6:.2f} MB")

# ================= 3. 增量 == 全量（语义口径）=================
# 选择路径（gather 无算术）增量==全量逐位；PCA 投影是 GEMV，cuBLAS 对不同
# 批次用不同 kernel → K 维归约顺序不同 → scale 有 fp 尾差（~1e-7 相对），
# 格点偶差 1 步。语义口径：格点一致率 + unpack allclose + select 集合一致。
S0, step = 3000, [64, 1, 200, 337]
idx0 = idxer_pca3.build_block_index(k_real[:S0])
cur = S0
for sn in step:
    idx0 = idxer_pca3.update_block_index(idx0, k_real[cur : cur + sn])
    cur += sn
idx_full = idxer_pca3.build_block_index(k_real[:cur])
grid_eq = (idx0["kq_q"][:cur] == idx_full["kq_q"][:cur]).float().mean().item()
kq_i = kq_unpack(idx0["kq_q"][:cur], idx0["kq_sc"][:cur], idx0["kq_mn"][:cur])
kq_f = kq_unpack(idx_full["kq_q"][:cur], idx_full["kq_sc"][:cur], idx_full["kq_mn"][:cur])
ok_m = torch.equal(idx0["kmin"][: idx0["nblk"]], idx_full["kmin"][: idx_full["nblk"]]) and \
    torch.equal(idx0["kmax"][: idx0["nblk"]], idx_full["kmax"][: idx_full["nblk"]])
ok_close = torch.allclose(kq_i, kq_f, rtol=1e-4, atol=1e-5)
# select 集合一致（真实语义要求）
q_t3 = d["q"].float()[-1:]
sel_i = idxer_pca3.select(idx0, q_t3, cur - 1)
sel_f = idxer_pca3.select(idx_full, q_t3, cur - 1)
ok_sel = all(
    set(sel_i[h].tolist()) == set(sel_f[h].tolist()) for h in range(Hkv)
)
print(f"[3] PCA 增量 vs 全量: 格点一致率 {grid_eq:.6f} / unpack allclose {ok_close} "
      f"/ kmin-max 逐位 {ok_m} / select 集合一致 {ok_sel}（cur={cur}）")
assert grid_eq > 0.999 and ok_close and ok_m and ok_sel

# ================= 4. select_decode_batched（pool flat 索引）vs per-request =================
S_list = [2049, 2500, 5000, S - 64]
nreq = len(S_list)
t_list = [Si - 1 for Si in S_list]
indices = [idxer_pca3.build_block_index(k_real[:Si]) for Si in S_list]
nd2 = idxer_pca3.nd2
S_cap = max(S_list) + 128
NBLK_CAP = (S_cap + prof_sel.block_size - 1) // prof_sel.block_size
pool = {
    "kq_q": torch.zeros(nreq, S_cap, Hkv, nd2, dtype=torch.uint8, device=dev),
    "kq_sc": torch.zeros(nreq, S_cap, Hkv, device=dev),
    "kq_mn": torch.zeros(nreq, S_cap, Hkv, device=dev),
    "kmin": torch.zeros(nreq, NBLK_CAP, Hkv, prof_sel.coarse_dim, device=dev),
    "kmax": torch.zeros(nreq, NBLK_CAP, Hkv, prof_sel.coarse_dim, device=dev),
}
for r, (Si, idx) in enumerate(zip(S_list, indices)):
    pool["kq_q"][r, :Si] = idx["kq_q"]
    pool["kq_sc"][r, :Si] = idx["kq_sc"]
    pool["kq_mn"][r, :Si] = idx["kq_mn"]
    pool["kmin"][r, : idx["nblk"]] = idx["kmin"]
    pool["kmax"][r, : idx["nblk"]] = idx["kmax"]
qpos = d["qpos"].cuda()
q_rows = []
for t in t_list:
    in_range = torch.nonzero(qpos <= t).squeeze(1)
    q_rows.append(d["q"].float()[int(in_range[-1])])
q_b = torch.stack(q_rows)  # [n, H, D]
sel_b = idxer_pca3.select_decode_batched(pool, torch.arange(nreq, device=dev), S_list, q_b)
mism = 0
for r, (Si, idx, t) in enumerate(zip(S_list, indices, t_list)):
    sel_e = idxer_pca3.select(idx, q_b[r : r + 1], t)
    for h in range(Hkv):
        a = set(sel_e[h].tolist())
        bset = {x for x in sel_b[r, h].tolist() if x < Si}
        if a != bset:
            mism += 1
            print(f"  row{r}(S={Si}) h{h} diff {len(a ^ bset)}")
    real = (sel_b[r] < Si).sum(dim=-1)
    assert torch.all(real == prof_sel.token_budget), f"row{r} 实选 {real.tolist()}"
print(f"[4] pool 批量（nd2={nd2} flat 索引）vs per-request: {nreq * Hkv - mism}/{nreq * Hkv} head 集合一致")
assert mism == 0

# ================= 5. select 延迟（L2 打分维 32→16）=================
def bench_select(idxer, k, q, t, reps=10):
    index = idxer.build_block_index(k)
    for _ in range(3):
        idxer.select(index, q, t)
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    ts = []
    for _ in range(reps):
        st.record()
        idxer.select(index, q, t)
        en.record()
        torch.cuda.synchronize()
        ts.append(st.elapsed_time(en))
    return sorted(ts)[reps // 2]

q1 = d["q"].float()[-1:]
t_tr = S - 1
lat_sel = bench_select(idxer_sel, k_real, q1, t_tr)
lat_pca = bench_select(idxer_pca3, k_real, q1, t_tr)
# 合成 131K microbench（已标注：仅延迟口径，非质量）
g = torch.Generator(device=dev).manual_seed(7)
k131 = (torch.randn(131072, Hkv, D, generator=g, device=dev) * 0.3).float()
q131 = (torch.randn(1, q1.shape[1], D, generator=g, device=dev) * 0.3).float()
lat_sel131 = bench_select(idxer_sel, k131, q131, 131071, reps=5)
lat_pca131 = bench_select(idxer_pca3, k131, q131, 131071, reps=5)
results["latency"].update({
    "trace_ms_sel": round(lat_sel, 3), "trace_ms_pca": round(lat_pca, 3),
    "trace_speedup": round(lat_sel / lat_pca, 3),
    "syn131k_ms_sel": round(lat_sel131, 3), "syn131k_ms_pca": round(lat_pca131, 3),
    "syn131k_speedup": round(lat_sel131 / lat_pca131, 3),
})
print(f"[5] select 延迟: trace S={S} {lat_sel:.3f}→{lat_pca:.3f} ms（{lat_sel / lat_pca:.2f}×）；"
      f"合成 131K {lat_sel131:.3f}→{lat_pca131:.3f} ms（{lat_sel131 / lat_pca131:.2f}×）")

json.dump(results, open("/home/wangyuanshuo02/sglang/tli_m9_results.json", "w"), indent=1)
print("\nALL PASS — saved tli_m9_results.json")
