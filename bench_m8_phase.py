# M8 第一步：select_decode_batched 阶段分解归因（kernel 级 microbench，合成数据口径——
# 形状/配置与生产一致（n=32, S_cap=131072, Hkv=8, d'=32, K1=128, budget=1024），
# 数值合成（randn），仅用于瓶颈归因；输出与真实 select_decode_batched 对拍保证
# 插桩副本忠实。7 个 phase 逐段 CUDA Event 计时 + 每 phase 字节流量估算。
import sys

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
import torch

from sglang.srt.layers.attention.tli.config import TLIProfile
from sglang.srt.layers.attention.tli.indexer import TLIIndexer

dev = "cuda:0"
torch.manual_seed(0)

# ---- 合成池（生产形状）----
n, R = 32, 32
S_cap, Hkv, D = 131072, 8, 128
prof = TLIProfile()
idxer = TLIIndexer(prof, head_dim=D).to(dev)
nd2, d1, bs = idxer.nd2, prof.coarse_dim, prof.block_size
Hkv_ = Hkv
NBLK_CAP = S_cap // bs  # 2048

pool = {
    "kq_q": torch.randint(0, 16, (R, S_cap, Hkv, nd2), dtype=torch.uint8, device=dev),
    "kq_sc": torch.rand(R, S_cap, Hkv, device=dev) * 0.01,
    "kq_mn": (torch.rand(R, S_cap, Hkv, device=dev) - 0.5) * 0.1,
    "kmin": (torch.rand(R, NBLK_CAP, Hkv, d1, device=dev) - 0.5) * 0.2 - 0.1,
    "kmax": (torch.rand(R, NBLK_CAP, Hkv, d1, device=dev) - 0.5) * 0.2 + 0.1,
}
rows = torch.arange(n, device=dev)
S_list = [S_cap] * n
q = torch.randn(n, Hkv * 4, D, device=dev) * 0.3
H = q.shape[1]
G = H // Hkv

# ---- 对拍：插桩副本 vs 真实函数 ----
ref = idxer.select_decode_batched(pool, rows, S_list, q)

p = prof
device = dev
K1 = min(p.k1_blocks, NBLK_CAP)
kq_pool, kq_sc_pool, kq_mn_pool = pool["kq_q"], pool["kq_sc"], pool["kq_mn"]
kmin_pool, kmax_pool = pool["kmin"], pool["kmax"]
S_t = torch.tensor(S_list, device=device, dtype=torch.long)
t_t = S_t - 1
nblk_t = (S_t + bs - 1) // bs
blk_id = torch.arange(NBLK_CAP, device=device)
blk_end = (blk_id + 1) * bs - 1
d_off = torch.arange(nd2, device=device)
h_off = torch.arange(Hkv, device=device).view(1, Hkv, 1) * nd2
h_off_s = torch.arange(Hkv, device=device)

Tc = min((K1 * Hkv + p.sliding_blocks) * bs, NBLK_CAP * bs, S_cap)  # 静态宽度
PH = ["P1_kminmax_gather", "P2_L1_einsum", "P3_topkK1+onehot", "P4_compact_topkmin",
      "P5_L2_gather+deq+einsum", "P6_misc_mask"]
# P7_partition_topk 的时间 = 每 rep 的 e_end − 最后一个打点（P6_misc_mask）
ev = {k: [] for k in PH}


def run_once(record):
    def mark(name, ev0):
        if record:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            ev[name].append(e)

    e_start = torch.cuda.Event(enable_timing=True)
    e_start.record()
    # P1: L1 输入物化（rows gather）
    kmin_b = kmin_pool[rows]
    kmax_b = kmax_pool[rows]
    mark(PH[0], None)
    # P2: L1 einsum
    qs = q[..., idxer.idx1]
    qg = qs.clamp(min=0).reshape(n, Hkv, G, p.coarse_dim)
    qn = qs.clamp(max=0).reshape(n, Hkv, G, p.coarse_dim)
    sc1 = torch.einsum("ahgd,amhd->ahm", qg, kmax_b) + torch.einsum(
        "ahgd,amhd->ahm", qn, kmin_b)
    mark(PH[1], None)
    # P3: mask + topk K1 + onehot
    valid_blk = (blk_id.view(1, -1) < nblk_t.view(-1, 1)) & (
        blk_end.view(1, -1) <= t_t.view(-1, 1))
    sc1m = sc1.masked_fill(~valid_blk.unsqueeze(1), float("-inf"))
    cand_blk = torch.topk(sc1m, K1, dim=-1).indices
    onehot = torch.zeros(n, NBLK_CAP, dtype=torch.bool, device=device)
    sel_src = torch.gather(valid_blk, 1, cand_blk.reshape(n, -1))
    onehot.scatter_(1, cand_blk.reshape(n, -1), sel_src)
    f_blk = (t_t.view(-1, 1) // bs - torch.arange(p.sliding_blocks, device=device)).clamp(min=0)
    onehot.scatter_(1, f_blk, True)
    mark(PH[2], None)
    # P4: 候选压实（topk-min over S_cap）
    sel_mask = onehot.repeat_interleave(bs, dim=1)[:, :S_cap]
    pos = torch.arange(S_cap, device=device)
    cand = sel_mask & (pos.view(1, S_cap) < S_t.view(-1, 1))
    seq_m = torch.where(cand, pos.view(1, S_cap), torch.full_like(pos, S_cap))
    tok = torch.topk(seq_m, Tc, dim=-1, largest=False).values
    valid = tok < S_cap
    tok_c = tok.clamp(max=S_cap - 1)
    mark(PH[3], None)
    # P5: L2 flat gather + dequant + einsum
    q2 = idxer._q_refine(q).reshape(n, Hkv, G, nd2).sum(2)
    s2 = torch.empty(n, Hkv, Tc, dtype=torch.float32, device=device)
    rows_l = rows.to(torch.long)
    chunk = max(1, (256 << 20) // max(Tc * Hkv * nd2 * 4, 1))
    for r0 in range(0, n, chunk):
        r1 = min(r0 + chunk, n)
        m = r1 - r0
        flat = (rows_l[r0:r1].view(m, 1, 1, 1) * (S_cap * Hkv * nd2)
                + tok_c[r0:r1].view(m, Tc, 1, 1) * (Hkv * nd2)
                + h_off.view(1, Hkv, 1) + d_off.view(1, 1, nd2))
        flat_s = (rows_l[r0:r1].view(m, 1, 1) * (S_cap * Hkv)
                  + tok_c[r0:r1].view(m, Tc, 1) * Hkv + h_off_s.view(1, 1, Hkv))
        grid_c = kq_pool.reshape(-1)[flat.view(-1)].view(m, Tc, Hkv, nd2)
        sc_c = kq_sc_pool.reshape(-1)[flat_s.view(-1)].view(m, Tc, Hkv)
        mn_c = kq_mn_pool.reshape(-1)[flat_s.view(-1)].view(m, Tc, Hkv)
        kq_c = grid_c.float() * sc_c.unsqueeze(-1) + mn_c.unsqueeze(-1)
        s2[r0:r1] = torch.einsum("ahd,athd->aht", q2[r0:r1], kq_c)
    mark(PH[4], None)
    # P6/P7: partition
    causal = tok <= t_t.view(-1, 1)
    SENT = S_cap
    far_lo = p.sink_blocks * bs
    far_hi_t = (t_t + 1 - p.near_len).clamp(min=far_lo)
    sw_lo_t = (t_t - p.sliding_window + 1).clamp(min=0)
    near_floor = p.sliding_window + far_lo
    far_cap = max(0, p.token_budget - near_floor)
    k2_far_t = (far_hi_t - far_lo).clamp(max=min(p.far_tokens, far_cap))
    F_t = (t_t + 1 - sw_lo_t).clamp(min=0)
    k2n_t = (p.token_budget - k2_far_t - F_t).clamp(min=0)
    W_far = min(p.far_tokens, far_cap, Tc)
    W_near = min(p.token_budget, Tc)
    W_forced = min(p.sliding_window, Tc)
    in_far = (tok >= far_lo) & (tok < far_hi_t.view(-1, 1)) & valid & causal
    in_near = valid & causal & ~in_far & (tok < sw_lo_t.view(-1, 1))
    far_sc = s2.masked_fill(~in_far.unsqueeze(1), float("-inf"))
    near_sc = s2.masked_fill(~in_near.unsqueeze(1), float("-inf"))
    tok_e = tok_c.unsqueeze(1).expand(n, Hkv, Tc)
    mark(PH[5], None)  # P6 misc mask
    parts = []
    if W_far > 0:
        i_f = torch.topk(far_sc, W_far, dim=-1).indices
        sc_f = torch.gather(far_sc, 2, i_f)
        sel_f = torch.gather(tok_e, 2, i_f)
        rank_f = torch.arange(W_far, device=device).view(1, 1, -1) < k2_far_t.view(-1, 1, 1)
        keep_f = (sc_f != float("-inf")) & rank_f
        parts.append(torch.where(~keep_f, torch.full_like(sel_f, SENT), sel_f))
    if W_near > 0:
        i_n = torch.topk(near_sc, W_near, dim=-1).indices
        sc_n = torch.gather(near_sc, 2, i_n)
        sel_n = torch.gather(tok_e, 2, i_n)
        rank_n = torch.arange(W_near, device=device).view(1, 1, -1) < k2n_t.view(-1, 1, 1)
        keep_n = (sc_n != float("-inf")) & rank_n
        parts.append(torch.where(~keep_n, torch.full_like(sel_n, SENT), sel_n))
    if W_forced > 0:
        f_pos = sw_lo_t.view(-1, 1) + torch.arange(W_forced, device=device)
        f_pad = torch.arange(W_forced, device=device).view(1, -1) >= F_t.view(-1, 1)
        forced = torch.where(f_pad, torch.full_like(f_pos, SENT), f_pos)
        parts.append(forced.unsqueeze(1).expand(n, Hkv, W_forced))
    out = torch.cat(parts, dim=-1)
    e_end = torch.cuda.Event(enable_timing=True)
    e_end.record()
    return out, e_start, e_end


out, e_start, e_end = run_once(record=False)
torch.cuda.synchronize()
assert torch.equal(out, ref), "插桩副本与真实函数输出不一致！"
print("对拍 PASS：插桩副本逐位一致")

REPS = 20
for _ in range(3):
    run_once(record=False)
torch.cuda.synchronize()
marks = []  # (start_ev, [phase evs], end_ev)
for _ in range(REPS):
    _, es, ee = run_once(record=True)
    marks.append((es, ee))
torch.cuda.synchronize()

# 逐 phase 汇总：重跑一轮，每 rep 统一 start 事件 + 各 phase 打点 + e_end
ev = {k: [] for k in PH}
starts, ends = [], []
for _ in range(REPS):
    e0 = torch.cuda.Event(enable_timing=True)
    e0.record()
    starts.append(e0)
    _, es, ee = run_once(record=True)
    ends.append(ee)
torch.cuda.synchronize()

times = {k: [] for k in PH}
times["P7_partition_topk"] = []
for r in range(REPS):
    prev = starts[r]
    for k in PH:
        cur = ev[k][r]
        times[k].append(prev.elapsed_time(cur))
        prev = cur
    times["P7_partition_topk"].append(prev.elapsed_time(ends[r]))
total = [starts[r].elapsed_time(ends[r]) for r in range(REPS)]
med_total = sorted(total)[REPS // 2]

print(f"\n{'phase':>26s} {'median_ms':>10s} {'pct':>6s}   流量估算")
tr = {
    PH[0]: f"kmin/kmax gather 读+写 {(n*NBLK_CAP*Hkv*d1*4*2*2)/1e6:.0f}MB",
    PH[1]: f"einsum 读 {((n*Hkv*G*d1 + n*NBLK_CAP*Hkv*d1)*4*2)/1e6:.0f}MB 写 {n*Hkv*NBLK_CAP*4/1e6:.0f}MB",
    PH[2]: "topk [32,8,2048] k=128 + scatter",
    PH[3]: f"topk-min [32,131072] k={Tc} + repeat_interleave",
    PH[4]: f"kq gather {n*Tc*Hkv*nd2/1e6:.0f}MB + kq_c 物化 {n*Tc*Hkv*nd2*4/1e6:.0f}MB×2(rw)",
    PH[5]: "mask/misc",
    "P7_partition_topk": "topk far/near + gather + where",
}
allk = PH + ["P7_partition_topk"]
for k in allk:
    m = sorted(times[k])[REPS // 2]
    print(f"{k:>26s} {m:10.3f} {m/med_total*100:5.1f}%   {tr[k]}")
print(f"{'合计(中位)':>26s} {med_total:10.3f}")
print(f"\nTc={Tc}, K1={K1}, n={n}, S_cap={S_cap}")
