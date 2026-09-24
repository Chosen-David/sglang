# M4 批量化 decode 对拍测试（GPU，真实 trace 数据）：
#   1. select_decode_batched vs per-request select（eager / L1+L2 kernel 两口径，
#      位置集合应完全一致——真实数据下 L1 候选池并集覆盖 ~全部块，
#      far/near 池充足，无 -inf 垃圾位分歧）
#   2. 共享 pool 行的增量维护 vs 全量重建（update_block_index on pool 行 view）
#   3. backend 多请求 forward_decode：批量路径 vs per-request 路径（fp32
#      累加顺序噪声级差异）；+ 行生命周期（请求退出 → 行回收 → 复用重建）
#   4. 批量路径 vs dense 参考（质量口径 cos）
# 数据：/tmp/trace/qwen3-8b/lb_hotpotqa_0/layer03.pt（真实 Qwen3-8B 权重 trace）
import sys

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
import os

import torch

from sglang.srt.layers.attention.tli.config import TLIProfile
from sglang.srt.layers.attention.tli.indexer import (
    TLIIndexer, quant4, quant4_pack, kq_unpack,
)
from sglang.srt.layers.attention.tli.backend import TLISparseAttnBackend

dev = "cuda:0"
d = torch.load("/tmp/trace/qwen3-8b/lb_hotpotqa_0/layer03.pt", map_location=dev)
k_real = d["k"].float()  # [S, Hkv, D]
q_real = d["q"].float()  # [nq, H, D]
qpos = d["qpos"].cuda()
S, Hkv, D = k_real.shape
H = q_real.shape[1]
G = H // Hkv
assert S > 5000, f"trace 太短: S={S}"

torch.manual_seed(0)
v_real = (torch.randn(S, Hkv, D, device=dev) * 0.05).float()

prof = TLIProfile()
idxer = TLIIndexer(prof, head_dim=D).to(dev)

# ================= 0. M6 uint8+scale 三张量：逐位重建 + 显存口径 =================
# 核心主张：kq_unpack(*quant4_pack(x)) 与 quant4(x) 逐位一致（格点 0-15 在
# fp32 精确可表，重建与量化走同一 IEEE 运算序列）→ 4bit 存储零数值漂移；
# 每 token-head 显存 nd2*4B(fp32) → nd2*1B(uint8) + 8B(双 scale) = 3.2×
x0 = k_real[:2048][..., idxer.idx2]  # [2048, Hkv, nd2] 真实 trace
g0, sc0, mn0 = quant4_pack(x0)
assert g0.dtype == torch.uint8
assert torch.equal(kq_unpack(g0, sc0, mn0), quant4(x0)), "M6 重建与 quant4 不逐位一致"
b_old, b_new = x0.numel() * 4, g0.numel() + 8 * sc0.numel()
print(f"[0] M6 4bit 存储: 重建与 quant4 逐位一致；"
      f"{b_old // 2048 // Hkv}B → {b_new // 2048 // Hkv}B /token-head（{b_old / b_new:.2f}×）")

# 4 个不同长度的"请求"（覆盖 span=0 边界 / span<far_tokens / 常规 / 长序列；
# 最长取 S-64，给第二步 decode 留出 k_real 索引余量）
S_list = [2049, 2500, 5000, S - 64]
n = len(S_list)
t_list = [S_i - 1 for S_i in S_list]

# ================= 1. 批量 select vs per-request select =================
# per-request 索引（独立 dict）+ 共享 pool（同内容写入）
indices = [idxer.build_block_index(k_real[:S_i]) for S_i in S_list]
S_cap = max(S_list) + 128
NBLK_CAP = (S_cap + prof.block_size - 1) // prof.block_size
nd2 = 2 * prof.delta
pool_kq = torch.zeros(n, S_cap, Hkv, nd2, dtype=torch.uint8, device=dev)
pool_ksc = torch.zeros(n, S_cap, Hkv, device=dev)
pool_kmn = torch.zeros(n, S_cap, Hkv, device=dev)
pool_kmin = torch.zeros(n, NBLK_CAP, Hkv, prof.coarse_dim, device=dev)
pool_kmax = torch.zeros_like(pool_kmin)
for r, (S_i, idx) in enumerate(zip(S_list, indices)):
    pool_kq[r, :S_i] = idx["kq_q"]
    pool_ksc[r, :S_i] = idx["kq_sc"]
    pool_kmn[r, :S_i] = idx["kq_mn"]
    pool_kmin[r, : idx["nblk"]] = idx["kmin"]
    pool_kmax[r, : idx["nblk"]] = idx["kmax"]

q_rows = []
for t in t_list:
    in_range = torch.nonzero(qpos <= t).squeeze(1)
    r_ = int(in_range[-1])
    q_rows.append(q_real[r_])
q_b = torch.stack(q_rows)  # [n, H, D]（每请求取位置 ≤ t 的真实 q 行）

rows_t = torch.arange(n, device=dev)
pool_dict = {
    "kq_q": pool_kq, "kq_sc": pool_ksc, "kq_mn": pool_kmn,
    "kmin": pool_kmin, "kmax": pool_kmax,
}
sel_b = idxer.select_decode_batched(pool_dict, rows_t, S_list, q_b)

mism_eager = mism_kern = 0
for r, (S_i, idx, t) in enumerate(zip(S_list, indices, t_list)):
    sel_e = idxer.select(idx, q_b[r : r + 1], t)  # eager 口径
    sel_k = idxer.select(
        idx, q_b[r : r + 1], t, use_l1_kernel=True, use_l2_kernel=True
    )  # L1+L2 fused kernel 口径（哨兵语义）
    for h in range(Hkv):
        a = set(sel_e[h].tolist())
        b = {x for x in sel_b[r, h].tolist() if x < S_i}  # 剥哨兵
        c = {x for x in sel_k[h].tolist() if x < S_i}
        if a != b:
            mism_eager += 1
            print(f"  row{r}(S={S_i}) h{h} eager diff {len(a ^ b)}")
        if b != c:
            mism_kern += 1
            print(f"  row{r}(S={S_i}) h{h} kernel diff {len(b ^ c)}")
    # 每行每个 head 实选数应恰为 token_budget（池充足时）
    real = (sel_b[r] < S_i).sum(dim=-1)
    assert torch.all(real == prof.token_budget), (
        f"row{r} 实选 {real.tolist()} != budget {prof.token_budget}"
    )
print(f"[1] 批量 select vs eager: {n * Hkv - mism_eager}/{n * Hkv} head 集合一致；"
      f"vs L1+L2 kernel: {n * Hkv - mism_kern}/{n * Hkv}")
assert mism_eager == 0, "批量选择与 eager per-request 不一致"
assert mism_kern == 0, "批量选择与 kernel per-request 不一致"

# ================= 2. pool 行增量维护 vs 全量重建 =================
row = 0
S0 = 3000
idx0 = idxer.build_block_index(k_real[:S0])
pool_kq2 = torch.zeros(1, S_cap, Hkv, nd2, dtype=torch.uint8, device=dev)
pool_ksc2 = torch.zeros(1, S_cap, Hkv, device=dev)
pool_kmn2 = torch.zeros(1, S_cap, Hkv, device=dev)
pool_kmin2 = torch.zeros(1, NBLK_CAP, Hkv, prof.coarse_dim, device=dev)
pool_kmax2 = torch.zeros_like(pool_kmin2)
pool_kq2[0, :S0] = idx0["kq_q"]
pool_ksc2[0, :S0] = idx0["kq_sc"]
pool_kmn2[0, :S0] = idx0["kq_mn"]
pool_kmin2[0, : idx0["nblk"]] = idx0["kmin"]
pool_kmax2[0, : idx0["nblk"]] = idx0["kmax"]
# 步进覆盖：对齐开新块 / 单 token / 尾块+多新块
cur = S0
for step_n in [64, 1, 200, 337]:
    vd = {
        "kmin": pool_kmin2[0], "kmax": pool_kmax2[0],
        "kq_q": pool_kq2[0], "kq_sc": pool_ksc2[0], "kq_mn": pool_kmn2[0],
        "nblk": (cur + prof.block_size - 1) // prof.block_size, "S": cur,
    }
    idxer.update_block_index(vd, k_real[cur : cur + step_n])
    cur += step_n
S_new = cur
ref = idxer.build_block_index(k_real[:S_new])
assert torch.equal(pool_kq2[0, :S_new], ref["kq_q"]), "pool 行增量 kq_q != 全量"
assert torch.equal(pool_ksc2[0, :S_new], ref["kq_sc"]), "pool 行增量 kq_sc != 全量"
assert torch.equal(pool_kmn2[0, :S_new], ref["kq_mn"]), "pool 行增量 kq_mn != 全量"
assert torch.equal(
    pool_kmin2[0, : ref["nblk"]], ref["kmin"]
), "pool 行增量 kmin != 全量"
assert torch.equal(
    pool_kmax2[0, : ref["nblk"]], ref["kmax"]
), "pool 行增量 kmax != 全量"
print(f"[2] pool 行增量维护: S={S0}→{S_new} 与全量重建逐位一致")

# ---- 2b. update_pool_rows_decode 批量增量 vs 逐行 update_block_index ----
# 4 行不同起点（覆盖对齐开新块 / 尾块合并），两份独立 pool 逐位对拍
m = 4
S_starts = [2048, 2050, 3000, 3641]  # 2048 对齐；2050 尾块 2；3000 尾 48；3641 尾 57
pools = []
for side in range(2):
    kq_ = torch.zeros(m, S_cap, Hkv, nd2, dtype=torch.uint8, device=dev)
    ksc_ = torch.zeros(m, S_cap, Hkv, device=dev)
    kmn2_ = torch.zeros(m, S_cap, Hkv, device=dev)
    kmn_ = torch.zeros(m, NBLK_CAP, Hkv, prof.coarse_dim, device=dev)
    kmx_ = torch.zeros_like(kmn_)
    for r, S_st in enumerate(S_starts):
        idx_r = idxer.build_block_index(k_real[:S_st])
        kq_[r, :S_st] = idx_r["kq_q"]
        ksc_[r, :S_st] = idx_r["kq_sc"]
        kmn2_[r, :S_st] = idx_r["kq_mn"]
        kmn_[r, : idx_r["nblk"]] = idx_r["kmin"]
        kmx_[r, : idx_r["nblk"]] = idx_r["kmax"]
    pools.append({"kq_q": kq_, "kq_sc": ksc_, "kq_mn": kmn2_, "kmin": kmn_, "kmax": kmx_})
k_step = k_real[S_starts[0] : S_starts[0] + m].float()  # [m, Hkv, D] 各行新 token
rows_m = torch.arange(m, device=dev)
# 侧 A：逐行 update_block_index（view 口径）
for r, S_st in enumerate(S_starts):
    vd = {
        "kmin": pools[0]["kmin"][r], "kmax": pools[0]["kmax"][r],
        "kq_q": pools[0]["kq_q"][r], "kq_sc": pools[0]["kq_sc"][r],
        "kq_mn": pools[0]["kq_mn"][r],
        "nblk": (S_st + prof.block_size - 1) // prof.block_size, "S": S_st,
    }
    idxer.update_block_index(vd, k_step[r : r + 1])
# 侧 B：批量 update_pool_rows_decode
idxer.update_pool_rows_decode(pools[1], rows_m, S_starts, k_step)
S_ends = [s + 1 for s in S_starts]
for key in ("kq_q", "kq_sc", "kq_mn", "kmin", "kmax"):
    for r, S_e in enumerate(S_ends):
        nb = (S_e + prof.block_size - 1) // prof.block_size
        w = S_e if key.startswith("kq") else nb
        assert torch.equal(pools[0][key][r, :w], pools[1][key][r, :w]), (
            f"批量增量 {key} row{r} != 逐行"
        )
print(f"[2b] update_pool_rows_decode 批量增量: 4 行（对齐/非对齐混合）与逐行逐位一致")

# ================= 3. backend 多请求 forward_decode =================
class FakePool:
    def __init__(self, k_pool, v_pool):
        self.k_pool, self.v_pool = k_pool, v_pool

    def get_kv_buffer(self, layer_id):
        return (self.k_pool, self.v_pool)

    def set_kv_buffer(self, layer, locs, k, v):
        k = k.view(-1, self.k_pool.shape[1], self.k_pool.shape[2])
        v = v.view(-1, self.v_pool.shape[1], self.v_pool.shape[2])
        self.k_pool[locs] = k.float()
        self.v_pool[locs] = v.float()


class FakeRTP:
    def __init__(self, req_to_token):
        self.req_to_token = req_to_token


class FakeMC:
    head_dim = D
    num_key_value_heads = Hkv
    num_hidden_layers = 36


POOL_N = S + 256
perm = torch.randperm(POOL_N, device=dev)
k_pool = torch.zeros(POOL_N, Hkv, D, device=dev)
v_pool = torch.zeros(POOL_N, Hkv, D, device=dev)
k_pool[perm[:S]] = k_real
v_pool[perm[:S]] = v_real
kvp = FakePool(k_pool, v_pool)
NREQ = n
req_ids = [1, 3, 5, 7]
req_to_token = torch.zeros(max(req_ids) + 2, S + 256, dtype=torch.long, device=dev)
for r_i, S_i in enumerate(S_list):
    req_to_token[req_ids[r_i], :S_i] = perm[:S_i]


class FakeRunner:
    def __init__(self):
        self.model_config = FakeMC()
        self.device = dev
        self.token_to_kv_pool = kvp
        self.req_to_token_pool = FakeRTP(req_to_token)


class FakeLayer:
    layer_id = 3
    scaling = D**-0.5


class FakeFB:
    token_to_kv_pool = kvp
    req_to_token_pool = FakeRTP(req_to_token)
    out_cache_loc = None
    seq_lens = None
    req_pool_indices = None


# 每"请求"当前 token 的 K/V 写入池尾（模拟 decode 步：seq_len = S_i + 1 之前
# 先把当前 token 放进 pool——简化：直接用 k_real[S_i-1] 已在池内，S_i 即含当前）
def make_fb(seq_lens):
    fb = FakeFB()
    fb.seq_lens = torch.tensor(seq_lens, device=dev)
    fb.req_pool_indices = torch.tensor(req_ids, device=dev)
    fb.out_cache_loc = torch.stack(
        [req_to_token[req_ids[i], s - 1] for i, s in enumerate(seq_lens)]
    )
    return fb


q_dec = q_b  # [n, H, D]
q_2d = q_dec.reshape(n, H * D)
k_cur = torch.stack([k_real[S_i - 1] for S_i in S_list])
v_cur = torch.stack([v_real[S_i - 1] for S_i in S_list])
fb = make_fb(S_list)


def run_fw(batched):
    be = TLISparseAttnBackend(runner=FakeRunner())
    be.profile.use_batch_select = batched
    be.profile.use_l1_kernel = not batched  # per-request 路径带 kernel（M3-b 形态）
    be.profile.use_l2_kernel = not batched
    o = be.forward_decode(
        q_2d.clone(), k_cur.clone().reshape(n, Hkv * D),
        v_cur.clone().reshape(n, Hkv * D), FakeLayer(), fb, save_kv_cache=True,
    )
    return o.reshape(n, H, D), be


out_batched, be_b = run_fw(True)
out_per, be_p = run_fw(False)
diff = (out_batched - out_per).abs().max().item()
print(f"[3] 多请求 forward_decode 批量 vs per-request(带kernel): max|diff| = {diff:.2e}")
assert diff < 1e-4, f"批量与 per-request 输出不一致: {diff}"
# 增量第二步（批量路径下 pool 行已就绪 → 增量分支）再对拍一轮
S_list2 = [s + 1 for s in S_list]
k_cur2 = torch.stack([k_real[S_i] for S_i in S_list])
v_cur2 = torch.stack([v_real[S_i] for S_i in S_list])
# 为每请求的新 token 分配互异新槽位 perm[S_i]：req_to_token[req, S_i] 未
# 初始化时为 0——4 行 out_cache_loc 全撞 slot 0（重复索引 scatter 获胜行
# 不确定），既污染共享 K/V 池（slot 0 属于另一逻辑位置）又让增量维护读
# 到错误 k_new；perm[S_i] 槽内恰为 k_real[S_i]，set_kv_buffer 写入无副作用
for r_i, S_i in enumerate(S_list):
    req_to_token[req_ids[r_i], S_i] = perm[S_i]
fb2 = make_fb(S_list2)


def run_fw2(be, batched):
    be.profile.use_batch_select = batched
    be.profile.use_l1_kernel = not batched
    be.profile.use_l2_kernel = not batched
    return be.forward_decode(
        q_2d.clone(), k_cur2.clone().reshape(n, Hkv * D),
        v_cur2.clone().reshape(n, Hkv * D), FakeLayer(), fb2, save_kv_cache=True,
    ).reshape(n, H, D)


o2b = run_fw2(be_b, True)
o2p = run_fw2(be_p, False)
diff2 = (o2b - o2p).abs().max().item()
print(f"[3] 第二步（增量索引路径）: max|diff| = {diff2:.2e}")
assert diff2 < 1e-4, f"增量步批量与 per-request 不一致: {diff2}"

# 行生命周期：请求 3/5/7 退出 → 行回收 → 新请求 9 复用首行 → 全量重建
fb3 = FakeFB()
fb3.seq_lens = torch.tensor([S_list[0]], device=dev)
fb3.req_pool_indices = torch.tensor([req_ids[0]], device=dev)
fb3.out_cache_loc = req_to_token[req_ids[0], S_list[0] - 1 : S_list[0]]
be_b.init_forward_metadata(fb3)
pool_l = be_b.index_pools[FakeLayer().layer_id]
freed = [r for r in (1, 3, 5) if r not in pool_l["row_of"].values() and pool_l["S"][r] == -1]
print(f"[3] 行回收: 释放 {len(freed)} 行（请求 3/5/7 退出）")
assert len(freed) >= 2, "行回收失败"

# ================= 4. 批量路径 vs dense 参考（质量口径） =================
def dense_ref(q_row, S_len):
    k_e = k_real[:S_len].transpose(0, 1)
    v_e = v_real[:S_len].transpose(0, 1)
    q_g = q_row.reshape(1, Hkv, G, D)
    att = torch.einsum("bhgd,hsd->bhgs", q_g, k_e) * (D**-0.5)
    att = torch.softmax(att, dim=-1)
    return torch.einsum("bhgs,hsd->bhgd", att, v_e).reshape(H, D)


cos = torch.nn.functional.cosine_similarity(
    out_batched.reshape(n, -1),
    torch.stack([dense_ref(q_dec[r], S_i) for r, S_i in enumerate(S_list)]).reshape(n, -1),
    dim=-1,
)
print(f"[4] 批量稀疏 vs dense: cos mean={cos.mean():.5f} min={cos.min():.5f}（随机 V 放大失真，仅参考）")
# 论文口径 = 行级 mass coverage（M2b 同款；sel_b 即批量路径实际选择——
# [1] 已验证其与 eager/kernel 集合一致，q 行位置与 t 对齐）
covs = []
for r, (S_i, t) in enumerate(zip(S_list, t_list)):
    qg = q_b[r].reshape(1, Hkv, G, D)
    s = torch.einsum("bhgd,chd->bhgc", qg, k_real[: t + 1]).sum(-2) * (D**-0.5)
    p_ = torch.softmax(s, dim=-1)[0]  # [Hkv, t+1]
    sel_r = sel_b[r].clamp(max=t)  # 哨兵 lane 清零（clamp 折叠成 t 会重复计质量）
    covs.append(
        p_.gather(1, sel_r).masked_fill(sel_b[r] >= S_i, 0).sum().item()
        / p_.sum().item()
    )
print(f"[4] 行级 mass coverage: mean={sum(covs) / len(covs):.5f} min={min(covs):.5f}")
# 剩余 mass 口径（剔除强制 token）：sink [0, sink_bs) 与滑窗 [sw_lo, t]
# 恒被选中、与索引器质量无关；总口径被 sink mass（H1 实测 0.37-0.71）
# 稀释至接近饱和，区分度低。竞争区 coverage = 选中 token ∩ 竞争区
# [sink_bs, sw_lo) 的 mass / 竞争区总 mass——索引器真实分辨力的口径
# （E4c 的 L1 块级 far capture 是同思想在 L1/远端子集上的特例）
sink_hi = prof.sink_blocks * prof.block_size
covs_res, res_frac = [], []
for r, (S_i, t) in enumerate(zip(S_list, t_list)):
    qg = q_b[r].reshape(1, Hkv, G, D)
    s = torch.einsum("bhgd,chd->bhgc", qg, k_real[: t + 1]).sum(-2) * (D**-0.5)
    p_ = torch.softmax(s, dim=-1)[0]  # [Hkv, t+1]
    sw_lo = max(0, t - prof.sliding_window + 1)
    contested = (torch.arange(t + 1, device=dev) >= sink_hi) & (
        torch.arange(t + 1, device=dev) < sw_lo
    )
    denom = p_[:, contested].sum().item()
    sel_pos = sel_b[r].clamp(max=t)
    in_c = contested[sel_pos] & (sel_b[r] < S_i)  # 哨兵/垃圾 lane 剔除
    num = p_.gather(1, sel_pos).masked_fill(~in_c, 0).sum().item()
    covs_res.append(num / denom)
    res_frac.append(denom / p_.sum().item())
print(f"[4] 剩余 mass coverage（竞争区 = 总 mass 去除 sink+滑窗）: "
      f"mean={sum(covs_res) / len(covs_res):.5f} min={min(covs_res):.5f}")
print(f"[4] 竞争区占总 mass 比例: mean={sum(res_frac) / len(res_frac):.5f} "
      f"min={min(res_frac):.5f}（总口径 0.99+ 主要由强制位贡献的直接证据）")
# 阈值口径：L03 是 far-heavy 层（E4c：L1 块上界 far capture 0.52-0.74，
# far mass 集中单 head），行级 pooled coverage 固有方差 0.95-1.0
# （eager 同行 exact 同值 0.95121@t=16892 / 0.99952@t=S-1）；批量与
# eager 的语义一致已由 [1] 集合相等保证，此处仅设 sanity 下界
assert min(covs) > 0.94 and sum(covs) / len(covs) > 0.98, f"批量选择质量异常: {covs}"
assert sum(covs_res) / len(covs_res) > 0.9, f"竞争区 coverage 异常低: {covs_res}"

print("ALL PASS")
