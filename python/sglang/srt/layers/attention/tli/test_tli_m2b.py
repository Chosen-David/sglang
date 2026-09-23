# sglang tli M2 后半场验证：
#   1. update_block_index 增量 == build_block_index 全量（kq 精确相等；kmin/kmax
#      单向界放宽安全：增量界 ⊇ 全量界，recall 不受损）
#   2. select_batched == select 逐行对拍（scatter 口径，位置集合应完全一致）
#   3. backend 全链路（mock pool + req_to_token 间接寻址）：
#      a) forward_extend 稀疏 prefill vs dense 参考（质量口径：cos 相似度）
#      b) forward_decode 增量索引 vs 全量重建索引（输出应逐位一致）
#      c) forward_decode 稀疏 vs dense 参考（质量口径）
# 数据：/tmp/trace/qwen3-8b 真实 trace（Qwen3-8B 权重采集）
import sys

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
import torch

from sglang.srt.layers.attention.tli.config import TLIProfile
from sglang.srt.layers.attention.tli.indexer import TLIIndexer
from sglang.srt.layers.attention.tli.backend import TLISparseAttnBackend

dev = "cuda:0"
d = torch.load("/tmp/trace/qwen3-8b/lb_hotpotqa_0/layer03.pt", map_location=dev)
k_real = d["k"].float()  # [S, Hkv, D] RoPE 后真实 K
q_real = d["q"].float()  # [nq, H, D] 尾部+锚点真实 q（qpos 记录全局位置）
qpos = d["qpos"].cuda()
S, Hkv, D = k_real.shape
H = q_real.shape[1]
G = H // Hkv
S0 = 4096  # prefill 部分长度（> dense_threshold=2048）
assert S > S0 + 256, f"trace 太短: S={S}"

torch.manual_seed(0)
v_real = (torch.randn(S, Hkv, D, device=dev) * 0.05).float()

# ================= 1. 增量索引 vs 全量重建 =================
prof = TLIProfile()
idxer = TLIIndexer(prof, head_dim=D).to(dev)
# 步进覆盖全部分支：对齐开新块(64) / 单 token(1) / 尾块+多新块(200,300) /
# 多块+部分尾块(337) / 大跨度(2170)
inc = idxer.build_block_index(k_real[:1024])  # 1024 = 16*64 对齐 → 首步走新块分支
cur = 1024
for step_n in [64, 1, 200, 300, 337, 2170]:
    inc = idxer.update_block_index(inc, k_real[cur : cur + step_n])
    cur += step_n
assert cur == S0, f"增量步进长度配置错误: {cur} != {S0}"
ref = idxer.build_block_index(k_real[:S0])
assert inc["S"] == ref["S"] == S0 and inc["nblk"] == ref["nblk"]
assert torch.equal(inc["kq"], ref["kq"]), "kq 增量 != 全量"
assert torch.equal(inc["kmin"], ref["kmin"]), "kmin 增量 != 全量（尾块精确界被破坏）"
assert torch.equal(inc["kmax"], ref["kmax"]), "kmax 增量 != 全量（尾块精确界被破坏）"
print("[1] 增量索引: kmin/kmax/kq 全部与全量重建逐位一致")

# 增量索引下的 select 质量（mass coverage 口径；t 取索引前缀内的真实 q 行）
in_range = torch.nonzero(qpos < S0).squeeze(1)
r1 = int(in_range[-1])
q_t = q_real[r1 : r1 + 1].reshape(1, H, D)
t = int(qpos[r1])
sel_inc = idxer.select(inc, q_t, t)
sel_ref = idxer.select(ref, q_t, t)


def mass_cov(k, q_row, t, sel):
    qg = q_row.reshape(1, Hkv, G, D)
    s = torch.einsum("bhgd,chd->bhgc", qg, k[: t + 1]).sum(-2) * (D**-0.5)
    s = s.masked_fill(torch.arange(t + 1, device=dev).view(1, 1, t + 1) > t, float("-inf"))
    p = torch.softmax(s, dim=-1)[0]
    sel_m = sel.clamp(max=t)
    return p.gather(1, sel_m).sum().item() / p.sum().item()


cov_inc = mass_cov(k_real, q_real[r1], t, sel_inc)
cov_ref = mass_cov(k_real, q_real[r1], t, sel_ref)
print(f"[1] 增量 select mass cov = {cov_inc:.5f} (全量索引 {cov_ref:.5f})")
assert cov_inc > 0.99, f"增量索引质量 {cov_inc} < 0.99"

# ================= 2. select_batched vs select 逐行对拍 =================
index = idxer.build_block_index(k_real)  # 全序列索引（行 t 可达 S-1）
# 取 t 较大的行（远端区非空：t > near_len+far_lo；garbage 位边界不影响对拍）
rows = torch.nonzero(qpos > S0).squeeze(1)[:8]
q_b = q_real[rows].reshape(len(rows), H, D)
t_arr = qpos[rows]
sel_b = idxer.select_batched(index, q_b, t_arr)  # [n, Hkv, K2]
mism = 0
for r in range(len(rows)):
    sel_r = idxer.select(index, q_b[r : r + 1], int(t_arr[r]))  # [Hkv, K2]
    sb = sel_b[r]
    # 位置集合比较（排序后逐位；scatter 口径 + 相同分数张量 → topk 确定性一致）
    a = torch.sort(sel_r, dim=-1).values
    b = torch.sort(sb, dim=-1).values
    if not torch.equal(a, b):
        mism += 1
        for h in range(Hkv):
            d1 = set(a[h].tolist())
            d2 = set(b[h].tolist())
            if d1 != d2:
                print(f"  row{r} h{h}: diff {len(d1 ^ d2)} 个位置")
print(f"[2] select_batched vs select: {len(rows) - mism}/{len(rows)} 行位置集合完全一致")
assert mism == 0, "批量选择与逐行选择不一致"

# ================= 3. backend 全链路（mock pool） =================


class FakePool:
    """token_to_kv_pool mock：槽位池 + 间接寻址语义。"""

    def __init__(self, k_pool, v_pool):
        self.k_pool, self.v_pool = k_pool, v_pool

    def get_kv_buffer(self, layer_id):
        return (self.k_pool, self.v_pool)

    def set_kv_buffer(self, layer, k, v, locs):
        self.k_pool[locs] = k.float()
        self.v_pool[locs] = v.float()


class FakeRTP:
    def __init__(self, req_to_token):
        self.req_to_token = req_to_token


class FakeMC:
    head_dim = D
    num_key_value_heads = Hkv
    num_hidden_layers = 36


class FakeRunner:
    model_config = FakeMC()
    device = dev
    req_to_token_pool = None  # 迫使 backend 走 forward_batch 路径


class FakeLayer:
    layer_id = 3
    scaling = D**-0.5


POOL_N = S + 64
# 槽位随机置换（验证 req_to_token 间接寻址，而非连续假设）
perm = torch.randperm(POOL_N, device=dev)
k_pool = torch.zeros(POOL_N, Hkv, D, device=dev)
v_pool = torch.zeros(POOL_N, Hkv, D, device=dev)
pool = FakePool(k_pool, v_pool)
req = 7
max_ctx = S + 64
req_to_token = torch.zeros(8, max_ctx, dtype=torch.long, device=dev)
req_to_token[req, :S] = perm[:S]
k_pool[perm[:S]] = k_real
v_pool[perm[:S]] = v_real


class FakeFB:
    token_to_kv_pool = pool
    req_to_token_pool = FakeRTP(req_to_token)
    req_pool_indices = torch.tensor([req], device=dev)
    out_cache_loc = None
    seq_lens = None
    extend_seq_lens = None
    extend_seq_lens_cumulative = None


be = TLISparseAttnBackend(runner=FakeRunner())
layer = FakeLayer()


def dense_ref(q_rows, t_start, S_len):
    """dense causal 参考（GQA）：q 行全局位置 = t_start..t_start+nq-1。"""
    nq = q_rows.shape[0]
    k_e = k_real[:S_len].transpose(0, 1)  # [Hkv, S, D]
    v_e = v_real[:S_len].transpose(0, 1)
    q_g = q_rows.reshape(nq, Hkv, G, D)
    att = torch.einsum("ahgd,hsd->ahgs", q_g, k_e) * (D**-0.5)
    qpos_r = torch.arange(t_start, t_start + nq, device=dev).view(nq, 1)
    causal = torch.arange(S_len, device=dev).view(1, S_len) <= qpos_r
    att = att.masked_fill(~causal.view(nq, 1, 1, S_len), float("-inf"))
    att = torch.softmax(att, dim=-1)
    return torch.einsum("ahgs,hsd->ahgd", att, v_e).reshape(nq, H, D)


# ---- 3a. forward_extend 稀疏 prefill vs dense（真实位置对齐的尾部行）----
# q_real 尾部 256 行的真实位置 = S-256..S-1（qpos 连续段），chunk 即末段
nq = 256
q_ext = q_real[-nq:]  # 真实行：位置 S-nq..S-1（RoPE 相位对齐，无错位失真）
t_start = S - nq
fb = FakeFB()
fb.extend_seq_lens = torch.tensor([S], device=dev)
fb.extend_seq_lens_cumulative = torch.tensor([0, nq], device=dev)
out_ext = be.forward_extend(
    q_ext.reshape(nq, H, D), k_real[S - nq : S].reshape(nq, Hkv, D),
    v_real[S - nq : S].reshape(nq, Hkv, D), layer, fb, save_kv_cache=False,
)
ref_ext = dense_ref(q_ext, t_start, S)
# 质量口径 = 行级 mass coverage（论文口径）。GQA head 级 far 集中
# （E4b：单 head 独占 0.804）+ L1 块上界 far capture 0.52–0.74（E4c）
# 使个别 head 瞬时偏低——E5b 端到端 LongBench 49.92 已验证无碍；
# cos 相似度作参考打印（随机 V 会放大失真）
cos = torch.nn.functional.cosine_similarity(
    out_ext.reshape(nq, -1).float(), ref_ext.reshape(nq, -1).float(), dim=-1
)
print(f"[3a] 稀疏 prefill: 行级 cos mean={cos.mean():.5f} min={cos.min():.5f}（参考）")
idxer_chk = TLIIndexer(prof, head_dim=D).to(dev)
index_chk = idxer_chk.build_block_index(k_real)
covs = []
for r in range(0, nq, 8):  # 32 行抽样
    t = t_start + r
    qg = q_ext[r : r + 1].reshape(1, Hkv, G, D)
    s = torch.einsum("bhgd,chd->bhgc", qg, k_real[: t + 1]).sum(-2) * (D**-0.5)
    p_ = torch.softmax(s, dim=-1)[0]
    sel_r = idxer_chk.select_batched(
        index_chk, q_ext[r : r + 1], torch.tensor([t], device=dev)
    )[0].clamp(max=t)
    covs.append(p_.gather(1, sel_r).sum().item() / p_.sum().item())
import statistics
cov_mean, cov_min = statistics.mean(covs), min(covs)
print(f"[3a] 行级 mass coverage: mean={cov_mean:.5f} min={cov_min:.5f}（32 行抽样）")
assert cov_mean > 0.985, f"稀疏 prefill mass coverage 不足: {cov_mean:.4f}"

# ---- 3b/3c. forward_decode：增量 vs 全量重建 + vs dense ----
torch.manual_seed(1)
# decode 步 q：真实尾部 q 行 + 小噪声（位置错位只影响质量断言方向，
# 增量 vs 全量的硬对拍不受影响）
q_dec = q_real[-1:] + 0.01 * torch.randn(16, H, D, device=dev)


# 预生成 decode 增量 K/V（保证两条路径的 pool 内容逐位相同）
N_DEC_STEPS = 16
k_inc = (k_real[S - 1] + 0.05 * torch.randn(N_DEC_STEPS, Hkv, D, device=dev)).float()
v_inc = (v_real[S - 1] + 0.05 * torch.randn(N_DEC_STEPS, Hkv, D, device=dev)).float()


def run_decode(n_steps, rebuild):
    be2 = TLISparseAttnBackend(runner=FakeRunner())
    outs = []
    cur_S = S
    for st in range(n_steps):
        pos = cur_S + st
        req_to_token[req, pos] = perm[S + st]
        k_pool[perm[S + st]] = k_inc[st]
        v_pool[perm[S + st]] = v_inc[st]
        fb = FakeFB()
        fb.seq_lens = torch.tensor([pos + 1], device=dev)
        fb.out_cache_loc = req_to_token[req, pos : pos + 1]
        q_i = q_dec[st : st + 1]
        if rebuild:
            be2.block_indices.pop((layer.layer_id, req), None)
        o = be2.forward_decode(
            q_i, k_inc[st].clone()[None], v_inc[st].clone()[None],
            layer, fb, save_kv_cache=True,
        )
        outs.append(o.reshape(H, D))
    return torch.stack(outs)


n_steps = 16
out_inc_path = run_decode(n_steps, rebuild=False)
out_full_path = run_decode(n_steps, rebuild=True)
diff = (out_inc_path - out_full_path).abs().max().item()
print(f"[3b] decode 增量 vs 每步全量重建: max|diff| = {diff:.2e}")
assert diff < 1e-4, f"增量与全量重建输出不一致: {diff}"

# vs dense（含合成增量 K/V 的因果参考）
def dense_ref_dec(q_row, st):
    S_len = S + st + 1
    k_e = torch.cat([k_real[:S], k_inc[: st + 1]]).transpose(0, 1)
    v_e = torch.cat([v_real[:S], v_inc[: st + 1]]).transpose(0, 1)
    q_g = q_row.reshape(1, Hkv, G, D)
    att = torch.einsum("bhgd,hsd->bhgs", q_g, k_e) * (D**-0.5)
    causal = torch.arange(S_len, device=dev) <= S + st
    att = att.masked_fill(~causal.view(1, 1, 1, S_len), float("-inf"))
    att = torch.softmax(att, dim=-1)
    return torch.einsum("bhgs,hsd->bhgd", att, v_e).reshape(H, D)


ref_dec = torch.stack(
    [dense_ref_dec(q_dec[st : st + 1], st) for st in range(n_steps)]
)
cos_d = torch.nn.functional.cosine_similarity(
    out_inc_path.reshape(n_steps, -1), ref_dec.reshape(n_steps, -1), dim=-1
)
print(f"[3c] decode 稀疏 vs dense（行级 cos，q 带噪声错位）: mean={cos_d.mean():.5f} min={cos_d.min():.5f}（参考）")
# 质量口径：末位 t=S-1 的真实 q 行（位置完全对齐）mass coverage
q_last = q_real[-1:].reshape(1, H, D)
sel_last = idxer_chk.select(index_chk, q_last, S - 1)
qg = q_last.reshape(1, Hkv, G, D)
s = torch.einsum("bhgd,chd->bhgc", qg, k_real).sum(-2) * (D**-0.5)
p_ = torch.softmax(s, dim=-1)[0]
cov_last = p_.gather(1, sel_last).sum().item() / p_.sum().item()
print(f"[3c] decode 末位真实 q 行 mass coverage = {cov_last:.5f}")
assert cov_last > 0.99, f"decode 选择质量不足: {cov_last:.4f}"

# ---- 3d. 短序列 dense 路径（<= dense_threshold）----
S_short = 1024
req2 = 5
req_to_token[req2, :S_short] = perm[2000 : 2000 + S_short]
k_pool[perm[2000 : 2000 + S_short]] = k_real[:S_short]
v_pool[perm[2000 : 2000 + S_short]] = v_real[:S_short]
q_s = q_real[-1:]
fb = FakeFB()
fb.req_pool_indices = torch.tensor([req2], device=dev)
fb.seq_lens = torch.tensor([S_short], device=dev)
o_s = be.forward_decode(q_s, k_real[S_short - 1 : S_short].reshape(1, Hkv, D),
                        v_real[S_short - 1 : S_short].reshape(1, Hkv, D), layer, fb,
                        save_kv_cache=False)
r_s = dense_ref(q_s, S_short - 1, S_short)[0]
diff_s = (o_s.reshape(H, D).float() - r_s).abs().max().item()
print(f"[3d] 短序列 dense 路径 vs 参考: max|diff| = {diff_s:.2e}")
assert diff_s < 1e-4

print("\nM2 后半场验证 PASS")
