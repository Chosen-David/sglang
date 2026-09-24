# M5 CUDA graph decode 对拍测试（GPU，真实 trace 数据）：
#   A. 图内路径语义（不真捕获）：out_graph 稳态维护 + forward_decode graph
#      分支（Python 执行）vs eager 路径多步逐位对拍（含首步全量重建 /
#      常规增量 / S 跳变重建 / pool 内容逐位一致）；每步静态 buffer 均带
#      pad 行（req=0/seq_len=1 → 哨兵行 0 / slot 0），对拍只看真实行
#   B. 真 CUDA graph 捕获：torch.cuda.graph 录制 forward_decode 图内路径
#      （capture 区域内出现 .tolist()/.item()/H2D 会直接报错——可录制性
#      的硬验证）+ 多步 replay 与 eager 对拍
#   D. 短序列行为：S ≤ token_budget 数学等价 dense；1024 < S ≤
#      dense_threshold 的行统一稀疏 mass coverage 不降（已知偏差口径）
# 数据：/tmp/trace/qwen3-8b/lb_hotpotqa_0/layer03.pt（真实 Qwen3-8B 权重 trace）
import sys

sys.path.insert(0, "/home/wangyuanshuo02/sglang/python")
import torch

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

POOL_N = S + 512
# slot 0 保留为牺牲位（框架 pad 行 out_cache_loc=0 的写入目标）——
# 真实引擎的 KV 分配器同样保留 slot 0，此处对齐该语义
perm = torch.randperm(POOL_N - 1, device=dev) + 1  # 全部 ≥ 1
k_pool = torch.zeros(POOL_N, Hkv, D, device=dev)
v_pool = torch.zeros(POOL_N, Hkv, D, device=dev)
k_pool[perm[: S + 64]] = torch.cat(
    [k_real, k_real[-1:] + 0.05 * torch.randn(64, Hkv, D, device=dev)]
)
v_pool[perm[: S + 64]] = torch.cat(
    [v_real, v_real[-1:] + 0.05 * torch.randn(64, Hkv, D, device=dev)]
)


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
    num_hidden_layers = 4  # 单测省显存：init_cuda_graph_state 建池层数


class FakeRunner:
    def __init__(self, req_to_token):
        self.model_config = FakeMC()
        self.device = dev
        self.token_to_kv_pool = FakePool(k_pool, v_pool)
        self.req_to_token_pool = FakeRTP(req_to_token)


class FakeLayer:
    layer_id = 3
    scaling = D**-0.5


class FakeGraphFB:
    """capture/replay 共享的静态 buffer 持有者（模拟 DecodeInputBuffers
    + build_replay_fb_view）：seq_lens/seq_lens_cpu/req_pool_indices/
    out_cache_loc 是静态张量，replay 前 fill，图内读同一引用。
    fill 自动 pad：未提供的行 = runner 的 ZERO/FILL_SENTINEL 策略
    （req=0 / seq_len=1 / out_cache_loc=0）。"""

    def __init__(self, bs):
        self.batch_size = bs
        self.seq_lens = torch.ones(bs, dtype=torch.int32, device=dev)  # fill=1
        self.seq_lens_cpu = torch.ones(bs, dtype=torch.int64)  # cpu 镜像
        self.req_pool_indices = torch.zeros(bs, dtype=torch.long, device=dev)
        self.out_cache_loc = torch.zeros(bs, dtype=torch.long, device=dev)
        self.num_padding = 0
        self.spec_info = None

    def fill(self, req_ids, seq_lens):
        bs = self.batch_size
        real = len(seq_lens)
        self.seq_lens.fill_(1)
        self.seq_lens_cpu.fill_(1)
        self.req_pool_indices.zero_()
        self.out_cache_loc.zero_()
        for i in range(real):
            self.seq_lens[i] = seq_lens[i]
            self.seq_lens_cpu[i] = seq_lens[i]
            self.req_pool_indices[i] = req_ids[i]
            self.out_cache_loc[i] = req_to_token[req_ids[i], seq_lens[i] - 1]
        self.num_padding = bs - real


max_ctx = S + 256
req_to_token = torch.zeros(16, max_ctx, dtype=torch.long, device=dev)

# 3 个不同长度的稀疏请求（全部 > dense_threshold，严格对拍前提）
S_list = [2500, 5000, S - 20]
n = len(S_list)
req_ids = [2, 4, 6]
for r_i, S_i in enumerate(S_list):
    req_to_token[req_ids[r_i], :S_i] = perm[:S_i]

q_rows = []
for t0 in [S_i - 1 for S_i in S_list]:
    in_range = torch.nonzero(qpos <= t0).squeeze(1)
    q_rows.append(q_real[int(in_range[-1])])
q_b = torch.stack(q_rows)  # [n, H, D]


N_STEPS = 8
JUMP_STEP, JUMP_N = 4, 5  # 第 4 步请求 1 发生 S 跳变 +5（chunked/spec 模拟）

# 预生成每步 q（两侧共用同一张量——若每次现生成，randn_like 噪声不同会
# 引入伪差异，m5 首版测试即栽在这：4.85e-03 假不一致）
_Q_STEPS = [q_b.clone()] + [q_b + 0.01 * torch.randn_like(q_b) for _ in range(N_STEPS - 1)]


def q_step(step):
    return _Q_STEPS[step]


# ---- 两侧共用的当前 token 分配（每步调用前执行；模拟 scheduler 写
# req_to_token + KV 落池由 backend save_kv_cache 完成）----
def alloc_cur_tokens(seq_lens):
    for i, s in enumerate(seq_lens):
        req_to_token[req_ids[i], s - 1] = perm[s - 1]


def cur_kv(seq_lens, bs_cap):
    k_full = torch.zeros(bs_cap, Hkv, D, device=dev)
    v_full = torch.zeros(bs_cap, Hkv, D, device=dev)
    for i, s in enumerate(seq_lens):
        k_full[i] = k_pool[req_to_token[req_ids[i], s - 1]]
        v_full[i] = v_pool[req_to_token[req_ids[i], s - 1]]
    return k_full, v_full


# ================= A. 图内路径语义 vs eager（Python 执行） =================
be_e = TLISparseAttnBackend(runner=FakeRunner(req_to_token))
be_g = TLISparseAttnBackend(runner=FakeRunner(req_to_token))
MAX_BS = 8
be_g.init_cuda_graph_state(MAX_BS, MAX_BS)
for pool_l in be_g.index_pools.values():
    assert 0 not in pool_l["free"] and 0 not in pool_l["row_of"].values()
print("[A0] init_cuda_graph_state: 哨兵行 0 保留；池预扩 S_cap =",
      be_g.index_pools[3]["S_cap"], "R_cap =", be_g.index_pools[3]["R_cap"])

fb_g = FakeGraphFB(MAX_BS)


def eager_step(be, step, seq_lens):
    fb = type("FB", (), {})()
    fb.seq_lens = torch.tensor(seq_lens, device=dev)
    fb.req_pool_indices = torch.tensor(req_ids, device=dev)
    fb.out_cache_loc = torch.stack(
        [req_to_token[req_ids[i], s - 1] for i, s in enumerate(seq_lens)]
    )
    k_cur, v_cur = cur_kv(seq_lens, n)
    be.init_forward_metadata(fb)
    o = be.forward_decode(
        q_step(step).reshape(n, H * D), k_cur.reshape(n, Hkv * D),
        v_cur.reshape(n, Hkv * D), FakeLayer(), fb, save_kv_cache=True,
    )
    return o.reshape(n, H, D)


def graph_step(be, fb, step, seq_lens):
    """模拟 runner replay：fill 静态 buffer → out_graph（host 稳态维护）→
    图内路径（阶段 A 直接 Python 执行；阶段 B 由 graph replay 执行）。
    q/k/v 按捕获形状 MAX_BS 全宽传入（pad 行零值，输出垃圾但有限）。"""
    fb.fill(req_ids, list(seq_lens))
    q_full = torch.zeros(MAX_BS, H, D, device=dev)
    q_full[:n] = q_step(step)
    k_full, v_full = cur_kv(seq_lens, MAX_BS)
    be.init_forward_metadata_out_graph(fb)
    o = be.forward_decode(
        q_full.reshape(MAX_BS, H * D), k_full.reshape(MAX_BS, Hkv * D),
        v_full.reshape(MAX_BS, Hkv * D), FakeLayer(), fb, save_kv_cache=True,
    )
    o = o.reshape(MAX_BS, H, D)
    assert torch.isfinite(o).all(), "图内路径输出含 NaN/Inf（pad 行哨兵语义失效？）"
    return o[:n]


cur = list(S_list)
diffs = []
for step in range(N_STEPS):
    if step == JUMP_STEP:
        cur[1] += JUMP_N  # 跳变：eager 走多 token update，graph 走 out_graph 重建
        req_to_token[req_ids[1], cur[1] - JUMP_N : cur[1]] = perm[cur[1] - JUMP_N : cur[1]]
    alloc_cur_tokens(cur)
    o_e = eager_step(be_e, step, cur)
    o_g = graph_step(be_g, fb_g, step, cur)
    diffs.append((o_e - o_g).abs().max().item())
    cur = [s + 1 for s in cur]
print(f"[A1] 图内路径 vs eager 多步对拍（{N_STEPS} 步含首步重建/增量/跳变/"
      f"每步 {MAX_BS - n} pad 行）: max|diff| = {max(diffs):.2e}")
assert max(diffs) < 1e-4, f"图内路径与 eager 不一致: {max(diffs)}"

# ---- A3. 混跑一致性：同一 backend 实例 eager 步 ↔ graph 步交替 ----
# （真实引擎：短行批被 veto 回退 eager，长行批走 graph——两种路径的
# pool bookkeeping 语义必须无缝互换）
cur = list(S_list)
diffs_mix = []
for step in range(4):
    alloc_cur_tokens(cur)
    if step % 2 == 0:
        # eager 步（模拟 veto 回退：_use_graph_path=False）
        be_g._use_graph_path = False
        fb = type("FB", (), {})()
        fb.seq_lens = torch.tensor(cur, device=dev)
        fb.req_pool_indices = torch.tensor(req_ids, device=dev)
        fb.out_cache_loc = torch.stack(
            [req_to_token[req_ids[i], s - 1] for i, s in enumerate(cur)]
        )
        k_cur, v_cur = cur_kv(cur, n)
        o = be_g.forward_decode(
            q_step(step).reshape(n, H * D), k_cur.reshape(n, Hkv * D),
            v_cur.reshape(n, Hkv * D), FakeLayer(), fb, save_kv_cache=True,
        )
        o_m = o.reshape(n, H, D)
    else:
        o_m = graph_step(be_g, fb_g, step, cur)
    o_e = eager_step(be_e, step, cur)
    diffs_mix.append((o_e - o_m).abs().max().item())
    cur = [s + 1 for s in cur]
print(f"[A3] 混跑（eager↔graph 交替，同一 backend）vs eager 参考: "
      f"max|diff| = {max(diffs_mix):.2e}")
assert max(diffs_mix) < 1e-4, f"混跑不一致: {max(diffs_mix)}"

# pool 内容逐位一致（增量路径 == 重建+增量）
pool_e, pool_g = be_e.index_pools[3], be_g.index_pools[3]
row_e, row_g = pool_e["row_of"][req_ids[1]], pool_g["row_of"][req_ids[1]]
L_now = cur[1] - 1
assert torch.equal(pool_e["kq"][row_e, :L_now], pool_g["kq"][row_g, :L_now]), "pool kq 不一致"
nb = (L_now + 63) // 64
assert torch.equal(pool_e["kmin"][row_e, :nb], pool_g["kmin"][row_g, :nb]), "pool kmin 不一致"
print(f"[A2] pool 行内容（graph 增量 vs eager）逐位一致（S={L_now}，含跳变重建）")

# ================= B. 真 CUDA graph 捕获 + replay =================
be_c = TLISparseAttnBackend(runner=FakeRunner(req_to_token))
BS_CAP = 4
be_c.init_cuda_graph_state(BS_CAP, BS_CAP)
fb_c = FakeGraphFB(BS_CAP)
q_s = torch.zeros(BS_CAP, H, D, device=dev)
k_s = torch.zeros(BS_CAP, Hkv, D, device=dev)
v_s = torch.zeros(BS_CAP, Hkv, D, device=dev)

# capture（dummy：req=0/seq_len=1 → 哨兵行 0）
be_c.init_forward_metadata_out_graph(fb_c, in_capture=True)
assert (be_c._graph_rows_l[3][:BS_CAP] == 0).all(), "capture dummy 应全映射哨兵行 0"
s_warm = torch.cuda.Stream()
s_warm.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s_warm):
    _ = be_c.forward_decode(
        q_s.reshape(BS_CAP, H * D), k_s.reshape(BS_CAP, Hkv * D),
        v_s.reshape(BS_CAP, Hkv * D), FakeLayer(), fb_c, save_kv_cache=True,
    )
torch.cuda.current_stream().wait_stream(s_warm)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    out_g = be_c.forward_decode(
        q_s.reshape(BS_CAP, H * D), k_s.reshape(BS_CAP, Hkv * D),
        v_s.reshape(BS_CAP, Hkv * D), FakeLayer(), fb_c, save_kv_cache=True,
    )  # capture 区域内任何 host 同步（tolist/item/H2D）会在此直接报错
print("[B0] CUDA graph 捕获成功（形状静态化 + 零 host 同步通过）")

# replay 多步（捕获形状 BS_CAP=4，真实 n=3 → 每步 1 pad 行）
cur = list(S_list)
diffs_b = []
for step in range(N_STEPS):
    if step == JUMP_STEP:
        cur[1] += JUMP_N
        req_to_token[req_ids[1], cur[1] - JUMP_N : cur[1]] = perm[cur[1] - JUMP_N : cur[1]]
    alloc_cur_tokens(cur)
    fb_c.fill(req_ids, cur)
    q_s.zero_(); k_s.zero_(); v_s.zero_()
    q_s[:n] = q_step(step)
    k_s[:n] = torch.stack([k_pool[req_to_token[req_ids[i], s - 1]] for i, s in enumerate(cur)])
    v_s[:n] = torch.stack([v_pool[req_to_token[req_ids[i], s - 1]] for i, s in enumerate(cur)])
    be_c.init_forward_metadata_out_graph(fb_c)
    g.replay()
    torch.cuda.synchronize()
    o_c = out_g.reshape(BS_CAP, H, D)[:n]
    assert torch.isfinite(out_g).all(), f"step{step} replay 输出含 NaN/Inf"
    o_e = eager_step(be_e, step, cur)
    diffs_b.append((o_e - o_c).abs().max().item())
    cur = [s + 1 for s in cur]
print(f"[B1] graph replay vs eager（{N_STEPS} 步，捕获形状 {BS_CAP} 含 1 pad 行）: "
      f"max|diff| = {max(diffs_b):.2e}")
assert max(diffs_b) < 1e-4, f"graph replay 与 eager 不一致: {max(diffs_b)}"

# ================= D. 短序列行为（统一稀疏的已知偏差口径） =================
S_short_list = [600, 1500]  # 600 ≤ 1024（数学等价 dense）；1500 ∈ (1024, 2048]
req_ids_s = [8, 9]
for r_i, S_i in enumerate(S_short_list):
    base = 1000 + r_i * 4096
    req_to_token[req_ids_s[r_i], :S_i] = perm[base : base + S_i]
    req_to_token[req_ids_s[r_i], S_i - 1] = perm[base + S_i - 1]
q_s2 = torch.stack([q_real[-1], q_real[-2]])
fb_s = FakeGraphFB(4)
be_s = TLISparseAttnBackend(runner=FakeRunner(req_to_token))
be_s.init_cuda_graph_state(4, 4)
cur_s = list(S_short_list)
fb_s.fill(req_ids_s, cur_s)
q_full = torch.zeros(4, H, D, device=dev); q_full[:2] = q_s2
k_full = torch.zeros(4, Hkv, D, device=dev)
v_full = torch.zeros(4, Hkv, D, device=dev)
for i, s in enumerate(cur_s):
    base = 1000 + i * 4096
    k_full[i] = k_pool[perm[base + s - 1]]
    v_full[i] = v_pool[perm[base + s - 1]]
be_s.init_forward_metadata_out_graph(fb_s)
o_s = be_s.forward_decode(
    q_full.reshape(4, H * D), k_full.reshape(4, Hkv * D), v_full.reshape(4, Hkv * D),
    FakeLayer(), fb_s, save_kv_cache=True,
).reshape(4, H, D)[:2]

# D 段只验证图内路径的短行行为；S=1500（token_budget < S ≤
# dense_threshold）的批已被 veto_cuda_graph 拦回 eager dense 路径，不进图
for r_i, S_i in enumerate(S_short_list):
    t = S_i - 1
    qg = q_s2[r_i].reshape(1, Hkv, G, D)
    s_attn = torch.einsum("bhgd,chd->bhgc", qg, k_real[: t + 1]).sum(-2) * (D**-0.5)
    p_ = torch.softmax(s_attn, dim=-1)[0]
    sel_r = be_s.indexers[3].select_decode_batched(
        be_s.index_pools[3]["kq"], be_s.index_pools[3]["kmin"], be_s.index_pools[3]["kmax"],
        be_s._graph_rows_l[3], fb_s.seq_lens.to(torch.long), q_full.float(),
    )[r_i]
    cov = (
        p_.gather(1, sel_r.clamp(max=t)).masked_fill(sel_r >= S_i, 0).sum().item()
        / p_.sum().item()
    )
    veto = be_s.veto_cuda_graph(fb_s)
    tag = "等价 dense" if S_i <= 1024 else "短行（veto 回退 eager）"
    print(f"[D] S={S_i}（{tag}）: 图内路径 mass coverage = {cov:.5f}, veto_cuda_graph = {veto}")
    if S_i <= 1024:
        assert cov > 0.999, f"S={S_i} 短序列质量异常: {cov}"
    else:
        # 短行批必须被 veto（回退 eager dense 路径），图内路径质量损失不进主路径
        assert veto is True, f"S={S_i} 短行批未被 veto: 图内路径会以 4bit 近端排名损失质量"

print("\nM5 CUDA graph 验证 ALL PASS")
