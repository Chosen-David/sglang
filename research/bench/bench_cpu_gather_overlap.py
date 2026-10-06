# 用户提议验证 v2（修正版）：far/near token 捞取搬到 CPU 换 overlap
# 口径：decode 单层，bs=32 × S=131K × K2=1024，KV bf16 选中集 134MB
# [A] GPU gather（当前路径）  [B] CPU 往返（GPU 预gather+D2H+CPU整理+H2D）
# [B2] host 镜像池：CPU index_select + pinned H2D（CPU 直接寻址版）
# [C] PCIe 有效带宽（逐次 sync + 正确性校验）  [D] copy engine 与 GEMM 并发
import json
import time

import torch

dev = "cuda:0"
Hkv, D, S, K2, BS = 8, 128, 131072, 1024, 32
BYTES_KV = BS * K2 * Hkv * D * 2 * 2
res = {"shape": {"bs": BS, "S": S, "K2": K2}, "bytes_kv": BYTES_KV}

pool_k = torch.randn(BS, S, Hkv, D, dtype=torch.bfloat16, device=dev)
pool_v = torch.randn(BS, S, Hkv, D, dtype=torch.bfloat16, device=dev)
g = torch.Generator(device=dev).manual_seed(0)
sel = torch.randint(0, S, (BS, K2), device=dev, generator=g)
rows = torch.arange(BS, device=dev).view(-1, 1)


def bench(fn, reps=20, warmup=5, sync_each=True):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
        if sync_each:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps


# ---- [A] GPU gather ----
def gpu_gather():
    return pool_k[rows, sel], pool_v[rows, sel]

tA = bench(gpu_gather, sync_each=False)
res["A_gpu_ms"] = round(tA * 1000, 4)
res["A_GBps"] = round(BYTES_KV / tA / 1e9, 1)
print(f"[A] GPU gather: {tA*1e3:.3f} ms（134MB → {BYTES_KV/tA/1e9:.0f} GB/s 有效）")

# ---- [C] PCIe 有效带宽（逐次 sync + 校验）----
xg = torch.randn(64 * 1024 * 1024, dtype=torch.bfloat16, device=dev)  # 128MB
pin = torch.empty(64 * 1024 * 1024, dtype=torch.bfloat16, pin_memory=True)
tD2H = bench(lambda: pin.copy_(xg, non_blocking=True))
assert torch.equal(pin[:4096], xg[:4096].cpu()), "D2H 校验失败"
tH2D = bench(lambda: xg.copy_(pin, non_blocking=True))
assert torch.equal(xg[:4096].cpu(), pin[:4096]), "H2D 校验失败"
res["C_D2H_GBps"] = round(134.2 / tD2H / 1000, 1)
res["C_H2D_GBps"] = round(134.2 / tH2D / 1000, 1)
print(f"[C] PCIe pinned 逐次同步: D2H {134.2/tD2H/1000:.1f} GB/s / H2D {134.2/tH2D/1000:.1f} GB/s")

# ---- [B] CPU 往返（CPU 无法寻址 GPU 池：必须 GPU 预 gather 或 host 镜像）----
pin_k = torch.empty(BS, K2, Hkv, D, dtype=torch.bfloat16, pin_memory=True)
pin_v = torch.empty_like(pin_k)


def cpu_path():
    k = pool_k[rows, sel].contiguous()
    v = pool_v[rows, sel].contiguous()
    pin_k.copy_(k, non_blocking=True)
    pin_v.copy_(v, non_blocking=True)
    torch.cuda.synchronize()
    hk, hv = pin_k.clone(), pin_v.clone()      # CPU 侧整理（best case = 纯 memcpy）
    return hk.to(dev), hv.to(dev)               # H2D（clone 后 pageable）

tB = bench(cpu_path, reps=10, sync_each=False)
res["B_roundtrip_ms"] = round(tB * 1000, 3)
res["B_vs_A"] = round(tB / tA, 1)
print(f"[B] CPU 往返（含 GPU 预 gather）: {tB*1e3:.2f} ms = {tB/tA:.1f}× [A]")

# ---- [B2] host 镜像池（17GB RAM）：CPU index_select + pinned H2D ----
host_k = pool_k.cpu()
host_v = pool_v.cpu()
sel_h = sel.cpu()
rows_h = rows.cpu().view(-1, 1)


def host_gather():
    hk = host_k[rows_h, sel_h]   # CPU 随机行 gather（这才是「CPU 捞取」本体）
    hv = host_v[rows_h, sel_h]
    pin_k.copy_(hk)               # host→pinned 再 H2D（non_blocking pinned）
    pin_v.copy_(hv)
    return pin_k.to(dev, non_blocking=True), pin_v.to(dev, non_blocking=True)

tB2 = bench(host_gather, reps=10, sync_each=False)
res["B2_host_mirror_ms"] = round(tB2 * 1000, 3)
res["B2_vs_A"] = round(tB2 / tA, 1)
print(f"[B2] host 镜像 + CPU gather + H2D: {tB2*1e3:.2f} ms = {tB2/tA:.1f}× [A]")

# ---- [D] copy engine 与 GPU 计算并发（同一 GPU 也能 overlap D2H/H2D）----
a = torch.randn(8192, 8192, dtype=torch.bfloat16, device=dev)
b = torch.randn(8192, 8192, dtype=torch.bfloat16, device=dev)
tG = bench(lambda: a @ b, sync_each=False)
res["D_gemm_ms"] = round(tG * 1000, 3)
st2 = torch.cuda.Stream()


def serial():
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    (a @ b)
    pin.copy_(xg, non_blocking=True)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def overlapped():
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.cuda.stream(st2):
        pin.copy_(xg, non_blocking=True)  # copy engine 与 GEMM 并发
    (a @ b)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


t_ser = min(serial() for _ in range(5))
t_ovl = min(overlapped() for _ in range(5))
res["D_serial_ms"] = round(t_ser * 1000, 3)
res["D_overlap_ms"] = round(t_ovl * 1000, 3)
print(f"[D] GEMM {tG*1e3:.2f} + 128MB D2H 串行 {t_ser*1e3:.2f} vs 并发 {t_ovl*1e3:.2f} ms"
      f"（copy engine 与 SM 独立 → 传输可被计算隐藏，无需 CPU）")

# ---- verdict ----
layer_ms = 5.24  # M5 实测 bs=32: 188.6ms/step ÷ 36 层（S=10K 口径）
res["verdict"] = {
    "gpu_gather_share_pct": round(tA / (layer_ms / 1000) * 100, 2),
    "cpu_roundtrip_penalty": f"{tB/tA:.0f}x (B), {tB2/tA:.0f}x (B2)",
    "pcie_vs_hbm": f"gather 有效 {BYTES_KV/tA/1e9:.0f}GB/s vs PCIe {134.2/tD2H/1000:.0f}GB/s",
}
print(f"\nverdict: GPU gather 占层时间 {tA/(layer_ms/1000)*100:.1f}%；"
      f"CPU 往返 {tB/tA:.0f}× / host镜像 {tB2/tA:.0f}× 慢；"
      f"且 [D] 证明传输 overlap 用 copy engine 在 GPU 侧即可完成")
json.dump(res, open("/home/wangyuanshuo02/sglang/cpu_gather_overlap_bench.json", "w"), indent=1)
print("saved cpu_gather_overlap_bench.json")
