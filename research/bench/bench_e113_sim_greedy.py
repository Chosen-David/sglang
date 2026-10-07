# E113：sim_greedy 贪心链 kernel 化 microbench（备用脚本，#150）
# 三个臂：
#   A. reference：HEAD _greedy_cluster_pass（GPU Python 逐步循环）——基线，逐 token 分解
#   B. cudagraph：C1 原型（单 token 步 CUDA Graph 捕获重放，Kcap 预分配 + live mask）
#      —— 即设计文档 §7.2-C1 的可行性验证，落地后可平移进 indexer
#   C. fused：V1 CUDA persistent kernel 占位（TODO，kernel 落地后接入；缺位时 SKIP）
# 附：决策等价对拍（assign 逐位 diff 率 = 设计文档 §3 铁律门 1 的微型版）
# 口径：生产 dd=32（tail32，cmp_ratio=4）；Kcap=T 预分配；sim 注册表值 0.9。
# 数据：默认合成（kernel 级 microbench 允许合成，JSON 标注 synthetic）；
#       有 dump 时可 --dump 接真实 k（协议同 E108 probe）。
# 输出：research/bench/e113_sim_greedy_bench.json
import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "two-level-attention"))

DEV = "cuda:0"


def reference_pass(x, sim, sums, cnt, sq, k_live):
    """HEAD 原样语义（tli_indexer._greedy_cluster_pass 的静态副本，避免 import 拖依赖）。
    x: [T,H,dd]；返回 (sums,cnt,sq,k_live,assign[H,T])。"""
    T, H, dd = x.shape
    K = sums.shape[1]
    dev = x.device
    EPS = 1e-9
    assign = torch.zeros(H, T, dtype=torch.long, device=dev)
    x_n = x.norm(dim=-1)
    ar_h = torch.arange(H, device=dev)
    live_row = torch.arange(K, device=dev).unsqueeze(0)
    for i in range(T):
        xi = x[i]
        xi_n = x_n[i]
        norm = sq.clamp(min=0).sqrt()
        dot = torch.bmm(sums, xi.unsqueeze(-1)).squeeze(-1)
        cos = dot / (norm * xi_n.unsqueeze(-1) + EPS)
        cos = cos.masked_fill(live_row >= k_live.unsqueeze(-1), float("-inf"))
        a = cos.argmax(-1)
        m = cos.gather(1, a.unsqueeze(-1)).squeeze(-1)
        upd = m >= sim
        w = upd.float()
        dot_a = dot.gather(1, a.unsqueeze(-1)).squeeze(-1)
        flat_upd = ar_h * K + a
        sums.view(-1, dd).index_add_(0, flat_upd, xi * w.unsqueeze(-1))
        cnt.view(-1).index_add_(0, flat_upd, w)
        sq.view(-1).index_add_(0, flat_upd, (2 * dot_a + xi_n * xi_n) * w)
        nw = (~upd).float()
        flat_new = ar_h * K + k_live
        sums.view(-1, dd).index_add_(0, flat_new, xi * nw.unsqueeze(-1))
        cnt.view(-1).index_add_(0, flat_new, nw)
        sq.view(-1).index_add_(0, flat_new, xi_n * xi_n * nw)
        assign[:, i] = torch.where(upd, a, k_live)
        k_live = k_live + (~upd).long()
    return sums, cnt, sq, k_live, assign


def cudagraph_pass(x, sim, sums, cnt, sq, k_live, unroll=1):
    """C1 原型：固定 shape 单 token 步 + CUDA Graph 重放。
    语义与 reference 完全同序同 op（仅把 Python 循环换成 graph 重放 + device 索引），
    状态 sum/cnt/sq/k_live 全 static。Kcap = sums.shape[1]（须 >= T，预分配）。"""
    T, H, dd = x.shape
    K = sums.shape[1]
    assert K >= T, "C1 要求 Kcap >= T（预分配），无运行时扩容"
    dev = x.device
    EPS = 1e-9
    T_pad = (T + unroll - 1) // unroll * unroll  # unroll>1 时补零 token（见 main 对拍口径）

    # static buffers
    s_sums = sums.clone()
    s_cnt = cnt.clone()
    s_sq = sq.clone()
    s_klive = k_live.clone()
    assign = torch.zeros(H, T, dtype=torch.long, device=dev)

    # 语义等价关键：xi_n 必须与参考实现同源——参考是 x.norm(dim=-1) 全量 [T,H] 一次算好，
    # 归约树序由 [T,H,dd] 形状决定；图内逐 token [H,dd] 现算会产生不同的归约次序，
    # fp 边界翻转级联放大（实测 1.6% assign diff 的根因）→ 预计算后按 pos 索引。
    x_n = x.norm(dim=-1)                                       # [T,H]
    x_n_flat = x_n.reshape(T, H).contiguous()
    # x_flat [T, H*dd]，token i 的行由 device 索引 pos 取（graph 内安全）
    x_flat = x.reshape(T, H * dd).contiguous()
    if T_pad > T:
        # unroll 补齐：零 token（norm=0 → cos=0 < sim → 恒新建空簇，不改 sums 实簇行；
        # k_live 每头膨胀 T_pad-T，对拍时扣除——见 main 的 klive_match 计算）
        pad = T_pad - T
        x_flat = torch.cat([x_flat, x_flat.new_zeros(pad, H * dd)])
        x_n_flat = torch.cat([x_n_flat, x_n_flat.new_zeros(pad, H)])
    pos = torch.zeros(1, dtype=torch.long, device=dev)
    # 常量索引 buffer 提到图外（静态），减图内 kernel 数
    live_row = torch.arange(K, device=dev).unsqueeze(0)
    ar_h = torch.arange(H, device=dev)

    def one_step():
        xi = x_flat.index_select(0, pos).view(H, dd)          # 动态索引 → static buf
        xi_n = x_n_flat.index_select(0, pos).view(H)           # [H]，与参考同源
        norm = s_sq.clamp(min=0).sqrt()                        # [H,K]
        dot = torch.bmm(s_sums, xi.unsqueeze(-1)).squeeze(-1)  # [H,K]
        cos = dot / (norm * xi_n.unsqueeze(-1) + EPS)
        cos = cos.masked_fill(live_row >= s_klive.unsqueeze(-1), float("-inf"))
        a = cos.argmax(-1)
        m = cos.gather(1, a.unsqueeze(-1)).squeeze(-1)
        upd = m >= sim
        w = upd.float()
        dot_a = dot.gather(1, a.unsqueeze(-1)).squeeze(-1)
        flat_upd = ar_h * K + a
        s_sums.view(-1, dd).index_add_(0, flat_upd, xi * w.unsqueeze(-1))
        s_cnt.view(-1).index_add_(0, flat_upd, w)
        s_sq.view(-1).index_add_(0, flat_upd, (2 * dot_a + xi_n * xi_n) * w)
        nw = (~upd).float()
        flat_new = ar_h * K + s_klive
        s_sums.view(-1, dd).index_add_(0, flat_new, xi * nw.unsqueeze(-1))
        s_cnt.view(-1).index_add_(0, flat_new, nw)
        s_sq.view(-1).index_add_(0, flat_new, xi_n * xi_n * nw)
        chosen = torch.where(upd, a, s_klive)                  # [H]
        assign.index_copy_(1, pos, chosen.unsqueeze(1))        # assign[:, pos] = chosen
        s_klive.add_((~upd).long())
        pos.add_(1)

    # 捕获前侧流预热：cublas handle / kernel 首次创建不能发生在 graph capture 内
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        one_step()
    torch.cuda.current_stream().wait_stream(s)

    # capture：捕获会执行一步并污染 static 状态 + 推进 pos → 捕获后全部复位再 replay。
    # unroll=U 时单图内串 U 个 token 步，把每图固定重放开销摊薄 U 倍（C1b）
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(unroll):
            one_step()
    s_sums.copy_(sums)
    s_cnt.copy_(cnt)
    s_sq.copy_(sq)
    s_klive.copy_(k_live)
    pos.zero_()
    assign.zero_()

    T_pad2 = (T + unroll - 1) // unroll * unroll
    n_replay = T_pad2 // unroll
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_replay):
        g.replay()
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    return s_sums, s_cnt, s_sq, s_klive, assign[:, :T], wall


def cpu_pass(x, sim, sums, cnt, sq, k_live):
    """C2 对照臂：CPU 单线程（torch.set_num_threads(1)）跑同语义链。
    决定性测量：CPU 每 token 开销 vs GPU 各臂——数据采集期 CPU 化的可行性依据。"""
    T, H, dd = x.shape
    K = sums.shape[1]
    dev = x.device
    EPS = 1e-9
    assign = torch.zeros(H, T, dtype=torch.long, device=dev)
    x_n = x.norm(dim=-1)
    ar_h = torch.arange(H, device=dev)
    live_row = torch.arange(K, device=dev).unsqueeze(0)
    for i in range(T):
        xi = x[i]
        xi_n = x_n[i]
        norm = sq.clamp(min=0).sqrt()
        dot = torch.bmm(sums, xi.unsqueeze(-1)).squeeze(-1)
        cos = dot / (norm * xi_n.unsqueeze(-1) + EPS)
        cos = cos.masked_fill(live_row >= k_live.unsqueeze(-1), float("-inf"))
        a = cos.argmax(-1)
        m = cos.gather(1, a.unsqueeze(-1)).squeeze(-1)
        upd = m >= sim
        w = upd.float()
        dot_a = dot.gather(1, a.unsqueeze(-1)).squeeze(-1)
        flat_upd = ar_h * K + a
        sums.view(-1, dd).index_add_(0, flat_upd, xi * w.unsqueeze(-1))
        cnt.view(-1).index_add_(0, flat_upd, w)
        sq.view(-1).index_add_(0, flat_upd, (2 * dot_a + xi_n * xi_n) * w)
        nw = (~upd).float()
        flat_new = ar_h * K + k_live
        sums.view(-1, dd).index_add_(0, flat_new, xi * nw.unsqueeze(-1))
        cnt.view(-1).index_add_(0, flat_new, nw)
        sq.view(-1).index_add_(0, flat_new, xi_n * xi_n * nw)
        assign[:, i] = torch.where(upd, a, k_live)
        k_live = k_live + (~upd).long()
    return sums, cnt, sq, k_live, assign


def make_inputs(T, H, dd, seed=0, dump_path=None):
    if dump_path:
        # TODO：接 /tmp/trace/qwen3-8b dump（协议常量同 E108 probe）
        raise NotImplementedError("--dump 接线待真实 kernel 联调时补")
    g = torch.Generator(device=DEV).manual_seed(seed)
    # 合成：token 有簇结构（每 c_grain 个 token 共享一个基向量 + 噪声），
    # 使 sim=0.9 下 C_over_N 落在 ~0.2-0.4 的生产量级（K̄~2.3K@T=16K）
    n_base = max(1, T // 4)
    base = torch.randn(n_base, H, dd, generator=g, device=DEV)
    tok = torch.randint(0, n_base, (T, H), generator=g, device=DEV)
    noise = torch.randn(T, H, dd, generator=g, device=DEV) * 0.15
    x = (base[tok, torch.arange(H, device=DEV).unsqueeze(0)] + noise).float()
    return x


def bench_arm(fn, args_factory, reps=3, warmup=True):
    """fn 会原地修改 state tensor（index_add_）→ warmup 与 timed 每次经
    args_factory() 取全新 state，杜绝污染轨迹（曾致 assign 对比假 diff 1.6%：
    timed 轮从 warmup 轮残留的 k_live/sums 续跑）。"""
    out = None
    if warmup:
        fn(*args_factory())
        torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn(*args_factory())
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return out, min(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ts", type=int, nargs="+", default=[4096, 8192, 16384])
    ap.add_argument("--sims", type=float, nargs="+", default=[0.9])
    ap.add_argument("--dds", type=int, nargs="+", default=[32])
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--dump", type=str, default=None)
    ap.add_argument("--out", type=str,
                    default=os.path.join(os.path.dirname(__file__), "e113_sim_greedy_bench.json"))
    ap.add_argument("--graph", action="store_true", help="启用 C1 CUDA Graph 臂")
    ap.add_argument("--cpu", action="store_true", help="启用 C2 CPU 单线程对照臂")
    args = ap.parse_args()
    n_threads_orig = torch.get_num_threads()

    res = {"meta": {
        "bench": "E113 sim_greedy kernel microbench（#150 备用脚本）",
        "device": torch.cuda.get_device_name(0),
        "data": "synthetic（簇结构合成；--dump 接真实 k 待补）" if not args.dump else "dump",
        "note": "reference=HEAD Python 循环基线；cudagraph=C1 原型；fused=V1 kernel TODO",
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, "grid": []}

    for T in args.Ts:
        for sim in args.sims:
            for dd in args.dds:
                x = make_inputs(T, args.H, dd)
                Kcap = T
                sums0 = x.new_zeros(args.H, Kcap, dd)
                cnt0 = x.new_zeros(args.H, Kcap)
                sq0 = x.new_zeros(args.H, Kcap)
                klive0 = torch.zeros(args.H, dtype=torch.long, device=DEV)

                row = {"T": T, "sim": sim, "dd": dd, "H": args.H}

                # 每次调用全新 state 的工厂（贪心链原地改 state，复用即污染轨迹）
                def gpu_args():
                    return (x, sim,
                            x.new_zeros(args.H, Kcap, dd), x.new_zeros(args.H, Kcap),
                            x.new_zeros(args.H, Kcap),
                            torch.zeros(args.H, dtype=torch.long, device=DEV))

                # A. reference（reps=1：T 大时循环很慢，C_over_N 由它顺便给出）
                (s1, c1, q1, k1, a1), t_ref = bench_arm(reference_pass, gpu_args, reps=1)
                row["reference_s"] = round(t_ref, 4)
                row["us_per_token"] = round(t_ref / T * 1e6, 2)
                row["C_over_N"] = round((k1.max().item()) / T, 4)

                # B. C1 CUDA Graph 原型
                if args.graph:
                    try:
                        (s2, c2, q2, k2, a2, tg) = cudagraph_pass(*gpu_args(), unroll=8)
                        row["cudagraph_s"] = round(tg, 4)
                        row["cudagraph_speedup"] = round(t_ref / tg, 2)
                        # 决策等价对拍（铁律门 1 微型版）
                        row["assign_diff_rate"] = float((a1 != a2).float().mean().item())
                        # unroll pad 步给每头 k_live 膨胀 T_pad-T（零 token 新建空簇）→ 扣除后对拍
                        U = 8
                        infl = ((T + U - 1) // U * U - T)
                        row["klive_match"] = bool(
                            ((k2 - infl).clamp(min=0) == k1).all().item())
                    except Exception as e:  # graph 捕获失败不阻塞基线
                        row["cudagraph_error"] = repr(e)[:200]

                # C2. CPU 单线程对照臂（C2 可行性决定性测量；每 worker set_num_threads(1)）
                if args.cpu:
                    xc = x.cpu()
                    torch.set_num_threads(1)

                    def cpu_args():
                        return (xc, sim,
                                xc.new_zeros(args.H, Kcap, dd), xc.new_zeros(args.H, Kcap),
                                xc.new_zeros(args.H, Kcap),
                                torch.zeros(args.H, dtype=torch.long))

                    try:
                        (_, _, _, k3, a3), t_cpu = bench_arm(cpu_pass, cpu_args, reps=1)
                        row["cpu_1thread_s"] = round(t_cpu, 4)
                        row["cpu_us_per_token"] = round(t_cpu / T * 1e6, 2)
                        # 注意：CPU 与 GPU fp 路径天然不同，diff 率仅供 C2 语义参考
                        # （C2 生产化时参考实现也应在 CPU 侧，同路径对拍）
                        row["cpu_vs_gpu_assign_diff"] = float(
                            (a1.cpu() != a3).float().mean().item())
                    finally:
                        try:
                            torch.set_num_threads(n_threads_orig)
                        except Exception:
                            pass  # 部分 torch 版本并行区启动后禁改——后续 GPU 臂不受影响

                # C. V1 fused kernel 占位
                row["fused"] = "TODO（CUDA persistent kernel 落地后接入）"

                res["grid"].append(row)
                print(json.dumps(row, ensure_ascii=False))

    with open(args.out, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
